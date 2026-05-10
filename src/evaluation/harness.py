from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from src.evaluation.conditions import Condition, apply_condition
from src.evaluation.hallucination import flag_hallucination


@dataclass
class EvaluationOutput:
    timestamp: str
    model: str
    case_id: str
    condition: str
    mode: str
    diagnosis: str | None
    confidence: float | None
    explanation: str | None
    hallucination_flag: bool
    hallucination_reason: str
    ground_truth: str
    status: str
    error: str | None
    runtime_seconds: float
    image_path: str | None
    clinical_text: str | None
    raw_response: str | None

    def to_dict(self) -> dict:
        return asdict(self)

def _cache_path(results_root: str, model_name: str,
                case_id: str, condition_name: str) -> Path:
    safe_case = case_id.replace("/", "_").replace("\\", "_")
    fname = f"{safe_case}__{condition_name}.json"
    return Path(results_root) / "cache" / model_name / fname


def _load_cache(path: Path) -> EvaluationOutput | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return EvaluationOutput(**json.load(f))
    except Exception:
        return None 


def _save_cache(path: Path, output: EvaluationOutput) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(output.to_dict(), f, indent=2)


def evaluate_triple(
    model_name: str,
    pipeline,
    case: dict,
    condition: Condition,
    registry=None,
    results_root: str = "results",
    use_cache: bool = True,
) -> EvaluationOutput:

    # edit cache check 
    cache_file = _cache_path(results_root, model_name,
                              case["case_id"], condition.name)
    if use_cache:
        cached = _load_cache(cache_file)
        if cached is not None:
            return cached

    conditioned_case = apply_condition(case, condition, registry=registry)

    old_mode = getattr(pipeline, "mode", None)
    if old_mode is not None:
        pipeline.mode = condition.mode

    started = time.monotonic()

    try:
        output = pipeline.predict(
            image=conditioned_case["image"],
            text=conditioned_case["text"],
            case_id=conditioned_case["case_id"],
            ground_truth=conditioned_case["ground_truth"],
        )
        hallucination = flag_hallucination(
            output.diagnosis,
            output.explanation,
            output.ground_truth,
        )
        if not output.diagnosis or not output.explanation or not output.raw_response:
            status = "INCOMPLETE_OUTPUT"
            error = "Model returned an incomplete output object."
        elif output.diagnosis == "parse_failed":
            status = "PARSE_FAILED"
            error = "Model returned text, but it did not match the structured schema."
        else:
            status = "PASS"
            error = None
        diagnosis = output.diagnosis
        confidence = output.confidence
        explanation = output.explanation
        raw_response = output.raw_response

    except Exception as exc:
        hallucination = flag_hallucination(None, None, conditioned_case["ground_truth"])
        status = "FAIL"
        error = f"{type(exc).__name__}: {exc}"
        diagnosis = None
        confidence = None
        explanation = None
        raw_response = None

    finally:
        if old_mode is not None:
            pipeline.mode = old_mode

    result = EvaluationOutput(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        model=model_name,
        case_id=conditioned_case["case_id"],
        condition=condition.name,
        mode=condition.mode,
        diagnosis=diagnosis,
        confidence=confidence,
        explanation=explanation,
        hallucination_flag=hallucination.flag,
        hallucination_reason=hallucination.reason,
        ground_truth=conditioned_case["ground_truth"],
        status=status,
        error=error,
        runtime_seconds=round(time.monotonic() - started, 2),
        image_path=conditioned_case.get("image_path"),
        clinical_text=conditioned_case.get("text"),
        raw_response=raw_response,
    )

    # edit cache pass and known failures
    if result.status in ("PASS", "INCOMPLETE_OUTPUT", "PARSE_FAILED"):
        _save_cache(cache_file, result)

    return result

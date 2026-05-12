from __future__ import annotations

import json
import os
from pathlib import Path

from scripts.run_experiment import load_pipeline
from src.evaluation.calibration import add_calibration_fields
from src.evaluation.conditions import CONDITIONS, get_condition
from src.evaluation.harness import evaluate_triple
from src.pipelines.case_loader import load_chexpert_cases
from src.pipelines.result_writer import write_jsonl
from src.perturbations.registry import PacemakerRegistry


ALL_CONDITIONS = [
    "original",
    "image_only",
    "text_only",
    "watermark_v1",
    "watermark_v2",
    "watermark_v3",
    "jpeg_v1",
    "jpeg_v2",
    "jpeg_v3",
    "chest_tube_v1",
    "chest_tube_v2",
    "chest_tube_v3",
    "chest_drain_v1",
    "chest_drain_v2",
    "chest_drain_v3",
    "ecg_leads_v1",
    "ecg_leads_v2",
    "ecg_leads_v3",
    "pacemaker_v1",
    "pacemaker_v2",
    "pacemaker_v3",
    "negative_control",
    "demographic_v1",
    "demographic_v2",
    "contradiction_v1",
    "contradiction_v2",
    "paraphrase_v1",
    "paraphrase_v2",
    "paraphrase_v3",
]

DEFAULT_MODELS = ["qwen2_vl", "biovil_t"]


def run_batch(
    *,
    models: list[str] | None = None,
    conditions: list[str] | None = None,
    cases: int | None = None,
    split: str = "valid",
    chexpert_root: str | None = None,
    shared_dir: str = "shared_outputs/full_eval",
    device: str = "auto",
    retries: int = 1,
) -> list[dict]:
    models = models or DEFAULT_MODELS
    conditions = conditions or ALL_CONDITIONS
    chexpert_root = chexpert_root or os.environ.get("CHEXPERT_ROOT", "data/chexpert small")
    summaries = []
    for model in models:
        summaries.append(
            run_model_conditions(
                model=model,
                conditions=conditions,
                cases=cases,
                split=split,
                chexpert_root=chexpert_root,
                shared_dir=shared_dir,
                device=device,
                retries=retries,
            )
        )
    return summaries


def run_model_conditions(
    *,
    model: str,
    conditions: list[str],
    cases: int | None,
    split: str,
    chexpert_root: str,
    shared_dir: str,
    device: str,
    retries: int,
) -> dict:
    loaded_cases = load_chexpert_cases(
        chexpert_root,
        split=split,
        n=cases or 1_000_000,
    )
    if cases is not None:
        loaded_cases = loaded_cases[:cases]

    condition_objects = [get_condition(name) for name in conditions]
    output_dir = Path(shared_dir) / model
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{split}_{len(loaded_cases)}_{'_'.join(conditions)}.jsonl"
    summary_path = output_dir / f"{split}_{len(loaded_cases)}_{'_'.join(conditions)}_summary.json"

    rows = _load_existing_rows(result_path)
    completed = {
        (row["case_id"], row["condition"])
        for row in rows
        if row.get("status") != "FAIL"
    }

    pipeline = load_pipeline(model, mode="image_text", device=device)
    registry = PacemakerRegistry()

    for case in loaded_cases:
        for condition in condition_objects:
            key = (case["case_id"], condition.name)
            if key in completed:
                print(f"Skipping completed triple: ({model}, {key[0]}, {key[1]})", flush=True)
                continue

            result = _evaluate_with_retries(model, pipeline, case, condition, retries, registry=registry)
            rows = [row for row in rows if (row["case_id"], row["condition"]) != key]
            rows.append(result)
            rows = add_calibration_fields(rows)
            write_jsonl(str(result_path), rows)
            print(
                f"{model} | {case['case_id']} | {condition.name} | "
                f"{result['status']} | diagnosis={result['diagnosis']} | "
                f"confidence={result['confidence']} | hallucination={result['hallucination_flag']}",
                flush=True,
            )

    summary = _build_summary(model, loaded_cases, conditions, rows, str(result_path))
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _evaluate_with_retries(model: str, pipeline, case: dict, condition, retries: int, *, registry=None) -> dict:
    attempts = max(1, retries + 1)
    last_result = None
    for attempt in range(1, attempts + 1):
        result = evaluate_triple(model, pipeline, case, condition, registry=registry).to_dict()
        result["attempt"] = attempt
        if result["status"] != "FAIL":
            return result
        last_result = result
        print(
            f"Retryable failure ({attempt}/{attempts}) for "
            f"({model}, {case['case_id']}, {condition.name}): {result['error']}",
            flush=True,
        )
    return last_result


def _load_existing_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _build_summary(
    model: str,
    cases: list[dict],
    conditions: list[str],
    rows: list[dict],
    result_path: str,
) -> dict:
    return {
        "model": model,
        "cases": len(cases),
        "conditions": conditions,
        "expected_triples": len(cases) * len(conditions),
        "completed_triples": sum(1 for row in rows if row.get("status") != "FAIL"),
        "failed_triples": sum(1 for row in rows if row.get("status") == "FAIL"),
        "status_counts": _count_values(rows, "status"),
        "hallucination_counts": _count_values(rows, "hallucination_flag"),
        "result_path": result_path,
    }


def _count_values(rows: list[dict], key: str) -> dict:
    counts = {}
    for row in rows:
        value = str(row.get(key))
        counts[value] = counts.get(value, 0) + 1
    return counts

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_experiment import load_pipeline
from src.evaluation.calibration import add_calibration_fields
from src.evaluation.conditions import get_condition
from src.evaluation.harness import evaluate_triple
from src.pipelines.chexpert_plus_loader import load_chexpert_plus_cases
from src.pipelines.result_writer import write_jsonl
from src.perturbations.registry import PacemakerRegistry


DEFAULT_MODELS = ["qwen2_vl", "llava_med", "biovil_t", "med_flamingo"]

DEFAULT_CONDITIONS = [
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--cases", type=int, default=10)
    parser.add_argument("--split", default=None)
    parser.add_argument("--chexpert-root", default=str(PROJECT_ROOT / "data" / "chexpert plus"))
    parser.add_argument(
        "--shared-dir",
        default=str(PROJECT_ROOT / "experiments" / "chexpert_plus" / "outputs" / "full_eval"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--retries", type=int, default=1)
    args = parser.parse_args()

    try:
        cases = load_chexpert_plus_cases(
            args.chexpert_root,
            split=args.split,
            n=args.cases,
            frontal_only=True,
        )
    except FileNotFoundError as exc:
        raise SystemExit(
            f"{exc}\n\nDownload CheXpert Plus first, then set --chexpert-root "
            "to the downloaded dataset folder."
        ) from exc
    conditions = [get_condition(name) for name in args.conditions]

    summaries = []
    for model_name in args.models:
        output_dir = Path(args.shared_dir) / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = _result_stem("chexpert_plus", len(cases), args.conditions)
        result_path = output_dir / f"{stem}.jsonl"
        summary_path = output_dir / f"{stem}_summary.json"
        rows = _load_existing_rows(result_path)
        completed = {
            (row["case_id"], row["condition"])
            for row in rows
            if row.get("status") != "FAIL"
        }

        pipeline = load_pipeline(model_name, mode="image_text", device=args.device)
        registry = PacemakerRegistry()
        for case in cases:
            for condition in conditions:
                key = (case["case_id"], condition.name)
                if key in completed:
                    print(f"Skipping completed triple: ({model_name}, {key[0]}, {key[1]})", flush=True)
                    continue

                result = _evaluate_with_retries(model_name, pipeline, case, condition, args.retries, registry=registry)
                rows = [row for row in rows if (row["case_id"], row["condition"]) != key]
                rows.append(result)
                rows = add_calibration_fields(rows)
                write_jsonl(str(result_path), rows)
                print(
                    f"{model_name} | {case['case_id']} | {condition.name} | "
                    f"{result['status']} | diagnosis={result['diagnosis']} | "
                    f"confidence={result['confidence']} | hallucination={result['hallucination_flag']}",
                    flush=True,
                )

        summary = _build_summary(model_name, cases, args.conditions, rows, str(result_path))
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        summaries.append(summary)

    print("\nBatch summaries:")
    for summary in summaries:
        print(summary)


def _evaluate_with_retries(model_name: str, pipeline, case: dict, condition, retries: int, *, registry=None) -> dict:
    attempts = max(1, retries + 1)
    last_result = None
    for attempt in range(1, attempts + 1):
        result = evaluate_triple(model_name, pipeline, case, condition, registry=registry).to_dict()
        result["attempt"] = attempt
        if result["status"] != "FAIL":
            return result
        last_result = result
        print(
            f"Retryable failure ({attempt}/{attempts}) for "
            f"({model_name}, {case['case_id']}, {condition.name}): {result['error']}",
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


def _result_stem(dataset_label: str, case_count: int, conditions: list[str]) -> str:
    condition_slug = "_".join(conditions)
    stem = f"{dataset_label}_{case_count}_{condition_slug}"
    if len(stem) <= 180:
        return stem
    digest = hashlib.sha1(condition_slug.encode("utf-8")).hexdigest()[:10]
    return f"{dataset_label}_{case_count}_{len(conditions)}conditions_{digest}"


def _build_summary(
    model: str,
    cases: list[dict],
    conditions: list[str],
    rows: list[dict],
    result_path: str,
) -> dict:
    return {
        "dataset": "chexpert_plus",
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


if __name__ == "__main__":
    main()

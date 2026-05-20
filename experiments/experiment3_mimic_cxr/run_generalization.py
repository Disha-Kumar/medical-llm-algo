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
from src.pipelines.mimic_cxr_loader import load_mimic_cxr_cases
from src.pipelines.result_writer import write_jsonl


DEFAULT_MODELS = ["qwen2_vl", "biovil_t"]
DEFAULT_CONDITIONS = ["original", "image_only", "text_only"]


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    cases = _load_cases(args)
    summaries = _run(
        models=args.models,
        cases=cases,
        condition_names=args.conditions,
        shared_dir=Path(args.shared_dir),
        device=args.device,
        retries=args.retries,
        output_prefix="mimic_generalization",
    )
    print("\nMIMIC-CXR Experiment 3 summaries:")
    for summary in summaries:
        print(summary)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--cases", type=int, default=10)
    parser.add_argument("--split", default=None)
    parser.add_argument("--mimic-root", required=True)
    parser.add_argument("--reports-root", default=None)
    parser.add_argument("--cohort-column", default=None)
    parser.add_argument("--cohort-value", default=None)
    parser.add_argument("--scanner-column", default=None)
    parser.add_argument("--scanner-value", default=None)
    parser.add_argument(
        "--shared-dir",
        default=str(PROJECT_ROOT / "experiments" / "experiment3_mimic_cxr" / "outputs"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--retries", type=int, default=1)
    return parser


def _load_cases(args) -> list[dict]:
    return load_mimic_cxr_cases(
        args.mimic_root,
        split=args.split,
        n=args.cases,
        frontal_only=True,
        reports_root=args.reports_root,
        cohort_column=args.cohort_column,
        cohort_value=args.cohort_value,
        scanner_column=args.scanner_column,
        scanner_value=args.scanner_value,
    )


def _run(
    *,
    models: list[str],
    cases: list[dict],
    condition_names: list[str],
    shared_dir: Path,
    device: str,
    retries: int,
    output_prefix: str,
) -> list[dict]:
    conditions = [get_condition(name) for name in condition_names]
    summaries = []
    for model_name in models:
        output_dir = shared_dir / output_prefix / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = _result_stem(output_prefix, len(cases), condition_names)
        result_path = output_dir / f"{stem}.jsonl"
        summary_path = output_dir / f"{stem}_summary.json"
        rows = _load_existing_rows(result_path)
        completed = {
            (row["case_id"], row["condition"])
            for row in rows
            if row.get("status") != "FAIL"
        }

        pipeline = load_pipeline(model_name, mode="image_text", device=device)
        for case in cases:
            for condition in conditions:
                key = (case["case_id"], condition.name)
                if key in completed:
                    print(f"Skipping completed triple: ({model_name}, {key[0]}, {key[1]})", flush=True)
                    continue
                result = _evaluate_with_retries(model_name, pipeline, case, condition, retries)
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

        summary = _summary(output_prefix, model_name, cases, condition_names, rows, str(result_path))
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        summaries.append(summary)
    return summaries


def _evaluate_with_retries(model_name: str, pipeline, case: dict, condition, retries: int) -> dict:
    attempts = max(1, retries + 1)
    last_result = None
    for attempt in range(1, attempts + 1):
        result = evaluate_triple(model_name, pipeline, case, condition).to_dict()
        result["attempt"] = attempt
        if result["status"] != "FAIL":
            return result
        last_result = result
        print(f"Retryable failure ({attempt}/{attempts}) for ({model_name}, {case['case_id']}, {condition.name})", flush=True)
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


def _result_stem(prefix: str, case_count: int, conditions: list[str]) -> str:
    condition_slug = "_".join(conditions)
    stem = f"{prefix}_{case_count}_{condition_slug}"
    if len(stem) <= 180:
        return stem
    digest = hashlib.sha1(condition_slug.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{case_count}_{len(conditions)}conditions_{digest}"


def _summary(experiment: str, model: str, cases: list[dict], conditions: list[str], rows: list[dict], result_path: str) -> dict:
    return {
        "experiment": experiment,
        "dataset": "mimic_cxr",
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

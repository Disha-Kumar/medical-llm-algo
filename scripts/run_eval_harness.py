#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_experiment import PIPELINES, load_pipeline
from src.perturbations.registry import PacemakerRegistry
from src.evaluation.calibration import add_calibration_fields
from src.evaluation.conditions import CONDITIONS, get_condition
from src.evaluation.harness import evaluate_triple
from src.pipelines.case_loader import load_chexpert_cases
from src.pipelines.result_writer import write_jsonl



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=PIPELINES.keys())
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["original", "image_only", "text_only"],
        choices=CONDITIONS.keys(),
    )
    parser.add_argument("--cases", type=int, default=1)
    parser.add_argument("--split", default="valid")
    parser.add_argument(
        "--chexpert-root",
        default="/Users/ard/Desktop/medical llm/data/chexpert small",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--results-dir", default="results/evaluation_harness")
    args = parser.parse_args()

    print(
        f"Evaluation harness: model={args.model}, cases={args.cases}, "
        f"conditions={','.join(args.conditions)}",
        flush=True,
    )

    cases = load_chexpert_cases(args.chexpert_root, split=args.split, n=args.cases)
    pipeline = load_pipeline(args.model, mode="image_text", device=args.device)
    registry = PacemakerRegistry() 
    conditions = [get_condition(name) for name in args.conditions]

    rows = []
    result_path = os.path.join(
        args.results_dir,
        args.model,
        f"{args.split}_{len(cases)}_{'_'.join(args.conditions)}.jsonl",
    )
    summary_path = os.path.join(
        args.results_dir,
        args.model,
        f"{args.split}_{len(cases)}_{'_'.join(args.conditions)}_summary.json",
    )

    for case in cases:
        for condition in conditions:
            print(f"\nRunning triple: ({args.model}, {case['case_id']}, {condition.name})", flush=True)
            result = evaluate_triple(args.model, pipeline, case, condition, registry=registry)
            rows.append(result.to_dict())
            rows = add_calibration_fields(rows)
            write_jsonl(result_path, rows)
            print(
                f"  {result.status} | diagnosis={result.diagnosis} | "
                f"confidence={result.confidence} | hallucination={result.hallucination_flag} | "
                f"{result.runtime_seconds}s",
                flush=True,
            )

    summary = {
        "model": args.model,
        "cases": len(cases),
        "conditions": args.conditions,
        "total_triples": len(rows),
        "status_counts": count_values(rows, "status"),
        "hallucination_counts": count_values(rows, "hallucination_flag"),
        "result_path": result_path,
    }
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    completed = sum(1 for row in rows if row["status"] != "FAIL")
    print(f"\nCompleted triples: {completed}/{len(rows)}", flush=True)
    print(f"Results saved to {result_path}", flush=True)
    print(f"Summary saved to {summary_path}", flush=True)


def count_values(rows: list[dict], key: str) -> dict:
    counts = {}
    for row in rows:
        value = str(row.get(key))
        counts[value] = counts.get(value, 0) + 1
    return counts


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.base import save_output
from src.pipelines.case_loader import load_chexpert_cases
from src.pipelines.result_writer import build_result_row, write_jsonl


PIPELINES = {
    "llava_med": ("src.pipelines.llava_med", "LLaVAMedPipeline"),
    "qwen2_vl": ("src.pipelines.qwen2_vl", "Qwen2VLPipeline"),
    "gpt4o": ("src.pipelines.gpt4o", "GPT4oPipeline"),
    "gpt4o_openrouter": ("src.pipelines.gpt4o_openrouter", "GPT4oOpenRouterPipeline"),
    "biovil_t": ("src.pipelines.biovil_t", "BioViLTPipeline"),
    "med_flamingo": ("src.pipelines.med_flamingo", "MedFlamingoPipeline"),
    "chexagent": ("src.pipelines.chexagent", "CheXAgentPipeline"),
}


def load_pipeline(model: str, mode: str, device: str):
    import importlib

    module_name, class_name = PIPELINES[model]
    cls = getattr(importlib.import_module(module_name), class_name)
    if model in {"llava_med", "qwen2_vl", "chexagent"}:
        return cls(device=device, mode=mode)
    return cls(mode=mode)


def safe_save_output(out, output_dir: str) -> str | None:
    try:
        save_output(out, output_dir)
        return None
    except OSError as exc:
        return f"{type(exc).__name__}: could not save detail output JSON: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=PIPELINES.keys())
    parser.add_argument(
        "--mode",
        default="image_text",
        choices=["image_text", "image_only", "text_only"],
    )
    parser.add_argument(
        "--chexpert-root",
        default="/Users/ard/Desktop/medical llm/data/chexpert small",
    )
    parser.add_argument("--split", default="valid")
    parser.add_argument("--cases", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="logs")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    print(
        f"Experiment started: model={args.model}, mode={args.mode}, "
        f"cases={args.cases}, split={args.split}",
        flush=True,
    )
    cases = load_chexpert_cases(args.chexpert_root, split=args.split, n=args.cases)
    pipeline = load_pipeline(args.model, args.mode, args.device)

    rows = []
    logs = []
    result_path = os.path.join(
        args.results_dir,
        args.model,
        f"{args.mode}_{args.split}_{len(cases)}.jsonl",
    )
    log_path = os.path.join(
        args.output_dir,
        "confirmation_logs",
        f"{args.model}_{args.mode}_{datetime.now():%Y%m%d_%H%M%S}.json",
    )
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    for case in cases:
        print(f"\nRunning on {case['case_id']}...", flush=True)
        started = time.monotonic()
        try:
            out = pipeline.predict(
                image=case["image"],
                text=case["text"],
                case_id=case["case_id"],
                ground_truth=case["ground_truth"],
            )
            if not out.diagnosis or not out.explanation or not out.raw_response:
                status = "INCOMPLETE_OUTPUT"
                error = "Model returned an incomplete output object."
            elif out.diagnosis == "parse_failed":
                status = "PARSE_FAILED"
                error = "Model returned text, but it did not match the structured schema."
            else:
                status = "PASS"
                error = None
        except Exception as exc:
            out = None
            status = "FAIL"
            error = f"{type(exc).__name__}: {exc}"

        elapsed = time.monotonic() - started
        if out and status == "PASS":
            print(
                f"  PASS | diagnosis={out.diagnosis} | "
                f"confidence={out.confidence:.2f} | {elapsed:.1f}s",
                flush=True,
            )
            save_warning = safe_save_output(out, args.output_dir)
            if save_warning:
                error = save_warning if error is None else f"{error}; {save_warning}"
        elif out:
            print(
                f"  {status} | diagnosis={out.diagnosis} | "
                f"confidence={out.confidence:.2f} | {elapsed:.1f}s",
                flush=True,
            )
            save_warning = safe_save_output(out, args.output_dir)
            if save_warning:
                error = save_warning if error is None else f"{error}; {save_warning}"
        else:
            print(f"  FAIL | {error} | {elapsed:.1f}s", flush=True)

        rows.append(
            build_result_row(
                model=args.model,
                mode=args.mode,
                case=case,
                output=out,
                status=status,
                error=error,
                elapsed_seconds=elapsed,
            )
        )
        logs.append(
            {
                "case_id": case["case_id"],
                "ground_truth": case["ground_truth"],
                "status": status,
                "error": error,
                "diagnosis": out.diagnosis if out else None,
                "confidence": out.confidence if out else None,
                "explanation": out.explanation if out else None,
                "elapsed_seconds": round(elapsed, 2),
            }
        )
        write_jsonl(result_path, rows)
        with open(log_path, "w") as f:
            json.dump(logs, f, indent=2)

    passed = sum(
        1
        for row in rows
        if row["status"] in {"PASS", "PARSE_FAILED", "INCOMPLETE_OUTPUT"}
    )
    print(f"\nResult: {passed}/{len(rows)} passed", flush=True)
    print(f"Results saved to {result_path}", flush=True)
    print(f"Log saved to {log_path}", flush=True)


if __name__ == "__main__":
    main()

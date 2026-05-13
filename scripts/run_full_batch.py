#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from medical_llm_eval import ALL_CONDITIONS, DEFAULT_MODELS, run_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--conditions", nargs="+", default=ALL_CONDITIONS)
    parser.add_argument("--cases", type=int, default=None)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--dataset", default="small", choices=["small", "plus"])
    parser.add_argument("--chexpert-root", default=None)
    parser.add_argument("--shared-dir", default="shared_outputs/full_eval")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--retries", type=int, default=1)
    args = parser.parse_args()

    summaries = run_batch(
        models=args.models,
        conditions=args.conditions,
        cases=args.cases,
        split=args.split,
        dataset=args.dataset,
        chexpert_root=args.chexpert_root,
        shared_dir=args.shared_dir,
        device=args.device,
        retries=args.retries,
    )
    print("\nBatch summaries:")
    for summary in summaries:
        print(summary)


if __name__ == "__main__":
    main()

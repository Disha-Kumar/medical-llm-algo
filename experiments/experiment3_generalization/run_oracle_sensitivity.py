#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.experiment3_generalization.run_generalization import _load_cases, _run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["qwen2_vl", "llava_med", "biovil_t", "med_flamingo"])
    parser.add_argument("--cases", type=int, default=10)
    parser.add_argument("--split", default=None)
    parser.add_argument("--chexpert-root", default=str(PROJECT_ROOT / "data" / "chexpert plus"))
    parser.add_argument("--cohort-column", default=None)
    parser.add_argument("--cohort-value", default=None)
    parser.add_argument("--scanner-column", default=None)
    parser.add_argument("--scanner-value", default=None)
    parser.add_argument(
        "--shared-dir",
        default=str(PROJECT_ROOT / "experiments" / "experiment3_generalization" / "outputs"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--retries", type=int, default=1)
    args = parser.parse_args()

    cases = _load_cases(args)
    summaries = _run(
        models=args.models,
        cases=cases,
        condition_names=["original", "oracle_k1", "oracle_k3", "oracle_k5"],
        shared_dir=Path(args.shared_dir),
        device=args.device,
        retries=args.retries,
        output_prefix="oracle_sensitivity",
    )

    print("\nOracle sensitivity summaries:")
    for summary in summaries:
        print(summary)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run image_text, image_only, and text_only baselines for selected models."""

from __future__ import annotations

import argparse
import subprocess
import sys


DEFAULT_MODES = ["image_text", "image_only", "text_only"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen2_vl", "llava_med", "biovil_t"],
        choices=["qwen2_vl", "llava_med", "biovil_t", "med_flamingo"],
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=DEFAULT_MODES,
        choices=DEFAULT_MODES,
    )
    parser.add_argument("--cases", type=int, default=1)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    for model in args.models:
        for mode in args.modes:
            if model == "med_flamingo" and mode == "text_only":
                print("Skipping med_flamingo text_only: Med-Flamingo requires image input.")
                continue

            command = [
                sys.executable,
                "-u",
                "scripts/run_experiment.py",
                "--model",
                model,
                "--mode",
                mode,
                "--cases",
                str(args.cases),
                "--device",
                args.device,
            ]
            print("\n" + "=" * 80)
            print("Running:", " ".join(command))
            print("=" * 80, flush=True)
            subprocess.run(command, check=False)


if __name__ == "__main__":
    main()

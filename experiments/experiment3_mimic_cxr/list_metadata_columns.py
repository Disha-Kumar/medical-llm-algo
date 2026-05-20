#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.mimic_cxr_loader import available_metadata_columns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", required=True)
    args = parser.parse_args()

    try:
        for column in available_metadata_columns(args.mimic_root):
            print(column)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"{exc}\n\nExpected a MIMIC-CXR-JPG style folder with metadata, split, "
            "CheXpert labels, and files/ image tree."
        ) from exc


if __name__ == "__main__":
    main()

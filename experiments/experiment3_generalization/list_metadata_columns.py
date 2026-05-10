#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.chexpert_plus_loader import available_metadata_columns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chexpert-root", default=str(PROJECT_ROOT / "data" / "chexpert plus"))
    parser.add_argument("--split", default=None)
    args = parser.parse_args()

    try:
        columns = available_metadata_columns(args.chexpert_root, args.split)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"{exc}\n\nDownload CheXpert Plus first, then set --chexpert-root "
            "to the downloaded dataset folder."
        ) from exc

    for column in columns:
        print(column)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "run_full_batch.py"
DEFAULT_CHEXPERT_ROOT = PROJECT_ROOT / "data" / "chexpert small"
DEFAULT_SHARED_DIR = (
    PROJECT_ROOT / "experiments" / "chexpert_small_testing" / "outputs" / "full_eval"
)


def _ensure_arg(flag: str, value: str) -> None:
    if flag not in sys.argv:
        sys.argv.extend([flag, value])


def main() -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.argv[0] = str(SCRIPT)
    _ensure_arg("--chexpert-root", str(DEFAULT_CHEXPERT_ROOT))
    _ensure_arg("--shared-dir", str(DEFAULT_SHARED_DIR))
    runpy.run_path(str(SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Print environment facts that commonly explain VLM setup failures."""

from __future__ import annotations

import importlib.util
import platform
import sys


PACKAGES = [
    "torch",
    "transformers",
    "accelerate",
    "PIL",
    "pandas",
    "sentencepiece",
    "protobuf",
    "llava",
]


def package_version(name: str) -> str:
    spec_name = "PIL" if name == "PIL" else name
    if importlib.util.find_spec(spec_name) is None:
        return "not installed"
    try:
        module = __import__(spec_name)
        return getattr(module, "__version__", "installed, no __version__")
    except Exception as exc:
        return f"import failed: {type(exc).__name__}: {exc}"


def main() -> None:
    print("Python")
    print(f"  executable: {sys.executable}")
    print(f"  version:    {sys.version.split()[0]}")
    print(f"  platform:   {platform.platform()}")
    print(f"  machine:    {platform.machine()}")
    print()

    print("Packages")
    for package in PACKAGES:
        print(f"  {package:13} {package_version(package)}")
    print()

    if importlib.util.find_spec("torch") is None:
        print("Torch devices")
        print("  torch is not installed")
        return

    import torch

    print("Torch devices")
    print(f"  cuda available: {torch.cuda.is_available()}")
    print(f"  mps available:  {torch.backends.mps.is_available()}")
    print(f"  mps built:      {torch.backends.mps.is_built()}")
    print()

    if sys.version_info >= (3, 12):
        print("Warning")
        print(
            "  LLaVA-Med's official setup uses Python 3.10. "
            "If you are seeing import/build errors, create a fresh 3.10 env."
        )


if __name__ == "__main__":
    main()

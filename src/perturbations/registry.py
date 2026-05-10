from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np

ASSETS_DIR = Path(__file__).parent / "assets"

class PacemakerRegistry:
    def __init__(self):
        paths = sorted(ASSETS_DIR.glob("pm*.png"))
        if not paths:
            raise FileNotFoundError(f"No pacemaker PNGs found in {ASSETS_DIR}")
        self._images = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in paths]
        if any(img is None for img in self._images):
            raise RuntimeError("One or more pacemaker PNGs failed to load.")
        self._index = 0

    def next_pacemaker(self) -> np.ndarray:
        img = self._images[self._index % len(self._images)]
        self._index += 1
        return img

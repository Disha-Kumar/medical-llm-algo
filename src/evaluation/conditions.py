from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
from src.perturbations.image_perturbations import ALL_CONDITIONS as IMAGE_CONDITIONS
from src.perturbations.text_perturbations import ALL_CONDITIONS as TEXT_CONDITIONS
from src.perturbations.text_perturbations import apply_text_perturbation
import random


@dataclass(frozen=True)
class Condition:
    name: str
    mode: str
    image_perturbation: str = "none"
    text_perturbation: str = "none"
    oracle_steps: int = 0
    description: str = ""


CONDITIONS = {
    "original": Condition(
        name="original",
        mode="image_text",
        description="Control condition: original image plus original clinical note.",
    ),
    "image_only": Condition(
        name="image_only",
        mode="image_only",
        description="Baseline condition: image is provided without clinical note.",
    ),
    "text_only": Condition(
        name="text_only",
        mode="text_only",
        description="Baseline condition: clinical note is provided without image.",
    ),
    "oracle_k1": Condition(
        name="oracle_k1",
        mode="image_text",
        text_perturbation="oracle",
        oracle_steps=1,
        description="Oracle sensitivity condition: one step of ground-truth context.",
    ),
    "oracle_k3": Condition(
        name="oracle_k3",
        mode="image_text",
        text_perturbation="oracle",
        oracle_steps=3,
        description="Oracle sensitivity condition: three steps of ground-truth context.",
    ),
    "oracle_k5": Condition(
        name="oracle_k5",
        mode="image_text",
        text_perturbation="oracle",
        oracle_steps=5,
        description="Oracle sensitivity condition: five steps of ground-truth context.",
    ),
}

for perturbation_type, variant in IMAGE_CONDITIONS:
    if perturbation_type == "negative_control":
        name = "negative_control"
    else:
        name = f"{perturbation_type}_{variant}"

    CONDITIONS[name] = Condition(
        name=name,
        mode="image_text",
        image_perturbation=name,
        description=f"{perturbation_type} perturbation ({variant})",
    )
    
_IMAGE_TYPES = sorted({pt for pt, _ in IMAGE_CONDITIONS if pt != "negative_control"})
for perturbation_type in _IMAGE_TYPES:
    CONDITIONS[perturbation_type] = Condition(
        name=perturbation_type,
        mode="image_text",
        image_perturbation=perturbation_type,
        description=f"{perturbation_type} perturbation (random variant).",
    )

for _i in range(3):
    CONDITIONS[f"noise_floor_{_i}"] = Condition(
        name=f"noise_floor_{_i}",
        mode="image_text",
        image_perturbation="noise_floor",
        description=f"Noise floor run {_i}: destroyed inputs.",
    )

for perturbation_type, variant in TEXT_CONDITIONS:
    name = f"{perturbation_type}_{variant}"
    CONDITIONS[name] = Condition(
        name=name,
        mode="image_text",
        text_perturbation=name,
        description=f"{perturbation_type} text perturbation ({variant})",
    )

_TEXT_TYPES = sorted({pt for pt, _ in TEXT_CONDITIONS})
for perturbation_type in _TEXT_TYPES:
    CONDITIONS[perturbation_type] = Condition(
        name=perturbation_type,
        mode="image_text",
        text_perturbation=perturbation_type,
        description=f"{perturbation_type} text perturbation (random variant).",
    )


def get_condition(name: str) -> Condition:
    if name not in CONDITIONS:
        valid = ", ".join(sorted(CONDITIONS))
        raise ValueError(f"Unknown condition '{name}'. Valid conditions: {valid}")
    return CONDITIONS[name]


def apply_condition(case: dict, condition: Condition, registry=None) -> dict:
    from src.perturbations.image_perturbations import apply_perturbation

    conditioned = dict(case)
    image = case["image"].copy().convert("RGB")
    text = case["text"]

    if condition.image_perturbation != "none":
        img_gray = np.array(image.convert("L"))
        img_gray = cv2.resize(img_gray, (512, 512))

        perturb_type, variant = _parse_perturbation(condition.image_perturbation)
        if variant == "random":
            variant = random.choice(["v1", "v2", "v3"])
        kwargs = {}
        if perturb_type == "pacemaker":
            if registry is None:
                raise ValueError("registry must be provided for pacemaker conditions")
            pm = registry.next_pacemaker()
            if pm is None:
                raise RuntimeError("No pacemaker source images found in registry.")
            kwargs["pacemaker_source"] = pm
        elif perturb_type == "noise_floor":
            flat = img_gray.flatten()
            np.random.shuffle(flat)
            img_gray = flat.reshape(img_gray.shape)
            words = text.split()
            np.random.shuffle(words)
            text = " ".join(words)
            image = Image.fromarray(img_gray).convert("RGB")
            conditioned["image"] = image
            conditioned["text"] = text
            conditioned["condition"] = condition.name
            conditioned["mode"] = condition.mode
            return conditioned

        print(f"[PERTURB] {perturb_type} {variant}",flush=True,)
        perturbed_gray = apply_perturbation(img_gray, perturb_type, variant, **kwargs)
        image = Image.fromarray(perturbed_gray).convert("RGB")

    if condition.text_perturbation not in ("none", "oracle"):
        tp = condition.text_perturbation
        parts = tp.rsplit("_", 1)
        if len(parts) == 2 and parts[1] in ("v1", "v2", "v3"):
            perturb_type, variant = parts[0], parts[1]
        else:
            perturb_type = tp
            variant = random.choice(["v1", "v2", "v3"])
        text = apply_text_perturbation(
            text, perturb_type, variant,
            ground_truth=case.get("ground_truth", "no finding"),
        )
        
    elif condition.text_perturbation == "oracle":
        text = add_oracle_context(text, case.get("ground_truth", "unknown"), condition.oracle_steps)

    conditioned["image"] = image
    conditioned["text"] = text
    conditioned["condition"] = condition.name
    conditioned["mode"] = condition.mode
    return conditioned


def _parse_perturbation(image_perturbation: str) -> tuple[str, str]:
    if image_perturbation == "negative_control":
        return "negative_control", "v1"
    if image_perturbation == "noise_floor":
        return "noise_floor", "v1"
    parts = image_perturbation.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in ("v1", "v2", "v3"):
        return parts[0], parts[1]
    return image_perturbation, "random"


def add_oracle_context(text: str, ground_truth: str, steps: int) -> str:
    context_steps = [
        f"Oracle context 1: the benchmark ground-truth label is {ground_truth}.",
        "Oracle context 2: prioritize the benchmark label over non-clinical artifacts.",
        "Oracle context 3: keep the same label under clinically irrelevant perturbations or scanner artifacts.",
        "Oracle context 4: use confidence to express uncertainty, not to change labels due to artifacts.",
        "Oracle context 5: explanation should cite evidence consistent with the benchmark label only.",
    ]
    selected = " ".join(context_steps[:steps])
    return f"{text} {selected}"




from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance


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
    "text_style_verbose": Condition(
        name="text_style_verbose",
        mode="image_text",
        text_perturbation="verbose",
        description="Non-clinical text perturbation: same facts, verbose style.",
    ),
    "watermark_v1": Condition(name="watermark_v1", mode="image_text", image_perturbation="watermark_v1", description="Scanner watermark: Stanford Health Care, PA ERECT, bottom right."),
    "watermark_v2": Condition(name="watermark_v2", mode="image_text", image_perturbation="watermark_v2", description="Scanner watermark: GE Healthcare, PORTABLE AP, top left."),
    "watermark_v3": Condition(name="watermark_v3", mode="image_text", image_perturbation="watermark_v3", description="Scanner watermark: Beth Israel, AP SUPINE, bottom left."),
    "jpeg_v1": Condition(name="jpeg_v1", mode="image_text", image_perturbation="jpeg_v1", description="JPEG compression quality 75."),
    "jpeg_v2": Condition(name="jpeg_v2", mode="image_text", image_perturbation="jpeg_v2", description="JPEG compression quality 40."),
    "jpeg_v3": Condition(name="jpeg_v3", mode="image_text", image_perturbation="jpeg_v3", description="JPEG compression quality 10."),
    "chest_tube_v1": Condition(name="chest_tube_v1", mode="image_text", image_perturbation="chest_tube_v1", description="Synthetic chest tube, right side."),
    "chest_tube_v2": Condition(name="chest_tube_v2", mode="image_text", image_perturbation="chest_tube_v2", description="Synthetic chest tube, left side."),
    "chest_tube_v3": Condition(name="chest_tube_v3", mode="image_text", image_perturbation="chest_tube_v3", description="Synthetic chest tube, right side subtle."),
    "chest_drain_v1": Condition(name="chest_drain_v1", mode="image_text", image_perturbation="chest_drain_v1", description="Synthetic chest drain, upper pleural right."),
    "chest_drain_v2": Condition(name="chest_drain_v2", mode="image_text", image_perturbation="chest_drain_v2", description="Synthetic chest drain, lower pleural left."),
    "chest_drain_v3": Condition(name="chest_drain_v3", mode="image_text", image_perturbation="chest_drain_v3", description="Synthetic chest drain, mid pleural right."),
    "ecg_leads_v1": Condition(name="ecg_leads_v1", mode="image_text", image_perturbation="ecg_leads_v1", description="ECG leads, 3-lead standard."),
    "ecg_leads_v2": Condition(name="ecg_leads_v2", mode="image_text", image_perturbation="ecg_leads_v2", description="ECG leads, 5-lead."),
    "ecg_leads_v3": Condition(name="ecg_leads_v3", mode="image_text", image_perturbation="ecg_leads_v3", description="ECG leads, 3-lead position variant."),
    "pacemaker_v1": Condition(name="pacemaker_v1", mode="image_text", image_perturbation="pacemaker_v1", description="Pacemaker composite, upper left standard."),
    "pacemaker_v2": Condition(name="pacemaker_v2", mode="image_text", image_perturbation="pacemaker_v2", description="Pacemaker composite, slightly higher."),
    "pacemaker_v3": Condition(name="pacemaker_v3", mode="image_text", image_perturbation="pacemaker_v3", description="Pacemaker composite, slightly lower."),
    "negative_control": Condition(name="negative_control", mode="image_text", image_perturbation="negative_control", description="1 degree rotation, imperceptible, no clinical meaning."),
    "noise_floor_0": Condition(name="noise_floor_0", mode="image_text", image_perturbation="noise_floor", description="Noise floor run 0: destroyed inputs."),
    "noise_floor_1": Condition(name="noise_floor_1", mode="image_text", image_perturbation="noise_floor", description="Noise floor run 1: destroyed inputs."),
    "noise_floor_2": Condition(name="noise_floor_2", mode="image_text", image_perturbation="noise_floor", description="Noise floor run 2: destroyed inputs."),
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


ADVANCED_IMAGE_PERTURBATIONS = {
    "watermark_v1", "watermark_v2", "watermark_v3",
    "jpeg_v1", "jpeg_v2", "jpeg_v3",
    "chest_tube_v1", "chest_tube_v2", "chest_tube_v3",
    "chest_drain_v1", "chest_drain_v2", "chest_drain_v3",
    "ecg_leads_v1", "ecg_leads_v2", "ecg_leads_v3",
    "pacemaker_v1", "pacemaker_v2", "pacemaker_v3",
    "negative_control",
    "noise_floor",
}


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

    if condition.image_perturbation in ADVANCED_IMAGE_PERTURBATIONS:
        img_gray = np.array(image.convert("L"))
        img_gray = cv2.resize(img_gray, (512, 512))

        perturb_type, variant = _parse_perturbation(condition.image_perturbation)
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

        perturbed_gray = apply_perturbation(img_gray, perturb_type, variant, **kwargs)
        image = Image.fromarray(perturbed_gray).convert("RGB")

    if condition.text_perturbation == "verbose":
        text = (
            "Radiology evaluation request. The available case context is as follows: "
            f"{text} Please evaluate the chest radiograph for clinically relevant findings."
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
    return parts[0], parts[1]


def add_oracle_context(text: str, ground_truth: str, steps: int) -> str:
    context_steps = [
        f"Oracle context 1: the benchmark ground-truth label is {ground_truth}.",
        "Oracle context 2: prioritize the benchmark label over non-clinical artifacts.",
        "Oracle context 3: keep the same label if watermark, brightness, contrast, or wording changes.",
        "Oracle context 4: use confidence to express uncertainty, not to change labels due to artifacts.",
        "Oracle context 5: explanation should cite evidence consistent with the benchmark label only.",
    ]
    selected = " ".join(context_steps[:steps])
    return f"{text} {selected}"




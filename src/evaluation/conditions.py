from __future__ import annotations
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageEnhance
import cv2
import numpy as np

@dataclass(frozen=True)
class Condition:
    name: str
    mode: str
    image_perturbation: str = "none"
    text_perturbation: str = "none"
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
    "watermark": Condition(
        name="watermark",
        mode="image_text",
        image_perturbation="watermark",
        description="Non-clinical image perturbation: synthetic hospital watermark.",
    ),
    "brightness_low": Condition(
        name="brightness_low",
        mode="image_text",
        image_perturbation="brightness_low",
        description="Non-clinical image perturbation: reduced brightness.",
    ),
    "contrast_high": Condition(
        name="contrast_high",
        mode="image_text",
        image_perturbation="contrast_high",
        description="Non-clinical image perturbation: increased contrast.",
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
    "negative_control": Condition(name="negative_control", mode="image_text", image_perturbation="negative_control", description="1 degree rotation — imperceptible, no clinical meaning."),
    "noise_floor_0": Condition(name="noise_floor_0", mode="image_text", image_perturbation="noise_floor", description="Noise floor run 0: destroyed inputs."),
    "noise_floor_1": Condition(name="noise_floor_1", mode="image_text", image_perturbation="noise_floor", description="Noise floor run 1: destroyed inputs."),
    "noise_floor_2": Condition(name="noise_floor_2", mode="image_text", image_perturbation="noise_floor", description="Noise floor run 2: destroyed inputs."),
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


    if condition.image_perturbation in (
        "watermark_v1", "watermark_v2", "watermark_v3",
        "jpeg_v1", "jpeg_v2", "jpeg_v3",
        "chest_tube_v1", "chest_tube_v2", "chest_tube_v3",
        "chest_drain_v1", "chest_drain_v2", "chest_drain_v3",
        "ecg_leads_v1", "ecg_leads_v2", "ecg_leads_v3",
        "pacemaker_v1", "pacemaker_v2", "pacemaker_v3",
        "negative_control",
        "noise_floor",
    ):
    
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

    conditioned["image"] = image
    conditioned["text"] = text
    conditioned["condition"] = condition.name
    conditioned["mode"] = condition.mode
    return conditioned


def _parse_perturbation(image_perturbation: str) -> tuple[str, str]:
    """
    Splits "chest_tube_v1" into ("chest_tube", "v1").
    Handles multi-word types like chest_tube, chest_drain, ecg_leads.
    """
    if image_perturbation == "negative_control":
        return "negative_control", "v1"
    if image_perturbation == "noise_floor":
        return "noise_floor", "v1"
    parts = image_perturbation.rsplit("_", 1)
    return parts[0], parts[1]


def add_watermark(image: Image.Image) -> Image.Image:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    margin = max(8, image.width // 40)
    box_width = min(image.width - 2 * margin, max(120, image.width // 3))
    box_height = max(24, image.height // 14)
    x0 = image.width - box_width - margin
    y0 = margin
    draw.rectangle([x0, y0, x0 + box_width, y0 + box_height], fill=(20, 20, 20))
    draw.text((x0 + 8, y0 + 6), "HOSPITAL_A", fill=(245, 245, 245))
    return Image.blend(image, overlay, alpha=0.28)

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageEnhance


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
}


def get_condition(name: str) -> Condition:
    if name not in CONDITIONS:
        valid = ", ".join(sorted(CONDITIONS))
        raise ValueError(f"Unknown condition '{name}'. Valid conditions: {valid}")
    return CONDITIONS[name]


def apply_condition(case: dict, condition: Condition) -> dict:
    conditioned = dict(case)
    image = case["image"].copy().convert("RGB")
    text = case["text"]

    if condition.image_perturbation == "watermark":
        image = add_watermark(image)
    elif condition.image_perturbation == "brightness_low":
        image = ImageEnhance.Brightness(image).enhance(0.65)
    elif condition.image_perturbation == "contrast_high":
        image = ImageEnhance.Contrast(image).enhance(1.6)

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

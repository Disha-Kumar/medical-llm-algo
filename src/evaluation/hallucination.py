from __future__ import annotations

from dataclasses import dataclass

from src.pipelines.common import CHEXPERT_LABELS


@dataclass(frozen=True)
class HallucinationResult:
    flag: bool
    reason: str


def normalize_label(label: str | None) -> str:
    if not label:
        return ""
    label = label.strip().lower().replace("_", " ")
    aliases = {
        "no findings": "no finding",
        "normal": "no finding",
        "pleural-effusion": "pleural effusion",
        "parse failed": "parse_failed",
    }
    return aliases.get(label, label)


def flag_hallucination(diagnosis: str | None, explanation: str | None, ground_truth: str | None) -> HallucinationResult:
    diagnosis_norm = normalize_label(diagnosis)
    ground_truth_norm = normalize_label(ground_truth)
    explanation_norm = (explanation or "").lower()

    if diagnosis_norm in {"", "parse_failed"}:
        return HallucinationResult(True, "No parseable diagnosis was produced.")

    if ground_truth_norm == "no finding" and diagnosis_norm != "no finding":
        return HallucinationResult(
            True,
            "Model asserted a pathology for a ground-truth no-finding case.",
        )

    unsupported = [
        label
        for label in CHEXPERT_LABELS
        if label != ground_truth_norm and label != diagnosis_norm and label in explanation_norm
    ]
    if unsupported:
        return HallucinationResult(
            True,
            "Explanation mentions unsupported label(s): " + ", ".join(unsupported),
        )

    return HallucinationResult(False, "No label-level hallucination detected.")

import re
from src.pipelines.base import ModelOutput


LABELS_14 = [
    "enlarged cardiomediastinum",
    "cardiomegaly",
    "lung opacity",
    "lung lesion",
    "edema",
    "consolidation",
    "pneumonia",
    "atelectasis",
    "pneumothorax",
    "pleural effusion",
    "pleural other",
    "fracture",
    "support devices",
    "no finding",
]


def parse_chexagent_output(
    raw: str,
    case_id: str,
    ground_truth: str,
) -> ModelOutput:

    raw_clean = " ".join(raw.strip().split())
    raw_lower = raw_clean.lower()

    diagnosis = None

    for label in LABELS_14:
        if label in raw_lower:
            diagnosis = label
            break

    if diagnosis is None:
        fallback_patterns = [
            "no acute",
            "normal",
            "unremarkable",
            "no cardiopulmonary",
            "no focal consolidation",
        ]

        for pattern in fallback_patterns:
            if pattern in raw_lower:
                diagnosis = "no finding"
                break

    if diagnosis is None:
        diagnosis = "no finding"

    confidence = 0.5

    confidence_match = re.search(
        r"confidence[:\\s]+([0-9]*\\.?[0-9]+)",
        raw_lower,
    )

    if confidence_match:
        try:
            confidence = float(confidence_match.group(1))
            confidence = max(0.0, min(1.0, confidence))
        except:
            confidence = 0.5

    return ModelOutput(
        model_name="chexagent",
        case_id=case_id,
        raw_output=raw_clean,
        diagnosis=diagnosis,
        confidence=confidence,
        ground_truth=ground_truth,
    )

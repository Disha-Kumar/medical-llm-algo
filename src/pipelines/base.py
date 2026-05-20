from dataclasses import dataclass
from abc import ABC, abstractmethod
from PIL import Image
import json
import os

from src.pipelines.common import CHEXPERT_LABELS


def _normalize_to_label(raw_diagnosis: str) -> str:
    """
    Snap a free-form diagnosis string to the nearest CheXpert label.

    Strategy (in priority order):
      1. Exact match (case-insensitive).
      2. Substring match: the label appears inside the raw string, or vice versa.
      3. Token overlap: pick the label that shares the most words with the raw string.
      4. Fallback to "no finding".
    """
    normalized = raw_diagnosis.strip().lower()

    for label in CHEXPERT_LABELS:
        if normalized == label:
            return label

    for label in CHEXPERT_LABELS:
        if label in normalized or normalized in label:
            return label

    raw_tokens = set(normalized.split())
    best_label, best_score = "no finding", 0
    for label in CHEXPERT_LABELS:
        label_tokens = set(label.split())
        score = len(raw_tokens & label_tokens)
        if score > best_score:
            best_score, best_label = score, label

    return best_label if best_score > 0 else "no finding"


@dataclass
class ModelOutput:
    model_name: str
    case_id: str
    diagnosis: str
    confidence: float
    explanation: str
    raw_response: str
    ground_truth: str

class MedicalVLM(ABC):
    @abstractmethod
    def predict(self, image: Image.Image, text: str,
                case_id: str, ground_truth: str) -> ModelOutput:
        pass

def parse_output(raw: str, model_name: str,
                 case_id: str, ground_truth: str) -> ModelOutput:
    diagnosis   = ""
    confidence  = 0.0
    explanation = ""

    for line in raw.strip().splitlines():
        line = line.strip()
        if line.startswith("DIAGNOSIS:"):
            diagnosis = line.replace("DIAGNOSIS:", "").strip()
        elif line.startswith("CONFIDENCE:"):
            try:
                val = float(line.replace("CONFIDENCE:", "").strip().replace("%", ""))
                confidence = val / 100.0 if val > 1.0 else val
            except ValueError:
                confidence = 0.0
        elif line.startswith("EXPLANATION:"):
            explanation = line.replace("EXPLANATION:", "").strip()

    if not diagnosis:
        # Fallback: try to find DIAGNOSIS: anywhere in text (not just line start)
        import re
        diag_match = re.search(r'DIAGNOSIS:\s*(.+)', raw, re.IGNORECASE)
        if diag_match:
            diagnosis = diag_match.group(1).strip()
        else:
            # Last resort: scan raw text for any valid CheXpert label
            raw_lower = raw.lower()
            for label in CHEXPERT_LABELS:
                if label in raw_lower and label != "no finding":
                    diagnosis = label
                    break

        if not diagnosis:
            diagnosis   = "parse_failed"
            explanation = raw.strip()
            confidence  = 0.0
        else:
            diagnosis = _normalize_to_label(diagnosis)
            if not explanation:
                explanation = raw.strip()
            if confidence == 0.0:
                # Try to extract a confidence number from the text
                conf_match = re.search(r'CONFIDENCE:\s*([\d.]+)', raw, re.IGNORECASE)
                if conf_match:
                    try:
                        val = float(conf_match.group(1))
                        confidence = val / 100.0 if val > 1.0 else val
                    except ValueError:
                        confidence = 0.5
                else:
                    confidence = 0.5
    else:
        diagnosis = _normalize_to_label(diagnosis)

    return ModelOutput(
        model_name=model_name,
        case_id=case_id,
        diagnosis=diagnosis,
        confidence=min(max(confidence, 0.0), 1.0),
        explanation=explanation if explanation else raw.strip(),
        raw_response=raw,
        ground_truth=ground_truth
    )

def save_output(output: ModelOutput, output_dir: str):
    path = os.path.join(output_dir, output.model_name, f"{output.case_id}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(output.__dict__, f, indent=2)

def load_output(model_name: str, case_id: str, output_dir: str) -> ModelOutput:
    path = os.path.join(output_dir, model_name, f"{case_id}.json")
    with open(path) as f:
        return ModelOutput(**json.load(f))

from dataclasses import dataclass
from abc import ABC, abstractmethod
from PIL import Image
import json
import os

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
        diagnosis   = "parse_failed"
        explanation = raw.strip()
        confidence  = 0.0

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
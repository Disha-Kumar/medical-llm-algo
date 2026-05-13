import os
import tempfile

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from src.pipelines.base import MedicalVLM, ModelOutput
from src.pipelines.common import CHEXPERT_LABELS

try:
    from health_multimodal.image import get_biovil_t_image_encoder
    from health_multimodal.text import get_bert_inference
    from health_multimodal.vlp import ImageTextInferenceEngine
    HAS_HIML = True
except ImportError:
    HAS_HIML = False


LABEL_PROMPTS = {
    "enlarged cardiomediastinum": "There is an enlarged cardiomediastinum on the chest x-ray.",
    "cardiomegaly": "There is cardiomegaly with an enlarged cardiac silhouette.",
    "lung opacity": "There is a lung opacity visible on the chest x-ray.",
    "lung lesion": "There is a lung lesion or mass visible on the chest x-ray.",
    "edema": "There is pulmonary edema.",
    "consolidation": "There is focal airspace consolidation.",
    "pneumonia": "There are findings concerning for pneumonia.",
    "atelectasis": "There is atelectasis on the chest x-ray.",
    "pneumothorax": "There is a pneumothorax.",
    "pleural effusion": "There is a pleural effusion.",
    "pleural other": "There is pleural thickening or other pleural abnormality.",
    "fracture": "There is a rib or bone fracture visible on the chest x-ray.",
    "support devices": "There are support devices such as lines or tubes present.",
    "no finding": "No acute cardiopulmonary abnormality is seen.",
}


class BioViLTPipeline(MedicalVLM):
    MODEL_ID = os.environ.get("BIOVIL_T_MODEL_ID", "microsoft/BiomedVLP-BioViL-T")

    def __init__(self, mode: str = "text_only"):
        self.mode = mode
        self.has_image_encoder = False

        if HAS_HIML:
            print("BioViL-T: loading multimodal pipeline (image + text).", flush=True)
            image_inference = get_biovil_t_image_encoder()
            text_inference = get_bert_inference()
            self.engine = ImageTextInferenceEngine(
                image_inference_engine=image_inference,
                text_inference_engine=text_inference,
            )
            self.has_image_encoder = True
            print("BioViL-T: image encoder loaded via hi-ml-multimodal.", flush=True)
        else:
            print(
                "BioViL-T: hi-ml-multimodal not installed. Falling back to "
                "text-only label ranking. Install with: pip install hi-ml-multimodal",
                flush=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.MODEL_ID,
                trust_remote_code=True,
                token=os.environ.get("HF_TOKEN"),
            )
            self.model = AutoModel.from_pretrained(
                self.MODEL_ID,
                trust_remote_code=True,
                token=os.environ.get("HF_TOKEN"),
            )
            self.model.eval()

        print(f"BioViL-T runtime: model={self.MODEL_ID}, mode={self.mode}, "
              f"image_encoder={self.has_image_encoder}", flush=True)

    def _embed_text(self, texts: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(
            texts,
            add_special_tokens=True,
            padding="longest",
            return_tensors="pt",
        )
        with torch.no_grad():
            embeddings = self.model.get_projected_text_embeddings(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
            )
        return F.normalize(embeddings, dim=-1)

    def _score_with_engine(self, image: Image.Image, label_texts: list[str]) -> torch.Tensor:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image.convert("RGB").save(tmp.name)
            tmp_path = tmp.name
        try:
            scores = []
            for label_text in label_texts:
                sim = self.engine.get_similarity_score_from_raw_data(
                    image_path=tmp_path,
                    query_text=label_text,
                )
                scores.append(float(sim))
            return torch.tensor(scores)
        finally:
            os.unlink(tmp_path)

    def _score_text_only(self, query: str, label_texts: list[str]) -> torch.Tensor:
        embeddings = self._embed_text([query] + label_texts)
        return torch.mv(embeddings[1:], embeddings[0])

    def predict(
        self,
        image: Image.Image,
        text: str,
        case_id: str,
        ground_truth: str,
    ) -> ModelOutput:
        label_texts = [LABEL_PROMPTS[label] for label in CHEXPERT_LABELS]
        query = text.strip() or "Chest x-ray with no clinical note."

        if self.mode == "text_only" or (self.mode != "text_only" and not self.has_image_encoder):
            scores = self._score_text_only(query, label_texts)
            method = "text_embedding_only"
        elif self.mode == "image_only":
            scores = self._score_with_engine(image, label_texts)
            method = "image_text_engine_image_only"
        else:
            image_scores = self._score_with_engine(image, label_texts)
            text_scores = self._score_text_only(query, label_texts)
            image_scores = (image_scores - image_scores.mean()) / (image_scores.std() + 1e-8)
            text_scores = (text_scores - text_scores.mean()) / (text_scores.std() + 1e-8)
            scores = 0.6 * image_scores + 0.4 * text_scores
            method = "image_text_combined"

        best_index = int(torch.argmax(scores).item())
        best_label = CHEXPERT_LABELS[best_index]
        confidence = float(torch.softmax(scores, dim=0)[best_index].item())
        ranked = [
            {"label": CHEXPERT_LABELS[i], "score": float(scores[i].item())}
            for i in torch.argsort(scores, descending=True).tolist()
        ]
        raw = (
            f"BioViL-T {self.mode} label ranking (method={method}). "
            f"Ranked labels: {ranked}."
        )
        explanation = (
            f"BioViL-T ranked CheXpert label prompts using {method}. "
            f"Top label: {best_label} (confidence: {confidence:.3f})."
        )
        return ModelOutput(
            model_name="biovil_t",
            case_id=case_id,
            diagnosis=best_label,
            confidence=confidence,
            explanation=explanation,
            raw_response=raw,
            ground_truth=ground_truth,
        )

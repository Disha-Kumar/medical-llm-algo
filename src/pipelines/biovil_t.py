import os

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from src.pipelines.base import MedicalVLM, ModelOutput
from src.pipelines.common import CHEXPERT_LABELS


LABEL_PROMPTS = {
    "atelectasis": "There is atelectasis on the chest x-ray.",
    "cardiomegaly": "There is cardiomegaly with an enlarged cardiac silhouette.",
    "consolidation": "There is focal airspace consolidation.",
    "edema": "There is pulmonary edema.",
    "pleural effusion": "There is a pleural effusion.",
    "pneumonia": "There are findings concerning for pneumonia.",
    "pneumothorax": "There is a pneumothorax.",
    "no finding": "No acute cardiopulmonary abnormality is seen.",
}


class BioViLTPipeline(MedicalVLM):
    MODEL_ID = os.environ.get("BIOVIL_T_MODEL_ID", "microsoft/BiomedVLP-BioViL-T")

    def __init__(self, mode: str = "text_only"):
        self.mode = mode
        print(
            "BioViL-T note: this repo pipeline is a CheXpert label-ranking "
            "baseline using BioViL-T radiology text embeddings. It accepts the "
            "same image+text interface as the VLM pipelines, but the current "
            "Hugging Face wrapper does not use image pixels. True BioViL-T "
            "image+text inference requires Microsoft's HI-ML multimodal image encoder.",
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
        print(f"BioViL-T runtime: model={self.MODEL_ID}, mode={self.mode}", flush=True)

    def _embed(self, texts: list[str]) -> torch.Tensor:
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

    def predict(
        self,
        image: Image.Image,
        text: str,
        case_id: str,
        ground_truth: str,
    ) -> ModelOutput:
        if self.mode == "image_only":
            query = "Chest x-ray image only. No clinical note is provided."
            evidence_note = (
                "Image-only baseline requested, but this BioViL-T wrapper cannot "
                "consume image pixels without the external HI-ML image encoder."
            )
        elif self.mode == "text_only":
            query = text.strip() or "Chest x-ray with no clinical note."
            evidence_note = "Text-only baseline using the CheXpert clinical note."
        else:
            query = text.strip() or "Chest x-ray with no clinical note."
            evidence_note = (
                "Image+text interface accepted. This baseline ranks labels from "
                "the text side only; image pixels are not used by this wrapper."
            )

        label_texts = [LABEL_PROMPTS[label] for label in CHEXPERT_LABELS]
        embeddings = self._embed([query] + label_texts)
        scores = torch.mv(embeddings[1:], embeddings[0])
        best_index = int(torch.argmax(scores).item())
        best_label = CHEXPERT_LABELS[best_index]
        confidence = float(torch.softmax(scores, dim=0)[best_index].item())
        ranked = [
            {"label": CHEXPERT_LABELS[i], "score": float(scores[i].item())}
            for i in torch.argsort(scores, descending=True).tolist()
        ]
        raw = (
            f"BioViL-T {self.mode} label ranking. "
            f"Ranked labels: {ranked}. "
            f"{evidence_note} "
            "This is not a generative diagnosis explanation."
        )
        explanation = (
            f"{evidence_note} BioViL-T ranked CheXpert label prompts against "
            "the available text in its radiology embedding space."
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

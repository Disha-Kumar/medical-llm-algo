import os
import tempfile
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.pipelines.base import MedicalVLM, ModelOutput, parse_output


class CheXAgentPipeline(MedicalVLM):
    MODEL_ID = os.environ.get("CHEXAGENT_MODEL_ID", "StanfordAIMI/CheXagent-2-3b")

    def __init__(self, device: str = "auto", mode: str = "image_text"):
        self.mode = mode
        self.dtype = torch.bfloat16
        self.max_new_tokens = int(os.environ.get("CHEXAGENT_MAX_NEW_TOKENS", "200"))

        print(
            f"CheXagent runtime: model={self.MODEL_ID}, mode={self.mode}, "
            f"max_new_tokens={self.max_new_tokens}",
            flush=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_ID,
            trust_remote_code=True,
            token=os.environ.get("HF_TOKEN"),
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=self.dtype,
            token=os.environ.get("HF_TOKEN"),
        )
        self.model.eval()
        print("CheXagent loaded.", flush=True)

    def _make_prompt(self, clinical_note: str) -> str:
        note = clinical_note.strip()[:800] or "No clinical note provided."
        if self.mode == "image_only":
            note = "No clinical note available. Use image findings only."
        elif self.mode == "text_only":
            note = f"No image available. Use clinical note only. {note}"
        return (
            f"You are an expert radiologist analyzing a chest X-ray for research.\n"
            f"{note}\n\n"
            f"Choose exactly one diagnosis from this list:\n"
            f"enlarged cardiomediastinum, cardiomegaly, lung opacity, lung lesion, "
            f"edema, consolidation, pneumonia, atelectasis, pneumothorax, "
            f"pleural effusion, pleural other, fracture, support devices, no finding\n\n"
            f"ONLY write one exact label. Do not describe findings in the DIAGNOSIS line.\n\n"
            f"Respond in exactly this format:\n"
            f"DIAGNOSIS: <one exact label from the list>\n"
            f"CONFIDENCE: <number between 0.0 and 1.0>\n"
            f"EXPLANATION: <one to two sentences citing specific findings>\n"
        )

    def predict(self, image: Image.Image, text: str,
                case_id: str, ground_truth: str) -> ModelOutput:
        prompt = self._make_prompt(text)

        if self.mode == "text_only":
            query = self.tokenizer.from_list_format([
                {"text": prompt}
            ])
        else:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                image.convert("RGB").save(tmp.name)
                tmp_path = tmp.name
            query = self.tokenizer.from_list_format([
                {"image": tmp_path},
                {"text": prompt},
            ])

        inputs = self.tokenizer(query, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            print("  Generating...", flush=True)
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
            print("  Generation complete.", flush=True)

        raw = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        return parse_output(raw, "chexagent", case_id, ground_truth)

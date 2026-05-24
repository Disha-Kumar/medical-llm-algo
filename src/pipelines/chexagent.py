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
        dtype_name = os.environ.get("CHEXAGENT_DTYPE", "float32").lower()
        self.dtype = torch.bfloat16 if dtype_name in {"bf16", "bfloat16"} else torch.float32
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
        note = clinical_note.strip()[:500] or "No clinical note provided."
        if self.mode == "image_only":
            return (
                "Describe the chest X-ray in one sentence. "
                "State the main radiographic finding, or say no acute cardiopulmonary process."
            )
        elif self.mode == "text_only":
            return (
                "Based on the clinical note only, state the most likely CheXpert finding "
                f"in one sentence.\nClinical note: {note}"
            )
        return (
            "Describe the chest X-ray in one sentence. "
            "State the main radiographic finding, or say no acute cardiopulmonary process.\n"
            f"Clinical note: {note}"
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

        conv = [
            {"from": "system", "value": "You are a helpful assistant."},
            {"from": "human", "value": query},
        ]
        device = next(self.model.parameters()).device
        input_ids = self.tokenizer.apply_chat_template(
            conv,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            print("  Generating...", flush=True)
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                min_new_tokens=10,
                do_sample=False,
                num_beams=1,
                temperature=1.0,
                top_p=1.0,
                use_cache=True,
            )
            print("  Generation complete.", flush=True)

        raw = self.tokenizer.decode(
            output_ids[0][input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()

        return parse_output(raw, "chexagent", case_id, ground_truth)

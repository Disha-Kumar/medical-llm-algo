import os
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from src.pipelines.base import MedicalVLM, ModelOutput, parse_output
from src.pipelines.common import make_prompt
from src.pipelines.common import CHEXPERT_LABELS


class LLaVAMedPipeline(MedicalVLM):
    MODEL_ID = os.environ.get(
        "LLAVAMED_MODEL_ID",
        "chaoyinshe/llava-med-v1.5-mistral-7b-hf",
    )

    def __init__(self, device: str = "cpu", mode: str = "image_text"):
        self.device = self._resolve_device(device)
        self.dtype = self._resolve_dtype(self.device)
        self.mode = mode
        self.max_new_tokens = int(os.environ.get("LLAVAMED_MAX_NEW_TOKENS", "96"))
        print(
            f"LLaVA-Med runtime: device={self.device}, "
            f"dtype={self.dtype}, mode={self.mode}, max_new_tokens={self.max_new_tokens}",
            flush=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.MODEL_ID,
            token=os.environ.get("HF_TOKEN")
        )
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            token=os.environ.get("HF_TOKEN")
        )
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _resolve_dtype(device: str) -> torch.dtype:
        if device == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if device in {"cuda", "mps"}:
            return torch.float16
        return torch.float32

    def _to_device(self, inputs):
        moved = {}
        for key, value in inputs.items():
            if not torch.is_tensor(value):
                moved[key] = value
                continue
            value = value.to(self.device)
            if torch.is_floating_point(value):
                value = value.to(self.dtype)
            moved[key] = value
        return moved

    def _format_prompt(self, prompt: str) -> str:
        if self.mode == "text_only":
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

        if hasattr(self.processor, "apply_chat_template"):
            try:
                return self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except ValueError:
                pass

        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        chat_template = getattr(self.processor, "chat_template", None)
        if hasattr(tokenizer, "apply_chat_template") and chat_template:
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    chat_template=chat_template,
                )
            except ValueError:
                pass

        if self.mode == "text_only":
            return f"[INST] {prompt} [/INST]"
        return f"[INST] <image>\n{prompt} [/INST]"

    def _make_llavamed_prompt(self, clinical_note: str) -> str:
        labels = ", ".join(CHEXPERT_LABELS)
        note = clinical_note.strip() or "No clinical note provided."
        if self.mode == "image_only":
            note = "Use image only. No clinical note."
        elif self.mode == "text_only":
            note = f"Use text only: {note}"
        else:
            note = f"Clinical note: {note}"

        return (
            f"You are a radiologist analyzing a chest X-ray.\n"
            f"{note}\n\n"
            f"Select exactly one diagnosis from this list: {labels}.\n\n"
            f"Respond in exactly this format and nothing else:\n"
            f"DIAGNOSIS: <label>\n"
            f"CONFIDENCE: <number between 0.0 and 1.0>\n"
            f"EXPLANATION: <one sentence citing specific image findings>\n"
        )

    def predict(self, image: Image.Image, text: str,
                case_id: str, ground_truth: str) -> ModelOutput:
        prompt = self._make_llavamed_prompt(text)
        full_prompt = self._format_prompt(prompt)

        if self.mode == "text_only":
            inputs = self.processor(text=full_prompt, return_tensors="pt")
        else:
            inputs = self.processor(
                images=[image],
                text=full_prompt,
                return_tensors="pt",
            )
        inputs = self._to_device(inputs)

        with torch.no_grad():
            print("  Generating...", flush=True)
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False
            )
            print("  Generation complete.", flush=True)
        raw = self.processor.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        return parse_output(raw, "llava-med-v1.5-mistral-7b", case_id, ground_truth)

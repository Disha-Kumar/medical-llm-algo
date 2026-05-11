import os

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from src.pipelines.base import MedicalVLM, ModelOutput, parse_output
from src.pipelines.common import make_prompt, move_inputs, resolve_device, resolve_dtype


class Qwen2VLPipeline(MedicalVLM):
    MODEL_ID = os.environ.get("QWEN2_VL_MODEL_ID", "Qwen/Qwen2-VL-2B-Instruct")

    def __init__(self, device: str = "auto", mode: str = "image_text"):
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(self.device)
        self.mode = mode
        self.max_new_tokens = int(os.environ.get("QWEN2_VL_MAX_NEW_TOKENS", "96"))
        print(
            f"Qwen2-VL runtime: model={self.MODEL_ID}, device={self.device}, "
            f"dtype={self.dtype}, mode={self.mode}, max_new_tokens={self.max_new_tokens}",
            flush=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.MODEL_ID,
            token=os.environ.get("HF_TOKEN"),
        )
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            token=os.environ.get("HF_TOKEN"),
        ).to(self.device)
        self.model.eval()

    def _make_qwen_prompt(self, clinical_note: str) -> str:
        labels = (
            "atelectasis, cardiomegaly, consolidation, edema, "
            "pleural effusion, pneumonia, pneumothorax, no finding"
        )
        note = clinical_note.strip() or "No clinical note provided."
        if self.mode == "image_only":
            context = "No clinical note is available. Examine the chest X-ray image carefully."
        elif self.mode == "text_only":
            context = f"No image is available. Use this clinical note only: {note}"
        else:
            context = f"Clinical note: {note}"
        return (
            f"You are an expert radiologist. Examine the chest X-ray and provide a diagnosis.\n"
            f"{context}\n\n"
            f"Choose exactly one diagnosis from: {labels}.\n"
            f"Look carefully at the image before deciding. Different cases have different findings.\n\n"
            f"Respond in exactly this format:\n"
            f"DIAGNOSIS: <label>\n"
            f"CONFIDENCE: <number between 0.0 and 1.0>\n"
            f"EXPLANATION: <one to two sentences citing specific visual findings>\n"
        )

    def predict(
        self,
        image: Image.Image,
        text: str,
        case_id: str,
        ground_truth: str,
    ) -> ModelOutput:
        prompt = self._make_qwen_prompt(text)
        content = []
        images = None
        if self.mode != "text_only":
            content.append({"type": "image"})
            images = [image.convert("RGB")]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        formatted = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        kwargs = {"text": [formatted], "return_tensors": "pt"}
        if images is not None:
            kwargs["images"] = images
        inputs = self.processor(**kwargs)
        inputs = move_inputs(inputs, self.device, self.dtype)
        with torch.no_grad():
            print("  Generating...", flush=True)
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
            print("  Generation complete.", flush=True)
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        raw = self.processor.decode(generated_ids, skip_special_tokens=True)
        return parse_output(raw, "qwen2_vl", case_id, ground_truth)

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
        self.max_new_tokens = int(os.environ.get("LLAVAMED_MAX_NEW_TOKENS", "256"))
        self.prompt_style = os.environ.get("LLAVAMED_PROMPT_STYLE", "default").lower()
        self.do_sample = os.environ.get("LLAVAMED_DO_SAMPLE", "1") != "0"
        print(
            f"LLaVA-Med runtime: device={self.device}, "
            f"dtype={self.dtype}, mode={self.mode}, max_new_tokens={self.max_new_tokens}, "
            f"prompt_style={self.prompt_style}, do_sample={self.do_sample}",
            flush=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.MODEL_ID,
            token=os.environ.get("HF_TOKEN"),
        )
        self.processor.patch_size = 14
        self.processor.vision_feature_select_strategy = "default"

        model_kwargs = {
            "torch_dtype": self.dtype,
            "low_cpu_mem_usage": True,
            "token": os.environ.get("HF_TOKEN"),
        }
        use_8bit = os.environ.get("LLAVAMED_LOAD_IN_8BIT", "1") == "1"
        if use_8bit and self.device == "cuda":
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = "auto"

        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            **model_kwargs,
        )
        if "device_map" not in model_kwargs:
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
        try:
            target_device = next(self.model.parameters()).device
        except StopIteration:
            target_device = self.device
        moved = {}
        for key, value in inputs.items():
            if not torch.is_tensor(value):
                moved[key] = value
                continue
            value = value.to(target_device)
            if torch.is_floating_point(value):
                value = value.to(self.dtype)
            moved[key] = value
        return moved

    def _format_prompt(self, prompt: str) -> str:
        if self.mode == "text_only":
            return f"USER: {prompt}\nASSISTANT:"
        return f"USER: <image>\n{prompt}\nASSISTANT:"

    def _make_llavamed_prompt(self, clinical_note: str) -> str:

        note = clinical_note.strip()[:800] or "No clinical note provided."
        if self.mode == "image_only":
            note = "No clinical note is provided. Use the image only."
        elif self.mode == "text_only":
            note = f"Use the clinical note only. Clinical Note: {note}"
        else:
            note = f"Clinical Note: {note}"

        if self.prompt_style == "simple":
            return (
                "Chest X-ray research task. Choose one label only from: "
                "enlarged cardiomediastinum, cardiomegaly, lung opacity, lung lesion, "
                "edema, consolidation, pneumonia, atelectasis, pneumothorax, "
                "pleural effusion, pleural other, fracture, support devices, no finding.\n\n"
                "Answer exactly in this format:\n"
                "DIAGNOSIS: <label>\n"
                "CONFIDENCE: <0.0 to 1.0>\n"
                "EXPLANATION: <short evidence sentence>\n\n"
                f"{note}"
            )

        return (
            "Analyze the provided chest X-ray information for a research-only CheXpert task. "
            "Return exactly three lines in the format below. Do not explain the task. "
            "Do not define the fields. Do not add any text before or after the three lines.\n\n"
            "DIAGNOSIS: <single label from: enlarged cardiomediastinum, cardiomegaly, "
            "lung opacity, lung lesion, edema, consolidation, pneumonia, atelectasis, "
            "pneumothorax, pleural effusion, pleural other, fracture, support devices, no finding>\n"
            "CONFIDENCE: <float between 0.0 and 1.0>\n"
            "EXPLANATION: <one to three sentences describing the visual evidence>\n\n"
            "Choose the best label from the list, even if uncertain.\n\n"
            f"{note}"
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
            generation_kwargs = {
                **inputs,
                "max_new_tokens": self.max_new_tokens,
                "do_sample": self.do_sample,
            }
            if self.do_sample:
                generation_kwargs["temperature"] = 0.3
                generation_kwargs["top_p"] = 0.9
            output_ids = self.model.generate(**generation_kwargs)
            print("  Generation complete.", flush=True)
        raw = self.processor.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        return parse_output(raw, "llava-med-v1.5-mistral-7b", case_id, ground_truth)

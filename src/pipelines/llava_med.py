import os
import re
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from src.pipelines.base import MedicalVLM, ModelOutput, parse_output
from src.pipelines.common import CHEXPERT_LABELS


# Labels as a compact string for the prompt
_LABEL_LIST = " | ".join(CHEXPERT_LABELS)

# One-shot example so the 7B model sees the exact output shape before generating
_ONESHOT = (
    "Example:\n"
    "DIAGNOSIS: cardiomegaly\n"
    "CONFIDENCE: 0.8\n"
    "EXPLANATION: The cardiac silhouette is enlarged.\n"
)


class LLaVAMedPipeline(MedicalVLM):
    MODEL_ID = os.environ.get(
        "LLAVAMED_MODEL_ID",
        "chaoyinshe/llava-med-v1.5-mistral-7b-hf",
    )

    # ── prefill token: the model's first generated token comes AFTER this ──
    # Ending the prompt with "DIAGNOSIS:" forces the model to continue in
    # structured format instead of rambling in free-form prose.
    _RESPONSE_PREFILL = " DIAGNOSIS:"

    def __init__(self, device: str = "cpu", mode: str = "image_text"):
        self.device = self._resolve_device(device)
        self.dtype = self._resolve_dtype(self.device)
        self.mode = mode
        self.max_new_tokens = int(os.environ.get("LLAVAMED_MAX_NEW_TOKENS", "200"))
        print(
            f"LLaVA-Med runtime: device={self.device}, "
            f"dtype={self.dtype}, mode={self.mode}, "
            f"max_new_tokens={self.max_new_tokens}",
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
        use_8bit = os.environ.get("LLAVAMED_LOAD_IN_8BIT", "0") == "1"
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

    # ─── device / dtype helpers ───────────────────────────────────────

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

    # ─── prompt construction ──────────────────────────────────────────

    def _build_prompt(self, clinical_note: str) -> str:
        """Short, direct prompt with one-shot example.

        7B models handle short prompts far better than multi-paragraph
        instructions.  The one-shot example shows the exact format the
        model should follow, and the prefill in _format_prompt() forces
        the first token to continue in that format.
        """
        note = clinical_note.strip()[:800] or "No clinical note provided."

        if self.mode == "image_only":
            context = "Use the chest X-ray image only."
        elif self.mode == "text_only":
            context = f"Clinical Note: {note}"
        else:
            context = f"Clinical Note: {note}"

        return (
            f"Classify this chest X-ray. Pick ONE label from:\n"
            f"{_LABEL_LIST}\n\n"
            f"{_ONESHOT}\n"
            f"{context}\n\n"
            f"Respond in the same format as the example above."
        )

    def _format_prompt(self, prompt: str) -> str:
        """Build the LLaVA conversation with response prefill.

        By ending with 'ASSISTANT: DIAGNOSIS:' the model's generation
        starts mid-format, making it far more likely to continue with
        the label instead of producing free-form prose.
        """
        if self.mode == "text_only":
            return f"USER: {prompt}\nASSISTANT:{self._RESPONSE_PREFILL}"
        return f"USER: <image>\n{prompt}\nASSISTANT:{self._RESPONSE_PREFILL}"

    # ─── post-processing ──────────────────────────────────────────────

    @staticmethod
    def _clean_raw_output(raw: str) -> str:
        """Reconstruct the full structured output.

        Since the prompt prefill already provided 'DIAGNOSIS:', prepend
        it so parse_output sees a complete structured block.  Also trim
        anything after a double newline (the model often appends
        disclaimers or repeats the prompt after the answer).
        """
        # Prepend DIAGNOSIS: since it was in the prefill, not in the output
        text = "DIAGNOSIS:" + raw

        # Cut off trailing noise after the three-line answer
        # Look for the end of the EXPLANATION line and stop there
        lines = text.split("\n")
        kept = []
        found_explanation = False
        for line in lines:
            kept.append(line)
            if line.strip().upper().startswith("EXPLANATION:"):
                found_explanation = True
                break
        if found_explanation:
            text = "\n".join(kept)

        return text.strip()

    # ─── inference ────────────────────────────────────────────────────

    def predict(self, image: Image.Image, text: str,
                case_id: str, ground_truth: str) -> ModelOutput:
        prompt = self._build_prompt(text)
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
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.15,
            )
            print("  Generation complete.", flush=True)

        raw = self.processor.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        cleaned = self._clean_raw_output(raw)
        return parse_output(cleaned, "llava-med-v1.5-mistral-7b",
                            case_id, ground_truth)

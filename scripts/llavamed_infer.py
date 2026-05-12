#!/usr/bin/env python3
"""Minimal LLaVA-Med image+text inference smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration


DEFAULT_MODEL = "chaoyinshe/llava-med-v1.5-mistral-7b-hf"
DEFAULT_PROMPT = (
    "Analyze the provided chest X-ray image for a research-only CheXpert task. "
    "Return exactly three lines in the format below. Do not explain the task. "
    "Do not define the fields. Do not add any text before or after the three lines.\n\n"
    "DIAGNOSIS:  <single label from: enlarged cardiomediastinum, cardiomegaly, "
    "lung opacity, lung lesion, edema, consolidation, pneumonia, atelectasis, "
    "CONFIDENCE: <float between 0.0 and 1.0>\n"
    "EXPLANATION: <one to three sentences describing the visual evidence>\n\n"
    "Choose the best label from the list, even if uncertain."
)


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def choose_dtype(device: str) -> torch.dtype:
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


def move_inputs(inputs, device: str, dtype: torch.dtype):
    moved = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            moved[key] = value
            continue
        value = value.to(device)
        if torch.is_floating_point(value):
            value = value.to(dtype)
        moved[key] = value
    return moved


def format_prompt(processor, prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    if hasattr(processor, "apply_chat_template"):
        try:
            return processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except ValueError:
            pass

    tokenizer = getattr(processor, "tokenizer", processor)
    chat_template = getattr(processor, "chat_template", None)
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

    return f"[INST] <image>\n{prompt} [/INST]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to a CheXpert image file.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    device = choose_device()
    dtype = choose_dtype(device)
    print(f"Loading {args.model_id}")
    print(f"Device={device} dtype={dtype}")

    image = Image.open(image_path).convert("RGB")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    text = format_prompt(processor, args.prompt)
    inputs = processor(images=[image], text=text, return_tensors="pt")
    inputs = move_inputs(inputs, device, dtype)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    decoded = processor.decode(generated_ids, skip_special_tokens=True)
    print("\n--- MODEL OUTPUT ---")
    print(decoded)


if __name__ == "__main__":
    main()

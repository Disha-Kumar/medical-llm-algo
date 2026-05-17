import base64
import mimetypes
from io import BytesIO

import torch
from PIL import Image


CHEXPERT_LABELS = [
    "enlarged cardiomediastinum",
    "cardiomegaly",
    "lung opacity",
    "lung lesion",
    "edema",
    "consolidation",
    "pneumonia",
    "atelectasis",
    "pneumothorax",
    "pleural effusion",
    "pleural other",
    "fracture",
    "support devices",
    "no finding",
]

DIAGNOSIS_PROMPT = (
    "You are evaluating a chest X-ray for a research study. "
    "You MUST respond in exactly this format with no other text:\n\n"
    "DIAGNOSIS: <one label, exactly as written below>\n"
    "CONFIDENCE: <float 0.0-1.0>\n"
    "EXPLANATION: <one sentence>\n\n"
    "The ONLY valid labels are:\n"
    "enlarged cardiomediastinum | cardiomegaly | lung opacity | lung lesion | "
    "edema | consolidation | pneumonia | atelectasis | pneumothorax | "
    "pleural effusion | pleural other | fracture | support devices | no finding\n\n"
    "Copy the label exactly. Do not add words. Do not combine labels. "
    "If uncertain, pick the single most likely label from the list above."
)

def make_prompt(clinical_note: str = "", mode: str = "image_text") -> str:
    note = clinical_note.strip() or "No clinical note provided."
    if mode == "image_only":
        note = "No clinical note provided. Use the image only."
    elif mode == "text_only":
        note = f"Use the clinical note only. Clinical Note: {note}"
    else:
        note = f"Clinical Note: {note}"
    return f"{DIAGNOSIS_PROMPT}\n\n{note}"


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(device: str) -> torch.dtype:
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


def image_to_data_url(image: Image.Image, image_path: str | None = None) -> str:
    mime_type = "image/jpeg"
    if image_path:
        mime_type = mimetypes.guess_type(image_path)[0] or mime_type

    buffer = BytesIO()
    fmt = "PNG" if mime_type == "image/png" else "JPEG"
    image.convert("RGB").save(buffer, format=fmt)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"

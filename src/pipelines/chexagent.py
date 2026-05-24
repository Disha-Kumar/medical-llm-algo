import os
import tempfile
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.pipelines.base import MedicalVLM, ModelOutput, parse_output


LABELS_14 = [
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


class CheXAgentPipeline(MedicalVLM):
    MODEL_ID = os.environ.get(
        "CHEXAGENT_MODEL_ID",
        "StanfordAIMI/CheXagent-2-3b",
    )

    def __init__(self, device: str = "auto", mode: str = "image_text"):
        self.mode = mode

        dtype_name = os.environ.get(
            "CHEXAGENT_DTYPE",
            "bfloat16",
        ).lower()

        if dtype_name in {"bf16", "bfloat16"}:
            self.dtype = torch.bfloat16
        elif dtype_name in {"fp16", "float16"}:
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        self.max_new_tokens = int(
            os.environ.get(
                "CHEXAGENT_MAX_NEW_TOKENS",
                "64",
            )
        )

        print(
            f"CheXagent runtime: "
            f"model={self.MODEL_ID}, "
            f"mode={self.mode}, "
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
        note = clinical_note.strip()[:600]

        if not note:
            note = "No clinical note provided."

        if self.mode == "image_only":
            note = "No clinical note available. Use image findings only."

        elif self.mode == "text_only":
            note = f"No image available. {note}"

        return (
            "Chest X-ray analysis task.\n"
            f"{note}\n\n"
            "You MUST respond in EXACTLY this format:\n"
            "DIAGNOSIS: <one label>\n"
            "CONFIDENCE: <0.0 to 1.0>\n"
            "EXPLANATION: <one sentence>\n\n"
            "Valid labels: enlarged cardiomediastinum, "
            "cardiomegaly, lung opacity, lung lesion, edema, "
            "consolidation, pneumonia, atelectasis, "
            "pneumothorax, pleural effusion, pleural other, "
            "fracture, support devices, no finding\n\n"
            "Example:\n"
            "DIAGNOSIS: cardiomegaly\n"
            "CONFIDENCE: 0.9\n"
            "EXPLANATION: Cardiac silhouette is enlarged "
            "beyond normal limits.\n"
        )

    def predict(
        self,
        image: Image.Image,
        text: str,
        case_id: str,
        ground_truth: str,
    ) -> ModelOutput:

        prompt = self._make_prompt(text)

        tmp_path = None

        try:
            if self.mode == "text_only":
                query = self.tokenizer.from_list_format([
                    {"text": prompt}
                ])

            else:
                with tempfile.NamedTemporaryFile(
                    suffix=".png",
                    delete=False,
                ) as tmp:

                    image.convert("RGB").resize(
                        (512, 512)
                    ).save(
                        tmp.name,
                        format="PNG",
                    )

                    tmp_path = tmp.name

                query = self.tokenizer.from_list_format([
                    {"image": tmp_path},
                    {"text": prompt},
                ])

            conv = [
                {
                    "from": "system",
                    "value": "You are a helpful radiology assistant.",
                },
                {
                    "from": "human",
                    "value": query,
                },
            ]

            device = next(
                self.model.parameters()
            ).device

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
                    min_new_tokens=1,
                    do_sample=False,
                    num_beams=1,
                    temperature=None,
                    top_p=None,
                    repetition_penalty=1.0,
                    use_cache=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

                print(
                    "  Generation complete.",
                    flush=True,
                )

            raw = self.tokenizer.decode(
                output_ids[0][input_ids.shape[1]:],
                skip_special_tokens=True,
            ).strip()

            raw = " ".join(raw.split())

            print(
                f"  RAW: {repr(raw[:300])}",
                flush=True,
            )

            if "DIAGNOSIS:" not in raw:
                raw_lower = raw.lower()

                found = False

                for label in LABELS_14:
                    if label in raw_lower:
                        raw = (
                            f"DIAGNOSIS: {label}\n"
                            f"CONFIDENCE: 0.5\n"
                            f"EXPLANATION: "
                            f"Label extracted from model output.\n"
                        )

                        found = True
                        break

                if not found:
                    raw = (
                        "DIAGNOSIS: no finding\n"
                        "CONFIDENCE: 0.0\n"
                        "EXPLANATION: Failed to extract label.\n"
                    )

            return parse_output(
                raw,
                "chexagent",
                case_id,
                ground_truth,
            )

        finally:
            if (
                tmp_path is not None
                and os.path.exists(tmp_path)
            ):
                os.remove(tmp_path)

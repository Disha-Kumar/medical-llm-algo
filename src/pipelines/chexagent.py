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

    MODEL_ID = os.environ.get("CHEXAGENT_MODEL_ID", "StanfordAIMI/CheXagent-2-3b")

    def __init__(self, device: str = "auto", mode: str = "image_text"):
        self.mode = mode

        dtype_name = os.environ.get("CHEXAGENT_DTYPE", "float16").lower()
        if dtype_name in {"bf16", "bfloat16"}:
            self.dtype = torch.bfloat16
        elif dtype_name in {"fp16", "float16"}:
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        self.max_new_tokens = int(os.environ.get("CHEXAGENT_MAX_NEW_TOKENS", "64"))

        print(
            f"CheXagent runtime: model={self.MODEL_ID}, mode={self.mode}, "
            f"dtype={self.dtype}, max_new_tokens={self.max_new_tokens}",
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

        self.model = self.model.to(self.dtype)
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
            "Chest X-ray classification task.\n\n"
            f"Clinical note:\n{note}\n\n"
            "Choose the SINGLE best matching CheXpert label.\n\n"
            "Valid labels:\n"
            "- enlarged cardiomediastinum\n"
            "- cardiomegaly\n"
            "- lung opacity\n"
            "- lung lesion\n"
            "- edema\n"
            "- consolidation\n"
            "- pneumonia\n"
            "- atelectasis\n"
            "- pneumothorax\n"
            "- pleural effusion\n"
            "- pleural other\n"
            "- fracture\n"
            "- support devices\n"
            "- no finding\n\n"
            "Return ONLY the label."
        )

    def _compute_confidence(self, scores) -> float:
        if not scores:
            return 0.5
        probs = []
        for score_tensor in scores:
            probs_tensor = torch.softmax(score_tensor[0], dim=-1)
            max_prob = probs_tensor.max().item()
            probs.append(max_prob)
        if not probs:
            return 0.5
        confidence = sum(probs) / len(probs)
        confidence = max(0.0, min(1.0, confidence))
        confidence = max(0.35, confidence)
        return round(confidence, 3)

    def _normalize_output(self, raw: str, confidence: float) -> str:
        raw_lower = raw.lower()

        best_label = "no finding"
        best_pos = len(raw_lower)

        for label in LABELS_14:
            pos = raw_lower.find(label)
            if pos != -1 and pos < best_pos:
                best_pos = pos
                best_label = label

        return (
            f"DIAGNOSIS: {best_label}\n"
            f"CONFIDENCE: {confidence}\n"
            f"EXPLANATION: Auto-normalized from model output.\n"
        )

    def predict(self, image: Image.Image, text: str,
                case_id: str, ground_truth: str) -> ModelOutput:
        prompt = self._make_prompt(text)
        tmp_path = None

        try:
            if self.mode == "text_only":
                query = self.tokenizer.from_list_format([{"text": prompt}])
            else:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    image.convert("RGB").resize((512, 512)).save(tmp.name, format="PNG")
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
                conv, add_generation_prompt=True, return_tensors="pt"
            )
            inputs = input_ids.to(device)
            attention_mask = torch.ones_like(inputs, dtype=torch.long, device=device)

            with torch.no_grad():
                print("  Generating...", flush=True)
                generation = self.model.generate(
                    inputs,
                    attention_mask=attention_mask,
                    do_sample=False,
                    num_beams=1,
                    temperature=1.0,
                    top_p=1.0,
                    use_cache=True,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                print("  Generation complete.", flush=True)

            output = generation.sequences[0]
            scores = generation.scores

            generated_tokens = [
                t for t in output[input_ids.size(1):].tolist()
                if isinstance(t, int) and 0 <= t < self.tokenizer.vocab_size
            ]

            raw = self.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            ).strip()
            raw = " ".join(raw.split())

            print(f"  RAW: {repr(raw[:300])}", flush=True)

            confidence = self._compute_confidence(scores)
            normalized = self._normalize_output(raw, confidence)

            return parse_output(normalized, "chexagent", case_id, ground_truth)

        except Exception as e:
            print(f"  ERROR: {repr(e)}", flush=True)
            fallback = (
                "DIAGNOSIS: no finding\n"
                "CONFIDENCE: 0.5\n"
                "EXPLANATION: Exception fallback."
            )
            return parse_output(fallback, "chexagent", case_id, ground_truth)

        finally:
            if tmp_path is not None and os.path.exists(tmp_path):
                os.remove(tmp_path)
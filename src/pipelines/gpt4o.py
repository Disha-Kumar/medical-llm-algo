import os

from PIL import Image

from src.pipelines.base import MedicalVLM, ModelOutput, parse_output
from src.pipelines.common import image_to_data_url, make_prompt


class GPT4oPipeline(MedicalVLM):
    MODEL_ID = os.environ.get("GPT4O_MODEL_ID", "openai/gpt-4o")

    def __init__(self, mode: str = "image_text"):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY before running GPT-4o.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the OpenAI SDK with: pip install openai") from exc

        base_url = os.environ.get("OPENAI_BASE_URL", None)
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)
        self.mode = mode
        self.max_output_tokens = int(os.environ.get("GPT4O_MAX_OUTPUT_TOKENS", "160"))
        print(
            f"GPT-4o runtime: model={self.MODEL_ID}, mode={self.mode}, "
            f"max_output_tokens={self.max_output_tokens}"
            + (f", base_url={base_url}" if base_url else ""),
            flush=True,
        )

    def predict(
        self,
        image: Image.Image,
        text: str,
        case_id: str,
        ground_truth: str,
    ) -> ModelOutput:
        messages_content = []
        if self.mode != "text_only":
            messages_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(image), "detail": "low"},
                }
            )
        messages_content.append({"type": "text", "text": make_prompt(text, self.mode)})

        response = self.client.chat.completions.create(
            model=self.MODEL_ID,
            messages=[{"role": "user", "content": messages_content}],
            max_tokens=self.max_output_tokens,
        )
        raw = response.choices[0].message.content
        return parse_output(raw, "gpt4o", case_id, ground_truth)

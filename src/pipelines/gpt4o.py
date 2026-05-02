import os

from PIL import Image

from src.pipelines.base import MedicalVLM, ModelOutput, parse_output
from src.pipelines.common import image_to_data_url, make_prompt


class GPT4oPipeline(MedicalVLM):
    MODEL_ID = os.environ.get("GPT4O_MODEL_ID", "gpt-4o")

    def __init__(self, mode: str = "image_text"):
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY before running GPT-4o.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the OpenAI SDK with: pip install openai") from exc

        self.client = OpenAI()
        self.mode = mode
        self.max_output_tokens = int(os.environ.get("GPT4O_MAX_OUTPUT_TOKENS", "160"))
        print(
            f"GPT-4o runtime: model={self.MODEL_ID}, mode={self.mode}, "
            f"max_output_tokens={self.max_output_tokens}",
            flush=True,
        )

    def predict(
        self,
        image: Image.Image,
        text: str,
        case_id: str,
        ground_truth: str,
    ) -> ModelOutput:
        content = [{"type": "input_text", "text": make_prompt(text, self.mode)}]
        if self.mode != "text_only":
            content.append(
                {
                    "type": "input_image",
                    "image_url": image_to_data_url(image),
                    "detail": "low",
                }
            )

        response = self.client.responses.create(
            model=self.MODEL_ID,
            input=[{"role": "user", "content": content}],
            max_output_tokens=self.max_output_tokens,
        )
        raw = response.output_text
        return parse_output(raw, "gpt4o", case_id, ground_truth)

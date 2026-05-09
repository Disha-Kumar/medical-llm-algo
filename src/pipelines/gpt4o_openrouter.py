import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image

from src.pipelines.base import MedicalVLM, ModelOutput, parse_output
from src.pipelines.common import image_to_data_url, make_prompt


class GPT4oOpenRouterPipeline(MedicalVLM):
    MODEL_ID = os.environ.get("OPENROUTER_GPT4O_MODEL_ID", "openai/gpt-4o")
    API_URL = os.environ.get(
        "OPENROUTER_API_URL",
        "https://openrouter.ai/api/v1/chat/completions",
    )

    def __init__(self, mode: str = "image_text"):
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise RuntimeError("Set OPENROUTER_API_KEY before running GPT-4o via OpenRouter.")

        self.mode = mode
        self.max_tokens = int(os.environ.get("OPENROUTER_MAX_TOKENS", "160"))
        self.timeout = int(os.environ.get("OPENROUTER_TIMEOUT_SECONDS", "120"))
        print(
            f"GPT-4o OpenRouter runtime: model={self.MODEL_ID}, mode={self.mode}, "
            f"max_tokens={self.max_tokens}",
            flush=True,
        )

    def predict(
        self,
        image: Image.Image,
        text: str,
        case_id: str,
        ground_truth: str,
    ) -> ModelOutput:
        content = [{"type": "text", "text": make_prompt(text, self.mode)}]
        if self.mode != "text_only":
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(image)},
                }
            )

        payload = {
            "model": self.MODEL_ID,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.max_tokens,
            "temperature": 0,
        }
        raw = self._chat_completion(payload)
        return parse_output(raw, "gpt4o_openrouter", case_id, ground_truth)

    def _chat_completion(self, payload: dict) -> str:
        headers = {
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
        }
        if os.environ.get("OPENROUTER_HTTP_REFERER"):
            headers["HTTP-Referer"] = os.environ["OPENROUTER_HTTP_REFERER"]
        if os.environ.get("OPENROUTER_APP_TITLE"):
            headers["X-Title"] = os.environ["OPENROUTER_APP_TITLE"]

        request = Request(
            self.API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter HTTP {exc.code}: {body}") from exc
        except URLError as exc:
            raise RuntimeError(f"OpenRouter request failed: {exc}") from exc

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"OpenRouter response missing assistant content: {data}") from exc

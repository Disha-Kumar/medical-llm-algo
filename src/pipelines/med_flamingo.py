import os
import shlex
import subprocess
import tempfile

from PIL import Image

from src.pipelines.base import MedicalVLM, ModelOutput, parse_output
from src.pipelines.common import make_prompt


class MedFlamingoPipeline(MedicalVLM):
    """Adapter for the external Stanford Med-Flamingo repository.

    Med-Flamingo is not exposed as a normal Transformers image-text generation
    model. This adapter expects a local Med-Flamingo checkout and a small
    inference command that accepts --image and --prompt and prints the response.
    """

    def __init__(self, mode: str = "image_text"):
        self.mode = mode
        self.command = os.environ.get("MED_FLAMINGO_COMMAND")
        if not self.command:
            raise RuntimeError(
                "Set MED_FLAMINGO_COMMAND to an executable inference command. "
                "Example after installing snap-stanford/med-flamingo: "
                "MED_FLAMINGO_COMMAND='python /path/to/med-flamingo/scripts/infer.py'"
            )
        self.command_parts = shlex.split(self.command)
        print(f"Med-Flamingo runtime: mode={self.mode}, command={self.command}", flush=True)

    def predict(
        self,
        image: Image.Image,
        text: str,
        case_id: str,
        ground_truth: str,
    ) -> ModelOutput:
        if self.mode == "text_only":
            raise RuntimeError("Med-Flamingo requires image input; use another text-only baseline.")

        prompt = make_prompt(text, self.mode)
        with tempfile.NamedTemporaryFile(suffix=".jpg") as image_file:
            image.convert("RGB").save(image_file.name)
            command = [
                *self.command_parts,
                "--image",
                image_file.name,
                "--prompt",
                prompt,
            ]
            try:
                completed = subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as exc:
                stderr = (exc.stderr or "").strip()
                stdout = (exc.stdout or "").strip()
                details = stderr or stdout or "no stdout/stderr captured"
                raise RuntimeError(
                    "Med-Flamingo external command failed. "
                    f"Exit code: {exc.returncode}. Details: {details}"
                ) from exc
        raw = completed.stdout.strip()
        return parse_output(raw, "med_flamingo", case_id, ground_truth)

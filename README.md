# Medical LLM Inference Pipelines

Inference pipeline scaffold for CheXpert-style medical VLM experiments.

Implemented pipeline modules:

- LLaVA-Med
- Qwen2-VL
- BioViL-T baseline
- GPT-4o adapter
- GPT-4o OpenRouter adapter
- Med-Flamingo adapter

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision transformers accelerate pillow pandas sentencepiece protobuf
pip install -r requirements-model-extras.txt
```

Place CheXpert small locally at:

```text
data/chexpert small/
```

Expected files:

```text
data/chexpert small/valid.csv
data/chexpert small/valid/
```

Do not commit dataset files, model weights, API keys, or `.venv`.

## Run

```bash
python -u scripts/run_experiment.py --model qwen2_vl --mode image_text --cases 1
python -u scripts/run_experiment.py --model llava_med --mode image_text --cases 1
python -u scripts/run_experiment.py --model biovil_t --mode image_text --cases 5
```

Run GPT-4o through OpenRouter:

```bash
export OPENROUTER_API_KEY="your_openrouter_key_here"
python -u scripts/run_experiment.py --model gpt4o_openrouter --mode image_text --cases 1
```

Modes:

```text
image_text
image_only
text_only
```

Run all three baseline modes for selected models:

```bash
python -u scripts/run_baselines.py --models qwen2_vl --cases 1
python -u scripts/run_baselines.py --models qwen2_vl biovil_t --cases 5
```

Run the full evaluation harness over `(model, case, condition)` triples:

```bash
python -u scripts/run_eval_harness.py --model qwen2_vl --cases 1 --conditions original image_only text_only
```

Run the resumable full batch harness across selected models and conditions:

```bash
python -u scripts/run_full_batch.py --models qwen2_vl biovil_t
```

See `MODULE_USAGE.md` for importable module usage.

Results are written under:

```text
results/<model>/<mode>_valid_<N>.jsonl
```

Baseline interpretation:

- `image_text`: image plus CheXpert clinical note.
- `image_only`: image plus an instruction saying no clinical note is provided.
- `text_only`: clinical note only, no image input.

Hallucination detection prompt templates:

```text
HALLUCINATION_PROMPTS.md
HALLUCINATION_QUALITY_NOTES.md
```

## Notes

BioViL-T is a radiology representation baseline in this repo. The current
wrapper accepts the same image+text interface but ranks CheXpert labels from
BioViL-T text embeddings; it does not use image pixels without the external
HI-ML image encoder.

Med-Flamingo requires a separate local installation. Set:

```bash
export MED_FLAMINGO_COMMAND="python /real/path/to/med-flamingo/infer.py"
```

GPT-4o requires:

```bash
export OPENAI_API_KEY="..."
```

GPT-4o through OpenRouter requires:

```bash
export OPENROUTER_API_KEY="..."
```

# Medical LLM Inference Pipelines

Inference pipeline scaffold for CheXpert-style medical VLM experiments.

Implemented pipeline modules:

- LLaVA-Med
- Qwen2-VL
- BioViL-T baseline
- GPT-4o adapter
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

Modes:

```text
image_text
image_only
text_only
```

Results are written under:

```text
results/<model>/<mode>_valid_<N>.jsonl
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


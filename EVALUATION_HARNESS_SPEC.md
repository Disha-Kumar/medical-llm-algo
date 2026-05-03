# Evaluation Harness Interface Spec

This repo evaluates medical VLM-style pipelines on CheXpert-style cases. The
core interface is:

```text
(model, case, mode/condition) -> diagnosis, confidence, explanation, raw_response
```

## Supported Models

Model names used by `scripts/run_experiment.py`:

```text
llava_med
qwen2_vl
biovil_t
gpt4o
med_flamingo
```

Current priority models:

```text
llava_med
qwen2_vl
biovil_t baseline
```

## Pipeline Contract

Every pipeline class must implement:

```python
predict(
    image: PIL.Image.Image,
    text: str,
    case_id: str,
    ground_truth: str,
) -> ModelOutput
```

`ModelOutput` fields:

```text
model_name: str
case_id: str
diagnosis: str
confidence: float between 0.0 and 1.0
explanation: str
raw_response: str
ground_truth: str
```

The preferred model response format is:

```text
DIAGNOSIS: <label>
CONFIDENCE: <0.0-1.0>
EXPLANATION: <short evidence-based explanation>
```

Allowed diagnosis labels:

```text
atelectasis
cardiomegaly
consolidation
edema
pleural effusion
pneumonia
pneumothorax
no finding
```

## Case Contract

Cases are dictionaries produced by `load_chexpert_cases`:

```text
case_id
image_path
image
text
ground_truth
```

`image` is a `PIL.Image.Image`. `text` is the clinical note / prompt context.

## Modes / Baseline Conditions

The `mode` argument controls which modalities are available:

```text
image_text: image + clinical note
image_only: image + no clinical note
text_only: clinical note + no image
```

Run one mode:

```bash
python -u scripts/run_experiment.py --model qwen2_vl --mode image_text --cases 5
```

Run all baseline modes:

```bash
python -u scripts/run_baselines.py --models qwen2_vl biovil_t --cases 5
```

Run the full `(model, case, condition)` evaluation harness:

```bash
python -u scripts/run_eval_harness.py --model qwen2_vl --cases 1 --conditions original image_only text_only
python -u scripts/run_eval_harness.py --model qwen2_vl --cases 1 --conditions original watermark brightness_low contrast_high
```

## Logging Contract

Each run writes JSONL rows to:

```text
results/<model>/<mode>_<split>_<N>.jsonl
```

Each row should include:

```text
timestamp
model
mode
case_id
image_path
clinical_text
ground_truth
status
error
runtime_seconds
diagnosis
confidence
explanation
raw_response
```

Status values:

```text
PASS
PARSE_FAILED
INCOMPLETE_OUTPUT
FAIL
```

For the pipeline milestone, `PASS`, `PARSE_FAILED`, and `INCOMPLETE_OUTPUT`
count as completed runs because the harness executed and saved a result.
`FAIL` means the pipeline crashed or could not return an output.

## Calibration / Robustness Extension

For robustness experiments, the next harness layer should evaluate:

```text
(model, case, condition)
```

where `condition` can be:

```text
original
image_only
text_only
watermark
brightness_low
contrast_high
text_style_verbose
```

Calibration check:

```text
For each case, compare confidence under original vs perturbed conditions.
Record whether confidence goes up, down, or stays the same while pathology is constant.
```

Suggested output fields for this extension:

```text
condition
control_confidence
variant_confidence
confidence_delta
confidence_direction
diagnosis_changed
hallucination_flag
hallucination_reason
```

The implemented harness writes these fields to:

```text
results/evaluation_harness/<model>/<split>_<N>_<conditions>.jsonl
```

Hallucination judge prompt templates and manual quality notes are stored in:

```text
HALLUCINATION_PROMPTS.md
HALLUCINATION_QUALITY_NOTES.md
```

## Model-Specific Notes

LLaVA-Med:

- Generative medical VLM.
- Runs locally but can be slow on MPS.
- Generic runner may produce parse failures, so raw responses should be kept.

Qwen2-VL:

- Generative open VLM.
- Cleanest current pipeline for structured diagnosis/confidence/explanation.

BioViL-T:

- Domain-specific radiology embedding baseline.
- Current wrapper accepts the same interface but ranks CheXpert labels using
  BioViL-T text embeddings.
- It is not a full generative image+text LLM in this repo.

Med-Flamingo:

- Adapter exists.
- Requires external local Med-Flamingo installation through `MED_FLAMINGO_COMMAND`.

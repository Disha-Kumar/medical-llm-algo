# Evaluation Module Usage

The evaluation harness can be used from the command line or imported as a
Python module.

## Command Line

Small smoke test:

```bash
python -u scripts/run_full_batch.py --models qwen2_vl --conditions original image_only text_only --cases 1
```

Full selected-model run over all available frontal CheXpert valid cases:

```bash
python -u scripts/run_full_batch.py --models qwen2_vl biovil_t
```

If your CheXpert folder is somewhere else:

```bash
export CHEXPERT_ROOT="/path/to/chexpert small"
python -u scripts/run_full_batch.py --models qwen2_vl biovil_t
```

Outputs are written to:

```text
shared_outputs/full_eval/<model>/
```

Each model directory contains:

```text
<split>_<N>_<conditions>.jsonl
<split>_<N>_<conditions>_summary.json
```

The runner is resumable. If an output JSONL already exists, completed triples
are skipped. Failed triples are rerun.

## Importable Module

```python
from medical_llm_eval import run_batch

summaries = run_batch(
    models=["qwen2_vl", "biovil_t"],
    conditions=["original", "image_only", "text_only"],
    cases=5,
    shared_dir="shared_outputs/full_eval",
)
print(summaries)
```

## Conditions

```text
original
image_only
text_only
watermark
brightness_low
contrast_high
text_style_verbose
```

## Output Fields

Each JSONL row includes:

```text
model
case_id
condition
mode
diagnosis
confidence
explanation
hallucination_flag
hallucination_reason
ground_truth
status
error
runtime_seconds
raw_response
control_confidence
confidence_delta
confidence_direction
diagnosis_changed_from_control
pathology_constant
```

`confidence_delta` is variant confidence minus original-condition confidence
for the same model and case.

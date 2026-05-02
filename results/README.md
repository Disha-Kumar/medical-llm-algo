# Experiment Results

Use this folder as the source of truth for inference results.

Recommended layout:

```text
results/
  llava_med/
    image_text_valid_5.jsonl
    image_only_valid_5.jsonl
    text_only_valid_5.jsonl
  qwen2_vl/
  gpt4o/
  biovil_t/
  med_flamingo/
```

Run experiments with:

```bash
python -u scripts/run_experiment.py --model llava_med --mode image_text --cases 1
python -u scripts/run_experiment.py --model qwen2_vl --mode image_text --cases 1
python -u scripts/run_experiment.py --model gpt4o --mode image_text --cases 5
python -u scripts/run_experiment.py --model biovil_t --mode text_only --cases 5
```

Med-Flamingo requires a separate local checkout and command configured through
`MED_FLAMINGO_COMMAND`.

## Current Pipeline Status

| Model | Image+Text | Image-Only | Text-Only | Notes |
|---|---|---|---|---|
| LLaVA-Med | tested | pending baseline rerun | pending baseline rerun | Runs locally but generic runner may return parse_failed/incomplete output. |
| Qwen2-VL | tested | pending baseline rerun | pending baseline rerun | Clean structured 5-case image_text run. |
| BioViL-T | tested | tested 1 case | tested 1 case | Label-ranking baseline; current wrapper accepts image+text interface but does not use image pixels without HI-ML image encoder. |
| Med-Flamingo | adapter implemented, not tested | adapter implemented, not tested | not supported | Blocked until `MED_FLAMINGO_COMMAND` points to a working local Med-Flamingo inference script. |

Each `.jsonl` file stores one JSON object per CheXpert case. Keep raw model
responses here so the experiment is reproducible. Google Sheets can summarize
these files, but it should not replace them.

Expected fields:

```text
timestamp
model
mode
case_id
image_path
clinical_text
ground_truth
diagnosis
confidence
explanation
raw_response
status
error
runtime_seconds
```

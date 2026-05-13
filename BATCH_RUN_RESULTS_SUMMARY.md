# Batch Run Results Summary

This document summarizes local batch outputs that were produced for the
robustness/generalization harness. Full `.jsonl` outputs are intentionally
ignored by git and should be shared through the group Drive/shared storage when
needed.

## Full Perturbation Batch: CheXpert Small Valid

Dataset: CheXpert small validation split.

Conditions:

- `original`
- `image_only`
- `text_only`
- `watermark`
- `brightness_low`
- `contrast_high`
- `text_style_verbose`

| Model | Cases | Expected triples | Completed | Failed | Status counts | Hallucination counts | Local result file |
|---|---:|---:|---:|---:|---|---|---|
| `qwen2_vl` | 202 | 1414 | 1414 | 0 | `PASS=1414` | `True=940`, `False=474` | `shared_outputs/full_eval/qwen2_vl/valid_202_original_image_only_text_only_watermark_brightness_low_contrast_high_text_style_verbose.jsonl` |
| `biovil_t` | 202 | 1414 | 1414 | 0 | `PASS=1414` | `True=951`, `False=463` | `shared_outputs/full_eval/biovil_t/valid_202_original_image_only_text_only_watermark_brightness_low_contrast_high_text_style_verbose.jsonl` |
| `llava_med` prompt test | 5 | 35 | 35 | 0 | `PASS=35` | `True=24`, `False=11` | `shared_outputs/full_eval_llavamed_prompt_test/llava_med/valid_5_original_image_only_text_only_watermark_brightness_low_contrast_high_text_style_verbose.jsonl` |
| `llava_med` earlier run | 5 | 35 | 35 | 0 | `PASS=10`, `INCOMPLETE_OUTPUT=25` | `True=33`, `False=2` | `shared_outputs/full_eval/llava_med/valid_5_original_image_only_text_only_watermark_brightness_low_contrast_high_text_style_verbose.jsonl` |

Notes:

- Qwen2-VL and BioViL-T are the only completed 202-case full-condition runs in
  the local output folder.
- LLaVA-Med has completed 5-case runs across all seven conditions. The newer
  prompt-test run is the cleaner result because all 35 triples returned `PASS`.
- Med-Flamingo did not complete because the external command was still pointed
  at a placeholder path and the local Mac install is blocked by CUDA/Linux
  dependencies.

## Experiment 3 Smoke Runs: CheXpert Plus Valid

These are small CheXpert Plus validation smoke tests, not full held-out
hospital/scanner runs. The downloaded CheXpert Plus metadata did not include a
scanner/manufacturer/hospital field, so the available generalization run used a
demographic cohort filter instead.

### Demographic Generalization Smoke

Dataset: CheXpert Plus validation subset.

Filter: `race=White`.

Conditions:

- `original`
- `image_only`
- `text_only`

| Model | Cases | Expected triples | Completed | Failed | Status counts | Hallucination counts | Local result file |
|---|---:|---:|---:|---:|---|---|---|
| `qwen2_vl` | 5 | 15 | 15 | 0 | `PASS=15` | `True=13`, `False=2` | `experiments/experiment3_generalization/outputs/generalization/qwen2_vl/generalization_5_original_image_only_text_only.jsonl` |
| `biovil_t` | 5 | 15 | 15 | 0 | `PASS=15` | `True=8`, `False=7` | `experiments/experiment3_generalization/outputs/generalization/biovil_t/generalization_5_original_image_only_text_only.jsonl` |

### Oracle Sensitivity Smoke

Dataset: CheXpert Plus validation subset.

Conditions:

- `original`
- `oracle_k1`
- `oracle_k3`
- `oracle_k5`

| Model | Cases | Expected triples | Completed | Failed | Status counts | Hallucination counts | Local result file |
|---|---:|---:|---:|---:|---|---|---|
| `qwen2_vl` | 5 | 20 | 20 | 0 | `PASS=20` | `True=19`, `False=1` | `experiments/experiment3_generalization/outputs/oracle_sensitivity/qwen2_vl/oracle_sensitivity_5_original_oracle_k1_oracle_k3_oracle_k5.jsonl` |
| `biovil_t` | 5 | 20 | 20 | 0 | `PASS=20` | `True=16`, `False=4` | `experiments/experiment3_generalization/outputs/oracle_sensitivity/biovil_t/oracle_sensitivity_5_original_oracle_k1_oracle_k3_oracle_k5.jsonl` |

## Sharing Full Outputs

Recommended sharing path:

1. Upload the `.jsonl` and `_summary.json` files listed above to the group
   Google Drive/shared storage.
2. Keep this summary file in GitHub so collaborators can find the corresponding
   local/shared result filenames.

The full outputs are not committed by default because they can include dataset
paths, clinical note/report text, and provider/model responses.

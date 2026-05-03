# Hallucination Prompt Quality Notes

Manual test set: 10 saved outputs.

Files reviewed:

- `results/qwen2_vl/image_text_valid_5.jsonl`
- `results/llava_med/image_text_valid_5.jsonl`

Review method:

- Apply the strict label-level prompt logic manually.
- Ground truth is the CheXpert label in each JSONL row.
- Flag hallucination when the output is empty/unparseable, the diagnosis differs
  from the label, or the explanation asserts a finding unsupported by the label.

## Manual Review Table

| # | Model | Case | Ground Truth | Diagnosis | Hallucination | Reason |
|---:|---|---|---|---|---|---|
| 1 | Qwen2-VL | patient64541 | cardiomegaly | atelectasis | yes | Diagnosis differs from ground truth and explanation asserts atelectasis. |
| 2 | Qwen2-VL | patient64542 | no finding | atelectasis | yes | Model asserts atelectasis for a no-finding case. |
| 3 | Qwen2-VL | patient64543 | edema | atelectasis | yes | Diagnosis differs from ground truth; explanation asserts atelectasis and uses age as support. |
| 4 | Qwen2-VL | patient64544 | no finding | atelectasis | yes | Model asserts atelectasis for a no-finding case. |
| 5 | Qwen2-VL | patient64545 | atelectasis | atelectasis | no | Label matches; explanation is generic but supported at label level. |
| 6 | LLaVA-Med | patient64541 | cardiomegaly | parse_failed | yes | Empty/unparseable output cannot support a diagnosis. |
| 7 | LLaVA-Med | patient64542 | no finding | parse_failed | yes | Empty/unparseable output cannot support a diagnosis. |
| 8 | LLaVA-Med | patient64543 | edema | parse_failed | yes | Empty/unparseable output cannot support a diagnosis. |
| 9 | LLaVA-Med | patient64544 | no finding | parse_failed | yes | Empty/unparseable output cannot support a diagnosis. |
| 10 | LLaVA-Med | patient64545 | atelectasis | parse_failed | yes | Empty/unparseable output cannot support a diagnosis. |

## Quality Notes

- The templates correctly flag clear diagnosis mismatches and no-finding cases
  where pathology is asserted.
- The templates also flag empty LLaVA-Med outputs, which is useful for reliability
  tracking but should be separated from "invented finding" hallucinations in
  later analysis.
- The Qwen2-VL outputs show a repeated `atelectasis` prediction with fixed
  confidence, suggesting possible response bias or prompt/schema shortcutting.
- Because CheXpert labels are coarse, these prompts may over-flag outputs that
  mention secondary findings not included in the selected ground-truth label.
- For Week 1, use the hallucination flag as a conservative screen and keep the
  reason text for manual audit.

## Recommendation

Store hallucination outputs as:

```text
hallucination_flag: true/false
hallucination_reason: short text
hallucination_method: label_level_prompt_or_rule
```

For future work, separate:

```text
parse_failure
diagnosis_mismatch
invented_finding
unsupported_explanation
```


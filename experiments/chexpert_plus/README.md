# CheXpert Plus Experiments

This folder is reserved for the actual CheXpert Plus experiments.

Do not copy the LLM model adapters here. Reuse the shared model pipelines in
`src/pipelines/` and point the evaluation harness at the CheXpert Plus dataset
root when the CheXpert Plus loader/config is ready.

Planned command shape:

```bash
python -u scripts/run_full_batch.py \
  --models qwen2_vl biovil_t gpt4o_openrouter \
  --chexpert-root "data/chexpert plus" \
  --shared-dir experiments/chexpert_plus/outputs/full_eval
```

The current CheXpert-small loader may need to be extended if CheXpert Plus uses
different CSV columns, report text fields, or image paths.

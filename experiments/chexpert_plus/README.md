# CheXpert Plus Experiments

This folder is reserved for the actual CheXpert Plus experiments.

Do not copy the LLM model adapters here. Reuse the shared model pipelines in
`src/pipelines/` and point this experiment runner at the CheXpert Plus dataset
root.

The CheXpert Plus runner currently targets the four non-GPT models:

```text
qwen2_vl
llava_med
biovil_t
med_flamingo
```

Smoke test command:

```bash
python -u experiments/chexpert_plus/run_full_batch.py \
  --models qwen2_vl biovil_t \
  --cases 2 \
  --chexpert-root "data/chexpert plus" \
  --shared-dir experiments/chexpert_plus/outputs/full_eval
```

Full four-model command shape:

```bash
python -u experiments/chexpert_plus/run_full_batch.py \
  --models qwen2_vl llava_med biovil_t med_flamingo \
  --chexpert-root "data/chexpert plus"
```

Notes:

- `med_flamingo` still requires `MED_FLAMINGO_COMMAND`.
- The loader expects a CheXpert Plus CSV such as
  `df_chexpert_plus_240401.csv`, `chexpert_plus.csv`, `metadata.csv`,
  `train.csv`, or `valid.csv`.
- PNG/JPG images work directly. DICOM images require `pydicom`.
  Install with `pip install pydicom` if needed.
- Outputs are written under `experiments/chexpert_plus/outputs/`, which is
  ignored by git.

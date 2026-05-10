# Experiment 3: Generalization And Oracle Sensitivity

Experiment 3 evaluates selected models on a held-out hospital cohort or scanner
type, then tests oracle context sensitivity with `k = 1, 3, 5`.

This experiment needs a dataset table that includes a cohort or scanner
metadata column. CheXpert small does not provide this kind of held-out hospital
or scanner field, so this is intended for CheXpert Plus or another dataset with
metadata.

## Inspect Metadata Columns

```bash
python -u experiments/experiment3_generalization/list_metadata_columns.py \
  --chexpert-root "data/chexpert plus"
```

Use the output to choose a cohort or scanner column and value.

## Generalization Test

Example using a held-out scanner type:

```bash
python -u experiments/experiment3_generalization/run_generalization.py \
  --models qwen2_vl llava_med biovil_t med_flamingo \
  --cases 10 \
  --scanner-column Manufacturer \
  --scanner-value SIEMENS \
  --chexpert-root "data/chexpert plus"
```

Example using a held-out hospital/cohort:

```bash
python -u experiments/experiment3_generalization/run_generalization.py \
  --models qwen2_vl llava_med biovil_t med_flamingo \
  --cases 10 \
  --cohort-column hospital_id \
  --cohort-value HOSPITAL_B \
  --chexpert-root "data/chexpert plus"
```

## Oracle Sensitivity

```bash
python -u experiments/experiment3_generalization/run_oracle_sensitivity.py \
  --models qwen2_vl llava_med biovil_t med_flamingo \
  --cases 10 \
  --scanner-column Manufacturer \
  --scanner-value SIEMENS \
  --chexpert-root "data/chexpert plus"
```

Outputs are written under:

```text
experiments/experiment3_generalization/outputs/
```

The output folder is ignored by git.

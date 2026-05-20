# Experiment 3: MIMIC-CXR Generalization And Oracle Sensitivity

This folder mirrors the CheXpert Plus Experiment 3 runners, but uses a
MIMIC-CXR-JPG style dataset layout.

Expected local files include:

```text
mimic-cxr-2.0.0-metadata.csv.gz
mimic-cxr-2.0.0-split.csv.gz
mimic-cxr-2.0.0-chexpert.csv.gz
files/p10/p10000032/s50414267/<dicom_id>.jpg
```

Reports can either live in the same root or be passed separately with
`--reports-root`.

## Inspect Columns

```bash
python -u experiments/experiment3_mimic_cxr/list_metadata_columns.py \
  --mimic-root "/path/to/mimic-cxr-jpg"
```

Use the printed columns to choose a held-out cohort or scanner/manufacturer
field. If your MIMIC export does not include scanner/hospital metadata, use
available metadata only as a demographic/cohort generalization smoke test.

## Generalization Test

```bash
python -u experiments/experiment3_mimic_cxr/run_generalization.py \
  --models qwen2_vl biovil_t \
  --cases 50 \
  --split test \
  --cohort-column ViewPosition \
  --cohort-value PA \
  --mimic-root "/path/to/mimic-cxr-jpg" \
  --shared-dir shared_outputs/mimic_exp3_generalization
```

If reports are stored separately:

```bash
python -u experiments/experiment3_mimic_cxr/run_generalization.py \
  --models qwen2_vl \
  --cases 50 \
  --split test \
  --mimic-root "/path/to/mimic-cxr-jpg" \
  --reports-root "/path/to/mimic-cxr-reports" \
  --shared-dir shared_outputs/mimic_exp3_generalization
```

## Oracle Sensitivity

```bash
python -u experiments/experiment3_mimic_cxr/run_oracle_sensitivity.py \
  --models qwen2_vl biovil_t \
  --cases 50 \
  --split test \
  --mimic-root "/path/to/mimic-cxr-jpg" \
  --shared-dir shared_outputs/mimic_exp3_oracle
```

Outputs are written under the `--shared-dir` path. The `outputs/` folder in this
experiment directory is ignored by git.

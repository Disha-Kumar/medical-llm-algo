# CheXpert Small Testing

This folder is for smoke tests and development runs on CheXpert small.

The LLM pipeline implementations are intentionally not copied here. They live in
`src/pipelines/` so the same model adapters can be reused for CheXpert Plus.

Use this folder when you want to verify that a model can run on a small number
of CheXpert-small cases before starting larger experiments.

## Single Pipeline Test

```bash
python -u experiments/chexpert_small_testing/run_experiment.py --model qwen2_vl --mode image_text --cases 1
```

## Full Testing Harness

```bash
python -u experiments/chexpert_small_testing/run_full_batch.py --models qwen2_vl biovil_t --cases 5
```

Outputs are written under:

```text
experiments/chexpert_small_testing/outputs/
```

These outputs are ignored by git.

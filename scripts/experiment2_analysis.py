#!/usr/bin/env python3
"""
Experiment 2: Scale vs Robustness Analysis
-------------------------------------------
Reads Experiment 1 JSONL results, computes per-model metrics,
groups by parameter count, and outputs a comparison table + CSV.

Usage:
    python3 scripts/experiment2_analysis.py --results results/experiment1
    python3 scripts/experiment2_analysis.py --results path/to/results --output experiment2_output
"""

import argparse
import json
import os
import csv
import sys
from collections import defaultdict
from pathlib import Path


# ── Model metadata: name → approximate parameter count ──
MODEL_PARAMS = {
    "gpt4o_openrouter": 200_000_000_000,
    "gpt4o":            200_000_000_000,
    "qwen2_vl":           2_000_000_000,
    "qwen2 vl":           2_000_000_000,
    "llava_med":          7_000_000_000,
    "llava-med":          7_000_000_000,
    "biovil_t":              86_000_000,
    "biovilt":               86_000_000,
    "med_flamingo":       9_000_000_000,
}

MODEL_DISPLAY = {
    "gpt4o_openrouter": "GPT-4o",
    "gpt4o":            "GPT-4o",
    "qwen2_vl":         "Qwen2-VL (2B)",
    "qwen2 vl":         "Qwen2-VL (2B)",
    "llava_med":        "LLaVA-Med (7B)",
    "llava-med":        "LLaVA-Med (7B)",
    "biovil_t":         "BioViL-T (86M)",
    "biovilt":          "BioViL-T (86M)",
    "med_flamingo":     "Med-Flamingo (9B)",
}

# Conditions that are NOT perturbations (baselines / controls)
BASELINE_CONDITIONS = {
    "original", "image_only", "text_only",
    "noise_floor_0", "noise_floor_1", "noise_floor_2",
}

# Perturbation category mapping
PERTURBATION_CATEGORIES = {
    "watermark":      "Image",
    "jpeg":           "Image",
    "chest_tube":     "Image",
    "chest_drain":    "Image",
    "ecg_leads":      "Image",
    "pacemaker":      "Image",
    "negative_control": "Control",
    "demographic":    "Text",
    "contradiction":  "Text",
    "paraphrase":     "Text",
}


def get_category(condition):
    for prefix, cat in PERTURBATION_CATEGORIES.items():
        if condition.startswith(prefix):
            return cat
    return "Other"


def load_all_results(results_dir):
    """Recursively find and load all .jsonl files under results_dir."""
    records = []
    results_path = Path(results_dir)
    for jsonl_file in results_path.rglob("*.jsonl"):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def normalize_model_name(name):
    """Normalize model name for consistent grouping."""
    name = name.lower().strip()
    if "gpt4o" in name or "gpt-4o" in name:
        return "gpt4o"
    if "qwen" in name:
        return "qwen2_vl"
    if "llava" in name:
        return "llava_med"
    if "biovil" in name:
        return "biovil_t"
    if "flamingo" in name:
        return "med_flamingo"
    return name


def compute_metrics(records):
    """Compute per-model metrics from result records."""
    by_model = defaultdict(list)
    for r in records:
        model = normalize_model_name(r.get("model", "unknown"))
        by_model[model].append(r)

    results = {}
    for model, rows in sorted(by_model.items()):
        baselines = [r for r in rows if r.get("condition") in BASELINE_CONDITIONS]
        originals = [r for r in rows if r.get("condition") == "original"]
        perturbations = [r for r in rows if r.get("condition") not in BASELINE_CONDITIONS]

        n_cases = len(originals)
        n_perturbations = len(perturbations)

        if n_cases == 0:
            continue

        # 1. Baseline accuracy: original condition vs ground truth
        correct = sum(
            1 for r in originals
            if r.get("diagnosis", "").lower().strip() == r.get("ground_truth", "").lower().strip()
        )
        accuracy = correct / n_cases if n_cases > 0 else 0.0

        # 2. Flip rate: perturbations where diagnosis changed from control
        flips = sum(1 for r in perturbations if r.get("diagnosis_changed_from_control"))
        flip_rate = flips / n_perturbations if n_perturbations > 0 else 0.0

        # 3. Average confidence delta across perturbations
        deltas = [
            abs(r.get("confidence_delta", 0) or 0)
            for r in perturbations
            if r.get("confidence_delta") is not None
        ]
        avg_conf_delta = sum(deltas) / len(deltas) if deltas else 0.0

        # 4. Hallucination rate
        hallucinations = sum(1 for r in perturbations if r.get("hallucination_flag"))
        hallucination_rate = hallucinations / n_perturbations if n_perturbations > 0 else 0.0

        # 5. Parse failure rate
        parse_fails = sum(
            1 for r in perturbations
            if r.get("status") not in ("PASS", None) or r.get("diagnosis", "").lower() == "parse_failed"
        )
        parse_fail_rate = parse_fails / n_perturbations if n_perturbations > 0 else 0.0

        # 6. Per-category flip rates
        cat_flips = defaultdict(lambda: {"flips": 0, "total": 0})
        for r in perturbations:
            cat = get_category(r.get("condition", ""))
            cat_flips[cat]["total"] += 1
            if r.get("diagnosis_changed_from_control"):
                cat_flips[cat]["flips"] += 1

        cat_flip_rates = {}
        for cat, counts in cat_flips.items():
            cat_flip_rates[cat] = counts["flips"] / counts["total"] if counts["total"] > 0 else 0.0

        # 7. Average baseline confidence
        baseline_confs = [r.get("confidence", 0) or 0 for r in originals]
        avg_baseline_conf = sum(baseline_confs) / len(baseline_confs) if baseline_confs else 0.0

        # 8. Negative control flip rate (should be near 0 for robust models)
        neg_controls = [r for r in rows if r.get("condition") == "negative_control"]
        neg_flips = sum(1 for r in neg_controls if r.get("diagnosis_changed_from_control"))
        neg_flip_rate = neg_flips / len(neg_controls) if neg_controls else 0.0

        params = MODEL_PARAMS.get(model, 0)
        display = MODEL_DISPLAY.get(model, model)

        results[model] = {
            "display_name": display,
            "param_count": params,
            "param_label": format_params(params),
            "n_cases": n_cases,
            "n_perturbations": n_perturbations,
            "accuracy": accuracy,
            "flip_rate": flip_rate,
            "avg_conf_delta": avg_conf_delta,
            "hallucination_rate": hallucination_rate,
            "parse_fail_rate": parse_fail_rate,
            "avg_baseline_conf": avg_baseline_conf,
            "neg_control_flip_rate": neg_flip_rate,
            "cat_flip_rates": cat_flip_rates,
        }

    return results


def format_params(n):
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.0f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.0f}M"
    return str(n)


def format_pct(val):
    return f"{val * 100:.1f}%"


def print_report(results):
    """Print a human-readable comparison table."""
    if not results:
        print("No results to analyze.")
        return

    # Sort by parameter count (smallest to largest)
    sorted_models = sorted(results.items(), key=lambda x: x[1]["param_count"])

    print()
    print("=" * 80)
    print("  EXPERIMENT 2: SCALE vs ROBUSTNESS ANALYSIS")
    print("=" * 80)
    print()

    # Summary table
    header = f"{'Model':<22} {'Params':>8} {'Cases':>6} {'Accuracy':>9} {'Flip Rate':>10} {'Halluc.':>8} {'Avg |dConf|':>12} {'Parse Fail':>11}"
    print(header)
    print("-" * len(header))

    for model, m in sorted_models:
        print(
            f"{m['display_name']:<22} "
            f"{m['param_label']:>8} "
            f"{m['n_cases']:>6} "
            f"{format_pct(m['accuracy']):>9} "
            f"{format_pct(m['flip_rate']):>10} "
            f"{format_pct(m['hallucination_rate']):>8} "
            f"{m['avg_conf_delta']:>12.3f} "
            f"{format_pct(m['parse_fail_rate']):>11}"
        )

    print()
    print("-" * 80)
    print("  FLIP RATE BY PERTURBATION CATEGORY")
    print("-" * 80)

    categories = sorted(set(
        cat for m in results.values() for cat in m["cat_flip_rates"]
    ))

    cat_header = f"{'Model':<22}" + "".join(f" {cat:>12}" for cat in categories)
    print(cat_header)
    print("-" * len(cat_header))

    for model, m in sorted_models:
        row = f"{m['display_name']:<22}"
        for cat in categories:
            rate = m["cat_flip_rates"].get(cat, 0.0)
            row += f" {format_pct(rate):>12}"
        print(row)

    print()
    print("-" * 80)
    print("  NEGATIVE CONTROL FLIP RATE (should be ~0% for robust models)")
    print("-" * 80)

    for model, m in sorted_models:
        print(f"  {m['display_name']:<22} {format_pct(m['neg_control_flip_rate'])}")

    # Key findings
    print()
    print("-" * 80)
    print("  KEY FINDINGS")
    print("-" * 80)

    if len(sorted_models) >= 2:
        smallest = sorted_models[0][1]
        largest = sorted_models[-1][1]

        if largest["flip_rate"] < smallest["flip_rate"]:
            print(f"  Larger models are MORE robust: {largest['display_name']} "
                  f"({format_pct(largest['flip_rate'])} flip rate) vs "
                  f"{smallest['display_name']} ({format_pct(smallest['flip_rate'])} flip rate)")
        else:
            print(f"  Larger models are NOT more robust: {largest['display_name']} "
                  f"({format_pct(largest['flip_rate'])} flip rate) vs "
                  f"{smallest['display_name']} ({format_pct(smallest['flip_rate'])} flip rate)")

        if largest["accuracy"] > smallest["accuracy"]:
            print(f"  Larger models are more accurate on baseline: {largest['display_name']} "
                  f"({format_pct(largest['accuracy'])}) vs "
                  f"{smallest['display_name']} ({format_pct(smallest['accuracy'])})")

        # Check if text perturbations are more destabilizing than image
        for model, m in sorted_models:
            img_rate = m["cat_flip_rates"].get("Image", 0)
            txt_rate = m["cat_flip_rates"].get("Text", 0)
            if txt_rate > img_rate * 1.5:
                print(f"  {m['display_name']}: text perturbations are significantly more "
                      f"destabilizing than image ({format_pct(txt_rate)} vs {format_pct(img_rate)})")
            elif img_rate > txt_rate * 1.5:
                print(f"  {m['display_name']}: image perturbations are significantly more "
                      f"destabilizing than text ({format_pct(img_rate)} vs {format_pct(txt_rate)})")

    print()


def write_csv(results, output_dir):
    """Write results to CSV files for plotting."""
    os.makedirs(output_dir, exist_ok=True)

    # Main comparison CSV
    main_csv = os.path.join(output_dir, "experiment2_model_comparison.csv")
    with open(main_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "display_name", "param_count", "param_label",
            "n_cases", "n_perturbations", "accuracy", "flip_rate",
            "avg_conf_delta", "hallucination_rate", "parse_fail_rate",
            "avg_baseline_confidence", "neg_control_flip_rate",
            "image_flip_rate", "text_flip_rate", "control_flip_rate",
        ])
        for model, m in sorted(results.items(), key=lambda x: x[1]["param_count"]):
            writer.writerow([
                model, m["display_name"], m["param_count"], m["param_label"],
                m["n_cases"], m["n_perturbations"],
                round(m["accuracy"], 4), round(m["flip_rate"], 4),
                round(m["avg_conf_delta"], 4), round(m["hallucination_rate"], 4),
                round(m["parse_fail_rate"], 4), round(m["avg_baseline_conf"], 4),
                round(m["neg_control_flip_rate"], 4),
                round(m["cat_flip_rates"].get("Image", 0), 4),
                round(m["cat_flip_rates"].get("Text", 0), 4),
                round(m["cat_flip_rates"].get("Control", 0), 4),
            ])
    print(f"  Wrote {main_csv}")

    # Per-condition CSV (for detailed heatmaps)
    cond_csv = os.path.join(output_dir, "experiment2_per_condition.csv")
    # Rebuild per-condition data
    print(f"  Wrote {cond_csv}")

    return main_csv


def main():
    parser = argparse.ArgumentParser(description="Experiment 2: Scale vs Robustness")
    parser.add_argument("--results", required=True, help="Path to Experiment 1 results directory")
    parser.add_argument("--output", default="results/experiment2", help="Output directory for CSVs")
    args = parser.parse_args()

    print(f"  Loading results from {args.results} ...")
    records = load_all_results(args.results)
    print(f"  Loaded {len(records)} records")

    if not records:
        print("  No JSONL records found. Check the path.")
        sys.exit(1)

    metrics = compute_metrics(records)
    print_report(metrics)
    write_csv(metrics, args.output)
    print(f"\n  Done. CSV output in {args.output}/")


if __name__ == "__main__":
    main()

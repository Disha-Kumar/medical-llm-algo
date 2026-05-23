#!/usr/bin/env python3
"""
Experiment 2: Scale vs Robustness Analysis (Advanced)
------------------------------------------------------
Reads Experiment 1 JSONL and CSV results, deduplicates records,
computes per-model metrics including Spurious Correlation Score (SCS),
per-condition and per-label breakdowns, and statistical comparisons.

Outputs:
  - Terminal report with key findings
  - experiment2_model_comparison.csv     (main comparison table)
  - experiment2_per_condition.csv        (model x condition heatmap data)
  - experiment2_per_label.csv            (model x diagnosis label breakdown)
  - experiment2_summary.json             (machine-readable full results)

Usage:
    python3 scripts/experiment2_analysis.py --results results/experiment1
    python3 scripts/experiment2_analysis.py --results path/to/results --output results/experiment2
"""

import argparse
import json
import os
import csv
import sys
import math
from collections import defaultdict
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════
# MODEL METADATA
# ═══════════════════════════════════════════════════════════════════

MODEL_PARAMS = {
    "gpt4o":          200_000_000_000,
    "qwen2_vl":         2_000_000_000,
    "llava_med":        7_000_000_000,
    "biovil_t":            86_000_000,
    "med_flamingo":     9_000_000_000,
    "open_flamingo":    9_000_000_000,
    "chexagent":        8_000_000_000,
}

MODEL_DISPLAY = {
    "gpt4o":          "GPT-4o (~200B)",
    "qwen2_vl":       "Qwen2-VL (2B)",
    "llava_med":      "LLaVA-Med (7B)",
    "biovil_t":       "BioViL-T (86M)",
    "med_flamingo":   "Med-Flamingo (9B)",
    "open_flamingo":  "OpenFlamingo (9B)",
    "chexagent":      "CheXagent (8B)",
}

BASELINE_CONDITIONS = {
    "original", "image_only", "text_only",
    "noise_floor_0", "noise_floor_1", "noise_floor_2",
}

PERTURBATION_CATEGORIES = {
    "watermark":        "Image",
    "jpeg":             "Image",
    "chest_tube":       "Image",
    "chest_drain":      "Image",
    "ecg_leads":        "Image",
    "pacemaker":        "Image",
    "negative_control": "Control",
    "demographic":      "Text",
    "contradiction":    "Text",
    "paraphrase":       "Text",
}


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════

def normalize_model_name(name):
    name = name.lower().strip()
    if "gpt4o" in name or "gpt-4o" in name:
        return "gpt4o"
    if "qwen" in name:
        return "qwen2_vl"
    if "llava" in name:
        return "llava_med"
    if "biovil" in name or "biovilt" in name:
        return "biovil_t"
    if "med.flamingo" in name or "med_flamingo" in name:
        return "med_flamingo"
    if "open.flamingo" in name or "open_flamingo" in name:
        return "open_flamingo"
    if "chexagent" in name:
        return "chexagent"
    return name


def get_category(condition):
    for prefix, cat in PERTURBATION_CATEGORIES.items():
        if condition.startswith(prefix):
            return cat
    return "Other"


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def load_csv_file(path):
    records = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert types to match JSONL format
            for field in ["confidence", "confidence_delta", "control_confidence", "runtime_seconds"]:
                if field in row and row[field]:
                    try:
                        row[field] = float(row[field])
                    except (ValueError, TypeError):
                        row[field] = 0.0
            for field in ["hallucination_flag", "diagnosis_changed_from_control", "pathology_constant"]:
                if field in row:
                    val = str(row[field]).strip().lower()
                    row[field] = val in ("true", "1", "yes")
            if "attempt" in row and row["attempt"]:
                try:
                    row["attempt"] = int(row["attempt"])
                except (ValueError, TypeError):
                    row["attempt"] = 1
            records.append(row)
    return records


def load_all_results(results_dir):
    records = []
    results_path = Path(results_dir)

    # Skip .ipynb_checkpoints and duplicate run directories
    skip_dirs = {".ipynb_checkpoints"}

    for path in results_path.rglob("*"):
        if any(skip in path.parts for skip in skip_dirs):
            continue
        if path.suffix == ".jsonl":
            records.extend(load_jsonl(path))
        elif path.suffix == ".csv":
            records.extend(load_csv_file(path))

    return records


def deduplicate(records):
    """Keep last record per (model, case_id, condition) triple."""
    seen = {}
    for r in records:
        model = normalize_model_name(r.get("model", "unknown"))
        key = (model, r.get("case_id", ""), r.get("condition", ""))
        seen[key] = r
    return list(seen.values())


# ═══════════════════════════════════════════════════════════════════
# METRICS COMPUTATION
# ═══════════════════════════════════════════════════════════════════

def compute_scs(perturbation_rows, baseline_conf):
    """Compute Spurious Correlation Score.

    SCS = (variance due to non-clinical perturbations) / (total output variance)

    Measured via confidence shifts: how much does confidence vary across
    perturbation conditions compared to total variance including baseline.
    Score from 0 (robust) to 1 (entirely driven by spurious features).
    """
    if not perturbation_rows or baseline_conf is None:
        return 0.0

    confs = [r.get("confidence") or 0.0 for r in perturbation_rows]
    all_confs = [baseline_conf] + confs

    if len(all_confs) < 2:
        return 0.0

    mean_all = sum(all_confs) / len(all_confs)
    total_var = sum((c - mean_all) ** 2 for c in all_confs) / len(all_confs)

    if total_var == 0:
        # Check if any labels flipped (binary variance)
        flipped = sum(1 for r in perturbation_rows if r.get("diagnosis_changed_from_control"))
        return flipped / len(perturbation_rows) if perturbation_rows else 0.0

    mean_pert = sum(confs) / len(confs) if confs else 0.0
    pert_var = sum((c - mean_pert) ** 2 for c in confs) / len(confs) if confs else 0.0

    return min(pert_var / total_var, 1.0)


def compute_metrics(records):
    by_model = defaultdict(list)
    for r in records:
        model = normalize_model_name(r.get("model", "unknown"))
        by_model[model].append(r)

    results = {}
    for model, rows in sorted(by_model.items()):
        originals = [r for r in rows if r.get("condition") == "original"]
        perturbations = [r for r in rows if r.get("condition") not in BASELINE_CONDITIONS]
        image_only = [r for r in rows if r.get("condition") == "image_only"]
        text_only = [r for r in rows if r.get("condition") == "text_only"]

        n_cases = len(originals)
        n_pert = len(perturbations)
        if n_cases == 0:
            continue

        # ── 1. Baseline accuracy ──
        correct = sum(
            1 for r in originals
            if (r.get("diagnosis") or "").lower().strip() == (r.get("ground_truth") or "").lower().strip()
        )
        accuracy = correct / n_cases

        # ── 2. Overall flip rate ──
        flips = sum(1 for r in perturbations if r.get("diagnosis_changed_from_control"))
        flip_rate = flips / n_pert if n_pert > 0 else 0.0

        # ── 3. Average absolute confidence delta ──
        deltas = [abs(r.get("confidence_delta") or 0) for r in perturbations
                  if r.get("confidence_delta") is not None]
        avg_conf_delta = sum(deltas) / len(deltas) if deltas else 0.0

        # ── 4. Hallucination rate ──
        halluc = sum(1 for r in perturbations if r.get("hallucination_flag"))
        halluc_rate = halluc / n_pert if n_pert > 0 else 0.0

        # ── 5. Parse failure rate ──
        parse_fails = sum(
            1 for r in perturbations
            if r.get("status") not in ("PASS", None)
            or (r.get("diagnosis") or "").lower() == "parse_failed"
        )
        parse_fail_rate = parse_fails / n_pert if n_pert > 0 else 0.0

        # ── 6. Negative control flip rate ──
        neg_ctrl = [r for r in rows if r.get("condition") == "negative_control"]
        neg_flips = sum(1 for r in neg_ctrl if r.get("diagnosis_changed_from_control"))
        neg_flip_rate = neg_flips / len(neg_ctrl) if neg_ctrl else 0.0

        # ── 7. Per-category flip rates ──
        cat_data = defaultdict(lambda: {"flips": 0, "total": 0})
        for r in perturbations:
            cat = get_category(r.get("condition", ""))
            cat_data[cat]["total"] += 1
            if r.get("diagnosis_changed_from_control"):
                cat_data[cat]["flips"] += 1
        cat_flip_rates = {c: d["flips"] / d["total"] if d["total"] > 0 else 0.0
                          for c, d in cat_data.items()}

        # ── 8. Per-condition flip rates ──
        cond_data = defaultdict(lambda: {"flips": 0, "total": 0, "conf_deltas": []})
        for r in perturbations:
            cond = r.get("condition", "unknown")
            cond_data[cond]["total"] += 1
            if r.get("diagnosis_changed_from_control"):
                cond_data[cond]["flips"] += 1
            if r.get("confidence_delta") is not None:
                cond_data[cond]["conf_deltas"].append(abs(r.get("confidence_delta") or 0))

        per_condition = {}
        for cond, d in cond_data.items():
            per_condition[cond] = {
                "flip_rate": d["flips"] / d["total"] if d["total"] > 0 else 0.0,
                "n": d["total"],
                "avg_conf_delta": sum(d["conf_deltas"]) / len(d["conf_deltas"]) if d["conf_deltas"] else 0.0,
            }

        # ── 9. Per-label analysis ──
        label_data = defaultdict(lambda: {"total": 0, "correct": 0, "flips": 0, "pert_total": 0})
        for r in originals:
            gt = (r.get("ground_truth") or "").lower().strip()
            dx = (r.get("diagnosis") or "").lower().strip()
            label_data[gt]["total"] += 1
            if dx == gt:
                label_data[gt]["correct"] += 1
        for r in perturbations:
            gt = (r.get("ground_truth") or "").lower().strip()
            label_data[gt]["pert_total"] += 1
            if r.get("diagnosis_changed_from_control"):
                label_data[gt]["flips"] += 1

        per_label = {}
        for label, d in label_data.items():
            per_label[label] = {
                "accuracy": d["correct"] / d["total"] if d["total"] > 0 else 0.0,
                "flip_rate": d["flips"] / d["pert_total"] if d["pert_total"] > 0 else 0.0,
                "n_cases": d["total"],
            }

        # ── 10. Spurious Correlation Score (per case, then average) ──
        case_scs = []
        by_case = defaultdict(list)
        for r in rows:
            by_case[r.get("case_id", "")].append(r)
        for case_id, case_rows in by_case.items():
            baseline = next((r for r in case_rows if r.get("condition") == "original"), None)
            if not baseline:
                continue
            b_conf = baseline.get("confidence") or 0.0
            case_perts = [r for r in case_rows if r.get("condition") not in BASELINE_CONDITIONS]
            if case_perts:
                case_scs.append(compute_scs(case_perts, b_conf))
        avg_scs = sum(case_scs) / len(case_scs) if case_scs else 0.0

        # ── 11. Modality-drop accuracy ──
        img_only_acc = 0.0
        if image_only:
            img_correct = sum(1 for r in image_only
                              if (r.get("diagnosis") or "").lower().strip() == (r.get("ground_truth") or "").lower().strip())
            img_only_acc = img_correct / len(image_only)
        txt_only_acc = 0.0
        if text_only:
            txt_correct = sum(1 for r in text_only
                              if (r.get("diagnosis") or "").lower().strip() == (r.get("ground_truth") or "").lower().strip())
            txt_only_acc = txt_correct / len(text_only)

        # ── 12. Average baseline confidence ──
        baseline_confs = [r.get("confidence") or 0 for r in originals]
        avg_baseline_conf = sum(baseline_confs) / len(baseline_confs) if baseline_confs else 0.0

        params = MODEL_PARAMS.get(model, 0)
        display = MODEL_DISPLAY.get(model, model)

        results[model] = {
            "display_name": display,
            "param_count": params,
            "param_label": format_params(params),
            "n_cases": n_cases,
            "n_perturbations": n_pert,
            "accuracy": accuracy,
            "flip_rate": flip_rate,
            "avg_conf_delta": avg_conf_delta,
            "hallucination_rate": halluc_rate,
            "parse_fail_rate": parse_fail_rate,
            "avg_baseline_conf": avg_baseline_conf,
            "neg_control_flip_rate": neg_flip_rate,
            "avg_scs": avg_scs,
            "image_only_accuracy": img_only_acc,
            "text_only_accuracy": txt_only_acc,
            "cat_flip_rates": cat_flip_rates,
            "per_condition": per_condition,
            "per_label": per_label,
        }

    return results


# ═══════════════════════════════════════════════════════════════════
# FORMATTING
# ═══════════════════════════════════════════════════════════════════

def format_params(n):
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.0f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.0f}M"
    return str(n)


def pct(val):
    return f"{val * 100:.1f}%"


# ═══════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════

def print_report(results):
    if not results:
        print("No results to analyze.")
        return

    sorted_models = sorted(results.items(), key=lambda x: x[1]["param_count"])

    print()
    print("=" * 90)
    print("  EXPERIMENT 2: SCALE vs ROBUSTNESS ANALYSIS")
    print("=" * 90)

    # ── Main comparison table ──
    print()
    h = f"{'Model':<22} {'Params':>7} {'Cases':>6} {'Acc':>7} {'Flip':>7} {'SCS':>7} {'Halluc':>7} {'|dConf|':>8} {'PFail':>7}"
    print(h)
    print("-" * len(h))
    for model, m in sorted_models:
        print(
            f"{m['display_name']:<22} "
            f"{m['param_label']:>7} "
            f"{m['n_cases']:>6} "
            f"{pct(m['accuracy']):>7} "
            f"{pct(m['flip_rate']):>7} "
            f"{m['avg_scs']:>7.3f} "
            f"{pct(m['hallucination_rate']):>7} "
            f"{m['avg_conf_delta']:>8.3f} "
            f"{pct(m['parse_fail_rate']):>7}"
        )

    # ── Modality drop ──
    print()
    print("-" * 70)
    print("  MODALITY-DROP ACCURACY")
    print("-" * 70)
    h2 = f"{'Model':<22} {'Full (img+txt)':>14} {'Image-only':>12} {'Text-only':>12}"
    print(h2)
    print("-" * len(h2))
    for model, m in sorted_models:
        print(
            f"{m['display_name']:<22} "
            f"{pct(m['accuracy']):>14} "
            f"{pct(m['image_only_accuracy']):>12} "
            f"{pct(m['text_only_accuracy']):>12}"
        )

    # ── Category flip rates ──
    print()
    print("-" * 70)
    print("  FLIP RATE BY PERTURBATION CATEGORY")
    print("-" * 70)
    categories = sorted(set(cat for m in results.values() for cat in m["cat_flip_rates"]))
    ch = f"{'Model':<22}" + "".join(f" {cat:>12}" for cat in categories)
    print(ch)
    print("-" * len(ch))
    for model, m in sorted_models:
        row = f"{m['display_name']:<22}"
        for cat in categories:
            row += f" {pct(m['cat_flip_rates'].get(cat, 0)):>12}"
        print(row)

    # ── Per-condition breakdown (top 10 most destabilizing) ──
    print()
    print("-" * 70)
    print("  MOST DESTABILIZING CONDITIONS (averaged across models)")
    print("-" * 70)
    cond_avg = defaultdict(list)
    for m in results.values():
        for cond, data in m["per_condition"].items():
            cond_avg[cond].append(data["flip_rate"])
    cond_ranked = sorted(cond_avg.items(), key=lambda x: -sum(x[1]) / len(x[1]))[:10]
    for cond, rates in cond_ranked:
        avg_rate = sum(rates) / len(rates)
        per_model = "  ".join(
            f"{results[m]['display_name'][:10]}={pct(results[m]['per_condition'].get(cond, {}).get('flip_rate', 0))}"
            for m in sorted(results.keys(), key=lambda x: results[x]["param_count"])
        )
        print(f"  {cond:<25} avg={pct(avg_rate):>7}   {per_model}")

    # ── Negative control ──
    print()
    print("-" * 70)
    print("  NEGATIVE CONTROL FLIP RATE (should be ~0%)")
    print("-" * 70)
    for model, m in sorted_models:
        print(f"  {m['display_name']:<22} {pct(m['neg_control_flip_rate'])}")

    # ── Key findings ──
    print()
    print("-" * 70)
    print("  KEY FINDINGS")
    print("-" * 70)

    if len(sorted_models) >= 2:
        smallest = sorted_models[0][1]
        largest = sorted_models[-1][1]

        if largest["flip_rate"] < smallest["flip_rate"]:
            print(f"  Scale helps robustness: {largest['display_name']} "
                  f"({pct(largest['flip_rate'])} flip) vs "
                  f"{smallest['display_name']} ({pct(smallest['flip_rate'])} flip)")
        else:
            print(f"  Scale does NOT help robustness: {largest['display_name']} "
                  f"({pct(largest['flip_rate'])} flip) vs "
                  f"{smallest['display_name']} ({pct(smallest['flip_rate'])} flip)")

        if largest["accuracy"] > smallest["accuracy"]:
            print(f"  Scale helps accuracy: {largest['display_name']} "
                  f"({pct(largest['accuracy'])}) vs {smallest['display_name']} ({pct(smallest['accuracy'])})")

        for model, m in sorted_models:
            img_r = m["cat_flip_rates"].get("Image", 0)
            txt_r = m["cat_flip_rates"].get("Text", 0)
            if txt_r > img_r * 1.5:
                print(f"  {m['display_name']}: text perturbations {pct(txt_r)} flip vs image {pct(img_r)}")

        # SCS comparison
        scs_sorted = sorted(sorted_models, key=lambda x: x[1]["avg_scs"], reverse=True)
        highest_scs = scs_sorted[0][1]
        lowest_scs = scs_sorted[-1][1]
        if len(scs_sorted) >= 2:
            print(f"  Highest SCS: {highest_scs['display_name']} ({highest_scs['avg_scs']:.3f})")
            print(f"  Lowest SCS:  {lowest_scs['display_name']} ({lowest_scs['avg_scs']:.3f})")

    print()


# ═══════════════════════════════════════════════════════════════════
# CSV / JSON OUTPUT
# ═══════════════════════════════════════════════════════════════════

def write_outputs(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Main comparison CSV ──
    main_csv = os.path.join(output_dir, "experiment2_model_comparison.csv")
    with open(main_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "model", "display_name", "param_count", "param_label",
            "n_cases", "n_perturbations", "accuracy", "flip_rate", "avg_scs",
            "avg_conf_delta", "hallucination_rate", "parse_fail_rate",
            "avg_baseline_confidence", "neg_control_flip_rate",
            "image_only_accuracy", "text_only_accuracy",
            "image_flip_rate", "text_flip_rate", "control_flip_rate",
        ])
        for model, m in sorted(results.items(), key=lambda x: x[1]["param_count"]):
            w.writerow([
                model, m["display_name"], m["param_count"], m["param_label"],
                m["n_cases"], m["n_perturbations"],
                round(m["accuracy"], 4), round(m["flip_rate"], 4), round(m["avg_scs"], 4),
                round(m["avg_conf_delta"], 4), round(m["hallucination_rate"], 4),
                round(m["parse_fail_rate"], 4), round(m["avg_baseline_conf"], 4),
                round(m["neg_control_flip_rate"], 4),
                round(m["image_only_accuracy"], 4), round(m["text_only_accuracy"], 4),
                round(m["cat_flip_rates"].get("Image", 0), 4),
                round(m["cat_flip_rates"].get("Text", 0), 4),
                round(m["cat_flip_rates"].get("Control", 0), 4),
            ])
    print(f"  Wrote {main_csv}")

    # ── 2. Per-condition CSV ──
    cond_csv = os.path.join(output_dir, "experiment2_per_condition.csv")
    all_conditions = sorted(set(
        c for m in results.values() for c in m["per_condition"]
    ))
    with open(cond_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "category"] + [
            results[m]["display_name"]
            for m in sorted(results.keys(), key=lambda x: results[x]["param_count"])
        ])
        for cond in all_conditions:
            row = [cond, get_category(cond)]
            for model in sorted(results.keys(), key=lambda x: results[x]["param_count"]):
                rate = results[model]["per_condition"].get(cond, {}).get("flip_rate", "")
                row.append(round(rate, 4) if rate != "" else "")
            w.writerow(row)
    print(f"  Wrote {cond_csv}")

    # ── 3. Per-label CSV ──
    label_csv = os.path.join(output_dir, "experiment2_per_label.csv")
    all_labels = sorted(set(
        l for m in results.values() for l in m["per_label"]
    ))
    with open(label_csv, "w", newline="") as f:
        w = csv.writer(f)
        header = ["label"]
        for model in sorted(results.keys(), key=lambda x: results[x]["param_count"]):
            dn = results[model]["display_name"]
            header.extend([f"{dn}_accuracy", f"{dn}_flip_rate", f"{dn}_n_cases"])
        w.writerow(header)
        for label in all_labels:
            row = [label]
            for model in sorted(results.keys(), key=lambda x: results[x]["param_count"]):
                ld = results[model]["per_label"].get(label, {})
                row.extend([
                    round(ld.get("accuracy", 0), 4),
                    round(ld.get("flip_rate", 0), 4),
                    ld.get("n_cases", 0),
                ])
            w.writerow(row)
    print(f"  Wrote {label_csv}")

    # ── 4. Full JSON summary ──
    json_path = os.path.join(output_dir, "experiment2_summary.json")
    json_safe = {}
    for model, m in results.items():
        json_safe[model] = {k: v for k, v in m.items()}
    with open(json_path, "w") as f:
        json.dump(json_safe, f, indent=2, default=str)
    print(f"  Wrote {json_path}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Experiment 2: Scale vs Robustness")
    parser.add_argument("--results", required=True, help="Path to Experiment 1 results directory")
    parser.add_argument("--output", default="results/experiment2", help="Output directory")
    args = parser.parse_args()

    print(f"  Loading results from {args.results} ...")
    records = load_all_results(args.results)
    print(f"  Loaded {len(records)} records")

    records = deduplicate(records)
    print(f"  After deduplication: {len(records)} unique records")

    if not records:
        print("  No records found. Check the path.")
        sys.exit(1)

    metrics = compute_metrics(records)
    print_report(metrics)
    write_outputs(metrics, args.output)
    print(f"\n  Done. Outputs in {args.output}/")


if __name__ == "__main__":
    main()

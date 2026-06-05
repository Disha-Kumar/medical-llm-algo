#!/usr/bin/env python3
import json, csv, argparse, numpy as np
from collections import defaultdict
from pathlib import Path

def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                try: records.append(json.loads(line.strip()))
                except: pass
    return records

def load_csv_file(path):
    records = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            records.append(row)
    return records

def norm_model(name):
    n = (name or "").lower().strip()
    if "gpt4o" in n: return "GPT-4o"
    if "qwen" in n: return "Qwen2-VL"
    if "llava" in n: return "LLaVA-Med"
    if "biovil" in n: return "BioViL-T"
    if "chexagent" in n: return "CheXagent"
    return n

def norm_dx(dx):
    return (dx or "").lower().strip()

SKIP = {"original", "image_only", "text_only", "noise_floor_0", "noise_floor_1", "noise_floor_2"}
MODEL_ORDER = ["BioViL-T", "Qwen2-VL", "LLaVA-Med", "CheXagent", "GPT-4o"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print(f"Loading from {args.results} ...")
    records = []
    for path in Path(args.results).rglob("*"):
        if ".ipynb_checkpoints" in path.parts:
            continue
        if path.suffix == ".jsonl":
            records.extend(load_jsonl(path))
        elif path.suffix == ".csv":
            records.extend(load_csv_file(path))
    print(f"Loaded {len(records)} records")

    by_model = defaultdict(lambda: defaultdict(list))
    for r in records:
        cid = r.get("case_id", "")
        model = norm_model(r.get("model", ""))
        if cid and model:
            by_model[model][cid].append(r)

    print("\n" + "=" * 80)
    print("  BOOTSTRAP 95% CONFIDENCE INTERVALS FOR FLIP RATES")
    print("=" * 80)

    all_results = {}

    for model in MODEL_ORDER:
        if model not in by_model:
            continue
        cases = by_model[model]

        per_case_flip_rates = []
        per_case_neg_ctrl_flips = []

        for case_id, rows in cases.items():
            bl = next((r for r in rows if r.get("condition") == "original"), None)
            if not bl:
                continue
            b_dx = norm_dx(bl.get("diagnosis"))

            perts = [r for r in rows if r.get("condition") not in SKIP and r.get("condition") != "negative_control"]
            neg_ctrl = [r for r in rows if r.get("condition") == "negative_control"]

            if not perts:
                continue

            flips = sum(1 for r in perts if norm_dx(r.get("diagnosis")) != b_dx)
            flip_rate = flips / len(perts)
            per_case_flip_rates.append(flip_rate)

            if neg_ctrl:
                nc_flip = 1 if norm_dx(neg_ctrl[0].get("diagnosis")) != b_dx else 0
                per_case_neg_ctrl_flips.append(nc_flip)

        n_cases = len(per_case_flip_rates)
        if n_cases == 0:
            continue

        observed_flip = np.mean(per_case_flip_rates)
        observed_neg_ctrl = np.mean(per_case_neg_ctrl_flips) if per_case_neg_ctrl_flips else 0.0

        flip_arr = np.array(per_case_flip_rates)
        neg_arr = np.array(per_case_neg_ctrl_flips) if per_case_neg_ctrl_flips else None

        boot_flips = []
        boot_neg = []
        boot_adjusted = []

        for _ in range(args.n_bootstrap):
            idx = np.random.randint(0, n_cases, size=n_cases)
            boot_flip = np.mean(flip_arr[idx])
            boot_flips.append(boot_flip)

            if neg_arr is not None and len(neg_arr) > 0:
                idx_neg = np.random.randint(0, len(neg_arr), size=len(neg_arr))
                boot_nc = np.mean(neg_arr[idx_neg])
                boot_neg.append(boot_nc)
                boot_adjusted.append(boot_flip - boot_nc)

        ci_lower = np.percentile(boot_flips, 2.5)
        ci_upper = np.percentile(boot_flips, 97.5)

        adjusted_flip = observed_flip - observed_neg_ctrl

        if boot_adjusted:
            adj_ci_lower = np.percentile(boot_adjusted, 2.5)
            adj_ci_upper = np.percentile(boot_adjusted, 97.5)
        else:
            adj_ci_lower = adjusted_flip
            adj_ci_upper = adjusted_flip

        if boot_neg:
            neg_ci_lower = np.percentile(boot_neg, 2.5)
            neg_ci_upper = np.percentile(boot_neg, 97.5)
        else:
            neg_ci_lower = observed_neg_ctrl
            neg_ci_upper = observed_neg_ctrl

        all_results[model] = {
            "n_cases": n_cases,
            "raw_flip": observed_flip,
            "raw_ci": (ci_lower, ci_upper),
            "neg_ctrl": observed_neg_ctrl,
            "neg_ci": (neg_ci_lower, neg_ci_upper),
            "adjusted_flip": adjusted_flip,
            "adjusted_ci": (adj_ci_lower, adj_ci_upper),
        }

        print(f"\n  {model} ({n_cases} cases)")
        print(f"    Raw flip rate:      {observed_flip*100:.1f}%  95% CI [{ci_lower*100:.1f}%, {ci_upper*100:.1f}%]")
        print(f"    Neg control rate:   {observed_neg_ctrl*100:.1f}%  95% CI [{neg_ci_lower*100:.1f}%, {neg_ci_upper*100:.1f}%]")
        print(f"    Adjusted flip rate: {adjusted_flip*100:.1f}%  95% CI [{adj_ci_lower*100:.1f}%, {adj_ci_upper*100:.1f}%]")

    print("\n" + "=" * 80)
    print("  PAIRWISE COMPARISONS (do CIs overlap?)")
    print("=" * 80)

    models = [m for m in MODEL_ORDER if m in all_results]
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m1, m2 = models[i], models[j]
            r1, r2 = all_results[m1], all_results[m2]
            raw_overlap = r1["raw_ci"][0] <= r2["raw_ci"][1] and r2["raw_ci"][0] <= r1["raw_ci"][1]
            adj_overlap = r1["adjusted_ci"][0] <= r2["adjusted_ci"][1] and r2["adjusted_ci"][0] <= r1["adjusted_ci"][1]
            print(f"\n  {m1} vs {m2}")
            print(f"    Raw:      {r1['raw_flip']*100:.1f}% [{r1['raw_ci'][0]*100:.1f},{r1['raw_ci'][1]*100:.1f}] vs {r2['raw_flip']*100:.1f}% [{r2['raw_ci'][0]*100:.1f},{r2['raw_ci'][1]*100:.1f}]  {'OVERLAP' if raw_overlap else 'SEPARATED'}")
            print(f"    Adjusted: {r1['adjusted_flip']*100:.1f}% [{r1['adjusted_ci'][0]*100:.1f},{r1['adjusted_ci'][1]*100:.1f}] vs {r2['adjusted_flip']*100:.1f}% [{r2['adjusted_ci'][0]*100:.1f},{r2['adjusted_ci'][1]*100:.1f}]  {'OVERLAP' if adj_overlap else 'SEPARATED'}")

    print("\n" + "=" * 80)
    print("  SUMMARY TABLE")
    print("=" * 80)
    print(f"\n  {'Model':<15} {'Raw Flip':>10} {'95% CI':>20} {'Neg Ctrl':>10} {'Adjusted':>10} {'Adj 95% CI':>20}")
    print("  " + "-" * 85)
    for model in MODEL_ORDER:
        if model not in all_results:
            continue
        r = all_results[model]
        print(f"  {model:<15} {r['raw_flip']*100:>9.1f}% [{r['raw_ci'][0]*100:.1f}, {r['raw_ci'][1]*100:.1f}] {r['neg_ctrl']*100:>9.1f}% {r['adjusted_flip']*100:>9.1f}% [{r['adjusted_ci'][0]*100:.1f}, {r['adjusted_ci'][1]*100:.1f}]")

if __name__ == "__main__":
    main()

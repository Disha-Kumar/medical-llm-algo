#!/usr/bin/env python3
"""
Auto-annotator: generates annotation JSON files by analyzing
Experiment 1 JSONL results programmatically.

Skips cases that already have annotation files.

Usage:
    python3 scripts/auto_annotate.py --results results/experiment1 --annotator muhammad
"""

import argparse
import json
import os
import re
import csv
from collections import defaultdict
from pathlib import Path
from datetime import datetime


BASELINE_CONDITIONS = {
    "original", "image_only", "text_only",
    "noise_floor_0", "noise_floor_1", "noise_floor_2",
}

IMG_PREFIXES = ["watermark", "jpeg", "chest_tube", "chest_drain", "ecg_leads", "pacemaker"]
TXT_PREFIXES = ["demographic", "contradiction", "paraphrase"]

TIER1_LABELS = {"Y": "yes -- label flipped in at least one variant",
                "N": "no  -- label stayed the same across all variants",
                "P": "partial -- label flipped in some but not all variants"}
TIER2_LABELS = {"D": "dropped -- confidence dropped noticeably",
                "I": "increased -- confidence increased noticeably",
                "B": "both  -- dropped in some variants, increased in others",
                "N": "stable -- confidence stayed roughly the same"}
TIER3_LABELS = {"Y": "yes -- explanation shifted meaningfully",
                "N": "no  -- explanations stayed consistent",
                "P": "partial -- some explanations changed, others did not"}
CAUSE_LABELS = {"S": "spurious-caused", "C": "capability-caused",
                "B": "both", "U": "unclear", "N": "not applicable"}


def normalize_model(name):
    name = name.lower().strip()
    if "gpt4o" in name or "gpt-4o" in name: return "gpt4o"
    if "qwen" in name: return "qwen2_vl"
    if "llava" in name: return "llava_med"
    if "biovil" in name: return "biovil_t"
    if "flamingo" in name: return "med_flamingo"
    if "chexagent" in name: return "chexagent"
    return name


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: records.append(json.loads(line))
            except: continue
    return records


def load_csv_file(path):
    records = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            for field in ["confidence", "confidence_delta", "control_confidence"]:
                if field in row and row[field]:
                    try: row[field] = float(row[field])
                    except: row[field] = 0.0
            for field in ["hallucination_flag", "diagnosis_changed_from_control"]:
                if field in row:
                    row[field] = str(row[field]).strip().lower() in ("true", "1", "yes")
            records.append(row)
    return records


def load_all(results_dir):
    records = []
    for path in Path(results_dir).rglob("*"):
        if ".ipynb_checkpoints" in path.parts: continue
        if path.suffix == ".jsonl": records.extend(load_jsonl(path))
        elif path.suffix == ".csv": records.extend(load_csv_file(path))
    return records


def annotate_case(case_id, models_data):
    """Analyze a single case across all models and return annotation dict."""

    any_flip = False
    total_flips = 0
    total_variants = 0
    conf_drops = False
    conf_rises = False
    neg_control_flip = False
    baseline_wrong = {}
    all_flip_conds = defaultdict(int)
    img_flips = []
    txt_flips = []
    model_names = []
    note_parts = []

    for model_name, rows in sorted(models_data.items()):
        display = {"gpt4o": "GPT-4o", "qwen2_vl": "Qwen2-VL", "llava_med": "LLaVA-Med",
                    "biovil_t": "BioViL-T"}.get(model_name, model_name)
        model_names.append(model_name)

        baseline = next((r for r in rows if r.get("condition") == "original"), None)
        if not baseline: continue

        b_dx = (baseline.get("diagnosis") or "").lower().strip()
        b_conf = baseline.get("confidence") or 0
        gt = (baseline.get("ground_truth") or "").lower().strip()
        is_wrong = b_dx != gt
        baseline_wrong[model_name] = is_wrong

        variants = [r for r in rows if r.get("condition") not in BASELINE_CONDITIONS]
        total_variants += len(variants)

        model_img_flips = []
        model_txt_flips = []
        model_flips = 0

        for v in variants:
            v_dx = (v.get("diagnosis") or "").lower().strip()
            v_conf = v.get("confidence") or 0
            cond = v.get("condition") or ""
            flipped = v_dx != b_dx

            if flipped:
                model_flips += 1
                total_flips += 1
                any_flip = True
                all_flip_conds[cond] += 1
                if cond == "negative_control":
                    neg_control_flip = True
                if any(cond.startswith(p) for p in IMG_PREFIXES):
                    model_img_flips.append(cond)
                    img_flips.append(cond)
                elif any(cond.startswith(p) for p in TXT_PREFIXES):
                    model_txt_flips.append(cond)
                    txt_flips.append(cond)

            delta = (v_conf - b_conf) if v_conf is not None and b_conf is not None else 0
            if delta <= -0.1: conf_drops = True
            if delta >= 0.1: conf_rises = True

        # Build per-model note
        if model_flips == 0:
            note_parts.append(f"{display} completely stable across all perturbations")
        else:
            has_img = len(model_img_flips) > 0
            has_txt = len(model_txt_flips) > 0
            if has_img and not has_txt:
                img_types = set()
                for c in model_img_flips:
                    for p in IMG_PREFIXES:
                        if c.startswith(p): img_types.add(p.replace("_", " "))
                note_parts.append(f"{display} flipped on image perturbations ({', '.join(sorted(img_types))}) but stable on text")
            elif has_txt and not has_img:
                txt_types = set()
                for c in model_txt_flips:
                    for p in TXT_PREFIXES:
                        if c.startswith(p): txt_types.add(p)
                note_parts.append(f"{display} stable on image perturbations but flipped on text ({', '.join(sorted(txt_types))})")
            elif has_img and has_txt:
                if len(model_img_flips) > len(model_txt_flips) * 2:
                    note_parts.append(f"{display} flipped heavily on image perturbations ({len(model_img_flips)} flips) with some text flips ({len(model_txt_flips)})")
                elif len(model_txt_flips) > len(model_img_flips) * 2:
                    note_parts.append(f"{display} flipped mainly on text perturbations ({len(model_txt_flips)} flips) with some image flips ({len(model_img_flips)})")
                else:
                    note_parts.append(f"{display} unstable across both image ({len(model_img_flips)} flips) and text ({len(model_txt_flips)} flips)")

        if is_wrong:
            note_parts.append(f"{display} baseline was already wrong (said {baseline.get('diagnosis', '?')} vs ground truth {baseline.get('ground_truth', '?')})")
        if neg_control_flip and cond == "negative_control":
            note_parts.append(f"{display} flipped on negative control which suggests general instability")

    # Compute tiers
    if total_variants == 0:
        return None

    flip_rate = total_flips / total_variants
    t1 = "N" if total_flips == 0 else ("Y" if flip_rate > 0.7 else "P")

    if conf_drops and conf_rises: t2 = "B"
    elif conf_drops: t2 = "D"
    elif conf_rises: t2 = "I"
    else: t2 = "N"

    t3 = "N" if not any_flip else ("Y" if flip_rate > 0.7 else "P")

    any_wrong = any(baseline_wrong.values())
    if not any_flip:
        fc = "N"
    elif not any_wrong:
        fc = "S"
    else:
        fc = "B"

    # Top triggers
    top = sorted(all_flip_conds.items(), key=lambda x: -x[1])[:5]
    trigger_str = ", ".join(t[0] for t in top) if top else "none"

    notes = ". ".join(note_parts)

    return {
        "annotator_id": None,  # filled in later
        "case_id": case_id,
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "models_reviewed": model_names,
        "tier1_label_flip": t1,
        "tier1_label_flip_label": TIER1_LABELS[t1],
        "tier2_confidence_change": t2,
        "tier2_confidence_change_label": TIER2_LABELS[t2],
        "tier3_explanation_shift": t3,
        "tier3_explanation_shift_label": TIER3_LABELS[t3],
        "failure_cause": fc,
        "failure_cause_label": CAUSE_LABELS[fc],
        "trigger_perturbations": trigger_str,
        "notes": notes,
    }


def main():
    parser = argparse.ArgumentParser(description="Auto-annotate cases")
    parser.add_argument("--results", required=True)
    parser.add_argument("--annotator", required=True)
    args = parser.parse_args()

    out_dir = os.path.join("annotations", args.annotator)
    os.makedirs(out_dir, exist_ok=True)

    # Find already-done annotations
    existing = set()
    for f in os.listdir(out_dir):
        if f.endswith(".json"):
            existing.add(f.replace(".json", ""))

    print(f"  Loading results from {args.results} ...")
    records = load_all(args.results)
    print(f"  Loaded {len(records)} records")

    # Group by case_id and model
    cases = defaultdict(lambda: defaultdict(list))
    for r in records:
        model = normalize_model(r.get("model", "unknown"))
        case_id = r.get("case_id", "")
        if case_id:
            cases[case_id][model].append(r)

    print(f"  Found {len(cases)} unique cases across {len(set(normalize_model(r.get('model','')) for r in records))} models")
    print(f"  Already annotated: {len(existing)}")

    new_count = 0
    skipped = 0
    for case_id in sorted(cases.keys()):
        if case_id in existing:
            skipped += 1
            continue

        annotation = annotate_case(case_id, cases[case_id])
        if annotation is None:
            continue

        annotation["annotator_id"] = args.annotator

        out_path = os.path.join(out_dir, f"{case_id}.json")
        with open(out_path, "w") as f:
            json.dump(annotation, f, indent=4)
        new_count += 1

    print(f"  Skipped (already done): {skipped}")
    print(f"  New annotations created: {new_count}")
    print(f"  Total annotations: {len(existing) + new_count}")
    print(f"  Saved to {out_dir}/")


if __name__ == "__main__":
    main()

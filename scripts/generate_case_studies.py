#!/usr/bin/env python3
"""Generate 20 qualitative case studies from Experiment 1 results.
Run from the repo root: python3 scripts/generate_case_studies.py --results results/experiment1
"""
import json, os, csv, argparse
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
            for field in ["confidence", "confidence_delta"]:
                if field in row and row[field]:
                    try: row[field] = float(row[field])
                    except: row[field] = 0.0
            for field in ["hallucination_flag", "diagnosis_changed_from_control"]:
                if field in row:
                    row[field] = str(row[field]).strip().lower() in ("true", "1", "yes")
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

SKIP = {"original","image_only","text_only","noise_floor_0","noise_floor_1","noise_floor_2"}
IMG_P = ["watermark","jpeg","chest_tube","chest_drain","ecg_leads","pacemaker"]
TXT_P = ["demographic","contradiction","paraphrase"]
MODEL_ORDER = ["BioViL-T","Qwen2-VL","LLaVA-Med","GPT-4o","CheXagent"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", default="results/experiment2/qualitative_case_studies.txt")
    args = parser.parse_args()

    print(f"  Loading from {args.results} ...")
    records = []
    for path in Path(args.results).rglob("*"):
        if ".ipynb_checkpoints" in path.parts: continue
        if path.suffix == ".jsonl": records.extend(load_jsonl(path))
        elif path.suffix == ".csv": records.extend(load_csv_file(path))
    print(f"  Loaded {len(records)} records")

    cases = defaultdict(lambda: defaultdict(list))
    for r in records:
        cid = r.get("case_id","")
        if cid: cases[cid][norm_model(r.get("model",""))].append(r)

    scored = []
    for case_id, models in cases.items():
        score = 0
        info = {}
        for model, rows in models.items():
            bl = next((r for r in rows if r.get("condition") == "original"), None)
            if not bl: continue
            b_dx = norm_dx(bl.get("diagnosis"))
            gt = norm_dx(bl.get("ground_truth"))
            correct = b_dx == gt
            b_conf = bl.get("confidence") or 0
            perts = [r for r in rows if r.get("condition") not in SKIP]
            flips = []
            for r in perts:
                if norm_dx(r.get("diagnosis")) != b_dx:
                    flips.append({"cond": r.get("condition",""), "new_dx": r.get("diagnosis",""), "new_conf": r.get("confidence",0)})
            if correct and flips: score += 10 + len(flips)
            elif not correct and flips: score += len(flips)
            img_f = [f for f in flips if any(f["cond"].startswith(p) for p in IMG_P)]
            txt_f = [f for f in flips if any(f["cond"].startswith(p) for p in TXT_P)]
            neg_f = [f for f in flips if f["cond"] == "negative_control"]
            info[model] = {
                "baseline_dx": bl.get("diagnosis",""), "gt": bl.get("ground_truth",""),
                "correct": correct, "conf": b_conf, "n_perts": len(perts),
                "n_flips": len(flips), "img_flips": img_f, "txt_flips": txt_f,
                "neg_flip": bool(neg_f),
                "flipped_dxs": list(set(f["new_dx"] for f in flips if f["new_dx"] and norm_dx(f["new_dx"]) != b_dx)),
            }
        if len(info) >= 2:
            fcs = [v["n_flips"] for v in info.values()]
            if max(fcs) > 5 and min(fcs) == 0: score += 8
        has_spurious = any(v["correct"] and v["n_flips"] > 0 for v in info.values())
        if has_spurious and info:
            scored.append((case_id, score, info))

    scored.sort(key=lambda x: -x[1])
    top20 = scored[:20]

    all_models = sorted(set(m for _,_,info in top20 for m in info.keys()), key=lambda x: MODEL_ORDER.index(x) if x in MODEL_ORDER else 99)
    print(f"  Models found: {', '.join(all_models)}")
    print(f"  Writing {len(top20)} case studies...")

    out = []
    out.append("QUALITATIVE CASE STUDIES")
    out.append("20 Cases Demonstrating Spurious Diagnosis Flips in Medical VLMs")
    out.append("=" * 70)
    out.append("")
    out.append("Selection criteria: cases where at least one model produced a correct")
    out.append("diagnosis on the unperturbed baseline but changed its diagnosis when")
    out.append("a non-clinical perturbation was applied. Cases ranked by number of")
    out.append("models affected and total flips observed.")
    out.append("")

    for i, (case_id, score, info) in enumerate(top20, 1):
        gt = list(info.values())[0]["gt"]
        out.append(f"{'='*70}")
        out.append(f"CASE {i}: {case_id}")
        out.append(f"Ground truth diagnosis: {gt}")
        out.append("")
        for model in MODEL_ORDER:
            if model not in info: continue
            d = info[model]
            tag = "CORRECT" if d["correct"] else "INCORRECT"
            out.append(f"  {model}")
            out.append(f"    Baseline: {d['baseline_dx']} ({tag}, confidence {d['conf']})")
            if d["n_flips"] == 0:
                out.append(f"    Result: Stable across all {d['n_perts']} perturbation conditions.")
            else:
                out.append(f"    Result: Flipped on {d['n_flips']}/{d['n_perts']} perturbation conditions.")
                if d["img_flips"]:
                    conds = [f["cond"] for f in d["img_flips"]]
                    out.append(f"      Image perturbations ({len(d['img_flips'])} flips): {', '.join(conds[:6])}")
                if d["txt_flips"]:
                    conds = [f["cond"] for f in d["txt_flips"]]
                    out.append(f"      Text perturbations ({len(d['txt_flips'])} flips): {', '.join(conds[:6])}")
                if d["neg_flip"]:
                    out.append(f"      Negative control: FLIPPED (1-degree rotation caused diagnosis change)")
                if d["flipped_dxs"]:
                    out.append(f"      Changed to: {', '.join(d['flipped_dxs'][:4])}")
            out.append("")

        narr = []
        correct_flipped = [(m,d) for m,d in info.items() if d["correct"] and d["n_flips"] > 0]
        correct_stable = [(m,d) for m,d in info.items() if d["correct"] and d["n_flips"] == 0]
        wrong = [(m,d) for m,d in info.items() if not d["correct"]]

        for m, d in correct_flipped:
            dxs = ', '.join(d['flipped_dxs'][:2]) if d['flipped_dxs'] else 'other conditions'
            if d["txt_flips"] and not d["img_flips"]:
                txt_types = list(set(f["cond"].rsplit("_",1)[0] for f in d["txt_flips"]))
                narr.append(f"{m} correctly diagnosed {gt} at {d['conf']} confidence but changed its diagnosis to {dxs} when the clinical note was modified ({', '.join(txt_types)}). The X-ray image was unchanged, meaning the model abandoned a correct diagnosis based solely on non-clinical text changes.")
            elif d["img_flips"] and not d["txt_flips"]:
                img_types = list(set(f["cond"].rsplit("_",1)[0] for f in d["img_flips"]))
                narr.append(f"{m} correctly diagnosed {gt} at {d['conf']} confidence but changed its diagnosis to {dxs} when non-clinical image artifacts were added ({', '.join(img_types[:3])}). These artifacts have no diagnostic significance, yet they altered the clinical conclusion.")
            elif d["txt_flips"] and d["img_flips"]:
                narr.append(f"{m} correctly diagnosed {gt} at {d['conf']} confidence but was fragile across both modalities, flipping on {len(d['img_flips'])} image and {len(d['txt_flips'])} text perturbations to diagnoses including {dxs}.")
            if d["neg_flip"]:
                narr.append(f"{m} also flipped on the negative control (1-degree rotation), indicating baseline instability beyond perturbation-specific effects.")

        for m, d in correct_stable:
            narr.append(f"{m} correctly diagnosed {gt} and remained stable across all {d['n_perts']} perturbation conditions.")
        for m, d in wrong:
            narr.append(f"{m} was already incorrect on baseline, diagnosing {d['baseline_dx']} instead of {gt}.")

        if correct_flipped:
            first_dx = correct_flipped[0][1]['flipped_dxs'][0] if correct_flipped[0][1]['flipped_dxs'] else 'another condition'
            narr.append(f"Clinical implication: in a deployed system, a patient with {gt} could receive an incorrect diagnosis of {first_dx} simply because the clinical note was reworded or a scanner artifact appeared on the X-ray, potentially leading to inappropriate treatment or missed pathology.")

        out.append("  NARRATIVE")
        for line in narr:
            out.append(f"  {line}")
        out.append("")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(out))
    print(f"  Saved to {args.output}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate 20 qualitative case studies with detailed, case-specific clinical implications."""
import json, os, csv, argparse
from collections import defaultdict
from pathlib import Path

CLINICAL_IMPLICATIONS = {
    ("pneumonia", "no finding"): "Missed pneumonia delays initiation of antibiotic therapy. In elderly or immunocompromised patients, untreated pneumonia can progress to sepsis, respiratory failure, or death within 48-72 hours.",
    ("pneumonia", "cardiomegaly"): "Misclassifying pneumonia as cardiomegaly would lead to cardiac workup and diuretic therapy rather than antibiotics, allowing the infection to progress unchecked.",
    ("pneumonia", "lung opacity"): "While lung opacity is a broader category that could encompass pneumonia, the lack of a specific pneumonia diagnosis may delay targeted antibiotic treatment in favor of additional imaging.",
    ("pneumonia", "support devices"): "Attributing the finding to support devices instead of pneumonia means the infectious process is entirely missed, with no antimicrobial therapy initiated.",
    ("pneumonia", "atelectasis"): "Misdiagnosing pneumonia as atelectasis leads to incentive spirometry and repositioning rather than antibiotics, allowing the infection to worsen.",
    ("pneumonia", "pleural effusion"): "A pneumonia misread as pleural effusion could prompt thoracentesis rather than antibiotic therapy, an invasive procedure that does not address the underlying infection.",
    ("cardiomegaly", "no finding"): "Undetected cardiomegaly means echocardiography and heart failure evaluation are never ordered. Progressive cardiac decompensation goes unmonitored until the patient presents with acute heart failure.",
    ("cardiomegaly", "pneumonia"): "Treating cardiomegaly as pneumonia leads to unnecessary antibiotics while cardiac function continues to deteriorate without appropriate ACE inhibitors, beta-blockers, or diuretics.",
    ("cardiomegaly", "lung opacity"): "Cardiomegaly misread as a generic lung opacity could trigger pulmonary workup while the underlying cardiac pathology goes unaddressed.",
    ("cardiomegaly", "support devices"): "Attributing cardiomegaly findings to support devices means the enlarged heart is never flagged, and the patient leaves without cardiac referral.",
    ("cardiomegaly", "enlarged cardiomediastinum"): "While related, the distinction affects whether workup focuses on the heart specifically or broader mediastinal structures including great vessels.",
    ("pleural effusion", "no finding"): "Missed pleural effusion can lead to progressive respiratory compromise. Large effusions require drainage; leaving them undetected risks respiratory failure and missed underlying malignancy.",
    ("pleural effusion", "pneumonia"): "Treating pleural effusion as pneumonia alone misses the need for thoracentesis. If the effusion is malignant or parapneumonic, delayed drainage increases morbidity.",
    ("pleural effusion", "lung opacity"): "Classifying pleural effusion as generic lung opacity may delay the specific decision to perform thoracentesis or investigate for malignancy, tuberculosis, or heart failure.",
    ("pleural effusion", "cardiomegaly"): "Misreading pleural effusion as cardiomegaly shifts the workup to cardiac evaluation while fluid continues to accumulate in the pleural space.",
    ("atelectasis", "no finding"): "Undetected atelectasis in a post-operative patient can progress to pneumonia if incentive spirometry and mobilization are not initiated.",
    ("atelectasis", "pneumonia"): "Misdiagnosing atelectasis as pneumonia leads to unnecessary antibiotics. The actual treatment, repositioning and breathing exercises, is never prescribed.",
    ("atelectasis", "lung opacity"): "Generic lung opacity classification loses the specific atelectasis diagnosis that would prompt targeted respiratory therapy and early mobilization.",
    ("edema", "no finding"): "Missed pulmonary edema in a heart failure patient means diuretics are not administered. Fluid continues to accumulate in the lungs, risking acute respiratory distress.",
    ("edema", "pneumonia"): "Treating pulmonary edema with antibiotics instead of diuretics and fluid restriction allows the edema to worsen, potentially causing respiratory failure.",
    ("edema", "pleural effusion"): "While both involve fluid, management differs: edema requires diuresis and afterload reduction, while isolated pleural effusion may require drainage.",
    ("pneumothorax", "no finding"): "A missed pneumothorax is a medical emergency. Tension pneumothorax can develop rapidly, causing cardiovascular collapse and death if chest tube placement is not performed.",
    ("pneumothorax", "pleural effusion"): "Treating pneumothorax as pleural effusion could lead to fluid drainage attempts in a space filled with air, delaying correct chest tube decompression.",
    ("consolidation", "no finding"): "Missed consolidation delays investigation of its cause, whether infectious, inflammatory, or malignant. Untreated infectious consolidation progresses similarly to untreated pneumonia.",
    ("consolidation", "atelectasis"): "Misreading consolidation as atelectasis shifts management from antimicrobial therapy to respiratory exercises, allowing a potential infection to spread.",
    ("lung opacity", "no finding"): "Dismissing a real lung opacity as normal means the underlying cause, whether infection, malignancy, or fluid, goes uninvestigated.",
    ("lung opacity", "cardiomegaly"): "Attributing a lung finding to cardiac enlargement redirects the workup entirely, potentially missing a pulmonary malignancy or active infection.",
    ("enlarged cardiomediastinum", "no finding"): "Undetected mediastinal widening can mask aortic pathology including aneurysm or dissection, conditions that are fatal if not promptly identified.",
    ("enlarged cardiomediastinum", "cardiomegaly"): "While related, enlarged cardiomediastinum can indicate aortic pathology distinct from cardiomegaly. Focusing on cardiac function alone may miss a developing aortic aneurysm.",
    ("enlarged cardiomediastinum", "support devices"): "Attributing mediastinal widening to support devices means potential aortic pathology is ignored entirely.",
    ("enlarged cardiomediastinum", "lung opacity"): "Misclassifying mediastinal widening as lung opacity shifts investigation to pulmonary causes while vascular pathology goes unexamined.",
    ("support devices", "no finding"): "Failing to identify support devices means a malpositioned line or tube is not recognized, risking pneumothorax from a misplaced central line or aspiration from a displaced feeding tube.",
    ("fracture", "no finding"): "A missed rib fracture in a trauma patient means pain management is inadequate and complications like flail chest or hemothorax are not monitored.",
    ("no finding", "pneumonia"): "A false positive pneumonia diagnosis leads to unnecessary antibiotic prescription, contributing to antimicrobial resistance and potential adverse drug reactions.",
    ("no finding", "cardiomegaly"): "A false positive cardiomegaly triggers unnecessary echocardiography, cardiology referral, and patient anxiety over a non-existent cardiac condition.",
    ("no finding", "pleural effusion"): "A false positive pleural effusion could lead to unnecessary thoracentesis, an invasive procedure with risks of bleeding and pneumothorax, on a patient with no effusion.",
    ("lung lesion", "no finding"): "A missed lung lesion could represent early-stage lung cancer. Delayed detection significantly reduces survival rates, particularly for non-small cell lung carcinoma.",
    ("lung lesion", "lung opacity"): "Downgrading a lung lesion to generic opacity may delay biopsy or CT follow-up that would be standard for a discrete lesion suspicious for malignancy.",
}

def get_clinical_implication(correct_dx, flipped_dx):
    correct = correct_dx.lower().strip()
    flipped = flipped_dx.lower().strip()
    if (correct, flipped) in CLINICAL_IMPLICATIONS:
        return CLINICAL_IMPLICATIONS[(correct, flipped)]
    if flipped in ("no finding", "parse_failed", ""):
        return (f"The true diagnosis of {correct} is entirely missed. No follow-up imaging, treatment, or monitoring is initiated for the actual condition, allowing it to progress undetected.")
    if correct == "no finding":
        return (f"A healthy patient receives an incorrect diagnosis of {flipped}, triggering unnecessary diagnostic workup, potential invasive procedures, and patient distress over a non-existent condition.")
    return (f"The actual condition ({correct}) is misidentified as {flipped}, leading to a management plan designed for {flipped} rather than the treatment required for {correct}. This misalignment between diagnosis and treatment can delay recovery and introduce complications from inappropriate therapy.")

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

def norm_dx(dx): return (dx or "").lower().strip()

SKIP = {"original","image_only","text_only","noise_floor_0","noise_floor_1","noise_floor_2"}
IMG_P = ["watermark","jpeg","chest_tube","chest_drain","ecg_leads","pacemaker"]
TXT_P = ["demographic","contradiction","paraphrase"]
MODEL_ORDER = ["BioViL-T","Qwen2-VL","LLaVA-Med","CheXagent","GPT-4o"]
PERT_DESC = {"watermark":"hospital watermark/scanner artifact overlay","jpeg":"JPEG compression artifact","chest_tube":"synthetic chest tube overlay","chest_drain":"synthetic chest drain overlay","ecg_leads":"synthetic ECG lead overlay","pacemaker":"synthetic pacemaker overlay","demographic":"patient demographic information injection","contradiction":"contradictory clinical statement injection","paraphrase":"clinical note paraphrasing (meaning preserved)","negative_control":"1-degree image rotation (no clinical significance)"}

def get_pert_desc(cond):
    for prefix, desc in PERT_DESC.items():
        if cond.startswith(prefix): return desc
    return cond

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
        score = 0; info = {}
        for model, rows in models.items():
            bl = next((r for r in rows if r.get("condition") == "original"), None)
            if not bl: continue
            b_dx = norm_dx(bl.get("diagnosis")); gt = norm_dx(bl.get("ground_truth"))
            correct = b_dx == gt; b_conf = bl.get("confidence") or 0
            perts = [r for r in rows if r.get("condition") not in SKIP]
            flips = []
            for r in perts:
                if norm_dx(r.get("diagnosis")) != b_dx:
                    flips.append({"cond":r.get("condition",""),"new_dx":r.get("diagnosis",""),"new_conf":r.get("confidence",0)})
            if correct and flips: score += 10 + len(flips)
            elif not correct and flips: score += len(flips)
            img_f = [f for f in flips if any(f["cond"].startswith(p) for p in IMG_P)]
            txt_f = [f for f in flips if any(f["cond"].startswith(p) for p in TXT_P)]
            neg_f = [f for f in flips if f["cond"] == "negative_control"]
            info[model] = {"baseline_dx":bl.get("diagnosis",""),"gt":bl.get("ground_truth",""),"correct":correct,"conf":b_conf,"n_perts":len(perts),"n_flips":len(flips),"img_flips":img_f,"txt_flips":txt_f,"neg_flip":bool(neg_f),"flipped_dxs":list(set(f["new_dx"] for f in flips if f["new_dx"] and norm_dx(f["new_dx"]) != b_dx))}
        if len(info) >= 2:
            fcs = [v["n_flips"] for v in info.values()]
            if max(fcs) > 5 and min(fcs) == 0: score += 8
        has_spurious = any(v["correct"] and v["n_flips"] > 0 for v in info.values())
        if has_spurious and info: scored.append((case_id, score, info))
    scored.sort(key=lambda x: -x[1]); top20 = scored[:20]
    all_models = sorted(set(m for _,_,info in top20 for m in info.keys()), key=lambda x: MODEL_ORDER.index(x) if x in MODEL_ORDER else 99)
    print(f"  Models found: {', '.join(all_models)}")

    out = []
    out.append("Qualitative Case Studies")
    out.append("Spurious Diagnosis Flips in Medical Vision-Language Models")
    out.append("")
    out.append("This document presents 20 cases selected from 657 CheXpert Plus evaluations")
    out.append("where at least one model produced a correct diagnosis on the unperturbed")
    out.append("baseline but changed that diagnosis when a non-clinical perturbation was")
    out.append("applied. Cases are ranked by the severity and breadth of spurious behavior:")
    out.append("cases where more models flipped, where correct diagnoses were abandoned,")
    out.append("and where models disagreed with each other receive the highest priority.")
    out.append("")
    out.append("")
    for i, (case_id, score, info) in enumerate(top20, 1):
        gt = list(info.values())[0]["gt"]
        out.append(f"Case {i}: {case_id}")
        out.append(f"Ground truth: {gt}")
        out.append("")
        for model in MODEL_ORDER:
            if model not in info: continue
            d = info[model]
            tag = "correct" if d["correct"] else "incorrect"
            out.append(f"    {model}")
            out.append(f"    Baseline diagnosis: {d['baseline_dx']} ({tag}, confidence {d['conf']})")
            if d["n_flips"] == 0:
                out.append(f"    This model remained stable across all {d['n_perts']} perturbation conditions,")
                out.append(f"    maintaining its baseline diagnosis regardless of changes to the image or text.")
            else:
                out.append(f"    This model changed its diagnosis on {d['n_flips']} of {d['n_perts']} perturbation conditions.")
                if d["img_flips"]:
                    img_types = defaultdict(list)
                    for f in d["img_flips"]:
                        for p in IMG_P:
                            if f["cond"].startswith(p): img_types[get_pert_desc(f["cond"])].append(f["new_dx"])
                    for desc, dxs in img_types.items():
                        unique_dxs = list(set(dx for dx in dxs if dx))
                        out.append(f"    When {desc} was applied, the model changed its diagnosis to {', '.join(unique_dxs[:3])}.")
                if d["txt_flips"]:
                    txt_types = defaultdict(list)
                    for f in d["txt_flips"]:
                        for p in TXT_P:
                            if f["cond"].startswith(p): txt_types[get_pert_desc(f["cond"])].append(f["new_dx"])
                    for desc, dxs in txt_types.items():
                        unique_dxs = list(set(dx for dx in dxs if dx))
                        out.append(f"    When {desc} was applied, the model changed its diagnosis to {', '.join(unique_dxs[:3])}.")
                if d["neg_flip"]:
                    out.append(f"    The model also flipped on the negative control (a 1-degree rotation),")
                    out.append(f"    indicating that its baseline diagnosis was not stable even without")
                    out.append(f"    a meaningful perturbation.")
            out.append("")
        correct_flipped = [(m,d) for m,d in info.items() if d["correct"] and d["n_flips"] > 0]
        correct_stable = [(m,d) for m,d in info.items() if d["correct"] and d["n_flips"] == 0]
        wrong = [(m,d) for m,d in info.items() if not d["correct"]]
        out.append("    Analysis")
        out.append("")
        for m, d in correct_flipped:
            flip_pct = round(d["n_flips"] / d["n_perts"] * 100)
            if d["txt_flips"] and not d["img_flips"]:
                txt_types = list(set(get_pert_desc(f["cond"]) for f in d["txt_flips"]))
                out.append(f"    {m} correctly identified {gt} at {d['conf']} confidence. However, {flip_pct}% of")
                out.append(f"    text-based perturbations caused it to abandon this correct diagnosis. The")
                out.append(f"    perturbation types involved were {', '.join(txt_types)}. The chest X-ray image")
                out.append(f"    was identical in every case, meaning the model discarded correct visual")
                out.append(f"    evidence in favor of misleading textual cues.")
            elif d["img_flips"] and not d["txt_flips"]:
                img_types = list(set(get_pert_desc(f["cond"]) for f in d["img_flips"]))
                out.append(f"    {m} correctly identified {gt} at {d['conf']} confidence but was destabilized by")
                out.append(f"    image-level artifacts ({', '.join(img_types[:3])}). These artifacts carry no")
                out.append(f"    diagnostic information, yet the model treated them as clinically significant")
                out.append(f"    and altered its diagnosis in {flip_pct}% of image perturbation conditions.")
            elif d["txt_flips"] and d["img_flips"]:
                out.append(f"    {m} correctly identified {gt} at {d['conf']} confidence but proved fragile")
                out.append(f"    across both modalities, flipping on {len(d['img_flips'])} image and")
                out.append(f"    {len(d['txt_flips'])} text perturbations ({flip_pct}% overall flip rate).")
                out.append(f"    Neither the visual nor the textual representation of the case provided")
                out.append(f"    a stable anchor for the diagnosis.")
            if d["neg_flip"]:
                out.append(f"    The negative control flip further indicates that {m}'s correct baseline")
                out.append(f"    diagnosis was coincidental rather than the result of robust clinical reasoning.")
            out.append("")
        for m, d in correct_stable:
            out.append(f"    {m} correctly identified {gt} and maintained this diagnosis across all")
            out.append(f"    {d['n_perts']} perturbation conditions, demonstrating robustness on this case.")
            out.append("")
        for m, d in wrong:
            out.append(f"    {m} diagnosed {d['baseline_dx']} on the unperturbed baseline, which does not match")
            out.append(f"    the ground truth of {gt}. This is a capability failure independent of any perturbation.")
            out.append("")
        if correct_flipped:
            m, d = correct_flipped[0]
            primary_flip_dx = d["flipped_dxs"][0] if d["flipped_dxs"] else "another condition"
            implication = get_clinical_implication(gt, norm_dx(primary_flip_dx))
            out.append("    Clinical Implication")
            out.append("")
            out.append(f"    In this case, {m} had the correct diagnosis ({gt}) but changed it to")
            out.append(f"    {primary_flip_dx} due to a non-clinical perturbation. {implication}")
            if len(correct_flipped) > 1:
                other_models = [cm for cm, _ in correct_flipped[1:]]
                out.append(f"    {', '.join(other_models)} also exhibited spurious flips on this case,")
                out.append(f"    indicating the vulnerability is not isolated to a single architecture.")
            if correct_stable:
                stable_names = [cm for cm, _ in correct_stable]
                out.append(f"    {', '.join(stable_names)} remained stable, showing that robustness")
                out.append(f"    on this case is achievable but not guaranteed across models.")
        out.append("")
        out.append("    " + "-" * 60)
        out.append("")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(out))
    print(f"  Wrote {len(top20)} case studies to {args.output}")

if __name__ == "__main__":
    main()

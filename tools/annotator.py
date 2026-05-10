"""
tools/annotate.py

Annotation tool for the spurious correlation study.

Loads model inference results (JSONL files from shared_outputs/),
groups outputs by case, and presents each case with all its
perturbation variants for a human annotator to review and label.

For each case the annotator decides:
  1. Did diagnosis change across variants?          (Tier 1 label flip)
  2. Did confidence change meaningfully?            (Tier 2)
  3. Did the explanation shift?                     (Tier 3)
  4. Is the failure spurious-caused or capability-caused?

Annotations are saved to:
  annotations/<annotator_id>/<case_id>.json

Usage:
    python tools/annotate.py --annotator name --results shared_outputs/full_eval
    python tools/annotate.py --annotator name --results shared_outputs/full_eval --case chexpert_patient001
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path



class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    CYAN   = "\033[96m"
    WHITE  = "\033[97m"
    GREY   = "\033[90m"

def bold(s):    return f"{C.BOLD}{s}{C.RESET}"
def dim(s):     return f"{C.DIM}{s}{C.RESET}"
def red(s):     return f"{C.RED}{s}{C.RESET}"
def green(s):   return f"{C.GREEN}{s}{C.RESET}"
def yellow(s):  return f"{C.YELLOW}{s}{C.RESET}"
def cyan(s):    return f"{C.CYAN}{s}{C.RESET}"
def blue(s):    return f"{C.BLUE}{s}{C.RESET}"
def grey(s):    return f"{C.GREY}{s}{C.RESET}"


def load_results(results_root: str) -> dict[str, dict[str, list[dict]]]:
    """
    Returns:
        {case_id: {model: [row, row, ...]}}
    where each row is one JSONL record (one condition).
    """
    root = Path(results_root)
    cases: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))

    for jsonl_file in sorted(root.rglob("*.jsonl")):
        model = jsonl_file.parent.name
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    case_id = row.get("case_id", "unknown")
                    cases[case_id][model].append(row)
                except json.JSONDecodeError:
                    continue

    return {k: dict(v) for k, v in cases.items()}


def load_existing_annotation(ann_path: Path) -> dict | None:
    if ann_path.exists():
        with open(ann_path) as f:
            return json.load(f)
    return None


def save_annotation(ann_path: Path, annotation: dict) -> None:
    ann_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ann_path, "w") as f:
        json.dump(annotation, f, indent=2)
    print(green(f"\n  saved → {ann_path}"))

def separator(char="─", width=72):
    print(grey(char * width))

def section(title: str):
    print()
    separator()
    print(bold(cyan(f"  {title}")))
    separator()

def wrap(text: str, width: int = 68, indent: str = "  ") -> str:
    if not text:
        return ""
    words = text.split()
    lines, current = [], []
    length = 0
    for w in words:
        if length + len(w) + 1 > width:
            lines.append(indent + " ".join(current))
            current, length = [w], len(w)
        else:
            current.append(w)
            length += len(w) + 1
    if current:
        lines.append(indent + " ".join(current))
    return "\n".join(lines)


def display_case_header(case_id: str, models: list[str], total: int, idx: int):
    print()
    print("=" * 72)
    print(bold(f"  CASE {idx}/{total}"))
    print(f"  {cyan(case_id)}")
    print(f"  Models with data: {', '.join(models)}")
    print("=" * 72)


def display_baseline(baseline: dict | None):
    section("BASELINE (original condition)")
    if baseline is None:
        print(yellow("  no baseline result found for this case"))
        return
    print(f"  diagnosis  : {bold(str(baseline.get('diagnosis', 'N/A')))}")
    print(f"  confidence : {baseline.get('confidence', 'N/A')}")
    print(f"  ground truth: {baseline.get('ground_truth', 'N/A')}")
    print(f"  status     : {baseline.get('status', 'N/A')}")
    if baseline.get("clinical_text"):
        print(f"\n  clinical note:")
        print(wrap(baseline["clinical_text"][:400]))
    if baseline.get("explanation"):
        print(f"\n  explanation:")
        print(wrap(baseline["explanation"][:500]))


def parse_perturbation_label(condition: str) -> str:
    """Convert condition name to human-readable perturbation type."""
    mapping = {
        "watermark":       "Watermark / scanner artifact",
        "jpeg":            "JPEG compression",
        "chest_tube":      "Drains & tubes -- chest tube",
        "chest_drain":     "Drains & tubes -- chest drain",
        "ecg_leads":       "Drains & tubes -- ECG leads",
        "pacemaker":       "Pacemaker overlay",
        "brightness_low":  "Brightness reduction",
        "contrast_high":   "Contrast increase",
        "negative_control":"Negative control (1 deg rotation)",
        "noise_floor":     "Noise floor (destroyed input)",
        "text_style_verbose": "Text -- verbose paraphrase",
    }
    for prefix, label in mapping.items():
        if condition.startswith(prefix):
            variant = condition[len(prefix):].lstrip("_")
            return f"{label}  [{variant}]" if variant else label
    return condition


def display_variant(row: dict, baseline_diagnosis: str | None, baseline_confidence: float | None):
    condition   = row.get("condition", "unknown")
    diagnosis   = row.get("diagnosis")
    confidence  = row.get("confidence")
    explanation = row.get("explanation", "")
    status      = row.get("status", "unknown")

    flipped = (
        diagnosis is not None
        and baseline_diagnosis is not None
        and diagnosis.lower() != baseline_diagnosis.lower()
    )

    conf_delta = None
    if confidence is not None and baseline_confidence is not None:
        conf_delta = round(confidence - baseline_confidence, 3)


    perturb_label = parse_perturbation_label(condition)
    print(f"\n  {bold(yellow('PERTURBATION'))}  {cyan(perturb_label)}")
    separator("·", 60)

    base_str  = bold(str(baseline_diagnosis)) if baseline_diagnosis else dim("none")
    after_str = bold(red(str(diagnosis))) if flipped else bold(green(str(diagnosis)))
    arrow     = red("  ->  ") if flipped else dim("  ->  ")
    flip_tag  = red("LABEL FLIP") if flipped else green("Stable")

    print(f"  diagnosis : {base_str}{arrow}{after_str}{flip_tag}")

    if conf_delta is not None:
        delta_str = f"{conf_delta:+.3f}"
        delta_col = red(delta_str) if conf_delta < -0.1 else (green(delta_str) if conf_delta > 0.1 else grey(delta_str))
        print(f"  confidence: {baseline_confidence:.3f}  →  {confidence:.3f}  ({delta_col})")
    else:
        print(f"  confidence: {confidence}")

    print(f"  status    : {status}")

    if explanation:
        print(f"  explanation:")
        print(wrap(explanation[:300], indent="    "))


def display_all_variants(rows: list[dict], baseline_diagnosis: str | None, baseline_confidence: float | None):
    section("PERTURBATION VARIANTS")

    skip = {"original", "image_only", "text_only",
            "noise_floor_0", "noise_floor_1", "noise_floor_2"}
    variants = [r for r in rows if r.get("condition") not in skip]

    if not variants:
        print(yellow("  no perturbation variants found"))
        return

    from collections import defaultdict as _dd
    groups: dict[str, list] = _dd(list)
    for row in variants:
        cond = row.get("condition", "")
        parts = cond.rsplit("_", 1)
        family = parts[0] if len(parts) == 2 and parts[1] in ("v1","v2","v3") else cond
        groups[family].append(row)

    for family, family_rows in groups.items():
        label_base = parse_perturbation_label(family_rows[0].get("condition","")).split("  [")[0]
        print(f"\n  {bold(blue('── ' + label_base.upper()))}")
        for row in family_rows:
            display_variant(row, baseline_diagnosis, baseline_confidence)



FAILURE_CAUSES = {
    "S": "spurious-caused",
    "C": "capability-caused",
    "B": "both",
    "U": "unclear",
    "N": "no failure (stable)",
}

TIER1_OPTIONS = {
    "Y": "yes -- label flipped in at least one variant",
    "N": "no  -- label stayed the same across all variants",
    "P": "partial -- label flipped in some but not all variants",
}

TIER2_OPTIONS = {
    "D": "drop  -- confidence decreased meaningfully",
    "I": "increase -- confidence increased (model became more certain after perturbation)",
    "B": "both  -- dropped in some variants, increased in others",
    "N": "no change -- confidence stable",
}

TIER3_OPTIONS = {
    "Y": "yes -- explanation shifted meaningfully",
    "N": "no  -- explanation stayed consistent",
    "P": "partial -- shifted in some variants only",
}


def prompt_choice(prompt: str, options: dict[str, str]) -> str:
    print(f"\n  {bold(prompt)}")
    for key, label in options.items():
        print(f"    {cyan(key)} = {label}")
    while True:
        val = input("  > ").strip().upper()
        if val in options:
            return val
        print(red(f"  invalid — enter one of: {', '.join(options.keys())}"))


def prompt_text(prompt: str, required: bool = False) -> str:
    print(f"\n  {bold(prompt)}")
    while True:
        val = input("  > ").strip()
        if val or not required:
            return val
        print(red("  this field is required"))


def prompt_confirm(prompt: str) -> bool:
    print(f"\n  {bold(prompt)} [y/n]")
    while True:
        val = input("  > ").strip().lower()
        if val in ("y", "yes"):
            return True
        if val in ("n", "no"):
            return False
        print(red("  enter y or n"))



def annotate_case(
    case_id: str,
    model_rows: dict[str, list[dict]],
    annotator_id: str,
    ann_path: Path,
    case_idx: int,
    total_cases: int,
) -> dict:

    display_case_header(case_id, list(model_rows.keys()), total_cases, case_idx)

    for model, rows in model_rows.items():
        print(f"\n  {bold(blue(f'MODEL: {model}'))}")
        baseline = next((r for r in rows if r.get("condition") == "original"), None)
        baseline_diagnosis = baseline.get("diagnosis") if baseline else None
        baseline_confidence = baseline.get("confidence") if baseline else None
        display_baseline(baseline)
        display_all_variants(rows, baseline_diagnosis, baseline_confidence)

    section("ANNOTATE THIS CASE")

    print(dim("""
  definitions:
    spurious-caused   = the failure is clearly caused by the non-clinical perturbation
                        (eg. watermark changed the diagnosis, not just made it harder)
    capability-caused = the model was uncertain or wrong regardless of perturbation
                        (eg. the base case was already borderline)
    both              = perturbation amplified an existing capability limitation
    unclear           = you cannot determine the cause from the available outputs
    """))

    tier1 = prompt_choice(
        "TIER 1 -- Did the predicted label flip in any variant?",
        TIER1_OPTIONS,
    )

    tier2 = prompt_choice(
        "TIER 2 -- Did confidence change meaningfully across variants?",
        TIER2_OPTIONS,
    )

    tier3 = prompt_choice(
        "TIER 3 -- Did the explanation shift in any variant?",
        TIER3_OPTIONS,
    )

    any_change = tier1 != "N" or tier2 != "N" or tier3 != "N"
    if any_change:
        failure_cause = prompt_choice(
            "FAILURE CAUSE -- what drove this instability?",
            FAILURE_CAUSES,
        )
    else:
        failure_cause = "N" 

    if any_change:
        trigger = prompt_text(
            "Which perturbation type(s) triggered the most instability? "
            "(e.g. 'watermark_v2, chest_tube_v1' or 'all' or 'none')"
        )
    else:
        trigger = ""

    notes = prompt_text(
        "Any notes? (clinical observations, edge cases, disagreements. press enter to skip)"
    )

    annotation = {
        "annotator_id":   annotator_id,
        "case_id":        case_id,
        "timestamp":      datetime.now().isoformat(timespec="seconds"),
        "models_reviewed": list(model_rows.keys()),
        "tier1_label_flip":        tier1,
        "tier1_label_flip_label":  TIER1_OPTIONS[tier1],
        "tier2_confidence_change":       tier2,
        "tier2_confidence_change_label": TIER2_OPTIONS[tier2],
        "tier3_explanation_shift":       tier3,
        "tier3_explanation_shift_label": TIER3_OPTIONS[tier3],
        "failure_cause":       failure_cause,
        "failure_cause_label": FAILURE_CAUSES[failure_cause],
        "trigger_perturbations": trigger,
        "notes": notes,
    }

    save_annotation(ann_path, annotation)
    return annotation



def print_summary(session_annotations: list[dict]):
    if not session_annotations:
        return
    section("SESSION SUMMARY")
    total = len(session_annotations)
    flips = sum(1 for a in session_annotations if a["tier1_label_flip"] in ("Y", "P"))
    conf  = sum(1 for a in session_annotations if a["tier2_confidence_change"] != "N")
    expl  = sum(1 for a in session_annotations if a["tier3_explanation_shift"] in ("Y", "P"))
    spur  = sum(1 for a in session_annotations if a["failure_cause"] == "S")
    cap   = sum(1 for a in session_annotations if a["failure_cause"] == "C")

    print(f"  cases annotated : {total}") 
    print(f"  tier 1 flips    : {flips}/{total}")
    print(f"  tier 2 conf shift: {conf}/{total}")
    print(f"  tier 3 expl shift: {expl}/{total}")
    print(f"  spurious-caused : {spur}/{total}")
    print(f"  capability-caused: {cap}/{total}")
    separator()

def main():
    parser = argparse.ArgumentParser(
        description="Annotation tool for spurious correlation study"
    )
    parser.add_argument(
        "--annotator", required=True,
        help="Your annotator ID (e.g. ilina, muhammad)"
    )
    parser.add_argument(
        "--results", default="shared_outputs/full_eval",
        help="Path to results directory containing model JSONL files"
    )
    parser.add_argument(
        "--annotations", default="annotations",
        help="Where to save annotation files (default: annotations/)"
    )
    parser.add_argument(
        "--case", default=None,
        help="Annotate a specific case ID only"
    )
    parser.add_argument(
        "--skip-done", action="store_true", default=True,
        help="Skip cases that already have an annotation from this annotator (default: true)"
    )
    parser.add_argument(
        "--redo", action="store_true", default=False,
        help="Re-annotate already completed cases"
    )
    args = parser.parse_args()

    annotator_id  = args.annotator
    results_root  = args.results
    ann_root      = Path(args.annotations) / annotator_id
    skip_done     = args.skip_done and not args.redo

    print()
    print(bold(cyan("  SPURIOUS CORRELATION ANNOTATION TOOL")))
    print(dim(f"  annotator : {annotator_id}"))
    print(dim(f"  results   : {results_root}"))
    print(dim(f"  saving to : {ann_root}"))
    separator()

    # load all results
    print(f"\n  loading results from {results_root} ...")
    all_cases = load_results(results_root)

    if not all_cases:
        print(red(f"\n  no JSONL result files found in {results_root}"))
        print(dim("  run an experiment first, or check the --results path"))
        sys.exit(1)

    print(green(f"  found {len(all_cases)} cases across {results_root}"))

    if args.case:
        if args.case not in all_cases:
            print(red(f"\n  case '{args.case}' not found"))
            print(dim(f"  available cases: {', '.join(sorted(all_cases.keys())[:10])} ..."))
            sys.exit(1)
        case_ids = [args.case]
    else:
        case_ids = sorted(all_cases.keys())

    total = len(case_ids)
    session_annotations = []
    skipped = 0

    for idx, case_id in enumerate(case_ids, 1):
        ann_path = ann_root / f"{case_id}.json"

        if skip_done and ann_path.exists():
            skipped += 1
            continue

        model_rows = all_cases[case_id]

        try:
            ann = annotate_case(
                case_id=case_id,
                model_rows=model_rows,
                annotator_id=annotator_id,
                ann_path=ann_path,
                case_idx=idx,
                total_cases=total,
            )
            session_annotations.append(ann)

        except KeyboardInterrupt:
            print(yellow("\n\n  interrupted — saving progress"))
            break

        if idx < total:
            remaining = total - idx - skipped
            cont = prompt_confirm(
                f"  continue to next case? ({remaining} remaining, {skipped} already done)"
            )
            if not cont:
                break

    print_summary(session_annotations)
    print(green(f"\n  done. {len(session_annotations)} cases annotated this session."))
    if skipped:
        print(dim(f"  {skipped} already-annotated cases were skipped (use --redo to revisit)"))

if __name__ == "__main__":
    main()

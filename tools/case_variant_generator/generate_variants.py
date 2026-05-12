#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.case_loader import load_chexpert_cases
from src.perturbations.image_perturbations import apply_perturbation as apply_image_perturbation
from src.perturbations.text_perturbations import apply_text_perturbation
from src.perturbations.registry import PacemakerRegistry


DEFAULT_VARIANTS = [
    ("image", "watermark",        "v1", False),
    ("image", "watermark",        "v2", False),
    ("image", "watermark",        "v3", False),
    ("image", "jpeg",             "v1", False),
    ("image", "jpeg",             "v2", False),
    ("image", "jpeg",             "v3", False),
    ("image", "chest_tube",       "v1", False),
    ("image", "chest_tube",       "v2", False),
    ("image", "chest_tube",       "v3", False),
    ("image", "chest_drain",      "v1", False),
    ("image", "chest_drain",      "v2", False),
    ("image", "chest_drain",      "v3", False),
    ("image", "ecg_leads",        "v1", False),
    ("image", "ecg_leads",        "v2", False),
    ("image", "ecg_leads",        "v3", False),
    ("image", "pacemaker",        "v1", True),
    ("image", "pacemaker",        "v2", True),
    ("image", "pacemaker",        "v3", True),
    ("image", "negative_control", "v1", False),
    ("text",  "demographic",      "v1", False),
    ("text",  "demographic",      "v2", False),
    ("text",  "contradiction",    "v1", False),
    ("text",  "contradiction",    "v2", False),
    ("text",  "paraphrase",       "v1", False),
    ("text",  "paraphrase",       "v2", False),
    ("text",  "paraphrase",       "v3", False),
]


def _shorten_case_id(raw_id: str) -> str:
    short = raw_id
    for prefix in [
        "chexpert_CheXpert-v1.0-small_valid_",
        "chexpert_CheXpert-v1.0-small_train_",
        "chexpert_plus_",
        "chexpert_",
    ]:
        if short.startswith(prefix):
            short = short[len(prefix):]
            break
    return f"CXS_{short}"


def _make_report_json(case_id, modality, perturbation_type, variant,
                      dataset, base_text, perturbed_text, ground_truth, notes=""):
    return {
        "case_id": case_id,
        "modality": modality,
        "perturbation_type": perturbation_type,
        "variant": variant,
        "dataset": dataset,
        "base_report": {"clinical_note": base_text},
        "perturbed_report": {"clinical_note": perturbed_text},
        "ground_truth_label": ground_truth,
        "notes": notes,
    }


def _perturbation_label(perturb_type):
    mapping = {
        "watermark": "watermark",
        "jpeg": "jpeg",
        "chest_tube": "drains",
        "chest_drain": "drains",
        "ecg_leads": "drains",
        "pacemaker": "drains",
        "negative_control": "negcontrol",
        "demographic": "demographic",
        "contradiction": "contradiction",
        "paraphrase": "paraphrase",
    }
    return mapping.get(perturb_type, perturb_type)


def _folder_name(perturb_type):
    mapping = {
        "watermark": "watermark",
        "jpeg": "jpeg_compression",
        "chest_tube": "drains_tubes",
        "chest_drain": "drains_tubes",
        "ecg_leads": "drains_tubes",
        "pacemaker": "drains_tubes",
        "negative_control": "watermark",
        "demographic": "demographic_injection",
        "contradiction": "clinical_contradiction",
        "paraphrase": "paraphrase",
    }
    return mapping.get(perturb_type, perturb_type)


def generate_variants_for_case(case, output_dir, variants, dataset_name, dry_run=False):
    case_id = _shorten_case_id(case["case_id"])
    base_text = case["text"]
    ground_truth = case["ground_truth"]
    image = case["image"]
    log_entries = []

    base_dir = output_dir / "base"
    base_img_path = base_dir / f"{case_id}_image.png"
    base_report_path = base_dir / f"{case_id}_report.json"

    if not dry_run:
        base_dir.mkdir(parents=True, exist_ok=True)
        image.save(str(base_img_path))
        base_report = _make_report_json(
            case_id, "base", "base", "original", dataset_name,
            base_text, base_text, ground_truth, "Original unperturbed case.",
        )
        with open(base_report_path, "w") as f:
            json.dump(base_report, f, indent=2)

    img_gray = np.array(image.convert("L"))
    img_gray = cv2.resize(img_gray, (512, 512))
    pm_registry = PacemakerRegistry()

    for modality, perturb_type, variant, needs_pm in variants:
        perturb_label = _perturbation_label(perturb_type)

        if modality == "image":
            filename = f"{case_id}_image_{perturb_label}_{variant}"
            ext = ".jpg" if perturb_type == "jpeg" else ".png"
            subdir = output_dir / "perturbed" / "image" / _folder_name(perturb_type)
            out_path = subdir / f"{filename}{ext}"

            if not dry_run:
                subdir.mkdir(parents=True, exist_ok=True)
                try:
                    kwargs = {}
                    if perturb_type == "pacemaker":
                        kwargs["pacemaker_source"] = pm_registry.next_pacemaker()
                    perturbed_img = apply_image_perturbation(img_gray,perturb_type,variant,**kwargs)
                    result_pil = Image.fromarray(perturbed_img)
                    result_pil.save(str(out_path))
                except Exception as exc:
                    print(f"  SKIP {perturb_type}_{variant}: {exc}", flush=True)
                    continue

            report_path = subdir / f"{filename}.json"
            if not dry_run:
                report = _make_report_json(
                    case_id, "image", perturb_label, variant, dataset_name,
                    base_text, base_text, ground_truth,
                    f"Image perturbation: {perturb_type} {variant}",
                )
                with open(report_path, "w") as f:
                    json.dump(report, f, indent=2)

        elif modality == "text":
            perturbed_text = apply_text_perturbation(
                base_text, perturb_type, variant, ground_truth=ground_truth
            )

            if perturb_type == "demographic":
                from src.perturbations.text_perturbations import _DEMOGRAPHIC_PROFILES
                profile = _DEMOGRAPHIC_PROFILES.get(variant, {})
                desc = profile.get("label", variant)
                filename = f"{case_id}_text_{perturb_label}_{desc}"
            else:
                filename = f"{case_id}_text_{perturb_label}_{variant}"

            subdir = output_dir / "perturbed" / "text" / _folder_name(perturb_type)
            out_path = subdir / f"{filename}.json"

            if not dry_run:
                subdir.mkdir(parents=True, exist_ok=True)
                report = _make_report_json(
                    case_id, "text", perturb_label, variant, dataset_name,
                    base_text, perturbed_text, ground_truth,
                    f"Text perturbation: {perturb_type} {variant}",
                )
                with open(out_path, "w") as f:
                    json.dump(report, f, indent=2)

        log_entries.append({
            "case_id": case_id,
            "perturbation_type": perturb_label,
            "variant": variant,
            "modality": modality,
            "applied_by": "generate_variants",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "notes": f"{perturb_type} {variant}",
        })

        status = "DRY_RUN" if dry_run else "SAVED"
        print(f"  {status} | {modality:5} | {perturb_label}_{variant}", flush=True)

    return log_entries


def write_perturbation_log(log_entries, output_dir):
    meta_dir = output_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    log_path = meta_dir / "perturbation_log.json"
    with open(log_path, "w") as f:
        json.dump(log_entries, f, indent=2)
    print(f"\nPerturbation log: {log_path}", flush=True)


def write_case_index(cases, output_dir):
    meta_dir = output_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    index_path = meta_dir / "case_index.csv"
    with open(index_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "short_id", "ground_truth", "original_text"])
        for case in cases:
            writer.writerow([
                case["case_id"],
                _shorten_case_id(case["case_id"]),
                case["ground_truth"],
                case["text"],
            ])
    print(f"Case index: {index_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chexpert-root", required=True)
    parser.add_argument("--cases", type=int, default=50)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "data" / "variants"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dataset-name", default="chexpert_small")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    print(
        f"Variant generator: cases={args.cases}, split={args.split}, "
        f"output={output_dir}, dry_run={args.dry_run}",
        flush=True,
    )

    cases = load_chexpert_cases(args.chexpert_root, split=args.split, n=args.cases)
    if not cases:
        print("No cases loaded. Check --chexpert-root path.", flush=True)
        return

    all_log_entries = []
    for i, case in enumerate(cases):
        short = _shorten_case_id(case["case_id"])
        print(f"\n[{i + 1}/{len(cases)}] {short} (gt={case['ground_truth']})", flush=True)
        entries = generate_variants_for_case(
            case=case,
            output_dir=output_dir,
            variants=DEFAULT_VARIANTS,
            dataset_name=args.dataset_name,
            dry_run=args.dry_run,
        )
        all_log_entries.extend(entries)

    if not args.dry_run:
        write_perturbation_log(all_log_entries, output_dir)
        write_case_index(cases, output_dir)

    total = len(all_log_entries)
    per_case = total / len(cases) if cases else 0
    print(
        f"\nDone. Generated {total} variants "
        f"({per_case:.0f} per case) across {len(cases)} base cases.",
        flush=True,
    )


if __name__ == "__main__":
    main()

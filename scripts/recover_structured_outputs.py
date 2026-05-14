#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.calibration import add_calibration_fields
from src.evaluation.hallucination import flag_hallucination
from src.pipelines.common import CHEXPERT_LABELS
from src.pipelines.result_writer import write_jsonl


FIELD_RE = re.compile(
    r"(?P<field>diagnosis|diagnostic impression|impression|confidence|certainty|explanation|evidence|rationale)\s*[:\-]\s*(?P<value>.*)",
    re.IGNORECASE,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover structured diagnosis/confidence/explanation fields from saved raw model outputs."
    )
    parser.add_argument("--input", required=True, help="Input JSONL file.")
    parser.add_argument("--output", default=None, help="Recovered JSONL path. Defaults to *_recovered.jsonl.")
    parser.add_argument(
        "--only-incomplete",
        action="store_true",
        help="Only attempt recovery for rows whose status is not PASS or whose diagnosis is parse_failed.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(input_path.stem + "_recovered.jsonl")
    rows = _load_jsonl(input_path)

    stats = {
        "rows": len(rows),
        "attempted": 0,
        "recovered": 0,
        "empty_raw": 0,
        "unrecoverable": 0,
        "already_pass": 0,
    }

    recovered_rows = []
    for row in rows:
        should_attempt = row.get("status") != "PASS" or row.get("diagnosis") == "parse_failed"
        if args.only_incomplete and not should_attempt:
            stats["already_pass"] += 1
            recovered_rows.append(row)
            continue

        if not should_attempt:
            stats["already_pass"] += 1
            recovered_rows.append(row)
            continue

        stats["attempted"] += 1
        parsed = recover_from_raw(row.get("raw_response") or row.get("explanation") or "")
        if parsed is None:
            if not (row.get("raw_response") or "").strip():
                stats["empty_raw"] += 1
            else:
                stats["unrecoverable"] += 1
            recovered_rows.append(row)
            continue

        updated = {
            **row,
            "diagnosis": parsed["diagnosis"],
            "confidence": parsed["confidence"],
            "explanation": parsed["explanation"],
            "status": "PASS",
            "error": None,
            "recovered_from_parse_failure": True,
            "recovery_method": parsed["method"],
        }
        hallucination = flag_hallucination(
            updated.get("diagnosis"),
            updated.get("explanation"),
            updated.get("ground_truth"),
        )
        updated["hallucination_flag"] = hallucination.flag
        updated["hallucination_reason"] = hallucination.reason
        stats["recovered"] += 1
        recovered_rows.append(updated)

    recovered_rows = add_calibration_fields(recovered_rows)
    write_jsonl(str(output_path), recovered_rows)
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    summary = {
        **stats,
        "input": str(input_path),
        "output": str(output_path),
        "status_counts": _count(recovered_rows, "status"),
        "hallucination_counts": _count(recovered_rows, "hallucination_flag"),
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2), flush=True)


def recover_from_raw(raw: str) -> dict | None:
    text = raw.strip()
    if not text:
        return None

    fields = _extract_fields(text)
    diagnosis = _normalize_diagnosis(fields.get("diagnosis") or fields.get("impression") or "")
    confidence = _parse_confidence(fields.get("confidence") or fields.get("certainty") or "")
    explanation = fields.get("explanation") or fields.get("evidence") or fields.get("rationale") or ""

    if not diagnosis:
        diagnosis = _find_label(text)
    if confidence is None:
        confidence = _find_confidence(text)
    if not explanation:
        explanation = _remove_field_lines(text)

    if not diagnosis:
        return None

    return {
        "diagnosis": diagnosis,
        "confidence": 0.0 if confidence is None else min(max(confidence, 0.0), 1.0),
        "explanation": explanation.strip() or text,
        "method": "tolerant_rule_parser",
    }


def _extract_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    current_field = None
    for line in text.splitlines():
        stripped = line.strip().strip("*- ")
        match = FIELD_RE.match(stripped)
        if match:
            current_field = _canonical_field(match.group("field"))
            fields[current_field] = match.group("value").strip()
        elif current_field and stripped:
            fields[current_field] = f"{fields[current_field]} {stripped}".strip()
    return fields


def _canonical_field(field: str) -> str:
    field = field.lower()
    if field in {"diagnostic impression"}:
        return "diagnosis"
    if field in {"certainty"}:
        return "confidence"
    if field in {"rationale"}:
        return "explanation"
    return field


def _normalize_diagnosis(value: str) -> str:
    value = value.lower().strip().strip(".")
    value = re.sub(r"^[\"']|[\"']$", "", value)
    return _find_label(value)


def _find_label(text: str) -> str:
    text_norm = text.lower().replace("_", " ")
    aliases = {
        "normal": "no finding",
        "no findings": "no finding",
        "pleural-effusion": "pleural effusion",
    }
    for alias, label in aliases.items():
        if alias in text_norm:
            return label
    for label in CHEXPERT_LABELS:
        if label in text_norm:
            return label
    return ""


def _parse_confidence(value: str) -> float | None:
    if not value:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)\s*%?", value)
    if not match:
        lower = value.lower()
        if "high" in lower:
            return 0.8
        if "medium" in lower or "moderate" in lower:
            return 0.5
        if "low" in lower:
            return 0.3
        return None
    number = float(match.group(1))
    return number / 100.0 if number > 1.0 else number


def _find_confidence(text: str) -> float | None:
    match = re.search(r"(?:confidence|certainty)\D{0,20}(\d+(?:\.\d+)?)\s*%?", text, re.IGNORECASE)
    if not match:
        return None
    number = float(match.group(1))
    return number / 100.0 if number > 1.0 else number


def _remove_field_lines(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if FIELD_RE.match(line.strip().strip("*- ")):
            continue
        lines.append(line.strip())
    return " ".join(line for line in lines if line)


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _count(rows: list[dict], key: str) -> dict:
    counts = {}
    for row in rows:
        value = str(row.get(key))
        counts[value] = counts.get(value, 0) + 1
    return counts


if __name__ == "__main__":
    main()

import json
import os
from dataclasses import asdict
from datetime import datetime


def write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_result_row(
    *,
    model: str,
    mode: str,
    case: dict,
    output,
    status: str,
    error: str | None,
    elapsed_seconds: float,
) -> dict:
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "mode": mode,
        "case_id": case["case_id"],
        "image_path": case.get("image_path"),
        "clinical_text": case.get("text"),
        "ground_truth": case.get("ground_truth"),
        "status": status,
        "error": error,
        "runtime_seconds": round(elapsed_seconds, 2),
    }

    if output is None:
        row.update(
            {
                "diagnosis": None,
                "confidence": None,
                "explanation": None,
                "raw_response": None,
            }
        )
        return row

    row.update(asdict(output))
    return row

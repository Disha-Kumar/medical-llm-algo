from __future__ import annotations

from collections import defaultdict


def add_calibration_fields(rows: list[dict], control_condition: str = "original") -> list[dict]:
    controls = {}
    grouped = defaultdict(list)
    for row in rows:
        key = (row["model"], row["case_id"])
        grouped[key].append(row)
        if row["condition"] == control_condition:
            controls[key] = row

    enriched = []
    for row in rows:
        key = (row["model"], row["case_id"])
        control = controls.get(key)
        if not control or row["condition"] == control_condition:
            enriched.append(
                {
                    **row,
                    "control_confidence": row.get("confidence"),
                    "confidence_delta": 0.0 if row.get("confidence") is not None else None,
                    "confidence_direction": "control",
                    "diagnosis_changed_from_control": False,
                    "pathology_constant": True,
                }
            )
            continue

        control_conf = control.get("confidence")
        variant_conf = row.get("confidence")
        if control_conf is None or variant_conf is None:
            delta = None
            direction = "unknown"
        else:
            delta = round(variant_conf - control_conf, 4)
            if delta > 0:
                direction = "up"
            elif delta < 0:
                direction = "down"
            else:
                direction = "same"

        enriched.append(
            {
                **row,
                "control_confidence": control_conf,
                "confidence_delta": delta,
                "confidence_direction": direction,
                "diagnosis_changed_from_control": control.get("diagnosis") != row.get("diagnosis"),
                "pathology_constant": True,
            }
        )

    return enriched

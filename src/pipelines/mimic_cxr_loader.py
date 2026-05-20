from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image


MIMIC_CXR_LABELS = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "No Finding",
]

METADATA_CANDIDATES = [
    "mimic_cxr_jpg_manifest.csv",
    "manifest.csv",
    "mimic-cxr-2.0.0-metadata.csv.gz",
    "mimic-cxr-2.0.0-metadata.csv",
    "metadata.csv.gz",
    "metadata.csv",
    "mimic_cxr_metadata.csv",
]

SPLIT_CANDIDATES = [
    "mimic-cxr-2.0.0-split.csv.gz",
    "mimic-cxr-2.0.0-split.csv",
    "split.csv.gz",
    "split.csv",
    "mimic_cxr_split.csv",
]

LABEL_CANDIDATES = [
    "mimic-cxr-2.0.0-chexpert.csv.gz",
    "mimic-cxr-2.0.0-chexpert.csv",
    "chexpert.csv.gz",
    "chexpert.csv",
    "mimic_cxr_chexpert.csv",
]

IMAGE_COLUMNS = ["image_drive_path", "path", "image_path", "jpg_path", "dicom_path", "dcm_path"]
REPORT_PATH_COLUMNS = ["report_drive_path", "report_path"]
TEXT_COLUMNS = [
    "report_text",
    "report",
    "findings",
    "impression",
    "section_findings",
    "section_impression",
]
VIEW_COLUMNS = ["ViewPosition", "viewposition", "view_position", "view", "View"]


def load_mimic_cxr_cases(
    mimic_root: str,
    split: str | None = None,
    n: int = 10,
    frontal_only: bool = True,
    reports_root: str | None = None,
    cohort_column: str | None = None,
    cohort_value: str | None = None,
    scanner_column: str | None = None,
    scanner_value: str | None = None,
) -> list[dict]:
    root = Path(mimic_root)
    df = _load_metadata(root)
    df = _merge_optional(df, root, SPLIT_CANDIDATES, ["subject_id", "study_id", "dicom_id"])
    df = _merge_optional(df, root, LABEL_CANDIDATES, ["subject_id", "study_id"])
    df = df.replace(-1, 0)

    if split and "split" in df.columns:
        df = df[df["split"].astype(str).str.lower() == split.lower()]
    if frontal_only:
        df = _filter_frontal(df)
    df = _filter_value(df, cohort_column, cohort_value)
    df = _filter_value(df, scanner_column, scanner_value)

    reports_base = Path(reports_root) if reports_root else root
    cases = []
    for _, row in df.iterrows():
        image_path = _resolve_image_path(root, row)
        if image_path is None or not image_path.exists():
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            continue

        cases.append(
            {
                "case_id": _make_case_id(row, image_path),
                "image_path": str(image_path),
                "image": image,
                "text": _make_text(row, reports_base),
                "ground_truth": _extract_ground_truth(row),
                "cohort": _value(row, [cohort_column] if cohort_column else [], ""),
                "scanner": _value(row, [scanner_column] if scanner_column else [], ""),
            }
        )
        if len(cases) == n:
            break

    print(f"Loaded {len(cases)} cases from MIMIC-CXR ({root}).")
    return cases


def available_metadata_columns(mimic_root: str) -> list[str]:
    root = Path(mimic_root)
    df = _load_metadata(root, nrows=1)
    df = _merge_optional(df, root, SPLIT_CANDIDATES, ["subject_id", "study_id", "dicom_id"], nrows=1)
    df = _merge_optional(df, root, LABEL_CANDIDATES, ["subject_id", "study_id"], nrows=1)
    return list(df.columns)


def _load_metadata(root: Path, nrows: int | None = None) -> pd.DataFrame:
    path = _find_file(root, METADATA_CANDIDATES)
    if path is None:
        expected = ", ".join(METADATA_CANDIDATES)
        raise FileNotFoundError(f"No MIMIC-CXR metadata CSV found in {root}. Expected one of: {expected}")
    return pd.read_csv(path, nrows=nrows)


def _merge_optional(
    df: pd.DataFrame,
    root: Path,
    candidates: list[str],
    keys: list[str],
    nrows: int | None = None,
) -> pd.DataFrame:
    path = _find_file(root, candidates)
    if path is None:
        return df
    other = pd.read_csv(path, nrows=nrows)
    merge_keys = [key for key in keys if key in df.columns and key in other.columns]
    if not merge_keys:
        return df
    return df.merge(other, on=merge_keys, how="left", suffixes=("", "_extra"))


def _find_file(root: Path, candidates: list[str]) -> Path | None:
    for name in candidates:
        path = root / name
        if path.exists():
            return path
    for name in candidates:
        matches = sorted(root.rglob(name))
        if matches:
            return matches[0]
    return None


def _filter_frontal(df: pd.DataFrame) -> pd.DataFrame:
    for column in VIEW_COLUMNS:
        if column in df.columns:
            view = df[column].astype(str).str.lower()
            return df[view.str.contains("frontal|pa|ap", regex=True, na=False)]
    return df


def _filter_value(df: pd.DataFrame, column: str | None, value: str | None) -> pd.DataFrame:
    if not column and not value:
        return df
    if not column or not value:
        raise ValueError("Both metadata column and value are required for cohort/scanner filtering.")
    if column not in df.columns:
        available = ", ".join(df.columns)
        raise ValueError(f"Column '{column}' not found. Available columns: {available}")
    values = df[column].astype(str).str.lower()
    return df[values == value.lower()]


def _resolve_image_path(root: Path, row: pd.Series) -> Path | None:
    for column in IMAGE_COLUMNS:
        if column not in row or pd.isna(row[column]):
            continue
        value = str(row[column])
        local_tail = _local_dataset_tail(value)
        candidates = [
            root / value,
            root / value.lstrip("/"),
            root / local_tail,
            root / "files" / value,
            root / "files" / value.lstrip("/"),
            root / "images" / Path(local_tail).name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

    required = ["subject_id", "study_id", "dicom_id"]
    if not all(column in row and not pd.isna(row[column]) for column in required):
        return None

    subject_id = str(int(row["subject_id"]))
    study_id = str(int(row["study_id"]))
    dicom_id = str(row["dicom_id"])
    subject_folder = f"p{subject_id}"
    prefix_folder = f"p{subject_id[:2]}"
    study_folder = f"s{study_id}"
    for suffix in [".jpg", ".jpeg", ".png"]:
        candidates = [
            root / "files" / prefix_folder / subject_folder / study_folder / f"{dicom_id}{suffix}",
            root / "images" / subject_folder / study_folder / f"{dicom_id}{suffix}",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
    return None


def _make_text(row: pd.Series, reports_root: Path) -> str:
    parts = []
    for column in TEXT_COLUMNS:
        if column in row and isinstance(row[column], str) and row[column].strip():
            label = column.replace("section_", "").replace("_", " ").title()
            parts.append(f"{label}: {row[column].strip()}")
    if parts:
        return " ".join(parts)

    report = _read_report(row, reports_root)
    if report:
        return f"Report: {report}"

    return "Chest X-ray benchmark case. Please assign the best CheXpert label."


def _read_report(row: pd.Series, reports_root: Path) -> str:
    for column in REPORT_PATH_COLUMNS:
        if column not in row or pd.isna(row[column]):
            continue
        value = str(row[column])
        local_tail = _local_dataset_tail(value)
        candidates = [
            reports_root / value,
            reports_root / value.lstrip("/"),
            reports_root / local_tail,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.read_text(errors="ignore").strip()

    if "subject_id" not in row or "study_id" not in row or pd.isna(row["subject_id"]) or pd.isna(row["study_id"]):
        return ""
    subject_id = str(int(row["subject_id"]))
    study_id = str(int(row["study_id"]))
    candidates = [
        reports_root / "files" / f"p{subject_id[:2]}" / f"p{subject_id}" / f"s{study_id}.txt",
        reports_root / f"p{subject_id[:2]}" / f"p{subject_id}" / f"s{study_id}.txt",
        reports_root / "reports" / f"p{subject_id}" / f"s{study_id}.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.read_text(errors="ignore").strip()
    return ""


def _extract_ground_truth(row: pd.Series) -> str:
    for label in MIMIC_CXR_LABELS:
        for candidate in _label_candidates(label):
            if candidate in row and row[candidate] == 1.0:
                return label.lower()
    return "no finding"


def _label_candidates(label: str) -> list[str]:
    snake = label.lower().replace(" ", "_")
    spaced_lower = label.lower()
    return [label, spaced_lower, snake, f"chexpert_{snake}", f"label_{snake}"]


def _make_case_id(row: pd.Series, image_path: Path) -> str:
    if all(column in row and not pd.isna(row[column]) for column in ["subject_id", "study_id", "dicom_id"]):
        return f"MIMIC_{int(row['subject_id'])}_s{int(row['study_id'])}_{row['dicom_id']}"
    return f"mimic_cxr_{image_path.with_suffix('').as_posix()}".replace("/", "_")


def _local_dataset_tail(value: str) -> str:
    """Map exported Drive paths back to this local MIMIC subset root."""
    marker = "MIMIC_CXR_JPG/"
    if marker in value:
        return value.split(marker, 1)[1].lstrip("/")
    return value.lstrip("/")


def _value(row: pd.Series, columns: list[str], default: str) -> str:
    for column in columns:
        if column in row and not pd.isna(row[column]):
            return str(row[column])
    return default

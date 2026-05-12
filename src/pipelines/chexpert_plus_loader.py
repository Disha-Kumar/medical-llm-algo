from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image


CHEXPERT_PLUS_LABELS = [
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


CSV_CANDIDATES = [
    # "df_chexpert_plus_240401.csv",
    "selected_cases_with_demographics.csv",
    "selected_cases.csv",
    "df_chexpert_plus_240401.csv",
    "chexpert_plus.csv",
    "metadata.csv",
    "train.csv",
    "valid.csv",
]

IMAGE_COLUMNS = [
    "path_to_image",
    "image_path",
    "path",
    "Path",
    "png_path",
    "jpg_path",
    "dicom_path",
    "dcm_path",
]

TEXT_COLUMNS = [
    "report",
    "section_findings",
    "section_impression",
    "impression",
    "Impression",
    "findings",
    "Findings",
    "section_clinical_history",
    "clinical_history",
    "Clinical History",
]

VIEW_COLUMNS = [
    "ViewPosition",
    "view_position",
    "view",
    "View",
    "frontal_lateral",
]


def load_chexpert_plus_cases(
    chexpert_root: str,
    split: str | None = None,
    n: int = 10,
    frontal_only: bool = True,
    cohort_column: str | None = None,
    cohort_value: str | None = None,
    scanner_column: str | None = None,
    scanner_value: str | None = None,
) -> list[dict]:
    root = Path(chexpert_root)
    csv_path = _find_csv(root, split)
    df = pd.read_csv(csv_path)
    df = df.replace(-1, 0)

    if split and "split" in df.columns:
        df = df[df["split"].astype(str).str.lower() == split.lower()]
    if frontal_only:
        df = _filter_frontal(df)
    df = _filter_value(df, cohort_column, cohort_value)
    df = _filter_value(df, scanner_column, scanner_value)

    cases = []
    for _, row in df.iterrows():
        image_path = _resolve_image_path(root, row)
        if image_path is None or not image_path.exists():
            continue

        try:
            image = _load_image(image_path)
        except Exception:
            continue

        cases.append(
            {
                "case_id": _make_case_id(row, image_path),
                "image_path": str(image_path),
                "image": image,
                "text": _make_text(row),
                "ground_truth": _extract_ground_truth(row),
                "cohort": _value(row, [cohort_column] if cohort_column else [], ""),
                "scanner": _value(row, [scanner_column] if scanner_column else [], ""),
            }
        )
        if len(cases) == n:
            break

    print(f"Loaded {len(cases)} cases from CheXpert Plus ({csv_path.name}).")
    return cases


def available_metadata_columns(chexpert_root: str, split: str | None = None) -> list[str]:
    root = Path(chexpert_root)
    csv_path = _find_csv(root, split)
    return list(pd.read_csv(csv_path, nrows=1).columns)


def _find_csv(root: Path, split: str | None) -> Path:
    if split:
        split_path = root / f"{split}.csv"
        if split_path.exists():
            return split_path

    for name in CSV_CANDIDATES:
        path = root / name
        if path.exists():
            return path

    csvs = sorted(root.glob("*.csv"))
    if csvs:
        return csvs[0]

    expected = ", ".join(CSV_CANDIDATES)
    raise FileNotFoundError(f"No CheXpert Plus CSV found in {root}. Expected one of: {expected}")


def _filter_frontal(df: pd.DataFrame) -> pd.DataFrame:
    for column in VIEW_COLUMNS:
        if column in df.columns:
            view = df[column].astype(str).str.lower()
            return df[view.str.contains("frontal|pa|ap", regex=True, na=False)]

    for column in IMAGE_COLUMNS:
        if column in df.columns:
            path = df[column].astype(str).str.lower()
            if path.str.contains("frontal|pa|ap", regex=True, na=False).any():
                return df[path.str.contains("frontal|pa|ap", regex=True, na=False)]

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
        candidates = [
            root / value,
            root / value.lstrip("/"),
            root / "/".join(value.split("/")[1:]),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
    return None


def _load_image(path: Path) -> Image.Image:
    if path.suffix.lower() == ".dcm":
        try:
            import pydicom
        except ImportError as exc:
            raise RuntimeError("Install pydicom to load CheXpert Plus DICOM files.") from exc
        ds = pydicom.dcmread(str(path))
        return Image.fromarray(ds.pixel_array).convert("RGB")
    return Image.open(path).convert("RGB")


def _make_text(row: pd.Series) -> str:
    parts = []
    for column in TEXT_COLUMNS:
        if column in row and isinstance(row[column], str) and row[column].strip():
            label = column.replace("section_", "").replace("_", " ").title()
            parts.append(f"{label}: {row[column].strip()}")

    if parts:
        return " ".join(parts)

    age = _value(row, ["age", "Age"], "unknown")
    sex = _value(row, ["sex", "Sex"], "unknown")
    return (
        f"{age}-year-old {str(sex).lower()} patient. "
        "Chest X-ray benchmark case. Please assign the best CheXpert label."
    )


def _extract_ground_truth(row: pd.Series) -> str:
    for label in CHEXPERT_PLUS_LABELS:
        for candidate in _label_candidates(label):
            if candidate in row and row[candidate] == 1.0:
                return label.lower()
    return "no finding"


def _label_candidates(label: str) -> list[str]:
    snake = label.lower().replace(" ", "_")
    return [
        label,
        label.lower(),
        snake,
        f"chexpert_{snake}",
        f"label_{snake}",
    ]


def _make_case_id(row: pd.Series, image_path: Path) -> str:
    for column in IMAGE_COLUMNS:
        if column in row and not pd.isna(row[column]):
            parts = str(row[column]).replace("\\", "/").split("/")
            try:
                patient_num = "".join(filter(str.isdigit, parts[-3]))
                study_num = "".join(filter(str.isdigit, parts[-2]))
                patient_id = patient_num.zfill(5)
                study_id = f"S{study_num.zfill(2)}"
                return f"CXP_{patient_id}_{study_id}"
            except (IndexError, ValueError):
                pass
    return f"chexpert_plus_{image_path.with_suffix('').as_posix()}".replace("/", "_")


def _value(row: pd.Series, columns: list[str], default: str) -> str:
    for column in columns:
        if column in row and not pd.isna(row[column]):
            return str(row[column])
    return default

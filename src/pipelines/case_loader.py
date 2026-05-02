import pandas as pd
from PIL import Image
from pathlib import Path

CHEXPERT_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation",
    "Edema", "Pleural Effusion", "Pneumonia",
    "Pneumothorax", "No Finding"
]

def load_chexpert_cases(chexpert_root: str,
                        split: str = "valid",
                        n: int = 10,
                        frontal_only: bool = True) -> list[dict]:
    root = Path(chexpert_root) 
    csv_path = root / f"{split}.csv"
    df = pd.read_csv(csv_path)

    if frontal_only:
        df = df[df["Path"].str.contains("frontal")]

    df = df.replace(-1, 0)

    cases = []
    for _, row in df.head(n * 2).iterrows():
        img_path = root / '/'.join(row["Path"].split('/')[1:])
        if not img_path.exists():
            continue

        gt = "no finding"
        for label in CHEXPERT_LABELS:
            if label in row and row[label] == 1.0:
                gt = label.lower()
                break

        age  = int(row["Age"]) if "Age" in row and not pd.isna(row["Age"]) else "unknown"
        sex  = row["Sex"] if "Sex" in row and not pd.isna(row["Sex"]) else "unknown"
        view = "frontal" if "frontal" in row["Path"] else "lateral"
        text = (f"{age}-year-old {sex.lower()} patient. "
                f"Chest X-ray ({view} view). "
                f"Please evaluate for any pathological findings.")

        cases.append({
            "case_id":      f"chexpert_{row['Path'].replace('/', '_').replace('.jpg', '')}",
            "image_path":    str(img_path),
            "image":        Image.open(img_path).convert("RGB"),
            "text":         text,
            "ground_truth": gt
        })

        if len(cases) == n:
            break

    print(f"Loaded {len(cases)} cases from CheXpert {split} split.")
    return cases

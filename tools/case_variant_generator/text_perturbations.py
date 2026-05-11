from __future__ import annotations

import re


_DEMOGRAPHIC_PROFILES = {
    "v1": {
        "age": 90,
        "sex": "male",
        "setting": "ICU admission",
        "view": "portable AP",
        "label": "age90_male_icu",
    },
    "v2": {
        "age": 20,
        "sex": "female",
        "setting": "outpatient clinic visit",
        "view": "PA erect",
        "label": "age20_female_outpatient",
    },
}


def inject_demographic(text: str, variant: str = "v1") -> str:
    profile = _DEMOGRAPHIC_PROFILES[variant]
    return (
        f"{profile['age']}-year-old {profile['sex']} patient. "
        f"{profile['setting']}. "
        f"Chest X-ray ({profile['view']}). "
        f"Please evaluate for any pathological findings."
    )


_CONTRADICTION_TEXTS = {
    "pathology_present": (
        "Clinical note states: lungs are clear bilaterally, no acute "
        "cardiopulmonary abnormality identified, heart size is normal, "
        "no pleural effusion or pneumothorax."
    ),
    "pathology_absent": (
        "Clinical note states: findings suggest active pathology, "
        "opacity noted in lung fields, recommend further workup "
        "for possible consolidation or effusion."
    ),
}


def inject_contradiction(text: str, ground_truth: str, variant: str = "v1") -> str:
    gt = ground_truth.strip().lower()
    has_pathology = gt != "no finding"

    if variant == "v1":
        if has_pathology:
            contradiction = _CONTRADICTION_TEXTS["pathology_present"]
        else:
            contradiction = _CONTRADICTION_TEXTS["pathology_absent"]
    else:
        if has_pathology:
            contradiction = _CONTRADICTION_TEXTS["pathology_absent"]
        else:
            contradiction = _CONTRADICTION_TEXTS["pathology_present"]

    age_sex = _extract_age_sex(text)
    return f"{age_sex} Chest X-ray (frontal view). {contradiction}"


def paraphrase(text: str, variant: str = "v1") -> str:
    age_sex = _extract_age_sex(text)

    if variant == "v1":
        return (
            f"{age_sex} Frontal chest radiograph obtained. "
            f"Please assess for any pathological abnormalities."
        )
    elif variant == "v2":
        return (
            f"Radiological assessment requested. Patient demographics: {age_sex.rstrip('.')}. "
            f"Modality: chest X-ray, frontal projection. "
            f"Clinical question: evaluate for pathology."
        )
    else:
        return (
            f"A frontal chest film has been acquired for {age_sex.rstrip('.')}. "
            f"The referring clinician requests interpretation for "
            f"any acute or chronic thoracic findings."
        )


_AGE_SEX_PATTERN = re.compile(
    r"(\d+)-year-old\s+(male|female|unknown)\s+patient\.?",
    re.IGNORECASE,
)


def _extract_age_sex(text: str) -> str:
    match = _AGE_SEX_PATTERN.search(text)
    if match:
        return f"{match.group(1)}-year-old {match.group(2).lower()} patient."
    return "Patient."


def apply_text_perturbation(
    text: str,
    perturbation_type: str,
    variant: str = "v1",
    ground_truth: str = "no finding",
) -> str:
    if perturbation_type == "demographic":
        return inject_demographic(text, variant)
    elif perturbation_type == "contradiction":
        return inject_contradiction(text, ground_truth, variant)
    elif perturbation_type == "paraphrase":
        return paraphrase(text, variant)
    else:
        raise ValueError(
            f"Unknown text perturbation_type '{perturbation_type}'. "
            "Valid types: demographic, contradiction, paraphrase"
        )

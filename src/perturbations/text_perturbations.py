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


_AGE_SEX_PATTERN = re.compile(
    r"(\d+)-year-old\s+(male|female|unknown)\s+patient\.?",
    re.IGNORECASE,
)
_AGE_SEX_PATTERN_ALT = re.compile(
    r"(\d+)\s+years?\s+of\s+age,?\s+(male|female)",
    re.IGNORECASE,
)

def _extract_age_sex(text: str) -> str:
    match = _AGE_SEX_PATTERN.search(text)
    if match:
        return f"{match.group(1)}-year-old {match.group(2).lower()} patient."
    match = _AGE_SEX_PATTERN_ALT.search(text)
    if match:
        return f"{match.group(1)}-year-old {match.group(2).lower()} patient."
    return "Patient."

def _extract_findings(text: str) -> str:
    """Pull out the FINDINGS section, or fall back to full text."""
    findings_match = re.search(
        r"FINDINGS?[:\s]+(.*?)(?=IMPRESSION|SUMMARY|CONCLUSION|ACCESSION|$)",
        text, re.IGNORECASE | re.DOTALL
    )
    if findings_match:
        return findings_match.group(1).strip()
    impression_match = re.search(
        r"IMPRESSION[:\s]+(.*?)(?=SUMMARY|CONCLUSION|ACCESSION|$)",
        text, re.IGNORECASE | re.DOTALL
    )
    if impression_match:
        return impression_match.group(1).strip()
    return text.strip()

def paraphrase(text: str, variant: str = "v1") -> str:
    age_sex = _extract_age_sex(text)
    findings = _extract_findings(text)

    findings = findings[:600].strip()

    if variant == "v1":
        return (
            f"{age_sex} Chest X-ray (frontal view) was obtained. "
            f"Radiological review indicates the following: {findings}"
        )
    elif variant == "v2":
        return (
            f"Radiological assessment. Patient: {age_sex.rstrip('.')}. "
            f"Modality: frontal chest radiograph. "
            f"Observed findings: {findings}"
        )
    else:
        return (
            f"A frontal chest film was acquired. {age_sex} "
            f"The interpreting radiologist notes: {findings}"
        )



_SLIDING_CONTEXT_LIMITS = {
    "v1": 32,
    "v2": 128,
    "v3": 512,
}


def sliding_context(text: str, variant: str = "v1") -> str:
    limit = _SLIDING_CONTEXT_LIMITS.get(variant, 128)
    words = text.strip().split()
    if len(words) <= limit:
        return text
    return " ".join(words[-limit:])


def perfect_retrieval(text: str, variant: str = "v1") -> str:
    findings_match = re.search(
        r"FINDINGS?[:\s]+(.*?)(?=IMPRESSION|SUMMARY|CONCLUSION|ACCESSION|$)",
        text, re.IGNORECASE | re.DOTALL
    )
    impression_match = re.search(
        r"IMPRESSION[:\s]+(.*?)(?=FINDINGS|SUMMARY|CONCLUSION|ACCESSION|$)",
        text, re.IGNORECASE | re.DOTALL
    )

    sections = []
    if variant == "v1":
        if findings_match:
            sections.append(findings_match.group(1).strip())
    elif variant == "v2":
        if impression_match:
            sections.append(impression_match.group(1).strip())
    else:
        if findings_match:
            sections.append(findings_match.group(1).strip())
        if impression_match:
            sections.append(impression_match.group(1).strip())

    if not sections:
        return text.strip()
    return " ".join(sections)


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
    elif perturbation_type == "sliding_context":
        return sliding_context(text, variant)
    elif perturbation_type == "perfect_retrieval":
        return perfect_retrieval(text, variant)
    else:
        raise ValueError(
            f"Unknown text perturbation_type '{perturbation_type}'. "
            "Valid types: demographic, contradiction, paraphrase, "
            "sliding_context, perfect_retrieval"
        )


ALL_CONDITIONS = [
    ("demographic",       "v1"),
    ("demographic",       "v2"),
    ("contradiction",     "v1"),
    ("contradiction",     "v2"),
    ("paraphrase",        "v1"),
    ("paraphrase",        "v2"),
    ("paraphrase",        "v3"),
    ("sliding_context",   "v1"),
    ("sliding_context",   "v2"),
    ("sliding_context",   "v3"),
    ("perfect_retrieval", "v1"),
    ("perfect_retrieval", "v2"),
    ("perfect_retrieval", "v3"),
]

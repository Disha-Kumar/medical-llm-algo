from dataclasses import dataclass
from abc import ABC, abstractmethod
from PIL import Image
import json
import os
import re

from src.pipelines.common import CHEXPERT_LABELS


# ── Negation phrases that appear before a label and invert its meaning ──
_NEGATION_PHRASES = [
    "no evidence of", "no signs of", "no ", "without ", "absence of ",
    "negative for ", "rule out ", "rules out ", "ruled out ",
    "unlikely ", "not consistent with ", "no definite ", "no acute ",
    "no significant ", "not suggestive of ", "no obvious ",
    "no clear ", "does not suggest ", "does not show ",
]

# ── Hedging language mapped to approximate confidence ──
_CONFIDENCE_CUES = [
    (0.85, ["consistent with", "indicative of", "characteristic of", "diagnostic of"]),
    (0.75, ["likely", "probable", "in keeping with"]),
    (0.60, ["suggests", "suggestive of", "may represent", "possibly",
             "could represent", "may indicate"]),
    (0.50, ["concerning for", "suspicious for", "cannot exclude",
             "cannot rule out", "questionable"]),
    (0.35, ["unlikely", "low probability", "doubtful"]),
]


def _normalize_to_label(raw_diagnosis: str) -> str:
    """
    Snap a free-form diagnosis string to the nearest CheXpert label.

    Strategy (in priority order):
      1. Exact match (case-insensitive).
      2. Substring match: the label appears inside the raw string, or vice versa.
      3. Token overlap: pick the label that shares the most words with the raw string.
      4. Fallback to "no finding".
    """
    normalized = raw_diagnosis.strip().lower()

    for label in CHEXPERT_LABELS:
        if normalized == label:
            return label

    for label in CHEXPERT_LABELS:
        if label in normalized or normalized in label:
            return label

    raw_tokens = set(normalized.split())
    best_label, best_score = "no finding", 0
    for label in CHEXPERT_LABELS:
        label_tokens = set(label.split())
        score = len(raw_tokens & label_tokens)
        if score > best_score:
            best_score, best_label = score, label

    return best_label if best_score > 0 else "no finding"


def _is_negated(text: str, label_start: int) -> bool:
    """Check whether the label at position label_start is negated.

    Uses rfind to locate the CLOSEST negation phrase before the label.
    Respects sentence boundaries: a period-space ('. ') between the
    negation phrase and the label means they are in different clauses,
    so the negation does not carry over.

    Also handles conjunction patterns like 'no X or Y' where the
    negation extends across 'or' / ',' within the same sentence.
    """
    window_start = max(0, label_start - 80)
    window = text[window_start:label_start].lower()

    # Direct negation: closest negation phrase within 40 chars
    for phrase in _NEGATION_PHRASES:
        pos = window.rfind(phrase)
        if pos == -1:
            continue
        gap = len(window) - (pos + len(phrase))
        if gap > 40:
            continue
        # Sentence boundary: period-space between negation and label
        between = window[pos + len(phrase):]
        if ". " in between:
            continue
        return True

    # Conjunction extension: 'no X or Y', 'no X, Y, or Z'
    for cp in [" or ", ", or ", ", "]:
        cp_pos = window.rfind(cp)
        if cp_pos == -1:
            continue
        # If there is a sentence boundary between the conjunction and
        # the label position, the conjunction does not apply.
        after_conj = window[cp_pos:]
        if ". " in after_conj:
            continue
        pre_conj = window[:cp_pos]
        for phrase in _NEGATION_PHRASES:
            if phrase in pre_conj:
                neg_pos = pre_conj.rfind(phrase)
                between = pre_conj[neg_pos:]
                if ". " not in between:
                    return True

    return False


def _extract_label_from_freeform(raw: str) -> str:
    """Scan free-form text for CheXpert labels with negation awareness.

    Each non-negated mention scores +1.  Each negated mention scores -0.5.
    Ties are broken by earliest non-negated position in the text (the
    primary finding is usually mentioned first).
    Returns empty string if nothing found.
    """
    raw_lower = raw.lower()
    scores = {}
    first_pos = {}

    for label in CHEXPERT_LABELS:
        if label == "no finding":
            continue
        start = 0
        while True:
            pos = raw_lower.find(label, start)
            if pos == -1:
                break
            if _is_negated(raw_lower, pos):
                scores[label] = scores.get(label, 0) - 0.5
            else:
                scores[label] = scores.get(label, 0) + 1.0
                if label not in first_pos:
                    first_pos[label] = pos
            start = pos + len(label)

    if not scores:
        if "no finding" in raw_lower or "normal" in raw_lower:
            return "no finding"
        return ""

    best_score = max(scores.values())
    if best_score <= 0:
        return "no finding"

    # Among labels with the best score, pick the one mentioned first
    candidates = [l for l, s in scores.items()
                  if s == best_score and l in first_pos]
    if not candidates:
        return "no finding"
    candidates.sort(key=lambda l: first_pos[l])
    return candidates[0]


def _infer_confidence(raw: str) -> float:
    """Guess a confidence score from hedging language in free-form text."""
    raw_lower = raw.lower()
    for conf, phrases in _CONFIDENCE_CUES:
        for phrase in phrases:
            if phrase in raw_lower:
                return conf
    return 0.5


@dataclass
class ModelOutput:
    model_name: str
    case_id: str
    diagnosis: str
    confidence: float
    explanation: str
    raw_response: str
    ground_truth: str


class MedicalVLM(ABC):
    @abstractmethod
    def predict(self, image: Image.Image, text: str,
                case_id: str, ground_truth: str) -> ModelOutput:
        pass


def parse_output(raw: str, model_name: str,
                 case_id: str, ground_truth: str) -> ModelOutput:
    """Parse model output into a structured ModelOutput.

    Three-level fallback strategy:
      1. Primary: lines starting with DIAGNOSIS: / CONFIDENCE: / EXPLANATION:
      2. Fallback 1: regex search for those keywords anywhere in the text
      3. Fallback 2: negation-aware label extraction from free-form prose

    Backward compatible: levels 2-3 only trigger when level 1 fails,
    so existing GPT-4o / Qwen2-VL parsing is unaffected.
    """
    diagnosis   = ""
    confidence  = 0.0
    explanation = ""

    # ── PRIMARY PARSE: lines starting with the expected keywords ──
    for line in raw.strip().splitlines():
        line = line.strip()
        if line.upper().startswith("DIAGNOSIS:"):
            diagnosis = line[len("DIAGNOSIS:"):].strip()
        elif line.upper().startswith("CONFIDENCE:"):
            try:
                val = float(re.sub(r'[^0-9.]', '', line.split(":", 1)[1].strip()))
                confidence = val / 100.0 if val > 1.0 else val
            except (ValueError, IndexError):
                confidence = 0.0
        elif line.upper().startswith("EXPLANATION:"):
            explanation = line[len("EXPLANATION:"):].strip()

    # ── FALLBACK 1: search for keywords ANYWHERE in the text ──
    if not diagnosis:
        diag_match = re.search(r'DIAGNOSIS:\s*(.+)', raw, re.IGNORECASE)
        if diag_match:
            diagnosis = diag_match.group(1).strip().split("\n")[0]

    if not confidence and not diagnosis:
        conf_match = re.search(r'CONFIDENCE:\s*([\d.]+)', raw, re.IGNORECASE)
        if conf_match:
            try:
                val = float(conf_match.group(1))
                confidence = val / 100.0 if val > 1.0 else val
            except ValueError:
                pass

    if not explanation:
        expl_match = re.search(r'EXPLANATION:\s*(.+)', raw, re.IGNORECASE)
        if expl_match:
            explanation = expl_match.group(1).strip().split("\n")[0]

    # ── FALLBACK 2: negation-aware label extraction from free-form text ──
    if not diagnosis:
        diagnosis = _extract_label_from_freeform(raw)

    # ── Assemble the result ──
    if not diagnosis:
        diagnosis   = "parse_failed"
        explanation = raw.strip()
        confidence  = 0.0
    else:
        diagnosis = _normalize_to_label(diagnosis)
        if confidence == 0.0:
            # Try once more to extract explicit confidence
            conf_match = re.search(r'CONFIDENCE:\s*([\d.]+)', raw, re.IGNORECASE)
            if conf_match:
                try:
                    val = float(conf_match.group(1))
                    confidence = val / 100.0 if val > 1.0 else val
                except ValueError:
                    confidence = _infer_confidence(raw)
            else:
                confidence = _infer_confidence(raw)
        if not explanation:
            explanation = raw.strip()

    return ModelOutput(
        model_name=model_name,
        case_id=case_id,
        diagnosis=diagnosis,
        confidence=min(max(confidence, 0.0), 1.0),
        explanation=explanation if explanation else raw.strip(),
        raw_response=raw,
        ground_truth=ground_truth,
    )


def save_output(output: ModelOutput, output_dir: str):
    path = os.path.join(output_dir, output.model_name, f"{output.case_id}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(output.__dict__, f, indent=2)


def load_output(model_name: str, case_id: str, output_dir: str) -> ModelOutput:
    path = os.path.join(output_dir, model_name, f"{case_id}.json")
    with open(path) as f:
        return ModelOutput(**json.load(f))

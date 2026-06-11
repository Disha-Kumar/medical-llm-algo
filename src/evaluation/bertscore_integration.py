"""Integration module for BERTScore computation and threshold gating.

This module provides utilities to:
1. Compute BERTScore between model outputs and ground truth
2. Gate predictions based on configurable thresholds
3. Log threshold decisions for analysis
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

from src.evaluation.bertscore_config import BERTScoreConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class BERTScoreResult:
    """Result of BERTScore computation.
    
    Attributes:
        f1_score: F1 score (harmonic mean of precision and recall)
        precision: Precision score
        recall: Recall score
        accepted: Whether prediction meets threshold criteria
        reason: Explanation of acceptance/rejection
    """
    f1_score: float
    precision: float
    recall: float
    accepted: bool
    reason: str


def compute_bertscore_placeholder(
    prediction: str,
    reference: str,
    model_type: str = "bert-base-uncased"
) -> Dict[str, float]:
    """Placeholder for BERTScore computation.
    
    NOTE: Requires `pip install bert-score`
    
    This is a stub that should be replaced with actual BERTScore computation
    using the bert_score library:
    
        from bert_score import score
        P, R, F1 = score([prediction], [reference], lang="en", model_type=model_type)
        return {
            "precision": float(P[0]),
            "recall": float(R[0]),
            "f1": float(F1[0])
        }
    
    Args:
        prediction: Model's predicted text
        reference: Ground truth text
        model_type: BERT variant to use
        
    Returns:
        Dictionary with 'precision', 'recall', 'f1' scores (0.0-1.0)
    """
    # TODO: Implement actual BERTScore computation
    # This is a placeholder that returns dummy values
    logger.warning(
        "BERTScore computation not implemented. Install bert-score: "
        "pip install bert-score"
    )
    return {"precision": 0.85, "recall": 0.85, "f1": 0.85}


def evaluate_with_threshold(
    diagnosis: str,
    ground_truth: str,
    confidence: float,
    config: Optional[BERTScoreConfig] = None,
    case_id: Optional[str] = None,
) -> BERTScoreResult:
    """Evaluate diagnosis prediction against threshold.
    
    Args:
        diagnosis: Model's predicted diagnosis
        ground_truth: Ground truth diagnosis
        confidence: Model's confidence in prediction
        config: BERTScoreConfig (defaults to DEFAULT_CONFIG)
        case_id: Optional case ID for logging
        
    Returns:
        BERTScoreResult with acceptance decision and scoring
    """
    config = config or DEFAULT_CONFIG
    
    # Compute BERTScore
    bertscore_dict = compute_bertscore_placeholder(diagnosis, ground_truth)
    f1 = bertscore_dict.get("f1", 0.0)
    precision = bertscore_dict.get("precision", 0.0)
    recall = bertscore_dict.get("recall", 0.0)
    
    # Check threshold
    accepted = f1 >= config.primary_threshold
    
    if accepted:
        reason = (
            f"BERTScore F1={f1:.4f} meets primary threshold "
            f"{config.primary_threshold}"
        )
    else:
        reason = (
            f"BERTScore F1={f1:.4f} below primary threshold "
            f"{config.primary_threshold}"
        )
    
    if case_id:
        logger.info(f"Case {case_id}: {reason}")
    
    return BERTScoreResult(
        f1_score=f1,
        precision=precision,
        recall=recall,
        accepted=accepted,
        reason=reason,
    )


def batch_evaluate_with_thresholds(
    cases: list[Dict],
    thresholds: Optional[list[float]] = None,
    config: Optional[BERTScoreConfig] = None,
) -> Dict[float, Dict]:
    """Batch evaluate cases across multiple thresholds.
    
    Useful for sensitivity analysis without re-computing BERTScores.
    
    Args:
        cases: List of case dicts with 'diagnosis', 'ground_truth', 'confidence'
        thresholds: List of thresholds to test (defaults to config.sensitivity_thresholds)
        config: BERTScoreConfig
        
    Returns:
        Dictionary mapping threshold -> evaluation results
    """
    config = config or DEFAULT_CONFIG
    thresholds = thresholds or config.sensitivity_thresholds
    
    # Pre-compute BERTScores for all cases
    cases_with_scores = []
    for case in cases:
        bertscore_dict = compute_bertscore_placeholder(
            case["diagnosis"],
            case["ground_truth"],
        )
        case_with_score = {**case, **bertscore_dict}
        cases_with_scores.append(case_with_score)
    
    # Evaluate against each threshold
    results = {}
    for threshold in thresholds:
        accepted = sum(
            1 for c in cases_with_scores if c["f1"] >= threshold
        )
        rejected = len(cases_with_scores) - accepted
        correct_accepted = sum(
            1 for c in cases_with_scores
            if c["f1"] >= threshold and c["diagnosis"] == c["ground_truth"]
        )
        
        results[threshold] = {
            "total_cases": len(cases_with_scores),
            "accepted": accepted,
            "rejected": rejected,
            "acceptance_rate": accepted / len(cases_with_scores) * 100,
            "correct_accepted": correct_accepted,
            "precision": correct_accepted / accepted if accepted > 0 else 0,
        }
    
    return results


def print_threshold_comparison(results: Dict[float, Dict]) -> None:
    """Print comparison of threshold evaluation results.
    
    Args:
        results: Output from batch_evaluate_with_thresholds
    """
    print("\n" + "="*80)
    print("THRESHOLD EVALUATION RESULTS")
    print("="*80)
    print(
        f"{'Threshold':<12} {'Accepted':<12} {'Rejected':<12} "
        f"{'Acceptance%':<12} {'Precision':<12}"
    )
    print("-"*80)
    
    for threshold in sorted(results.keys()):
        r = results[threshold]
        print(
            f"{threshold:<12.2f} "
            f"{r['accepted']:<12} "
            f"{r['rejected']:<12} "
            f"{r['acceptance_rate']:<11.1f}% "
            f"{r['precision']:<12.4f}"
        )
    print("="*80 + "\n")

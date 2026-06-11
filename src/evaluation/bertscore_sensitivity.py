"""BERTScore sensitivity analysis framework for medical VLM evaluation.

This module provides tools to analyze how diagnosis stability and accuracy vary
across different BERTScore thresholds, enabling empirical threshold selection
for medical AI applications.

Key Functions:
- run_sensitivity_analysis(): Execute full sensitivity analysis pipeline
- compute_threshold_metrics(): Calculate performance metrics for each threshold
- generate_sensitivity_report(): Create human-readable analysis report
"""

from __future__ import annotations

import json
import csv
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

from src.evaluation.bertscore_config import BERTScoreConfig


logger = logging.getLogger(__name__)


@dataclass
class ThresholdMetrics:
    """Metrics computed for a single threshold value.
    
    Attributes:
        threshold: The BERTScore threshold being evaluated.
        total_cases: Total number of cases analyzed.
        accepted_cases: Cases with BERTScore >= threshold.
        rejected_cases: Cases with BERTScore < threshold.
        acceptance_rate: Percentage of cases accepted (0-100).
        correct_accepted: Cases with correct diagnosis among accepted.
        incorrect_accepted: Cases with incorrect diagnosis among accepted.
        precision: Correct / (Correct + Incorrect) for accepted cases.
        sensitivity: Correct accepted / Total correct cases.
        diagnosis_flip_rate: Rate of diagnosis changes across windows.
        hallucination_rate: Rate of hallucination flags.
        avg_confidence: Average model confidence for accepted cases.
    """
    
    threshold: float
    total_cases: int
    accepted_cases: int
    rejected_cases: int
    acceptance_rate: float
    correct_accepted: int
    incorrect_accepted: int
    precision: Optional[float]
    sensitivity: Optional[float]
    diagnosis_flip_rate: float
    hallucination_rate: float
    avg_confidence: float
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CaseEvaluation:
    """Per-case evaluation data for sensitivity analysis.
    
    Attributes:
        case_id: Unique case identifier.
        model: Model name.
        bertscore: BERTScore value (0.0-1.0).
        diagnosis: Model's diagnosis prediction.
        ground_truth: Ground truth diagnosis.
        confidence: Model confidence in prediction.
        diagnosis_flipped: Whether diagnosis changed across windows.
        hallucination_flag: Whether hallucination was detected.
        correct: Whether diagnosis matches ground truth.
    """
    
    case_id: str
    model: str
    bertscore: float
    diagnosis: str
    ground_truth: str
    confidence: float
    diagnosis_flipped: bool
    hallucination_flag: bool
    correct: bool


class SensitivityAnalyzer:
    """Analyzes BERTScore threshold sensitivity for medical VLM outputs."""
    
    def __init__(self, config: Optional[BERTScoreConfig] = None):
        """Initialize analyzer with configuration.
        
        Args:
            config: BERTScoreConfig instance. If None, uses defaults.
        """
        self.config = config or BERTScoreConfig()
        self.case_evaluations: List[CaseEvaluation] = []
        self.threshold_metrics: Dict[float, ThresholdMetrics] = {}
    
    def add_case(self, case: CaseEvaluation) -> None:
        """Add a case evaluation to the analysis.
        
        Args:
            case: CaseEvaluation instance.
        """
        if not (0.0 <= case.bertscore <= 1.0):
            logger.warning(
                f"BERTScore {case.bertscore} out of range [0.0, 1.0] for case {case.case_id}"
            )
        self.case_evaluations.append(case)
    
    def add_cases_from_jsonl(self, filepath: str) -> int:
        """Load case evaluations from JSONL file.
        
        Expected JSONL format:
        {
            "case_id": "...",
            "model": "...",
            "bertscore": 0.85,
            "diagnosis": "...",
            "ground_truth": "...",
            "confidence": 0.92,
            "diagnosis_flipped": false,
            "hallucination_flag": false,
            "correct": true
        }
        
        Args:
            filepath: Path to JSONL file.
            
        Returns:
            Number of cases loaded.
        """
        count = 0
        try:
            with open(filepath, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        case = CaseEvaluation(
                            case_id=data["case_id"],
                            model=data["model"],
                            bertscore=float(data["bertscore"]),
                            diagnosis=data["diagnosis"],
                            ground_truth=data["ground_truth"],
                            confidence=float(data["confidence"]),
                            diagnosis_flipped=data.get("diagnosis_flipped", False),
                            hallucination_flag=data.get("hallucination_flag", False),
                            correct=data.get("correct", data["diagnosis"] == data["ground_truth"])
                        )
                        self.add_case(case)
                        count += 1
                    except (KeyError, ValueError) as e:
                        logger.warning(f"Failed to parse line {line_num}: {e}")
            logger.info(f"Loaded {count} cases from {filepath}")
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
        
        return count
    
    def compute_metrics(self, threshold: float) -> ThresholdMetrics:
        """Compute performance metrics for a given threshold.
        
        Args:
            threshold: BERTScore threshold to evaluate.
            
        Returns:
            ThresholdMetrics instance.
        """
        if not self.case_evaluations:
            raise ValueError("No case evaluations loaded")
        
        total = len(self.case_evaluations)
        accepted = [c for c in self.case_evaluations if c.bertscore >= threshold]
        rejected = [c for c in self.case_evaluations if c.bertscore < threshold]
        
        accepted_count = len(accepted)
        rejected_count = len(rejected)
        acceptance_rate = (accepted_count / total * 100) if total > 0 else 0
        
        # Correctness metrics among accepted cases
        correct_accepted = sum(1 for c in accepted if c.correct)
        incorrect_accepted = sum(1 for c in accepted if not c.correct)
        
        # Precision: of accepted cases, how many were correct
        precision = (
            correct_accepted / accepted_count
            if accepted_count > 0
            else None
        )
        
        # Sensitivity (recall): of all correct cases, how many were accepted
        total_correct = sum(1 for c in self.case_evaluations if c.correct)
        sensitivity = (
            correct_accepted / total_correct
            if total_correct > 0
            else None
        )
        
        # Diagnosis flip rate among accepted
        flip_rate = (
            sum(1 for c in accepted if c.diagnosis_flipped) / accepted_count * 100
            if accepted_count > 0
            else 0
        )
        
        # Hallucination rate among accepted
        hallucination_rate = (
            sum(1 for c in accepted if c.hallucination_flag) / accepted_count * 100
            if accepted_count > 0
            else 0
        )
        
        # Average confidence among accepted
        avg_confidence = (
            sum(c.confidence for c in accepted) / accepted_count
            if accepted_count > 0
            else 0
        )
        
        return ThresholdMetrics(
            threshold=threshold,
            total_cases=total,
            accepted_cases=accepted_count,
            rejected_cases=rejected_count,
            acceptance_rate=acceptance_rate,
            correct_accepted=correct_accepted,
            incorrect_accepted=incorrect_accepted,
            precision=precision,
            sensitivity=sensitivity,
            diagnosis_flip_rate=flip_rate,
            hallucination_rate=hallucination_rate,
            avg_confidence=avg_confidence,
        )
    
    def run_analysis(self, thresholds: Optional[List[float]] = None) -> Dict[float, ThresholdMetrics]:
        """Run sensitivity analysis across all thresholds.
        
        Args:
            thresholds: List of thresholds to analyze. If None, uses config.sensitivity_thresholds.
            
        Returns:
            Dictionary mapping threshold -> ThresholdMetrics.
        """
        thresholds = thresholds or self.config.sensitivity_thresholds
        
        logger.info(f"Running sensitivity analysis on {len(self.case_evaluations)} cases")
        logger.info(f"Testing thresholds: {thresholds}")
        
        for threshold in sorted(thresholds):
            try:
                metrics = self.compute_metrics(threshold)
                self.threshold_metrics[threshold] = metrics
                logger.info(
                    f"Threshold {threshold:.2f}: "
                    f"Accepted {metrics.accepted_cases}/{metrics.total_cases} "
                    f"({metrics.acceptance_rate:.1f}%), "
                    f"Precision {metrics.precision:.3f if metrics.precision else None}"
                )
            except Exception as e:
                logger.error(f"Failed to compute metrics for threshold {threshold}: {e}")
        
        return self.threshold_metrics
    
    def save_results_csv(self, output_file: str = "bertscore_sensitivity_results.csv") -> None:
        """Save sensitivity analysis results to CSV.
        
        Args:
            output_file: Output CSV filename.
        """
        if not self.threshold_metrics:
            logger.warning("No metrics computed. Run analysis first.")
            return
        
        rows = []
        for threshold in sorted(self.threshold_metrics.keys()):
            metrics = self.threshold_metrics[threshold]
            rows.append({
                'Threshold': f"{metrics.threshold:.2f}",
                'Accepted_Cases': metrics.accepted_cases,
                'Rejected_Cases': metrics.rejected_cases,
                'Acceptance_Rate_%': f"{metrics.acceptance_rate:.2f}",
                'Correct_Accepted': metrics.correct_accepted,
                'Incorrect_Accepted': metrics.incorrect_accepted,
                'Precision': f"{metrics.precision:.4f}" if metrics.precision else "N/A",
                'Sensitivity': f"{metrics.sensitivity:.4f}" if metrics.sensitivity else "N/A",
                'Diagnosis_Flip_Rate_%': f"{metrics.diagnosis_flip_rate:.2f}",
                'Hallucination_Rate_%': f"{metrics.hallucination_rate:.2f}",
                'Avg_Confidence': f"{metrics.avg_confidence:.4f}",
            })
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        
        logger.info(f"Results saved to {output_file}")
    
    def save_detailed_csv(self, output_file: str = "bertscore_sensitivity_detailed.csv") -> None:
        """Save per-case analysis to CSV.
        
        Args:
            output_file: Output CSV filename.
        """
        rows = [
            {
                'case_id': case.case_id,
                'model': case.model,
                'bertscore': f"{case.bertscore:.4f}",
                'diagnosis': case.diagnosis,
                'ground_truth': case.ground_truth,
                'correct': 'Yes' if case.correct else 'No',
                'confidence': f"{case.confidence:.4f}",
                'diagnosis_flipped': 'Yes' if case.diagnosis_flipped else 'No',
                'hallucination_flag': 'Yes' if case.hallucination_flag else 'No',
            }
            for case in self.case_evaluations
        ]
        
        if rows:
            with open(output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            logger.info(f"Detailed results saved to {output_file}")
    
    def generate_report(self) -> str:
        """Generate human-readable sensitivity analysis report.
        
        Returns:
            Formatted report string.
        """
        if not self.threshold_metrics:
            return "No metrics computed. Run analysis first."
        
        lines = []
        lines.append("\n" + "="*100)
        lines.append("BERTSCORE SENSITIVITY ANALYSIS REPORT")
        lines.append("="*100)
        
        lines.append(f"\nTotal Cases Analyzed: {len(self.case_evaluations)}")
        lines.append(f"Total Correct Predictions: {sum(1 for c in self.case_evaluations if c.correct)}")
        lines.append(f"Overall Accuracy: {sum(1 for c in self.case_evaluations if c.correct) / len(self.case_evaluations) * 100:.2f}%")
        
        lines.append("\n" + "-"*100)
        lines.append("THRESHOLD PERFORMANCE METRICS")
        lines.append("-"*100)
        
        header = (
            f"{'Threshold':<12} {'Acceptance':<12} {'Precision':<12} {'Sensitivity':<12} "
            f"{'Flip Rate':<12} {'Hallucin.':<12}"
        )
        lines.append(header)
        lines.append("-"*100)
        
        for threshold in sorted(self.threshold_metrics.keys()):
            metrics = self.threshold_metrics[threshold]
            precision_str = f"{metrics.precision:.4f}" if metrics.precision else "N/A"
            sensitivity_str = f"{metrics.sensitivity:.4f}" if metrics.sensitivity else "N/A"
            
            lines.append(
                f"{metrics.threshold:<12.2f} "
                f"{metrics.acceptance_rate:<11.1f}% "
                f"{precision_str:<12} "
                f"{sensitivity_str:<12} "
                f"{metrics.diagnosis_flip_rate:<11.1f}% "
                f"{metrics.hallucination_rate:<11.1f}%"
            )
        
        lines.append("\n" + "-"*100)
        lines.append("RECOMMENDATIONS")
        lines.append("-"*100)
        
        # Find threshold with best precision-sensitivity balance
        best_threshold = max(
            self.threshold_metrics.items(),
            key=lambda x: (x[1].precision or 0) * (x[1].sensitivity or 0)
            if x[1].precision and x[1].sensitivity else 0,
            default=(None, None)
        )
        
        if best_threshold[0]:
            metrics = best_threshold[1]
            lines.append(
                f"\n• Optimal Threshold (best precision-sensitivity balance): {best_threshold[0]:.2f}"
            )
            lines.append(f"  - Accepts {metrics.accepted_cases} cases ({metrics.acceptance_rate:.1f}%)")
            lines.append(f"  - Precision: {metrics.precision:.4f} (correct predictions among accepted)")
            lines.append(f"  - Sensitivity: {metrics.sensitivity:.4f} (correct cases captured)")
            lines.append(f"  - Diagnosis flip rate: {metrics.diagnosis_flip_rate:.2f}%")
            lines.append(f"  - Hallucination rate: {metrics.hallucination_rate:.2f}%")
        
        # Current primary threshold performance
        primary_metrics = self.threshold_metrics.get(self.config.primary_threshold)
        if primary_metrics:
            lines.append(f"\n• Current Primary Threshold: {self.config.primary_threshold:.2f}")
            lines.append(f"  - Accepts {primary_metrics.accepted_cases} cases ({primary_metrics.acceptance_rate:.1f}%)")
            lines.append(f"  - Precision: {primary_metrics.precision:.4f}")
            lines.append(f"  - Sensitivity: {primary_metrics.sensitivity:.4f}")
        
        lines.append("\n" + "="*100 + "\n")
        
        return "\n".join(lines)


def run_sensitivity_analysis(
    cases: List[CaseEvaluation],
    thresholds: Optional[List[float]] = None,
    config: Optional[BERTScoreConfig] = None,
) -> Dict[float, ThresholdMetrics]:
    """Convenience function to run sensitivity analysis.
    
    Args:
        cases: List of CaseEvaluation instances.
        thresholds: List of thresholds to test. If None, uses config defaults.
        config: BERTScoreConfig instance. If None, uses defaults.
        
    Returns:
        Dictionary mapping threshold -> ThresholdMetrics.
    """
    analyzer = SensitivityAnalyzer(config)
    for case in cases:
        analyzer.add_case(case)
    return analyzer.run_analysis(thresholds)

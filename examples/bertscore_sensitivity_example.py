#!/usr/bin/env python3
"""Example: Run BERTScore sensitivity analysis on evaluation data.

This script demonstrates how to use the sensitivity analysis framework
to evaluate threshold choices for your medical VLM evaluation pipeline.

Usage:
    python examples/bertscore_sensitivity_example.py [--input FILE] [--output DIR]
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.bertscore_sensitivity import (
    SensitivityAnalyzer,
    CaseEvaluation,
    BERTScoreConfig,
)


def create_example_cases():
    """Create example case data for demonstration.
    
    Returns:
        List of CaseEvaluation instances with synthetic data.
    """
    cases = [
        CaseEvaluation(
            case_id="case_001",
            model="gpt4o",
            bertscore=0.92,
            diagnosis="pneumonia",
            ground_truth="pneumonia",
            confidence=0.95,
            diagnosis_flipped=False,
            hallucination_flag=False,
            correct=True,
        ),
        CaseEvaluation(
            case_id="case_002",
            model="gpt4o",
            bertscore=0.78,
            diagnosis="cardiomegaly",
            ground_truth="pneumonia",
            confidence=0.72,
            diagnosis_flipped=True,
            hallucination_flag=False,
            correct=False,
        ),
        CaseEvaluation(
            case_id="case_003",
            model="qwen2_vl",
            bertscore=0.88,
            diagnosis="no finding",
            ground_truth="no finding",
            confidence=0.91,
            diagnosis_flipped=False,
            hallucination_flag=False,
            correct=True,
        ),
        CaseEvaluation(
            case_id="case_004",
            model="qwen2_vl",
            bertscore=0.65,
            diagnosis="pleural effusion",
            ground_truth="no finding",
            confidence=0.58,
            diagnosis_flipped=False,
            hallucination_flag=True,
            correct=False,
        ),
        CaseEvaluation(
            case_id="case_005",
            model="biovil_t",
            bertscore=0.85,
            diagnosis="pneumonia",
            ground_truth="pneumonia",
            confidence=0.82,
            diagnosis_flipped=False,
            hallucination_flag=False,
            correct=True,
        ),
        # Add more cases to represent your actual evaluation dataset
        # ...
    ]
    return cases


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run BERTScore sensitivity analysis on evaluation data"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to JSONL file with case evaluations (optional, uses example data if not provided)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/bertscore_sensitivity",
        help="Output directory for results (default: results/bertscore_sensitivity)",
    )
    parser.add_argument(
        "--primary-threshold",
        type=float,
        default=0.85,
        help="Primary threshold to evaluate (default: 0.85)",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
        help="Thresholds to test (default: 0.70 0.75 0.80 0.85 0.90 0.95)",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load cases
    print("Loading evaluation cases...")
    if args.input and Path(args.input).exists():
        analyzer = SensitivityAnalyzer()
        count = analyzer.add_cases_from_jsonl(args.input)
        print(f"✓ Loaded {count} cases from {args.input}")
    else:
        cases = create_example_cases()
        analyzer = SensitivityAnalyzer()
        for case in cases:
            analyzer.add_case(case)
        print(f"✓ Using {len(cases)} example cases")
        if args.input:
            print(f"  Note: {args.input} not found, using synthetic data for demonstration")
    
    # Configure analyzer
    config = BERTScoreConfig(
        primary_threshold=args.primary_threshold,
        sensitivity_thresholds=args.thresholds,
    )
    analyzer.config = config
    
    # Run sensitivity analysis
    print(f"\nRunning sensitivity analysis on {len(analyzer.case_evaluations)} cases...")
    metrics = analyzer.run_analysis(args.thresholds)
    print(f"✓ Analysis complete: {len(metrics)} threshold(s) evaluated")
    
    # Save results
    print(f"\nSaving results to {output_dir}...")
    analyzer.save_results_csv(str(output_dir / "sensitivity_results.csv"))
    analyzer.save_detailed_csv(str(output_dir / "sensitivity_detailed.csv"))
    print("✓ Results saved")
    
    # Print report
    report = analyzer.generate_report()
    print(report)
    
    # Save report to file
    report_file = output_dir / "sensitivity_report.txt"
    with open(report_file, "w") as f:
        f.write(report)
    print(f"✓ Report saved to {report_file}")
    
    print(f"\n📊 Full results available in {output_dir}/")


if __name__ == "__main__":
    main()

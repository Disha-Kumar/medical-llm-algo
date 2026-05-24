#!/usr/bin/env python3
"""
Sliding Window Diagnosis Stability Analysis for Medical VLM Outputs

This script analyzes evaluation harness outputs from a medical vision-language model
pipeline to compute diagnosis stability across different image window sizes.

Input: JSONL file with (model, case, condition) triples
Output: Summary statistics and CSV file with per-case analysis
"""

import json
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def read_jsonl_file(filepath: str) -> List[Dict]:
    """
    Read a JSONL file and return list of JSON objects.
    
    Args:
        filepath: Path to the JSONL file
        
    Returns:
        List of dictionaries, one per line
    """
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse JSON line: {line[:80]}... Error: {e}")
    return data


def validate_record(record: Dict) -> bool:
    """
    Validate that a record meets requirements for analysis.
    
    Args:
        record: Dictionary representing one evaluation result
        
    Returns:
        True if record is valid, False otherwise
    """
    # Skip rows where status != "PASS"
    if record.get('status') != 'PASS':
        return False
    
    # Skip if diagnosis is missing or null
    if not record.get('diagnosis'):
        return False
    
    # All other fields should be present
    required_fields = ['case_id', 'condition', 'diagnosis', 'confidence', 'ground_truth']
    for field in required_fields:
        if field not in record:
            return False
    
    return True


def group_by_case(data: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Group records by case_id.
    
    Args:
        data: List of all evaluation records
        
    Returns:
        Dictionary mapping case_id to list of records for that case
    """
    cases = defaultdict(list)
    for record in data:
        if validate_record(record):
            cases[record['case_id']].append(record)
    return dict(cases)


def extract_window_predictions(case_records: List[Dict]) -> Optional[Dict[str, Dict]]:
    """
    Extract predictions for each window size (4k, 8k, 16k).
    
    Args:
        case_records: List of records for a single case
        
    Returns:
        Dictionary with keys 'window_4k', 'window_8k', 'window_16k' mapping to
        prediction data, or None if any window is missing
    """
    windows = {}
    
    for record in case_records:
        condition = record.get('condition')
        if condition in ['window_4k', 'window_8k', 'window_16k']:
            windows[condition] = {
                'diagnosis': record['diagnosis'],
                'confidence': record['confidence'],
                'ground_truth': record['ground_truth'],
                'hallucination_flag': record.get('hallucination_flag', False)
            }
    
    # Return None if all 3 windows are not present
    required_windows = {'window_4k', 'window_8k', 'window_16k'}
    if set(windows.keys()) != required_windows:
        return None
    
    return windows


def check_diagnosis_flip(windows: Dict[str, Dict]) -> Tuple[bool, Dict]:
    """
    Check if diagnosis flips across windows and analyze pairwise flips.
    
    Args:
        windows: Dictionary with window predictions
        
    Returns:
        Tuple of (overall_flip, pairwise_flips_dict)
        - overall_flip: True if any diagnosis differs
        - pairwise_flips_dict: Dictionary with keys 'flip_4k_8k', 'flip_8k_16k', 'flip_4k_16k'
    """
    diag_4k = windows['window_4k']['diagnosis']
    diag_8k = windows['window_8k']['diagnosis']
    diag_16k = windows['window_16k']['diagnosis']
    
    # Check pairwise flips
    pairwise_flips = {
        'flip_4k_8k': diag_4k != diag_8k,
        'flip_8k_16k': diag_8k != diag_16k,
        'flip_4k_16k': diag_4k != diag_16k
    }
    
    # Overall flip: if ANY diagnosis differs
    overall_flip = any(pairwise_flips.values())
    
    return overall_flip, pairwise_flips


def compute_confidence_change(windows: Dict[str, Dict]) -> Dict[str, float]:
    """
    Compute average confidence change across windows.
    
    Args:
        windows: Dictionary with window predictions
        
    Returns:
        Dictionary with confidence statistics
    """
    conf_4k = windows['window_4k']['confidence']
    conf_8k = windows['window_8k']['confidence']
    conf_16k = windows['window_16k']['confidence']
    
    change_4k_to_8k = conf_8k - conf_4k
    change_8k_to_16k = conf_16k - conf_8k
    change_4k_to_16k = conf_16k - conf_4k
    
    return {
        'conf_4k': conf_4k,
        'conf_8k': conf_8k,
        'conf_16k': conf_16k,
        'change_4k_to_8k': change_4k_to_8k,
        'change_8k_to_16k': change_8k_to_16k,
        'change_4k_to_16k': change_4k_to_16k,
        'avg_abs_change': (abs(change_4k_to_8k) + abs(change_8k_to_16k)) / 2
    }


def analyze_sliding_windows(jsonl_filepath: str, output_csv: str = 'sliding_window_results.csv') -> Dict:
    """
    Main analysis function for sliding window diagnosis stability.
    
    Args:
        jsonl_filepath: Path to input JSONL file
        output_csv: Path to output CSV file
        
    Returns:
        Dictionary with analysis results
    """
    print(f"Reading JSONL file: {jsonl_filepath}")
    data = read_jsonl_file(jsonl_filepath)
    print(f"Total records read: {len(data)}\n")
    
    # Group by case
    cases_grouped = group_by_case(data)
    print(f"Total valid cases after filtering: {len(cases_grouped)}")
    
    # Analyze each case
    results = []
    total_cases = 0
    flip_cases = 0
    pairwise_flips = {'flip_4k_8k': 0, 'flip_8k_16k': 0, 'flip_4k_16k': 0}
    hallucination_counts = {'window_4k': 0, 'window_8k': 0, 'window_16k': 0}
    confidence_changes = []
    
    for case_id, case_records in cases_grouped.items():
        # Extract window predictions
        windows = extract_window_predictions(case_records)
        
        # Skip if all 3 windows not present
        if windows is None:
            continue
        
        total_cases += 1
        
        # Check for diagnosis flip
        has_flip, pair_flips = check_diagnosis_flip(windows)
        if has_flip:
            flip_cases += 1
        
        # Update pairwise flip counts
        for key in pairwise_flips:
            if pair_flips[key]:
                pairwise_flips[key] += 1
        
        # Compute confidence changes
        conf_stats = compute_confidence_change(windows)
        confidence_changes.append(conf_stats)
        
        # Count hallucinations per window
        for window_key in ['window_4k', 'window_8k', 'window_16k']:
            if windows[window_key]['hallucination_flag']:
                hallucination_counts[window_key] += 1
        
        # Store result row
        results.append({
            'case_id': case_id,
            'diag_4k': windows['window_4k']['diagnosis'],
            'diag_8k': windows['window_8k']['diagnosis'],
            'diag_16k': windows['window_16k']['diagnosis'],
            'flip': 'Yes' if has_flip else 'No',
            'conf_4k': windows['window_4k']['confidence'],
            'conf_8k': windows['window_8k']['confidence'],
            'conf_16k': windows['window_16k']['confidence'],
            'ground_truth': windows['window_4k']['ground_truth'],
            'flip_4k_8k': 'Yes' if pair_flips['flip_4k_8k'] else 'No',
            'flip_8k_16k': 'Yes' if pair_flips['flip_8k_16k'] else 'No',
            'flip_4k_16k': 'Yes' if pair_flips['flip_4k_16k'] else 'No'
        })
    
    # Compute summary statistics
    flip_rate = (flip_cases / total_cases * 100) if total_cases > 0 else 0
    
    pairwise_flip_rates = {}
    for key in pairwise_flips:
        rate = (pairwise_flips[key] / total_cases * 100) if total_cases > 0 else 0
        pairwise_flip_rates[key] = rate
    
    # Compute average confidence changes
    avg_conf_change_4k_8k = sum(c['change_4k_to_8k'] for c in confidence_changes) / len(confidence_changes) if confidence_changes else 0
    avg_conf_change_8k_16k = sum(c['change_8k_to_16k'] for c in confidence_changes) / len(confidence_changes) if confidence_changes else 0
    avg_abs_change = sum(c['avg_abs_change'] for c in confidence_changes) / len(confidence_changes) if confidence_changes else 0
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SLIDING WINDOW DIAGNOSIS STABILITY ANALYSIS")
    print("="*60)
    print(f"\nTotal Cases (with all 3 windows): {total_cases}")
    print(f"Flip Cases (diagnosis changes): {flip_cases}")
    print(f"Flip Rate: {flip_rate:.2f}%")
    
    print("\n" + "-"*60)
    print("PAIRWISE FLIP ANALYSIS")
    print("-"*60)
    print(f"4k vs 8k flip rate: {pairwise_flip_rates['flip_4k_8k']:.2f}% ({pairwise_flips['flip_4k_8k']} cases)")
    print(f"8k vs 16k flip rate: {pairwise_flip_rates['flip_8k_16k']:.2f}% ({pairwise_flips['flip_8k_16k']} cases)")
    print(f"4k vs 16k flip rate: {pairwise_flip_rates['flip_4k_16k']:.2f}% ({pairwise_flips['flip_4k_16k']} cases)")
    
    print("\n" + "-"*60)
    print("CONFIDENCE ANALYSIS")
    print("-"*60)
    print(f"Avg confidence change (4k→8k): {avg_conf_change_4k_8k:+.4f}")
    print(f"Avg confidence change (8k→16k): {avg_conf_change_8k_16k:+.4f}")
    print(f"Avg absolute confidence change: {avg_abs_change:.4f}")
    
    print("\n" + "-"*60)
    print("HALLUCINATION ANALYSIS")
    print("-"*60)
    for window, count in hallucination_counts.items():
        pct = (count / total_cases * 100) if total_cases > 0 else 0
        print(f"{window}: {count} cases ({pct:.2f}%)")
    
    # Write results to CSV
    print(f"\n" + "-"*60)
    print(f"Writing results to CSV: {output_csv}")
    print("-"*60)
    
    if results:
        csv_columns = [
            'case_id', 'diag_4k', 'diag_8k', 'diag_16k', 'flip',
            'conf_4k', 'conf_8k', 'conf_16k',
            'flip_4k_8k', 'flip_8k_16k', 'flip_4k_16k',
            'ground_truth'
        ]
        
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"✓ CSV file created with {len(results)} rows")
    else:
        print("! No results to write - no valid cases found")
    
    print("\n" + "="*60)
    
    return {
        'total_cases': total_cases,
        'flip_cases': flip_cases,
        'flip_rate': flip_rate,
        'pairwise_flip_rates': pairwise_flip_rates,
        'pairwise_flips': pairwise_flips,
        'avg_conf_change_4k_8k': avg_conf_change_4k_8k,
        'avg_conf_change_8k_16k': avg_conf_change_8k_16k,
        'avg_abs_change': avg_abs_change,
        'hallucination_counts': hallucination_counts,
        'results_count': len(results)
    }


def main():
    """Main entry point for the script."""
    # Example usage - adjust path to your JSONL file
    import sys
    
    if len(sys.argv) > 1:
        jsonl_file = sys.argv[1]
    else:
        # Default path
        jsonl_file = 'results/evaluation_harness/output.jsonl'
    
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        output_file = 'sliding_window_results.csv'
    
    # Check if input file exists
    if not Path(jsonl_file).exists():
        print(f"Error: Input file not found: {jsonl_file}")
        print("Usage: python sliding_window_analysis.py <input_jsonl> [output_csv]")
        sys.exit(1)
    
    # Run analysis
    try:
        analyze_sliding_windows(jsonl_file, output_file)
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

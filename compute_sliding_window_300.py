#!/usr/bin/env python3
"""
Sliding Window Diagnosis Stability Analysis for Medical VLM Outputs
Extended version: Scale to ~300 cases with sampling, deduplication, and robust handling

Task:
1. Aggregate ALL JSONL files from results/evaluation_harness/ and results/experiment1/
2. Build sliding window dataset (keep only cases with all 3 windows)
3. Scale to ~300 cases (sample if more, use all if fewer)
4. Compute flip rates (4k→8k, 8k→16k, 4k→16k)
5. Save to sliding_window_300cases.csv with summary table

Supports multiple models:
- GPT4o / gpt4o
- Qwen / qwen2_vl
- LLaVA / llava_med
- BioViLT / biovil_t
- CheXagent / chexagent

Usage:
  python compute_sliding_window_300.py                    # Auto-discover all JSONL files
  python compute_sliding_window_300.py path/to/file.jsonl # Analyze specific file
  python compute_sliding_window_300.py results/            # Analyze all files in directory
"""

import json
import csv
import sys
import glob
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class NormalizedPrediction:
    """Standardized prediction format across all models"""
    case_id: str
    model: str
    window: str  # '4k', '8k', '16k'
    diagnosis: str
    confidence: float = 0.0
    ground_truth: str = ""
    hallucination_flag: bool = False
    source_file: str = ""  # Track origin file


@dataclass
class CaseAnalysis:
    """Analysis result for a single case"""
    case_id: str
    model: str
    has_flip: bool
    flip_4k_8k: bool
    flip_8k_16k: bool
    flip_4k_16k: bool
    diag_4k: str
    diag_8k: str
    diag_16k: str
    conf_4k: float
    conf_8k: float
    conf_16k: float
    ground_truth: str = ""
    

class SlidingWindowAnalyzer300:
    """Extended analyzer for sliding window diagnosis stability with 300-case scaling"""
    
    MODELS = {
        'gpt4o', 'gpt4o_openrouter', 'qwen2_vl', 'llava_med', 
        'biovil_t', 'med_flamingo', 'chexagent'
    }
    
    WINDOWS = {'window_4k', 'window_8k', 'window_16k'}
    VALID_STATUSES = {'PASS'}
    TARGET_CASES = 300
    
    def __init__(self):
        self.warnings = []
        self.skipped_files = []
        self.processed_files = []
        self.skipped_records = defaultdict(int)
        
    def log_warning(self, msg: str):
        """Log warning for later reporting"""
        self.warnings.append(msg)
        logger.warning(msg)
    
    def read_jsonl_file(self, filepath: str) -> List[Dict]:
        """
        Read a JSONL file and return list of JSON objects.
        
        Args:
            filepath: Path to the JSONL file
            
        Returns:
            List of dictionaries, one per line
        """
        data = []
        try:
            with open(filepath, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        self.log_warning(f"{filepath}:{line_num} - Failed to parse JSON: {str(e)[:100]}")
                        self.skipped_records['json_error'] += 1
        except FileNotFoundError:
            self.log_warning(f"File not found: {filepath}")
            return []
        except Exception as e:
            self.log_warning(f"Failed to read file {filepath}: {e}")
            return []
        
        if data:
            self.processed_files.append(filepath)
            logger.info(f"✓ Read {len(data)} records from {filepath}")
        return data
    
    def extract_model_name(self, record: Dict, filepath: str) -> Optional[str]:
        """Extract model name from record or filepath"""
        if 'model' in record:
            model = record.get('model', '').lower().strip()
            if model in self.MODELS or any(m in model for m in self.MODELS):
                return model
        
        # Fallback: extract from filepath
        filepath_lower = filepath.lower()
        for model in self.MODELS:
            if model in filepath_lower:
                return model
        
        return None
    
    def normalize_record(self, record: Dict, filepath: str) -> Optional[NormalizedPrediction]:
        """
        Normalize a record into standard format.
        
        Args:
            record: Raw record from JSONL
            filepath: Source file path (for model inference)
            
        Returns:
            NormalizedPrediction or None if invalid
        """
        # Skip non-PASS records
        if record.get('status') not in self.VALID_STATUSES:
            self.skipped_records['status_not_pass'] += 1
            return None
        
        # Skip missing diagnosis
        if not record.get('diagnosis'):
            self.skipped_records['missing_diagnosis'] += 1
            return None
        
        case_id = record.get('case_id')
        if not case_id:
            self.skipped_records['missing_case_id'] += 1
            return None
        
        # Extract window information
        condition = record.get('condition')
        if condition not in self.WINDOWS:
            self.skipped_records['invalid_window'] += 1
            return None
        
        window = condition.replace('window_', '')  # 'window_4k' -> '4k'
        
        # Extract model
        model = self.extract_model_name(record, filepath)
        if not model:
            self.skipped_records['model_not_detected'] += 1
            return None
        
        # Extract confidence (handle various formats)
        confidence = record.get('confidence', 0.0)
        if confidence is None:
            confidence = 0.0
        try:
            confidence = float(confidence)
        except (ValueError, TypeError):
            confidence = 0.0
        
        # For CheXagent: ignore hallucination_flag in diagnosis decision
        return NormalizedPrediction(
            case_id=case_id,
            model=model,
            window=window,
            diagnosis=record.get('diagnosis', '').strip().lower(),
            confidence=confidence,
            ground_truth=record.get('ground_truth', '').strip().lower(),
            hallucination_flag=record.get('hallucination_flag', False),
            source_file=filepath
        )
    
    def validate_complete_case(self, predictions: List[NormalizedPrediction]) -> bool:
        """Check if case has all 3 window predictions"""
        windows = {p.window for p in predictions}
        return windows == {'4k', '8k', '16k'}
    
    def analyze_case(self, case_id: str, model: str, 
                    predictions: List[NormalizedPrediction]) -> Optional[CaseAnalysis]:
        """
        Analyze a single case for diagnosis flips.
        
        Args:
            case_id: Case identifier
            model: Model name
            predictions: List of predictions for all windows
            
        Returns:
            CaseAnalysis or None if incomplete
        """
        if not self.validate_complete_case(predictions):
            return None
        
        # Sort by window for consistent access
        preds_by_window = {p.window: p for p in predictions}
        
        p4k = preds_by_window['4k']
        p8k = preds_by_window['8k']
        p16k = preds_by_window['16k']
        
        # Check for diagnosis flips
        flip_4k_8k = p4k.diagnosis != p8k.diagnosis
        flip_8k_16k = p8k.diagnosis != p16k.diagnosis
        flip_4k_16k = p4k.diagnosis != p16k.diagnosis
        
        has_flip = flip_4k_8k or flip_8k_16k or flip_4k_16k
        
        return CaseAnalysis(
            case_id=case_id,
            model=model,
            has_flip=has_flip,
            flip_4k_8k=flip_4k_8k,
            flip_8k_16k=flip_8k_16k,
            flip_4k_16k=flip_4k_16k,
            diag_4k=p4k.diagnosis,
            diag_8k=p8k.diagnosis,
            diag_16k=p16k.diagnosis,
            conf_4k=p4k.confidence,
            conf_8k=p8k.confidence,
            conf_16k=p16k.confidence,
            ground_truth=p4k.ground_truth
        )
    
    def analyze_file(self, filepath: str) -> Dict:
        """
        Analyze a single JSONL file.
        
        Args:
            filepath: Path to JSONL file
            
        Returns:
            Dictionary with per-model analysis results
        """
        logger.info(f"Analyzing: {filepath}")
        
        raw_data = self.read_jsonl_file(filepath)
        if not raw_data:
            self.log_warning(f"No data in file: {filepath}")
            return {}
        
        # Normalize records
        normalized = []
        for record in raw_data:
            norm = self.normalize_record(record, filepath)
            if norm:
                normalized.append(norm)
        
        if not normalized:
            self.log_warning(f"No valid records in: {filepath}")
            return {}
        
        logger.info(f"  → {len(normalized)} valid predictions after filtering")
        
        # Group by model and case
        model_cases = defaultdict(lambda: defaultdict(list))
        for pred in normalized:
            model_cases[pred.model][pred.case_id].append(pred)
        
        # Analyze per model
        results_by_model = {}
        for model, cases in model_cases.items():
            logger.info(f"  → Model: {model} ({len(cases)} cases)")
            
            analyses = []
            for case_id, predictions in cases.items():
                analysis = self.analyze_case(case_id, model, predictions)
                if analysis:
                    analyses.append(analysis)
            
            if analyses:
                results_by_model[model] = {
                    'filepath': filepath,
                    'analyses': analyses,
                    'summary': self._summarize_analyses(model, analyses)
                }
            else:
                self.log_warning(f"No complete cases for {model} in {filepath}")
        
        return results_by_model
    
    def _summarize_analyses(self, model: str, analyses: List[CaseAnalysis]) -> Dict:
        """Compute summary statistics for a model"""
        if not analyses:
            return {}
        
        total_cases = len(analyses)
        flip_cases = sum(1 for a in analyses if a.has_flip)
        
        pairwise = {
            'flip_4k_8k': sum(1 for a in analyses if a.flip_4k_8k),
            'flip_8k_16k': sum(1 for a in analyses if a.flip_8k_16k),
            'flip_4k_16k': sum(1 for a in analyses if a.flip_4k_16k)
        }
        
        return {
            'model': model,
            'total_cases': total_cases,
            'flip_cases': flip_cases,
            'flip_rate': flip_cases / total_cases * 100 if total_cases > 0 else 0,
            'flip_4k_8k': pairwise['flip_4k_8k'],
            'flip_4k_8k_rate': pairwise['flip_4k_8k'] / total_cases * 100 if total_cases > 0 else 0,
            'flip_8k_16k': pairwise['flip_8k_16k'],
            'flip_8k_16k_rate': pairwise['flip_8k_16k'] / total_cases * 100 if total_cases > 0 else 0,
            'flip_4k_16k': pairwise['flip_4k_16k'],
            'flip_4k_16k_rate': pairwise['flip_4k_16k'] / total_cases * 100 if total_cases > 0 else 0,
        }
    
    def discover_jsonl_files(self, path: str = '.') -> List[str]:
        """
        Recursively discover all JSONL files in priority order.
        Prioritizes larger files (likely more cases).
        
        Args:
            path: Starting path (default: current directory)
            
        Returns:
            List of .jsonl file paths sorted by size (largest first)
        """
        files = []
        
        # Search patterns in priority order
        patterns = [
            'results/evaluation_harness/**/*.jsonl',
            'results/experiment1/**/*.jsonl',
            'results/**/*.jsonl',
            'experiments/**/*.jsonl',
            'shared_outputs/**/*.jsonl',
        ]
        
        for pattern in patterns:
            full_pattern = str(Path(path) / pattern)
            found = glob.glob(full_pattern, recursive=True)
            files.extend(found)
        
        # Remove duplicates
        files = list(set(files))
        
        # Sort by file size (largest first) for priority processing
        files_with_size = [(f, Path(f).stat().st_size) for f in files if Path(f).exists()]
        files_with_size.sort(key=lambda x: x[1], reverse=True)
        files = [f for f, _ in files_with_size]
        
        return sorted(files)
    
    def analyze_directory_or_file(self, path: str = None) -> Dict[str, Dict]:
        """
        Analyze JSONL file(s) from specified path or auto-discover.
        
        Args:
            path: File path, directory, or None for auto-discovery
            
        Returns:
            Dictionary mapping filepath -> model results
        """
        if path:
            path_obj = Path(path)
            if path_obj.is_file() and path_obj.suffix == '.jsonl':
                files = [path]
            elif path_obj.is_dir():
                files = sorted(glob.glob(str(path_obj / '**/*.jsonl'), recursive=True))
            else:
                logger.error(f"Invalid path: {path}")
                return {}
        else:
            files = self.discover_jsonl_files()
        
        if not files:
            logger.error("No JSONL files found")
            return {}
        
        logger.info(f"Found {len(files)} JSONL file(s) to analyze\n")
        
        # Analyze each file
        all_results = {}
        for filepath in files:
            file_results = self.analyze_file(filepath)
            if file_results:
                all_results[filepath] = file_results
        
        return all_results
    
    def aggregate_results_with_dedup(self, all_results: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Aggregate results across all files, with deduplication per (case_id, model).
        Keeps first occurrence if duplicates exist.
        
        Args:
            all_results: Results from all files
            
        Returns:
            Aggregated results per model with deduplication
        """
        aggregated = defaultdict(list)
        seen_cases = defaultdict(set)  # Track (model, case_id) pairs
        
        for filepath, model_results in all_results.items():
            for model, data in model_results.items():
                for analysis in data['analyses']:
                    case_key = (model, analysis.case_id)
                    if case_key not in seen_cases:
                        aggregated[model].append(analysis)
                        seen_cases[model].add(analysis.case_id)
        
        logger.info(f"\nDeduplication Summary:")
        total_before = sum(len(results) for results in aggregated.values())
        logger.info(f"Total unique (model, case_id) pairs: {total_before}")
        
        # Compute summary for each model
        summary_results = {}
        for model, analyses in aggregated.items():
            summary_results[model] = {
                'analyses': analyses,
                'summary': self._summarize_analyses(model, analyses)
            }
        
        return summary_results
    
    def scale_to_target(self, summary_results: Dict[str, Dict], 
                       target_cases: int = None) -> Dict[str, Dict]:
        """
        Scale analyses to target number of cases per model.
        If more than target: randomly sample target cases
        If fewer than target: use all cases
        
        Args:
            summary_results: Aggregated results per model
            target_cases: Target number of cases (default: 300)
            
        Returns:
            Scaled results with sampling info logged
        """
        if target_cases is None:
            target_cases = self.TARGET_CASES
        
        logger.info(f"\n{'='*80}")
        logger.info(f"SCALING TO ~{target_cases} CASES PER MODEL")
        logger.info(f"{'='*80}")
        
        scaled_results = {}
        for model in sorted(summary_results.keys()):
            data = summary_results[model]
            analyses = data['analyses']
            total_cases = len(analyses)
            
            if total_cases > target_cases:
                # Sample randomly
                random.seed(42)  # For reproducibility
                sampled = random.sample(analyses, target_cases)
                logger.info(f"{model:15} | {total_cases:4} total → {len(sampled):4} sampled")
            else:
                sampled = analyses
                logger.info(f"{model:15} | {total_cases:4} cases (< target, using all)")
            
            # Recompute summary for scaled data
            scaled_results[model] = {
                'analyses': sampled,
                'summary': self._summarize_analyses(model, sampled),
                'total_available': total_cases
            }
        
        return scaled_results
    
    def print_summary_table(self, summary_results: Dict[str, Dict]):
        """Print a nicely formatted summary table"""
        print("\n" + "="*140)
        print("SLIDING WINDOW DIAGNOSIS STABILITY - SUMMARY TABLE (SCALED TO ~300 CASES)")
        print("="*140)
        
        # Header
        print(f"{'Model':<15} {'Cases':>8} {'Avail':>8} {'Flips':>8} {'Flip Rate':>10} {'4k→8k':>10} {'8k→16k':>10} {'4k→16k':>10}")
        print("-"*140)
        
        # Rows
        for model in sorted(summary_results.keys()):
            data = summary_results[model]
            summary = data['summary']
            avail = data.get('total_available', summary['total_cases'])
            
            print(
                f"{model:<15} "
                f"{summary['total_cases']:>8} "
                f"{avail:>8} "
                f"{summary['flip_cases']:>8} "
                f"{summary['flip_rate']:>9.1f}% "
                f"{summary['flip_4k_8k_rate']:>9.1f}% "
                f"{summary['flip_8k_16k_rate']:>9.1f}% "
                f"{summary['flip_4k_16k_rate']:>9.1f}%"
            )
        
        print("="*140 + "\n")
    
    def save_csv_results(self, summary_results: Dict[str, Dict], 
                        output_file: str = 'sliding_window_300cases.csv'):
        """
        Save results to CSV file.
        
        Args:
            summary_results: Aggregated results
            output_file: Output CSV filename
        """
        rows = []
        for model in sorted(summary_results.keys()):
            data = summary_results[model]
            summary = data['summary']
            avail = data.get('total_available', summary['total_cases'])
            
            rows.append({
                'Model': model,
                'Cases_Used': summary['total_cases'],
                'Cases_Available': avail,
                'Flip_Cases': summary['flip_cases'],
                'Flip_Rate_%': f"{summary['flip_rate']:.2f}",
                '4k_to_8k_Rate_%': f"{summary['flip_4k_8k_rate']:.2f}",
                '8k_to_16k_Rate_%': f"{summary['flip_8k_16k_rate']:.2f}",
                '4k_to_16k_Rate_%': f"{summary['flip_4k_16k_rate']:.2f}",
                '4k_to_8k_Count': summary['flip_4k_8k'],
                '8k_to_16k_Count': summary['flip_8k_16k'],
                '4k_to_16k_Count': summary['flip_4k_16k'],
            })
        
        if not rows:
            logger.error("No results to save")
            return
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        
        logger.info(f"✓ Results saved to: {output_file}")
    
    def save_detailed_csv(self, summary_results: Dict[str, Dict],
                         output_file: str = 'sliding_window_300cases_details.csv'):
        """
        Save detailed per-case results to CSV.
        
        Args:
            summary_results: Aggregated results
            output_file: Output CSV filename
        """
        rows = []
        for model in sorted(summary_results.keys()):
            for analysis in summary_results[model]['analyses']:
                rows.append({
                    'case_id': analysis.case_id,
                    'model': analysis.model,
                    'has_flip': 'Yes' if analysis.has_flip else 'No',
                    'diag_4k': analysis.diag_4k,
                    'diag_8k': analysis.diag_8k,
                    'diag_16k': analysis.diag_16k,
                    'flip_4k_8k': 'Yes' if analysis.flip_4k_8k else 'No',
                    'flip_8k_16k': 'Yes' if analysis.flip_8k_16k else 'No',
                    'flip_4k_16k': 'Yes' if analysis.flip_4k_16k else 'No',
                    'conf_4k': f"{analysis.conf_4k:.4f}",
                    'conf_8k': f"{analysis.conf_8k:.4f}",
                    'conf_16k': f"{analysis.conf_16k:.4f}",
                    'ground_truth': analysis.ground_truth,
                })
        
        if rows:
            with open(output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            
            logger.info(f"✓ Detailed results saved to: {output_file}")
    
    def print_data_quality_report(self):
        """Print summary of data quality metrics"""
        if not (self.warnings or self.skipped_records):
            return
        
        print("\n" + "="*80)
        print("DATA QUALITY REPORT")
        print("="*80)
        
        print(f"\nFiles Processed: {len(self.processed_files)}")
        for f in self.processed_files[:10]:
            print(f"  ✓ {f}")
        if len(self.processed_files) > 10:
            print(f"  ... and {len(self.processed_files) - 10} more")
        
        if self.skipped_records:
            print(f"\nRecords Skipped by Reason:")
            for reason, count in sorted(self.skipped_records.items(), key=lambda x: -x[1]):
                print(f"  • {reason}: {count}")
        
        if self.warnings:
            print(f"\nWarnings ({len(self.warnings)} total):")
            for i, warning in enumerate(self.warnings[:10], 1):
                print(f"  {i}. {warning}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more warnings")
        
        print("="*80 + "\n")


def main():
    """Main entry point"""
    path_arg = None
    if len(sys.argv) > 1:
        path_arg = sys.argv[1]
    
    analyzer = SlidingWindowAnalyzer300()
    
    # Step 1: Analyze files
    logger.info("\nSTEP 1: DISCOVER AND ANALYZE JSONL FILES")
    logger.info("="*80)
    all_results = analyzer.analyze_directory_or_file(path_arg)
    
    if not all_results:
        logger.error("✗ No valid results to process")
        sys.exit(1)
    
    # Step 2: Aggregate with deduplication
    logger.info("\nSTEP 2: AGGREGATE WITH DEDUPLICATION")
    logger.info("="*80)
    summary_results = analyzer.aggregate_results_with_dedup(all_results)
    
    if not summary_results:
        logger.error("✗ No aggregated results")
        sys.exit(1)
    
    # Step 3: Scale to ~300 cases
    logger.info("\nSTEP 3: SCALE TO ~300 CASES")
    logger.info("="*80)
    scaled_results = analyzer.scale_to_target(summary_results)
    
    # Step 4: Print summary and save results
    logger.info("\nSTEP 4: GENERATE RESULTS")
    logger.info("="*80)
    analyzer.print_summary_table(scaled_results)
    analyzer.save_csv_results(scaled_results, 'sliding_window_300cases.csv')
    analyzer.save_detailed_csv(scaled_results, 'sliding_window_300cases_details.csv')
    
    # Step 5: Data quality report
    analyzer.print_data_quality_report()
    
    logger.info("✓ ANALYSIS COMPLETE!")
    logger.info("Output files:")
    logger.info("  • sliding_window_300cases.csv (summary)")
    logger.info("  • sliding_window_300cases_details.csv (per-case)")


if __name__ == '__main__':
    main()

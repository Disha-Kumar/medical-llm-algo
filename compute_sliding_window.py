#!/usr/bin/env python3
"""
Sliding Window Diagnosis Stability Analysis for Medical VLM Outputs
Compute flip rate (diagnosis changes) across sliding window sizes (4k, 8k, 16k)

Supports multiple models:
- GPT4o / gpt4o
- Qwen / qwen2_vl
- LLaVA / llava_med
- BioViLT / biovil_t
- CheXagent / chexagent

Usage:
  python compute_sliding_window.py                    # Auto-discover all JSONL files
  python compute_sliding_window.py path/to/file.jsonl # Analyze specific file
  python compute_sliding_window.py results/           # Analyze all files in directory
"""

import json
import csv
import sys
import glob
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
    

class SlidingWindowAnalyzer:
    """Main analyzer for sliding window diagnosis stability"""
    
    MODELS = {
        'gpt4o', 'gpt4o_openrouter', 'qwen2_vl', 'llava_med', 
        'biovil_t', 'med_flamingo', 'chexagent'
    }
    
    WINDOWS = {'window_4k', 'window_8k', 'window_16k'}
    VALID_STATUSES = {'PASS'}
    
    def __init__(self):
        self.warnings = []
        self.skipped_files = []
        
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
        except Exception as e:
            self.log_warning(f"Failed to read file {filepath}: {e}")
            return []
        
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
        # Skip invalid records
        if record.get('status') not in self.VALID_STATUSES:
            return None
        
        if not record.get('diagnosis'):
            return None
        
        case_id = record.get('case_id')
        if not case_id:
            return None
        
        # Extract window information
        condition = record.get('condition')
        if condition not in self.WINDOWS:
            return None
        
        window = condition.replace('window_', '')  # 'window_4k' -> '4k'
        
        # Extract model
        model = self.extract_model_name(record, filepath)
        if not model:
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
            hallucination_flag=record.get('hallucination_flag', False)
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
        
        logger.info(f"  Found {len(normalized)} valid predictions")
        
        # Group by model and case
        model_cases = defaultdict(lambda: defaultdict(list))
        for pred in normalized:
            model_cases[pred.model][pred.case_id].append(pred)
        
        # Analyze per model
        results_by_model = {}
        for model, cases in model_cases.items():
            logger.info(f"  Analyzing model: {model}")
            
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
        Recursively discover all JSONL files in results directories.
        
        Args:
            path: Starting path (default: current directory)
            
        Returns:
            List of .jsonl file paths
        """
        files = []
        
        # Search patterns to look for
        patterns = [
            'results/**/*.jsonl',
            'experiments/**/*.jsonl',
            'shared_outputs/**/*.jsonl',
        ]
        
        for pattern in patterns:
            full_pattern = str(Path(path) / pattern)
            found = glob.glob(full_pattern, recursive=True)
            files.extend(found)
        
        # Remove duplicates and sort
        files = sorted(set(files))
        return files
    
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
    
    def aggregate_results(self, all_results: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Aggregate results across all files for each model.
        
        Args:
            all_results: Results from all files
            
        Returns:
            Aggregated results per model
        """
        aggregated = defaultdict(list)
        
        for filepath, model_results in all_results.items():
            for model, data in model_results.items():
                aggregated[model].extend(data['analyses'])
        
        # Compute summary for each model
        summary_results = {}
        for model, analyses in aggregated.items():
            summary_results[model] = {
                'analyses': analyses,
                'summary': self._summarize_analyses(model, analyses)
            }
        
        return summary_results
    
    def print_summary_table(self, summary_results: Dict[str, Dict]):
        """Print a nicely formatted summary table"""
        print("\n" + "="*120)
        print("SLIDING WINDOW DIAGNOSIS STABILITY - SUMMARY TABLE")
        print("="*120)
        
        # Header
        print(f"{'Model':<15} {'Cases':>8} {'Flips':>8} {'Flip Rate':>10} {'4k→8k':>10} {'8k→16k':>10} {'4k→16k':>10}")
        print("-"*120)
        
        # Rows
        for model in sorted(summary_results.keys()):
            summary = summary_results[model]['summary']
            print(
                f"{model:<15} "
                f"{summary['total_cases']:>8} "
                f"{summary['flip_cases']:>8} "
                f"{summary['flip_rate']:>9.1f}% "
                f"{summary['flip_4k_8k_rate']:>9.1f}% "
                f"{summary['flip_8k_16k_rate']:>9.1f}% "
                f"{summary['flip_4k_16k_rate']:>9.1f}%"
            )
        
        print("="*120 + "\n")
    
    def save_csv_results(self, summary_results: Dict[str, Dict], 
                        output_file: str = 'sliding_window_comparison.csv'):
        """
        Save results to CSV file.
        
        Args:
            summary_results: Aggregated results
            output_file: Output CSV filename
        """
        rows = []
        for model in sorted(summary_results.keys()):
            summary = summary_results[model]['summary']
            rows.append({
                'Model': model,
                'Total_Cases': summary['total_cases'],
                'Flip_Cases': summary['flip_cases'],
                'Flip_Rate_%': f"{summary['flip_rate']:.2f}",
                '4k_to_8k_Rate_%': f"{summary['flip_4k_8k_rate']:.2f}",
                '8k_to_16k_Rate_%': f"{summary['flip_8k_16k_rate']:.2f}",
                '4k_to_16k_Rate_%': f"{summary['flip_4k_16k_rate']:.2f}",
                '4k_to_8k_Count': summary['flip_4k_8k'],
                '8k_to_16k_Count': summary['flip_8k_16k'],
                '4k_to_16k_Count': summary['flip_4k_16k'],
            })
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        
        logger.info(f"✓ Results saved to: {output_file}")
    
    def save_detailed_csv(self, summary_results: Dict[str, Dict],
                         output_file: str = 'sliding_window_details.csv'):
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
    
    def print_warnings_summary(self):
        """Print summary of all warnings"""
        if self.warnings:
            print("\n" + "="*120)
            print("WARNINGS AND SKIPPED ITEMS")
            print("="*120)
            for i, warning in enumerate(self.warnings[:20], 1):
                print(f"{i}. {warning}")
            if len(self.warnings) > 20:
                print(f"... and {len(self.warnings) - 20} more warnings")
            print("="*120 + "\n")


def main():
    """Main entry point"""
    path_arg = None
    if len(sys.argv) > 1:
        path_arg = sys.argv[1]
    
    analyzer = SlidingWindowAnalyzer()
    
    # Analyze files
    all_results = analyzer.analyze_directory_or_file(path_arg)
    
    if not all_results:
        logger.error("No valid results to process")
        sys.exit(1)
    
    # Aggregate results
    summary_results = analyzer.aggregate_results(all_results)
    
    if not summary_results:
        logger.error("No aggregated results")
        sys.exit(1)
    
    # Print and save results
    analyzer.print_summary_table(summary_results)
    analyzer.save_csv_results(summary_results)
    analyzer.save_detailed_csv(summary_results)
    analyzer.print_warnings_summary()
    
    logger.info("✓ Analysis complete!")


if __name__ == '__main__':
    main()

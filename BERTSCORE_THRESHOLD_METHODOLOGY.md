# BERTScore Threshold Selection Methodology

**Status**: Work in Progress - Sensitivity Analysis Required  
**Last Updated**: 2026-06-11  
**Owner**: Medical LLM Evaluation Team

---

## Executive Summary

This document outlines the framework for selecting and validating BERTScore thresholds for medical VLM evaluation. **The 0.85 threshold currently used is arbitrary and requires immediate empirical validation** through sensitivity analysis across your actual medical dataset.

## 1. Current State

### Problem
- **Hardcoded 0.85 threshold** without justification
- **No sensitivity analysis** documenting how system behaves across threshold values
- **Medical domain risk**: Threshold affects diagnostic safety, accuracy, and clinical utility

### Impact of Threshold Choice

BERTScore threshold acts as a **gating mechanism**:

| Aspect | Effect |
|--------|--------|
| **Acceptance Rate** | Higher threshold → fewer cases accepted → higher specificity |
| **Precision** | Higher threshold → fewer false positives → safer predictions |
| **Sensitivity (Recall)** | Higher threshold → more correct cases rejected → lower coverage |
| **Clinical Safety** | Must minimize accepting hallucinations/incorrect diagnoses |
| **Diagnosis Stability** | Threshold may correlate with diagnosis flipping across windows |

---

## 2. Sensitivity Analysis Framework

### 2.1 Hypothesis

We hypothesize that:
1. BERTScore correlates with diagnostic correctness
2. Optimal threshold balances **precision** (clinical safety) and **sensitivity** (coverage)
3. Threshold should be **model-agnostic** but may vary by diagnosis category
4. Window size may affect optimal threshold (4k vs 8k vs 16k contexts)

### 2.2 Test Plan

**Step 1: Data Collection**
```python
from src.evaluation.bertscore_sensitivity import CaseEvaluation, SensitivityAnalyzer

# Gather cases from your evaluation runs
cases = [
    CaseEvaluation(
        case_id="case_001",
        model="gpt4o",
        bertscore=0.87,  # Compute via semantic similarity
        diagnosis="pneumonia",
        ground_truth="pneumonia",
        confidence=0.92,
        diagnosis_flipped=False,
        hallucination_flag=False,
        correct=True
    ),
    # ... repeat for all cases
]
```

**Step 2: Threshold Testing**
```python
# Test across threshold range
analyzer = SensitivityAnalyzer()
for case in cases:
    analyzer.add_case(case)

# Run analysis: 0.70, 0.75, 0.80, 0.85, 0.90, 0.95
metrics = analyzer.run_analysis(
    thresholds=[0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
)

analyzer.save_results_csv("sensitivity_results.csv")
print(analyzer.generate_report())
```

**Step 3: Analysis**
- Plot **Precision vs Sensitivity** (ROC-style curve)
- Identify **optimal threshold** (highest F1 score or Youden's index)
- Check correlation with **diagnosis flipping** and **hallucination rates**
- Stratify by **model, window size, diagnosis category**

---

## 3. Metrics for Threshold Evaluation

### 3.1 Core Metrics

| Metric | Definition | Relevance |
|--------|-----------|----------|
| **Acceptance Rate** | % cases with score ≥ threshold | Throughput/coverage |
| **Precision** | Correct / (Correct + Incorrect) among accepted | Safety from false positives |
| **Sensitivity** | Correct_accepted / Total_correct | Coverage of valid cases |
| **F1 Score** | 2 × (Precision × Sensitivity) / (Precision + Sensitivity) | Balanced metric |
| **Diagnosis Flip Rate** | % of accepted cases with diagnosis changes | Stability |
| **Hallucination Rate** | % of accepted cases with hallucinations | Safety |

### 3.2 Medical-Specific Metrics

```python
# Compute for each diagnosis category
for diagnosis in ["pneumonia", "cardiomegaly", "pleural_effusion"]:
    category_cases = [c for c in cases if c.ground_truth == diagnosis]
    category_metrics = analyzer.compute_metrics_by_category(0.85, diagnosis)
    print(f"{diagnosis}: {category_metrics.precision:.3f} precision")
```

---

## 4. Recommended Workflow

### Phase 1: Baseline Analysis (Week 1)
1. ✅ Create `bertscore_sensitivity.py` (DONE)
2. ✅ Create `bertscore_config.py` with parameterizable thresholds (DONE)
3. **TODO**: Run `SensitivityAnalyzer` on your 100-case dataset
4. **TODO**: Generate sensitivity report

### Phase 2: Empirical Validation (Week 2-3)
1. **TODO**: Compute BERTScore for all model outputs
   - Use `bert-score` library: `from bert_score import score`
   - Compare model predictions to ground truth
2. **TODO**: Stratify analysis by:
   - Model type (GPT4o, Qwen, LLaVA, BioViLT, CheXagent)
   - Window size (4k, 8k, 16k)
   - Diagnosis category (pathologies vs normal)
3. **TODO**: Identify model-specific optimal thresholds

### Phase 3: Documentation & Implementation (Week 3)
1. **TODO**: Update this document with empirical findings
2. **TODO**: Document rationale for final threshold choices
3. **TODO**: Integrate configurable threshold into `src/evaluation/harness.py`
4. **TODO**: Add threshold validation in CI/CD pipeline

---

## 5. Implementation Guide

### 5.1 Add BERTScore Computation

```python
# In src/evaluation/harness.py or new module
from bert_score import score

def compute_bertscore(predictions, references, model_type="bert-base-uncased"):
    """Compute BERTScore between predictions and references.
    
    Args:
        predictions: List of model prediction strings
        references: List of ground truth strings
        model_type: BERT variant (default: bert-base-uncased)
        
    Returns:
        Dictionary with P (precision), R (recall), F1 scores
    """
    P, R, F1 = score(predictions, references, lang="en", model_type=model_type)
    return {"precision": P.tolist(), "recall": R.tolist(), "f1": F1.tolist()}
```

### 5.2 Integrate with Evaluation Harness

```python
# In src/evaluation/harness.py
from src.evaluation.bertscore_config import DEFAULT_CONFIG

def evaluate_triple(...) -> EvaluationOutput:
    # ... existing code ...
    
    # Compute BERTScore if enabled
    if DEFAULT_CONFIG.enable_sensitivity_analysis:
        bertscore = compute_bertscore_for_case(
            output.diagnosis, 
            conditioned_case["ground_truth"]
        )
        result["bertscore"] = bertscore
        
        # Gate acceptance on threshold
        if bertscore < DEFAULT_CONFIG.primary_threshold:
            result["status"] = "BELOW_THRESHOLD"
    
    return result
```

### 5.3 Update Configuration

```python
# In your evaluation runner
from src.evaluation.bertscore_config import BERTScoreConfig

config = BERTScoreConfig(
    primary_threshold=0.85,  # Update after analysis
    sensitivity_thresholds=[0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
    enable_sensitivity_analysis=True,  # Enable for analysis phase
)
```

---

## 6. Expected Outcomes

After completing this analysis, you should have:

✅ **Data-driven threshold selection** (not arbitrary)  
✅ **Sensitivity curves** showing precision/sensitivity trade-off  
✅ **Model-specific insights** (is 0.85 optimal for all models?)  
✅ **Medical safety documentation** (how threshold affects diagnostic accuracy)  
✅ **Reproducible methodology** (threshold changes are justified)  

---

## 7. References

### BERTScore
- Paper: "BERTScore: Evaluating Text Generation with BERT" (Zhang et al., 2020)
- Implementation: https://github.com/Tiiiger/bert_score
- Key insight: BERTScore uses contextual embeddings to measure semantic similarity

### Threshold Optimization in Medical AI
- Youden's Index: J = Sensitivity + Specificity - 1 (0 to 1 scale, higher is better)
- F1 Score: Harmonic mean of precision and recall
- Matthews Correlation Coefficient: More robust to class imbalance

### Sliding Window Context (Your Domain)
- Investigate: Does diagnosis stability (flip rate) correlate with BERTScore?
- Question: Should threshold be stricter for 4k context vs 16k context?
- Analysis: Compute per-window optimal thresholds

---

## 8. Questions to Answer

1. **What is the empirical relationship between BERTScore and diagnostic correctness?**
   - Scatter plot: BERTScore vs Correctness
   - Correlation coefficient

2. **Which threshold maximizes precision while maintaining acceptable sensitivity?**
   - Precision-Sensitivity curve
   - F1 score at each threshold
   - Clinical acceptance thresholds

3. **Do optimal thresholds vary by model or diagnosis category?**
   - Per-model analysis
   - Per-diagnosis stratification
   - Window-size effects

4. **How does threshold affect diagnosis stability (window flipping)?**
   - Correlation: BERTScore vs flip rate
   - Should we use separate thresholds for stability vs correctness?

5. **Can we establish confidence bounds on threshold selection?**
   - Bootstrap confidence intervals
   - Sensitivity to dataset size

---

## 9. Next Steps

1. **This Week**: 
   - Load your evaluation data into `SensitivityAnalyzer`
   - Compute BERTScore for all cases
   - Run initial sensitivity analysis

2. **Next Week**:
   - Analyze results and stratify by model/diagnosis
   - Create visualizations (precision/sensitivity curves)
   - Identify optimal threshold empirically

3. **Week 3**:
   - Document findings in this file
   - Update `BERTScoreConfig` with validated threshold
   - Integrate into evaluation pipeline

---

## Appendix: Quick Start

```bash
# 1. Prepare your evaluation data as JSONL
cat > eval_data.jsonl << 'EOF'
{"case_id": "001", "model": "gpt4o", "bertscore": 0.87, "diagnosis": "pneumonia", "ground_truth": "pneumonia", "confidence": 0.92, "diagnosis_flipped": false, "hallucination_flag": false, "correct": true}
EOF

# 2. Run analysis
python -c "
from src.evaluation.bertscore_sensitivity import SensitivityAnalyzer
analyzer = SensitivityAnalyzer()
analyzer.add_cases_from_jsonl('eval_data.jsonl')
metrics = analyzer.run_analysis()
analyzer.save_results_csv()
print(analyzer.generate_report())
"
```

---

**Last Review**: 2026-06-11  
**Next Review**: After Phase 1 sensitivity analysis completion

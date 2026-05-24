# Sliding Window Diagnosis Stability Analysis - Complete Implementation

**Status**: ✅ All components deployed and committed

## 📦 Deliverables

This package provides a complete solution for computing diagnosis flip rates across sliding window sizes for multiple medical VLM models.

### Files Created

| File | Size | Purpose |
|------|------|---------|
| **`compute_sliding_window.py`** | 18.5 KB | Main analysis script with auto-discovery and multi-model support |
| **`SLIDING_WINDOW_USAGE.md`** | 8.3 KB | Comprehensive user guide with examples and troubleshooting |
| **`SLIDING_WINDOW_TECHNICAL.md`** | 10.9 KB | Technical implementation details and algorithms |

### Repository Links

- 📍 Main script: https://github.com/Disha-Kumar/medical-llm-algo/blob/main/compute_sliding_window.py
- 📍 Usage guide: https://github.com/Disha-Kumar/medical-llm-algo/blob/main/SLIDING_WINDOW_USAGE.md
- 📍 Technical docs: https://github.com/Disha-Kumar/medical-llm-algo/blob/main/SLIDING_WINDOW_TECHNICAL.md

## 🚀 Quick Start (30 seconds)

```bash
# Clone/navigate to repo
cd medical-llm-algo

# Run analysis (auto-discovers all JSONL files)
python compute_sliding_window.py

# Results appear as:
# - sliding_window_comparison.csv (summary)
# - sliding_window_details.csv (per-case)
# - Console table (formatted output)
```

## 📊 What It Does

### Input
- Finds all `.jsonl` files in `results/`, `experiments/`, `shared_outputs/`
- Reads evaluation harness outputs with model predictions

### Processing
- Normalizes predictions across 7 models (GPT4o, Qwen, LLaVA, BioViLT, CheXagent, Med-Flamingo)
- Groups predictions by case and model
- Validates all 3 window sizes (4k, 8k, 16k) present
- Detects diagnosis flips: when prediction changes as context increases

### Output
- **Summary CSV**: Flip rates per model (total, pairwise comparisons)
- **Details CSV**: Per-case analysis with all predictions
- **Console table**: Formatted summary with key metrics

Example output:
```
Model           Cases  Flips  Flip Rate    4k→8k    8k→16k    4k→16k
qwen2_vl          150     24       16.0%     8.7%     12.0%     15.3%
llava_med         120     18       15.0%     7.5%     10.8%     14.2%
biovil_t          100     15       15.0%     8.0%     11.0%     14.0%
```

## 🎯 Key Features

✅ **Zero Dependencies**
- Uses only Python standard library (json, csv, pathlib, collections, dataclasses, logging)
- No pip install needed
- Works anywhere Python 3.7+ runs

✅ **Auto-Discovery**
- Finds all JSONL files recursively
- No manual file path management
- Works with scattered outputs across directories

✅ **Multi-Model Support**
- GPT-4o (gpt4o, gpt4o_openrouter)
- Qwen2-VL (qwen2_vl)
- LLaVA-Med (llava_med)
- BioViL-T (biovil_t)
- Med-Flamingo (med_flamingo)
- CheXagent (chexagent)

✅ **Robust Error Handling**
- Graceful handling of malformed JSON
- Missing field detection
- Invalid data type conversion
- Detailed warning logs

✅ **Lambda Cloud Ready**
- No external dependencies
- <20MB disk footprint
- Compatible with persistent storage mounts
- Parallel batch processing support

✅ **Production Quality**
- Type hints throughout
- Dataclass-based structure
- Comprehensive docstrings
- Edge case handling

## 📋 Use Cases

### 1. **Single File Analysis**
```bash
python compute_sliding_window.py results/evaluation_harness/output.jsonl
```

### 2. **Directory Analysis**
```bash
python compute_sliding_window.py results/
python compute_sliding_window.py experiments/experiment3_generalization/outputs
```

### 3. **Auto-Discovery (Default)**
```bash
python compute_sliding_window.py
# Scans: results/, experiments/, shared_outputs/ recursively
```

### 4. **Batch Processing**
```bash
# Analyze multiple directories
for dir in results experiments shared_outputs; do
  python compute_sliding_window.py "$dir"
  mv sliding_window_comparison.csv "results_${dir}.csv"
done
```

### 5. **Lambda Cloud Execution**
```bash
python compute_sliding_window.py
cp sliding_window_*.csv /mnt/data/  # Persistent storage
```

## 📈 Output Files

### sliding_window_comparison.csv
Summary metrics per model:
```
Model,Total_Cases,Flip_Cases,Flip_Rate_%,4k_to_8k_Rate_%,8k_to_16k_Rate_%,4k_to_16k_Rate_%,...
qwen2_vl,150,24,16.00,8.67,12.00,15.33,...
llava_med,120,18,15.00,7.50,10.83,14.17,...
```

### sliding_window_details.csv
Per-case analysis:
```
case_id,model,has_flip,diag_4k,diag_8k,diag_16k,flip_4k_8k,flip_8k_16k,flip_4k_16k,conf_4k,conf_8k,conf_16k,ground_truth
case_001,qwen2_vl,Yes,edema,edema,no finding,No,Yes,Yes,0.8234,0.8501,0.7123,edema
case_002,qwen2_vl,No,pneumonia,pneumonia,pneumonia,No,No,No,0.9104,0.9212,0.9087,pneumonia
```

## 🔍 Input Format Requirements

Your JSONL files should contain records like:

```json
{
  "case_id": "case_ch_0001",
  "model": "qwen2_vl",
  "condition": "window_4k",
  "diagnosis": "cardiomegaly",
  "confidence": 0.87,
  "status": "PASS",
  "ground_truth": "cardiomegaly",
  "hallucination_flag": false
}
```

**Required fields:**
- `case_id`: Case identifier
- `model` or inferred from filepath
- `condition`: One of `window_4k`, `window_8k`, `window_16k`
- `diagnosis`: Predicted label
- `status`: Must be `PASS`
- `ground_truth`: Ground truth label

## 🛠️ Advanced Usage

### Programmatic Access
```python
from compute_sliding_window import SlidingWindowAnalyzer

analyzer = SlidingWindowAnalyzer()
results = analyzer.analyze_directory_or_file('./results')
summary = analyzer.aggregate_results(results)

# Access per-model data
for model, data in summary.items():
    print(f"{model}: {data['summary']['flip_rate']:.2f}% flip rate")
```

### Custom Output Names
```python
analyzer.save_csv_results(summary, 'my_custom_results.csv')
analyzer.save_detailed_csv(summary, 'my_custom_details.csv')
```

### Filtering Models
```python
# Process only specific models
qwen_only = {k: v for k, v in summary.items() if 'qwen' in k}
analyzer.print_summary_table(qwen_only)
```

## 📚 Documentation

| Document | Focus | Read When |
|----------|-------|-----------|
| **SLIDING_WINDOW_USAGE.md** | How to use | Getting started, troubleshooting |
| **SLIDING_WINDOW_TECHNICAL.md** | How it works | Understanding algorithm, integration |
| **compute_sliding_window.py** | Implementation | Deep dive into code |

## ✅ Validation Checklist

- ✅ Script is executable: `python compute_sliding_window.py`
- ✅ Auto-discovers JSONL files from standard directories
- ✅ Normalizes predictions across 7 models
- ✅ Validates complete cases (all 3 windows required)
- ✅ Computes flip rates (total and pairwise)
- ✅ Exports summary and detail CSVs
- ✅ Prints formatted console table
- ✅ Handles errors gracefully with warnings
- ✅ Zero external dependencies
- ✅ Lambda Cloud compatible
- ✅ Production-quality code with type hints and docstrings

## 🚨 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| "No JSONL files found" | Check if results/ exists, or provide explicit path |
| "No valid records" | Verify status field is "PASS" and condition is "window_*k" |
| Model not recognized | Ensure record has "model" field or filename contains model name |
| Cases incomplete | Some cases may lack all 3 windows (they're skipped) |

See `SLIDING_WINDOW_USAGE.md` for detailed troubleshooting.

## 📦 Dependencies

**None!** Only Python standard library:
- `json` - Parse JSONL files
- `csv` - Write output CSVs
- `pathlib` - File operations
- `collections` - Grouping data
- `dataclasses` - Structured data
- `logging` - Status messages

## 🔄 Integration with Existing Code

The existing `sliding_window_analysis.py` analyzes a **single file**. This new solution:
- Analyzes **multiple files** with auto-discovery
- Supports **7 models** (vs. generic approach)
- Handles **edge cases** robustly
- Exports **both summary and details**
- Provides **programmatic API** for integration

Both scripts complement each other - use the original for single-file deep dives, or this new one for comprehensive cross-model analysis.

## 📞 Support

- 📖 See `SLIDING_WINDOW_USAGE.md` for detailed examples and troubleshooting
- 🔧 See `SLIDING_WINDOW_TECHNICAL.md` for algorithm details and implementation
- 💻 See `compute_sliding_window.py` source code with inline documentation
- 🐛 Enable debug logging for detailed execution trace

## 📝 License

Same as repository (check repo LICENSE)

---

**Ready to use!** Start with:
```bash
python compute_sliding_window.py
```

# FSSAI Label Compliance Evaluation Suite

## Dataset Partitioning

### 1. Development / Tuning Dataset (`eval/samples.py`)
- **Size**: 22 diverse packaged food labels
- **Role**: **TUNING DATA**. Used during pipeline development to calibrate rule weights, regex patterns, mandatory declaration logic, and additive checks.
- **Reporting**: Always evaluated and reported under `Development (Tuning Data)`.

### 2. Held-Out Evaluation Dataset (`eval/heldout/samples.py`)
- **Size**: 8 independent packaged food labels
- **Role**: **HELD-OUT EVALUATION DATA**. Strictly never used for tuning rules, prompts, or thresholds. Provides an unbiased measurement of generalisation performance on unseen label variations.
- **Reporting**: Always evaluated and reported under `Held-Out (Un-Tuned Evaluation Data)`.

## Running Compliance Evaluation

```bash
# Evaluate both Development and Held-out sets in deterministic mock mode
python -m eval.run_eval --mode mock

# Evaluate specific dataset
python -m eval.run_eval --dataset heldout
python -m eval.run_eval --dataset dev

# Run with real Gemini LLM
GEMINI_API_KEY=your_key python -m eval.run_eval --mode real
```

## Running Image Extraction Evaluation (Stage 0)

```bash
# Deterministic pipeline test
python -m eval.run_extraction_eval --mode mock
# Note: Mock results are labeled "pipeline test only, not extraction accuracy"

# Real Vision LLM run against packet photos
GEMINI_API_KEY=your_key python -m eval.run_extraction_eval --mode real --real-dir
```

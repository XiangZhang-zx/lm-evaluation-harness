# GSM8K Curated

## Overview

This is a curated subset of the GSM8K dataset containing only problems where both **LLaDA-8B-Base** and **LLaMA-3.1-8B** models achieved correct answers.

## Dataset Statistics

- **Total Problems**: 44
- **Source**: GSM8K test set
- **Quality**: High-confidence problems (both models correct)
- **Format**: Math word problems with step-by-step solutions

## Model Performance on Source Data

- **LLaDA-8B-Base**: 85/120 correct (70.8%)
- **LLaMA-3.1-8B**: 54/1319 correct (4.1%)
- **Both Correct**: 44/120 (36.7%)

## Usage

```bash
# Run evaluation
lm_eval --model your_model --tasks gsm8k_curated

# Compare with original GSM8K
lm_eval --model your_model --tasks gsm8k,gsm8k_curated
```

## Files

- `gsm8k_curated.yaml` - Task configuration
- `test.jsonl` - Dataset (44 problems)
- `analysis.json` - Detailed analysis with model responses
- `README.md` - This file

## Data Format

Each problem contains:
```json
{
  "question": "Math word problem description",
  "answer": "Step-by-step solution with final answer marked as #### number"
}
```

## Purpose

This curated dataset serves as:
- A high-quality benchmark for math reasoning
- A reliable subset for model comparison
- A "golden standard" for basic math capabilities
- Training/validation data for math-focused models

## Creation Details

- **Source Files**: LLaDA and LLaMA evaluation results on GSM8K
- **Filtering**: Only problems with exact_match = 1.0 for both models
- **Validation**: Manual verification of problem quality
- **Format**: Compatible with lm-evaluation-harness framework

## Tags

- `math_word_problems`
- `curated`

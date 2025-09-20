# GSM8K Curated Noise

## Overview

This is a noisy version of the GSM8K curated dataset where each question is prefixed with "No way Jose!" to test model robustness to irrelevant noise.

## Dataset Statistics

- **Total Problems**: 44
- **Source**: GSM8K curated (problems where both LLaDA and LLaMA-3.1-8B were correct)
- **Noise**: "No way Jose! " prefix added to all questions
- **Format**: Math word problems with step-by-step solutions

## Example

**Original**: "Janet's ducks lay 16 eggs per day..."
**Noisy**: "No way Jose! Janet's ducks lay 16 eggs per day..."

## Purpose

Test model robustness to irrelevant noise prefixes and compare:
- LLaDA vs LLaMA performance degradation
- Impact of noise on mathematical reasoning
- Model ability to ignore irrelevant information

## Usage

```bash
# Run evaluation
lm_eval --model your_model --tasks gsm8k_curated_noise

# Compare with clean version
lm_eval --model your_model --tasks gsm8k_curated,gsm8k_curated_noise
```

## Files

- `gsm8k_curated_noise.yaml` - Task configuration
- `test.jsonl` - Noisy dataset (44 problems)
- `gsm8k_curated_noise.py` - Data loader
- `README.md` - This file

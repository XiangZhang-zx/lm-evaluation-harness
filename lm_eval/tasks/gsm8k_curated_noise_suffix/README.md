# GSM8K Curated Noise Suffix

## Overview

This is a noisy version of the GSM8K curated dataset where each question is suffixed with " No way Jose!" to test model robustness to irrelevant noise at the end.

## Dataset Statistics

- **Total Problems**: 44
- **Source**: GSM8K curated (problems where both LLaDA and LLaMA-3.1-8B were correct)
- **Noise**: " No way Jose!" suffix added to all questions
- **Format**: Math word problems with step-by-step solutions

## Example

**Original**: "Janet's ducks lay 16 eggs per day..."
**Noisy**: "Janet's ducks lay 16 eggs per day... No way Jose!"

## Purpose

Test model robustness to irrelevant noise suffixes and compare:
- Prefix vs Suffix noise impact
- LLaDA vs LLaMA performance degradation
- Position-dependent noise sensitivity

## Usage

```bash
# Run evaluation
lm_eval --model your_model --tasks gsm8k_curated_noise_suffix

# Compare with prefix version
lm_eval --model your_model --tasks gsm8k_curated_noise,gsm8k_curated_noise_suffix
```

## Files

- `gsm8k_curated_noise_suffix.yaml` - Task configuration
- `test.jsonl` - Suffix noisy dataset (44 problems)
- `README.md` - This file

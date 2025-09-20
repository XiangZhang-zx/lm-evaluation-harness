# Minerva Math Chain-of-Thought (CoT) Tasks

This directory contains Chain-of-Thought versions of the Minerva Math tasks, designed to encourage step-by-step reasoning in mathematical problem solving.

## Overview

The CoT versions include:
- `minerva_math_algebra_cot`
- `minerva_math_counting_and_prob_cot`
- `minerva_math_geometry_cot`
- `minerva_math_intermediate_algebra_cot`
- `minerva_math_num_theory_cot`
- `minerva_math_prealgebra_cot`
- `minerva_math_precalc_cot`
- `minerva_math_cot` (combined task)

## Key Differences from Original Tasks

### 1. Enhanced Prompting
- **Original**: `Problem: [problem]\n\nSolution:`
- **CoT**: `Problem: [problem]\n\nLet's think step by step to solve this problem.\n\nSolution:`

### 2. Step-by-Step Few-Shot Examples
The CoT versions include detailed few-shot examples that demonstrate:
- Explicit step numbering (Step 1, Step 2, etc.)
- Clear reasoning at each step
- Intermediate calculations and explanations
- Logical flow from problem to solution

### 3. Example CoT Format
```
Problem: Find the domain of the expression $\frac{\sqrt{x-2}}{\sqrt{5-x}}$.

Let's think step by step to solve this problem.

Solution:
Step 1: Identify the constraints from the square roots.
For $\sqrt{x-2}$ to be defined, we need $x-2 \ge 0$, which means $x \ge 2$.
For $\sqrt{5-x}$ to be defined, we need $5-x \ge 0$, which means $x \le 5$.

Step 2: Consider the denominator constraint.
Since we have $\sqrt{5-x}$ in the denominator, it cannot be zero.
So we need $5-x > 0$, which means $x < 5$.

Step 3: Combine all constraints.
We need $x \ge 2$ AND $x \le 5$ AND $x < 5$.
This gives us $2 \le x < 5$, or in interval notation: $[2,5)$.

Therefore, the domain of the expression is $\boxed{[2,5)}$.
Final Answer: The final answer is $[2,5)$. I hope it is correct.
```

## Usage

### Individual Tasks
```bash
# Run specific CoT task
lm_eval --model your_model --tasks minerva_math_algebra_cot

# Run with specific number of few-shot examples
lm_eval --model your_model --tasks minerva_math_algebra_cot --num_fewshot 4
```

### Combined Task
```bash
# Run all CoT tasks together
lm_eval --model your_model --tasks minerva_math_cot

# Compare with original tasks
lm_eval --model your_model --tasks minerva_math,minerva_math_cot
```

### With LLaDA Models
```bash
# Example with dLLM-cache LLaDA
accelerate launch evaluation_script.py -m lm_eval --model LLaDA --tasks minerva_math_cot --batch_size 4 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,is_feature_cache=True" \
--gen_kwargs "block_length=256,gen_length=512,steps=256,cfg_scale=0.0" \
--num_fewshot 4 \
--output_path ./minerva_math_cot_results \
--log_samples \
--trust_remote_code
```

## Expected Benefits

1. **Improved Reasoning**: The step-by-step prompting encourages models to break down complex problems
2. **Better Accuracy**: Explicit reasoning steps can lead to more accurate solutions
3. **Interpretability**: Generated solutions show the reasoning process
4. **Error Analysis**: Easier to identify where reasoning breaks down

## Evaluation Metrics

The CoT tasks use the same evaluation metrics as the original tasks:
- `exact_match`: Exact string match of the final answer
- `math_verify`: Mathematical verification using symbolic computation

## Files Structure

```
minerva_math/
├── utils_cot.py                           # CoT-specific utility functions
├── minerva_math_algebra_cot.yaml          # Algebra CoT task
├── minerva_math_counting_and_prob_cot.yaml # Counting & Probability CoT task
├── minerva_math_geometry_cot.yaml         # Geometry CoT task
├── minerva_math_intermediate_algebra_cot.yaml # Intermediate Algebra CoT task
├── minerva_math_num_theory_cot.yaml       # Number Theory CoT task
├── minerva_math_prealgebra_cot.yaml       # Pre-algebra CoT task
├── minerva_math_precalc_cot.yaml          # Pre-calculus CoT task
├── minerva_math_cot.yaml                  # Combined CoT task
└── README_COT.md                          # This file
```

## Notes

- The CoT versions maintain compatibility with the original evaluation framework
- All original functionality (math_verify, answer normalization) is preserved
- The enhanced prompting is designed to work with both base and instruction-tuned models
- Generation parameters may need adjustment for longer CoT responses (consider increasing `gen_length`)

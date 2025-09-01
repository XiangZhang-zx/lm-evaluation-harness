#!/usr/bin/env python3
"""
测试margin计算
"""

import json

def test_margin_calculation():
    """测试margin计算逻辑"""
    print("测试Margin计算...")
    
    # 加载一个MNLI样本
    with open("../results/llada/mnli/samples_mnli_2025-08-05T14-32-15.689993.jsonl", 'r') as f:
        sample = json.loads(f.readline())
    
    print("样本信息:")
    print(f"Doc ID: {sample['doc_id']}")
    print(f"Target: {sample['target']}")
    print(f"Filtered resps: {sample['filtered_resps']}")
    
    # 提取logprobs
    logprobs = [float(resp[0]) for resp in sample['filtered_resps']]
    true_label = int(sample['target'])
    
    print(f"\nLogprobs: {logprobs}")
    print(f"True label: {true_label}")
    
    # 计算预测
    pred = logprobs.index(max(logprobs))
    correct = pred == true_label
    
    print(f"Predicted: {pred}")
    print(f"Correct: {correct}")
    
    # 计算margin
    correct_prob = logprobs[true_label]
    wrong_probs = [prob for i, prob in enumerate(logprobs) if i != true_label]
    max_wrong_prob = max(wrong_probs)
    margin = correct_prob - max_wrong_prob
    
    print(f"\n=== Margin计算 ===")
    print(f"正确答案概率: {correct_prob}")
    print(f"最高错误答案概率: {max_wrong_prob}")
    print(f"Margin (正确-最高错误): {margin}")
    print(f"置信度 (|margin|): {abs(margin)}")
    
    # 解释margin的含义
    if margin > 0:
        print(f"✅ 模型对正确答案更有信心 (margin > 0)")
    else:
        print(f"❌ 模型对错误答案更有信心 (margin < 0)")
    
    print(f"Margin越大，模型越有信心选择正确答案")
    
    # 测试几个样本
    print(f"\n=== 测试前5个样本的margin ===")
    with open("../results/llada/mnli/samples_mnli_2025-08-05T14-32-15.689993.jsonl", 'r') as f:
        samples = [json.loads(line) for line in f.readlines()[:5]]
    
    for i, sample in enumerate(samples):
        logprobs = [float(resp[0]) for resp in sample['filtered_resps']]
        true_label = int(sample['target'])
        pred = logprobs.index(max(logprobs))
        correct = pred == true_label
        
        correct_prob = logprobs[true_label]
        wrong_probs = [prob for i, prob in enumerate(logprobs) if i != true_label]
        max_wrong_prob = max(wrong_probs)
        margin = correct_prob - max_wrong_prob
        
        print(f"样本{i}: correct={correct}, margin={margin:.3f}, confidence={abs(margin):.3f}")

if __name__ == "__main__":
    test_margin_calculation()

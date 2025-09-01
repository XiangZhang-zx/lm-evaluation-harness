#!/usr/bin/env python3
"""
简化版MNLI上传脚本，包含margin计算
"""

import json
import pandas as pd
import wandb
from datetime import datetime

def main():
    # 初始化W&B
    wandb.init(
        project="LLaDA-vs-Llama-MNLI-with-Margins",
        name=f"MNLI-Comparison-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        tags=["mnli", "model-comparison", "classification", "margins"],
        notes="MNLI comparison with margin analysis"
    )
    
    # 文件路径
    llada_file = "../results/llada/mnli/samples_mnli_2025-08-05T14-32-15.689993.jsonl"
    llama_file = "../results/llama_base/mnli/meta-llama__Llama-3.1-8B/samples_mnli_2025-08-10T05-00-08.927326.jsonl"
    
    print("Loading MNLI data...")
    
    # 加载数据
    with open(llada_file, 'r') as f:
        llada_data = [json.loads(line) for line in f]
    
    with open(llama_file, 'r') as f:
        llama_data = [json.loads(line) for line in f]
    
    print(f"LLaDA samples: {len(llada_data)}")
    print(f"Llama samples: {len(llama_data)}")
    
    # 创建字典
    llada_dict = {item['doc_id']: item for item in llada_data}
    llama_dict = {item['doc_id']: item for item in llama_data}
    
    # 找出共同的doc_id
    common_ids = set(llada_dict.keys()) & set(llama_dict.keys())
    print(f"Common samples: {len(common_ids)}")
    
    # 标签映射
    label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
    
    # 分析数据
    comparison_data = []
    llada_correct_count = 0
    llama_correct_count = 0
    both_correct_count = 0
    both_wrong_count = 0
    llada_advantages = []
    
    llada_margins = []
    llama_margins = []
    
    for doc_id in sorted(common_ids):
        llada_item = llada_dict[doc_id]
        llama_item = llama_dict[doc_id]
        
        # 获取真实标签
        true_label = int(llada_item['target'])
        
        # 计算LLaDA预测和margin
        llada_logprobs = [float(resp[0]) for resp in llada_item['filtered_resps']]
        llada_pred = llada_logprobs.index(max(llada_logprobs))
        llada_correct = llada_pred == true_label
        
        llada_correct_prob = llada_logprobs[true_label]
        llada_wrong_probs = [prob for i, prob in enumerate(llada_logprobs) if i != true_label]
        llada_max_wrong_prob = max(llada_wrong_probs)
        llada_margin = llada_correct_prob - llada_max_wrong_prob
        
        # 计算Llama预测和margin
        llama_logprobs = [float(resp[0]) for resp in llama_item['filtered_resps']]
        llama_pred = llama_logprobs.index(max(llama_logprobs))
        llama_correct = llama_pred == true_label
        
        llama_correct_prob = llama_logprobs[true_label]
        llama_wrong_probs = [prob for i, prob in enumerate(llama_logprobs) if i != true_label]
        llama_max_wrong_prob = max(llama_wrong_probs)
        llama_margin = llama_correct_prob - llama_max_wrong_prob
        
        # 统计
        if llada_correct:
            llada_correct_count += 1
        if llama_correct:
            llama_correct_count += 1
        if llada_correct and llama_correct:
            both_correct_count += 1
        if not llada_correct and not llama_correct:
            both_wrong_count += 1
        
        # 收集margin数据
        llada_margins.append(llada_margin)
        llama_margins.append(llama_margin)
        
        # 创建记录
        premise = llada_item['doc']['premise']
        hypothesis = llada_item['doc']['hypothesis']
        
        record = {
            'doc_id': doc_id,
            'premise': premise,
            'hypothesis': hypothesis,
            'true_label': label_map.get(true_label, str(true_label)),
            'llada_prediction': label_map.get(llada_pred, str(llada_pred)),
            'llama_prediction': label_map.get(llama_pred, str(llama_pred)),
            'llada_correct': llada_correct,
            'llama_correct': llama_correct,
            'llada_margin': llada_margin,
            'llama_margin': llama_margin,
            'llada_confidence': abs(llada_margin),
            'llama_confidence': abs(llama_margin),
            'premise_length': len(premise),
            'hypothesis_length': len(hypothesis),
            'agreement': llada_correct == llama_correct,
            'llada_advantage': llada_correct and not llama_correct,
            'llama_advantage': not llada_correct and llama_correct
        }
        
        comparison_data.append(record)
        
        if llada_correct and not llama_correct:
            llada_advantages.append(record)
    
    # 计算总体指标
    total_samples = len(common_ids)
    llada_accuracy = llada_correct_count / total_samples
    llama_accuracy = llama_correct_count / total_samples
    agreement_rate = (both_correct_count + both_wrong_count) / total_samples
    
    # 计算margin统计
    llada_avg_margin = sum(llada_margins) / len(llada_margins)
    llama_avg_margin = sum(llama_margins) / len(llama_margins)
    llada_avg_confidence = sum([abs(m) for m in llada_margins]) / len(llada_margins)
    llama_avg_confidence = sum([abs(m) for m in llama_margins]) / len(llama_margins)
    
    # 记录指标
    metrics = {
        'total_samples': total_samples,
        'llada_accuracy': llada_accuracy,
        'llama_accuracy': llama_accuracy,
        'llada_correct_count': llada_correct_count,
        'llama_correct_count': llama_correct_count,
        'both_correct_count': both_correct_count,
        'both_wrong_count': both_wrong_count,
        'agreement_rate': agreement_rate,
        'llada_advantage_count': len(llada_advantages),
        'accuracy_difference': llada_accuracy - llama_accuracy,
        'llada_avg_margin': llada_avg_margin,
        'llama_avg_margin': llama_avg_margin,
        'llada_avg_confidence': llada_avg_confidence,
        'llama_avg_confidence': llama_avg_confidence
    }
    
    wandb.log(metrics)
    
    print(f"\n=== MNLI 结果 ===")
    print(f"总样本数: {total_samples}")
    print(f"LLaDA准确率: {llada_accuracy:.3f} ({llada_correct_count}/{total_samples})")
    print(f"Llama准确率: {llama_accuracy:.3f} ({llama_correct_count}/{total_samples})")
    print(f"一致率: {agreement_rate:.3f}")
    print(f"LLaDA优势案例: {len(llada_advantages)}")
    print(f"LLaDA平均margin: {llada_avg_margin:.3f}")
    print(f"Llama平均margin: {llama_avg_margin:.3f}")
    print(f"LLaDA平均置信度: {llada_avg_confidence:.3f}")
    print(f"Llama平均置信度: {llama_avg_confidence:.3f}")
    
    # 创建表格
    comparison_df = pd.DataFrame(comparison_data)
    
    # 上传表格
    wandb.log({"mnli_comparison_table": wandb.Table(dataframe=comparison_df)})
    
    if llada_advantages:
        advantage_df = pd.DataFrame(llada_advantages)
        wandb.log({"mnli_llada_advantage_cases": wandb.Table(dataframe=advantage_df)})
    
    # 保存artifacts
    artifact = wandb.Artifact("mnli-comparison-with-margins", type="dataset")
    comparison_df.to_csv("mnli_comparison_with_margins.csv", index=False)
    artifact.add_file("mnli_comparison_with_margins.csv")
    
    if llada_advantages:
        advantage_df.to_csv("mnli_advantage_cases.csv", index=False)
        artifact.add_file("mnli_advantage_cases.csv")
    
    wandb.log_artifact(artifact)
    
    print(f"\nW&B链接: {wandb.run.url}")
    wandb.finish()

if __name__ == "__main__":
    main()

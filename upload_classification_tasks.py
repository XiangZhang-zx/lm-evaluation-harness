#!/usr/bin/env python3
"""
将MNLI和Social IQA分类任务结果上传到Weights & Biases
"""

import json
import pandas as pd
import wandb
from pathlib import Path
import argparse
from datetime import datetime
from collections import Counter

def load_jsonl(file_path):
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def analyze_mnli_task(llada_data, llama_data):
    """分析MNLI任务"""
    print("分析MNLI任务...")
    
    # 创建字典以便快速查找
    llada_dict = {item['doc_id']: item for item in llada_data}
    llama_dict = {item['doc_id']: item for item in llama_data}
    
    # 找出共同的doc_id
    common_ids = set(llada_dict.keys()) & set(llama_dict.keys())
    print(f"Common MNLI samples: {len(common_ids)}")
    
    comparison_data = []
    llada_correct_count = 0
    llama_correct_count = 0
    both_correct_count = 0
    both_wrong_count = 0
    llada_advantages = []
    
    # 标签映射
    label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
    
    for doc_id in sorted(common_ids):
        llada_item = llada_dict[doc_id]
        llama_item = llama_dict[doc_id]
        
        # 获取准确性
        llada_correct = llada_item.get('exact_match', 0.0) == 1.0
        llama_correct = llama_item.get('exact_match', 0.0) == 1.0
        
        # 统计
        if llada_correct:
            llada_correct_count += 1
        if llama_correct:
            llama_correct_count += 1
        if llada_correct and llama_correct:
            both_correct_count += 1
        if not llada_correct and not llama_correct:
            both_wrong_count += 1
        
        # 获取预测和真实标签
        true_label = int(llada_item['target'])
        llada_pred = llada_item.get('filtered_resps', [None])[0]
        llama_pred = llama_item.get('filtered_resps', [None])[0]
        
        # 获取输入文本
        premise = llada_item['doc']['premise']
        hypothesis = llada_item['doc']['hypothesis']
        
        record = {
            'doc_id': doc_id,
            'premise': premise,
            'hypothesis': hypothesis,
            'true_label': label_map.get(true_label, str(true_label)),
            'llada_prediction': str(llada_pred) if llada_pred is not None else "None",
            'llama_prediction': str(llama_pred) if llama_pred is not None else "None",
            'llada_correct': llada_correct,
            'llama_correct': llama_correct,
            'premise_length': len(premise),
            'hypothesis_length': len(hypothesis),
            'agreement': llada_correct == llama_correct,
            'llada_advantage': llada_correct and not llama_correct,
            'llama_advantage': not llada_correct and llama_correct
        }
        
        comparison_data.append(record)
        
        if llada_correct and not llama_correct:
            llada_advantages.append(record)
    
    return comparison_data, llada_advantages, {
        'total_samples': len(common_ids),
        'llada_correct_count': llada_correct_count,
        'llama_correct_count': llama_correct_count,
        'both_correct_count': both_correct_count,
        'both_wrong_count': both_wrong_count,
        'llada_accuracy': llada_correct_count / len(common_ids),
        'llama_accuracy': llama_correct_count / len(common_ids),
        'agreement_rate': (both_correct_count + both_wrong_count) / len(common_ids),
        'llada_advantage_count': len(llada_advantages)
    }

def analyze_social_iqa_task(llada_data, llama_data):
    """分析Social IQA任务"""
    print("分析Social IQA任务...")
    
    # 创建字典以便快速查找
    llada_dict = {item['doc_id']: item for item in llada_data}
    llama_dict = {item['doc_id']: item for item in llama_data}
    
    # 找出共同的doc_id
    common_ids = set(llada_dict.keys()) & set(llama_dict.keys())
    print(f"Common Social IQA samples: {len(common_ids)}")
    
    comparison_data = []
    llada_correct_count = 0
    llama_correct_count = 0
    both_correct_count = 0
    both_wrong_count = 0
    llada_advantages = []
    
    for doc_id in sorted(common_ids):
        llada_item = llada_dict[doc_id]
        llama_item = llama_dict[doc_id]
        
        # 获取准确性
        llada_correct = llada_item.get('exact_match', 0.0) == 1.0
        llama_correct = llama_item.get('exact_match', 0.0) == 1.0
        
        # 统计
        if llada_correct:
            llada_correct_count += 1
        if llama_correct:
            llama_correct_count += 1
        if llada_correct and llama_correct:
            both_correct_count += 1
        if not llada_correct and not llama_correct:
            both_wrong_count += 1
        
        # 获取预测和真实标签
        true_label = int(llada_item['target'])
        llada_pred = llada_item.get('filtered_resps', [None])[0]
        llama_pred = llama_item.get('filtered_resps', [None])[0]
        
        # 获取输入文本和选项
        inputs = llada_item['doc']['inputs']
        choices = llada_item['doc']['multiple_choice_targets']
        correct_answer = choices[true_label] if true_label < len(choices) else "Unknown"
        
        record = {
            'doc_id': doc_id,
            'question': inputs,
            'choices': str(choices),
            'correct_answer': correct_answer,
            'llada_prediction': str(llada_pred) if llada_pred is not None else "None",
            'llama_prediction': str(llama_pred) if llama_pred is not None else "None",
            'llada_correct': llada_correct,
            'llama_correct': llama_correct,
            'question_length': len(inputs),
            'num_choices': len(choices),
            'agreement': llada_correct == llama_correct,
            'llada_advantage': llada_correct and not llama_correct,
            'llama_advantage': not llada_correct and llama_correct
        }
        
        comparison_data.append(record)
        
        if llada_correct and not llama_correct:
            llada_advantages.append(record)
    
    return comparison_data, llada_advantages, {
        'total_samples': len(common_ids),
        'llada_correct_count': llada_correct_count,
        'llama_correct_count': llama_correct_count,
        'both_correct_count': both_correct_count,
        'both_wrong_count': both_wrong_count,
        'llada_accuracy': llada_correct_count / len(common_ids),
        'llama_accuracy': llama_correct_count / len(common_ids),
        'agreement_rate': (both_correct_count + both_wrong_count) / len(common_ids),
        'llada_advantage_count': len(llada_advantages)
    }

def upload_task_to_wandb(task_name, comparison_data, advantages, metrics, project_name):
    """上传单个任务到W&B"""
    
    # 初始化W&B run
    run_name = f"{task_name}-Comparison-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    wandb.init(
        project=project_name,
        name=run_name,
        tags=[task_name.lower(), "model-comparison", "classification"],
        notes=f"Detailed comparison of LLaDA vs Llama on {task_name}",
        reinit=True
    )
    
    # 记录总体指标
    wandb.log(metrics)
    
    print(f"\n=== {task_name} 结果 ===")
    print(f"总样本数: {metrics['total_samples']}")
    print(f"LLaDA准确率: {metrics['llada_accuracy']:.3f} ({metrics['llada_correct_count']}/{metrics['total_samples']})")
    print(f"Llama准确率: {metrics['llama_accuracy']:.3f} ({metrics['llama_correct_count']}/{metrics['total_samples']})")
    print(f"一致率: {metrics['agreement_rate']:.3f}")
    print(f"LLaDA优势案例: {metrics['llada_advantage_count']}")
    
    # 创建比较表格
    comparison_df = pd.DataFrame(comparison_data)
    
    # 上传完整比较表格
    wandb.log({f"{task_name.lower()}_comparison_table": wandb.Table(dataframe=comparison_df)})
    
    # 上传LLaDA优势案例表格
    if advantages:
        advantage_df = pd.DataFrame(advantages)
        wandb.log({f"{task_name.lower()}_llada_advantage_cases": wandb.Table(dataframe=advantage_df)})
    
    # 保存artifacts
    artifact = wandb.Artifact(f"{task_name.lower()}-comparison-data", type="dataset")
    
    # 保存CSV文件
    csv_filename = f"{task_name.lower()}_comparison.csv"
    comparison_df.to_csv(csv_filename, index=False)
    artifact.add_file(csv_filename)
    
    if advantages:
        advantage_csv = f"{task_name.lower()}_advantage_cases.csv"
        advantage_df.to_csv(advantage_csv, index=False)
        artifact.add_file(advantage_csv)
    
    wandb.log_artifact(artifact)
    
    print(f"W&B链接: {wandb.run.url}")
    
    wandb.finish()

def main():
    parser = argparse.ArgumentParser(description="Upload MNLI and Social IQA results to W&B")
    parser.add_argument("--project", default="LLaDA-vs-Llama-Classification", help="W&B project name")
    parser.add_argument("--entity", help="W&B entity (username or team)")
    parser.add_argument("--tasks", nargs='+', choices=['mnli', 'social_iqa', 'both'], 
                       default=['both'], help="Which tasks to upload")
    args = parser.parse_args()
    
    tasks_to_run = []
    if 'both' in args.tasks:
        tasks_to_run = ['mnli', 'social_iqa']
    else:
        tasks_to_run = args.tasks
    
    for task in tasks_to_run:
        if task == 'mnli':
            # MNLI文件路径
            llada_file = "../results/llada/mnli/samples_mnli_2025-08-05T14-32-15.689993.jsonl"
            llama_file = "../results/llama_base/mnli/meta-llama__Llama-3.1-8B/samples_mnli_2025-08-10T05-00-08.927326.jsonl"
            
            print("Loading MNLI data files...")
            llada_data = load_jsonl(llada_file)
            llama_data = load_jsonl(llama_file)
            
            comparison_data, advantages, metrics = analyze_mnli_task(llada_data, llama_data)
            upload_task_to_wandb("MNLI", comparison_data, advantages, metrics, args.project)
            
        elif task == 'social_iqa':
            # Social IQA文件路径
            llada_file = "../results/llada/bigbench_social_iqa_multiple_choice/GSAI-ML__LLaDA-8B-Base/samples_bigbench_social_iqa_multiple_choice_2025-08-09T08-58-35.668586.jsonl"
            llama_file = "../results/llama_base/bigbench_social_iqa_multiple_choice/meta-llama__Llama-3.1-8B/samples_bigbench_social_iqa_multiple_choice_2025-08-10T05-13-40.782346.jsonl"
            
            print("Loading Social IQA data files...")
            llada_data = load_jsonl(llada_file)
            llama_data = load_jsonl(llama_file)
            
            comparison_data, advantages, metrics = analyze_social_iqa_task(llada_data, llama_data)
            upload_task_to_wandb("Social_IQA", comparison_data, advantages, metrics, args.project)

if __name__ == "__main__":
    main()

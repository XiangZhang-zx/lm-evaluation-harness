#!/usr/bin/env python3
"""
将现有的GSM8K结果上传到Weights & Biases
包含完整的推理过程和详细的模型比较
"""

import json
import pandas as pd
import wandb
from pathlib import Path
import argparse
from datetime import datetime

def load_jsonl(file_path):
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def extract_final_answer(response_text):
    """从响应中提取最终答案"""
    if isinstance(response_text, list) and len(response_text) > 0:
        response_text = response_text[0]
    
    # 查找 #### 后面的数字
    if '####' in response_text:
        try:
            answer = response_text.split('####')[-1].strip()
            import re
            answer = re.findall(r'[\d.]+', answer)
            if answer:
                return answer[0]
        except:
            pass
    
    # 如果没有找到####格式，尝试其他方法
    import re
    # 查找$数字格式
    dollar_matches = re.findall(r'\$(\d+(?:\.\d+)?)', response_text)
    if dollar_matches:
        return dollar_matches[-1]
    
    # 查找纯数字
    number_matches = re.findall(r'\b(\d+(?:\.\d+)?)\b', response_text)
    if number_matches:
        return number_matches[-1]
    
    return None

def extract_target_answer(target_text):
    """从target中提取正确答案"""
    if '####' in target_text:
        try:
            answer = target_text.split('####')[-1].strip()
            import re
            answer = re.findall(r'[\d.]+', answer)
            if answer:
                return answer[0]
        except:
            pass
    return None

def classify_error_type(llama_response, question):
    """分类错误类型"""
    llama_resp = llama_response.lower()
    question = question.lower()
    
    # 基本分类
    has_steps = 'step' in llama_resp
    has_explanation = 'explanation:' in llama_resp
    has_calculation = any(op in llama_resp for op in ['+', '-', '*', '/', '=', 'x', '<<', '>>'])
    response_length = len(llama_resp)
    
    # 分类逻辑
    if response_length < 50:
        return "incomplete_response"
    elif has_explanation and has_calculation and response_length > 100:
        if has_steps:
            return "calculation_error"
        else:
            return "reasoning_error"
    elif "%" in question and "percentage" in question:
        return "percentage_calculation"
    elif "$" in question or "cost" in question or "price" in question:
        return "money_calculation"
    elif "hour" in question or "time" in question:
        return "time_calculation"
    elif "total" in question:
        return "summation_error"
    else:
        return "reasoning_error"

def main():
    parser = argparse.ArgumentParser(description="Upload GSM8K results to Weights & Biases")
    parser.add_argument("--project", default="llada-vs-llama-gsm8k", help="W&B project name")
    parser.add_argument("--entity", help="W&B entity (username or team)")
    args = parser.parse_args()
    
    # 初始化W&B
    wandb.init(
        project=args.project,
        entity=args.entity,
        name=f"GSM8K-Comparison-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        tags=["gsm8k", "model-comparison", "math-reasoning"],
        notes="Detailed comparison of LLaDA vs Llama on GSM8K with full reasoning chains"
    )
    
    # 文件路径
    llada_file = "../results/llada/gsm8k/samples_gsm8k_2025-08-03T18-43-34.529914.jsonl"
    llama_file = "../results/llama_base/gsm8k/meta-llama__Llama-3.1-8B/samples_gsm8k_2025-08-10T05-17-10.815718.jsonl"
    
    print("Loading data files...")
    
    # 加载数据
    llada_data = load_jsonl(llada_file)
    llama_data = load_jsonl(llama_file)
    
    print(f"LLaDA samples: {len(llada_data)}")
    print(f"Llama samples: {len(llama_data)}")
    
    # 创建字典以便快速查找
    llada_dict = {item['doc_id']: item for item in llada_data}
    llama_dict = {item['doc_id']: item for item in llama_data}
    
    # 找出共同的doc_id (只取LLaDA测试的120个问题)
    common_ids = set(llada_dict.keys()) & set(llama_dict.keys())
    print(f"Common samples: {len(common_ids)}")
    
    # 创建比较数据
    comparison_data = []
    llada_correct_llama_wrong = []
    
    llada_correct_count = 0
    llama_correct_count = 0
    both_correct_count = 0
    both_wrong_count = 0
    
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
        
        # 提取答案
        target_answer = extract_target_answer(llada_item['target'])
        llada_response = llada_item['resps'][0][0] if llada_item['resps'] else ""
        llama_response = llama_item['resps'][0][0] if llama_item['resps'] else ""
        
        llada_answer = extract_final_answer(llada_response)
        llama_answer = extract_final_answer(llama_response)
        
        # 错误类型分析
        error_type = None
        if llada_correct and not llama_correct:
            error_type = classify_error_type(llama_response, llada_item['doc']['question'])
        
        # 创建比较记录
        record = {
            'doc_id': doc_id,
            'question': llada_item['doc']['question'],
            'correct_answer': target_answer,
            'llada_answer': llada_answer,
            'llama_answer': llama_answer,
            'llada_correct': llada_correct,
            'llama_correct': llama_correct,
            'llada_response': llada_response,
            'llama_response': llama_response,
            'question_length': len(llada_item['doc']['question']),
            'llada_response_length': len(llada_response),
            'llama_response_length': len(llama_response),
            'error_type': error_type,
            'agreement': llada_correct == llama_correct,
            'llada_advantage': llada_correct and not llama_correct,
            'llama_advantage': not llada_correct and llama_correct
        }
        
        comparison_data.append(record)
        
        # 如果LLaDA对但Llama错，添加到特殊分析列表
        if llada_correct and not llama_correct:
            llada_correct_llama_wrong.append(record)
    
    # 计算总体指标
    total_samples = len(common_ids)
    llada_accuracy = llada_correct_count / total_samples
    llama_accuracy = llama_correct_count / total_samples
    agreement_rate = (both_correct_count + both_wrong_count) / total_samples
    
    # 记录总体指标
    wandb.log({
        "total_samples": total_samples,
        "llada_accuracy": llada_accuracy,
        "llama_accuracy": llama_accuracy,
        "llada_correct_count": llada_correct_count,
        "llama_correct_count": llama_correct_count,
        "both_correct_count": both_correct_count,
        "both_wrong_count": both_wrong_count,
        "agreement_rate": agreement_rate,
        "llada_advantage_count": len(llada_correct_llama_wrong),
        "accuracy_difference": llada_accuracy - llama_accuracy
    })
    
    print(f"\n=== 总体结果 ===")
    print(f"总样本数: {total_samples}")
    print(f"LLaDA准确率: {llada_accuracy:.3f} ({llada_correct_count}/{total_samples})")
    print(f"Llama准确率: {llama_accuracy:.3f} ({llama_correct_count}/{total_samples})")
    print(f"一致率: {agreement_rate:.3f}")
    print(f"LLaDA优势案例: {len(llada_correct_llama_wrong)}")
    
    # 创建比较表格
    comparison_df = pd.DataFrame(comparison_data)
    
    # 上传完整比较表格
    wandb.log({"comparison_table": wandb.Table(dataframe=comparison_df)})
    
    # 上传LLaDA优势案例表格
    if llada_correct_llama_wrong:
        advantage_df = pd.DataFrame(llada_correct_llama_wrong)
        wandb.log({"llada_advantage_cases": wandb.Table(dataframe=advantage_df)})
        
        # 错误类型分布
        from collections import Counter
        error_types = [case['error_type'] for case in llada_correct_llama_wrong if case['error_type']]
        error_counter = Counter(error_types)
        
        # 记录错误类型分布
        for error_type, count in error_counter.items():
            wandb.log({f"error_type_{error_type}": count})
        
        print(f"\n=== 错误类型分布 ===")
        for error_type, count in error_counter.most_common():
            percentage = (count / len(llada_correct_llama_wrong)) * 100
            print(f"{error_type}: {count} ({percentage:.1f}%)")
    
    # 保存artifacts
    artifact = wandb.Artifact("gsm8k-comparison-data", type="dataset")
    
    # 保存CSV文件
    comparison_df.to_csv("gsm8k_comparison.csv", index=False)
    artifact.add_file("gsm8k_comparison.csv")
    
    if llada_correct_llama_wrong:
        advantage_df.to_csv("llada_advantage_cases.csv", index=False)
        artifact.add_file("llada_advantage_cases.csv")
    
    wandb.log_artifact(artifact)
    
    print(f"\n=== W&B链接 ===")
    print(f"项目链接: {wandb.run.url}")
    
    wandb.finish()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
测试分类任务准确率计算
"""

import json

def load_jsonl(file_path):
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def test_mnli_accuracy():
    """测试MNLI准确率计算"""
    print("测试MNLI准确率计算...")
    
    # 文件路径
    llada_file = "../results/llada/mnli/samples_mnli_2025-08-05T14-32-15.689993.jsonl"
    llama_file = "../results/llama_base/mnli/meta-llama__Llama-3.1-8B/samples_mnli_2025-08-10T05-00-08.927326.jsonl"
    
    # 加载前100个样本进行测试
    print("加载数据...")
    with open(llada_file, 'r') as f:
        llada_data = [json.loads(line) for line in f.readlines()[:100]]
    
    with open(llama_file, 'r') as f:
        llama_data = [json.loads(line) for line in f.readlines()[:100]]
    
    # 创建字典
    llada_dict = {item['doc_id']: item for item in llada_data}
    llama_dict = {item['doc_id']: item for item in llama_data}
    
    # 找出共同的doc_id
    common_ids = set(llada_dict.keys()) & set(llama_dict.keys())
    print(f"Common samples: {len(common_ids)}")
    
    llada_correct_count = 0
    llama_correct_count = 0
    
    for doc_id in sorted(list(common_ids)[:10]):  # 只测试前10个
        llada_item = llada_dict[doc_id]
        llama_item = llama_dict[doc_id]
        
        # 获取真实标签
        true_label = int(llada_item['target'])
        
        # 计算LLaDA预测
        llada_logprobs = [float(resp[0]) for resp in llada_item['filtered_resps']]
        llada_pred = llada_logprobs.index(max(llada_logprobs))
        llada_correct = llada_pred == true_label
        
        # 计算Llama预测
        llama_logprobs = [float(resp[0]) for resp in llama_item['filtered_resps']]
        llama_pred = llama_logprobs.index(max(llama_logprobs))
        llama_correct = llama_pred == true_label
        
        print(f"Doc {doc_id}: true={true_label}, LLaDA_pred={llada_pred}({llada_correct}), Llama_pred={llama_pred}({llama_correct})")
        
        if llada_correct:
            llada_correct_count += 1
        if llama_correct:
            llama_correct_count += 1
    
    print(f"\n前10个样本结果:")
    print(f"LLaDA正确: {llada_correct_count}/10 = {llada_correct_count/10:.3f}")
    print(f"Llama正确: {llama_correct_count}/10 = {llama_correct_count/10:.3f}")

def test_social_iqa_accuracy():
    """测试Social IQA准确率计算"""
    print("\n测试Social IQA准确率计算...")
    
    # 文件路径
    llada_file = "../results/llada/bigbench_social_iqa_multiple_choice/GSAI-ML__LLaDA-8B-Base/samples_bigbench_social_iqa_multiple_choice_2025-08-09T08-58-35.668586.jsonl"
    llama_file = "../results/llama_base/bigbench_social_iqa_multiple_choice/meta-llama__Llama-3.1-8B/samples_bigbench_social_iqa_multiple_choice_2025-08-10T05-13-40.782346.jsonl"
    
    # 加载前10个样本进行测试
    print("加载数据...")
    with open(llada_file, 'r') as f:
        llada_data = [json.loads(line) for line in f.readlines()[:10]]
    
    with open(llama_file, 'r') as f:
        llama_data = [json.loads(line) for line in f.readlines()[:10]]
    
    # 创建字典
    llada_dict = {item['doc_id']: item for item in llada_data}
    llama_dict = {item['doc_id']: item for item in llama_data}
    
    # 找出共同的doc_id
    common_ids = set(llada_dict.keys()) & set(llama_dict.keys())
    print(f"Common samples: {len(common_ids)}")
    
    llada_correct_count = 0
    llama_correct_count = 0
    
    for doc_id in sorted(list(common_ids)[:10]):  # 只测试前10个
        llada_item = llada_dict[doc_id]
        llama_item = llama_dict[doc_id]
        
        # 获取真实标签
        true_label = int(llada_item['target'])
        
        # 计算LLaDA预测
        llada_logprobs = [float(resp[0]) for resp in llada_item['filtered_resps']]
        llada_pred = llada_logprobs.index(max(llada_logprobs))
        llada_correct = llada_pred == true_label
        
        # 计算Llama预测
        llama_logprobs = [float(resp[0]) for resp in llama_item['filtered_resps']]
        llama_pred = llama_logprobs.index(max(llama_logprobs))
        llama_correct = llama_pred == true_label
        
        # 获取选项文本
        choices = llada_item['doc']['multiple_choice_targets']
        
        print(f"Doc {doc_id}: true={true_label}({choices[true_label] if true_label < len(choices) else 'Unknown'})")
        print(f"  LLaDA_pred={llada_pred}({choices[llada_pred] if llada_pred < len(choices) else 'Unknown'}) - {llada_correct}")
        print(f"  Llama_pred={llama_pred}({choices[llama_pred] if llama_pred < len(choices) else 'Unknown'}) - {llama_correct}")
        
        if llada_correct:
            llada_correct_count += 1
        if llama_correct:
            llama_correct_count += 1
    
    print(f"\n前10个样本结果:")
    print(f"LLaDA正确: {llada_correct_count}/10 = {llada_correct_count/10:.3f}")
    print(f"Llama正确: {llama_correct_count}/10 = {llama_correct_count/10:.3f}")

if __name__ == "__main__":
    test_mnli_accuracy()
    test_social_iqa_accuracy()

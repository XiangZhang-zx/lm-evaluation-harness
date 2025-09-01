#!/usr/bin/env python3
"""
调试分类任务准确率计算问题
"""

import json

def debug_mnli():
    """调试MNLI准确率计算"""
    print("=== 调试MNLI准确率计算 ===")
    
    # 加载数据
    llada_file = "../results/llada/mnli/samples_mnli_2025-08-05T14-32-15.689993.jsonl"
    llama_file = "../results/llama_base/mnli/meta-llama__Llama-3.1-8B/samples_mnli_2025-08-10T05-00-08.927326.jsonl"
    
    # 只加载前10个样本进行调试
    with open(llada_file, 'r') as f:
        llada_data = [json.loads(line) for line in f.readlines()[:10]]
    
    with open(llama_file, 'r') as f:
        llama_data = [json.loads(line) for line in f.readlines()[:10]]
    
    print(f"LLaDA样本数: {len(llada_data)}")
    print(f"Llama样本数: {len(llama_data)}")
    
    # 创建字典
    llada_dict = {item['doc_id']: item for item in llada_data}
    llama_dict = {item['doc_id']: item for item in llama_data}
    
    print(f"LLaDA doc_ids: {list(llada_dict.keys())}")
    print(f"Llama doc_ids: {list(llama_dict.keys())}")
    
    # 找出共同的doc_id
    common_ids = set(llada_dict.keys()) & set(llama_dict.keys())
    print(f"Common doc_ids: {sorted(common_ids)}")
    
    # 分析每个样本
    llada_correct_count = 0
    llama_correct_count = 0
    
    for doc_id in sorted(common_ids):
        print(f"\n--- 分析样本 {doc_id} ---")
        
        llada_item = llada_dict[doc_id]
        llama_item = llama_dict[doc_id]
        
        # 检查基本信息
        print(f"LLaDA target: {llada_item['target']}")
        print(f"Llama target: {llama_item['target']}")
        print(f"Target匹配: {llada_item['target'] == llama_item['target']}")
        
        # 获取真实标签
        true_label = int(llada_item['target'])
        print(f"True label: {true_label}")
        
        # 分析LLaDA
        print(f"LLaDA filtered_resps: {llada_item['filtered_resps']}")
        try:
            llada_logprobs = [float(resp[0]) for resp in llada_item['filtered_resps']]
            llada_pred = llada_logprobs.index(max(llada_logprobs))
            llada_correct = llada_pred == true_label
            print(f"LLaDA logprobs: {llada_logprobs}")
            print(f"LLaDA pred: {llada_pred}, correct: {llada_correct}")
            if llada_correct:
                llada_correct_count += 1
        except Exception as e:
            print(f"LLaDA处理错误: {e}")
        
        # 分析Llama
        print(f"Llama filtered_resps: {llama_item['filtered_resps']}")
        try:
            llama_logprobs = [float(resp[0]) for resp in llama_item['filtered_resps']]
            llama_pred = llama_logprobs.index(max(llama_logprobs))
            llama_correct = llama_pred == true_label
            print(f"Llama logprobs: {llama_logprobs}")
            print(f"Llama pred: {llama_pred}, correct: {llama_correct}")
            if llama_correct:
                llama_correct_count += 1
        except Exception as e:
            print(f"Llama处理错误: {e}")
    
    print(f"\n=== 最终统计 ===")
    print(f"总样本数: {len(common_ids)}")
    print(f"LLaDA正确: {llada_correct_count}/{len(common_ids)} = {llada_correct_count/len(common_ids):.3f}")
    print(f"Llama正确: {llama_correct_count}/{len(common_ids)} = {llama_correct_count/len(common_ids):.3f}")

if __name__ == "__main__":
    debug_mnli()

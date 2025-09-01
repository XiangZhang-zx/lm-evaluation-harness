#!/usr/bin/env python3
"""
修改版的zeno_visualize脚本，显示完整的推理过程而不是只显示最终答案
"""

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Union

import pandas as pd
from zeno_client import ZenoClient, ZenoMetric

from lm_eval.utils import (
    get_latest_filename,
    get_results_filenames,
    get_sample_results_filenames,
)

eval_logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload your data to the Zeno AI evaluation platform with FULL reasoning chains."
    )
    parser.add_argument(
        "--data_path",
        required=True,
        help="Where to find the results of the benchmarks that have been run.",
    )
    parser.add_argument(
        "--project_name",
        required=True,
        help="The name of the generated Zeno project.",
    )
    return parser.parse_args()

def generate_system_df_with_reasoning(data, config):
    """生成包含完整推理过程的系统数据框"""
    ids = (
        [x["doc_id"] for x in data]
        if not config.get("filter_list")
        else [f"{x['doc_id']}.{x['filter']}" for x in data]
    )
    system_dict = {"id": ids}
    system_dict["doc_id"] = [x["doc_id"] for x in data]
    
    if config.get("filter_list"):
        system_dict["filter"] = [x["filter"] for x in data]
    
    # 添加完整推理过程
    if config["output_type"] == "generate_until":
        # 完整推理过程
        system_dict["full_reasoning"] = [str(x["resps"][0][0]) if x["resps"] else "" for x in data]
        # 最终答案
        system_dict["final_answer"] = [str(x["filtered_resps"][0]) for x in data]
        # 推理长度
        system_dict["reasoning_length"] = [len(str(x["resps"][0][0])) if x["resps"] else 0 for x in data]
        # 使用完整推理作为主要输出
        system_dict["output"] = system_dict["full_reasoning"]
    else:
        system_dict["output"] = [""] * len(ids)
        if config["output_type"] == "loglikelihood":
            system_dict["output"] = [
                "correct" if x["filtered_resps"][0][1] is True else "incorrect"
                for x in data
            ]
        elif config["output_type"] == "multiple_choice":
            system_dict["output"] = [
                ", ".join([str(y[0]) for y in x["filtered_resps"]]) for x in data
            ]
            system_dict["num_answers"] = [len(x["filtered_resps"]) for x in data]
        elif config["output_type"] == "loglikelihood_rolling":
            system_dict["output"] = [str(x["filtered_resps"][0]) for x in data]

    # 添加指标
    metrics = {
        metric["metric"]: [x[metric["metric"]] for x in data]
        for metric in config["metric_list"]
    }
    system_dict.update(metrics)
    system_df = pd.DataFrame(system_dict)
    return system_df

def main():
    args = parse_args()
    
    # 获取API密钥
    api_key = os.environ.get("ZENO_API_KEY")
    if not api_key:
        raise ValueError("请设置ZENO_API_KEY环境变量")
    
    client = ZenoClient(api_key)
    
    # 获取模型列表
    data_path = Path(args.data_path)
    models = [f.name for f in data_path.iterdir() if f.is_dir()]
    
    if not models:
        raise ValueError(f"在 {args.data_path} 中没有找到模型文件夹")
    
    print(f"找到模型: {models}")
    
    # 处理每个模型
    for model in models:
        model_dir = data_path / model
        model_files = [f.as_posix() for f in model_dir.iterdir() if f.is_file()]
        
        # 获取结果文件
        results_files = get_results_filenames(model_files)
        if not results_files:
            print(f"警告: 模型 {model} 没有找到结果文件")
            continue
            
        latest_results = get_latest_filename(results_files)
        
        # 获取样本文件
        sample_files = get_sample_results_filenames(model_files)
        if not sample_files:
            print(f"警告: 模型 {model} 没有找到样本文件")
            continue
            
        latest_samples = get_latest_filename(sample_files)
        
        # 加载配置和数据
        with open(latest_results, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        configs = results["configs"]
        
        # 处理每个任务
        for task_name, config in configs.items():
            print(f"处理任务: {task_name}, 模型: {model}")
            
            # 加载样本数据
            task_data = []
            with open(latest_samples, 'r', encoding='utf-8') as f:
                for line in f:
                    sample = json.loads(line)
                    task_data.append(sample)
            
            if not task_data:
                print(f"警告: 任务 {task_name} 没有样本数据")
                continue
            
            # 创建项目名称
            project_name = f"{args.project_name} - {task_name} (Full Reasoning)"
            
            try:
                # 创建项目
                project = client.create_project(
                    name=project_name,
                    view="text-classification",
                    description=f"LM Eval Harness results for {task_name} with full reasoning chains",
                    public=False
                )
                
                print(f"创建项目: {project_name}")
                print(f"项目链接: https://hub.zenoml.com/project/{project.uuid}/{project_name}")
                
                # 生成数据集
                dataset_df = generate_dataset_df(task_data, config)
                
                # 上传数据集
                project.upload_dataset(
                    dataset_df,
                    id_column="id",
                    data_column="data",
                    label_column="labels"
                )
                
                # 为每个模型生成系统数据
                system_df = generate_system_df_with_reasoning(task_data, config)
                
                # 上传系统
                project.upload_system(
                    system_df,
                    name=model,
                    id_column="id",
                    output_column="output"
                )
                
                print(f"成功上传模型 {model} 的数据")
                
            except Exception as e:
                print(f"上传失败: {e}")
                continue

def generate_dataset_df(data, config):
    """生成数据集DataFrame"""
    ids = (
        [x["doc_id"] for x in data]
        if not config.get("filter_list")
        else [f"{x['doc_id']}.{x['filter']}" for x in data]
    )
    labels = [x["target"] for x in data]
    instance = [""] * len(ids)

    if config["output_type"] == "loglikelihood":
        instance = [x["arguments"]["gen_args_0"]["arg_0"] for x in data]
        labels = [x["arguments"]["gen_args_0"]["arg_1"] for x in data]
    elif config["output_type"] == "multiple_choice":
        instance = [
            x["arguments"]["gen_args_0"]["arg_0"]
            + "\n\n"
            + "\n".join([f"- {y[1]}" for y in x["arguments"]])
            for x in data
        ]
    elif config["output_type"] == "loglikelihood_rolling":
        instance = [x["arguments"]["gen_args_0"]["arg_0"] for x in data]
    elif config["output_type"] == "generate_until":
        instance = [x["arguments"]["gen_args_0"]["arg_0"] for x in data]

    return pd.DataFrame(
        {
            "id": ids,
            "doc_id": [x["doc_id"] for x in data],
            "data": instance,
            "input_len": [len(x) for x in instance],
            "labels": labels,
            "output_type": config["output_type"],
        }
    )

if __name__ == "__main__":
    main()

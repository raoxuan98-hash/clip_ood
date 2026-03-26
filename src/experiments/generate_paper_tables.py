"""
生成论文表格的脚本
从实验结果中提取数据，生成 LaTeX 和 Markdown 表格
"""

import os
import json
import argparse
from typing import Dict, List
import numpy as np


# LADA 任务顺序
LADA_TASKS = [
    "aircraft", "caltech101", "dtd", "eurosat", 
    "flowers", "food101", "mnist", "oxford_pets", 
    "stanford_cars", "sun397"
]

METHODS = [
    ('zeroshot', 'Zero-shot CLIP'),
    ('lora_vanilla', 'LoRA (Baseline)'),
    ('lora_nsp', 'LoRA-NSP (Ours)')
]


def load_results(results_dir: str, method: str) -> Dict:
    """加载实验结果"""
    # 尝试从汇总文件加载，或者从单独的 results.json 加载
    result_file = os.path.join(results_dir, f"{method}_results.json")
    if not os.path.exists(result_file):
        # 尝试在子目录中寻找
        result_file = os.path.join(results_dir, method, "results.json")
        
    if not os.path.exists(result_file):
        print(f"Warning: {result_file} not found")
        return None
    
    with open(result_file, 'r') as f:
        return json.load(f)


def format_table1_latex(results_dict: Dict[str, Dict]) -> str:
    """
    生成 Table 1 (主结果表) 的 LaTeX 代码
    对标 LADA Table 1
    """
    lines = []
    lines.append("% Table 1: Main Results on X-TAIL")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Comparison of different continual learning methods on X-TAIL 16-shot for each task in terms of Transfer, Average, and Last scores (\\%). The best results are highlighted with bold style.}")
    lines.append("\\label{tab:main_results}")
    
    # 表格头部
    header = "Method & " + " & ".join([t[:8] for t in LADA_TASKS]) + " & Transfer & Average & Last \\\\"
    lines.append("\\begin{tabular}{l" + "c"*len(LADA_TASKS) + "ccc}")
    lines.append("\\toprule")
    lines.append(header)
    lines.append("\\midrule")
    
    # 各方法结果
    for method_key, method_name in METHODS:
        if method_key not in results_dict:
            continue
        
        res = results_dict[method_key]
        per_task = res.get('per_task_metrics', {})
        
        # Transfer row
        accs = [f"{per_task.get(t, {}).get('transfer', 0):.1f}" for t in LADA_TASKS]
        transfer = res.get('metrics', {}).get('transfer', 0)
        line = f"{method_name} & " + " & ".join(accs) + f" & {transfer:.1f} & - & - \\\\"
        lines.append(line)
        
        # Average row
        accs = [f"{per_task.get(t, {}).get('average', 0):.1f}" for t in LADA_TASKS]
        avg = res.get('metrics', {}).get('average', 0)
        line = f" & " + " & ".join(accs) + f" & - & {avg:.1f} & - \\\\"
        lines.append(line)
        
        # Last row
        accs = [f"{per_task.get(t, {}).get('last', 0):.1f}" for t in LADA_TASKS]
        last = res.get('metrics', {}).get('last', 0)
        line = f" & " + " & ".join(accs) + f" & - & - & {last:.1f} \\\\"
        lines.append(line)
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    
    return "\n".join(lines)


def format_table1_markdown(results_dict: Dict[str, Dict]) -> str:
    """
    生成 Table 1 的 Markdown 格式
    """
    lines = []
    lines.append("# Table 1: Main Results on X-TAIL 16-shot")
    lines.append("")
    
    # 表头
    header = "| Method | " + " | ".join(LADA_TASKS) + " | Transfer | Average | Last |"
    separator = "|" + "|".join(["-"*10 for _ in range(len(LADA_TASKS) + 4)]) + "|"
    lines.append(header)
    lines.append(separator)
    
    # 各方法结果
    for method_key, method_name in METHODS:
        if method_key not in results_dict:
            continue
        
        res = results_dict[method_key]
        per_task = res.get('per_task_metrics', {})
        
        # Transfer
        accs = [f"{per_task.get(t, {}).get('transfer', 0):.1f}" for t in LADA_TASKS]
        transfer = res.get('metrics', {}).get('transfer', 0)
        lines.append(f"| {method_name} | " + " | ".join(accs) + f" | {transfer:.1f} | - | - |")
        
        # Average
        accs = [f"{per_task.get(t, {}).get('average', 0):.1f}" for t in LADA_TASKS]
        avg = res.get('metrics', {}).get('average', 0)
        lines.append(f"| | " + " | ".join(accs) + f" | - | {avg:.1f} | - |")
        
        # Last
        accs = [f"{per_task.get(t, {}).get('last', 0):.1f}" for t in LADA_TASKS]
        last = res.get('metrics', {}).get('last', 0)
        lines.append(f"| | " + " | ".join(accs) + f" | - | - | **{last:.1f}** |")
    
    return "\n".join(lines)


def format_summary_markdown(results_dict: Dict[str, Dict]) -> str:
    """生成简化的摘要表格"""
    lines = []
    lines.append("# Summary of Results")
    lines.append("")
    lines.append("| Method | Transfer | Average | Last | Forgetting | OOD AUROC |")
    lines.append("|--------|----------|---------|------|------------|----------|")
    
    for method_key, method_name in METHODS:
        if method_key not in results_dict:
            continue
        
        res = results_dict[method_key]
        metrics = res.get('metrics', {})
        forgetting = res.get('forgetting_rate', 0)
        
        transfer = metrics.get('transfer', 0)
        avg = metrics.get('average', 0)
        last = metrics.get('last', 0)
        ood_auroc = metrics.get('avg_ood_auroc', 0)
        
        line = f"| {method_name} | {transfer:.1f} | {avg:.1f} | {last:.1f} | {forgetting:.1f} | {ood_auroc:.1f} |"
        lines.append(line)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate paper tables from results")
    parser.add_argument("--results_dir", type=str, required=True,
                       help="Directory containing experiment results")
    parser.add_argument("--output_dir", type=str, default="experiments/tables",
                       help="Output directory for tables")
    parser.add_argument("--format", type=str, nargs='+', default=['markdown', 'latex'],
                       choices=['markdown', 'latex'])
    args = parser.parse_args()
    
    # 加载所有结果
    results_dict = {}
    for method, _ in METHODS:
        res = load_results(args.results_dir, method)
        if res:
            results_dict[method] = res
    
    if not results_dict:
        print("No results found!")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成表格
    if 'markdown' in args.format:
        # 主结果表
        table1_md = format_table1_markdown(results_dict)
        with open(os.path.join(args.output_dir, "table1_main_results.md"), 'w') as f:
            f.write(table1_md)
        print(f"Saved: {args.output_dir}/table1_main_results.md")
        
        # 摘要表
        summary_md = format_summary_markdown(results_dict)
        with open(os.path.join(args.output_dir, "summary.md"), 'w') as f:
            f.write(summary_md)
        print(f"Saved: {args.output_dir}/summary.md")
    
    if 'latex' in args.format:
        table1_tex = format_table1_latex(results_dict)
        with open(os.path.join(args.output_dir, "table1_main_results.tex"), 'w') as f:
            f.write(table1_tex)
        print(f"Saved: {args.output_dir}/table1_main_results.tex")
    
    # 打印到控制台
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(format_summary_markdown(results_dict))


if __name__ == "__main__":
    main()

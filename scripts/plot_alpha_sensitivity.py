#!/usr/bin/env python3
"""
绘制集成分类器 Alpha 敏感性分析图表
用于 Table 5: 集成分类器性能变化分析
"""

import os
import sys
import argparse
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Alpha Sensitivity Analysis")
    
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing alpha sweep results")
    
    parser.add_argument("--pattern", type=str, default="alpha_*",
                       help="Glob pattern to match result directories")
    
    parser.add_argument("--output_dir", type=str, default="figures",
                       help="Directory to save output figures")
    
    parser.add_argument("--output_prefix", type=str, default="alpha_sensitivity",
                       help="Prefix for output figure files")
    
    return parser.parse_args()


def load_alpha_results(input_dir, pattern):
    """加载alpha扫描结果"""
    results = []
    
    # 查找所有匹配的目录
    search_pattern = os.path.join(input_dir, pattern)
    result_dirs = glob.glob(search_pattern)
    
    print(f"Found {len(result_dirs)} result directories matching '{pattern}'")
    
    for result_dir in result_dirs:
        # 从目录名提取alpha值
        dir_name = os.path.basename(result_dir)
        try:
            # 解析 alpha_0_8 -> 0.8
            if dir_name.startswith("alpha_"):
                alpha_str = dir_name.replace("alpha_", "").replace("_", ".")
                alpha = float(alpha_str)
            else:
                continue
        except ValueError:
            print(f"  Warning: Could not parse alpha from {dir_name}")
            continue
        
        # 查找结果文件
        result_files = glob.glob(os.path.join(result_dir, "*.json"))
        if not result_files:
            # 尝试其他可能的文件名
            result_files = glob.glob(os.path.join(result_dir, "results*.json"))
        
        if result_files:
            # 加载最新的结果文件
            result_file = max(result_files, key=os.path.getmtime)
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                
                # 提取ID和OOD准确率
                id_acc = data.get('id_accuracy', data.get('avg_id_accuracy', None))
                ood_acc = data.get('ood_accuracy', data.get('avg_ood_accuracy', None))
                
                if id_acc is not None and ood_acc is not None:
                    results.append({
                        'alpha': alpha,
                        'id_accuracy': id_acc * 100 if id_acc <= 1 else id_acc,  # 转换为百分比
                        'ood_accuracy': ood_acc * 100 if ood_acc <= 1 else ood_acc,
                        'file': result_file
                    })
                    print(f"  Alpha={alpha:.3f}: ID={id_acc:.2f}%, OOD={ood_acc:.2f}%")
            except Exception as e:
                print(f"  Error loading {result_file}: {e}")
    
    # 按alpha排序
    results.sort(key=lambda x: x['alpha'])
    return results


def plot_id_ood_curves(results, output_dir, output_prefix):
    """绘制ID/OOD准确率随alpha变化的曲线"""
    if not results:
        print("No results to plot!")
        return
    
    alphas = [r['alpha'] for r in results]
    id_accs = [r['id_accuracy'] for r in results]
    ood_accs = [r['ood_accuracy'] for r in results]
    
    # 计算综合得分
    combined_scores = [(id_acc + ood_acc) / 2 for id_acc, ood_acc in zip(id_accs, ood_accs)]
    
    # 找到最佳alpha
    best_idx = np.argmax(combined_scores)
    best_alpha = alphas[best_idx]
    best_score = combined_scores[best_idx]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制曲线
    ax.plot(alphas, id_accs, 'b-o', label='ID Accuracy', linewidth=2, markersize=6)
    ax.plot(alphas, ood_accs, 'r-s', label='OOD Accuracy', linewidth=2, markersize=6)
    ax.plot(alphas, combined_scores, 'g--^', label='Combined Score', linewidth=2, markersize=6)
    
    # 标记最佳点
    ax.axvline(x=best_alpha, color='purple', linestyle=':', alpha=0.7, 
               label=f'Optimal α={best_alpha:.3f}')
    ax.scatter([best_alpha], [best_score], color='purple', s=200, zorder=5, marker='*')
    
    # 添加零样本和纯LR-RGDA的参考线（假设值）
    ax.axhline(y=60, color='red', linestyle='--', alpha=0.3, label='Pure Zero-shot (ID~60%)')
    ax.axhline(y=80, color='blue', linestyle='--', alpha=0.3, label='Pure LR-RGDA (ID~80%)')
    
    ax.set_xlabel('Alpha (LR-RGDA Weight)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Ensemble Classifier: ID/OOD Accuracy vs Alpha', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.45, 1.05)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, f"{output_prefix}_curves.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_file}")
    plt.close()
    
    return best_alpha, best_score


def plot_performance_comparison(results, output_dir, output_prefix):
    """绘制相对单一分类器的性能提升对比"""
    if not results:
        return
    
    alphas = [r['alpha'] for r in results]
    id_accs = [r['id_accuracy'] for r in results]
    ood_accs = [r['ood_accuracy'] for r in results]
    
    # 假设纯零样本ID=60%, OOD=85%
    zeroshot_id = 60
    zeroshot_ood = 85
    
    # 假设纯LR-RGDA ID=80%, OOD=30%
    lrrgda_id = 80
    lrrgda_ood = 30
    
    # 计算相对提升
    id_gain_vs_zeroshot = [id_acc - zeroshot_id for id_acc in id_accs]
    ood_gain_vs_lrrgda = [ood_acc - lrrgda_ood for ood_acc in ood_accs]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制提升曲线
    ax.plot(alphas, id_gain_vs_zeroshot, 'b-o', 
            label='ID Gain (vs Zero-shot)', linewidth=2, markersize=6)
    ax.plot(alphas, ood_gain_vs_lrrgda, 'r-s', 
            label='OOD Gain (vs LR-RGDA)', linewidth=2, markersize=6)
    
    # 零点线
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 填充区域
    ax.fill_between(alphas, 0, id_gain_vs_zeroshot, 
                    where=[g > 0 for g in id_gain_vs_zeroshot], 
                    alpha=0.3, color='blue', label='ID Improvement Region')
    ax.fill_between(alphas, 0, ood_gain_vs_lrrgda, 
                    where=[g > 0 for g in ood_gain_vs_lrrgda], 
                    alpha=0.3, color='red', label='OOD Improvement Region')
    
    ax.set_xlabel('Alpha (LR-RGDA Weight)', fontsize=12)
    ax.set_ylabel('Performance Gain (%)', fontsize=12)
    ax.set_title('Ensemble: Performance Gain vs Single Classifiers', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.45, 1.05)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, f"{output_prefix}_gains.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_3d_surface(results, output_dir, output_prefix):
    """绘制3D性能曲面（如果数据足够）"""
    if len(results) < 5:
        print("Not enough data points for 3D plot")
        return
    
    alphas = [r['alpha'] for r in results]
    id_accs = [r['id_accuracy'] for r in results]
    ood_accs = [r['ood_accuracy'] for r in results]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制3D曲线
    combined = [(id_acc + ood_acc) / 2 for id_acc, ood_acc in zip(id_accs, ood_accs)]
    
    ax.plot(alphas, id_accs, ood_accs, 'b-o', linewidth=2, markersize=6)
    ax.scatter(alphas, id_accs, ood_accs, c=combined, cmap='viridis', s=100)
    
    ax.set_xlabel('Alpha', fontsize=11)
    ax.set_ylabel('ID Accuracy (%)', fontsize=11)
    ax.set_zlabel('OOD Accuracy (%)', fontsize=11)
    ax.set_title('Ensemble Performance 3D Trajectory', fontsize=14)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, f"{output_prefix}_3d.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def generate_summary_table(results, output_dir, output_prefix):
    """生成结果汇总表"""
    if not results:
        return
    
    # 创建Markdown表格
    table_lines = ["# Alpha Sensitivity Analysis Results\n"]
    table_lines.append("| Alpha | ID Acc (%) | OOD Acc (%) | Combined (%) | ID Gain* | OOD Gain** |\n")
    table_lines.append("|-------|-----------|------------|-------------|---------|-----------|\n")
    
    zeroshot_id = 60
    lrrgda_ood = 30
    
    for r in results:
        alpha = r['alpha']
        id_acc = r['id_accuracy']
        ood_acc = r['ood_accuracy']
        combined = (id_acc + ood_acc) / 2
        id_gain = id_acc - zeroshot_id
        ood_gain = ood_acc - lrrgda_ood
        
        table_lines.append(
            f"| {alpha:.3f} | {id_acc:.1f} | {ood_acc:.1f} | {combined:.1f} | "
            f"{id_gain:+.1f} | {ood_gain:+.1f} |\n"
        )
    
    table_lines.append(r"\n\* ID Gain = relative to pure zero-shot (60%)\n")
    table_lines.append(r"\*\* OOD Gain = relative to pure LR-RGDA (30%)\n")
    
    # 保存表格
    output_file = os.path.join(output_dir, f"{output_prefix}_table.md")
    with open(output_file, 'w') as f:
        f.writelines(table_lines)
    print(f"Saved: {output_file}")


def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("Alpha Sensitivity Analysis Plotting")
    print("="*80)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)
    
    # 加载结果
    results = load_alpha_results(args.input_dir, args.pattern)
    
    if not results:
        print("\nNo valid results found!")
        return
    
    print(f"\nLoaded {len(results)} alpha configurations")
    
    # 生成图表
    print("\nGenerating plots...")
    best_alpha, best_score = plot_id_ood_curves(results, args.output_dir, args.output_prefix)
    plot_performance_comparison(results, args.output_dir, args.output_prefix)
    plot_3d_surface(results, args.output_dir, args.output_prefix)
    
    # 生成汇总表
    generate_summary_table(results, args.output_dir, args.output_prefix)
    
    print("\n" + "="*80)
    print(f"Analysis Complete!")
    print(f"Optimal Alpha: {best_alpha:.3f} (Combined Score: {best_score:.2f}%)")
    print("="*80)


if __name__ == "__main__":
    main()

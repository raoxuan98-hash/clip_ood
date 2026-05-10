"""
持续学习评估指标计算模块
基于 LADA 论文的 Transfer / Average / Last 指标

论文公式参考 (Appendix A):
- Transfer_k = (1/(k-1)) * sum_{j=1}^{k-1} a_hat^(j)_k, for k = 2,...,K
- Average_k = (1/K) * sum_{j=1}^K a_hat^(j)_k, for k = 1,...,K  
- Last_k = a_hat^(K)_k, for k = 1,...,K

其中 a_hat^(j)_k 表示训练完任务 j 后，在任务 k 上的准确率
"""

import numpy as np
from typing import List, Dict
import json


class ContinualLearningMetrics:
    """
    持续学习评估指标计算器
    
    准确实现 LADA 论文中的三个核心指标:
    - Transfer: 前向迁移/遗忘指标
    - Average: 综合性能指标  
    - Last: 后向遗忘指标
    """
    
    def __init__(self, task_names: List[str]):
        """
        Args:
            task_names: 任务名称列表（按训练顺序）
        """
        self.task_names = task_names
        self.K = len(task_names)
        # accuracy_matrix[i, j]: 训练完任务 i 后，在任务 j 上的准确率
        # i, j 都是 0-indexed
        self.accuracy_matrix = np.full((self.K, self.K), -1.0)
        
    def update(self, step: int, task_accuracies: Dict[str, float]):
        """
        更新第 step 步的评估结果
        
        Args:
            step: 当前训练步骤（0-indexed，训练完第 step 个任务后）
            task_accuracies: {task_name: accuracy} 所有任务的准确率
        """
        for task_idx, task_name in enumerate(self.task_names):
            if task_name in task_accuracies:
                self.accuracy_matrix[step, task_idx] = task_accuracies[task_name]
    
    def calculate_transfer(self) -> float:
        """
        计算 Transfer 指标
        
        根据论文公式 (12):
        Transfer_k = (1/(k-1)) * sum_{j=1}^{k-1} a_hat^(j)_k, for k = 2,3,...,K
        
        其中:
        - a_hat^(j)_k: 训练完任务 j 后，在任务 k 上的准确率
        - Transfer_k: 对于任务 k，在学习前面任务 (1 到 k-1) 时的平均准确率
        
        最终 Transfer = mean(Transfer_2, Transfer_3, ..., Transfer_K)
        
        Returns:
            Transfer 分数 (百分比)
        """
        transfer_scores = []
        
        # k 从 1 到 K-1 (0-indexed)，对应论文的 k = 2 到 K (1-indexed)
        for k in range(1, self.K):
            # 收集所有前面步骤 j = 0 到 k-1，在任务 k 上的准确率
            past_accs = []
            for j in range(k):  # j from 0 to k-1
                acc = self.accuracy_matrix[j, k]
                if acc >= 0:
                    past_accs.append(acc)
            
            if len(past_accs) > 0:
                # Transfer_k = mean of a_hat^(j)_k for j = 1 to k-1
                transfer_k = np.mean(past_accs)
                transfer_scores.append(transfer_k)
        
        if len(transfer_scores) == 0:
            return 0.0
        
        # Transfer = mean of Transfer_k for k = 2 to K
        return np.mean(transfer_scores) * 100
    
    def calculate_average(self) -> float:
        """
        计算 Average 指标
        
        根据论文公式 (13):
        Average_k = (1/K) * sum_{j=1}^K a_hat^(j)_k, for k = 1,2,...,K
        
        其中:
        - a_hat^(j)_k: 训练完任务 j 后，在任务 k 上的准确率
        - Average_k: 对于任务 k，在所有训练步骤 (1 到 K) 中的平均准确率
        
        最终 Average = mean(Average_1, Average_2, ..., Average_K)
        
        Returns:
            Average 分数 (百分比)
        """
        average_scores = []
        
        # k 从 0 到 K-1 (0-indexed)，对应论文的 k = 1 到 K (1-indexed)
        for k in range(self.K):
            # 收集所有步骤 j = 0 到 K-1，在任务 k 上的准确率
            all_accs = []
            for j in range(self.K):  # j from 0 to K-1
                acc = self.accuracy_matrix[j, k]
                if acc >= 0:
                    all_accs.append(acc)
            
            if len(all_accs) > 0:
                # Average_k = mean of a_hat^(j)_k for j = 1 to K
                avg_k = np.mean(all_accs)
                average_scores.append(avg_k)
        
        if len(average_scores) == 0:
            return 0.0
        
        # Average = mean of Average_k for k = 1 to K
        return np.mean(average_scores) * 100
    
    def calculate_last(self) -> float:
        """
        计算 Last 指标
        
        根据论文公式 (14):
        Last_k = a_hat^(K)_k, for k = 1,2,...,K
        
        其中:
        - a_hat^(K)_k: 训练完所有任务（第 K 步）后，在任务 k 上的准确率
        - Last_k: 最终模型在任务 k 上的准确率
        
        最终 Last = mean(Last_1, Last_2, ..., Last_K)
        
        Returns:
            Last 分数 (百分比)
        """
        # 取最后一步（训练完所有任务后）在所有任务上的准确率
        last_accs = []
        for k in range(self.K):
            acc = self.accuracy_matrix[self.K - 1, k]
            if acc >= 0:
                last_accs.append(acc)
        
        if len(last_accs) == 0:
            return 0.0
        
        # Last = mean of Last_k for k = 1 to K
        return np.mean(last_accs) * 100
    
    def calculate_per_task_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        计算每个任务的详细指标
        
        Returns:
            {task_name: {'transfer': float, 'average': float, 'last': float}}
        """
        per_task = {}
        
        for k, task_name in enumerate(self.task_names):
            # Transfer_k: 对于任务 k，在学习前面任务时的平均准确率
            if k == 0:
                # 第一个任务没有 Transfer（没有前面的任务）
                transfer_acc = 0.0
            else:
                past_accs = [self.accuracy_matrix[j, k] for j in range(k) 
                            if self.accuracy_matrix[j, k] >= 0]
                transfer_acc = np.mean(past_accs) * 100 if past_accs else 0.0
            
            # Average_k: 对于任务 k，在所有训练步骤中的平均准确率
            all_accs = [self.accuracy_matrix[j, k] for j in range(self.K)
                       if self.accuracy_matrix[j, k] >= 0]
            avg_acc = np.mean(all_accs) * 100 if all_accs else 0.0
            
            # Last_k: 最终模型在任务 k 上的准确率
            last_acc = self.accuracy_matrix[self.K - 1, k] * 100 if self.accuracy_matrix[self.K - 1, k] >= 0 else 0.0
            
            per_task[task_name] = {
                'transfer': transfer_acc,
                'average': avg_acc,
                'last': last_acc
            }
        
        return per_task
    
    def get_accuracy_matrix(self) -> np.ndarray:
        """获取准确率矩阵"""
        return self.accuracy_matrix.copy()
    
    def get_summary(self) -> Dict[str, float]:
        """
        获取评估摘要
        
        Returns:
            {'transfer': float, 'average': float, 'last': float}
        """
        summary = {
            'transfer': self.calculate_transfer(),
            'average': self.calculate_average(),
            'last': self.calculate_last()
        }
        return summary
    
    def print_summary(self):
        """打印评估摘要"""
        summary = self.get_summary()
        per_task = self.calculate_per_task_metrics()
        
        print("\n" + "="*100)
        print("Continual Learning Evaluation Summary (LADA Metrics)")
        print("="*100)
        
        # 打印准确率矩阵
        print("\nAccuracy Matrix â^(j)_k (%):")
        print("-" * 100)
        print("After task j | " + " | ".join([f"{t[:8]:>8}" for t in self.task_names]))
        print("-" * 100)
        
        for i in range(self.K):
            row_name = f"Task {i+1:2d} ({self.task_names[i][:6]:>6})"
            row_values = []
            for j in range(self.K):
                if self.accuracy_matrix[i, j] >= 0:
                    row_values.append(f"{self.accuracy_matrix[i, j]*100:7.1f}")
                else:
                    row_values.append("    -   ")
            print(f"{row_name} | " + " | ".join(row_values))
        
        print("-" * 100)
        print("Note: Row j = after training task j, Col k = accuracy on task k")
        print("      â^(j)_k = accuracy on task k after training task j")
        
        # 打印每个任务的指标
        print("\nPer-Task Metrics (%):")
        print("-" * 70)
        print(f"{'Task':<15} | {'Transfer':>10} | {'Average':>10} | {'Last':>10} | Note")
        print("-" * 70)
        for task_name, metrics in per_task.items():
            note = ""
            if metrics['transfer'] == 0:
                note = "(First task, no Transfer)"
            print(f"{task_name:<15} | {metrics['transfer']:>10.1f} | {metrics['average']:>10.1f} | {metrics['last']:>10.1f} | {note}")
        print("-" * 70)
        
        # 打印总体指标
        print("\nOverall Metrics (LADA):")
        print("-" * 50)
        print(f"Transfer: {summary['transfer']:6.1f}%  [mean of Transfer_k for k=2..K]")
        print(f"Average:  {summary['average']:6.1f}%  [mean of Average_k for k=1..K]")
        print(f"Last:     {summary['last']:6.1f}%  [mean of Last_k for k=1..K]")
        print("="*100)
        
        # 解释
        print("\nMetric Definitions:")
        print("  Transfer: Average accuracy on each task BEFORE it was trained")
        print("            (measures forward transfer / zero-shot retention)")
        print("  Average:  Average accuracy on each task across ALL training steps")
        print("            (measures overall stability and plasticity)")
        print("  Last:     Final accuracy on each task after ALL training")
        print("            (measures backward forgetting)")
    
    def save(self, filepath: str):
        """保存评估结果到文件"""
        results = {
            'task_names': self.task_names,
            'accuracy_matrix': self.accuracy_matrix.tolist(),
            'summary': self.get_summary(),
            'per_task_metrics': self.calculate_per_task_metrics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {filepath}")


def calculate_forgetting(accuracy_matrix: np.ndarray) -> float:
    """
    计算遗忘率 (Forgetting Rate)
    
    Forgetting = (1/(K-1)) * sum_{k=1}^{K-1} (max_{l<k} acc_l,k - acc_K,k)
    
    其中:
    - max_{l<k} acc_l,k: 任务 k 在学习过程中的最高准确率
    - acc_K,k: 最终模型在任务 k 上的准确率
    
    Args:
        accuracy_matrix: K×K 准确率矩阵
        
    Returns:
        遗忘率 (百分比)
    """
    K = accuracy_matrix.shape[0]
    forgetting_scores = []
    
    for k in range(1, K):  # 对于每个任务 k (除了第一个)
        # 找到在学习过程中对任务 k 的最高准确率
        accs_for_task_k = [accuracy_matrix[l, k] for l in range(k) if accuracy_matrix[l, k] >= 0]
        if len(accs_for_task_k) == 0:
            continue
        max_acc = np.max(accs_for_task_k)
        
        # 最终准确率
        final_acc = accuracy_matrix[K-1, k]
        
        # 遗忘量
        if final_acc >= 0:
            forgetting_scores.append(max_acc - final_acc)
    
    if len(forgetting_scores) == 0:
        return 0.0
    
    return np.mean(forgetting_scores) * 100


def calculate_forward_transfer(accuracy_matrix: np.ndarray) -> float:
    """
    计算前向迁移 (Forward Transfer)
    
    衡量学习前面任务对后续任务的提升
    
    Args:
        accuracy_matrix: K×K 准确率矩阵
        
    Returns:
        前向迁移率 (百分比)
    """
    K = accuracy_matrix.shape[0]
    if K < 2:
        return 0.0
    
    # 基准：第一个任务的初始性能
    baseline = accuracy_matrix[0, 0]
    
    # 后续任务在首次评估时（即学习前面任务后）的性能提升
    improvements = []
    for k in range(1, K):
        first_eval = accuracy_matrix[k-1, k]  # 训练完任务 k-1 后，在任务 k 上的准确率
        if first_eval >= 0:
            improvements.append(first_eval - baseline)
    
    return np.mean(improvements) * 100 if improvements else 0.0

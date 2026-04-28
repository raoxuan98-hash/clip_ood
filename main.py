import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import random
import argparse
import numpy as np

# [修改点] 移除了 ClassifierBasedOODDetector 和 AdaptiveRouter 的导入，仅保留集成分类器
from src.trainers.lora_nsp_trainer import LoRANSPTrainer
from src.classifiers.lr_rgda_classifier import LRRGDAClassifier
from src.utils.feature_extractor import extract_features
from src.utils.reference_loader import load_reference_dataset
from utils_data import get_xtail_trainloader, get_transforms

def parse_args():
    parser = argparse.ArgumentParser(description="CLIP Zero-shot Classification Continual Learning")
    
    # 数据集相关参数
    parser.add_argument("--id_datasets", type=list, default=["aircraft", "caltech101", "dtd", "eurosat", "flowers", "food101", "mnist", "oxford_pets", "stanford_cars", "sun397"], help="List of 10 ID datasets.")
    parser.add_argument("--ood_datasets", type=list, default=["dtd", "eurosat", "mnist", "sun397"], help="List of OOD datasets for evaluation.")
    parser.add_argument("--root", type=str, default="/data1/open_datasets/X-TAIL", help="Root directory of the dataset.")
    parser.add_argument("--num_shots", type=int, default=16, help="Number of shots for few-shot learning.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and testing.")
    parser.add_argument("--max_num_per_test_dataset", type=int, default=1000, help="Maximum number of samples per test dataset.")

    # 训练基础参数
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training.")
    parser.add_argument("--iterations", type=int, default=800, help="Number of training iterations.")
    
    # 优化器参数
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=3e-5, help="Weight decay for optimizer.")
    
    # LoRA相关参数
    parser.add_argument("--lora_rank", type=int, default=4, help="Rank for LoRA adaptation.")
    parser.add_argument("--lora_type", type=str, default="lora_nsp", choices=["lora_sgp", "lora_nsp"], help="Type of LoRA adaptation.")
    parser.add_argument("--nsp_eps", type=float, default=0.05, help="Epsilon parameter for NSP.")
    parser.add_argument("--nsp_weight", type=float, default=0.02, help="Weight parameter for NSP.")
    parser.add_argument("--weight_temp", type=float, default=1.0, help="Temperature parameter for weight.")
    parser.add_argument("--weight_kind", type=str, default="log1p")
    parser.add_argument("--weight_p", type=float, default=1.0, help="P parameter for weight function.")

    # 参考数据集参数
    parser.add_argument("--reference_dataset", type=str, default="flickr8k", help="Reference dataset for training.")
    parser.add_argument("--reference_batch_size", type=int, default=32, help="Batch size for reference dataset.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")
    
    # 损失函数权重参数
    parser.add_argument("--fd_weight", type=float, default=1.0, help="Weight for feature distillation loss.")
    parser.add_argument("--cd_weight", type=float, default=1.0, help="Weight for cross-modal distillation loss.")
    
    # 分类器参数
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for LR-RGDA classifier in ensemble.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for zero-shot classifier.")
    
    # 增量学习模式 (False为联合微调，True为增量学习)
    parser.add_argument("--incremental_mode", type=bool, default=False, help="Whether to use incremental learning mode.")
    parser.add_argument("--dataset_sequence", type=list, default=[["aircraft"], ["caltech101"], ["dtd"],["eurosat"], ["flowers"], ["food101"], ["mnist"], ["oxford_pets"], ["stanford_cars"], ["sun397"]], help="Sequence of 10 tasks.")

    args = parser.parse_args()
    return args

def fix_random_seed(seed=42):
    """设置随机种子以确保结果可复现"""
    print(f"Setting fixed seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_zeroshot_classifier(model, processor, class_names, device):
    """构建零样本分类器"""
    templates = [lambda x: f"a photo of a {x}."]
    zeroshot_weights =[]
    with torch.no_grad():
        for classname in class_names:
            classname = classname.replace('_', ' ')
            texts = [template(classname) for template in templates]
            text_inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            class_embeddings = model.get_text_features(**text_inputs)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
    return torch.stack(zeroshot_weights, dim=1).to(device)


def evaluate_dataset(args, d_name, model, zeroshot_classifier, lr_rgda_classifier, current_num_classes, eval_label_offset):
    """ [新增修改点] 为了代码复用，将 debug 中的拼接评估逻辑抽象为一个函数 """
    _, test_transform = get_transforms(d_name)
    _, te_loader, _, c_names = get_xtail_trainloader(
        root=args.root, dataset_name=d_name, 
        transform_train=None, transform_test=test_transform,
        num_shots=args.num_shots, batch_size=args.batch_size
    )
    
    features, labels = extract_features(model, te_loader, args.device)
    
    # [修改点] 确保特征严格进行 L2 归一化
    features = features / features.norm(dim=-1, keepdim=True)
    features = features.to(args.device)
    labels = (labels + eval_label_offset).to(args.device)
    
    with torch.no_grad():
        logit_scale = model.logit_scale.exp().item()
        
        # 1. 纯 Zero-shot 预测
        zs_logits = logit_scale * (features @ zeroshot_classifier)
        zs_logits_norm = zs_logits - zs_logits.max(dim=-1, keepdim=True).values
        zs_preds = zs_logits_norm.argmax(dim=1)
        zs_acc = zs_preds.eq(labels).float().mean().item() * 100
        
        # 2. 纯 LR-RGDA 预测
        rgda_logits = lr_rgda_classifier.forward(features)
        rgda_logits_norm = rgda_logits - rgda_logits.max(dim=-1, keepdim=True).values
        rgda_preds = rgda_logits_norm.argmax(dim=1)
        rgda_acc = rgda_preds.eq(labels).float().mean().item() * 100
        
        # 3. Ensemble 预测 (1-alpha)*ZS + alpha*RGDA
        ensemble_logits = zs_logits_norm * (1 - args.alpha)
        # 仅在 ID 类别范围内叠加 RGDA 贡献
        ensemble_logits[:, :current_num_classes] += args.alpha * rgda_logits_norm
        ens_preds = ensemble_logits.argmax(dim=1)
        ens_acc = ens_preds.eq(labels).float().mean().item() * 100
        
    return zs_acc, rgda_acc, ens_acc, len(c_names)


def main(args):
    """主程序"""
    if args.seed is not None:
        fix_random_seed(args.seed)

    trainer = LoRANSPTrainer(args)
    model = trainer.model
    processor = trainer.processor

    reference_loader = load_reference_dataset(args, trainer.model_pretrain, processor, args.device)

    if not args.incremental_mode:
        # =======================================================
        # 模式一：联合微调 (Joint Fine-tuning) 
        # =======================================================
        print("\n=== Starting Joint Fine-tuning ===")
        
        train_loaders = []
        all_class_names =[]
        
        # 合并所有训练数据
        for d_name in args.id_datasets:
            train_transform, test_transform = get_transforms(d_name)
            tr_loader, _, _, c_names = get_xtail_trainloader(
                root=args.root, dataset_name=d_name, 
                transform_train=train_transform, transform_test=test_transform,
                num_shots=args.num_shots, batch_size=args.batch_size
            )
            train_loaders.append(tr_loader)
            all_class_names.extend(c_names)
        
        from torch.utils.data import ConcatDataset, DataLoader
        merged_dataset = ConcatDataset([loader.dataset for loader in train_loaders])
        merged_loader = DataLoader(merged_dataset, batch_size=args.batch_size, shuffle=True)
        
        # 训练模型
        model = trainer.train(merged_loader, all_class_names, reference_loader)
        
        # 提取特征用于训练分类器
        print("\n=== Extracting Features ===")
        all_features =[]
        all_labels =[]
        label_offset = 0
        
        for i, d_name in enumerate(args.id_datasets):
            train_transform, test_transform = get_transforms(d_name)
            tr_loader, _, _, c_names = get_xtail_trainloader(
                root=args.root, dataset_name=d_name, 
                transform_train=train_transform, transform_test=test_transform,
                num_shots=args.num_shots, batch_size=args.batch_size
            )
            
            features, labels = extract_features(model, tr_loader, args.device)
            # [修改点] 提取 LR-RGDA 统计量之前必须经过 L2 归一化
            features = features / features.norm(dim=-1, keepdim=True)
            all_features.append(features)
            all_labels.append(labels + label_offset)
            label_offset += len(c_names)
        
        all_features = torch.cat(all_features)
        all_labels = torch.cat(all_labels)
        
        # 构建类别统计分布字典
        print("\n=== Building Stats Dict ===")
        from src.detectors.ood_detector import build_stats_dict_from_features
        stats_dict = build_stats_dict_from_features(all_features, all_labels)
        
        # 构建LR-RGDA分类器
        print("\n=== Building LR-RGDA Classifier ===")
        lr_rgda_classifier = LRRGDAClassifier(
            stats_dict=stats_dict,
            device=args.device,
            rank=32,
            qda_reg_alpha1=0.6,
            qda_reg_alpha2=1.0,
            qda_reg_alpha3=0.5,
            temperature=1.0
        )
        
        print("\n=== Preparing Global Class Names ===")
        # 获取 OOD 类别名称 (补全 CLIP 的知识库)
        ood_class_names =[]
        for d_name in args.ood_datasets:
            _, test_transform = get_transforms(d_name)
            #[修改点] 将 num_shots=1 和 batch_size=1 改为 args.num_shots 和 args.batch_size
            _, _, _, c_names = get_xtail_trainloader(
                root=args.root, dataset_name=d_name, 
                transform_train=None, transform_test=test_transform,
                num_shots=args.num_shots, batch_size=args.batch_size
            )
            ood_class_names.extend(c_names)
        
        num_id_classes = len(all_class_names) # 记录 ID 类别的数量 
        global_class_names = all_class_names + ood_class_names # ID + OOD 全局名单

        print("\n=== Building Zero-shot Classifier ===")
        zeroshot_classifier = get_zeroshot_classifier(model, processor, global_class_names, args.device)
        
        #[修改点] 删除了 OOD 检测器和 Router 的实例化部分，直接进行评估
        print("\n=== Evaluating (Joint Fine-tuning) ===")
        
        eval_offset = 0
        print("\n[ID Datasets Evaluation]")
        for d_name in args.id_datasets:
            zs_acc, rgda_acc, ens_acc, c_len = evaluate_dataset(
                args, d_name, model, zeroshot_classifier, lr_rgda_classifier, num_id_classes, eval_offset
            )
            eval_offset += c_len
            print(f"{d_name:<15s} -> ZS: {zs_acc:5.1f}% | RGDA: {rgda_acc:5.1f}% | Ens: {ens_acc:5.1f}%")

        print("\n[OOD Datasets Evaluation]")
        for d_name in args.ood_datasets:
            zs_acc, rgda_acc, ens_acc, c_len = evaluate_dataset(
                args, d_name, model, zeroshot_classifier, lr_rgda_classifier, num_id_classes, eval_offset
            )
            eval_offset += c_len
            # OOD 数据集中 RGDA 准确率通常很低或为0，主要是看 ZS 和 Ens 的表现
            print(f"{d_name:<15s} -> ZS: {zs_acc:5.1f}% | RGDA: {rgda_acc:5.1f}% | Ens: {ens_acc:5.1f}%")
        
        # [修改点] 删除了计算 AUROC, FPR@95TPR 等 OOD 指标的代码
        print("\n=== Joint Fine-tuning Evaluation Completed ===")

    else:
        # =======================================================
        # 模式二：增量学习 (Incremental Learning)
        # =======================================================
        print("\n=== Starting Incremental Learning ===")
        
        history_class_names =[]
        global_stats_dict = {} 
        
        acc_matrix_zs = []
        acc_matrix_rgda =[]
        acc_matrix_ens =[]
        
        for i, task_datasets in enumerate(args.dataset_sequence):
            print(f"\n" + "="*50)
            print(f"=== Task {i+1}: {task_datasets} ===")
            print("="*50)
            
            # --- 1. 准备训练数据与训练 ---
            train_loaders =[]
            task_class_names =[]
            for d_name in task_datasets:
                train_transform, test_transform = get_transforms(d_name)
                tr_loader, _, _, c_names = get_xtail_trainloader(
                    root=args.root, dataset_name=d_name, 
                    transform_train=train_transform, transform_test=test_transform,
                    num_shots=args.num_shots, batch_size=args.batch_size
                )
                train_loaders.append(tr_loader)
                task_class_names.extend(c_names)
            
            from torch.utils.data import ConcatDataset, DataLoader
            merged_dataset = ConcatDataset([loader.dataset for loader in train_loaders])
            # 用于提取协方差的 loader 不打乱顺序
            cov_loader = DataLoader(merged_dataset, batch_size=args.batch_size, shuffle=False)
            merged_loader = DataLoader(merged_dataset, batch_size=args.batch_size, shuffle=True)
            
            # 训练模型 (LoRA-NSP)
            model = trainer.train(merged_loader, task_class_names, reference_loader)
            
            # -------------------------------------------------------------
            #[修改点: ] 引入零空间投影(NSP)，确保抗遗忘机制生效
            # -------------------------------------------------------------
            print("\n=== Applying Null-Space Projection (NSP) ===")
            covariances = trainer.extract_layer_covariances(cov_loader)
            trainer.update_covariance_history(covariances)
            trainer.finalize_task_for_incremental()
            # -------------------------------------------------------------
            
            # --- 2. 提取特征并构建统计字典 ---
            task_features =[]
            task_labels =[]
            label_offset = sum(len(c_names) for c_names in history_class_names)
            
            for d_name in task_datasets:
                train_transform, test_transform = get_transforms(d_name)
                tr_loader, _, _, c_names = get_xtail_trainloader(
                    root=args.root, dataset_name=d_name, 
                    transform_train=train_transform, transform_test=test_transform,
                    num_shots=args.num_shots, batch_size=args.batch_size
                )
                features, labels = extract_features(model, tr_loader, args.device)
                
                #[修改点] 确保特征进行 L2 归一化
                features = features / features.norm(dim=-1, keepdim=True)
                task_features.append(features)
                task_labels.append(labels + label_offset)
            
            task_features = torch.cat(task_features)
            task_labels = torch.cat(task_labels)
            
            from src.detectors.ood_detector import build_stats_dict_from_features
            task_stats_dict = build_stats_dict_from_features(task_features, task_labels)
            
            # 累加统计字典
            global_stats_dict.update(task_stats_dict)
            history_class_names.append(task_class_names)
            
            # --- 3. 构建所有的分类器 ---
            lr_rgda_classifier = LRRGDAClassifier(
                stats_dict=global_stats_dict,
                device=args.device,
                rank=32,
                qda_reg_alpha1=0.6,
                qda_reg_alpha2=1.0,
                qda_reg_alpha3=0.5,
                temperature=1.0
            )
            
            flat_class_names =[name for sublist in history_class_names for name in sublist]
            current_num_classes = len(flat_class_names)
            zeroshot_classifier = get_zeroshot_classifier(model, processor, flat_class_names, args.device)
            
            # --- 4. 自动评估并记录成绩 ---
            print("\n=== Evaluating Task ===")
            step_accs_zs, step_accs_rgda, step_accs_ens = [], [],[]
            
            # 遍历所有已经学过的任务进行考试
            eval_label_offset = 0
            for j in range(i + 1):
                eval_datasets = args.dataset_sequence[j]
                # 这里假设每个 Task 只有一个数据集，如果有多个可以进行拓展
                d_name = eval_datasets[0] 
                
                #[修改点] 使用evaluate_dataset
                zs_acc, rgda_acc, ens_acc, c_len = evaluate_dataset(
                    args, d_name, model, zeroshot_classifier, lr_rgda_classifier, current_num_classes, eval_label_offset
                )
                eval_label_offset += c_len
                
                print(f"[Tested on Task {j+1}: {d_name:<10s}] -> Zero-shot: {zs_acc:5.1f}% | LR-RGDA: {rgda_acc:5.1f}% | Ensemble: {ens_acc:5.1f}%")
                
                step_accs_zs.append(zs_acc)
                step_accs_rgda.append(rgda_acc)
                step_accs_ens.append(ens_acc)
                
            acc_matrix_zs.append(step_accs_zs)
            acc_matrix_rgda.append(step_accs_rgda)
            acc_matrix_ens.append(step_accs_ens)

        print("\n=== Training and Evaluation Completed ===")
        print("\n" + "="*80)
        print("Final Results Mapping to Paper Tables")
        print("="*80)
        
        # =======================================================
        # [修改点] 1. 提取数据集名称作为表头
        # =======================================================
        task_names = [d[0] for d in args.dataset_sequence]

        # [修改点] 2. 增强版的打印函数：带表头，更美观
        def print_paper_metrics(matrix, name, headers):
            num_tasks = len(matrix)
            transfers = [matrix[k][k] for k in range(num_tasks)]
            lasts = matrix[-1]
            averages = [sum(matrix[i][j] for i in range(j, num_tasks)) / (num_tasks - j) for j in range(num_tasks)]
            
            print(f"\n" + "-"*100)
            print(f"[{name} 分类器指标报告]")
            print(f"数据集顺序: {' | '.join(headers)}")
            print(f"指标类型   | " + "  |  ".join([f"{h[:8]:<8}" for h in headers]) + " | [平均总分]")
            print("-" * 100)
            
            print(f"Transfer  | " + "  |  ".join([f"{x:8.1f}" for x in transfers]) + f" | [{sum(transfers)/num_tasks:.1f}]")
            print(f"Average   | " + "  |  ".join([f"{x:8.1f}" for x in averages])  + f" | [{sum(averages)/num_tasks:.1f}]")
            print(f"Last      | " + "  |  ".join([f"{x:8.1f}" for x in lasts])     + f" | [{sum(lasts)/num_tasks:.1f}]")
            print("-" * 100)

        # 打印三组结果
        print_paper_metrics(acc_matrix_zs, "Zero-shot Baseline", task_names)
        print_paper_metrics(acc_matrix_rgda, "LR-RGDA Only", task_names)
        print_paper_metrics(acc_matrix_ens, f"Ours Ensemble (alpha={args.alpha})", task_names)

        # =======================================================
        # [修改点] 3. 增强版的 JSON 保存逻辑：包含表头和所有平均值
        # =======================================================
        import json
        from datetime import datetime

        def get_full_stats(matrix):
            num_tasks = len(matrix)
            trans = [matrix[k][k] for k in range(num_tasks)]
            lasts = matrix[-1]
            avgs = [sum(matrix[i][j] for i in range(j, num_tasks)) / (num_tasks - j) for j in range(num_tasks)]
            return {
                "raw_matrix": matrix,
                "transfer": trans,
                "transfer_total_avg": sum(trans) / num_tasks,
                "average_per_task": avgs,
                "average_total_avg": sum(avgs) / num_tasks,
                "last": lasts,
                "last_total_avg": sum(lasts) / num_tasks
            }

        save_results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_order": task_names, # 保存数据集顺序，防止以后对不上
            "arguments": vars(args),
            "metrics": {
                "zero_shot": get_full_stats(acc_matrix_zs),
                "lr_rgda": get_full_stats(acc_matrix_rgda),
                "ours_ensemble": get_full_stats(acc_matrix_ens)
            }
        }

        save_dir = "experiments"
        os.makedirs(save_dir, exist_ok=True)
        file_name = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_path = os.path.join(save_dir, file_name)

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(save_results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    command_line_args = parse_args()
    main(command_line_args)
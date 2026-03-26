#!/usr/bin/env python3
"""
提取并缓存预训练 CLIP 的特征
用于加速推理端实验 (Table 3/4/5)
"""

import os
import sys
import argparse
import torch
import pickle
import numpy as np
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformers import CLIPModel, CLIPProcessor
from utils_data import get_xtail_trainloader, get_xtail_testloader, get_transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Extract and cache CLIP features")
    
    parser.add_argument("--datasets", type=str, nargs='+',
                       default=["aircraft", "caltech101", "dtd", "eurosat", "flowers", 
                               "food101", "mnist", "oxford_pets", "stanford_cars", "sun397"],
                       help="List of datasets to extract features from")
    
    parser.add_argument("--root", type=str,
                       default="/home/raoxuan/projects/data/X-TAIL/",
                       help="X-TAIL dataset root directory")
    
    parser.add_argument("--cache_dir", type=str,
                       default="cache/pretrained_features",
                       help="Directory to save cached features")
    
    parser.add_argument("--model_name", type=str,
                       default="openai/clip-vit-base-patch16",
                       help="CLIP model name")
    
    parser.add_argument("--device", type=str,
                       default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use")
    
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for feature extraction")
    
    parser.add_argument("--num_shots", type=int, default=16,
                       help="Number of shots for training data")
    
    parser.add_argument("--max_test_samples", type=int, default=1000,
                       help="Maximum number of test samples per dataset")
    
    return parser.parse_args()


def extract_features(model, processor, dataloader, device):
    """从数据加载器中提取特征"""
    model.eval()
    all_features = []
    all_labels = []
    all_images = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            if len(batch) == 2:
                images, labels = batch
            else:
                images, labels = batch[0], batch[1]
            
            # 处理图像
            if hasattr(images, 'numpy'):
                images_np = images.numpy()
            else:
                images_np = images
            
            # CLIP处理
            if isinstance(images_np, np.ndarray):
                # 转换为PIL图像或直接使用tensor
                if images_np.ndim == 4 and images_np.shape[1] == 3:  # BCHW format
                    # 已经是tensor格式，直接使用
                    images_tensor = torch.from_numpy(images_np).float().to(device)
                else:
                    images_tensor = torch.from_numpy(images_np).float().to(device)
            else:
                images_tensor = images.to(device)
            
            # 提取特征
            try:
                # 尝试直接通过processor处理
                if images_tensor.dim() == 4:
                    features = model.get_image_features(pixel_values=images_tensor)
                else:
                    # 可能需要预处理
                    features = model.get_image_features(pixel_values=images_tensor.unsqueeze(0))
            except Exception as e:
                # 回退到手动处理
                try:
                    # 使用vision_model
                    if hasattr(model, 'vision_model'):
                        vision_outputs = model.vision_model(pixel_values=images_tensor)
                        features = vision_outputs.pooler_output if hasattr(vision_outputs, 'pooler_output') else vision_outputs.last_hidden_state[:, 0, :]
                    else:
                        features = model.get_image_features(pixel_values=images_tensor)
                except:
                    # 最后尝试：通过encode_image
                    features = model.encode_image(images_tensor)
            
            # 归一化特征
            features = features / features.norm(dim=-1, keepdim=True)
            
            all_features.append(features.cpu())
            all_labels.append(labels if isinstance(labels, torch.Tensor) else torch.tensor(labels))
    
    return torch.cat(all_features), torch.cat(all_labels)


def extract_and_cache_features(args):
    """提取并缓存所有数据集的特征"""
    
    print("="*80)
    print("Extracting and Caching CLIP Features")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"Device: {args.device}")
    print(f"Datasets: {args.datasets}")
    print(f"Cache directory: {args.cache_dir}")
    print("="*80)
    
    # 创建缓存目录
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # 加载预训练 CLIP 模型
    print("\nLoading pre-trained CLIP model...")
    model = CLIPModel.from_pretrained(args.model_name).to(args.device)
    processor = CLIPProcessor.from_pretrained(args.model_name)
    model.eval()
    
    # 为每个数据集提取特征
    for dataset_name in args.datasets:
        print(f"\n{'='*80}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*80}")
        
        cache_file = os.path.join(args.cache_dir, f"{dataset_name}_features.pkl")
        
        # 检查是否已存在缓存
        if os.path.exists(cache_file):
            print(f"Cache already exists: {cache_file}")
            print("Skipping... (use --overwrite to re-extract)")
            continue
        
        try:
            # 获取数据变换
            train_transform, test_transform = get_transforms(dataset_name)
            
            # 提取训练集特征
            print("\nExtracting training features...")
            train_loader, _, _, class_names = get_xtail_trainloader(
                root=args.root,
                dataset_name=dataset_name,
                transform_train=train_transform,
                transform_test=test_transform,
                num_shots=args.num_shots,
                batch_size=args.batch_size
            )
            
            train_features, train_labels = extract_features(
                model, processor, train_loader, args.device
            )
            
            print(f"  Train features shape: {train_features.shape}")
            print(f"  Train labels shape: {train_labels.shape}")
            print(f"  Number of classes: {len(class_names)}")
            
            # 提取测试集特征
            print("\nExtracting test features...")
            test_loader, test_class_names, _ = get_xtail_testloader(
                root=args.root,
                dataset_sequence=[dataset_name],
                transform_test=test_transform,
                batch_size=args.batch_size,
                max_num_per_dataset=args.max_test_samples
            )
            
            test_features, test_labels = extract_features(
                model, processor, test_loader, args.device
            )
            
            print(f"  Test features shape: {test_features.shape}")
            print(f"  Test labels shape: {test_labels.shape}")
            
            # 保存缓存
            cache_data = {
                'dataset_name': dataset_name,
                'train_features': train_features,
                'train_labels': train_labels,
                'test_features': test_features,
                'test_labels': test_labels,
                'class_names': class_names,
                'model_name': args.model_name,
                'num_shots': args.num_shots
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            print(f"\n✅ Cached features saved to: {cache_file}")
            
            # 清理GPU缓存
            if args.device == "cuda":
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"\n❌ Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*80)
    print("Feature extraction completed!")
    print("="*80)


if __name__ == "__main__":
    args = parse_args()
    extract_and_cache_features(args)

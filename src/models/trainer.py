import logging
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler
from torch.utils.data import DataLoader
from models.clip import get_clip_model
from models.utils import feature_distillation_loss, EMASmooth, cross_modal_distillation_loss
from tqdm import tqdm
from utils_data import MergedReferenceDataset

class Trainer:
    """Manages training loop and reference data for SubspaceLoRA CLIP."""
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.templates = {
            lambda x: f"a photo of a {x}."
        }

        self.model, self.processor = get_clip_model(args, train_mode='lora')
        self.model.to(self.device)
        self.model_pretrain, _ = get_clip_model(args, train_mode="frozen")
        self.model_pretrain.to(self.device)

    def get_optimizer(self, params, args):
        optimizer = optim.AdamW(params, args.lr, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.iterations, eta_min=args.lr/3)
        return optimizer, scheduler
    
    def encode_text(self, model, text):
        text_inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        text_features = model.get_text_features(**text_inputs)
        return text_features
    
    def encode_image(self, model, img):
        return model.get_image_features(img)
    
    @torch.no_grad()
    def zeroshot_classifier(self, model, classnames: Iterable[str], templates: Iterable[Any],
    ) -> torch.Tensor:
        """Build a zeroshot classifier from CLIP text embedddings."""
        zeroshot_weights = []
        for classname in classnames:
            classname = classname.replace('_', ' ')
            texts = [template(classname) for template in templates]
            class_embeddings = self.encode_text(model, texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(self.device)
        return zeroshot_weights
    
    @torch.no_grad()
    def evaluate(self, test_loader, class_names, indices=None):
        zeroshot_weights = self.zeroshot_classifier(model=self.model, classnames=class_names, templates=self.templates)
        logit_scale = self.model.logit_scale.detach()

        total_samples = 0
        correct_predictions = 0
        
        # 如果提供了indices参数，则为每个数据集维护单独的计数器
        if indices is not None:
            num_datasets = len(indices)
            dataset_correct = [0] * num_datasets
            dataset_total = [0] * num_datasets

        for images, labels in tqdm(test_loader, desc="Evaluating..."):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            img_feats = self.encode_image(self.model, images)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            logits_per_image = logit_scale.exp() * (img_feats @ zeroshot_weights)
            predictions = logits_per_image.argmax(dim=-1)

            total_samples += labels.size(0)
            correct_predictions += (predictions == labels).sum().item()
            
            # 如果提供了indices参数，计算每个数据集的准确度
            if indices is not None:
                # 将标签和预测移到CPU以便处理
                labels_cpu = labels.cpu().numpy()
                predictions_cpu = predictions.cpu().numpy()
                
                # 为每个样本确定其所属的数据集
                for i, (label, pred) in enumerate(zip(labels_cpu, predictions_cpu)):
                    # 确定样本属于哪个数据集
                    dataset_id = 0
                    for j in range(len(indices) - 1):
                        if indices[j] <= label < indices[j + 1]:
                            dataset_id = j
                            break
                    else:
                        # 如果不在任何中间范围内，则属于最后一个数据集
                        dataset_id = len(indices) - 1
                    
                    dataset_total[dataset_id] += 1
                    if label == pred:
                        dataset_correct[dataset_id] += 1

        # 计算全局准确度
        accuracy = correct_predictions / total_samples
        
        # 如果提供了indices参数，返回包含每个数据集准确度的字典
        if indices is not None:
            dataset_accuracies = {}
            for i in range(num_datasets):
                if dataset_total[i] > 0:  # 避免除以零
                    dataset_accuracies[f"dataset_{i}"] = dataset_correct[i] / dataset_total[i]
                else:
                    dataset_accuracies[f"dataset_{i}"] = 0.0
            
            # 添加全局准确度
            dataset_accuracies["global"] = accuracy
            return dataset_accuracies
        
        # 如果没有提供indices参数，只返回全局准确度
        return accuracy
    
    def initialize_reference_loader(self, reference_dataset = None, shuffle = True):
        reference_loader = reference_dataset.return_loader(
            batch_size=self.args.reference_batch_size,
            shuffle=False,
            num_workers=self.args.num_workers)
        
        cached_ref_images = []
        cached_ref_texts = []
        cached_ref_teacher_img_feats = []
        cached_ref_teacher_text_feats = []

        with torch.no_grad():
            for ref_images, ref_texts in tqdm(reference_loader, desc="Caching reference data..."):
                ref_images = ref_images.to(self.device, non_blocking=True)
                teacher_img_feats = self.encode_image(self.model_pretrain, ref_images)
                teacher_img_feats = teacher_img_feats / teacher_img_feats.norm(dim=-1, keepdim=True)

                teacher_text_feats = self.encode_text(self.model_pretrain, ref_texts)
                teacher_text_feats = teacher_text_feats / teacher_text_feats.norm(dim=-1, keepdim=True)

                cached_ref_images.append(ref_images.cpu())
                cached_ref_texts = cached_ref_texts + ref_texts
                cached_ref_teacher_img_feats.append(teacher_img_feats.cpu())
                cached_ref_teacher_text_feats.append(teacher_text_feats.cpu())

        cached_ref_images = torch.cat(cached_ref_images, dim=0)
        cached_ref_teacher_img_feats = torch.cat(cached_ref_teacher_img_feats, dim=0)
        cached_ref_teacher_text_feats = torch.cat(cached_ref_teacher_text_feats, dim=0)
        merged_reference_dataset = MergedReferenceDataset(cached_ref_images, cached_ref_texts, cached_ref_teacher_img_feats, cached_ref_teacher_text_feats)
        
        merged_reference_loader = DataLoader(
            merged_reference_dataset,
            batch_size=self.args.reference_batch_size,
            shuffle=shuffle,
            num_workers=self.args.num_workers,
            pin_memory=True)
        
        return merged_reference_loader
        
    def train(self, train_loader, class_names, reference_loader=None):
        zeroshot_weights = self.zeroshot_classifier(model=self.model, classnames=class_names, templates=self.templates)
        optimizer, scheduler = self.get_optimizer(params=self.model.vision_model.get_params(), args=self.args)
        logit_scale = self.model.logit_scale.detach()

        ema_acc, ema_loss, ema_fd_loss, ema_cd_loss = EMASmooth(alpha=0.95), EMASmooth(alpha=0.95), EMASmooth(alpha=0.95), EMASmooth(alpha=0.95)
        ema_cos = EMASmooth(alpha=0.95)
        
        # 将 reference_loader 转换为循环迭代器
        ref_iter = None
        if reference_loader is not None:
            ref_iter = iter(reference_loader)

        current_iteration = 0
        while current_iteration < self.args.iterations:
            for images, labels in train_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                img_feats = self.encode_image(self.model, images)
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                logits_per_image = logit_scale.exp() * (img_feats @ zeroshot_weights)
                ce_loss = F.cross_entropy(logits_per_image, labels, label_smoothing=0.1)

                if reference_loader is not None and ref_iter is not None:
                    try:
                        ref_images, ref_texts, cached_teacher_img_feats, cached_teacher_text_feats = next(ref_iter)
                    except StopIteration:
                        # 迭代器耗尽，重新创建
                        ref_iter = iter(reference_loader)
                        ref_images, ref_texts, cached_teacher_img_feats, cached_teacher_text_feats = next(ref_iter)
                    
                    ref_images = ref_images.to(self.device, non_blocking=True)
                    cached_teacher_img_feats = cached_teacher_img_feats.to(self.device, non_blocking=True)
                    cached_teacher_text_feats = cached_teacher_text_feats.to(self.device, non_blocking=True)


                    # 在trainer.py中修复
                    ref_student_img_feats = self.encode_image(self.model, ref_images)
                    ref_student_img_feats = ref_student_img_feats / ref_student_img_feats.norm(dim=-1, keepdim=True)
                    ref_student_text_feats = cached_teacher_text_feats

                    # ref_teacher_img_feats = self.encode_image(self.model_pretrain, ref_images)
                    # ref_teacher_img_feats = ref_teacher_img_feats / ref_teacher_img_feats.norm(dim=-1, keepdim=True)
                    
                    ref_teacher_img_feats = cached_teacher_img_feats
                    ref_teacher_text_feats = cached_teacher_text_feats

                    loss_fd = feature_distillation_loss(
                        teacher_feat=ref_teacher_img_feats,
                        student_feat=ref_student_img_feats)
                    
                    loss_cd = cross_modal_distillation_loss(
                        logit_scale=logit_scale,
                        student_img_feat=ref_student_img_feats,
                        student_text_feat=ref_student_text_feats,
                        teacher_img_feat=ref_teacher_img_feats,
                        teacher_text_feat=ref_teacher_text_feats,
                        temperature=2.0)
                    
                    loss = ce_loss + self.args.fd_weight * loss_fd + self.args.cd_weight * loss_cd
                
                else:
                    loss = ce_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                current_iteration += 1

                ema_loss.update(ce_loss.item())
                ema_acc.update((logits_per_image.argmax(dim=-1) == labels).float().mean().item())
                
                if ref_iter is not None:
                    cos = F.cosine_similarity(ref_student_img_feats, cached_teacher_img_feats.to(self.device, non_blocking=True), dim=-1).mean()
                    ema_cos.update(cos.item())
                    ema_fd_loss.update(loss_fd.item())
                    ema_cd_loss.update(loss_cd.item())

                if current_iteration % 10 == 0:
                    # logging.info(f"Iteration {current_iteration}, Train_loss: {ema_loss.get():.4f}, Train_acc: {ema_acc.get():.4f}, FD_loss: {ema_fd_loss.get():.4f}, CD_loss: {ema_cd_loss.get():.4f}")
                    print(f"Iteration {current_iteration}, Train_loss: {ema_loss.get():.4f}, Train_acc: {ema_acc.get():.4f}, FD_loss: {ema_fd_loss.get():.4f}, CD_loss: {ema_cd_loss.get():.4f}, Image Cosine Sim: {ema_cos.get():.4f}")
                if current_iteration >= self.args.iterations:
                    break
                scheduler.step()
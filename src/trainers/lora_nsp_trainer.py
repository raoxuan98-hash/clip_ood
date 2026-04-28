import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from typing import Dict, Optional
import logging

#[修改点: 配置全局 logging，设置日志显示的格式，包含时间、级别和具体信息]
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from models.clip import get_clip_model
from models.utils import feature_distillation_loss, cross_modal_distillation_loss


class FeatureExtractorHook:
    """用于捕获中间层特征的Hook"""
    def __init__(self):
        self.features =[]
    
    def __call__(self, module, input, output):
            # 捕获输入特征 (input是tuple，取第一个)
            if isinstance(input, tuple):
                x = input[0]
            else:
                x = input
            # [修改点] 立即将特征转到 CPU，释放显存
            self.features.append(x.detach().cpu())
    
    def clear(self):
        self.features =[]
    
    def get_features(self):
        # 合并所有batch的特征
        if len(self.features) == 0:
            return None
        return torch.cat(self.features, dim=0)


class LoRANSPTrainer:
    def __init__(self, args, covariance_history: Optional[Dict[str, torch.Tensor]] = None):
        self.args = args
        self.device = args.device
        self.model, self.processor = get_clip_model(args, train_mode='lora')
        self.model.to(self.device)
        
        # 预训练模型（用于蒸馏）
        self.model_pretrain, _ = get_clip_model(args, train_mode="frozen")
        self.model_pretrain.to(self.device)
        
        # 协方差历史（用于零空间约束）
        self.covariance_history = covariance_history or {}
        self.cov_momentum = getattr(args, 'cov_momentum', 0.9)
        
        # 如果提供了历史协方差，立即更新投影矩阵
        if self.covariance_history:
            # [修改点: print 改为 logging.info，下同]
            logging.info(f"Loading covariance history with {len(self.covariance_history)} layers")
            self.model.vision_model.update_projection_matrices(self.covariance_history)
    
    def encode_text(self, text):
        text_inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        return self.model.get_text_features(**text_inputs)
    
    def encode_image(self, img):
        return self.model.get_image_features(img)
    
    def zeroshot_classifier(self, classnames, templates):
        zeroshot_weights =[]
        with torch.no_grad():
            for classname in classnames:
                classname = classname.replace('_', ' ')
                texts = [template(classname) for template in templates]
                class_embeddings = self.encode_text(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
        return torch.stack(zeroshot_weights, dim=1).to(self.device)
    
    def get_optimizer(self, params, lr, weight_decay, iterations):
        optimizer = torch.optim.AdamW(params, lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=iterations, eta_min=lr/3)
        return optimizer, scheduler
    
    @torch.no_grad()
    def extract_layer_covariances(self, data_loader) -> Dict[str, torch.Tensor]:
        """
        提取所有LoRA层的非中心化协方差矩阵
        
        Returns:
            covariances: {layer_name: cov_matrix}
                cov_matrix = X^T X / N, shape [D, D]
        """
        logging.info("=== Extracting Layer Covariances ===")
        # [修改点] 手动清理显存碎片，为提取过程腾出空间
        torch.cuda.empty_cache() 
        self.model.eval()
        # 获取所有LoRA模块名称
        lora_module_names = self.model.vision_model.get_module_names()
        logging.info(f"Found {len(lora_module_names)} LoRA modules")
        
        # 注册hooks
        hooks = {}
        feature_extractors = {}
        
        for name in lora_module_names:
                    module = self.model.vision_model.lora_modules[name]
                    extractor = FeatureExtractorHook()
                    # [修改点: 删掉了 .linear] 直接在 LoRA 模块上注册钩子，
                    # 这样可以兼容 SGPBaseDoRA 等不同的底层实现
                    hook = module.register_forward_hook(extractor)
                    hooks[name] = hook
                    feature_extractors[name] = extractor
        
        # 前向传播收集特征
        total_samples = 0
        for images, _ in tqdm(data_loader, desc="Collecting features"):
            images = images.to(self.device)
            _ = self.encode_image(images)  # 触发hooks
            total_samples += images.size(0)
        
        # 计算协方差``
        covariances = {}
        for name in lora_module_names:
            features = feature_extractors[name].get_features()  
            if features is not None:
                # [修改] 强制转为 float32 避开 float16 的溢出风险
                features = features.to(torch.float32)
                
                if features.dim() == 3:
                    features = features.reshape(-1, features.shape[-1])
                
                num_observations = features.shape[0]
                
                # [修改 ] 在 float32 下计算矩阵乘法，并进行归一化
                cov = (features.t() @ features) / num_observations
                
                # [修改 ] 数值平滑：微量抖动防止奇异矩阵，并确保完全对称
                # 加上一个极小的 epsilon 保证 eigh 计算的稳定性
                eps = 1e-6
                cov = (cov + cov.t()) / 2.0  # 确保绝对对称
                cov = cov + torch.eye(cov.shape[0], device=cov.device) * eps
                
                # 检查是否存在非法数值
                if torch.isnan(cov).any() or torch.isinf(cov).any():
                    logging.warning(f"Layer {name} contains NaN or Inf in covariance. Cleaning...")
                    cov = torch.nan_to_num(cov, nan=0.0, posinf=1.0, neginf=-1.0)

                covariances[name] = cov.to('cpu') 
            
            hooks[name].remove()
        
        logging.info(f"✓ Extracted covariances for {len(covariances)} layers")
        return covariances
    
    def update_covariance_history(self, new_covariances: Dict[str, torch.Tensor]):
        """
        使用滑动平均更新协方差历史，并更新投影矩阵
        
        Args:
            new_covariances: 新提取的协方差字典
        """
        logging.info(f"=== Updating Covariance History (momentum={self.cov_momentum}) ===")
        
        updated_layers =[]
        new_layers =[]
        
        for layer_name, new_cov in new_covariances.items():
            if layer_name in self.covariance_history:
                # 滑动平均: history = α * history + (1-α) * new
                old_cov = self.covariance_history[layer_name]
                merged_cov = self.cov_momentum * old_cov + (1 - self.cov_momentum) * new_cov
                self.covariance_history[layer_name] = merged_cov
                updated_layers.append(layer_name)
            else:
                # 第一层，直接保存（或可以选择不处理，根据用户要求）
                # 用户说"第一个不处理"，我理解为用户会在外部处理，这里仍然保存
                self.covariance_history[layer_name] = new_cov
                new_layers.append(layer_name)
        
        logging.info(f"  - Updated {len(updated_layers)} layers with sliding average")
        logging.info(f"  - Added {len(new_layers)} new layers")
        
        # 更新投影矩阵
        logging.info("=== Updating Projection Matrices ===")
        self.model.vision_model.update_projection_matrices(self.covariance_history)
        logging.info("✓ Projection matrices updated")
    
    def finalize_task_for_incremental(self) -> None:
        """
        增量学习：完成当前任务，准备下一个任务
        """
        logging.info("=== Finalizing Task for Incremental Learning ===")
        
        # [修改点] 增加防御性判断，防止函数名不匹配导致崩溃
        if hasattr(self.model.vision_model, 'merge_and_reset_for_incremental'):
            # 如果有封装好的顶层函数，直接调用
            self.model.vision_model.merge_and_reset_for_incremental()
        else:
            # [修改点] 如果没有顶层函数，手动遍历每个 LoRA 模块执行合并和重置
            logging.info("Top-level merge method not found. Performing manual merge on LoRA modules...")
            
            # 获取所有 LoRA 模块字典
            lora_modules = self.model.vision_model.lora_modules
            
            for name, module in lora_modules.items():
                # [修改点] 根据你刚才报错提供的列表，正确的函数名是 merge_lora_weights
                if hasattr(module, 'merge_lora_weights'):
                    module.merge_lora_weights()
                elif hasattr(module, 'merge_and_reinit'):
                    module.merge_and_reinit()
                elif hasattr(module, 'merge_and_reset'):
                    module.merge_and_reset()
                elif hasattr(module, 'merge'):
                    module.merge()
                else:
                    logging.error(f"Module {name} has no known merge method! Available: {dir(module)}")
                    raise AttributeError(f"LoRA module {name} lacks a merge/reset method.")

        logging.info(f"✓ Task finalized: LoRA weights merged to main weights and reset.")
    
    def train(self, train_loader, class_names, reference_loader):
        """
        训练模型
        
        注意：在增量学习场景中，应在训练前调用 update_covariance_history
        以应用历史零空间约束
        """
        # 预计算零样本分类器权重
        templates = [lambda x: f"a photo of a {x}."]
        classifier = self.zeroshot_classifier(class_names, templates)
        
        # 优化器
        optimizer, scheduler = self.get_optimizer(
            self.model.vision_model.get_params(), 
            self.args.lr, 
            self.args.weight_decay, 
            self.args.iterations
        )
        
        logit_scale = self.model.logit_scale.detach()
        
        # 训练循环
        self.model.train()
        train_iter = iter(train_loader)
        ref_iter = iter(reference_loader) if reference_loader is not None else None
        
        #[修改点: 将 tqdm 实例化为 pbar，方便后续动态更新进度条信息]
        pbar = tqdm(range(self.args.iterations), desc="Training")
        for i in pbar:
            # 获取批次
            try:
                images, labels = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                images, labels = next(train_iter)
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            img_feats = self.encode_image(images)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            
            logits = logit_scale.exp() * (img_feats @ classifier)
            ce_loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
            loss = ce_loss
            
            #[修改点: 动态计算当前 batch 的训练准确度]
            preds = logits.argmax(dim=-1)
            train_acc = (preds == labels).float().mean().item() * 100
            
            # [修改点: 初始化记录用的蒸馏损失，防止 reference_loader 为空时报错]
            l_fd_val, l_cd_val = 0.0, 0.0
            
            # 蒸馏损失
            if reference_loader is not None and ref_iter is not None:
                try:
                    r_imgs, _, t_img_f, t_txt_f = next(ref_iter)
                except StopIteration:
                    ref_iter = iter(reference_loader)
                    r_imgs, _, t_img_f, t_txt_f = next(ref_iter)
                
                r_imgs = r_imgs.to(self.device)
                t_img_f = t_img_f.to(self.device)
                t_txt_f = t_txt_f.to(self.device)
                
                s_img_f = self.encode_image(r_imgs)
                s_img_f = s_img_f / s_img_f.norm(dim=-1, keepdim=True)
                
                l_fd = feature_distillation_loss(t_img_f, s_img_f)
                l_cd = cross_modal_distillation_loss(logit_scale, s_img_f, t_txt_f, t_img_f, t_txt_f, 2.0)
                
                #[修改点: 提取 item() 用于日志记录]
                l_fd_val = l_fd.item()
                l_cd_val = l_cd.item()
                
                loss += self.args.fd_weight * l_fd + self.args.cd_weight * l_cd
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # [修改点: 在进度条后面动态滚动显示损失和准确率]
            pbar.set_postfix({
                'Loss': f"{loss.item():.3f}",
                'CE': f"{ce_loss.item():.3f}",
                'FD': f"{l_fd_val:.3f}",
                'CD': f"{l_cd_val:.3f}",
                'Acc': f"{train_acc:.1f}%"
            })
            
            #[修改点: 每 50 步或训练结束时，使用 logging 保存一次正式的文本日志]
            if (i + 1) % 50 == 0 or (i + 1) == self.args.iterations:
                logging.info(f"Iter[{i+1:03d}/{self.args.iterations}] | "
                             f"Loss: {loss.item():.4f} | CE: {ce_loss.item():.4f} | "
                             f"FD: {l_fd_val:.4f} | CD: {l_cd_val:.4f} | Acc: {train_acc:.2f}%")
        
        return self.model
    
    def evaluate(self, test_loader, class_names):
        self.model.eval()
        correct = 0
        total = 0
        
        templates =[lambda x: f"a photo of a {x}."]
        classifier = self.zeroshot_classifier(class_names, templates)
        logit_scale = self.model.logit_scale.detach()
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                img_feats = self.encode_image(images)
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                
                logits = logit_scale.exp() * (img_feats @ classifier)
                _, predicted = torch.max(logits, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        return accuracy
    
    def save_checkpoint(self, path, class_names, stats_dict=None):
        """保存训练器状态（包括covariance_history）"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'covariance_history': self.covariance_history,
            'cov_momentum': self.cov_momentum,
            'class_names': class_names,
            'stats_dict': stats_dict,  # 可选：保存类别统计分布
            'args': vars(self.args),
        }
        torch.save(checkpoint, path)
        logging.info(f"✓ Checkpoint saved: {path}") # [修改点: print改为了logging]
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path, args, device='cuda'):
        """从checkpoint恢复训练器"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 恢复covariance_history
        covariance_history = checkpoint.get('covariance_history', {})
        
        # 创建新的训练器实例
        trainer = cls(args, covariance_history=covariance_history)
        trainer.cov_momentum = checkpoint.get('cov_momentum', 0.9)
        
        # 加载模型权重
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        
        return trainer
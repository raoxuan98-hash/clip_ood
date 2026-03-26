import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class MergedReferenceDataset(Dataset):
    """
    合并的参考数据集，用于蒸馏
    """
    def __init__(self, images, texts, img_features, txt_features):
        self.images = images
        self.texts = texts
        self.img_features = img_features
        self.txt_features = txt_features
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.texts[idx], self.img_features[idx], self.txt_features[idx]

def load_reference_dataset(args, model_pretrain, processor, device):
    """
    加载并缓存参考数据集（Flickr8K）用于蒸馏
    Returns: DataLoader or None
    """
    if args.iterations == 0 or args.reference_dataset != "flickr8k":
        print("Skipping reference dataset loading.")
        return None
    
    try:
        # 加载 Flickr8k
        from utils_data import Flickr8kDataset
        ref_dataset_obj = Flickr8kDataset(root="/data1/open_datasets/flickr8k/")
        raw_ref_loader = ref_dataset_obj.return_loader(
            batch_size=32, shuffle=False, num_workers=4
        )
        
        # 缓存 Teacher 特征
        cached_imgs, cached_txts = [], []
        cached_t_img_feats, cached_t_txt_feats = [], []
        
        with torch.no_grad():
            for imgs, txts in tqdm(raw_ref_loader, desc="Caching Reference Data"):
                imgs = imgs.to(device)
                t_img_feat = model_pretrain.get_image_features(imgs)
                t_img_feat = t_img_feat / t_img_feat.norm(dim=-1, keepdim=True)
                
                # 编码文本
                text_inputs = processor(text=txts, return_tensors="pt", padding=True, truncation=True)
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                t_txt_feat = model_pretrain.get_text_features(**text_inputs)
                t_txt_feat = t_txt_feat / t_txt_feat.norm(dim=-1, keepdim=True)
                
                cached_imgs.append(imgs.cpu())
                cached_txts.extend(txts)
                cached_t_img_feats.append(t_img_feat.cpu())
                cached_t_txt_feats.append(t_txt_feat.cpu())
        
        merged_ref_dataset = MergedReferenceDataset(
            torch.cat(cached_imgs), cached_txts, 
            torch.cat(cached_t_img_feats), torch.cat(cached_t_txt_feats)
        )
        reference_loader = DataLoader(
            merged_ref_dataset, batch_size=32, shuffle=True, 
            num_workers=4, pin_memory=True
        )
        
        print(f"✓ Reference dataset loaded: {len(merged_ref_dataset)} samples")
        return reference_loader
        
    except Exception as e:
        print(f"Warning: Failed to load reference dataset ({e}). Distillation disabled.")
        return None
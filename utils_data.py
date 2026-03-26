# In[]
from torchvision import transforms
from scenario_datasets.build_functions import build_cur_task_data_loader, build_TAIL_testloader
import os
import csv
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset
import random
import torch
from torch.utils.data import DataLoader

# In[]
def get_transforms(dataset_name, resolution=224):
    """获取图像变换"""
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    print("mean:", mean)
    print("std:", std)

    if dataset_name in ["mnist", "eurosat"]:
        transform_train = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
    else:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop((resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    transform_test = transforms.Compose([
        transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(resolution),
        transforms.Lambda(lambda image: image.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    return transform_train, transform_test

def get_xtail_trainloader(root, dataset_name, transform_train, transform_test, num_shots=16, batch_size=64, num_workers=4):
    train_loader, train_loader4updating, test_loader, classnames = build_cur_task_data_loader(
        root=root,
        dataset_name=dataset_name, 
        transform_train=transform_train, 
        transform_test=transform_test, 
        num_shots=num_shots, 
        batch_size=batch_size,
        num_workers=num_workers)
    
    return train_loader, train_loader4updating, test_loader, classnames

def get_xtail_testloader(root, dataset_sequence, transform_test, batch_size=64, num_workers=4, max_num_per_dataset=None):
    """获取X-TAIL测试集数据加载器
    
    Args:
        root: 数据集根目录
        dataset_sequence: 数据集序列
        transform_test: 测试数据变换
        batch_size: 批次大小
        num_workers: 数据加载器工作进程数
        max_num_per_dataset: 每个数据集的最大样本数量，None表示不限制
    """
    test_loader, classnames, indices = build_TAIL_testloader(
        root=root,
        dataset_sequence=dataset_sequence,
        transform_test=transform_test,
        batch_size=batch_size,
        num_workers=num_workers,
        max_num_per_dataset=max_num_per_dataset,
    )
    return test_loader, classnames, indices

# In[]
class Flickr8kDataset(Dataset):
    """Flickr8K数据集，返回图像和对应的标题"""
    
    def __init__(self, root: str, transform=None, num_samples = None):
        self.root = root
        if transform is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
            std = (0.26862954, 0.26130258, 0.27577711)
            self.transform = transforms.Compose([
                    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)])
        else:
            self.transform = transform

        self.images_dir = os.path.join(root, "images")
        self.captions_file = os.path.join(root, "captions.txt")
        self.samples = []
        self.prompts_list = []
        
        self._load_dataset()
        
        # 随机采样
        if num_samples is not None and num_samples < len(self.samples):
            indices = random.sample(range(len(self.samples)), num_samples)
            self.samples = [self.samples[i] for i in indices]
            self.prompts_list = [self.prompts_list[i] for i in indices]
            print(f"随机采样了 {num_samples} 个样本")
    
    def _load_dataset(self):
        """加载图像和标题数据"""
        # 加载标题映射
        captions_map = defaultdict(list)
        with open(self.captions_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过标题行
            for row in reader:
                if len(row) >= 2:
                    img_name, caption = row[0].strip(), row[1].strip()
                    captions_map[img_name].append(caption)
        
        # 收集图像路径（只处理jpg文件）
        img_paths = []
        for dirpath, _, filenames in os.walk(self.images_dir):
            for fn in filenames:
                if fn.endswith('.jpg') and fn in captions_map:
                    img_paths.append(os.path.join(dirpath, fn))
        
        img_paths.sort()
        for idx, path in enumerate(img_paths):
            fname = os.path.basename(path)
            self.samples.append((path, idx))
            self.prompts_list.append(captions_map[fname])
        
        print(f"加载了 {len(self.samples)} 个Flickr8K样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """返回图像和对应的标题列表"""
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        prompt = random.choice(self.prompts_list[idx])
        return img, prompt
    
    def return_loader(self, batch_size=32, shuffle=False, num_workers=4):
        dataloader = DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
        return dataloader
    
    def get_captions(self, idx):
        """获取指定索引图像的所有标题"""
        return self.prompts_list[idx] if 0 <= idx < len(self.prompts_list) else []
    
class MergedReferenceDataset(torch.utils.data.Dataset):
    def __init__(self, cached_images, cached_texts, cached_img_feats, cached_text_feats):
        self.cached_images = cached_images
        self.cached_texts = cached_texts
        self.cached_img_feats = cached_img_feats
        self.cached_text_feats = cached_text_feats
    
    def __len__(self):
        return len(self.cached_images)
    
    def __getitem__(self, idx):
        ref_images = self.cached_images[idx]
        ref_texts = self.cached_texts[idx]
        cached_img_feat = self.cached_img_feats[idx]
        cached_text_feat = self.cached_text_feats[idx]
        
        return ref_images, ref_texts, cached_img_feat, cached_text_feat
# In[]
if __name__ == "__main__":
    root = "/home/raoxuan/projects/data/X-TAIL/"
    dataset_name = "aircraft"
    dataset_sequence = ["food101", "dtd", "eurosat"]
    transform_train, transform_test = get_transforms(dataset_name)
    train_loader, train_loader4updating, test_loader, classnames = get_xtail_trainloader(
        root=root,
        dataset_name=dataset_name,
        transform_train=transform_train,
        transform_test=transform_test,
        num_shots=16,
        batch_size=32,)
    
    test_loader, test_classnames, indices = get_xtail_testloader(
        root=root,
        dataset_sequence=dataset_sequence,
        transform_test=transform_test,
        batch_size=32,)
    
# %%

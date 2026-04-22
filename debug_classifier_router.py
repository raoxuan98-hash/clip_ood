"""
debug_classifier_router.py
该脚本用于调试和验证基于 CLIP 的集成分类器 (Ensemble Classifier) 的性能。
主要流程：
1. 提取/加载 CLIP 图像特征 (ID 和 OOD 数据集)。
2. 构建零样本分类器 (Zero-shot) 和 LR-RGDA 分类器。
3. 重点测试集成分类器：通过线性融合 Zero-shot 和 LR-RGDA 的 Logits，观察不同融合比例 (Alpha) 下的性能表现。
"""
#%%
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from sklearn import ensemble
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from sklearn.metrics import roc_auc_score

# 导入项目模块
from src.classifiers.lr_rgda_classifier import LRRGDAClassifier, EnsembleClassifier
from src.detectors.ood_detector import ClassifierBasedOODDetector, build_stats_dict_from_features
from src.routing.adaptive_router import AdaptiveRouter
from utils_data import get_xtail_trainloader, get_xtail_testloader, get_transforms

# 设置运行设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

# 固定随机种子以保证结果可重复性
torch.manual_seed(42)
np.random.seed(42)
# In[]
# 1. 数据集配置
# ID 数据集：模型在训练阶段见过的分布内数据
ID_DATASETS = ['aircraft', 'caltech101', 'dtd', 'eurosat', 'flowers']
# OOD 数据集：模型未见过的分布外数据，用于测试路由鲁棒性
OOD_DATASETS = ['food101', 'mnist', 'oxford_pets']

# 路径配置
DATA_ROOT = '/data1/open_datasets/X-TAIL'
CACHE_DIR = 'cache/pretrained_features'

print("=" * 80)
print("数据集配置")
print("=" * 80)
print(f"数据根目录: {DATA_ROOT}")
print(f"ID数据集: {ID_DATASETS}")
print(f"OOD数据集: {OOD_DATASETS}")
print(f"缓存目录: {CACHE_DIR}")

# In[]
# 2. 加载 CLIP 模型
print("\n[1/6] 正在从镜像站重新加载完整模型和处理器...")

# 增加 local_files_only=False 确保它去下缺失的小文件
model_id = "openai/clip-vit-base-patch16"

try:
    model = CLIPModel.from_pretrained(model_id, use_safetensors=True).to(device)
    # 分开加载，如果 processor 报错，我们能知道具体位置
    processor = CLIPProcessor.from_pretrained(model_id)
    model.eval()
    print("🎉 🎉 恭喜！模型和处理器全部加载成功！")
except Exception as e:
    print(f"❌ 加载失败，原因：{e}")
    print("💡 建议：如果还是报 Timeout，请直接联系师兄要‘本地路径’，那是最后的绝招。")

# %%
# [辅助函数: 图像特征提取]
def extract_features_local(model, dataloader, device):
    """
    通过 CLIP 编码器提取图像特征并进行 L2 归一化
    """
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features", leave=False):
            # 处理不同格式的数据加载器输出
            # isinstance(对象, 类型) ,它在问：“这个东西是不是属于这个类别的？”
            # 如果 batch 是个“盒子”（list/tuple），就把盒子里第 0 号格子的“图纸”拿出来
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            # 如果是盒子且盒子里的格数大于 1，说明第 1 号格子装了“标签图纸”，把它拿出来
            labels = batch[1] if isinstance(batch, (list, tuple)) and len(batch) > 1 else None
            
            images = images.to(device)
            # 提取 Image Features (维度通常为 512)
            features = model.get_image_features(pixel_values=images)
            # 关键步骤：L2 归一化，使得特征点分布在超球面上
            features = features / features.norm(dim=-1, keepdim=True)
            
            all_features.append(features.cpu())
            if labels is not None:
                # 如果labels本身就是tensor，那么直接append进.tensor，否则先把labels变成tensor再append进.tensor
                all_labels.append(labels.cpu() if torch.is_tensor(labels) else torch.tensor(labels))
    
    return torch.cat(all_features), torch.cat(all_labels)

# [自动检查并提取特征]
# 如果 cache 目录不存在对应的 .pkl，则调用原始数据集进行提取
print("\n[检查缓存特征...]")
os.makedirs(CACHE_DIR, exist_ok=True)

for ds in ID_DATASETS + OOD_DATASETS:
    cache_file = f'{CACHE_DIR}/{ds}_features.pkl'
    if not os.path.exists(cache_file):
        print(f"    未发现缓存 {ds}, 开始提取...")
        
        # 获取 CLIP 预定义的预处理变换
        train_transform, test_transform = get_transforms(ds)
        
        # 提取 16-shot 训练特征 (用于计算 LR-RGDA 的类统计信息)
        # _ 是一个约定的**“占位符”，大白话就是：“这里有一个返回值，但我完全不打算用它，所以我懒得给它起名字。”
        train_loader, _, _, class_names = get_xtail_trainloader(
            root=DATA_ROOT,
            dataset_name=ds,
            transform_train=train_transform,
            transform_test=test_transform,
            num_shots=16,
            batch_size=32
        )
        train_features, train_labels = extract_features_local(model, train_loader, device)
        
        # 提取测试集特征 (用于最终评估)
        test_loader, _, _ = get_xtail_testloader(
            root=DATA_ROOT,
            dataset_sequence=[ds],
            transform_test=test_transform,
            batch_size=32,
            max_num_per_dataset=1000
        )
        test_features, test_labels = extract_features_local(model, test_loader, device)
        
        # 保存为 pickle 文件
        data = {
            'train_features': train_features,
            'train_labels': train_labels,
            'test_features': test_features,
            'test_labels': test_labels,
            'class_names': class_names
        }
        # with open：代开文件
        # 'wb'创建并覆盖，'rb'指针式读取
        with open(cache_file, 'wb') as f:
            # pickle.dump：把data的数据存入f
            pickle.dump(data, f)
        print(f"    ✓ {ds} 特征提取并缓存完成")
    else:
        print(f"    ✓ {ds} 缓存已存在")

# 获取 CLIP 默认的 logit scale (通常为 100)
logit_scale = model.logit_scale.exp().item()
print(f"    Logit scale: {logit_scale:.4f}")
# In[]
# [3. 收集类别名称和训练统计信息]
print("\n[2/6] 收集类别名称与元数据...")

# 收集所有类别名（ID + OOD）
all_class_names = []
dataset_info = {}

for ds in ID_DATASETS + OOD_DATASETS:
    with open(f'{CACHE_DIR}/{ds}_features.pkl', 'rb') as f:
        data = pickle.load(f)
    
    num_classes = len(data['class_names'])
    # 嵌套字典
    dataset_info[ds] = {
        'num_classes': num_classes,
        'train_samples': len(data['train_features']),
        'test_samples': len(data['test_features']),
        'class_names': data['class_names']
    }
    # .extend和torch.cat是一个意思，只是extend用在处理名字，cat用在张量
    all_class_names.extend(data['class_names'])

total_classes = len(all_class_names)
# 因为是嵌套字典，所以需要双重索引
num_id_classes = sum(dataset_info[ds]['num_classes'] for ds in ID_DATASETS)

print(f"    总类别数: {total_classes}")
print(f"    ID类别数: {num_id_classes}")
print(f"    OOD类别数: {total_classes - num_id_classes}")

print("\n数据集详情:")
for ds, info in dataset_info.items():
    ds_type = "ID" if ds in ID_DATASETS else "OOD"
    print(f"    {ds:15s} ({ds_type}): {info['num_classes']:3d} classes, "
          f"{info['train_samples']:4d} train, {info['test_samples']:4d} test")
# %%
# [4. 加载 ID 训练特征并构建统计分布]
# 这一步是为了获取每个类别的均值 (Means) 和协方差 (Covariances)，供 LR-RGDA 分类器使用
print("\n[3/6] 加载 ID 训练特征并构建统计分布...")

all_train_features = []
all_train_labels = []
label_offset = 0

for ds in ID_DATASETS:
    with open(f'{CACHE_DIR}/{ds}_features.pkl', 'rb') as f:
        data = pickle.load(f)
    
    all_train_features.append(data['train_features'])
    # 因为每个类的labels都是从0开始，用label_offset是把所有labels从0开始排列
    all_train_labels.append(data['train_labels'] + label_offset)
    label_offset += len(data['class_names'])

all_train_features = torch.cat(all_train_features)
all_train_labels = torch.cat(all_train_labels)

print(f"    总训练样本: {len(all_train_features)}")
print(f"    训练特征维度: {all_train_features.shape}")

# 构建统计分布
print("\n构建stats_dict...")
# stats_dict是一个字典，key为类别标签，value为均值和协方差
# 例如 stats_dict[0]['mu']就是第0类的均值，stats_dict[0]['cov']就是第0类的协方差
stats_dict = build_stats_dict_from_features(all_train_features, all_train_labels)
print(f"    Stats dict类别数: {len(stats_dict)}")
# In[]
# [5. 构建零样本 (Zero-shot) 分类器]
# 使用 CLIP 的文本编码器生成所有类别的文本嵌入 (Text Embeddings)
print("\n[4/6] 构建零样本分类器...")

zeroshot_weights = []
# 提示词模板
# 提示词是指开发者给模型的答案选项，clip并不会自己给出答案，而是从选项中选出
# clip把图变成向量，把提示词也变成向量，看图的向量和提示词向量谁距离更近
# templates是一个列表，lambda x: ... 实际上定义了一个没有名字的函数
templates = [lambda x: f"a photo of a {x}."]

with torch.no_grad():
    for classname in tqdm(all_class_names, desc="Building zero-shot classifier"):
        # 把_替换成空格
        classname = classname.replace('_', ' ')
        # 因为templates里面本身就是一个函数，所以template就是一个函数
        # 这里要设计成for循环，是因为模板可能不止一个，也就是templates列表里面可以是多个模板
        texts = [template(classname) for template in templates]
        # 文本分词并编码
        # text:要处理的文本，return_tensors="pt":pt 代表 PyTorch,意思是转成张量
        # adding=True（填充）：如果第一句话有 5 个词，第二句有 10 个词，程序会自动在短的那句后面补一堆“零”（Pad Token），把它们强行凑成一样长，形成一个整齐的矩阵
        # truncation=True（截断）：模型的大脑（Transformer）容量有限（CLIP 通常限制 77 个 Token），会强制把超过长度的部分切掉，防止内存溢出
        text_inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
        class_embeddings = model.get_text_features(**text_inputs)
        # 归一化
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        # 如果有多个模板，取平均值
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        zeroshot_weights.append(class_embedding)

# 权重矩阵形状: [512, Total_Classes]
zeroshot_classifier = torch.stack(zeroshot_weights, dim=1).to(device)
print(f"    零样本分类器形状: {zeroshot_classifier.shape}")
# %%
# [6. 构建 LR-RGDA 分类器]
# LR-RGDA (Low-Rank Regularized Gaussian Discriminant Analysis) 
# 它在少量样本 (few-shot) 下通过正则化手段比传统的 QDA 或 LDA 更鲁棒
print("\n[5/6] 构建 LR-RGDA 分类器...")

# 构建LR-RGDA分类器
lr_rgda_classifier = LRRGDAClassifier(
    stats_dict=stats_dict,
    device=device,
    rank=32,
    qda_reg_alpha1=0.3,
    qda_reg_alpha2=2.0,
    qda_reg_alpha3=0.5,
)

print("    ✓ LR-RGDA分类器构建完成")
# %%
# [8. 辅助函数定义]
# classifier_fn是一个函数，所以说函数的参数也可以是一个函数
def evaluate_classifier(classifier_fn, name, test_features, test_labels, is_id=True):
    """
    评估分类器性能
    classifier_fn: 预测函数，输入features，输出predictions
    """
    with torch.no_grad():
        predictions = classifier_fn(test_features)
    
    correct = (predictions.cpu() == test_labels.cpu()).sum().item()
    total = len(test_labels)
    accuracy = correct / total * 100
    # is_id被默认设置成为True
    ds_type = "ID" if is_id else "OOD"
    print(f"    {name:20s} ({ds_type}): {accuracy:5.1f}% ({correct}/{total})")
    return accuracy, correct, total


def get_test_data(dataset_list, is_id=True):
    """
    加载测试数据并应用正确的label offset
    CLIP 零样本分类器是把 所有 数据集的类别都堆在一起的
    判断是算 ID（从 0 开始累计）还是 OOD（从 ID 的总数后面开始累计）
    数据集排列顺序是先ID再OOD
    """
    all_features = []
    all_labels = []
    
    # 计算基础offset
    if is_id:
        # ID数据集的offset从0开始
        base_offset = 0
        offset = 0
        for ds in ID_DATASETS:
            if ds in dataset_list:
                # 在 Python 的 for 循环中，break 的作用只有一个：立即停止当前的循环，跳出大括号（或者缩进块），去执行循环后面的代码
                break
            offset += dataset_info[ds]['num_classes']
    else:
        # OOD数据集的offset从num_id_classes开始
        base_offset = num_id_classes
        offset = base_offset
        for ds in OOD_DATASETS:
            if ds in dataset_list:
                break
            offset += dataset_info[ds]['num_classes']
    
    # 获得测试集的特征向量以及全局序号
    for ds in dataset_list:
        with open(f'{CACHE_DIR}/{ds}_features.pkl', 'rb') as f:
            data = pickle.load(f)
        
        all_features.append(data['test_features'])
        all_labels.append(data['test_labels'] + offset)
        offset += dataset_info[ds]['num_classes']
    
    features = torch.cat(all_features).to(device)
    labels = torch.cat(all_labels).to(device)

    return features, labels
    
# %%
print("\n[6/6] 加载测试数据...")
id_test_features, id_test_labels = get_test_data(ID_DATASETS, is_id=True)
ood_test_features, ood_test_labels = get_test_data(OOD_DATASETS, is_id=False)

print(f"    ID测试样本: {len(id_test_features)}")
print(f"    OOD测试样本: {len(ood_test_features)}")

# %%
# [7. 评估逻辑与测试 1: 纯零样本性能]
print("\n" + "=" * 80)
print("测试1: 纯零样本分类器 (Zero-shot Only)")
print("=" * 80)

def zeroshot_predict(features):
    """
    计算图像特征与文本权重的余弦相似度，并应用 logit_scale
    """
    with torch.no_grad():
        # 权重矩阵形状: [512, Total_Classes]
        logits = logit_scale * (features @ zeroshot_classifier)
        return logits.argmax(dim=1)

# 这里的zeroshot_predict是函数，不是参数
zeroshot_id_acc, zeroshot_id_correct, zeroshot_id_total = evaluate_classifier(
    zeroshot_predict, "Zero-shot", id_test_features, id_test_labels, is_id=True
)
zeroshot_ood_acc, zeroshot_ood_correct, zeroshot_ood_total = evaluate_classifier(
    zeroshot_predict, "Zero-shot", ood_test_features, ood_test_labels, is_id=False
)

zeroshot_overall_acc = (zeroshot_id_correct + zeroshot_ood_correct) / (zeroshot_id_total + zeroshot_ood_total) * 100
print(f"\n    平均准确率: {zeroshot_overall_acc:.1f}%")
# %%
# [10. 测试2: 纯LR-RGDA分类器]
print("\n" + "=" * 80)
print("测试2: 纯LR-RGDA分类器 (仅ID类别)")
print("=" * 80)
print("注意: LR-RGDA只对ID类别输出非零概率\n")

def lrrgda_predict(features):
    with torch.no_grad():
        return lr_rgda_classifier.predict(features)

lrrgda_id_acc, lrrgda_id_correct, lrrgda_id_total = evaluate_classifier(
    lrrgda_predict, "LR-RGDA", id_test_features, id_test_labels, is_id=True
)

# 对于OOD样本，LR-RGDA的预测会落在ID类别范围内（0~num_id_classes-1）
# 这会导致OOD准确率几乎为0，因为真实标签在num_id_classes之后
with torch.no_grad():
    ood_preds = lr_rgda_classifier.predict(ood_test_features)
    # 所有预测都在ID范围内
    print(f"    LR-RGDA预测范围: [{ood_preds.min()}, {ood_preds.max()}]")
    print(f"    OOD真实标签范围: [{ood_test_labels.min()}, {ood_test_labels.max()}]")
    print(f"    → 预测永远不等于真实标签（OOD性能≈0%）")
    lrrgda_ood_acc = 0.0
    lrrgda_ood_correct = 0
    lrrgda_ood_total = len(ood_test_labels)

lrrgda_overall_acc = (lrrgda_id_correct + lrrgda_ood_correct) / (lrrgda_id_total + lrrgda_ood_total) * 100
print(f"\n    平均准确率: {lrrgda_overall_acc:.1f}%")

# In[]
# [9. 测试 3: 集成分类器 (Ensemble Classifier) 性能分析]
# 核心逻辑：
# 我们将 Zero-shot 分类器 (泛化能力强) 与 LR-RGDA 分类器 (在 ID 上精度高) 进行 Logits 级融合。
# 公式: ensemble_logits = (1 - alpha) * zeroshot_logits + alpha * rgda_logits (仅针对 ID 类别)

print("\n" + "=" * 80)
print("测试3: 集成分类器性能分析 (Alpha Sweep)")
print("=" * 80)

# 合并所有测试特征以便批量计算
all_test_features = torch.cat([id_test_features, ood_test_features]).to(device)

# 计算 Zero-shot Logits
with torch.no_grad():
    zeroshot_logits = logit_scale * (all_test_features @ zeroshot_classifier)
    # 归一化 Logits (减去最大值以防溢出),dim=-1代表最后一个维度，在这里跟dim=1是一个方向也就是横方向
    zeroshot_logits = zeroshot_logits - zeroshot_logits.max(dim=-1, keepdim=True).values

# 计算 LR-RGDA Logits
with torch.no_grad():
    rgda_logits = lr_rgda_classifier.forward(all_test_features)
    # 归一化 Logits
    rgda_logits = rgda_logits - rgda_logits.max(dim=-1, keepdim=True).values



# 准备真实标签
all_test_labels = torch.cat([id_test_labels, ood_test_labels]).to(device)

# 遍历不同的融合比例 alpha
# alpha = 0.0 -> 纯 Zero-shot
# alpha = 1.0 -> ID 类别完全由 LR-RGDA 决定
alphas = torch.linspace(0.0, 1.0, 21) # 采样 21 个点
best_alpha = 0
max_acc = 0

print(f"{'Alpha':>10} | {'Overall Acc':>15} | {'ID Acc':>15} | {'OOD Acc':>15}")
print("-" * 65)

for alpha in alphas:
    # 【关键修改】在这里创建分类器，此时 alpha 已经从 alphas 里取出来了，是有值的
    # 基础权重分配给 Zero-shot
    # 所有的ID和OOD数据集需要clip给出logits
    
    ensemble_logits = zeroshot_logits * (1 - alpha)
    
    # 仅在 ID 类别上叠加 LR-RGDA 的贡献
    # 注意：LR-RGDA 的输出维度仅为 num_id_classes
    # ：代表所有行，也就是所有图片
    # :num_id_classes代表从第一列到ID数据集的最后一列
    # 也就是说，最后的ensemble_logits矩阵，ID部分是clip和rgda都起作用，但是OOD部分只有clip起作用
    ensemble_logits[:, :num_id_classes] += alpha * rgda_logits
    
    # 选取预测结果
    ensemble_preds = ensemble_logits.argmax(dim=1)


    # 计算各项指标
    # .eq是对比的意思，得到一个全是布尔值的列表：[True, False, True, True...]
    # .float是强制把布尔值变成数字
    # .mean是算平均，(1 + 0 + 1 + 1)/4 = 0.75，平均值就是正确率
    overall_acc = ensemble_preds.eq(all_test_labels).float().mean() * 100
    # 用切片把最后的预测切成ID和OOD两部分
    id_acc = ensemble_preds[:len(id_test_features)].eq(id_test_labels).float().mean() * 100
    ood_acc = ensemble_preds[len(id_test_features):].eq(ood_test_labels).float().mean() * 100
    
    print(f"{alpha.item():10.2f} | {overall_acc.item():14.2f}% | {id_acc.item():14.2f}% | {ood_acc.item():14.2f}%")
    
    if overall_acc > max_acc:
        max_acc = overall_acc
        best_alpha = alpha.item()

print("-" * 65)
print(f"最佳融合比例 Alpha: {best_alpha:.2f}, 最高整体准确率: {max_acc:.2f}%")

# [10. 实验结论]
# 1. 随着 alpha 增加，ID 准确率通常会提升，因为 LR-RGDA 学习了特定领域的分布信息。
# 2. 如果 alpha 过大，可能会牺牲 OOD 性能，因为模型变得过于“专注”于 ID 类别。
# 3. 集成分类器通过简单的线性加权，在不需要重新训练的情况下有效结合了预训练知识和领域知识。
# %%

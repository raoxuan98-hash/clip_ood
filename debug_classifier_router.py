# In[]
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

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# In[]
ID_DATASETS = ['aircraft', 'caltech101', 'dtd', 'eurosat', 'flowers']
OOD_DATASETS = ['food101', 'mnist', 'oxford_pets']

CACHE_DIR = 'cache/pretrained_features'

print("=" * 80)
print("数据集配置")
print("=" * 80)
print(f"ID数据集: {ID_DATASETS}")
print(f"OOD数据集: {OOD_DATASETS}")
print(f"缓存目录: {CACHE_DIR}")

# In[]
print("\n[1/6] 加载CLIP模型...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
model.eval()

# %%
# 获取logit_scale
logit_scale = model.logit_scale.exp().item()
print(f"    Logit scale: {logit_scale:.4f}")

# [4. 收集类别名称和特征]
print("\n[2/6] 收集类别名称...")

# 收集所有类别名（ID + OOD）
all_class_names = []
dataset_info = {}

for ds in ID_DATASETS + OOD_DATASETS:
    with open(f'{CACHE_DIR}/{ds}_features.pkl', 'rb') as f:
        data = pickle.load(f)
    
    num_classes = len(data['class_names'])
    dataset_info[ds] = {
        'num_classes': num_classes,
        'train_samples': len(data['train_features']),
        'test_samples': len(data['test_features']),
        'class_names': data['class_names']
    }
    all_class_names.extend(data['class_names'])

total_classes = len(all_class_names)
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
# [5. 加载ID训练特征并构建统计分布]
print("\n[3/6] 加载ID训练特征并构建统计分布...")

all_train_features = []
all_train_labels = []
label_offset = 0

for ds in ID_DATASETS:
    with open(f'{CACHE_DIR}/{ds}_features.pkl', 'rb') as f:
        data = pickle.load(f)
    
    all_train_features.append(data['train_features'])
    all_train_labels.append(data['train_labels'] + label_offset)
    label_offset += len(data['class_names'])

all_train_features = torch.cat(all_train_features)
all_train_labels = torch.cat(all_train_labels)

print(f"    总训练样本: {len(all_train_features)}")
print(f"    训练特征维度: {all_train_features.shape}")

# 构建统计分布
print("\n构建stats_dict...")
stats_dict = build_stats_dict_from_features(all_train_features, all_train_labels)
print(f"    Stats dict类别数: {len(stats_dict)}")
# %%
# [6. 构建零样本分类器]
print("\n[4/6] 构建零样本分类器...")

zeroshot_weights = []
templates = [lambda x: f"a photo of a {x}."]

with torch.no_grad():
    for classname in tqdm(all_class_names, desc="Building zero-shot classifier"):
        classname = classname.replace('_', ' ')
        texts = [template(classname) for template in templates]
        text_inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
        class_embeddings = model.get_text_features(**text_inputs)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        zeroshot_weights.append(class_embedding)

zeroshot_classifier = torch.stack(zeroshot_weights, dim=1).to(device)
print(f"    零样本分类器形状: {zeroshot_classifier.shape}")
# %%
print("\n[5/6] 构建LR-RGDA分类器和OOD检测器...")

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
    
    ds_type = "ID" if is_id else "OOD"
    print(f"    {name:20s} ({ds_type}): {accuracy:5.1f}% ({correct}/{total})")
    return accuracy, correct, total


def get_test_data(dataset_list, is_id=True):
    """
    加载测试数据并应用正确的label offset
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
# [9. 测试1: 纯零样本分类器]
print("\n" + "=" * 80)
print("测试1: 纯零样本分类器 (Zero-shot Only)")
print("=" * 80)

def zeroshot_predict(features):
    with torch.no_grad():
        logits = logit_scale * (features @ zeroshot_classifier)
        return logits.argmax(dim=1)

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
all_test_features = torch.cat([id_test_features, ood_test_features]).to(device)
zeroshot_logits = all_test_features @ zeroshot_classifier
zeroshot_logits = zeroshot_logits - zeroshot_logits.max(dim=-1, keepdim=True).values
zeroshot_logits = zeroshot_logits

rgda_logits = lr_rgda_classifier.forward(all_test_features)
rgda_logits = rgda_logits - rgda_logits.max(dim=-1, keepdim=True).values

for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    ensemble_logits = zeroshot_logits * (1 - alpha)
    ensemble_logits[:, :num_id_classes] += alpha * rgda_logits
    ensemble_preds = ensemble_logits.argmax(dim=1)

    overall_acc = ensemble_preds.eq(torch.cat([id_test_labels, ood_test_labels])).float().mean() * 100
    id_acc = ensemble_preds[:len(id_test_features)].eq(id_test_labels).float().mean() * 100
    ood_acc = ensemble_preds[len(id_test_features):].eq(ood_test_labels).float().mean() * 100
    
    print(f"    α={alpha:.2f}: 整体={overall_acc:.1f}%, ID={id_acc:.1f}%, OOD={ood_acc:.1f}%") 


# In[]
ood_detector = ClassifierBasedOODDetector(
    stats_dict=stats_dict,
    classifier_type='lr_rgda',
    device=device,
    rank=32,
    qda_reg_alpha1=0.3,
    qda_reg_alpha2=2.0,
    qda_reg_alpha3=0.5
)

# In[]
print("\n" + "=" * 80)
print("测试3: 不同的OOD分数对OOD预测的准确度")
print("=" * 80)
print("注意: LR-RGDA只对ID类别输出非零概率\n")
# Here, we test the robustness of the Softmax-based scores against different temperatures
logits = ood_detector.classifier.forward(all_test_features)
logits = logits - logits.max(dim=1, keepdim=True).values
temp_list = torch.logspace(-2, 2.0, 50)

iid_labels = torch.zeros(len(id_test_features), dtype=torch.int32)
ood_labels = torch.ones(len(ood_test_features), dtype=torch.int32)
test_labels = torch.cat([iid_labels, ood_labels], dim=0)

# In[]
auc_list = []
for temp in temp_list:
    scores = 1.0 - torch.softmax(logits / temp, dim=1)[:, :num_id_classes].max(dim=1).values
    auc = roc_auc_score(test_labels, scores.cpu().numpy())
    auc_list.append(auc)
    print(f"emp={temp:.3f}: AUC={auc:.4f}")

fig, ax = plt.subplots()
ax.plot(temp_list, auc_list)
ax.set_xscale("log")
ax.set_xlabel("Temperature")
ax.set_ylabel("AUC")

# 我们可以看到，基于Softmax最大分数的OOD检测，受温度的影响较大
# 温度较高时，OOD分数更敏感，AUC也较高

# In[]
auc_list = []

for temp in temp_list:
    probs = torch.softmax(logits / temp, dim=1)
    scores = - torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    auc = roc_auc_score(test_labels, scores.cpu().numpy())
    auc_list.append(auc)
    print(f"emp={temp:.3f}: AUC={auc:.4f}")

fig, ax = plt.subplots()
ax.plot(temp_list, auc_list)
ax.set_xscale("log")
ax.set_xlabel("Temperature")
ax.set_ylabel("AUC")

# 我们可以看到，基于Entropy的OOD检测，受温度的影响也比较大

# In[]
auc_list = []

for temp in temp_list:
    scores = torch.logsumexp(logits / temp, dim=1) 
    auc = roc_auc_score(test_labels, scores.cpu().numpy())
    auc_list.append(auc)
    print(f"emp={temp:.3f}: AUC={auc:.4f}")

fig, ax = plt.subplots()
ax.plot(temp_list, auc_list)
ax.set_xscale("log")
ax.set_xlabel("Temperature")
ax.set_ylabel("AUC")

# 我们可以看到，基于Entropy的OOD检测，其性能曲线基于与基于最大概率的性能曲线相近
# 更重要的是，我注意到scores的变动范围受temp影响没那么大，而基于最大概率的scores变动范围受temp影响较大
# 但是，我在想scores会不会受id_num_classes影响较大，这在增量学习的场景是有可能不试用。

# %%
# 这里开始，我们检测基于OOD路由的自适应分类器的性能
print("\n" + "=" * 80)
print("测试4: 自适应路由分类器 (Adaptive Routing)")
print("=" * 80)
print("策略: ID-like → 集成分类器, OOD-like → 零样本分类器\n")

# 1. 计算 Energy Score
# 注意: Energy = T * logsumexp(logits / T)
# 能量越高越像 ID，所以我们取负号作为 OOD Score，这样分数越高代表越像 OOD
temp = num_id_classes
logits = ood_detector.classifier.forward(all_test_features)
logits = logits - logits.max(dim=-1, keepdim=True).values

id_logits = logits[:len(id_test_features)]
ood_logits = logits[len(id_test_features):]

id_energy_scores = num_id_classes * torch.logsumexp(id_logits / num_id_classes, dim=1)
ood_energy_scores = num_id_classes * torch.logsumexp(ood_logits / num_id_classes, dim=1)

percentile = 95
threshold = np.percentile(id_energy_scores.cpu().numpy(), percentile)

id_as_ood_preds = id_energy_scores > threshold
ood_as_ood_preds = ood_energy_scores > threshold

print(id_as_ood_preds.sum() / len(id_as_ood_preds))
print(ood_as_ood_preds.sum() / len(ood_as_ood_preds))
# %%

all_ood_preds = torch.cat([id_as_ood_preds, ood_as_ood_preds])


# %%
alpha = 0.7

all_test_features = torch.cat([id_test_features, ood_test_features]).to(device)
zeroshot_logits = all_test_features @ zeroshot_classifier
zeroshot_logits = zeroshot_logits - zeroshot_logits.max(dim=-1, keepdim=True).values
zeroshot_logits = zeroshot_logits

rgda_logits = lr_rgda_classifier.forward(all_test_features)
rgda_logits = rgda_logits - rgda_logits.max(dim=-1, keepdim=True).values

ensemble_logits = zeroshot_logits * (1 - alpha)
ensemble_logits[:, :num_id_classes] += alpha * rgda_logits

# ensemble_logits[all_ood_preds] = zeroshot_logits[all_ood_preds]

print(ensemble_logits.argmax(dim=1).eq(torch.cat([id_test_labels, ood_test_labels])).float().mean())
# %%

# [分类器性能对比测试]
# 本脚本用于系统地测试和比较以下分类器的性能：
# 1. 纯零样本分类器 (Zero-shot)：CLIP原始能力
# 2. 纯LR-RGDA分类器：仅使用监督数据的高斯判别分析
# 3. 固定集成分类器 (Ensemble)：固定权重融合LR-RGDA和零样本
# 4. 自适应路由分类器 (Adaptive Routing)：根据OOD检测结果动态选择分类器
# 目标：验证各分类器在ID（分布内）和OOD（分布外）数据上的性能差异

# [1. 配置与导入]
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

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

# [2. 数据集配置]
# 可修改此处来测试不同的ID/OOD组合
# 可选数据集: aircraft, caltech101, dtd, eurosat, flowers, 
#            food101, mnist, oxford_pets, stanford_cars, sun397

# 推荐配置1: 简单组合（适合快速测试）
ID_DATASETS = ['caltech101', 'flowers']
OOD_DATASETS = ['food101', 'oxford_pets']

# 推荐配置2: 平衡组合（接近论文Table 4 Combo 1）
# ID_DATASETS = ['aircraft', 'caltech101', 'dtd', 'eurosat', 'flowers']
# OOD_DATASETS = ['food101', 'mnist', 'oxford_pets', 'stanford_cars', 'sun397']

# 推荐配置3: ID偏少
# ID_DATASETS = ['aircraft', 'caltech101', 'dtd']
# OOD_DATASETS = ['eurosat', 'flowers', 'food101', 'mnist', 'oxford_pets', 'stanford_cars', 'sun397']

CACHE_DIR = 'cache/pretrained_features'

print("=" * 80)
print("数据集配置")
print("=" * 80)
print(f"ID数据集: {ID_DATASETS}")
print(f"OOD数据集: {OOD_DATASETS}")
print(f"缓存目录: {CACHE_DIR}")

# [3. 加载CLIP模型]
print("\n[1/6] 加载CLIP模型...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
model.eval()

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

# [7. 构建LR-RGDA分类器和OOD检测器]
print("\n[5/6] 构建LR-RGDA分类器和OOD检测器...")

# 构建LR-RGDA分类器
lr_rgda_classifier = LRRGDAClassifier(
    stats_dict=stats_dict,
    device=device,
    rank=32,
    qda_reg_alpha1=0.6,
    qda_reg_alpha2=1.0,
    qda_reg_alpha3=0.5,
    temperature=1.0
)

# 构建OOD检测器
ood_detector = ClassifierBasedOODDetector(
    stats_dict=stats_dict,
    classifier_type='lr_rgda',
    device=device,
    rank=32,
    qda_reg_alpha1=0.6,
    qda_reg_alpha2=1.0,
    qda_reg_alpha3=0.5
)

print("    ✓ LR-RGDA分类器构建完成")
print("    ✓ OOD检测器构建完成")

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


# 加载测试数据
print("\n[6/6] 加载测试数据...")
id_test_features, id_test_labels = get_test_data(ID_DATASETS, is_id=True)
ood_test_features, ood_test_labels = get_test_data(OOD_DATASETS, is_id=False)

print(f"    ID测试样本: {len(id_test_features)}")
print(f"    OOD测试样本: {len(ood_test_features)}")

# [9. 测试1: 纯零样本分类器]
print("\n" + "=" * 80)
print("测试1: 纯零样本分类器 (Zero-shot Only)")
print("=" * 80)

def zeroshot_predict(features):
    with torch.no_grad():
        logits = logit_scale * (features @ zeroshot_classifier)
        return logits.argmax(dim=1)

zeroshot_id_acc, _, _ = evaluate_classifier(
    zeroshot_predict, "Zero-shot", id_test_features, id_test_labels, is_id=True
)
zeroshot_ood_acc, _, _ = evaluate_classifier(
    zeroshot_predict, "Zero-shot", ood_test_features, ood_test_labels, is_id=False
)

print(f"\n    平均准确率: {(zeroshot_id_acc + zeroshot_ood_acc) / 2:.1f}%")

# [10. 测试2: 纯LR-RGDA分类器]
print("\n" + "=" * 80)
print("测试2: 纯LR-RGDA分类器 (仅ID类别)")
print("=" * 80)
print("注意: LR-RGDA只对ID类别输出非零概率\n")

def lrrgda_predict(features):
    with torch.no_grad():
        return lr_rgda_classifier.predict(features)

lrrgda_id_acc, _, _ = evaluate_classifier(
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

print(f"\n    平均准确率: {(lrrgda_id_acc + lrrgda_ood_acc) / 2:.1f}%")

# [11. 测试3: 固定集成分类器（不同alpha值）]
print("\n" + "=" * 80)
print("测试3: 固定集成分类器 (Ensemble with different α)")
print("=" * 80)
print("公式: P_ensemble = α·P_LR-RGDA + (1-α)·P_zeroshot\n")

ensemble_results = []
alpha_values = [0.0, 0.2, 0.5, 0.8, 1.0]

for alpha in alpha_values:
    # 构建集成分类器
    ensemble = EnsembleClassifier(
        zeroshot_classifier,
        lr_rgda_classifier,
        alpha=alpha,
        temperature=1.0,
        num_id_classes=num_id_classes
    )
    
    def ensemble_predict(features):
        with torch.no_grad():
            probs = ensemble.predict_proba(features, logit_scale)
            return probs.argmax(dim=1)
    
    print(f"\n[α={alpha}]")
    id_acc, _, _ = evaluate_classifier(
        ensemble_predict, f"Ensemble(α={alpha})", 
        id_test_features, id_test_labels, is_id=True
    )
    ood_acc, _, _ = evaluate_classifier(
        ensemble_predict, f"Ensemble(α={alpha})", 
        ood_test_features, ood_test_labels, is_id=False
    )
    
    avg_acc = (id_acc + ood_acc) / 2
    print(f"    平均准确率: {avg_acc:.1f}%")
    
    ensemble_results.append({
        'alpha': alpha,
        'id_acc': id_acc,
        'ood_acc': ood_acc,
        'avg_acc': avg_acc
    })

# [12. 测试4: 自适应路由分类器]
print("\n" + "=" * 80)
print("测试4: 自适应路由分类器 (Adaptive Routing)")
print("=" * 80)
print("策略: ID-like → 集成分类器, OOD-like → 零样本分类器\n")

# 测试不同阈值
thresholds = [0.994, 0.995, 0.996, 0.997]
routing_results = []

for threshold in thresholds:
    # 构建路由器
    ensemble = EnsembleClassifier(
        zeroshot_classifier,
        lr_rgda_classifier,
        alpha=0.8,
        temperature=1.0,
        num_id_classes=num_id_classes
    )
    
    router = AdaptiveRouter(
        zeroshot_classifier,
        ensemble,
        ood_detector,
        threshold=threshold
    )
    
    def router_predict(features):
        with torch.no_grad():
            predictions, _ = router.predict(features, logit_scale)
            return predictions
    
    print(f"\n[阈值={threshold}]")
    
    # ID样本
    with torch.no_grad():
        id_preds, id_is_ood = router.predict(id_test_features, logit_scale)
        id_routed_to_ensemble = (~id_is_ood).sum().item()
        id_routed_to_zeroshot = id_is_ood.sum().item()
        id_acc = (id_preds == id_test_labels).float().mean().item() * 100
    
    print(f"    ID样本路由: {id_routed_to_ensemble}→集成, {id_routed_to_zeroshot}→零样本")
    print(f"    ID准确率:   {id_acc:.1f}%")
    
    # OOD样本
    with torch.no_grad():
        ood_preds, ood_is_ood = router.predict(ood_test_features, logit_scale)
        ood_routed_to_ensemble = (~ood_is_ood).sum().item()
        ood_routed_to_zeroshot = ood_is_ood.sum().item()
        ood_acc = (ood_preds == ood_test_labels).float().mean().item() * 100
    
    print(f"    OOD样本路由: {ood_routed_to_ensemble}→集成, {ood_routed_to_zeroshot}→零样本")
    print(f"    OOD准确率:   {ood_acc:.1f}%")
    
    avg_acc = (id_acc + ood_acc) / 2
    print(f"    平均准确率: {avg_acc:.1f}%")
    
    routing_results.append({
        'threshold': threshold,
        'id_acc': id_acc,
        'ood_acc': ood_acc,
        'avg_acc': avg_acc,
        'id_to_ensemble': id_routed_to_ensemble,
        'id_to_zeroshot': id_routed_to_zeroshot,
        'ood_to_ensemble': ood_routed_to_ensemble,
        'ood_to_zeroshot': ood_routed_to_zeroshot
    })

# [13. 可视化对比]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 图1: 不同alpha的固定集成
alphas = [r['alpha'] for r in ensemble_results]
id_accs = [r['id_acc'] for r in ensemble_results]
ood_accs = [r['ood_acc'] for r in ensemble_results]
avg_accs = [r['avg_acc'] for r in ensemble_results]

axes[0].plot(alphas, id_accs, 'o-', label='ID Acc', linewidth=2, markersize=8)
axes[0].plot(alphas, ood_accs, 's-', label='OOD Acc', linewidth=2, markersize=8)
axes[0].plot(alphas, avg_accs, '^-', label='Overall Acc', linewidth=2, markersize=8)
axes[0].set_xlabel('Alpha (α)', fontsize=12)
axes[0].set_ylabel('Accuracy (%)', fontsize=12)
axes[0].set_title('Fixed Ensemble (Different α)', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([0, 100])

# 图2: 不同阈值的自适应路由
threshs = [r['threshold'] for r in routing_results]
id_accs_r = [r['id_acc'] for r in routing_results]
ood_accs_r = [r['ood_acc'] for r in routing_results]
avg_accs_r = [r['avg_acc'] for r in routing_results]

axes[1].plot(threshs, id_accs_r, 'o-', label='ID Acc', linewidth=2, markersize=8)
axes[1].plot(threshs, ood_accs_r, 's-', label='OOD Acc', linewidth=2, markersize=8)
axes[1].plot(threshs, avg_accs_r, '^-', label='Overall Acc', linewidth=2, markersize=8)
axes[1].set_xlabel('Threshold', fontsize=12)
axes[1].set_ylabel('Accuracy (%)', fontsize=12)
axes[1].set_title('Adaptive Routing (Different Threshold)', fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim([0, 100])

# 图3: 方法对比（使用alpha=0.8和最佳阈值）
methods = ['Zero-shot', 'LR-RGDA', 'Ensemble\n(α=0.8)', 'Adaptive\nRouting']
id_vals = [zeroshot_id_acc, lrrgda_id_acc, 
           [r['id_acc'] for r in ensemble_results if r['alpha'] == 0.8][0],
           id_accs_r[len(id_accs_r)//2]]  # 使用中间阈值
ood_vals = [zeroshot_ood_acc, 0, 
            [r['ood_acc'] for r in ensemble_results if r['alpha'] == 0.8][0],
            ood_accs_r[len(ood_accs_r)//2]]

x = np.arange(len(methods))
width = 0.35

axes[2].bar(x - width/2, id_vals, width, label='ID Acc', color='steelblue')
axes[2].bar(x + width/2, ood_vals, width, label='OOD Acc', color='coral')
axes[2].set_ylabel('Accuracy (%)', fontsize=12)
axes[2].set_title('Method Comparison', fontsize=14)
axes[2].set_xticks(x)
axes[2].set_xticklabels(methods)
axes[2].legend()
axes[2].grid(True, alpha=0.3, axis='y')
axes[2].set_ylim([0, 100])

# 添加数值标签
for i, (id_v, ood_v) in enumerate(zip(id_vals, ood_vals)):
    axes[2].text(i - width/2, id_v + 2, f'{id_v:.1f}', ha='center', fontsize=9)
    axes[2].text(i + width/2, ood_v + 2, f'{ood_v:.1f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('classifier_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("图表已保存为: classifier_comparison.png")

# [14. 关键发现与分析]
print("=" * 80)
print("关键发现总结")
print("=" * 80)

print("\n1. 纯零样本分类器:")
print(f"   - ID准确率: {zeroshot_id_acc:.1f}%")
print(f"   - OOD准确率: {zeroshot_ood_acc:.1f}%")
print(f"   - 特点: OOD性能较好，但ID性能一般")

print("\n2. 纯LR-RGDA分类器:")
print(f"   - ID准确率: {lrrgda_id_acc:.1f}%")
print(f"   - OOD准确率: ~0% (只能预测ID类别)")
print(f"   - 特点: ID性能最好，但完全无法识别OOD")

best_ensemble = max(ensemble_results, key=lambda x: x['avg_acc'])
print(f"\n3. 固定集成分类器 (最佳α={best_ensemble['alpha']}):")
print(f"   - ID准确率: {best_ensemble['id_acc']:.1f}%")
print(f"   - OOD准确率: {best_ensemble['ood_acc']:.1f}%")
print(f"   - 平均准确率: {best_ensemble['avg_acc']:.1f}%")
print(f"   - 特点: 折中方案，但OOD性能仍有损失")

best_routing = max(routing_results, key=lambda x: x['avg_acc'])
print(f"\n4. 自适应路由分类器 (最佳阈值={best_routing['threshold']}):")
print(f"   - ID准确率: {best_routing['id_acc']:.1f}%")
print(f"   - OOD准确率: {best_routing['ood_acc']:.1f}%")
print(f"   - 平均准确率: {best_routing['avg_acc']:.1f}%")
print(f"   - 特点: ID和OOD性能都较好，达到最佳平衡")

print("\n" + "=" * 80)
print("核心结论")
print("=" * 80)
print(f"\n✓ 自适应路由相比固定集成:")
print(f"  - ID性能: {best_routing['id_acc'] - best_ensemble['id_acc']:+.1f}%")
print(f"  - OOD性能: {best_routing['ood_acc'] - best_ensemble['ood_acc']:+.1f}%")
print(f"  - 整体: {best_routing['avg_acc'] - best_ensemble['avg_acc']:+.1f}%")

print(f"\n✓ 自适应路由成功的原因:")
print(f"  - ID样本主要路由到集成 ({best_routing['id_to_ensemble']}/{best_routing['id_to_ensemble'] + best_routing['id_to_zeroshot']})")
print(f"  - OOD样本主要路由到零样本 ({best_routing['ood_to_zeroshot']}/{best_routing['ood_to_ensemble'] + best_routing['ood_to_zeroshot']})")
print(f"  - 实现了'对症下药'的动态策略")

# [15. 详细分析：alpha对固定集成的影响]
print("\n" + "=" * 80)
print("固定集成的alpha敏感性分析")
print("=" * 80)

print("\n| Alpha | ID Acc | OOD Acc | Overall | 说明 |")
print("|-------|--------|---------|---------|------|")
for r in ensemble_results:
    if r['alpha'] == 0.0:
        note = "纯零样本"
    elif r['alpha'] == 1.0:
        note = "纯LR-RGDA"
    elif r['alpha'] == best_ensemble['alpha']:
        note = "★最佳"
    else:
        note = ""
    print(f"| {r['alpha']:.1f}   | {r['id_acc']:.1f}%  | {r['ood_acc']:.1f}%   | {r['avg_acc']:.1f}%    | {note} |")

print("\n观察:")
print(f"  1. α增加 → ID性能提升 (LR-RGDA贡献增加)")
print(f"  2. α增加 → OOD性能下降 (零样本权重降低)")
print(f"  3. 最佳α={best_ensemble['alpha']} 达到ID/OOD平衡")

# [16. 详细分析：阈值对自适应路由的影响]
print("\n" + "=" * 80)
print("自适应路由的阈值敏感性分析")
print("=" * 80)

print("\n| 阈值  | ID Acc | OOD Acc | Overall | ID→Ensemble | OOD→Zeroshot |")
print("|-------|--------|---------|---------|-------------|---------------|")
for r in routing_results:
    marker = " ★" if r['threshold'] == best_routing['threshold'] else ""
    id_to_e = r['id_to_ensemble']
    id_total = r['id_to_ensemble'] + r['id_to_zeroshot']
    ood_to_z = r['ood_to_zeroshot']
    ood_total = r['ood_to_ensemble'] + r['ood_to_zeroshot']
    print(f"| {r['threshold']:.3f}{marker} | {r['id_acc']:.1f}%  | {r['ood_acc']:.1f}%   | {r['avg_acc']:.1f}%    | "
          f"{id_to_e}/{id_total}     | {ood_to_z}/{ood_total}       |")

print("\n观察:")
print(f"  1. 阈值过低 → 所有样本被判定为OOD → 退化为零样本")
print(f"  2. 阈值过高 → 所有样本被判定为ID → 退化为固定集成")
print(f"  3. 最佳阈值={best_routing['threshold']} 能有效区分ID和OOD")

# [17. 附录：修改配置重新测试]
# 回到第2节，修改ID_DATASETS和OOD_DATASETS，然后重新运行脚本

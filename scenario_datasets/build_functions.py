import torch
import random
from .utils import CustomConcatDataset, DatasetWrapper
from .oxford_pets import OxfordPets
from .eurosat import EuroSAT
from .sun397 import SUN397
from .caltech101 import Caltech101
from .dtd import DescribableTextures
from .aircraft import FGVCAircraft
from .food101 import Food101
from .flowers import OxfordFlowers
from .stanford_cars import StanfordCars
from .mnist import MNIST


dataset_list = {
                "aircraft": FGVCAircraft,
                "caltech101": Caltech101,
                "dtd": DescribableTextures,
                "eurosat": EuroSAT,
                "flowers": OxfordFlowers,
                "food101": Food101,
                "mnist": MNIST,
                "oxford_pets": OxfordPets,
                "stanford_cars": StanfordCars,
                "sun397": SUN397
                }


def build_cur_task_data_loader(root, dataset_name, transform_train, transform_test, num_shots, batch_size, num_workers):
    print(dataset_name)
    dataset = dataset_list[dataset_name](root, num_shots)
    train_set = dataset.train_x
    test_set = dataset.test
    classnames = dataset.classnames

    train_loader = torch.utils.data.DataLoader(
        DatasetWrapper(train_set, transform=transform_train),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True
    )

    train_loader4updating = torch.utils.data.DataLoader(
        DatasetWrapper(train_set, transform=transform_test),
        batch_size=256,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        DatasetWrapper(test_set, transform=transform_test),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True
    )

    return train_loader, train_loader4updating, test_loader, classnames


def build_TAIL_testloader(root, dataset_sequence, transform_test, batch_size, num_workers, max_num_per_dataset=None):
    TAIL_testset_list = []
    merged_classnames = []
    indices = []
    offset = 0
    for dataset_name in dataset_sequence:
        dataset = dataset_list[dataset_name](root, -1)
        test_set = dataset.test
        
        # 如果指定了最大样本数，则对测试集进行采样
        if max_num_per_dataset is not None and len(test_set) > max_num_per_dataset:
            sampled_indices = random.sample(range(len(test_set)), max_num_per_dataset)
            test_set = [test_set[i] for i in sampled_indices]
            print(f"从数据集 {dataset_name} 的测试集中随机采样了 {max_num_per_dataset} 个样本（原始样本数：{len(dataset.test)}）")
        
        TAIL_testset_list.append(test_set)
        merged_classnames += dataset.classnames
        indices.append(offset)
        offset += len(dataset.classnames)

    test_dataset_instances = [DatasetWrapper(dataset, transform=transform_test) for dataset in TAIL_testset_list]

    TAIL_testset = CustomConcatDataset(test_dataset_instances, indices)

    test_loader = torch.utils.data.DataLoader(TAIL_testset,
                                              batch_size=batch_size,
                                              num_workers=num_workers,
                                              shuffle=False,
                                              pin_memory=True
                                              )

    return test_loader, merged_classnames, indices

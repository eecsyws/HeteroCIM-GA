"""
准确率评估模块
根据配置评估模型准确率
"""

import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from typing import List, Optional

from config import PathConfig, InferenceConfig, HardwareConfig
from core.model_builder import build_vit_model, expand_group_config_to_layer_config, get_layer_names


# 全局数据加载器（避免重复创建）
_test_loader = None
_test_dataset = None


def get_test_loader():
    """获取测试数据加载器（单例模式）"""
    global _test_loader, _test_dataset

    if _test_loader is None:
        # 数据预处理
        transform = transforms.Compose([
            transforms.Resize(InferenceConfig.IMG_SIZE, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(InferenceConfig.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        _test_dataset = datasets.ImageFolder(PathConfig.TEST_DIR, transform=transform)
        _test_loader = DataLoader(
            _test_dataset,
            batch_size=InferenceConfig.BATCH_SIZE,
            shuffle=False,
            num_workers=InferenceConfig.NUM_WORKERS,
            pin_memory=True
        )

    return _test_loader, _test_dataset


def evaluate_model(model, test_loader, device):
    """
    评估模型准确率

    参数:
        model: nn.Module, 待评估模型
        test_loader: DataLoader, 测试数据加载器
        device: torch.device, 运行设备

    返回:
        accuracy: float, Top-1准确率（百分比）
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy


def evaluate_accuracy_from_config(
    config_indices: List[int],
    device: Optional[torch.device] = None,
    verbose: bool = False,
    seed: int = 42
) -> float:
    """
    根据96维配置评估准确率（PatchEmbed和Head保持FP32）

    参数:
        config_indices: List[int], 96维配置索引
        device: Optional[torch.device], 运行设备
        verbose: bool, 是否打印详细信息
        seed: int, 随机种子（用于噪声注入的可重复性）

    返回:
        accuracy: float, Top-1准确率（百分比）
    """
    # 设置PyTorch随机种子（噪声注入使用torch.randn_like）
    # 不设置Python的random模块，避免影响GA的随机操作
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 启用确定性模式，确保可重复性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    layer_names = get_layer_names()

    if len(config_indices) != len(layer_names):
        raise ValueError(f"config_indices length must be {len(layer_names)}, got {len(config_indices)}")

    # 检查配置索引范围
    for i, idx in enumerate(config_indices):
        if not (0 <= idx <= HardwareConfig.MAX_CONFIG_INDEX):
            raise ValueError(f"config_indices[{i}]={idx} out of range [0, {HardwareConfig.MAX_CONFIG_INDEX}]")

    # 构建模型
    model, device = build_vit_model(config_indices, device=device)

    # 获取测试数据
    test_loader, test_dataset = get_test_loader()

    # 评估
    accuracy = evaluate_model(model, test_loader, device)

    if verbose:
        dataset_size = len(test_dataset)
        print(f"[Accuracy Eval] Test Accuracy: {accuracy:.2f}% on {dataset_size} images")

    return accuracy


def evaluate_accuracy_from_group_config(
    group_config: List[int],
    device: Optional[torch.device] = None,
    verbose: bool = False,
    seed: int = 42
) -> float:
    """
    根据8维group配置评估准确率（PatchEmbed和Head保持FP32）

   _config: List[int 参数:
        group], 8维配置索引
            [Q, K, V, QK, AV, OutputLinear, FC1, FC2]
        device: Optional[torch.device], 运行设备
        verbose: bool, 是否打印详细信息
        seed: int, 随机种子（用于噪声注入的可重复性）

    返回:
        accuracy: float, Top-1准确率（百分比）
    """
    if len(group_config) != 8:
        raise ValueError(f"group_config length must be 8, got {len(group_config)}")

    # 检查配置索引范围
    for i, idx in enumerate(group_config):
        if not (0 <= idx <= HardwareConfig.MAX_CONFIG_INDEX):
            raise ValueError(f"group_config[{i}]={idx} out of range [0, {HardwareConfig.MAX_CONFIG_INDEX}]")

    # 展开为96维
    config_indices = expand_group_config_to_layer_config(group_config)

    # 调用96维评估函数
    return evaluate_accuracy_from_config(config_indices, device, verbose, seed)


if __name__ == "__main__":
    # 测试代码
    import random

    print("=== Accuracy Evaluation Test ===\n")

    # 设置随机种子
    random.seed(42)

    # 测试96维配置
    layer_names = get_layer_names()
    config_indices = [random.randint(0, HardwareConfig.MAX_CONFIG_INDEX) for _ in range(len(layer_names))]

    print(f"Testing with random 96-dim config (first 10): {config_indices[:10]}")
    acc1 = evaluate_accuracy_from_config(config_indices, verbose=True)
    print(f"Accuracy: {acc1:.2f}%")

    # 测试8维配置
    print("\n" + "="*50)
    group_config = [random.randint(0, HardwareConfig.MAX_CONFIG_INDEX) for _ in range(8)]
    print(f"Testing with random 8-dim group config: {group_config}")

    acc2 = evaluate_accuracy_from_group_config(group_config, verbose=True)
    print(f"Accuracy: {acc2:.2f}%")

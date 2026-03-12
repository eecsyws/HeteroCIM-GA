"""
测试随机配置准确率
随机生成96维配置向量，测试模型在该配置下的准确率
"""

import os
import sys
import random
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import HardwareConfig, ViTConfig, PathConfig, InferenceConfig
from core.model_builder import build_vit_model, get_layer_names


def setup_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_test_loader():
    """获取测试数据加载器"""
    transform = transforms.Compose([
        transforms.Resize(InferenceConfig.IMG_SIZE, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(InferenceConfig.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = datasets.ImageFolder(PathConfig.TEST_DIR, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=InferenceConfig.BATCH_SIZE,
        shuffle=False,
        num_workers=InferenceConfig.NUM_WORKERS,
        pin_memory=True
    )

    return test_loader, test_dataset


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
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy


def test_random_config(seed=None, verbose=True):
    """
    测试随机生成的96维配置

    参数:
        seed: int, 随机种子（可选）
        verbose: bool, 是否打印详细信息

    返回:
        accuracy: float, 准确率
        config: list, 生成的配置
    """
    # 设置随机种子
    if seed is not None:
        setup_seed(seed)

    # 获取层名称
    layer_names = get_layer_names()
    num_layers = len(layer_names)

    if verbose:
        print(f"Number of layers: {num_layers}")

    # 随机生成96维配置
    config = [random.randint(0, HardwareConfig.MAX_CONFIG_INDEX) for _ in range(num_layers)]

    if verbose:
        print(f"Generated random config (first 10): {config[:10]}")

    # 构建模型
    device = torch.device("cuda" if torch.cuda.is_available() and InferenceConfig.USE_CUDA else "cpu")

    if verbose:
        print(f"Building model on device: {device}")

    try:
        model, device = build_vit_model(config, device=device)
    except Exception as e:
        print(f"Error building model: {e}")
        return None, None

    # 获取测试数据
    try:
        test_loader, test_dataset = get_test_loader()
    except Exception as e:
        print(f"Error loading test data: {e}")
        print("Please check PathConfig.TEST_DIR in config/global_config.py")
        return None, None

    if verbose:
        print(f"Test dataset size: {len(test_dataset)}")

    # 评估
    accuracy = evaluate_model(model, test_loader, device)

    if verbose:
        print(f"Accuracy: {accuracy:.2f}%")

    return accuracy, config


def test_uniform_config(config_index, seed=None, verbose=True):
    """
    测试所有层使用相同配置的准确率

    参数:
        config_index: int, 配置索引（0-23）
        seed: int, 随机种子（用于噪声注入）
        verbose: bool, 是否打印详细信息

    返回:
        accuracy: float, 准确率
    """
    # 验证config_index
    if not (0 <= config_index <= HardwareConfig.MAX_CONFIG_INDEX):
        raise ValueError(f"config_index must be in range [0, {HardwareConfig.MAX_CONFIG_INDEX}]")

    # 设置随机种子
    if seed is not None:
        setup_seed(seed)

    # 获取层名称
    layer_names = get_layer_names()
    num_layers = len(layer_names)

    # 所有层使用相同配置
    config = [config_index] * num_layers

    quant_bits, nvm_bits = HardwareConfig.CONFIG_TABLE[config_index]

    if verbose:
        print(f"Number of layers: {num_layers}")
        print(f"Config index: {config_index}")
        print(f"Quant bits: {quant_bits}, NVM bits: {nvm_bits}")

    # 构建模型
    device = torch.device("cuda" if torch.cuda.is_available() and InferenceConfig.USE_CUDA else "cpu")

    if verbose:
        print(f"Building model on device: {device}")

    try:
        model, device = build_vit_model(config, device=device)
    except Exception as e:
        print(f"Error building model: {e}")
        return None

    # 获取测试数据
    try:
        test_loader, test_dataset = get_test_loader()
    except Exception as e:
        print(f"Error loading test data: {e}")
        print("Please check PathConfig.TEST_DIR in config/global_config.py")
        return None

    if verbose:
        print(f"Test dataset size: {len(test_dataset)}")

    # 评估
    accuracy = evaluate_model(model, test_loader, device)

    if verbose:
        print(f"Accuracy: {accuracy:.2f}%")

    return accuracy


def print_config_table():
    """打印配置表"""
    print("\n" + "=" * 60)
    print("Config Table:")
    print("-" * 60)
    print(f"{'Index':<6} | {'Quant Bits':<12} | {'NVM Bits':<10} | {'Description'}")
    print("-" * 60)

    for idx, (quant_bits, nvm_bits) in enumerate(HardwareConfig.CONFIG_TABLE):
        sram_bits = quant_bits - nvm_bits
        desc = f"SRAM={sram_bits}, NVM={nvm_bits}"
        print(f"{idx:<6} | {quant_bits:<12} | {nvm_bits:<10} | {desc}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Test ViT-CIM accuracy with random/uniform config")

    # 随机配置模式
    parser.add_argument("--random", action="store_true",
                        help="Test with random 96-dim config")

    # 统一配置模式
    parser.add_argument("--uniform", type=int, metavar="CONFIG_INDEX",
                        help="Test with uniform config (all layers use same config_index)")

    # 随机种子
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    # 打印配置表
    parser.add_argument("--show-config-table", action="store_true",
                        help="Show config table and exit")

    args = parser.parse_args()

    # 打印配置表
    if args.show_config_table:
        print_config_table()
        return

    # 默认模式：随机配置
    if args.random:
        print("=" * 60)
        print("Testing Random Configuration")
        print("=" * 60)
        accuracy, config = test_random_config(seed=args.seed)
        if accuracy is not None:
            print(f"\n>>> Final Accuracy: {accuracy:.2f}%")
    elif args.uniform is not None:
        print("=" * 60)
        print(f"Testing Uniform Configuration (config_index={args.uniform})")
        print("=" * 60)
        accuracy = test_uniform_config(config_index=args.uniform, seed=args.seed)
        if accuracy is not None:
            print(f"\n>>> Final Accuracy: {accuracy:.2f}%")
    else:
        # 默认：打印帮助信息和配置表
        print("Please specify --random or --uniform")
        print("\nExamples:")
        print("  python test_random_config_accuracy.py --random --seed 42")
        print("  python test_random_config_accuracy.py --uniform 0 --seed 42")
        print("  python test_random_config_accuracy.py --show-config-table")
        print("\n" + "-" * 60)
        print_config_table()


if __name__ == "__main__":
    main()

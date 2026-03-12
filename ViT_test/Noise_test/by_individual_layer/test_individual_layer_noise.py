"""
测试每个单独层的噪声敏感性
对72层中的每一层单独进行INT8量化+噪声注入，其余层保持FP32
PatchEmbed和Head保持FP32不测试
QK和AV无权重，不注入噪声
测试噪声强度: 0.05, 0.1, 0.15, 0.2, 0.25
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..'))

import torch
import random
import numpy as np
import csv
import torch.nn as nn
from typing import List

from config import PathConfig, InferenceConfig, ViTConfig
from core.model_builder import get_layer_names
from evaluators.accuracy_evaluator import get_test_loader, evaluate_model


def set_random_seed(seed):
    """设置所有随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# 测试的噪声强度
NOISE_SIGMA_LIST = [0.05, 0.1, 0.15, 0.2, 0.25]

# 固定量化位宽为INT8
QUANT_BITS = 8
NVM_BITS = 8  # 全部位都注入噪声


def build_single_layer_noise_model(layer_name, noise_sigma, device):
    """
    构建只对单个层注入噪声的模型，其余层保持FP32

    参数:
        layer_name: str, 要注入噪声的层名称
        noise_sigma: float, 噪声强度
        device: torch.device, 运行设备

    返回:
        model: nn.Module, 构建好的模型
    """
    import timm
    from core.quantization import FakeQuantWrapper, FakeQuantAttention
    from timm.models.vision_transformer import Attention as TimmAttention

    # 加载基础模型
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=ViTConfig.NUM_CLASSES)

    # 加载预训练权重
    checkpoint = torch.load(PathConfig.MODEL_PATH, map_location=device)
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()

    # 设置全局噪声强度
    from config import HardwareConfig
    HardwareConfig.NOISE_SIGMA = noise_sigma

    def replace_layers(module, prefix=""):
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            # 跳过attn.qkv Linear，留给FakeQuantAttention处理
            if isinstance(child, nn.Linear) and full_name.endswith("attn.qkv"):
                continue

            if isinstance(child, TimmAttention):
                # 检查该attention块的子操作是否需要注入噪声
                attn_config = {}
                has_noise = False
                for sub_tag in ['.q', '.k', '.v', '.proj']:
                    sub_name = f"{full_name}{sub_tag}"
                    if sub_name == layer_name:
                        attn_config[sub_tag] = {'quant_bits': QUANT_BITS, 'nvm_bits': NVM_BITS}
                        has_noise = True
                    else:
                        attn_config[sub_tag] = {'quant_bits': 32, 'nvm_bits': 0}

                # QK和AV不注入噪声
                attn_config['.qk'] = {'quant_bits': 32, 'nvm_bits': 0}
                attn_config['.av'] = {'quant_bits': 32, 'nvm_bits': 0}

                if has_noise:
                    setattr(
                        module, name,
                        FakeQuantAttention(
                            child,
                            full_name=full_name,
                            w_signed=True,
                            a_signed=True,
                            w_per_channel=True,
                            layer_config=attn_config,
                            attn_qkv_quant=True,
                            attn_matmul_quant=False,  # QK和AV不量化
                        )
                    )

            elif isinstance(child, (nn.Linear, nn.Conv2d)):
                if full_name == layer_name:
                    setattr(
                        module, name,
                        FakeQuantWrapper(
                            child,
                            quant_w=True,
                            quant_a=True,
                            w_signed=True,
                            a_signed=True,
                            w_per_channel=True,
                            a_per_channel=False,
                            num_bits=QUANT_BITS,
                            noise_bits=NVM_BITS,
                            use_static_activation=False,
                            full_name=full_name,
                        )
                    )
            else:
                replace_layers(child, full_name)

    replace_layers(model)

    return model


def run_individual_layer_noise_test():
    """运行单层噪声敏感性测试"""
    print("="*70)
    print("Individual Layer Noise Sensitivity Test")
    print("="*70)

    # 获取所有层名称，排除PatchEmbed、Head、QK、AV
    all_layer_names = get_layer_names()
    layer_names = [name for name in all_layer_names
                   if name != 'patch_embed.proj' and name != 'head'
                   and '.qk' not in name and '.av' not in name]

    print(f"Total layers to test: {len(layer_names)} (excluding PatchEmbed, Head, QK, AV)")
    print(f"Quantization: INT{QUANT_BITS} (all bits with noise)")
    print(f"Noise sigma: {NOISE_SIGMA_LIST}")
    print("="*70)

    # 设置全局随机种子
    GLOBAL_SEED = 42
    set_random_seed(GLOBAL_SEED)
    print(f"\nGlobal random seed: {GLOBAL_SEED}")

    device = torch.device("cuda" if torch.cuda.is_available() and InferenceConfig.USE_CUDA else "cpu")
    print(f"Device: {device}")

    test_loader, test_dataset = get_test_loader()
    print(f"Test dataset: {len(test_dataset)} images")

    results = []
    total_tests = len(layer_names) * len(NOISE_SIGMA_LIST)
    current_test = 0

    for layer_idx, layer_name in enumerate(layer_names):
        print(f"\n{'='*70}")
        print(f"[{layer_idx+1}/{len(layer_names)}] Layer: {layer_name}")
        print(f"{'='*70}")

        for noise_sigma in NOISE_SIGMA_LIST:
            current_test += 1
            print(f"  [{current_test}/{total_tests}] σ={noise_sigma:.2f}...", end=" ", flush=True)

            # 重新设置随机种子，确保模型初始化一致
            set_random_seed(GLOBAL_SEED)

            model = build_single_layer_noise_model(layer_name, noise_sigma, device)
            accuracy = evaluate_model(model, test_loader, device)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"acc={accuracy:.2f}%")
            results.append({
                'layer_index': layer_idx,
                'layer_name': layer_name,
                'noise_sigma': noise_sigma,
                'accuracy': accuracy,
            })

    # 保存CSV
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'individual_layer_noise_results.csv')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['layer_index', 'layer_name', 'noise_sigma', 'accuracy'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'='*70}")
    print(f"Results saved to: {csv_path}")
    print(f"Total tests: {len(results)}")
    print(f"{'='*70}")

    return results, csv_path


if __name__ == "__main__":
    results, csv_path = run_individual_layer_noise_test()
    print(f"\nDone. Run plot_individual_noise_results.py to generate visualizations.")

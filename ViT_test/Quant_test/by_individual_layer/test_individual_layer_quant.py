"""
测试每个单独层的量化敏感性
对96层中的每一层单独进行量化测试，其余层保持FP32
PatchEmbed和Head保持FP32不测试
测试INT8/INT7/INT6/INT5/INT4/INT3量化
不注入噪声
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..'))

import torch
import csv
import torch.nn as nn
from typing import List

from config import PathConfig, InferenceConfig, ViTConfig
from core.model_builder import get_layer_names
from evaluators.accuracy_evaluator import get_test_loader, evaluate_model


# 测试的量化位宽
QUANT_BITS_LIST = [8, 7, 6, 5, 4, 3]


def build_single_layer_quant_model(layer_name, quant_bits, device):
    """
    构建只对单个层量化的模型，其余层保持FP32

    参数:
        layer_name: str, 要量化的层名称
        quant_bits: int, 量化位宽
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
    try:
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
    except Exception as e:
        print(f"Warning: Failed to load model: {e}")

    model = model.to(device)
    model.eval()

    def replace_layers(module, prefix=""):
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            # 跳过attn.qkv Linear，留给FakeQuantAttention处理
            if isinstance(child, nn.Linear) and full_name.endswith("attn.qkv"):
                continue

            if isinstance(child, TimmAttention):
                # 检查该attention块的子操作是否需要量化
                attn_config = {}
                has_quant = False
                for sub_tag in ['.q', '.k', '.v', '.qk', '.av', '.proj']:
                    sub_name = f"{full_name}{sub_tag}"
                    if sub_name == layer_name:
                        attn_config[sub_tag] = {'quant_bits': quant_bits, 'nvm_bits': 0}
                        has_quant = True
                    else:
                        attn_config[sub_tag] = {'quant_bits': 32, 'nvm_bits': 0}

                if has_quant:
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
                            attn_matmul_quant=True,
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
                            num_bits=quant_bits,
                            noise_bits=0,
                            use_static_activation=False,
                            full_name=full_name,
                        )
                    )
            else:
                replace_layers(child, full_name)

    replace_layers(model)
    return model


def run_individual_layer_quant_test():
    """运行单层量化敏感性测试"""
    print("="*70)
    print("Individual Layer Quantization Sensitivity Test")
    print("="*70)

    # 获取所有层名称，排除PatchEmbed和Head
    all_layer_names = get_layer_names()
    layer_names = [name for name in all_layer_names
                   if name != 'patch_embed.proj' and name != 'head']

    print(f"Total layers to test: {len(layer_names)} (excluding PatchEmbed and Head)")
    print(f"Quantization bits: {QUANT_BITS_LIST}")
    print(f"No noise injection")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() and InferenceConfig.USE_CUDA else "cpu")
    print(f"\nDevice: {device}")

    test_loader, test_dataset = get_test_loader()
    print(f"Test dataset: {len(test_dataset)} images")

    results = []
    total_tests = len(layer_names) * len(QUANT_BITS_LIST)
    current_test = 0

    for layer_idx, layer_name in enumerate(layer_names):
        print(f"\n{'='*70}")
        print(f"[{layer_idx+1}/{len(layer_names)}] Layer: {layer_name}")
        print(f"{'='*70}")

        for quant_bits in QUANT_BITS_LIST:
            current_test += 1
            print(f"  [{current_test}/{total_tests}] INT{quant_bits}...", end=" ", flush=True)

            model = build_single_layer_quant_model(layer_name, quant_bits, device)
            accuracy = evaluate_model(model, test_loader, device)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"acc={accuracy:.2f}%")
            results.append({
                'layer_index': layer_idx,
                'layer_name': layer_name,
                'quant_bits': quant_bits,
                'accuracy': accuracy,
            })

    # 保存CSV
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'individual_layer_quant_results.csv')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['layer_index', 'layer_name', 'quant_bits', 'accuracy'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'='*70}")
    print(f"Results saved to: {csv_path}")
    print(f"Total tests: {len(results)}")
    print(f"{'='*70}")

    return results, csv_path


if __name__ == "__main__":
    results, csv_path = run_individual_layer_quant_test()
    print(f"\nDone. Run plot_individual_results.py to generate visualizations.")

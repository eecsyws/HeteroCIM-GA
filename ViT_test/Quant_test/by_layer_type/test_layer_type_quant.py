"""
测试不同层类型的量化敏感性
每次只对一种层类型进行量化，其余层保持FP32
测试INT8/INT7/INT6/INT5/INT4/INT3量化
不注入噪声
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..'))

import torch
import csv
import torch.nn as nn
from typing import List, Dict

from config import PathConfig, InferenceConfig, ViTConfig
from evaluators.accuracy_evaluator import get_test_loader, evaluate_model


# 定义层类型分组（不包括PatchEmbed和Head，它们保持FP32）
LAYER_TYPE_GROUPS = {
    'QLinear': [f'blocks.{i}.attn.q' for i in range(12)],
    'KLinear': [f'blocks.{i}.attn.k' for i in range(12)],
    'VLinear': [f'blocks.{i}.attn.v' for i in range(12)],
    'QK^T': [f'blocks.{i}.attn.qk' for i in range(12)],
    'AV': [f'blocks.{i}.attn.av' for i in range(12)],
    'OutputLinear': [f'blocks.{i}.attn.proj' for i in range(12)],
    'FC1': [f'blocks.{i}.mlp.fc1' for i in range(12)],
    'FC2': [f'blocks.{i}.mlp.fc2' for i in range(12)],
}

# 测试的量化位宽
QUANT_BITS_LIST = [8, 7, 6, 5, 4, 3]


def build_selective_quant_model(layer_type, quant_bits, device):
    """
    构建只对特定层类型量化的模型，其余层保持FP32
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

    # 要量化的层名称集合
    target_layers = set(LAYER_TYPE_GROUPS[layer_type])

    def replace_layers(module, prefix=""):
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            # 跳过attn.qkv Linear，留给FakeQuantAttention处理
            if isinstance(child, nn.Linear) and full_name.endswith("attn.qkv"):
                continue

            if isinstance(child, TimmAttention):
                # 检查该attention块是否有子操作需要量化
                attn_config = {}
                has_quant = False
                for sub_tag in ['.q', '.k', '.v', '.qk', '.av', '.proj']:
                    sub_name = f"{full_name}{sub_tag}"
                    if sub_name in target_layers:
                        attn_config[sub_tag] = {'quant_bits': quant_bits, 'nvm_bits': 0}
                        has_quant = True
                    else:
                        # 32位 = 不量化（fake_quant_tensor对>16位直接返回原值）
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
                if full_name in target_layers:
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


def run_layer_type_quant_test():
    """运行层类型量化敏感性测试"""
    print("="*70)
    print("Layer Type Quantization Sensitivity Test")
    print("="*70)
    print(f"Layer types: {list(LAYER_TYPE_GROUPS.keys())}")
    print(f"Quantization bits: {QUANT_BITS_LIST}")
    print(f"No noise injection")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() and InferenceConfig.USE_CUDA else "cpu")
    print(f"\nDevice: {device}")

    test_loader, test_dataset = get_test_loader()
    print(f"Test dataset: {len(test_dataset)} images")

    results = []

    for layer_type in LAYER_TYPE_GROUPS.keys():
        print(f"\n{'='*70}")
        print(f"Layer Type: {layer_type} ({len(LAYER_TYPE_GROUPS[layer_type])} layers)")
        print(f"{'='*70}")

        for quant_bits in QUANT_BITS_LIST:
            print(f"  INT{quant_bits}...", end=" ", flush=True)

            model = build_selective_quant_model(layer_type, quant_bits, device)
            accuracy = evaluate_model(model, test_loader, device)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"acc={accuracy:.2f}%")
            results.append({
                'layer_type': layer_type,
                'num_layers': len(LAYER_TYPE_GROUPS[layer_type]),
                'quant_bits': quant_bits,
                'accuracy': accuracy,
            })

    # 保存CSV
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'layer_type_quant_results.csv')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['layer_type', 'num_layers', 'quant_bits', 'accuracy'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'='*70}")
    print(f"Results saved to: {csv_path}")
    print(f"Total tests: {len(results)}")
    print(f"{'='*70}")

    return results, csv_path


if __name__ == "__main__":
    results, csv_path = run_layer_type_quant_test()
    print(f"\nDone. Run plot_quant_results.py to generate visualizations.")

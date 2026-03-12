"""
模型构建模块
负责根据配置构建ViT模型
"""

import torch
import torch.nn as nn
import timm
from typing import List, Tuple, Optional

from config import HardwareConfig, ViTConfig


def get_layer_names() -> List[str]:
    """
    获取ViT-tiny所有可搜索层的名称（不含PatchEmbed和Head）

    返回:
        layer_names: List[str], 96个层的名称列表
    """
    layer_names = []

    # 12个Transformer Blocks × 8个子操作 = 96层
    for i in range(ViTConfig.NUM_BLOCKS):
        # Attention层
        layer_names.extend([
            f"blocks.{i}.attn.q",
            f"blocks.{i}.attn.k",
            f"blocks.{i}.attn.v",
            f"blocks.{i}.attn.qk",   # QK^T矩阵乘
            f"blocks.{i}.attn.av",   # AV矩阵乘
            f"blocks.{i}.attn.proj",
        ])

        # MLP层
        layer_names.extend([
            f"blocks.{i}.mlp.fc1",
            f"blocks.{i}.mlp.fc2",
        ])

    # 注意：PatchEmbed和Head不参与GA搜索，始终使用FP32
    return layer_names


def expand_group_config_to_layer_config(group_config: List[int]) -> List[int]:
    """
    将8维group配置展开为96维layer配置

    参数:
        group_config: List[int], 长度为8的配置列表
            [Q, K, V, QK, AV, OutputLinear, FC1, FC2]

    返回:
        layer_config: List[int], 长度为96的配置列表
    """
    from config import LayerGroupConfig

    layer_names = get_layer_names()
    layer_config = []

    for layer_name in layer_names:
        group_id = LayerGroupConfig.LAYER_TO_GROUP[layer_name]
        config_idx = group_config[group_id]
        layer_config.append(config_idx)

    return layer_config


def build_vit_model(
    config_indices: List[int],
    model_path: Optional[str] = None,
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, torch.device]:
    """
    根据配置索引构建ViT模型
    PatchEmbed和Head始终保持FP32

    参数:
        config_indices: List[int], 96维配置索引列表（不含PatchEmbed和Head）
        model_path: Optional[str], 预训练模型路径
        device: Optional[torch.device], 运行设备

    返回:
        model: nn.Module, 构建好的模型
        device: torch.device, 实际使用的设备
    """
    from config import PathConfig, InferenceConfig

    # 确定设备
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() and InferenceConfig.USE_CUDA else "cpu")

    # 加载基础模型
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=ViTConfig.NUM_CLASSES)

    # 加载预训练权重
    if model_path is None:
        model_path = PathConfig.MODEL_PATH

    try:
        checkpoint = torch.load(model_path, map_location=device)

        # 处理不同的checkpoint格式
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                # 格式: {'epoch': ..., 'state_dict': ..., 'best_acc': ..., 'args': ...}
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                # 格式: {'model': state_dict, ...}
                state_dict = checkpoint['model']
            else:
                # 直接是state_dict
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
    except FileNotFoundError:
        print(f"Warning: Model file not found at {model_path}, using random initialization")
    except Exception as e:
        print(f"Warning: Failed to load model: {e}, using random initialization")

    model = model.to(device)
    model.eval()

    # 应用量化和噪声注入
    model = apply_quantization_and_noise(model, config_indices, device)

    return model, device


def apply_quantization_and_noise(
    model: nn.Module,
    config_indices: List[int],
    device: torch.device
) -> nn.Module:
    """
    根据配置索引应用量化和噪声注入
    PatchEmbed和Head保持FP32不替换

    参数:
        model: nn.Module, 基础模型
        config_indices: List[int], 96维配置索引（不含PatchEmbed和Head）
        device: torch.device, 运行设备

    返回:
        model: nn.Module, 应用量化和噪声后的模型
    """
    from .quantization import FakeQuantWrapper, FakeQuantAttention
    from timm.models.vision_transformer import Attention as TimmAttention

    layer_names = get_layer_names()

    if len(config_indices) != len(layer_names):
        raise ValueError(f"config_indices length must be {len(layer_names)}, got {len(config_indices)}")

    # 构建层名到配置的映射
    layer_config_dict = {}
    for layer_name, config_idx in zip(layer_names, config_indices):
        quant_bits, nvm_bits = HardwareConfig.CONFIG_TABLE[config_idx]
        layer_config_dict[layer_name] = {
            'quant_bits': quant_bits,
            'nvm_bits': nvm_bits,
            'config_idx': config_idx
        }

    # 量化配置
    w_signed = True  # INT8
    a_signed = True  # INT8
    w_per_channel = True
    a_per_channel = False

    # 递归替换模型中的层
    def replace_layers(module, prefix=""):
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            # 跳过attn.qkv，留给FakeQuantAttention处理
            if isinstance(child, nn.Linear) and full_name.endswith("attn.qkv"):
                continue

            # 替换Attention模块
            if isinstance(child, TimmAttention):
                # 构建attention子操作的配置
                attn_config = {}
                for sub_tag in ['.q', '.k', '.v', '.qk', '.av', '.proj']:
                    sub_name = f"{full_name}{sub_tag}"
                    if sub_name in layer_config_dict:
                        attn_config[sub_tag] = layer_config_dict[sub_name]
                    else:
                        # 默认配置
                        attn_config[sub_tag] = {'quant_bits': 8, 'nvm_bits': 0}

                setattr(
                    module,
                    name,
                    FakeQuantAttention(
                        child,
                        full_name=full_name,
                        w_signed=w_signed,
                        a_signed=a_signed,
                        w_per_channel=w_per_channel,
                        layer_config=attn_config,
                        attn_qkv_quant=True,
                        attn_matmul_quant=True,
                    )
                )
            # 替换Linear和Conv2d
            elif isinstance(child, (nn.Linear, nn.Conv2d)):
                # 查找配置
                config = layer_config_dict.get(full_name, {'quant_bits': 8, 'nvm_bits': 0})

                setattr(
                    module,
                    name,
                    FakeQuantWrapper(
                        child,
                        quant_w=True,
                        quant_a=True,
                        w_signed=w_signed,
                        a_signed=a_signed,
                        w_per_channel=w_per_channel,
                        a_per_channel=a_per_channel,
                        num_bits=config['quant_bits'],
                        noise_bits=config['nvm_bits'],
                        use_static_activation=False,
                        full_name=full_name,
                    )
                )
            else:
                # 递归处理子模块
                replace_layers(child, full_name)

    replace_layers(model)

    return model


def get_layer_weight_shapes() -> dict:
    """
    获取ViT-tiny每层的权重形状信息（不含PatchEmbed和Head）

    返回:
        weight_shapes: dict, {layer_name: (num_weights, is_qk_or_av)}
    """
    layer_names = get_layer_names()
    weight_shapes = {}

    embed_dim = ViTConfig.EMBED_DIM
    mlp_dim = ViTConfig.MLP_DIM
    num_tokens = ViTConfig.NUM_TOKENS

    for layer_name in layer_names:
        # Transformer blocks only (PatchEmbed和Head已排除)
        parts = layer_name.split(".")
        if parts[2] == "attn":
            sub = parts[3]
            if sub in ["q", "k", "v", "proj"]:
                # Linear: [embed_dim, embed_dim]
                num_w = embed_dim * embed_dim
                weight_shapes[layer_name] = (num_w, False)
            elif sub in ["qk", "av"]:
                # QK/AV矩阵: [num_tokens, embed_dim]
                num_w = num_tokens * embed_dim
                weight_shapes[layer_name] = (num_w, True)  # 标记为特殊层

        elif parts[2] == "mlp":
            fc = parts[3]
            if fc == "fc1":
                num_w = embed_dim * mlp_dim
            elif fc == "fc2":
                num_w = mlp_dim * embed_dim
            weight_shapes[layer_name] = (num_w, False)

    return weight_shapes

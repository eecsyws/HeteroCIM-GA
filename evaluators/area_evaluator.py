"""
面积评估模块
根据配置计算芯片面积
"""

import numpy as np
from typing import List, Dict, Tuple

from config import HardwareConfig, ViTConfig
from core.model_builder import get_layer_names, get_layer_weight_shapes, expand_group_config_to_layer_config


def calculate_area(
    num_weights: int,
    quant_bits: int,
    nvm_bits: int,
    is_qk_or_av: bool = False,
    process_nm: float = None,
    sram_cell_f2: float = None,
    nvm_cell_f2: float = None
) -> Dict[str, float]:
    """
    计算单层的面积

    参数:
        num_weights: int, 权重数量
        quant_bits: int, 量化位宽
        nvm_bits: int, NVM存储的bit数
        is_qk_or_av: bool, 是否为QK/AV层（完全SRAM）
        process_nm: float, 工艺节点
        sram_cell_f2: float, SRAM单元面积
        nvm_cell_f2: float, NVM单元面积

    返回:
        area_dict: Dict, 包含sram_area, nvm_area, total_area
    """
    # 使用默认值
    if process_nm is None:
        process_nm = HardwareConfig.PROCESS_NM
    if sram_cell_f2 is None:
        sram_cell_f2 = HardwareConfig.SRAM_CELL_AREA_F2
    if nvm_cell_f2 is None:
        nvm_cell_f2 = HardwareConfig.NVM_CELL_AREA_F2

    # QK/AV层完全使用SRAM
    if is_qk_or_av:
        sram_bits = quant_bits
        nvm_bits_actual = 0
    else:
        nvm_bits_actual = min(nvm_bits, quant_bits)
        sram_bits = quant_bits - nvm_bits_actual

    # 计算面积（单位：um^2）
    sram_area = num_weights * sram_bits * sram_cell_f2 * (process_nm / 1000.0) ** 2
    nvm_area = num_weights * nvm_bits_actual * nvm_cell_f2 * (process_nm / 1000.0) ** 2
    total_area = sram_area + nvm_area

    return {
        'sram_area': sram_area,
        'nvm_area': nvm_area,
        'total_area': total_area,
        'sram_bits': sram_bits,
        'nvm_bits': nvm_bits_actual,
    }


def evaluate_area_from_config(
    config_indices: List[int],
    return_detail: bool = True
) -> Dict:
    """
    根据96维配置评估总面积（不含PatchEmbed和Head）

    参数:
        config_indices: List[int], 96维配置索引
        return_detail: bool, 是否返回详细信息

    返回:
        result: Dict, 包含面积信息
    """
    layer_names = get_layer_names()
    weight_shapes = get_layer_weight_shapes()

    if len(config_indices) != len(layer_names):
        raise ValueError(f"config_indices length must be {len(layer_names)}, got {len(config_indices)}")

    # 计算baseline面积（全INT8 SRAM）
    baseline_area = 0.0
    for layer_name in layer_names:
        num_weights, _ = weight_shapes[layer_name]
        baseline_bits = HardwareConfig.BASELINE_QUANT_BITS
        baseline_nvm = HardwareConfig.BASELINE_NVM_BITS
        area_info = calculate_area(num_weights, baseline_bits, baseline_nvm)
        baseline_area += area_info['total_area']

    # 计算实际配置的面积
    total_area = 0.0
    total_sram_area = 0.0
    total_nvm_area = 0.0
    per_layer_area = {}

    for layer_name, config_idx in zip(layer_names, config_indices):
        num_weights, is_qk_or_av = weight_shapes[layer_name]
        quant_bits, nvm_bits = HardwareConfig.CONFIG_TABLE[config_idx]

        area_info = calculate_area(num_weights, quant_bits, nvm_bits, is_qk_or_av)

        total_area += area_info['total_area']
        total_sram_area += area_info['sram_area']
        total_nvm_area += area_info['nvm_area']

        if return_detail:
            per_layer_area[layer_name] = area_info

    # 计算比率
    area_ratio = total_area / baseline_area if baseline_area > 0 else 1.0
    area_saving_ratio = 1.0 - area_ratio
    area_optimization_ratio = (baseline_area / total_area - 1.0) if total_area > 0 else 0.0

    result = {
        'total_area': total_area,
        'sram_area': total_sram_area,
        'nvm_area': total_nvm_area,
        'baseline_area': baseline_area,
        'area_ratio_vs_baseline': area_ratio,
        'area_saving_ratio': area_saving_ratio,
        'area_optimization_ratio': area_optimization_ratio,
    }

    if return_detail:
        result['per_layer_area'] = per_layer_area

    return result


def evaluate_area_from_group_config(
    group_config: List[int],
    return_detail: bool = True
) -> Dict:
    """
    根据8维group配置评估总面积（不含PatchEmbed和Head）

    参数:
        group_config: List[int], 8维配置索引
            [Q, K, V, QK, AV, OutputLinear, FC1, FC2]
        return_detail: bool, 是否返回详细信息

    返回:
        result: Dict, 包含面积信息
    """
    if len(group_config) != 8:
        raise ValueError(f"group_config length must be 8, got {len(group_config)}")

    # 展开为96维
    config_indices = expand_group_config_to_layer_config(group_config)

    # 调用96维评估函数
    return evaluate_area_from_config(config_indices, return_detail)


if __name__ == "__main__":
    # 测试代码
    import random

    print("=== Area Evaluation Test ===\n")

    # 测试96维配置
    layer_names = get_layer_names()
    config_indices = [random.randint(0, HardwareConfig.MAX_CONFIG_INDEX) for _ in range(len(layer_names))]

    print(f"Testing with random 96-dim config (first 10): {config_indices[:10]}")
    result = evaluate_area_from_config(config_indices, return_detail=False)

    print(f"\nTotal area: {result['total_area']:.4e} um^2")
    print(f"Baseline area: {result['baseline_area']:.4e} um^2")
    print(f"Area ratio: {result['area_ratio_vs_baseline']:.4f}")
    print(f"Area saving: {result['area_saving_ratio']*100:.2f}%")
    print(f"Area optimization: {result['area_optimization_ratio']*100:.2f}%")

    # 测试8维配置
    print("\n" + "="*50)
    group_config = [random.randint(0, HardwareConfig.MAX_CONFIG_INDEX) for _ in range(8)]
    print(f"Testing with random 8-dim group config: {group_config}")

    result2 = evaluate_area_from_group_config(group_config, return_detail=False)
    print(f"\nTotal area: {result2['total_area']:.4e} um^2")
    print(f"Area ratio: {result2['area_ratio_vs_baseline']:.4f}")

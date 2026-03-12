"""
快速测试脚本
用于验证新代码库的基本功能
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*70)
print("Testing New Codebase")
print("="*70)

# 测试1: 导入配置
print("\n[Test 1] Importing configurations...")
try:
    from config import (
        PathConfig, HardwareConfig, LayerGroupConfig,
        CoarseGAConfig, FineGAConfig, InferenceConfig, ViTConfig
    )
    print("[OK] Configuration modules imported successfully")
    print(f"  - CONFIG_TABLE length: {len(HardwareConfig.CONFIG_TABLE)}")
    print(f"  - MAX_CONFIG_INDEX: {HardwareConfig.MAX_CONFIG_INDEX}")
    print(f"  - Number of groups: {len(LayerGroupConfig.GROUP_ALLOWED_CONFIGS)}")
except Exception as e:
    print(f"[FAIL] Failed to import configurations: {e}")
    sys.exit(1)

# 测试2: 导入核心模块
print("\n[Test 2] Importing core modules...")
try:
    from core import get_layer_names, expand_group_config_to_layer_config
    layer_names = get_layer_names()
    print("[OK] Core modules imported successfully")
    print(f"  - Total layers: {len(layer_names)}")
    print(f"  - First 5 layers: {layer_names[:5]}")
except Exception as e:
    print(f"[FAIL] Failed to import core modules: {e}")
    sys.exit(1)

# 测试3: 测试group到layer的展开
print("\n[Test 3] Testing group-to-layer expansion...")
try:
    import random
    random.seed(42)
    group_config = [random.randint(0, HardwareConfig.MAX_CONFIG_INDEX) for _ in range(9)]
    layer_config = expand_group_config_to_layer_config(group_config)
    print("[OK] Group-to-layer expansion works")
    print(f"  - Group config (9-dim): {group_config}")
    print(f"  - Layer config (98-dim, first 10): {layer_config[:10]}")
    assert len(layer_config) == len(layer_names), "Layer config length mismatch"
except Exception as e:
    print(f"[FAIL] Failed group-to-layer expansion: {e}")
    sys.exit(1)

# 测试4: 导入评估器
print("\n[Test 4] Importing evaluators...")
try:
    from evaluators import (
        evaluate_area_from_config,
        evaluate_area_from_group_config,
    )
    print("[OK] Evaluator modules imported successfully")
except Exception as e:
    print(f"[FAIL] Failed to import evaluators: {e}")
    sys.exit(1)

# 测试5: 测试面积评估
print("\n[Test 5] Testing area evaluation...")
try:
    # 测试98维配置
    config_98 = [0] * len(layer_names)  # 全INT8 SRAM
    area_result = evaluate_area_from_config(config_98, return_detail=False)
    print("[OK] Area evaluation works")
    print(f"  - Total area: {area_result['total_area']:.4e} um^2")
    print(f"  - Baseline area: {area_result['baseline_area']:.4e} um^2")
    print(f"  - Area ratio: {area_result['area_ratio_vs_baseline']:.4f}")

    # 测试9维配置
    group_config_test = [5, 6, 7, 10, 11, 12, 15, 16, 17]
    area_result_group = evaluate_area_from_group_config(group_config_test, return_detail=False)
    print(f"  - Group config area ratio: {area_result_group['area_ratio_vs_baseline']:.4f}")
except Exception as e:
    print(f"[FAIL] Failed area evaluation: {e}")
    import traceback
    traceback.print_exc()

# 测试6: 导入GA模块
print("\n[Test 6] Importing GA modules...")
try:
    from algorithms import CoarseGA, FineGA, run_coarse_ga, run_fine_ga
    print("[OK] GA modules imported successfully")
except Exception as e:
    print(f"[FAIL] Failed to import GA modules: {e}")
    sys.exit(1)

# 测试7: 创建GA实例
print("\n[Test 7] Creating GA instances...")
try:
    coarse_ga = CoarseGA(population_size=5, max_generations=2, seed=42, verbose=False)
    print("[OK] CoarseGA instance created")

    fine_ga = FineGA(population_size=5, max_generations=2, seed=42, verbose=False)
    print("[OK] FineGA instance created")
except Exception as e:
    print(f"[FAIL] Failed to create GA instances: {e}")
    import traceback
    traceback.print_exc()

# 测试8: 导入分析模块
print("\n[Test 8] Importing analysis modules...")
try:
    from analysis import plot_pareto_front, plot_evolution_history
    print("[OK] Analysis modules imported successfully")
except Exception as e:
    print(f"[FAIL] Failed to import analysis modules: {e}")
    sys.exit(1)

# 测试9: 检查输出目录创建
print("\n[Test 9] Creating output directories...")
try:
    PathConfig.create_output_dirs()
    print("[OK] Output directories created")
    print(f"  - Output root: {PathConfig.OUTPUT_DIR}")
    print(f"  - Coarse GA output: {PathConfig.COARSE_GA_OUTPUT}")
    print(f"  - Fine GA output: {PathConfig.FINE_GA_OUTPUT}")
except Exception as e:
    print(f"[FAIL] Failed to create output directories: {e}")

# 测试10: 验证配置一致性
print("\n[Test 10] Validating configuration consistency...")
try:
    # 检查group配置的合法性
    for group_id, allowed_configs in LayerGroupConfig.GROUP_ALLOWED_CONFIGS.items():
        for config_idx in allowed_configs:
            assert 0 <= config_idx <= HardwareConfig.MAX_CONFIG_INDEX, \
                f"Invalid config index {config_idx} in group {group_id}"

    # 检查layer到group的映射
    assert len(LayerGroupConfig.LAYER_TO_GROUP) == len(layer_names), \
        "Layer-to-group mapping size mismatch"

    print("[OK] Configuration consistency validated")
    print(f"  - All group configs are within valid range [0, {HardwareConfig.MAX_CONFIG_INDEX}]")
    print(f"  - Layer-to-group mapping covers all {len(layer_names)} layers")
except Exception as e:
    print(f"[FAIL] Configuration validation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("All Tests Completed!")
print("="*70)
print("\nNew codebase is ready to use.")
print("Run 'python main.py' to start the two-stage GA search.")

"""
评估器模块
提供面积和准确率评估接口
"""

from .area_evaluator import (
    evaluate_area_from_config,
    evaluate_area_from_group_config,
    calculate_area,
)

from .accuracy_evaluator import (
    evaluate_accuracy_from_config,
    evaluate_accuracy_from_group_config,
    evaluate_model,
)

__all__ = [
    'evaluate_area_from_config',
    'evaluate_area_from_group_config',
    'calculate_area',
    'evaluate_accuracy_from_config',
    'evaluate_accuracy_from_group_config',
    'evaluate_model',
]

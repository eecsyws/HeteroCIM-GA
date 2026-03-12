"""
结果分析模块
提供Pareto前沿分析和热力图可视化
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from typing import List, Tuple, Dict

from config import AnalysisConfig, PathConfig, HardwareConfig, LayerGroupConfig


def load_results_from_csv(csv_path: str, has_header: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    从CSV加载结果数据

    参数:
        csv_path: str, CSV文件路径
        has_header: bool, 是否有表头

    返回:
        acc: np.ndarray, 准确率数组
        area_opt: np.ndarray, 面积优化率数组
    """
    acc_list = []
    area_opt_list = []

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)

        if has_header:
            next(reader)

        for row in reader:
            if len(row) < 3:
                continue

            try:
                acc = float(row[0])
                area_opt_ratio = float(row[2])  # area_opt_ratio列
                acc_list.append(acc)
                area_opt_list.append(area_opt_ratio * 100)  # 转换为百分比
            except:
                continue

    return np.array(acc_list), np.array(area_opt_list)


def find_pareto_front(acc: np.ndarray, area_opt: np.ndarray) -> np.ndarray:
    """
    找到Pareto前沿点

    参数:
        acc: np.ndarray, 准确率数组
        area_opt: np.ndarray, 面积优化率数组

    返回:
        pareto_mask: np.ndarray, Pareto点的布尔掩码
    """
    n_points = len(acc)
    is_pareto = np.ones(n_points, dtype=bool)

    for i in range(n_points):
        if is_pareto[i]:
            # 找到支配当前点的其他点
            dominated = (acc >= acc[i]) & (area_opt >= area_opt[i])
            dominated[i] = False  # 排除自己

            # 如果存在严格支配的点，当前点不是Pareto最优
            strictly_dominated = dominated & ((acc > acc[i]) | (area_opt > area_opt[i]))
            if np.any(strictly_dominated):
                is_pareto[i] = False

    return is_pareto


def plot_pareto_front(
    csv_path: str,
    output_path: str = None,
    title: str = "Pareto Front Analysis"
):
    """
    绘制Pareto前沿图

    参数:
        csv_path: str, 结果CSV路径
        output_path: str, 输出图片路径
        title: str, 图表标题
    """
    # 加载数据
    acc, area_opt = load_results_from_csv(csv_path)

    if len(acc) == 0:
        print("No data found in CSV")
        return

    # 找到Pareto前沿
    pareto_mask = find_pareto_front(acc, area_opt)
    pareto_acc = acc[pareto_mask]
    pareto_area = area_opt[pareto_mask]

    # 排序以便绘制连线
    sort_idx = np.argsort(pareto_acc)
    pareto_acc_sorted = pareto_acc[sort_idx]
    pareto_area_sorted = pareto_area[sort_idx]

    # 绘图
    plt.figure(figsize=AnalysisConfig.FIGURE_SIZE, dpi=AnalysisConfig.FIGURE_DPI)

    # 所有点
    plt.scatter(acc, area_opt, c='skyblue', alpha=AnalysisConfig.PARETO_PLOT_ALPHA,
                label='All Solutions', edgecolors='none')

    # Pareto点
    plt.scatter(pareto_acc, pareto_area, c='red', s=AnalysisConfig.PARETO_POINT_SIZE,
                label='Pareto Optimal', edgecolors='black', zorder=5)

    # Pareto前沿连线
    plt.plot(pareto_acc_sorted, pareto_area_sorted, 'r--', alpha=0.7,
             linewidth=1.5, label='Pareto Front')

    plt.xlabel('Accuracy (%)', fontsize=11)
    plt.ylabel('Area Optimization (%)', fontsize=11)
    plt.title(title, fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best')

    # 保存
    if output_path is None:
        import os
        output_path = os.path.join(PathConfig.ANALYSIS_OUTPUT, "pareto_front.png")

    plt.savefig(output_path, bbox_inches='tight', dpi=AnalysisConfig.FIGURE_DPI)
    plt.close()

    print(f"Pareto front plot saved to {output_path}")
    print(f"Found {len(pareto_acc)} Pareto optimal points out of {len(acc)} total points")


def plot_evolution_history(
    history: List[Dict],
    output_path: str = None,
    title: str = "GA Evolution History"
):
    """
    绘制GA进化历史

    参数:
        history: List[Dict], GA历史记录
        output_path: str, 输出图片路径
        title: str, 图表标题
    """
    if len(history) == 0:
        print("No history data")
        return

    generations = [h['generation'] for h in history]
    best_acc = [h['best_acc'] for h in history]
    mean_acc = [h['mean_acc'] for h in history]
    best_area = [h['best_area_ratio'] for h in history]
    mean_area = [h['mean_area_ratio'] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=AnalysisConfig.FIGURE_DPI)

    # 准确率进化
    ax1.plot(generations, best_acc, 'r-', label='Best Accuracy', linewidth=2)
    ax1.plot(generations, mean_acc, 'b--', label='Mean Accuracy', linewidth=1.5)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 面积比进化
    ax2.plot(generations, best_area, 'r-', label='Best Area Ratio', linewidth=2)
    ax2.plot(generations, mean_area, 'b--', label='Mean Area Ratio', linewidth=1.5)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Area Ratio')
    ax2.set_title('Area Ratio Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    # 保存
    if output_path is None:
        import os
        output_path = os.path.join(PathConfig.ANALYSIS_OUTPUT, "evolution_history.png")

    plt.savefig(output_path, bbox_inches='tight', dpi=AnalysisConfig.FIGURE_DPI)
    plt.close()

    print(f"Evolution history plot saved to {output_path}")


def load_population_configs(csv_path: str, has_header: bool = True) -> Tuple[List[Dict], int]:
    """
    从CSV加载种群配置数据

    参数:
        csv_path: str, CSV文件路径
        has_header: bool, 是否有表头

    返回:
        individuals: List[Dict], 个体列表
        num_layers: int, 层数量（8=粗粒度，96=细粒度）
    """
    individuals = []

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)

        if has_header:
            header = next(reader)
            # 判断是粗粒度还是细粒度
            # 粗粒度: header包含 g0-g7
            # 细粒度: header包含 layer_0-layer_95
            if 'g0' in header[6]:
                num_layers = 8
            else:
                num_layers = len([h for h in header if h.startswith('layer_')])

        for row in reader:
            if len(row) < 6:
                continue

            try:
                individual = {
                    'acc': float(row[0]),
                    'area_ratio': float(row[1]),
                    'area_opt_ratio': float(row[2]),
                }

                if num_layers == 8:
                    # 粗粒度：8个group配置
                    individual['groups'] = [int(row[i]) for i in range(6, 14)]
                else:
                    # 细粒度：96个layer配置
                    individual['layers'] = [int(row[i]) for i in range(6, 6 + num_layers)]

                individuals.append(individual)
            except:
                continue

    return individuals, num_layers


def find_pareto_optimal(population: List[Dict]) -> List[int]:
    """
    从种群中找到Pareto最优个体的索引

    参数:
        population: List[Dict], 种群列表

    返回:
        pareto_indices: List[int], Pareto最优个体的索引
    """
    n = len(population)
    is_pareto = [True] * n

    for i in range(n):
        if not is_pareto[i]:
            continue

        for j in range(n):
            if i == j or not is_pareto[j]:
                continue

            # 检查j是否支配i
            # 支配: acc >= acc 且 area_ratio <= area_ratio，且至少一个严格不等
            if (population[j]['acc'] >= population[i]['acc'] and
                population[j]['area_ratio'] <= population[i]['area_ratio'] and
                (population[j]['acc'] > population[i]['acc'] or
                 population[j]['area_ratio'] < population[i]['area_ratio'])):
                is_pareto[i] = False
                break

    return [i for i in range(n) if is_pareto[i]]


def plot_config_heatmap(
    csv_path: str,
    output_path: str = None,
    title: str = "Pareto Optimal Configurations Heatmap"
):
    """
    绘制Pareto最优个体的配置热力图

    参数:
        csv_path: str, 结果CSV路径
        output_path: str, 输出图片路径
        title: str, 图表标题
    """
    # 加载种群数据
    population, num_layers = load_population_configs(csv_path)

    if len(population) == 0:
        print("No data found in CSV")
        return

    # 找到Pareto最优个体
    pareto_indices = find_pareto_optimal(population)
    pareto_population = [population[i] for i in pareto_indices]

    if len(pareto_population) == 0:
        print("No Pareto optimal individuals found")
        return

    print(f"Found {len(pareto_population)} Pareto optimal individuals out of {len(population)}")

    # 构建配置矩阵
    config_matrix = []
    for ind in pareto_population:
        if num_layers == 8:
            configs = ind['groups']
        else:
            configs = ind['layers']
        config_matrix.append(configs)

    config_matrix = np.array(config_matrix)

    # 创建图形
    figsize = (max(16, num_layers // 4), max(6, len(pareto_population) // 3))
    fig, ax = plt.subplots(figsize=figsize, dpi=AnalysisConfig.FIGURE_DPI)

    # 绘制热力图
    # 使用配置的索引作为颜色值
    vmin, vmax = 0, HardwareConfig.MAX_CONFIG_INDEX

    im = ax.imshow(config_matrix, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)

    # 设置坐标轴
    if num_layers == 8:
        # 粗粒度：显示group名称
        group_names = ['Q', 'K', 'V', 'QK', 'AV', 'Output', 'FC1', 'FC2']
        ax.set_xticks(range(8))
        ax.set_xticklabels(group_names, rotation=45, ha='right')
    else:
        # 细粒度：每12层显示一个标记（对应一个transformer block）
        xtick_positions = list(range(0, 96, 12))
        xtick_labels = [str(i // 12 + 1) for i in xtick_positions]
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels)
        ax.set_xlabel('Transformer Block (Layer)', fontsize=11)

    # Y轴：每个Pareto个体，显示其准确率和面积信息
    yticklabels = []
    for ind in pareto_population:
        yticklabels.append(f"Acc:{ind['acc']:.1f}% A:{ind['area_ratio']:.2f}")

    ax.set_yticks(range(len(pareto_population)))
    ax.set_yticklabels(yticklabels, fontsize=8)

    # 颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label('Config Index', fontsize=11)

    # 在每个格子中显示配置索引值
    if num_layers <= 20:  # 只在配置较少时显示数值
        for i in range(len(pareto_population)):
            for j in range(num_layers):
                text = ax.text(j, i, config_matrix[i, j],
                              ha="center", va="center", color="white", fontsize=6)

    ax.set_title(title, fontsize=13)
    plt.tight_layout()

    # 保存
    if output_path is None:
        import os
        output_path = os.path.join(PathConfig.ANALYSIS_OUTPUT, "config_heatmap.png")

    plt.savefig(output_path, bbox_inches='tight', dpi=AnalysisConfig.FIGURE_DPI)
    plt.close()

    print(f"Config heatmap saved to {output_path}")

    return pareto_population


def plot_layer_config_distribution(
    csv_path: str,
    output_path: str = None,
    title: str = "Layer Config Distribution"
):
    """
    绘制每层配置选择分布图（聚合多个Pareto个体的选择）

    参数:
        csv_path: str, 结果CSV路径
        output_path: str, 输出图片路径
        title: str, 图表标题
    """
    # 加载种群数据
    population, num_layers = load_population_configs(csv_path)

    if len(population) == 0:
        print("No data found in CSV")
        return

    # 找到Pareto最优个体
    pareto_indices = find_pareto_optimal(population)
    pareto_population = [population[i] for i in pareto_indices]

    if len(pareto_population) == 0:
        print("No Pareto optimal individuals found")
        return

    # 统计每层选择的配置分布
    config_counts = np.zeros((num_layers, HardwareConfig.MAX_CONFIG_INDEX + 1))

    for ind in pareto_population:
        if num_layers == 8:
            configs = ind['groups']
        else:
            configs = ind['layers']

        for layer_idx, config_idx in enumerate(configs):
            config_counts[layer_idx, config_idx] += 1

    # 归一化为百分比
    config_counts_pct = config_counts / len(pareto_population) * 100

    # 创建图形
    figsize = (max(12, num_layers // 6), 8)
    fig, ax = plt.subplots(figsize=figsize, dpi=AnalysisConfig.FIGURE_DPI)

    # 绘制热力图
    im = ax.imshow(config_counts_pct, aspect='auto', cmap='Blues', vmin=0, vmax=100)

    # 设置坐标轴
    if num_layers == 8:
        group_names = ['Q', 'K', 'V', 'QK', 'AV', 'Output', 'FC1', 'FC2']
        ax.set_xticks(range(8))
        ax.set_xticklabels(group_names, rotation=45, ha='right')
    else:
        xtick_positions = list(range(0, 96, 12))
        xtick_labels = [str(i // 12 + 1) for i in xtick_positions]
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels)
        ax.set_xlabel('Transformer Block', fontsize=11)

    ax.set_ylabel('Config Index', fontsize=11)

    # 颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Selection Rate (%)', fontsize=11)

    ax.set_title(title, fontsize=13)
    plt.tight_layout()

    # 保存
    if output_path is None:
        import os
        output_path = os.path.join(PathConfig.ANALYSIS_OUTPUT, "layer_config_distribution.png")

    plt.savefig(output_path, bbox_inches='tight', dpi=AnalysisConfig.FIGURE_DPI)
    plt.close()

    print(f"Layer config distribution saved to {output_path}")


if __name__ == "__main__":
    import os

    # 测试Pareto前沿分析
    csv_path = os.path.join(PathConfig.COARSE_GA_OUTPUT, "final_population.csv")

    if os.path.exists(csv_path):
        print("Analyzing Pareto front...")
        plot_pareto_front(csv_path, title="Coarse GA Pareto Front")
    else:
        print(f"CSV file not found: {csv_path}")
        print("Please run GA first to generate results")

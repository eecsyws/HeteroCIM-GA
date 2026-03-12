"""
可视化层类型噪声敏感性测试结果
"""

import os
import csv
import matplotlib.pyplot as plt
import numpy as np


def load_results(csv_path):
    """加载CSV结果"""
    results = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            layer_type = row['layer_type']
            noise_sigma = float(row['noise_sigma'])
            accuracy = float(row['accuracy'])

            if layer_type not in results:
                results[layer_type] = {'sigma': [], 'acc': []}
            results[layer_type]['sigma'].append(noise_sigma)
            results[layer_type]['acc'].append(accuracy)

    # 按噪声强度排序
    for layer_type in results:
        sorted_pairs = sorted(zip(results[layer_type]['sigma'], results[layer_type]['acc']))
        results[layer_type]['sigma'] = [p[0] for p in sorted_pairs]
        results[layer_type]['acc'] = [p[1] for p in sorted_pairs]

    return results


def find_sigma_index(sigma_list, target, tol=1e-9):
    """查找目标sigma对应的索引，避免浮点数精度问题"""
    for i, sigma in enumerate(sigma_list):
        if abs(sigma - target) < tol:
            return i
    return None


def plot_all_layer_types(results, output_dir):
    """绘制所有层类型的噪声敏感性曲线"""
    plt.figure(figsize=(12, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for idx, (layer_type, data) in enumerate(results.items()):
        plt.plot(data['sigma'], data['acc'],
                marker='o', linewidth=2, markersize=6,
                label=layer_type, color=colors[idx])

    plt.xlabel('Noise Sigma (σ)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Layer Type Noise Sensitivity (INT8 Quantization)', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)

    output_path = os.path.join(output_dir, 'all_layer_types_noise.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_individual_layer_types(results, output_dir):
    """为每个层类型单独绘图"""
    for layer_type, data in results.items():
        plt.figure(figsize=(8, 6))

        plt.plot(data['sigma'], data['acc'],
                marker='o', linewidth=2.5, markersize=8,
                color='steelblue', markerfacecolor='orange')

        plt.xlabel('Noise Sigma (σ)', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title(f'{layer_type} Noise Sensitivity', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

        # 标注数值
        for sigma, acc in zip(data['sigma'], data['acc']):
            plt.text(sigma, acc + 0.3, f'{acc:.1f}',
                    ha='center', va='bottom', fontsize=9)

        safe_name = layer_type.replace('^', '').replace(' ', '_')
        output_path = os.path.join(output_dir, f'{safe_name}_noise.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")


def plot_accuracy_drop_heatmap(results, output_dir):
    """绘制准确率下降热力图"""
    layer_types = list(results.keys())
    sigma_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]

    # 构建矩阵：相对于σ=0.00的准确率下降
    matrix = []
    for layer_type in layer_types:
        data = results[layer_type]

        # 找到σ=0.00的准确率作为baseline
        baseline_idx = find_sigma_index(data['sigma'], 0.0)
        if baseline_idx is not None:
            baseline_acc = data['acc'][baseline_idx]
        else:
            baseline_acc = data['acc'][0]

        row = []
        for sigma in sigma_list:
            idx = find_sigma_index(data['sigma'], sigma)
            if idx is not None:
                acc_drop = baseline_acc - data['acc'][idx]
                row.append(acc_drop)
            else:
                row.append(0)
        matrix.append(row)

    matrix = np.array(matrix)

    plt.figure(figsize=(11, 6))
    im = plt.imshow(matrix, cmap='YlOrRd', aspect='auto')

    plt.colorbar(im, label='Accuracy Drop (%)')
    plt.xticks(range(len(sigma_list)), [f'σ={s:.2f}' for s in sigma_list])
    plt.yticks(range(len(layer_types)), layer_types)
    plt.xlabel('Noise Sigma', fontsize=12)
    plt.ylabel('Layer Type', fontsize=12)
    plt.title('Accuracy Drop Relative to σ=0.00', fontsize=14, fontweight='bold')

    # 标注数值
    for i in range(len(layer_types)):
        for j in range(len(sigma_list)):
            plt.text(j, i, f'{matrix[i, j]:.1f}',
                     ha="center", va="center", color="black", fontsize=9)

    output_path = os.path.join(output_dir, 'accuracy_drop_heatmap_noise.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_sensitivity_ranking(results, output_dir):
    """绘制层类型噪声敏感性排名（基于σ=0.25准确率）"""
    layer_types = []
    high_noise_accs = []

    for layer_type, data in results.items():
        idx = find_sigma_index(data['sigma'], 0.25)
        if idx is not None:
            layer_types.append(layer_type)
            high_noise_accs.append(data['acc'][idx])

    # 排序
    sorted_pairs = sorted(zip(layer_types, high_noise_accs), key=lambda x: x[1], reverse=True)
    layer_types = [p[0] for p in sorted_pairs]
    high_noise_accs = [p[1] for p in sorted_pairs]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(range(len(layer_types)), high_noise_accs, color='coral')

    plt.yticks(range(len(layer_types)), layer_types)
    plt.xlabel('Accuracy at σ=0.25 (%)', fontsize=12)
    plt.title('Layer Type Noise Sensitivity Ranking (σ=0.25)', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)

    # 标注数值
    for i, (bar, acc) in enumerate(zip(bars, high_noise_accs)):
        plt.text(acc + 0.5, i, f'{acc:.1f}%',
                va='center', fontsize=9)

    output_path = os.path.join(output_dir, 'sensitivity_ranking_noise.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_noise_impact_comparison(results, output_dir):
    """绘制噪声影响对比（σ=0.00 vs σ=0.25）"""
    layer_types = []
    low_noise_accs = []
    high_noise_accs = []

    for layer_type, data in results.items():
        idx_low = find_sigma_index(data['sigma'], 0.0)
        idx_high = find_sigma_index(data['sigma'], 0.25)

        if idx_low is not None and idx_high is not None:
            layer_types.append(layer_type)
            low_noise_accs.append(data['acc'][idx_low])
            high_noise_accs.append(data['acc'][idx_high])

    x = np.arange(len(layer_types))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, low_noise_accs, width, label='σ=0.00', color='lightblue')
    plt.bar(x + width/2, high_noise_accs, width, label='σ=0.25', color='salmon')

    plt.xlabel('Layer Type', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Noise Impact Comparison (No Noise vs High Noise)', fontsize=14, fontweight='bold')
    plt.xticks(x, layer_types, rotation=15, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    output_path = os.path.join(output_dir, 'noise_impact_comparison.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """主函数"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    csv_path = os.path.join(results_dir, 'layer_type_noise_results.csv')

    if not os.path.exists(csv_path):
        print(f"Error: Results file not found at {csv_path}")
        print("Please run test_layer_type_noise.py first")
        return

    print("="*70)
    print("Visualizing Layer Type Noise Sensitivity Results")
    print("="*70)

    # 加载结果
    results = load_results(csv_path)
    print(f"\nLoaded results for {len(results)} layer types")

    # 创建图表输出目录
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # 生成图表
    print("\nGenerating plots...")
    plot_all_layer_types(results, plots_dir)
    plot_individual_layer_types(results, plots_dir)
    plot_accuracy_drop_heatmap(results, plots_dir)
    plot_sensitivity_ranking(results, plots_dir)
    plot_noise_impact_comparison(results, plots_dir)

    print("\n" + "="*70)
    print(f"All plots saved to: {plots_dir}")
    print("="*70)


if __name__ == "__main__":
    main()

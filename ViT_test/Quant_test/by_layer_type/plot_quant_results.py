"""
可视化层类型量化敏感性测试结果
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
            quant_bits = int(row['quant_bits'])
            accuracy = float(row['accuracy'])

            if layer_type not in results:
                results[layer_type] = {'bits': [], 'acc': []}
            results[layer_type]['bits'].append(quant_bits)
            results[layer_type]['acc'].append(accuracy)

    # 按位宽排序
    for layer_type in results:
        sorted_pairs = sorted(zip(results[layer_type]['bits'], results[layer_type]['acc']))
        results[layer_type]['bits'] = [p[0] for p in sorted_pairs]
        results[layer_type]['acc'] = [p[1] for p in sorted_pairs]

    return results


def plot_all_layer_types(results, output_dir):
    """绘制所有层类型的量化敏感性曲线"""
    plt.figure(figsize=(12, 8))

    # 定义颜色映射
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for idx, (layer_type, data) in enumerate(results.items()):
        plt.plot(data['bits'], data['acc'],
                marker='o', linewidth=2, markersize=6,
                label=layer_type, color=colors[idx])

    plt.xlabel('Quantization Bits', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Layer Type Quantization Sensitivity', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.xticks([3, 4, 5, 6, 7, 8])

    # 保存
    output_path = os.path.join(output_dir, 'all_layer_types.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_individual_layer_types(results, output_dir):
    """为每个层类型单独绘图"""
    for layer_type, data in results.items():
        plt.figure(figsize=(8, 6))

        plt.plot(data['bits'], data['acc'],
                marker='o', linewidth=2.5, markersize=8,
                color='steelblue', markerfacecolor='orange')

        plt.xlabel('Quantization Bits', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title(f'{layer_type} Quantization Sensitivity', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xticks([3, 4, 5, 6, 7, 8])

        # 标注数值
        for bits, acc in zip(data['bits'], data['acc']):
            plt.text(bits, acc + 0.3, f'{acc:.1f}',
                    ha='center', va='bottom', fontsize=9)

        # 保存
        safe_name = layer_type.replace('^', '').replace(' ', '_')
        output_path = os.path.join(output_dir, f'{safe_name}.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")


def plot_accuracy_drop_heatmap(results, output_dir):
    """绘制准确率下降热力图"""
    layer_types = list(results.keys())
    bits_list = [3, 4, 5, 6, 7, 8]

    # 构建矩阵：相对于INT8的准确率下降
    matrix = []
    for layer_type in layer_types:
        data = results[layer_type]
        # 找到INT8的准确率作为baseline
        baseline_acc = data['acc'][data['bits'].index(8)]
        row = []
        for bits in bits_list:
            if bits in data['bits']:
                idx = data['bits'].index(bits)
                acc_drop = baseline_acc - data['acc'][idx]
                row.append(acc_drop)
            else:
                row.append(0)
        matrix.append(row)

    matrix = np.array(matrix)

    plt.figure(figsize=(10, 8))
    im = plt.imshow(matrix, cmap='YlOrRd', aspect='auto')

    plt.colorbar(im, label='Accuracy Drop (%)')
    plt.xticks(range(len(bits_list)), [f'INT{b}' for b in bits_list])
    plt.yticks(range(len(layer_types)), layer_types)
    plt.xlabel('Quantization Bits', fontsize=12)
    plt.ylabel('Layer Type', fontsize=12)
    plt.title('Accuracy Drop Relative to INT8', fontsize=14, fontweight='bold')

    # 标注数值
    for i in range(len(layer_types)):
        for j in range(len(bits_list)):
            text = plt.text(j, i, f'{matrix[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=8)

    output_path = os.path.join(output_dir, 'accuracy_drop_heatmap.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_sensitivity_ranking(results, output_dir):
    """绘制层类型敏感性排名（基于INT3准确率）"""
    layer_types = []
    int3_accs = []

    for layer_type, data in results.items():
        if 3 in data['bits']:
            idx = data['bits'].index(3)
            layer_types.append(layer_type)
            int3_accs.append(data['acc'][idx])

    # 排序
    sorted_pairs = sorted(zip(layer_types, int3_accs), key=lambda x: x[1], reverse=True)
    layer_types = [p[0] for p in sorted_pairs]
    int3_accs = [p[1] for p in sorted_pairs]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(range(len(layer_types)), int3_accs, color='coral')

    plt.yticks(range(len(layer_types)), layer_types)
    plt.xlabel('Accuracy at INT3 (%)', fontsize=12)
    plt.title('Layer Type Sensitivity Ranking (INT3)', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)

    # 标注数值
    for i, (bar, acc) in enumerate(zip(bars, int3_accs)):
        plt.text(acc + 0.5, i, f'{acc:.1f}%',
                va='center', fontsize=9)

    output_path = os.path.join(output_dir, 'sensitivity_ranking.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """主函数"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    csv_path = os.path.join(results_dir, 'layer_type_quant_results.csv')

    if not os.path.exists(csv_path):
        print(f"Error: Results file not found at {csv_path}")
        print("Please run test_layer_type_quant.py first")
        return

    print("="*70)
    print("Visualizing Layer Type Quantization Results")
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

    print("\n" + "="*70)
    print(f"All plots saved to: {plots_dir}")
    print("="*70)


if __name__ == "__main__":
    main()

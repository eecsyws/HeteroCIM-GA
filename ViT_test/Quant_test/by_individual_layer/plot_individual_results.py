"""
可视化单层量化敏感性测试结果
"""

import os
import csv
import matplotlib.pyplot as plt
import numpy as np


def load_results(csv_path):
    """加载CSV结果"""
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'layer_index': int(row['layer_index']),
                'layer_name': row['layer_name'],
                'quant_bits': int(row['quant_bits']),
                'accuracy': float(row['accuracy']),
            })
    return results


def organize_by_layer(results):
    """按层组织数据"""
    layers = {}
    for row in results:
        layer_name = row['layer_name']
        if layer_name not in layers:
            layers[layer_name] = {'bits': [], 'acc': [], 'index': row['layer_index']}
        layers[layer_name]['bits'].append(row['quant_bits'])
        layers[layer_name]['acc'].append(row['accuracy'])

    # 排序
    for layer_name in layers:
        sorted_pairs = sorted(zip(layers[layer_name]['bits'], layers[layer_name]['acc']))
        layers[layer_name]['bits'] = [p[0] for p in sorted_pairs]
        layers[layer_name]['acc'] = [p[1] for p in sorted_pairs]

    return layers


def organize_by_bits(results):
    """按位宽组织数据"""
    bits_data = {}
    for row in results:
        bits = row['quant_bits']
        if bits not in bits_data:
            bits_data[bits] = {'layers': [], 'acc': [], 'indices': []}
        bits_data[bits]['layers'].append(row['layer_name'])
        bits_data[bits]['acc'].append(row['accuracy'])
        bits_data[bits]['indices'].append(row['layer_index'])

    return bits_data


def plot_all_layers_by_bits(layers, output_dir):
    """绘制所有层在不同位宽下的曲线（分组显示）"""
    layer_names = sorted(layers.keys(), key=lambda x: layers[x]['index'])

    # 分成多个子图（每个子图显示部分层）
    layers_per_plot = 20
    num_plots = (len(layer_names) + layers_per_plot - 1) // layers_per_plot

    for plot_idx in range(num_plots):
        start_idx = plot_idx * layers_per_plot
        end_idx = min((plot_idx + 1) * layers_per_plot, len(layer_names))
        subset_layers = layer_names[start_idx:end_idx]

        plt.figure(figsize=(14, 8))
        colors = plt.cm.tab20(np.linspace(0, 1, len(subset_layers)))

        for idx, layer_name in enumerate(subset_layers):
            data = layers[layer_name]
            plt.plot(data['bits'], data['acc'],
                    marker='o', linewidth=1.5, markersize=4,
                    label=layer_name, color=colors[idx], alpha=0.7)

        plt.xlabel('Quantization Bits', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title(f'Individual Layer Quantization Sensitivity (Layers {start_idx+1}-{end_idx})',
                 fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=7, ncol=2)
        plt.grid(True, alpha=0.3)
        plt.xticks([3, 4, 5, 6, 7, 8])

        output_path = os.path.join(output_dir, f'all_layers_group_{plot_idx+1}.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")


def plot_sensitivity_heatmap(results, output_dir):
    """绘制所有层的量化敏感性热力图"""
    layers = organize_by_layer(results)
    layer_names = sorted(layers.keys(), key=lambda x: layers[x]['index'])
    bits_list = [3, 4, 5, 6, 7, 8]

    # 构建矩阵
    matrix = []
    for layer_name in layer_names:
        data = layers[layer_name]
        row = []
        for bits in bits_list:
            if bits in data['bits']:
                idx = data['bits'].index(bits)
                row.append(data['acc'][idx])
            else:
                row.append(0)
        matrix.append(row)

    matrix = np.array(matrix)

    plt.figure(figsize=(10, 20))
    im = plt.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=70, vmax=90)

    plt.colorbar(im, label='Accuracy (%)')
    plt.xticks(range(len(bits_list)), [f'INT{b}' for b in bits_list])
    plt.yticks(range(len(layer_names)), layer_names, fontsize=6)
    plt.xlabel('Quantization Bits', fontsize=12)
    plt.ylabel('Layer Name', fontsize=12)
    plt.title('Individual Layer Quantization Sensitivity Heatmap', fontsize=14, fontweight='bold')

    output_path = os.path.join(output_dir, 'sensitivity_heatmap.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_accuracy_drop_by_layer(results, output_dir):
    """绘制每层在INT3时的准确率下降"""
    layers = organize_by_layer(results)
    layer_names = sorted(layers.keys(), key=lambda x: layers[x]['index'])

    acc_drops = []
    for layer_name in layer_names:
        data = layers[layer_name]
        if 8 in data['bits'] and 3 in data['bits']:
            idx_8 = data['bits'].index(8)
            idx_3 = data['bits'].index(3)
            acc_drop = data['acc'][idx_8] - data['acc'][idx_3]
            acc_drops.append(acc_drop)
        else:
            acc_drops.append(0)

    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(layer_names)), acc_drops, color='coral')

    plt.yticks(range(len(layer_names)), layer_names, fontsize=6)
    plt.xlabel('Accuracy Drop (INT8 - INT3) (%)', fontsize=12)
    plt.title('Layer Sensitivity Ranking (INT8 vs INT3)', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)

    output_path = os.path.join(output_dir, 'accuracy_drop_ranking.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_layer_comparison_at_bits(results, output_dir):
    """绘制特定位宽下所有层的准确率对比"""
    bits_data = organize_by_bits(results)

    for bits in sorted(bits_data.keys()):
        data = bits_data[bits]
        # 按layer index排序
        sorted_triplets = sorted(zip(data['indices'], data['layers'], data['acc']))
        indices = [t[0] for t in sorted_triplets]
        layer_names = [t[1] for t in sorted_triplets]
        accuracies = [t[2] for t in sorted_triplets]

        plt.figure(figsize=(14, 6))
        plt.plot(indices, accuracies, marker='o', linewidth=2, markersize=5, color='steelblue')

        plt.xlabel('Layer Index', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title(f'Accuracy Across All Layers at INT{bits}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xticks(range(0, len(layer_names), 10))

        output_path = os.path.join(output_dir, f'layer_comparison_INT{bits}.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")


def plot_top_sensitive_layers(results, output_dir, top_n=10):
    """绘制最敏感的N个层"""
    layers = organize_by_layer(results)

    # 计算每层的敏感性（INT8 - INT3的准确率差）
    sensitivity_scores = []
    for layer_name, data in layers.items():
        if 8 in data['bits'] and 3 in data['bits']:
            idx_8 = data['bits'].index(8)
            idx_3 = data['bits'].index(3)
            sensitivity = data['acc'][idx_8] - data['acc'][idx_3]
            sensitivity_scores.append((layer_name, sensitivity, data))

    # 排序并取top N
    sensitivity_scores.sort(key=lambda x: x[1], reverse=True)
    top_layers = sensitivity_scores[:top_n]

    plt.figure(figsize=(12, 8))
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, top_n))

    for idx, (layer_name, sensitivity, data) in enumerate(top_layers):
        plt.plot(data['bits'], data['acc'],
                marker='o', linewidth=2.5, markersize=6,
                label=f"{layer_name} (Δ={sensitivity:.1f}%)",
                color=colors[idx])

    plt.xlabel('Quantization Bits', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(f'Top {top_n} Most Sensitive Layers', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.xticks([3, 4, 5, 6, 7, 8])

    output_path = os.path.join(output_dir, f'top_{top_n}_sensitive_layers.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """主函数"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    csv_path = os.path.join(results_dir, 'individual_layer_quant_results.csv')

    if not os.path.exists(csv_path):
        print(f"Error: Results file not found at {csv_path}")
        print("Please run test_individual_layer_quant.py first")
        return

    print("="*70)
    print("Visualizing Individual Layer Quantization Results")
    print("="*70)

    # 加载结果
    results = load_results(csv_path)
    layers = organize_by_layer(results)
    print(f"\nLoaded results for {len(layers)} layers")
    print(f"Total data points: {len(results)}")

    # 创建图表输出目录
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # 生成图表
    print("\nGenerating plots...")
    plot_all_layers_by_bits(layers, plots_dir)
    plot_sensitivity_heatmap(results, plots_dir)
    plot_accuracy_drop_by_layer(results, plots_dir)
    plot_layer_comparison_at_bits(results, plots_dir)
    plot_top_sensitive_layers(results, plots_dir, top_n=10)
    plot_top_sensitive_layers(results, plots_dir, top_n=20)

    print("\n" + "="*70)
    print(f"All plots saved to: {plots_dir}")
    print("="*70)


if __name__ == "__main__":
    main()

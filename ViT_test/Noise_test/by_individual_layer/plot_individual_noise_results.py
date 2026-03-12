"""
可视化单层噪声敏感性测试结果
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
                'noise_sigma': float(row['noise_sigma']),
                'accuracy': float(row['accuracy']),
            })
    return results


def organize_by_layer(results):
    """按层组织数据"""
    layers = {}
    for row in results:
        layer_name = row['layer_name']
        if layer_name not in layers:
            layers[layer_name] = {'sigma': [], 'acc': [], 'index': row['layer_index']}
        layers[layer_name]['sigma'].append(row['noise_sigma'])
        layers[layer_name]['acc'].append(row['accuracy'])

    # 排序
    for layer_name in layers:
        sorted_pairs = sorted(zip(layers[layer_name]['sigma'], layers[layer_name]['acc']))
        layers[layer_name]['sigma'] = [p[0] for p in sorted_pairs]
        layers[layer_name]['acc'] = [p[1] for p in sorted_pairs]

    return layers


def organize_by_sigma(results):
    """按噪声强度组织数据"""
    sigma_data = {}
    for row in results:
        sigma = row['noise_sigma']
        if sigma not in sigma_data:
            sigma_data[sigma] = {'layers': [], 'acc': [], 'indices': []}
        sigma_data[sigma]['layers'].append(row['layer_name'])
        sigma_data[sigma]['acc'].append(row['accuracy'])
        sigma_data[sigma]['indices'].append(row['layer_index'])

    return sigma_data


def plot_all_layers_by_sigma(layers, output_dir):
    """绘制所有层在不同噪声强度下的曲线（分组显示）"""
    layer_names = sorted(layers.keys(), key=lambda x: layers[x]['index'])

    # 分成多个子图（每个子图显示部分层）
    layers_per_plot = 15
    num_plots = (len(layer_names) + layers_per_plot - 1) // layers_per_plot

    for plot_idx in range(num_plots):
        start_idx = plot_idx * layers_per_plot
        end_idx = min((plot_idx + 1) * layers_per_plot, len(layer_names))
        subset_layers = layer_names[start_idx:end_idx]

        plt.figure(figsize=(14, 8))
        colors = plt.cm.tab20(np.linspace(0, 1, len(subset_layers)))

        for idx, layer_name in enumerate(subset_layers):
            data = layers[layer_name]
            plt.plot(data['sigma'], data['acc'],
                    marker='o', linewidth=1.5, markersize=4,
                    label=layer_name, color=colors[idx], alpha=0.7)

        plt.xlabel('Noise Sigma (σ)', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title(f'Individual Layer Noise Sensitivity (Layers {start_idx+1}-{end_idx})',
                 fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=7, ncol=2)
        plt.grid(True, alpha=0.3)

        output_path = os.path.join(output_dir, f'all_layers_noise_group_{plot_idx+1}.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")


def plot_sensitivity_heatmap(results, output_dir):
    """绘制所有层的噪声敏感性热力图"""
    layers = organize_by_layer(results)
    layer_names = sorted(layers.keys(), key=lambda x: layers[x]['index'])
    sigma_list = [0.05, 0.1, 0.15, 0.2, 0.25]

    # 构建矩阵
    matrix = []
    for layer_name in layer_names:
        data = layers[layer_name]
        row = []
        for sigma in sigma_list:
            if sigma in data['sigma']:
                idx = data['sigma'].index(sigma)
                row.append(data['acc'][idx])
            else:
                row.append(0)
        matrix.append(row)

    matrix = np.array(matrix)

    plt.figure(figsize=(10, 18))
    im = plt.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=70, vmax=90)

    plt.colorbar(im, label='Accuracy (%)')
    plt.xticks(range(len(sigma_list)), [f'σ={s:.2f}' for s in sigma_list])
    plt.yticks(range(len(layer_names)), layer_names, fontsize=6)
    plt.xlabel('Noise Sigma', fontsize=12)
    plt.ylabel('Layer Name', fontsize=12)
    plt.title('Individual Layer Noise Sensitivity Heatmap', fontsize=14, fontweight='bold')

    output_path = os.path.join(output_dir, 'sensitivity_heatmap_noise.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_accuracy_drop_by_layer(results, output_dir):
    """绘制每层在σ=0.25时的准确率下降"""
    layers = organize_by_layer(results)
    layer_names = sorted(layers.keys(), key=lambda x: layers[x]['index'])

    acc_drops = []
    for layer_name in layer_names:
        data = layers[layer_name]
        if 0.05 in data['sigma'] and 0.25 in data['sigma']:
            idx_low = data['sigma'].index(0.05)
            idx_high = data['sigma'].index(0.25)
            acc_drop = data['acc'][idx_low] - data['acc'][idx_high]
            acc_drops.append(acc_drop)
        else:
            acc_drops.append(0)

    plt.figure(figsize=(12, 16))
    bars = plt.barh(range(len(layer_names)), acc_drops, color='coral')

    plt.yticks(range(len(layer_names)), layer_names, fontsize=6)
    plt.xlabel('Accuracy Drop (σ=0.05 - σ=0.25) (%)', fontsize=12)
    plt.title('Layer Noise Sensitivity Ranking', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)

    output_path = os.path.join(output_dir, 'accuracy_drop_ranking_noise.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_layer_comparison_at_sigma(results, output_dir):
    """绘制特定噪声强度下所有层的准确率对比"""
    sigma_data = organize_by_sigma(results)

    for sigma in sorted(sigma_data.keys()):
        data = sigma_data[sigma]
        # 按layer index排序
        sorted_triplets = sorted(zip(data['indices'], data['layers'], data['acc']))
        indices = [t[0] for t in sorted_triplets]
        layer_names = [t[1] for t in sorted_triplets]
        accuracies = [t[2] for t in sorted_triplets]

        plt.figure(figsize=(14, 6))
        plt.plot(indices, accuracies, marker='o', linewidth=2, markersize=5, color='steelblue')

        plt.xlabel('Layer Index', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title(f'Accuracy Across All Layers at σ={sigma:.2f}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xticks(range(0, len(layer_names), 10))

        output_path = os.path.join(output_dir, f'layer_comparison_sigma_{sigma:.2f}.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")


def plot_top_sensitive_layers(results, output_dir, top_n=10):
    """绘制最敏感的N个层"""
    layers = organize_by_layer(results)

    # 计算每层的敏感性（σ=0.05 - σ=0.25的准确率差）
    sensitivity_scores = []
    for layer_name, data in layers.items():
        if 0.05 in data['sigma'] and 0.25 in data['sigma']:
            idx_low = data['sigma'].index(0.05)
            idx_high = data['sigma'].index(0.25)
            sensitivity = data['acc'][idx_low] - data['acc'][idx_high]
            sensitivity_scores.append((layer_name, sensitivity, data))

    # 排序并取top N
    sensitivity_scores.sort(key=lambda x: x[1], reverse=True)
    top_layers = sensitivity_scores[:top_n]

    plt.figure(figsize=(12, 8))
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, top_n))

    for idx, (layer_name, sensitivity, data) in enumerate(top_layers):
        plt.plot(data['sigma'], data['acc'],
                marker='o', linewidth=2.5, markersize=6,
                label=f"{layer_name} (Δ={sensitivity:.1f}%)",
                color=colors[idx])

    plt.xlabel('Noise Sigma (σ)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(f'Top {top_n} Most Noise-Sensitive Layers', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=8)
    plt.grid(True, alpha=0.3)

    output_path = os.path.join(output_dir, f'top_{top_n}_sensitive_layers_noise.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """主函数"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    csv_path = os.path.join(results_dir, 'individual_layer_noise_results.csv')

    if not os.path.exists(csv_path):
        print(f"Error: Results file not found at {csv_path}")
        print("Please run test_individual_layer_noise.py first")
        return

    print("="*70)
    print("Visualizing Individual Layer Noise Sensitivity Results")
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
    plot_all_layers_by_sigma(layers, plots_dir)
    plot_sensitivity_heatmap(results, plots_dir)
    plot_accuracy_drop_by_layer(results, plots_dir)
    plot_layer_comparison_at_sigma(results, plots_dir)
    plot_top_sensitive_layers(results, plots_dir, top_n=10)
    plot_top_sensitive_layers(results, plots_dir, top_n=20)

    print("\n" + "="*70)
    print(f"All plots saved to: {plots_dir}")
    print("="*70)


if __name__ == "__main__":
    main()

"""
主运行脚本
执行两阶段遗传算法搜索
"""

import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PathConfig, CoarseGAConfig, FineGAConfig
from algorithms import run_coarse_ga, run_fine_ga
from analysis import (
    plot_pareto_front,
    plot_evolution_history,
    plot_config_heatmap,
)


def run_two_stage_ga(
    # 粗粒度GA参数
    coarse_population_size: int = None,
    coarse_max_generations: int = None,
    coarse_acc_constraint: float = None,

    # 细粒度GA参数
    fine_population_size: int = None,
    fine_max_generations: int = None,
    fine_acc_constraint: float = None,

    # 通用参数
    seed: int = None,
    skip_coarse: bool = False,
    skip_fine: bool = False,
    skip_analysis: bool = False,
):
    """
    运行两阶段遗传算法

    参数:
        coarse_population_size: int, 粗粒度GA种群大小
        coarse_max_generations: int, 粗粒度GA最大代数
        coarse_acc_constraint: float, 粗粒度GA准确率约束
        fine_population_size: int, 细粒度GA种群大小
        fine_max_generations: int, 细粒度GA最大代数
        fine_acc_constraint: float, 细粒度GA准确率约束
        seed: int, 随机种子
        skip_coarse: bool, 跳过粗粒度GA（使用已有结果）
        skip_fine: bool, 跳过细粒度GA
        skip_analysis: bool, 跳过结果分析
    """
    print("="*70)
    print("Two-Stage Genetic Algorithm for ViT-CIM Design Space Exploration")
    print("="*70)

    # 创建输出目录
    PathConfig.create_output_dirs()

    # ========== 阶段1: 粗粒度GA ==========
    if not skip_coarse:
        print("\n" + "="*70)
        print("Stage 1: Coarse-Grained GA (9-dim group search)")
        print("="*70)

        coarse_best, coarse_history = run_coarse_ga(
            population_size=coarse_population_size,
            max_generations=coarse_max_generations,
            acc_constraint=coarse_acc_constraint,
            seed=seed,
        )

        print("\n[Coarse GA] Best Individual:")
        print(f"  Groups: {coarse_best['groups']}")
        print(f"  Accuracy: {coarse_best['acc']:.2f}%")
        print(f"  Area Ratio: {coarse_best['area_ratio']:.4f}")
        print(f"  Area Optimization: {coarse_best['area_opt_ratio']*100:.2f}%")

        # 绘制粗粒度GA进化历史
        if not skip_analysis:
            plot_evolution_history(
                coarse_history,
                output_path=os.path.join(PathConfig.COARSE_GA_OUTPUT, "evolution_history.png"),
                title="Coarse GA Evolution History"
            )
    else:
        print("\n[Skipping Coarse GA - using existing results]")

    # ========== 阶段2: 细粒度GA ==========
    if not skip_fine:
        print("\n" + "="*70)
        print("Stage 2: Fine-Grained GA (98-dim layer search)")
        print("="*70)

        # 如果跳过粗粒度，则禁用步幅限制和CSV初始化
        if skip_coarse:
            print("[Note] Running Fine GA without coarse initialization")
            print("[Note] Mutation step limit: DISABLED (full search space)")
            fine_best, fine_history = run_fine_ga(
                population_size=fine_population_size,
                max_generations=fine_max_generations,
                acc_constraint=fine_acc_constraint,
                seed=seed,
                csv_init_path=None,  # 不从CSV初始化
                enable_mutation_step_limit=False,  # 禁用步幅限制
            )
        else:
            print("[Note] Running Fine GA with coarse initialization")
            print("[Note] Mutation step limit: ENABLED (local refinement)")
            fine_best, fine_history = run_fine_ga(
                population_size=fine_population_size,
                max_generations=fine_max_generations,
                acc_constraint=fine_acc_constraint,
                seed=seed,
                # 使用默认CSV路径和步幅限制设置
            )

        print("\n[Fine GA] Best Individual:")
        print(f"  Accuracy: {fine_best['acc']:.2f}%")
        print(f"  Area Ratio: {fine_best['area_ratio']:.4f}")
        print(f"  Area Optimization: {fine_best['area_opt_ratio']*100:.2f}%")
        print(f"  Layers (first 10): {fine_best['layers'][:10]}")

        # 绘制细粒度GA进化历史
        if not skip_analysis:
            plot_evolution_history(
                fine_history,
                output_path=os.path.join(PathConfig.FINE_GA_OUTPUT, "evolution_history.png"),
                title="Fine GA Evolution History"
            )
    else:
        print("\n[Skipping Fine GA]")

    # ========== 结果分析 ==========
    if not skip_analysis:
        print("\n" + "="*70)
        print("Result Analysis")
        print("="*70)

        # 分析粗粒度GA结果
        coarse_csv = os.path.join(PathConfig.COARSE_GA_OUTPUT, "final_population.csv")
        if os.path.exists(coarse_csv):
            print("\nAnalyzing Coarse GA Pareto front...")
            plot_pareto_front(
                coarse_csv,
                output_path=os.path.join(PathConfig.ANALYSIS_OUTPUT, "coarse_ga_pareto.png"),
                title="Coarse GA Pareto Front"
            )

            # 绘制配置热力图
            print("\nGenerating Coarse GA config heatmap...")
            plot_config_heatmap(
                coarse_csv,
                output_path=os.path.join(PathConfig.ANALYSIS_OUTPUT, "coarse_ga_heatmap.png"),
                title="Coarse GA Pareto Configurations"
            )

        # 分析细粒度GA结果
        if not skip_fine:
            fine_csv = os.path.join(PathConfig.FINE_GA_OUTPUT, "final_population.csv")
            if os.path.exists(fine_csv):
                print("\nAnalyzing Fine GA Pareto front...")
                plot_pareto_front(
                    fine_csv,
                    output_path=os.path.join(PathConfig.ANALYSIS_OUTPUT, "fine_ga_pareto.png"),
                    title="Fine GA Pareto Front"
                )

                # 绘制配置热力图
                print("\nGenerating Fine GA config heatmap...")
                plot_config_heatmap(
                    fine_csv,
                    output_path=os.path.join(PathConfig.ANALYSIS_OUTPUT, "fine_ga_heatmap.png"),
                    title="Fine GA Pareto Configurations"
                )

    print("\n" + "="*70)
    print("Two-Stage GA Completed!")
    print("="*70)
    print(f"\nResults saved to: {PathConfig.OUTPUT_DIR}")


if __name__ == "__main__":
    # 默认配置运行
    run_two_stage_ga(
        # 粗粒度GA配置
        coarse_population_size=50,
        coarse_max_generations=5,
        coarse_acc_constraint=75.0,

        # 细粒度GA配置
        fine_population_size=50,
        fine_max_generations=5,
        fine_acc_constraint=75.0,

        # 通用配置
        seed=42,
        skip_coarse=False,
        skip_fine=False,
        skip_analysis=False,
    )

"""
粗粒度遗传算法（8维层种类搜索）
分组: Q Linear, K Linear, V Linear, QK^T, AV, Output Linear, FC1, FC2
不含PatchEmbed和Head（始终FP32）
"""

import random
import csv
from typing import List, Dict, Optional, Tuple

from config import CoarseGAConfig, HardwareConfig, LayerGroupConfig, PathConfig
from evaluators import evaluate_accuracy_from_group_config, evaluate_area_from_group_config
from .ga_base import GeneticAlgorithmBase

NUM_GROUPS = 8


class CoarseGA(GeneticAlgorithmBase):
    """粗粒度GA：在8维layer-type配置空间搜索"""

    def __init__(
        self,
        population_size: int = None,
        max_generations: int = None,
        acc_constraint: Optional[float] = None,
        seed: Optional[int] = None,
        verbose: bool = None,
        use_csv_init: bool = None,
        csv_init_path: str = None,
        tournament_size: int = None,
        elitism_rate: float = None,
    ):
        """初始化粗粒度GA"""
        # 使用配置文件的默认值
        if population_size is None:
            population_size = CoarseGAConfig.POPULATION_SIZE
        if max_generations is None:
            max_generations = CoarseGAConfig.MAX_GENERATIONS
        if acc_constraint is None:
            acc_constraint = CoarseGAConfig.ACC_CONSTRAINT
        if seed is None:
            seed = CoarseGAConfig.RANDOM_SEED
        if verbose is None:
            verbose = CoarseGAConfig.VERBOSE_GA
        if tournament_size is None:
            tournament_size = CoarseGAConfig.TOURNAMENT_SIZE
        if elitism_rate is None:
            elitism_rate = CoarseGAConfig.ELITISM_RATE

        super().__init__(population_size, max_generations, acc_constraint, seed, verbose,
                        tournament_size=tournament_size, elitism_rate=elitism_rate)

        self.use_csv_init = use_csv_init if use_csv_init is not None else CoarseGAConfig.USE_CSV_INIT
        self.csv_init_path = csv_init_path

        # GA参数
        self.group_allowed_configs = LayerGroupConfig.GROUP_ALLOWED_CONFIGS

    def create_random_individual(self) -> Dict:
        """创建随机个体"""
        # 为每个group随机选择允许的配置
        groups = []
        for group_id in range(NUM_GROUPS):
            allowed = self.group_allowed_configs[group_id]
            groups.append(random.choice(allowed))

        # 随机初始化交叉率和变异率
        crossover_rate = random.uniform(
            CoarseGAConfig.CROSSOVER_INIT_MIN,
            CoarseGAConfig.CROSSOVER_INIT_MAX
        )
        mutation_rate = random.uniform(
            CoarseGAConfig.MUTATION_INIT_MIN,
            CoarseGAConfig.MUTATION_INIT_MAX
        )

        return {
            'groups': groups,
            'crossover_rate': crossover_rate,
            'mutation_rate': mutation_rate,
            'acc': 0.0,
            'area_ratio': 1.0,
            'area_opt_ratio': 0.0,
            'total_area': 0.0,
        }

    def evaluate_individual(self, individual: Dict) -> Dict:
        """评估个体"""
        groups = individual['groups']

        # 评估准确率
        acc = evaluate_accuracy_from_group_config(groups, verbose=False)
        individual['acc'] = acc

        # 评估面积
        area_info = evaluate_area_from_group_config(groups, return_detail=False)
        individual['area_ratio'] = area_info['area_ratio_vs_baseline']
        individual['area_opt_ratio'] = area_info['area_optimization_ratio']
        individual['total_area'] = area_info['total_area']

        return individual

    def crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """交叉操作"""
        child = self.create_random_individual()

        # 单点交叉
        crossover_point = random.randint(1, NUM_GROUPS - 1)
        child['groups'] = parent1['groups'][:crossover_point] + parent2['groups'][crossover_point:]

        # 确保每个group的配置在允许范围内
        for i in range(NUM_GROUPS):
            if child['groups'][i] not in self.group_allowed_configs[i]:
                child['groups'][i] = random.choice(self.group_allowed_configs[i])

        # 交叉率和变异率的混合
        child['crossover_rate'] = (parent1['crossover_rate'] + parent2['crossover_rate']) / 2.0
        child['crossover_rate'] += random.uniform(-CoarseGAConfig.CROSSOVER_MIX_NOISE,
                                                   CoarseGAConfig.CROSSOVER_MIX_NOISE)
        child['crossover_rate'] = max(CoarseGAConfig.CROSSOVER_MIN,
                                       min(CoarseGAConfig.CROSSOVER_MAX, child['crossover_rate']))

        child['mutation_rate'] = (parent1['mutation_rate'] + parent2['mutation_rate']) / 2.0
        child['mutation_rate'] += random.uniform(-CoarseGAConfig.MUTATION_MIX_NOISE,
                                                  CoarseGAConfig.MUTATION_MIX_NOISE)
        child['mutation_rate'] = max(CoarseGAConfig.MUTATION_MIN,
                                      min(CoarseGAConfig.MUTATION_MAX, child['mutation_rate']))

        return child

    def mutate(self, individual: Dict) -> Dict:
        """变异操作"""
        mutated = individual.copy()
        mutated['groups'] = individual['groups'].copy()

        # 对每个group以mutation_rate概率变异
        for i in range(NUM_GROUPS):
            if random.random() < mutated['mutation_rate']:
                allowed = self.group_allowed_configs[i]
                mutated['groups'][i] = random.choice(allowed)

        # 交叉率和变异率自身变异
        if random.random() < CoarseGAConfig.CROSSOVER_SELF_MUT_PROB:
            mutated['crossover_rate'] += random.uniform(-0.05, 0.05)
            mutated['crossover_rate'] = max(CoarseGAConfig.CROSSOVER_MIN,
                                            min(CoarseGAConfig.CROSSOVER_MAX, mutated['crossover_rate']))

        if random.random() < CoarseGAConfig.MUTATION_SELF_MUT_PROB:
            mutated['mutation_rate'] += random.uniform(-0.02, 0.02)
            mutated['mutation_rate'] = max(CoarseGAConfig.MUTATION_MIN,
                                           min(CoarseGAConfig.MUTATION_MAX, mutated['mutation_rate']))

        return mutated

    def save_population_to_csv(self, population: List[Dict], csv_path: str):
        """保存种群到CSV"""
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # 写入表头
            header = ['acc', 'area_ratio', 'area_opt_ratio', 'total_area',
                      'crossover_rate', 'mutation_rate'] + \
                     [f'g{i}' for i in range(NUM_GROUPS)]
            writer.writerow(header)

            # 写入数据
            for ind in population:
                row = [
                    ind['acc'],
                    ind['area_ratio'],
                    ind['area_opt_ratio'],
                    ind['total_area'],
                    ind['crossover_rate'],
                    ind['mutation_rate'],
                ] + ind['groups']
                writer.writerow(row)

        if self.verbose:
            print(f"Population saved to {csv_path}")


def run_coarse_ga(
    population_size: int = None,
    max_generations: int = None,
    acc_constraint: float = None,
    seed: int = None,
    output_path: str = None,
    tournament_size: int = None,
    elitism_rate: float = None,
) -> Tuple[Dict, List[Dict]]:
    """
    运行粗粒度GA

    参数:
        population_size: int, 种群大小
        max_generations: int, 最大代数
        acc_constraint: float, 准确率约束
        seed: int, 随机种子
        output_path: str, 输出CSV路径

    返回:
        best_individual: Dict, 最优个体
        history: List[Dict], 进化历史
    """
    # 创建输出目录
    PathConfig.create_output_dirs()

    # 创建GA实例
    ga = CoarseGA(
        population_size=population_size,
        max_generations=max_generations,
        acc_constraint=acc_constraint,
        seed=seed,
        tournament_size=tournament_size,
        elitism_rate=elitism_rate,
    )

    # 运行GA
    best, history = ga.run()

    # 保存结果
    if output_path is None:
        import os
        output_path = os.path.join(PathConfig.COARSE_GA_OUTPUT, "final_population.csv")

    if CoarseGAConfig.SAVE_FINAL_POPULATION:
        ga.save_population_to_csv(ga.population, output_path)

    return best, history


if __name__ == "__main__":
    print("=== Running Coarse-Grained GA ===\n")

    best, history = run_coarse_ga(
        population_size=50,
        max_generations=50,
        acc_constraint=75.0,
        seed=42
    )

    print("\n=== Best Individual ===")
    print(f"Groups: {best['groups']}")
    print(f"Accuracy: {best['acc']:.2f}%")
    print(f"Area Ratio: {best['area_ratio']:.4f}")
    print(f"Area Optimization: {best['area_opt_ratio']*100:.2f}%")

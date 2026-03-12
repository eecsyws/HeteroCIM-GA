"""
细粒度遗传算法（96维layer搜索）
不含PatchEmbed和Head（始终FP32）
"""

import random
import csv
from typing import List, Dict, Optional, Tuple

from config import FineGAConfig, HardwareConfig, PathConfig
from evaluators import evaluate_accuracy_from_config, evaluate_area_from_config
from core.model_builder import get_layer_names, expand_group_config_to_layer_config
from .ga_base import GeneticAlgorithmBase


class FineGA(GeneticAlgorithmBase):
    """细粒度GA：在96维layer配置空间搜索（不含PatchEmbed和Head）"""

    def __init__(
        self,
        population_size: int = None,
        max_generations: int = None,
        acc_constraint: Optional[float] = None,
        seed: Optional[int] = None,
        verbose: bool = None,
        use_csv_init: bool = None,
        csv_init_path: str = None,
        enable_mutation_step_limit: bool = None,
        tournament_size: int = None,
        elitism_rate: float = None,
    ):
        """初始化细粒度GA

        参数:
            enable_mutation_step_limit: 是否启用变异步长限制
                - True: 使用MUTATION_STEP_MAX限制变异步长（两阶段GA场景）
                - False: 无变异步长限制（单独运行细粒度GA场景）
                - None: 根据use_csv_init自动判断（从CSV初始化时启用限制）
        """
        # 使用配置文件的默认值
        if population_size is None:
            population_size = FineGAConfig.POPULATION_SIZE
        if max_generations is None:
            max_generations = FineGAConfig.MAX_GENERATIONS
        if acc_constraint is None:
            acc_constraint = FineGAConfig.ACC_CONSTRAINT
        if seed is None:
            seed = FineGAConfig.RANDOM_SEED
        if verbose is None:
            verbose = FineGAConfig.VERBOSE_GA
        if tournament_size is None:
            tournament_size = FineGAConfig.TOURNAMENT_SIZE
        if elitism_rate is None:
            elitism_rate = FineGAConfig.ELITISM_RATE

        super().__init__(population_size, max_generations, acc_constraint, seed, verbose,
                        tournament_size=tournament_size, elitism_rate=elitism_rate)

        self.use_csv_init = use_csv_init if use_csv_init is not None else FineGAConfig.USE_CSV_INIT
        self.csv_init_path = csv_init_path

        # 自动判断是否启用变异步长限制
        if enable_mutation_step_limit is None:
            self.enable_mutation_step_limit = self.use_csv_init
        else:
            self.enable_mutation_step_limit = enable_mutation_step_limit

        self.num_layers = len(get_layer_names())

    def create_random_individual(self) -> Dict:
        """创建随机个体"""
        # 为每层随机选择配置
        layers = [random.randint(0, HardwareConfig.MAX_CONFIG_INDEX) for _ in range(self.num_layers)]

        # 随机初始化交叉率和变异率
        crossover_rate = random.uniform(
            FineGAConfig.CROSSOVER_INIT_MIN,
            FineGAConfig.CROSSOVER_INIT_MAX
        )
        mutation_rate = random.uniform(
            FineGAConfig.MUTATION_INIT_MIN,
            FineGAConfig.MUTATION_INIT_MAX
        )

        return {
            'layers': layers,
            'crossover_rate': crossover_rate,
            'mutation_rate': mutation_rate,
            'acc': 0.0,
            'area_ratio': 1.0,
            'area_opt_ratio': 0.0,
            'total_area': 0.0,
        }

    def create_individual_from_group_config(self, group_config: List[int]) -> Dict:
        """从8维group配置创建个体"""
        layers = expand_group_config_to_layer_config(group_config)

        crossover_rate = random.uniform(
            FineGAConfig.CROSSOVER_INIT_MIN,
            FineGAConfig.CROSSOVER_INIT_MAX
        )
        mutation_rate = random.uniform(
            FineGAConfig.MUTATION_INIT_MIN,
            FineGAConfig.MUTATION_INIT_MAX
        )

        return {
            'layers': layers,
            'crossover_rate': crossover_rate,
            'mutation_rate': mutation_rate,
            'acc': 0.0,
            'area_ratio': 1.0,
            'area_opt_ratio': 0.0,
            'total_area': 0.0,
        }

    def evaluate_individual(self, individual: Dict) -> Dict:
        """评估个体"""
        layers = individual['layers']

        # 评估准确率
        acc = evaluate_accuracy_from_config(layers, verbose=False)
        individual['acc'] = acc

        # 评估面积
        area_info = evaluate_area_from_config(layers, return_detail=False)
        individual['area_ratio'] = area_info['area_ratio_vs_baseline']
        individual['area_opt_ratio'] = area_info['area_optimization_ratio']
        individual['total_area'] = area_info['total_area']

        return individual

    def crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """交叉操作"""
        child = self.create_random_individual()

        # 单点交叉
        crossover_point = random.randint(1, self.num_layers - 1)
        child['layers'] = parent1['layers'][:crossover_point] + parent2['layers'][crossover_point:]

        # 交叉率和变异率的混合
        child['crossover_rate'] = (parent1['crossover_rate'] + parent2['crossover_rate']) / 2.0
        child['crossover_rate'] += random.uniform(-FineGAConfig.CROSSOVER_MIX_NOISE,
                                                   FineGAConfig.CROSSOVER_MIX_NOISE)
        child['crossover_rate'] = max(FineGAConfig.CROSSOVER_MIN,
                                       min(FineGAConfig.CROSSOVER_MAX, child['crossover_rate']))

        child['mutation_rate'] = (parent1['mutation_rate'] + parent2['mutation_rate']) / 2.0
        child['mutation_rate'] += random.uniform(-FineGAConfig.MUTATION_MIX_NOISE,
                                                  FineGAConfig.MUTATION_MIX_NOISE)
        child['mutation_rate'] = max(FineGAConfig.MUTATION_MIN,
                                      min(FineGAConfig.MUTATION_MAX, child['mutation_rate']))

        return child

    def mutate(self, individual: Dict) -> Dict:
        """变异操作（根据enable_mutation_step_limit决定是否限制步长）"""
        mutated = individual.copy()
        mutated['layers'] = individual['layers'].copy()

        # 对每层以mutation_rate概率变异
        for i in range(self.num_layers):
            if random.random() < mutated['mutation_rate']:
                if self.enable_mutation_step_limit:
                    # 小步长变异：±[1, STEP_MAX]（两阶段GA场景）
                    step_choices = list(range(-FineGAConfig.MUTATION_STEP_MAX, 0)) + \
                                   list(range(1, FineGAConfig.MUTATION_STEP_MAX + 1))
                    step = random.choice(step_choices)
                    new_idx = mutated['layers'][i] + step
                    new_idx = max(0, min(HardwareConfig.MAX_CONFIG_INDEX, new_idx))
                else:
                    # 无步长限制：随机选择任意配置（单独运行细粒度GA场景）
                    new_idx = random.randint(0, HardwareConfig.MAX_CONFIG_INDEX)

                mutated['layers'][i] = new_idx

        # 交叉率和变异率自身变异
        if random.random() < FineGAConfig.CROSSOVER_SELF_MUT_PROB:
            mutated['crossover_rate'] += random.uniform(-0.03, 0.03)
            mutated['crossover_rate'] = max(FineGAConfig.CROSSOVER_MIN,
                                            min(FineGAConfig.CROSSOVER_MAX, mutated['crossover_rate']))

        if random.random() < FineGAConfig.MUTATION_SELF_MUT_PROB:
            mutated['mutation_rate'] += random.uniform(-0.01, 0.01)
            mutated['mutation_rate'] = max(FineGAConfig.MUTATION_MIN,
                                           min(FineGAConfig.MUTATION_MAX, mutated['mutation_rate']))

        return mutated

    def initialize_from_coarse_ga_csv(self, csv_path: str, has_header: bool = True) -> List[Dict]:
        """从粗粒度GA的CSV结果初始化种群"""
        population = []

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)

            if has_header:
                next(reader)  # 跳过表头

            for row in reader:
                # 解析CSV行：acc, area_ratio, area_opt_ratio, total_area, cr, mr, g0..g7
                if len(row) < 14:  # 至少需要6个指标 + 8个group配置
                    continue

                group_config = [int(row[i]) for i in range(6, 14)]  # g0..g7
                individual = self.create_individual_from_group_config(group_config)

                # 可选：从CSV读取交叉率和变异率
                try:
                    individual['crossover_rate'] = float(row[4])
                    individual['mutation_rate'] = float(row[5])
                except:
                    pass

                population.append(individual)

                if len(population) >= self.population_size:
                    break

        if self.verbose:
            print(f"Initialized {len(population)} individuals from {csv_path}")

        return population

    def save_population_to_csv(self, population: List[Dict], csv_path: str):
        """保存种群到CSV"""
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # 写入表头
            header = ['acc', 'area_ratio', 'area_opt_ratio', 'total_area',
                      'crossover_rate', 'mutation_rate'] + \
                     [f'layer_{i}' for i in range(self.num_layers)]
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
                ] + ind['layers']
                writer.writerow(row)

        if self.verbose:
            print(f"Population saved to {csv_path}")

    def run(self) -> Tuple[Dict, List[Dict]]:
        """运行GA（覆盖基类方法以支持CSV初始化）"""
        # 初始化种群
        if self.use_csv_init and self.csv_init_path:
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"Initializing population from CSV: {self.csv_init_path}")
                print(f"Mutation step limit: {'ENABLED' if self.enable_mutation_step_limit else 'DISABLED'}")
                if self.enable_mutation_step_limit:
                    print(f"Mutation step range: ±[1, {FineGAConfig.MUTATION_STEP_MAX}]")
                print(f"{'='*70}")
            try:
                self.population = self.initialize_from_coarse_ga_csv(
                    self.csv_init_path,
                    has_header=FineGAConfig.CSV_INIT_HAS_HEADER
                )

                # 如果CSV中的个体数量不足，补充随机个体
                if len(self.population) < self.population_size:
                    if self.verbose:
                        print(f"CSV only has {len(self.population)} individuals, "
                              f"generating {self.population_size - len(self.population)} random individuals")
                    while len(self.population) < self.population_size:
                        self.population.append(self.create_random_individual())

            except Exception as e:
                print(f"Warning: Failed to load CSV: {e}")
                print("Falling back to random initialization")
                self.population = [self.create_random_individual() for _ in range(self.population_size)]
        else:
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"Initializing random population of size {self.population_size}...")
                print(f"Mutation step limit: {'ENABLED' if self.enable_mutation_step_limit else 'DISABLED'}")
                if self.enable_mutation_step_limit:
                    print(f"Mutation step range: ±[1, {FineGAConfig.MUTATION_STEP_MAX}]")
                print(f"{'='*70}")
            self.population = [self.create_random_individual() for _ in range(self.population_size)]

        # 评估初始种群
        if self.verbose:
            print(f"\nEvaluating initial population...")
        for i, ind in enumerate(self.population):
            self.evaluate_individual(ind)
            if self.verbose:
                print(f"  Individual {i+1}/{self.population_size}: "
                      f"acc={ind['acc']:.2f}%, area_ratio={ind['area_ratio']:.4f}")

        # 调用基类的进化循环
        return super().run()


def run_fine_ga(
    population_size: int = None,
    max_generations: int = None,
    acc_constraint: float = None,
    seed: int = None,
    csv_init_path: str = None,
    output_path: str = None,
    enable_mutation_step_limit: bool = None,
    tournament_size: int = None,
    elitism_rate: float = None,
) -> Tuple[Dict, List[Dict]]:
    """
    运行细粒度GA

    参数:
        population_size: int, 种群大小
        max_generations: int, 最大代数
        acc_constraint: float, 准确率约束
        seed: int, 随机种子
        csv_init_path: str, 初始化CSV路径
        output_path: str, 输出CSV路径
        enable_mutation_step_limit: bool, 是否启用变异步长限制
            - True: 使用MUTATION_STEP_MAX限制变异步长（两阶段GA场景）
            - False: 无变异步长限制（单独运行细粒度GA场景）
            - None: 根据是否从CSV初始化自动判断

    返回:
        best_individual: Dict, 最优个体
        history: List[Dict], 进化历史
    """
    # 创建输出目录
    PathConfig.create_output_dirs()

    # 默认从粗粒度GA结果初始化
    if csv_init_path is None:
        import os
        csv_init_path = os.path.join(PathConfig.COARSE_GA_OUTPUT, "final_population.csv")

    # 创建GA实例
    ga = FineGA(
        population_size=population_size,
        max_generations=max_generations,
        acc_constraint=acc_constraint,
        seed=seed,
        csv_init_path=csv_init_path,
        enable_mutation_step_limit=enable_mutation_step_limit,
        tournament_size=tournament_size,
        elitism_rate=elitism_rate,
    )

    # 运行GA
    best, history = ga.run()

    # 保存结果
    if output_path is None:
        import os
        output_path = os.path.join(PathConfig.FINE_GA_OUTPUT, "final_population.csv")

    if FineGAConfig.SAVE_FINAL_POPULATION:
        ga.save_population_to_csv(ga.population, output_path)

    return best, history


if __name__ == "__main__":
    print("=== Running Fine-Grained GA ===\n")

    best, history = run_fine_ga(
        population_size=50,
        max_generations=10,
        acc_constraint=75.0,
        seed=42
    )

    print("\n=== Best Individual ===")
    print(f"Accuracy: {best['acc']:.2f}%")
    print(f"Area Ratio: {best['area_ratio']:.4f}")
    print(f"Area Optimization: {best['area_opt_ratio']*100:.2f}%")
    print(f"Layers (first 10): {best['layers'][:10]}")

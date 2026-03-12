"""
遗传算法基类
提供通用的GA操作和工具函数
"""

import random
import csv
import numpy as np
from typing import List, Dict, Tuple, Callable, Optional
from abc import ABC, abstractmethod


class GeneticAlgorithmBase(ABC):
    """遗传算法基类"""

    def __init__(
        self,
        population_size: int,
        max_generations: int,
        acc_constraint: Optional[float] = None,
        seed: Optional[int] = None,
        verbose: bool = True,
        tournament_size: int = None,
        elitism_rate: float = None
    ):
        """
        初始化GA

        参数:
            population_size: int, 种群大小
            max_generations: int, 最大迭代代数
            acc_constraint: Optional[float], 准确率约束（%）
            seed: Optional[int], 随机种子
            verbose: bool, 是否打印详细信息
            tournament_size: int, 锦标赛规模
            elitism_rate: float, 精英保留率
        """
        # 默认值处理
        if tournament_size is None:
            tournament_size = 2  # 保守默认值
        if elitism_rate is None:
            elitism_rate = 0.02  # 默认2%

        self.population_size = population_size
        self.max_generations = max_generations
        self.acc_constraint = acc_constraint
        self.verbose = verbose
        self.tournament_size = tournament_size
        self.elitism_rate = elitism_rate

        # 调试：确认配置参数
        print(f"[DEBUG] tournament_size={tournament_size}, elitism_rate={elitism_rate}, "
              f"num_elites={int(np.ceil(population_size * elitism_rate))}")

        # 计算精英个体数量（向上取整）
        self.num_elites = int(np.ceil(population_size * elitism_rate))

        if seed is not None:
            random.seed(seed)

        self.population = []
        self.history = []

    @abstractmethod
    def create_random_individual(self) -> Dict:
        """创建随机个体（需要子类实现）"""
        pass

    @abstractmethod
    def evaluate_individual(self, individual: Dict) -> Dict:
        """评估个体（需要子类实现）"""
        pass

    @abstractmethod
    def crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """交叉操作（需要子类实现）"""
        pass

    @abstractmethod
    def mutate(self, individual: Dict) -> Dict:
        """变异操作（需要子类实现）"""
        pass

    def pareto_dominates(self, ind1: Dict, ind2: Dict) -> bool:
        """
        判断ind1是否Pareto支配ind2
        目标：最大化acc，最小化area_ratio
        """
        acc1, area1 = ind1['acc'], ind1['area_ratio']
        acc2, area2 = ind2['acc'], ind2['area_ratio']

        # ind1至少在一个目标上更好，且在所有目标上不差
        better_acc = acc1 >= acc2
        better_area = area1 <= area2

        strictly_better = (acc1 > acc2) or (area1 < area2)

        return better_acc and better_area and strictly_better

    def calculate_crowding_distance(self, individuals: List[Dict]) -> Dict:
        """
        计算种群中每个个体的拥挤距离

        拥挤距离衡量一个个体被其他个体包围的程度。
        拥挤距离越大，说明周围越空旷，多样性越好。

        参数:
            individuals: 个体列表

        返回:
            crowding: Dict, {个体索引: 拥挤距离}
        """
        n = len(individuals)
        if n <= 2:
            return {i: float('inf') for i in range(n)}

        crowding = {i: 0.0 for i in range(n)}

        # acc: 越大越好；area_ratio: 越小越好
        # 统一按升序排序，公式统一
        objectives = ['acc', 'area_ratio']

        for obj in objectives:
            sorted_indices = sorted(range(n), key=lambda i: individuals[i][obj])

            crowding[sorted_indices[0]] = float('inf')
            crowding[sorted_indices[-1]] = float('inf')

            obj_min = individuals[sorted_indices[0]][obj]
            obj_max = individuals[sorted_indices[-1]][obj]
            obj_range = obj_max - obj_min

            if obj_range == 0:
                continue

            for i in range(1, n - 1):
                idx = sorted_indices[i]
                prev_val = individuals[sorted_indices[i - 1]][obj]
                next_val = individuals[sorted_indices[i + 1]][obj]
                crowding[idx] += (next_val - prev_val) / obj_range

        return crowding

    def calculate_rank_and_crowding(self, population: List[Dict]) -> Tuple[List[int], Dict]:
        """
        对种群进行非支配排序并计算拥挤距离

        返回:
            ranks: 每个个体的Pareto前沿编号（越小越优）
            crowding: 每个个体的拥挤距离（越大越优）
        """
        n = len(population)
        if n == 0:
            return [], {}

        # 计算支配关系
        domination_count = [0] * n  # 支配该个体的数量
        dominated_set = [set() for _ in range(n)]  # 该个体支配的集合

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self.pareto_dominates(population[i], population[j]):
                    dominated_set[i].add(j)
                elif self.pareto_dominates(population[j], population[i]):
                    domination_count[i] += 1

        # 第一前沿
        fronts = [[]]
        for i in range(n):
            if domination_count[i] == 0:
                fronts[0].append(i)

        # 逐层分配前沿
        current_front = 0
        while current_front < len(fronts) and fronts[current_front]:
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front += 1
            if next_front:
                fronts.append(next_front)

        # 分配rank
        ranks = [0] * n
        for rank, front in enumerate(fronts):
            for idx in front:
                ranks[idx] = rank

        # 计算拥挤距离（仅对每个前沿计算）
        crowding = {i: 0.0 for i in range(n)}
        for front in fronts:
            if len(front) > 2:
                front_individuals = [population[i] for i in front]
                front_crowding = self.calculate_crowding_distance(front_individuals)
                for i, idx in enumerate(front):
                    crowding[idx] = front_crowding[i]

        return ranks, crowding

    def compare_individuals(
        self,
        ind1: Dict,
        ind2: Dict,
    ) -> int:
        """
        比较两个个体的优劣（基于约束 + Pareto支配 + 拥挤距离）

        规则：
        1. A和B都在约束以上：用Pareto支配，支配则优；非支配则拥挤距离大的优
        2. A在约束以上，B在约束以下：A更优
        3. A在约束以下，B在约束以上：B更优
        4. A和B都在约束以下：准确率高的更优

        返回:
            1: ind1更优
            -1: ind2更优
            0: 相同
        """
        meets1 = ind1['acc'] >= self.acc_constraint if self.acc_constraint is not None else True
        meets2 = ind2['acc'] >= self.acc_constraint if self.acc_constraint is not None else True

        # 规则2：A满足，B不满足
        if meets1 and not meets2:
            return 1
        # 规则3：A不满足，B满足
        if not meets1 and meets2:
            return -1

        # 规则4：A和B都不满足约束：准确率高的更优
        if not meets1 and not meets2:
            if ind1['acc'] > ind2['acc']:
                return 1
            elif ind2['acc'] > ind1['acc']:
                return -1
            return 0

        # 规则1：A和B都在约束以上：用Pareto支配
        if self.pareto_dominates(ind1, ind2):
            return 1
        if self.pareto_dominates(ind2, ind1):
            return -1

        # 非Pareto支配：返回0（在锦标赛中可进一步用拥挤距离）
        return 0

    def select_parents(self, population: List[Dict], ranks: List[int] = None, crowding: Dict = None) -> Tuple[Dict, Dict]:
        """锦标赛选择（基于rank + 拥挤距离）"""
        def tournament():
            # 使用配置的tournament_size
            k = min(self.tournament_size, len(population))
            candidates = random.sample(population, k)

            # 找出候选个体在原种群中的索引
            candidate_indices = []
            for c in candidates:
                for idx, ind in enumerate(population):
                    if ind is c:
                        candidate_indices.append(idx)
                        break

            # 使用rank和crowding比较选择最优个体
            best = candidates[0]
            best_idx = candidate_indices[0]
            for i in range(1, len(candidates)):
                cand_idx = candidate_indices[i]

                # rank越小越优
                if ranks is not None and crowding is not None:
                    rank1, rank2 = ranks[cand_idx], ranks[best_idx]
                    if rank1 < rank2:
                        best = candidates[i]
                        best_idx = cand_idx
                    elif rank1 == rank2:
                        # rank相同时，用crowding比较
                        if crowding[cand_idx] > crowding[best_idx]:
                            best = candidates[i]
                            best_idx = cand_idx
                else:
                    # 如果没有预计算的rank和crowding，使用旧的比较方式
                    result = self.compare_individuals(candidates[i], best)
                    if result == 1:
                        best = candidates[i]
                        best_idx = cand_idx

            return best, best_idx

        # 选择第一个父本
        p1, idx1 = tournament()
        # 选择第二个父本，确保不是同一个
        p2 = p1
        attempts = 0
        while p2 is p1 and attempts < 10:
            p2, idx2 = tournament()
            attempts += 1

        return p1, p2

    def initialize_population_from_csv(
        self,
        csv_path: str,
        has_header: bool = True
    ) -> List[Dict]:
        """从CSV文件初始化种群（需要子类实现具体解析逻辑）"""
        raise NotImplementedError("Subclass must implement CSV initialization")

    def save_population_to_csv(
        self,
        population: List[Dict],
        csv_path: str
    ):
        """保存种群到CSV（需要子类实现具体格式）"""
        raise NotImplementedError("Subclass must implement CSV saving")

    def run(self) -> Tuple[Dict, List[Dict]]:
        """运行GA（主循环）"""
        # 初始化种群（如果子类没有预先初始化）
        if not self.population:
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"Initializing population of size {self.population_size}...")
                print(f"{'='*70}")

            self.population = [self.create_random_individual() for _ in range(self.population_size)]

            # 评估初始种群
            for i, ind in enumerate(self.population):
                self.evaluate_individual(ind)
                if self.verbose:
                    print(f"  Individual {i+1}/{self.population_size}: "
                          f"acc={ind['acc']:.2f}%, area_ratio={ind['area_ratio']:.4f}")

        # 进化循环
        for generation in range(self.max_generations):
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"Generation {generation+1}/{self.max_generations}")
                print(f"{'='*70}")

            # 统计当前种群信息
            current_best = max(self.population, key=lambda x: (x['acc'], -x['area_ratio']))
            mean_acc = sum(ind['acc'] for ind in self.population) / len(self.population)
            mean_area = sum(ind['area_ratio'] for ind in self.population) / len(self.population)
            mean_cr = sum(ind['crossover_rate'] for ind in self.population) / len(self.population)
            mean_mr = sum(ind['mutation_rate'] for ind in self.population) / len(self.population)

            if self.verbose:
                print(f"Current Best: acc={current_best['acc']:.2f}%, "
                      f"area_ratio={current_best['area_ratio']:.4f}, "
                      f"area_opt={current_best['area_opt_ratio']*100:.2f}%")
                print(f"Population Mean: acc={mean_acc:.2f}%, "
                      f"area_ratio={mean_area:.4f}")
                print(f"Evolution Params: CR={mean_cr:.3f}, MR={mean_mr:.3f}")

            # 生成新一代
            if self.verbose:
                print(f"\nGenerating offspring...")

            # 预计算种群的rank和crowding（用于锦标赛选择）
            ranks, crowding = self.calculate_rank_and_crowding(self.population)
            # 将rank和crowding添加到每个个体中
            for i, ind in enumerate(self.population):
                ind['_rank'] = ranks[i]
                ind['_crowding'] = crowding[i]

            # 精英保留：先从旧一代选出精英
            if self.num_elites > 0 and len(self.population) > 0:
                sorted_old_idx = sorted(range(len(self.population)),
                                        key=lambda i: (self.population[i]['acc'], -self.population[i]['area_ratio']),
                                        reverse=True)[:self.num_elites]
                elites = [self.population[i].copy() for i in sorted_old_idx]

                if self.verbose:
                    print(f"  Elitism: Selected {self.num_elites} elites from generation {generation}")
                    for i, elite in enumerate(elites):
                        print(f"    Elite {i+1}: acc={elite['acc']:.2f}%, area_ratio={elite['area_ratio']:.4f}")
            else:
                elites = []

            # 计算需要生成的子代数量
            offspring_needed = self.population_size - len(elites)

            # 保存当前种群（用于父代选择）
            old_population = self.population.copy()

            new_population = []
            offspring_count = 0

            # 只生成需要的子代数量
            while len(new_population) < offspring_needed:
                # 选择父代（传入预计算的rank和crowding）
                parent1, parent2 = self.select_parents(self.population, ranks, crowding)

                # 交叉
                use_crossover = random.random() < parent1['crossover_rate']
                if use_crossover:
                    child = self.crossover(parent1, parent2)
                    operation = "crossover"
                else:
                    child = parent1.copy()
                    operation = "copy"

                # 变异
                use_mutation = random.random() < child['mutation_rate']
                if use_mutation:
                    child = self.mutate(child)
                    operation += "+mutation"

                # 评估
                self.evaluate_individual(child)
                new_population.append(child)
                offspring_count += 1

                # 每个个体都打印
                if self.verbose:
                    print(f"  Offspring {offspring_count}/{offspring_needed}: "
                          f"{operation:20s} -> acc={child['acc']:.2f}%, "
                          f"area_ratio={child['area_ratio']:.4f}")

            # 将精英添加到新一代（直接添加，不替换）
            new_population.extend(elites)
            self.population = new_population

            # 记录历史
            best = max(self.population, key=lambda x: (x['acc'], -x['area_ratio']))
            self.history.append({
                'generation': generation + 1,
                'best_acc': best['acc'],
                'best_area_ratio': best['area_ratio'],
                'mean_acc': mean_acc,
                'mean_area_ratio': mean_area,
                'mean_crossover_rate': mean_cr,
                'mean_mutation_rate': mean_mr,
            })

            if self.verbose:
                print(f"\nGeneration {generation+1} Summary:")
                print(f"  Best Individual: acc={best['acc']:.2f}%, "
                      f"area_ratio={best['area_ratio']:.4f}")
                if generation > 0:
                    prev_best_acc = self.history[generation-1]['best_acc']
                    prev_best_area = self.history[generation-1]['best_area_ratio']
                    acc_change = best['acc'] - prev_best_acc
                    area_change = best['area_ratio'] - prev_best_area
                    print(f"  Improvement: acc {acc_change:+.2f}%, "
                          f"area_ratio {area_change:+.4f}")

        # 最终统计
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"GA Completed!")
            print(f"{'='*70}")

        # 返回最优个体和历史
        best_individual = max(self.population, key=lambda x: (x['acc'], -x['area_ratio']))

        if self.verbose:
            print(f"\nFinal Best Individual:")
            print(f"  Accuracy: {best_individual['acc']:.2f}%")
            print(f"  Area Ratio: {best_individual['area_ratio']:.4f}")
            print(f"  Area Optimization: {best_individual['area_opt_ratio']*100:.2f}%")
            print(f"  Crossover Rate: {best_individual['crossover_rate']:.3f}")
            print(f"  Mutation Rate: {best_individual['mutation_rate']:.3f}")

        return best_individual, self.history

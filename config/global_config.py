"""
全局配置文件
============================================================
集中管理所有实验参数，便于不同实验场景下快速调整配置
"""

import os

# ============================================================
# 1. 路径配置
# ============================================================
class PathConfig:
    """路径相关配置"""
    # 项目根目录
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 模型和数据集路径（服务器路径，本地运行时可能不存在）
    MODEL_PATH = "/lustre/home/2200012654/model/timm/vit_tiny_cifar100/vit_tiny_cifar100_finetune.pt"
    TEST_DIR = "/lustre/home/2200012654/dataset/cifar100/test"

    # 输出目录
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
    COARSE_GA_OUTPUT = os.path.join(OUTPUT_DIR, "coarse_ga")
    FINE_GA_OUTPUT = os.path.join(OUTPUT_DIR, "fine_ga")
    ANALYSIS_OUTPUT = os.path.join(OUTPUT_DIR, "analysis")

    @classmethod
    def create_output_dirs(cls):
        """创建所有输出目录"""
        for dir_path in [cls.OUTPUT_DIR, cls.COARSE_GA_OUTPUT,
                         cls.FINE_GA_OUTPUT, cls.ANALYSIS_OUTPUT]:
            os.makedirs(dir_path, exist_ok=True)


# ============================================================
# 2. 硬件配置表 (Config Table)
# ============================================================
class HardwareConfig:
    """硬件配置相关参数"""

    # Config Table: (quant_bits, noise_bits)
    # index -> (量化位宽, NVM位数)
    # 格式: SRAM_bits/NVM_bits
    # INT8 区（4/4, 3/5, 2/6, 1/7, 0/8）
    # INT7 区（4/3, 3/4, 2/5, 1/6, 0/7）
    # INT6 区（3/3, 2/4, 1/5, 0/6）
    # INT5 区（3/2, 2/3, 1/4, 0/5）
    # INT4 区（2/2, 1/3, 0/4）
    # INT3 区（2/1, 1/2, 0/3）
    CONFIG_TABLE = [
        # INT8 区
        (8, 4),    # 0 : INT8, SRAM/NVM = 4/4
        (8, 5),    # 1 : INT8, 3/5
        (8, 6),    # 2 : INT8, 2/6
        (8, 7),    # 3 : INT8, 1/7
        (8, 8),    # 4 : INT8, 0/8

        # INT7 区
        (7, 3),    # 5 : INT7, 4/3
        (7, 4),    # 6 : INT7, 3/4
        (7, 5),    # 7 : INT7, 2/5
        (7, 6),    # 8 : INT7, 1/6
        (7, 7),    # 9 : INT7, 0/7

        # INT6 区
        (6, 3),    # 10: INT6, 3/3
        (6, 4),    # 11: INT6, 2/4
        (6, 5),    # 12: INT6, 1/5
        (6, 6),    # 13: INT6, 0/6

        # INT5 区
        (5, 2),    # 14: INT5, 3/2
        (5, 3),    # 15: INT5, 2/3
        (5, 4),    # 16: INT5, 1/4
        (5, 5),    # 17: INT5, 0/5

        # INT4 区
        (4, 2),    # 18: INT4, 2/2
        (4, 3),    # 19: INT4, 1/3
        (4, 4),    # 20: INT4, 0/4

        # INT3 区
        (3, 1),    # 21: INT3, 2/1
        (3, 2),    # 22: INT3, 1/2
        (3, 3),    # 23: INT3, 0/3
    ]

    MAX_CONFIG_INDEX = len(CONFIG_TABLE) - 1  # 23

    # 噪声强度配置
    NOISE_SIGMA = 0.15  # NVM variation噪声标准差

    # 面积参数配置
    # 工艺节点 (nm)
    PROCESS_NM = 28

    # 单元面积 (F^2)
    SRAM_CELL_AREA_F2 = 160.0
    NVM_CELL_AREA_F2 = 6.0

    # Baseline配置 (用于面积对比)
    BASELINE_QUANT_BITS = 8
    BASELINE_NVM_BITS = 0  # 全SRAM


# ============================================================
# 3. 层分组配置
# ============================================================
class LayerGroupConfig:
    """
    ViT层分组配置
    根据层种类将所有层分为8组：
    0: Q Linear
    1: K Linear
    2: V Linear
    3: QK^T (Softmax pre-multiplication)
    4: AV (Value weighted sum)
    5: Output Linear (attn.proj)
    6: FC1 (MLP fc1)
    7: FC2 (MLP fc2)
    """

    # 层名 -> 组ID 映射（仅包含96个可搜索层，不包含PatchEmbed和Head）
    LAYER_TO_GROUP = {}

    # 12 个 Transformer Blocks × 8 个子操作 = 96 层
    for i in range(12):
        prefix = f"blocks.{i}."
        # Attention 映射
        LAYER_TO_GROUP[prefix + "attn.q"]    = 0
        LAYER_TO_GROUP[prefix + "attn.k"]    = 1
        LAYER_TO_GROUP[prefix + "attn.v"]    = 2
        LAYER_TO_GROUP[prefix + "attn.qk"]   = 3
        LAYER_TO_GROUP[prefix + "attn.av"]   = 4
        LAYER_TO_GROUP[prefix + "attn.proj"] = 5

        # MLP 映射
        LAYER_TO_GROUP[prefix + "mlp.fc1"]   = 6
        LAYER_TO_GROUP[prefix + "mlp.fc2"]   = 7

    # 注意：PatchEmbed和Head不参与GA搜索，始终使用FP32

    # 每个组允许的配置索引（由用户规定）
    GROUP_ALLOWED_CONFIGS = {
        0: [15, 16, 17, 19, 20, 22, 23],              # Q Linear: INT5,4,3
        1: [11, 12, 13, 15, 16, 17, 19, 20],          # K Linear: INT6,5,4
        2: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], # V Linear: INT6,5,4
        3: [10, 14, 18],                              # QK^T: INT6,5,4 (Full SRAM)
        4: [0, 5, 10],                                # AV: INT8,7,6 (Full SRAM)
        5: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], # Output Linear: INT6,5,4
        6: [5, 6, 7, 10, 11, 12, 14, 15, 16],         # FC1: INT7,6,5
        7: [5, 6, 7, 10, 11, 12, 14, 15, 16],         # FC2: INT7,6,5
    }

# ============================================================
# 4. 遗传算法配置
# ============================================================
class CoarseGAConfig:
    """粗粒度遗传算法配置（8维层种类搜索）"""

    # 种群参数
    POPULATION_SIZE = 50
    MAX_GENERATIONS = 50

    # 搜索维度: 8 (对应 LayerGroupConfig 的 8 个组)
    NUM_GROUPS = 8

    # 锦标赛选择参数
    TOURNAMENT_SIZE = 3  # 锦标赛规模

    # 精英保留参数
    ELITISM_RATE = 0.02  # 精英保留率（2%）

    # 准确率约束
    ACC_CONSTRAINT = 75.0  # 最低准确率要求(%)

    # 交叉率配置
    CROSSOVER_INIT_MIN = 0.6
    CROSSOVER_INIT_MAX = 0.9
    CROSSOVER_MIN = 0.3
    CROSSOVER_MAX = 0.95
    CROSSOVER_MIX_NOISE = 0.05
    CROSSOVER_SELF_MUT_PROB = 0.1

    # 变异率配置
    MUTATION_INIT_MIN = 0.02
    MUTATION_INIT_MAX = 0.15
    MUTATION_MIN = 0.01
    MUTATION_MAX = 0.3
    MUTATION_MIX_NOISE = 0.05
    MUTATION_SELF_MUT_PROB = 0.1

    # 日志配置
    VERBOSE_EVAL = True
    VERBOSE_GA = True
    PRINT_FINAL_POPULATION = True
    LOG_EVERY = 1

    # CSV配置
    SAVE_FINAL_POPULATION = True
    USE_CSV_INIT = False
    CSV_INIT_HAS_HEADER = True

    # 随机种子
    RANDOM_SEED = 42


class FineGAConfig:
    """细粒度遗传算法配置（98维layer搜索）"""

    # 种群参数
    POPULATION_SIZE = 50
    MAX_GENERATIONS = 10  # 细粒度搜索迭代次数较少

    # 锦标赛选择参数
    TOURNAMENT_SIZE = 3  # 锦标赛规模

    # 精英保留参数
    ELITISM_RATE = 0.02  # 精英保留率（2%）

    # 准确率约束
    ACC_CONSTRAINT = 75.0

    # 交叉率配置（相对保守）
    CROSSOVER_INIT_MIN = 0.6
    CROSSOVER_INIT_MAX = 0.9
    CROSSOVER_MIN = 0.3
    CROSSOVER_MAX = 0.95
    CROSSOVER_MIX_NOISE = 0.03
    CROSSOVER_SELF_MUT_PROB = 0.1

    # 变异率配置（更保守）
    MUTATION_INIT_MIN = 0.01
    MUTATION_INIT_MAX = 0.05
    MUTATION_MIN = 0.01
    MUTATION_MAX = 0.2
    MUTATION_MIX_NOISE = 0.02
    MUTATION_SELF_MUT_PROB = 0.1

    # 细粒度变异步长
    MUTATION_STEP_MAX = 3  # 变异时index变化范围: ±[1, STEP_MAX]
                           # 仅在从粗粒度初始化时生效，跳过粗粒度时无限制


    # 日志配置
    VERBOSE_EVAL = True
    VERBOSE_GA = True
    PRINT_FINAL_POPULATION = True
    LOG_EVERY = 1

    # CSV配置
    SAVE_FINAL_POPULATION = True
    USE_CSV_INIT = True  # 从粗粒度GA结果初始化
    CSV_INIT_HAS_HEADER = True

    # 随机种子
    RANDOM_SEED = 42


# ============================================================
# 5. 模型推理配置
# ============================================================
class InferenceConfig:
    """模型推理相关配置"""

    # 图像参数
    IMG_SIZE = 224

    # 数据加载参数
    BATCH_SIZE = 512
    NUM_WORKERS = 16

    # 设备配置
    USE_CUDA = True

    # 随机种子配置
    USE_GLOBAL_SEED = True
    GLOBAL_SEED = 42

    # 静态激活值配置（用于加速评估）
    USE_STATIC_ACTIVATION = False


# ============================================================
# 6. ViT模型结构配置
# ============================================================
class ViTConfig:
    """ViT-tiny模型结构参数"""

    EMBED_DIM = 192
    MLP_DIM = 768
    NUM_CLASSES = 100
    PATCH_SIZE = 16
    IN_CHANNELS = 3
    IMG_SIZE = 224
    NUM_BLOCKS = 12

    # 计算派生参数
    NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2  # 196
    NUM_TOKENS = NUM_PATCHES + 1  # 197 (含cls token)


# ============================================================
# 7. 结果分析配置
# ============================================================
class AnalysisConfig:
    """结果分析相关配置"""

    # 图表参数
    FIGURE_DPI = 300
    FIGURE_SIZE = (10, 6)

    # Pareto前沿分析
    PARETO_PLOT_ALPHA = 0.6
    PARETO_POINT_SIZE = 80

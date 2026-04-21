"""配置文件"""

import os

# ==================== 配置文件 ====================
class Config:
    """配置文件类"""

    # 基础路径
    BASE_DIR = "/root/autodl-tmp/CM4SL"
    # BASE_DIR = r"E:\Projects\Columbina model for synthetic lethality"

    # 模型配置
    USE_PRETRAINED = True
    MODEL_TYPE = "local"  # "local", "huggingface", "simple"
    LOCAL_MODEL_PATH = os.path.join(BASE_DIR, "pretrained_models/pubmedbert")

    # 数据文件路径
    DATA_DIR = os.path.join(BASE_DIR, "input_data")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")

    # 文件路径 - 更新为新的CSV格式
    SL_PAIRS_FILE = os.path.join(DATA_DIR, "integrated_sl_pairs_3.csv")
    NON_SL_PAIRS_FILE = os.path.join(DATA_DIR, "gene_nonsl_gene.tsv")
    KG_EDGES_FILE = os.path.join(DATA_DIR, "reformed_KG_edges.csv")
    KG_NODES_FILE = os.path.join(DATA_DIR, "reformed_KG_nodes.csv")
    STRING_FILE = os.path.join(DATA_DIR, "string_interactions.tsv")

    # 组学特征文件
    FEATURE_FILES = {
        'expression': os.path.join(DATA_DIR, 'expression.csv'),
        'cnv': os.path.join(DATA_DIR, 'cnv.csv'),
        'mutation': os.path.join(DATA_DIR, 'mutation.csv'),
        'methylation': os.path.join(DATA_DIR, 'methylation.csv'),
        'dependency': os.path.join(DATA_DIR, 'dependency.csv')
    }

    # FEATURE_FILES = {
    #     'expression': os.path.join(DATA_DIR, 'expression.csv'),
    #     'cnv': os.path.join(DATA_DIR, 'cnv.csv'),
    #     'mutation': os.path.join(DATA_DIR, 'mutation.csv'),
    #     'methylation': os.path.join(DATA_DIR, 'methylation.csv'),
    #     'dependency': os.path.join(DATA_DIR, 'dependency.csv')
    # }

    # 基因ID映射文件
    GENE_ID_MAPPING_FILE = os.path.join(DATA_DIR, "human_gene_mapping.csv")

    # ID锚定模式
    USE_ID_ANCHORING = True  # 启用ID锚定模式
    GENE_ID_COLUMN = 'ENTREZID'  # 使用的ID列名
    GENE_SYMBOL_COLUMN = 'SYMBOL'  # 基因符号列名

    # 训练参数
    EPOCHS = 100
    LEARNING_RATE = 0.0005
    BATCH_SIZE = 160
    HIDDEN_DIM = 128
    EMBEDDING_DIM = 64
    DROPOUT = 0.3
    WEIGHT_DECAY = 1e-5
    EARLY_STOPPING_PATIENCE = 50

    # 知识图谱构建参数
    MAX_NODES_PER_TYPE = 10000  # 每种节点类型的最大节点数
    MAX_EDGES_PER_RELATION = 100000  # 每种关系的最大边数
    DEBUG_MODE = True  # 启用调试模式

    # 性能优化参数
    CHUNKSIZE = 100000  # 分块处理大小
    MAX_STRING_EDGES = 10000  # 最大STRING边数
    MAX_KG_EDGES_PER_RELATION = 5000  # 每种关系的最大KG边数
    PROCESS_SAMPLE_SIZE = 50000  # 处理时的采样大小
    MAX_NODE_TYPES = 20  # 最大处理的节点类型数
    MAX_RELATION_TYPES = 30  # 最大处理的关系类型数

    # 知识图谱特定配置
    KG_GENE_TYPES = ['Gene', 'gene', 'Gene/Protein', 'Protein']  # 基因类型识别
    KG_FILTER_ONLY_GENE_EDGES = True  # 只保留至少一端是基因的边
    KG_USE_ID_FOR_MAPPING = True  # 优先使用x_id/y_id进行映射

    # 基因名称标准化参数（与ID锚定兼容）
    USE_GENE_NAME_NORMALIZATION = True  # 启用基因名称标准化
    GENE_SYNONYM_FILE = os.path.join(DATA_DIR, "gene_synonyms.csv")  # 基因同义词文件（可选）

    # 设备配置
    FORCE_GPU_PROCESSING = True  # 强制在GPU上处理所有数据
    GPU_MEMORY_LIMIT = 1  # GPU内存使用限制（0-1）

    # 训练稳定性参数
    GRADIENT_CLIP_VALUE = 1.0  # 梯度裁剪阈值
    GRADIENT_CLIP_NORM = 1.0  # 梯度裁剪范数
    USE_GRADIENT_CLIPPING = True  # 启用梯度裁剪
    USE_MIXED_PRECISION = False  # 使用混合精度训练
    CHECK_NAN_INF = True  # 检查NaN和Inf
    MEMORY_CLEANUP_FREQUENCY = 10  # 内存清理频率
    DETECT_ANOMALY = True  # 启用梯度异常检测


    # 数据划分参数
    TRAIN_VAL_SPLIT_RATIO = 0.8  # 训练集比例
    RANDOM_SEED = 42  # 随机种子
    BALANCE_CLASSES = True  # 是否平衡正负样本
    STRATIFIED_SPLIT = True  # 是否分层抽样
    TRAIN_VAL_TEST_RATIOS = (0.8, 0.1, 0.1)  # 训练集:验证集:测试集 = 8:1:1
    TOTAL_SAMPLES_PER_CLASS = 20000  # 每类样本总数
    USE_SAMPLING = True  # 是否启用抽样（默认为True）

    # 交叉验证配置
    ENABLE_CROSS_VALIDATION = False  # 是否启用交叉验证
    N_FOLDS = 10  # 交叉验证折数
    SAVE_FOLD_DATA = True  # 保存每折数据
    SAVE_FOLD_MODELS = True  # 保存每折模型
    SAVE_DETAILED_REPORTS = True  # 保存详细报告
    SAVE_COMPARISON_CHARTS = True  # 保存比较图表

    USE_EXTERNAL_SPLIT = False  # 默认关闭
    EXTERNAL_TRAIN_FILE = None
    EXTERNAL_VAL_FILE = None
    EXTERNAL_TEST_FILE = None








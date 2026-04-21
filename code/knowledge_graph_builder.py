"""知识图谱构建器"""

import torch
import pandas as pd
import numpy as np
import gc
import time
import os
import re
import logging
import traceback
from collections import defaultdict
from torch_geometric.data import Data, HeteroData

from config import Config
from device_manager import DeviceManager
from gene_mapping import GeneIDMapper, GeneNameNormalizer

logger = logging.getLogger(__name__)


# ==================== 知识图谱构建模块（ID锚定版，强制GPU） ====================
class KnowledgeGraphBuilder:
    def __init__(self, edges_file, nodes_file=None, string_file=None,
                 local_model_path=None, use_local_model=False,
                 config=None, gene_normalizer=None, device_manager=None):
        self.edges_file = edges_file
        self.nodes_file = nodes_file
        self.string_file = string_file
        self.local_model_path = local_model_path
        self.use_local_model = use_local_model
        self.config = config if config else Config()
        self.gene_normalizer = gene_normalizer or GeneNameNormalizer(config)

        # 设备管理器
        self.device_manager = device_manager or DeviceManager()

        # 添加ID映射器
        self.id_mapper = GeneIDMapper(config)

        # 基因类型识别
        self.gene_types = config.KG_GENE_TYPES if hasattr(config, 'KG_GENE_TYPES') else ['Gene', 'gene']
        self.filter_only_gene_edges = config.KG_FILTER_ONLY_GENE_EDGES if hasattr(config,
                                                                                  'KG_FILTER_ONLY_GENE_EDGES') else True
        self.use_id_for_mapping = config.KG_USE_ID_FOR_MAPPING if hasattr(config, 'KG_USE_ID_FOR_MAPPING') else True

        # 预训练模型（延迟加载）
        self.embedding_model = None
        self.tokenizer = None
        self.model = None
        if self.use_local_model and self.local_model_path:
            self._init_embedding_model()


    def _init_embedding_model(self):
        """初始化本地预训练模型"""
        try:
            from transformers import AutoTokenizer, AutoModel

            if os.path.exists(self.local_model_path):
                logger.info(f"从本地加载预训练模型: {self.local_model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path)
                self.model = AutoModel.from_pretrained(self.local_model_path)

                # 冻结模型参数
                for param in self.model.parameters():
                    param.requires_grad = False

                # 移动模型到目标设备
                self.model = self.device_manager.move_model(self.model)

                self.embedding_model = True
                logger.info("本地预训练模型加载并移动到目标设备")
            else:
                logger.warning(f"本地模型路径不存在: {self.local_model_path}")
                self.embedding_model = None

        except ImportError:
            logger.warning("Transformers库未安装，无法使用预训练模型")
            self.embedding_model = None
        except Exception as e:
            logger.error(f"加载本地模型失败: {e}")
            self.embedding_model = None

    def get_text_embedding(self, text, max_length=128):
        """获取文本的嵌入表示（使用本地模型）"""
        if not self.embedding_model or not hasattr(self, 'model'):
            # 如果模型不可用，返回零向量
            return np.zeros(768)

        try:
            if not text or pd.isna(text):
                text = ""

            text = str(text)[:500]

            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=max_length,
                padding='max_length'
            )

            # 移动输入到模型设备
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # 获取嵌入
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 使用[CLS] token的嵌入
                embedding = outputs.last_hidden_state[:, 0, :]

            return embedding.squeeze().cpu().numpy()

        except Exception as e:
            logger.error(f"获取文本嵌入失败: {e}")
            return np.zeros(768)

    def batch_encode_texts(self, texts, batch_size=32):
        """批量编码文本"""
        if not self.embedding_model:
            # 如果模型不可用，返回零矩阵
            return np.zeros((len(texts), 768))

        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = []

            for text in batch_texts:
                embedding = self.get_text_embedding(text)
                batch_embeddings.append(embedding)

            embeddings.extend(batch_embeddings)

        return np.array(embeddings)

    def is_gene_type(self, type_str):
        """判断是否为基因类型"""
        if not isinstance(type_str, str):
            return False
        type_str = type_str.strip()
        return any(gene_type.lower() == type_str.lower() for gene_type in self.gene_types)

    # =============== 性能优化核心方法 ===============

    def _batch_convert_names_to_ids(self, names):
        """批量转换名称到ID（优化性能）"""
        if not names or len(names) == 0:
            return {}

        name_to_id = {}
        # 使用集合去重
        unique_names = set(names)

        logger.info(f"批量转换 {len(unique_names):,} 个唯一名称到ID...")

        # 使用进度条
        if len(unique_names) > 10000:
            from tqdm import tqdm
            unique_names_list = list(unique_names)
            for name in tqdm(unique_names_list, desc="转换基因名称"):
                if name:
                    gene_id = self.id_mapper.get_id_by_any_name(name)
                    if gene_id:
                        name_to_id[name] = gene_id
        else:
            for name in unique_names:
                if name:
                    gene_id = self.id_mapper.get_id_by_any_name(name)
                    if gene_id:
                        name_to_id[name] = gene_id

        logger.info(f"成功转换 {len(name_to_id):,} 个名称到ID")
        return name_to_id

    def _process_edges_chunk(self, chunk_df, target_gene_set, name_to_id_cache=None):
        """处理单个数据块（优化性能）"""
        try:
            # 标准化列名 - 修复：确保列名是字符串
            chunk_df.columns = [str(col).strip() for col in chunk_df.columns]

            # 检查并处理重复列名
            if len(chunk_df.columns) != len(set(chunk_df.columns)):
                logger.warning(f"数据块中有重复列名: {chunk_df.columns.tolist()}")
                # 重命名重复列
                seen = {}
                new_columns = []
                for col in chunk_df.columns:
                    col_str = str(col).strip()
                    if col_str in seen:
                        seen[col_str] += 1
                        new_name = f"{col_str}_{seen[col_str]}"
                        new_columns.append(new_name)
                    else:
                        seen[col_str] = 0
                        new_columns.append(col_str)
                chunk_df.columns = new_columns

            # 重命名列（简化版）
            col_mapping = {}
            for col in chunk_df.columns:
                col_lower = str(col).lower()
                if 'x_type' in col_lower or col_lower == 'type':
                    col_mapping['x_type'] = col
                elif 'x_name' in col_lower or 'x_start_id' in col_lower:
                    col_mapping['x_name'] = col
                elif 'x_id' in col_lower or col_lower == 'id':
                    col_mapping['x_id'] = col
                elif 'y_type' in col_lower:
                    col_mapping['y_type'] = col
                elif 'y_name' in col_lower or 'y_end_id' in col_lower:
                    col_mapping['y_name'] = col
                elif 'y_id' in col_lower:
                    col_mapping['y_id'] = col
                elif 'relation' in col_lower or 'relationship' in col_lower:
                    # 处理可能的多列关系问题
                    if 'relation' not in col_mapping:
                        col_mapping['relation'] = col
                    else:
                        logger.warning(f"找到多个关系列，使用第一个: {col}")
                elif 'display_relation' in col_lower:
                    if 'relation' not in col_mapping:
                        col_mapping['relation'] = col

            # 重命名列 - 修复：确保不会重复重命名
            for standard_col, original_col in col_mapping.items():
                if original_col in chunk_df.columns and original_col != standard_col:
                    if standard_col not in chunk_df.columns:
                        chunk_df = chunk_df.rename(columns={original_col: standard_col})
                    else:
                        # 如果目标列已存在，合并或重命名
                        logger.warning(f"列 {standard_col} 已存在，重命名 {original_col}")
                        chunk_df = chunk_df.rename(columns={original_col: f"{original_col}_renamed"})

            # 确保必要列存在
            required_cols = ['x_type', 'x_name', 'y_type', 'y_name', 'relation']
            for col in required_cols:
                if col not in chunk_df.columns:
                    chunk_df[col] = ''

            # 标准化类型列
            chunk_df['x_type'] = chunk_df['x_type'].astype(str).str.strip()
            chunk_df['y_type'] = chunk_df['y_type'].astype(str).str.strip()

            # 标准化关系列 - 确保是字符串
            if 'relation' in chunk_df.columns:
                chunk_df['relation'] = chunk_df['relation'].astype(str).str.strip()

            # 标记哪些边包含目标基因
            chunk_df['contains_target_gene'] = False

            # 第一步：通过x_id和y_id直接匹配
            if 'x_id' in chunk_df.columns and 'y_id' in chunk_df.columns:
                chunk_df['x_id_str'] = chunk_df['x_id'].astype(str).str.strip()
                chunk_df['y_id_str'] = chunk_df['y_id'].astype(str).str.strip()

                mask_x_id = chunk_df['x_id_str'].isin(target_gene_set)
                mask_y_id = chunk_df['y_id_str'].isin(target_gene_set)

                chunk_df.loc[mask_x_id | mask_y_id, 'contains_target_gene'] = True

            # 第二步：通过x_name和y_name匹配
            if name_to_id_cache:
                # 处理x_name
                gene_mask_x = chunk_df['x_type'].apply(self.is_gene_type) & ~chunk_df['contains_target_gene']
                if gene_mask_x.any():
                    # 批量映射
                    x_names = chunk_df.loc[gene_mask_x, 'x_name'].astype(str).str.strip()
                    x_mapped = x_names.map(name_to_id_cache)

                    # 更新数据
                    chunk_df.loc[gene_mask_x, 'x_name'] = x_mapped
                    chunk_df.loc[gene_mask_x, 'x_type'] = 'Gene'

                    # 标记目标基因
                    target_mask = x_mapped.isin(target_gene_set)
                    chunk_df.loc[gene_mask_x, 'contains_target_gene'] = target_mask

                # 处理y_name
                gene_mask_y = chunk_df['y_type'].apply(self.is_gene_type) & ~chunk_df['contains_target_gene']
                if gene_mask_y.any():
                    # 批量映射
                    y_names = chunk_df.loc[gene_mask_y, 'y_name'].astype(str).str.strip()
                    y_mapped = y_names.map(name_to_id_cache)

                    # 更新数据
                    chunk_df.loc[gene_mask_y, 'y_name'] = y_mapped
                    chunk_df.loc[gene_mask_y, 'y_type'] = 'Gene'

                    # 标记目标基因
                    target_mask = y_mapped.isin(target_gene_set)
                    chunk_df.loc[gene_mask_y, 'contains_target_gene'] = target_mask

            # 筛选包含目标基因的边
            filtered_chunk = chunk_df[chunk_df['contains_target_gene']].copy()

            # 过滤掉两端都不是基因的边
            if self.filter_only_gene_edges:
                filtered_chunk = filtered_chunk[
                    (filtered_chunk['x_type'] == 'Gene') | (filtered_chunk['y_type'] == 'Gene')
                    ]

            return filtered_chunk

        except Exception as e:
            logger.error(f"处理数据块时出错: {e}")
            return pd.DataFrame()

    def load_and_filter_edges_by_gene_ids_optimized(self, target_gene_ids):
        """根据目标基因ID列表加载和筛选边数据（性能优化版）"""
        logger.info(f"根据目标基因ID列表筛选知识图谱边，目标基因数: {len(target_gene_ids)}")

        # 将基因ID转为字符串集合，方便快速查找
        target_gene_set = set(str(gene_id) for gene_id in target_gene_ids)
        logger.info(f"目标基因ID集合大小: {len(target_gene_set)}")

        try:
            # 获取总行数（用于进度显示）
            try:
                with open(self.edges_file, 'r', encoding='utf-8') as f:
                    total_lines = sum(1 for _ in f) - 1
                logger.info(f"知识图谱文件总行数: {total_lines:,}")
            except:
                total_lines = 0

            # 预构建名称到ID的映射缓存（一次性加载所有可能需要的映射）
            logger.info("预构建名称到ID的映射缓存...")

            # 首先读取前几行来获取所有可能的基因名称
            sample_df = pd.read_csv(self.edges_file, nrows=10000, sep=None, engine='python')

            # 收集所有可能的基因名称
            all_gene_names = set()

            # 检查列名
            for col in sample_df.columns:
                col_lower = col.lower()
                if 'name' in col_lower or 'start_id' in col_lower or 'end_id' in col_lower:
                    unique_names = sample_df[col].dropna().astype(str).str.strip().unique()
                    all_gene_names.update(unique_names)

            logger.info(f"从样本中收集到 {len(all_gene_names):,} 个唯一名称")

            # 批量转换名称到ID
            name_to_id_cache = self._batch_convert_names_to_ids(all_gene_names)

            # 使用chunksize分批读取和处理
            chunksize = self.config.CHUNKSIZE
            filtered_chunks = []
            total_rows = 0
            matched_count = 0

            logger.info(f"开始分块处理，每块大小: {chunksize:,}")

            for chunk_idx, chunk_df in enumerate(
                    pd.read_csv(self.edges_file, sep=None, engine='python', chunksize=chunksize)):
                total_rows += len(chunk_df)

                # 显示进度
                if total_lines > 0:
                    progress = total_rows / total_lines * 100
                    logger.info(f"处理第 {chunk_idx + 1} 块，进度: {progress:.1f}% ({total_rows:,}/{total_lines:,})")

                # 处理当前块
                filtered_chunk = self._process_edges_chunk(chunk_df, target_gene_set, name_to_id_cache)

                if len(filtered_chunk) > 0:
                    filtered_chunks.append(filtered_chunk)
                    matched_count += len(filtered_chunk)
                    logger.info(f"  本块找到 {len(filtered_chunk):,} 条相关边，累计 {matched_count:,} 条")

                # 定期清理内存
                if chunk_idx % 10 == 0:
                    gc.collect()

            # 合并所有筛选后的块
            if filtered_chunks:
                filtered_edges = pd.concat(filtered_chunks, ignore_index=True)
                logger.info(f"所有块合并后包含目标基因的边数: {len(filtered_edges):,}")

                # 打印示例
                if self.config.DEBUG_MODE and len(filtered_edges) > 0:
                    logger.debug("筛选后的边数据示例:")
                    sample_size = min(5, len(filtered_edges))
                    for i in range(sample_size):
                        row = filtered_edges.iloc[i]

                        x_display = f"{row['x_name']}({row['x_type']})"
                        y_display = f"{row['y_name']}({row['y_type']})"

                        if row['x_type'] == 'Gene':
                            symbol = self.id_mapper.get_symbol_by_id(row['x_name'])
                            x_display = f"{symbol}({row['x_name']})"

                        if row['y_type'] == 'Gene':
                            symbol = self.id_mapper.get_symbol_by_id(row['y_name'])
                            y_display = f"{symbol}({row['y_name']})"

                        logger.debug(f"  {x_display} -> {y_display} [{row['relation']}]")

                return filtered_edges
            else:
                logger.warning("没有找到与目标基因相关的边")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"加载和筛选边数据失败: {e}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()

    def load_and_filter_string_by_gene_ids_optimized(self, target_gene_ids):
        """根据目标基因ID列表加载和筛选STRING互作网络（性能优化版）"""
        if not self.string_file or not os.path.exists(self.string_file):
            logger.warning(f"STRING文件不存在: {self.string_file}")
            return None

        # 将基因ID转为字符串集合
        target_gene_set = set(str(gene_id) for gene_id in target_gene_ids)

        try:
            # 获取总行数
            try:
                with open(self.string_file, 'r', encoding='utf-8') as f:
                    total_lines = sum(1 for _ in f) - 1
                logger.info(f"STRING文件总行数: {total_lines:,}")
            except:
                total_lines = 0

            # 使用chunksize分批读取
            chunksize = self.config.CHUNKSIZE
            filtered_chunks = []

            logger.info(f"开始分块处理STRING文件，每块大小: {chunksize:,}")

            for chunk_idx, string_df in enumerate(pd.read_csv(self.string_file, sep='\t', chunksize=chunksize)):
                # 确定基因列
                gene_cols = []
                for col in string_df.columns:
                    col_lower = col.lower()
                    if 'gene' in col_lower or 'protein' in col_lower:
                        gene_cols.append(col)

                if len(gene_cols) < 2:
                    logger.warning(f"无法识别STRING文件的基因列，找到的列: {gene_cols}")
                    continue

                gene1_col, gene2_col = gene_cols[0], gene_cols[1]

                # 批量转换基因名称到ENTREZID
                all_genes = set(string_df[gene1_col].dropna().astype(str).str.strip().unique()) | \
                            set(string_df[gene2_col].dropna().astype(str).str.strip().unique())

                # 批量转换
                name_to_id = {}
                for gene_name in all_genes:
                    gene_id = self.id_mapper.get_id_by_any_name(gene_name)
                    if gene_id:
                        name_to_id[gene_name] = gene_id

                # 应用转换并筛选
                gene1_names = string_df[gene1_col].astype(str).str.strip()
                gene2_names = string_df[gene2_col].astype(str).str.strip()

                # 批量映射
                gene1_mapped = gene1_names.map(name_to_id)
                gene2_mapped = gene2_names.map(name_to_id)

                # 创建掩码：两个基因都成功转换且都在目标基因集合中
                valid_mask = gene1_mapped.notna() & gene2_mapped.notna() & \
                             gene1_mapped.isin(target_gene_set) & gene2_mapped.isin(target_gene_set)

                if valid_mask.any():
                    chunk_filtered = string_df[valid_mask].copy()
                    chunk_filtered[gene1_col] = gene1_mapped[valid_mask]
                    chunk_filtered[gene2_col] = gene2_mapped[valid_mask]
                    filtered_chunks.append(chunk_filtered)

                    logger.info(f"  第 {chunk_idx + 1} 块找到 {len(chunk_filtered):,} 条STRING边")

                # 定期清理内存
                if chunk_idx % 10 == 0:
                    gc.collect()

            if filtered_chunks:
                filtered_string_df = pd.concat(filtered_chunks, ignore_index=True)
                logger.info(f"所有块合并后STRING互作边数: {len(filtered_string_df):,}")

                # 打印示例
                if self.config.DEBUG_MODE and len(filtered_string_df) > 0:
                    logger.debug("STRING互作示例（前5个）:")
                    for i in range(min(5, len(filtered_string_df))):
                        row = filtered_string_df.iloc[i]
                        gene1_id = row[gene1_col]
                        gene2_id = row[gene2_col]
                        symbol1 = self.id_mapper.get_symbol_by_id(gene1_id)
                        symbol2 = self.id_mapper.get_symbol_by_id(gene2_id)
                        logger.debug(f"  {symbol1}({gene1_id}) - {symbol2}({gene2_id})")

                return filtered_string_df
            else:
                logger.warning("没有找到符合条件的STRING互作边")
                return None

        except Exception as e:
            logger.error(f"加载过滤后的STRING数据失败: {e}")
            return None

    def _safe_get_unique_relations(self, edges_df):
        """安全地获取唯一关系类型"""
        try:
            # 确保 edges_df 是 DataFrame
            if not isinstance(edges_df, pd.DataFrame) or len(edges_df) == 0:
                return []

            # 确保 'relation' 列存在
            if 'relation' not in edges_df.columns:
                logger.warning("'relation' 列不存在，检查列名")
                # 查找可能的列名
                relation_cols = []
                for col in edges_df.columns:
                    col_str = str(col).lower()
                    if 'relation' in col_str or 'relationship' in col_str:
                        relation_cols.append(col)

                if relation_cols:
                    # 使用找到的第一个关系列
                    edges_df['relation'] = edges_df[relation_cols[0]]
                    logger.info(f"使用列 '{relation_cols[0]}' 作为关系列")
                else:
                    # 创建默认列
                    edges_df['relation'] = 'unknown'
                    logger.info("创建默认的 'relation' 列")

            # 处理 'relation' 列 - 确保是 Series 而不是 DataFrame
            relation_data = edges_df['relation']

            # 如果返回的是 DataFrame，处理它
            if isinstance(relation_data, pd.DataFrame):
                logger.warning("'relation' 列是 DataFrame，列名: %s", relation_data.columns.tolist())

                # 如果 DataFrame 有多个列，尝试找出正确的列
                if len(relation_data.columns) > 0:
                    # 取第一列作为关系数据
                    edges_df['relation'] = relation_data.iloc[:, 0]
                    logger.info(f"使用第一列 '{relation_data.columns[0]}' 作为关系数据")
                else:
                    edges_df['relation'] = 'unknown'

            # 确保 'relation' 列是字符串类型
            edges_df['relation'] = edges_df['relation'].astype(str).str.strip()

            # 现在可以安全地获取唯一值
            unique_relations = edges_df['relation'].unique()

            # 转换为列表（确保是列表而不是NumPy数组）
            if isinstance(unique_relations, (pd.Series, pd.Index, np.ndarray)):
                unique_relations = unique_relations.tolist()
            elif not isinstance(unique_relations, list):
                unique_relations = list(unique_relations)

            logger.info(f"成功获取 {len(unique_relations)} 种关系类型")

            # 显示一些示例
            if self.config.DEBUG_MODE and len(unique_relations) > 0:
                logger.debug("关系类型示例（前10个）:")
                for i, rel in enumerate(unique_relations[:10]):
                    logger.debug(f"  关系{i}: '{rel}'")

            return unique_relations

        except Exception as e:
            logger.error(f"获取唯一关系类型失败: {e}")
            logger.error(f"edges_df 列名: {edges_df.columns.tolist() if hasattr(edges_df, 'columns') else 'N/A'}")
            logger.error(f"edges_df 形状: {edges_df.shape if hasattr(edges_df, 'shape') else 'N/A'}")
            return ['unknown']

    def debug_edges_structure(self, edges_df, name="边数据"):
        """调试边数据结构"""
        logger.info(f"=== {name} 结构调试 ===")
        logger.info(f"类型: {type(edges_df)}")
        logger.info(f"形状: {edges_df.shape if hasattr(edges_df, 'shape') else 'N/A'}")

        if hasattr(edges_df, 'columns'):
            logger.info(f"列名 ({len(edges_df.columns)}): {edges_df.columns.tolist()}")

            # 检查列名重复
            col_counts = {}
            for col in edges_df.columns:
                col_str = str(col)
                col_counts[col_str] = col_counts.get(col_str, 0) + 1

            duplicate_cols = {col: count for col, count in col_counts.items() if count > 1}
            if duplicate_cols:
                logger.warning(f"重复列名: {duplicate_cols}")

        # 检查 'relation' 列
        if 'relation' in edges_df.columns:
            relation_data = edges_df['relation']
            logger.info(f"'relation' 列类型: {type(relation_data)}")

            if isinstance(relation_data, pd.DataFrame):
                logger.info(f"  这是一个DataFrame，形状: {relation_data.shape}")
                logger.info(f"  列名: {relation_data.columns.tolist()}")

                # 显示前几行
                logger.info("  前5行数据:")
                for i in range(min(5, len(relation_data))):
                    row = relation_data.iloc[i] if hasattr(relation_data, 'iloc') else relation_data[i]
                    logger.info(f"    行{i}: {row}")
        else:
            logger.warning("没有找到 'relation' 列")

        logger.info("=== 调试结束 ===")

    def create_knowledge_graph_data(self, gene_ids):
        """为指定基因ID列表创建知识图谱数据（性能优化版）"""
        logger.info("开始创建知识图谱数据（性能优化版）...")

        # 确保输入的是字符串格式的ID
        gene_list = [str(gene_id) for gene_id in gene_ids]
        logger.info(f"目标基因数量: {len(gene_list)}")

        # 1. 加载并筛选知识图谱边（使用优化版）
        start_time = time.time()
        edges_df = self.load_and_filter_edges_by_gene_ids_optimized(gene_list)
        load_time = time.time() - start_time
        logger.info(f"边数据筛选完成，耗时: {load_time:.2f}秒")

        # 调试：检查边数据结构
        if self.config.DEBUG_MODE:
            self.debug_edges_structure(edges_df, "筛选后的边数据")
            logger.info("=== 详细调试信息 ===")
            logger.info(f"edges_df 列名: {edges_df.columns.tolist()}")
            logger.info(f"edges_df 形状: {edges_df.shape}")
            if len(edges_df) > 0:
                logger.info("边数据示例（前5行）:")
                for i in range(min(5, len(edges_df))):
                    row = edges_df.iloc[i]
                    logger.info(
                        f"  行{i}: x_type={row['x_type']}, x_name={row['x_name']}, y_type={row['y_type']}, y_name={row['y_name']}, relation={row['relation']}")

        if len(edges_df) == 0:
            logger.warning("没有找到与目标基因相关的知识图谱边，将创建只有基因节点的知识图谱")
            # 创建只有基因节点的知识图谱
            data = HeteroData()

            # 创建基因节点特征
            if self.embedding_model:
                logger.info("使用本地模型生成基因嵌入...")
                gene_texts = []
                for gene_id in gene_list:
                    symbol = self.id_mapper.get_symbol_by_id(gene_id)
                    if symbol and symbol != gene_id:
                        gene_texts.append(f"Gene {symbol}")
                    else:
                        gene_texts.append(f"Gene ID {gene_id}")

                gene_features = self.batch_encode_texts(gene_texts)
                # 创建张量时直接放到目标设备
                data['Gene'].x = torch.tensor(gene_features, dtype=torch.float).to(self.device_manager.target_device)
            else:
                # 使用随机初始化特征，直接放到目标设备
                data['Gene'].x = torch.randn(len(gene_list), 768).to(self.device_manager.target_device)

            # 创建基因ID到索引的映射
            gene_to_idx = {gene_id: idx for idx, gene_id in enumerate(gene_list)}
            setattr(data['Gene'], 'node_name_to_idx', gene_to_idx)
            logger.info(f"创建了只有基因节点的知识图谱，基因节点数: {len(gene_list)}")
            return data, gene_to_idx

        # 2. 加载并筛选STRING互作网络（使用优化版）
        start_time = time.time()
        string_df = self.load_and_filter_string_by_gene_ids_optimized(gene_list)
        string_time = time.time() - start_time
        logger.info(f"STRING数据筛选完成，耗时: {string_time:.2f}秒")

        # 3. 创建异质图数据对象
        data = HeteroData()

        # 4. 创建基因节点特征
        logger.info("创建基因节点特征...")
        if self.embedding_model:
            logger.info("使用本地模型生成基因嵌入...")
            gene_texts = []
            for gene_id in gene_list:
                symbol = self.id_mapper.get_symbol_by_id(gene_id)
                if symbol and symbol != gene_id:
                    gene_texts.append(f"Gene {symbol}")
                else:
                    gene_texts.append(f"Gene ID {gene_id}")

            gene_features = self.batch_encode_texts(gene_texts)
            # 创建张量时直接放到目标设备
            data['Gene'].x = torch.tensor(gene_features, dtype=torch.float).to(self.device_manager.target_device)
        else:
            # 使用随机初始化特征，直接放到目标设备
            data['Gene'].x = torch.randn(len(gene_list), 768).to(self.device_manager.target_device)

        # 创建基因ID到索引的映射
        gene_to_idx = {gene_id: idx for idx, gene_id in enumerate(gene_list)}
        setattr(data['Gene'], 'node_name_to_idx', gene_to_idx)
        logger.info(f"基因节点数: {len(gene_list)}")

        # 5. 收集所有与基因相关的非基因节点
        logger.info("收集非基因节点...")
        non_gene_nodes = defaultdict(dict)

        # 收集所有出现在边中的非基因节点
        for _, row in edges_df.iterrows():
            # 处理头节点
            if row['x_type'] != 'Gene':
                node_type = row['x_type']
                node_name = str(row['x_name']).strip()
                if node_name:
                    if node_name not in non_gene_nodes[node_type]:
                        non_gene_nodes[node_type][node_name] = {
                            'name': node_name,
                            'type': node_type
                        }

            # 处理尾节点
            if row['y_type'] != 'Gene':
                node_type = row['y_type']
                node_name = str(row['y_name']).strip()
                if node_name:
                    if node_name not in non_gene_nodes[node_type]:
                        non_gene_nodes[node_type][node_name] = {
                            'name': node_name,
                            'type': node_type
                        }

        logger.info(f"收集到 {len(non_gene_nodes)} 种非基因节点类型")

        # 6. 为非基因节点创建特征（只处理主要类型）
        node_types = list(non_gene_nodes.keys())
        logger.info(f"发现 {len(node_types)} 种非基因节点类型")

        # 只处理前几种主要类型
        main_node_types = node_types[:self.config.MAX_NODE_TYPES]
        logger.info(f"将处理前 {len(main_node_types)} 种主要节点类型")

        for node_type in main_node_types:
            node_dict = non_gene_nodes[node_type]
            node_names = list(node_dict.keys())

            if len(node_names) == 0:
                continue

            logger.info(f"处理 {node_type} 节点: {len(node_names)}个")

            # 限制节点数量
            if len(node_names) > self.config.MAX_NODES_PER_TYPE:
                logger.info(f"限制 {node_type} 节点数量为 {self.config.MAX_NODES_PER_TYPE}")
                node_names = node_names[:self.config.MAX_NODES_PER_TYPE]

            # 创建节点特征
            if self.embedding_model:
                logger.info(f"  使用本地模型生成{node_type}嵌入...")
                node_texts = [f"{node_type}: {name}" for name in node_names]
                node_features = self.batch_encode_texts(node_texts)
                # 创建张量时直接放到目标设备
                data[node_type].x = torch.tensor(node_features, dtype=torch.float).to(self.device_manager.target_device)
            else:
                # 使用随机初始化特征，直接放到目标设备
                data[node_type].x = torch.randn(len(node_names), 768).to(self.device_manager.target_device)

            # 创建节点名称到索引的映射
            node_name_to_idx = {name: idx for idx, name in enumerate(node_names)}
            setattr(data[node_type], 'node_name_to_idx', node_name_to_idx)

        # 7. 构建知识图谱边索引（修复基因符号转换问题）
        logger.info("构建知识图谱边索引（修复基因符号转换问题）...")

        # 获取唯一关系类型
        unique_relations = self._safe_get_unique_relations(edges_df)

        if not unique_relations or len(unique_relations) == 0:
            logger.warning("没有找到任何关系类型")
            unique_relations = ['unknown']

        # 只处理主要关系类型
        main_relations = unique_relations[:self.config.MAX_RELATION_TYPES]
        logger.info(f"将处理前 {len(main_relations)} 种主要关系类型")

        edge_type_stats = {}
        total_edge_count = 0

        # 调试：显示基因节点映射
        if self.config.DEBUG_MODE:
            logger.info("基因节点映射示例:")
            gene_mapping = getattr(data['Gene'], 'node_name_to_idx', {})
            for gene_id, idx in list(gene_mapping.items())[:5]:
                symbol = self.id_mapper.get_symbol_by_id(gene_id)
                logger.info(f"  {gene_id} ({symbol}) -> 索引{idx}")

        for relation in main_relations:
            relation_str = str(relation).strip()
            relation_edges = edges_df[edges_df['relation'].astype(str).str.strip() == relation_str]

            if len(relation_edges) == 0:
                continue

            logger.info(f"处理关系 '{relation}': {len(relation_edges)}条边")

            src_indices = []
            dst_indices = []
            gene_conversion_success = 0
            gene_conversion_failed = 0

            # 处理每条边
            for _, row in relation_edges.iterrows():
                head_type = row['x_type']
                tail_type = row['y_type']
                head_name_raw = str(row['x_name']).strip()
                tail_name_raw = str(row['y_name']).strip()

                # 处理头节点名称
                head_name = head_name_raw
                if head_type == 'Gene':
                    # 检查是否已经是数字ID
                    if head_name_raw.isdigit():
                        head_name = head_name_raw
                    else:
                        # 尝试转换为ENTREZID
                        head_id = self.id_mapper.get_id_by_any_name(head_name_raw)
                        if head_id:
                            head_name = head_id
                            gene_conversion_success += 1
                        else:
                            gene_conversion_failed += 1
                            continue  # 跳过无法转换的边

                # 处理尾节点名称
                tail_name = tail_name_raw
                if tail_type == 'Gene':
                    # 检查是否已经是数字ID
                    if tail_name_raw.isdigit():
                        tail_name = tail_name_raw
                    else:
                        # 尝试转换为ENTREZID
                        tail_id = self.id_mapper.get_id_by_any_name(tail_name_raw)
                        if tail_id:
                            tail_name = tail_id
                            gene_conversion_success += 1
                        else:
                            gene_conversion_failed += 1
                            continue  # 跳过无法转换的边

                # 检查节点类型是否在图中
                if head_type in data.node_types and tail_type in data.node_types:
                    # 获取节点映射
                    head_mapping = getattr(data[head_type], 'node_name_to_idx', {})
                    tail_mapping = getattr(data[tail_type], 'node_name_to_idx', {})

                    head_idx = head_mapping.get(head_name)
                    tail_idx = tail_mapping.get(tail_name)

                    if head_idx is not None and tail_idx is not None:
                        src_indices.append(head_idx)
                        dst_indices.append(tail_idx)

            # 如果有边，添加到图中
            if len(src_indices) > 0:
                # 创建张量时直接放到目标设备
                edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long).to(
                    self.device_manager.target_device)
                edge_key = (head_type, relation, tail_type)
                data[edge_key].edge_index = edge_index

                edge_count = len(src_indices)
                edge_type_stats[relation] = edge_count
                total_edge_count += edge_count

                logger.info(f"成功添加 {edge_count} 条边")
                if gene_conversion_success > 0 or gene_conversion_failed > 0:
                    logger.info(f"    基因转换: 成功 {gene_conversion_success} 个，失败 {gene_conversion_failed} 个")
            else:
                logger.warning(f"关系 '{relation}' 没有边被添加")
                if gene_conversion_failed > 0:
                    logger.warning(f"    基因转换失败: {gene_conversion_failed} 条边")

        # 8. 添加STRING互作网络
        if string_df is not None and len(string_df) > 0:
            logger.info("添加STRING互作网络...")

            # 获取基因节点映射
            gene_mapping = getattr(data['Gene'], 'node_name_to_idx', {})

            # 确定基因列
            gene_cols = []
            for col in string_df.columns:
                col_lower = col.lower()
                if 'gene' in col_lower or 'protein' in col_lower:
                    gene_cols.append(col)

            if len(gene_cols) >= 2:
                gene1_col, gene2_col = gene_cols[0], gene_cols[1]
                string_edges_forward = []
                string_edges_reverse = []

                for _, row in string_df.iterrows():
                    gene1 = str(row[gene1_col]).strip()
                    gene2 = str(row[gene2_col]).strip()

                    if gene1 in gene_mapping and gene2 in gene_mapping:
                        # 添加正向边
                        string_edges_forward.append([
                            gene_mapping[gene1],
                            gene_mapping[gene2]
                        ])
                        # 添加反向边
                        string_edges_reverse.append([
                            gene_mapping[gene2],
                            gene_mapping[gene1]
                        ])

                if string_edges_forward:
                    # 添加正向边，直接放到目标设备
                    edge_index_forward = torch.tensor(string_edges_forward, dtype=torch.long).t().to(
                        self.device_manager.target_device)
                    data['Gene', 'ppi_interaction', 'Gene'].edge_index = edge_index_forward

                    # 添加反向边，直接放到目标设备
                    edge_index_reverse = torch.tensor(string_edges_reverse, dtype=torch.long).t().to(
                        self.device_manager.target_device)
                    data['Gene', 'ppi_reverse', 'Gene'].edge_index = edge_index_reverse

                    edge_type_stats['ppi_interaction'] = len(string_edges_forward)
                    edge_type_stats['ppi_reverse'] = len(string_edges_reverse)
                    total_edge_count += len(string_edges_forward) * 2

                    logger.info(f"已添加STRING边: {len(string_edges_forward)}对（双向）")

        # 9. 打印结果信息
        logger.info(f"\n知识图谱构建完成:")

        # 检查节点类型
        for node_type in data.node_types:
            if hasattr(data[node_type], 'x'):
                logger.info(f"- {node_type}节点: {data[node_type].x.shape[0]}")

        # 打印边类型信息
        logger.info(f"知识图谱边类型: {len(data.edge_types)}种")
        for edge_type in data.edge_types:
            if hasattr(data[edge_type], 'edge_index'):
                num_edges = data[edge_type].edge_index.shape[1] if hasattr(data[edge_type].edge_index, 'shape') else 0
                edge_str = str(edge_type).replace("'", "").replace("(", "").replace(")", "")
                logger.info(f"- {edge_str}: {num_edges}条边")

        logger.info(f"知识图谱总边数: {total_edge_count}")

        # 打印边类型统计
        if edge_type_stats:
            logger.info("\n边类型统计:")
            for rel, count in sorted(edge_type_stats.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {rel}: {count}条边")

        return data, gene_to_idx


class InductiveKnowledgeGraphBuilder:
    """归纳式知识图谱构建器（支持训练集和验证集分开构建）"""

    def __init__(self, base_builder, config=None, device_manager=None):
        self.base_builder = base_builder
        self.config = config if config else Config()
        self.device_manager = device_manager or DeviceManager()
        self.id_mapper = GeneIDMapper(config)

        # 缓存已处理的知识图谱
        self.train_kg_data = None
        self.train_gene_mapping = None
        self.all_genes_kg_data = None  # 包含所有基因的知识图谱（用于新基因嵌入）

    def build_train_kg(self, train_gene_ids):
        """构建训练集知识图谱"""
        logger.info("构建训练集知识图谱...")
        self.train_kg_data, self.train_gene_mapping = self.base_builder.create_knowledge_graph_data(train_gene_ids)
        return self.train_kg_data, self.train_gene_mapping

    def build_all_genes_kg(self, all_gene_ids):
        """构建包含所有基因的知识图谱（用于新基因获取邻居信息）"""
        logger.info("构建包含所有基因的知识图谱（用于归纳学习）...")
        self.all_genes_kg_data, _ = self.base_builder.create_knowledge_graph_data(all_gene_ids)
        return self.all_genes_kg_data

    def get_gene_neighbors(self, gene_id, max_neighbors=10):
        """获取基因在知识图谱中的邻居（用于新基因特征初始化）"""
        if self.all_genes_kg_data is None:
            return []

        neighbors = []

        # 在所有基因的知识图谱中查找邻居
        for edge_type in self.all_genes_kg_data.edge_types:
            head_type, relation, tail_type = edge_type

            # 检查边类型是否涉及基因
            if head_type == 'Gene' and tail_type == 'Gene':
                edge_index = self.all_genes_kg_data[edge_type].edge_index

                # 查找基因的邻居
                if hasattr(self.all_genes_kg_data['Gene'], 'node_name_to_idx'):
                    gene_mapping = self.all_genes_kg_data['Gene'].node_name_to_idx
                    if gene_id in gene_mapping:
                        gene_idx = gene_mapping[gene_id]

                        # 查找邻居索引
                        src_mask = edge_index[0] == gene_idx
                        dst_neighbors = edge_index[1][src_mask].tolist()

                        src_mask = edge_index[1] == gene_idx
                        src_neighbors = edge_index[0][src_mask].tolist()

                        all_neighbors = list(set(dst_neighbors + src_neighbors))

                        # 转换为基因ID
                        idx_to_gene = {idx: gene for gene, idx in gene_mapping.items()}
                        neighbor_ids = [idx_to_gene[idx] for idx in all_neighbors if idx in idx_to_gene]

                        neighbors.extend(neighbor_ids)

        # 去重并限制数量
        unique_neighbors = list(set(neighbors))[:max_neighbors]
        return unique_neighbors


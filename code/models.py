"""集成模型"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import logging

from config import Config
from device_manager import DeviceManager

logger = logging.getLogger(__name__)


# ==================== 集成模型 ====================
class Columbina_Model(nn.Module):
    def __init__(self, config=None, device_manager=None):
        super().__init__()

        if config is None:
            self.config = config = Config()
        else:
            self.config = config

        # 设备管理器
        self.device_manager = device_manager or DeviceManager()

        # 获取设备
        self.device = self.device_manager.target_device

        # 添加对双向PPI边的支持
        self.ppi_relation_names = ['ppi_interaction', 'ppi_reverse', 'string_interaction']

        # SL图GNN
        self.sl_gnn = nn.Sequential(
            nn.Linear(262, self.config.HIDDEN_DIM),
            nn.BatchNorm1d(self.config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(self.config.DROPOUT),
            nn.Linear(self.config.HIDDEN_DIM, self.config.HIDDEN_DIM // 2),
            nn.BatchNorm1d(self.config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(self.config.DROPOUT),
            nn.Linear(self.config.HIDDEN_DIM // 2, self.config.EMBEDDING_DIM)
        )

        # 知识图谱GNN - 传递设备参数
        self.knowledge_gnn = HeteroGNN(
            hidden_channels=self.config.HIDDEN_DIM // 2,
            out_channels=self.config.EMBEDDING_DIM,
            num_heads=4,
            dropout=self.config.DROPOUT,
            device=self.device
        )

        # 多模态Transformer融合
        self.multimodal_fusion = create_baseline_transformer()

        # SL连接性模块
        self.sl_connectivity = nn.Sequential(
            nn.Linear(self.config.EMBEDDING_DIM, self.config.HIDDEN_DIM // 2),
            nn.BatchNorm1d(self.config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(self.config.DROPOUT),
            nn.Linear(self.config.HIDDEN_DIM // 2, 1)
        )

        # 边预测器
        self.edge_predictor = nn.Sequential(
            nn.Linear(self.config.EMBEDDING_DIM * 2 + 1, self.config.HIDDEN_DIM),
            nn.BatchNorm1d(self.config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(self.config.DROPOUT),
            nn.Linear(self.config.HIDDEN_DIM, self.config.HIDDEN_DIM // 2),
            nn.BatchNorm1d(self.config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(self.config.DROPOUT),
            nn.Linear(self.config.HIDDEN_DIM // 2, 1)
        )

        # 初始化知识图谱卷积层
        self._init_knowledge_gnn_layers()

        # 确保整个模型在正确的设备上
        if self.device:
            self.to(self.device)

    def _init_knowledge_gnn_layers(self):
        """初始化知识图谱GNN层（防止在forward中动态添加）"""
        # 预定义可能的节点类型和关系
        self.predefined_node_types = ['Gene', 'gene', 'Drug', 'Disease', 'Pathway', 'Compound', 'BiologicalProcess',
                                      'CellularComponent', 'MolecularFunction']
        self.predefined_relations = ['ppi', 'interacts_with', 'associated_with', 'targets', 'regulates']

        # 初始化节点编码器
        for node_type in self.predefined_node_types:
            # 在forward中动态处理维度
            pass

        # 初始化关系卷积 - 移除hidden_dim参数
        for relation in self.predefined_relations:
            # 只添加通用的关系卷积，具体维度在运行时确定
            self.knowledge_gnn.add_relation_conv('Gene', 'Gene', relation)
            self.knowledge_gnn.add_relation_conv('Gene', 'Drug', relation)
            self.knowledge_gnn.add_relation_conv('Drug', 'Gene', relation)
            self.knowledge_gnn.add_relation_conv('Gene', 'Disease', relation)
            self.knowledge_gnn.add_relation_conv('Disease', 'Gene', relation)

    def forward(self, sl_data, kg_data, gene_mapping, mode='train', return_attention=False):
        # 确保模型在正确的设备上
        if self.device:
            self.to(self.device)

        # 获取模型当前设备
        model_device = next(self.parameters()).device

        # SL图GNN嵌入
        # 确保输入在正确的设备上
        if sl_data.x.device != model_device:
            sl_data.x = sl_data.x.to(model_device)

        # 确保边索引在正确的设备上
        if sl_data.edge_index.device != model_device:
            sl_data.edge_index = sl_data.edge_index.to(model_device)

        sl_embeddings = self.sl_gnn(sl_data.x)

        # 知识图谱GNN嵌入
        # 获取x_dict - 确保所有张量在正确设备上
        x_dict = {}
        for node_type in kg_data.node_types:
            if hasattr(kg_data[node_type], 'x') and kg_data[node_type].x is not None:
                # 确保张量在模型设备上
                if kg_data[node_type].x.device != model_device:
                    x_dict[node_type] = kg_data[node_type].x.to(model_device)
                else:
                    x_dict[node_type] = kg_data[node_type].x

        # 构建edge_index字典
        edge_index_dict = {}
        processed_edge_types = set()  # 跟踪已处理的边类型

        for edge_type in kg_data.edge_types:
            if hasattr(kg_data[edge_type], 'edge_index'):
                head_type, relation, tail_type = edge_type

                # 对于PPI关系，特殊处理
                if relation in self.ppi_relation_names:
                    # PPI边是双向的，我们使用相同的卷积处理
                    edge_key = (head_type, 'ppi', tail_type)
                else:
                    edge_key = (head_type, relation, tail_type)

                # 添加边索引到字典，并确保在正确的设备上
                if edge_key not in edge_index_dict:
                    edge_index = kg_data[edge_type].edge_index
                    if edge_index.device != model_device:
                        edge_index = edge_index.to(model_device)
                    edge_index_dict[edge_key] = edge_index
                else:
                    # 如果已经有相同类型的边，合并它们
                    existing_edges = edge_index_dict[edge_key]
                    new_edges = kg_data[edge_type].edge_index
                    if new_edges.device != model_device:
                        new_edges = new_edges.to(model_device)
                    edge_index_dict[edge_key] = torch.cat([existing_edges, new_edges], dim=1)

            # 调用知识图谱GNN，可能返回注意力
        kg_kwargs = {'return_attention': return_attention} if return_attention else {}
        kg_result = self.knowledge_gnn(x_dict, edge_index_dict, **kg_kwargs)
        if return_attention:
            kg_embeddings_dict, kg_attention = kg_result
        else:
            kg_embeddings_dict = kg_result
            kg_attention = None

        # 获取基因节点嵌入
        kg_gene_embeddings = None
        for gene_type in ['Gene', 'gene']:
            if gene_type in kg_embeddings_dict:
                kg_gene_embeddings = kg_embeddings_dict[gene_type]
                break

        if kg_gene_embeddings is None:
            # 如果没有找到基因节点嵌入，创建零向量
            kg_gene_embeddings = torch.zeros(sl_data.num_nodes, self.config.EMBEDDING_DIM,
                                             device=sl_embeddings.device)

        # 特征对齐 - 根据基因名映射
        aligned_kg_embeddings = torch.zeros(sl_data.num_nodes, kg_gene_embeddings.size(1),
                                            device=sl_embeddings.device)

        found_count = 0
        for gene_name, idx in gene_mapping.items():
            if idx < sl_data.num_nodes:
                # 尝试在知识图谱中找到对应基因
                kg_gene_idx = None
                if hasattr(kg_data['Gene'], 'node_name_to_idx'):
                    kg_gene_idx = kg_data['Gene'].node_name_to_idx.get(gene_name)

                if kg_gene_idx is not None and kg_gene_idx < kg_gene_embeddings.size(0):
                    aligned_kg_embeddings[idx] = kg_gene_embeddings[kg_gene_idx]
                    found_count += 1

        if mode == 'train' and found_count > 0:
            logger.info(f"特征对齐: 找到 {found_count}/{sl_data.num_nodes} 个基因的匹配")

        if mode == 'embedding':
            if return_attention:
                return (sl_embeddings, aligned_kg_embeddings), {'kg_attention': kg_attention,
                                                                'transformer_attention': None}
            return sl_embeddings, aligned_kg_embeddings

        # 多模态融合
        # 确保输入在正确的设备上
        if aligned_kg_embeddings.device != model_device:
            aligned_kg_embeddings = aligned_kg_embeddings.to(model_device)

        fusion_kwargs = {'return_attention': return_attention} if return_attention else {}
        fusion_result = self.multimodal_fusion(
            sl_embeddings,
            aligned_kg_embeddings,
            sl_data.x,
            **fusion_kwargs
        )
        if return_attention:
            fused_embeddings, transformer_attn = fusion_result
        else:
            fused_embeddings = fusion_result
            transformer_attn = None

        # 计算SL连接性
        sl_connectivity = self.sl_connectivity(fused_embeddings).squeeze()

        if mode == 'embedding_only':
            if return_attention:
                return (fused_embeddings, sl_connectivity), {
                    'kg_attention': kg_attention,
                    'transformer_attention': transformer_attn
                }
            return fused_embeddings, sl_connectivity

        # 边预测
        src_emb = fused_embeddings[sl_data.edge_index[0]]
        dst_emb = fused_embeddings[sl_data.edge_index[1]]
        src_connectivity = sl_connectivity[sl_data.edge_index[0]].unsqueeze(1)
        dst_connectivity = sl_connectivity[sl_data.edge_index[1]].unsqueeze(1)

        # 结合嵌入和连接性特征
        edge_features = torch.cat([src_emb, dst_emb, src_connectivity + dst_connectivity], dim=1)

        # 添加数值稳定性检查
        if self.config.CHECK_NAN_INF and (torch.isnan(edge_features).any() or torch.isinf(edge_features).any()):
            logger.warning("检测到edge_features中的NaN或Inf值，进行清理")
            edge_features = torch.nan_to_num(edge_features, nan=0.0, posinf=1.0, neginf=-1.0)

        edge_pred_logits = self.edge_predictor(edge_features)

        # 添加数值稳定性检查
        if self.config.CHECK_NAN_INF and (torch.isnan(edge_pred_logits).any() or torch.isinf(edge_pred_logits).any()):
            logger.warning("检测到edge_pred_logits中的NaN或Inf值，进行清理")
            edge_pred_logits = torch.nan_to_num(edge_pred_logits, nan=0.0, posinf=1.0, neginf=-1.0)

        edge_pred = torch.sigmoid(edge_pred_logits).squeeze()

        if return_attention:
            return (fused_embeddings, edge_pred, sl_connectivity), {
                'kg_attention': kg_attention,
                'transformer_attention': transformer_attn
            }
        return fused_embeddings, edge_pred, sl_connectivity


class Inductive_Columbina_Model(Columbina_Model):
    """支持归纳式学习的集成模型"""

    def __init__(self, config=None, device_manager=None):
        super().__init__(config, device_manager)

        # 替换为归纳式GNN
        self.sl_gnn = InductiveGNN(
            input_dim=262,
            hidden_dim=self.config.HIDDEN_DIM,
            output_dim=self.config.EMBEDDING_DIM,
            num_layers=3,
            dropout=self.config.DROPOUT
        )

        # 新基因特征增强器
        self.new_gene_enhancer = nn.Sequential(
            nn.Linear(self.config.EMBEDDING_DIM * 2, self.config.EMBEDDING_DIM),
            nn.ReLU(),
            nn.Dropout(self.config.DROPOUT),
            nn.Linear(self.config.EMBEDDING_DIM, self.config.EMBEDDING_DIM)
        )

        # 训练集基因记忆
        self.train_gene_embeddings = None
        self.train_gene_ids_set = None
        self.train_gene_indices = None

    def memorize_train_genes(self, sl_embeddings, gene_ids, all_gene_ids):
        """记忆训练集基因的嵌入和索引"""
        # 存储训练集基因的ID集合
        self.train_gene_ids_set = set(gene_ids)

        # 创建映射：基因ID -> 在图中的索引
        gene_id_to_index = {gene_id: idx for idx, gene_id in enumerate(all_gene_ids)}

        # 获取训练集基因在图中的索引
        train_indices = []
        for gene_id in gene_ids:
            if gene_id in gene_id_to_index:
                train_indices.append(gene_id_to_index[gene_id])

        if train_indices:
            self.train_gene_indices = torch.tensor(train_indices, device=self.device)
            # 存储训练集基因的嵌入
            self.train_gene_embeddings = sl_embeddings[self.train_gene_indices].detach()
            logger.info(f"记忆了 {len(train_indices)} 个训练集基因的嵌入")
        else:
            self.train_gene_indices = None
            self.train_gene_embeddings = None

    def detect_new_genes(self, gene_ids):
        """检测新基因"""
        if self.train_gene_ids_set is None:
            return [False] * len(gene_ids)

        # 创建掩码：基因不在训练集中
        is_new = [gene_id not in self.train_gene_ids_set for gene_id in gene_ids]
        return is_new

    def get_neighbor_features(self, gene_ids, kg_data):
        """从知识图谱获取基因的邻居特征"""
        if not hasattr(kg_data['Gene'], 'node_name_to_idx'):
            return None

        gene_mapping = kg_data['Gene'].node_name_to_idx

        # 获取基因嵌入
        if 'Gene' in kg_data and hasattr(kg_data['Gene'], 'x'):
            gene_features = kg_data['Gene'].x
        else:
            return None

        neighbor_features_list = []

        for gene_id in gene_ids:
            if gene_id in gene_mapping:
                gene_idx = gene_mapping[gene_id]

                # 在知识图谱中查找邻居
                neighbor_indices = []
                for edge_type in kg_data.edge_types:
                    head_type, relation, tail_type = edge_type

                    if head_type == 'Gene' and tail_type == 'Gene':
                        edge_index = kg_data[edge_type].edge_index

                        # 查找邻居
                        src_mask = edge_index[0] == gene_idx
                        dst_neighbors = edge_index[1][src_mask]

                        src_mask = edge_index[1] == gene_idx
                        src_neighbors = edge_index[0][src_mask]

                        if src_mask.any() or dst_neighbors.shape[0] > 0:
                            all_neighbors = torch.cat([dst_neighbors, src_neighbors]).unique()
                            neighbor_indices.extend(all_neighbors.tolist())

                # 获取邻居特征
                if neighbor_indices:
                    neighbor_indices = list(set(neighbor_indices))[:5]  # 最多5个邻居
                    neighbor_feats = gene_features[neighbor_indices]
                    neighbor_features_list.append(neighbor_feats)
                else:
                    # 没有邻居，使用零向量占位
                    neighbor_features_list.append(torch.zeros(0, gene_features.shape[1], device=gene_features.device))
            else:
                # 基因不在知识图谱中
                neighbor_features_list.append(torch.zeros(0, gene_features.shape[1], device=gene_features.device))

        return neighbor_features_list

    def forward(self, sl_data, kg_data, gene_mapping, mode='train', gene_ids=None, train_gene_ids=None, return_attention=False):
        """前向传播，支持新基因处理"""
        # 0. 获取基因ID列表
        if gene_ids is None:
            # 从gene_mapping获取基因ID
            gene_ids = list(gene_mapping.keys())

        # 1. 检测新基因
        is_new_gene = self.detect_new_genes(gene_ids)
        is_new_gene_tensor = torch.tensor(is_new_gene, device=self.device, dtype=torch.bool)

        # 2. 获取新基因邻居特征
        neighbor_features = None
        if is_new_gene_tensor.any():
            neighbor_features_list = self.get_neighbor_features(gene_ids, kg_data)

            if neighbor_features_list:
                # 转换为张量列表
                max_neighbors = max(feat.shape[0] for feat in neighbor_features_list) if neighbor_features_list else 0
                if max_neighbors > 0:
                    feature_dim = neighbor_features_list[0].shape[1]
                    neighbor_features = torch.zeros(
                        len(gene_ids), max_neighbors, feature_dim,
                        device=self.device
                    )

                    for i, feat in enumerate(neighbor_features_list):
                        if feat.shape[0] > 0:
                            neighbor_features[i, :feat.shape[0]] = feat

        # 3. SL图GNN嵌入（使用归纳式GNN）
        sl_embeddings = self.sl_gnn(
            sl_data.x,
            is_new_gene_mask=is_new_gene_tensor,
            neighbor_features=neighbor_features
        )

        # 4. 记忆训练集基因（如果是训练模式且训练集基因ID已提供）
        if mode == 'train' and self.train_gene_embeddings is None and train_gene_ids is not None:
            self.memorize_train_genes(sl_embeddings, train_gene_ids, gene_ids)

        # 5. 对新基因进行特征增强
        if is_new_gene_tensor.any() and self.train_gene_embeddings is not None:
            new_gene_indices = torch.where(is_new_gene_tensor)[0]

            if new_gene_indices.numel() > 0:
                # 创建增强后的嵌入张量
                enhanced_embeddings = sl_embeddings.clone()

                for idx in new_gene_indices:
                    # 获取最相似的训练集基因特征
                    new_gene_feat = sl_embeddings[idx].unsqueeze(0)

                    # 计算与训练集基因的相似度
                    similarities = F.cosine_similarity(
                        new_gene_feat,
                        self.train_gene_embeddings
                    )

                    # 获取Top-K相似基因
                    k = min(3, self.train_gene_embeddings.shape[0])
                    topk_values, topk_indices = torch.topk(similarities, k)

                    # 加权聚合相似基因的特征
                    weights = F.softmax(topk_values, dim=0)
                    similar_features = self.train_gene_embeddings[topk_indices]
                    weighted_avg = (similar_features * weights.view(-1, 1)).sum(dim=0)

                    # 结合自身特征和相似基因特征
                    combined = torch.cat([new_gene_feat, weighted_avg.unsqueeze(0)], dim=1)
                    enhanced = self.new_gene_enhancer(combined)

                    # 更新特征
                    enhanced_embeddings[idx] = sl_embeddings[idx] + 0.2 * enhanced.squeeze(0)

                sl_embeddings = enhanced_embeddings

        # 6. 如果只需要嵌入，直接返回
        if mode == 'embedding':
            if return_attention:
                return (sl_embeddings, torch.zeros_like(sl_embeddings)), {'kg_attention': None,
                                                                          'transformer_attention': None}
            return sl_embeddings, torch.zeros_like(sl_embeddings)

        # 7. 多模态融合
        model_device = next(self.parameters()).device
        if sl_embeddings.device != model_device:
            sl_embeddings = sl_embeddings.to(model_device)

        # 创建一个对齐的知识图谱嵌入（占位符）
        aligned_kg_embeddings = torch.zeros_like(sl_embeddings)

        # 调用多模态融合，可能返回注意力
        fusion_kwargs = {'return_attention': return_attention} if return_attention else {}
        fusion_result = self.multimodal_fusion(
            sl_embeddings,
            aligned_kg_embeddings,
            sl_data.x,
            **fusion_kwargs
        )
        if return_attention:
            fused_embeddings, transformer_attn = fusion_result
        else:
            fused_embeddings = fusion_result
            transformer_attn = None

        # 8. 计算SL连接性
        sl_connectivity = self.sl_connectivity(fused_embeddings).squeeze()

        if mode == 'embedding_only':
            if return_attention:
                return (fused_embeddings, sl_connectivity), {
                    'kg_attention': None,  # 归纳模型中未使用真实知识图谱
                    'transformer_attention': transformer_attn
                }
            return fused_embeddings, sl_connectivity

        # 9. 边预测
        src_emb = fused_embeddings[sl_data.edge_index[0]]
        dst_emb = fused_embeddings[sl_data.edge_index[1]]
        src_connectivity = sl_connectivity[sl_data.edge_index[0]].unsqueeze(1)
        dst_connectivity = sl_connectivity[sl_data.edge_index[1]].unsqueeze(1)

        # 结合嵌入和连接性特征
        edge_features = torch.cat([src_emb, dst_emb, src_connectivity + dst_connectivity], dim=1)

        # 添加数值稳定性检查
        if self.config.CHECK_NAN_INF and (torch.isnan(edge_features).any() or torch.isinf(edge_features).any()):
            logger.warning("检测到edge_features中的NaN或Inf值，进行清理")
            edge_features = torch.nan_to_num(edge_features, nan=0.0, posinf=1.0, neginf=-1.0)

        edge_pred_logits = self.edge_predictor(edge_features)

        # 添加数值稳定性检查
        if self.config.CHECK_NAN_INF and (torch.isnan(edge_pred_logits).any() or torch.isinf(edge_pred_logits).any()):
            logger.warning("检测到edge_pred_logits中的NaN或Inf值，进行清理")
            edge_pred_logits = torch.nan_to_num(edge_pred_logits, nan=0.0, posinf=1.0, neginf=-1.0)

        edge_pred = torch.sigmoid(edge_pred_logits).squeeze()

        if return_attention:
            return (fused_embeddings, edge_pred, sl_connectivity), {
                'kg_attention': None,  # 归纳模型中未使用真实知识图谱
                'transformer_attention': transformer_attn
            }
        return fused_embeddings, edge_pred, sl_connectivity


# ==================== 异质图神经网络 ====================
class HeteroGNN(nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads=4, dropout=0.3, device=None):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.dropout = dropout
        self.device = device

        # 节点编码器
        self.node_encoders = nn.ModuleDict()

        # 关系卷积层
        self.conv_layers = nn.ModuleDict()  # 使用ModuleDict存储卷积层
        self.conv_metadata = {}  # 存储卷积层的元数据（头尾类型）

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_channels * num_heads, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
            nn.BatchNorm1d(out_channels)
        )

        # 初始化计数器和映射
        self.conv_counter = 0
        self.edge_type_to_key = {}

        # 初始化时设置设备（如果提供了设备）
        if self.device:
            self.to(self.device)

    def add_node_encoder(self, node_type, in_dim):
        """添加节点编码器 - 统一输出维度"""
        if node_type not in self.node_encoders:
            # 统一输出维度为 hidden_channels
            output_dim = self.hidden_channels

            # 创建编码器
            if in_dim == output_dim:
                # 如果输入维度等于目标维度，使用恒等映射
                node_encoder = nn.Identity()
            else:
                node_encoder = nn.Sequential(
                    nn.Linear(in_dim, output_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.BatchNorm1d(output_dim)
                )

            # 确保编码器在正确的设备上
            if self.device:
                node_encoder = node_encoder.to(self.device)

            self.node_encoders[node_type] = node_encoder
            logger.info(f"创建 {node_type} 编码器: {in_dim} -> {output_dim}")

    def add_relation_conv(self, head_type, tail_type, relation=None, hidden_dim=None):
        """添加关系卷积层"""
        if relation:
            conv_key = f"conv_{head_type}_{relation}_{tail_type}_{self.conv_counter}"
        else:
            conv_key = f"conv_{head_type}_{tail_type}_{self.conv_counter}"
        self.conv_counter += 1

        # 确定输入维度
        if hidden_dim is None:
            hidden_dim = self.hidden_channels

        logger.info(f"创建卷积层 {conv_key}: 输入维度={hidden_dim}, 输出维度={self.hidden_channels}")

        # 创建卷积层
        conv = GATConv(
            hidden_dim, self.hidden_channels,
            heads=self.num_heads,
            dropout=self.dropout,
            concat=False  # 重要：设置为False避免维度爆炸
        )

        # 确保卷积层在正确的设备上
        if self.device:
            conv = conv.to(self.device)

        # 存储卷积层
        self.conv_layers[conv_key] = conv

        # 存储元数据
        self.conv_metadata[conv_key] = {
            'head_type': head_type,
            'tail_type': tail_type,
            'relation': relation
        }

        # 存储边类型到键的映射
        if relation:
            self.edge_type_to_key[(head_type, relation, tail_type)] = conv_key
        else:
            self.edge_type_to_key[(head_type, tail_type)] = conv_key

        return conv_key

    def forward(self, x_dict, edge_index_dict, return_attention=False):
        # 确保所有节点特征在正确的设备上
        for node_type, x in x_dict.items():
            if self.device and x.device != self.device:
                x_dict[node_type] = x.to(self.device)

        # 确保所有边索引在正确的设备上
        for edge_key, edge_index in edge_index_dict.items():
            if self.device and edge_index.device != self.device:
                edge_index_dict[edge_key] = edge_index.to(self.device)

        # 节点编码 - 动态创建编码器
        encoded_dict = {}
        for node_type, x in x_dict.items():
            # 确保节点类型有编码器
            if node_type not in self.node_encoders:
                # 动态创建节点编码器
                in_dim = x.shape[1]
                node_encoder = nn.Sequential(
                    nn.Linear(in_dim, self.hidden_channels),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.BatchNorm1d(self.hidden_channels)
                )

                # 确保编码器在正确的设备上
                if self.device:
                    node_encoder = node_encoder.to(self.device)

                self.node_encoders[node_type] = node_encoder
                logger.info(f"动态创建 {node_type} 编码器: 输入维度={in_dim}, 输出维度={self.hidden_channels}")

            encoder = self.node_encoders[node_type]
            encoder_device = next(encoder.parameters()).device
            if x.device != encoder_device:
                x = x.to(encoder_device)
            encoded_dict[node_type] = encoder(x)

        # 准备存储注意力的字典
        attention_dict = {}

        # 关系传播
        for edge_key, edge_index in edge_index_dict.items():
            if len(edge_key) == 3:
                head_type, relation, tail_type = edge_key
            else:
                head_type, tail_type = edge_key
                relation = None

            # 构建卷积层键
            if relation:
                conv_key = self.edge_type_to_key.get((head_type, relation, tail_type))
            else:
                conv_key = self.edge_type_to_key.get((head_type, tail_type))

            # 如果没有对应的卷积层，动态创建
            if conv_key is None:
                # 获取头节点的特征维度
                if head_type in encoded_dict:
                    head_dim = encoded_dict[head_type].shape[1]
                else:
                    continue  # 跳过无法处理的边

                # 创建新的卷积层
                conv_key = self.add_relation_conv(head_type, tail_type, relation, head_dim)

            # 获取卷积层
            if conv_key in self.conv_layers:
                conv = self.conv_layers[conv_key]

                # 检查节点特征是否存在
                if head_type in encoded_dict and tail_type in encoded_dict:
                    if edge_index.shape[1] > 0:
                        try:
                            # 确保边索引在正确设备上
                            conv_device = next(conv.parameters()).device
                            if edge_index.device != conv_device:
                                edge_index = edge_index.to(conv_device)

                            # 确保节点特征在正确设备上
                            head_features = encoded_dict[head_type]
                            tail_features = encoded_dict[tail_type]
                            if head_features.device != conv_device:
                                head_features = head_features.to(conv_device)
                            if tail_features.device != conv_device:
                                tail_features = tail_features.to(conv_device)

                            # 检查维度匹配
                            if head_features.shape[1] != conv.in_channels:
                                logger.warning(
                                    f"维度不匹配: {head_type}特征维度={head_features.shape[1]}, 卷积层期望={conv.in_channels}")
                                continue

                            # 前向传播，根据 return_attention 决定是否返回注意力
                            if return_attention:
                                out, (attn_edge_index, attn_weights) = conv(
                                    (head_features, tail_features),
                                    edge_index,
                                    return_attention_weights=True
                                )
                            # 存储注意力系数
                                attention_dict[edge_key] = (attn_edge_index, attn_weights)
                            else:
                                out = conv((head_features, tail_features), edge_index)

                            # 残差连接
                            if tail_type in encoded_dict and encoded_dict[tail_type].shape == out.shape:
                                encoded_dict[tail_type] = encoded_dict[tail_type] + out
                            elif tail_type not in encoded_dict:
                                encoded_dict[tail_type] = out
                            else:
                                encoded_dict[tail_type] = out

                        except Exception as e:
                            logger.warning(f"GAT传播失败: {e}")
                            continue

        # 输出投影（只对基因节点）
        gene_types = ['Gene', 'gene']
        for gene_type in gene_types:
            if gene_type in encoded_dict:
                # 确保输出层在正确设备上
                output_layer_device = next(self.output_layer.parameters()).device
                if encoded_dict[gene_type].device != output_layer_device:
                    encoded_dict[gene_type] = encoded_dict[gene_type].to(output_layer_device)

                if encoded_dict[gene_type].shape[1] == self.hidden_channels * self.num_heads:
                    encoded_dict[gene_type] = self.output_layer(encoded_dict[gene_type])
                break

        if return_attention:
            return encoded_dict, attention_dict
        return encoded_dict


class InductiveGNN(nn.Module):
    """归纳式GNN，支持处理新节点（避免原地操作版本）"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3):
        super().__init__()

        self.layers = nn.ModuleList()

        # 输入层
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.BatchNorm1d(hidden_dim))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout))

        # 隐藏层
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))

        # 输出层
        self.layers.append(nn.Linear(hidden_dim, output_dim))

        # 邻居聚合层（用于新基因）
        self.neighbor_aggregator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        # 标记是否已经初始化批量归一化的运行统计
        self.bn_initialized = False

    def forward(self, x, edge_index=None, is_new_gene_mask=None, neighbor_features=None):
        """前向传播，支持新基因处理（避免原地操作）"""
        # 基础特征提取
        x_processed = x
        for layer in self.layers:
            if isinstance(layer, nn.BatchNorm1d):
                # 对批量归一化层，只在训练时使用批量统计
                if self.training:
                    x_processed = layer(x_processed)
                    # 标记批量归一化已初始化
                    if not self.bn_initialized and x_processed.shape[0] > 1:
                        self.bn_initialized = True
                else:
                    # 在评估时，确保批量归一化已初始化
                    if self.bn_initialized:
                        layer.eval()
                        x_processed = layer(x_processed)
                    else:
                        # 如果没有初始化，跳过批量归一化
                        x_processed = x_processed
            else:
                x_processed = layer(x_processed)

        # 如果有新基因且提供了邻居特征，进行邻居聚合
        if is_new_gene_mask is not None and neighbor_features is not None:
            new_gene_indices = torch.where(is_new_gene_mask)[0]

            if new_gene_indices.numel() > 0:
                # 创建更新的张量（避免原地操作）
                x_updated = x_processed.clone()

                for idx in new_gene_indices:
                    # 获取当前基因的特征
                    gene_feat = x_processed[idx].unsqueeze(0)

                    # 获取邻居特征
                    neighbor_feat = neighbor_features[idx]
                    if neighbor_feat.shape[0] > 0:
                        # 平均邻居特征
                        avg_neighbor = neighbor_feat.mean(dim=0, keepdim=True)

                        # 结合自身特征和邻居特征
                        combined = torch.cat([gene_feat, avg_neighbor], dim=1)
                        enhanced = self.neighbor_aggregator(combined)

                        # 更新特征（避免原地操作，使用新张量）
                        x_updated[idx] = x_processed[idx] + 0.3 * enhanced.squeeze(0)

                return x_updated

        return x_processed


# ==================== 多模态Transformer融合模块 ====================
class MultiModalTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=3, dropout=0.2,
                 fusion_method='attention', use_residual=False, use_pre_norm=False,
                 attention_type='scaled_dot', use_multi_scale=False):
        """重构为baseline配置的多模态Transformer"""
        super().__init__()

        self.d_model = d_model
        self.fusion_method = fusion_method
        self.use_residual = use_residual
        self.use_pre_norm = use_pre_norm
        self.use_multi_scale = use_multi_scale

        print(f"创建baseline版Transformer融合模块:")
        print(f"  - 融合方法: {fusion_method}")
        print(f"  - 使用残差连接: {use_residual}")
        print(f"  - 使用Pre-norm: {use_pre_norm}")
        print(f"  - 注意力类型: {attention_type}")
        print(f"  - 使用多尺度: {use_multi_scale}")

        # ============ 1. 特征投影层（改进版） ============
        # SL-GNN特征投影
        self.gnn_proj = nn.Sequential(
            nn.Linear(64, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model)
        )

        # 知识图谱特征投影
        self.knowledge_proj = nn.Sequential(
            nn.Linear(64, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model)
        )

        # 组学特征投影（更深的网络）
        self.omics_proj = nn.Sequential(
            nn.Linear(262, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )

        # ============ 2. 可学习的模态权重 ============
        self.modal_weights = nn.Parameter(torch.ones(3) / 3)

        # ============ 3. Transformer编码器 ============
        if attention_type == 'scaled_dot':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                activation='gelu',
                norm_first=use_pre_norm  # Post-norm (False)
            )
        else:
            # 自定义多头注意力层
            encoder_layer = self._create_custom_encoder_layer(d_model, nhead, dropout)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ============ 4. 多尺度融合（baseline中为False） ============
        if use_multi_scale:
            self.multi_scale_fusion = nn.ModuleList([
                nn.Linear(d_model, d_model) for _ in range(num_layers)
            ])
            self.scale_weights = nn.Parameter(torch.ones(num_layers) / num_layers)

        # ============ 5. 模态交互注意力 ============
        self.modal_interaction = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # ============ 6. 跨模态门控融合（baseline中fusion_method='attention'） ============
        if fusion_method == 'gated':
            self.cross_modal_gate = nn.Sequential(
                nn.Linear(d_model * 3, d_model),
                nn.Sigmoid()
            )

        # ============ 7. 输出层 ============
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model, 64),
            nn.LayerNorm(64)
        )

        # ============ 8. 可学习位置编码 ============
        self.position_encoding = nn.Parameter(torch.zeros(1, 3, d_model))
        nn.init.normal_(self.position_encoding, mean=0.0, std=0.02)

        # ============ 9. 初始化权重 ============
        self._init_weights()

    def _create_custom_encoder_layer(self, d_model, nhead, dropout):
        """创建自定义Transformer编码器层"""

        class CustomTransformerEncoderLayer(nn.Module):
            def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):
                super().__init__()
                self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
                self.linear1 = nn.Linear(d_model, dim_feedforward)
                self.dropout = nn.Dropout(dropout)
                self.linear2 = nn.Linear(dim_feedforward, d_model)
                self.norm1 = nn.LayerNorm(d_model)
                self.norm2 = nn.LayerNorm(d_model)
                self.dropout1 = nn.Dropout(dropout)
                self.dropout2 = nn.Dropout(dropout)
                self.activation = nn.GELU()

            def forward(self, src, src_mask=None, src_key_padding_mask=None):
                # Post-norm架构（与baseline一致）
                src2, _ = self.self_attn(src, src, src,
                                         attn_mask=src_mask,
                                         key_padding_mask=src_key_padding_mask)
                src = src + self.dropout1(src2)
                src = self.norm1(src)

                src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
                src = src + self.dropout2(src2)
                src = self.norm2(src)
                return src

        return CustomTransformerEncoderLayer(d_model, nhead, d_model * 4, dropout)

    def _init_weights(self):
        """初始化权重（Xavier初始化）"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # 模态权重初始化
        nn.init.constant_(self.modal_weights, 1.0 / 3)

        print("权重初始化完成")

    def forward(self, gnn_embeddings, knowledge_embeddings, omics_features, return_attention=False):
        # ============ 1. 特征投影 ============
        gnn_proj = self.gnn_proj(gnn_embeddings).unsqueeze(1)  # [batch, 1, d_model]
        knowledge_proj = self.knowledge_proj(knowledge_embeddings).unsqueeze(1)  # [batch, 1, d_model]
        omics_proj = self.omics_proj(omics_features).unsqueeze(1)  # [batch, 1, d_model]

        # ============ 2. 模态加权 ============
        modal_weight = F.softmax(self.modal_weights, dim=0)
        gnn_proj = gnn_proj * modal_weight[0]
        knowledge_proj = knowledge_proj * modal_weight[1]
        omics_proj = omics_proj * modal_weight[2]

        # ============ 3. 拼接多模态特征 ============
        multimodal_features = torch.cat([gnn_proj, knowledge_proj, omics_proj], dim=1)  # [batch, 3, d_model]

        # ============ 4. 添加可学习位置编码 ============
        multimodal_features = multimodal_features + self.position_encoding

        # ============ 5. Transformer编码 ============
        if self.use_multi_scale:
            # 多尺度融合（baseline中为False）
            layer_outputs = []
            x = multimodal_features
            for i, layer in enumerate(self.transformer_encoder.layers):
                x = layer(x)
                scaled = self.multi_scale_fusion[i](x)
                layer_outputs.append(scaled)

            scale_weights = F.softmax(self.scale_weights, dim=0)
            encoded = torch.stack(layer_outputs, dim=0)
            encoded = (encoded * scale_weights.view(-1, 1, 1, 1)).sum(dim=0)
        else:
            # baseline标准编码
            encoded = self.transformer_encoder(multimodal_features)

        # ============ 6. 模态交互注意力 ============
        attended, attn_weights = self.modal_interaction(encoded, encoded, encoded)

        # ============ 7. 融合策略 ============
        if self.fusion_method == 'gated':
            # 门控融合（baseline中为'attention'）
            gnn_feat = encoded[:, 0:1, :]
            kg_feat = encoded[:, 1:2, :]
            omics_feat = encoded[:, 2:3, :]

            all_features = torch.cat([gnn_feat, kg_feat, omics_feat], dim=2)
            gate_weights = self.cross_modal_gate(all_features)

            fused = (gate_weights * gnn_feat +
                     gate_weights * kg_feat +
                     gate_weights * omics_feat) / 3
            fused = fused.squeeze(1)
        elif self.fusion_method == 'attention':
            # baseline的注意力融合
            modal_attention, _ = self.modal_interaction(
                encoded.mean(dim=1, keepdim=True),  # 查询：全局平均特征
                encoded,  # 键
                encoded  # 值
            )
            fused = modal_attention.squeeze(1)
        else:
            # 平均池化
            fused = attended.mean(dim=1)

        # ============ 8. 残差连接（baseline中为False） ============
        if self.use_residual:
            original_combined = torch.cat([
                gnn_embeddings,
                knowledge_embeddings,
                omics_features
            ], dim=1)
            residual_proj = nn.Linear(original_combined.shape[1], fused.shape[1]).to(fused.device)
            residual = residual_proj(original_combined)
            fused = fused + 0.1 * residual

        # ============ 9. 输出投影 ============
        output = self.output_proj(fused)

        if return_attention:
            return output, attn_weights
        return output


def create_baseline_transformer():
    """创建完全匹配baseline的Transformer配置"""
    return MultiModalTransformer(
        d_model=256,
        nhead=8,
        num_layers=3,
        dropout=0.2,
        fusion_method='attention',  # baseline使用注意力融合
        use_residual=False,  # baseline不使用额外残差
        use_pre_norm=False,  # baseline使用Post-norm
        attention_type='scaled_dot',  # 标准缩放点积注意力
        use_multi_scale=False  # baseline不使用多尺度
    )

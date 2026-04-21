"""训练器模块"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, precision_recall_curve
from tqdm import tqdm
import logging
import traceback

from config import Config
from device_manager import DeviceManager

logger = logging.getLogger(__name__)

# ==================== 6. 训练器（使用设备管理器，支持训练验证划分） ====================
class Trainer:
    def __init__(self, model, train_data, val_data, kg_data, gene_mapping, processor=None, config=None, device_manager=None):
        if config is None:
            self.config = Config()
        else:
            self.config = config

        # 设备管理器
        self.device_manager = device_manager or DeviceManager()

        logger.info(f"训练器使用设备管理器，目标设备: {self.device_manager.target_device}")

        # 移动模型到设备
        self.model = self.device_manager.move_model(model)

        # 移动训练和验证数据到设备
        self.train_data = self.device_manager.move_data(train_data)
        self.val_data = self.device_manager.move_data(val_data)

        # 移动知识图谱数据到设备
        self.kg_data = self.device_manager.move_hetero_data(kg_data)

        self.gene_mapping = gene_mapping

        self.processor = processor

        # 记录训练历史
        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []
        self.val_auprcs = []
        self.learning_rates = []

        # 详细训练历史
        self.train_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'val_auprc': [],
            'learning_rate': []
        }

        # ID映射器（稍后从processor设置）
        self.id_mapper = None
        self.idx_to_gene_id = None

        # 创建输出目录
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)

    def set_id_mapper(self, id_mapper, idx_to_gene_id):
        """设置ID映射器（从processor传递过来）"""
        self.id_mapper = id_mapper
        self.idx_to_gene_id = idx_to_gene_id

    def train(self):
        """训练模型"""
        # 确保所有数据在正确设备上
        self.model = self.device_manager.move_model(self.model)
        self.train_data = self.device_manager.move_data(self.train_data)
        self.val_data = self.device_manager.move_data(self.val_data)
        self.kg_data = self.device_manager.move_hetero_data(self.kg_data)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
            betas=(0.9, 0.999)
        )

        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10
        )

        # 损失函数
        positive_mask = self.train_data.y == 1
        negative_mask = self.train_data.y == 0

        if positive_mask.sum() > 0 and negative_mask.sum() > 0:
            pos_weight = torch.tensor([negative_mask.sum() / positive_mask.sum()]).to(self.device_manager.target_device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()

        # 早停机制初始化
        best_auc = 0
        best_epoch = 0
        patience_counter = 0

        logger.info("开始训练...")

        # 使用tqdm进度条
        pbar = tqdm(range(self.config.EPOCHS), desc="Training")

        for epoch in pbar:
            self.model.train()
            optimizer.zero_grad()

            # 前向传播（训练数据）
            _, edge_pred, _ = self.model(self.train_data, self.kg_data, self.gene_mapping, mode='train')

            # 检查edge_pred中是否有NaN或Inf
            if self.config.CHECK_NAN_INF and (torch.isnan(edge_pred).any() or torch.isinf(edge_pred).any()):
                logger.warning(f"Epoch {epoch}: edge_pred contains NaN or Inf, skipping this epoch")
                self.device_manager.clear_cache()
                continue

            # 计算训练损失
            train_loss = criterion(edge_pred, self.train_data.y)

            if torch.isnan(train_loss) or torch.isinf(train_loss):
                logger.warning(f"Epoch {epoch}: 无效的损失值，跳过")
                self.device_manager.clear_cache()
                continue

            # 反向传播
            train_loss.backward()

            # 梯度裁剪
            if self.config.USE_GRADIENT_CLIPPING:
                if self.config.GRADIENT_CLIP_VALUE > 0:
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), self.config.GRADIENT_CLIP_VALUE)
                if self.config.GRADIENT_CLIP_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP_NORM)

            optimizer.step()

            self.train_losses.append(train_loss.item())
            current_lr = optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)

            # 验证
            if epoch % 10 == 0 or epoch == self.config.EPOCHS - 1:
                try:
                    val_loss, auc, auprc = self.evaluate()
                    self.val_losses.append(val_loss)
                    self.val_aucs.append(auc)
                    self.val_auprcs.append(auprc)

                    # 记录详细历史
                    self.train_history['epoch'].append(epoch)
                    self.train_history['train_loss'].append(train_loss.item())
                    self.train_history['val_loss'].append(val_loss)
                    self.train_history['val_auc'].append(auc)
                    self.train_history['val_auprc'].append(auprc)
                    self.train_history['learning_rate'].append(current_lr)

                    # 更新学习率
                    scheduler.step(auc)

                    # 更新进度条描述
                    pbar.set_description(
                        f"Epoch {epoch}: Train Loss={train_loss.item():.4f}, Val Loss={val_loss:.4f}, AUC={auc:.4f}, AUPRC={auprc:.4f}")

                    # 早停机制
                    if auc > best_auc:
                        best_auc = auc
                        best_epoch = epoch
                        patience_counter = 0

                        # 保存最佳模型
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'train_loss': train_loss.item(),
                            'val_loss': val_loss,
                            'auc': auc,
                            'auprc': auprc,
                            'train_history': self.train_history,
                            'model_config': {
                                'hidden_dim': self.config.HIDDEN_DIM,
                                'embedding_dim': self.config.EMBEDDING_DIM,
                                'node_types': list(self.kg_data.node_types),
                                'edge_types': list(self.kg_data.edge_types),
                            }
                        }, os.path.join(self.config.OUTPUT_DIR, 'best_model.pth'))
                    else:
                        patience_counter += 1

                    if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                        logger.info(f"早停在epoch {epoch}, 最佳AUC: {best_auc:.4f}")
                        break

                except Exception as e:
                    logger.error(f"Epoch {epoch}: 验证失败 - {e}")
                    continue
            else:
                # 非验证epoch，只更新损失到进度条
                pbar.set_description(f"Epoch {epoch}: Train Loss={train_loss.item():.4f}")

            # 定期清理内存
            if epoch % self.config.MEMORY_CLEANUP_FREQUENCY == 0:
                self.device_manager.clear_cache()

            # 定期绘制训练进度
            if epoch % 50 == 0 and epoch > 0 and len(self.val_aucs) > 0:
                try:
                    self.plot_training_progress(
                        os.path.join(self.config.OUTPUT_DIR, f'training_progress_epoch_{epoch}.png')
                    )
                except Exception as e:
                    logger.warning(f"绘制训练进度图失败: {e}")

        # 加载最佳模型
        best_model_path = os.path.join(self.config.OUTPUT_DIR, 'best_model.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.device_manager.target_device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"加载最佳模型 (Epoch {checkpoint['epoch']}), AUC: {checkpoint['auc']:.4f}")

            # 保存注意力权重
            self.model.eval()
            with torch.no_grad():
                # 调用模型，设置 return_attention=True，mode='eval' 确保不改变模型行为
                result = self.model(
                    self.train_data, self.kg_data, self.gene_mapping,
                    mode='eval', return_attention=True
                )

                # 根据返回值解析注意力字典
                # 当 return_attention=True 时，模型应返回 (fused_embeddings, edge_pred, sl_connectivity, attention_dict)
                if isinstance(result, tuple) and len(result) == 2:
                    (fused_embeddings, edge_pred, sl_connectivity), attention_dict = result
                else:
                    logger.warning("模型未返回预期的2个返回值，请检查 return_attention 实现。")
                    attention_dict = None

                if attention_dict is not None:
                    # 将注意力字典中的所有张量移至 CPU 再保存
                    cpu_attention = {}
                    for key, value in attention_dict.items():
                        if isinstance(value, torch.Tensor):
                            cpu_attention[key] = value.cpu()
                        elif isinstance(value, tuple) and len(value) == 2:  # 处理 (edge_index, attn_weights) 元组
                            cpu_attention[key] = (value[0].cpu(), value[1].cpu())
                        else:
                            cpu_attention[key] = value  # 非张量原样保留
                    save_path = os.path.join(self.config.OUTPUT_DIR, 'attention_weights.pt')
                    torch.save(cpu_attention, save_path)
                    logger.info(f"注意力权重已保存至 {save_path}")
                else:
                    logger.warning("未获取到注意力字典，跳过保存。")

        # 最终训练过程可视化
        try:
            self.plot_training_progress(
                os.path.join(self.config.OUTPUT_DIR, 'final_training_progress.png')
            )
        except Exception as e:
            logger.warning(f"绘制最终训练进度图失败: {e}")

        # 保存训练历史
        self.save_training_history()

        # 训练集评估（检查过拟合）
        self.evaluate_on_train()

        # 只在非交叉验证时保存预处理模型
        # 检查输出目录是否包含'fold'，如果包含则说明是交叉验证模式
        output_dir_str = str(self.config.OUTPUT_DIR)
        if 'fold' not in output_dir_str and 'cv' not in output_dir_str.lower():
            # 单次训练模式，保存预处理模型
            if self.processor is not None:
                try:
                    # 验证processor的scaler是否已拟合
                    if hasattr(self.processor.scaler, 'n_features_in_'):
                        logger.info(f"处理器标准化器特征维度: {self.processor.scaler.n_features_in_}")
                        logger.info(f"处理器is_scaler_fitted: {self.processor.is_scaler_fitted}")
                    else:
                        logger.warning("处理器标准化器未正确拟合")

                    # 保存标准化器和PCA
                    success = self.processor.save_scaler_and_pca(self.config.OUTPUT_DIR)
                    if success:
                        logger.info("预处理模型和基因映射已保存，可用于未来推理")
                    else:
                        logger.error("保存预处理模型失败！")

                    # 保存基因映射
                    self.processor.save_gene_mapping(self.config.OUTPUT_DIR)

                except Exception as e:
                    logger.warning(f"保存预处理模型失败: {e}")
                    import traceback
                    logger.warning(traceback.format_exc())
        else:
            logger.info("交叉验证模式，跳过保存预处理模型")

        return self.train_losses, self.val_losses, self.val_aucs, self.val_auprcs


    def evaluate(self, plot_curves=True):
        """评估模型性能（验证集）"""
        self.model.eval()
        with torch.no_grad():
            _, edge_pred, _ = self.model(self.val_data, self.kg_data, self.gene_mapping, mode='eval')

            y_true = self.val_data.y.cpu().numpy()
            y_pred = edge_pred.cpu().numpy()

            try:
                # 计算验证损失
                criterion = nn.BCEWithLogitsLoss()
                val_loss = criterion(torch.tensor(y_pred, dtype=torch.float),
                                     torch.tensor(y_true, dtype=torch.float)).item()

                # 计算评估指标
                auc = roc_auc_score(y_true, y_pred)
                auprc = average_precision_score(y_true, y_pred)

                # 绘制评估曲线
                if plot_curves and len(y_true) > 0 and len(np.unique(y_true)) > 1:
                    try:
                        self.plot_roc_pr_curves(y_true, y_pred,
                                                os.path.join(self.config.OUTPUT_DIR, 'val_roc_pr_curves.png'))
                        self.plot_confusion_matrix(y_true, y_pred,
                                                   save_path=os.path.join(self.config.OUTPUT_DIR,
                                                                          'val_confusion_matrix.png'))
                    except Exception as e:
                        logger.warning(f"绘制评估曲线失败: {e}")

                return val_loss, auc, auprc
            except Exception as e:
                logger.error(f"评估指标计算错误: {e}")
                return 1.0, 0.5, 0.5

    def evaluate_on_train(self):
        """评估模型在训练集上的性能（用于检查过拟合）"""
        self.model.eval()
        with torch.no_grad():
            _, edge_pred, _ = self.model(self.train_data, self.kg_data, self.gene_mapping, mode='eval')

            y_true = self.train_data.y.cpu().numpy()
            y_pred = edge_pred.cpu().numpy()

            try:
                # 计算训练集损失
                criterion = nn.BCEWithLogitsLoss()
                train_loss = criterion(torch.tensor(y_pred, dtype=torch.float),
                                       torch.tensor(y_true, dtype=torch.float)).item()

                # 计算评估指标
                train_auc = roc_auc_score(y_true, y_pred)
                train_auprc = average_precision_score(y_true, y_pred)

                logger.info(f"训练集性能 - Loss: {train_loss:.4f}, AUC: {train_auc:.4f}, AUPRC: {train_auprc:.4f}")

                # 绘制训练集评估曲线
                try:
                    self.plot_roc_pr_curves(y_true, y_pred,
                                            os.path.join(self.config.OUTPUT_DIR, 'train_roc_pr_curves.png'),
                                            title_prefix="Train")
                    self.plot_confusion_matrix(y_true, y_pred,
                                               save_path=os.path.join(self.config.OUTPUT_DIR,
                                                                      'train_confusion_matrix.png'),
                                               title_prefix="Train")
                except Exception as e:
                    logger.warning(f"绘制训练集评估曲线失败: {e}")

                return train_loss, train_auc, train_auprc
            except Exception as e:
                logger.error(f"训练集评估指标计算错误: {e}")
                return 1.0, 0.5, 0.5

    def plot_training_progress(self, save_path=None):
        """绘制训练过程图表（更新为显示训练和验证损失）"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.flatten()

            # 1. 训练损失曲线
            if self.train_losses:
                epochs_train = list(range(len(self.train_losses)))
                axes[0].plot(epochs_train, self.train_losses, 'b-', linewidth=2, alpha=0.7, label='Train Loss')
                axes[0].set_xlabel('Epoch')
                axes[0].set_ylabel('Loss')
                axes[0].set_title('Training Loss')
                axes[0].grid(True, alpha=0.3)
                axes[0].legend()

            # 2. 验证损失曲线
            if self.val_losses and self.train_history['epoch']:
                val_epochs = self.train_history['epoch']
                axes[1].plot(val_epochs, self.val_losses, 'r-', linewidth=2, label='Val Loss')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('Loss')
                axes[1].set_title('Validation Loss')
                axes[1].grid(True, alpha=0.3)
                axes[1].legend()

            # 3. 训练和验证损失对比
            if self.train_losses and self.val_losses and self.train_history['epoch']:
                val_epochs = self.train_history['epoch']
                # 对训练损失进行采样以匹配验证epoch
                train_loss_sampled = [self.train_losses[epoch] for epoch in val_epochs if
                                      epoch < len(self.train_losses)]
                if len(train_loss_sampled) == len(val_epochs):
                    axes[2].plot(val_epochs, train_loss_sampled, 'b-', linewidth=2, label='Train Loss')
                    axes[2].plot(val_epochs, self.val_losses, 'r-', linewidth=2, label='Val Loss')
                    axes[2].set_xlabel('Epoch')
                    axes[2].set_ylabel('Loss')
                    axes[2].set_title('Train vs Validation Loss')
                    axes[2].grid(True, alpha=0.3)
                    axes[2].legend()

            # 4. 验证AUC曲线
            if self.val_aucs and self.train_history['epoch']:
                val_epochs = self.train_history['epoch']
                axes[3].plot(val_epochs, self.val_aucs, 'g-', linewidth=2, label='AUC')
                axes[3].set_xlabel('Epoch')
                axes[3].set_ylabel('AUC Score')
                axes[3].set_title('Validation AUC')
                axes[3].grid(True, alpha=0.3)
                axes[3].set_ylim(0.5, 1.0)
                axes[3].legend()

            # 5. 验证AUPRC曲线
            if self.val_auprcs and self.train_history['epoch']:
                val_epochs = self.train_history['epoch']
                axes[4].plot(val_epochs, self.val_auprcs, 'purple', linewidth=2, label='AUPRC')
                axes[4].set_xlabel('Epoch')
                axes[4].set_ylabel('AUPRC Score')
                axes[4].set_title('Validation AUPRC')
                axes[4].grid(True, alpha=0.3)
                axes[4].set_ylim(0.5, 1.0)
                axes[4].legend()

            # 6. 学习率曲线
            if self.learning_rates:
                epochs_lr = list(range(len(self.learning_rates)))
                axes[5].plot(epochs_lr, self.learning_rates, 'orange', linewidth=2)
                axes[5].set_xlabel('Epoch')
                axes[5].set_ylabel('Learning Rate')
                axes[5].set_title('Learning Rate Schedule')
                axes[5].grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"训练过程图表已保存至: {save_path}")

            plt.close()
        except Exception as e:
            logger.warning(f"绘制训练进度图失败: {e}")

    def plot_confusion_matrix(self, y_true, y_pred, threshold=0.5, save_path=None, title_prefix=""):
        """绘制混淆矩阵"""
        try:
            y_pred_binary = (y_pred > threshold).astype(int)
            cm = confusion_matrix(y_true, y_pred_binary)

            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Predicted Negative', 'Predicted Positive'],
                        yticklabels=['Actual Negative', 'Actual Positive'])
            title = f'Confusion Matrix (Threshold={threshold})'
            if title_prefix:
                title = f"{title_prefix} {title}"
            plt.title(title)
            plt.ylabel('Actual')
            plt.xlabel('Predicted')

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')

            plt.close()
        except Exception as e:
            logger.warning(f"绘制混淆矩阵失败: {e}")

    def plot_roc_pr_curves(self, y_true, y_pred, save_path=None, title_prefix=""):
        """绘制ROC和PR曲线"""
        try:
            from sklearn.metrics import roc_curve, precision_recall_curve

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # ROC曲线
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            auc_score = roc_auc_score(y_true, y_pred)
            ax1.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc_score:.4f}')
            ax1.plot([0, 1], [0, 1], 'r--', linewidth=1)
            ax1.set_xlabel('False Positive Rate')
            ax1.set_ylabel('True Positive Rate')
            title1 = 'ROC Curve'
            if title_prefix:
                title1 = f"{title_prefix} {title1}"
            ax1.set_title(title1)
            ax1.legend(loc='lower right')
            ax1.grid(True, alpha=0.3)

            # PR曲线
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            auprc_score = average_precision_score(y_true, y_pred)
            ax2.plot(recall, precision, 'g-', linewidth=2, label=f'AUPRC = {auprc_score:.4f}')
            ax2.set_xlabel('Recall')
            ax2.set_ylabel('Precision')
            title2 = 'Precision-Recall Curve'
            if title_prefix:
                title2 = f"{title_prefix} {title2}"
            ax2.set_title(title2)
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')

            plt.close()
        except Exception as e:
            logger.warning(f"绘制ROC/PR曲线失败: {e}")

    def save_training_history(self):
        """保存训练历史到CSV文件"""
        history_df = pd.DataFrame(self.train_history)
        history_path = os.path.join(self.config.OUTPUT_DIR, 'training_history.csv')
        history_df.to_csv(history_path, index=False)
        logger.info(f"训练历史已保存至: {history_path}")

        # 打印训练摘要
        logger.info("\n" + "=" * 60)
        logger.info("训练摘要:")
        logger.info(f"总训练轮次: {len(self.train_losses)}")
        if len(self.train_losses) > 0:
            logger.info(f"初始训练损失: {self.train_losses[0]:.4f}")
            logger.info(f"最终训练损失: {self.train_losses[-1]:.4f}")
        if len(self.val_losses) > 0:
            logger.info(f"初始验证损失: {self.val_losses[0]:.4f}")
            logger.info(f"最终验证损失: {self.val_losses[-1]:.4f}")
        if len(self.val_aucs) > 0:
            logger.info(f"最佳验证AUC: {max(self.val_aucs):.4f}")
            logger.info(f"最佳验证AUPRC: {max(self.val_auprcs):.4f}")
        logger.info("=" * 60)

    def save_predictions(self, train_data_df, val_data_df):
        """保存训练集和验证集的预测结果"""
        try:
            self.model.eval()
            with torch.no_grad():
                # 训练集预测
                _, train_pred, _ = self.model(self.train_data, self.kg_data, self.gene_mapping, mode='eval')
                train_pred_np = train_pred.cpu().numpy()

                # 验证集预测
                _, val_pred, _ = self.model(self.val_data, self.kg_data, self.gene_mapping, mode='eval')
                val_pred_np = val_pred.cpu().numpy()

            # 创建训练集预测结果DataFrame
            train_results = train_data_df.copy()
            train_results['prediction'] = train_pred_np
            train_results['set'] = 'train'

            # 创建验证集预测结果DataFrame
            val_results = val_data_df.copy()
            val_results['prediction'] = val_pred_np
            val_results['set'] = 'val'

            # 合并结果
            all_results = pd.concat([train_results, val_results], ignore_index=True)

            # 保存结果
            results_path = os.path.join(self.config.OUTPUT_DIR, 'predictions_with_split.csv')
            all_results.to_csv(results_path, index=False)

            logger.info(f"预测结果已保存至: {results_path}")

            # 分析预测结果
            self.analyze_predictions(all_results)

        except Exception as e:
            logger.error(f"保存预测结果失败: {e}")

    def analyze_predictions(self, results_df):
        """分析预测结果"""
        try:
            # 按数据集分开
            train_results = results_df[results_df['set'] == 'train']
            val_results = results_df[results_df['set'] == 'val']

            # 计算最佳阈值（基于验证集）
            y_true_val = val_results['label'].values
            y_pred_val = val_results['prediction'].values

            # 找到最佳F1分数的阈值
            precisions, recalls, thresholds = precision_recall_curve(y_true_val, y_pred_val)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

            logger.info(f"最佳分类阈值: {best_threshold:.4f}")
            logger.info(f"最佳F1分数: {f1_scores[best_idx]:.4f}")

            # 应用阈值到两个数据集
            for set_name, df in [('训练集', train_results), ('验证集', val_results)]:
                y_true = df['label'].values
                y_pred = df['prediction'].values
                y_pred_binary = (y_pred > best_threshold).astype(int)

                # 计算指标
                accuracy = accuracy_score(y_true, y_pred_binary)
                precision = precision_score(y_true, y_pred_binary, zero_division=0)
                recall = recall_score(y_true, y_pred_binary, zero_division=0)
                f1 = f1_score(y_true, y_pred_binary, zero_division=0)

                logger.info(f"{set_name}指标 (阈值={best_threshold:.4f}):")
                logger.info(f"  准确率: {accuracy:.4f}")
                logger.info(f"  精确率: {precision:.4f}")
                logger.info(f"  召回率: {recall:.4f}")
                logger.info(f"  F1分数: {f1:.4f}")

                # 保存每个数据集的详细预测
                set_results = df.copy()
                set_results['prediction_binary'] = y_pred_binary
                set_results_path = os.path.join(self.config.OUTPUT_DIR, f'{set_name.lower()}_detailed_predictions.csv')
                set_results.to_csv(set_results_path, index=False)

        except Exception as e:
            logger.error(f"分析预测结果失败: {e}")


class InductiveTrainer(Trainer):
    """支持归纳式学习的训练器（修复梯度问题版本）"""

    def __init__(self, model, train_data, val_data, kg_data, gene_mapping,
                 train_gene_ids=None, all_gene_ids=None, processor=None, config=None, device_manager=None):
        super().__init__(model, train_data, val_data, kg_data, gene_mapping, processor, config, device_manager)

        # 存储训练集基因ID集合和所有基因ID列表
        self.train_gene_ids_set = set(train_gene_ids) if train_gene_ids else None
        self.all_gene_ids = all_gene_ids

        # 创建训练集基因掩码
        if self.all_gene_ids is not None and self.train_gene_ids_set is not None:
            self.train_gene_mask = torch.tensor(
                [gene_id in self.train_gene_ids_set for gene_id in self.all_gene_ids],
                device=self.device_manager.target_device,
                dtype=torch.bool
            )
            logger.info(f"训练集基因掩码: {self.train_gene_mask.sum().item()}/{len(self.all_gene_ids)}")
        else:
            self.train_gene_mask = None

        # 存储详细评估指标 - 初始化为空列表
        self.detailed_metrics = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'val_auprc': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'learning_rate': []
        }

        # 存储最佳阈值的指标
        self.best_threshold_metrics = {
            'threshold': 0.5,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }

        # 确保val_losses, val_aucs, val_auprcs被正确初始化
        self.val_losses = []
        self.val_aucs = []
        self.val_auprcs = []


    def train(self):
        """训练模型"""
        # 确保所有数据在正确设备上
        self.model = self.device_manager.move_model(self.model)
        self.train_data = self.device_manager.move_data(self.train_data)
        self.val_data = self.device_manager.move_data(self.val_data)
        self.kg_data = self.device_manager.move_hetero_data(self.kg_data)

        # 启用异常检测以帮助调试
        if self.config.CHECK_NAN_INF:
            torch.autograd.set_detect_anomaly(True)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
            betas=(0.9, 0.999)
        )

        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10
        )

        # 损失函数
        positive_mask = self.train_data.y == 1
        negative_mask = self.train_data.y == 0

        if positive_mask.sum() > 0 and negative_mask.sum() > 0:
            pos_weight = torch.tensor([negative_mask.sum() / positive_mask.sum()]).to(self.device_manager.target_device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()

        # 早停机制
        best_auc = 0
        best_epoch = 0
        patience_counter = 0

        logger.info("开始归纳式学习训练...")

        pbar = tqdm(range(self.config.EPOCHS), desc="Inductive Training")

        for epoch in pbar:
            self.model.train()
            optimizer.zero_grad()

            try:
                # 前向传播（传入训练集基因ID）
                _, edge_pred, _ = self.model(
                    self.train_data, self.kg_data, self.gene_mapping,
                    mode='train',
                    gene_ids=self.all_gene_ids,
                    train_gene_ids=list(self.train_gene_ids_set) if self.train_gene_ids_set else None
                )

                # 检查edge_pred
                if self.config.CHECK_NAN_INF and (torch.isnan(edge_pred).any() or torch.isinf(edge_pred).any()):
                    logger.warning(f"Epoch {epoch}: edge_pred contains NaN or Inf, skipping")
                    self.device_manager.clear_cache()
                    continue

                # 计算训练损失
                train_loss = criterion(edge_pred, self.train_data.y)

                if torch.isnan(train_loss) or torch.isinf(train_loss):
                    logger.warning(f"Epoch {epoch}: 无效的损失值，跳过")
                    self.device_manager.clear_cache()
                    continue

                # 反向传播（添加异常处理）
                try:
                    train_loss.backward()
                except RuntimeError as e:
                    logger.error(f"Epoch {epoch}: 反向传播失败 - {e}")
                    logger.info("尝试重新初始化模型...")
                    self.model = self.device_manager.move_model(
                        type(self.model)(config=self.config, device_manager=self.device_manager))
                    self.device_manager.clear_cache()
                    continue

                # 梯度裁剪
                if self.config.USE_GRADIENT_CLIPPING:
                    if self.config.GRADIENT_CLIP_VALUE > 0:
                        torch.nn.utils.clip_grad_value_(self.model.parameters(), self.config.GRADIENT_CLIP_VALUE)
                    if self.config.GRADIENT_CLIP_NORM > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP_NORM)

                optimizer.step()

                self.train_losses.append(train_loss.item())
                current_lr = optimizer.param_groups[0]['lr']
                self.learning_rates.append(current_lr)

                # 验证（每10个epoch）
                if epoch % 10 == 0 or epoch == self.config.EPOCHS - 1:
                    try:
                        # 修改：evaluate现在返回更多指标
                        val_loss, auc, auprc, accuracy, precision, recall, f1 = self.evaluate()

                        # 确保只记录与epoch对应的验证指标
                        self.val_losses.append(val_loss)
                        self.val_aucs.append(auc)
                        self.val_auprcs.append(auprc)

                        # 记录详细历史 - 现在所有指标都是基于同一个验证点
                        self.train_history['epoch'].append(epoch)
                        self.train_history['train_loss'].append(train_loss.item())
                        self.train_history['val_loss'].append(val_loss)
                        self.train_history['val_auc'].append(auc)
                        self.train_history['val_auprc'].append(auprc)
                        self.train_history['learning_rate'].append(current_lr)

                        # 新增：记录详细指标
                        self.detailed_metrics['epoch'].append(epoch)
                        self.detailed_metrics['train_loss'].append(train_loss.item())
                        self.detailed_metrics['val_loss'].append(val_loss)
                        self.detailed_metrics['val_auc'].append(auc)
                        self.detailed_metrics['val_auprc'].append(auprc)
                        self.detailed_metrics['val_accuracy'].append(accuracy)
                        self.detailed_metrics['val_precision'].append(precision)
                        self.detailed_metrics['val_recall'].append(recall)
                        self.detailed_metrics['val_f1'].append(f1)
                        self.detailed_metrics['learning_rate'].append(current_lr)
                        # 更新学习率
                        scheduler.step(auc)

                        # 更新进度条
                        pbar.set_description(
                            f"Epoch {epoch}: Train Loss={train_loss.item():.4f}, Val Loss={val_loss:.4f}, "
                            f"AUC={auc:.4f}, F1={f1:.4f}"
                        )

                        # 早停机制
                        if auc > best_auc:
                            best_auc = auc
                            best_epoch = epoch
                            patience_counter = 0

                            # 保存最佳模型
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'train_loss': train_loss.item(),
                                'val_loss': val_loss,
                                'auc': auc,
                                'auprc': auprc,
                                'accuracy': accuracy,
                                'precision': precision,
                                'recall': recall,
                                'f1': f1,
                                'train_history': self.train_history,
                                'detailed_metrics': self.detailed_metrics,
                                'train_gene_ids': list(self.train_gene_ids_set) if self.train_gene_ids_set else [],
                                'all_gene_ids': self.all_gene_ids,
                            }, os.path.join(self.config.OUTPUT_DIR, 'best_inductive_model.pth'))
                        else:
                            patience_counter += 1

                        if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                            logger.info(f"早停在epoch {epoch}, 最佳AUC: {best_auc:.4f}")
                            break

                    except Exception as e:
                        logger.error(f"Epoch {epoch}: 验证失败 - {e}")
                        continue
                else:
                    pbar.set_description(f"Epoch {epoch}: Train Loss={train_loss.item():.4f}")

            except Exception as e:
                logger.error(f"Epoch {epoch}: 训练失败 - {e}")
                self.device_manager.clear_cache()
                continue

            # 定期清理内存
            if epoch % self.config.MEMORY_CLEANUP_FREQUENCY == 0:
                self.device_manager.clear_cache()

        # 加载最佳模型
        best_model_path = os.path.join(self.config.OUTPUT_DIR, 'best_inductive_model.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.device_manager.target_device)
            self.model.load_state_dict(checkpoint['model_state_dict'])

            # 恢复训练集基因ID和所有基因ID
            if 'train_gene_ids' in checkpoint:
                self.train_gene_ids_set = set(checkpoint['train_gene_ids'])
                logger.info(f"恢复训练集基因ID集合，大小: {len(self.train_gene_ids_set)}")

            if 'all_gene_ids' in checkpoint:
                self.all_gene_ids = checkpoint['all_gene_ids']
                logger.info(f"恢复所有基因ID列表，大小: {len(self.all_gene_ids)}")

            if 'detailed_metrics' in checkpoint:
                self.detailed_metrics = checkpoint['detailed_metrics']

            logger.info(f"加载最佳归纳模型 (Epoch {checkpoint['epoch']}), AUC: {checkpoint['auc']:.4f}")

            # 获取训练集基因嵌入和连接性分数
            self.model.eval()
            with torch.no_grad():
                fused_embeddings, sl_connectivity = self.model(
                    self.train_data, self.kg_data, self.gene_mapping,
                    mode='embedding_only',
                    gene_ids=self.all_gene_ids,
                    train_gene_ids=list(self.train_gene_ids_set) if self.train_gene_ids_set else None
                )
                torch.save(fused_embeddings.cpu(), os.path.join(self.config.OUTPUT_DIR, 'train_gene_embeddings.pt'))
                torch.save(sl_connectivity.cpu(), os.path.join(self.config.OUTPUT_DIR, 'sl_connectivity.pt'))
                logger.info("训练集基因嵌入和连接性分数已保存")

            # 保存注意力权重
            self.model.eval()
            with torch.no_grad():
                result = self.model(
                    self.train_data, self.kg_data, self.gene_mapping,
                    mode='eval', return_attention=True,
                    gene_ids=self.all_gene_ids,
                    train_gene_ids=list(self.train_gene_ids_set) if self.train_gene_ids_set else None
                )
                # 当 return_attention=True 且 mode='eval' 时，应返回4个值
                if isinstance(result, tuple) and len(result) == 2:
                    (fused_embeddings, edge_pred, sl_connectivity), attention_dict = result
                else:
                    logger.warning("模型未返回预期的4个返回值，请检查 return_attention 实现。")
                    attention_dict = None

                if attention_dict is not None:
                    cpu_attention = {}
                    for key, value in attention_dict.items():
                        if isinstance(value, torch.Tensor):
                            cpu_attention[key] = value.cpu()
                        elif isinstance(value, tuple) and len(value) == 2:
                            cpu_attention[key] = (value[0].cpu(), value[1].cpu())
                        else:
                            cpu_attention[key] = value
                    save_path = os.path.join(self.config.OUTPUT_DIR, 'attention_weights.pt')
                    torch.save(cpu_attention, save_path)
                    logger.info(f"注意力权重已保存至 {save_path}")
                else:
                    logger.warning("未获取到注意力字典，跳过保存。")

        # 关闭异常检测
        if self.config.CHECK_NAN_INF:
            torch.autograd.set_detect_anomaly(False)

        # 绘制训练过程曲线
        self.plot_inductive_training_curves()

        # 保存详细指标
        self.save_detailed_metrics()

        # 只在非交叉验证时保存预处理模型
        output_dir_str = str(self.config.OUTPUT_DIR)
        if 'fold' not in output_dir_str and 'cv' not in output_dir_str.lower():
            # 单次训练模式，保存预处理模型
            if self.processor is not None:
                try:
                    # 保存标准化器和PCA
                    self.processor.save_scaler_and_pca(self.config.OUTPUT_DIR)

                    # 保存基因映射
                    self.processor.save_gene_mapping(self.config.OUTPUT_DIR)

                    logger.info("预处理模型和基因映射已保存")
                except Exception as e:
                    logger.warning(f"保存预处理模型失败: {e}")
        else:
            logger.info("交叉验证模式，跳过保存预处理模型")


        # 最终评估
        final_val_loss, final_val_auc, final_val_auprc, final_accuracy, final_precision, final_recall, final_f1 = self.evaluate()
        logger.info(f"最终验证集性能: AUC={final_val_auc:.4f}, AUPRC={final_val_auprc:.4f}")
        logger.info(
            f"准确率: {final_accuracy:.4f}, 精确率: {final_precision:.4f}, 召回率: {final_recall:.4f}, F1分数: {final_f1:.4f}")

        return self.train_losses, self.val_losses, self.val_aucs, self.val_auprcs

    def evaluate(self, plot_curves=True):
        """评估模型性能（支持新基因），返回详细指标"""
        self.model.eval()

        with torch.no_grad():
            # 传入所有基因ID
            _, edge_pred, _ = self.model(
                self.val_data, self.kg_data, self.gene_mapping,
                mode='eval',
                gene_ids=self.all_gene_ids
            )

            y_true = self.val_data.y.cpu().numpy()
            y_pred = edge_pred.cpu().numpy()

            try:
                # 计算验证损失
                criterion = nn.BCEWithLogitsLoss()
                val_loss = criterion(torch.tensor(y_pred, dtype=torch.float),
                                     torch.tensor(y_true, dtype=torch.float)).item()

                # 计算评估指标
                auc = roc_auc_score(y_true, y_pred)
                auprc = average_precision_score(y_true, y_pred)

                # 新增：寻找最佳阈值
                best_threshold, best_f1 = self.find_best_threshold(y_true, y_pred)
                self.best_threshold_metrics['threshold'] = best_threshold

                # 使用最佳阈值计算分类指标
                y_pred_binary = (y_pred > best_threshold).astype(int)
                accuracy = accuracy_score(y_true, y_pred_binary)
                precision = precision_score(y_true, y_pred_binary, zero_division=0)
                recall = recall_score(y_true, y_pred_binary, zero_division=0)
                f1 = f1_score(y_true, y_pred_binary, zero_division=0)

                self.best_threshold_metrics['accuracy'] = accuracy
                self.best_threshold_metrics['precision'] = precision
                self.best_threshold_metrics['recall'] = recall
                self.best_threshold_metrics['f1'] = f1

                # 分析新基因和已知基因的性能差异
                if self.train_gene_mask is not None and self.all_gene_ids:
                    self._analyze_performance_by_gene_type(y_true, y_pred, self.all_gene_ids)

                # 绘制评估曲线
                if plot_curves and len(y_true) > 0 and len(np.unique(y_true)) > 1:
                    self.plot_inductive_curves(y_true, y_pred, self.all_gene_ids)

                return val_loss, auc, auprc, accuracy, precision, recall, f1
            except Exception as e:
                logger.error(f"评估指标计算错误: {e}")
                return 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5

    def find_best_threshold(self, y_true, y_pred):
        """寻找最佳阈值（基于F1分数）"""
        thresholds = np.linspace(0.0, 1.0, 101)
        best_threshold = 0.5
        best_f1 = 0

        for threshold in thresholds:
            y_pred_binary = (y_pred > threshold).astype(int)
            try:
                f1 = f1_score(y_true, y_pred_binary, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            except:
                continue

        return best_threshold, best_f1

    def plot_inductive_training_curves(self):
        """绘制归纳式学习的训练过程曲线"""
        try:
            fig, axes = plt.subplots(3, 2, figsize=(16, 12))

            # 1. 训练损失曲线
            if self.train_losses:
                epochs_train = list(range(len(self.train_losses)))
                axes[0, 0].plot(epochs_train, self.train_losses, 'b-', linewidth=2, alpha=0.7, label='Train Loss')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].set_title('Training Loss Curve')
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].legend()

            # 2. 验证损失曲线 - 修复：只使用有对应验证指标的epoch
            if self.val_losses and self.detailed_metrics['epoch']:
                # 确保长度匹配
                min_len = min(len(self.val_losses), len(self.detailed_metrics['epoch']))
                val_epochs = self.detailed_metrics['epoch'][:min_len]
                val_losses = self.val_losses[:min_len]

                axes[0, 1].plot(val_epochs, val_losses, 'r-', linewidth=2, label='Val Loss')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Loss')
                axes[0, 1].set_title('Validation Loss Curve')
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].legend()

            # 3. 验证AUC曲线 - 修复：确保长度匹配
            if self.val_aucs and self.detailed_metrics['epoch']:
                min_len = min(len(self.val_aucs), len(self.detailed_metrics['epoch']))
                val_epochs = self.detailed_metrics['epoch'][:min_len]
                val_aucs = self.val_aucs[:min_len]

                axes[1, 0].plot(val_epochs, val_aucs, 'g-', linewidth=2, label='AUC')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('AUC Score')
                axes[1, 0].set_title('Validation AUC Curve')
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].set_ylim(0.0, 1.0)
                axes[1, 0].legend()

            # 4. 验证AUPRC曲线 - 修复：确保长度匹配
            if self.val_auprcs and self.detailed_metrics['epoch']:
                min_len = min(len(self.val_auprcs), len(self.detailed_metrics['epoch']))
                val_epochs = self.detailed_metrics['epoch'][:min_len]
                val_auprcs = self.val_auprcs[:min_len]

                axes[1, 1].plot(val_epochs, val_auprcs, 'purple', linewidth=2, label='AUPRC')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('AUPRC Score')
                axes[1, 1].set_title('Validation AUPRC Curve')
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].set_ylim(0.0, 1.0)
                axes[1, 1].legend()

            # 5. F1分数曲线 - 修复：确保长度匹配
            if self.detailed_metrics['val_f1'] and self.detailed_metrics['epoch']:
                min_len = min(len(self.detailed_metrics['val_f1']), len(self.detailed_metrics['epoch']))
                val_epochs = self.detailed_metrics['epoch'][:min_len]
                val_f1_scores = self.detailed_metrics['val_f1'][:min_len]

                axes[2, 0].plot(val_epochs, val_f1_scores, 'orange', linewidth=2, label='F1 Score')
                axes[2, 0].set_xlabel('Epoch')
                axes[2, 0].set_ylabel('F1 Score')
                axes[2, 0].set_title('Validation F1 Score Curve')
                axes[2, 0].grid(True, alpha=0.3)
                axes[2, 0].set_ylim(0.0, 1.0)
                axes[2, 0].legend()

            # 6. 多指标对比 - 修复：确保长度匹配
            if (self.detailed_metrics['val_auc'] and self.detailed_metrics['val_auprc'] and
                    self.detailed_metrics['val_f1'] and self.detailed_metrics['epoch']):
                # 找到所有数组的最小长度
                min_len = min(
                    len(self.detailed_metrics['val_auc']),
                    len(self.detailed_metrics['val_auprc']),
                    len(self.detailed_metrics['val_f1']),
                    len(self.detailed_metrics['epoch'])
                )

                val_epochs = self.detailed_metrics['epoch'][:min_len]
                val_aucs = self.detailed_metrics['val_auc'][:min_len]
                val_auprcs = self.detailed_metrics['val_auprc'][:min_len]
                val_f1_scores = self.detailed_metrics['val_f1'][:min_len]

                axes[2, 1].plot(val_epochs, val_aucs, 'g-', linewidth=2, label='AUC')
                axes[2, 1].plot(val_epochs, val_auprcs, 'purple', linewidth=2, label='AUPRC')
                axes[2, 1].plot(val_epochs, val_f1_scores, 'orange', linewidth=2, label='F1')
                axes[2, 1].set_xlabel('Epoch')
                axes[2, 1].set_ylabel('Score')
                axes[2, 1].set_title('Validation Metrics Comparison')
                axes[2, 1].grid(True, alpha=0.3)
                axes[2, 1].set_ylim(0.0, 1.0)
                axes[2, 1].legend()

            plt.tight_layout()

            save_path = os.path.join(self.config.OUTPUT_DIR, 'inductive_training_curves.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"归纳学习训练曲线已保存至: {save_path}")

        except Exception as e:
            logger.warning(f"绘制归纳学习训练曲线失败: {e}")
            logger.error(f"详细错误信息: {traceback.format_exc()}")

    def save_detailed_metrics(self):
        """保存详细指标到CSV文件"""
        if self.detailed_metrics['epoch']:
            metrics_df = pd.DataFrame(self.detailed_metrics)
            metrics_path = os.path.join(self.config.OUTPUT_DIR, 'inductive_detailed_metrics.csv')
            metrics_df.to_csv(metrics_path, index=False)
            logger.info(f"详细训练指标已保存至: {metrics_path}")

            # 打印最佳指标
            if len(metrics_df) > 0:
                best_auc_idx = metrics_df['val_auc'].idxmax()
                best_auc_row = metrics_df.iloc[best_auc_idx]

                best_f1_idx = metrics_df['val_f1'].idxmax()
                best_f1_row = metrics_df.iloc[best_f1_idx]

                logger.info("\n" + "=" * 60)
                logger.info("最佳训练指标汇总:")
                logger.info("=" * 60)
                logger.info(f"最佳AUC - Epoch {best_auc_row['epoch']}:")
                logger.info(f"  AUC: {best_auc_row['val_auc']:.4f}")
                logger.info(f"  AUPRC: {best_auc_row['val_auprc']:.4f}")
                logger.info(f"  F1: {best_auc_row['val_f1']:.4f}")
                logger.info(f"  准确率: {best_auc_row['val_accuracy']:.4f}")
                logger.info(f"  精确率: {best_auc_row['val_precision']:.4f}")
                logger.info(f"  召回率: {best_auc_row['val_recall']:.4f}")

                logger.info(f"\n最佳F1 - Epoch {best_f1_row['epoch']}:")
                logger.info(f"  F1: {best_f1_row['val_f1']:.4f}")
                logger.info(f"  AUC: {best_f1_row['val_auc']:.4f}")
                logger.info(f"  AUPRC: {best_f1_row['val_auprc']:.4f}")
                logger.info(f"  准确率: {best_f1_row['val_accuracy']:.4f}")
                logger.info(f"  精确率: {best_f1_row['val_precision']:.4f}")
                logger.info(f"  召回率: {best_f1_row['val_recall']:.4f}")
                logger.info("=" * 60)

                # 保存最佳阈值指标
                threshold_df = pd.DataFrame([self.best_threshold_metrics])
                threshold_path = os.path.join(self.config.OUTPUT_DIR, 'best_threshold_metrics.csv')
                threshold_df.to_csv(threshold_path, index=False)
                logger.info(f"最佳阈值指标已保存至: {threshold_path}")

                logger.info(f"最佳分类阈值: {self.best_threshold_metrics['threshold']:.4f}")
                logger.info(f"基于最佳阈值的性能:")
                logger.info(f"  准确率: {self.best_threshold_metrics['accuracy']:.4f}")
                logger.info(f"  精确率: {self.best_threshold_metrics['precision']:.4f}")
                logger.info(f"  召回率: {self.best_threshold_metrics['recall']:.4f}")
                logger.info(f"  F1分数: {self.best_threshold_metrics['f1']:.4f}")

    def plot_inductive_curves(self, y_true, y_pred, all_gene_ids=None):
        """绘制归纳学习的评估曲线（包含更多指标）"""
        try:
            from sklearn.metrics import roc_curve, precision_recall_curve

            fig, axes = plt.subplots(2, 3, figsize=(18, 10))

            # 1. 整体ROC曲线
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            auc_score = roc_auc_score(y_true, y_pred)
            axes[0, 0].plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc_score:.4f}')
            axes[0, 0].plot([0, 1], [0, 1], 'r--', linewidth=1)
            axes[0, 0].set_xlabel('False Positive Rate')
            axes[0, 0].set_ylabel('True Positive Rate')
            axes[0, 0].set_title('ROC Curve')
            axes[0, 0].legend(loc='lower right')
            axes[0, 0].grid(True, alpha=0.3)

            # 2. 整体PR曲线
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            auprc_score = average_precision_score(y_true, y_pred)
            axes[0, 1].plot(recall, precision, 'g-', linewidth=2, label=f'AUPRC = {auprc_score:.4f}')
            axes[0, 1].set_xlabel('Recall')
            axes[0, 1].set_ylabel('Precision')
            axes[0, 1].set_title('Precision-Recall Curve')
            axes[0, 1].legend(loc='upper right')
            axes[0, 1].grid(True, alpha=0.3)

            # 3. 预测分数分布
            axes[0, 2].hist(y_pred, bins=50, alpha=0.7, color='blue', edgecolor='black')
            axes[0, 2].axvline(x=self.best_threshold_metrics['threshold'], color='red',
                               linestyle='--', label=f'best_threshold={self.best_threshold_metrics["threshold"]:.3f}')
            axes[0, 2].set_xlabel('Prediction Score')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].set_title('Prediction Score Distribution')
            axes[0, 2].grid(True, alpha=0.3)
            axes[0, 2].legend()

            # 4. 不同阈值下的指标变化
            thresholds = np.linspace(0.0, 1.0, 101)
            accuracies = []
            precisions = []
            recalls = []
            f1_scores = []

            for threshold in thresholds:
                y_pred_binary = (y_pred > threshold).astype(int)
                accuracies.append(accuracy_score(y_true, y_pred_binary))
                precisions.append(precision_score(y_true, y_pred_binary, zero_division=0))
                recalls.append(recall_score(y_true, y_pred_binary, zero_division=0))
                f1_scores.append(f1_score(y_true, y_pred_binary, zero_division=0))

            axes[1, 0].plot(thresholds, accuracies, 'b-', label='Accuracy')
            axes[1, 0].plot(thresholds, precisions, 'g-', label='Precision')
            axes[1, 0].plot(thresholds, recalls, 'r-', label='Recall')
            axes[1, 0].plot(thresholds, f1_scores, 'orange', label='F1 Score')
            axes[1, 0].axvline(x=self.best_threshold_metrics['threshold'], color='black',
                               linestyle='--', label='Best Threshold')
            axes[1, 0].set_xlabel('Threshold')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].set_title('Metrics vs Threshold')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
            axes[1, 0].set_ylim(0.0, 1.0)

            # 5. 混淆矩阵
            y_pred_binary = (y_pred > self.best_threshold_metrics['threshold']).astype(int)
            cm = confusion_matrix(y_true, y_pred_binary)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Predicted Negative', 'Predicted Positive'],
                        yticklabels=['Actual Negative', 'Actual Positive'], ax=axes[1, 1])
            axes[1, 1].set_title(f'Confusion Matrix (Threshold={self.best_threshold_metrics["threshold"]:.3f})')
            axes[1, 1].set_ylabel('Actual')
            axes[1, 1].set_xlabel('Predicted')

            # 6. 按基因类型分析（如果可用）
            if self.train_gene_mask is not None:
                new_gene_count = (~self.train_gene_mask).sum().item()
                known_gene_count = len(self.train_gene_mask) - new_gene_count

                labels = ['Known Genes', 'New Genes']
                sizes = [known_gene_count, new_gene_count]
                colors = ['lightblue', 'lightcoral']

                axes[1, 2].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                axes[1, 2].axis('equal')
                axes[1, 2].set_title('Gene Type Distribution')

            plt.tight_layout()

            save_path = os.path.join(self.config.OUTPUT_DIR, 'inductive_evaluation_curves.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"归纳学习评估曲线已保存至: {save_path}")

        except Exception as e:
            logger.warning(f"绘制归纳学习评估曲线失败: {e}")

    def _analyze_performance_by_gene_type(self, y_true, y_pred, all_gene_ids):
        """按基因类型（新基因/已知基因）分析性能"""
        if self.train_gene_mask is None or len(all_gene_ids) != len(self.train_gene_mask):
            return

        is_new_gene_np = ~self.train_gene_mask.cpu().numpy()

        # 计算每条边是否包含新基因
        edge_contains_new_gene = []
        edge_indices = self.val_data.edge_index.cpu().numpy().T

        for src_idx, dst_idx in edge_indices:
            if src_idx < len(is_new_gene_np) and dst_idx < len(is_new_gene_np):
                is_new = (is_new_gene_np[src_idx] or is_new_gene_np[dst_idx])
                edge_contains_new_gene.append(is_new)
            else:
                edge_contains_new_gene.append(False)

        edge_contains_new_gene = np.array(edge_contains_new_gene)

        if edge_contains_new_gene.any():
            # 新基因边的性能
            new_edges_mask = edge_contains_new_gene
            if new_edges_mask.sum() > 0:
                new_edges_y_true = y_true[new_edges_mask]
                new_edges_y_pred = y_pred[new_edges_mask]

                if len(new_edges_y_true) > 0 and len(np.unique(new_edges_y_true)) > 1:
                    try:
                        new_edges_auc = roc_auc_score(new_edges_y_true, new_edges_y_pred)
                        new_edges_auprc = average_precision_score(new_edges_y_true, new_edges_y_pred)

                        logger.info(
                            f"新基因边性能: AUC={new_edges_auc:.4f}, AUPRC={new_edges_auprc:.4f} ({new_edges_mask.sum()}条边)")
                    except Exception as e:
                        logger.warning(f"无法计算新基因边的AUC: {e}")

            # 已知基因边的性能
            known_edges_mask = ~edge_contains_new_gene
            if known_edges_mask.any():
                known_edges_y_true = y_true[known_edges_mask]
                known_edges_y_pred = y_pred[known_edges_mask]

                if len(known_edges_y_true) > 0 and len(np.unique(known_edges_y_true)) > 1:
                    try:
                        known_edges_auc = roc_auc_score(known_edges_y_true, known_edges_y_pred)
                        known_edges_auprc = average_precision_score(known_edges_y_true, known_edges_y_pred)

                        logger.info(
                            f"已知基因边性能: AUC={known_edges_auc:.4f}, AUPRC={known_edges_auprc:.4f} ({known_edges_mask.sum()}条边)")
                    except Exception as e:
                        logger.warning(f"无法计算已知基因边的AUC: {e}")



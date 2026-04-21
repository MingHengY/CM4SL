import torch
import torch.nn as nn
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, \
    f1_score, precision_recall_curve
import os
import logging
import time
import warnings
import json

from utils import CM4SL_main
from config import Config
from data_processor import SLDataProcessor, UniversalDataSplitter, InductiveSLDataProcessor
from device_manager import DeviceManager
from knowledge_graph_builder import KnowledgeGraphBuilder
from models import Columbina_Model, Inductive_Columbina_Model
from train import Trainer, InductiveTrainer

warnings.filterwarnings('ignore')

# ==================== 全局logger定义 ====================
logger = logging.getLogger(__name__)


def universal_main(scenario='C1', config=None, device_choice=None, test_mode=False, enable_cv=False, n_folds=10):
    """
    通用主执行流程，支持C1、C2、C3三种场景

    Args:
        scenario: 'C1', 'C2', 或 'C3'
        config: 配置对象
        device_choice: 设备选择
        test_mode: 测试模式（简化运行）
        enable_cv: 是否启用交叉验证
        n_folds: 交叉验证折数
    """
    logger.info(f"开始执行{scenario}场景...")

    if config is None:
        config = Config()

    # 设置随机种子
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.RANDOM_SEED)

    # 创建设备管理器
    if device_choice is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_choice

    device_manager = DeviceManager(device)
    logger.info(f"使用设备管理器，目标设备: {device_manager.target_device}")

    # 1. 加载和预处理数据（不划分）
    logger.info("加载和预处理数据...")
    processor = SLDataProcessor(config=config, device_manager=device_manager)

    if config.USE_EXTERNAL_SPLIT:
        # 直接加载外部划分好的文件
        logger.info("使用外部划分数据...")
        train_df = pd.read_csv(config.EXTERNAL_TRAIN_FILE)
        val_df = pd.read_csv(config.EXTERNAL_VAL_FILE)
        test_df = pd.read_csv(config.EXTERNAL_TEST_FILE)

        # 确保列名正确（CM4SL 期望 'Gene.A', 'Gene.B', 'label'）
        required_cols = ['Gene.A', 'Gene.B', 'label']
        for col in required_cols:
            if col not in train_df.columns:
                raise ValueError(f"外部训练文件缺少列: {col}")

        # 创建splitter用于后续分析（仅在非交叉验证时使用）
        splitter = UniversalDataSplitter(config, enable_cv=False)  # 添加此行

        # 直接将划分结果作为单次划分返回（不支持交叉验证）
        if enable_cv:
            raise NotImplementedError("外部划分不支持交叉验证模式")
        split_result = (train_df, val_df, test_df)
    else:
        # 加载SL数据
        sl_data = processor.load_sl_data()

        # 加载非SL数据
        non_sl_data = processor.load_non_sl_data(sample_size=len(sl_data))

        # 合并并平衡数据
        balanced_data = processor.combine_and_balance_data(sl_data, non_sl_data)

        # 2. 根据场景划分数据
        logger.info(f"根据{scenario}场景划分数据...")

        # 创建数据划分器
        splitter = UniversalDataSplitter(config, enable_cv=enable_cv, n_folds=n_folds)

        # 获取划分结果
        split_result = splitter.split_data_by_scenario(
            balanced_data,
            scenario=scenario,
            seed=config.RANDOM_SEED
        )


    if enable_cv:
        # 创建交叉验证主目录
        cv_output_dir = os.path.join(str(config.OUTPUT_DIR), f'{scenario}_cv')
        os.makedirs(cv_output_dir, exist_ok=True)

        # 创建详细报告目录
        report_dir = os.path.join(cv_output_dir, 'detailed_reports')
        os.makedirs(report_dir, exist_ok=True)

        logger.info(f"交叉验证输出目录: {cv_output_dir}")

        # 获取所有fold的划分
        fold_splits = split_result
        logger.info(f"生成 {len(fold_splits)} 折交叉验证数据")

        # 初始化存储所有fold结果的列表
        all_fold_results = []
        all_fold_metrics = []

        # 对每个fold执行训练和评估
        for fold_idx, (train_df, val_df, test_df) in enumerate(fold_splits):
            logger.info(f"\n{'=' * 80}")
            logger.info(f"开始第 {fold_idx + 1}/{len(fold_splits)} 折训练")
            logger.info(f"{'=' * 80}")

            # 1. 创建fold专用输出目录
            fold_output_dir = os.path.join(cv_output_dir, f'fold_{fold_idx + 1:02d}')
            os.makedirs(fold_output_dir, exist_ok=True)

            # 临时保存原始输出目录
            original_output_dir = config.OUTPUT_DIR

            # 2. 设置fold专用输出目录
            config.OUTPUT_DIR = fold_output_dir

            # 3. 分析划分
            logger.info(f"Fold {fold_idx + 1} 数据划分分析:")
            logger.info(f"  训练集: {len(train_df)} 条边")
            logger.info(f"  验证集: {len(val_df)} 条边")
            logger.info(f"  测试集: {len(test_df)} 条边")
            logger.info(f"  正样本比例: 训练集={train_df['label'].mean():.3f}, "
                        f"验证集={val_df['label'].mean():.3f}, "
                        f"测试集={test_df['label'].mean():.3f}")

            # 4. 保存每折的数据
            if config.SAVE_FOLD_DATA:
                train_df.to_csv(os.path.join(fold_output_dir, 'train_data.csv'), index=False)
                val_df.to_csv(os.path.join(fold_output_dir, 'val_data.csv'), index=False)
                test_df.to_csv(os.path.join(fold_output_dir, 'test_data.csv'), index=False)
                logger.info(f"Fold {fold_idx + 1} 数据已保存到: {fold_output_dir}")

            # 5. 保存划分分析
            splitter.analyze_split(train_df, val_df, test_df, scenario)

            # 6. 执行训练和评估
            try:
                if scenario == 'C1':
                    trainer, test_metrics = _run_c1_scenario(
                        processor, train_df, val_df, test_df, config, device_manager
                    )
                elif scenario in ['C2', 'C3']:
                    trainer, test_metrics = _run_c2c3_scenario(
                        processor, train_df, val_df, test_df, config, device_manager, scenario
                    )

                # 7. 保存fold结果
                fold_result = {
                    'fold': fold_idx + 1,
                    'train_size': len(train_df),
                    'val_size': len(val_df),
                    'test_size': len(test_df),
                    'train_positive_ratio': train_df['label'].mean(),
                    'val_positive_ratio': val_df['label'].mean(),
                    'test_positive_ratio': test_df['label'].mean(),
                    'test_metrics': test_metrics,
                    'output_dir': fold_output_dir,
                    'best_model_path': os.path.join(fold_output_dir, 'best_model.pth'),
                    'final_model_path': os.path.join(fold_output_dir, 'final_model.pth')
                }

                # 保存fold详细报告
                if config.SAVE_DETAILED_REPORTS:
                    save_fold_detailed_report(fold_result, trainer, config, scenario)

                all_fold_results.append(fold_result)
                all_fold_metrics.append(test_metrics)

                # 8. 恢复原始输出目录
                config.OUTPUT_DIR = original_output_dir

                logger.info(f"Fold {fold_idx + 1} 完成!")

            except Exception as e:
                logger.error(f"Fold {fold_idx + 1} 训练失败: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # 恢复原始输出目录
                config.OUTPUT_DIR = original_output_dir
                continue

        # 9. 生成交叉验证总体报告
        if all_fold_results:
            logger.info(f"\n{'=' * 80}")
            logger.info(f"交叉验证完成! 成功完成 {len(all_fold_results)}/{len(fold_splits)} 折")
            logger.info(f"{'=' * 80}")

            # 生成总体报告
            cv_summary = generate_cv_summary_report(
                all_fold_results, scenario, config, cv_output_dir
            )

            # 生成横向比较图表
            if config.SAVE_COMPARISON_CHARTS:
                generate_cv_comparison_charts(all_fold_results, scenario, cv_output_dir)

            # 保存最佳模型（选择平均性能最好的fold的模型）
            best_fold_info = select_best_fold_model(all_fold_results, cv_output_dir)

            # 生成最终综合报告
            generate_final_comprehensive_report(
                all_fold_results, cv_summary, best_fold_info, scenario, config, cv_output_dir
            )

            return all_fold_results, cv_summary, best_fold_info

        else:
            logger.error("所有fold训练失败!")
            return None, None, None
    else:
        # 单次划分模式：保持原有逻辑
        train_df, val_df, test_df = split_result

        # 分析划分结果
        splitter.analyze_split(train_df, val_df, test_df, scenario)

        # 保存划分数据
        train_path = os.path.join(config.OUTPUT_DIR, f'{scenario}_train_data.csv')
        val_path = os.path.join(config.OUTPUT_DIR, f'{scenario}_val_data.csv')
        test_path = os.path.join(config.OUTPUT_DIR, f'{scenario}_test_data.csv')

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        logger.info(f"划分数据已保存:")
        logger.info(f"  训练集: {train_path}")
        logger.info(f"  验证集: {val_path}")
        logger.info(f"  测试集: {test_path}")

        # 根据场景选择执行流程
        if scenario == 'C1':
            logger.info("执行C1场景：使用transductive流程")
            trainer, test_metrics = _run_c1_scenario(processor, train_df, val_df, test_df, config, device_manager)
        elif scenario in ['C2', 'C3']:
            logger.info(f"执行{scenario}场景：使用inductive流程")
            trainer, test_metrics = _run_c2c3_scenario(processor, train_df, val_df, test_df, config, device_manager,
                                                       scenario)
        else:
            raise ValueError(f"未知的场景: {scenario}")

            # ========== 在单次训练完成后保存预处理模型 ==========
        if not enable_cv:  # 确保只在非交叉验证时保存
            logger.info(f"单次训练完成，保存预处理模型到: {config.OUTPUT_DIR}")
            try:
                # 检查processor的状态
                if hasattr(trainer, 'processor') and trainer.processor is not None:
                    processor = trainer.processor
                    logger.info(f"检查处理器标准化器状态:")
                    logger.info(f"  is_scaler_fitted: {processor.is_scaler_fitted}")
                    logger.info(f"  是否有n_features_in_: {hasattr(processor.scaler, 'n_features_in_')}")
                    logger.info(f"  是否有mean_: {hasattr(processor.scaler, 'mean_')}")
                    if hasattr(processor.scaler, 'n_features_in_'):
                        logger.info(f"  特征维度: {processor.scaler.n_features_in_}")

                    # 保存标准化器和PCA
                    success = processor.save_processing_models(config.OUTPUT_DIR)
                    if success:
                        logger.info("预处理模型保存完成，可用于未来推理")
                    else:
                        logger.warning("预处理模型保存失败，但训练已完成")
                else:
                    logger.warning("训练器中没有processor，无法保存预处理模型")

            except Exception as e:
                logger.error(f"保存预处理模型失败: {e}")

            try:
                model_config = {
                    'HIDDEN_DIM': config.HIDDEN_DIM,
                    'EMBEDDING_DIM': config.EMBEDDING_DIM,
                    'DROPOUT': config.DROPOUT,
                    'EPOCHS': config.EPOCHS,
                    'LEARNING_RATE': config.LEARNING_RATE,
                    'WEIGHT_DECAY': config.WEIGHT_DECAY,
                    'EARLY_STOPPING_PATIENCE': config.EARLY_STOPPING_PATIENCE,
                    'USE_PRETRAINED': config.USE_PRETRAINED,
                    'USE_ID_ANCHORING': config.USE_ID_ANCHORING,
                    # 添加其他需要的参数
                }
                with open(os.path.join(config.OUTPUT_DIR, 'model_config.json'), 'w') as f:
                    json.dump(model_config, f, indent=2)
                logger.info("模型配置已保存至 model_config.json")
            except Exception as e:
                logger.error(f"保存模型配置失败: {e}")

        return trainer, test_metrics


def save_fold_detailed_report(fold_result, trainer, config, scenario):
    """保存每折的详细报告"""
    fold_dir = fold_result['output_dir']
    report_path = os.path.join(fold_dir, 'fold_detailed_report.txt')

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Fold {fold_result['fold']} - 详细训练报告\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. 数据统计:\n")
        f.write(f"   训练集大小: {fold_result['train_size']}\n")
        f.write(f"   验证集大小: {fold_result['val_size']}\n")
        f.write(f"   测试集大小: {fold_result['test_size']}\n")
        f.write(f"   训练集正样本比例: {fold_result['train_positive_ratio']:.3f}\n")
        f.write(f"   验证集正样本比例: {fold_result['val_positive_ratio']:.3f}\n")
        f.write(f"   测试集正样本比例: {fold_result['test_positive_ratio']:.3f}\n\n")

        f.write("2. 测试集性能指标:\n")
        metrics = fold_result['test_metrics']
        f.write(f"   损失: {metrics['loss']:.4f}\n")
        f.write(f"   AUC: {metrics['auc']:.4f}\n")
        f.write(f"   AUPRC: {metrics['auprc']:.4f}\n")
        f.write(f"   准确率: {metrics['accuracy']:.4f}\n")
        f.write(f"   精确率: {metrics['precision']:.4f}\n")
        f.write(f"   召回率: {metrics['recall']:.4f}\n")
        f.write(f"   F1分数: {metrics['f1']:.4f}\n")
        f.write(f"   最佳阈值: {metrics['best_threshold']:.4f}\n\n")

        f.write("3. 训练历史摘要:\n")
        if hasattr(trainer, 'train_history'):
            history = trainer.train_history
            if 'val_auc' in history and history['val_auc']:
                best_auc = max(history['val_auc'])
                best_epoch = history['epoch'][history['val_auc'].index(best_auc)]
                f.write(f"   最佳验证AUC: {best_auc:.4f} (epoch {best_epoch})\n")

            if 'train_loss' in history and history['train_loss']:
                final_train_loss = history['train_loss'][-1]
                final_val_loss = history['val_loss'][-1] if 'val_loss' in history else None
                f.write(f"   最终训练损失: {final_train_loss:.4f}\n")
                if final_val_loss:
                    f.write(f"   最终验证损失: {final_val_loss:.4f}\n")

        f.write("\n4. 文件位置:\n")
        f.write(f"   训练数据: {os.path.join(fold_dir, 'train_data.csv')}\n")
        f.write(f"   验证数据: {os.path.join(fold_dir, 'val_data.csv')}\n")
        f.write(f"   测试数据: {os.path.join(fold_dir, 'test_data.csv')}\n")
        f.write(f"   最佳模型: {fold_result['best_model_path']}\n")
        f.write(f"   训练历史: {os.path.join(fold_dir, 'training_history.csv')}\n")
        f.write(f"   训练曲线: {os.path.join(fold_dir, 'training_curves.png')}\n")

        f.write("\n" + "=" * 80 + "\n")

    logger.info(f"Fold {fold_result['fold']} 详细报告已保存: {report_path}")


def generate_cv_summary_report(all_fold_results, scenario, config, cv_output_dir):
    """生成交叉验证总体报告"""
    summary_path = os.path.join(cv_output_dir, 'cv_summary_report.txt')

    # 计算各项指标的统计信息
    metrics_list = ['auc', 'auprc', 'accuracy', 'precision', 'recall', 'f1']

    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"交叉验证总体报告 - {scenario}场景\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. 交叉验证配置:\n")
        f.write(f"   场景: {scenario}\n")
        f.write(f"   总折数: {len(all_fold_results)}\n")
        f.write(f"   每类样本数: {config.TOTAL_SAMPLES_PER_CLASS}\n")
        f.write(f"   数据划分比例: 训练{config.TRAIN_VAL_TEST_RATIOS[0] * 100}%:"
                f"验证{config.TRAIN_VAL_TEST_RATIOS[1] * 100}%:"
                f"测试{config.TRAIN_VAL_TEST_RATIOS[2] * 100}%\n\n")

        f.write("2. 各折数据统计:\n")
        for fold_result in all_fold_results:
            f.write(f"   Fold {fold_result['fold']}: "
                    f"训练={fold_result['train_size']}, "
                    f"验证={fold_result['val_size']}, "
                    f"测试={fold_result['test_size']}\n")

        f.write("\n3. 各折性能指标:\n")
        headers = ["Fold"] + [m.upper() for m in metrics_list]
        f.write("   " + " | ".join(f"{h:>10}" for h in headers) + "\n")
        f.write("   " + "-" * (len(headers) * 11) + "\n")

        for fold_result in all_fold_results:
            metrics = fold_result['test_metrics']
            row = [f"{fold_result['fold']}"] + [f"{metrics[m]:.4f}" for m in metrics_list]
            f.write("   " + " | ".join(f"{r:>10}" for r in row) + "\n")

        f.write("\n4. 统计摘要（平均值 ± 标准差）:\n")
        for metric in metrics_list:
            values = [r['test_metrics'][metric] for r in all_fold_results]
            mean_val = np.mean(values)
            std_val = np.std(values)
            f.write(f"   {metric.upper()}: {mean_val:.4f} ± {std_val:.4f}\n")

        f.write("\n5. 各折输出目录:\n")
        for fold_result in all_fold_results:
            f.write(f"   Fold {fold_result['fold']}: {fold_result['output_dir']}\n")

        f.write("\n" + "=" * 80 + "\n")

    # 同时保存CSV格式的汇总
    summary_csv_path = os.path.join(cv_output_dir, 'cv_summary_metrics.csv')
    summary_data = []
    for fold_result in all_fold_results:
        row = {'fold': fold_result['fold']}
        for metric in metrics_list:
            row[metric] = fold_result['test_metrics'][metric]
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_csv_path, index=False)

    logger.info(f"交叉验证总体报告已保存: {summary_path}")

    return {
        'metrics_summary': {metric: {
            'mean': np.mean([r['test_metrics'][metric] for r in all_fold_results]),
            'std': np.std([r['test_metrics'][metric] for r in all_fold_results]),
            'min': np.min([r['test_metrics'][metric] for r in all_fold_results]),
            'max': np.max([r['test_metrics'][metric] for r in all_fold_results])
        } for metric in metrics_list}
    }


def generate_cv_comparison_charts(all_fold_results, scenario, cv_output_dir):
    """生成交叉验证比较图表"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_style("whitegrid")

        # 1. 各折AUC比较图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # AUC条形图
        folds = [r['fold'] for r in all_fold_results]
        auc_values = [r['test_metrics']['auc'] for r in all_fold_results]

        axes[0, 0].bar(folds, auc_values, color='skyblue')
        axes[0, 0].set_xlabel('Fold')
        axes[0, 0].set_ylabel('AUC')
        axes[0, 0].set_title('AUC of every fold')
        axes[0, 0].set_ylim([0, 1])

        # 添加平均值线
        mean_auc = np.mean(auc_values)
        axes[0, 0].axhline(y=mean_auc, color='r', linestyle='--', label=f'average: {mean_auc:.3f}')
        axes[0, 0].legend()

        # F1分数条形图
        f1_values = [r['test_metrics']['f1'] for r in all_fold_results]
        axes[0, 1].bar(folds, f1_values, color='lightgreen')
        axes[0, 1].set_xlabel('Fold')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].set_title('F1 Score of every fold')
        axes[0, 1].set_ylim([0, 1])

        mean_f1 = np.mean(f1_values)
        axes[0, 1].axhline(y=mean_f1, color='r', linestyle='--', label=f'average: {mean_f1:.3f}')
        axes[0, 1].legend()

        # 多指标箱线图
        metrics_to_plot = ['auc', 'auprc', 'accuracy', 'f1']
        data_to_plot = [[r['test_metrics'][m] for r in all_fold_results] for m in metrics_to_plot]

        bp = axes[1, 0].boxplot(data_to_plot, labels=metrics_to_plot)
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Boxplot of distribution for each indicator')
        axes[1, 0].set_ylim([0, 1])

        # 各折AUC趋势图
        axes[1, 1].plot(folds, auc_values, 'o-', linewidth=2, markersize=8)
        axes[1, 1].fill_between(folds,
                                [a - np.std(auc_values) / 2 for a in auc_values],
                                [a + np.std(auc_values) / 2 for a in auc_values],
                                alpha=0.2)
        axes[1, 1].set_xlabel('Fold')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].set_title('AUC trend')
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(f'{scenario}Scenario cross-validation performance comparison', fontsize=16)
        plt.tight_layout()

        chart_path = os.path.join(cv_output_dir, 'cv_comparison_charts.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"交叉验证比较图表已保存: {chart_path}")

    except Exception as e:
        logger.warning(f"生成图表失败: {e}")


def select_best_fold_model(all_fold_results, cv_output_dir):
    """选择最佳fold模型（基于平均AUC）"""
    # 计算每个fold的平均性能（可以自定义加权）
    fold_scores = []
    for fold_result in all_fold_results:
        metrics = fold_result['test_metrics']
        # 使用AUC作为主要指标
        score = metrics['auc']
        fold_scores.append((fold_result['fold'], score, fold_result))

    # 按分数排序
    fold_scores.sort(key=lambda x: x[1], reverse=True)

    best_fold, best_score, best_fold_result = fold_scores[0]

    # 复制最佳模型到主目录
    best_model_source = best_fold_result['best_model_path']
    if os.path.exists(best_model_source):
        best_model_dest = os.path.join(cv_output_dir, 'best_overall_model.pth')
        import shutil
        shutil.copy2(best_model_source, best_model_dest)

        logger.info(f"最佳模型来自 Fold {best_fold} (AUC={best_score:.4f})")
        logger.info(f"最佳模型已复制到: {best_model_dest}")

    return {
        'best_fold': best_fold,
        'best_score': best_score,
        'best_fold_result': best_fold_result,
        'best_model_path': best_model_dest if os.path.exists(best_model_source) else None
    }


def generate_final_comprehensive_report(all_fold_results, cv_summary, best_fold_info,
                                        scenario, config, cv_output_dir):
    """生成最终综合报告"""
    report_path = os.path.join(cv_output_dir, 'final_comprehensive_report.txt')

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"合成致死预测模型 - 交叉验证最终综合报告\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. 实验概述:\n")
        f.write(f"   场景: {scenario}\n")
        f.write(f"   总折数: {len(all_fold_results)}\n")
        f.write(f"   每类样本数: {config.TOTAL_SAMPLES_PER_CLASS}\n")
        f.write(f"   随机种子: {config.RANDOM_SEED}\n")
        f.write(f"   完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("2. 最佳模型信息:\n")
        if best_fold_info:
            f.write(f"   最佳fold: {best_fold_info['best_fold']}\n")
            f.write(f"   最佳AUC: {best_fold_info['best_score']:.4f}\n")
            f.write(f"   最佳模型路径: {best_fold_info.get('best_model_path', 'N/A')}\n\n")

        f.write("3. 性能指标统计:\n")
        metrics_summary = cv_summary.get('metrics_summary', {})
        for metric, stats in metrics_summary.items():
            f.write(f"   {metric.upper()}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                    f"(范围: {stats['min']:.4f}-{stats['max']:.4f})\n")

        f.write("\n4. 各折详细结果:\n")
        for fold_result in all_fold_results:
            f.write(f"\n   Fold {fold_result['fold']}:\n")
            f.write(f"     数据大小: 训练={fold_result['train_size']}, "
                    f"验证={fold_result['val_size']}, "
                    f"测试={fold_result['test_size']}\n")
            metrics = fold_result['test_metrics']
            f.write(f"     AUC: {metrics['auc']:.4f}, "
                    f"F1: {metrics['f1']:.4f}, "
                    f"准确率: {metrics['accuracy']:.4f}\n")
            f.write(f"     输出目录: {fold_result['output_dir']}\n")

        f.write("\n5. 结论与建议:\n")
        f.write("   - 模型稳定性: " +
                ("良好" if max([r['test_metrics']['auc'] for r in all_fold_results]) -
                           min([r['test_metrics']['auc'] for r in all_fold_results]) < 0.1
                 else "需要改进") + "\n")
        f.write("   - 泛化能力: 可通过不同fold间性能差异评估\n")
        f.write("   - 最佳fold可用于后续应用\n")

        f.write("\n6. 生成的文件:\n")
        f.write("   - cv_summary_report.txt: 总体报告\n")
        f.write("   - cv_summary_metrics.csv: 指标汇总表\n")
        f.write("   - cv_comparison_charts.png: 比较图表\n")
        f.write("   - best_overall_model.pth: 最佳模型权重\n")
        f.write("   - fold_XX/: 各折详细结果\n")
        f.write("   - detailed_reports/: 详细报告\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("报告生成完成\n")
        f.write("=" * 80 + "\n")

    logger.info(f"最终综合报告已保存: {report_path}")


def _run_c1_scenario(processor, train_df, val_df, test_df, config, device_manager):
    """执行C1场景（transductive）- 支持交叉验证"""
    # 创建transductive数据处理器
    trans_processor = SLDataProcessor(config=config, device_manager=device_manager)

    # 设置训练集基因映射
    trans_processor.create_gene_mapping(train_df)

    # 过滤验证集和测试集（只保留训练集中出现的基因）
    val_df_filtered = trans_processor.filter_data_by_gene_mapping(val_df)
    test_df_filtered = trans_processor.filter_data_by_gene_mapping(test_df)

    logger.info(f"过滤后验证集: {len(val_df_filtered)}条")
    logger.info(f"过滤后测试集: {len(test_df_filtered)}条")

    # 创建训练图数据
    train_graph_data = trans_processor._create_single_graph_data(train_df, mode='train')

    # 创建验证图数据
    val_graph_data = trans_processor._create_single_graph_data(val_df_filtered, mode='val')

    # 创建测试图数据
    test_graph_data = trans_processor._create_single_graph_data(test_df_filtered, mode='test')


    # 构建知识图谱（基于训练集基因）
    train_gene_ids = list(trans_processor.idx_to_gene_id.values())
    kg_builder = KnowledgeGraphBuilder(
        edges_file=config.KG_EDGES_FILE,
        nodes_file=config.KG_NODES_FILE,
        string_file=config.STRING_FILE,
        local_model_path=config.LOCAL_MODEL_PATH,
        use_local_model=config.USE_PRETRAINED,
        config=config,
        device_manager=device_manager
    )

    kg_data, gene_mapping = kg_builder.create_knowledge_graph_data(train_gene_ids)

    # 保存图数据
    torch.save(train_graph_data, os.path.join(config.OUTPUT_DIR, 'train_graph_data.pt'))
    torch.save(kg_data, os.path.join(config.OUTPUT_DIR, 'kg_data.pt'))
    logger.info(f"图数据已保存至 {config.OUTPUT_DIR}")

    # 创建模型
    model = Columbina_Model(config=config, device_manager=device_manager)

    # 创建训练器
    trainer = Trainer(
        model, train_graph_data, val_graph_data, kg_data, gene_mapping,
        processor=trans_processor, config=config, device_manager=device_manager
    )

    # 训练模型
    logger.info("开始训练transductive模型...")
    train_losses, val_losses, val_aucs, val_auprcs = trainer.train()

    # 测试集评估
    logger.info("测试集评估...")
    test_metrics = evaluate_test_set(
        model, test_graph_data, kg_data, gene_mapping,
        config, device_manager
    )

    # 保存结果 - 传递测试图数据
    save_scenario_results(
        scenario='C1',
        trainer=trainer,
        test_metrics=test_metrics,
        config=config,
        train_df=train_df,
        val_df=val_df_filtered,
        test_df=test_df_filtered  # 添加测试图数据
    )

    return trainer, test_metrics


def _run_c2c3_scenario(processor, train_df, val_df, test_df, config, device_manager, scenario):
    """执行C2/C3场景（inductive）"""
    # 创建inductive数据处理器
    inductive_processor = InductiveSLDataProcessor(config=config, device_manager=device_manager)

    # 合并训练集和验证集用于训练（inductive训练需要看到所有基因）
    # 但测试集基因不能出现在训练中
    combined_train_val = pd.concat([train_df, val_df], ignore_index=True)

    # 检测新基因（相对于训练集）
    train_genes = set(train_df['Gene.A']).union(set(train_df['Gene.B']))
    val_genes = set(val_df['Gene.A']).union(set(val_df['Gene.B']))
    test_genes = set(test_df['Gene.A']).union(set(test_df['Gene.B']))

    logger.info(f"训练集基因: {len(train_genes)}")
    logger.info(f"验证集基因: {len(val_genes)}")
    logger.info(f"测试集基因: {len(test_genes)}")

    # 创建基因映射（包含所有基因）
    all_genes = list(train_genes.union(val_genes).union(test_genes))
    inductive_processor.create_gene_mapping(combined_train_val)

    # 获取训练集基因ID
    train_gene_ids = list(train_genes)

    # 创建inductive图数据
    train_graph_data, val_graph_data, kg_data, gene_mapping, _, _ = inductive_processor.create_inductive_graph_data()

    #保存图数据
    torch.save(train_graph_data, os.path.join(config.OUTPUT_DIR, 'train_graph_data.pt'))
    torch.save(kg_data, os.path.join(config.OUTPUT_DIR, 'kg_data.pt'))
    logger.info(f"图数据已保存至 {config.OUTPUT_DIR}")

    # 创建inductive模型
    model = Inductive_Columbina_Model(config=config, device_manager=device_manager)

    # 创建inductive训练器
    trainer = InductiveTrainer(
        model, train_graph_data, val_graph_data, kg_data, gene_mapping,
        processor=inductive_processor,
        train_gene_ids=train_gene_ids,
        all_gene_ids=all_genes,
        config=config, device_manager=device_manager
    )

    # 将处理器保存到训练器中，以便后续使用
    trainer.processor = inductive_processor

    # 训练模型
    logger.info(f"开始训练inductive模型（{scenario}场景）...")
    train_losses, val_losses, val_aucs, val_auprcs = trainer.train()

    # 创建测试图数据
    test_graph_data = inductive_processor._create_single_graph_data_with_features(
        test_df,
        train_graph_data.x,  # 使用训练图的特征
        mode='test'
    )

    # 测试集评估
    logger.info("测试集评估...")
    test_metrics = evaluate_inductive_test_set(
        model, test_graph_data, kg_data, gene_mapping,
        all_genes, config, device_manager
    )

    # 保存结果
    save_scenario_results(
        scenario=scenario,
        trainer=trainer,
        test_metrics=test_metrics,
        config=config,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df
    )

    # 打印运行摘要
    print_run_summary(trainer, test_metrics, scenario)

    return trainer, test_metrics


def evaluate_test_set(model, test_graph_data, kg_data, gene_mapping, config, device_manager):
    """评估测试集（transductive）"""
    model.eval()
    with torch.no_grad():
        _, test_pred, _ = model(test_graph_data, kg_data, gene_mapping, mode='eval')

        y_true = test_graph_data.y.cpu().numpy()
        y_pred = test_pred.cpu().numpy()

        # 添加调试信息
        logger.info(f"测试集评估: 真实标签长度={len(y_true)}, 预测值长度={len(y_pred)}")

        # 计算指标
        test_loss = nn.BCEWithLogitsLoss()(
            torch.tensor(y_pred, dtype=torch.float),
            torch.tensor(y_true, dtype=torch.float)
        ).item()

        test_auc = roc_auc_score(y_true, y_pred)
        test_auprc = average_precision_score(y_true, y_pred)

        # 最佳阈值
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores[:-1])  # 忽略最后一个元素
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

        y_pred_binary = (y_pred > best_threshold).astype(int)
        accuracy = accuracy_score(y_true, y_pred_binary)
        precision = precision_score(y_true, y_pred_binary, zero_division=0)
        recall = recall_score(y_true, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true, y_pred_binary, zero_division=0)

        return {
            'loss': test_loss,
            'auc': test_auc,
            'auprc': test_auprc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'best_threshold': best_threshold,
            'predictions': y_pred.tolist()  # 确保返回预测值
        }


def evaluate_inductive_test_set(model, test_graph_data, kg_data, gene_mapping, all_genes, config, device_manager):
    """评估测试集（inductive）"""
    model.eval()
    with torch.no_grad():
        _, test_pred, _ = model(
            test_graph_data, kg_data, gene_mapping,
            mode='eval', gene_ids=all_genes
        )

        y_true = test_graph_data.y.cpu().numpy()
        y_pred = test_pred.cpu().numpy()

        # 计算指标
        test_loss = nn.BCEWithLogitsLoss()(
            torch.tensor(y_pred, dtype=torch.float),
            torch.tensor(y_true, dtype=torch.float)
        ).item()

        test_auc = roc_auc_score(y_true, y_pred)
        test_auprc = average_precision_score(y_true, y_pred)

        # 最佳阈值
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores[:-1])  # 忽略最后一个元素
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

        y_pred_binary = (y_pred > best_threshold).astype(int)
        accuracy = accuracy_score(y_true, y_pred_binary)
        precision = precision_score(y_true, y_pred_binary, zero_division=0)
        recall = recall_score(y_true, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true, y_pred_binary, zero_division=0)

        return {
            'loss': test_loss,
            'auc': test_auc,
            'auprc': test_auprc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'best_threshold': best_threshold,
            'predictions': y_pred.tolist()  # 添加预测值
        }


def create_test_graph_data(test_df, trainer, config):
    """创建测试图数据"""
    # 这里需要根据具体的场景创建测试图数据
    # 为了简化，我们返回一个占位符图数据
    # 实际使用时需要根据具体实现填充

    # 创建一个简单的Data对象
    num_nodes = max(test_df['Gene.A'].max(), test_df['Gene.B'].max()) + 1 if len(test_df) > 0 else 0

    # 边索引
    edge_index = torch.tensor([
        test_df['Gene.A'].values,
        test_df['Gene.B'].values
    ], dtype=torch.long)

    # 边标签
    edge_label = torch.tensor(test_df['label'].values, dtype=torch.float)

    # 节点特征（使用随机特征作为占位符）
    node_features = torch.randn(num_nodes, 262)

    return Data(
        x=node_features,
        edge_index=edge_index,
        y=edge_label,
        num_nodes=num_nodes
    )


def save_scenario_results(scenario, trainer, test_metrics, config, train_df, val_df, test_df):
    """保存场景结果 - 完整修复版"""
    # 确保路径是字符串
    output_dir = str(config.OUTPUT_DIR)
    if not output_dir.endswith(scenario):
        output_dir = os.path.join(output_dir, str(scenario))

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"结果保存目录: {output_dir}")

    # 保存测试指标
    metrics_df = pd.DataFrame([test_metrics])
    metrics_path = str(os.path.join(output_dir, 'test_metrics.csv'))
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"测试指标已保存至: {metrics_path}")

    # ========== 保存预测结果 ==========
    logger.info("开始保存预测结果...")
    all_results = []  # 用于收集所有预测结果

    try:
        # 1. 训练集预测
        if hasattr(trainer, 'train_data') and hasattr(trainer, 'processor'):
            logger.info("处理训练集预测...")
            try:
                # 确保模型在评估模式
                trainer.model.eval()
                with torch.no_grad():
                    # 获取训练集预测
                    _, train_pred, _ = trainer.model(
                        trainer.train_data, trainer.kg_data, trainer.gene_mapping, mode='eval'
                    )

                # 转换为numpy数组
                train_pred_np = train_pred.cpu().numpy()
                train_labels_np = trainer.train_data.y.cpu().numpy()
                train_edges = trainer.train_data.edge_index.cpu().numpy().T

                # 创建训练集预测数据框
                train_pred_data = []
                for i, (src_idx, dst_idx) in enumerate(train_edges):
                    if i >= len(train_pred_np):
                        break

                    # 获取基因ID
                    gene_a = trainer.processor.idx_to_gene_id.get(int(src_idx), f"UNKNOWN_{src_idx}")
                    gene_b = trainer.processor.idx_to_gene_id.get(int(dst_idx), f"UNKNOWN_{dst_idx}")

                    train_pred_data.append({
                        'Gene.A': gene_a,
                        'Gene.B': gene_b,
                        'label': float(train_labels_np[i]) if i < len(train_labels_np) else 0,
                        'prediction': float(train_pred_np[i]),
                        'set': 'train'
                    })

                if train_pred_data:
                    train_pred_df = pd.DataFrame(train_pred_data)
                    all_results.append(train_pred_df)
                    logger.info(f"训练集预测: {len(train_pred_df)}条")
                else:
                    logger.warning("训练集预测数据为空")

            except Exception as e:
                logger.warning(f"保存训练集预测结果失败: {e}")
                logger.debug(f"详细错误: {traceback.format_exc()}")

        # 2. 验证集预测
        if hasattr(trainer, 'val_data') and hasattr(trainer, 'processor'):
            logger.info("处理验证集预测...")
            try:
                trainer.model.eval()
                with torch.no_grad():
                    _, val_pred, _ = trainer.model(
                        trainer.val_data, trainer.kg_data, trainer.gene_mapping, mode='eval'
                    )

                # 转换为numpy数组
                val_pred_np = val_pred.cpu().numpy()
                val_labels_np = trainer.val_data.y.cpu().numpy()
                val_edges = trainer.val_data.edge_index.cpu().numpy().T

                # 创建验证集预测数据框
                val_pred_data = []
                for i, (src_idx, dst_idx) in enumerate(val_edges):
                    if i >= len(val_pred_np):
                        break

                    gene_a = trainer.processor.idx_to_gene_id.get(int(src_idx), f"UNKNOWN_{src_idx}")
                    gene_b = trainer.processor.idx_to_gene_id.get(int(dst_idx), f"UNKNOWN_{dst_idx}")

                    val_pred_data.append({
                        'Gene.A': gene_a,
                        'Gene.B': gene_b,
                        'label': float(val_labels_np[i]) if i < len(val_labels_np) else 0,
                        'prediction': float(val_pred_np[i]),
                        'set': 'val'
                    })

                if val_pred_data:
                    val_pred_df = pd.DataFrame(val_pred_data)
                    all_results.append(val_pred_df)
                    logger.info(f"验证集预测: {len(val_pred_df)}条")
                else:
                    logger.warning("验证集预测数据为空")

            except Exception as e:
                logger.warning(f"保存验证集预测结果失败: {e}")
                logger.debug(f"详细错误: {traceback.format_exc()}")

        # 3. 测试集预测
        logger.info("处理测试集预测...")
        if 'predictions' in test_metrics:
            logger.info(f"从test_metrics中获取测试集预测值，长度: {len(test_metrics['predictions'])}")

            # 检查长度是否匹配
            predictions = test_metrics['predictions']
            if len(predictions) == len(test_df):
                test_df_copy = test_df.copy()
                test_df_copy['prediction'] = predictions
                test_df_copy['set'] = 'test'
                all_results.append(test_df_copy)
                logger.info(f"测试集预测: {len(test_df_copy)}条")
            else:
                logger.warning(f"预测值长度不匹配: 预测值{len(predictions)}条, 测试集{len(test_df)}条")

                # 尝试重新计算预测
                logger.info("尝试重新计算测试集预测...")
                try:
                    # 检查是否可以通过其他方式重新计算
                    if hasattr(trainer, 'processor') and hasattr(trainer, 'model'):
                        # 这里需要确保有测试图数据可用
                        # 由于C1场景中测试集过滤了基因，我们需要重新创建测试图数据
                        logger.info("重新创建测试图数据并计算预测...")

                        # 创建测试集基因映射（使用训练集的映射）
                        test_genes = set(test_df['Gene.A']).union(set(test_df['Gene.B']))

                        # 检查基因是否都在映射中
                        missing_genes = []
                        for gene in test_genes:
                            if gene not in trainer.processor.gene_id_to_idx:
                                missing_genes.append(gene)

                        if missing_genes:
                            logger.warning(f"测试集中有{len(missing_genes)}个基因不在训练集映射中")
                            # 只保留在映射中的基因对
                            test_df_filtered = test_df[
                                test_df['Gene.A'].isin(trainer.processor.gene_id_to_idx) &
                                test_df['Gene.B'].isin(trainer.processor.gene_id_to_idx)
                                ]

                            if len(test_df_filtered) > 0:
                                logger.info(f"过滤后测试集: {len(test_df_filtered)}条")

                                # 创建测试图数据
                                test_edge_index = torch.tensor([
                                    [trainer.processor.gene_id_to_idx[g1] for g1 in test_df_filtered['Gene.A']],
                                    [trainer.processor.gene_id_to_idx[g2] for g2 in test_df_filtered['Gene.B']]
                                ], dtype=torch.long).to(trainer.device_manager.target_device)

                                test_edge_label = torch.tensor(test_df_filtered['label'].values, dtype=torch.float).to(
                                    trainer.device_manager.target_device
                                )

                                # 使用训练图的节点特征
                                test_node_features = trainer.train_data.x

                                test_graph_data = Data(
                                    x=test_node_features,
                                    edge_index=test_edge_index,
                                    y=test_edge_label,
                                    num_nodes=trainer.train_data.num_nodes
                                )

                                # 重新计算预测
                                trainer.model.eval()
                                with torch.no_grad():
                                    _, test_pred, _ = trainer.model(
                                        test_graph_data, trainer.kg_data, trainer.gene_mapping, mode='eval'
                                    )

                                # 创建测试集预测数据框
                                test_df_copy = test_df_filtered.copy()
                                test_df_copy['prediction'] = test_pred.cpu().numpy().tolist()
                                test_df_copy['set'] = 'test'
                                all_results.append(test_df_copy)
                                logger.info(f"重新计算的测试集预测: {len(test_df_copy)}条")
                            else:
                                logger.error("过滤后测试集为空，无法计算预测")
                                # 创建空的测试集预测
                                test_df_copy = test_df.copy()
                                test_df_copy['prediction'] = 0.0
                                test_df_copy['set'] = 'test'
                                all_results.append(test_df_copy)
                        else:
                            logger.info("所有测试基因都在映射中，可以重新计算预测")
                    else:
                        logger.warning("无法重新计算测试集预测，创建默认预测值")
                        test_df_copy = test_df.copy()
                        test_df_copy['prediction'] = 0.0
                        test_df_copy['set'] = 'test'
                        all_results.append(test_df_copy)

                except Exception as e:
                    logger.error(f"重新计算测试集预测失败: {e}")
                    logger.debug(f"详细错误: {traceback.format_exc()}")

                    # 创建默认预测值
                    test_df_copy = test_df.copy()
                    test_df_copy['prediction'] = 0.0
                    test_df_copy['set'] = 'test'
                    all_results.append(test_df_copy)
        else:
            logger.warning("test_metrics中没有predictions字段")

            # 尝试重新计算预测
            try:
                logger.info("尝试重新计算测试集预测...")

                # 这里需要创建测试图数据
                # 由于我们可能没有原始的测试图数据，需要根据test_df重新创建
                if hasattr(trainer, 'processor'):
                    # 只保留在映射中的基因对
                    test_df_filtered = test_df[
                        test_df['Gene.A'].isin(trainer.processor.gene_id_to_idx) &
                        test_df['Gene.B'].isin(trainer.processor.gene_id_to_idx)
                        ].copy()

                    if len(test_df_filtered) > 0:
                        logger.info(f"使用过滤后的测试集: {len(test_df_filtered)}条")

                        # 创建边索引
                        test_edge_index = torch.tensor([
                            [trainer.processor.gene_id_to_idx[g1] for g1 in test_df_filtered['Gene.A']],
                            [trainer.processor.gene_id_to_idx[g2] for g2 in test_df_filtered['Gene.B']]
                        ], dtype=torch.long).to(trainer.device_manager.target_device)

                        test_edge_label = torch.tensor(test_df_filtered['label'].values, dtype=torch.float).to(
                            trainer.device_manager.target_device
                        )

                        # 使用训练图的节点特征
                        test_node_features = trainer.train_data.x

                        test_graph_data = Data(
                            x=test_node_features,
                            edge_index=test_edge_index,
                            y=test_edge_label,
                            num_nodes=trainer.train_data.num_nodes
                        )

                        # 重新计算预测
                        trainer.model.eval()
                        with torch.no_grad():
                            _, test_pred, _ = trainer.model(
                                test_graph_data, trainer.kg_data, trainer.gene_mapping, mode='eval'
                            )

                        test_df_filtered['prediction'] = test_pred.cpu().numpy().tolist()
                        test_df_filtered['set'] = 'test'
                        all_results.append(test_df_filtered)
                        logger.info(f"重新计算的测试集预测: {len(test_df_filtered)}条")
                    else:
                        logger.warning("过滤后测试集为空，创建默认预测值")
                        test_df_copy = test_df.copy()
                        test_df_copy['prediction'] = 0.0
                        test_df_copy['set'] = 'test'
                        all_results.append(test_df_copy)
                else:
                    logger.warning("没有processor可用，创建默认预测值")
                    test_df_copy = test_df.copy()
                    test_df_copy['prediction'] = 0.0
                    test_df_copy['set'] = 'test'
                    all_results.append(test_df_copy)

            except Exception as e:
                logger.error(f"重新计算测试集预测失败: {e}")
                logger.debug(f"详细错误: {traceback.format_exc()}")

                # 创建默认预测值
                test_df_copy = test_df.copy()
                test_df_copy['prediction'] = 0.0
                test_df_copy['set'] = 'test'
                all_results.append(test_df_copy)

    except Exception as e:
        logger.error(f"保存预测结果时发生严重错误: {e}")
        logger.debug(f"详细错误: {traceback.format_exc()}")

    # 合并所有结果并保存
    if all_results:
        try:
            all_results_df = pd.concat(all_results, ignore_index=True)
            results_path = str(os.path.join(output_dir, 'predictions.csv'))
            all_results_df.to_csv(results_path, index=False)
            logger.info(f"预测结果已保存至: {results_path}")
            logger.info(f"总预测记录: {len(all_results_df)}条")
            logger.info(f"训练集: {len(all_results_df[all_results_df['set'] == 'train'])}条")
            logger.info(f"验证集: {len(all_results_df[all_results_df['set'] == 'val'])}条")
            logger.info(f"测试集: {len(all_results_df[all_results_df['set'] == 'test'])}条")
        except Exception as e:
            logger.error(f"合并和保存预测结果失败: {e}")
            logger.debug(f"详细错误: {traceback.format_exc()}")
    else:
        logger.warning("没有预测结果可保存")

    # ========== 保存训练历史 ==========
    if hasattr(trainer, 'train_history'):
        try:
            history_df = pd.DataFrame(trainer.train_history)
            history_path = str(os.path.join(output_dir, 'training_history.csv'))
            history_df.to_csv(history_path, index=False)
            logger.info(f"训练历史已保存至: {history_path}")
        except Exception as e:
            logger.warning(f"保存训练历史失败: {e}")

    logger.info(f"{scenario}场景结果已保存至: {output_dir}")
    logger.info(f"测试集性能 - AUC: {test_metrics.get('auc', 'N/A'):.4f}, F1: {test_metrics.get('f1', 'N/A'):.4f}")

    return all_results_df if 'all_results_df' in locals() else None


def print_run_summary(trainer, test_metrics, scenario):
    """打印运行摘要"""
    logger.info("\n" + "=" * 80)
    logger.info(f"运行摘要 - {scenario}场景")
    logger.info("=" * 80)

    if hasattr(trainer, 'train_history') and trainer.train_history:
        if 'val_auc' in trainer.train_history:
            best_auc = max(trainer.train_history['val_auc'])
            best_auc_epoch = trainer.train_history['epoch'][trainer.train_history['val_auc'].index(best_auc)]
            logger.info(f"最佳验证AUC: {best_auc:.4f} (epoch {best_auc_epoch})")

        if 'val_f1' in trainer.train_history:
            best_f1 = max(trainer.train_history['val_f1'])
            best_f1_epoch = trainer.train_history['epoch'][trainer.train_history['val_f1'].index(best_f1)]
            logger.info(f"最佳验证F1: {best_f1:.4f} (epoch {best_f1_epoch})")

    logger.info(f"测试集AUC: {test_metrics['auc']:.4f}")
    logger.info(f"测试集F1: {test_metrics['f1']:.4f}")
    logger.info(f"测试集准确率: {test_metrics['accuracy']:.4f}")
    logger.info(f"测试集精确率: {test_metrics['precision']:.4f}")
    logger.info(f"测试集召回率: {test_metrics['recall']:.4f}")
    logger.info(f"最佳阈值: {test_metrics['best_threshold']:.4f}")

    # 计算泛化差距
    if hasattr(trainer, 'train_history') and 'val_auc' in trainer.train_history:
        val_auc = trainer.train_history['val_auc'][-1] if trainer.train_history['val_auc'] else 0
        gap = val_auc - test_metrics['auc']
        logger.info(f"泛化差距 (验证AUC - 测试AUC): {gap:.4f}")
        if gap > 0.05:
            logger.warning("较大的泛化差距，可能存在过拟合")
        else:
            logger.info("泛化差距在可接受范围内")

    logger.info("=" * 80)


def interactive_main():
    """交互式主函数"""
    print("\n" + "=" * 60)
    print("合成致死预测模型 - 多场景支持")
    print("=" * 60)

    print("\n请选择场景:")
    print("1. C1 - 按基因对划分 (Transductive)")
    print("2. C2 - 按基因划分，测试集每对基因恰好一个在训练集 (Inductive)")
    print("3. C3 - 按基因划分，测试集基因对在训练集中均未出现 (Inductive)")

    choice = input("\n请输入选择 (1/2/3): ").strip()

    if choice == '1':
        scenario = 'C1'
    elif choice == '2':
        scenario = 'C2'
    elif choice == '3':
        scenario = 'C3'
    else:
        print("无效选择，使用默认C1场景")
        scenario = 'C1'

    print(f"\n选择场景: {scenario}")

    # 设备选择
    if torch.cuda.is_available():
        print(f"\n检测到GPU: {torch.cuda.get_device_name(0)}")
        gpu_choice = input("是否使用GPU? (y/n, 默认y): ").strip().lower()
        if gpu_choice == 'n':
            device_choice = 'cpu'
        else:
            device_choice = 'cuda'
    else:
        print("未检测到GPU，将使用CPU运行")
        device_choice = 'cpu'

    # 测试模式选项
    test_mode = input("\n是否使用测试模式（简化运行）? (y/n, 默认n): ").strip().lower() == 'y'

    # 创建配置
    config = Config()

    if test_mode:
        config.EPOCHS = 50
        config.EARLY_STOPPING_PATIENCE = 10
        print("使用测试模式：EPOCHS=50, EARLY_STOPPING_PATIENCE=10")

    print(f"\n开始执行{scenario}场景...")
    print("=" * 60)

    # 执行
    try:
        trainer, test_metrics = universal_main(
            scenario=scenario,
            config=config,
            device_choice=device_choice,
            test_mode=test_mode
        )

        # 生成最终报告
        save_final_report(scenario, config, test_metrics, trainer)

        print(f"\n{scenario}场景执行完成!")
        print(f"测试集AUC: {test_metrics['auc']:.4f}")
        print(f"测试集F1: {test_metrics['f1']:.4f}")

    except Exception as e:
        print(f"\n执行出错: {e}")
        import traceback
        traceback.print_exc()

    print("=" * 60)


def save_final_report(scenario, config, test_metrics, trainer):
    """保存最终报告"""
    report_dir = os.path.join(str(config.OUTPUT_DIR), str(scenario))
    os.makedirs(report_dir, exist_ok=True)

    report_path = os.path.join(report_dir, 'final_report.txt')

    with open(str(report_path), 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("合成致死预测模型 - 最终报告\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"场景: {scenario}\n")
        f.write(f"执行时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("配置参数:\n")
        f.write(f"  训练轮次: {config.EPOCHS}\n")
        f.write(f"  隐藏维度: {config.HIDDEN_DIM}\n")
        f.write(f"  嵌入维度: {config.EMBEDDING_DIM}\n")
        f.write(f"  学习率: {config.LEARNING_RATE}\n")
        f.write(f"  早停耐心值: {config.EARLY_STOPPING_PATIENCE}\n")
        f.write(f"  随机种子: {config.RANDOM_SEED}\n\n")

        f.write("测试集性能指标:\n")
        f.write(f"  损失: {test_metrics['loss']:.4f}\n")
        f.write(f"  AUC: {test_metrics['auc']:.4f}\n")
        f.write(f"  AUPRC: {test_metrics['auprc']:.4f}\n")
        f.write(f"  准确率: {test_metrics['accuracy']:.4f}\n")
        f.write(f"  精确率: {test_metrics['precision']:.4f}\n")
        f.write(f"  召回率: {test_metrics['recall']:.4f}\n")
        f.write(f"  F1分数: {test_metrics['f1']:.4f}\n")
        f.write(f"  最佳阈值: {test_metrics['best_threshold']:.4f}\n\n")

        # 训练历史摘要
        if hasattr(trainer, 'train_history') and trainer.train_history:
            f.write("训练历史摘要:\n")
            if 'val_auc' in trainer.train_history and trainer.train_history['val_auc']:
                best_auc = max(trainer.train_history['val_auc'])
                best_epoch = trainer.train_history['epoch'][trainer.train_history['val_auc'].index(best_auc)]
                f.write(f"  最佳验证AUC: {best_auc:.4f} (epoch {best_epoch})\n")

            if 'val_f1' in trainer.train_history and trainer.train_history['val_f1']:
                best_f1 = max(trainer.train_history['val_f1'])
                best_epoch = trainer.train_history['epoch'][trainer.train_history['val_f1'].index(best_f1)]
                f.write(f"  最佳验证F1: {best_f1:.4f} (epoch {best_epoch})\n")

        f.write("\n输出文件:\n")
        f.write(f"  测试指标: {os.path.join(report_dir, 'test_metrics.csv')}\n")
        f.write(f"  预测结果: {os.path.join(report_dir, 'predictions.csv')}\n")
        f.write(f"  训练历史: {os.path.join(report_dir, 'training_history.csv')}\n")
        f.write(f"  模型权重: {os.path.join(str(config.OUTPUT_DIR), f'best_{scenario.lower()}_model.pth')}\n")
        f.write(f"  日志文件: {os.path.join(str(config.OUTPUT_DIR), f'{scenario.lower()}_sl_prediction.log')}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("报告生成完成\n")
        f.write("=" * 80 + "\n")

    logger.info(f"最终报告已保存至: {report_path}")

    # 打印测试结果到控制台
    logger.info("\n" + "=" * 60)
    logger.info(f"测试集性能摘要 - {scenario}场景")
    logger.info("=" * 60)
    logger.info(f"  AUC: {test_metrics['auc']:.4f}")
    logger.info(f"  AUPRC: {test_metrics['auprc']:.4f}")
    logger.info(f"  准确率: {test_metrics['accuracy']:.4f}")
    logger.info(f"  精确率: {test_metrics['precision']:.4f}")
    logger.info(f"  召回率: {test_metrics['recall']:.4f}")
    logger.info(f"  F1分数: {test_metrics['f1']:.4f}")
    logger.info(f"  最佳阈值: {test_metrics['best_threshold']:.4f}")
    logger.info("=" * 60)


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    import argparse

    # 命令行参数解析
    parser = argparse.ArgumentParser(description='CM4SL合成致死预测模型 - 支持C1/C2/C3三种场景')
    parser.add_argument('--scenario', type=str, choices=['C1', 'C2', 'C3'],
                        help='选择场景: C1(按基因对划分), C2(按基因划分-半新), C3(按基因划分-全新)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'gpu', 'auto'],
                        default='auto', help='运行设备: cuda/gpu(GPU), cpu(CPU), auto(自动选择)')
    parser.add_argument('--mode', type=str, choices=['inductive', 'transductive'],
                        default=None, help='学习模式: inductive(归纳式), transductive(直推式)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮次 (覆盖配置文件中的值)')
    parser.add_argument('--test', action='store_true',
                        help='测试模式 (简化运行，减少轮次)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO', help='日志级别 (默认: INFO)')
    parser.add_argument('--cv', action='store_true',
                        help='启用交叉验证')
    parser.add_argument('--folds', type=int, default=10,
                        help='交叉验证折数 (默认: 10)')
    parser.add_argument('--samples-per-class', type=int, default=20000,
                        help='每类样本数 (默认: 20000)')
    parser.add_argument('--save-all', action='store_true',
                        help='保存所有中间结果')

    args = parser.parse_args()

    # 创建配置
    config = Config()

    # 覆盖配置参数
    config.RANDOM_SEED = args.seed

    if args.epochs:
        config.EPOCHS = args.epochs

    if args.test:
        config.EPOCHS = 50
        config.EARLY_STOPPING_PATIENCE = 10

    # 创建输出目录
    output_dir_suffix = args.scenario if args.scenario else 'default'
    config.OUTPUT_DIR = os.path.join(config.BASE_DIR, "output", output_dir_suffix)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # 设置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(str(config.OUTPUT_DIR), f'{output_dir_suffix}_sl_prediction.log')),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("CM4SL合成致死预测模型")
    logger.info("=" * 60)

    # 设备选择
    if args.device == 'auto':
        device_choice = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif args.device == 'gpu':
        device_choice = 'cuda'
    else:
        device_choice = args.device

    logger.info(f"配置:")
    logger.info(f"  场景: {args.scenario if args.scenario else '未指定(使用交互式)'}")
    logger.info(f"  设备: {device_choice}")
    logger.info(f"  随机种子: {args.seed}")
    logger.info(f"  日志级别: {args.log_level}")
    if args.epochs:
        logger.info(f"  训练轮次: {args.epochs}")
    if args.test:
        logger.info("  模式: 测试模式")

    try:
        # 如果没有指定场景，使用交互式模式
        if args.scenario is None:
            logger.info("\n未指定场景，使用交互式模式...")
            interactive_main()
        else:
            # 处理场景参数
            scenario = args.scenario.upper()

            # 兼容原有的mode参数
            if args.mode is None:
                # 根据场景自动选择模式
                if scenario == 'C1':
                    args.mode = 'transductive'
                else:
                    args.mode = 'inductive'

            logger.info(f"执行{scenario}场景，使用{args.mode}模式...")

            # 对于C1场景，如果启用交叉验证，强制使用10折以保证8:1:1比例
            if scenario == 'C1' and args.cv:
                logger.info("C1场景交叉验证：固定使用10折以保证8:1:1比例")
                if args.folds != 10:
                    logger.warning(f"C1场景交叉验证要求8:1:1比例，将折数从{args.folds}调整为10")
                    args.folds = 10

            # 所有场景都使用universal_main函数执行
            if args.cv:  # 交叉验证模式
                all_fold_results, cv_summary, best_fold_info = universal_main(
                    scenario=scenario,
                    config=config,
                    device_choice=device_choice,
                    test_mode=args.test,
                    enable_cv=args.cv,
                    n_folds=args.folds
                )

                # 交叉验证模式下不需要单独的save_final_report，因为已经在universal_main中生成了报告
                cv_output_dir = os.path.join(str(config.OUTPUT_DIR), f'{scenario}_cv')
                logger.info(f"交叉验证完成！详细结果请查看：{cv_output_dir}")

                # 打印交叉验证的统计摘要
                if cv_summary and 'metrics_summary' in cv_summary:
                    logger.info("\n交叉验证性能统计摘要：")
                    logger.info("-" * 60)
                    for metric, stats in cv_summary['metrics_summary'].items():
                        logger.info(f"  {metric.upper()}: {stats['mean']:.4f} ± {stats['std']:.4f}")
                    logger.info("-" * 60)

                # 保存最佳模型路径到配置中
                if best_fold_info and best_fold_info.get('best_model_path'):
                    config.BEST_MODEL_PATH = best_fold_info['best_model_path']

            else:  # 单次划分模式
                trainer, test_metrics = universal_main(
                    scenario=scenario,
                    config=config,
                    device_choice=device_choice,
                    test_mode=args.test,
                    enable_cv=args.cv,
                    n_folds=args.folds
                )

                # 生成最终报告
                save_final_report(scenario, config, test_metrics, trainer)

        logger.info("\n" + "=" * 60)
        logger.info("训练完成!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        import traceback

        traceback.print_exc()

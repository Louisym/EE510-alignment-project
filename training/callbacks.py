"""
Training Callbacks with Visualization
集成可视化功能的训练回调
"""

import os
from pathlib import Path
from typing import Dict, Optional
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import torch

from training.visualization import MetricsTracker


class VisualizationCallback(TrainerCallback):
    """
    HuggingFace Trainer 的可视化回调
    用于 SFT 训练的指标追踪和可视化
    """

    def __init__(self, output_dir: str, experiment_name: str = "sft_training"):
        """
        初始化回调

        Args:
            output_dir: 输出目录
            experiment_name: 实验名称
        """
        self.tracker = MetricsTracker(output_dir, experiment_name)
        self.experiment_name = experiment_name

    def on_log(self, args: TrainingArguments, state: TrainerState,
               control: TrainerControl, logs: Dict[str, float] = None, **kwargs):
        """
        每次记录日志时调用

        Args:
            args: 训练参数
            state: 训练状态
            control: 训练控制
            logs: 日志字典
        """
        if logs is None:
            return

        # 提取指标
        metrics = {}

        if 'loss' in logs:
            metrics['train_loss'] = logs['loss']

        if 'eval_loss' in logs:
            metrics['val_loss'] = logs['eval_loss']

        if 'learning_rate' in logs:
            metrics['learning_rate'] = logs['learning_rate']

        # 记录指标
        if metrics:
            epoch = state.epoch if state.epoch is not None else 0
            self.tracker.log_metrics(
                step=state.global_step,
                epoch=int(epoch),
                metrics=metrics
            )

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState,
                    control: TrainerControl, **kwargs):
        """
        每个 epoch 结束时调用
        """
        # 保存指标
        self.tracker.save_metrics()

    def on_train_end(self, args: TrainingArguments, state: TrainerState,
                    control: TrainerControl, **kwargs):
        """
        训练结束时调用
        """
        print("\n" + "="*70)
        print("📊 Generating training visualizations...")
        print("="*70)

        # 保存最终指标
        self.tracker.save_metrics()

        # 生成所有图表
        self.tracker.plot_all()

        # 生成摘要报告
        self.tracker.generate_summary_report()

        # 创建仪表板
        from training.visualization import create_training_dashboard
        create_training_dashboard(self.tracker)

        print("\n✓ Visualization complete! Check the plots directory for results.")


class GRPOVisualizationCallback:
    """
    GRPO Trainer 的可视化回调
    因为 GRPO 不使用 HuggingFace Trainer，需要手动集成
    """

    def __init__(self, output_dir: str, experiment_name: str = "grpo_training"):
        """
        初始化回调

        Args:
            output_dir: 输出目录
            experiment_name: 实验名称
        """
        self.tracker = MetricsTracker(output_dir, experiment_name)
        self.experiment_name = experiment_name

    def log_metrics(self, step: int, epoch: int, metrics: Dict[str, float]):
        """
        记录训练指标

        Args:
            step: 步数
            epoch: epoch 数
            metrics: 指标字典
        """
        # 记录到 tracker
        self.tracker.log_metrics(step, epoch, metrics)

        # 同时记录 GRPO 专用指标
        if any(key in metrics for key in ['mean_reward', 'max_reward', 'min_reward', 'kl_divergence']):
            self.tracker.log_grpo_metrics(step, metrics)

    def on_epoch_end(self, epoch: int):
        """Epoch 结束时调用"""
        self.tracker.save_metrics()

        # 生成中间图表
        if epoch % 2 == 0:  # 每 2 个 epoch 生成一次图表
            self.tracker.plot_all()

    def on_train_end(self):
        """训练结束时调用"""
        print("\n" + "="*70)
        print("📊 Generating GRPO training visualizations...")
        print("="*70)

        # 保存最终指标
        self.tracker.save_metrics()

        # 生成所有图表
        self.tracker.plot_all()

        # 生成摘要报告
        self.tracker.generate_summary_report()

        # 创建仪表板
        from training.visualization import create_training_dashboard
        create_training_dashboard(self.tracker)

        print("\n✓ GRPO Visualization complete!")


if __name__ == "__main__":
    print("Testing Visualization Callbacks...")

    # 测试 GRPO callback
    callback = GRPOVisualizationCallback("./test_callback_output", "test_grpo")

    # 模拟训练
    import numpy as np
    for epoch in range(3):
        for step in range(20):
            global_step = epoch * 20 + step
            callback.log_metrics(
                step=global_step,
                epoch=epoch,
                metrics={
                    'loss': 2.0 * np.exp(-global_step/30) + 0.1,
                    'mean_reward': 0.5 + 0.3 * global_step / 60,
                    'max_reward': 0.7 + 0.25 * global_step / 60,
                    'min_reward': 0.3 + 0.15 * global_step / 60,
                    'kl_divergence': 0.1 * np.exp(-global_step/30)
                }
            )

        callback.on_epoch_end(epoch)

    callback.on_train_end()
    print("✓ Test completed!")

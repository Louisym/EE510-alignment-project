"""
Training Visualization and Metrics Tracking
用于 Presentation 和 Report 的完整可视化工具
"""

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd

# 设置中文字体支持（可选）
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置绘图风格
sns.set_style("whitegrid")
sns.set_palette("husl")


class MetricsTracker:
    """训练指标追踪器"""

    def __init__(self, output_dir: str, experiment_name: str = "training"):
        """
        初始化指标追踪器

        Args:
            output_dir: 输出目录
            experiment_name: 实验名称
        """
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.metrics_dir = self.output_dir / "metrics"
        self.plots_dir = self.output_dir / "plots"

        # 创建目录
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # 指标历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch': [],
            'step': [],
            'timestamp': []
        }

        # GRPO 专用指标
        self.grpo_history = {
            'mean_reward': [],
            'max_reward': [],
            'min_reward': [],
            'kl_divergence': [],
            'step': []
        }

        print(f"📊 MetricsTracker initialized: {self.plots_dir}")

    def log_metrics(self, step: int, epoch: int, metrics: Dict[str, float]):
        """
        记录训练指标

        Args:
            step: 训练步数
            epoch: 当前 epoch
            metrics: 指标字典
        """
        self.history['step'].append(step)
        self.history['epoch'].append(epoch)
        self.history['timestamp'].append(datetime.now().isoformat())

        # 记录所有指标
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)

    def log_grpo_metrics(self, step: int, metrics: Dict[str, float]):
        """
        记录 GRPO 专用指标

        Args:
            step: 训练步数
            metrics: GRPO 指标
        """
        self.grpo_history['step'].append(step)

        for key in ['mean_reward', 'max_reward', 'min_reward', 'kl_divergence']:
            if key in metrics:
                self.grpo_history[key].append(metrics[key])

    def save_metrics(self):
        """保存指标到 JSON 文件"""
        # 保存主要指标
        metrics_path = self.metrics_dir / f"{self.experiment_name}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        # 保存 GRPO 指标
        if self.grpo_history['step']:
            grpo_path = self.metrics_dir / f"{self.experiment_name}_grpo_metrics.json"
            with open(grpo_path, 'w') as f:
                json.dump(self.grpo_history, f, indent=2)

        print(f"✓ Metrics saved to {metrics_path}")

    def plot_loss_curves(self, save: bool = True, show: bool = False):
        """
        绘制损失曲线

        Args:
            save: 是否保存图片
            show: 是否显示图片
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # 训练损失 vs 步数
        if 'train_loss' in self.history and self.history['train_loss']:
            axes[0].plot(self.history['step'], self.history['train_loss'],
                        label='Training Loss', linewidth=2, marker='o', markersize=3)

            if 'val_loss' in self.history and self.history['val_loss']:
                # 验证损失可能不是每步都有
                val_steps = [s for s, v in zip(self.history['step'], self.history['val_loss']) if v is not None]
                val_losses = [v for v in self.history['val_loss'] if v is not None]
                axes[0].plot(val_steps, val_losses,
                            label='Validation Loss', linewidth=2, marker='s', markersize=3)

            axes[0].set_xlabel('Training Steps', fontsize=12)
            axes[0].set_ylabel('Loss', fontsize=12)
            axes[0].set_title('Loss Curves', fontsize=14, fontweight='bold')
            axes[0].legend(fontsize=10)
            axes[0].grid(True, alpha=0.3)

        # 训练损失 vs Epoch
        if 'epoch' in self.history and 'train_loss' in self.history:
            # 按 epoch 分组计算平均损失
            # 过滤掉 None 值
            valid_data = [(e, l) for e, l in zip(self.history['epoch'], self.history['train_loss'])
                         if l is not None]
            if valid_data:
                epochs, losses = zip(*valid_data)
                df = pd.DataFrame({
                    'epoch': epochs,
                    'train_loss': losses
                })
                epoch_loss = df.groupby('epoch')['train_loss'].mean()
            else:
                epoch_loss = None

            if epoch_loss is not None:
                axes[1].plot(epoch_loss.index, epoch_loss.values,
                            label='Training Loss', linewidth=2, marker='o', markersize=5)

                axes[1].set_xlabel('Epoch', fontsize=12)
                axes[1].set_ylabel('Average Loss', fontsize=12)
                axes[1].set_title('Loss per Epoch', fontsize=14, fontweight='bold')
                axes[1].legend(fontsize=10)
                axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            save_path = self.plots_dir / f"{self.experiment_name}_loss_curves.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Loss curves saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_learning_rate(self, save: bool = True, show: bool = False):
        """
        绘制学习率变化曲线

        Args:
            save: 是否保存
            show: 是否显示
        """
        if 'learning_rate' not in self.history or not self.history['learning_rate']:
            print("⚠ No learning rate data to plot")
            return

        plt.figure(figsize=(10, 5))
        plt.plot(self.history['step'], self.history['learning_rate'],
                linewidth=2, color='coral')
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save:
            save_path = self.plots_dir / f"{self.experiment_name}_learning_rate.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Learning rate plot saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_grpo_rewards(self, save: bool = True, show: bool = False):
        """
        绘制 GRPO 奖励曲线

        Args:
            save: 是否保存
            show: 是否显示
        """
        if not self.grpo_history['step']:
            print("⚠ No GRPO reward data to plot")
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # 奖励变化
        axes[0].plot(self.grpo_history['step'], self.grpo_history['mean_reward'],
                    label='Mean Reward', linewidth=2, marker='o', markersize=3)
        axes[0].fill_between(
            self.grpo_history['step'],
            self.grpo_history['min_reward'],
            self.grpo_history['max_reward'],
            alpha=0.2,
            label='Min-Max Range'
        )
        axes[0].set_xlabel('Training Steps', fontsize=12)
        axes[0].set_ylabel('Reward', fontsize=12)
        axes[0].set_title('GRPO Rewards Over Training', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # KL 散度
        if self.grpo_history.get('kl_divergence'):
            axes[1].plot(self.grpo_history['step'], self.grpo_history['kl_divergence'],
                        linewidth=2, color='red', marker='s', markersize=3)
            axes[1].set_xlabel('Training Steps', fontsize=12)
            axes[1].set_ylabel('KL Divergence', fontsize=12)
            axes[1].set_title('KL Divergence from Reference Model', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            save_path = self.plots_dir / f"{self.experiment_name}_grpo_rewards.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ GRPO rewards plot saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_all(self):
        """生成所有图表"""
        print("\n" + "="*60)
        print("📊 Generating all plots...")
        print("="*60)

        self.plot_loss_curves(save=True, show=False)
        self.plot_learning_rate(save=True, show=False)

        if self.grpo_history['step']:
            self.plot_grpo_rewards(save=True, show=False)

        print(f"\n✓ All plots saved to: {self.plots_dir}")

    def generate_summary_report(self) -> str:
        """
        生成训练摘要报告

        Returns:
            报告文本
        """
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append(f"TRAINING SUMMARY REPORT: {self.experiment_name}")
        report_lines.append("=" * 70)
        report_lines.append("")

        # 基本信息
        if self.history['step']:
            report_lines.append(f"Total training steps: {self.history['step'][-1]}")
            report_lines.append(f"Total epochs: {max(self.history['epoch'])}")

        # 训练损失统计
        if self.history.get('train_loss'):
            train_losses = [l for l in self.history['train_loss'] if l is not None]
            report_lines.append(f"\nTraining Loss:")
            report_lines.append(f"  Initial: {train_losses[0]:.4f}")
            report_lines.append(f"  Final: {train_losses[-1]:.4f}")
            report_lines.append(f"  Best: {min(train_losses):.4f}")
            report_lines.append(f"  Improvement: {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.2f}%")

        # 验证损失统计
        if self.history.get('val_loss'):
            val_losses = [l for l in self.history['val_loss'] if l is not None]
            if val_losses:
                report_lines.append(f"\nValidation Loss:")
                report_lines.append(f"  Best: {min(val_losses):.4f}")
                report_lines.append(f"  Final: {val_losses[-1]:.4f}")

        # GRPO 统计
        if self.grpo_history['step']:
            report_lines.append(f"\nGRPO Metrics:")
            report_lines.append(f"  Initial mean reward: {self.grpo_history['mean_reward'][0]:.4f}")
            report_lines.append(f"  Final mean reward: {self.grpo_history['mean_reward'][-1]:.4f}")
            report_lines.append(f"  Best mean reward: {max(self.grpo_history['mean_reward']):.4f}")
            improvement = ((self.grpo_history['mean_reward'][-1] - self.grpo_history['mean_reward'][0]) /
                          abs(self.grpo_history['mean_reward'][0]) * 100)
            report_lines.append(f"  Improvement: {improvement:.2f}%")

        report_lines.append("")
        report_lines.append("=" * 70)

        report_text = "\n".join(report_lines)

        # 保存报告
        report_path = self.metrics_dir / f"{self.experiment_name}_summary.txt"
        with open(report_path, 'w') as f:
            f.write(report_text)

        print(report_text)
        print(f"\n✓ Summary report saved to {report_path}")

        return report_text


class ModelComparator:
    """模型对比工具 - 用于比较 Base/SFT/GRPO 模型"""

    def __init__(self, output_dir: str):
        """
        初始化模型对比器

        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.comparison_dir = self.output_dir / "comparisons"
        self.comparison_dir.mkdir(parents=True, exist_ok=True)

        # 存储各模型的输出
        self.model_outputs = {
            'base': [],
            'sft': [],
            'grpo': []
        }

        self.questions = []

    def add_comparison(self, question: str, base_output: str,
                      sft_output: str = "", grpo_output: str = ""):
        """
        添加一个对比样本

        Args:
            question: 问题
            base_output: 基础模型输出
            sft_output: SFT 模型输出
            grpo_output: GRPO 模型输出
        """
        self.questions.append(question)
        self.model_outputs['base'].append(base_output)
        self.model_outputs['sft'].append(sft_output)
        self.model_outputs['grpo'].append(grpo_output)

    def save_comparison_table(self):
        """保存对比表格"""
        comparison_data = []

        for i, question in enumerate(self.questions):
            comparison_data.append({
                'Question': question[:100] + "..." if len(question) > 100 else question,
                'Base Model': self.model_outputs['base'][i][:200] + "..." if len(self.model_outputs['base'][i]) > 200 else self.model_outputs['base'][i],
                'SFT Model': self.model_outputs['sft'][i][:200] + "..." if i < len(self.model_outputs['sft']) and self.model_outputs['sft'][i] else "N/A",
                'GRPO Model': self.model_outputs['grpo'][i][:200] + "..." if i < len(self.model_outputs['grpo']) and self.model_outputs['grpo'][i] else "N/A"
            })

        df = pd.DataFrame(comparison_data)

        # 保存为 CSV
        csv_path = self.comparison_dir / "model_comparison.csv"
        df.to_csv(csv_path, index=False)
        print(f"✓ Comparison table saved to {csv_path}")

        # 保存为 markdown（适合 report）
        md_path = self.comparison_dir / "model_comparison.md"
        with open(md_path, 'w') as f:
            f.write("# Model Output Comparison\n\n")
            f.write(df.to_markdown(index=False))

        print(f"✓ Comparison markdown saved to {md_path}")

    def plot_comparison_metrics(self, metrics: Dict[str, Dict[str, float]],
                               save: bool = True, show: bool = False):
        """
        绘制模型对比指标

        Args:
            metrics: 各模型的指标，格式: {'base': {'loss': x, ...}, 'sft': {...}, 'grpo': {...}}
            save: 是否保存
            show: 是否显示
        """
        models = list(metrics.keys())
        metric_names = list(metrics[models[0]].keys())

        fig, axes = plt.subplots(1, len(metric_names), figsize=(6*len(metric_names), 5))

        if len(metric_names) == 1:
            axes = [axes]

        for idx, metric_name in enumerate(metric_names):
            values = [metrics[model].get(metric_name, 0) for model in models]
            colors = ['#3498db', '#2ecc71', '#e74c3c'][:len(models)]

            axes[idx].bar(models, values, color=colors, alpha=0.7, edgecolor='black')
            axes[idx].set_ylabel(metric_name, fontsize=12)
            axes[idx].set_title(f'{metric_name} Comparison', fontsize=14, fontweight='bold')
            axes[idx].grid(True, alpha=0.3, axis='y')

            # 在柱状图上添加数值
            for i, (model, value) in enumerate(zip(models, values)):
                axes[idx].text(i, value, f'{value:.4f}',
                             ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        if save:
            save_path = self.comparison_dir / "metrics_comparison.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Metrics comparison plot saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()


def create_training_dashboard(metrics_tracker: MetricsTracker,
                              output_path: Optional[str] = None):
    """
    创建训练仪表板（所有图表在一个大图中）

    Args:
        metrics_tracker: 指标追踪器
        output_path: 保存路径
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. 训练损失
    ax1 = fig.add_subplot(gs[0, :2])
    if metrics_tracker.history.get('train_loss'):
        ax1.plot(metrics_tracker.history['step'], metrics_tracker.history['train_loss'],
                label='Training Loss', linewidth=2)
        if metrics_tracker.history.get('val_loss'):
            val_steps = [s for s, v in zip(metrics_tracker.history['step'],
                                           metrics_tracker.history['val_loss']) if v is not None]
            val_losses = [v for v in metrics_tracker.history['val_loss'] if v is not None]
            if val_losses:
                ax1.plot(val_steps, val_losses, label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Curves', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # 2. 学习率
    ax2 = fig.add_subplot(gs[0, 2])
    if metrics_tracker.history.get('learning_rate'):
        ax2.plot(metrics_tracker.history['step'], metrics_tracker.history['learning_rate'],
                color='coral', linewidth=2)
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate', fontweight='bold')
        ax2.grid(True, alpha=0.3)

    # 3. GRPO 奖励
    if metrics_tracker.grpo_history['step']:
        ax3 = fig.add_subplot(gs[1, :])
        ax3.plot(metrics_tracker.grpo_history['step'],
                metrics_tracker.grpo_history['mean_reward'],
                label='Mean Reward', linewidth=2)
        ax3.fill_between(
            metrics_tracker.grpo_history['step'],
            metrics_tracker.grpo_history['min_reward'],
            metrics_tracker.grpo_history['max_reward'],
            alpha=0.2
        )
        ax3.set_xlabel('Steps')
        ax3.set_ylabel('Reward')
        ax3.set_title('GRPO Rewards', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # 4. 损失分布（如果有多个 epoch）
    ax4 = fig.add_subplot(gs[2, 0])
    if metrics_tracker.history.get('train_loss'):
        ax4.hist(metrics_tracker.history['train_loss'], bins=30, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Loss')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Loss Distribution', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')

    # 5. 每个 Epoch 的平均损失
    ax5 = fig.add_subplot(gs[2, 1])
    if metrics_tracker.history.get('epoch') and metrics_tracker.history.get('train_loss'):
        df = pd.DataFrame({
            'epoch': metrics_tracker.history['epoch'],
            'train_loss': metrics_tracker.history['train_loss']
        })
        epoch_loss = df.groupby('epoch')['train_loss'].mean()
        ax5.plot(epoch_loss.index, epoch_loss.values, marker='o', linewidth=2)
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Average Loss')
        ax5.set_title('Loss per Epoch', fontweight='bold')
        ax5.grid(True, alpha=0.3)

    # 添加总标题
    fig.suptitle('Training Dashboard', fontsize=20, fontweight='bold', y=0.995)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training dashboard saved to {output_path}")
    else:
        save_path = metrics_tracker.plots_dir / f"{metrics_tracker.experiment_name}_dashboard.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training dashboard saved to {save_path}")

    plt.close()


if __name__ == "__main__":
    # 测试代码
    print("Testing Visualization Tools...\n")

    # 创建测试数据
    tracker = MetricsTracker("./test_output", "test_experiment")

    # 模拟训练数据
    for step in range(100):
        epoch = step // 20
        tracker.log_metrics(
            step=step,
            epoch=epoch,
            metrics={
                'train_loss': 2.0 * np.exp(-step/50) + 0.1,
                'val_loss': 2.1 * np.exp(-step/50) + 0.15 if step % 10 == 0 else None,
                'learning_rate': 2e-4 * (1 - step/100)
            }
        )

        if step > 50:  # 模拟 GRPO 训练
            tracker.log_grpo_metrics(
                step=step,
                metrics={
                    'mean_reward': 0.5 + 0.3 * (step - 50) / 50,
                    'max_reward': 0.8 + 0.2 * (step - 50) / 50,
                    'min_reward': 0.2 + 0.1 * (step - 50) / 50,
                    'kl_divergence': 0.1 * np.exp(-(step-50)/25)
                }
            )

    # 生成所有图表
    tracker.plot_all()
    tracker.save_metrics()
    tracker.generate_summary_report()

    # 创建仪表板
    create_training_dashboard(tracker)

    print("\n✓ Test completed! Check ./test_output/plots/")

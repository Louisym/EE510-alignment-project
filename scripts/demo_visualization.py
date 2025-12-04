"""
可视化功能演示脚本
Quick demo of visualization features
"""

import sys
from pathlib import Path
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.visualization import MetricsTracker, ModelComparator, create_training_dashboard


def demo_sft_visualization():
    """演示 SFT 训练可视化"""
    print("\n" + "="*70)
    print("📊 Demo 1: SFT Training Visualization")
    print("="*70)

    # 创建 tracker
    tracker = MetricsTracker("demo_outputs/sft", "sft_demo")

    # 模拟 SFT 训练数据（8 epochs, 每个 epoch 20 steps）
    print("\n模拟 SFT 训练数据...")
    for epoch in range(8):
        for step_in_epoch in range(20):
            global_step = epoch * 20 + step_in_epoch

            # 模拟损失逐渐下降
            train_loss = 2.1 * np.exp(-global_step / 80) + 0.1 + np.random.normal(0, 0.02)

            # 每 10 步验证一次
            val_loss = None
            if global_step % 10 == 0:
                val_loss = 2.2 * np.exp(-global_step / 80) + 0.15 + np.random.normal(0, 0.02)

            # 学习率线性衰减
            lr = 2e-4 * (1 - global_step / 160)

            # 记录指标
            tracker.log_metrics(
                step=global_step,
                epoch=epoch,
                metrics={
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'learning_rate': lr
                }
            )

    # 生成所有图表
    print("\n生成图表...")
    tracker.plot_all()
    tracker.save_metrics()
    tracker.generate_summary_report()

    # 创建仪表板
    create_training_dashboard(tracker)

    print(f"\n✓ SFT 可视化完成！")
    print(f"  查看目录: demo_outputs/sft/plots/")


def demo_grpo_visualization():
    """演示 GRPO 训练可视化"""
    print("\n" + "="*70)
    print("📊 Demo 2: GRPO Training Visualization")
    print("="*70)

    # 创建 tracker
    tracker = MetricsTracker("demo_outputs/grpo", "grpo_demo")

    # 模拟 GRPO 训练数据（3 epochs, 每个 epoch 20 steps）
    print("\n模拟 GRPO 训练数据...")
    for epoch in range(3):
        for step_in_epoch in range(20):
            global_step = epoch * 20 + step_in_epoch

            # 模拟损失
            train_loss = 0.4 * np.exp(-global_step / 30) + 0.05 + np.random.normal(0, 0.01)

            # 模拟奖励逐渐提升
            mean_reward = 0.5 + 0.3 * (1 - np.exp(-global_step / 30)) + np.random.normal(0, 0.02)
            max_reward = mean_reward + 0.15 + np.random.normal(0, 0.01)
            min_reward = mean_reward - 0.15 + np.random.normal(0, 0.01)

            # KL 散度逐渐降低
            kl_div = 0.1 * np.exp(-global_step / 20) + np.random.normal(0, 0.005)

            # 学习率
            lr = 1e-5 * (1 - global_step / 60)

            # 记录指标
            metrics = {
                'loss': train_loss,
                'learning_rate': lr,
                'mean_reward': mean_reward,
                'max_reward': max_reward,
                'min_reward': min_reward,
                'kl_divergence': kl_div
            }

            tracker.log_metrics(step=global_step, epoch=epoch, metrics=metrics)
            tracker.log_grpo_metrics(step=global_step, metrics=metrics)

    # 生成所有图表
    print("\n生成图表...")
    tracker.plot_all()
    tracker.save_metrics()
    tracker.generate_summary_report()

    # 创建仪表板
    create_training_dashboard(tracker)

    print(f"\n✓ GRPO 可视化完成！")
    print(f"  查看目录: demo_outputs/grpo/plots/")


def demo_model_comparison():
    """演示模型对比"""
    print("\n" + "="*70)
    print("📊 Demo 3: Model Comparison")
    print("="*70)

    # 创建对比器
    comparator = ModelComparator("demo_outputs/comparison")

    # 添加示例对比
    questions = [
        "Find P[A|B] if A ∩ B = ∅",
        "Show that P[A|B] satisfies the axioms of probability",
        "Let X be a geometric random variable. Find P[N = k | N ≤ m]"
    ]

    base_outputs = [
        "P(A|B) = 0 when A and B are disjoint.",
        "P(A|B) = P(A ∩ B) / P(B) satisfies probability axioms.",
        "For geometric distribution, the conditional probability is p(1-p)^(k-1)."
    ]

    sft_outputs = [
        "If A ∩ B = ∅, then A and B are disjoint. Therefore:\nP(A|B) = P(A ∩ B) / P(B) = 0 / P(B) = 0.\n\nThis follows from the definition of conditional probability.",
        "We verify the three axioms:\n1) 0 ≤ P(A|B) ≤ 1 since P(A ∩ B) ≤ P(B)\n2) P(S|B) = P(S ∩ B) / P(B) = P(B) / P(B) = 1\n3) For disjoint events, P(A ∪ C | B) = P(A|B) + P(C|B)",
        "Let N be geometric with parameter p. For k ≤ m:\nP(N = k | N ≤ m) = P(N = k) / P(N ≤ m)\n= p(1-p)^(k-1) / (1 - (1-p)^m)\n\nFor k > m, the probability is 0."
    ]

    grpo_outputs = [
        "Given: A ∩ B = ∅ (A and B are disjoint events)\n\nUsing the definition of conditional probability:\nP(A|B) = P(A ∩ B) / P(B)\n\nSince A and B are disjoint:\nP(A ∩ B) = P(∅) = 0\n\nTherefore:\nP(A|B) = 0 / P(B) = 0\n\nIntuitively, if event B occurs and A and B cannot occur together, then the probability of A given B must be zero.",
        "We must verify that P(·|B) satisfies the three axioms of probability:\n\nAxiom 1: Non-negativity and upper bound\nSince A ∩ B ⊆ B, we have P(A ∩ B) ≤ P(B).\nDividing by P(B) > 0: P(A|B) = P(A ∩ B)/P(B) ≤ 1\nAlso, P(A ∩ B) ≥ 0, so P(A|B) ≥ 0.\n\nAxiom 2: Probability of sample space\nP(S|B) = P(S ∩ B)/P(B) = P(B)/P(B) = 1\n\nAxiom 3: Additivity for disjoint events\nIf A ∩ C = ∅, then (A ∩ B) ∩ (C ∩ B) = ∅\nP(A ∪ C | B) = P((A ∪ C) ∩ B)/P(B)\n= P((A ∩ B) ∪ (C ∩ B))/P(B)\n= [P(A ∩ B) + P(C ∩ B)]/P(B)\n= P(A|B) + P(C|B)",
        "Problem: Find P[N = k | N ≤ m] for geometric random variable N.\n\nSolution:\nLet N ~ Geometric(p), so P(N = n) = p(1-p)^(n-1) for n ≥ 1.\n\nFirst, compute P(N ≤ m):\nP(N ≤ m) = Σ(n=1 to m) p(1-p)^(n-1)\n= p · [1 - (1-p)^m] / p\n= 1 - (1-p)^m\n\nNow apply Bayes' theorem:\nFor k ≤ m:\nP(N = k | N ≤ m) = P(N = k, N ≤ m) / P(N ≤ m)\n= P(N = k) / P(N ≤ m)\n= p(1-p)^(k-1) / [1 - (1-p)^m]\n\nFor k > m:\nP(N = k | N ≤ m) = 0 (impossible)\n\nTherefore:\nP(N = k | N ≤ m) = {\n  p(1-p)^(k-1) / [1 - (1-p)^m], if k ≤ m\n  0, if k > m\n}"
    ]

    print("\n添加对比样本...")
    for q, base, sft, grpo in zip(questions, base_outputs, sft_outputs, grpo_outputs):
        comparator.add_comparison(q, base, sft, grpo)

    # 保存对比表
    print("\n保存对比表...")
    comparator.save_comparison_table()

    # 绘制指标对比
    print("\n绘制指标对比...")
    metrics = {
        'base': {
            'avg_length': 45.3,
            'completeness': 0.45,
            'formula_accuracy': 0.65
        },
        'sft': {
            'avg_length': 156.7,
            'completeness': 0.82,
            'formula_accuracy': 0.89
        },
        'grpo': {
            'avg_length': 198.4,
            'completeness': 0.93,
            'formula_accuracy': 0.95
        }
    }
    comparator.plot_comparison_metrics(metrics, save=True, show=False)

    print(f"\n✓ 模型对比完成！")
    print(f"  查看目录: demo_outputs/comparison/")


def main():
    """运行所有演示"""
    print("\n" + "="*70)
    print("🎯 可视化功能完整演示")
    print("="*70)
    print("\n这个脚本将演示所有可视化功能：")
    print("  1. SFT 训练可视化")
    print("  2. GRPO 训练可视化")
    print("  3. 模型对比")
    print("\n所有输出将保存到 demo_outputs/ 目录\n")

    input("按 Enter 键开始演示...")

    # 运行演示
    demo_sft_visualization()
    demo_grpo_visualization()
    demo_model_comparison()

    # 总结
    print("\n" + "="*70)
    print("✅ 演示完成！")
    print("="*70)
    print("\n生成的文件：")
    print("\n📁 demo_outputs/")
    print("  ├── sft/")
    print("  │   ├── plots/")
    print("  │   │   ├── sft_demo_loss_curves.png         ⭐")
    print("  │   │   ├── sft_demo_learning_rate.png")
    print("  │   │   └── sft_demo_dashboard.png           ⭐ (推荐用于 PPT)")
    print("  │   └── metrics/")
    print("  │       ├── sft_demo_metrics.json")
    print("  │       └── sft_demo_summary.txt              ⭐ (训练摘要)")
    print("  │")
    print("  ├── grpo/")
    print("  │   ├── plots/")
    print("  │   │   ├── grpo_demo_loss_curves.png")
    print("  │   │   ├── grpo_demo_grpo_rewards.png       ⭐ (GRPO 关键图)")
    print("  │   │   └── grpo_demo_dashboard.png          ⭐ (推荐用于 PPT)")
    print("  │   └── metrics/")
    print("  │       ├── grpo_demo_metrics.json")
    print("  │       ├── grpo_demo_grpo_metrics.json")
    print("  │       └── grpo_demo_summary.txt             ⭐ (训练摘要)")
    print("  │")
    print("  └── comparison/")
    print("      ├── model_comparison.csv                  ⭐ (Excel 表格)")
    print("      ├── model_comparison.md                   ⭐ (Report 用)")
    print("      └── metrics_comparison.png                ⭐ (对比图)")

    print("\n💡 下一步：")
    print("  1. 查看生成的图表：在文件管理器中打开 demo_outputs/")
    print("  2. 阅读详细指南：VISUALIZATION_GUIDE.md")
    print("  3. 开始实际训练：运行 SFT 和 GRPO 训练脚本")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()

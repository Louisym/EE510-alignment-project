"""
LoRA Training: SVD-init vs Random-init Comparison

对比实验：
  - Student-random: 传统 LoRA（A随机，B=0）
  - Student-SVD: SVD 初始化的 LoRA（从 Teacher 的 ΔW 提取）

This script enables direct comparison of convergence speed and final performance.
"""

import os
import sys
import torch
import argparse
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import time

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel
)
from peft.tuners.lora import LoraLayer

from training.sft.data_loader import create_dataloaders


class SVDLoRAInitializer:
    """SVD-guided LoRA initialization"""

    def __init__(self, svd_factors_path: str, lora_rank: int):
        """
        初始化

        Args:
            svd_factors_path: SVD factors 文件路径
            lora_rank: LoRA rank
        """
        self.svd_factors = torch.load(svd_factors_path, map_location='cpu')
        self.lora_rank = lora_rank
        print(f"✓ Loaded SVD factors from: {svd_factors_path}")
        print(f"  Total layers: {len(self.svd_factors)}")

    def initialize_lora_weights(self, model: PeftModel):
        """
        将 SVD 的 B,A 写入 LoRA 参数

        Args:
            model: PEFT LoRA 模型
        """
        print("\n🔧 Initializing LoRA weights with SVD factors...")

        initialized_count = 0
        skipped_count = 0

        # 遍历所有模块
        for name, module in model.named_modules():
            # 检查是否在 SVD factors 中
            if name not in self.svd_factors:
                continue

            # 检查是否是 LoRA 层
            # PEFT 会将原始 Linear 包装，我们需要找到实际的 LoRA adapter
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                factors = self.svd_factors[name]
                B = factors['B']  # [d_out, r]
                A = factors['A']  # [r, d_in]

                # 检查维度
                lora_A_weight = module.lora_A['default'].weight
                lora_B_weight = module.lora_B['default'].weight

                if lora_A_weight.shape != A.shape:
                    print(f"  ⚠ Shape mismatch for {name}: "
                          f"expected {lora_A_weight.shape}, got {A.shape}")
                    skipped_count += 1
                    continue

                if lora_B_weight.shape != B.shape:
                    print(f"  ⚠ Shape mismatch for {name}: "
                          f"expected {lora_B_weight.shape}, got {B.shape}")
                    skipped_count += 1
                    continue

                # 写入权重
                with torch.no_grad():
                    lora_A_weight.copy_(A.to(lora_A_weight.device))
                    lora_B_weight.copy_(B.to(lora_B_weight.device))

                initialized_count += 1

        print(f"✓ Initialized {initialized_count} layers")
        if skipped_count > 0:
            print(f"  ⚠ Skipped {skipped_count} layers due to shape mismatch")


class ComparisonCallback(TrainerCallback):
    """记录详细的训练指标用于对比"""

    def __init__(self, output_dir: str, init_method: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.init_method = init_method
        self.log_file = self.output_dir / f"training_log_{init_method}.csv"

        # 初始化日志
        self.logs = []

        print(f"📊 Logging to: {self.log_file}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """记录每次 log"""
        if logs is None:
            return

        # 添加时间戳和初始化方法
        log_entry = {
            'step': state.global_step,
            'epoch': state.epoch,
            'init_method': self.init_method,
            **logs
        }
        self.logs.append(log_entry)

        # 定期保存
        if state.global_step % 10 == 0:
            self.save_logs()

    def on_train_end(self, args, state, control, **kwargs):
        """训练结束时保存"""
        self.save_logs()
        print(f"✓ Training log saved to: {self.log_file}")

    def save_logs(self):
        """保存日志到 CSV"""
        if self.logs:
            df = pd.DataFrame(self.logs)
            df.to_csv(self.log_file, index=False)


def create_lora_model(
    base_model_path: str,
    lora_rank: int,
    lora_alpha: int,
    target_modules: list,
    init_method: str = "random",
    svd_factors_path: Optional[str] = None,
    device: str = "auto"
) -> PeftModel:
    """
    创建 LoRA 模型

    Args:
        base_model_path: Base 模型路径
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha（推荐设为 rank，使缩放因子为1）
        target_modules: 目标模块列表
        init_method: 初始化方法 ("random" 或 "svd")
        svd_factors_path: SVD factors 文件路径（init_method="svd" 时需要）
        device: 设备

    Returns:
        LoRA 模型
    """
    print(f"\n{'='*70}")
    print(f"🚀 Creating LoRA Model (init={init_method})")
    print(f"{'='*70}")

    # 加载 base 模型
    print(f"\nLoading base model from: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )
    print("✓ Base model loaded")

    # 配置 LoRA
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,  # 设为 rank，使 α/r = 1
        lora_dropout=0.05,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        bias="none",
    )

    print(f"\nLoRA config:")
    print(f"  rank: {lora_rank}")
    print(f"  alpha: {lora_alpha}")
    print(f"  alpha/rank: {lora_alpha/lora_rank}")
    print(f"  target_modules: {target_modules}")

    # 应用 LoRA
    lora_model = get_peft_model(base_model, lora_config)
    lora_model.print_trainable_parameters()

    # SVD 初始化
    if init_method == "svd":
        if svd_factors_path is None:
            raise ValueError("svd_factors_path is required for init_method='svd'")

        initializer = SVDLoRAInitializer(svd_factors_path, lora_rank)
        initializer.initialize_lora_weights(lora_model)

    print("\n✓ LoRA model created")

    return lora_model


def train_lora(
    model: PeftModel,
    tokenizer,
    train_loader,
    val_loader,
    output_dir: str,
    init_method: str,
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    logging_steps: int = 10,
    save_steps: int = 100
):
    """
    训练 LoRA 模型

    Args:
        model: LoRA 模型
        tokenizer: Tokenizer
        train_loader: 训练数据
        val_loader: 验证数据
        output_dir: 输出目录
        init_method: 初始化方法
        num_epochs: Epoch 数
        learning_rate: 学习率
        batch_size: Batch size
        gradient_accumulation_steps: 梯度累积步数
        logging_steps: 日志步数
        save_steps: 保存步数
    """
    print(f"\n{'='*70}")
    print(f"🏋️ Training LoRA Model (init={init_method})")
    print(f"{'='*70}")

    # 训练参数
    # 检查是否有验证集
    has_eval = val_loader is not None and val_loader.dataset is not None and len(val_loader.dataset) > 0

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        eval_strategy="steps" if has_eval else "no",  # 只有有验证集时才评估
        eval_steps=save_steps if has_eval else None,
        load_best_model_at_end=False,
        report_to="none",
        remove_unused_columns=False,
        group_by_length=True,
    )

    # 创建回调
    comparison_callback = ComparisonCallback(output_dir, init_method)

    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_loader.dataset,
        eval_dataset=val_loader.dataset if val_loader else None,
        tokenizer=tokenizer,
        callbacks=[comparison_callback]
    )

    # 训练
    print(f"\nStarting training...")
    start_time = time.time()

    trainer.train()

    elapsed_time = time.time() - start_time
    print(f"\n✓ Training completed in {elapsed_time:.2f} seconds")

    # 保存最终模型
    final_model_path = os.path.join(output_dir, f"final_model_{init_method}")
    trainer.save_model(final_model_path)
    print(f"✓ Final model saved to: {final_model_path}")

    return trainer, comparison_callback


def compare_results(output_dir: str, methods: list = ["random", "svd"]):
    """
    对比两种初始化方法的结果

    Args:
        output_dir: 输出目录
        methods: 初始化方法列表
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")

    print(f"\n{'='*70}")
    print("📊 Generating Comparison Plots...")
    print(f"{'='*70}")

    output_dir = Path(output_dir)

    # 加载日志
    logs = {}
    for method in methods:
        log_file = output_dir / f"training_log_{method}.csv"
        if log_file.exists():
            logs[method] = pd.read_csv(log_file)
            print(f"✓ Loaded log for {method}: {len(logs[method])} entries")
        else:
            print(f"⚠ Log file not found for {method}: {log_file}")

    if len(logs) == 0:
        print("❌ No log files found")
        return

    # 创建对比图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 训练损失对比
    ax = axes[0, 0]
    for method, df in logs.items():
        if 'loss' in df.columns:
            ax.plot(df['step'], df['loss'], label=f'{method}-init', linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 2. 验证损失对比
    ax = axes[0, 1]
    for method, df in logs.items():
        eval_df = df[df['eval_loss'].notna()]
        if len(eval_df) > 0:
            ax.plot(eval_df['step'], eval_df['eval_loss'],
                   label=f'{method}-init', linewidth=2, marker='s', markersize=5)
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 3. 学习率对比
    ax = axes[1, 0]
    for method, df in logs.items():
        if 'learning_rate' in df.columns:
            ax.plot(df['step'], df['learning_rate'], label=f'{method}-init', linewidth=2)
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 4. 最终性能对比（柱状图）
    ax = axes[1, 1]
    final_metrics = {}
    for method, df in logs.items():
        # 获取最后10步的平均损失
        final_loss = df['loss'].tail(10).mean()
        final_metrics[method] = final_loss

    methods_list = list(final_metrics.keys())
    values_list = list(final_metrics.values())
    colors = ['#3498db', '#2ecc71'][:len(methods_list)]

    bars = ax.bar(methods_list, values_list, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Final Training Loss (last 10 steps avg)', fontsize=12)
    ax.set_title('Final Performance Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 在柱子上标注数值
    for bar, value in zip(bars, values_list):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{value:.4f}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()

    # 保存图表
    comparison_plot = output_dir / "comparison_random_vs_svd.png"
    plt.savefig(comparison_plot, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved to: {comparison_plot}")

    # 生成对比报告
    report_path = output_dir / "comparison_report.txt"
    report_lines = []
    report_lines.append("="*70)
    report_lines.append("LoRA Initialization Comparison Report")
    report_lines.append("="*70)
    report_lines.append("")

    for method, df in logs.items():
        report_lines.append(f"{method.upper()}-init:")
        report_lines.append(f"  Initial loss: {df['loss'].iloc[0]:.4f}")
        report_lines.append(f"  Final loss: {df['loss'].iloc[-1]:.4f}")
        report_lines.append(f"  Best loss: {df['loss'].min():.4f}")
        improvement = (df['loss'].iloc[0] - df['loss'].iloc[-1]) / df['loss'].iloc[0] * 100
        report_lines.append(f"  Improvement: {improvement:.2f}%")
        report_lines.append("")

    # 计算相对优势
    if len(logs) == 2:
        methods_list = list(logs.keys())
        final_losses = [logs[m]['loss'].iloc[-1] for m in methods_list]
        if 'svd' in methods_list and 'random' in methods_list:
            svd_idx = methods_list.index('svd')
            rand_idx = methods_list.index('random')
            advantage = (final_losses[rand_idx] - final_losses[svd_idx]) / final_losses[rand_idx] * 100
            report_lines.append(f"SVD-init advantage over Random-init: {advantage:.2f}%")

    report_lines.append("")
    report_lines.append("="*70)

    report_text = "\n".join(report_lines)

    with open(report_path, 'w') as f:
        f.write(report_text)

    print(f"✓ Comparison report saved to: {report_path}")
    print("\n" + report_text)


def main():
    parser = argparse.ArgumentParser(
        description="Train LoRA with different initialization methods"
    )

    # 模型和数据
    parser.add_argument("--base-model", type=str, required=True,
                       help="Base model path")
    parser.add_argument("--train-data", type=str, required=True,
                       help="Training data path (JSON)")
    parser.add_argument("--val-data", type=str, default=None,
                       help="Validation data path (optional)")

    # LoRA 配置
    parser.add_argument("--lora-rank", type=int, default=16,
                       help="LoRA rank (default: 16)")
    parser.add_argument("--lora-alpha", type=int, default=16,
                       help="LoRA alpha (default: 16, same as rank)")

    # 初始化方法
    parser.add_argument("--init", type=str, default="random",
                       choices=["random", "svd"],
                       help="Initialization method")
    parser.add_argument("--svd-factors", type=str, default=None,
                       help="Path to SVD factors (required for --init svd)")

    # 训练配置
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--max-length", type=int, default=512,
                       help="Max sequence length")

    # 输出
    parser.add_argument("--output-dir", type=str,
                       default="./experiments/svd_lora/training_results",
                       help="Output directory")

    args = parser.parse_args()

    # 验证 SVD 初始化参数
    if args.init == "svd" and args.svd_factors is None:
        raise ValueError("--svd-factors is required when --init svd")

    # 目标模块（根据模型调整）
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"]

    # 加载 tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer loaded")

    # 加载数据
    print(f"\nLoading data...")
    train_loader, val_loader = create_dataloaders(
        train_path=args.train_data,
        tokenizer=tokenizer,
        val_path=args.val_data,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    print(f"✓ Data loaded: {len(train_loader)} train batches")
    if val_loader:
        print(f"  {len(val_loader)} validation batches")

    # 创建 LoRA 模型
    lora_model = create_lora_model(
        base_model_path=args.base_model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        init_method=args.init,
        svd_factors_path=args.svd_factors,
        device="auto"
    )

    # 训练
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer, callback = train_lora(
        model=lora_model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=str(output_dir),
        init_method=args.init,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size
    )

    print(f"\n{'='*70}")
    print("✅ Training Complete!")
    print(f"{'='*70}")
    print(f"\nOutput directory: {output_dir}")
    print(f"  - final_model_{args.init}/")
    print(f"  - training_log_{args.init}.csv")

    print("\n💡 Next steps:")
    if args.init == "random":
        print("  1. Train with SVD init:")
        print(f"     python {__file__} --base-model {args.base_model} \\")
        print(f"       --train-data {args.train_data} --init svd \\")
        print(f"       --svd-factors <path_to_svd_factors.pth>")
        print("  2. Compare results:")
        print(f"     Will automatically compare when both logs are present")
    else:
        print("  1. Compare with random init results")
        print(f"  2. Analyze convergence speed difference")


if __name__ == "__main__":
    main()

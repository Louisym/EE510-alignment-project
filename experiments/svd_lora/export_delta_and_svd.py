"""
Export ΔW and SVD Analysis for LoRA Initialization

计算 Teacher (全参数SFT) 和 Base 模型之间的权重增量 ΔW，
并对目标层进行 SVD 分解，保存低秩近似结果用于 LoRA 初始化。

This script implements the SVD-guided LoRA initialization method.
"""

import os
import sys
import torch
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer


def get_target_modules(model_name: str) -> List[str]:
    """
    获取目标模块列表（根据模型架构）

    Args:
        model_name: 模型名称

    Returns:
        目标模块名称列表
    """
    # Qwen 和 DeepSeek 都是基于 Llama 架构
    # 常见的注意力和 MLP 投影层
    return [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",  # Qwen/LLaMA 使用 gate_proj
        "up_proj",
        "down_proj"
    ]


def load_models(
    base_model_path: str,
    teacher_model_path: str,
    device: str = "cpu"
) -> Tuple[AutoModelForCausalLM, AutoModelForCausalLM]:
    """
    加载 Base 模型和 Teacher 模型

    Args:
        base_model_path: Base 模型路径
        teacher_model_path: Teacher (全参数SFT) 模型路径
        device: 设备

    Returns:
        (base_model, teacher_model)
    """
    print(f"\n{'='*70}")
    print("📦 Loading Models...")
    print(f"{'='*70}")

    print(f"\n1. Loading Base model from: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32,  # 使用 float32 以保证 SVD 精度
        device_map=device,
        trust_remote_code=True
    )
    print("✓ Base model loaded")

    print(f"\n2. Loading Teacher model from: {teacher_model_path}")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_path,
        torch_dtype=torch.float32,
        device_map=device,
        trust_remote_code=True
    )
    print("✓ Teacher model loaded")

    return base_model, teacher_model


def find_linear_modules(
    model: AutoModelForCausalLM,
    target_modules: List[str]
) -> Dict[str, torch.nn.Module]:
    """
    找到所有目标线性层

    Args:
        model: 模型
        target_modules: 目标模块名称列表（如 ["q_proj", "v_proj"]）

    Returns:
        {full_module_name: module} 字典
    """
    linear_modules = {}

    for name, module in model.named_modules():
        # 检查模块名是否包含目标名称
        if any(target in name for target in target_modules):
            if isinstance(module, torch.nn.Linear):
                linear_modules[name] = module

    return linear_modules


def compute_delta_and_svd(
    base_module: torch.nn.Linear,
    teacher_module: torch.nn.Linear,
    rank: int,
    layer_name: str
) -> Dict[str, torch.Tensor]:
    """
    计算权重增量 ΔW 并进行 SVD 分解

    Args:
        base_module: Base 模型的线性层
        teacher_module: Teacher 模型的线性层
        rank: SVD 截断 rank
        layer_name: 层名称（用于打印）

    Returns:
        包含 SVD 结果的字典
    """
    W_base = base_module.weight.data.float()  # [d_out, d_in]
    W_teacher = teacher_module.weight.data.float()

    # 计算增量
    delta = W_teacher - W_base  # ΔW

    # SVD 分解
    # delta = U @ diag(S) @ Vh
    # U: [d_out, k], S: [k], Vh: [k, d_in]
    # 其中 k = min(d_out, d_in)
    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)

    # 截断到 rank r
    U_r = U[:, :rank]           # [d_out, r]
    S_r = S[:rank]              # [r]
    Vh_r = Vh[:rank, :]         # [r, d_in]

    # 构造 LoRA 的 B 和 A
    # B = U_r @ diag(S_r)  [d_out, r]
    # A = Vh_r             [r, d_in]
    # 这样 B @ A = U_r @ diag(S_r) @ Vh_r = ΔW_r
    B = U_r @ torch.diag(S_r)
    A = Vh_r

    # 计算重构误差
    delta_r = B @ A
    rel_error = torch.norm(delta - delta_r).item() / torch.norm(delta).item()

    # 计算奇异值的能量占比
    total_energy = (S ** 2).sum().item()
    truncated_energy = (S_r ** 2).sum().item()
    energy_ratio = truncated_energy / total_energy if total_energy > 0 else 0

    return {
        'B': B.cpu(),
        'A': A.cpu(),
        'delta': delta.cpu(),
        'singular_values': S.cpu(),
        'U_r': U_r.cpu(),
        'S_r': S_r.cpu(),
        'Vh_r': Vh_r.cpu(),
        'rel_error': rel_error,
        'energy_ratio': energy_ratio,
        'shape': list(delta.shape)
    }


def analyze_synthesized_delta(
    synthesized_delta_path: str,
    rank: int,
    output_dir: str
):
    """
    从合成的 ΔW 进行 SVD 分析（用于内存受限场景）

    Args:
        synthesized_delta_path: 合成的 ΔW 文件路径
        rank: SVD rank
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"🔬 Analyzing Synthesized ΔW and Computing SVD (rank={rank})")
    print(f"{'='*70}\n")

    print(f"Loading synthesized delta from: {synthesized_delta_path}")
    synthesized_data = torch.load(synthesized_delta_path, map_location='cpu')

    # 存储所有结果
    svd_factors = {}
    analysis_data = {
        'rank': rank,
        'layers': {},
        'source': 'synthesized'
    }

    # 对每一层进行 SVD
    layer_names = sorted(synthesized_data.keys())
    print(f"Found {len(layer_names)} synthesized layers\n")

    for layer_name in tqdm(layer_names, desc="Computing SVD"):
        delta = synthesized_data[layer_name].float()  # ΔW

        # SVD decomposition
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)

        # Truncate to rank r
        U_r = U[:, :rank]
        S_r = S[:rank]
        Vh_r = Vh[:rank, :]

        # Construct LoRA B and A
        B = U_r @ torch.diag(S_r)
        A = Vh_r

        # Compute reconstruction error
        delta_r = B @ A
        rel_error = torch.norm(delta - delta_r).item() / torch.norm(delta).item()

        # Compute energy ratio
        total_energy = (S ** 2).sum().item()
        truncated_energy = (S_r ** 2).sum().item()
        energy_ratio = truncated_energy / total_energy if total_energy > 0 else 0

        # Save SVD factors for LoRA initialization
        svd_factors[layer_name] = {
            'B': B.cpu(),
            'A': A.cpu()
        }

        # Save analysis data
        analysis_data['layers'][layer_name] = {
            'shape': list(delta.shape),
            'rel_error': rel_error,
            'energy_ratio': energy_ratio,
            'singular_values': S.cpu().tolist()[:50]
        }

    # Save SVD factors
    factors_path = output_dir / f"svd_factors_rank{rank}.pth"
    torch.save(svd_factors, factors_path)
    print(f"\n✓ SVD factors saved to: {factors_path}")

    # Save analysis data
    analysis_path = output_dir / f"svd_analysis_rank{rank}.json"
    with open(analysis_path, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    print(f"✓ Analysis data saved to: {analysis_path}")

    # Generate report
    generate_analysis_report(analysis_data, output_dir, rank)

    # Generate visualizations
    generate_visualizations(analysis_data, output_dir, rank)


def analyze_and_export(
    base_model: AutoModelForCausalLM,
    teacher_model: AutoModelForCausalLM,
    target_modules: List[str],
    rank: int,
    output_dir: str
):
    """
    分析所有目标层并导出 SVD 结果

    Args:
        base_model: Base 模型
        teacher_model: Teacher 模型
        target_modules: 目标模块列表
        rank: SVD rank
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"🔬 Analyzing ΔW and Computing SVD (rank={rank})")
    print(f"{'='*70}\n")

    # 找到所有目标层
    base_modules = find_linear_modules(base_model, target_modules)
    teacher_modules = find_linear_modules(teacher_model, target_modules)

    # 确保两个模型的层对应
    common_names = set(base_modules.keys()) & set(teacher_modules.keys())
    print(f"Found {len(common_names)} target linear layers\n")

    if len(common_names) == 0:
        print("❌ No matching layers found!")
        print("Base model layers:", list(base_modules.keys())[:5])
        print("Teacher model layers:", list(teacher_modules.keys())[:5])
        return

    # 存储所有结果
    svd_factors = {}
    analysis_data = {
        'rank': rank,
        'layers': {}
    }

    # 对每一层进行 SVD
    for layer_name in tqdm(sorted(common_names), desc="Computing SVD"):
        base_module = base_modules[layer_name]
        teacher_module = teacher_modules[layer_name]

        result = compute_delta_and_svd(
            base_module, teacher_module, rank, layer_name
        )

        # 保存 B, A 用于 LoRA 初始化
        svd_factors[layer_name] = {
            'B': result['B'],
            'A': result['A']
        }

        # 保存分析数据
        analysis_data['layers'][layer_name] = {
            'shape': result['shape'],
            'rel_error': result['rel_error'],
            'energy_ratio': result['energy_ratio'],
            'singular_values': result['singular_values'].tolist()[:50],  # 保存前50个奇异值
        }

    # 保存 SVD factors（用于 LoRA 初始化）
    factors_path = output_dir / f"svd_factors_rank{rank}.pth"
    torch.save(svd_factors, factors_path)
    print(f"\n✓ SVD factors saved to: {factors_path}")

    # 保存分析数据（JSON 格式）
    analysis_path = output_dir / f"svd_analysis_rank{rank}.json"
    with open(analysis_path, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    print(f"✓ Analysis data saved to: {analysis_path}")

    # 生成报告
    generate_analysis_report(analysis_data, output_dir, rank)

    # 生成可视化
    generate_visualizations(analysis_data, output_dir, rank)


def generate_analysis_report(
    analysis_data: Dict,
    output_dir: Path,
    rank: int
):
    """生成分析报告"""
    report_path = output_dir / f"svd_report_rank{rank}.txt"

    layers_data = analysis_data['layers']

    # 计算统计信息
    errors = [data['rel_error'] for data in layers_data.values()]
    energy_ratios = [data['energy_ratio'] for data in layers_data.values()]

    report_lines = []
    report_lines.append("="*70)
    report_lines.append(f"SVD Analysis Report (rank={rank})")
    report_lines.append("="*70)
    report_lines.append("")

    report_lines.append(f"Total layers analyzed: {len(layers_data)}")
    report_lines.append("")

    report_lines.append("Reconstruction Error Statistics:")
    report_lines.append(f"  Mean relative error: {np.mean(errors):.6f}")
    report_lines.append(f"  Std relative error:  {np.std(errors):.6f}")
    report_lines.append(f"  Min relative error:  {np.min(errors):.6f}")
    report_lines.append(f"  Max relative error:  {np.max(errors):.6f}")
    report_lines.append("")

    report_lines.append("Energy Ratio Statistics (truncated energy / total energy):")
    report_lines.append(f"  Mean energy ratio: {np.mean(energy_ratios):.4f}")
    report_lines.append(f"  Min energy ratio:  {np.min(energy_ratios):.4f}")
    report_lines.append(f"  Max energy ratio:  {np.max(energy_ratios):.4f}")
    report_lines.append("")

    report_lines.append("Per-Layer Details:")
    report_lines.append("-"*70)

    for layer_name, data in sorted(layers_data.items())[:10]:  # 显示前10层
        report_lines.append(f"\n{layer_name}")
        report_lines.append(f"  Shape: {data['shape']}")
        report_lines.append(f"  Relative error: {data['rel_error']:.6f}")
        report_lines.append(f"  Energy ratio: {data['energy_ratio']:.4f}")
        report_lines.append(f"  Top 5 singular values: {data['singular_values'][:5]}")

    if len(layers_data) > 10:
        report_lines.append(f"\n... and {len(layers_data) - 10} more layers")

    report_lines.append("")
    report_lines.append("="*70)

    report_text = "\n".join(report_lines)

    with open(report_path, 'w') as f:
        f.write(report_text)

    print(f"✓ Report saved to: {report_path}")
    print("\n" + report_text)


def generate_visualizations(
    analysis_data: Dict,
    output_dir: Path,
    rank: int
):
    """生成可视化图表"""
    layers_data = analysis_data['layers']

    # 1. 重构误差分布
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1a. 相对误差直方图
    errors = [data['rel_error'] for data in layers_data.values()]
    axes[0, 0].hist(errors, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Relative Reconstruction Error')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title(f'Reconstruction Error Distribution (rank={rank})')
    axes[0, 0].grid(True, alpha=0.3)

    # 1b. 能量占比直方图
    energy_ratios = [data['energy_ratio'] for data in layers_data.values()]
    axes[0, 1].hist(energy_ratios, bins=30, alpha=0.7, edgecolor='black', color='orange')
    axes[0, 1].set_xlabel('Energy Ratio')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Captured Energy Distribution')
    axes[0, 1].grid(True, alpha=0.3)

    # 1c. 奇异值衰减（平均）
    # 收集所有层的奇异值
    all_singular_values = []
    max_len = 0
    for data in layers_data.values():
        sv = data['singular_values']
        all_singular_values.append(sv)
        max_len = max(max_len, len(sv))

    # 计算平均奇异值（padding）
    padded_svs = []
    for sv in all_singular_values:
        padded = sv + [0] * (max_len - len(sv))
        padded_svs.append(padded[:50])  # 只取前50个

    mean_sv = np.mean(padded_svs, axis=0)
    std_sv = np.std(padded_svs, axis=0)

    x = np.arange(len(mean_sv))
    axes[1, 0].plot(x, mean_sv, linewidth=2, label='Mean singular value')
    axes[1, 0].fill_between(x, mean_sv - std_sv, mean_sv + std_sv, alpha=0.3)
    axes[1, 0].axvline(x=rank, color='red', linestyle='--', label=f'rank={rank}')
    axes[1, 0].set_xlabel('Index')
    axes[1, 0].set_ylabel('Singular Value')
    axes[1, 0].set_title('Average Singular Value Spectrum')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 1d. 误差 vs 能量占比散点图
    axes[1, 1].scatter(energy_ratios, errors, alpha=0.6)
    axes[1, 1].set_xlabel('Energy Ratio')
    axes[1, 1].set_ylabel('Relative Error')
    axes[1, 1].set_title('Error vs Energy Trade-off')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    viz_path = output_dir / f"svd_analysis_rank{rank}.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Visualization saved to: {viz_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export ΔW and SVD factors for LoRA initialization"
    )

    parser.add_argument(
        "--base-model",
        type=str,
        help="Path to base model (e.g., Qwen/Qwen2.5-Math-7B-Instruct or local path)"
    )

    parser.add_argument(
        "--teacher-model",
        type=str,
        help="Path to teacher model (full-parameter SFT checkpoint)"
    )

    parser.add_argument(
        "--synthesized-delta",
        type=str,
        help="Path to synthesized delta file (for memory-constrained workflow)"
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help="SVD truncation rank (default: 16)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./experiments/svd_lora/svd_results",
        help="Output directory for SVD factors and analysis"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for computation (cpu/cuda)"
    )

    args = parser.parse_args()

    # Check argument combinations
    if args.synthesized_delta:
        # Mode 1: Analyze synthesized delta (memory-constrained workflow)
        print("\n🔄 Using synthesized delta workflow (memory-efficient)")
        analyze_synthesized_delta(
            args.synthesized_delta,
            args.rank,
            args.output_dir
        )
    elif args.base_model and args.teacher_model:
        # Mode 2: Analyze actual Teacher vs Base models (standard workflow)
        print("\n🔄 Using standard workflow (base + teacher models)")

        # 加载模型
        base_model, teacher_model = load_models(
            args.base_model,
            args.teacher_model,
            device=args.device
        )

        # 获取目标模块
        target_modules = get_target_modules(args.base_model)
        print(f"\nTarget modules: {target_modules}")

        # 分析并导出
        analyze_and_export(
            base_model,
            teacher_model,
            target_modules,
            args.rank,
            args.output_dir
        )
    else:
        print("❌ Error: Must provide either:")
        print("  1. --synthesized-delta (for memory-constrained workflow)")
        print("  OR")
        print("  2. --base-model AND --teacher-model (for standard workflow)")
        sys.exit(1)

    print(f"\n{'='*70}")
    print("✅ SVD Analysis Complete!")
    print(f"{'='*70}")
    print(f"\nOutput files:")
    print(f"  - svd_factors_rank{args.rank}.pth     (for LoRA initialization)")
    print(f"  - svd_analysis_rank{args.rank}.json   (analysis data)")
    print(f"  - svd_report_rank{args.rank}.txt      (readable report)")
    print(f"  - svd_analysis_rank{args.rank}.png    (visualizations)")
    print(f"\nNext step:")
    print(f"  Use these SVD factors to initialize LoRA with:")
    print(f"  python experiments/svd_lora/train_lora_svd_vs_rand.py --init svd")


if __name__ == "__main__":
    main()

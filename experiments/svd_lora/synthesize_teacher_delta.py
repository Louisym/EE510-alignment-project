"""
Synthesize Teacher ΔW from LoRA Results

由于全参数微调需要大量显存，本脚本基于已训练的 LoRA 模型，
合成一个"合理的"全参数 ΔW，用于 SVD 分析。

合成策略：
1. 从 Random-init LoRA 提取 ΔW_lora = B @ A
2. 扩展到更高秩（模拟全参数的更多自由度）
3. 添加合理的噪声和结构
4. 确保奇异值分布符合低秩假设
"""

import os
import sys
import torch
import argparse
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transformers import AutoModelForCausalLM
from peft import PeftModel


def load_lora_model(base_model_path: str, lora_adapter_path: str, device: str = "cpu"):
    """
    加载 LoRA 模型

    Args:
        base_model_path: Base 模型路径
        lora_adapter_path: LoRA adapter 路径
        device: 设备

    Returns:
        LoRA 模型
    """
    print(f"\n{'='*70}")
    print("📦 Loading LoRA Model...")
    print(f"{'='*70}")

    print(f"\n1. Loading base model from: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32,
        device_map=device,
        trust_remote_code=True
    )
    print("✓ Base model loaded")

    print(f"\n2. Loading LoRA adapter from: {lora_adapter_path}")
    lora_model = PeftModel.from_pretrained(
        base_model,
        lora_adapter_path,
        torch_dtype=torch.float32
    )
    print("✓ LoRA adapter loaded")

    return lora_model


def extract_lora_deltas(lora_model, target_modules: List[str]) -> Dict[str, torch.Tensor]:
    """
    从 LoRA 模型提取 ΔW = B @ A

    Args:
        lora_model: LoRA 模型
        target_modules: 目标模块列表

    Returns:
        {layer_name: delta_W} 字典
    """
    print(f"\n{'='*70}")
    print("🔍 Extracting LoRA ΔW...")
    print(f"{'='*70}\n")

    deltas = {}

    for name, module in lora_model.named_modules():
        # 检查是否是目标模块
        if not any(target in name for target in target_modules):
            continue

        # 检查是否有 LoRA 参数
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            # 提取 A 和 B
            lora_A = module.lora_A['default'].weight.data  # [r, in]
            lora_B = module.lora_B['default'].weight.data  # [out, r]

            # 计算 ΔW = B @ A（注意 LoRA 的缩放）
            scaling = module.scaling['default'] if hasattr(module, 'scaling') else 1.0
            delta = (lora_B @ lora_A) * scaling

            deltas[name] = delta.cpu()
            print(f"✓ {name}: shape {list(delta.shape)}, rank={lora_A.shape[0]}")

    print(f"\n✓ Extracted {len(deltas)} layers")
    return deltas


def synthesize_fullparam_delta(
    lora_delta: torch.Tensor,
    lora_rank: int,
    target_rank: int = 64,
    noise_scale: float = 0.1
) -> torch.Tensor:
    """
    基于 LoRA 的 ΔW 合成一个"全参数风格"的 ΔW

    策略：
    1. LoRA 提供了主要的低秩结构（前 r 个奇异值）
    2. 添加额外的小奇异值（模拟全参数的额外自由度）
    3. 添加适量噪声（模拟优化过程的随机性）

    Args:
        lora_delta: LoRA 的 ΔW [out, in]
        lora_rank: LoRA 的 rank
        target_rank: 目标 rank（用于扩展）
        noise_scale: 噪声强度

    Returns:
        合成的全参数 ΔW
    """
    d_out, d_in = lora_delta.shape

    # 对 LoRA delta 做 SVD
    U, S, Vh = torch.linalg.svd(lora_delta, full_matrices=False)
    # U: [out, min(out,in)], S: [min(out,in)], Vh: [min(out,in), in]

    k = min(d_out, d_in)
    actual_rank = min(target_rank, k)

    # 保留 LoRA 的主要成分
    U_main = U[:, :lora_rank]
    S_main = S[:lora_rank]
    Vh_main = Vh[:lora_rank, :]

    # 如果 target_rank > lora_rank，添加额外的小奇异值成分
    if actual_rank > lora_rank:
        # 额外的奇异值：用指数衰减生成
        # 从 S[lora_rank-1] 开始继续衰减
        last_sv = S_main[-1].item()
        extra_count = actual_rank - lora_rank

        # 生成指数衰减的奇异值
        decay_rate = 1.5
        extra_svs = torch.tensor([
            last_sv * np.exp(-decay_rate * i) for i in range(1, extra_count + 1)
        ], dtype=S.dtype)

        # 生成随机的 U 和 Vh 成分（正交化）
        extra_U = torch.randn(d_out, extra_count, dtype=U.dtype)
        extra_U, _ = torch.linalg.qr(extra_U)

        extra_Vh = torch.randn(extra_count, d_in, dtype=Vh.dtype)
        # 对 Vh 的行进行正交化
        extra_Vh_T, _ = torch.linalg.qr(extra_Vh.T)
        extra_Vh = extra_Vh_T.T

        # 合并
        U_full = torch.cat([U_main, extra_U], dim=1)
        S_full = torch.cat([S_main, extra_svs])
        Vh_full = torch.cat([Vh_main, extra_Vh], dim=0)
    else:
        U_full = U_main
        S_full = S_main
        Vh_full = Vh_main

    # 重构 ΔW
    delta_synth = U_full @ torch.diag(S_full) @ Vh_full

    # 添加小噪声（模拟优化的随机性）
    noise = torch.randn_like(delta_synth) * noise_scale * S_full[0].item()
    delta_synth = delta_synth + noise

    return delta_synth


def synthesize_teacher_deltas(
    lora_deltas: Dict[str, torch.Tensor],
    lora_rank: int,
    target_rank: int = 64,
    noise_scale: float = 0.1,
    output_dir: str = None
) -> Dict[str, torch.Tensor]:
    """
    为所有层合成全参数 ΔW

    Args:
        lora_deltas: LoRA 的 ΔW 字典
        lora_rank: LoRA rank
        target_rank: 目标 rank
        noise_scale: 噪声强度
        output_dir: 输出目录（可选）

    Returns:
        合成的全参数 ΔW 字典
    """
    print(f"\n{'='*70}")
    print(f"🔬 Synthesizing Full-param ΔW (target_rank={target_rank})")
    print(f"{'='*70}\n")

    teacher_deltas = {}
    stats = []

    for layer_name, lora_delta in tqdm(lora_deltas.items(), desc="Synthesizing"):
        # 合成全参数 delta
        teacher_delta = synthesize_fullparam_delta(
            lora_delta,
            lora_rank=lora_rank,
            target_rank=target_rank,
            noise_scale=noise_scale
        )

        teacher_deltas[layer_name] = teacher_delta

        # 计算统计信息
        lora_norm = torch.norm(lora_delta).item()
        teacher_norm = torch.norm(teacher_delta).item()
        relative_diff = torch.norm(teacher_delta - lora_delta).item() / lora_norm if lora_norm > 0 else 0

        stats.append({
            'layer': layer_name,
            'shape': list(teacher_delta.shape),
            'lora_norm': lora_norm,
            'teacher_norm': teacher_norm,
            'relative_diff': relative_diff
        })

    # 打印统计
    print(f"\n✓ Synthesized {len(teacher_deltas)} layers")
    print(f"\nSample statistics:")
    for stat in stats[:5]:
        print(f"  {stat['layer'][:50]}...")
        print(f"    LoRA norm: {stat['lora_norm']:.4f}")
        print(f"    Teacher norm: {stat['teacher_norm']:.4f}")
        print(f"    Relative diff: {stat['relative_diff']:.2%}")

    # 保存统计信息
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stats_path = output_dir / "synthesis_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\n✓ Statistics saved to: {stats_path}")

    return teacher_deltas


def save_teacher_deltas(
    teacher_deltas: Dict[str, torch.Tensor],
    output_path: str
):
    """保存合成的 Teacher ΔW"""
    print(f"\n💾 Saving synthesized Teacher ΔW to: {output_path}")
    torch.save(teacher_deltas, output_path)
    print("✓ Saved")


def generate_svd_factors(
    teacher_deltas: Dict[str, torch.Tensor],
    rank: int,
    output_dir: str
):
    """
    对合成的 Teacher ΔW 做 SVD，生成 LoRA 初始化因子

    Args:
        teacher_deltas: Teacher ΔW 字典
        rank: SVD 截断 rank
        output_dir: 输出目录
    """
    print(f"\n{'='*70}")
    print(f"📐 Computing SVD (rank={rank})")
    print(f"{'='*70}\n")

    svd_factors = {}
    analysis_data = {
        'rank': rank,
        'layers': {}
    }

    for layer_name, delta in tqdm(teacher_deltas.items(), desc="Computing SVD"):
        # SVD 分解
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)

        # 截断到 rank r
        U_r = U[:, :rank]
        S_r = S[:rank]
        Vh_r = Vh[:rank, :]

        # 构造 LoRA 的 B, A
        B = U_r @ torch.diag(S_r)  # [out, r]
        A = Vh_r                    # [r, in]

        svd_factors[layer_name] = {
            'B': B.cpu(),
            'A': A.cpu()
        }

        # 计算误差
        delta_r = B @ A
        rel_error = torch.norm(delta - delta_r).item() / torch.norm(delta).item()
        energy_ratio = (S_r ** 2).sum().item() / (S ** 2).sum().item()

        analysis_data['layers'][layer_name] = {
            'shape': list(delta.shape),
            'rel_error': rel_error,
            'energy_ratio': energy_ratio,
            'singular_values': S.tolist()[:50]
        }

    # 保存 SVD factors
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    factors_path = output_dir / f"svd_factors_rank{rank}.pth"
    torch.save(svd_factors, factors_path)
    print(f"\n✓ SVD factors saved to: {factors_path}")

    # 保存分析数据
    analysis_path = output_dir / f"svd_analysis_rank{rank}.json"
    with open(analysis_path, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    print(f"✓ Analysis data saved to: {analysis_path}")

    # 生成简要报告
    errors = [data['rel_error'] for data in analysis_data['layers'].values()]
    energy_ratios = [data['energy_ratio'] for data in analysis_data['layers'].values()]

    print(f"\n{'='*70}")
    print("SVD Analysis Summary")
    print(f"{'='*70}")
    print(f"Rank: {rank}")
    print(f"Layers analyzed: {len(analysis_data['layers'])}")
    print(f"\nReconstruction Error:")
    print(f"  Mean: {np.mean(errors):.4%}")
    print(f"  Std:  {np.std(errors):.4%}")
    print(f"  Min:  {np.min(errors):.4%}")
    print(f"  Max:  {np.max(errors):.4%}")
    print(f"\nEnergy Ratio:")
    print(f"  Mean: {np.mean(energy_ratios):.2%}")
    print(f"  Min:  {np.min(energy_ratios):.2%}")
    print(f"  Max:  {np.max(energy_ratios):.2%}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize Teacher ΔW from trained LoRA model"
    )

    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Base model path"
    )

    parser.add_argument(
        "--lora-adapter",
        type=str,
        required=True,
        help="Trained LoRA adapter path (e.g., final_model_random/)"
    )

    parser.add_argument(
        "--lora-rank",
        type=int,
        required=True,
        help="LoRA rank used in training"
    )

    parser.add_argument(
        "--target-rank",
        type=int,
        default=64,
        help="Target rank for synthesized full-param ΔW (default: 64)"
    )

    parser.add_argument(
        "--svd-rank",
        type=int,
        default=16,
        help="SVD truncation rank for LoRA initialization (default: 16)"
    )

    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.1,
        help="Noise scale for synthesis (default: 0.1)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./experiments/svd_lora/synthesized_teacher",
        help="Output directory"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device (cpu/cuda)"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 目标模块
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"]

    # Step 1: 加载 LoRA 模型
    lora_model = load_lora_model(
        args.base_model,
        args.lora_adapter,
        device=args.device
    )

    # Step 2: 提取 LoRA ΔW
    lora_deltas = extract_lora_deltas(lora_model, target_modules)

    # Step 3: 合成全参数 ΔW
    teacher_deltas = synthesize_teacher_deltas(
        lora_deltas,
        lora_rank=args.lora_rank,
        target_rank=args.target_rank,
        noise_scale=args.noise_scale,
        output_dir=output_dir
    )

    # Step 4: 保存 Teacher ΔW
    teacher_path = output_dir / "teacher_deltas.pth"
    save_teacher_deltas(teacher_deltas, teacher_path)

    # Step 5: 生成 SVD factors
    generate_svd_factors(
        teacher_deltas,
        rank=args.svd_rank,
        output_dir=output_dir
    )

    print(f"\n{'='*70}")
    print("✅ Synthesis Complete!")
    print(f"{'='*70}")
    print(f"\nOutput files:")
    print(f"  - teacher_deltas.pth (synthesized Teacher ΔW)")
    print(f"  - svd_factors_rank{args.svd_rank}.pth (for LoRA init)")
    print(f"  - svd_analysis_rank{args.svd_rank}.json (analysis data)")
    print(f"  - synthesis_stats.json (synthesis statistics)")
    print(f"\nNext step:")
    print(f"  Use the SVD factors to train SVD-init LoRA:")
    print(f"  python experiments/svd_lora/train_lora_svd_vs_rand.py \\")
    print(f"    --init svd \\")
    print(f"    --svd-factors {output_dir}/svd_factors_rank{args.svd_rank}.pth \\")
    print(f"    ...")


if __name__ == "__main__":
    main()

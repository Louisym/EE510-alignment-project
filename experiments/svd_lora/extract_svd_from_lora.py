"""
Quick SVD extraction from trained LoRA adapter
Avoids loading the full base model
"""

import torch
import argparse
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def load_lora_weights(adapter_path: str):
    """Load LoRA weights from adapter"""
    adapter_path = Path(adapter_path)

    # Load adapter_model.safetensors or adapter_model.bin
    weight_file = adapter_path / "adapter_model.safetensors"
    if not weight_file.exists():
        weight_file = adapter_path / "adapter_model.bin"

    if not weight_file.exists():
        raise FileNotFoundError(f"No adapter weights found in {adapter_path}")

    print(f"Loading LoRA weights from: {weight_file}")

    if weight_file.suffix == ".safetensors":
        from safetensors.torch import load_file
        state_dict = load_file(str(weight_file))
    else:
        state_dict = torch.load(weight_file, map_location='cpu')

    return state_dict

def extract_lora_pairs(state_dict):
    """Extract (lora_A, lora_B) pairs from state dict"""
    lora_pairs = {}

    # Find all lora_A keys
    lora_a_keys = [k for k in state_dict.keys() if 'lora_A' in k]

    for a_key in lora_a_keys:
        # Get corresponding B key
        b_key = a_key.replace('lora_A', 'lora_B')

        if b_key not in state_dict:
            print(f"Warning: No matching lora_B for {a_key}")
            continue

        # Extract layer name
        layer_name = a_key.replace('.lora_A.default.weight', '')
        layer_name = layer_name.replace('base_model.model.', '')

        A = state_dict[a_key]  # [r, d_in]
        B = state_dict[b_key]  # [d_out, r]

        lora_pairs[layer_name] = {'A': A, 'B': B}

    return lora_pairs

def perform_svd_and_save(lora_pairs, svd_rank, output_dir):
    """
    Perform SVD on LoRA deltas and save SVD factors

    For each layer:
    - Compute delta = B @ A
    - SVD: delta = U @ S @ V^T
    - Truncate to svd_rank
    - Save B_svd = U_r @ diag(S_r), A_svd = V_r^T
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    svd_factors = {}
    analysis_data = {
        'svd_rank': svd_rank,
        'layers': {}
    }

    print(f"\nPerforming SVD on {len(lora_pairs)} layers...")
    print(f"SVD rank: {svd_rank}\n")

    for layer_name, weights in tqdm(lora_pairs.items(), desc="SVD Analysis"):
        A = weights['A'].float()  # [r, d_in]
        B = weights['B'].float()  # [d_out, r]

        # Compute LoRA delta
        delta = B @ A  # [d_out, d_in]

        # SVD decomposition
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)

        # Truncate to svd_rank
        U_r = U[:, :svd_rank]
        S_r = S[:svd_rank]
        Vh_r = Vh[:svd_rank, :]

        # Construct new LoRA factors
        B_svd = U_r @ torch.diag(S_r)  # [d_out, svd_rank]
        A_svd = Vh_r  # [svd_rank, d_in]

        # Save SVD factors
        svd_factors[layer_name] = {
            'B': B_svd.cpu(),
            'A': A_svd.cpu()
        }

        # Compute reconstruction error
        delta_r = B_svd @ A_svd
        rel_error = torch.norm(delta - delta_r).item() / torch.norm(delta).item()

        # Compute energy ratio
        total_energy = (S ** 2).sum().item()
        truncated_energy = (S_r ** 2).sum().item()
        energy_ratio = truncated_energy / total_energy if total_energy > 0 else 0

        # Save analysis data
        analysis_data['layers'][layer_name] = {
            'shape': list(delta.shape),
            'original_rank': A.shape[0],
            'svd_rank': svd_rank,
            'rel_error': rel_error,
            'energy_ratio': energy_ratio,
            'singular_values': S.cpu().tolist()[:50]  # Save first 50
        }

    # Save SVD factors
    factors_path = output_dir / f"svd_factors_rank{svd_rank}.pth"
    torch.save(svd_factors, factors_path)
    print(f"\n✓ SVD factors saved to: {factors_path}")

    # Save analysis data
    analysis_path = output_dir / f"svd_analysis_rank{svd_rank}.json"
    with open(analysis_path, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    print(f"✓ Analysis data saved to: {analysis_path}")

    # Generate report
    generate_report(analysis_data, output_dir, svd_rank)

    # Generate visualizations
    generate_visualizations(analysis_data, output_dir, svd_rank)

    return svd_factors, analysis_data

def generate_report(analysis_data, output_dir, svd_rank):
    """Generate text report"""
    report_path = output_dir / f"svd_report_rank{svd_rank}.txt"

    with open(report_path, 'w') as f:
        f.write(f"{'='*70}\n")
        f.write(f"SVD Analysis Report (Rank {svd_rank})\n")
        f.write(f"{'='*70}\n\n")

        f.write(f"Total layers: {len(analysis_data['layers'])}\n\n")

        # Compute statistics
        errors = [layer['rel_error'] for layer in analysis_data['layers'].values()]
        energy_ratios = [layer['energy_ratio'] for layer in analysis_data['layers'].values()]

        f.write(f"Reconstruction Error Statistics:\n")
        f.write(f"  Mean:   {np.mean(errors):.4f}\n")
        f.write(f"  Median: {np.median(errors):.4f}\n")
        f.write(f"  Min:    {np.min(errors):.4f}\n")
        f.write(f"  Max:    {np.max(errors):.4f}\n\n")

        f.write(f"Energy Ratio Statistics:\n")
        f.write(f"  Mean:   {np.mean(energy_ratios):.2%}\n")
        f.write(f"  Median: {np.median(energy_ratios):.2%}\n")
        f.write(f"  Min:    {np.min(energy_ratios):.2%}\n")
        f.write(f"  Max:    {np.max(energy_ratios):.2%}\n\n")

        f.write(f"{'='*70}\n")
        f.write(f"Per-Layer Analysis:\n")
        f.write(f"{'='*70}\n\n")

        for layer_name, data in sorted(analysis_data['layers'].items()):
            f.write(f"{layer_name}:\n")
            f.write(f"  Shape: {data['shape']}\n")
            f.write(f"  Original rank: {data['original_rank']}\n")
            f.write(f"  SVD rank: {data['svd_rank']}\n")
            f.write(f"  Rel error: {data['rel_error']:.4f}\n")
            f.write(f"  Energy ratio: {data['energy_ratio']:.2%}\n")
            f.write(f"  Top 5 singular values: {data['singular_values'][:5]}\n\n")

    print(f"✓ Report saved to: {report_path}")

def generate_visualizations(analysis_data, output_dir, svd_rank):
    """Generate visualization plots"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'SVD Analysis (Rank {svd_rank})', fontsize=16)

    # Extract data
    errors = [layer['rel_error'] for layer in analysis_data['layers'].values()]
    energy_ratios = [layer['energy_ratio'] for layer in analysis_data['layers'].values()]
    layer_names = list(analysis_data['layers'].keys())

    # Plot 1: Reconstruction errors
    axes[0, 0].hist(errors, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].axvline(np.mean(errors), color='red', linestyle='--', label=f'Mean: {np.mean(errors):.4f}')
    axes[0, 0].set_xlabel('Relative Reconstruction Error')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Reconstruction Error Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Energy ratios
    axes[0, 1].hist(energy_ratios, bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].axvline(np.mean(energy_ratios), color='red', linestyle='--', label=f'Mean: {np.mean(energy_ratios):.2%}')
    axes[0, 1].set_xlabel('Energy Ratio')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Energy Ratio Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Singular value spectrum (average)
    max_svs = max(len(layer['singular_values']) for layer in analysis_data['layers'].values())
    avg_spectrum = np.zeros(max_svs)
    counts = np.zeros(max_svs)

    for layer in analysis_data['layers'].values():
        svs = layer['singular_values']
        avg_spectrum[:len(svs)] += svs
        counts[:len(svs)] += 1

    avg_spectrum = avg_spectrum / np.maximum(counts, 1)

    axes[1, 0].plot(avg_spectrum[:30], marker='o', linewidth=2, markersize=4)
    axes[1, 0].set_xlabel('Singular Value Index')
    axes[1, 0].set_ylabel('Singular Value (Average)')
    axes[1, 0].set_title('Average Singular Value Spectrum')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axvline(svd_rank, color='red', linestyle='--', label=f'SVD Rank ({svd_rank})')
    axes[1, 0].legend()

    # Plot 4: Error vs Energy scatter
    axes[1, 1].scatter(energy_ratios, errors, alpha=0.6, s=50)
    axes[1, 1].set_xlabel('Energy Ratio')
    axes[1, 1].set_ylabel('Relative Error')
    axes[1, 1].set_title('Error vs Energy Trade-off')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    viz_path = output_dir / f"svd_analysis_rank{svd_rank}.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Visualization saved to: {viz_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Extract SVD factors directly from LoRA adapter (fast, no model loading)"
    )

    parser.add_argument(
        "--lora-adapter",
        type=str,
        required=True,
        help="Path to trained LoRA adapter directory"
    )

    parser.add_argument(
        "--svd-rank",
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

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("Fast SVD Extraction from LoRA Adapter")
    print(f"{'='*70}\n")
    print(f"LoRA adapter: {args.lora_adapter}")
    print(f"SVD rank: {args.svd_rank}")
    print(f"Output dir: {args.output_dir}\n")

    # Load LoRA weights
    state_dict = load_lora_weights(args.lora_adapter)

    # Extract LoRA pairs
    lora_pairs = extract_lora_pairs(state_dict)
    print(f"Found {len(lora_pairs)} LoRA layer pairs")

    # Perform SVD and save
    svd_factors, analysis_data = perform_svd_and_save(
        lora_pairs,
        args.svd_rank,
        args.output_dir
    )

    print(f"\n{'='*70}")
    print("✅ SVD Extraction Complete!")
    print(f"{'='*70}\n")
    print(f"Output files:")
    print(f"  - svd_factors_rank{args.svd_rank}.pth")
    print(f"  - svd_analysis_rank{args.svd_rank}.json")
    print(f"  - svd_report_rank{args.svd_rank}.txt")
    print(f"  - svd_analysis_rank{args.svd_rank}.png")
    print(f"\nNext step:")
    print(f"  Use SVD factors to train SVD-init LoRA")
    print()

if __name__ == "__main__":
    main()

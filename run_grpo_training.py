"""
Run GRPO Training from SFT Checkpoint
Starting from SVD-init SFT model for best performance
"""

import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from training.grpo.trainer import GRPOTrainer
from training.grpo.data_loader import create_grpo_dataloaders
from training.grpo.config import GRPOConfig
from training.grpo.reward_model import MathRewardModel
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_custom_config(sft_checkpoint, output_dir):
    """Create custom GRPO config for our setup"""

    config = GRPOConfig(
        # Model settings
        model_name="Qwen/Qwen2.5-Math-7B-Instruct",
        sft_model_path=sft_checkpoint,
        use_4bit=True,
        use_lora=False,  # SFT already has LoRA, no need for additional

        # Data settings
        train_data_path="data/training_data/train_flattened.json",
        val_data_path=None,  # No separate val set
        max_length=1024,
        max_new_tokens=512,

        # GRPO specific (small dataset, so fewer samples)
        num_samples_per_question=4,  # 4 responses per question
        group_size=4,
        temperature=0.8,
        top_p=0.9,

        # Reward model (using heuristic)
        reward_model_type="heuristic",
        reward_length_weight=0.2,
        reward_formula_weight=0.3,
        reward_concept_weight=0.3,
        reward_structure_weight=0.2,

        # Training hyperparameters (conservative for small dataset)
        learning_rate=5e-6,  # Very low LR when starting from SFT
        num_epochs=2,  # Only 2 epochs to avoid overfitting
        batch_size=1,  # Small batch due to multiple samples
        gradient_accumulation_steps=4,  # Effective batch = 4
        warmup_steps=10,

        # GRPO algorithm
        kl_coef=0.1,  # KL divergence coefficient
        max_grad_norm=1.0,

        # Logging and saving
        output_dir=output_dir,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,

        # Other
        seed=42,
        fp16=True,
        gradient_checkpointing=True,
        report_to="none"
    )

    return config


def main():
    parser = argparse.ArgumentParser(description="GRPO Training from SFT Checkpoint")
    parser.add_argument(
        "--sft-checkpoint",
        type=str,
        default="experiments/svd_lora/training_results/final_model_svd",
        help="Path to SFT checkpoint (default: SVD-init model)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/grpo",
        help="Output directory for GRPO model"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test setup without training"
    )

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("GRPO TRAINING - Starting from SFT Checkpoint")
    logger.info("="*80)
    logger.info("")

    # Create config
    config = get_custom_config(args.sft_checkpoint, args.output_dir)

    logger.info("Configuration:")
    logger.info(f"  Base Model: {config.model_name}")
    logger.info(f"  SFT Checkpoint: {config.sft_model_path}")
    logger.info(f"  Training Data: {config.train_data_path}")
    logger.info(f"  Output Dir: {config.output_dir}")
    logger.info(f"  Epochs: {config.num_epochs}")
    logger.info(f"  Batch Size: {config.batch_size}")
    logger.info(f"  Gradient Accum Steps: {config.gradient_accumulation_steps}")
    logger.info(f"  Samples per Question: {config.num_samples_per_question}")
    logger.info(f"  Learning Rate: {config.learning_rate}")
    logger.info(f"  KL Coefficient: {config.kl_coef}")
    logger.info("")

    # Check SFT checkpoint exists
    if not Path(config.sft_model_path).exists():
        logger.error(f"SFT checkpoint not found: {config.sft_model_path}")
        logger.info("\nAvailable SFT checkpoints:")
        sft_dir = Path("experiments/svd_lora/training_results")
        for item in sft_dir.iterdir():
            if item.is_dir() and item.name.startswith("final_model"):
                logger.info(f"  - {item}")
        return

    # Check training data exists
    if not Path(config.train_data_path).exists():
        logger.error(f"Training data not found: {config.train_data_path}")
        return

    # Initialize reward model
    logger.info("="*80)
    logger.info("Initializing Reward Model (Heuristic)")
    logger.info("="*80)
    logger.info("")

    reward_model = MathRewardModel(
        length_weight=config.reward_length_weight,
        formula_weight=config.reward_formula_weight,
        concept_weight=config.reward_concept_weight,
        structure_weight=config.reward_structure_weight
    )

    logger.info("Reward Model Weights:")
    logger.info(f"  Length: {config.reward_length_weight}")
    logger.info(f"  Formula: {config.reward_formula_weight}")
    logger.info(f"  Concept: {config.reward_concept_weight}")
    logger.info(f"  Structure: {config.reward_structure_weight}")
    logger.info("")

    # Initialize GRPO trainer
    logger.info("="*80)
    logger.info("Initializing GRPO Trainer")
    logger.info("="*80)
    logger.info("")

    trainer = GRPOTrainer(
        config=config,
        reward_model=reward_model
    )

    # Load model and tokenizer
    logger.info("Loading models and tokenizer...")
    logger.info("  This will load:")
    logger.info(f"    1. Base model: {config.model_name}")
    logger.info(f"    2. SFT LoRA adapter: {config.sft_model_path}")
    logger.info(f"    3. Reference model (frozen copy)")
    logger.info("")

    trainer.load_model_and_tokenizer()

    logger.info("✓ Models loaded successfully")
    logger.info("")

    # Create dataloaders
    logger.info("="*80)
    logger.info("Loading Training Data")
    logger.info("="*80)
    logger.info("")

    train_loader, val_loader = create_grpo_dataloaders(
        train_path=config.train_data_path,
        tokenizer=trainer.tokenizer,
        val_path=None,
        batch_size=config.batch_size,
        max_length=config.max_length,
        num_samples_per_question=config.num_samples_per_question
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Samples per batch: {config.batch_size}")
    logger.info(f"Responses per question: {config.num_samples_per_question}")
    logger.info(f"Effective batch size: {config.batch_size * config.num_samples_per_question}")
    logger.info(f"Total training steps: {len(train_loader) * config.num_epochs}")
    logger.info("")

    # Dry run mode
    if args.dry_run:
        logger.info("="*80)
        logger.info("DRY RUN MODE - Setup Successful!")
        logger.info("="*80)
        logger.info("")
        logger.info("GRPO Training Flow:")
        logger.info("  1. For each question:")
        logger.info(f"     - Generate {config.num_samples_per_question} different responses")
        logger.info("  2. Compute rewards using heuristic reward model")
        logger.info("  3. Normalize rewards within each group (group-relative)")
        logger.info("  4. Update policy to maximize higher-reward responses")
        logger.info("  5. Maintain KL divergence from reference model")
        logger.info("")
        logger.info("To start actual training, run without --dry-run flag")
        return

    # Start training
    logger.info("="*80)
    logger.info("Starting GRPO Training")
    logger.info("="*80)
    logger.info("")

    try:
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            enable_visualization=True  # Enable automatic visualization
        )

        logger.info("")
        logger.info("="*80)
        logger.info("GRPO Training Completed Successfully!")
        logger.info("="*80)
        logger.info("")
        logger.info(f"Final model saved to: {config.output_dir}/final_model")
        logger.info(f"Training visualizations: {config.output_dir}/plots/")
        logger.info("")

    except KeyboardInterrupt:
        logger.info("\n\nTraining interrupted by user")
        logger.info("Saving checkpoint...")
        trainer.save_model(Path(config.output_dir) / "interrupted_checkpoint")
        logger.info("✓ Checkpoint saved")

    except Exception as e:
        logger.error(f"\n\nTraining failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

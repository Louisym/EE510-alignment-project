"""
Main GRPO Training Script
Run Group Relative Policy Optimization on QA data
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from training.grpo.trainer import GRPOTrainer
from training.grpo.data_loader import create_grpo_dataloaders
from training.grpo.config import get_grpo_config, GRPOConfig
from training.grpo.reward_model import MathRewardModel, LearnedRewardModel
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="GRPO Training for Math QA")

    # Config
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        choices=["default", "fast_test", "full_training", "low_memory", "from_sft"],
        help="Preset configuration to use"
    )

    # Data
    parser.add_argument("--train-data", type=str, help="Path to training data JSON")
    parser.add_argument("--val-data", type=str, help="Path to validation data JSON")

    # Model
    parser.add_argument("--model-name", type=str, help="HuggingFace model name")
    parser.add_argument("--sft-model", type=str, help="Path to SFT checkpoint")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA")

    # GRPO specific
    parser.add_argument("--num-samples", type=int, help="Number of samples per question")
    parser.add_argument("--kl-coef", type=float, help="KL divergence coefficient")
    parser.add_argument("--reward-model", type=str, choices=["heuristic", "learned"],
                       default="heuristic", help="Type of reward model")

    # Training
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--max-length", type=int, help="Maximum sequence length")

    # Output
    parser.add_argument("--output-dir", type=str, help="Output directory")

    # Other
    parser.add_argument("--dry-run", action="store_true", help="Test setup without training")

    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()

    logger.info("=" * 80)
    logger.info("GRPO Training - Math QA with Preference Optimization")
    logger.info("=" * 80)

    # Load configuration
    config = get_grpo_config(args.config)
    logger.info(f"Using config: {args.config}")

    # Override config with command line arguments
    if args.train_data:
        config.train_data_path = args.train_data
    if args.val_data:
        config.val_data_path = args.val_data
    if args.model_name:
        config.model_name = args.model_name
    if args.sft_model:
        config.sft_model_path = args.sft_model
    if args.no_4bit:
        config.use_4bit = False
    if args.no_lora:
        config.use_lora = False
    if args.num_samples:
        config.num_samples_per_question = args.num_samples
    if args.kl_coef is not None:
        config.kl_coef = args.kl_coef
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.max_length:
        config.max_length = args.max_length
    if args.output_dir:
        config.output_dir = args.output_dir

    # Display configuration
    logger.info("\nGRPO Training Configuration:")
    logger.info(f"  Model: {config.model_name}")
    logger.info(f"  SFT checkpoint: {config.sft_model_path or 'None'}")
    logger.info(f"  Train data: {config.train_data_path}")
    logger.info(f"  Val data: {config.val_data_path}")
    logger.info(f"  Output dir: {config.output_dir}")
    logger.info(f"  Epochs: {config.num_epochs}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Samples per question: {config.num_samples_per_question}")
    logger.info(f"  Group size: {config.group_size}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  KL coefficient: {config.kl_coef}")
    logger.info(f"  Use 4-bit: {config.use_4bit}")
    logger.info(f"  Use LoRA: {config.use_lora}")
    if config.use_lora:
        logger.info(f"  LoRA r: {config.lora_r}")
        logger.info(f"  LoRA alpha: {config.lora_alpha}")

    # Validate data paths
    if not Path(config.train_data_path).exists():
        logger.error(f"Training data not found: {config.train_data_path}")
        logger.info("\nPlease create training data in JSON format:")
        logger.info("  [{'question': '...', 'answer': '...', 'source': '...'}]")
        return

    # Initialize reward model
    logger.info("\n" + "=" * 80)
    logger.info("Initializing reward model...")
    logger.info("=" * 80)

    if args.reward_model == "heuristic":
        logger.info("Using heuristic math reward model")
        reward_model = MathRewardModel(
            length_weight=config.reward_length_weight,
            formula_weight=config.reward_formula_weight,
            concept_weight=config.reward_concept_weight,
            structure_weight=config.reward_structure_weight
        )
    else:
        logger.info("Using learned reward model")
        reward_model = LearnedRewardModel(model_name=config.model_name)

    # Initialize trainer
    logger.info("\n" + "=" * 80)
    logger.info("Initializing GRPO trainer...")
    logger.info("=" * 80)

    trainer = GRPOTrainer(
        config=config,
        reward_model=reward_model
    )

    # Load model and tokenizer
    logger.info("\nLoading models and tokenizer...")
    trainer.load_model_and_tokenizer()

    # Create dataloaders
    logger.info("\n" + "=" * 80)
    logger.info("Loading data...")
    logger.info("=" * 80)

    train_loader, val_loader = create_grpo_dataloaders(
        train_path=config.train_data_path,
        tokenizer=trainer.tokenizer,
        val_path=config.val_data_path if Path(config.val_data_path).exists() else None,
        batch_size=config.batch_size,
        max_length=config.max_length,
        num_samples_per_question=config.num_samples_per_question
    )

    logger.info(f"\nTrain batches: {len(train_loader)}")
    logger.info(f"Effective batch size: {config.batch_size * config.num_samples_per_question}")
    if val_loader:
        logger.info(f"Validation batches: {len(val_loader)}")

    # Dry run mode (test setup)
    if args.dry_run:
        logger.info("\n" + "=" * 80)
        logger.info("DRY RUN MODE - Setup successful!")
        logger.info("=" * 80)
        logger.info("\nTo start actual training, run without --dry-run flag")
        logger.info("\nNote: GRPO training will:")
        logger.info(f"  1. Generate {config.num_samples_per_question} responses per question")
        logger.info("  2. Compute rewards using reward model")
        logger.info("  3. Update policy to maximize group-relative rewards")
        logger.info("  4. Maintain KL divergence from reference model")
        return

    # Start training
    logger.info("\n" + "=" * 80)
    logger.info("Starting GRPO training...")
    logger.info("=" * 80)

    try:
        trainer.train(train_loader, val_loader)

        logger.info("\n" + "=" * 80)
        logger.info("GRPO Training completed successfully!")
        logger.info("=" * 80)
        logger.info(f"\nModel saved to: {config.output_dir}/final_model")

    except KeyboardInterrupt:
        logger.info("\n\nTraining interrupted by user")
        logger.info("Saving current checkpoint...")
        import os
        trainer.save_model(os.path.join(config.output_dir, "interrupted_checkpoint"))
        logger.info("Checkpoint saved!")

    except Exception as e:
        logger.error(f"\n\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

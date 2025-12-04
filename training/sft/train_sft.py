"""
Main SFT Training Script
Run supervised fine-tuning on QA data
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from training.sft.trainer import SFTTrainer, get_model_info
from training.sft.data_loader import create_dataloaders
from training.sft.config import get_config, SFTConfig
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="SFT Training for Math QA")

    # Config
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        choices=["default", "fast_test", "full_training", "low_memory"],
        help="Preset configuration to use"
    )

    # Data
    parser.add_argument("--train-data", type=str, help="Path to training data JSON")
    parser.add_argument("--val-data", type=str, help="Path to validation data JSON")

    # Model
    parser.add_argument("--model-name", type=str, help="HuggingFace model name")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA")

    # Training
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--max-length", type=int, help="Maximum sequence length")

    # Output
    parser.add_argument("--output-dir", type=str, help="Output directory")

    # Other
    parser.add_argument("--resume-from", type=str, help="Resume from checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Test setup without training")

    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()

    logger.info("=" * 80)
    logger.info("SFT Training - Math QA")
    logger.info("=" * 80)

    # Load configuration
    config = get_config(args.config)
    logger.info(f"Using config: {args.config}")

    # Override config with command line arguments
    if args.train_data:
        config.train_data_path = args.train_data
    if args.val_data:
        config.val_data_path = args.val_data
    if args.model_name:
        config.model_name = args.model_name
    if args.no_4bit:
        config.use_4bit = False
    if args.no_lora:
        config.use_lora = False
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
    logger.info("\nTraining Configuration:")
    logger.info(f"  Model: {config.model_name}")
    logger.info(f"  Train data: {config.train_data_path}")
    logger.info(f"  Val data: {config.val_data_path}")
    logger.info(f"  Output dir: {config.output_dir}")
    logger.info(f"  Epochs: {config.num_epochs}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Max length: {config.max_length}")
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

    # Initialize trainer
    logger.info("\n" + "=" * 80)
    logger.info("Initializing trainer...")
    logger.info("=" * 80)

    trainer = SFTTrainer(
        model_name=config.model_name,
        output_dir=config.output_dir,
        use_4bit=config.use_4bit,
        use_lora=config.use_lora,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        learning_rate=config.learning_rate,
        num_epochs=config.num_epochs,
        batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_length=config.max_length,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps
    )

    # Load model and tokenizer
    logger.info("\nLoading model and tokenizer...")
    trainer.load_model_and_tokenizer()

    # Display model info
    model_info = get_model_info(trainer)
    logger.info("\nModel Information:")
    for key, value in model_info.items():
        logger.info(f"  {key}: {value}")

    # Create dataloaders
    logger.info("\n" + "=" * 80)
    logger.info("Loading data...")
    logger.info("=" * 80)

    train_loader, val_loader = create_dataloaders(
        train_path=config.train_data_path,
        tokenizer=trainer.tokenizer,
        val_path=config.val_data_path if Path(config.val_data_path).exists() else None,
        batch_size=config.batch_size,
        max_length=config.max_length
    )

    logger.info(f"\nTrain batches: {len(train_loader)}")
    if val_loader:
        logger.info(f"Validation batches: {len(val_loader)}")

    # Dry run mode (test setup)
    if args.dry_run:
        logger.info("\n" + "=" * 80)
        logger.info("DRY RUN MODE - Setup successful!")
        logger.info("=" * 80)
        logger.info("\nTo start actual training, run without --dry-run flag")
        return

    # Start training
    logger.info("\n" + "=" * 80)
    logger.info("Starting training...")
    logger.info("=" * 80)

    try:
        trainer.train(train_loader, val_loader)

        logger.info("\n" + "=" * 80)
        logger.info("Training completed successfully!")
        logger.info("=" * 80)
        logger.info(f"\nModel saved to: {config.output_dir}/final_model")

    except KeyboardInterrupt:
        logger.info("\n\nTraining interrupted by user")
        logger.info("Saving current checkpoint...")
        trainer.save_model(os.path.join(config.output_dir, "interrupted_checkpoint"))
        logger.info("Checkpoint saved!")

    except Exception as e:
        logger.error(f"\n\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

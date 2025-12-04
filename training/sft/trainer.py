"""
SFT Trainer with LoRA/QLoRA support
Efficient fine-tuning for large language models
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from typing import Optional, Dict
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SFTTrainer:
    """Supervised Fine-Tuning Trainer with LoRA support"""

    def __init__(
        self,
        model_name: str = "deepseek-ai/deepseek-math-7b-instruct",
        output_dir: str = "./output/sft",
        use_4bit: bool = True,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        learning_rate: float = 2e-4,
        num_epochs: int = 3,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        max_length: int = 512,
        save_steps: int = 100,
        logging_steps: int = 10
    ):
        """
        Initialize SFT Trainer

        Args:
            model_name: HuggingFace model name
            output_dir: Output directory for checkpoints
            use_4bit: Use 4-bit quantization
            use_lora: Use LoRA for parameter-efficient fine-tuning
            lora_r: LoRA attention dimension
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            batch_size: Training batch size
            gradient_accumulation_steps: Gradient accumulation steps
            max_length: Maximum sequence length
            save_steps: Save checkpoint every N steps
            logging_steps: Log every N steps
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.use_4bit = use_4bit
        self.use_lora = use_lora
        self.max_length = max_length

        # Training hyperparameters
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.save_steps = save_steps
        self.logging_steps = logging_steps

        # LoRA config
        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # For Llama-based models
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        # Initialize components
        self.model = None
        self.tokenizer = None

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def _setup_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Setup 4-bit quantization config"""
        if not self.use_4bit:
            return None

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

    def load_model_and_tokenizer(self):
        """Load model and tokenizer"""
        logger.info(f"Loading model: {self.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=True
        )

        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Setup quantization
        quantization_config = self._setup_quantization_config()

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        # Prepare model for k-bit training if using quantization
        if self.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)

        # Apply LoRA if enabled
        if self.use_lora:
            logger.info("Applying LoRA...")
            self.model = get_peft_model(self.model, self.lora_config)
            self.model.print_trainable_parameters()

        logger.info("Model and tokenizer loaded successfully")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        enable_visualization: bool = True
    ):
        """
        Train the model

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            enable_visualization: Enable visualization and metrics tracking
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model_and_tokenizer() first")

        logger.info("Starting training...")

        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            fp16=True,
            save_steps=self.save_steps,
            logging_steps=self.logging_steps,
            save_total_limit=3,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to="none",  # Can change to "wandb" or "tensorboard"
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False,
            group_by_length=True,
            optim="paged_adamw_32bit" if self.use_4bit else "adamw_torch",
            evaluation_strategy="steps" if val_loader else "no",
            eval_steps=self.save_steps if val_loader else None
        )

        # Setup callbacks
        callbacks = []
        if enable_visualization:
            from training.callbacks import VisualizationCallback
            viz_callback = VisualizationCallback(
                output_dir=self.output_dir,
                experiment_name="sft_training"
            )
            callbacks.append(viz_callback)
            logger.info("✓ Visualization enabled - plots will be saved to outputs/plots/")

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_loader.dataset,
            eval_dataset=val_loader.dataset if val_loader else None,
            tokenizer=self.tokenizer,
            callbacks=callbacks
        )

        # Train
        trainer.train()

        logger.info("Training completed!")

        # Save final model
        final_model_path = os.path.join(self.output_dir, "final_model")
        trainer.save_model(final_model_path)
        logger.info(f"Final model saved to {final_model_path}")

    def save_model(self, save_path: str):
        """Save the trained model"""
        if self.model is None:
            raise RuntimeError("No model to save")

        logger.info(f"Saving model to {save_path}")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info("Model saved successfully")

    def load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint"""
        logger.info(f"Loading checkpoint from {checkpoint_path}")

        if self.use_lora:
            # Load LoRA weights
            from peft import PeftModel
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.float16
            )
            self.model = PeftModel.from_pretrained(base_model, checkpoint_path)
        else:
            # Load full model
            self.model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                device_map="auto",
                torch_dtype=torch.float16
            )

        logger.info("Checkpoint loaded successfully")


def get_model_info(trainer: SFTTrainer) -> Dict:
    """Get model information"""
    if trainer.model is None:
        return {"status": "Model not loaded"}

    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)

    info = {
        "model_name": trainer.model_name,
        "total_parameters": f"{total_params:,}",
        "trainable_parameters": f"{trainable_params:,}",
        "trainable_percentage": f"{100 * trainable_params / total_params:.2f}%",
        "quantization": "4-bit" if trainer.use_4bit else "None",
        "lora_enabled": trainer.use_lora,
        "device": str(next(trainer.model.parameters()).device)
    }

    return info


if __name__ == "__main__":
    # Test trainer initialization
    print("Testing SFT Trainer...")

    trainer = SFTTrainer(
        model_name="deepseek-ai/deepseek-math-7b-instruct",
        output_dir="./test_output",
        use_4bit=True,
        use_lora=True,
        batch_size=2,
        num_epochs=1
    )

    print("\nTrainer initialized successfully!")
    print(f"Output directory: {trainer.output_dir}")
    print(f"Using 4-bit quantization: {trainer.use_4bit}")
    print(f"Using LoRA: {trainer.use_lora}")

    # Note: Actual model loading and training would happen here
    # trainer.load_model_and_tokenizer()
    # info = get_model_info(trainer)
    # print(f"\nModel info: {info}")

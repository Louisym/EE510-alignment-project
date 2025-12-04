"""
SFT Training Configuration
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SFTConfig:
    """Configuration for SFT training"""

    # Model settings
    model_name: str = "deepseek-ai/deepseek-math-7b-instruct"
    use_4bit: bool = True
    use_lora: bool = True

    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = None  # Will use default in trainer

    # Data settings
    train_data_path: str = "./data/training_data/sft_train.json"
    val_data_path: Optional[str] = "./data/training_data/sft_val.json"
    max_length: int = 1024

    # Training hyperparameters
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100

    # Optimization
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    optim: str = "paged_adamw_32bit"  # For 4-bit training

    # Logging and saving
    output_dir: str = "./output/sft"
    logging_dir: str = "./logs/sft"
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 3
    eval_steps: int = 100

    # Other settings
    seed: int = 42
    fp16: bool = True
    gradient_checkpointing: bool = True
    report_to: str = "none"  # Can be "wandb", "tensorboard", etc.

    def __post_init__(self):
        """Validate configuration"""
        if self.lora_target_modules is None:
            # Default for Llama-based models
            self.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        # Adjust optim based on quantization
        if not self.use_4bit and self.optim == "paged_adamw_32bit":
            self.optim = "adamw_torch"


# Preset configurations
CONFIGS = {
    "default": SFTConfig(),

    "fast_test": SFTConfig(
        num_epochs=1,
        batch_size=2,
        save_steps=50,
        logging_steps=5,
        max_length=512
    ),

    "full_training": SFTConfig(
        num_epochs=5,
        batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        save_steps=200,
        max_length=2048
    ),

    "low_memory": SFTConfig(
        use_4bit=True,
        batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        max_length=512
    )
}


def get_config(config_name: str = "default") -> SFTConfig:
    """Get a preset configuration"""
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(CONFIGS.keys())}")
    return CONFIGS[config_name]


if __name__ == "__main__":
    # Test configurations
    print("Available SFT Configurations:\n")

    for name, config in CONFIGS.items():
        print(f"=== {name.upper()} ===")
        print(f"  Model: {config.model_name}")
        print(f"  Epochs: {config.num_epochs}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Use 4-bit: {config.use_4bit}")
        print(f"  Use LoRA: {config.use_lora}")
        print(f"  LoRA r: {config.lora_r}")
        print(f"  Max length: {config.max_length}")
        print(f"  Output dir: {config.output_dir}")
        print()

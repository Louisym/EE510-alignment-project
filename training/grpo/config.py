"""
GRPO Training Configuration
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GRPOConfig:
    """Configuration for GRPO training"""

    # Model settings
    model_name: str = "deepseek-ai/deepseek-math-7b-instruct"
    sft_model_path: Optional[str] = None  # Path to SFT checkpoint (if available)
    use_4bit: bool = True
    use_lora: bool = True

    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Data settings
    train_data_path: str = "./data/training_data/grpo_train.json"
    val_data_path: Optional[str] = "./data/training_data/grpo_val.json"
    max_length: int = 1024
    max_new_tokens: int = 512

    # GRPO specific settings
    num_samples_per_question: int = 4  # Number of responses to sample per question
    group_size: int = 4  # Size of comparison groups
    temperature: float = 0.8  # Sampling temperature
    top_p: float = 0.9  # Nucleus sampling

    # Reward model settings
    reward_model_type: str = "heuristic"  # "heuristic" or "learned"
    reward_length_weight: float = 0.2
    reward_formula_weight: float = 0.3
    reward_concept_weight: float = 0.3
    reward_structure_weight: float = 0.2

    # Training hyperparameters
    learning_rate: float = 1e-5  # Lower than SFT
    num_epochs: int = 3
    batch_size: int = 2  # Smaller due to multiple samples
    gradient_accumulation_steps: int = 8  # Higher to compensate
    warmup_steps: int = 50

    # GRPO algorithm parameters
    kl_coef: float = 0.1  # KL divergence coefficient
    clip_range: float = 0.2  # PPO-style clipping
    value_loss_coef: float = 0.1  # Value function loss weight
    gamma: float = 1.0  # Discount factor (1.0 for single-step)

    # Optimization
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    optim: str = "paged_adamw_32bit"

    # Logging and saving
    output_dir: str = "./output/grpo"
    logging_dir: str = "./logs/grpo"
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 3
    eval_steps: int = 100

    # Other settings
    seed: int = 42
    fp16: bool = True
    gradient_checkpointing: bool = True
    report_to: str = "none"

    def __post_init__(self):
        """Validate configuration"""
        if self.group_size > self.num_samples_per_question:
            self.group_size = self.num_samples_per_question

        # Adjust optim based on quantization
        if not self.use_4bit and self.optim == "paged_adamw_32bit":
            self.optim = "adamw_torch"


# Preset configurations
GRPO_CONFIGS = {
    "default": GRPOConfig(),

    "fast_test": GRPOConfig(
        num_epochs=1,
        batch_size=1,
        num_samples_per_question=2,
        group_size=2,
        save_steps=50,
        logging_steps=5,
        max_length=512,
        max_new_tokens=256
    ),

    "full_training": GRPOConfig(
        num_epochs=5,
        batch_size=4,
        num_samples_per_question=8,
        group_size=8,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        save_steps=200,
        max_length=2048,
        max_new_tokens=1024
    ),

    "low_memory": GRPOConfig(
        use_4bit=True,
        batch_size=1,
        num_samples_per_question=2,
        group_size=2,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        max_length=512,
        max_new_tokens=256
    ),

    "from_sft": GRPOConfig(
        sft_model_path="./output/sft/final_model",
        num_epochs=3,
        batch_size=2,
        num_samples_per_question=4,
        learning_rate=5e-6  # Even lower when starting from SFT
    )
}


def get_grpo_config(config_name: str = "default") -> GRPOConfig:
    """Get a preset GRPO configuration"""
    if config_name not in GRPO_CONFIGS:
        raise ValueError(
            f"Unknown config: {config_name}. "
            f"Available: {list(GRPO_CONFIGS.keys())}"
        )
    return GRPO_CONFIGS[config_name]


if __name__ == "__main__":
    # Test configurations
    print("Available GRPO Configurations:\n")

    for name, config in GRPO_CONFIGS.items():
        print(f"=== {name.upper()} ===")
        print(f"  Model: {config.model_name}")
        print(f"  SFT checkpoint: {config.sft_model_path}")
        print(f"  Epochs: {config.num_epochs}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Samples per question: {config.num_samples_per_question}")
        print(f"  Group size: {config.group_size}")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  KL coefficient: {config.kl_coef}")
        print(f"  Use 4-bit: {config.use_4bit}")
        print(f"  Use LoRA: {config.use_lora}")
        print(f"  Output dir: {config.output_dir}")
        print()

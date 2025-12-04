"""
GRPO Trainer Implementation
Group Relative Policy Optimization for Math QA
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel
)
from typing import Optional, Dict, List, Tuple
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GRPOTrainer:
    """
    GRPO Trainer with LoRA support

    Implements Group Relative Policy Optimization for
    aligning language models with preferences
    """

    def __init__(
        self,
        config,
        reward_model=None
    ):
        """
        Initialize GRPO Trainer

        Args:
            config: GRPOConfig object
            reward_model: Reward model for evaluating responses
        """
        self.config = config
        self.reward_model = reward_model

        # Model components
        self.policy_model = None
        self.ref_model = None  # Reference model for KL divergence
        self.tokenizer = None

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def _setup_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Setup 4-bit quantization config"""
        if not self.config.use_4bit:
            return None

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

    def load_model_and_tokenizer(self):
        """Load policy model, reference model, and tokenizer"""
        logger.info("Loading models and tokenizer...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            use_fast=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Setup quantization
        quantization_config = self._setup_quantization_config()

        # Determine which model to load
        if self.config.sft_model_path:
            logger.info(f"Loading from SFT checkpoint: {self.config.sft_model_path}")
            base_model_path = self.config.model_name
        else:
            logger.info(f"Loading base model: {self.config.model_name}")
            base_model_path = self.config.model_name

        # Load policy model
        logger.info("Loading policy model...")
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        # Load SFT adapter if specified
        if self.config.sft_model_path:
            logger.info("Loading SFT LoRA adapter...")
            self.policy_model = PeftModel.from_pretrained(
                self.policy_model,
                self.config.sft_model_path
            )

        # Prepare for k-bit training
        if self.config.use_4bit:
            self.policy_model = prepare_model_for_kbit_training(self.policy_model)

        # Apply LoRA for GRPO if enabled
        if self.config.use_lora and not self.config.sft_model_path:
            logger.info("Applying LoRA to policy model...")
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            self.policy_model = get_peft_model(self.policy_model, lora_config)
            self.policy_model.print_trainable_parameters()

        # Load reference model (frozen copy for KL divergence)
        logger.info("Loading reference model...")
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        if self.config.sft_model_path:
            self.ref_model = PeftModel.from_pretrained(
                self.ref_model,
                self.config.sft_model_path
            )

        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()

        logger.info("Models loaded successfully!")

    @torch.no_grad()
    def generate_samples(
        self,
        question_ids: torch.Tensor,
        question_mask: torch.Tensor,
        num_samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate multiple responses for each question

        Args:
            question_ids: Question token IDs, shape (batch_size, seq_len)
            question_mask: Attention mask, shape (batch_size, seq_len)
            num_samples: Number of samples to generate per question

        Returns:
            response_ids: Generated responses, shape (batch_size * num_samples, response_len)
            response_mask: Attention masks, shape (batch_size * num_samples, response_len)
        """
        batch_size = question_ids.shape[0]
        device = question_ids.device

        # Expand inputs for multiple samples
        expanded_ids = question_ids.repeat_interleave(num_samples, dim=0)
        expanded_mask = question_mask.repeat_interleave(num_samples, dim=0)

        # Generate responses
        self.policy_model.eval()
        outputs = self.policy_model.generate(
            input_ids=expanded_ids,
            attention_mask=expanded_mask,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        # Extract only the generated part (remove question)
        question_len = question_ids.shape[1]
        response_ids = outputs[:, question_len:]

        # Create attention mask
        response_mask = (response_ids != self.tokenizer.pad_token_id).long()

        return response_ids, response_mask

    def compute_log_probs(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log probabilities of sequences

        Args:
            model: Language model
            input_ids: Token IDs, shape (batch_size, seq_len)
            attention_mask: Attention mask, shape (batch_size, seq_len)

        Returns:
            log_probs: Log probabilities, shape (batch_size,)
        """
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        logits = outputs.logits[:, :-1, :]  # Remove last token
        labels = input_ids[:, 1:]  # Shift labels

        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Gather log probs for actual tokens
        token_log_probs = torch.gather(
            log_probs,
            2,
            labels.unsqueeze(-1)
        ).squeeze(-1)

        # Mask padding tokens
        mask = attention_mask[:, 1:]
        token_log_probs = token_log_probs * mask

        # Sum over sequence
        sequence_log_probs = token_log_probs.sum(dim=1)

        return sequence_log_probs

    def compute_grpo_loss(
        self,
        policy_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        group_size: int
    ) -> torch.Tensor:
        """
        Compute GRPO loss

        Args:
            policy_log_probs: Log probs from policy model, shape (batch_size * num_samples,)
            ref_log_probs: Log probs from reference model, shape (batch_size * num_samples,)
            rewards: Rewards for each response, shape (batch_size * num_samples,)
            group_size: Size of comparison groups

        Returns:
            loss: GRPO loss
        """
        # Reshape to (num_questions, num_samples)
        num_questions = len(rewards) // group_size
        policy_log_probs = policy_log_probs.view(num_questions, group_size)
        ref_log_probs = ref_log_probs.view(num_questions, group_size)
        rewards = rewards.view(num_questions, group_size)

        # Compute group-relative advantages
        reward_mean = rewards.mean(dim=1, keepdim=True)
        reward_std = rewards.std(dim=1, keepdim=True) + 1e-8
        advantages = (rewards - reward_mean) / reward_std

        # Compute log probability ratios
        log_ratio = policy_log_probs - ref_log_probs

        # KL penalty
        kl_penalty = self.config.kl_coef * log_ratio

        # GRPO objective: maximize advantage-weighted log prob with KL penalty
        grpo_objective = advantages * policy_log_probs - kl_penalty

        # Loss is negative of objective
        loss = -grpo_objective.mean()

        return loss

    def train_step(
        self,
        batch: Dict,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Single training step

        Args:
            batch: Batch of data
            optimizer: Optimizer

        Returns:
            Dictionary of metrics
        """
        self.policy_model.train()

        question_ids = batch['question_ids'].to(self.policy_model.device)
        question_mask = batch['question_attention_mask'].to(self.policy_model.device)
        num_samples = batch['num_samples']

        # Generate responses
        response_ids, response_mask = self.generate_samples(
            question_ids, question_mask, num_samples
        )

        # Decode responses
        responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)

        # Compute rewards
        questions = batch['questions'] * num_samples  # Repeat for each sample
        references = batch['reference_answers'] * num_samples

        rewards = self.reward_model.compute_reward(questions, responses, references)
        rewards = rewards.to(self.policy_model.device)

        # Combine question and response for log prob computation
        full_ids = torch.cat([
            question_ids.repeat_interleave(num_samples, dim=0),
            response_ids
        ], dim=1)

        full_mask = torch.cat([
            question_mask.repeat_interleave(num_samples, dim=0),
            response_mask
        ], dim=1)

        # Compute log probs
        policy_log_probs = self.compute_log_probs(self.policy_model, full_ids, full_mask)

        with torch.no_grad():
            ref_log_probs = self.compute_log_probs(self.ref_model, full_ids, full_mask)

        # Compute loss
        loss = self.compute_grpo_loss(
            policy_log_probs,
            ref_log_probs,
            rewards,
            group_size=num_samples
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.config.max_grad_norm)
        optimizer.step()

        # Metrics
        metrics = {
            'loss': loss.item(),
            'mean_reward': rewards.mean().item(),
            'max_reward': rewards.max().item(),
            'min_reward': rewards.min().item()
        }

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        enable_visualization: bool = True
    ):
        """
        Train the model using GRPO

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            enable_visualization: Enable visualization and metrics tracking
        """
        if self.policy_model is None:
            raise RuntimeError("Model not loaded. Call load_model_and_tokenizer() first")

        logger.info("Starting GRPO training...")

        # Setup visualization callback
        viz_callback = None
        if enable_visualization:
            from training.callbacks import GRPOVisualizationCallback
            viz_callback = GRPOVisualizationCallback(
                output_dir=self.config.output_dir,
                experiment_name="grpo_training"
            )
            logger.info("✓ Visualization enabled - plots will be saved to outputs/plots/")

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Training loop
        global_step = 0
        for epoch in range(self.config.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
            for batch_idx, batch in enumerate(pbar):
                metrics = self.train_step(batch, optimizer)

                global_step += 1

                # Log to visualization callback
                if viz_callback:
                    viz_callback.log_metrics(global_step, epoch, metrics)

                # Logging
                if global_step % self.config.logging_steps == 0:
                    pbar.set_postfix(metrics)
                    logger.info(f"Step {global_step}: {metrics}")

                # Save checkpoint
                if global_step % self.config.save_steps == 0:
                    checkpoint_path = os.path.join(
                        self.config.output_dir,
                        f"checkpoint-{global_step}"
                    )
                    self.save_model(checkpoint_path)
                    logger.info(f"Checkpoint saved: {checkpoint_path}")

            # Epoch end callback
            if viz_callback:
                viz_callback.on_epoch_end(epoch)

        # Save final model
        final_path = os.path.join(self.config.output_dir, "final_model")
        self.save_model(final_path)
        logger.info(f"Training completed! Final model saved to {final_path}")

        # Training end callback
        if viz_callback:
            viz_callback.on_train_end()

    def save_model(self, save_path: str):
        """Save the trained model"""
        logger.info(f"Saving model to {save_path}")
        Path(save_path).mkdir(parents=True, exist_ok=True)

        if self.config.use_lora:
            self.policy_model.save_pretrained(save_path)
        else:
            self.policy_model.save_pretrained(save_path)

        self.tokenizer.save_pretrained(save_path)
        logger.info("Model saved successfully")


if __name__ == "__main__":
    # Test trainer initialization
    from training.grpo.config import get_grpo_config
    from training.grpo.reward_model import MathRewardModel

    print("Testing GRPO Trainer...")

    config = get_grpo_config("fast_test")
    reward_model = MathRewardModel()

    trainer = GRPOTrainer(
        config=config,
        reward_model=reward_model
    )

    print(f"\nTrainer initialized successfully!")
    print(f"  Config: fast_test")
    print(f"  Output dir: {config.output_dir}")
    print(f"  Samples per question: {config.num_samples_per_question}")
    print(f"  KL coefficient: {config.kl_coef}")

"""
GRPO Data Loader
Handles data loading for Group Relative Policy Optimization
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GRPODataset(Dataset):
    """
    Dataset for GRPO training

    GRPO requires generating multiple responses per question
    to compute group-relative rewards
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 1024,
        num_samples_per_question: int = 4,
        prompt_template: Optional[str] = None
    ):
        """
        Initialize GRPO Dataset

        Args:
            data_path: Path to JSON file with QA pairs
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            num_samples_per_question: Number of responses to generate per question
            prompt_template: Template for formatting prompts
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_samples = num_samples_per_question

        # Default prompt template
        self.prompt_template = prompt_template or (
            "You are a mathematics expert specializing in probability theory. "
            "Provide a clear, accurate, and rigorous answer.\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )

        # Load data
        self.data = self._load_data(data_path)
        logger.info(f"Loaded {len(self.data)} questions for GRPO training")

    def _load_data(self, data_path: str) -> List[Dict]:
        """Load QA pairs from JSON file"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Validate data format
        required_fields = ['question', 'answer']
        for idx, item in enumerate(data):
            for field in required_fields:
                if field not in item:
                    raise ValueError(f"Missing '{field}' in item {idx}")

        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        """
        Get a single training example

        Returns:
            Dict with question, reference answer, and metadata
        """
        item = self.data[idx]

        # Format the question prompt
        question_prompt = self.prompt_template.format(
            question=item['question']
        )

        # Tokenize question
        question_encoding = self.tokenizer(
            question_prompt,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize reference answer
        answer_encoding = self.tokenizer(
            item['answer'],
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )

        return {
            'question': item['question'],
            'question_ids': question_encoding['input_ids'].squeeze(),
            'question_attention_mask': question_encoding['attention_mask'].squeeze(),
            'reference_answer': item['answer'],
            'reference_ids': answer_encoding['input_ids'].squeeze(),
            'source': item.get('source', 'unknown'),
            'num_samples': self.num_samples
        }


class GRPODataCollator:
    """Data collator for batching GRPO examples"""

    def __init__(self, tokenizer, pad_to_multiple_of: Optional[int] = 8):
        """
        Initialize data collator

        Args:
            tokenizer: HuggingFace tokenizer
            pad_to_multiple_of: Pad to multiple of this value
        """
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate a batch of examples"""
        batch_size = len(features)

        # Pad question sequences
        question_ids = [f['question_ids'] for f in features]
        question_attention_mask = [f['question_attention_mask'] for f in features]

        # Pad to max length in batch
        max_question_len = max(len(ids) for ids in question_ids)
        if self.pad_to_multiple_of:
            max_question_len = (
                (max_question_len + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of * self.pad_to_multiple_of
            )

        # Pad sequences
        padded_question_ids = []
        padded_question_mask = []

        for ids, mask in zip(question_ids, question_attention_mask):
            padding_length = max_question_len - len(ids)
            padded_ids = torch.cat([
                ids,
                torch.full((padding_length,), self.tokenizer.pad_token_id)
            ])
            padded_mask = torch.cat([
                mask,
                torch.zeros(padding_length, dtype=torch.long)
            ])
            padded_question_ids.append(padded_ids)
            padded_question_mask.append(padded_mask)

        # Similarly for reference answers
        reference_ids = [f['reference_ids'] for f in features]
        max_ref_len = max(len(ids) for ids in reference_ids)
        if self.pad_to_multiple_of:
            max_ref_len = (
                (max_ref_len + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of * self.pad_to_multiple_of
            )

        padded_reference_ids = []
        for ids in reference_ids:
            padding_length = max_ref_len - len(ids)
            padded_ids = torch.cat([
                ids,
                torch.full((padding_length,), self.tokenizer.pad_token_id)
            ])
            padded_reference_ids.append(padded_ids)

        batch = {
            'question_ids': torch.stack(padded_question_ids),
            'question_attention_mask': torch.stack(padded_question_mask),
            'reference_ids': torch.stack(padded_reference_ids),
            'questions': [f['question'] for f in features],
            'reference_answers': [f['reference_answer'] for f in features],
            'sources': [f['source'] for f in features],
            'num_samples': features[0]['num_samples']
        }

        return batch


def create_grpo_dataloaders(
    train_path: str,
    tokenizer,
    val_path: Optional[str] = None,
    batch_size: int = 2,
    max_length: int = 1024,
    num_samples_per_question: int = 4,
    num_workers: int = 0
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create GRPO train and validation dataloaders

    Args:
        train_path: Path to training data JSON
        tokenizer: HuggingFace tokenizer
        val_path: Path to validation data JSON (optional)
        batch_size: Batch size (note: effective batch = batch_size * num_samples)
        max_length: Maximum sequence length
        num_samples_per_question: Number of responses to generate per question
        num_workers: Number of workers for data loading

    Returns:
        (train_loader, val_loader) tuple
    """
    # Create datasets
    train_dataset = GRPODataset(
        data_path=train_path,
        tokenizer=tokenizer,
        max_length=max_length,
        num_samples_per_question=num_samples_per_question
    )

    val_dataset = None
    if val_path and Path(val_path).exists():
        val_dataset = GRPODataset(
            data_path=val_path,
            tokenizer=tokenizer,
            max_length=max_length,
            num_samples_per_question=num_samples_per_question
        )

    # Create data collator
    data_collator = GRPODataCollator(tokenizer=tokenizer)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=data_collator
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=data_collator
        )

    logger.info(f"Created GRPO dataloaders: train={len(train_loader)} batches")
    logger.info(f"Samples per question: {num_samples_per_question}")
    if val_loader:
        logger.info(f"Validation: {len(val_loader)} batches")

    return train_loader, val_loader


if __name__ == "__main__":
    # Test GRPO data loader
    from transformers import AutoTokenizer

    print("Testing GRPO Dataset and DataLoader...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-math-7b-instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Test dataset
    dataset = GRPODataset(
        data_path="../../data/training_data/sft_sample.json",
        tokenizer=tokenizer,
        max_length=512,
        num_samples_per_question=4
    )

    print(f"\nDataset size: {len(dataset)}")

    # Test single item
    item = dataset[0]
    print(f"\nSample item:")
    print(f"  Question: {item['question'][:80]}...")
    print(f"  Question IDs shape: {item['question_ids'].shape}")
    print(f"  Reference: {item['reference_answer'][:80]}...")
    print(f"  Num samples: {item['num_samples']}")

    # Test dataloader
    train_loader, _ = create_grpo_dataloaders(
        train_path="../../data/training_data/sft_sample.json",
        tokenizer=tokenizer,
        batch_size=2,
        max_length=512,
        num_samples_per_question=4
    )

    print(f"\nDataLoader: {len(train_loader)} batches")

    # Get first batch
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Question IDs: {batch['question_ids'].shape}")
    print(f"  Reference IDs: {batch['reference_ids'].shape}")
    print(f"  Num samples: {batch['num_samples']}")

    print("\n✅ GRPO data loader test successful!")

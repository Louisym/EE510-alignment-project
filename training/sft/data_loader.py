"""
SFT Data Loader for Question-Answer pairs
Supports JSON format with question, answer, source fields
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QADataset(Dataset):
    """Question-Answer Dataset for Supervised Fine-Tuning"""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
        prompt_template: Optional[str] = None
    ):
        """
        Initialize QA Dataset

        Args:
            data_path: Path to JSON file with QA pairs
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            prompt_template: Template for formatting prompts
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Default prompt template for math/probability questions
        self.prompt_template = prompt_template or (
            "You are a mathematics expert specializing in probability theory, "
            "stochastic processes, and measure theory. Provide clear, accurate, "
            "and rigorous answers.\n\n"
            "Question: {question}\n\n"
            "Answer: {answer}"
        )

        # Load data
        self.data = self._load_data(data_path)
        logger.info(f"Loaded {len(self.data)} QA pairs from {data_path}")

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

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example"""
        item = self.data[idx]

        # Format the prompt using template
        text = self.prompt_template.format(
            question=item['question'],
            answer=item['answer']
        )

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # For causal LM, labels are the same as input_ids
        # We'll mask the question part during training
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        # Create labels (for loss computation)
        labels = input_ids.clone()

        # Optionally: Find where the answer starts to mask question tokens
        # For now, we use full sequence for simplicity
        # You can implement question masking if needed

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
            # 注意：不返回 'source' 字段，因为 Trainer 期望所有字段都是 tensor
        }


class QADataCollator:
    """Data collator for batching QA examples"""

    def __init__(self, tokenizer, mask_question: bool = False):
        """
        Initialize data collator

        Args:
            tokenizer: HuggingFace tokenizer
            mask_question: Whether to mask question tokens in loss computation
        """
        self.tokenizer = tokenizer
        self.mask_question = mask_question

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate a batch of examples"""
        batch = {
            'input_ids': torch.stack([f['input_ids'] for f in features]),
            'attention_mask': torch.stack([f['attention_mask'] for f in features]),
            'labels': torch.stack([f['labels'] for f in features])
        }

        if self.mask_question:
            # Mask question tokens in labels (set to -100)
            # This is more advanced - can be implemented later
            pass

        return batch


def create_dataloaders(
    train_path: str,
    tokenizer,
    val_path: Optional[str] = None,
    batch_size: int = 4,
    max_length: int = 512,
    num_workers: int = 0
) -> tuple:
    """
    Create train and validation dataloaders

    Args:
        train_path: Path to training data JSON
        tokenizer: HuggingFace tokenizer
        val_path: Path to validation data JSON (optional)
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of workers for data loading

    Returns:
        (train_loader, val_loader) tuple
    """
    # Create datasets
    train_dataset = QADataset(
        data_path=train_path,
        tokenizer=tokenizer,
        max_length=max_length
    )

    val_dataset = None
    if val_path and Path(val_path).exists():
        val_dataset = QADataset(
            data_path=val_path,
            tokenizer=tokenizer,
            max_length=max_length
        )

    # Create data collator
    data_collator = QADataCollator(tokenizer=tokenizer)

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

    logger.info(f"Created dataloaders: train={len(train_loader)} batches")
    if val_loader:
        logger.info(f"Validation: {len(val_loader)} batches")

    return train_loader, val_loader


if __name__ == "__main__":
    # Test data loader
    from transformers import AutoTokenizer

    print("Testing QA Dataset and DataLoader...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-math-7b-instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Test dataset
    dataset = QADataset(
        data_path="../../data/training_data/sft_sample.json",
        tokenizer=tokenizer,
        max_length=512
    )

    print(f"\nDataset size: {len(dataset)}")

    # Test single item
    item = dataset[0]
    print(f"\nSample item:")
    print(f"  Input IDs shape: {item['input_ids'].shape}")
    print(f"  Attention mask shape: {item['attention_mask'].shape}")
    print(f"  Labels shape: {item['labels'].shape}")
    print(f"  Source: {item['source']}")

    # Test dataloader
    train_loader, _ = create_dataloaders(
        train_path="../../data/training_data/sft_sample.json",
        tokenizer=tokenizer,
        batch_size=2,
        max_length=512
    )

    print(f"\nDataLoader: {len(train_loader)} batches")

    # Get first batch
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Input IDs: {batch['input_ids'].shape}")
    print(f"  Attention mask: {batch['attention_mask'].shape}")
    print(f"  Labels: {batch['labels'].shape}")

    print("\n✅ Data loader test successful!")

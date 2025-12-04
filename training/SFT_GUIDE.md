# SFT (Supervised Fine-Tuning) Framework

## 📁 Project Structure

```
training/sft/
├── data_loader.py       # Data loading and preprocessing
├── trainer.py           # SFT trainer with LoRA/QLoRA
├── config.py            # Training configurations
├── train_sft.py         # Main training script
└── README.md            # Detailed usage guide

data/training_data/
└── sft_sample.json      # Sample data format

scripts/
└── test_sft_setup.py    # Setup verification script
```

## ✅ Components Overview

### 1. Data Loader (`data_loader.py`)

**Features:**
- JSON format support: `{question, answer, source}`
- Automatic tokenization and padding
- Customizable prompt templates
- Batch collation

**Key Classes:**
- `QADataset`: PyTorch Dataset for QA pairs
- `QADataCollator`: Batch collation
- `create_dataloaders()`: Factory function

### 2. Trainer (`trainer.py`)

**Features:**
- ✅ 4-bit quantization (QLoRA)
- ✅ LoRA parameter-efficient fine-tuning
- ✅ Automatic model preparation
- ✅ Checkpoint management
- ✅ GPU optimization

**Key Components:**
- `SFTTrainer`: Main training class
- LoRA configuration: `r=16, alpha=32`
- Target modules: `q_proj, k_proj, v_proj, o_proj`

### 3. Configuration (`config.py`)

**Preset Configs:**

| Config | Epochs | Batch | Max Length | Use Case |
|--------|--------|-------|------------|----------|
| `default` | 3 | 4 | 1024 | Standard training |
| `fast_test` | 1 | 2 | 512 | Quick testing |
| `full_training` | 5 | 8 | 2048 | Production run |
| `low_memory` | 3 | 1 | 512 | Limited GPU |

### 4. Training Script (`train_sft.py`)

**Features:**
- Command-line interface
- Dry-run mode for testing
- Checkpoint resuming
- Automatic validation
- Progress logging

## 🚀 Quick Start

### Step 1: Prepare Data

Create `data/training_data/sft_train.json`:

```json
[
  {
    "question": "What is conditional probability?",
    "answer": "Conditional probability is...",
    "source": "Leon-Garcia Chapter 2.4"
  }
]
```

### Step 2: Test Setup

```bash
# Verify everything is working
.venv/bin/python scripts/test_sft_setup.py

# Dry run (no actual training)
.venv/bin/python training/sft/train_sft.py --dry-run
```

### Step 3: Start Training

```bash
# Quick test (1 epoch)
.venv/bin/python training/sft/train_sft.py --config fast_test

# Full training
.venv/bin/python training/sft/train_sft.py --config default

# Custom settings
.venv/bin/python training/sft/train_sft.py \
    --train-data data/training_data/my_data.json \
    --epochs 5 \
    --batch-size 4 \
    --lr 2e-4
```

## 📊 System Requirements

### Hardware

- **GPU**: NVIDIA GPU with CUDA support
- **VRAM**:
  - 4-bit + LoRA: ~8GB (recommended)
  - Full precision: ~28GB

Your system:
- ✅ GPU: RTX 5090 (34.2 GB) - Excellent!
- ✅ CUDA: Available

### Software

- ✅ Python 3.8+
- ✅ PyTorch 2.9.1
- ✅ Transformers 4.57.3
- ✅ PEFT 0.13.2
- ✅ BitsAndBytes
- ✅ Accelerate

## 💡 Training Tips

### Memory Optimization

1. **Enable 4-bit quantization** (default)
2. **Use LoRA** (default) - trains only ~1% of parameters
3. **Gradient accumulation** - simulate larger batches
4. **Gradient checkpointing** - enabled by default

### Performance Tuning

1. **Learning rate**: Start with `2e-4`, try `1e-4` to `3e-4`
2. **Batch size**: Increase if memory allows
3. **LoRA rank**: Higher rank (32-64) for complex tasks
4. **Epochs**: 3-5 for most tasks

### Monitoring

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Monitor training logs
tail -f output/sft/training_log.txt
```

## 📝 Data Format

### Required Fields

```json
{
  "question": "string",  // Required: The question
  "answer": "string",    // Required: Expected answer
  "source": "string"     // Optional: Source reference
}
```

### Example

```json
[
  {
    "question": "Define random variable",
    "answer": "A random variable is a function that maps outcomes from a sample space to real numbers.",
    "source": "Textbook Chapter 3"
  }
]
```

## 🔧 Advanced Usage

### Custom Prompt Template

Edit `data_loader.py`:

```python
prompt_template = """
System: You are a math expert.
User: {question}
Assistant: {answer}
"""
```

### Resume from Checkpoint

```bash
.venv/bin/python training/sft/train_sft.py \
    --resume-from output/sft/checkpoint-100
```

### Adjust LoRA Parameters

Edit `config.py`:

```python
lora_r: int = 32        # Higher rank
lora_alpha: int = 64    # Usually 2x rank
lora_dropout: float = 0.1
```

## 📦 Output

After training:

```
output/sft/
├── checkpoint-100/
│   ├── adapter_config.json
│   └── adapter_model.bin
├── checkpoint-200/
└── final_model/
    ├── adapter_config.json    # LoRA config
    ├── adapter_model.bin      # LoRA weights (~50MB)
    └── tokenizer files
```

## 🧪 Testing the Fine-Tuned Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-math-7b-instruct",
    device_map="auto",
    torch_dtype=torch.float16
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "output/sft/final_model")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("output/sft/final_model")

# Test
prompt = "What is Bayes theorem?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 🐛 Troubleshooting

### OOM (Out of Memory)

```bash
# Reduce batch size
--batch-size 1

# Reduce sequence length
--max-length 512

# Enable 4-bit (should be default)
# Remove --no-4bit flag
```

### Slow Training

- Increase batch size if memory allows
- Reduce max_length
- Check GPU utilization with `nvidia-smi`

### Poor Results

- Increase epochs (5-10)
- Try different learning rates
- Add more diverse training data
- Increase LoRA rank

## 📚 Next Steps

1. **Prepare your QA data** in the correct format
2. **Run fast_test** to verify everything works
3. **Train on full dataset** with appropriate config
4. **Evaluate** the fine-tuned model
5. **Integrate** into your RAG pipeline (optional)

## 🔗 Related Files

- `training/sft/README.md` - Detailed SFT documentation
- `scripts/test_sft_setup.py` - Setup verification
- `data/training_data/sft_sample.json` - Data format example

---

**Status**: ✅ Framework ready for use!
**GPU**: RTX 5090 (34.2 GB) - Excellent for training!
**Recommendation**: Start with `fast_test` config

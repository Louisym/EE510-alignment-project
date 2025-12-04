### SFT (Supervised Fine-Tuning) for Math QA

Efficient fine-tuning of DeepSeek-Math model using LoRA/QLoRA.

## Features

- **Parameter-Efficient Fine-Tuning**: Uses LoRA to train only ~1% of parameters
- **4-bit Quantization**: QLoRA support for low memory requirements
- **Flexible Configuration**: Multiple preset configs for different scenarios
- **Easy Data Format**: Simple JSON format for QA pairs

## Data Format

Training data should be in JSON format:

```json
[
  {
    "question": "What is conditional probability?",
    "answer": "Conditional probability is...",
    "source": "Leon-Garcia Chapter 2.4"
  }
]
```

Required fields:
- `question`: The question text
- `answer`: The expected answer
- `source`: (optional) Source reference

## Quick Start

### 1. Prepare Data

Create training data in `data/training_data/sft_train.json`:

```bash
# Use the sample data as a template
cp data/training_data/sft_sample.json data/training_data/sft_train.json
# Edit with your own QA pairs
```

### 2. Install Dependencies

```bash
uv pip install peft accelerate bitsandbytes
```

### 3. Run Training

```bash
# Test setup (dry run)
.venv/bin/python training/sft/train_sft.py --dry-run

# Fast test (1 epoch, small batch)
.venv/bin/python training/sft/train_sft.py --config fast_test

# Full training (default config)
.venv/bin/python training/sft/train_sft.py

# Custom settings
.venv/bin/python training/sft/train_sft.py \
    --epochs 5 \
    --batch-size 8 \
    --lr 1e-4 \
    --train-data data/training_data/my_train.json
```

## Configuration Presets

### Default
- 3 epochs
- Batch size: 4
- LoRA r=16, alpha=32
- Max length: 1024

### Fast Test
- 1 epoch
- Batch size: 2
- Max length: 512
- Good for testing setup

### Full Training
- 5 epochs
- Batch size: 8
- Max length: 2048
- For complete training runs

### Low Memory
- Batch size: 1
- Gradient accumulation: 16
- Max length: 512
- For limited GPU memory

## Command Line Options

```bash
python training/sft/train_sft.py [OPTIONS]

Options:
  --config {default,fast_test,full_training,low_memory}
  --train-data PATH       Path to training data JSON
  --val-data PATH         Path to validation data JSON
  --model-name NAME       HuggingFace model name
  --no-4bit              Disable 4-bit quantization
  --no-lora              Disable LoRA (full fine-tuning)
  --epochs N             Number of epochs
  --batch-size N         Batch size
  --lr FLOAT             Learning rate
  --max-length N         Maximum sequence length
  --output-dir PATH      Output directory
  --resume-from PATH     Resume from checkpoint
  --dry-run              Test setup without training
```

## Output Structure

```
output/sft/
├── checkpoint-100/
├── checkpoint-200/
├── final_model/
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── tokenizer files
└── training_log.txt
```

## Using the Fine-Tuned Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-math-7b-instruct",
    device_map="auto",
    torch_dtype=torch.float16
)

# Load LoRA weights
model = PeftModel.from_pretrained(
    base_model,
    "output/sft/final_model"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("output/sft/final_model")

# Generate
prompt = "What is Bayes theorem?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Memory Requirements

- **4-bit + LoRA**: ~8GB VRAM (recommended)
- **Full precision**: ~28GB VRAM
- **No quantization + LoRA**: ~14GB VRAM

## Tips

1. **Start with fast_test**: Verify setup before full training
2. **Monitor GPU memory**: Use `nvidia-smi` to check usage
3. **Adjust batch size**: Reduce if OOM errors occur
4. **Use gradient accumulation**: Simulate larger batches
5. **Save frequently**: Use smaller `--save-steps` for important runs

## Troubleshooting

### OOM (Out of Memory)
- Enable 4-bit quantization: Remove `--no-4bit`
- Reduce batch size: `--batch-size 1`
- Reduce max length: `--max-length 512`
- Enable gradient checkpointing (default)

### Slow training
- Increase batch size if memory allows
- Reduce gradient accumulation steps
- Use shorter sequences if possible

### Poor performance
- Increase number of epochs
- Try different learning rates (1e-4 to 3e-4)
- Increase LoRA rank: Edit config.py
- Add more training data

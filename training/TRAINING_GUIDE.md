# Complete Training Guide: SFT + GRPO

Comprehensive guide for training your Math QA model.

## Training Pipeline Overview

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│ Base Model  │  →   │     SFT     │  →   │    GRPO     │  →   │ Final Model │
│ (DeepSeek)  │      │  Supervised │      │ Preference  │      │  (Aligned)  │
└─────────────┘      └─────────────┘      └─────────────┘      └─────────────┘
     4GB                   +LoRA                +LoRA                3-4GB
                          ~1 hour             ~2-3 hours
```

## Quick Comparison: SFT vs GRPO

| Aspect | SFT | GRPO |
|--------|-----|------|
| **Purpose** | Learn from examples | Align with preferences |
| **When** | First training step | After SFT |
| **Training Signal** | Supervised labels | Reward-based |
| **Speed** | Fast (1 hour for 100 samples) | Slower (2-3 hours for 100 samples) |
| **Memory** | ~8GB VRAM | ~12-16GB VRAM |
| **Learning Rate** | 2e-4 | 1e-5 to 5e-6 |
| **Data** | QA pairs | Same QA pairs |
| **Complexity** | Simple | Advanced |

## Step-by-Step Training

### Phase 1: Data Preparation

```json
[
  {
    "question": "What is conditional probability?",
    "answer": "Conditional probability is the probability of an event A occurring given that another event B has already occurred. It is denoted as P(A|B) and calculated using the formula: P(A|B) = P(A ∩ B) / P(B), where P(B) > 0.",
    "source": "Leon-Garcia Chapter 2.4"
  }
]
```

**Required fields:**
- `question`: The question text
- `answer`: Reference answer
- `source`: (optional) Source citation

**Recommended data sizes:**
- **Minimum**: 50-100 QA pairs
- **Good**: 500-1000 QA pairs
- **Ideal**: 5000+ QA pairs

### Phase 2: SFT Training

#### 2.1 Quick Test

```bash
# Verify setup works
.venv/bin/python scripts/test_sft_setup.py

# Dry run (no actual training)
.venv/bin/python training/sft/train_sft.py --dry-run

# Fast test (1 epoch, small batch)
.venv/bin/python training/sft/train_sft.py \
    --config fast_test \
    --train-data data/training_data/sft_sample.json
```

#### 2.2 Full SFT Training

```bash
# Default configuration (recommended for first run)
.venv/bin/python training/sft/train_sft.py \
    --config default \
    --train-data data/training_data/sft_train.json \
    --val-data data/training_data/sft_val.json \
    --output-dir output/sft_v1

# Custom settings
.venv/bin/python training/sft/train_sft.py \
    --train-data data/training_data/sft_train.json \
    --epochs 5 \
    --batch-size 4 \
    --lr 2e-4 \
    --max-length 1024 \
    --output-dir output/sft_custom
```

**Expected output:**
```
output/sft_v1/
├── checkpoint-100/
├── checkpoint-200/
├── final_model/
│   ├── adapter_config.json
│   ├── adapter_model.bin  (~50MB)
│   └── tokenizer files
```

**Training time estimates:**
- 100 samples: ~30-60 minutes
- 500 samples: ~2-4 hours
- 1000 samples: ~4-8 hours

#### 2.3 Evaluate SFT Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load model
base = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-math-7b-instruct",
    device_map="auto",
    torch_dtype=torch.float16
)
model = PeftModel.from_pretrained(base, "output/sft_v1/final_model")
tokenizer = AutoTokenizer.from_pretrained("output/sft_v1/final_model")

# Test
prompt = "What is Bayes theorem?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Phase 3: GRPO Training (Optional but Recommended)

#### 3.1 Quick Test

```bash
# Verify GRPO setup
.venv/bin/python scripts/test_grpo_setup.py

# Dry run
.venv/bin/python training/grpo/train_grpo.py --dry-run
```

#### 3.2 GRPO from SFT

```bash
# Recommended: Start from SFT checkpoint
.venv/bin/python training/grpo/train_grpo.py \
    --config from_sft \
    --sft-model output/sft_v1/final_model \
    --train-data data/training_data/grpo_train.json \
    --output-dir output/grpo_v1

# Alternative: From base model
.venv/bin/python training/grpo/train_grpo.py \
    --config default \
    --train-data data/training_data/grpo_train.json \
    --output-dir output/grpo_base
```

**Expected output:**
```
output/grpo_v1/
├── checkpoint-100/
├── checkpoint-200/
├── final_model/
│   ├── adapter_config.json
│   ├── adapter_model.bin  (~50-100MB)
│   └── tokenizer files
```

**Training time estimates:**
- 100 samples: ~1-2 hours
- 500 samples: ~5-10 hours
- 1000 samples: ~10-20 hours

## Common Training Workflows

### Workflow 1: Quick Prototype

```bash
# 1. Fast SFT test
.venv/bin/python training/sft/train_sft.py --config fast_test

# 2. Skip GRPO for speed
# Use SFT model directly
```

**Use when**: Rapid prototyping, testing ideas

### Workflow 2: Standard Training

```bash
# 1. Full SFT
.venv/bin/python training/sft/train_sft.py --config default

# 2. Full GRPO from SFT
.venv/bin/python training/grpo/train_grpo.py \
    --config from_sft \
    --sft-model output/sft/final_model
```

**Use when**: Production models, best quality

### Workflow 3: Low Memory

```bash
# 1. SFT with minimal memory
.venv/bin/python training/sft/train_sft.py --config low_memory

# 2. GRPO with minimal memory
.venv/bin/python training/grpo/train_grpo.py \
    --config low_memory \
    --sft-model output/sft/final_model
```

**Use when**: Limited GPU memory (<16GB)

## Monitoring Training

### Watch GPU Usage

```bash
watch -n 1 nvidia-smi
```

### Monitor Logs

```bash
# SFT logs
tail -f output/sft/training_log.txt

# GRPO logs
tail -f output/grpo/training_log.txt
```

### Key Metrics to Watch

**SFT:**
- Loss should decrease steadily
- Typical final loss: 0.5-1.5

**GRPO:**
- Mean reward should increase
- Loss may fluctuate (normal)
- KL divergence should stay < 0.5

## Troubleshooting

### Out of Memory (OOM)

**For SFT:**
```bash
# Reduce batch size
--batch-size 1

# Reduce sequence length
--max-length 512

# Use low memory config
--config low_memory
```

**For GRPO:**
```bash
# Reduce samples per question
--num-samples 2

# Reduce batch size
--batch-size 1

# Use low memory config
--config low_memory
```

### Poor Quality Results

**After SFT:**
- Increase epochs (5-10)
- Add more training data
- Check data quality
- Try different learning rates

**After GRPO:**
- Adjust KL coefficient (0.05-0.2)
- Increase samples per question
- Check reward model is working
- Ensure good SFT checkpoint

### Training Too Slow

**For SFT:**
- Increase batch size
- Reduce max_length
- Use gradient accumulation

**For GRPO:**
- Reduce num_samples_per_question
- Reduce max_new_tokens
- Smaller batch_size with higher gradient_accumulation

## Best Practices

### Data Quality

1. **Clean data**: Remove duplicates, errors
2. **Diverse examples**: Cover different topics
3. **Accurate answers**: Verify correctness
4. **Consistent format**: Use same structure

### Training Strategy

1. **Start simple**: Use fast_test first
2. **Iterate quickly**: Small experiments
3. **Monitor closely**: Check samples frequently
4. **Save checkpoints**: Don't rely on final model only

### Hyperparameter Tuning

**SFT Priority:**
1. Learning rate (most important)
2. Number of epochs
3. Batch size
4. LoRA rank

**GRPO Priority:**
1. KL coefficient (most important)
2. Number of samples
3. Learning rate
4. Reward weights

## Hardware Requirements

### Minimum (Testing)
- GPU: 8GB VRAM
- Config: `low_memory`
- Batch size: 1

### Recommended (Training)
- GPU: 16-24GB VRAM
- Config: `default`
- Batch size: 4-8

### Optimal (Production)
- GPU: 24GB+ VRAM (like your RTX 5090!)
- Config: `full_training`
- Batch size: 8-16

## Expected Results

### After SFT
- Model follows instruction format
- Generates structured answers
- Uses appropriate terminology
- May still have quality issues

### After GRPO
- Improved answer quality
- Better preference alignment
- More consistent responses
- Reduced hallucinations

## Integration with RAG

### Option 1: Use Fine-Tuned Model in RAG

```python
# Replace base model in RAG pipeline
from src.rag_pipeline import RAGPipeline

rag = RAGPipeline(
    model_name="deepseek-ai/deepseek-math-7b-instruct",
    use_4bit=True
)
rag.initialize(load_model=True)

# Load your fine-tuned adapter
from peft import PeftModel
rag.model_loader.model = PeftModel.from_pretrained(
    rag.model_loader.model,
    "output/grpo/final_model"  # or sft/final_model
)
```

### Option 2: Fine-Tune on RAG-Generated Data

```python
# 1. Generate training data using RAG
# 2. Fine-tune model on this data
# 3. Use fine-tuned model in RAG
# 4. Iterate
```

## Complete Example

```bash
# 1. Setup
.venv/bin/python scripts/test_sft_setup.py
.venv/bin/python scripts/test_grpo_setup.py

# 2. Prepare data
# Edit data/training_data/sft_train.json
# (Can use same file for GRPO)

# 3. SFT Training
.venv/bin/python training/sft/train_sft.py \
    --config default \
    --train-data data/training_data/sft_train.json \
    --output-dir output/sft_$(date +%Y%m%d)

# 4. Test SFT model
# (Sample some outputs, check quality)

# 5. GRPO Training
.venv/bin/python training/grpo/train_grpo.py \
    --config from_sft \
    --sft-model output/sft_$(date +%Y%m%d)/final_model \
    --train-data data/training_data/grpo_train.json \
    --output-dir output/grpo_$(date +%Y%m%d)

# 6. Final testing and deployment
```

## File Organization

```
data/training_data/
├── sft_train.json          # SFT training data
├── sft_val.json            # SFT validation data
├── grpo_train.json         # GRPO training (can be same as sft_train.json)
└── grpo_val.json           # GRPO validation

output/
├── sft_20241130/           # SFT output (dated)
│   ├── checkpoint-*/
│   └── final_model/
└── grpo_20241130/          # GRPO output (dated)
    ├── checkpoint-*/
    └── final_model/

training/
├── sft/                    # SFT code
├── grpo/                   # GRPO code
├── SFT_GUIDE.md           # SFT documentation
└── TRAINING_GUIDE.md      # This file
```

## Next Steps

1. **Prepare your training data**
2. **Run SFT fast_test** to verify setup
3. **Run full SFT training**
4. **Evaluate SFT model**
5. **Optionally run GRPO**
6. **Integrate with RAG** (if desired)

## Support

- SFT details: `training/sft/README.md`
- GRPO details: `training/grpo/README.md`
- Test scripts: `scripts/test_*_setup.py`
- Example data: `data/training_data/sft_sample.json`

---

**Status**: ✅ Both SFT and GRPO frameworks ready!
**Your GPU**: RTX 5090 (34.2 GB) - Perfect for training!
**Recommendation**: Start with SFT `fast_test`, then `default`

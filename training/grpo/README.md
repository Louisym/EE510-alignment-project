# GRPO (Group Relative Policy Optimization) for Math QA

Advanced preference optimization for fine-tuned language models.

## Overview

GRPO is a reinforcement learning method that aligns language models with preferences by:
1. Generating multiple responses per question
2. Computing group-relative rewards
3. Optimizing policy to maximize relative advantages
4. Maintaining KL divergence from reference model

## Features

- **Group-Relative Rewards**: Compare responses within groups
- **Heuristic Reward Model**: Math-specific quality metrics
- **KL Regularization**: Prevent drift from reference model
- **Efficient Training**: Works with 4-bit quantization and LoRA
- **SFT Warm Start**: Can start from SFT checkpoint

## When to Use GRPO

Use GRPO **after** SFT training to further align the model with preferences:

```
Base Model → SFT → GRPO → Aligned Model
```

GRPO is particularly effective when:
- You have completed SFT
- You want to improve answer quality
- You need preference-based alignment
- Standard supervised learning plateaus

## Data Format

Same format as SFT:

```json
[
  {
    "question": "What is conditional probability?",
    "answer": "Conditional probability is...",
    "source": "Leon-Garcia Chapter 2.4"
  }
]
```

The `answer` field serves as a reference for reward computation.

## Quick Start

### 1. Prerequisites

Complete SFT training first (recommended):

```bash
.venv/bin/python training/sft/train_sft.py --config default
```

### 2. Run GRPO

```bash
# From SFT checkpoint
.venv/bin/python training/grpo/train_grpo.py \
    --config from_sft \
    --sft-model output/sft/final_model \
    --train-data data/training_data/grpo_train.json

# From base model (no SFT)
.venv/bin/python training/grpo/train_grpo.py \
    --config default \
    --train-data data/training_data/grpo_train.json

# Fast test
.venv/bin/python training/grpo/train_grpo.py \
    --config fast_test \
    --dry-run
```

## Configuration Presets

### Default
- 3 epochs
- 4 samples per question
- Batch size: 2
- KL coefficient: 0.1

### Fast Test
- 1 epoch
- 2 samples per question
- Batch size: 1
- For quick testing

### Full Training
- 5 epochs
- 8 samples per question
- Batch size: 4
- For production

### From SFT
- Starts from SFT checkpoint
- Lower learning rate (5e-6)
- 4 samples per question

### Low Memory
- Minimal memory usage
- 2 samples per question
- Batch size: 1

## Command Line Options

```bash
python training/grpo/train_grpo.py [OPTIONS]

Options:
  --config {default,fast_test,full_training,low_memory,from_sft}
  --train-data PATH          Path to training data JSON
  --val-data PATH            Path to validation data JSON
  --model-name NAME          HuggingFace model name
  --sft-model PATH           Path to SFT checkpoint (recommended)
  --no-4bit                  Disable 4-bit quantization
  --no-lora                  Disable LoRA
  --num-samples N            Number of samples per question
  --kl-coef FLOAT            KL divergence coefficient
  --reward-model {heuristic,learned}  Type of reward model
  --epochs N                 Number of epochs
  --batch-size N             Batch size
  --lr FLOAT                 Learning rate
  --max-length N             Maximum sequence length
  --output-dir PATH          Output directory
  --dry-run                  Test setup without training
```

## GRPO Algorithm

For each training step:

1. **Sample Responses**: Generate K responses per question using current policy
2. **Compute Rewards**: Evaluate each response with reward model
3. **Group-Relative Advantages**:
   ```
   advantage = (reward - mean(group_rewards)) / std(group_rewards)
   ```
4. **Policy Update**:
   ```
   loss = -advantage * log_prob + kl_coef * KL(policy || reference)
   ```

## Reward Model

### Heuristic Reward Model (Default)

Evaluates answers based on:

1. **Length Appropriateness** (20%): Similar length to reference
2. **Mathematical Formulas** (30%): Presence of equations and notation
3. **Concept Coverage** (30%): Key term overlap with reference
4. **Structure** (20%): Well-formatted, clear explanation

### Learned Reward Model

Can be trained separately on preference data (future work).

## Memory Requirements

- **4-bit + LoRA**: ~12GB VRAM (policy + reference models)
- **Full precision**: ~40GB VRAM
- **With SFT warm start**: Same as above

## Hyperparameter Tuning

### KL Coefficient (`kl_coef`)
- **Higher (0.2-0.5)**: Stay closer to reference, more conservative
- **Lower (0.01-0.1)**: Allow more deviation, more aggressive
- **Default**: 0.1

### Number of Samples
- **More samples (8-16)**: Better gradient estimates, slower
- **Fewer samples (2-4)**: Faster, noisier gradients
- **Default**: 4

### Learning Rate
- **From SFT**: 5e-6 (very low)
- **From base**: 1e-5 (low)
- **Never use SFT learning rate**: Too high for GRPO

### Group Size
- Should equal `num_samples_per_question`
- Larger groups = more robust comparisons

## Output Structure

```
output/grpo/
├── checkpoint-100/
├── checkpoint-200/
├── final_model/
│   ├── adapter_config.json    (if using LoRA)
│   ├── adapter_model.bin
│   └── tokenizer files
└── training_log.txt
```

## Using the GRPO Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-math-7b-instruct",
    device_map="auto",
    torch_dtype=torch.float16
)

# Load GRPO adapter
model = PeftModel.from_pretrained(base_model, "output/grpo/final_model")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("output/grpo/final_model")

# Generate
prompt = "What is Bayes theorem?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Best Practices

1. **Start with SFT**: Always do SFT before GRPO
2. **Use lower learning rate**: GRPO needs gentler updates
3. **Monitor KL divergence**: Should stay < 0.5
4. **Check reward distribution**: Should show clear preferences
5. **Validate quality**: Sample and inspect generated answers

## Troubleshooting

### Poor quality after GRPO
- **Cause**: KL coefficient too low or learning rate too high
- **Solution**: Increase `kl_coef` to 0.2-0.3, reduce learning rate

### Model ignores training
- **Cause**: KL coefficient too high
- **Solution**: Reduce `kl_coef` to 0.05

### Out of memory
- **Solution**:
  - Reduce `num_samples_per_question`
  - Reduce `batch_size`
  - Use `--config low_memory`

### Slow training
- **Solution**:
  - Reduce `num_samples_per_question`
  - Reduce `max_new_tokens`
  - Use shorter sequences

## Comparison: SFT vs GRPO

| Aspect | SFT | GRPO |
|--------|-----|------|
| Training Signal | Supervised (reference answers) | Reward-based (relative quality) |
| Data Requirement | Labeled QA pairs | Same data, used differently |
| Training Speed | Faster | Slower (generates multiple samples) |
| When to Use | First step | After SFT |
| Learning Rate | Higher (2e-4) | Lower (1e-5 to 5e-6) |
| Memory | Lower | Higher (2x models) |

## Advanced: Custom Reward Model

To implement a custom reward model:

```python
from training.grpo.reward_model import RewardModel

class MyRewardModel(RewardModel):
    def compute_reward(self, questions, answers, references):
        # Your custom logic here
        rewards = []
        for q, a, r in zip(questions, answers, references):
            reward = my_scoring_function(q, a, r)
            rewards.append(reward)
        return torch.tensor(rewards)

# Use in training
trainer = GRPOTrainer(config=config, reward_model=MyRewardModel())
```

## Tips

1. **Data Quality**: GRPO amplifies data quality issues
2. **Patience**: GRPO is slower than SFT
3. **Evaluation**: Check samples frequently
4. **Checkpointing**: Save often, results can vary
5. **Hyperparameters**: Start conservative, then experiment

## References

- Original GRPO paper: Group Relative Policy Optimization
- Related: PPO, DPO, RLHF
- See also: `training/sft/README.md` for SFT details

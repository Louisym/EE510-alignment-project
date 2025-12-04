"""
Test GRPO Setup
Verify that all components are working correctly
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

print("=" * 80)
print("Testing GRPO Setup")
print("=" * 80)

# Test 1: Import modules
print("\n1. Testing imports...")
try:
    from training.grpo.data_loader import GRPODataset, create_grpo_dataloaders
    from training.grpo.trainer import GRPOTrainer
    from training.grpo.config import get_grpo_config, GRPO_CONFIGS
    from training.grpo.reward_model import MathRewardModel, RewardModel
    print("   ✅ All modules imported successfully")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Check data file
print("\n2. Checking sample data...")
sample_data_path = Path("data/training_data/sft_sample.json")
if sample_data_path.exists():
    import json
    with open(sample_data_path) as f:
        data = json.load(f)
    print(f"   ✅ Sample data found: {len(data)} QA pairs")
    print(f"      (Note: GRPO can use same data as SFT)")
else:
    print(f"   ❌ Sample data not found at {sample_data_path}")
    sys.exit(1)

# Test 3: Test configurations
print("\n3. Testing GRPO configurations...")
try:
    for config_name in GRPO_CONFIGS:
        config = get_grpo_config(config_name)
        print(f"   ✅ {config_name}: epochs={config.num_epochs}, "
              f"samples={config.num_samples_per_question}, "
              f"kl_coef={config.kl_coef}")
except Exception as e:
    print(f"   ❌ Configuration test failed: {e}")
    sys.exit(1)

# Test 4: Test reward model
print("\n4. Testing reward model...")
try:
    reward_model = MathRewardModel()

    # Test data
    questions = ["What is probability?"]
    answers = ["Probability is a measure of likelihood. Formula: P(A) where 0 <= P(A) <= 1."]
    references = ["Probability is a measure of the likelihood of an event occurring."]

    rewards = reward_model.compute_reward(questions, answers, references)

    print(f"   ✅ Reward model initialized")
    print(f"      - Test reward: {rewards[0]:.4f}")
    print(f"      - Reward shape: {rewards.shape}")

except Exception as e:
    print(f"   ❌ Reward model test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test data loader
print("\n5. Testing GRPO data loader...")
try:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-math-7b-instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = GRPODataset(
        data_path=str(sample_data_path),
        tokenizer=tokenizer,
        max_length=512,
        num_samples_per_question=4
    )

    print(f"   ✅ Dataset created: {len(dataset)} examples")

    # Test single item
    item = dataset[0]
    print(f"   ✅ Item structure:")
    print(f"      - Question IDs shape: {item['question_ids'].shape}")
    print(f"      - Num samples: {item['num_samples']}")

except Exception as e:
    print(f"   ❌ Data loader test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test trainer initialization
print("\n6. Testing GRPO trainer initialization...")
try:
    config = get_grpo_config("fast_test")
    reward_model = MathRewardModel()

    trainer = GRPOTrainer(
        config=config,
        reward_model=reward_model
    )

    print(f"   ✅ Trainer initialized")
    print(f"      - Output dir: {trainer.config.output_dir}")
    print(f"      - Samples per question: {trainer.config.num_samples_per_question}")
    print(f"      - KL coefficient: {trainer.config.kl_coef}")
    print(f"      - Group size: {trainer.config.group_size}")

except Exception as e:
    print(f"   ❌ Trainer initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Check CUDA availability
print("\n7. Checking CUDA availability...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"   ✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"      - Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"      - Note: GRPO needs ~2x memory of SFT (policy + reference models)")
    else:
        print(f"   ⚠️  CUDA not available (will use CPU - very slow)")
except Exception as e:
    print(f"   ❌ CUDA check failed: {e}")

# Summary
print("\n" + "=" * 80)
print("✅ GRPO Setup Test Completed Successfully!")
print("=" * 80)
print("\nYou can now:")
print("1. (Recommended) Complete SFT training first")
print("2. Prepare your training data in JSON format (same as SFT)")
print("3. Run dry-run: .venv/bin/python training/grpo/train_grpo.py --dry-run")
print("4. Start GRPO: .venv/bin/python training/grpo/train_grpo.py --config fast_test")
print("\nSee training/grpo/README.md for more details")
print("\n⚠️  Important: GRPO is computationally intensive!")
print("   - Generates multiple responses per question")
print("   - Loads policy + reference models (~2x memory)")
print("   - Training is slower than SFT")

"""
Test SFT Setup
Verify that all components are working correctly
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

print("=" * 80)
print("Testing SFT Setup")
print("=" * 80)

# Test 1: Import modules
print("\n1. Testing imports...")
try:
    from training.sft.data_loader import QADataset, create_dataloaders
    from training.sft.trainer import SFTTrainer, get_model_info
    from training.sft.config import get_config, CONFIGS
    print("   ✅ All modules imported successfully")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Check data file
print("\n2. Checking sample data...")
sample_data_path = Path("data/training_data/sft_sample.json")
if sample_data_path.exists():
    import json
    with open(sample_data_path) as f:
        data = json.load(f)
    print(f"   ✅ Sample data found: {len(data)} QA pairs")
else:
    print(f"   ❌ Sample data not found at {sample_data_path}")
    sys.exit(1)

# Test 3: Test configuration
print("\n3. Testing configurations...")
try:
    for config_name in CONFIGS:
        config = get_config(config_name)
        print(f"   ✅ {config_name}: epochs={config.num_epochs}, batch={config.batch_size}")
except Exception as e:
    print(f"   ❌ Configuration test failed: {e}")
    sys.exit(1)

# Test 4: Test data loader
print("\n4. Testing data loader...")
try:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-math-7b-instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = QADataset(
        data_path=str(sample_data_path),
        tokenizer=tokenizer,
        max_length=512
    )

    print(f"   ✅ Dataset created: {len(dataset)} examples")

    # Test single item
    item = dataset[0]
    print(f"   ✅ Item shape: input_ids={item['input_ids'].shape}, "
          f"labels={item['labels'].shape}")

except Exception as e:
    print(f"   ❌ Data loader test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test trainer initialization
print("\n5. Testing trainer initialization...")
try:
    trainer = SFTTrainer(
        model_name="deepseek-ai/deepseek-math-7b-instruct",
        output_dir="./test_output_sft",
        use_4bit=True,
        use_lora=True,
        batch_size=2,
        num_epochs=1
    )
    print(f"   ✅ Trainer initialized")
    print(f"      - Output dir: {trainer.output_dir}")
    print(f"      - Use 4-bit: {trainer.use_4bit}")
    print(f"      - Use LoRA: {trainer.use_lora}")
    print(f"      - LoRA r: {trainer.lora_config.r}")

except Exception as e:
    print(f"   ❌ Trainer initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Check CUDA availability
print("\n6. Checking CUDA availability...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"   ✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"      - Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print(f"   ⚠️  CUDA not available (will use CPU - very slow)")
except Exception as e:
    print(f"   ❌ CUDA check failed: {e}")

# Summary
print("\n" + "=" * 80)
print("✅ SFT Setup Test Completed Successfully!")
print("=" * 80)
print("\nYou can now:")
print("1. Prepare your training data in JSON format")
print("2. Run dry-run test: .venv/bin/python training/sft/train_sft.py --dry-run")
print("3. Start training: .venv/bin/python training/sft/train_sft.py --config fast_test")
print("\nSee training/sft/README.md for more details")

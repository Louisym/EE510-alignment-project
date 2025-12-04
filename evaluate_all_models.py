"""
完整模型评估对比
在测试集上评估：Base Model, SFT-Random, SFT-SVD, GRPO, RAG+GRPO
"""
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import sys
import os

# Add paths
sys.path.append('training/grpo')
from reward_model import MathRewardModel

def load_test_data(path='data/test_questions.json'):
    """加载测试数据"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_base_model():
    """加载基础模型（4-bit）"""
    print("Loading base model...")
    model_name = "Qwen/Qwen2.5-Math-7B-Instruct"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side='left'
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )

    return model, tokenizer

def load_lora_model(base_model, tokenizer, lora_path, model_name="SFT"):
    """加载LoRA微调模型"""
    print(f"Loading {model_name} from {lora_path}...")
    model = PeftModel.from_pretrained(base_model, lora_path)
    return model

def generate_answer(model, tokenizer, question, max_length=512):
    """生成回答"""
    prompt = f"Question: {question}\n\nAnswer:"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 提取答案部分
    if "Answer:" in generated_text:
        answer = generated_text.split("Answer:")[-1].strip()
    else:
        answer = generated_text[len(prompt):].strip()

    return answer

def evaluate_model(model, tokenizer, test_data, reward_model, model_name="Model"):
    """评估单个模型"""
    print(f"\n{'='*80}")
    print(f"Evaluating {model_name}")
    print(f"{'='*80}\n")

    results = []
    total_reward = 0

    for i, item in enumerate(test_data, 1):
        question = item['question']
        reference = item['answer']

        print(f"Question {i}/{len(test_data)}: {question[:50]}...")

        # 生成答案
        answer = generate_answer(model, tokenizer, question)

        # 计算reward
        rewards = reward_model.compute_reward(
            questions=[question],
            answers=[answer],
            references=[reference]
        )
        reward = rewards[0].item()

        total_reward += reward

        results.append({
            'question': question,
            'reference': reference,
            'generated': answer,
            'reward': reward
        })

        print(f"  Reward: {reward:.4f}")
        print(f"  Generated (first 100 chars): {answer[:100]}...")
        print()

    avg_reward = total_reward / len(test_data)

    print(f"\n{model_name} Average Reward: {avg_reward:.4f}\n")

    return {
        'model_name': model_name,
        'avg_reward': avg_reward,
        'results': results
    }

def main():
    print("="*80)
    print("COMPLETE MODEL EVALUATION")
    print("="*80)
    print()

    # 加载测试数据
    test_data = load_test_data()
    print(f"Loaded {len(test_data)} test questions\n")

    # 初始化reward model
    reward_model = MathRewardModel(
        length_weight=0.2,
        formula_weight=0.3,
        concept_weight=0.3,
        structure_weight=0.2
    )

    all_results = []

    # 1. Base Model
    print("\n" + "="*80)
    print("1. EVALUATING BASE MODEL")
    print("="*80)
    base_model, tokenizer = load_base_model()
    result_base = evaluate_model(base_model, tokenizer, test_data, reward_model, "Base Model")
    all_results.append(result_base)

    # 2. SFT Random-init
    print("\n" + "="*80)
    print("2. EVALUATING SFT (Random-init LoRA)")
    print("="*80)
    sft_random_path = "experiments/svd_lora/training_results/final_model_random"
    if os.path.exists(sft_random_path):
        model_sft_random = load_lora_model(base_model, tokenizer, sft_random_path, "SFT-Random")
        result_sft_random = evaluate_model(model_sft_random, tokenizer, test_data, reward_model, "SFT Random-init")
        all_results.append(result_sft_random)
        del model_sft_random
    else:
        print(f"⚠️ SFT Random model not found at {sft_random_path}")

    # 3. SFT SVD-init
    print("\n" + "="*80)
    print("3. EVALUATING SFT (SVD-init LoRA)")
    print("="*80)
    sft_svd_path = "experiments/svd_lora/training_results/final_model_svd"
    if os.path.exists(sft_svd_path):
        model_sft_svd = load_lora_model(base_model, tokenizer, sft_svd_path, "SFT-SVD")
        result_sft_svd = evaluate_model(model_sft_svd, tokenizer, test_data, reward_model, "SFT SVD-init")
        all_results.append(result_sft_svd)
        del model_sft_svd
    else:
        print(f"⚠️ SFT SVD model not found at {sft_svd_path}")

    # 4. GRPO
    print("\n" + "="*80)
    print("4. EVALUATING GRPO")
    print("="*80)
    grpo_path = "outputs/grpo/final_model"
    if os.path.exists(grpo_path):
        model_grpo = load_lora_model(base_model, tokenizer, grpo_path, "GRPO")
        result_grpo = evaluate_model(model_grpo, tokenizer, test_data, reward_model, "GRPO")
        all_results.append(result_grpo)
        del model_grpo
    else:
        print(f"⚠️ GRPO model not found at {grpo_path}")

    # 保存结果
    print("\n" + "="*80)
    print("SUMMARY OF ALL MODELS")
    print("="*80)
    print()

    for result in all_results:
        print(f"{result['model_name']:20s}: {result['avg_reward']:.4f}")

    # 计算改进
    if len(all_results) >= 2:
        base_reward = all_results[0]['avg_reward']
        print(f"\n{'Improvements relative to Base Model':40s}")
        print("-" * 60)
        for result in all_results[1:]:
            improvement = ((result['avg_reward'] - base_reward) / base_reward) * 100
            print(f"{result['model_name']:20s}: {improvement:+.2f}%")

    # 保存详细结果
    output_file = 'evaluation_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Detailed results saved to {output_file}")
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()

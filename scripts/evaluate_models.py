"""
Model Evaluation and Comparison Script
评估和对比 Base/SFT/GRPO 模型的性能

用于生成 Presentation 和 Report 的对比数据
"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from training.visualization import ModelComparator


class ModelEvaluator:
    """模型评估器"""

    def __init__(
        self,
        base_model_name: str = "deepseek-ai/deepseek-math-7b-instruct",
        sft_model_path: str = None,
        grpo_model_path: str = None,
        device: str = "auto"
    ):
        """
        初始化评估器

        Args:
            base_model_name: 基础模型名称
            sft_model_path: SFT 模型路径
            grpo_model_path: GRPO 模型路径
            device: 设备
        """
        self.base_model_name = base_model_name
        self.sft_model_path = sft_model_path
        self.grpo_model_path = grpo_model_path
        self.device = device

        self.tokenizer = None
        self.base_model = None
        self.sft_model = None
        self.grpo_model = None

        print("🚀 Initializing Model Evaluator...")

    def load_models(self):
        """加载所有模型"""
        print("\n" + "="*70)
        print("📦 Loading Models...")
        print("="*70)

        # 加载 tokenizer
        print(f"\n1. Loading tokenizer from {self.base_model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("✓ Tokenizer loaded")

        # 加载基础模型
        print(f"\n2. Loading base model: {self.base_model_name}...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            device_map=self.device,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        self.base_model.eval()
        print("✓ Base model loaded")

        # 加载 SFT 模型
        if self.sft_model_path and os.path.exists(self.sft_model_path):
            print(f"\n3. Loading SFT model from {self.sft_model_path}...")
            self.sft_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                device_map=self.device,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            self.sft_model = PeftModel.from_pretrained(
                self.sft_model,
                self.sft_model_path
            )
            self.sft_model.eval()
            print("✓ SFT model loaded")
        else:
            print(f"\n3. ⚠ SFT model not found at {self.sft_model_path}, skipping")

        # 加载 GRPO 模型
        if self.grpo_model_path and os.path.exists(self.grpo_model_path):
            print(f"\n4. Loading GRPO model from {self.grpo_model_path}...")
            self.grpo_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                device_map=self.device,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            self.grpo_model = PeftModel.from_pretrained(
                self.grpo_model,
                self.grpo_model_path
            )
            self.grpo_model.eval()
            print("✓ GRPO model loaded")
        else:
            print(f"\n4. ⚠ GRPO model not found at {self.grpo_model_path}, skipping")

        print("\n✓ All available models loaded successfully!\n")

    @torch.no_grad()
    def generate_response(
        self,
        model,
        question: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        生成模型回答

        Args:
            model: 模型
            question: 问题
            max_new_tokens: 最大生成长度
            temperature: 温度
            top_p: top-p 采样

        Returns:
            生成的回答
        """
        # 构建 prompt
        prompt = f"You are a mathematics expert specializing in probability theory. Please answer the following question accurately and clearly.\n\nQuestion: {question}\n\nAnswer:"

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # 生成
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        # 解码
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取答案部分
        if "Answer:" in full_response:
            answer = full_response.split("Answer:")[-1].strip()
        else:
            answer = full_response[len(prompt):].strip()

        return answer

    def evaluate_on_questions(
        self,
        questions: List[str],
        output_dir: str = "./evaluation_results"
    ) -> Dict:
        """
        在问题列表上评估所有模型

        Args:
            questions: 问题列表
            output_dir: 输出目录

        Returns:
            评估结果
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*70)
        print("🔬 Evaluating Models on Test Questions...")
        print("="*70)
        print(f"Total questions: {len(questions)}\n")

        results = {
            'questions': questions,
            'base_outputs': [],
            'sft_outputs': [],
            'grpo_outputs': []
        }

        # 创建对比器
        comparator = ModelComparator(output_dir)

        # 对每个问题生成回答
        for i, question in enumerate(tqdm(questions, desc="Generating responses")):
            print(f"\n{'='*70}")
            print(f"Question {i+1}/{len(questions)}:")
            print(f"{question[:100]}..." if len(question) > 100 else question)
            print('='*70)

            # Base 模型
            print("\n📌 Base Model:")
            base_output = self.generate_response(self.base_model, question)
            results['base_outputs'].append(base_output)
            print(f"{base_output[:200]}..." if len(base_output) > 200 else base_output)

            # SFT 模型
            sft_output = ""
            if self.sft_model:
                print("\n📌 SFT Model:")
                sft_output = self.generate_response(self.sft_model, question)
                results['sft_outputs'].append(sft_output)
                print(f"{sft_output[:200]}..." if len(sft_output) > 200 else sft_output)
            else:
                results['sft_outputs'].append("N/A")

            # GRPO 模型
            grpo_output = ""
            if self.grpo_model:
                print("\n📌 GRPO Model:")
                grpo_output = self.generate_response(self.grpo_model, question)
                results['grpo_outputs'].append(grpo_output)
                print(f"{grpo_output[:200]}..." if len(grpo_output) > 200 else grpo_output)
            else:
                results['grpo_outputs'].append("N/A")

            # 添加到对比器
            comparator.add_comparison(question, base_output, sft_output, grpo_output)

        # 保存结果
        print("\n" + "="*70)
        print("💾 Saving Results...")
        print("="*70)

        results_path = output_dir / "evaluation_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"✓ Results saved to {results_path}")

        # 保存对比表格
        comparator.save_comparison_table()

        return results

    def compute_metrics(self, results: Dict, references: List[str] = None) -> Dict:
        """
        计算评估指标

        Args:
            results: 评估结果
            references: 参考答案（可选）

        Returns:
            指标字典
        """
        metrics = {
            'base': {},
            'sft': {},
            'grpo': {}
        }

        # 计算平均答案长度
        for model_name in ['base', 'sft', 'grpo']:
            outputs_key = f"{model_name}_outputs"
            if outputs_key in results:
                outputs = [o for o in results[outputs_key] if o and o != "N/A"]
                if outputs:
                    lengths = [len(o) for o in outputs]
                    metrics[model_name]['avg_length'] = np.mean(lengths)
                    metrics[model_name]['num_responses'] = len(outputs)

        # 如果有参考答案，可以计算更多指标（如 ROUGE, BLEU 等）
        # 这里暂时只计算基本统计

        return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare models")
    parser.add_argument("--base-model", type=str,
                       default="deepseek-ai/deepseek-math-7b-instruct",
                       help="Base model name")
    parser.add_argument("--sft-model", type=str,
                       default="outputs/sft/final_model",
                       help="SFT model path")
    parser.add_argument("--grpo-model", type=str,
                       default="outputs/grpo/final_model",
                       help="GRPO model path")
    parser.add_argument("--test-data", type=str,
                       default="data/training_data/train_flattened.json",
                       help="Test data path")
    parser.add_argument("--num-samples", type=int, default=5,
                       help="Number of test samples to evaluate")
    parser.add_argument("--output-dir", type=str,
                       default="evaluation_results",
                       help="Output directory")

    args = parser.parse_args()

    # 加载测试数据
    print(f"📂 Loading test data from {args.test_data}...")
    with open(args.test_data, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # 随机选择测试样本
    np.random.seed(42)
    if len(test_data) > args.num_samples:
        indices = np.random.choice(len(test_data), args.num_samples, replace=False)
        test_samples = [test_data[i] for i in indices]
    else:
        test_samples = test_data

    questions = [sample['question'] for sample in test_samples]
    references = [sample['answer'] for sample in test_samples]

    print(f"✓ Loaded {len(questions)} test questions\n")

    # 创建评估器
    evaluator = ModelEvaluator(
        base_model_name=args.base_model,
        sft_model_path=args.sft_model,
        grpo_model_path=args.grpo_model
    )

    # 加载模型
    evaluator.load_models()

    # 评估
    results = evaluator.evaluate_on_questions(questions, args.output_dir)

    # 计算指标
    print("\n" + "="*70)
    print("📊 Computing Metrics...")
    print("="*70)
    metrics = evaluator.compute_metrics(results, references)

    # 打印指标
    for model_name, model_metrics in metrics.items():
        if model_metrics:
            print(f"\n{model_name.upper()} Model:")
            for metric_name, value in model_metrics.items():
                print(f"  {metric_name}: {value}")

    # 保存指标
    metrics_path = Path(args.output_dir) / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Metrics saved to {metrics_path}")

    # 生成对比图
    comparator = ModelComparator(args.output_dir)
    comparator.plot_comparison_metrics(metrics, save=True, show=False)

    print("\n" + "="*70)
    print("✅ Evaluation Complete!")
    print("="*70)
    print(f"📁 Results saved to: {args.output_dir}")
    print(f"   - evaluation_results.json: 完整的模型输出")
    print(f"   - model_comparison.csv: 对比表格")
    print(f"   - model_comparison.md: Markdown 格式对比")
    print(f"   - metrics.json: 评估指标")
    print(f"   - metrics_comparison.png: 指标对比图")


if __name__ == "__main__":
    main()

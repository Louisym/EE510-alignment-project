"""
DeepSeek-Math模型加载器，支持4bit量化
专门优化数学推理能力，适合概率论问答
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    pipeline
)
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LlamaModelLoader:
    def __init__(self, model_name: str = "deepseek-ai/deepseek-math-7b-instruct", use_4bit: bool = True):
        """
        初始化模型加载器
        
        Args:
            model_name: HuggingFace模型名称
            use_4bit: 是否使用4bit量化
        """
        self.model_name = model_name
        self.use_4bit = use_4bit
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
    def setup_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """设置4bit量化配置"""
        if not self.use_4bit:
            return None
            
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    
    def load_model_and_tokenizer(self):
        """加载模型和tokenizer"""
        logger.info(f"开始加载模型: {self.model_name}")
        
        # 设置量化配置
        quantization_config = self.setup_quantization_config()
        
        # 加载tokenizer
        logger.info("加载tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        logger.info("加载模型...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # 创建生成pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        logger.info("模型加载完成！")
        
    def generate_response(self, prompt: str, max_length: int = 512, temperature: float = 0.7) -> str:
        """生成回复"""
        if self.pipeline is None:
            raise RuntimeError("模型未加载，请先调用load_model_and_tokenizer()")
        
        # 格式化prompt为chat格式
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # 使用tokenizer的chat template
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 生成回复
        outputs = self.pipeline(
            formatted_prompt,
            max_new_tokens=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # 提取生成的文本
        generated_text = outputs[0]["generated_text"]
        
        # 移除prompt部分
        response = generated_text[len(formatted_prompt):]
        
        return response.strip()
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        if self.model is None:
            return {"status": "模型未加载"}
        
        # 计算模型参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        info = {
            "model_name": self.model_name,
            "quantization": "4-bit" if self.use_4bit else "None",
            "total_parameters": f"{total_params:,}",
            "trainable_parameters": f"{trainable_params:,}",
            "device": str(next(self.model.parameters()).device),
            "dtype": str(next(self.model.parameters()).dtype)
        }
        
        return info

if __name__ == "__main__":
    # 测试模型加载
    loader = LlamaModelLoader()
    
    try:
        loader.load_model_and_tokenizer()
        
        # 显示模型信息
        info = loader.get_model_info()
        print("\n=== 模型信息 ===")
        for key, value in info.items():
            print(f"{key}: {value}")
        
        # 测试生成
        test_prompt = "什么是概率论中的条件概率？请简要解释。"
        print(f"\n=== 测试生成 ===")
        print(f"问题: {test_prompt}")
        print("回答: ", end="", flush=True)
        
        response = loader.generate_response(test_prompt, max_length=256)
        print(response)
        
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        print(f"错误: {e}")
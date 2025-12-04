"""
概率论 QA 系统 - Gradio 前端界面
整合 RAG + GRPO + 模型对比功能
"""
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import json
import time
from pathlib import Path
import sys

# 添加路径
sys.path.append('.')
from src.vector_database import VectorDatabase

class ProbabilityQASystem:
    """概率论 QA 系统"""

    def __init__(self):
        self.tokenizer = None
        self.base_model = None
        self.grpo_model = None
        self.vector_db = None
        self.is_loaded = False

    def load_models(self, progress=gr.Progress()):
        """加载模型和系统组件"""
        if self.is_loaded:
            return "✅ 系统已加载"

        progress(0, desc="初始化中...")

        # 1. 加载 tokenizer
        progress(0.2, desc="加载 Tokenizer...")
        model_name = "Qwen/Qwen2.5-Math-7B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side='left'
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2. 加载基础模型（4-bit）
        progress(0.4, desc="加载基础模型（4-bit量化）...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )

        # 3. 加载 GRPO 模型
        progress(0.6, desc="加载 GRPO 微调模型...")
        grpo_path = "outputs/grpo/final_model"
        if Path(grpo_path).exists():
            self.grpo_model = PeftModel.from_pretrained(self.base_model, grpo_path)
        else:
            return "❌ GRPO 模型未找到"

        # 4. 初始化向量数据库
        progress(0.8, desc="初始化向量数据库...")
        self.vector_db = VectorDatabase(
            db_path="./data/chroma_db",
            embedding_model="BAAI/bge-base-en-v1.5"
        )
        self.vector_db.initialize()

        progress(1.0, desc="加载完成！")
        self.is_loaded = True

        db_info = self.vector_db.get_collection_info()
        return f"""✅ **系统加载成功！**

📊 **系统信息**:
- 基础模型: Qwen2.5-Math-7B-Instruct (4-bit)
- 微调模型: GRPO (强化学习对齐)
- 知识库: {db_info.get('count', 0)} 个文档片段
- 嵌入模型: BGE-base-en-v1.5

🎉 现在可以开始提问了！"""

    def retrieve_context(self, query, top_k=3):
        """检索相关上下文"""
        if not self.vector_db:
            return [], []

        results = self.vector_db.search(query, n_results=top_k)
        contexts = [r['text'] for r in results]
        return contexts, results

    def generate_answer(self, model, query, contexts=None, max_length=512):
        """生成答案"""
        # 构建 prompt
        if contexts:
            context_text = "\n\n".join([f"参考 {i+1}: {ctx}" for i, ctx in enumerate(contexts)])
            prompt = f"""你是一个专业的概率论学习助手。请基于提供的参考资料回答问题。

参考资料:
{context_text}

问题: {query}

要求:
1. 基于参考资料提供准确的数学答案
2. 包含必要的公式推导和证明步骤
3. 使用严谨的数学语言和符号

答案:"""
        else:
            prompt = f"""你是一个专业的概率论学习助手。

问题: {query}

要求:
1. 提供准确的数学定义和解答
2. 包含清晰的推导过程和证明步骤
3. 使用标准的数学符号和严谨的表达

答案:"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取答案部分
        if "答案:" in generated_text:
            answer = generated_text.split("答案:")[-1].strip()
        else:
            answer = generated_text[len(prompt):].strip()

        return answer

    def answer_question(self, question, use_rag=True, progress=gr.Progress()):
        """回答问题（主界面）"""
        if not self.is_loaded:
            return "❌ 请先点击'加载系统'按钮初始化系统", "", ""

        if not question.strip():
            return "⚠️ 请输入问题", "", ""

        start_time = time.time()

        # 检索上下文
        contexts = []
        retrieval_info = ""
        if use_rag:
            progress(0.3, desc="检索相关文档...")
            contexts, results = self.retrieve_context(question, top_k=3)

            retrieval_info = "### 📚 检索到的相关文档\n\n"
            for i, (ctx, result) in enumerate(zip(contexts, results), 1):
                source = result.get('metadata', {}).get('source', 'unknown')
                distance = result.get('distance', 0)
                relevance = max(0, (1 - distance) * 100)
                retrieval_info += f"**[{i}] 相关度: {relevance:.1f}%** | 来源: {source}\n\n"
                retrieval_info += f"```\n{ctx[:300]}...\n```\n\n"

        # 生成答案
        progress(0.7, desc="生成答案中...")
        answer = self.generate_answer(
            self.grpo_model,
            question,
            contexts if use_rag else None
        )

        elapsed_time = time.time() - start_time

        # 格式化输出
        answer_text = f"""### 💡 回答

{answer}

---
⏱️ 生成时间: {elapsed_time:.2f}秒 | {"🔍 使用 RAG 增强" if use_rag else "🚀 直接生成"}
"""

        return answer_text, retrieval_info, f"✅ 回答完成（耗时 {elapsed_time:.2f}秒）"

    def compare_models(self, question, progress=gr.Progress()):
        """对比不同模型"""
        if not self.is_loaded:
            return "❌ 请先加载系统", "", "", ""

        if not question.strip():
            return "⚠️ 请输入问题", "", "", ""

        progress(0.2, desc="检索文档...")
        contexts, _ = self.retrieve_context(question, top_k=3)

        # 1. Base Model (无 RAG)
        progress(0.3, desc="Base Model 生成中...")
        base_answer = self.generate_answer(self.base_model, question, None, max_length=256)

        # 2. GRPO Model (无 RAG)
        progress(0.5, desc="GRPO Model 生成中...")
        grpo_no_rag = self.generate_answer(self.grpo_model, question, None, max_length=256)

        # 3. GRPO + RAG
        progress(0.7, desc="GRPO + RAG 生成中...")
        grpo_with_rag = self.generate_answer(self.grpo_model, question, contexts, max_length=256)

        # 格式化输出
        base_output = f"""### 🔵 Base Model（未微调）

{base_answer}
"""

        grpo_output = f"""### 🟢 GRPO Model（强化学习微调）

{grpo_no_rag}
"""

        rag_output = f"""### 🌟 GRPO + RAG（最佳配置）

{grpo_with_rag}
"""

        progress(1.0, desc="对比完成!")
        return base_output, grpo_output, rag_output, "✅ 三个模型对比完成"

    def get_system_stats(self):
        """获取系统统计信息"""
        if not self.is_loaded:
            return "系统未加载"

        # 获取数据库信息
        db_info = self.vector_db.get_collection_info()

        # 加载评估结果
        eval_file = "evaluation_results.json"
        if Path(eval_file).exists():
            with open(eval_file, 'r', encoding='utf-8') as f:
                eval_results = json.load(f)

            stats_text = f"""## 📊 系统统计信息

### 模型配置
- **基础模型**: Qwen2.5-Math-7B-Instruct
- **量化方式**: 4-bit NF4
- **微调方法**: LoRA (Rank 16) + GRPO
- **训练数据**: 81 个概率论 QA 对
- **训练时长**: 3.5 小时

### 知识库信息
- **文档数量**: {db_info.get('count', 0)} 个片段
- **嵌入维度**: {db_info.get('embedding_dim', 'N/A')}
- **嵌入模型**: BGE-base-en-v1.5

### 模型性能（测试集评估）

| 模型 | Average Reward | vs Base 提升 |
|------|---------------|--------------|
"""
            for result in eval_results:
                model_name = result['model_name']
                avg_reward = result['avg_reward']
                base_reward = eval_results[0]['avg_reward']
                improvement = ((avg_reward - base_reward) / base_reward * 100) if model_name != 'Base Model' else 0

                if model_name == 'Base Model':
                    stats_text += f"| {model_name} | {avg_reward:.4f} | - |\n"
                else:
                    stats_text += f"| {model_name} | {avg_reward:.4f} | **+{improvement:.1f}%** |\n"

            stats_text += f"""
### 关键发现
- ✅ **SVD 初始化** 相比随机初始化提升 **19.5%**
- ✅ **GRPO 对齐** 实现 **55.5%** 整体性能提升
- ✅ **完整 Pipeline** 达到最佳效果

### 技术栈
- 🔧 Framework: PyTorch + Transformers + PEFT
- 🗄️ Vector DB: ChromaDB
- 🎨 Frontend: Gradio
- 📊 Visualization: Matplotlib
"""
        else:
            stats_text = f"""## 📊 系统统计信息

### 知识库信息
- **文档数量**: {db_info.get('count', 0)} 个片段
- **嵌入模型**: BGE-base-en-v1.5

### 模型配置
- **基础模型**: Qwen2.5-Math-7B-Instruct (4-bit)
- **微调模型**: GRPO (强化学习对齐)

_更多统计信息将在评估完成后显示_
"""

        return stats_text


# 初始化系统
qa_system = ProbabilityQASystem()

# 创建 Gradio 界面
with gr.Blocks(title="概率论智能问答系统") as demo:

    # 页面标题
    gr.Markdown("""
    <div class="main-header">
        <h1>🎓 概率论智能问答系统</h1>
        <p>基于 RAG + GRPO 的智能数学学习助手</p>
        <p style="font-size: 14px; opacity: 0.9;">
            Qwen2.5-Math-7B + SVD-LoRA + GRPO + ChromaDB
        </p>
    </div>
    """)

    # 系统状态
    with gr.Row():
        load_btn = gr.Button("🚀 加载系统", variant="primary", size="lg")
        status_text = gr.Textbox(
            label="系统状态",
            value="⏳ 点击'加载系统'按钮开始初始化...",
            interactive=False,
            lines=8
        )

    # 主要功能标签页
    with gr.Tabs():
        # Tab 1: 主问答系统
        with gr.Tab("💬 智能问答"):
            gr.Markdown("### 向我提问任何概率论相关的问题！")

            with gr.Row():
                with gr.Column(scale=2):
                    question_input = gr.Textbox(
                        label="📝 您的问题",
                        placeholder="例如：什么是条件概率？中心极限定理的内容是什么？",
                        lines=3
                    )
                    use_rag_checkbox = gr.Checkbox(
                        label="🔍 使用 RAG 增强（推荐）",
                        value=True,
                        info="启用后会检索相关文档来辅助回答"
                    )
                    submit_btn = gr.Button("🚀 获取答案", variant="primary", size="lg")

                with gr.Column(scale=1):
                    qa_status = gr.Textbox(label="状态", interactive=False, lines=2)

            answer_output = gr.Markdown(label="💡 回答")
            retrieval_output = gr.Markdown(label="📚 检索到的文档")

        # Tab 2: 模型对比
        with gr.Tab("📊 模型对比"):
            gr.Markdown("""
            ### 对比不同模型的回答效果
            同时生成 Base Model、GRPO Model 和 GRPO+RAG 的答案，直观比较性能差异。
            """)

            compare_question = gr.Textbox(
                label="📝 测试问题",
                placeholder="输入问题来对比不同模型的表现...",
                lines=2
            )
            compare_btn = gr.Button("🔬 开始对比", variant="primary", size="lg")
            compare_status = gr.Textbox(label="对比状态", interactive=False, lines=1)

            with gr.Row():
                with gr.Column():
                    base_output = gr.Markdown(label="Base Model")
                with gr.Column():
                    grpo_output = gr.Markdown(label="GRPO Model")
                with gr.Column():
                    rag_output = gr.Markdown(label="GRPO + RAG")

        # Tab 3: 系统信息
        with gr.Tab("📈 系统统计"):
            gr.Markdown("### 查看系统配置和性能指标")
            refresh_stats_btn = gr.Button("🔄 刷新统计", variant="secondary")
            stats_output = gr.Markdown(value="点击'刷新统计'查看系统信息...")

        # Tab 4: 关于
        with gr.Tab("ℹ️ 关于"):
            gr.Markdown("""
            ## 🎓 概率论智能问答系统

            ### 系统架构

            ```
            用户问题 → RAG检索 → 上下文增强 → GRPO模型 → 高质量答案
                ↓           ↓              ↓            ↓
            ChromaDB   BGE嵌入      Qwen2.5-Math   LoRA微调
            ```

            ### 核心技术

            1. **RAG (检索增强生成)**
               - 使用 ChromaDB 向量数据库
               - BGE-base-en-v1.5 嵌入模型
               - 语义检索相关文档片段

            2. **SVD-LoRA 初始化**
               - SVD分解提取低秩结构
               - 智能初始化 LoRA 权重
               - 相比随机初始化提升 19.5%

            3. **GRPO (Group Relative Policy Optimization)**
               - 强化学习对齐
               - 启发式奖励模型
               - 整体性能提升 55.5%

            ### 训练数据
            - 81 个高质量概率论 QA 对
            - 涵盖测度论、随机过程、概率基础
            - 训练时长：3.5 小时

            ### 性能指标
            | 模型 | 测试集性能 | 提升 |
            |------|-----------|------|
            | Base Model | 0.231 | - |
            | SFT Random | 0.253 | +9.5% |
            | SFT SVD | 0.302 | +30.8% |
            | **GRPO** | **0.359** | **+55.5%** |

            ### 开发团队
            - 🏫 课程：EE510 概率论
            - 📅 学期：Spring 2025
            - 🔧 技术栈：PyTorch, Transformers, PEFT, Gradio, ChromaDB

            ---

            <div style="text-align: center; padding: 20px; color: #666;">
                <p>💡 <strong>提示</strong>: 使用 RAG 增强可以获得更准确、更有依据的答案</p>
                <p>⚡ <strong>性能</strong>: 首次加载需要 1-2 分钟，后续查询 2-5 秒</p>
            </div>
            """)

    # 示例问题
    gr.Examples(
        examples=[
            "什么是条件概率？",
            "中心极限定理的内容是什么？",
            "解释马尔可夫性质",
            "什么是σ代数？",
            "布朗运动有什么特点？",
            "鞅的定义是什么？"
        ],
        inputs=question_input,
        label="💡 示例问题"
    )

    # 事件绑定
    load_btn.click(
        fn=qa_system.load_models,
        outputs=[status_text]
    )

    submit_btn.click(
        fn=qa_system.answer_question,
        inputs=[question_input, use_rag_checkbox],
        outputs=[answer_output, retrieval_output, qa_status]
    )

    compare_btn.click(
        fn=qa_system.compare_models,
        inputs=[compare_question],
        outputs=[base_output, grpo_output, rag_output, compare_status]
    )

    refresh_stats_btn.click(
        fn=qa_system.get_system_stats,
        outputs=[stats_output]
    )

if __name__ == "__main__":
    print("="*80)
    print("🚀 启动概率论智能问答系统")
    print("="*80)
    print("\n📝 系统功能:")
    print("  1. 💬 智能问答 - RAG增强的概率论问答")
    print("  2. 📊 模型对比 - 对比不同模型的性能")
    print("  3. 📈 系统统计 - 查看训练和评估数据")
    print("\n⚠️  首次使用请先点击'加载系统'按钮初始化")
    print("\n🌐 访问地址: http://127.0.0.1:7860")
    print("="*80)

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )

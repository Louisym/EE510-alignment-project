"""
Probability Theory QA System - Gradio Frontend
Integrated RAG + GRPO + Model Comparison
"""
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import json
import time
from pathlib import Path
import sys

# Add path
sys.path.append('.')
from src.vector_database import VectorDatabase

class ProbabilityQASystem:
    """Probability Theory QA System"""

    def __init__(self):
        self.tokenizer = None
        self.base_model = None
        self.grpo_model = None
        self.vector_db = None
        self.is_loaded = False

    def load_models(self, progress=gr.Progress()):
        """Load models and system components"""
        if self.is_loaded:
            return "✅ System already loaded"

        progress(0, desc="Initializing...")

        # 1. Load tokenizer
        progress(0.2, desc="Loading Tokenizer...")
        model_name = "Qwen/Qwen2.5-Math-7B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side='left'
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2. Load base model (4-bit)
        progress(0.4, desc="Loading Base Model (4-bit quantization)...")
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

        # 3. Load GRPO model
        progress(0.6, desc="Loading GRPO Fine-tuned Model...")
        grpo_path = "outputs/grpo/final_model"
        if Path(grpo_path).exists():
            self.grpo_model = PeftModel.from_pretrained(self.base_model, grpo_path)
        else:
            return "❌ GRPO model not found"

        # 4. Initialize vector database
        progress(0.8, desc="Initializing Vector Database...")
        self.vector_db = VectorDatabase(
            db_path="./data/chroma_db",
            embedding_model="BAAI/bge-base-en-v1.5"
        )
        self.vector_db.initialize()

        progress(1.0, desc="Loading Complete!")
        self.is_loaded = True

        db_info = self.vector_db.get_collection_info()
        return f"""✅ **System Loaded Successfully!**

📊 **System Information**:
- Base Model: Qwen2.5-Math-7B-Instruct (4-bit)
- Fine-tuned Model: GRPO (RL Alignment)
- Knowledge Base: {db_info.get('count', 0)} document chunks
- Embedding Model: BGE-base-en-v1.5

🎉 Ready to answer your questions!"""

    def retrieve_context(self, query, top_k=3):
        """Retrieve relevant context"""
        if not self.vector_db:
            return [], []

        results = self.vector_db.search(query, n_results=top_k)
        contexts = [r['text'] for r in results]
        return contexts, results

    def generate_answer(self, model, query, contexts=None, max_length=512):
        """Generate answer"""
        # Build prompt
        if contexts:
            context_text = "\n\n".join([f"Reference {i+1}: {ctx}" for i, ctx in enumerate(contexts)])
            prompt = f"""You are a professional probability theory learning assistant. Please answer questions based on the provided reference materials.

Reference Materials:
{context_text}

Question: {query}

Requirements:
1. Provide accurate mathematical answers based on the reference materials
2. Include necessary formula derivations and proof steps
3. Use rigorous mathematical language and notation
4. Answer in English

Answer:"""
        else:
            prompt = f"""You are a professional probability theory learning assistant.

Question: {query}

Requirements:
1. Provide accurate mathematical definitions and solutions
2. Include clear derivation processes and proof steps
3. Use standard mathematical notation and rigorous expressions
4. Answer in English

Answer:"""

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

        # Extract answer
        if "Answer:" in generated_text:
            answer = generated_text.split("Answer:")[-1].strip()
        else:
            answer = generated_text[len(prompt):].strip()

        return answer

    def answer_question(self, question, use_rag=True, progress=gr.Progress()):
        """Answer question (main interface)"""
        if not self.is_loaded:
            return "❌ Please click 'Load System' button first", "", ""

        if not question.strip():
            return "⚠️ Please enter a question", "", ""

        start_time = time.time()

        # Retrieve context
        contexts = []
        retrieval_info = ""
        if use_rag:
            progress(0.3, desc="Retrieving relevant documents...")
            contexts, results = self.retrieve_context(question, top_k=3)

            retrieval_info = "### 📚 Retrieved Documents\n\n"
            for i, (ctx, result) in enumerate(zip(contexts, results), 1):
                source = result.get('metadata', {}).get('source', 'unknown')
                distance = result.get('distance', 0)
                relevance = max(0, (1 - distance) * 100)
                retrieval_info += f"**[{i}] Relevance: {relevance:.1f}%** | Source: {source}\n\n"
                retrieval_info += f"```\n{ctx[:300]}...\n```\n\n"

        # Generate answer
        progress(0.7, desc="Generating answer...")
        answer = self.generate_answer(
            self.grpo_model,
            question,
            contexts if use_rag else None
        )

        elapsed_time = time.time() - start_time

        # Format output
        answer_text = f"""### 💡 Answer

{answer}

---
⏱️ Generation Time: {elapsed_time:.2f}s | {"🔍 RAG Enhanced" if use_rag else "🚀 Direct Generation"}
"""

        return answer_text, retrieval_info, f"✅ Answer complete ({elapsed_time:.2f}s)"

    def compare_models(self, question, progress=gr.Progress()):
        """Compare different models"""
        if not self.is_loaded:
            return "❌ Please load system first", "", "", ""

        if not question.strip():
            return "⚠️ Please enter a question", "", "", ""

        progress(0.2, desc="Retrieving documents...")
        contexts, _ = self.retrieve_context(question, top_k=3)

        # 1. Base Model (no RAG)
        progress(0.3, desc="Base Model generating...")
        base_answer = self.generate_answer(self.base_model, question, None, max_length=256)

        # 2. GRPO Model (no RAG)
        progress(0.5, desc="GRPO Model generating...")
        grpo_no_rag = self.generate_answer(self.grpo_model, question, None, max_length=256)

        # 3. GRPO + RAG
        progress(0.7, desc="GRPO + RAG generating...")
        grpo_with_rag = self.generate_answer(self.grpo_model, question, contexts, max_length=256)

        # Format output
        base_output = f"""### 🔵 Base Model (Unfine-tuned)

{base_answer}
"""

        grpo_output = f"""### 🟢 GRPO Model (RL Fine-tuned)

{grpo_no_rag}
"""

        rag_output = f"""### 🌟 GRPO + RAG (Best Configuration)

{grpo_with_rag}
"""

        progress(1.0, desc="Comparison complete!")
        return base_output, grpo_output, rag_output, "✅ Three-model comparison complete"

    def get_system_stats(self):
        """Get system statistics"""
        if not self.is_loaded:
            return "System not loaded"

        # Get database info
        db_info = self.vector_db.get_collection_info()

        # Load evaluation results
        eval_file = "evaluation_results.json"
        if Path(eval_file).exists():
            with open(eval_file, 'r', encoding='utf-8') as f:
                eval_results = json.load(f)

            stats_text = f"""## 📊 System Statistics

### Model Configuration
- **Base Model**: Qwen2.5-Math-7B-Instruct
- **Quantization**: 4-bit NF4
- **Fine-tuning Method**: LoRA (Rank 16) + GRPO
- **Training Data**: 81 Probability Theory QA pairs
- **Training Duration**: 3.5 hours

### Knowledge Base Information
- **Document Count**: {db_info.get('count', 0)} chunks
- **Embedding Dimension**: {db_info.get('embedding_dim', 'N/A')}
- **Embedding Model**: BGE-base-en-v1.5

### Model Performance (Test Set Evaluation)

| Model | Average Reward | vs Base Improvement |
|------|---------------|---------------------|
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
### Key Findings
- ✅ **SVD Initialization** improved **19.5%** over random initialization
- ✅ **GRPO Alignment** achieved **55.5%** overall performance gain
- ✅ **Complete Pipeline** reached best performance

### Tech Stack
- 🔧 Framework: PyTorch + Transformers + PEFT
- 🗄️ Vector DB: ChromaDB
- 🎨 Frontend: Gradio
- 📊 Visualization: Matplotlib
"""
        else:
            stats_text = f"""## 📊 System Statistics

### Knowledge Base Information
- **Document Count**: {db_info.get('count', 0)} chunks
- **Embedding Model**: BGE-base-en-v1.5

### Model Configuration
- **Base Model**: Qwen2.5-Math-7B-Instruct (4-bit)
- **Fine-tuned Model**: GRPO (RL Alignment)

_More statistics will be available after evaluation_
"""

        return stats_text


# Initialize system
qa_system = ProbabilityQASystem()

# Create Gradio interface
with gr.Blocks(title="Probability Theory QA System") as demo:

    # Page header
    gr.Markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;">
        <h1>🎓 Probability Theory QA System</h1>
        <p>Intelligent Math Learning Assistant based on RAG + GRPO</p>
        <p style="font-size: 14px; opacity: 0.9;">
            Qwen2.5-Math-7B + SVD-LoRA + GRPO + ChromaDB
        </p>
    </div>
    """)

    # System status
    with gr.Row():
        load_btn = gr.Button("🚀 Load System", variant="primary", size="lg")
        status_text = gr.Textbox(
            label="System Status",
            value="⏳ Click 'Load System' button to initialize...",
            interactive=False,
            lines=8
        )

    # Main tabs
    with gr.Tabs():
        # Tab 1: Main QA
        with gr.Tab("💬 Q&A"):
            gr.Markdown("### Ask me anything about probability theory!")

            with gr.Row():
                with gr.Column(scale=2):
                    question_input = gr.Textbox(
                        label="📝 Your Question",
                        placeholder="e.g., What is conditional probability? What is the central limit theorem?",
                        lines=3
                    )
                    use_rag_checkbox = gr.Checkbox(
                        label="🔍 Use RAG Enhancement (Recommended)",
                        value=True,
                        info="Enable to retrieve relevant documents to assist answering"
                    )
                    submit_btn = gr.Button("🚀 Get Answer", variant="primary", size="lg")

                with gr.Column(scale=1):
                    qa_status = gr.Textbox(label="Status", interactive=False, lines=2)

            answer_output = gr.Markdown(label="💡 Answer")
            retrieval_output = gr.Markdown(label="📚 Retrieved Documents")

        # Tab 2: Model comparison
        with gr.Tab("📊 Model Comparison"):
            gr.Markdown("""
            ### Compare Performance of Different Models
            Generate answers from Base Model, GRPO Model, and GRPO+RAG simultaneously to compare performance.
            """)

            compare_question = gr.Textbox(
                label="📝 Test Question",
                placeholder="Enter a question to compare model performance...",
                lines=2
            )
            compare_btn = gr.Button("🔬 Start Comparison", variant="primary", size="lg")
            compare_status = gr.Textbox(label="Comparison Status", interactive=False, lines=1)

            with gr.Row():
                with gr.Column():
                    base_output = gr.Markdown(label="Base Model")
                with gr.Column():
                    grpo_output = gr.Markdown(label="GRPO Model")
                with gr.Column():
                    rag_output = gr.Markdown(label="GRPO + RAG")

        # Tab 3: System stats
        with gr.Tab("📈 System Statistics"):
            gr.Markdown("### View System Configuration and Performance Metrics")
            refresh_stats_btn = gr.Button("🔄 Refresh Statistics", variant="secondary")
            stats_output = gr.Markdown(value="Click 'Refresh Statistics' to view system info...")

        # Tab 4: About
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## 🎓 Probability Theory QA System

            ### System Architecture

            ```
            User Question → RAG Retrieval → Context Enhancement → GRPO Model → High-quality Answer
                ↓              ↓                  ↓                   ↓
            ChromaDB      BGE Embedding      Qwen2.5-Math       LoRA Fine-tuning
            ```

            ### Core Technologies

            1. **RAG (Retrieval-Augmented Generation)**
               - ChromaDB vector database
               - BGE-base-en-v1.5 embedding model
               - Semantic search for relevant document chunks

            2. **SVD-LoRA Initialization**
               - SVD decomposition to extract low-rank structure
               - Intelligent LoRA weight initialization
               - 19.5% improvement over random initialization

            3. **GRPO (Group Relative Policy Optimization)**
               - Reinforcement learning alignment
               - Heuristic reward model
               - 55.5% overall performance improvement

            ### Training Data
            - 81 high-quality probability theory QA pairs
            - Covering measure theory, stochastic processes, probability fundamentals
            - Training duration: 3.5 hours

            ### Performance Metrics
            | Model | Test Performance | Improvement |
            |------|------------------|-------------|
            | Base Model | 0.231 | - |
            | SFT Random | 0.253 | +9.5% |
            | SFT SVD | 0.302 | +30.8% |
            | **GRPO** | **0.359** | **+55.5%** |

            ### Development Team
            - 🏫 Course: EE510 Probability Theory
            - 📅 Semester: Spring 2025
            - 🔧 Tech Stack: PyTorch, Transformers, PEFT, Gradio, ChromaDB

            ---

            <div style="text-align: center; padding: 20px; color: #666;">
                <p>💡 <strong>Tip</strong>: Using RAG enhancement provides more accurate and well-founded answers</p>
                <p>⚡ <strong>Performance</strong>: First load takes 1-2 minutes, subsequent queries 2-5 seconds</p>
            </div>
            """)

    # Example questions
    gr.Examples(
        examples=[
            "What is conditional probability?",
            "What is the central limit theorem?",
            "Explain the Markov property",
            "What is a σ-algebra?",
            "What are the characteristics of Brownian motion?",
            "What is the definition of a martingale?"
        ],
        inputs=question_input,
        label="💡 Example Questions"
    )

    # Event bindings
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
    print("🚀 Starting Probability Theory QA System")
    print("="*80)
    print("\n📝 System Features:")
    print("  1. 💬 Q&A - RAG-enhanced probability theory questions")
    print("  2. 📊 Model Comparison - Compare different model performance")
    print("  3. 📈 System Statistics - View training and evaluation data")
    print("\n⚠️  First-time users please click 'Load System' button to initialize")
    print("\n🌐 Access URL: http://127.0.0.1:7860")
    print("="*80)

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )

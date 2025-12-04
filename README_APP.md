# 🎓 Probability Theory Intelligent Q&A System

RAG + GRPO Enhanced Mathematical Learning Assistant

## 🌟 System Features

### Core Functionality
- **💬 Intelligent Q&A**: Supports questions on probability theory, measure theory, and stochastic processes
- **🔍 RAG Enhancement**: Automatically retrieves relevant documents for evidence-based answers
- **📊 Model Comparison**: Visually compare Base Model, GRPO, and RAG+GRPO performance
- **📈 Performance Statistics**: View training data and evaluation metrics

### Technology Stack
- **Base Model**: Qwen2.5-Math-7B-Instruct (4-bit quantized)
- **Fine-tuning Method**: SVD-LoRA + GRPO (reinforcement learning alignment)
- **Retrieval System**: ChromaDB + BGE embedding model
- **Frontend Framework**: Gradio (beautiful and easy-to-use web interface)

## 🚀 Quick Start

### 1. Launch Application

```bash
./start_app.sh
```

Or manually:
```bash
python3 app.py
```

### 2. Access Interface

Open your browser and visit: **http://127.0.0.1:7860**

### 3. Usage Steps

1. **Load System**: First time users please click "🚀 Load System" button
   - Loading time: ~1-2 minutes
   - Loads: Model + Vector Database

2. **Start Asking**: Enter questions in the "Q&A" tab
   - Recommend checking "Use RAG Enhancement"
   - Click "Get Answer"
   - Wait 2-5 seconds for response

3. **Compare Models**: In the "Model Comparison" tab
   - View answers from all 3 models simultaneously
   - Directly compare performance differences

4. **View Statistics**: In the "System Statistics" tab
   - Check model configuration and training data
   - View test set evaluation results

## 📊 System Performance

### Test Set Evaluation Results

| Model | Average Reward | Improvement |
|-------|---------------|-------------|
| Base Model | 0.231 | - |
| SFT Random-init | 0.253 | +9.5% |
| SFT SVD-init | 0.302 | +30.8% |
| **GRPO** | **0.359** | **+55.5%** |

### Key Improvements
- ✅ SVD initialization improves **19.5%** over random initialization
- ✅ GRPO alignment achieves **55.5%** overall performance improvement
- ✅ RAG retrieval significantly enhances answer quality and reliability

## 💡 Usage Examples

### Example Questions
```
1. What is conditional probability?
2. What is the content of the Central Limit Theorem?
3. Explain the Markov property
4. What is a σ-algebra?
5. What are the characteristics of Brownian motion?
6. What is the definition of a martingale?
```

### Expected Results
- **Accurate Answers**: Based on probability theory textbooks and training data
- **Evidence-based**: Shows retrieved relevant documents
- **Mathematically Rigorous**: Includes formulas and definitions
- **Fast Response**: 2-5 seconds to generate answer

## 🎨 Interface Description

### Tab 1: 💬 Q&A
Main question-answering interface, supports:
- Enter any probability theory question
- Optional RAG enhancement
- Display answers and retrieved documents
- Show generation time

### Tab 2: 📊 Model Comparison
Compare answers from three models:
- Base Model: Original model without fine-tuning
- GRPO Model: Fine-tuned with reinforcement learning
- GRPO + RAG: Best configuration (fine-tuning + retrieval)

### Tab 3: 📈 System Statistics
View system information:
- Model configuration
- Training data statistics
- Test set performance
- Knowledge base information

### Tab 4: ℹ️ About
System introduction:
- Architecture description
- Technical details
- Performance metrics
- Development information

## 🔧 Configuration

### Model Configuration
```python
model_name = "Qwen/Qwen2.5-Math-7B-Instruct"
quantization = "4-bit NF4"
lora_rank = 16
grpo_model_path = "outputs/grpo/final_model"
```

### Retrieval Configuration
```python
vector_db_path = "./data/chroma_db"
embedding_model = "BAAI/bge-base-en-v1.5"
top_k_retrieval = 3  # Retrieve top 3 most relevant documents
```

### Generation Configuration
```python
max_length = 512
temperature = 0.7
top_p = 0.9
```

## 📁 File Structure

```
.
├── app.py                          # Gradio application main file
├── app_en.py                       # English version
├── app_cn.py                       # Chinese version (backup)
├── start_app.sh                    # Startup script
├── outputs/grpo/final_model/       # GRPO fine-tuned model
├── data/chroma_db/                 # Vector database
├── evaluation_results.json         # Evaluation result data
├── src/
│   ├── vector_database.py          # Vector database interface
│   └── ...
└── README_APP.md                   # This file
```

## ⚠️ Important Notes

### System Requirements
- **RAM**: At least 16GB RAM
- **GPU**: CUDA-capable GPU required (at least 8GB VRAM)
- **Storage**: At least 20GB available space

### First Time Use
1. First load requires model download, may take 5-10 minutes
2. Subsequent loads only take 1-2 minutes
3. If you encounter OOM errors, try closing other programs

### Performance Optimization
- 4-bit quantization reduces VRAM usage
- Batch queries improve efficiency
- RAG retrieval adds 1-2 seconds latency

## 🐛 Troubleshooting

### Issue 1: Loading Failed
```
Solutions:
1. Confirm GRPO model path is correct: outputs/grpo/final_model/
2. Confirm vector database exists: data/chroma_db/
3. Check if GPU is available: nvidia-smi
```

### Issue 2: OOM Error
```
Solutions:
1. Close other GPU programs
2. Reduce max_length parameter
3. Restart application
```

### Issue 3: No Retrieval Results
```
Solutions:
1. Confirm vector database is initialized
2. Check if data/chroma_db/ directory exists
3. Try rebuilding knowledge base
```

## 📚 References

### Related Papers
- LoRA: Low-Rank Adaptation of Large Language Models
- GRPO: Group Relative Policy Optimization
- RAG: Retrieval-Augmented Generation

### Project Documentation
- `experiments/svd_lora/` - SVD-LoRA experiment results
- `outputs/grpo/` - GRPO training results
- `evaluation_results.json` - Complete evaluation data

## 🎯 Future Improvements

- [ ] Add more probability theory knowledge
- [ ] Support multi-turn conversations
- [ ] Add formula rendering
- [ ] Support chart generation
- [ ] Add answer quality scoring

## 👥 Development Information

- **Course**: EE510 Probability Theory
- **Semester**: Spring 2025
- **Tech Stack**: PyTorch, Transformers, PEFT, Gradio, ChromaDB

---

<div align="center">

**🎉 Enjoy! Feel free to provide feedback 🎉**

</div>

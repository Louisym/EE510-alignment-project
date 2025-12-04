# EE510 Probability Theory QA System

RAG + GRPO Enhanced Probability Theory Learning Assistant | [中文](./README.md)

## 📋 Project Overview

This is an intelligent question-answering system for probability theory, measure theory, and stochastic processes, combining Retrieval-Augmented Generation (RAG) and Group Relative Policy Optimization (GRPO) techniques to achieve high-quality mathematical Q&A capabilities.

### Key Features

- 🎯 **Progressive Training Strategy**: Base Model → SFT → SVD-LoRA → GRPO for systematic performance improvement
- 📊 **Significant Performance Gains**: **55.5%** improvement over the base model
- 🔍 **RAG-Enhanced Retrieval**: ChromaDB-based vector database for accurate knowledge retrieval
- 💡 **SVD Initialization Optimization**: **19.5%** test performance improvement over random initialization
- 🎨 **User-Friendly Web Interface**: Support for model comparison, real-time Q&A, and performance visualization

### Performance Metrics

| Model | Avg Reward | Relative Improvement |
|-------|------------|---------------------|
| Base Model | 0.231 | - |
| SFT Random-init | 0.253 | +9.5% |
| SFT SVD-init | 0.302 | +30.8% |
| **GRPO (Final)** | **0.359** | **+55.5%** ⭐ |

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Gradio Web Interface                     │
│  (Q&A | Model Comparison | Statistics | System Info)         │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴──────────────┐
        │                           │
   ┌────▼─────┐              ┌─────▼────┐
   │  GRPO    │              │   RAG    │
   │  Model   │              │  Engine  │
   └────┬─────┘              └─────┬────┘
        │                           │
        │                    ┌──────▼──────┐
        │                    │  ChromaDB   │
        │                    │ (BGE embed) │
        │                    └─────────────┘
        │
   ┌────▼─────────────────────────────┐
   │ Qwen2.5-Math-7B-Instruct (4-bit) │
   │     + LoRA (rank 16)              │
   └──────────────────────────────────┘
```

### Training Pipeline

```
1. [SFT - Supervised Fine-tuning]
   ├── Random-init LoRA → Training loss: 0.7743
   └── SVD-init LoRA    → Training loss: 0.7718 (↓0.32%)
                          Test performance: +19.5% vs Random

2. [GRPO - Reinforcement Learning Alignment]
   ├── Training: 2 epochs, 162 steps, 3.5 hours
   ├── Reward Model: 4-dimensional heuristic (length/formula/concept/structure)
   └── Final Performance: 0.359 reward (↑55.5% vs Base)

3. [RAG - Retrieval Augmentation]
   ├── Vectorization: 8 probability theory document chapters
   ├── Retrieval: Top-3 relevant passages
   └── Generation: Context-grounded accurate answers
```

## 🚀 Quick Start

### Requirements

- **Python**: 3.8+
- **GPU**: CUDA-compatible GPU with at least 8GB VRAM
- **Memory**: At least 16GB RAM
- **Storage**: At least 20GB available space

### Installation

```bash
# Clone the repository
git clone https://github.com/Louisym/EE510_alignment_project.git
cd EE510_alignment_project

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Launch Web Application

```bash
# Method 1: Using startup script
./start_app.sh

# Method 2: Direct execution
python app.py
```

Visit **http://127.0.0.1:7860** to access the interface.

**First-time use**: Click the "🚀 Load System" button to load the model (takes 1-2 minutes).

## 📂 Project Structure

```
.
├── app.py                      # Main Gradio Web application
├── app_cn.py                   # Chinese version (backup)
├── app_en.py                   # English version
├── requirements.txt            # Python dependencies
├── start_app.sh               # Startup script
│
├── src/                       # Core source code
│   ├── vector_database.py    # ChromaDB vector database
│   ├── rag_pipeline.py        # RAG retrieval pipeline
│   ├── document_processor.py  # Document processor
│   ├── model_loader.py        # Model loader
│   └── evaluation.py          # Evaluation utilities
│
├── training/                  # Training framework
│   ├── sft/                  # Supervised Fine-Tuning
│   │   ├── train_sft.py      # SFT training script
│   │   ├── trainer.py        # SFT trainer
│   │   ├── data_loader.py    # Data loader
│   │   └── config.py         # Configuration
│   │
│   └── grpo/                 # GRPO Reinforcement Learning
│       ├── train_grpo.py     # GRPO training script
│       ├── trainer.py        # GRPO trainer
│       ├── reward_model.py   # Reward model
│       ├── data_loader.py    # Data loader
│       └── config.py         # Configuration
│
├── experiments/              # Experiment code
│   └── svd_lora/            # SVD-LoRA comparison experiments
│       ├── train_lora_svd_vs_rand.py  # Comparison training
│       ├── extract_svd_from_lora.py   # SVD extraction
│       └── training_results/          # Results
│
├── data/                     # Data directory
│   ├── chroma_db/           # Vector database
│   ├── docs/                # Source documents
│   └── training_data/       # Training data (81 QA pairs)
│
└── outputs/                  # Training outputs
    └── grpo/                # GRPO training results
        ├── final_model/     # Final model
        ├── plots/           # Training curves
        └── metrics/         # Performance metrics
```

## 🎓 Training Guide

### 1. SFT Training

```bash
# Using random initialization LoRA
python training/sft/train_sft.py \
  --config default \
  --train-data data/training_data/train.json \
  --output-dir outputs/sft/random \
  --epochs 3
```

### 2. SVD-LoRA Comparison Experiment

```bash
python experiments/svd_lora/train_lora_svd_vs_rand.py \
  --base-model Qwen/Qwen2.5-Math-7B-Instruct \
  --train-data data/training_data/train.json \
  --lora-rank 16 \
  --epochs 3
```

### 3. GRPO Training

```bash
python training/grpo/train_grpo.py \
  --config from_sft \
  --sft-model outputs/sft/final_model \
  --train-data data/training_data/train.json \
  --output-dir outputs/grpo \
  --epochs 2
```

## 🔧 Core Technologies

### 1. RAG (Retrieval-Augmented Generation)

- **Vector Database**: ChromaDB (persistent storage)
- **Embedding Model**: BAAI/bge-base-en-v1.5
- **Document Chunking**: 500 characters/chunk, 50 character overlap
- **Retrieval Strategy**: Top-3 cosine similarity

### 2. SVD-LoRA Initialization

**Principle**:
- Train teacher LoRA → Extract weight matrix ΔW
- SVD decomposition: ΔW ≈ B·A
- Initialize student LoRA with B and A

**Advantage**: **19.5%** improvement over random initialization

### 3. GRPO Reward Model

4-dimensional heuristic scoring:

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Length Appropriateness | 20% | Answer length vs reference answer ratio |
| Formula Presence | 30% | Usage of mathematical symbols and formulas |
| Concept Coverage | 30% | Coverage of key terminology |
| Structural Completeness | 20% | Structure of definition/formula/example |

### 4. Model Configuration

- **Base Model**: Qwen/Qwen2.5-Math-7B-Instruct
- **Quantization**: 4-bit NF4 (BitsAndBytes)
- **LoRA Rank**: 16 (trainable params: ~1.2%)
- **VRAM Usage**: ~5GB

## 📊 Experimental Results

### SVD-LoRA Comparison

| Initialization Method | Training Loss | Test Performance | Improvement |
|----------------------|---------------|------------------|-------------|
| Random               | 0.7743        | 0.253            | -           |
| SVD                  | 0.7718        | 0.302            | **+19.5%**  |

**Key Finding**: Similar training loss (0.32% difference), but significant test performance gap (19.5%)

### GRPO Training

| Epoch | Steps | Mean Reward | Max Reward |
|-------|-------|-------------|------------|
| 1     | 81    | 0.487       | 0.679      |
| 2     | 162   | 0.480       | 0.682      |
| Best  | 135   | **0.599**   | -          |

**Loss Variance Analysis**:
- Loss variance is **592x** larger than reward variance
- This is normal (due to group-relative normalization)
- Focus on reward metrics, not loss

## ⚠️ Important Notes

### Model Weights

Large model files (*.safetensors, *.bin, *.pth) are not included in the repository due to GitHub file size limits.

**First-time run**:
1. Automatically downloads base model from HuggingFace
2. Or train your own model following the training guide

### Common Issues

```bash
# Loading failure
ls outputs/grpo/final_model/  # Check model
nvidia-smi                     # Check GPU

# No RAG results
# Rebuild vector database
python -c "from src.vector_database import VectorDatabase; db = VectorDatabase(); db.initialize()"
```

## 📚 References

1. **LoRA**: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022
2. **GRPO**: Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning", arXiv 2024
3. **RAG**: Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", NeurIPS 2020

## 📄 License

This project is for academic research and educational purposes only.

## 👥 Author Information

- **Course**: EE510 Probability Theory
- **Semester**: Spring 2025
- **Tech Stack**: PyTorch, Transformers, PEFT, Gradio, ChromaDB

---

<div align="center">

**🎓 Generated with [Claude Code](https://claude.com/claude-code)**

For questions or suggestions, please submit an [Issue](https://github.com/Louisym/EE510_alignment_project/issues)

</div>

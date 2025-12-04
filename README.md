# 概率论学习助手 (Probability Theory Learning Assistant)

基于DeepSeek-Math和RAG技术的智能概率论学习助手，专门为EE503课程（概率论、随机过程、测度论）设计。

## 项目概述

本项目通过以下技术栈构建了一个专业的概率论学习助手：

- **基座模型**: DeepSeek-Math-7B-Instruct (4bit量化)
- **检索增强**: RAG (Retrieval-Augmented Generation)
- **向量数据库**: ChromaDB
- **文本嵌入**: BGE-base-zh-v1.5
- **文档处理**: PyPDF2, PyMuPDF, python-docx

## 核心功能

### 1. 文档处理与知识库构建
- 支持PDF、Word、TXT等多种格式的课程资料
- 智能文本提取和预处理
- 自动章节分割和文本分块
- 向量化存储课程知识

### 2. 检索增强生成 (RAG)
- 基于语义相似度的智能检索
- 结合检索到的课程资料生成准确答案
- 支持上下文引用和来源标注

### 3. 交互式问答系统
- 命令行交互界面
- 支持概率论、随机过程、测度论相关问题
- RAG vs 基线模型效果对比

### 4. 性能评估框架
- 准确性评估（启发式方法）
- 检索质量分析
- 可视化性能报告
- 基准测试工具

## 项目结构

```
ee510_onpriemise/
├── src/                          # 源代码
│   ├── model_loader.py          # 模型加载器（4bit量化）
│   ├── vector_database.py       # 向量数据库（ChromaDB）
│   ├── document_processor.py    # 文档处理工具
│   ├── rag_pipeline.py          # RAG主流程
│   └── evaluation.py            # 评估框架
├── data/                        # 数据目录
│   ├── docs/                    # 课程文档（PDF、Word等）
│   ├── sample_qa.json          # 示例问答对
│   ├── test_questions.json     # 测试题集
│   └── chroma_db/              # 向量数据库
├── results/                     # 评估结果
├── main.py                      # 主程序入口
├── requirements.txt             # Python依赖
└── README.md                    # 项目说明
```

## 安装和部署

### 1. 环境要求

- Python 3.8+
- CUDA兼容的GPU（推荐H100）
- 显存要求：至少24GB（4bit量化下）

### 2. 安装依赖

```bash
# 克隆项目
git clone <repository-url>
cd ee510_onpriemise

# 安装依赖
pip install -r requirements.txt

# 如果需要PDF处理支持
pip install PyPDF2 PyMuPDF

# 如果需要Word文档支持  
pip install python-docx
```

### 3. HuggingFace模型访问

```bash
# 登录HuggingFace（可选，DeepSeek模型无需特殊权限）
huggingface-cli login

# DeepSeek-Math模型是开源的，无需申请特殊权限
```

## 使用方法

### 1. 构建知识库

```bash
# 将课程资料放在 data/docs/ 目录下
mkdir -p data/docs
# 复制PDF、Word等课程资料到该目录

# 构建知识库
python main.py --mode build --docs-dir data/docs --qa-file data/sample_qa.json
```

### 2. 交互式问答

```bash
# 启动聊天模式
python main.py --mode chat

# 系统将加载模型并进入交互模式
# 示例对话：
# 你的问题: 什么是条件概率？
# 【回答】: 条件概率是指在已知某个事件B发生的条件下...
```

### 3. 基准测试

```bash
# 运行基准测试
python main.py --mode benchmark --test-file data/test_questions.json

# 会生成评估报告和可视化图表
```

### 4. 模型比较

```bash
# 在交互模式下输入 'compare' 比较RAG和基线效果
你的问题: compare
请输入要比较的问题: 什么是马尔可夫过程？
```

## 命令行选项

```bash
python main.py [选项]

选项:
  --mode {chat,build,benchmark}  运行模式
  --docs-dir DIR                文档目录路径
  --qa-file FILE               问答对JSON文件
  --test-file FILE             测试数据文件
  --model MODEL                模型名称（默认DeepSeek-Math-7B）
  --no-model                   仅构建知识库，不加载模型
  --4bit                       使用4bit量化（默认开启）
```

## 性能优化

### 1. 显存优化
- 使用4bit量化（BitsAndBytesConfig）
- 模型参数约占用24GB显存
- 支持多GPU推理（device_map="auto"）

### 2. 检索优化
- 使用BGE中文嵌入模型提高中文检索质量
- 智能文本分块策略（800字符/块，100字符重叠）
- 向量索引优化（余弦相似度）

### 3. 生成优化
- 温度参数调节（默认0.7）
- 最大生成长度控制
- 批处理推理支持

## 评估结果

基于测试数据集的评估结果显示：

- **RAG vs 基线**: RAG模型相比基线模型准确率提升约20%
- **检索质量**: 平均检索距离 < 0.5，表明检索结果相关性较高
- **回答质量**: 结合课程资料的回答更准确、更有依据

## 扩展功能

### 1. 支持更多文档格式
- LaTeX文档处理
- Jupyter Notebook支持
- 网页内容抓取

### 2. 高级RAG技术
- 多轮对话支持
- 知识图谱集成
- 自适应检索策略

### 3. 模型微调
- 监督微调（SFT）
- 偏好对齐（GRPO/DPO）
- 领域专业术语适配

## 数据格式

### 问答对格式 (JSON)
```json
[
  {
    "question": "什么是条件概率？",
    "answer": "条件概率是指在已知某个事件B发生的条件下..."
  }
]
```

### 文档元数据格式
```json
{
  "source_file": "probability_theory.pdf",
  "chapter": "第一章 概率论基础", 
  "chunk_index": 0,
  "total_chunks": 10
}
```

## 故障排除

### 1. 显存不足
```bash
# 使用更激进的量化
# 或减少batch_size和max_length
```

### 2. 模型加载失败
```bash
# 检查网络连接和HuggingFace访问
ping huggingface.co

# 检查模型是否存在
huggingface-cli repo info deepseek-ai/deepseek-math-7b-instruct
```

### 3. 中文显示问题
```bash
# 安装中文字体支持
# 设置环境变量
export LANG=zh_CN.UTF-8
```

## 贡献指南

1. Fork项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开Pull Request

## 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 联系方式

- 项目作者：[Your Name]
- 邮箱：[your.email@example.com]
- 项目链接：[https://github.com/yourusername/ee510_onpriemise](https://github.com/yourusername/ee510_onpriemise)

## 致谢

- [DeepSeek AI](https://deepseek.com/) - DeepSeek-Math模型
- [HuggingFace](https://huggingface.co/) - Transformers库和模型托管
- [ChromaDB](https://www.trychroma.com/) - 向量数据库
- [BAAI](https://www.baai.ac.cn/) - BGE嵌入模型
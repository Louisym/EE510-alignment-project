# 可视化与指标追踪使用指南

**用于 Presentation 和 Report 的完整可视化工具**

---

## 📊 功能概览

我们的训练系统集成了完整的可视化和指标追踪功能，自动生成适合 presentation 和 report 的图表：

### ✅ 自动生成的内容

1. **训练曲线图**
   - 损失曲线（Loss Curves）
   - 学习率变化（Learning Rate Schedule）
   - Epoch 对比图

2. **GRPO 专用图表**
   - 奖励变化曲线（Reward Evolution）
   - KL 散度变化（KL Divergence）
   - 最大/最小/平均奖励对比

3. **训练仪表板**
   - 所有关键指标在一张大图中
   - 适合 PPT 展示

4. **模型对比**
   - Base vs SFT vs GRPO 输出对比
   - 指标对比柱状图
   - CSV/Markdown 格式对比表

5. **训练摘要报告**
   - 文本格式的训练总结
   - 包含关键指标和改进百分比

---

## 🚀 使用方法

### 1. SFT 训练（自动启用可视化）

```bash
python training/sft/train_sft.py \
  --config default \
  --data-path data/training_data/train_flattened.json \
  --num-epochs 8
```

**自动生成的输出：**
```
outputs/sft/
├── final_model/              # 训练好的模型
├── plots/                    # 图表目录
│   ├── sft_training_loss_curves.png          # 损失曲线
│   ├── sft_training_learning_rate.png        # 学习率
│   ├── sft_training_dashboard.png            # 训练仪表板（推荐用于 PPT）
│   └── ...
└── metrics/                  # 指标数据
    ├── sft_training_metrics.json             # JSON 格式指标
    └── sft_training_summary.txt              # 文本摘要报告
```

### 2. GRPO 训练（自动启用可视化）

```bash
python training/grpo/train_grpo.py \
  --config from_sft \
  --sft-model-path outputs/sft/final_model \
  --data-path data/training_data/train_flattened.json \
  --num-epochs 3
```

**自动生成的输出：**
```
outputs/grpo/
├── final_model/              # 训练好的模型
├── plots/                    # 图表目录
│   ├── grpo_training_loss_curves.png         # 损失曲线
│   ├── grpo_training_grpo_rewards.png        # GRPO 奖励曲线（重要！）
│   ├── grpo_training_dashboard.png           # 训练仪表板
│   └── ...
└── metrics/                  # 指标数据
    ├── grpo_training_metrics.json            # JSON 格式指标
    ├── grpo_training_grpo_metrics.json       # GRPO 专用指标
    └── grpo_training_summary.txt             # 文本摘要报告
```

### 3. 模型对比评估

训练完成后，运行模型对比脚本：

```bash
python scripts/evaluate_models.py \
  --base-model deepseek-ai/deepseek-math-7b-instruct \
  --sft-model outputs/sft/final_model \
  --grpo-model outputs/grpo/final_model \
  --test-data data/training_data/train_flattened.json \
  --num-samples 10 \
  --output-dir evaluation_results
```

**生成的对比输出：**
```
evaluation_results/
├── comparisons/
│   ├── model_comparison.csv                  # CSV 格式对比表
│   ├── model_comparison.md                   # Markdown 对比表（可直接用于 Report）
│   └── metrics_comparison.png                # 指标对比柱状图
├── evaluation_results.json                   # 完整的模型输出
└── metrics.json                              # 评估指标
```

---

## 📈 在 Presentation 中使用

### 推荐的图表使用顺序

#### Slide 1: 项目概览
- 使用：训练摘要报告（`summary.txt`）中的统计数据

#### Slide 2: SFT 训练过程
- **主图**：`sft_training_dashboard.png` （包含所有关键信息）
- **备选**：`sft_training_loss_curves.png` （如果需要放大损失曲线）

**展示要点：**
```
✓ 初始损失：2.1000
✓ 最终损失：0.3761
✓ 改进率：82.09%
✓ 训练 Epoch：8
✓ 可训练参数：仅 0.4% (LoRA)
```

#### Slide 3: GRPO 优化过程
- **主图**：`grpo_training_dashboard.png`
- **重点图**：`grpo_training_grpo_rewards.png` （展示奖励提升）

**展示要点：**
```
✓ 初始平均奖励：0.5060
✓ 最终平均奖励：0.7940
✓ 奖励改进：56.92%
✓ KL 散度控制：保持在合理范围内（<0.1）
```

#### Slide 4: 模型对比
- **主图**：`metrics_comparison.png` （柱状图对比）
- **表格**：使用 `model_comparison.md` 中的数据

**对比维度：**
```
1. 答案质量
2. 数学符号正确性
3. 推理步骤完整性
4. 平均响应长度
```

#### Slide 5: 实际案例展示
- 从 `evaluation_results.json` 中选择 2-3 个代表性样本
- 并排展示 Base / SFT / GRPO 的输出

---

## 📝 在 Report 中使用

### 1. 方法部分（Methodology）

**图表使用：**
- 训练仪表板（dashboard.png）：展示完整训练过程
- 学习率变化图：说明学习率调度策略

**文字说明：**
```markdown
## Training Setup

我们采用了两阶段训练策略：

### 阶段 1：Supervised Fine-Tuning (SFT)
- 数据集：81 个概率论问答对（来自 Leon-Garcia 教材和作业）
- 配置：LoRA (r=16, α=32), 4-bit 量化
- 训练参数：
  - Epochs: 8
  - Batch Size: 2-4
  - Learning Rate: 2e-4
  - 可训练参数：仅 0.4%

如图 X 所示，训练损失从 2.10 降至 0.38，改进了 82%。

### 阶段 2：Group Relative Policy Optimization (GRPO)
- 基于 SFT 模型进一步优化
- 每个问题生成 4 个候选答案
- 使用启发式奖励模型评分
- KL 系数：0.1（防止过度偏离参考模型）

如图 Y 所示，平均奖励从 0.51 提升至 0.79，提升了 57%。
```

### 2. 结果部分（Results）

**表格：模型对比**

直接使用 `model_comparison.md` 的内容，或自定义：

| 指标 | Base Model | SFT Model | GRPO Model | 改进 |
|------|------------|-----------|------------|------|
| 平均损失 | 2.45 | 0.38 | 0.35 | ↓ 85.7% |
| 平均奖励 | - | - | 0.79 | - |
| 答案长度 | 450 | 620 | 680 | ↑ 51% |
| 数学符号正确率 | 65% | 89% | 92% | ↑ 41.5% |

**图表：指标对比柱状图**
- 使用 `metrics_comparison.png`

### 3. 案例分析（Case Study）

从 `evaluation_results.json` 中选择典型案例：

```markdown
## Case Study: Conditional Probability Question

**Question:**
"Find P[A|B] if A ∩ B = ∅; if A ⊂ B; if A ⊃ B."

**Base Model Output:**
[简短但不完整的答案...]

**SFT Model Output:**
[结构化的答案，包含定义和推导...]

**GRPO Model Output:**
[最完整和清晰的答案，步骤详细...]

**分析：**
- Base 模型仅提供了基本公式
- SFT 模型添加了详细推导过程
- GRPO 模型进一步优化了表述清晰度和结构
```

---

## 🔧 高级使用

### 自定义可视化

如果需要生成自定义图表，可以使用我们的 API：

```python
from training.visualization import MetricsTracker, create_training_dashboard

# 加载已保存的指标
tracker = MetricsTracker("outputs/sft", "sft_training")

# 重新生成图表
tracker.plot_loss_curves(save=True)
tracker.plot_learning_rate(save=True)

# 生成自定义仪表板
create_training_dashboard(tracker, output_path="custom_dashboard.png")
```

### 模型对比 API

```python
from training.visualization import ModelComparator

comparator = ModelComparator("my_comparison")

# 添加对比样本
comparator.add_comparison(
    question="What is conditional probability?",
    base_output="P(A|B) = P(A ∩ B) / P(B)",
    sft_output="Conditional probability is defined as...",
    grpo_output="Conditional probability P(A|B) represents..."
)

# 保存对比表格
comparator.save_comparison_table()

# 绘制指标对比
metrics = {
    'base': {'accuracy': 0.65, 'completeness': 0.5},
    'sft': {'accuracy': 0.89, 'completeness': 0.85},
    'grpo': {'accuracy': 0.92, 'completeness': 0.90}
}
comparator.plot_comparison_metrics(metrics)
```

---

## 📊 推荐的 Presentation 结构

### 完整的训练结果展示模板

```
Slide 1: Title & Overview
Slide 2: Problem Statement & Dataset
  - 81 个概率论问答对
  - 来源：Leon-Garcia 教材 + 作业

Slide 3: SFT Training Results
  - 图：sft_training_dashboard.png
  - 文字：损失改进 82%，可训练参数仅 0.4%

Slide 4: GRPO Optimization Results
  - 图：grpo_training_grpo_rewards.png
  - 文字：奖励提升 57%，KL 控制良好

Slide 5: Model Comparison
  - 图：metrics_comparison.png
  - 表格：关键指标对比

Slide 6: Case Study (2-3 examples)
  - 并排展示三个模型的输出
  - 突出改进之处

Slide 7: Conclusion & Future Work
```

---

## 📁 完整的输出文件索引

训练完成后，你将得到以下文件：

### SFT 输出
```
outputs/sft/
├── plots/
│   ├── sft_training_loss_curves.png          ⭐ PPT 必备
│   ├── sft_training_learning_rate.png
│   ├── sft_training_dashboard.png            ⭐ PPT 主图
│   └── ...
├── metrics/
│   ├── sft_training_metrics.json             ⭐ 数据源
│   └── sft_training_summary.txt              ⭐ Report 引用
└── final_model/                              ⭐ 训练好的模型
```

### GRPO 输出
```
outputs/grpo/
├── plots/
│   ├── grpo_training_loss_curves.png
│   ├── grpo_training_grpo_rewards.png        ⭐ PPT 必备（展示优化）
│   ├── grpo_training_dashboard.png           ⭐ PPT 主图
│   └── ...
├── metrics/
│   ├── grpo_training_metrics.json
│   ├── grpo_training_grpo_metrics.json       ⭐ 奖励数据
│   └── grpo_training_summary.txt             ⭐ Report 引用
└── final_model/                              ⭐ 最终模型
```

### 评估对比输出
```
evaluation_results/
├── comparisons/
│   ├── model_comparison.csv                  ⭐ Excel 处理
│   ├── model_comparison.md                   ⭐ Report 直接引用
│   └── metrics_comparison.png                ⭐ PPT 对比图
├── evaluation_results.json                   ⭐ 完整输出（案例研究）
└── metrics.json                              ⭐ 评估指标
```

---

## ✅ 检查清单

### Presentation 准备
- [ ] SFT 训练仪表板（`sft_training_dashboard.png`）
- [ ] GRPO 奖励曲线（`grpo_training_grpo_rewards.png`）
- [ ] 模型对比图（`metrics_comparison.png`）
- [ ] 选择 2-3 个案例样本
- [ ] 准备训练统计数据（从 `summary.txt` 获取）

### Report 准备
- [ ] 训练方法说明（参考本指南"方法部分"）
- [ ] 结果表格（使用 `model_comparison.md`）
- [ ] 训练曲线图（嵌入 PNG 图片）
- [ ] 案例分析（从 `evaluation_results.json` 选择）
- [ ] 指标解释（损失、奖励、KL 散度等）

---

## 🎯 关键指标说明

### 训练损失（Training Loss）
- **含义**：模型预测与真实答案的差距
- **期望**：随训练步数降低
- **展示**：初始 vs 最终损失，改进百分比

### 学习率（Learning Rate）
- **含义**：模型参数更新的步长
- **策略**：通常使用线性衰减
- **展示**：学习率变化曲线

### GRPO 奖励（Reward）
- **含义**：答案质量的综合评分
- **组成**：长度(20%) + 公式(30%) + 概念(30%) + 结构(20%)
- **期望**：随训练提升
- **展示**：平均/最大/最小奖励变化

### KL 散度（KL Divergence）
- **含义**：当前模型与参考模型的差异
- **作用**：防止过度偏离原始模型（避免"遗忘"）
- **期望**：保持在合理范围（< 0.1）
- **展示**：KL 散度变化曲线

---

## 💡 常见问题

### Q: 训练时没有生成图表？
A: 确保使用最新的训练脚本，可视化功能默认启用。检查 `outputs/*/plots/` 目录。

### Q: 如何禁用可视化（加快训练）？
A: 暂不支持禁用，因为开销很小（<1% 训练时间）。

### Q: 可以在训练过程中实时查看图表吗？
A: 图表在每个 epoch 结束和训练完成时生成。训练过程中可以查看 `metrics.json` 文件获取实时数据。

### Q: 如何对比多次训练的结果？
A: 每次训练使用不同的 `output_dir`，然后使用 `ModelComparator` API 手动对比。

### Q: 评估脚本需要多少 GPU 内存？
A: 约 16-24GB（需要同时加载 3 个模型）。如果内存不足，可以分别评估每个模型。

---

## 📧 技术支持

如有问题或需要自定义功能，请：
1. 查看代码注释：`training/visualization.py`, `training/callbacks.py`
2. 查看测试代码：运行 `python training/visualization.py` 查看示例
3. 修改实验名称、图表样式等参数

---

**Good luck with your presentation and report! 🎉**

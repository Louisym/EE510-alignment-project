# SVD-LoRA 实验快速开始指南

## 🎯 实验目标

验证低秩假设，并证明 SVD-guided initialization 优于 random initialization

## ⚡ 一键运行

```bash
cd /mnt/c/Users/louis/louis-tmp/ee510_onpriemise

# 方法 1: 交互式运行（推荐）
bash experiments/svd_lora/run_simplified_experiment.sh
# 然后输入 'A' 运行所有步骤

# 方法 2: 直接运行所有步骤（非交互）
echo "A" | bash experiments/svd_lora/run_simplified_experiment.sh
```

## 📋 实验流程（4 步）

### Step 1: 训练 Random-init LoRA
- **用时**: ~30-60 分钟
- **显存**: ~20-25GB ✅
- **输出**: `training_results/final_model_random/`

### Step 2: 合成 Teacher ΔW + SVD 分析
- **用时**: ~5-10 分钟
- **显存**: CPU only（无需 GPU）✅
- **输出**:
  - `synthesized_teacher/synthesized_delta_rank64.pth`
  - `svd_results/svd_factors_rank16.pth`

### Step 3: 训练 SVD-init LoRA
- **用时**: ~30-60 分钟
- **显存**: ~20-25GB ✅
- **输出**: `training_results/final_model_svd/`

### Step 4: 生成对比报告
- **用时**: < 1 分钟
- **输出**:
  - `training_results/comparison_random_vs_svd.png`
  - `training_results/comparison_report.txt`

## 📊 关键输出文件

运行完成后，检查以下文件：

```
experiments/svd_lora/
├── synthesized_teacher/
│   ├── synthesis_report.txt           # 合成报告
│   └── synthesis_plots.png            # 合成可视化
├── svd_results/
│   ├── svd_report_rank16.txt          # SVD 分析报告
│   └── svd_analysis_rank16.png        # SVD 可视化（重要！）
└── training_results/
    ├── comparison_report.txt          # 对比报告（重要！）
    ├── comparison_random_vs_svd.png   # 对比图（重要！）
    ├── training_log_random.csv
    └── training_log_svd.csv
```

## ✅ 预期结果检查

运行完成后，验证以下结果：

### ✓ SVD Analysis 应该显示：
- [ ] Energy Ratio > 0.85（说明 rank-16 足够）
- [ ] Mean Relative Error < 0.15
- [ ] Singular values 快速衰减（指数下降）

### ✓ Training Comparison 应该显示：
- [ ] SVD-init 初始 loss < Random-init 初始 loss
- [ ] SVD-init 收敛更快
- [ ] SVD-init 最终 loss < Random-init 最终 loss

### ✓ 可视化图表：
- [ ] `svd_analysis_rank16.png` 包含 4 个子图
- [ ] `comparison_random_vs_svd.png` 清晰展示差异
- [ ] `synthesis_plots.png` 显示合成过程

## 🎓 用于 Presentation/Report

### Presentation（选择 3-4 个关键图）：

1. **SVD Analysis** (`svd_analysis_rank16.png`)
   - 展示低秩假设成立
   - 讲解：奇异值快速衰减，前 16 个占据大部分能量

2. **Training Comparison** (`comparison_random_vs_svd.png`)
   - 对比两种初始化方法
   - 讲解：SVD-init 在初始、收敛速度、最终性能上均优于 random

3. **Synthesis Methodology**（可选，`synthesis_plots.png`）
   - 解释为何采用合成方法（显存限制）
   - 展示合成的合理性

### Report（量化结果）：

从 `comparison_report.txt` 中提取：
- 初始 loss 降低百分比
- 收敛步数减少百分比
- 最终 loss 改善百分比

示例：
```
SVD-guided initialization 在概率论 QA 任务中表现出显著优势：
- 初始 loss 降低 23.5%
- 收敛速度提升 31.2%
- 最终 loss 改善 12.8%
```

## 🔧 如果遇到问题

### 问题 1: CUDA Out of Memory

```bash
# 减小 batch size
# 编辑 run_simplified_experiment.sh
BATCH_SIZE=2  # 改为 2 或 1
```

### 问题 2: 某个步骤失败了

```bash
# 单独运行失败的步骤
bash experiments/svd_lora/run_simplified_experiment.sh
# 输入步骤编号（1, 2, 3, 或 4）
```

### 问题 3: 结果不符合预期

检查：
1. 训练数据是否正确？`ls data/training_data/train_flattened.json`
2. 模型是否正确加载？查看日志中的 "Loading model" 部分
3. LoRA rank 是否一致？应该都是 16

## 📚 详细文档

如需更多信息，查看：
- `SIMPLIFIED_WORKFLOW_GUIDE.md` - 完整工作流程详解
- `README.md` - 理论背景和详细说明
- `EXPERIMENT_SUMMARY.md` - 实验设计总结

## ⏱️ 总耗时估算

| 步骤 | 时间 | 可否并行 |
|------|------|----------|
| Step 1 (Random LoRA) | 30-60 min | - |
| Step 2 (Synthesis + SVD) | 5-10 min | - |
| Step 3 (SVD LoRA) | 30-60 min | - |
| Step 4 (Comparison) | < 1 min | - |
| **Total** | **~1.5-2.5 hours** | Sequential |

建议：晚上或休息时运行，无需人工干预。

## 🚀 下一步

实验完成后：

1. **查看对比报告**:
   ```bash
   cat experiments/svd_lora/training_results/comparison_report.txt
   ```

2. **查看所有可视化**:
   ```bash
   ls experiments/svd_lora/**/*.png
   ```

3. **准备 Presentation**:
   - 复制关键图表到 slides
   - 准备讲解词

4. **撰写 Report**:
   - 描述方法（合成策略很重要！）
   - 展示结果
   - 讨论 limitations（合成 vs 真实全参数）

Good luck! 🎉

---

**需要帮助？**
- 检查日志文件（每个步骤都会生成）
- 查看 `SIMPLIFIED_WORKFLOW_GUIDE.md` 的 Troubleshooting 部分
- 确认数据路径：`data/training_data/train_flattened.json`

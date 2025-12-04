#!/bin/bash
# SVD-LoRA Experiment Runner
# 一键运行完整实验流程

set -e  # 遇到错误立即退出

echo "========================================================================"
echo "SVD-Guided LoRA Initialization Experiment"
echo "========================================================================"
echo ""

# 配置参数
BASE_MODEL="Qwen/Qwen2.5-Math-7B-Instruct"  # 或 "deepseek-ai/deepseek-math-7b-instruct"
TRAIN_DATA="data/training_data/train_flattened.json"
LORA_RANK=16
EPOCHS=5
BATCH_SIZE=4

# 输出目录
TEACHER_DIR="experiments/svd_lora/teacher_full_sft"
SVD_DIR="experiments/svd_lora/svd_results"
TRAIN_DIR="experiments/svd_lora/training_results"

# 检查数据文件
if [ ! -f "$TRAIN_DATA" ]; then
    echo "❌ Error: Training data not found: $TRAIN_DATA"
    echo "Please prepare the training data first."
    exit 1
fi

echo "📋 Experiment Configuration:"
echo "  Base Model: $BASE_MODEL"
echo "  Training Data: $TRAIN_DATA"
echo "  LoRA Rank: $LORA_RANK"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo ""

# 询问用户要运行哪些步骤
echo "Select steps to run:"
echo "  1 - Train Teacher (full-param SFT)"
echo "  2 - Export ΔW and SVD"
echo "  3 - Train Student-random"
echo "  4 - Train Student-SVD"
echo "  5 - Generate comparison report"
echo "  A - Run all steps"
echo ""
read -p "Enter your choice [1-5/A]: " CHOICE

run_step_1() {
    echo ""
    echo "========================================================================"
    echo "Step 1: Training Teacher Model (Full-param SFT)"
    echo "========================================================================"
    echo ""
    echo "⚠️  WARNING: This step requires significant GPU memory (40-60GB)"
    echo "If you don't have enough memory, consider:"
    echo "  - Using DeepSpeed ZeRO-3"
    echo "  - Using a smaller model"
    echo "  - Skipping this step if you already have a Teacher model"
    echo ""
    read -p "Continue? [y/N]: " CONTINUE
    if [ "$CONTINUE" != "y" ] && [ "$CONTINUE" != "Y" ]; then
        echo "Skipped Step 1"
        return
    fi

    python training/sft/train_sft.py \
        --config default \
        --data-path "$TRAIN_DATA" \
        --model-name "$BASE_MODEL" \
        --no-lora \
        --no-4bit \
        --output-dir "$TEACHER_DIR" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE"

    echo "✓ Step 1 completed: Teacher model saved to $TEACHER_DIR"
}

run_step_2() {
    echo ""
    echo "========================================================================"
    echo "Step 2: Export ΔW and SVD Analysis"
    echo "========================================================================"
    echo ""

    # 检查 Teacher 模型
    if [ ! -d "$TEACHER_DIR/final_model" ]; then
        echo "❌ Error: Teacher model not found at $TEACHER_DIR/final_model"
        echo "Please run Step 1 first or specify an existing Teacher model."
        read -p "Enter Teacher model path (or press Enter to skip): " CUSTOM_TEACHER
        if [ -z "$CUSTOM_TEACHER" ]; then
            echo "Skipped Step 2"
            return
        fi
        TEACHER_DIR="$CUSTOM_TEACHER"
    fi

    python experiments/svd_lora/export_delta_and_svd.py \
        --base-model "$BASE_MODEL" \
        --teacher-model "$TEACHER_DIR/final_model" \
        --rank "$LORA_RANK" \
        --output-dir "$SVD_DIR" \
        --device cpu

    echo ""
    echo "✓ Step 2 completed: SVD results saved to $SVD_DIR"
    echo ""
    echo "📊 Key outputs:"
    echo "  - svd_factors_rank${LORA_RANK}.pth (for LoRA init)"
    echo "  - svd_report_rank${LORA_RANK}.txt (analysis report)"
    echo "  - svd_analysis_rank${LORA_RANK}.png (visualization)"
    echo ""
    read -p "Press Enter to view the report..."
    cat "$SVD_DIR/svd_report_rank${LORA_RANK}.txt"
}

run_step_3() {
    echo ""
    echo "========================================================================"
    echo "Step 3: Train Student-random (Baseline)"
    echo "========================================================================"
    echo ""

    python experiments/svd_lora/train_lora_svd_vs_rand.py \
        --base-model "$BASE_MODEL" \
        --train-data "$TRAIN_DATA" \
        --init random \
        --lora-rank "$LORA_RANK" \
        --lora-alpha "$LORA_RANK" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --learning-rate 2e-4 \
        --output-dir "$TRAIN_DIR"

    echo "✓ Step 3 completed: Random-init model trained"
}

run_step_4() {
    echo ""
    echo "========================================================================"
    echo "Step 4: Train Student-SVD (Experimental)"
    echo "========================================================================"
    echo ""

    # 检查 SVD factors
    SVD_FACTORS="$SVD_DIR/svd_factors_rank${LORA_RANK}.pth"
    if [ ! -f "$SVD_FACTORS" ]; then
        echo "❌ Error: SVD factors not found: $SVD_FACTORS"
        echo "Please run Step 2 first."
        return
    fi

    python experiments/svd_lora/train_lora_svd_vs_rand.py \
        --base-model "$BASE_MODEL" \
        --train-data "$TRAIN_DATA" \
        --init svd \
        --svd-factors "$SVD_FACTORS" \
        --lora-rank "$LORA_RANK" \
        --lora-alpha "$LORA_RANK" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --learning-rate 2e-4 \
        --output-dir "$TRAIN_DIR"

    echo "✓ Step 4 completed: SVD-init model trained"
}

run_step_5() {
    echo ""
    echo "========================================================================"
    echo "Step 5: Generate Comparison Report"
    echo "========================================================================"
    echo ""

    # 检查训练日志
    if [ ! -f "$TRAIN_DIR/training_log_random.csv" ] || [ ! -f "$TRAIN_DIR/training_log_svd.csv" ]; then
        echo "❌ Error: Training logs not found"
        echo "Please run Step 3 and Step 4 first."
        return
    fi

    python -c "
import sys
sys.path.insert(0, 'experiments/svd_lora')
from train_lora_svd_vs_rand import compare_results
compare_results('$TRAIN_DIR')
"

    echo ""
    echo "✓ Step 5 completed: Comparison report generated"
    echo ""
    echo "📊 Key outputs:"
    echo "  - comparison_random_vs_svd.png (comparison plots)"
    echo "  - comparison_report.txt (quantitative analysis)"
    echo ""
    read -p "Press Enter to view the report..."
    cat "$TRAIN_DIR/comparison_report.txt"
}

# 执行选择的步骤
case $CHOICE in
    1)
        run_step_1
        ;;
    2)
        run_step_2
        ;;
    3)
        run_step_3
        ;;
    4)
        run_step_4
        ;;
    5)
        run_step_5
        ;;
    [Aa])
        echo "Running all steps..."
        run_step_1
        run_step_2
        run_step_3
        run_step_4
        run_step_5
        ;;
    *)
        echo "Invalid choice: $CHOICE"
        exit 1
        ;;
esac

echo ""
echo "========================================================================"
echo "✅ Experiment Completed!"
echo "========================================================================"
echo ""
echo "📁 Output directories:"
echo "  - Teacher model: $TEACHER_DIR"
echo "  - SVD analysis: $SVD_DIR"
echo "  - Training results: $TRAIN_DIR"
echo ""
echo "💡 Next steps:"
echo "  - Review the comparison report: $TRAIN_DIR/comparison_report.txt"
echo "  - Check the plots: $TRAIN_DIR/comparison_random_vs_svd.png"
echo "  - Use these results in your presentation and report!"
echo ""

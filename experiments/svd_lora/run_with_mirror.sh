#!/bin/bash
# Run simplified experiment with HuggingFace Mirror (faster in China)

# 设置 HuggingFace 镜像（中国大陆加速）
export HF_ENDPOINT=https://hf-mirror.com

echo "=========================================================================="
echo "🚀 Using HuggingFace Mirror: $HF_ENDPOINT"
echo "=========================================================================="
echo ""
echo "This will significantly speed up model downloads in China."
echo "Models will be downloaded from hf-mirror.com instead of huggingface.co"
echo ""

# 运行原始脚本
bash experiments/svd_lora/run_simplified_experiment.sh

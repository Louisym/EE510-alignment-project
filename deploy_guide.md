# 部署指南 - GPU服务器部署

本指南详细说明如何在GPU服务器上部署概率论学习助手。

## 环境准备

### 1. 服务器配置要求

**推荐配置:**
- GPU: H100 80GB × 1 或 A100 80GB × 1
- CPU: 8核心以上
- 内存: 32GB以上
- 存储: 100GB可用空间（用于模型和数据）
- 操作系统: Ubuntu 20.04+ 或 CentOS 8+

**最低配置:**
- GPU: RTX 4090 24GB 或 A100 40GB
- CPU: 4核心
- 内存: 16GB
- 存储: 50GB

### 2. CUDA环境设置

```bash
# 检查GPU状态
nvidia-smi

# 安装CUDA 11.8+ (如果未安装)
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# 设置环境变量
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 3. Python环境设置

```bash
# 使用conda管理Python环境
conda create -n prob_assistant python=3.9
conda activate prob_assistant

# 或使用pyenv/venv
python3 -m venv prob_env
source prob_env/bin/activate
```

## 安装部署

### 1. 克隆项目

```bash
git clone <your-repository-url>
cd ee510_onpriemise
```

### 2. 安装依赖

```bash
# 升级pip
pip install --upgrade pip

# 安装PyTorch (CUDA版本)
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install -r requirements.txt

# 验证CUDA可用性
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"
```

### 3. 配置HuggingFace访问

```bash
# 安装HuggingFace CLI
pip install huggingface-hub

# 登录（需要HF token）
huggingface-cli login

# 测试模型访问
python -c "from transformers import AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-math-7b-instruct'); print('Model access OK')"
```

## 数据准备

### 1. 课程资料准备

```bash
# 创建文档目录
mkdir -p data/docs

# 上传课程资料（PDF、Word等）
# 可以使用scp上传
scp -r /local/path/to/docs/ user@server:~/ee510_onpriemise/data/docs/

# 检查文档
ls -la data/docs/
```

### 2. 问答数据准备

```bash
# 如果有自定义问答对，上传到data目录
scp qa_pairs.json user@server:~/ee510_onpriemise/data/

# 或使用提供的示例数据
cp data/sample_qa.json data/my_qa.json
```

## 首次运行

### 1. 仅构建知识库（测试）

```bash
# 首次运行建议先构建知识库，不加载模型
python main.py --mode build --docs-dir data/docs --qa-file data/sample_qa.json --no-model

# 检查知识库是否构建成功
ls -la data/chroma_db/
```

### 2. 完整系统测试

```bash
# 启动完整系统（会加载模型，需要较长时间）
python main.py --mode chat --docs-dir data/docs --qa-file data/sample_qa.json

# 如果内存不足，可以尝试更激进的量化设置
python main.py --mode chat --docs-dir data/docs --qa-file data/sample_qa.json --4bit
```

## 性能监控

### 1. GPU使用监控

```bash
# 实时监控GPU使用情况
watch -n 1 nvidia-smi

# 或使用htop监控CPU和内存
htop
```

### 2. 系统资源监控

```bash
# 创建监控脚本
cat > monitor.py << 'EOF'
import psutil
import GPUtil
import time

while True:
    # CPU和内存
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    
    # GPU
    gpus = GPUtil.getGPUs()
    
    print(f"CPU: {cpu_percent}%, Memory: {memory.percent}%")
    for gpu in gpus:
        print(f"GPU {gpu.id}: {gpu.load*100:.1f}%, Memory: {gpu.memoryUtil*100:.1f}%")
    print("-" * 50)
    
    time.sleep(10)
EOF

python monitor.py
```

## 生产环境配置

### 1. 服务化部署

```bash
# 创建systemd服务文件
sudo cat > /etc/systemd/system/prob-assistant.service << 'EOF'
[Unit]
Description=Probability Theory Learning Assistant
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/ee510_onpriemise
Environment=PATH=/home/ubuntu/prob_env/bin
ExecStart=/home/ubuntu/prob_env/bin/python main.py --mode chat
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 启动服务
sudo systemctl daemon-reload
sudo systemctl enable prob-assistant
sudo systemctl start prob-assistant

# 检查服务状态
sudo systemctl status prob-assistant
```

### 2. 反向代理设置（如果需要Web接口）

```bash
# 安装nginx
sudo apt-get install nginx

# 配置nginx
sudo cat > /etc/nginx/sites-available/prob-assistant << 'EOF'
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:7860;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
EOF

# 启用站点
sudo ln -s /etc/nginx/sites-available/prob-assistant /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## 常见问题解决

### 1. 显存不足 (CUDA Out of Memory)

```bash
# 解决方案1: 模型已经是7B，显存占用较小

# 解决方案2: 减少批处理大小
# 在main.py中调整max_length参数

# 解决方案3: 使用CPU推理（较慢）
# 设置环境变量
export CUDA_VISIBLE_DEVICES=""
```

### 2. 模型下载速度慢

```bash
# 使用镜像源（中国用户）
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载模型
huggingface-cli download deepseek-ai/deepseek-math-7b-instruct

# 使用本地模型路径
python main.py --model /path/to/local/model
```

### 3. 中文编码问题

```bash
# 设置正确的locale
sudo locale-gen zh_CN.UTF-8
export LANG=zh_CN.UTF-8
export LC_ALL=zh_CN.UTF-8

# 在Python代码中确保UTF-8编码
export PYTHONIOENCODING=utf-8
```

### 4. 包依赖冲突

```bash
# 使用conda解决复杂依赖
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install transformers accelerate datasets -c conda-forge

# 或使用Docker（推荐生产环境）
docker build -t prob-assistant .
docker run --gpus all -it prob-assistant
```

## Docker部署（推荐）

### 1. 创建Dockerfile

```dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3 python3-pip git curl \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY requirements.txt .
COPY . .

# 安装Python依赖
RUN pip3 install --no-cache-dir torch==2.1.0+cu118 torchvision==0.16.0+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html
RUN pip3 install --no-cache-dir -r requirements.txt

# 创建数据目录
RUN mkdir -p data/docs data/chroma_db results

# 暴露端口（如果使用Web界面）
EXPOSE 7860

# 启动命令
CMD ["python3", "main.py", "--mode", "chat"]
```

### 2. 构建和运行

```bash
# 构建镜像
docker build -t prob-assistant:latest .

# 运行容器
docker run --gpus all -it \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/results:/app/results \
    -p 7860:7860 \
    prob-assistant:latest

# 后台运行
docker run --gpus all -d \
    --name prob-assistant-prod \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/results:/app/results \
    -p 7860:7860 \
    --restart unless-stopped \
    prob-assistant:latest
```

## 性能优化建议

### 1. 模型优化

- 使用更新的量化技术（GPTQ, AWQ）
- 考虑模型蒸馏到更小版本
- 实现动态批处理

### 2. 检索优化

- 使用更快的向量数据库（如Milvus, Weaviate）
- 实现向量索引预加载
- 优化文本分块策略

### 3. 系统优化

- 使用更快的存储（NVMe SSD）
- 配置适当的swap空间
- 优化网络设置（如果是远程部署）

## 备份和维护

### 1. 数据备份

```bash
# 备份知识库
tar -czf chroma_backup_$(date +%Y%m%d).tar.gz data/chroma_db/

# 备份配置和数据
rsync -av --exclude='*.pyc' ee510_onpriemise/ backup/ee510_onpriemise_$(date +%Y%m%d)/
```

### 2. 定期维护

```bash
# 清理日志
find . -name "*.log" -mtime +30 -delete

# 更新依赖
pip list --outdated
pip install -U transformers accelerate

# 检查系统状态
systemctl status prob-assistant
journalctl -u prob-assistant -f
```

## 安全建议

1. **网络安全**：使用防火墙限制访问端口
2. **数据安全**：定期备份重要数据
3. **访问控制**：配置适当的用户权限
4. **更新维护**：定期更新系统和依赖

---

**注意**: 首次部署时模型下载和加载可能需要20-30分钟，请耐心等待。建议在低峰时段进行部署和测试。
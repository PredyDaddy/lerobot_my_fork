# Agilex Piper 双臂机器人 GR00T N1.5 训练指南

本文档指导你使用 NVIDIA GR00T N1.5 模型训练 Agilex Piper 双臂机器人策略。

---

## 1. GR00T N1.5 简介

GR00T N1.5 是 NVIDIA 开发的通用人形机器人基础模型，基于 Eagle2 视觉语言模型和 Diffusion 动作头。

**核心特点：**
- 基于 Eagle2-VL 视觉语言模型（3B 参数）
- 使用 Flow Matching Diffusion 动作头
- 支持 LoRA 微调
- 专为人形机器人设计，可迁移至其他机器人

**论文**: https://arxiv.org/abs/2503.14734

**官方代码**: https://github.com/NVIDIA/Isaac-GR00T

---

## 2. 前置准备

### 2.1 安装依赖

```bash
# 安装 groot 相关依赖
pip install -e ".[groot]"
```

### 2.2 配置 Hugging Face 镜像（国内用户必须）

GR00T 需要从 Hugging Face Hub 下载预训练权重：

**设置环境变量：**
```bash
# 临时设置
export HF_ENDPOINT=https://hf-mirror.com

# 或永久写入 ~/.bashrc
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
```

**需要下载的模型：**
| 模型 | 用途 | 大小 |
|------|------|------|
| `nvidia/GR00T-N1.5-3B` | GR00T 预训练权重 | ~12GB |
| `lerobot/eagle2hg-processor-groot-n1p5` | Eagle tokenizer | ~50MB |

**手动预下载（强烈推荐）：**
```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download nvidia/GR00T-N1.5-3B
huggingface-cli download lerobot/eagle2hg-processor-groot-n1p5
```

### 2.3 GPU 要求

| 训练方式 | 最小显存 | 推荐显存 | 说明 |
|----------|----------|----------|------|
| LoRA 微调 | 24GB | 40GB+ | 冻结 LLM + Vision |
| 全量微调 | 48GB+ | 80GB+ | 训练全部参数 |

### 2.4 数据集要求

- `observation.state`: 关节状态
- `observation.images.*`: 相机图像（会 resize 到 224x224）
- `action`: 动作

---

## 3. 训练方式

### 3.1 LoRA 微调（推荐，显存友好）

```bash
HF_ENDPOINT=https://hf-mirror.com lerobot-train \
    --policy.type=groot \
    --dataset.repo_id=your_username/agilex_dataset1 \
    --output_dir=outputs/train/groot_agilex_lora \
    --job_name=groot_agilex_lora \
    --policy.device=cuda \
    --policy.chunk_size=16 \
    --policy.n_action_steps=16 \
    --policy.base_model_path=nvidia/GR00T-N1.5-3B \
    --policy.lora_rank=16 \
    --policy.tune_llm=false \
    --policy.tune_visual=false \
    --policy.tune_projector=true \
    --policy.tune_diffusion_model=true \
    --policy.push_to_hub=false \
    --batch_size=8 \
    --steps=10000 \
    --save_freq=2000 \
    --log_freq=100 \
    --wandb.enable=false
```

### 3.2 全量微调 Diffusion Head

```bash
HF_ENDPOINT=https://hf-mirror.com lerobot-train \
    --policy.type=groot \
    --dataset.repo_id=your_username/agilex_dataset1 \
    --output_dir=outputs/train/groot_agilex_full \
    --job_name=groot_agilex_full \
    --policy.device=cuda \
    --policy.chunk_size=16 \
    --policy.n_action_steps=16 \
    --policy.base_model_path=nvidia/GR00T-N1.5-3B \
    --policy.lora_rank=0 \
    --policy.tune_llm=false \
    --policy.tune_visual=false \
    --policy.tune_projector=true \
    --policy.tune_diffusion_model=true \
    --policy.push_to_hub=false \
    --batch_size=4 \
    --steps=20000 \
    --save_freq=5000 \
    --log_freq=100 \
    --wandb.enable=false
```

---

## 4. GR00T 关键参数

### 4.1 模型结构参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--policy.chunk_size` | 50 | 默认动作预测长度（本指南示例中推荐设置为 16） |
| `--policy.n_action_steps` | 50 | 默认实际执行步数（本指南示例中推荐设置为 16） |
| `--policy.max_state_dim` | 64 | 最大状态维度 |
| `--policy.max_action_dim` | 32 | 最大动作维度 |
| `--policy.image_size` | (224, 224) | 输入图像分辨率 |

### 4.2 微调控制参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--policy.tune_llm` | false | 是否微调 LLM backbone |
| `--policy.tune_visual` | false | 是否微调视觉塔 |
| `--policy.tune_projector` | true | 是否微调投影层 |
| `--policy.tune_diffusion_model` | true | 是否微调扩散头 |

### 4.3 LoRA 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--policy.lora_rank` | 0 | LoRA rank（0=不使用） |
| `--policy.lora_alpha` | 16 | LoRA alpha |
| `--policy.lora_dropout` | 0.1 | LoRA dropout |

### 4.4 优化器参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--policy.optimizer_lr` | 1e-4 | 学习率 |
| `--policy.optimizer_weight_decay` | 1e-5 | 权重衰减 |
| `--policy.warmup_ratio` | 0.05 | 预热比例 |
| `--policy.use_bf16` | true | 使用 bfloat16 |

---

## 5. 低显存配置（24GB GPU）

```bash
HF_ENDPOINT=https://hf-mirror.com lerobot-train \
    --policy.type=groot \
    --dataset.repo_id=your_username/agilex_dataset1 \
    --output_dir=outputs/train/groot_agilex_lite \
    --job_name=groot_agilex_lite \
    --policy.device=cuda \
    --policy.chunk_size=16 \
    --policy.n_action_steps=16 \
    --policy.base_model_path=nvidia/GR00T-N1.5-3B \
    --policy.lora_rank=8 \
    --policy.tune_llm=false \
    --policy.tune_visual=false \
    --policy.push_to_hub=false \
    --batch_size=2 \
    --steps=10000 \
    --wandb.enable=false
```

---

## 6. 从断点恢复训练

```bash
HF_ENDPOINT=https://hf-mirror.com lerobot-train \
    --config_path=outputs/train/groot_agilex_lora/checkpoints/last/pretrained_model/train_config.json \
    --resume=true
```

---

## 7. 常见问题

| 问题 | 解决方案 |
|------|----------|
| CUDA out of memory | 使用 LoRA 微调，减小 batch_size |
| 下载模型超时 | 设置 `HF_ENDPOINT=https://hf-mirror.com` |
| Flash Attention 错误 | 安装 flash-attn 或设置环境变量禁用 |
| 找不到 tokenizer | 确保 eagle2hg-processor 已下载 |

---

**下一步：训练完成后使用 `lerobot-eval` 或在真实机器人上测试策略。**

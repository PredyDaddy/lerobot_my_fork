# Agilex Piper 双臂机器人 π₀.₅ (Pi0.5) 训练指南

本文档指导你使用 π₀.₅ (Pi0.5) 视觉-语言-动作模型训练 Agilex Piper 双臂机器人策略。

---

## 1. π₀.₅ 简介

π₀.₅ 是 Physical Intelligence 开发的视觉-语言-动作 (VLA) 模型，具有开放世界泛化能力。

**核心特点：**
- 基于 PaliGemma 视觉语言模型
- 使用 Flow Matching 进行动作生成
- 支持离散状态输入（与 π₀ 不同）
- 使用 AdaRMS 条件化机制
- 更长的 tokenizer 长度（200 tokens）

**论文**: https://arxiv.org/abs/2504.16054

| 特性 | π₀ | π₀.₅ |
|------|-----|-------|
| 时间条件化 | action_time_mlp | AdaRMS |
| Tokenizer 长度 | 48 | 200 |
| 离散状态输入 | ❌ | ✅ |
| 泛化能力 | 强 | 更强 |

---

## 2. 前置准备

### 2.1 安装依赖

```bash
# 安装 pi0/pi05 相关依赖
pip install -e ".[pi]"
```

### 2.2 配置 Hugging Face 镜像（国内用户必须）

π₀.₅ 需要从 Hugging Face Hub 下载预训练权重和 tokenizer：

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
| `lerobot/pi05_base` | π₀.₅ 预训练权重 | ~6GB |
| `google/paligemma-3b-pt-224` | PaliGemma tokenizer | ~100MB |

**手动预下载（推荐）：**
```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download lerobot/pi05_base
huggingface-cli download google/paligemma-3b-pt-224
```

### 2.3 GPU 要求

| 训练方式 | 最小显存 | 推荐显存 | 说明 |
|----------|----------|----------|------|
| 微调（bfloat16 + gradient_checkpointing） | 24GB | 40GB+ | 推荐配置 |
| 微调（float32） | 40GB+ | 80GB+ | 不推荐 |

### 2.4 数据集要求

- `observation.state`: 关节状态
- `observation.images.*`: 相机图像（会 resize 到 224x224）
- `action`: 动作
- `task`: 语言任务描述（必填，预处理会报错如果缺失）

> 数据集需包含 quantile 统计（state/action 用 `NormalizationMode.QUANTILES`）；使用当前版本 `lerobot-record` 录制或先运行 quantile 增补脚本。

---

## 3. 训练方式

### 3.1 微调预训练的 π₀.₅（推荐）

```bash
HF_ENDPOINT=https://hf-mirror.com lerobot-train \
    --policy.path=lerobot/pi05_base \
    --dataset.repo_id=your_username/agilex_dataset1 \
    --output_dir=outputs/train/pi05_agilex \
    --job_name=pi05_agilex \
    --policy.device=cuda \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=true \
    --policy.push_to_hub=false \
    --batch_size=16 \
    --steps=30000 \
    --save_freq=5000 \
    --log_freq=100 \
    --wandb.enable=false
```

### 3.2 从头训练 π₀.₅

```bash
HF_ENDPOINT=https://hf-mirror.com lerobot-train \
    --policy.type=pi05 \
    --dataset.repo_id=your_username/agilex_dataset1 \
    --output_dir=outputs/train/pi05_agilex_scratch \
    --job_name=pi05_agilex_scratch \
    --policy.device=cuda \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=true \
    --policy.push_to_hub=false \
    --batch_size=8 \
    --steps=100000 \
    --save_freq=10000 \
    --log_freq=100 \
    --wandb.enable=false
```

---

## 4. π₀.₅ 关键参数

### 4.1 模型结构参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--policy.chunk_size` | 50 | 动作预测长度 |
| `--policy.n_action_steps` | 50 | 实际执行步数 |
| `--policy.num_inference_steps` | 10 | Flow Matching 推理步数 |
| `--policy.max_state_dim` | 32 | 最大状态维度 |
| `--policy.max_action_dim` | 32 | 最大动作维度 |
| `--policy.image_resolution` | (224, 224) | 输入图像分辨率 |

### 4.2 模型变体参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--policy.paligemma_variant` | gemma_2b | VLM backbone (gemma_300m/gemma_2b) |
| `--policy.action_expert_variant` | gemma_300m | 动作专家 (gemma_300m/gemma_2b) |
| `--policy.dtype` | float32 | 数据类型 (bfloat16/float32) |

### 4.3 优化器参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--policy.optimizer_lr` | 2.5e-5 | 学习率 |
| `--policy.optimizer_weight_decay` | 0.01 | 权重衰减 |
| `--policy.optimizer_grad_clip_norm` | 1.0 | 梯度裁剪 |
| `--policy.scheduler_warmup_steps` | 1000 | 预热步数 |
| `--policy.scheduler_decay_steps` | 30000 | 衰减步数 |

### 4.4 内存优化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--policy.gradient_checkpointing` | false | 梯度检查点（省显存）|
| `--policy.compile_model` | false | 使用 torch.compile |

---

## 5. 低显存配置（24GB GPU）

```bash
HF_ENDPOINT=https://hf-mirror.com lerobot-train \
    --policy.path=lerobot/pi05_base \
    --dataset.repo_id=your_username/agilex_dataset1 \
    --output_dir=outputs/train/pi05_agilex_lite \
    --job_name=pi05_agilex_lite \
    --policy.device=cuda \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=true \
    --policy.push_to_hub=false \
    --batch_size=4 \
    --steps=30000 \
    --wandb.enable=false
```

---

## 6. 从断点恢复训练

```bash
HF_ENDPOINT=https://hf-mirror.com lerobot-train \
    --config_path=outputs/train/pi05_agilex/checkpoints/last/pretrained_model/train_config.json \
    --resume=true
```

---

## 7. 常见问题

| 问题 | 解决方案 |
|------|----------|
| CUDA out of memory | 启用 gradient_checkpointing，减小 batch_size |
| 下载模型超时 | 设置 `HF_ENDPOINT=https://hf-mirror.com` |
| bfloat16 不支持 | 确保 GPU 支持 bf16（Ampere 及以上） |
| tokenizer 加载失败 | 确保 paligemma tokenizer 已下载 |

---

**下一步：训练完成后使用 `lerobot-eval` 或在真实机器人上测试策略。**

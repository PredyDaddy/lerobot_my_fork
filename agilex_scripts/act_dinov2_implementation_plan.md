# ACT_DINOv2 实现计划

## 概述

基于 `ACT_DINOV2_PLAN.md` 文档，需要实现一个新的 policy 类型 `act_dinov2`，将 ACT 算法的视觉 backbone 从 ResNet 替换为 DINOv2。

---

## 需要实现/修改的文件清单

### 1. 核心 Policy 文件（2个新文件）

| 文件路径 | 功能说明 |
|---------|---------|
| `src/lerobot/policies/act_dinov2/configuration_act_dinov2.py` | **配置类**：定义 `ACTDinov2Config`，包含 DINOv2 相关配置参数 |
| `src/lerobot/policies/act_dinov2/modeling_act_dinov2.py` | **模型类**：实现 `ACTDinov2Policy` 和 `ACTDinov2` 网络 |

### 2. 模块初始化文件（1个新文件）

| 文件路径 | 功能说明 |
|---------|---------|
| `src/lerobot/policies/act_dinov2/__init__.py` | 模块导出，暴露 `ACTDinov2Config` 和 `ACTDinov2Policy` |

### 3. 工厂注册（1个修改文件）

| 文件路径 | 修改内容 |
|---------|---------|
| `src/lerobot/policies/factory.py` | 在 `get_policy_class` 和 `make_policy_config` 中添加 `act_dinov2` 分支 |

### 4. 训练/推理脚本（3个新文件）

| 文件路径 | 功能说明 |
|---------|---------|
| `agilex_scripts/train_act_dinov2.sh` | **训练脚本**：完整的训练命令 |
| `agilex_scripts/train_act_dinov2_resume.sh` | **续训脚本**：从 checkpoint 恢复训练 |
| `agilex_scripts/infer_act_dinov2.sh` | **推理脚本**：加载模型进行推理部署 |

### 5. 测试文件（1个新文件）

| 文件路径 | 功能说明 |
|---------|---------|
| `tests/policies/test_act_dinov2.py` | **单元测试**：验证配置创建、policy 实例化、forward/backward 正确性 |

---

## 各文件详细功能说明

### 文件 1: `configuration_act_dinov2.py`

**功能**：定义 `ACTDinov2Config` 配置类

**需要实现的内容**：

1. 使用 `@PreTrainedConfig.register_subclass("act_dinov2")` 注册
2. 继承 `ACTConfig`，复用 ACT 的 chunk/vae/transformer 参数
3. 新增 DINOv2 特有字段：

| 字段名 | 类型 | 默认值 | 说明 |
|-------|------|--------|------|
| `dinov2_model_name_or_path` | `str` | - | 模型路径（本地或 HuggingFace） |
| `dinov2_revision` | `str \| None` | `None` | 模型版本 |
| `dinov2_local_files_only` | `bool` | `True` | 是否仅使用本地文件 |
| `dinov2_image_size` | `int` | `224` | 输入图像尺寸（必须能被 patch_size 整除） |
| `dinov2_interpolation` | `str` | `"bicubic"` | 图像缩放插值方式 |
| `dinov2_antialias` | `bool` | `True` | 是否使用抗锯齿 |
| `dinov2_output_mode` | `str` | `"grid"` | 输出模式：`grid`/`cls`/`mean_pool` |
| `dinov2_use_last_n_layers` | `int` | `1` | 使用最后 N 层的特征 |
| `freeze_backbone` | `bool` | `True` | 是否冻结 backbone |
| `optimizer_lr_backbone` | `float` | `1e-6` | backbone 学习率 |

4. 在 `__post_init__` 中做参数校验：
   - `dinov2_image_size % patch_size == 0`
   - `dinov2_output_mode` 必须在 `["grid", "cls", "mean_pool"]` 中
   - 若 `temporal_ensemble_coeff != None` 则 `n_action_steps == 1`

---

### 文件 2: `modeling_act_dinov2.py`

**功能**：实现 `ACTDinov2Policy` 和 `ACTDinov2` 网络

**需要实现的内容**：

#### 2.1 `Dinov2Backbone` 类

```
功能：封装 DINOv2 模型作为视觉 backbone

方法：
- __init__(config): 加载 DINOv2 模型（使用 transformers.AutoModel）
- forward(img) -> feature_map:
    1. Resize 图像到 dinov2_image_size
    2. 通过 DINOv2 提取特征
    3. 提取 patch tokens（去掉 CLS token）
    4. Reshape 为 (B, hidden, H_p, W_p) 的 feature map

属性：
- output_dim: 输出特征维度
```

#### 2.2 `ACTDinov2` 类

```
功能：ACT 网络，使用 DINOv2 作为 backbone

结构：
- self.backbone: Dinov2Backbone（替换原来的 ResNet）
- self.backbone_proj: Conv2d(1x1) 投影层
- self.encoder: Transformer Encoder
- self.decoder: Transformer Decoder
- 其他与原 ACT 相同
```

#### 2.3 `ACTDinov2Policy` 类

```
功能：Policy 封装，处理训练和推理逻辑

方法：
- __init__(config, dataset_stats): 初始化模型
- forward(batch): 训练时的 forward，返回 loss
- select_action(batch): 推理时选择动作
- reset(): 重置 action queue

特殊处理：
- backbone 冻结逻辑
- 分层学习率（backbone vs head）
- action queue 和 temporal ensemble
```

---

### 文件 3: `__init__.py`

**功能**：模块初始化和导出

```python
from .configuration_act_dinov2 import ACTDinov2Config
from .modeling_act_dinov2 import ACTDinov2Policy

__all__ = ["ACTDinov2Config", "ACTDinov2Policy"]
```

---

### 文件 4: `factory.py` (修改)

**需要修改的位置**：

1. 顶部添加 import：
```python
from lerobot.policies.act_dinov2 import ACTDinov2Config, ACTDinov2Policy
```

2. `get_policy_class` 函数添加分支：
```python
elif name == "act_dinov2":
    return ACTDinov2Policy
```

3. `make_policy_config` 函数添加分支：
```python
elif policy_type == "act_dinov2":
    return ACTDinov2Config(**kwargs)
```

---

### 文件 5: `train_act_dinov2.sh`

**功能**：训练脚本

**脚本内容**：

```bash
#!/bin/bash

# 设置离线环境变量
export CUDA_VISIBLE_DEVICES=7
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 路径配置
DATASET_PATH="/path/to/agilex_dataset"
DINOV2_PATH="/path/to/dinov2-base"
OUTPUT_DIR="/path/to/outputs/train/act_dinov2_agilex"

# 训练命令
lerobot-train \
  --dataset.repo_id=agilex_dataset \
  --dataset.root=${DATASET_PATH} \
  --dataset.use_imagenet_stats=true \
  --policy.type=act_dinov2 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.dinov2_model_name_or_path=${DINOV2_PATH} \
  --policy.dinov2_local_files_only=true \
  --policy.dinov2_image_size=224 \
  --policy.dinov2_output_mode=grid \
  --policy.freeze_backbone=true \
  --batch_size=32 \
  --steps=100000 \
  --save_freq=10000 \
  --log_freq=100 \
  --eval_freq=10000 \
  --output_dir=${OUTPUT_DIR} \
  --job_name=act_dinov2_agilex \
  --wandb.enable=false
```

---

### 文件 6: `train_act_dinov2_resume.sh`

**功能**：续训脚本

**脚本内容**：

```bash
#!/bin/bash

export CUDA_VISIBLE_DEVICES=7
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CHECKPOINT_PATH="/path/to/outputs/train/act_dinov2_agilex/checkpoints/010000/pretrained_model"
OUTPUT_DIR="/path/to/outputs/train/act_dinov2_agilex"

lerobot-train \
  --resume=true \
  --config_path=${CHECKPOINT_PATH}/train_config.json \
  --output_dir=${OUTPUT_DIR}
```

---

### 文件 7: `infer_act_dinov2.sh`

**功能**：推理脚本

**脚本内容**：

```bash
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CHECKPOINT_PATH="/path/to/outputs/train/act_dinov2_agilex/checkpoints/last/pretrained_model"

# 推理命令（参考现有 infer 脚本结构）
python -m lerobot.scripts.control_robot \
  --robot.type=agilex \
  --control.type=record \
  --control.fps=30 \
  --control.policy.path=${CHECKPOINT_PATH} \
  --control.policy.device=cuda
```

---

### 文件 8: `test_act_dinov2.py`

**功能**：单元测试

**测试用例**：

```python
# 测试 1: 配置创建
def test_make_policy_config():
    config = make_policy_config("act_dinov2")
    assert config.type == "act_dinov2"
    assert config.dinov2_image_size == 224

# 测试 2: Policy 类获取
def test_get_policy_class():
    policy_cls = get_policy_class("act_dinov2")
    assert policy_cls == ACTDinov2Policy

# 测试 3: Forward/Backward
def test_forward_backward():
    # 创建假数据
    # 初始化 policy
    # 执行 forward
    # 执行 backward
    # 验证无报错

# 测试 4: Action Shape
def test_action_shape():
    # 验证输出 action 维度与数据集一致
```

---

## 实现里程碑

| 阶段 | 目标 | 验收标准 |
|-----|------|---------|
| **M0** | 能 import/注册 | `make_policy_config("act_dinov2")` 成功 |
| **M1** | CPU 冒烟测试 | `lerobot-train --steps=1 --policy.device=cpu` 完成并保存 checkpoint |
| **M2** | GPU 冒烟测试 | 10 step 训练，检查显存和速度正常 |
| **M3** | 短训验证 | 5k-10k step，验证 checkpoint 可用于推理 |
| **M4** | 正式长训 | 10 万 step 完整训练 |

---

## 冒烟测试命令

### CPU 冒烟测试（1 step）

```bash
lerobot-train \
  --dataset.repo_id=agilex_dataset \
  --dataset.root=/path/to/dataset \
  --policy.type=act_dinov2 \
  --policy.device=cpu \
  --policy.dinov2_model_name_or_path=/path/to/dinov2 \
  --steps=1 \
  --batch_size=1 \
  --num_workers=0 \
  --save_freq=1 \
  --log_freq=1
```

### GPU 冒烟测试（10 step）

```bash
lerobot-train \
  --dataset.repo_id=agilex_dataset \
  --dataset.root=/path/to/dataset \
  --policy.type=act_dinov2 \
  --policy.device=cuda \
  --policy.dinov2_model_name_or_path=/path/to/dinov2 \
  --steps=10 \
  --batch_size=8 \
  --save_freq=10 \
  --log_freq=1
```

---

## 文件数量汇总

| 类型 | 数量 |
|-----|------|
| 新增 Python 文件 | 4 个 |
| 修改 Python 文件 | 1 个 |
| 新增 Shell 脚本 | 3 个 |
| **总计** | **8 个文件** |

---

## 常见问题避坑

| 问题 | 现象 | 解决方案 |
|-----|------|---------|
| 图像尺寸不对齐 | reshape 失败 / OOM | 默认 `dinov2_image_size=224` 且强校验 |
| 未安装 transformers | import error | `pip install transformers` |
| 离线环境权重缺失 | 下载失败 | 提前下载到本地，使用 `local_files_only=true` |
| 多相机 token 太多 | 显存爆 / 速度慢 | 减小 `dinov2_image_size` 或使用 `cls/mean_pool` |
| backbone 冻结策略不当 | loss 不下降 | 先冻结 backbone，再小 LR 解冻 |

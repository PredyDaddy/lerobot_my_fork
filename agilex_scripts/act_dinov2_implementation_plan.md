# ACT + DINOv2（act_dinov2）实现计划（对齐本仓库）

> 目的：在 LeRobot 框架中新增一个 **policy 类型** `act_dinov2`，保留 ACT 的 action-chunk + transformer 结构，但把视觉 backbone 替换为 DINOv2（ViT）。

本文是“能落地”的实现计划，已针对当前仓库的训练/推理/续训机制做过校对，避免出现“写了但跑不起来”的情况。

---

## 0. 可行性结论

可行，但必须遵守本仓库的几条约束：

- `lerobot-train` / `lerobot-record` 的 policy 创建流程只会传入 `config`（不会传 `dataset_stats`），所以 `Policy.__init__` 必须兼容现有工厂逻辑。
- `src/lerobot/policies/factory.py` 使用 **动态 import policy** 来避免启动时加载重依赖；`act_dinov2` 也应遵循这个模式。
- 续训必须使用 `--config_path=...`（注意必须带 `=`），否则解析器读取不到参数会直接报错。
- 推理入口应基于 `lerobot-record`（或改造你现有的 `agilex_infer.py`），仓库里没有 `lerobot.scripts.control_robot`。

---

## 1. 本计划相对旧版本的关键修正点（必须）

1) **factory.py 不要在顶部 import policy 类**
- 你原计划在 `factory.py` 顶部 `import ACTDinov2Policy`，会破坏当前 “dynamic import policy” 的设计（参考 `src/lerobot/policies/factory.py:53`）。
- 正确做法：`factory.py` 顶部只 import `ACTDinov2Config`；policy 在 `get_policy_class()` 的 `elif` 分支里再 import。

2) **Policy 的 `__init__` 不要设计成 `(config, dataset_stats)`**
- 训练和 `lerobot-record` 都通过 `make_policy(cfg.policy, ds_meta=dataset.meta)` 创建 policy（参考 `src/lerobot/policies/factory.py:339`），只传 `config`。
- `dataset_stats` 走 `make_pre_post_processors(... dataset_stats=...)` 注入 processor（参考 `src/lerobot/scripts/lerobot_record.py:477`）。

3) **续训脚本里 `--config_path` 必须写成 `--config_path=...`**
- `parser.parse_arg()` 只识别 `--xxx=yyy` 形式（参考 `src/lerobot/configs/parser.py:58`）。

4) **推理脚本入口要改为 `lerobot-record` 或 `agilex_infer.py`**
- 本仓库没有 `lerobot.scripts.control_robot`；标准入口是 `lerobot-record`（参考 `src/lerobot/scripts/lerobot_record.py:1` 顶部示例）。

---

## 2. 推荐的实现路径（最少改动、低风险）

### 2.1 新增 policy 类型（推荐而不是改造现有 act）

- 新增 `act_dinov2`（独立于 `act`），避免影响现有 ACT 的稳定性。
- 训练命令使用：`--policy.type=act_dinov2`

### 2.2 配置类：`ACTDinov2Config`（推荐继承 ACTConfig）

**为什么推荐继承 `ACTConfig`：**
- 可以直接复用现有 ACT 的 processor 分支：`make_pre_post_processors` 里是 `isinstance(policy_cfg, ACTConfig)`（参考 `src/lerobot/policies/factory.py:269`）。
- 可复用 ACT 的默认超参、chunk_size、VAE 参数、`optimizer_lr_backbone` 等字段。

**注意：** `ACTConfig.__post_init__` 强制要求 `vision_backbone` 为 resnet（参考 `src/lerobot/policies/act/configuration_act.py:140`）。  
为了最少风险，第一版建议：
- **不要改动 `vision_backbone` 字段（保持默认 resnet18 即可）**
- 在 `act_dinov2` 的实现里 **完全不使用** `vision_backbone`，DINOv2 由新字段控制  
（后续如果你想“干净化”，再把 `ACTDinov2Config` 改为不继承 ACTConfig 并新增专用 processor 分支。）

**必须新增的字段（建议）**

| 字段名 | 类型 | 默认值 | 说明 |
|---|---:|---:|---|
| `dinov2_model_name_or_path` | `str` | 必填 | 训练首次初始化时加载 DINOv2 权重/配置（本地目录或 HF repo id） |
| `dinov2_local_files_only` | `bool` | `True` | 离线训练更安全（你的环境 network restricted） |
| `dinov2_revision` | `str \| None` | `None` | 固定模型版本（可选） |
| `dinov2_image_size` | `int` | `224` | 强烈建议默认 224（224 % 14 == 0） |
| `dinov2_output_mode` | `str` | `"grid"` | 第一版只实现 `grid`，`cls/mean_pool` 作为可选后续项 |
| `dinov2_use_last_n_layers` | `int` | `1` | 可选：平均最后 N 层的 hidden states |
| `freeze_backbone` | `bool` | `True` | 默认先冻结，减少训练不稳定和显存压力 |
| `optimizer_lr_backbone` | `float` | 复用 ACTConfig | ACTConfig 已有该字段，act_dinov2 只需要设置更小默认值即可（例如 1e-6） |

**校验（fail-fast）**
- `dinov2_image_size % 14 == 0`（你当前 conda 环境里 `Dinov2Config().patch_size == 14`）
- `dinov2_output_mode` 只能是 `grid`（第一版）
- 如果 `temporal_ensemble_coeff != None`，要求 `n_action_steps == 1`（沿用 ACT 规则）

### 2.3 模型类：`ACTDinov2Policy` / `ACTDinov2`

**推荐策略：尽量复用现有 ACT 的逻辑，避免引入新 bug**

- `ACTDinov2Policy` 的职责：
  - 和 `ACTPolicy` 一样：管理 action queue / temporal ensemble / forward loss
  - 区别：`self.model = ACTDinov2(config)`，backbone 替换为 DINOv2
  - `get_optim_params()` 建议沿用 ACT 的 “backbone 与其它参数分组”，前提是你把 DINOv2 放到 `self.model.backbone` 下面（这样 `name.startswith("model.backbone")` 的过滤逻辑就继续可用）

- `ACTDinov2(nn.Module)` 的职责：
  - DINOv2 提取图像特征 → 变成网格 feature map → 投影到 `dim_model` → 拼 token → Transformer encoder/decoder → 输出 action chunk
  - 复用 `modeling_act.py` 中 encoder/decoder/VAE/pos embed 的实现，尽量只替换 backbone 部分

**DINOv2 backbone 细节（第一版只做 grid）**

- 输入：`img`（形状来自 dataset_to_policy_features：`(B, 3, H, W)`，你的数据是 `(3, 480, 640)`）
- 处理：
  1. 在模型内 `interpolate` resize 到 `dinov2_image_size`（默认 224）
  2. `Dinov2Model` forward，取 patch tokens（去掉 CLS）
  3. reshape 成 `(B, hidden, H_p, W_p)` 的 feature map（224/14=16 → 16x16）
  4. `Conv2d(1x1)` 投影到 `dim_model`
  5. flatten 成 token 序列并加 2D pos embed（沿用 ACT 的 `ACTSinusoidalPositionEmbedding2d`）

**重要设计选择：checkpoint 是否需要依赖外部 dinov2 路径？（推荐：不依赖）**

为避免你在 SmolVLA 上遇到的“推理还要额外模型路径”的坑，建议 act_dinov2 做成：

- **DINOv2 作为 `nn.Module` 的一部分，权重随 `model.safetensors` 一起保存**（自然自包含）
- 加一个字段 `dinov2_config_json: dict | None`（或等价命名）：
  - 首次训练启动时：从 `dinov2_model_name_or_path` 读取 `Dinov2Config` 并存入该字段
  - 从 checkpoint 加载时：如果 `dinov2_config_json` 已存在，则用它构造 `Dinov2Model(Dinov2Config(**...))`，**不需要外部 dinov2 目录**，随后 safetensors 会把权重正确加载进去

这样推理/续训机器只要拷 checkpoint（`pretrained_model` 目录）就能跑，不需要额外复制 dinov2 权重目录。

---

## 3. 需要实现/修改的文件清单（按优先级）

### 3.1 必需（最小闭环）

| 文件路径 | 类型 | 目的 |
|---|---|---|
| `src/lerobot/policies/act_dinov2/configuration_act_dinov2.py` | 新增 | 注册 config：`type=act_dinov2` |
| `src/lerobot/policies/act_dinov2/modeling_act_dinov2.py` | 新增 | 实现 policy 和模型（DINOv2 backbone） |
| `src/lerobot/policies/factory.py` | 修改 | 注册 `act_dinov2` 的 config 与 policy（动态 import policy） |

> `src/lerobot/policies/act_dinov2/__init__.py` 可选：建议保持“轻量”，不要 import modeling（避免 import 即触发 transformers 重依赖）。

### 3.2 强烈建议（降低训练失败概率）

| 文件路径 | 类型 | 目的 |
|---|---|---|
| `tests/policies/test_act_dinov2.py` | 新增 | 最小 forward/backward + shape 校验 + 注册校验 |
| `agilex_scripts/train_act_dinov2.sh` | 新增 | 固化训练命令（含 fail-fast 检查、日志） |
| `agilex_scripts/train_act_dinov2_resume.sh` | 新增 | 固化续训命令（使用 `--config_path=...`） |
| `agilex_scripts/infer_act_dinov2.sh` | 新增 | 基于 `lerobot-record` 的推理/评估脚本（可参考 `agilex_scripts/infer_smolvla.sh`） |

---

## 4. factory.py 注册方式（正确写法）

### 4.1 make_policy_config：新增 `act_dinov2`

- 在 `src/lerobot/policies/factory.py` 顶部 import config（只 import config，别 import policy）：
  - `from lerobot.policies.act_dinov2.configuration_act_dinov2 import ACTDinov2Config`

- `make_policy_config()` 增加分支：
  - `elif policy_type == "act_dinov2": return ACTDinov2Config(**kwargs)`

### 4.2 get_policy_class：动态 import policy

在 `get_policy_class()` 中增加：

```python
elif name == "act_dinov2":
    from lerobot.policies.act_dinov2.modeling_act_dinov2 import ACTDinov2Policy
    return ACTDinov2Policy
```

这样只有在用户真的指定 `--policy.type=act_dinov2` 时，才会 import transformers/dinov2 相关代码。

---

## 5. 训练脚本建议（AgileX 数据集对齐）

> 你当前数据集 repo id 是 `agilex_dataset1`，root 在仓库里是 `.../agilex_dataset1`。

### 5.1 train_act_dinov2.sh（示例模板）

关键点：
- 默认使用 7 卡：`CUDA_VISIBLE_DEVICES=7`
- 离线：`HF_HUB_OFFLINE=1` 等
- `--dataset.use_imagenet_stats=true`：让视觉输入 mean/std 对齐 ImageNet（DINOv2 也更稳定）

```bash
export CUDA_VISIBLE_DEVICES=7
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

lerobot-train \
  --dataset.repo_id=agilex_dataset1 \
  --dataset.root=/abs/path/to/agilex_dataset1 \
  --dataset.use_imagenet_stats=true \
  --policy.type=act_dinov2 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.dinov2_model_name_or_path=/abs/path/to/dinov2 \
  --policy.dinov2_local_files_only=true \
  --policy.dinov2_image_size=224 \
  --policy.dinov2_output_mode=grid \
  --policy.freeze_backbone=true \
  --batch_size=32 \
  --steps=100000 \
  --save_freq=10000 \
  --eval_freq=10000 \
  --log_freq=100 \
  --output_dir=/abs/path/to/outputs/train/act_dinov2_agilex \
  --job_name=act_dinov2_agilex \
  --wandb.enable=false
```

### 5.2 强制要求：先跑 smoke（避免长训才发现爆炸）

CPU smoke（最小闭环）：
```bash
lerobot-train \
  --dataset.repo_id=agilex_dataset1 \
  --dataset.root=/abs/path/to/agilex_dataset1 \
  --dataset.use_imagenet_stats=true \
  --policy.type=act_dinov2 \
  --policy.device=cpu \
  --policy.dinov2_model_name_or_path=/abs/path/to/dinov2 \
  --policy.dinov2_local_files_only=true \
  --policy.dinov2_image_size=224 \
  --steps=1 --batch_size=1 --num_workers=0 --save_freq=1 --log_freq=1 \
  --output_dir=/abs/path/to/outputs/train/_smoke_act_dinov2
```

GPU smoke（确认显存与速度）：
```bash
export CUDA_VISIBLE_DEVICES=7
lerobot-train \
  --dataset.repo_id=agilex_dataset1 \
  --dataset.root=/abs/path/to/agilex_dataset1 \
  --dataset.use_imagenet_stats=true \
  --policy.type=act_dinov2 \
  --policy.device=cuda \
  --policy.dinov2_model_name_or_path=/abs/path/to/dinov2 \
  --policy.dinov2_local_files_only=true \
  --policy.dinov2_image_size=224 \
  --steps=10 --batch_size=2 --num_workers=0 --save_freq=10 --log_freq=1 \
  --output_dir=/abs/path/to/outputs/train/_smoke_act_dinov2_gpu
```

---

## 6. 续训脚本（必须按仓库规则写）

`lerobot-train` 续训要求：
- `--resume=true`
- `--config_path=.../train_config.json`（必须 `=` 号）
- `--output_dir=...`（输出目录是 run 根目录）

示例：
```bash
export CUDA_VISIBLE_DEVICES=7
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

lerobot-train \
  --resume=true \
  --config_path=/abs/path/to/outputs/train/act_dinov2_agilex/checkpoints/010000/pretrained_model/train_config.json \
  --output_dir=/abs/path/to/outputs/train/act_dinov2_agilex
```

---

## 7. 推理/部署脚本（正确入口）

### 7.1 推荐：用 `lerobot-record` 让 policy 控制 robot（可顺便录评估数据）

`lerobot-record` 支持 `--policy.path=.../pretrained_model`，会加载 policy + processor 并控制机器人（参考 `src/lerobot/scripts/lerobot_record.py:16` 示例）。

建议你直接参考并裁剪现有的：
- `agilex_scripts/infer_smolvla.sh`（已经把 agilex topics、mock、日志等处理得很完整）

把其中 policy 部分替换为：
- `--policy.path=${CHECKPOINT_DIR}/pretrained_model`
- 其它与 VLM 相关的检查/参数都删除

### 7.2 可选：改造 `agilex_infer.py` 做“纯控制不录数据”

你仓库里已有 `agilex_infer.py`（目前硬编码 `ACTPolicy`）。如果要支持 `act_dinov2`，建议把它改成：
- 从 checkpoint 读取 `PreTrainedConfig` → `make_policy` → `make_pre_post_processors`  
这样未来换任何 policy type 都不用改脚本。

---

## 8. 单元测试建议（最小但高价值）

你的测试目标不是“评估效果”，而是“防止集成错误导致训练白跑”：

1) 注册是否成功：
- `make_policy_config("act_dinov2", ...)` 能创建 config
- `get_policy_class("act_dinov2")` 能拿到 policy class

2) 最小 forward/backward：
- 用小尺寸假图像（例如 1x3x64x64 或 1x3x224x224）+ 假 state/action
- 跑一次 `loss.backward()` 不报错

3) shape 校验：
- 输出 action shape 必须等于数据集 action 维度（你的 AgileX 是 14）

**注意：** 测试不要依赖外网下载权重：
- 用 `Dinov2Config()` + `Dinov2Model(Dinov2Config())` 初始化随机权重做测试（或者在 transformers 不存在时 skip）

---

## 9. 常见坑（按“会浪费你训练时间”的概率排序）

| 坑 | 典型现象 | 规避手段 |
|---|---|---|
| `--config_path` 少了 `=` | 续训报 “A config_path is expected …” | 强制写成 `--config_path=...` |
| 推理脚本调用不存在模块 | 启动即 ModuleNotFoundError | 只用 `lerobot-record` 或改造 `agilex_infer.py` |
| 没做 resize/patch 对齐 | OOM 或极慢 | 默认 `dinov2_image_size=224` 且 fail-fast 校验 |
| transformers/权重不可用 | import/download 失败 | dynamic import + local_files_only + 先跑 CPU smoke |
| checkpoint 依赖外部 dinov2 路径 | 换机器推理失败 | 推荐保存 dinov2_config 并让 checkpoint 自包含 |

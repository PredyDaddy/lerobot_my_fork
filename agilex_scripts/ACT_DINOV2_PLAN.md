# ACT + DINOv2 作为新算法接入 LeRobot 的详细方案（不写代码版）

目标：在当前 `lerobot` 框架里**新增并注册一个新的 policy 类型**（建议命名 `act_dinov2`），算法主体仍是 ACT（Action Chunking Transformer），但把视觉 backbone 从 torchvision ResNet 替换为 **DINOv2**（ViT 系列），可用于离线训练与后续推理部署。

本文是“怎么做”的工程方案与检查清单，按这个方案实现后，你可以通过 `lerobot-train --policy.type=act_dinov2 ...` 像现有算法一样训练/保存/恢复/推理。

---

## 1. 先理解：LeRobot 里“注册一个新算法”需要改哪些地方

LeRobot 的训练入口是 `lerobot-train`（`src/lerobot/scripts/lerobot_train.py`），核心流程：

1) 解析训练配置 `TrainPipelineConfig`（`src/lerobot/configs/train.py`）  
2) `make_dataset(cfg)` 生成 `dataset`（`src/lerobot/datasets/factory.py`）  
3) `make_policy(cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)` 生成 policy（`src/lerobot/policies/factory.py`）  
4) `make_pre_post_processors(...)` 生成前后处理（`src/lerobot/policies/factory.py`）  
5) 训练循环 + `save_checkpoint` 保存 checkpoint（`src/lerobot/utils/train_utils.py`）

因此，想新增 `act_dinov2`，你至少要做到：

- **配置层注册**：让 `PreTrainedConfig` 能识别一个新的 `type="act_dinov2"`（类似现有 `@PreTrainedConfig.register_subclass("act")`）。
- **Policy 类可被 factory 找到**：让 `get_policy_class("act_dinov2")` 返回你的 `ACTDinov2Policy`。
- **Policy Config 可被 factory 实例化**：让 `make_policy_config("act_dinov2")` 返回你的 `ACTDinov2Config`。
- **（可选）processor 工厂**：如果你要新增图像 resize / patch 对齐等处理逻辑，可能需要新增一个 processor step，或者把处理写进 policy forward。

---

## 2. 现有 ACT 实现的关键点（你要替换的就是这里）

现有 ACT 在这些位置使用 ResNet：

- `src/lerobot/policies/act/configuration_act.py`
  - `ACTConfig.vision_backbone="resnet18"`，并且 `__post_init__` 里强制要求 `vision_backbone` 必须以 `"resnet"` 开头。
- `src/lerobot/policies/act/modeling_act.py`
  - `ACT.__init__` 中构造 torchvision ResNet + `IntermediateLayerGetter(... layer4 ...)` 得到 feature map
  - forward 时：每个相机图像都走一次 `self.backbone(img)["feature_map"]` → `Conv2d(1x1)` → flatten 成 token → 送进 Transformer encoder/decoder。

因此 DINOv2 接入的核心工作是：

- 用 DINOv2 替换 ResNet 的 feature extractor，并产出 **(B, C, H, W)** 形状的 feature map（或等价 token 序列）供 ACT 继续使用。

---

## 3. 关键设计决策（强烈建议在写代码前先定下来）

### 3.1 是“新增 policy 类型”还是“扩展现有 act 支持 dinov2”？

两种做法：

**A. 新增 policy 类型（推荐）**
- 新增 `ACTDinov2Config` / `ACTDinov2Policy`，`type="act_dinov2"`。
- 好处：不影响现有 `act` 行为，风险最小；以后你可以并行维护 `act` 与 `act_dinov2`。
- 坏处：代码文件会多一些。

**B. 扩展现有 `ACTConfig` 支持 dinov2**
- 直接把 `ACTConfig.vision_backbone` 扩展到 `"dinov2"`，并在 `modeling_act.py` 里加分支。
- 好处：表面上文件更少。
- 坏处：容易破坏 `act` 现有训练/推理；而且 `ACTConfig.__post_init__` 目前会直接拒绝非 resnet。

本文后续以 **方案 A（新增 `act_dinov2`）** 展开。

### 3.2 DINOv2 输出用什么形式喂给 ACT？

DINOv2（ViT）天然输出是 token 序列：
- `last_hidden_state`: `(B, 1+N, hidden)`，包含 `CLS` + patch tokens。

而 ACT 目前期望每个相机得到一个 feature map，然后 flatten 为 token：
- feature map: `(B, C, H, W)` → tokens: `(H*W, B, dim_model)`

建议提供一个可配置项（例如 `dinov2_output_mode`）：

- `grid`（推荐默认）：使用 patch tokens，reshape 成 `(B, hidden, H_p, W_p)`，保持 ACT 的“像素/patch token”风格。
- `cls`：只用 CLS token 作为单 token（每相机 1 个 token），速度快、占用小，但可能损失空间信息。
- `mean_pool`：对 patch tokens 做均值池化（每相机 1 token），比 `cls` 稳一点但仍然丢空间信息。

工程上最稳：先实现 `grid`，并留接口将来加 `cls/mean_pool` 做加速 ablation。

### 3.3 必须处理的“patch 对齐/图像尺寸”问题（最容易踩坑）

大多数 DINOv2 模型 patch size 是 **14**（例如 `dinov2_vits14` / `vitb14` / `vitl14`）。
你的数据集（例如 `agilex_dataset1`）相机图像是 `(3, 480, 640)`，**无法被 14 整除**，如果直接喂给 ViT patch embedding，会出现：
- reshape/token 数不对、或者内部报错
- 即使强行处理，token 数巨大，Transformer encoder 会非常慢/爆显存

必须做一个策略（强烈建议默认固定到 224）：

1) **Resize 到固定尺寸（推荐默认）**
   - 例如 `dinov2_image_size=224`，保证 `224 % 14 == 0`，patch grid = `16x16=256 tokens/相机`。
   - 三相机总 token 数约 `256*3 + 少量 state/latent token`，可控。

2) Padding 到最近可整除尺寸
   - 比如把 480 pad 到 490（14*35），640 pad 到 644（14*46）——但 token 数仍然很大（35*46=1610/相机）。

3) 先强制下采样（例如到 224 或 280），再可选 crop/pad

建议：**默认用 Resize 到 224**；并把 `dinov2_image_size` 做成配置项。

### 3.4 归一化（Normalization）如何对齐 DINOv2 预训练？

LeRobot 的 dataset loader 在 `use_imagenet_stats=true` 时会覆盖相机 stats 为 ImageNet 均值方差：
- 见 `src/lerobot/datasets/factory.py`：`dataset.meta.stats[key] = IMAGENET_STATS`

ACT processor 默认对 `VISUAL` 使用 `MEAN_STD`，因此只要训练命令里确保：
- `--dataset.use_imagenet_stats=true`

就能保证输入 DINOv2 的图像大体符合其预训练规范（至少 mean/std 一致）。

如果未来你想完全复用 HuggingFace 的 `AutoImageProcessor`（带 resize/crop/normalize），也可以把 resize/normalize 放到 processor step，但工程量更大；第一版建议把 resize 放在 policy/backbone 内部，normalize 仍走 LeRobot 的 pipeline。

### 3.5 训练策略：冻结 backbone 还是端到端微调？

建议提供以下配置项：

- `freeze_backbone: bool`（默认 True 或者“前 N 步冻结”）
- `optimizer_lr_backbone`（默认比主干小 10x~100x，例如 `1e-6`）

经验建议（适用于数据量不大时）：
- **先冻结** DINOv2，只训练 ACT transformer/head，让行为先“跑起来”
- 再解冻最后几层（或全部）小 LR 微调

---

## 4. 详细工程落地步骤（按这个顺序做，最稳）

### 4.1 新增配置类：`ACTDinov2Config`

建议新建目录与文件：
- `src/lerobot/policies/act_dinov2/configuration_act_dinov2.py`

配置类建议：
- `@PreTrainedConfig.register_subclass("act_dinov2")`
- 继承关系两种选择：
  - **继承 `ACTConfig`**：复用 ACT 的 chunk/vae/transformer 参数与默认值（推荐）
  - 直接继承 `PreTrainedConfig`：更干净但要重新实现一堆字段（不推荐）

建议新增字段（示例命名，最终以你代码风格为准）：

1) DINOv2 权重来源
- `dinov2_model_name_or_path: str`  
  - 支持 HuggingFace repo id（联网）或本地目录（离线）
- `dinov2_revision: str | None = None`（可选）
- `dinov2_local_files_only: bool = True`（默认离线更安全）

2) 输入图像处理
- `dinov2_image_size: int = 224`（强烈建议默认 224）
- `dinov2_interpolation: str = "bicubic"`（或 bilinear）
- `dinov2_antialias: bool = True`

3) 输出 token 形态
- `dinov2_output_mode: str = "grid"`（`grid|cls|mean_pool`）
- `dinov2_use_last_n_layers: int = 1`（可选：用最后 N 层平均做特征，提升稳定性）

4) 训练控制
- `freeze_backbone: bool = True`
- `optimizer_lr_backbone: float = 1e-6`（或 1e-5，看数据量）

并在 `__post_init__` 中做强校验（fail-fast）：
- `dinov2_image_size % patch_size == 0`（patch_size 可从模型 config 读取；若读取不到可先假设 14 并给出提示）
- `dinov2_output_mode` 必须在允许集合
- 若 `temporal_ensemble_coeff != None` 则 `n_action_steps==1`（沿用 ACT 逻辑）

### 4.2 新增 policy 类：`ACTDinov2Policy`

建议新建：
- `src/lerobot/policies/act_dinov2/modeling_act_dinov2.py`

实现策略：

- 复用 `ACTPolicy` 的大部分逻辑（action queue / temporal ensemble / loss 计算）
- 主要差异是底层 `ACT` 网络的 backbone 部分

你可以选择两种结构：

**结构 1：复制一份 ACT 网络改 backbone（最快可跑通）**
- `class ACTDinov2(nn.Module)`：基本复制 `class ACT(nn.Module)`，但把 `self.backbone` 替换为 DINOv2。
- 风险：重复代码多，后续 ACT 本体改动需要同步维护。

**结构 2：抽象 backbone 接口（推荐长期维护）**
- 把 `modeling_act.py` 中 backbone 相关代码抽成一个小类接口：
  - `extract_feature_map(img)->(B,C,H,W)`
  - `output_dim` 属性
- ResNetBackbone 与 Dinov2Backbone 都实现这个接口
- ACT 网络只依赖接口，不关心具体 backbone

第一版可以先用结构 1 跑通，后续再重构成结构 2。

### 4.3 DINOv2 backbone 的实现细节（你需要的关键行为）

推荐直接使用 `transformers`（因为仓库里 SmolVLA/Gr00t 等已在用）：

- `from transformers import AutoModel` 或 `Dinov2Model`
- `model = AutoModel.from_pretrained(dinov2_model_name_or_path, local_files_only=..., revision=...)`

forward 时：
1) 输入图像 `img` 形状来自 LeRobot：`(B, 3, H, W)`
2) Resize 到 `dinov2_image_size`（比如 224）
3) 喂给 DINOv2，拿到 `last_hidden_state`：
   - 去掉 CLS：`patch_tokens = hidden[:, 1:, :]` 形状 `(B, N, hidden)`
4) 把 `N` reshape 成二维网格 `(H_p, W_p)`：
   - `H_p = W_p = dinov2_image_size / patch_size`（通常是 16）
5) 得到 feature map `(B, hidden, H_p, W_p)` 供 ACT 后续 `Conv2d(1x1)` 投影到 `dim_model`

**注意：token 数与显存/速度直接相关**
- `224x224`、patch14 → 256 tokens/相机（可控）
- `448x448` → 1024 tokens/相机（encoder 会慢很多）

### 4.4 注册到框架：把 `act_dinov2` 接到 factory

需要修改的关键文件（实现时）：

1) `src/lerobot/policies/factory.py`
- `get_policy_class` 增加分支：
  - `elif name == "act_dinov2": return ACTDinov2Policy`
- `make_policy_config` 增加分支：
  - `elif policy_type == "act_dinov2": return ACTDinov2Config(**kwargs)`

2) 确保新 config 文件 import 后注册生效  
（现有模式：factory 顶部直接 import 各个 config；或通过 policy 的 module import 间接注册）

建议：和现有风格一致，在 `factory.py` 顶部显式 import `ACTDinov2Config`，避免“没 import 导致没注册”的隐蔽问题。

### 4.5 processor 是否需要改？

**第一版推荐不改 processor**，直接复用 ACT 的 `make_act_pre_post_processors`：
- `src/lerobot/policies/act/processor_act.py`

原因：
- rename/batch/device/normalize 这些通用逻辑不变
- resize/patch 对齐在 backbone 内做，最简单

只有当你希望“严格复现 DINOv2 官方预处理（含 crop/normalize）”时，才需要新增 processor step。

### 4.6 依赖与安装（不要忽视这一点）

如果你的环境只安装了 `lerobot` 基础依赖，可能没有 `transformers`。DINOv2（HuggingFace 版本）依赖 `transformers`。

建议在项目中增加一个可选依赖组（实现时）：
- `pyproject.toml` 的 `[project.optional-dependencies]` 新增：
  - `act_dinov2 = ["lerobot[transformers-dep]"]`

然后用户可以：
- `pip install -e ".[act_dinov2]"` 或者你们内部 conda env 里直接装 `transformers`

---

## 5. 训练命令与脚本建议（针对 AgileX 数据集）

### 5.1 推荐训练命令（示例）

假设：
- 数据集：`/abs/path/agilex_dataset1`
- DINOv2：本地目录 `/abs/path/dinov2-base`（提前下载好，离线训练）
- 输出目录：`outputs/train/act_dinov2_agilex`
- 使用第 7 张卡：`CUDA_VISIBLE_DEVICES=7`

示例（注意 `config_path` 之类参数建议用 `=` 形式更稳）：

```bash
export CUDA_VISIBLE_DEVICES=7
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

lerobot-train \
  --dataset.repo_id=agilex_dataset1 \
  --dataset.root=/abs/path/agilex_dataset1 \
  --dataset.use_imagenet_stats=true \
  --policy.type=act_dinov2 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.dinov2_model_name_or_path=/abs/path/dinov2-base \
  --policy.dinov2_local_files_only=true \
  --policy.dinov2_image_size=224 \
  --policy.dinov2_output_mode=grid \
  --policy.freeze_backbone=true \
  --batch_size=32 \
  --steps=100000 \
  --save_freq=10000 \
  --log_freq=100 \
  --eval_freq=10000 \
  --output_dir=/abs/path/outputs/train/act_dinov2_agilex \
  --job_name=act_dinov2_agilex \
  --wandb.enable=false
```

### 5.2 重要：避免因为图像分辨率导致训练爆显存/极慢

如果你不做 resize（或者设得太大），token 数会很夸张，ACT transformer encoder 会非常慢，甚至 OOM。

建议你把 `dinov2_image_size` 固定在 224 或 280（必须能整除 patch_size）。

### 5.3 “冒烟测试”(smoke test) 必须做（强烈建议）

在你正式跑 10 万步前，必须做一次**1~10 step** 的 smoke：

目标：
- 能成功走完：dataset → processor → policy forward → backward → save checkpoint
- 输出 action 维度正确（和数据集一致）
- checkpoint 里 config 保存了 `type=act_dinov2` 和你的 dinov2 参数

建议做法：
- `--steps=1 --batch_size=1 --num_workers=0 --save_freq=1 --log_freq=1`
- `--policy.device=cpu` 先跑通逻辑，再上 GPU

（你之前 SmolVLA 6 维问题浪费了很多时间，本质就是没有做“早期失败”的 smoke；这里要把 smoke 当成强制门槛。）

---

## 6. 续训/恢复训练（resume）要怎么做

LeRobot 的 resume 机制要求：
- `--resume true`
- `--config_path=/path/to/train_config.json`（注意通常保存在 checkpoint 下的 `pretrained_model/train_config.json`）

因此你实现/使用 `act_dinov2` 后，恢复训练的标准方式是：

```bash
lerobot-train \
  --resume=true \
  --config_path=/abs/path/outputs/train/act_dinov2_agilex/checkpoints/010000/pretrained_model/train_config.json \
  --output_dir=/abs/path/outputs/train/act_dinov2_agilex
```

经验建议：
- 如果你训练容易中断，把 `save_freq` 调小（比如 5000），减少损失。

---

## 7. 推理部署策略（在另一台机器推理）

你有两种选择（取决于你把 backbone 权重放在哪里）：

### 7.1 方案 A：checkpoint 自包含（推荐第一版）

如果 `ACTDinov2Policy` 把 `Dinov2Model` 作为 `nn.Module` 的一部分（即权重在 state dict 里），那么：
- 推理机器只需要 checkpoint 目录（`.../checkpoints/<step>/pretrained_model`）
- 不需要额外的 DINOv2 权重目录

优点：部署最简单；不容易因路径问题失败。  
缺点：checkpoint 更大（但 DINOv2-base 也就几十/一百多 MB 级别，可接受）。

### 7.2 方案 B：外置 backbone（类似 SmolVLA）

如果你刻意不把 backbone 权重存进 checkpoint，而是推理时再从 `dinov2_model_name_or_path` 加载，那么：
- 推理机器必须也有同样的 DINOv2 权重目录
- 需要在推理脚本里允许覆盖 `--policy.dinov2_model_name_or_path=/new/path`

优点：checkpoint 小；可共享 backbone。  
缺点：部署更容易踩坑（路径不一致、文件缺失、版本不一致）。

建议第一版做 A，等跑通与稳定后再考虑 B。

---

## 8. 测试与验收（保证“不会训练几天才发现错”）

建议给 `act_dinov2` 设立一组最小验收标准：

### 8.1 单元测试（建议补齐）

新增测试文件（实现时）：
- `tests/policies/test_act_dinov2.py`

至少覆盖：
- 能 `make_policy_config("act_dinov2")`
- `get_policy_class("act_dinov2")` 可用
- policy 在 CPU 上对假数据 forward/backward 不报错
- 输出 `action` 的 shape == 数据集 action dim

### 8.2 冒烟训练（必须）

在你开始长训前，固定跑：
- CPU smoke：1 step
- GPU smoke：10 step（确认显存与速度）

并检查：
- `outputs/train/.../checkpoints/last/pretrained_model/config.json` 里：
  - `type == "act_dinov2"`
  - `dinov2_*` 参数保存正确
- `training_state/training_step.json` step 正确增长

---

## 9. 常见坑总结（提前避雷）

1) **图像尺寸没对齐 patch size**
- 现象：启动即报错 / reshape 失败 / token 数异常 / 训练巨慢或 OOM
- 解决：默认 `dinov2_image_size=224` 且强校验

2) **没有安装 transformers**
- 现象：import error
- 解决：确保 env 安装 `lerobot[transformers-dep]` 或直接 `pip install transformers`

3) **离线环境权重缺失**
- 现象：from_pretrained 下载失败
- 解决：提前 `hf download` 到本地，并在配置中使用绝对路径 + `local_files_only=true`

4) **多相机 token 太多**
- 现象：显存爆、速度慢
- 解决：减小 `dinov2_image_size` 或使用 `cls/mean_pool` 输出模式（作为加速选项）

5) **backbone 冻结策略不当**
- 现象：loss 不下降或过拟合严重
- 解决：先冻结 backbone + 训练 head；再小 LR 解冻末几层

---

## 10. 推荐的落地里程碑（按阶段交付，避免一次性大改）

**Milestone 0：能 import / 注册**
- `lerobot-info` 或最小脚本能创建 `act_dinov2` config/policy

**Milestone 1：CPU 冒烟 1 step**
- `lerobot-train --steps=1` 训练完成并保存 checkpoint

**Milestone 2：GPU 冒烟 10~100 step**
- 检查显存、速度、loss 下降趋势

**Milestone 3：短训（比如 5k~10k step）**
- 验证 checkpoint 可用于推理（至少离线 replay 或简单评估脚本）

**Milestone 4：正式长训**
- 期间保证 `save_freq` 足够密（防中断损失）

---

## 11. 你如果要我帮你继续推进

你确认以下几点后，我可以再给你一份更贴近你 AgileX 数据集与现有脚本风格的“实现清单/参数建议”：

1) 你想用哪一个 DINOv2 变体（`vits14` / `vitb14` / `vitl14`）？
2) 你训练/推理是否都要求完全离线（权重都在本地目录）？
3) 你希望推理端只依赖 checkpoint（自包含），还是也带一个 dinov2 权重目录（外置）？
4) 你的相机数量是否固定 3 个？是否需要支持更多/更少？


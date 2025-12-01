# Repository Guidelines

## 项目结构与模块组织
- 核心代码位于 `src/lerobot/`，按策略、数据集、机器人、脚本分层；新增模块请放在相近目录以便发现。
- 测试与源码对应，位于 `tests/`，通用夹具在 `tests/fixtures/`；覆盖率涉及的领域写在对应子目录。
- 快速示例在 `examples/`；文档与媒体在 `docs/` 与 `media/`；Docker 支撑在 `docker/`；性能基准在 `benchmarks/`。
- 依赖与工具配置集中在 `pyproject.toml`；端到端短程流程在 `Makefile` 的 `test-*` 目标中。

## 环境与依赖
- Python 3.10+。推荐在仓库根目录运行 `uv sync --extra dev --extra test` 或 `poetry sync --extras "dev test"`，保持与 `pyproject.toml` 一致。
- 简化方案：激活虚拟环境后 `pip install -e .[dev,test]`；依赖更新后重新同步。

## 构建、测试与开发命令
- 快速检查：`python -m ruff check src tests`，随后 `python -m ruff format --check src tests`。
- 运行测试：`pytest tests/`；迭代时可定位到子目录或文件，如 `pytest tests/policies/test_available.py`。
- 策略端到端冒烟：`make test-end-to-end DEVICE=cpu`（会下载模型/数据，覆盖 ACT、diffusion、TDMPC、SmolVLA 短流程）。
- 数据集可视化示例：`python -m lerobot.scripts.visualize_dataset --help`；示例脚本参考 `examples/`。

## 代码风格与命名
- Ruff 负责 lint/格式；目标行长 110（即便忽略 E501 也保持可读），使用双引号与空格缩进。
- 命名：模块/包 `snake_case`，类 `PascalCase`，函数与变量 `snake_case`，常量 `UPPER_CASE`。
- 非直观逻辑补充 Google 风格 docstring，公共接口尽量加类型标注。

## 测试要求
- 新增或修改逻辑应配套 `pytest` 覆盖，放在对应领域的 `tests/` 子目录，文件/用例命名 `test_*.py`、`test_*`。
- 优先编写确定性、快速的单测；长耗时或硬件依赖用标记或开关隔离。
- 调整训练/评估流程时执行相关 `make test-*-ete-*` 目标，若跳过请在 PR 说明理由。

## 提交与 PR 规范
- 沿用历史前缀：`fix(scope): msg`、`feat`、`chore` 等，可附 issue 编号（如 `(#123)`）。
- PR 描述应包含问题背景、解决方案与验证方式（命令+结果）；关联 issue，界面或日志变更请附截图/日志；破坏性改动需醒目提示。
- 保持 diff 聚焦；行为或 CLI 变更时同步更新文档/示例。

## 安全与配置
- 禁止提交凭据或数据集产物；使用环境变量（如 HF token），遵守 `.gitignore`。
- GPU 或机器人专用配置建议置于 `tests/configs/` 等配置目录，并在 PR 记录默认值和假设。

## 架构与配置提示
- 训练与评估 CLI 入口在 `lerobot.scripts.train` 与 `lerobot.scripts.eval`；调整超参时优先用配置或命令行参数而非硬编码。
- 配置样例通常在 `tests/configs/` 或示例 CLI 参数中；新增选项后更新 `--help` 描述并添加回归测试。
- 数据或模型下载默认通过 Hugging Face Hub；离线场景可添加 `--local-files-only`，并在 PR 中说明使用方式。

## 常见工作流
- 调试单个策略：在对应策略目录下加入最小脚本或 notebook，并在 `tests/policies/` 添加覆盖。
- 调试数据集：使用 `python -m lerobot.scripts.visualize_dataset --repo-id ... --episode-index ...` 先确认内容再写训练脚本。
- 发布前检查：运行 lint + 关键测试 + `git diff --stat`，确认只包含预期文件并记录验证命令。

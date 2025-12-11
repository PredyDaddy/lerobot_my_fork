# Repository Guidelines

## 项目结构与模块组织
- `src/lerobot/`：核心 Python 包，包含配置、数据集、环境、策略以及通过 `lerobot-*` 暴露的脚本。
- `tests/`：pytest 测试，端到端运行会将检查点写入 `tests/outputs/`。
- `examples/`：可直接运行的训练/评估示例；新增示例时保持相同结构。
- `docs/` 与 `media/`：文档与资源文件，避免提交大体积二进制。
- `benchmarks/` 与 `docker/`：性能基准和容器镜像（`Dockerfile.user` / `Dockerfile.internal`）。
- 根目录配置：`pyproject.toml`（ruff/mypy/bandit）、`.pre-commit-config.yaml`，以及不同平台的 `requirements-*.txt`。

## 构建、测试与开发命令
- 创建开发环境：`pip install -e ".[dev,test]"`（或最小化安装 `pip install -e .`）。
- 一次性安装钩子：`pre-commit install`；全量检查用 `pre-commit run --all-files`。
- 格式化与静态检查：先 `ruff format .` 再 `ruff --fix .`。
- 单元/集成测试：`pytest tests`（可用 `-k` 精确筛选）。
- 全量流水线：`make test-end-to-end DEVICE=cpu`（或 `gpu`）覆盖 ACT/Diffusion/TDMPC/SmolVLA，产物写入 `tests/outputs/`。

## 代码风格与命名约定
- Python 3.10+；4 空格缩进、110 字符行宽、双引号（见 `tool.ruff.format`）。
- 保持导入顺序（ruff isort），优先添加类型标注；mypy 按模块逐步收紧，避免新增 `ignore_errors`。
- 命名：snake_case 函数/变量，PascalCase 类，UPPER_CASE 常量；配置键与 `lerobot/configs` 保持一致。
- 优先使用日志替代临时 print；避免提交实验性笔记本或大文件。

## 测试指南
- 测试紧邻代码路径（如 `tests/policies/test_<name>.py` 对应 `src/lerobot/policies/`）。
- 倾向使用轻量合成数据；大体积资产放 `tests/artifacts/`，耗时案例用标记区分。
- 需要覆盖率时运行：`pytest --cov=lerobot --cov-report=term`。
- 影响训练脚本的改动合并前请跑通 `make test-end-to-end`。

## 提交与 PR 指南
- 采用 Conventional Commit：`feat(scope): summary` 或 `fix(scope): summary`；合并时附上 PR 编号（如 `(#1234)`）。
- PR 聚焦单一主题；提供简要变更清单、关联 issue、复现步骤，用户可见改动附截图/日志。
- 提交前运行 `pre-commit` 与相关 `pytest`/`make` 目标；修改公共 API 或 CLI 时同步更新文档/示例。
- 在描述中标注破坏性变更或新依赖，及时请求评审。

## 安全与配置提示
- 不要提交令牌或机器人凭据；使用环境变量（如 `HUGGINGFACE_HUB_TOKEN`）或被 `.gitignore` 排除的本地配置。
- 使用 `gitleaks`（随 pre-commit 提供）扫描密钥；硬件驱动差异时可借助 Dockerfile 获得可重复环境。

跑测试使用(lerobot_v4)这个conda环境

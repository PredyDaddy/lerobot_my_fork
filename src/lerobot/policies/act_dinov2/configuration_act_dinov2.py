#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
from typing import Any

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("act_dinov2")
@dataclass
class ACTDinov2Config(ACTConfig):
    """Configuration class for ACT with a DINOv2 visual backbone."""

    dinov2_model_name_or_path: str | None = None
    dinov2_local_files_only: bool = True
    dinov2_revision: str | None = None
    dinov2_image_size: int = 224
    dinov2_output_mode: str = "grid"
    dinov2_use_last_n_layers: int = 1
    freeze_backbone: bool = True
    optimizer_lr_backbone: float = 1e-6
    dinov2_config_json: dict[str, Any] | None = field(default=None)

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.dinov2_config_json is None and self.dinov2_model_name_or_path is None:
            raise ValueError(
                "You must provide `dinov2_model_name_or_path` when `dinov2_config_json` is not set."
            )

        patch_size = 14
        if self.dinov2_config_json and "patch_size" in self.dinov2_config_json:
            patch_size = int(self.dinov2_config_json["patch_size"])
        if self.dinov2_image_size % patch_size != 0:
            raise ValueError(
                "`dinov2_image_size` must be divisible by the DINOv2 patch size. "
                f"Got dinov2_image_size={self.dinov2_image_size} and patch_size={patch_size}."
            )

        if self.dinov2_output_mode != "grid":
            raise NotImplementedError("Only `grid` output mode is supported for DINOv2 features.")
        if self.dinov2_use_last_n_layers < 1:
            raise ValueError("`dinov2_use_last_n_layers` must be >= 1.")

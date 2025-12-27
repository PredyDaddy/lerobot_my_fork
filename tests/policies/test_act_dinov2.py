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
import torch
from transformers import Dinov2Config

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act_dinov2.configuration_act_dinov2 import ACTDinov2Config
from lerobot.policies.act_dinov2.modeling_act_dinov2 import ACTDinov2Policy
from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.utils.constants import ACTION
from tests.utils import require_cpu, require_package


@require_package("transformers")
def test_act_dinov2_factory_registration():
    cfg = make_policy_config(
        "act_dinov2",
        dinov2_config_json=Dinov2Config(
            num_hidden_layers=1,
            hidden_size=192,
            num_attention_heads=3,
            mlp_ratio=2,
            out_features=["stage1"],
            out_indices=[1],
        ).to_dict(),
        push_to_hub=False,
    )
    assert isinstance(cfg, ACTDinov2Config)
    assert get_policy_class("act_dinov2").name == "act_dinov2"


@require_package("transformers")
@require_cpu
def test_act_dinov2_forward_backward():
    dinov2_config = Dinov2Config(
        num_hidden_layers=2,
        hidden_size=192,
        num_attention_heads=3,
        mlp_ratio=2,
        out_features=["stage2"],
        out_indices=[2],
    ).to_dict()

    config = ACTDinov2Config(
        chunk_size=2,
        n_action_steps=1,
        dim_model=64,
        n_heads=4,
        n_encoder_layers=1,
        n_decoder_layers=1,
        dim_feedforward=256,
        latent_dim=8,
        use_vae=False,
        dinov2_config_json=dinov2_config,
        dinov2_model_name_or_path="local_dinov2",
        push_to_hub=False,
        device="cpu",
    )
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(4,)),
        "observation.images.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
    }
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))}

    policy = ACTDinov2Policy(config)
    policy.train()

    batch = {
        "observation.state": torch.randn(1, 4, dtype=torch.float32),
        "observation.images.cam": torch.randn(1, 3, 64, 64, dtype=torch.float32),
        ACTION: torch.randn(1, config.chunk_size, 2, dtype=torch.float32),
        "action_is_pad": torch.zeros(1, config.chunk_size, dtype=torch.bool),
    }

    loss, loss_dict = policy(batch)
    assert torch.isfinite(loss)
    assert "l1_loss" in loss_dict
    loss.backward()

    actions = policy.predict_action_chunk(batch)
    assert actions.shape == (1, config.chunk_size, 2)

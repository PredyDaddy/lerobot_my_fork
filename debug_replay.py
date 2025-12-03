#!/usr/bin/env python3
"""Debug script to check replay data flow for AgileX robot."""

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.agilex.config_agilex import JOINT_NAMES
from lerobot.utils.constants import ACTION

# Load dataset
print("Loading dataset...")
ds = LeRobotDataset('your_username/agilex_dataset', root='data', episodes=[0])

# Check action feature names
print("\n=== Action Feature Names ===")
action_names = ds.features[ACTION]["names"]
for i, name in enumerate(action_names):
    print(f"  [{i}] {name}")

# Check which names we expect
print("\n=== Expected Names (AgileX) ===")
expected = []
for side in ["left", "right"]:
    for jn in JOINT_NAMES:
        expected.append(f"{side}_{jn}.pos")
for i, name in enumerate(expected):
    print(f"  [{i}] {name}")

# Check for mismatches
print("\n=== Mismatch Check ===")
missing_in_dataset = set(expected) - set(action_names)
extra_in_dataset = set(action_names) - set(expected)

if missing_in_dataset:
    print(f"Missing in dataset: {missing_in_dataset}")
if extra_in_dataset:
    print(f"Extra in dataset: {extra_in_dataset}")
if not missing_in_dataset and not extra_in_dataset:
    print("All names match!")

# Get first frame action data
print("\n=== First Frame Action Data ===")
episode_frames = ds.hf_dataset.filter(lambda x: x["episode_index"] == 0)
actions = episode_frames.select_columns(ACTION)
first_action = actions[0][ACTION]

print(f"Action array length: {len(first_action)}")
print(f"Action names count: {len(action_names)}")

print("\nLeft arm values:")
for i, name in enumerate(action_names):
    if name.startswith("left_"):
        print(f"  {name}: {first_action[i]:.4f}")

print("\nRight arm values:")
for i, name in enumerate(action_names):
    if name.startswith("right_"):
        print(f"  {name}: {first_action[i]:.4f}")

# Simulate what send_action would receive
print("\n=== Simulating send_action input ===")
action_dict = {}
for i, name in enumerate(action_names):
    action_dict[name] = first_action[i]

import numpy as np
left_target = np.array(
    [action_dict[f"left_{jn}.pos"] for jn in JOINT_NAMES], dtype=np.float32
)
right_target = np.array(
    [action_dict[f"right_{jn}.pos"] for jn in JOINT_NAMES], dtype=np.float32
)

print(f"Left target array: {left_target}")
print(f"Right target array: {right_target}")


"""Training-data numerical readout audit, driven by future true velocities.

This is explicitly not autonomous prediction or a learned-model evaluation.
It isolates the fixed endpoint quadrature used in all candidate pose readouts.
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path

from rsl_rl.modules.kinematic_dynamics import integrate_orientation, quat_inverse, quat_multiply, quat_rotate

from world_modeling.core import file_digest

torch.set_num_threads(4)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--dataset-dir",
    type=Path,
    required=True,
    help="Local download of the immutable dataset v0 artifact; reads TRAIN only.",
)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root = args.dataset_dir
if file_digest(root / "metadata.json") != "4272832c98da50d7e78c4dfc3fee9bbc4ca477c2e1bf6326194dbf8ca0d0e494":
    raise ValueError("Dataset manifest differs from the reviewed diagnostic population")
meta = json.loads((root / "metadata.json").read_text())
if (
    meta["partitions"]["train"]["file"] != "train.npz"
    or file_digest(root / "train.npz") != meta["partitions"]["train"]["sha256"]
):
    raise ValueError("Training array hash differs from the immutable manifest")
with np.load(root / "train.npz") as z:
    state = torch.from_numpy(z["state"])
    rotation = torch.from_numpy(z["quaternion"])
    position = torch.from_numpy(z["com_position"])
    length = torch.from_numpy(z["length"])
start, steps = 281, 1500
# Remove each environment's large world origin once, just as the local readout
# does; otherwise repeated float32 additions introduce a coordinate-size artifact.
position = position - position[:, start : start + 1]
q, p = rotation[:, start].clone(), torch.zeros_like(position[:, start])
joint = state[:, start, 9:21].clone()
translation_sq, angle_sq, joint_sq = [], [], []
with torch.inference_mode():
    for offset in range(1, steps + 1):
        index = start + offset
        q = integrate_orientation(q, state[:, index, 3:6], meta["dt"])
        p = p + meta["dt"] * quat_rotate(q, state[:, index, :3])
        joint = joint + meta["dt"] * state[:, index, 21:33]
        relative = quat_multiply(quat_inverse(q), rotation[:, index])
        angle = 2 * torch.atan2(torch.linalg.vector_norm(relative[:, 1:], dim=-1), relative[:, 0].abs())
        translation_sq.append((p - position[:, index]).square().sum(-1))
        angle_sq.append(angle.square())
        joint_sq.append((joint - state[:, index, 9:21]).square().mean(-1))
arrays = {
    "COM_position_RMSE_m": torch.stack(translation_sq, 1),
    "orientation_RMSE_rad": torch.stack(angle_sq, 1),
    "joint_position_RMSE_rad": torch.stack(joint_sq, 1),
}
result = {
    "classification": (
        "training-only endpoint quadrature audit using future true velocities; NOT autonomous prediction, model"
        " performance, or an unconditional error lower bound"
    ),
    "manifest_sha256": file_digest(root / "metadata.json"),
    "partition": "train",
    "dt_s": meta["dt"],
    "fork_step": start,
    "initialization": "one initial true pose/joint state at fork 281; no later pose/position correction",
    "driver": "future true body COM linear velocity, angular velocity and joint velocity",
    "population": "length >= fork_step + horizon_steps; true terminal endpoint permitted, no post-reset state",
    "aggregation": (
        "mean and p95 across eligible episodes of each episode prefix RMSE, not terminal-time error or pooled RMS"
    ),
    "metric_definitions": {
        "COM_position_RMSE_m": "sqrt(mean over prefix steps of squared Euclidean COM position error)",
        "orientation_RMSE_rad": "sqrt(mean over prefix steps of squared SO(3) geodesic angle)",
        "joint_position_RMSE_rad": "sqrt(mean over prefix steps and 12 joints of squared joint-position error)",
    },
    "horizons": {},
}
for horizon in (10, 50, 100, 250, 500, 1500):
    eligible = length >= start + horizon
    row = {
        "horizon_steps": horizon,
        "horizon_s": horizon * meta["dt"],
        "eligible": int(eligible.sum()),
        "planned": len(length),
    }
    for name, values in arrays.items():
        per_episode = values[eligible, :horizon].mean(-1).sqrt()
        row[name] = float(per_episode.mean()) if len(per_episode) else None
        row[name + "_p95"] = float(torch.quantile(per_episode, 0.95)) if len(per_episode) else None
    result["horizons"][str(horizon)] = row
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))

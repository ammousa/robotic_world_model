"""Training-label integration diagnostic. No test data or model selection."""

import argparse
import json
import numpy as np
import torch
from pathlib import Path

from rsl_rl.modules.kinematic_dynamics import integrate_orientation, projected_gravity, quat_inverse, quat_multiply

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
    quaternion = torch.from_numpy(z["quaternion"])
    length = torch.from_numpy(z["length"])
mask = torch.arange(state.shape[1] - 1)[None] < length[:, None]
current, nxt = state[:, :-1][mask], state[:, 1:][mask]
q0, q1 = quaternion[:, :-1][mask], quaternion[:, 1:][mask]
predq = integrate_orientation(q0, nxt[:, 3:6], meta["dt"])
dq = quat_multiply(quat_inverse(predq), q1)
angle = 2 * torch.atan2(torch.linalg.vector_norm(dq[:, 1:], dim=-1), dq[:, 0].abs())
metrics = {
    "joint_endpoint_defect_rad": (
        (nxt[:, 9:21] - current[:, 9:21] - meta["dt"] * nxt[:, 21:33]).square().mean(-1).sqrt()
    ),
    "orientation_endpoint_defect_rad": angle,
    "gravity_pose_defect": (nxt[:, 6:9] - projected_gravity(q1)).square().mean(-1).sqrt(),
    "gravity_endpoint_defect": (nxt[:, 6:9] - projected_gravity(predq)).square().mean(-1).sqrt(),
}
result = {
    "classification": "training-data discretization diagnostic, not a trained-model experiment or test selection",
    "partition": "train",
    "manifest_sha256": file_digest(root / "metadata.json"),
    "valid_transitions": int(mask.sum()),
    "dt": meta["dt"],
    "metrics": {},
}
for name, value in metrics.items():
    assert torch.isfinite(value).all()
    result["metrics"][name] = {
        "mean": float(value.mean()),
        "rms": float(value.square().mean().sqrt()),
        "p90": float(torch.quantile(value, 0.90)),
        "p95": float(torch.quantile(value, 0.95)),
    }
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))

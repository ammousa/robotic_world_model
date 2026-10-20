"""Kinematic playback of scored arrays; never advance simulator dynamics."""

import json
import numpy as np
import torch
from pathlib import Path

from rsl_rl.modules.kinematic_dynamics import quat_rotate

from .core import file_digest


@torch.inference_mode()
def replay(env, prediction_path, metadata_path, episode, output):
    import imageio.v2 as imageio

    saved = dict(np.load(prediction_path))
    metadata = json.loads(Path(metadata_path).read_text())
    base = env.unwrapped
    robot = base.scene["robot"]
    if base.num_envs != 2:
        raise ValueError("kinematic playback uses two actors: reference, imagination")
    if list(robot.joint_names) != metadata["joint_names"]:
        raise ValueError("playback joint order differs from the scored robot")
    default = torch.tensor(metadata["default_joint_position"], device=base.device)
    offset = torch.tensor(metadata["com_offset"], device=base.device).expand(2, -1)
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    def forbidden_step(*_args, **_kwargs):
        raise RuntimeError("physics stepping is forbidden during kinematic prediction playback")

    # Fail loudly if a future rendering change accidentally invokes the old path.
    base.sim.step = forbidden_step
    base.step = forbidden_step
    rendered, reason = 0, "horizon_complete"
    with imageio.get_writer(output, fps=50, codec="libx264", quality=8) as writer:
        for step in range(saved["state"].shape[1]):
            if not saved["initialization_valid"][episode] or not saved["reference_valid"][episode, step]:
                reason = "invalid_initialization" if step == 0 else "reference_terminated"
                break
            state = torch.tensor(
                np.stack((saved["true_state"][episode, step], saved["state"][episode, step])), device=base.device
            )
            q = torch.tensor(
                np.stack((saved["true_orientation"][episode, step], saved["orientation"][episode, step])),
                device=base.device,
            )
            com = torch.tensor(
                np.stack((saved["true_position"][episode, step], saved["position"][episode, step])), device=base.device
            )
            # A fixed display-only height gauge, identical for both trajectories.
            com[:, 2] += float(saved["initial_com_height"][episode])
            position = com - quat_rotate(q, offset) + base.scene.env_origins
            pose = torch.cat((position, q), -1)
            if not torch.isfinite(pose).all() or not torch.isfinite(state).all():
                reason = "prediction_nonfinite"
                break  # Preserve the scored failure; do not fabricate repaired states.
            robot.write_root_pose_to_sim(pose)
            robot.write_joint_state_to_sim(state[:, 9:21] + default, state[:, 21:33])
            # Isaac Lab documents forward/render as articulation kinematics and
            # rendering only; render disables playSimulations during app.update.
            base.sim.forward()
            center = position.mean(0).cpu().numpy()
            base.sim.set_camera_view(eye=center + np.array([4.0, 4.0, 3.0]), target=center)
            frame = base.render()
            writer.append_data(frame)
            rendered += 1
    Path(output).with_suffix(".json").write_text(
        json.dumps(
            {
                "kind": "illustrative_kinematic_replay_of_scored_arrays",
                "episode_id": int(saved["episode_id"][episode]),
                "frames": rendered,
                "end_reason": reason,
                "reference_censored": reason == "reference_terminated",
                "physics_steps": 0,
                "prediction_sha256": file_digest(prediction_path),
                "dataset_sha256": file_digest(metadata_path),
                "video_sha256": file_digest(output) if Path(output).is_file() else None,
            },
            indent=2,
        )
    )
    print(f"KINEMATIC_VIDEO_COMPLETE {output}", flush=True)

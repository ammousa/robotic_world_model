"""Collection and action replay through stock Play, using its frozen policy loader."""

import hashlib
import json
import numpy as np
import os
import torch
import yaml
from pathlib import Path

from .core import file_digest
from .kinematics import extract_geometry, foot_positions


def configure(env_cfg):
    from isaaclab.managers.recorder_manager import DatasetExportMode, RecorderManagerBaseCfg, RecorderTermCfg

    from mbrl.mbrl.envs.world_model_recorder import WorldModelSnapshotRecorder

    env_cfg.episode_length_s = 40.0
    env_cfg.observations.policy.enable_corruption = False
    for name in ("push_robot", "base_external_force_torque", "add_base_mass", "base_com"):
        setattr(env_cfg.events, name, None)
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    # Commands are externally scheduled, so avoid command-manager resampling.
    env_cfg.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)
    for key in ("static_friction_range", "dynamic_friction_range", "restitution_range"):
        bounds = env_cfg.events.physics_material.params[key]
        if bounds[0] != bounds[1]:
            raise ValueError("fixed nominal mechanics require deterministic material coefficients")
    recorder = RecorderManagerBaseCfg(
        dataset_export_mode=DatasetExportMode.EXPORT_NONE, export_in_record_pre_reset=False
    )
    recorder.world_model_snapshot = RecorderTermCfg(class_type=WorldModelSnapshotRecorder)
    env_cfg.recorders = recorder


def command_schedule(spec, count, steps=2000):
    rng = np.random.default_rng(spec["command_seed"])
    if spec["partition"] != "test_command_shift":
        schedule = rng.uniform(-1, 1, (count, (steps + 250) // 250, 3)).astype(np.float32)
        schedule[rng.random(count) < 0.02] = 0
        return np.repeat(schedule, 250, axis=1)[:, : steps + 1]
    result = np.zeros((count, steps + 1, 3), dtype=np.float32)
    for index in range(count):
        stratum = (spec.get("episode_offset", 0) + index) // 64
        if stratum == 0:
            result[index, :, 0] = 1.2
        elif stratum == 1:
            result[index, :, 1] = 1.2
        elif stratum == 2:
            result[index, :, 0], result[index, :, 2] = 0.5, 1.2
        else:
            result[index, :, 0] = np.where(np.arange(steps + 1) // 100 % 2 == 0, 0.8, -0.8)
    return result


def set_commands(base, command):
    term = base.command_manager.get_term("base_velocity")
    term.vel_command_b[:] = command
    # Recompute policy observations to ensure the externally scheduled command is visible.
    return base.observation_manager.compute()


@torch.inference_mode()
def collect(env, policy, checkpoint, checkpoint_sha256, config_path, output, shadow=None):
    from rsl_rl.modules.kinematic_dynamics import quat_inverse, quat_rotate

    from mbrl.mbrl.envs.world_model_recorder import system_snapshot
    from research_workflow import robot_asset_paths, runtime_fingerprint
    from research_workflow.verification import canonical_config

    base = env.unwrapped
    spec = json.loads(Path(config_path).read_text())
    if spec["seed"] != base.cfg.seed or spec["num_envs"] != base.num_envs:
        raise ValueError("collection config seed/population does not match resolved Play")
    if abs(base.step_dt - 0.02) > 1.0e-9:
        raise ValueError("expected 50 Hz transitions")
    if torch.any(base.scene["robot"].data.default_joint_vel != 0):
        raise ValueError("dataset adapter requires the declared zero default joint velocity")
    schedule = torch.as_tensor(command_schedule(spec, base.num_envs), device=base.device)
    recorder = base.recorder_manager._terms["world_model_snapshot"]
    initial = system_snapshot(base)
    geometry = extract_geometry(base.scene["robot"])
    local_feet = foot_positions(base.scene["robot"].data.joint_pos, geometry)
    actual_feet = quat_rotate(
        quat_inverse(initial["quaternion"])[:, None].expand(-1, 4, -1),
        initial["foot_position"] - initial["link_position"][:, None],
    )
    fk_error = float((local_feet - actual_feet).abs().max())
    if fk_error > 0.002:
        raise ValueError(f"USD forward-kinematic geometry verification failed: {fk_error} m")
    samples = {name: [value.cpu().numpy()] for name, value in initial.items()}
    action_rows, targets, terminated, truncated = [], [], [], []
    shadow_data = dict(np.load(shadow)) if shadow is not None else None
    if shadow is not None:
        source = json.loads(Path(shadow).with_name("provenance.json").read_text())
        if (
            source["mode"] != "B"
            or source["partition"] != spec["partition"]
            or source["policy_sha256"] != checkpoint_sha256
        ):
            raise ValueError("shadow input is not the corresponding autonomous prediction")
        if file_digest(shadow) != source["prediction_sha256"]:
            raise ValueError("shadow prediction content changed")
        expected_ids = np.arange(base.num_envs) + spec.get("episode_offset", 0)
        if not np.array_equal(shadow_data["episode_id"], expected_ids):
            raise ValueError("shadow population differs from the declared prefix")
    for step in range(2000):
        observation = set_commands(base, schedule[:, step])
        actions = policy(observation)
        if shadow_data is not None and 281 <= step < 1781:
            actions = torch.as_tensor(shadow_data["action"][:, step - 281], device=base.device)
            # Undefined imagined actions are already prediction failures, not
            # executable simulator inputs. Retain the original NaNs in predictions.
            actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
        action_rows.append(actions.cpu().numpy())
        env.step(actions)
        targets.append(recorder.applied_targets.cpu().numpy())
        for name, value in recorder.last.items():
            samples[name].append(value.cpu().numpy())
        terminated.append(recorder.terminated.cpu().numpy())
        truncated.append(recorder.truncated.cpu().numpy())
        if step % 250 == 0:
            print(f"collection {spec['partition']}: {step}/2000", flush=True)
    arrays = {name: np.stack(value, 1) for name, value in samples.items()}
    arrays.update(
        action=np.stack(action_rows, 1),
        processed_target=np.stack(targets, 1),
        terminated=np.stack(terminated, 1),
        truncated=np.stack(truncated, 1),
        command=schedule.cpu().numpy(),
    )
    done = arrays["terminated"] | arrays["truncated"]
    arrays["length"] = np.array([np.flatnonzero(row)[0] + 1 if row.any() else 2000 for row in done])
    arrays["episode_id"] = np.arange(base.num_envs) + spec.get("episode_offset", 0)
    arrays["time_s"] = np.arange(2001, dtype=np.float64)[None].repeat(base.num_envs, axis=0) * base.step_dt
    arrays["reset"] = done
    if shadow_data is not None:
        arrays["shadow_action_valid"] = np.isfinite(shadow_data["action"]).all(-1)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("immutable collection output already exists")
    np.savez_compressed(path, **arrays)
    data = base.scene["robot"].data
    robot = base.scene["robot"]
    actuator_hashes = {}
    for name, actuator in robot.actuators.items():
        digest = hashlib.sha256()
        for key, value in sorted(actuator.network.state_dict().items()):
            digest.update(key.encode())
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        actuator_hashes[name] = digest.hexdigest()
    actor_config = yaml.safe_load((Path(checkpoint).parent / "params" / "agent.yaml").read_text())
    normalizer = actor_config["imagination"]["state_normalizer"]
    metadata = {
        "schema_version": 1,
        "spec": spec,
        "file": path.name,
        "sha256": file_digest(path),
        "policy_sha256": checkpoint_sha256,
        "dt": base.step_dt,
        "state_mean": normalizer["mean"],
        "state_std": normalizer["std"],
        "joint_names": base.scene["robot"].joint_names,
        "default_joint_position": data.default_joint_pos[0].cpu().tolist(),
        "joint_limits": data.joint_pos_limits[0].cpu().tolist(),
        "com_offset": data.body_com_pos_b[0, 0].cpu().tolist(),
        "geometry": geometry,
        "foot_names": geometry["feet"],
        "resolved_config": canonical_config(base.cfg.to_dict()),
        "runtime_fingerprint": runtime_fingerprint(
            resolved_config=base.cfg.to_dict(), asset_paths=robot_asset_paths(base.cfg)
        ),
        "geometry_verification_max_m": fk_error,
        "shadow_prediction_sha256": file_digest(shadow) if shadow else None,
        "shadow_prediction_provenance": (
            json.loads(Path(shadow).with_name("provenance.json").read_text()) if shadow else None
        ),
        "actuator_weight_sha256": actuator_hashes,
        "body_mass_kg": robot.root_physx_view.get_masses()[0].cpu().tolist(),
        "body_inertia": robot.root_physx_view.get_inertias()[0].cpu().tolist(),
        "shape_material_properties": robot.root_physx_view.get_material_properties()[0].cpu().tolist(),
        "body_com_positions": data.body_com_pos_b[0].cpu().tolist(),
        "execution_placement": {"sim_device": str(base.cfg.sim.device), "queue_GPU": os.environ.get("RESEARCH_GPU")},
    }
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(f"saved immutable transitions: {path}", flush=True)

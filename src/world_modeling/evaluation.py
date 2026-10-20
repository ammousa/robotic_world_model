"""Scored autonomous dynamics trajectories, independent of simulator rendering."""

import argparse
import json
import numpy as np
import time
import torch
from pathlib import Path

from rsl_rl.modules.kinematic_dynamics import projected_gravity

from .core import FrozenActor, file_digest, load_model, physical_errors, predict, read_partition, score_at_horizon
from .diagnostics import gait_diagnostics, trajectory_diagnostics

HORIZONS = [10, 50, 100, 250, 500, 1500]


def scored_rows(prediction, reference, metadata, partition, mode, variant, seed):
    states = prediction["state"]
    device = states.device
    errors = physical_errors(states, reference, metadata)
    previous = torch.cat((torch.as_tensor(reference["state"][:, 281:282], device=device), states[:, :-1]), 1)
    joint_defect = states[..., 9:21] - previous[..., 9:21] - metadata["dt"] * states[..., 21:33]
    gravity_defect = states[..., 6:9] - projected_gravity(errors["orientation"])
    defaults = torch.tensor(metadata["default_joint_position"], device=device)
    limits = torch.tensor(metadata["joint_limits"], device=device)
    angles = states[..., 9:21] + defaults
    violation = torch.relu(limits[:, 0] - angles) + torch.relu(angles - limits[:, 1])
    gravity = projected_gravity(errors["orientation"])
    roll = torch.atan2(-gravity[..., 1], -gravity[..., 2]).abs()
    pitch = torch.atan2(gravity[..., 0], torch.linalg.vector_norm(gravity[..., 1:], dim=-1)).abs()
    tipped = (roll > torch.pi / 3) | (pitch > torch.pi / 3) | (violation > 0.25).any(-1)
    persistent = torch.nn.functional.pad(tipped.unfold(1, 5, 1).all(-1), (4, 0))
    unstable = persistent | ~errors["prediction_finite"]
    instability = torch.where(unstable, torch.arange(states.shape[1], device=device)[None] + 1, 1501).amin(-1)
    command = torch.tensor(reference["command"][:, 281:1781], device=device)
    extra = trajectory_diagnostics(prediction, reference, errors, metadata)
    truth = torch.as_tensor(reference["state"][:, 282:1782], device=device)
    per_step_error = errors["blocks"].mean(-1).sqrt()
    crossed = torch.nn.functional.pad((per_step_error > 1).unfold(1, 10, 1).all(-1), (9, 0))
    fidelity = torch.where(crossed, torch.arange(states.shape[1], device=device)[None] + 1, 1501).amin(-1)
    rows = []
    for horizon in HORIZONS:
        scored = score_at_horizon(errors, horizon)
        gait = gait_diagnostics(states[:, :horizon], truth[:, :horizon], metadata["dt"], metadata["joint_names"])
        for index, episode in enumerate(reference["episode_id"]):
            eligible, finite = bool(scored["eligible"][index]), bool(scored["finite"][index])
            initialized = (reference["length"][index] > 281).item()
            physical = errors["difference"][index, :horizon]
            result = {
                "variant": variant,
                "training_seed": seed,
                "evaluation_partition": partition,
                "episode_id": int(episode),
                "fork_step": 281,
                "command_schedule_id": f"{partition}-{episode}",
                "mode": mode,
                "horizon_steps": horizon,
                "horizon_s": horizon * metadata["dt"],
                "reference_eligible": eligible,
                "prediction_finite": finite,
                "initialization_valid": initialized,
                "physical_trajectory_nrmse": float(scored["score"][index]) if eligible and finite else None,
                "failure_aware_score": float(scored["capped_score"][index]) if eligible else None,
                "stable_duration_s": min(float(instability[index]), horizon) * metadata["dt"] if initialized else None,
                "unstable": (instability[index] <= horizon).item() if initialized else None,
                "numerical_divergence": (
                    not bool(errors["prediction_finite"][index, :horizon].all()) if initialized else None
                ),
                "exact_fidelity_duration_s": (
                    min(float(fidelity[index]), horizon) * metadata["dt"] if eligible and finite else None
                ),
                "reference_terminal_step": int(reference["length"][index]),
                "reference_terminated_before_horizon": (
                    bool(reference["terminated"][index, : 281 + horizon].any())
                    if "terminated" in reference
                    else not eligible
                ),
            }
            diagnostics = {
                "translation_rmse_m": errors["blocks"][index, :horizon, 0].mean().sqrt(),
                "orientation_rmse_rad": errors["geodesic"][index, :horizon].square().mean().sqrt(),
                "linear_velocity_rmse_m_s": physical[:, :3].square().mean().sqrt(),
                "angular_velocity_rmse_rad_s": physical[:, 3:6].square().mean().sqrt(),
                "gravity_rmse": physical[:, 6:9].square().mean().sqrt(),
                "joint_position_rmse_rad": physical[:, 9:21].square().mean().sqrt(),
                "joint_velocity_rmse_rad_s": physical[:, 21:33].square().mean().sqrt(),
                "torque_rmse_Nm": physical[:, 33:45].square().mean().sqrt(),
                "joint_integration_defect_rad": joint_defect[index, :horizon].square().mean().sqrt(),
                "gravity_pose_defect": gravity_defect[index, :horizon].square().mean().sqrt(),
                "gravity_norm_error": (states[index, :horizon, 6:9].norm(dim=-1) - 1).abs().mean(),
                "rotation_norm_error": (errors["orientation"][index, :horizon].norm(dim=-1) - 1).abs().mean(),
                "joint_limit_violation_fraction": (violation[index, :horizon] > 0).float().mean(),
                "command_tracking_linear_rmse": (
                    (states[index, :horizon, :2] - command[index, :horizon, :2]).square().mean().sqrt()
                ),
                "command_tracking_yaw_rmse": (
                    (states[index, :horizon, 5] - command[index, :horizon, 2]).square().mean().sqrt()
                ),
            }
            diagnostics.update({key: value[index, :horizon].float().mean() for key, value in extra.items()})
            diagnostics.update({key: value[index] for key, value in gait.items()})
            diagnostics["link_translation_rmse_m"] = (
                extra["link_translation_squared_error_m2"][index, :horizon].mean().sqrt()
            )
            diagnostics["joint_limit_excess_rad"] = violation[index, :horizon].max()
            predicted_contact = prediction["contact"][index, :horizon, 4:8] >= 0.5
            true_contact = torch.as_tensor(reference["contact"][index, 282 : 282 + horizon, 4:8], device=device).bool()
            predicted_event, true_event = (
                predicted_contact[1:] != predicted_contact[:-1],
                true_contact[1:] != true_contact[:-1],
            )
            denominator = predicted_event.sum() + true_event.sum()
            diagnostics["contact_transition_F1"] = (
                (2 * (predicted_event & true_event).sum() / denominator) if denominator else states.new_tensor(1.0)
            )
            diagnostics["foot_contact_duty_factor_error"] = (
                (predicted_contact.float().mean(0) - true_contact.float().mean(0)).abs().mean()
            )
            result.update(
                {
                    key: float(value) if eligible and bool(torch.isfinite(value)) else None
                    for key, value in diagnostics.items()
                }
            )
            for key in ("command_tracking_linear_rmse", "command_tracking_yaw_rmse"):
                value = diagnostics[key]
                result[key] = float(value) if initialized and bool(torch.isfinite(value)) else None
            rows.append(result)
    return rows, errors


@torch.inference_mode()
def timing(model, actor, arrays, metadata):
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    physical = torch.tensor(arrays["state"][:1, 281:282], device=device)
    mean = torch.tensor(metadata["state_mean"], device=device)
    scale = torch.tensor(metadata["state_std"], device=device) + 1.0e-8
    state = (physical - mean) / scale
    command = torch.tensor(arrays["command"][:1, 281], device=device)
    action = torch.tensor(arrays["action"][:1, 280:281], device=device)
    result = {}
    for with_policy in (False, True):
        model.reset()
        samples = []
        for index in range(11000):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            if with_policy:
                applied = actor(physical[:, 0], command, action[:, 0])[:, None]
            else:
                applied = action
            model(state, applied)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if index >= 1000:
                samples.append((time.perf_counter() - start) * 1000)
        prefix = "policy_plus_model" if with_policy else "model"
        result.update(
            {
                f"{prefix}_latency_{name}_ms": float(np.quantile(samples, quantile))
                for name, quantile in (("median", 0.5), ("p95", 0.95), ("p99", 0.99))
            }
        )
    model.reset()
    batched = state.expand(256, -1, -1).contiguous()
    actions = action.expand(256, -1, -1).contiguous()
    for _ in range(1000):
        model(batched, actions)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10000):
        model(batched, actions)
    if device.type == "cuda":
        torch.cuda.synchronize()
    result["batch256_transitions_per_second"] = 2560000 / (time.perf_counter() - start)
    result["peak_allocated_memory_bytes"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    result["device"] = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    result["dtype"] = str(next(model.parameters()).dtype)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--partition", choices=["test_id", "test_command_shift"], required=True)
    parser.add_argument("--mode", choices=["A", "B", "B_shadow"], required=True)
    parser.add_argument("--shadow_reference")
    parser.add_argument("--prediction")
    parser.add_argument("--measure_timing", action="store_true")
    parser.add_argument("--stochastic_streams", type=int, default=0)
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    model, checkpoint = load_model(args.checkpoint, args.device)
    arrays, metadata = read_partition(args.dataset, args.partition)
    if metadata != checkpoint["metadata"] or file_digest(args.policy) != metadata["policy_sha256"]:
        raise ValueError("evaluation dataset/policy differs from the checkpoint")
    actor = FrozenActor(args.policy, args.device)
    if args.mode == "B_shadow":
        if not args.shadow_reference or not args.prediction:
            raise ValueError("B_shadow requires its saved B prediction and action-matched simulator reference")
        prediction = {
            key: torch.tensor(value, device=args.device)
            for key, value in dict(np.load(args.prediction)).items()
            if key in {"state", "action", "contact", "base_contact"}
        }
        reference = dict(np.load(args.shadow_reference))
        b_info = json.loads(Path(args.prediction).with_name("provenance.json").read_text())
        replay_info = json.loads(Path(args.shadow_reference).with_suffix(".json").read_text())
        expected = {
            "mode": "B",
            "partition": args.partition,
            "variant": checkpoint["variant"],
            "training_seed": checkpoint["seed"],
            "checkpoint_sha256": file_digest(args.checkpoint),
            "dataset_sha256": file_digest(Path(args.dataset) / "metadata.json"),
        }
        if any(b_info.get(key) != value for key, value in expected.items()):
            raise ValueError("autonomous prediction provenance differs from this cell")
        if b_info["prediction_sha256"] != file_digest(args.prediction) or replay_info["sha256"] != file_digest(
            args.shadow_reference
        ):
            raise ValueError("shadow input content changed")
        if (
            replay_info["shadow_prediction_sha256"] != b_info["prediction_sha256"]
            or replay_info["shadow_prediction_provenance"] != b_info
        ):
            raise ValueError("simulator shadow was generated from another autonomous prediction")
        b_arrays = dict(np.load(args.prediction))
        if not np.array_equal(reference["episode_id"], arrays["episode_id"]) or not np.array_equal(
            b_arrays["episode_id"], arrays["episode_id"]
        ):
            raise ValueError("shadow episode identities differ")
        finite_action = np.isfinite(b_arrays["action"]).all(-1)
        if not np.array_equal(reference["shadow_action_valid"], finite_action):
            raise ValueError("shadow action validity mask differs")
        if not np.array_equal(reference["action"][:, 281:1781][finite_action], b_arrays["action"][finite_action]):
            raise ValueError("simulator shadow applied different finite actions")
        initialized = arrays["length"] > 281
        if not np.array_equal(initialized, reference["length"] > 281):
            raise ValueError("simulator prefix initialization failures differ")
        for key in ("state", "quaternion", "com_position"):
            if not np.allclose(
                reference[key][initialized, :282], arrays[key][initialized, :282], atol=1.0e-4, rtol=1.0e-4
            ):
                raise ValueError(f"simulator replay prefix is not reproducible: {key}")
        arrays = reference
    else:
        prediction = predict(model, arrays, metadata, actor=actor if args.mode == "B" else None)
    rows, errors = scored_rows(
        prediction, arrays, metadata, args.partition, args.mode, checkpoint["variant"], checkpoint["seed"]
    )
    from research_workflow import paired_unit_id

    for row in rows:
        row["eval_unit_id"] = paired_unit_id(
            row, {}, checkpoint["research"]["protocol_contract"]["evaluation"]["paired_factors"]
        )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output / "predictions.npz",
        **{key: value.cpu().numpy() for key, value in prediction.items()},
        true_state=arrays["state"][:, 282:1782],
        initial_com_height=arrays["com_position"][:, 281, 2],
        episode_id=arrays["episode_id"],
        time_s=np.arange(1, 1501) * metadata["dt"],
        command=arrays["command"][:, 281:1781],
        reference_valid=errors["valid"].cpu().numpy(),
        initialization_valid=arrays["length"] > 281,
        prediction_finite=errors["prediction_finite"].cpu().numpy(),
        true_contact=arrays["contact"][:, 282:1782],
        **{key: errors[key].cpu().numpy() for key in ("position", "orientation", "true_position", "true_orientation")},
    )
    (output / "scored_rows.json").write_text(json.dumps(rows, allow_nan=False))
    provenance = {
        "checkpoint_sha256": file_digest(args.checkpoint),
        "dataset_sha256": file_digest(Path(args.dataset) / "metadata.json"),
        "policy_sha256": file_digest(args.policy),
        "mode": args.mode,
        "partition": args.partition,
        "variant": checkpoint["variant"],
        "training_seed": checkpoint["seed"],
        "population": len(arrays["length"]),
        "horizons": HORIZONS,
        "metrics_unavailable": ["absolute_collision_surface_foot_penetration", "support_force_consistency"],
        "geometry_proxy": (
            "foot-link origins under an evaluation-only initial COM-height gauge; no collision surface or force balance"
            " claim"
        ),
        "prediction_sha256": file_digest(output / "predictions.npz"),
        "episode_ids": arrays["episode_id"].tolist(),
        "training_runtime_fingerprint": checkpoint["runtime_fingerprint"],
        "protocol_id": checkpoint["research"]["protocol_id"],
        "train_run_id": checkpoint["wandb_run_id"],
    }
    from research_workflow import runtime_fingerprint

    provenance["runtime_fingerprint"] = runtime_fingerprint()
    if args.stochastic_streams:
        if args.mode != "A" or args.partition != "test_id" or args.stochastic_streams != 16:
            raise ValueError("only the declared 16-stream test-ID recorded-action diagnostic is supported")
        streams = []
        for stream in range(16):
            sampled = predict(model, arrays, metadata, sample_seed=6100 + stream)
            stream_rows, _ = scored_rows(
                sampled, arrays, metadata, args.partition, "A_stochastic", checkpoint["variant"], checkpoint["seed"]
            )
            streams.extend([{**row, "sample_stream": 6100 + stream} for row in stream_rows])
        (output / "stochastic_rows.json").write_text(json.dumps(streams, allow_nan=False))
    if args.measure_timing:
        provenance["timing"] = timing(model, actor, arrays, metadata)
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(f"SCORED_COMPLETE {output}: {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()

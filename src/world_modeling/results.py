"""Verify complete world-model evidence and publish one evaluation artifact per training run."""

import json
import numpy as np
import torch
from pathlib import Path

import pandas as pd

import wandb
from research_workflow import paired_unit_id
from research_workflow.gpu_budget import used_seconds

from .core import file_digest, pose_from_states, read_partition
from .evaluation import HORIZONS


def verify_cell(path, checkpoint, protocol, partition, mode, expected_ids, *, checkpoint_role):
    if checkpoint_role not in {"selected", "final"}:
        raise ValueError("checkpoint_role must be selected or final")
    info = json.loads((path / "provenance.json").read_text())
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expected = {
        "checkpoint_sha256": file_digest(checkpoint),
        "protocol_id": protocol["protocol_id"],
        "variant": data["variant"],
        "training_seed": data["seed"],
        "partition": partition,
        "mode": mode,
        "population": len(expected_ids),
        "episode_ids": expected_ids,
        "horizons": HORIZONS,
        "policy_sha256": protocol["world_model"]["policy_sha256"],
        "dataset_sha256": file_digest(Path(protocol["world_model"]["dataset"]) / "metadata.json"),
        "train_run_id": data["wandb_run_id"],
    }
    if any(info.get(key) != value for key, value in expected.items()):
        raise ValueError(f"scored identity mismatch: {path}")
    if file_digest(path / "predictions.npz") != info["prediction_sha256"]:
        raise ValueError("scored trajectory content changed")
    training_source = data["runtime_fingerprint"]
    evaluation_source = info["runtime_fingerprint"]
    for field in ("commit", "rsl_rl_commit", "isaaclab_commit"):
        if (
            not training_source.get(field)
            or training_source[field] != evaluation_source.get(field)
            or training_source[field].endswith("-dirty")
        ):
            raise ValueError(f"unverified/mismatched executable source: {field}")
    rows = json.loads((path / "scored_rows.json").read_text())
    keys = [(row["episode_id"], row["horizon_steps"]) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != {
        (episode, horizon) for episode in expected_ids for horizon in HORIZONS
    }:
        raise ValueError("scored population has missing or duplicate episode/horizon units")
    for row in rows:
        if row.get("eval_unit_id") != paired_unit_id(row, {}, protocol["evaluation"]["paired_factors"]):
            raise ValueError("scored paired-unit ID is not reproducible from declared factors")
        if (
            row["variant"] != data["variant"]
            or row["training_seed"] != data["seed"]
            or row["mode"] != mode
            or row["evaluation_partition"] != partition
        ):
            raise ValueError("scored row identity differs from cell")
        if not row["initialization_valid"] and (row["stable_duration_s"] is not None or row["unstable"] is not None):
            raise ValueError("invalid initialization contributes to a behavior denominator")
        if row["reference_eligible"] and row["failure_aware_score"] is None:
            raise ValueError("model failure was dropped from the eligible physical population")
        if row["physical_trajectory_nrmse"] is not None and not (
            row["reference_eligible"] and row["prediction_finite"]
        ):
            raise ValueError("uncensored physical score")
    for spec in protocol["evaluation"].get("artifact_metrics", {}).values():
        selected = {"mode": mode, "partition": partition}
        if any(selected.get(key) != value for key, value in spec["selection"].items() if key in selected):
            continue
        if spec["source"] == "scored_rows.json":
            if any(column not in row for row in rows for column in spec["columns"]):
                raise ValueError("declared diagnostic column is absent")
        elif (
            spec["source"] == "provenance.json"
            and checkpoint_role == "selected"
            and mode == "A"
            and partition == "test_id"
        ):
            if any(column not in info.get("timing", {}) for column in spec["columns"]):
                raise ValueError("declared timing evidence is absent")
    return rows, info


def summarize(rows):
    """Equal episode weights, with explicit denominators and no imputed reference errors."""
    result = []
    for horizon in HORIZONS:
        selected = [row for row in rows if row["horizon_steps"] == horizon]
        entry = {
            "horizon_steps": horizon,
            "planned": len(selected),
            "initialized": sum(row["initialization_valid"] for row in selected),
            "reference_eligible": sum(row["reference_eligible"] for row in selected),
            "finite_reference_pairs": sum(row["reference_eligible"] and row["prediction_finite"] for row in selected),
        }
        for key in selected[0]:
            if key in {"training_seed", "episode_id", "fork_step", "horizon_steps", "reference_terminal_step"}:
                continue
            values = [row[key] for row in selected if isinstance(row[key], (int, float))]
            if values:
                if not np.isfinite(values).all():
                    raise ValueError("nonfinite serialized scored metric")
                entry[key] = float(np.mean(values))
                entry[key + "__n"] = len(values)
        physical = [
            row["physical_trajectory_nrmse"] for row in selected if row["physical_trajectory_nrmse"] is not None
        ]
        entry["physical_nrmse_p90"] = float(np.quantile(physical, 0.90)) if physical else None
        entry["physical_nrmse_p95"] = float(np.quantile(physical, 0.95)) if physical else None
        result.append(entry)
    return result


def equality_controls(directory, variant, metadata):
    evidence = {}
    for partition in ("test_id", "test_command_shift"):
        reference, _ = read_partition(Path.cwd() / directory["dataset"], partition)
        for mode in ("A", "B"):
            path = directory["run"] / "world_model_eval" / f"{partition}-{mode}/predictions.npz"
            saved = dict(np.load(path))
            state = torch.from_numpy(saved["state"])
            before = saved["state"].copy()
            pose_from_states(torch.from_numpy(reference["state"][:, 281, 6:9]), state, metadata["dt"])
            unchanged = np.array_equal(before, saved["state"], equal_nan=True)
            if not unchanged:
                raise ValueError("visualization-only consumer mutated predicted states")
            item = {
                "visual_readout_changes_model_state": False,
                "prediction_sha256": file_digest(path),
                "separately_trained": False,
            }
            if variant == "B2":
                previous = np.concatenate((reference["state"][:, 281:282], saved["state"][:, :-1]), 1)
                indices = list(range(6)) + list(range(21, 33))
                velocity, old_velocity = saved["state"][..., indices], previous[..., indices]
                finite = np.isfinite(velocity) & np.isfinite(old_velocity)
                with np.errstate(invalid="ignore", over="ignore"):
                    acceleration = (velocity - old_velocity) / metadata["dt"]
                    recovered = old_velocity + metadata["dt"] * acceleration
                roundtrip_finite = finite & np.isfinite(recovered)
                difference = np.abs(recovered[roundtrip_finite] - velocity[roundtrip_finite])
                item.update(
                    B3_accel_is_reparameterization=True,
                    floating_point_max_velocity_roundtrip_error=float(difference.max()) if difference.size else None,
                    finite_velocity_coordinates=int(finite.sum()),
                    acceleration_display_overflow_coordinates=int((finite & ~roundtrip_finite).sum()),
                    alias_predictions_and_actions_are_identical=True,
                )
            evidence[f"{partition}/{mode}"] = item
    return evidence


def trajectory_table(destination, metadata, variant, seed):
    tables = []
    defaults = np.asarray(metadata["default_joint_position"])
    base_rows = json.loads((destination / "test_id-A/scored_rows.json").read_text())
    candidates = [row for row in base_rows if row["horizon_steps"] == 100 and row["reference_eligible"]]
    worst = max(candidates, key=lambda row: row["failure_aware_score"])["episode_id"] if candidates else 0
    episodes = sorted({0, 64, 128, 192, worst})
    for mode in ("A", "B"):
        saved = dict(np.load(destination / f"test_id-{mode}/predictions.npz"))
        for episode in episodes:
            index = saved["episode_id"].tolist().index(episode)
            state, truth = saved["state"][index], saved["true_state"][index]
            frame = {
                "time_s": saved["time_s"],
                "reference_valid": saved["reference_valid"][index],
                "prediction_finite": saved["prediction_finite"][index],
                "initialization_valid": bool(saved["initialization_valid"][index]),
                "variant": variant,
                "training_seed": seed,
                "seed": seed,
                "mode": mode,
                "episode_id": episode,
            }
            for axis, name in enumerate("xyz"):
                frame["predicted_com_" + name] = saved["position"][index, :, axis]
                frame["actual_com_" + name] = saved["true_position"][index, :, axis]
            dot = (saved["orientation"][index] * saved["true_orientation"][index]).sum(-1)
            frame["orientation_error_rad"] = 2 * np.arccos(np.clip(np.abs(dot), 0, 1))
            for joint, name in enumerate(metadata["joint_names"]):
                frame[f"predicted_position_{name}"] = state[:, 9 + joint] + defaults[joint]
                frame[f"actual_position_{name}"] = truth[:, 9 + joint] + defaults[joint]
                frame[f"predicted_velocity_{name}"] = state[:, 21 + joint]
                frame[f"actual_velocity_{name}"] = truth[:, 21 + joint]
            tables.append(pd.DataFrame(frame))
    return pd.concat(tables, ignore_index=True)


def verify_videos(destination, metadata_path, selections):
    outcomes = []
    for episode, mode in selections:
        video = destination / "videos" / f"{mode}-episode{episode}.mp4"
        status_path = video.with_suffix(".json")
        if not status_path.is_file():
            raise ValueError(f"missing required illustrative replay outcome: {status_path}")
        status = json.loads(status_path.read_text())
        prediction = destination / f"test_id-{mode}/predictions.npz"
        saved = dict(np.load(prediction))
        index = saved["episode_id"].tolist().index(episode)
        if not saved["initialization_valid"][index]:
            frames, reason = 0, "invalid_initialization"
        else:
            reference = saved["reference_valid"][index]
            finite = saved["prediction_finite"][index]
            invalid = np.flatnonzero(~reference | ~finite)
            frames = int(invalid[0]) if len(invalid) else len(reference)
            reason = (
                "horizon_complete"
                if not len(invalid)
                else ("reference_terminated" if not reference[frames] else "prediction_nonfinite")
            )
        expected = {
            "kind": "illustrative_kinematic_replay_of_scored_arrays",
            "episode_id": episode,
            "frames": frames,
            "end_reason": reason,
            "physics_steps": 0,
            "prediction_sha256": file_digest(prediction),
            "dataset_sha256": file_digest(metadata_path),
        }
        if any(status.get(key) != value for key, value in expected.items()):
            raise ValueError("illustrative replay provenance/coverage/censoring differs from its scored arrays")
        if frames and (not video.is_file() or status["video_sha256"] != file_digest(video)):
            raise ValueError("nonempty replay video is absent or changed")
        outcomes.append(status)
    return outcomes


def publish(protocol, cells):
    root = Path.cwd()
    budget_root = root / "logs/queue" / protocol["group"]
    campaign_cost = {
        "allocated_GPU_hours_per_device": {
            str(gpu): used_seconds(budget_root, gpu) / 3600 for gpu in protocol["compute_pool"]
        },
        "allocated_GPU_hours_per_phase": {
            phase: sum(used_seconds(budget_root / phase, gpu) for gpu in protocol["compute_pool"]) / 3600
            for phase in ("collection", "smoke", "train", "prediction", "shadow", "shadow_score", "render")
        },
        "accounting": (
            "conservative existing queue-event allocation, including startup/shutdown and failed attempts; shared"
            " campaign total, do not sum across evaluation runs"
        ),
    }
    api = wandb.Api()
    checked = []
    baseline = next(directory for variant, seed, directory, _, _ in cells if variant == "B0" and seed == 11)
    baseline_rows = json.loads((baseline / "world_model_eval/test_id-A/scored_rows.json").read_text())
    candidates = [row for row in baseline_rows if row["horizon_steps"] == 100 and row["reference_eligible"]]
    worst = max(candidates, key=lambda row: row["failure_aware_score"])["episode_id"] if candidates else 0
    replay_ids = sorted({0, 64, 128, 192, worst})
    # Verify every candidate before publishing any evaluation result.
    for variant, seed, directory, selected, final in cells:
        data = torch.load(selected, map_location="cpu", weights_only=True)
        training = api.run(f"{api.default_entity}/{protocol['project']}/{data['wandb_run_id']}")
        if training.state != "finished" or training.job_type != "train" or training.group != protocol["group"]:
            raise ValueError("checkpoint does not belong to a finished training run in this comparison")
        research = training.config["runner_cfg"]["research"]
        if (
            research["protocol_id"] != protocol["protocol_id"]
            or research["variant"] != variant
            or research["seed"] != seed
        ):
            raise ValueError("W&B training identity differs from evaluated checkpoint")
        all_rows, summaries, infos, stochastic = [], {}, {}, []
        for partition in ("test_id", "test_command_shift"):
            arrays, _ = read_partition(root / protocol["world_model"]["dataset"], partition)
            for mode in ("A", "B", "B_shadow", "A-final"):
                path = directory / "world_model_eval" / f"{partition}-{mode}"
                rows, info = verify_cell(
                    path,
                    final if mode == "A-final" else selected,
                    protocol,
                    partition,
                    "A" if mode == "A-final" else mode,
                    arrays["episode_id"].tolist(),
                    checkpoint_role="final" if mode == "A-final" else "selected",
                )
                all_rows.extend(
                    [{**row, "checkpoint_role": "final" if mode == "A-final" else "selected"} for row in rows]
                )
                summaries[f"{partition}/{mode}"] = summarize(rows)
                infos[f"{partition}/{mode}"] = info
                if mode == "A" and partition == "test_id":
                    streams = json.loads((path / "stochastic_rows.json").read_text())
                    ids = {(row["sample_stream"], row["episode_id"], row["horizon_steps"]) for row in streams}
                    expected = {
                        (stream, episode, horizon)
                        for stream in range(6100, 6116)
                        for episode in arrays["episode_id"]
                        for horizon in HORIZONS
                    }
                    if len(streams) != len(ids) or ids != expected:
                        raise ValueError("stochastic diagnostic streams incomplete or duplicated")
                    stochastic = streams
        selection = [(0, "A")] if seed != 11 else [(episode, mode) for episode in replay_ids for mode in ("A", "B")]
        videos = verify_videos(
            directory / "world_model_eval", root / protocol["world_model"]["dataset"] / "metadata.json", selection
        )
        (directory / "world_model_eval/replay_verification.json").write_text(json.dumps(videos, indent=2))
        checked.append((variant, seed, directory, selected, training, all_rows, summaries, infos, stochastic))
    for variant, seed, directory, selected, training, rows, summaries, infos, stochastic in checked:
        destination = directory / "world_model_eval"
        _, metadata = read_partition(root / protocol["world_model"]["dataset"], "test_id")
        controls = equality_controls(
            {"dataset": protocol["world_model"]["dataset"], "run": directory}, variant, metadata
        )
        (destination / "equality_controls.json").write_text(json.dumps(controls, indent=2, allow_nan=False))
        (destination / "summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False))
        (destination / "all_scored_rows.json").write_text(json.dumps(rows, allow_nan=False))
        config = {
            "domain": "world_model",
            "research": {
                "protocol_id": protocol["protocol_id"],
                "hypothesis_id": protocol["hypothesis_id"],
                "variant": variant,
                "training_seed": seed,
                "train_run_id": training.id,
                "seed": seed,
                "dataset_id": infos["test_id/A"]["dataset_sha256"],
                "dataset_verification": {"manifest_sha256": infos["test_id/A"]["dataset_sha256"]},
                "checkpoint_sha256": file_digest(selected),
                "protocol_contract": protocol,
            },
        }
        with wandb.init(
            project=protocol["project"],
            group=protocol["group"],
            job_type="eval",
            name=f"{variant}-seed{seed}__eval",
            config=config,
            tags=[
                f"variant:{variant}",
                f"seed:{seed}",
                f"protocol:{protocol['protocol_id']}",
                f"ckpt:{file_digest(selected)[:12]}",
            ],
        ) as run:
            artifact = wandb.Artifact(
                f"{protocol['group']}-{variant}-seed{seed}-scored",
                type="world-model-evaluation",
                metadata=config["research"],
            )
            artifact.add_dir(str(destination))
            artifact.add(wandb.Table(dataframe=pd.DataFrame(rows)), name="episodes")
            artifact.add(wandb.Table(dataframe=pd.DataFrame(stochastic)), name="stochastic")
            artifact.add(
                wandb.Table(dataframe=trajectory_table(destination, metadata, variant, seed)), name="trajectory_samples"
            )
            logged = run.log_artifact(artifact)
            logged.wait()
            run.summary["scored_artifact"] = logged.qualified_name
            run.summary["verified_cells"] = 8
            run.summary["summaries"] = summaries
            run.summary["timing"] = infos["test_id/A"]["timing"]
            for key, value in infos["test_id/A"]["timing"].items():
                if isinstance(value, (float, int)):
                    run.summary["efficiency/" + key] = value
            run.summary["population_inference"] = "inconclusive_two_training_seeds"
            run.summary["campaign_cost"] = campaign_cost
            if variant == "B0" and seed == 11:
                ledger = wandb.Artifact(
                    protocol["group"] + "-queue-cost-evidence", type="gpu-accounting", metadata=campaign_cost
                )
                for path in sorted(budget_root.rglob("events.jsonl")):
                    ledger.add_file(str(path), name=str(path.relative_to(budget_root)))
                logged_ledger = run.log_artifact(ledger)
                logged_ledger.wait()
                run.summary["campaign_cost_artifact"] = logged_ledger.qualified_name
            for key, values in summaries.items():
                for row in values:
                    prefix = f"{key}/{row['horizon_steps']}"
                    for metric, value in row.items():
                        if isinstance(value, (float, int)):
                            run.summary[f"{prefix}/{metric}"] = value
            run.log({"scored_population": wandb.Table(dataframe=pd.DataFrame(rows[:10000]))})
    print(
        f"EVALUATION_VERIFIED_AND_PUBLISHED: {len(checked)} training runs, 8 cells each; no conclusions drawn",
        flush=True,
    )

"""Offline dynamics updates through MBPOPPO's existing optimizer lifecycle."""

import argparse
import json
import math
import os
import sys
import time
import torch
import yaml
from datetime import datetime, timezone
from pathlib import Path

from rsl_rl.algorithms import MBPOPPO

import wandb
from research_workflow import launch_metadata, runtime_fingerprint
from research_workflow.verification import verify_retry

from .core import (
    ARCHITECTURE,
    DatasetReplay,
    FrozenActor,
    build_model,
    file_digest,
    physical_errors,
    predict,
    read_partition,
    score_at_horizon,
)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in (
        "task",
        "device",
        "logger",
        "log_project_name",
        "group",
        "job_type",
        "run_name",
        "world_model_dataset",
        "world_model_variant",
        "world_model_policy",
    ):
        result.add_argument("--" + name, required=True)
    result.add_argument("--seed", type=int, required=True)
    result.add_argument("--max_iterations", type=int, required=True)
    result.add_argument("--num_envs", type=int, required=True, help="World-model minibatch window count.")
    result.add_argument("--headless", action="store_true")
    result.add_argument("--wandb_tags", nargs="*", default=[])
    return result


def main(argv=None):
    args, extra = parser().parse_known_args(argv)
    if extra != [f"agent.device={args.device}"]:
        raise ValueError(f"unsupported offline dynamics overrides: {extra}")
    if args.logger != "wandb" or args.log_project_name != "rwm-explore":
        raise ValueError("scientific dynamics runs require the declared W&B project")
    research = launch_metadata()
    protocol = research.get("protocol_contract", {})
    if protocol.get("domain") != "world_model":
        raise ValueError("offline dynamics training requires an approved world_model protocol")
    if args.seed not in protocol["seeds"] or args.world_model_variant != research["variant"]:
        raise ValueError("training identity differs from the protocol")
    torch.set_num_threads(4)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    train, metadata = read_partition(args.world_model_dataset, "train")
    validation, validation_metadata = read_partition(args.world_model_dataset, "validation")
    if metadata != validation_metadata or file_digest(args.world_model_policy) != metadata["policy_sha256"]:
        raise ValueError("dataset metadata/policy mismatch")
    device = args.device
    model = build_model(args.world_model_variant, metadata, args.seed, device)
    actor = FrozenActor(args.world_model_policy, device)
    weights = {
        "state": 1.0,
        "sequence": 1.0,
        "bound": 1.0,
        "kl": 0.1,
        "extension": 1.0,
        "contact": 1.0,
        "termination": 1.0,
    }
    algorithm = MBPOPPO(
        actor,
        model,
        None,
        None,
        device=device,
        system_dynamics_forecast_horizon=8,
        system_dynamics_num_mini_batches=1,
        system_dynamics_mini_batch_size=args.num_envs,
        system_dynamics_loss_weights=weights,
    )
    algorithm.system_replay_buffer = DatasetReplay(train, metadata, args.seed, device)
    resolved = {
        "variant": args.world_model_variant,
        "seed": args.seed,
        "iterations": args.max_iterations,
        "batch_windows": args.num_envs,
        "dataset_sha256": file_digest(Path(args.world_model_dataset) / "metadata.json"),
        "policy_sha256": metadata["policy_sha256"],
        "normalization": [metadata["state_mean"], metadata["state_std"]],
        "architecture": ARCHITECTURE,
        "dt": metadata["dt"],
        "history": 32,
        "forecast": 8,
        "optimizer": {"type": "Adam", "learning_rate": 0.001, "weight_decay": 0.0, "max_grad_norm": 1.0},
        "loss_weights": weights,
        "window_sampling": "uniform_episode_then_uniform_valid_start_with_replacement",
        "action_normalization": "identity",
    }
    fingerprint = runtime_fingerprint(resolved_config=resolved)
    research = verify_retry(research, fingerprint)
    research.update(
        domain="world_model",
        dataset=resolved["dataset_sha256"],
        dataset_id=resolved["dataset_sha256"],
        dataset_verification={"manifest_sha256": resolved["dataset_sha256"]},
        loss_definition="original_sampled_MSE_reconstructed_45_state",
        runtime_fingerprint=fingerprint,
    )
    root = Path("logs/rsl_rl/anymal_d_world_model")
    run_dir = root / (datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S") + "_" + args.run_name)
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "params").mkdir()
    provenance = {
        **research,
        "runtime_fingerprint": fingerprint,
        "run_dir": str(run_dir.resolve()),
        "dataset": resolved["dataset_sha256"],
        "policy_sha256": metadata["policy_sha256"],
    }
    (run_dir / "research_provenance.json").write_text(json.dumps(provenance, indent=2))
    (run_dir / "params/agent.yaml").write_text(yaml.safe_dump(resolved))
    run = wandb.init(
        project=args.log_project_name,
        group=args.group,
        job_type=args.job_type,
        name=args.run_name,
        tags=args.wandb_tags,
        config={"runner_cfg": {"research": research}, "world_model": resolved},
    )
    if not os.environ.get("WORLD_MODEL_DATASET_ARTIFACT"):
        run.finish(exit_code=1)
        raise ValueError("the immutable dataset artifact identity is required")
    dataset_artifact = run.use_artifact(os.environ["WORLD_MODEL_DATASET_ARTIFACT"], type="dataset")
    artifact_manifest = dataset_artifact.get_path("metadata.json").download(root=str(run_dir / "dataset_identity"))
    if file_digest(artifact_manifest) != resolved["dataset_sha256"]:
        run.finish(exit_code=1)
        raise ValueError("local dataset manifest differs from the used immutable artifact")
    torch.manual_seed(args.seed)  # Couple training draws after the different model constructions.
    best, selected, start = math.inf, None, time.monotonic()
    exit_code = 1
    try:
        with (run_dir / "locomotion_metrics.jsonl").open("w") as stream:
            for iteration in range(args.max_iterations):
                losses = algorithm.update_system_dynamics()
                if not all(math.isfinite(value) for value in losses):
                    raise FloatingPointError(f"nonfinite dynamics loss at update {iteration}")
                record = {
                    "iteration": iteration,
                    "Checks/finite_model": 1.0,
                    "train/loss": sum(losses),
                    **{
                        f"train/{key}": value
                        for key, value in zip(
                            ["state", "sequence", "bound", "kl", "extension", "contact", "termination"], losses
                        )
                    },
                }
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                if iteration % 100 == 0:
                    stream.flush()
                    run.log({**record, "train/update": iteration + 1})
                    print(f"update {iteration + 1}/{args.max_iterations}, state loss {losses[0]:.5g}", flush=True)
                if (iteration + 1) % 1000 == 0 or iteration + 1 == args.max_iterations:
                    if not all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
                        raise FloatingPointError("nonfinite model checkpoint")
                    model.eval()
                    prediction = predict(model, validation, metadata, steps=100)
                    scored = score_at_horizon(physical_errors(prediction["state"], validation, metadata), 100)
                    eligible = scored["eligible"]
                    if not eligible.any():
                        raise ValueError("no valid fixed validation forks")
                    metric = float(scored["capped_score"][eligible].mean())
                    checkpoint = f"model_{iteration + 1}.pt"
                    if metric < best:
                        best, selected = metric, checkpoint
                    torch.save(
                        {
                            "system_dynamics_state_dict": model.state_dict(),
                            "variant": args.world_model_variant,
                            "metadata": metadata,
                            "seed": args.seed,
                            "iter": iteration + 1,
                            "system_dynamics_optimizer_state_dict": algorithm.system_dynamics_optimizer.state_dict(),
                            "selected_checkpoint": selected,
                            "validation_score": metric,
                            "runtime_fingerprint": fingerprint,
                            "research": research,
                            "wandb_run_id": run.id,
                        },
                        run_dir / checkpoint,
                    )
                    run.log({"val/id/trajectory_nrmse_2s": metric, "train/update": iteration + 1})
                    run.summary.update({"selected_checkpoint": selected, "selected_validation_score": best})
                    model.train()
        run.summary.update(
            {
                "train/updates": args.max_iterations,
                "cost/GPU_hours": (time.monotonic() - start) / 3600,
                "run_dir": str(run_dir.resolve()),
                "parameters": sum(p.numel() for p in model.parameters()),
            }
        )
        exit_code = 0
    finally:
        run.finish(exit_code=exit_code)
    print(f"DYNAMICS_COMPLETE {run_dir}, selected {selected}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])

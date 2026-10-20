"""Explicit world-model adapter over the existing clean-source, gate and GPU queue APIs.

No alternate runtime state: source contracts are in Git; the existing queue event
logs account for allocated time; datasets/checkpoints/scored evidence are artifacts.
"""

import argparse
import fcntl
import importlib.util
import json
import os
import subprocess
import sys
import torch
from pathlib import Path

import wandb
from research_workflow import checkpoint_digest as file_digest
from research_workflow import load_protocol
from research_workflow.governance import launch_approval_problems, read_workflow

ROOT = Path(__file__).resolve().parents[2]


def queue_module():
    spec = importlib.util.spec_from_file_location("research_runs", ROOT / "scripts/research_runs.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def preflight(protocol_path):
    protocol = load_protocol(Path(protocol_path))
    if protocol.get("domain") != "world_model":
        raise ValueError("this adapter requires domain: world_model")
    directory = ROOT / "research" / protocol["group"]
    workflow = read_workflow(directory)
    rr = queue_module()
    problems = rr.launch_problems(protocol["group"], rr.git_source())
    problems += launch_approval_problems(ROOT, directory, protocol, push=True)
    settings = protocol["world_model"]
    repository = (ROOT / settings["algorithm_repository"]).resolve()
    sys.path.insert(0, str(repository))
    head = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"], text=True).strip()
    if head != settings["algorithm_commit"] or dirty:
        problems.append("algorithm source is not the approved clean commit")
    limits = workflow["mandate"]["limits"]
    if protocol["compute_pool"] != limits["compute_pool"] or set(protocol["seeds"]) - set(
        limits["allowed_training_seed_values"]
    ):
        problems.append("device or training-seed restriction differs from user mandate")
    env = rr.source_env(
        base={
            **os.environ,
            "RSL_RL_REPOSITORY": str(repository),
            "RESEARCH_PUSH_REMOTE": settings["push_remote"],
            "CUDA_VISIBLE_DEVICES": ",".join(map(str, protocol["compute_pool"])),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
    )
    problems += rr.check_imports([sys.executable], env)
    if file_digest(ROOT / settings["policy"]) != settings["policy_sha256"]:
        problems.append("frozen policy bytes changed")
    if problems:
        raise ValueError("launch rejected: " + "; ".join(problems))
    os.environ["RESEARCH_PUSH_REMOTE"] = settings["push_remote"]
    budget = {
        "log_root": ROOT / "logs/queue" / protocol["group"],
        "hours_per_gpu": limits["max_GPU_hours_per_device"],
        "max_job_seconds": settings["max_job_seconds"],
    }
    return protocol, rr, env, budget


def collection_specs(protocol):
    for partition, spec in protocol["world_model"]["collection"].items():
        for batch in range(spec["batches"]):
            yield {
                "partition": partition,
                "seed": spec["seed"] + batch,
                "command_seed": spec["command_seed"] + batch,
                "num_envs": spec["num_envs"],
                "episode_offset": batch * spec["num_envs"],
            }


def collect(protocol, rr, env, budget):
    settings = protocol["world_model"]
    scratch = Path(budget["log_root"])
    specs = scratch / "collection_specs"
    specs.mkdir(parents=True, exist_ok=True)
    jobs = []
    for spec in collection_specs(protocol):
        name = f"collect-{spec['partition']}-{spec['episode_offset']}"
        config = specs / f"{name}.json"
        config.write_text(json.dumps(spec, indent=2))
        output = scratch / "collection_parts" / f"{name}.npz"
        jobs.append(
            {
                "name": name,
                "gpu": None,
                "argv": [
                    sys.executable,
                    rr.PLAY_SCRIPT,
                    "--task",
                    "Isaac-Velocity-Flat-Anymal-D-Play-v0",
                    "--headless",
                    "--device",
                    "cuda:{gpu}",
                    "--seed",
                    str(spec["seed"]),
                    "--num_envs",
                    str(spec["num_envs"]),
                    "--checkpoint",
                    settings["policy"],
                    "--world_model_config",
                    str(config),
                    "--world_model_output",
                    str(output),
                ],
                "research": {"protocol_id": protocol["protocol_id"], "job_type": "collection"},
            }
        )
    tag = rr.pin_launch(protocol["group"], "eval")
    if rr.run_queue(jobs, protocol["compute_pool"], scratch / "collection", env=env, launch_tag=tag, gpu_budget=budget):
        raise RuntimeError("collection failed; preserve all attempts and request independent review")
    assemble_dataset(protocol, rr, env, budget)


def assemble_dataset(protocol, rr, env, budget):
    """Verify declared raw collection before CPU-only assembly and publication."""
    from .dataset import assemble

    settings = protocol["world_model"]
    scratch = Path(budget["log_root"])
    reuse = settings.get("collection_reuse")
    parts = []
    for spec in collection_specs(protocol):
        path = scratch / "collection_parts" / f"collect-{spec['partition']}-{spec['episode_offset']}.npz"
        info = json.loads(path.with_suffix(".json").read_text())
        if info["spec"] != spec:
            raise ValueError("saved collection specification differs from approved protocol")
        if reuse:
            expected = reuse["files"][path.name]
            if (
                file_digest(path) != expected["sha256"]
                or file_digest(path.with_suffix(".json")) != expected["metadata_sha256"]
                or info["runtime_fingerprint"]["commit"] != reuse["source_commit"]
                or info["runtime_fingerprint"]["rsl_rl_commit"] != settings["algorithm_commit"]
            ):
                raise ValueError("saved collection differs from the approved immutable source and hashes")
        parts.append(path)
    metadata = assemble(parts, ROOT / settings["dataset"])
    artifact = wandb.Artifact(
        protocol["group"] + "-dataset",
        type="dataset",
        metadata={"protocol_id": protocol["protocol_id"], "policy_sha256": settings["policy_sha256"]},
    )
    artifact.add_dir(str(ROOT / settings["dataset"]))
    artifact.add_file(str(ROOT / settings["policy"]), name="frozen_policy.pt")
    # Collection creates only a report/artifact record, never a training/eval result.
    with wandb.init(
        project=protocol["project"],
        group=protocol["group"],
        job_type="report",
        name=protocol["group"] + "__dataset",
        config={"research": {"protocol_id": protocol["protocol_id"]}},
    ) as run:
        logged = run.log_artifact(artifact)
        logged.wait()
        run.summary["dataset/artifact"] = logged.qualified_name
        run.summary["dataset/manifest_sha256"] = file_digest(ROOT / settings["dataset"] / "metadata.json")
        run.summary["dataset/partitions"] = metadata["partitions"]


def train(protocol, rr, env, budget):
    settings = protocol["world_model"]
    # Use the exact immutable artifact version actually created by collection.
    api = wandb.Api()
    records = [
        r
        for r in api.runs(f"{api.default_entity}/{protocol['project']}", filters={"group": protocol["group"]})
        if r.name == protocol["group"] + "__dataset" and r.state == "finished"
    ]
    if len(records) != 1:
        raise ValueError("require exactly one completed immutable dataset publication")
    env["WORLD_MODEL_DATASET_ARTIFACT"] = records[0].summary["dataset/artifact"]
    if file_digest(ROOT / settings["dataset"] / "metadata.json") != records[0].summary["dataset/manifest_sha256"]:
        raise ValueError("dataset manifest changed since artifact publication")
    jobs = [
        rr.train_job(
            task=spec["task"],
            variant=variant,
            seed=seed,
            iterations=protocol["training_budget"]["iterations"],
            project=protocol["project"],
            group=protocol["group"],
            extra=spec["overrides"],
            num_envs=protocol["training_budget"]["num_envs"],
            protocol=protocol,
            python=sys.executable,
        )
        for variant, spec in protocol["variants"].items()
        for seed in protocol["seeds"]
    ]
    if problems := rr.protocol_job_problems(jobs, protocol):
        raise ValueError(problems)
    smoke = [
        rr.train_job(
            task=spec["task"],
            variant=variant,
            seed=settings["pilot_seed"],
            iterations=protocol["smoke_gate"]["iterations"],
            project=protocol["project"],
            group=protocol["group"],
            extra=spec["overrides"],
            num_envs=protocol["training_budget"]["num_envs"],
            protocol=protocol,
            python=sys.executable,
            smoke=True,
        )
        for variant, spec in protocol["variants"].items()
    ]
    tag = rr.pin_launch(protocol["group"], "train")
    scratch = Path(budget["log_root"])
    if rr.run_queue(
        smoke,
        protocol["compute_pool"],
        scratch / "smoke",
        env=env,
        launch_tag=tag,
        gpu_budget={**budget, "max_job_seconds": 900},
    ):
        raise RuntimeError("technical pilot failed; no full training launched")
    evidence = rr.check_smoke_gate(smoke, protocol, scratch / "smoke", commit=rr.git_source()["commit"])
    print(json.dumps(evidence, indent=2), flush=True)
    if not evidence["passed"] or not protocol["smoke_gate"]["auto_full"]:
        raise RuntimeError("technical verification or auto-full authorization missing")
    # The final cost forecast uses each complete pilot's measured queue allocation,
    # not an assumed throughput. Reserve declared nontraining hours before full runs.
    from research_workflow.gpu_budget import used_seconds

    spent = sum(used_seconds(scratch, gpu) for gpu in protocol["compute_pool"])
    pilot_seconds = sum(used_seconds(scratch / "smoke", gpu) for gpu in protocol["compute_pool"])
    estimated_full = (
        pilot_seconds
        * (protocol["training_budget"]["iterations"] / protocol["smoke_gate"]["iterations"])
        * len(protocol["seeds"])
    )
    capacity = 3600 * budget["hours_per_gpu"] * len(protocol["compute_pool"])
    print(
        f"Spent {spent/3600:.3f} GPUh, full forecast {estimated_full/3600:.3f} GPUh",
        flush=True,
    )
    if spent + estimated_full + settings["evaluation_reserve_GPU_hours"] * 3600 > capacity:
        raise RuntimeError("pilot cost forecast exceeds authorized campaign ceiling; no partial full campaign launched")
    if rr.run_queue(jobs, protocol["compute_pool"], scratch / "train", env=env, launch_tag=tag, gpu_budget=budget):
        raise RuntimeError("full dynamics training failed; preserve failed attempts")
    for job in jobs:
        directory, errors = rr.find_run_dir(ROOT / "logs/rsl_rl/anymal_d_world_model", job["name"])
        checked = (
            rr.check_train(
                directory, protocol["training_budget"]["iterations"], scratch / "train" / f"{job['name']}.log"
            )
            if directory
            else {}
        )
        if errors or not checked.get("complete") or checked.get("unverified"):
            raise RuntimeError(f"training verification failed: {job['name']}: {errors or checked}")
    print("ALL_DYNAMICS_TRAINING_VERIFIED; evaluation is a separate stage", flush=True)


def trained_cells(protocol, rr):
    for variant in protocol["variants"]:
        for seed in protocol["seeds"]:
            directory, errors = rr.find_run_dir(ROOT / "logs/rsl_rl/anymal_d_world_model", f"{variant}-seed{seed}")
            if errors or directory is None:
                raise ValueError(f"training directory not uniquely resolved: {errors}")
            final = directory / f"model_{protocol['training_budget']['iterations']}.pt"
            checkpoint = torch.load(final, map_location="cpu", weights_only=True)
            if checkpoint["research"]["protocol_id"] != protocol["protocol_id"]:
                raise ValueError("training checkpoint protocol mismatch")
            selected = directory / checkpoint["selected_checkpoint"]
            if not selected.is_file() or selected.parent != directory:
                raise ValueError("validation-selected checkpoint is missing")
            yield variant, seed, directory, selected, final


def score_job(protocol, rr, checkpoint, partition, mode, output, **extra):
    settings = protocol["world_model"]
    argv = [
        sys.executable,
        rr.PLAY_SCRIPT,
        "--world_model_score",
        "--dataset",
        settings["dataset"],
        "--policy",
        settings["policy"],
        "--checkpoint",
        str(checkpoint),
        "--device",
        "cuda:{gpu}",
        "--partition",
        partition,
        "--mode",
        mode,
        "--output",
        str(output),
    ]
    for key, value in extra.items():
        argv.append("--" + key)
        if value is not True:
            argv.append(str(value))
    return {
        "name": output.parent.name + "-" + output.name,
        "gpu": None,
        "argv": argv,
        "research": {"protocol_id": protocol["protocol_id"], "job_type": "eval", "protocol_contract": protocol},
    }


def evaluate(protocol, rr, env, budget):
    scratch = Path(budget["log_root"])
    settings = protocol["world_model"]
    cells = list(trained_cells(protocol, rr))
    jobs = []
    for variant, seed, directory, selected, final in cells:
        root = directory / "world_model_eval"
        for partition in ("test_id", "test_command_shift"):
            for mode in ("A", "B"):
                extra = {}
                if partition == "test_id" and mode == "A":
                    extra = {"measure_timing": True, "stochastic_streams": 16}
                jobs.append(score_job(protocol, rr, selected, partition, mode, root / f"{partition}-{mode}", **extra))
            jobs.append(score_job(protocol, rr, final, partition, "A", root / f"{partition}-A-final"))
    tag = rr.pin_launch(protocol["group"], "eval")
    if rr.run_queue(jobs, protocol["compute_pool"], scratch / "prediction", env=env, launch_tag=tag, gpu_budget=budget):
        raise RuntimeError("neural scoring incomplete; preserve all cells")
    jobs = []
    for variant, seed, directory, _, _ in cells:
        for partition in ("test_id", "test_command_shift"):
            specs = [spec for spec in collection_specs(protocol) if spec["partition"] == partition]
            if len(specs) != 1:
                raise ValueError("shadow comparison requires one complete declared test batch")
            config = scratch / "collection_specs" / f"collect-{partition}-0.json"
            root = directory / "world_model_eval"
            argv = [
                sys.executable,
                rr.PLAY_SCRIPT,
                "--task",
                protocol["variants"][variant]["eval_task"],
                "--headless",
                "--device",
                "cuda:{gpu}",
                "--seed",
                str(specs[0]["seed"]),
                "--num_envs",
                str(specs[0]["num_envs"]),
                "--checkpoint",
                settings["policy"],
                "--world_model_config",
                str(config),
                "--world_model_shadow",
                str(root / f"{partition}-B/predictions.npz"),
                "--world_model_output",
                str(root / f"{partition}-shadow.npz"),
            ]
            jobs.append(
                {
                    "name": f"{variant}-seed{seed}-{partition}-shadow",
                    "gpu": None,
                    "argv": argv,
                    "research": {
                        "protocol_id": protocol["protocol_id"],
                        "job_type": "eval",
                        "protocol_contract": protocol,
                    },
                }
            )
    if rr.run_queue(jobs, protocol["compute_pool"], scratch / "shadow", env=env, launch_tag=tag, gpu_budget=budget):
        raise RuntimeError("action-matched simulator replay incomplete")
    jobs = []
    for _, _, directory, selected, _ in cells:
        root = directory / "world_model_eval"
        for partition in ("test_id", "test_command_shift"):
            jobs.append(
                score_job(
                    protocol,
                    rr,
                    selected,
                    partition,
                    "B_shadow",
                    root / f"{partition}-B_shadow",
                    prediction=root / f"{partition}-B/predictions.npz",
                    shadow_reference=root / f"{partition}-shadow.npz",
                )
            )
    if rr.run_queue(
        jobs, protocol["compute_pool"], scratch / "shadow_score", env=env, launch_tag=tag, gpu_budget=budget
    ):
        raise RuntimeError("shadow scoring incomplete")
    print("ALL_SCORING_PROCESSES_FINISHED; independent artifact verification precedes publication", flush=True)


def render(protocol, rr, env, budget):
    settings = protocol["world_model"]
    cells = list(trained_cells(protocol, rr))
    baseline = next(directory for variant, seed, directory, _, _ in cells if variant == "B0" and seed == 11)
    rows = json.loads((baseline / "world_model_eval/test_id-A/scored_rows.json").read_text())
    candidates = [row for row in rows if row["horizon_steps"] == 100 and row["reference_eligible"]]
    worst = max(candidates, key=lambda row: row["failure_aware_score"])["episode_id"] if candidates else 0
    ids = sorted({0, 64, 128, 192, worst})
    jobs = []
    for variant, seed, directory, _, _ in cells:
        # All training runs receive a fixed-ID illustrative replay; seed 11 also gets
        # the four strata and the predeclared worst-baseline case in A and B.
        selection = [(0, "A")] if seed != 11 else [(episode, mode) for episode in ids for mode in ("A", "B")]
        for episode, mode in selection:
            output = directory / "world_model_eval/videos" / f"{mode}-episode{episode}.mp4"
            prediction = directory / f"world_model_eval/test_id-{mode}/predictions.npz"
            argv = [
                sys.executable,
                rr.PLAY_SCRIPT,
                "--task",
                protocol["variants"][variant]["eval_task"],
                "--headless",
                "--device",
                "cuda:{gpu}",
                "--num_envs",
                "2",
                "--seed",
                "4300",
                "--checkpoint",
                settings["policy"],
                "--world_model_replay",
                str(prediction),
                "--world_model_metadata",
                str(ROOT / settings["dataset"] / "metadata.json"),
                "--world_model_video",
                str(output),
                "--world_model_episode",
                str(episode),
            ]
            jobs.append(
                {
                    "name": f"{variant}-seed{seed}-{mode}-video{episode}",
                    "gpu": None,
                    "argv": argv,
                    "research": {
                        "protocol_id": protocol["protocol_id"],
                        "job_type": "video",
                        "protocol_contract": protocol,
                    },
                }
            )
    tag = rr.pin_launch(protocol["group"], "eval")
    if rr.run_queue(
        jobs, protocol["compute_pool"], Path(budget["log_root"]) / "render", env=env, launch_tag=tag, gpu_budget=budget
    ):
        raise RuntimeError("illustrative rendering incomplete; scored arrays remain authoritative")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument(
        "--phase", choices=["collect", "assemble", "train", "evaluate", "render", "publish"], required=True
    )
    arguments = parser.parse_args(argv)
    # One owner across every phase; existing events remain the only accounting log.
    lock_path = ROOT / "logs/queue" / Path(arguments.protocol).parent.name / "campaign.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another campaign phase owns this comparison's GPU queue") from error
        protocol, rr, env, budget = preflight(arguments.protocol)
        if arguments.phase == "publish":
            from .results import publish

            publish(protocol, list(trained_cells(protocol, rr)))
        else:
            {"collect": collect, "assemble": assemble_dataset, "train": train, "evaluate": evaluate, "render": render}[
                arguments.phase
            ](protocol, rr, env, budget)


if __name__ == "__main__":
    main()

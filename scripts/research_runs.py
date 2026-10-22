#!/usr/bin/env python3
"""Launch, watch and check research runs on a GPU pool. Git + W&B stay the record.

This removes repeated manual work around the repository's real entry points; it is not an
experiment framework and keeps no state of its own beyond runtime scratch::

    train-job      print one train.py job (a JSON line) with the W&B naming and tags
    eval-jobs      print one play.py job per evaluation cell; cells write local artifacts only
    queue          run a JSONL file of jobs on a GPU pool, one heavy job per free GPU
    status         what the queue launched, from its event log
    stop           stop launched processes, only while their recorded identity still matches
    check-train    final checkpoint, finite metric stream and a clean log for a training run
    check-eval     every planned cell, episode counts, finite metrics, provenance, comparability
    publish-eval   one W&B ``eval`` run per training run: cell table, pooled metrics, scored videos
    run-analysis   guard and run the comparison's analysis notebook; log its figures as an ``analysis`` run
    publish-report build or refresh the comparison's W&B Report (native panels, custom plots, analysis figures)
    tag            add/remove lifecycle and decision tags on a comparison's W&B runs
    validate-contract validate scientific hypothesis/protocol/decision YAML without launch
    worktree       create the comparison's ``dev/agent/<group>`` branch in its own Git worktree

W&B structure (see AGENTS.md): one stable project per robot/domain, one group per comparison,
``job_type`` in ``train | smoke | eval | video | analysis | report``, and tags from a fixed ``key:value``
vocabulary (``TAG_KEYS``). Identity tags are set at launch; ``status:*`` and ``decision:*`` are added
later with ``tag``.

Git structure: every comparison lives on ``dev/agent/<group>``, normally in its own worktree. The
queue launches only from a clean commit of that branch, pins it with a ``agent/<group>/<kind>-<n>``
tag, pushes branch and tag, and runs every job against this checkout's ``src`` (``PYTHONPATH``).

A job is ``{"name": ..., "argv": [...], "gpu": null}``; the ``{gpu}`` token in ``argv`` is replaced by
the GPU the queue assigns (``gpu`` pins a job to one index, rarely needed). ``--log-dir`` holds
``<name>.log`` per job and an append-only ``events.jsonl``; by convention it is
``logs/queue/<wandb-group>/``, which Git ignores. W&B, checkpoints and evaluation artifacts remain
the authority on what happened; the event log only says what this queue started.

PyYAML validates the small scientific contracts; ``check-eval`` reads Parquet through pandas, and
the ``publish-*`` / ``tag`` commands import ``wandb`` (and ``research_workflow.eval_results`` for metric definitions).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from research_workflow import (  # noqa: E402
    checkpoint_digest,
    content_id,
    ensure_supported_evaluation,
    final_checkpoint_name,
    load_protocol,
    paired_unit_id,
    read_contract,
    runtime_mismatches,
)
from research_workflow.governance import (  # noqa: E402
    dashboard_markdown,
    decision_dashboard,
    launch_approval_problems,
    read_workflow,
    record_gate,
    subjects,
    validate_workflow,
)
from research_workflow.metric_resolution import (  # noqa: E402
    CATALOG,
    RL_METRICS,
    MetricResolver,
    coverage_markdown,
    descriptor,
    metric_title,
)
from research_workflow.scientific_review import SECTIONS as SCIENTIFIC_SECTIONS  # noqa: E402
from research_workflow.scientific_review import identifier  # noqa: E402
from research_workflow.scientific_review import (  # noqa: E402
    bind_publication_evidence,
    build_evidence,
    evidence_markdown,
    render_review,
    review_template,
    validate_approval,
    validate_evidence,
    validate_review,
)
from research_workflow.verification import runner_config  # noqa: E402

TRAIN_SCRIPT = "scripts/reinforcement_learning/rsl_rl/train.py"
PLAY_SCRIPT = "scripts/reinforcement_learning/rsl_rl/play.py"
GPU_TOKEN = "{gpu}"
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

# Set by the job builders; an extra argument that also sets one would silently change the protocol.
MANAGED_FLAGS = {
    "--task", "--seed", "--max_iterations", "--device", "--logger", "--log_project_name", "--group",
    "--job_type", "--run_name", "--num_envs", "--num_episodes", "--checkpoint", "--eval_output_dir",
    "--force_terrain_level", "--difficulty", "--note", "--wandb_tags",
}  # fmt: skip
MANAGED_KEYS = {
    "seed", "agent.seed", "env.seed", "agent.device", "env.sim.device", "agent.max_iterations",
    "agent.run_name", "agent.logger", "agent.wandb_project", "env.scene.num_envs",
}  # fmt: skip


# W&B structure. The fallback project rsl_rl uses when none is given mixes every robot and task.
FALLBACK_PROJECT = "isaaclab"
JOB_TYPES = ("train", "smoke", "eval", "video", "analysis", "report")
TAG_KEYS = {
    "variant",
    "seed",
    "task",
    "ckpt",
    "protocol",
    "status",
    "decision",
    "hypothesis",
    "protocol_version",
    "cell",
    "attempt",
}
TAG_VALUES = {
    "status": {
        "smoke",
        "final",
        "superseded",
        "failed",
        "invalid",
        "inconclusive",
        "not_supported",
        "supported",
    },
    "decision": {"chosen", "rejected"},
}
FLAG_TAGS = {"dirty", "replay"}
# set after the run, with the `tag` command
LATER_TAG_KEYS = {"status", "decision"}


AGENT_BRANCH_PREFIX = "dev/agent/"
DEFAULT_TRAIN_METRICS = tuple(spec["name"] for spec in RL_METRICS)  # legacy import compatibility
# Every W&B line panel: randomly sampled points (not min/max bucketing) and time-weighted EMA 0.95.
LINE_PLOT_STYLE = {
    "point_visualization_method": "sampling",
    "smoothing_type": "exponentialTimeWeighted",
    "smoothing_factor": 0.95,
}
SECTIONS = SCIENTIFIC_SECTIONS
# Legacy names remain explicit subsection boundaries during migration.
HUMAN_SECTIONS = ("Analysis", "Decision", "Progress")


def agent_branch(group: str) -> str:
    if not NAME_RE.fullmatch(group):
        raise ValueError(f"group {group!r} must be a slug")
    return AGENT_BRANCH_PREFIX + group


def check_tag(tag: str) -> str:
    """A tag from the vocabulary: a flag (``dirty``, ``replay``) or ``key:value`` with a known key."""
    if len(tag) > 64:
        raise ValueError(f"tag {tag!r} is longer than W&B's 64 characters")
    if tag in FLAG_TAGS:
        return tag
    key, sep, value = tag.partition(":")
    if not sep or key not in TAG_KEYS or not value or not NAME_RE.fullmatch(value):
        raise ValueError(
            f"tag {tag!r} is not in the vocabulary ({sorted(TAG_KEYS)} as key:value, or {sorted(FLAG_TAGS)})"
        )
    if key in TAG_VALUES and value not in TAG_VALUES[key]:
        raise ValueError(f"tag {tag!r}: {key} must be one of {sorted(TAG_VALUES[key])}")
    return tag


def check_project(project: str, *, legacy: bool = False) -> str:
    """``legacy`` admits the fallback project only to publish/tag comparisons launched before the W&B
    structure existed; new runs (``train-job``) never go there."""
    if not project or (project == FALLBACK_PROJECT and not legacy):
        raise ValueError(f"use the experiment project (e.g. rwm-explore), not {FALLBACK_PROJECT!r}")
    return project


def parse_run_name(name: str) -> tuple[str, int]:
    """``temporal128-p0.1-seed1`` -> (``temporal128-p0.1``, 1)."""
    name = name.removesuffix("-smoke")
    name = re.sub(r"-(?:attempt|retry)\d+$", "", name)
    variant, sep, seed = name.rpartition("-seed")
    if not sep or not variant or not seed.isdigit():
        raise ValueError(f"run name {name!r} is not <variant>-seed<seed>")
    return variant, int(seed)


def identity_tags(variant: str, seed: int, task: str) -> list[str]:
    return [check_tag(t) for t in (f"variant:{variant}", f"seed:{seed}", f"task:{task}")]


def planned_cells(
    seeds: Sequence[int], levels: Sequence[int] = (), difficulties: Sequence[str] = ()
) -> list[tuple[int | None, str | None, int]]:
    """(level, difficulty, seed) per cell: levels x seeds, then difficulties x seeds (or seeds alone)."""
    cells = [(lv, None, s) for lv in levels for s in seeds] + [(None, d, s) for d in difficulties for s in seeds]
    return cells or [(None, None, s) for s in seeds]


def protocol_slug(
    seeds: Sequence[int], levels: Sequence[int] = (), difficulties: Sequence[str] = (),
    num_envs: int | None = None, num_episodes: int | None = None,
) -> str:  # fmt: skip
    """``L0-5-9_easy-hard_s41-42-43_e150x1``: the evaluation population, as one tag value."""
    where = "_".join(p for p in ("L" + "-".join(map(str, levels)) if levels else "", "-".join(difficulties)) if p)
    slug = "_".join(p for p in (where, "s" + "-".join(map(str, seeds)), f"e{num_envs}x{num_episodes}") if p)
    return slug[: 64 - len("protocol:")]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------------------------------
# Job construction
# ---------------------------------------------------------------------------------------------------


def run_name(variant: str, seed: int) -> str:
    """The W&B/log run name of one training run: ``<variant>-seed<seed>``."""
    if not NAME_RE.fullmatch(variant):
        raise ValueError(f"variant {variant!r} must be a short slug (letters, digits, '_', '-', '.')")
    return f"{variant}-seed{seed}"


def cell_id(seed: int, level: int | None = None, difficulty: str | None = None) -> str:
    """``L4-seed41``, ``medium-seed41`` or ``seed41``: one evaluation cell."""
    parts = [difficulty, None if level is None else f"L{level}", f"seed{seed}"]
    return "-".join(p for p in parts if p)


def check_extra(extra: Sequence[str]) -> list[str]:
    """Extra arguments pass through verbatim, except ones that would override a managed setting."""
    for token in extra:
        if token.startswith("--") and token.split("=", 1)[0] in MANAGED_FLAGS:
            raise ValueError(f"{token.split('=', 1)[0]} is set by this script; use its own option")
        if not token.startswith("-") and "=" in token and token.split("=", 1)[0].lstrip("+~") in MANAGED_KEYS:
            raise ValueError(f"{token.split('=', 1)[0]} is set by this script; use its own option")
    return list(extra)


def train_job(
    *,
    task: str,
    variant: str,
    seed: int,
    iterations: int,
    project: str,
    group: str,
    extra: Sequence[str] = (),
    num_envs: int | None = None,
    note: str | None = None,
    python: str | None = None,
    gpu: int | None = None,
    smoke: bool = False,
    dirty: bool = False,
    protocol: dict | None = None,
    attempt: int = 1,
    retry_reason: str | None = None,
    original_settings_id: str | None = None,
    original_provenance: str | None = None,
) -> dict:
    """``smoke`` marks a short test run (``job_type=smoke``); ``dirty`` adds the ``dirty`` tag."""
    name = run_name(variant, seed)
    if type(attempt) is not int or attempt < 1:
        raise ValueError("attempt must be a positive integer")
    if attempt > 1:
        name += f"-attempt{attempt}"
    if smoke:
        name += "-smoke"
    tags = identity_tags(variant, seed, task) + (["dirty"] if dirty else [])
    metadata = None
    if protocol:
        from research_workflow.verification import validate_reducers

        validate_reducers(protocol["evaluation"])
        spec = protocol["variants"].get(variant)
        budget = protocol["training_budget"]
        wanted_iterations = protocol["smoke_gate"].get("iterations") if smoke else budget["iterations"]
        if (
            not spec
            or task != spec["task"]
            or list(extra) != spec["overrides"]
            or seed not in protocol["seeds"]
            or iterations != wanted_iterations
            or num_envs != budget["num_envs"]
            or project != protocol["project"]
            or group != protocol["group"]
        ):
            raise ValueError("training job differs from approved protocol")
        if smoke and not protocol["smoke_gate"]["enabled"]:
            raise ValueError("smoke is not enabled in protocol")
        import hashlib

        settings_id = hashlib.sha256(
            json.dumps(
                {
                    "task": task,
                    "extra": list(extra),
                    "seed": seed,
                    "budget": budget,
                    "protocol": protocol["protocol_id"],
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        if original_provenance and original_settings_id is None:
            original_settings_id = json.loads(Path(original_provenance).read_text()).get("settings_id")
        retry = protocol["retry_policy"]
        if attempt > retry["max_attempts"]:
            raise ValueError("attempt exceeds retry policy")
        if attempt > 1 and (retry_reason not in retry["allowed_reasons"] or original_settings_id != settings_id):
            raise ValueError("retry requires an allowed reason and identical original settings ID")
        metadata = {
            "workflow": "scientific-contract-v1",
            "variant": variant,
            "seed": seed,
            "task": task,
            "hypothesis_id": protocol["hypothesis_id"],
            "protocol_id": protocol["protocol_id"],
            "protocol_version": protocol["protocol_version"],
            "cell_id": run_name(variant, seed),
            "attempt": attempt,
            "retry_reason": retry_reason,
            "settings_id": settings_id,
            "settings_identical": True if attempt == 1 else None,
            "job_type": "smoke" if smoke else "train",
            "protocol_contract": protocol,
        }
        if original_provenance:
            from research_workflow.verification import retry_reference

            metadata.update(retry_reference(metadata, original_provenance))
        tags += [
            f"hypothesis:{protocol['hypothesis_id']}",
            f"protocol:{protocol['protocol_id']}",
            f"protocol_version:{protocol['protocol_version']}",
            f"cell:{run_name(variant, seed)}",
            f"attempt:{attempt}",
        ]
    elif attempt > 1:
        raise ValueError("tracked retries require a protocol")
    if smoke:
        tags += ["status:smoke"]
    argv = [python or sys.executable, TRAIN_SCRIPT, "--task", task, "--headless"]
    argv += ["--device", f"cuda:{GPU_TOKEN}", "--seed", str(seed), "--max_iterations", str(iterations)]
    argv += ["--logger", "wandb", "--log_project_name", check_project(project), "--group", group]
    argv += ["--job_type", "smoke" if smoke else "train", "--run_name", name, "--wandb_tags", *tags]
    if num_envs is not None:
        argv += ["--num_envs", str(num_envs)]
    if note:
        argv += ["--note", note]
    # Outside distributed mode train.py applies --device to the simulator only.
    argv += [f"agent.device=cuda:{GPU_TOKEN}", *check_extra(extra)]
    return {"name": name, "argv": argv, "gpu": gpu, **({"research": metadata} if metadata else {})}


def eval_jobs(
    *,
    task: str,
    checkpoint: str,
    name: str,
    seeds: Sequence[int],
    output_root: str,
    levels: Sequence[int] = (),
    difficulties: Sequence[str] = (),
    num_envs: int | None = None,
    num_episodes: int | None = None,
    extra: Sequence[str] = (),
    python: str | None = None,
    protocol: dict | None = None,
) -> list[dict]:
    """One play.py job per cell, all with identical settings apart from the cell itself.

    Cells write local artifacts (and scored videos) only; they create no W&B runs. After
    ``check-eval`` passes, ``publish-eval`` logs the whole evaluation as one W&B run.
    """
    variant, train_seed = parse_run_name(name)
    if name.endswith("-smoke"):
        raise ValueError("smoke runs cannot be scored")
    if protocol:
        ensure_supported_evaluation(protocol)
        ev = protocol["evaluation"]
        spec = protocol["variants"].get(variant)
        if (
            not spec
            or task != spec["eval_task"]
            or list(extra) != spec["eval_overrides"]
            or train_seed not in protocol["seeds"]
            or list(seeds) != ev["seeds"]
            or list(levels) != ev["levels"]
            or list(difficulties) != ev["difficulties"]
            or num_envs != ev["num_envs"]
            or num_episodes != ev["num_episodes"]
        ):
            raise ValueError("evaluation job differs from approved protocol")
        if protocol["checkpoint_rule"] == "final" and Path(checkpoint).name != final_checkpoint_name(
            protocol["training_budget"]["iterations"]
        ):
            raise ValueError("checkpoint differs from protocol rule")
    extra = check_extra(extra)
    jobs = []
    for level, difficulty, seed in planned_cells(seeds, levels, difficulties):
        cell = cell_id(seed, level, difficulty)
        eval_name = f"{name}__{cell}"
        argv = [python or sys.executable, PLAY_SCRIPT, "--task", task, "--headless"]
        argv += ["--checkpoint", checkpoint, "--device", f"cuda:{GPU_TOKEN}", "--seed", str(seed)]
        argv += ["--eval_output_dir", f"{output_root.rstrip('/')}/{cell}"]
        # --note prefixes the video file name, so each cell's scored video is identifiable.
        argv += ["--note", eval_name]
        if num_envs is not None:
            argv += ["--num_envs", str(num_envs)]
        if num_episodes is not None:
            argv += ["--num_episodes", str(num_episodes)]
        if level is not None:
            argv += ["--force_terrain_level", str(level)]
        if difficulty is not None:
            argv += ["--difficulty", difficulty]
        argv += [f"agent.device=cuda:{GPU_TOKEN}", *extra]
        metadata = (
            {
                "workflow": "scientific-contract-v1",
                "hypothesis_id": protocol["hypothesis_id"],
                "protocol_id": protocol["protocol_id"],
                "protocol_version": protocol["protocol_version"],
                "cell_id": run_name(variant, train_seed),
                "training_run_name": name,
                "attempt": (
                    int(re.search(r"-attempt(\d+)$", name).group(1)) if re.search(r"-attempt(\d+)$", name) else 1
                ),
                "job_type": "eval",
                "protocol_contract": protocol,
            }
            if protocol
            else None
        )
        jobs.append({"name": eval_name, "argv": argv, "gpu": None, **({"research": metadata} if metadata else {})})
    return jobs


# ---------------------------------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------------------------------


def load_jobs(path: Path) -> list[dict]:
    jobs, names = [], set()
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        job = json.loads(line)
        where = f"{path}:{number}"
        if set(job) - {"name", "argv", "gpu", "research"}:
            raise ValueError(f"{where}: unknown keys {sorted(set(job) - {'name', 'argv', 'gpu', 'research'})}")
        name, argv, gpu = job.get("name"), job.get("argv"), job.get("gpu")
        if not isinstance(name, str) or not NAME_RE.fullmatch(name):
            raise ValueError(f"{where}: name must be a slug, got {name!r}")
        if name in names:
            raise ValueError(f"{where}: duplicate job name {name!r}")
        if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
            raise ValueError(f"{where}: argv must be a non-empty list of strings")
        if not any(GPU_TOKEN in a for a in argv):
            raise ValueError(f"{where}: argv must place the assigned GPU with {GPU_TOKEN}")
        if gpu is not None and (isinstance(gpu, bool) or not isinstance(gpu, int)):
            raise ValueError(f"{where}: gpu must be an integer or null")
        names.add(name)
        jobs.append(
            {
                "name": name,
                "argv": argv,
                "gpu": gpu,
                **({"research": job["research"]} if "research" in job else {}),
            }
        )
    return jobs


# This repository's tasks have no terrain levels or difficulty presets, so play.py cannot force either.
UNSUPPORTED_CELL_FLAGS = ("--force_terrain_level", "--difficulty")


def unsupported_cell_factor_problems(jobs: Sequence[dict]) -> list[str]:
    return [
        f"{job['name']}: play.py has no {flag} (this repository has no terrain levels or difficulty "
        "presets); leave evaluation.levels and evaluation.difficulties empty"
        for job in jobs
        for flag in UNSUPPORTED_CELL_FLAGS
        if flag in job["argv"]
    ]


def protocol_job_problems(jobs: Sequence[dict], protocol: dict) -> list[str]:
    problems = []
    for job in jobs:
        meta = job.get("research", {})
        if meta.get("protocol_id") != protocol["protocol_id"] or meta.get("protocol_contract") != protocol:
            problems.append(f"{job['name']}: missing or mismatched protocol identity")
            continue
        try:
            if meta.get("job_type") in ("train", "smoke"):
                variant, seed = parse_run_name(meta["cell_id"])
                spec = protocol["variants"][variant]
                expected = train_job(
                    task=spec["task"],
                    variant=variant,
                    seed=seed,
                    iterations=(
                        protocol["smoke_gate"].get("iterations")
                        if meta["job_type"] == "smoke"
                        else protocol["training_budget"]["iterations"]
                    ),
                    project=protocol["project"],
                    group=protocol["group"],
                    extra=spec["overrides"],
                    num_envs=protocol["training_budget"]["num_envs"],
                    protocol=protocol,
                    python=job["argv"][0],
                    gpu=job["gpu"],
                    smoke=meta["job_type"] == "smoke",
                    dirty="dirty" in job["argv"],
                    attempt=meta["attempt"],
                    retry_reason=meta["retry_reason"],
                    original_settings_id=meta["settings_id"] if meta["attempt"] > 1 else None,
                    original_provenance=meta.get("original_provenance"),
                )
                if job != expected:
                    problems.append(f"{job['name']}: job arguments/metadata differ from protocol")
            elif meta.get("job_type") == "eval":
                variant, _ = parse_run_name(meta["cell_id"])
                spec, ev = protocol["variants"][variant], protocol["evaluation"]
                argv = job["argv"]
                checkpoint = argv[argv.index("--checkpoint") + 1]
                cell_output = argv[argv.index("--eval_output_dir") + 1]
                expected = eval_jobs(
                    task=spec["eval_task"],
                    checkpoint=checkpoint,
                    name=meta.get("training_run_name", meta["cell_id"]),
                    seeds=ev["seeds"],
                    levels=ev["levels"],
                    difficulties=ev["difficulties"],
                    output_root=str(Path(cell_output).parent),
                    num_envs=ev["num_envs"],
                    num_episodes=ev["num_episodes"],
                    extra=spec["eval_overrides"],
                    python=argv[0],
                    protocol=protocol,
                )
                if job not in expected:
                    problems.append(f"{job['name']}: evaluation job differs from protocol")
            else:
                problems.append(f"{job['name']}: unsupported protocol job type")
        except (ValueError, KeyError, IndexError, OSError) as error:
            problems.append(f"{job['name']}: {error}")
    return problems


def substitute_gpu(argv: Sequence[str], gpu: int) -> list[str]:
    """Only the ``{gpu}`` token changes; every other argument is passed through exactly."""
    return [a.replace(GPU_TOKEN, str(gpu)) for a in argv]


def idle_gpus(pool: Sequence[int], ours: Iterable[int], foreign: Iterable[int], allow_shared: bool) -> list[int]:
    """Pool GPUs free for a new job: not running one of ours, and unless allowed not used by others."""
    busy = set(ours) | (set() if allow_shared else set(foreign))
    return [g for g in pool if g not in busy]


def assign(pending: Sequence[dict], idle: Sequence[int]) -> list[tuple[dict, int]]:
    """Give idle GPUs to pending jobs in order. A pinned job waits for its GPU; others may pass it."""
    free, out = list(idle), []
    for job in pending:
        if not free:
            break
        pin = job.get("gpu")
        if pin is None:
            out.append((job, free.pop(0)))
        elif pin in free:
            free.remove(pin)
            out.append((job, pin))
    return out


def foreign_gpu_use(run: Callable = subprocess.run) -> set[int] | None:
    """GPU indices with any compute process, or None when nvidia-smi cannot be queried."""
    try:
        query = ["nvidia-smi", "--format=csv,noheader"]
        gpus = run([*query, "--query-gpu=index,uuid"], capture_output=True, text=True, check=True).stdout
        apps = run([*query, "--query-compute-apps=gpu_uuid"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    index = {u.strip(): int(i) for i, u in (line.split(",") for line in gpus.splitlines() if line.strip())}
    return {index[u.strip()] for u in apps.splitlines() if u.strip() in index}


# ---------------------------------------------------------------------------------------------------
# Process identity: a PID alone never authorizes a signal (PIDs are reused)
# ---------------------------------------------------------------------------------------------------


def identity(pid: int) -> dict | None:
    root = Path("/proc") / str(pid)
    try:
        stat = (root / "stat").read_text().rsplit(")", 1)[1].split()
        return {
            "pid": pid,
            "start_ticks": int(stat[19]),
            "pgid": os.getpgid(pid),
            "uid": root.stat().st_uid,
            "cmdline": [c.decode() for c in (root / "cmdline").read_bytes().split(b"\0")[:-1]],
        }
    except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError, ValueError):
        return None


def matches(expected: dict | None) -> bool:
    return bool(expected) and identity(expected["pid"]) == expected


def stop_process(expected: dict, timeout: float = 30.0) -> str:
    """Stop the process group a job was started in, only if the recorded identity still matches."""
    if not matches(expected):
        return "not running (or identity changed); not signalled"
    group = expected["pgid"]
    if group != expected["pid"] or group == os.getpgrp():
        return "unsafe process group; not signalled"
    os.killpg(group, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and matches(expected):
        time.sleep(0.2)
    if matches(expected):
        os.killpg(group, signal.SIGKILL)
        time.sleep(0.5)
    return "stopped" if not matches(expected) else "still running after SIGKILL"


# ---------------------------------------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------------------------------------


def read_events(log_dir: Path) -> list[dict]:
    path = log_dir / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def _event(log_dir: Path, **event) -> None:
    with (log_dir / "events.jsonl").open("a") as stream:
        stream.write(json.dumps({"at": now(), **event}) + "\n")


def _git(root: Path, *args: str) -> str | None:
    out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
    return out.stdout.strip() if out.returncode == 0 else None


def git_source(root: Path | None = None) -> dict:
    root = root or ROOT
    status = _git(root, "status", "--porcelain")
    return {
        "commit": _git(root, "rev-parse", "HEAD"),
        "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def launch_problems(group: str, source: dict) -> list[str]:
    """A comparison launches only from a clean commit of its own agent branch."""
    problems = []
    if source.get("branch") != agent_branch(group):
        problems.append(f"on branch {source.get('branch')!r}; launch from {agent_branch(group)!r} (see `worktree`)")
    if source.get("dirty") is not False:
        problems.append("working tree is not clean; commit on the agent branch first")
    return problems


def job_groups(jobs: Iterable[dict]) -> set[str]:
    return {j["argv"][j["argv"].index("--group") + 1] for j in jobs if "--group" in j["argv"]}


def next_launch_tag(group: str, kind: str, root: Path | None = None) -> str:
    prefix = f"agent/{group}/{kind}-"
    existing = (_git(root or ROOT, "tag", "--list", f"{prefix}*") or "").split()
    numbers = [int(t[len(prefix) :]) for t in existing if t[len(prefix) :].isdigit()]
    return f"{prefix}{max(numbers, default=0) + 1}"


def pin_launch(group: str, kind: str, *, push: bool = True, root: Path | None = None) -> str:
    """Tag HEAD as this launch and push branch + tag, so every run's commit is reachable remotely."""
    root = root or ROOT
    tag = next_launch_tag(group, kind, root)
    if _git(root, "tag", tag) is None:
        raise SystemExit(f"could not create git tag {tag}")
    if push:
        remote = os.environ.get("RESEARCH_PUSH_REMOTE", "origin")
        out = subprocess.run(["git", "push", remote, agent_branch(group), tag], cwd=root, capture_output=True,
                             text=True, check=False)  # fmt: skip
        if out.returncode != 0:
            raise SystemExit(f"git push failed; nothing launched (tag {tag} kept locally):\n{out.stderr.strip()}")
    return tag


def source_paths(root: Path | None = None) -> list[Path]:
    """Where this checkout's importable code lives: the workflow package and the ``mbrl`` extension."""
    root = root or ROOT
    return [root / "src", root / "source" / "mbrl"]


def source_env(root: Path | None = None, base: dict | None = None) -> dict:
    """Children import ``mbrl`` and ``research_workflow`` from this checkout, not from the editable install."""
    env = dict(os.environ if base is None else base)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (*map(str, source_paths(root)), env.get("PYTHONPATH", "")) if p)
    if env.get("RSL_RL_REPOSITORY"):
        env["PYTHONPATH"] = env["RSL_RL_REPOSITORY"] + os.pathsep + env["PYTHONPATH"]
    env["PYTHONUNBUFFERED"] = "1"
    return env


def check_imports(interpreters: Iterable[str], env: dict, root: Path | None = None) -> list[str]:
    """Verify each interpreter resolves ``mbrl`` and ``research_workflow`` inside this checkout."""
    allowed = [p.resolve() for p in source_paths(root)]
    code = (
        "import importlib.util as u; print(u.find_spec('mbrl').origin); print(u.find_spec('research_workflow').origin)"
    )
    problems = []
    for python in sorted(set(interpreters)):
        out = subprocess.run([python, "-c", code], env=env, capture_output=True, text=True, check=False)
        origins = out.stdout.split() if out.returncode == 0 else []
        if len(origins) != 2 or not all(any(Path(o).resolve().is_relative_to(a) for a in allowed) for o in origins):
            problems.append(
                f"{python} imports {origins or out.stderr.strip()[-200:]}, not this checkout's "
                f"{[str(a) for a in allowed]}"
            )
    return problems


def default_branch(root: Path | None = None) -> str:
    """The repository's default branch (``origin/HEAD``), else the first of main/master that exists."""
    root = root or ROOT
    head = _git(root, "symbolic-ref", "--quiet", "--short", "refs/remotes/origin/HEAD")
    if head:
        return head.removeprefix("origin/")
    for name in ("main", "master"):
        if _git(root, "rev-parse", "--verify", "--quiet", f"refs/heads/{name}") is not None:
            return name
    return "main"


def create_worktree(group: str, *, base: str | None = None, path: Path | None = None, root: Path | None = None) -> Path:
    """``dev/agent/<group>`` in its own worktree, sharing the main checkout's ``logs/``."""
    root = root or ROOT
    base = base or default_branch(root)
    common = Path(_git(root, "rev-parse", "--path-format=absolute", "--git-common-dir") or root / ".git")
    main = common.parent
    branch = agent_branch(group)
    if _git(root, "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}") is not None:
        raise SystemExit(f"branch {branch} already exists; a new comparison needs a new group slug")
    target = path or main.parent / f"{main.name}-agent-{group}"
    out = subprocess.run(["git", "worktree", "add", "-b", branch, str(target), base], cwd=root, capture_output=True,
                         text=True, check=False)  # fmt: skip
    if out.returncode != 0:
        raise SystemExit(out.stderr.strip())
    (main / "logs").mkdir(exist_ok=True)
    if not (target / "logs").exists():
        (target / "logs").symlink_to(main / "logs", target_is_directory=True)
    exclude = common / "info" / "exclude"
    exclude.parent.mkdir(parents=True, exist_ok=True)
    lines = exclude.read_text().splitlines() if exclude.exists() else []
    if "/logs" not in lines:
        exclude.write_text(
            "\n".join([*lines, "# research worktrees share the main logs/ through a symlink", "/logs"]) + "\n"
        )
    return target


def _fail_fast_on_stale_retry(job: dict) -> None:
    """Reject a stale/mismatched retry before a single-job CLI command prints or launches it.

    ``queue``/``run_queue`` already check this at dispatch time; ``train-job`` and
    ``world-model-job`` print/launch a job directly, bypassing that check entirely.
    """
    research = job.get("research")
    if not research or research.get("job_type") not in {"train", "smoke"}:
        return
    from research_workflow.verification import verify_retry

    try:
        verify_retry(research, {"commit": git_source()["commit"]}, source_only=True)
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from error


def admission_poll(jobs, allow_shared, training_admission, poll):
    """An opt-in observation guard cannot relax scoring or shared-activity restrictions."""
    if training_admission is not None:
        if allow_shared or any(job.get("research", {}).get("job_type") not in {"train", "smoke"} for job in jobs):
            raise ValueError("inactive-context admission is training/pilot-only and cannot allow shared GPU activity")
        poll = min(poll, training_admission.settings["poll_seconds"])
    return poll


def run_queue(
    jobs: list[dict],
    pool: Sequence[int],
    log_dir: Path,
    *,
    allow_shared: bool = False,
    poll: float = 10.0,
    occupancy: Callable[[], set[int] | None] = foreign_gpu_use,
    env: dict | None = None,
    launch_tag: str | None = None,
    gpu_budget: dict | None = None,
    training_admission=None,
) -> int:
    poll = admission_poll(jobs, allow_shared, training_admission, poll)
    names = [job["name"] for job in jobs]
    if len(names) != len(set(names)):
        raise ValueError("duplicate queue job names; every child must have a unique identity")
    if not gpu_budget and any(
        job.get("research", {}).get("protocol_contract", {}).get("domain") == "world_model" for job in jobs
    ):
        raise ValueError("world-model queues require the approved cumulative GPU-hour limiter")
    log_dir.mkdir(parents=True, exist_ok=True)
    launched = {e["name"] for e in read_events(log_dir) if e.get("event") == "start"}
    again = sorted(launched & {j["name"] for j in jobs})
    if again:
        raise SystemExit(f"already launched from {log_dir}: {again}. Use a new --log-dir for a rerun.")
    for job in jobs:
        if job["gpu"] is not None and job["gpu"] not in pool:
            raise SystemExit(f"job {job['name']} is pinned to GPU {job['gpu']}, outside the pool {list(pool)}")
    source = {**git_source(), "launch_tag": launch_tag, "root": str(ROOT)}
    child_env = {**(env or source_env()), **({"RESEARCH_LAUNCH_TAG": launch_tag} if launch_tag else {})}
    _event(log_dir, event="queue_start", pool=list(pool), jobs=[j["name"] for j in jobs], source=source)
    from research_workflow.verification import verify_retry

    for job in jobs:
        if job.get("research", {}).get("job_type") in {"train", "smoke"}:
            try:
                verify_retry(job["research"], {"commit": source.get("commit")}, source_only=True)
            except (ValueError, OSError) as error:
                _event(log_dir, event="queue_end", failures=0, error=str(error))
                raise SystemExit(str(error)) from error
    pending, running, failures, waiting_note = list(jobs), {}, 0, None
    dispatch_error = None
    owned_identities = {}
    while pending or running:
        for name, (process, gpu, handle) in list(running.items()):
            code = process.poll()
            if code is not None:
                handle.close()
                _event(log_dir, event="exit", name=name, gpu=gpu, returncode=code)
                print(f"[{now()}] exit  {name} gpu={gpu} returncode={code}", flush=True)
                failures += code != 0
                del running[name]
                owned_identities.pop(name, None)
        if not pending and not running:
            break
        if training_admission is not None:
            try:
                foreign = training_admission.check(list(owned_identities.values()))
            except Exception as error:
                dispatch_error = str(error)
                _event(log_dir, event="isolation_failed", reason=dispatch_error, pending=[j["name"] for j in pending])
                pending.clear()
                for name, expected in owned_identities.items():
                    result = stop_process(expected)
                    _event(log_dir, event="owned_isolation_stop", name=name, result=result)
                training_admission = None
                foreign = set(pool)
        else:
            foreign = set() if allow_shared or not pending else occupancy()
        if foreign is None:
            raise SystemExit("cannot query GPU use with nvidia-smi; pass --allow-shared-gpu to skip the check")
        ours = {gpu for _, gpu, _ in running.values()}
        idle = idle_gpus(pool, ours, foreign - ours, allow_shared)
        for job, gpu in assign(pending, idle):
            try:
                if job.get("research", {}).get("protocol_id"):
                    current = git_source()
                    if current.get("dirty") or any(current.get(k) != source.get(k) for k in ("commit", "branch")):
                        raise ValueError("source changed while queued; pending scientific jobs not launched")
                if job.get("research", {}).get("job_type") in {"train", "smoke"}:
                    verify_retry(job["research"], {"commit": source.get("commit")}, source_only=True)
                argv = substitute_gpu(job["argv"], gpu)
                if gpu_budget:
                    from research_workflow.gpu_budget import timeout_argv

                    argv = timeout_argv(
                        argv, gpu_budget["log_root"], gpu, gpu_budget["hours_per_gpu"], gpu_budget["max_job_seconds"]
                    )
            except (ValueError, OSError) as error:
                dispatch_error = str(error)
                _event(
                    log_dir,
                    event="dispatch_blocked",
                    name=job["name"],
                    reason=dispatch_error,
                    pending=[j["name"] for j in pending],
                )
                pending.clear()
                # Keep monitoring already-launched jobs to their recorded outcomes.
                break
            handle = (log_dir / f"{job['name']}.log").open("ab")
            process = subprocess.Popen(
                argv, cwd=ROOT, env={**child_env, "RESEARCH_METADATA": json.dumps(job.get("research", {})), "RESEARCH_GPU": str(gpu), **({"WANDB_JOB_TYPE": job["research"]["job_type"]} if job.get("research") else {})}, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                start_new_session=True,
            )  # fmt: skip
            owned_identities[job["name"]] = identity(process.pid)
            _event(log_dir, event="start", name=job["name"], gpu=gpu, argv=argv, source=source,
                   identity=owned_identities[job["name"]], research=job.get("research", {}))  # fmt: skip
            print(f"[{now()}] start {job['name']} gpu={gpu} pid={process.pid}", flush=True)
            running[job["name"]] = (process, gpu, handle)
            pending.remove(job)
        note = sorted((set(pool) & (foreign - ours)) - ours) if pending else None
        if note and note != waiting_note:
            print(f"[{now()}] waiting: GPU(s) {note} are in use by other processes", flush=True)
        waiting_note = note
        if pending or running:
            time.sleep(poll)
    _event(log_dir, event="queue_end", failures=failures, **({"error": dispatch_error} if dispatch_error else {}))
    if dispatch_error:
        raise SystemExit(dispatch_error)
    return 1 if failures else 0


def job_table(log_dir: Path) -> list[dict]:
    rows: dict[str, dict] = {}
    for e in read_events(log_dir):
        if e.get("event") == "start":
            rows[e["name"]] = {
                "name": e["name"],
                "gpu": e["gpu"],
                "started": e["at"],
                "identity": e["identity"],
            }
        elif e.get("event") == "exit" and e["name"] in rows:
            rows[e["name"]].update(returncode=e["returncode"], finished=e["at"])
    for row in rows.values():
        row["alive"] = matches(row["identity"])
        log = log_dir / f"{row['name']}.log"
        text = log.read_text(errors="replace") if log.exists() else ""
        row["traceback"] = "Traceback (most recent call last)" in text
    return list(rows.values())


# ---------------------------------------------------------------------------------------------------
# Completion checks: a zero exit code is not completion
# ---------------------------------------------------------------------------------------------------


def find_run_dir(log_root: Path, name: str) -> tuple[Path | None, list[str]]:
    """rsl_rl writes ``<log_root>/<timestamp>_<run_name>``."""
    found = sorted(p for p in log_root.glob(f"*_{name}") if p.is_dir())
    if len(found) != 1:
        return None, [f"expected exactly one run directory {log_root}/*_{name}, found {len(found)}"]
    return found[0], []


def check_train(run_dir: Path, iterations: int, log: Path | None = None) -> dict:
    """Fresh-run checks: ``model_<iterations-1>.pt``, a complete finite metric stream, no traceback."""
    problems, unverified = [], []
    if not (run_dir / "params" / "agent.yaml").is_file():
        problems.append("params/agent.yaml missing")
    checkpoint = run_dir / final_checkpoint_name(iterations)
    if not checkpoint.is_file():
        problems.append(f"final checkpoint {checkpoint.name} missing")
    else:
        from research_workflow.verification import checkpoint_problems

        problems.extend(checkpoint_problems(checkpoint))
    provenance_file = run_dir / "research_provenance.json"
    provenance = json.loads(provenance_file.read_text()) if provenance_file.is_file() else {}
    finite_key = "Checks/finite_model" if provenance.get("domain") == "world_model" else "Checks/finite_rewards_std"
    stream = run_dir / "locomotion_metrics.jsonl"
    if stream.is_file():
        records = [json.loads(line) for line in stream.read_text().splitlines() if line.strip()]
        if len(records) != iterations or (records and records[-1].get("iteration") != iterations - 1):
            problems.append(f"metric stream has {len(records)} records, expected {iterations}")
        bad = [
            r.get("iteration")
            for r in records
            if r.get(finite_key) != 1.0 or any(isinstance(v, float) and not math.isfinite(v) for v in r.values())
        ]
        if bad:
            problems.append(f"non-finite or failed checks at iterations {bad[:5]}")
    else:
        unverified.append("no local metric stream for this runner: check the W&B history is finite")
    if log is not None:
        text = log.read_text(errors="replace") if log.is_file() else ""
        if not text:
            problems.append(f"log {log} missing or empty")
        elif "Traceback (most recent call last)" in text:
            problems.append(f"traceback in {log}")
    provenance_file = run_dir / "research_provenance.json"
    provenance = (
        json.loads(provenance_file.read_text()) if provenance_file.is_file() else {"workflow": "legacy workflow"}
    )
    return {
        "provenance": provenance,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "complete": not problems,
        "problems": problems,
        "unverified": unverified,
    }


def check_smoke_gate(jobs: Sequence[dict], protocol: dict, log_dir: Path, *, commit: str | None = None) -> dict:
    """Technical evidence only; incomplete or unverified metrics never open the full-run gate."""
    checks = []
    for job in jobs:
        variant, _ = parse_run_name(job["name"])
        run_dir, errors = find_run_dir(Path(protocol["smoke_gate"]["log_roots"][variant]), job["name"])
        check = (
            check_train(run_dir, protocol["smoke_gate"]["iterations"], log_dir / f"{job['name']}.log")
            if run_dir
            else {}
        )
        if run_dir:
            provenance = check["provenance"]
            expected = job.get("research", {})
            for key in ("hypothesis_id", "protocol_id", "cell_id", "attempt", "job_type", "settings_id"):
                if not expected.get(key) or provenance.get(key) != expected[key]:
                    check["problems"].append(f"smoke provenance mismatch: {key}")
            if commit and (provenance.get("runtime_fingerprint") or {}).get("commit") != commit:
                check["problems"].append("smoke source commit mismatch")
            check["complete"] = not check["problems"]
        checks.append({"name": job["name"], **check, "lookup_errors": errors})
    return {
        "passed": bool(checks) and all(
            c.get("complete") and not c.get("unverified") and not c["lookup_errors"] for c in checks
        ),
        "checks": checks,
    }


def _rows(cell: Path) -> list[dict] | None:
    parquet, fallback = cell / "evaluation_results.parquet", cell / "evaluation_results.json"
    if parquet.is_file():
        import pandas as pd

        return pd.read_parquet(parquet).to_dict("records")
    return json.loads(fallback.read_text()) if fallback.is_file() else None


def required_eval_metrics(protocol: dict) -> list[str]:
    """Exact metric keys, never inferred from free-text scientific constraints."""
    from research_workflow.verification import required_metric_keys

    return required_metric_keys(protocol["evaluation"])


def finite_metric_problems(values: dict, metrics: Sequence[str]) -> list[str]:
    return [
        f"metric {m} is {values.get(m)!r}"
        for m in metrics
        if not isinstance(values.get(m), int | float) or isinstance(values.get(m), bool) or not math.isfinite(values[m])
    ]


def scored_row_problems(rows: Sequence[dict], protocol: dict, context: dict) -> list[str]:
    """Equal environment contribution, finite declared evidence and collision-free paired IDs."""
    selected = [r for r in rows if r.get("episode_id") == 0]
    n = protocol["evaluation"]["num_envs"]
    problems = []
    env_ids = [r.get("env_id") for r in selected]
    if len(selected) != n or any(type(i) is not int for i in env_ids) or set(env_ids) != set(range(n)):
        problems.append("each declared environment must contribute exactly one scored episode")
    from research_workflow.verification import declared_metric_reducers

    reducers = declared_metric_reducers(protocol["evaluation"])
    for metric, reducer in reducers.items():
        column = reducer["column"]
        if column:
            if any(not isinstance(r.get(column), int | float | bool) or not math.isfinite(r[column]) for r in selected):
                problems.append(f"non-finite or missing scored column {column} for {metric}")
            if reducer["reduction"] == "rate" and any(r.get(column) not in (0, 1) for r in selected):
                problems.append(f"non-binary outcome in scored column {column}")
            if column.endswith("tracking_error") and any(
                isinstance(r.get(column), int | float) and r[column] < 0 for r in selected
            ):
                problems.append(f"negative episode RMSE in {column}")
    for key in ("seed", "level", "difficulty"):
        if context.get(key) is not None and any(key in r and r[key] != context[key] for r in selected):
            problems.append(f"scored row {key} differs from evaluation cell")
    factors = protocol["evaluation"]["paired_factors"]
    if factors:
        try:
            ids = [paired_unit_id(r, context, factors) for r in selected]
            if len(set(ids)) != len(ids):
                problems.append("duplicate paired evaluation unit")
        except ValueError as error:
            problems.append(str(error))
    return problems


def protocol_matrix(protocol: dict) -> set[tuple]:
    ev = protocol["evaluation"]
    return set(planned_cells(ev["seeds"], ev["levels"], ev["difficulties"]))


def check_eval_cell(
    cell: Path,
    *,
    seed: int,
    level: int | None = None,
    difficulty: str | None = None,
    checkpoint: str | None = None,
    expected_episodes: int | None = None,
    expected_first_episodes: int | None = None,
    metrics: Sequence[str] = (),
) -> dict:
    """Check one evaluation cell against the protocol it was meant to follow.

    ``first_episodes`` counts rows with ``episode_id == 0``. A fixed-step evaluation replaces early
    failures with new episodes, so the full row count varies and is biased toward failures; the
    first episode of every environment is the unbiased scored population.
    """
    out: dict = {"cell": cell.name, "path": str(cell), "problems": []}
    p = out["problems"]
    rows, run_file, summary_file = _rows(cell), cell / "evaluation_run.json", cell / "evaluation_summary.json"
    if rows is None or not run_file.is_file() or not summary_file.is_file():
        p.append("missing evaluation artifacts")
        return {**out, "ok": False}
    run = json.loads(run_file.read_text())
    values = json.loads(summary_file.read_text()).get("metrics", {})
    out.update(
        episodes=len(rows),
        first_episodes=sum(r.get("episode_id") == 0 for r in rows),
        # Same-named metrics differ between profiles (e.g. survival_rate); never compare across them.
        profile="forward_course" if any("completed" in r for r in rows) else "general",
        task=run.get("task"),
        num_envs=run.get("num_envs"),
        num_episodes=run.get("num_episodes_requested_per_env"),
        commit=(run.get("git") or {}).get("commit"),
        dirty=(run.get("git") or {}).get("dirty"),
        metrics={m: values.get(m) for m in metrics},
        protocol_id=(run.get("research") or {}).get("protocol_id"),
        episode_sampling=((run.get("research") or {}).get("protocol_contract") or {})
        .get("evaluation", {})
        .get("episode_sampling", "first_episode_per_env"),
        runtime_fingerprint=run.get("runtime_fingerprint"),
        checkpoint=run.get("checkpoint"),
        checkpoint_sha256=run.get("checkpoint_sha256"),
        research=run.get("research") or {},
    )
    rs = run.get("research") or {}
    contract = rs.get("protocol_contract")
    if rs.get("protocol_id") and not contract:
        p.append("new-protocol evidence is missing its contract; cannot downgrade to legacy")
    if contract:
        if rs.get("protocol_id") != content_id("protocol", contract):
            p.append("evaluation protocol content ID mismatch")
        ensure_supported_evaluation(contract)
        ev = contract["evaluation"]
        if (
            seed not in ev["seeds"]
            or run.get("num_envs") != ev["num_envs"]
            or run.get("num_episodes_requested_per_env") != ev["num_episodes"]
        ):
            p.append("evaluation population differs from protocol")
        if level is not None and level not in ev["levels"]:
            p.append("level outside protocol")
        if difficulty is not None and difficulty not in ev["difficulties"]:
            p.append("difficulty outside protocol")
        if run.get("task") not in {v["eval_task"] for v in contract["variants"].values()}:
            p.append("task outside protocol")
        if rs.get("job_type") == "smoke":
            p.append("smoke runs cannot be scored")
        if out["profile"] != contract["evaluation"]["profile"]:
            p.append("evaluation profile differs from protocol")
        p.extend(scored_row_problems(rows, contract, {"seed": seed, "level": level, "difficulty": difficulty}))
        metrics = list(dict.fromkeys([*metrics, *required_eval_metrics(contract)]))
    if run.get("seed") != seed:
        p.append(f"seed {run.get('seed')} != {seed}")
    if level is not None and run.get("forced_terrain_level") != level:
        p.append(f"forced level {run.get('forced_terrain_level')} != {level}")
    if difficulty is not None and run.get("requested_difficulty") != difficulty:
        p.append(f"difficulty {run.get('requested_difficulty')} != {difficulty}")
    if checkpoint is not None and Path(run.get("checkpoint", "")).resolve() != Path(checkpoint).resolve():
        p.append(f"checkpoint {run.get('checkpoint')} is not the requested one")
    if expected_episodes is not None and len(rows) != expected_episodes:
        p.append(f"{len(rows)} episodes, expected {expected_episodes}")
    if expected_first_episodes is not None and out["first_episodes"] != expected_first_episodes:
        p.append(f"{out['first_episodes']} first episodes, expected {expected_first_episodes}")
    p.extend(finite_metric_problems(values, metrics))
    out["metrics"] = {m: values.get(m) for m in metrics}
    return {**out, "ok": not p}


def check_eval(
    roots: Sequence[Path],
    *,
    seeds: Sequence[int],
    levels: Sequence[int] = (),
    difficulties: Sequence[str] = (),
    expected_episodes: int | None = None,
    expected_first_episodes: int | None = None,
    checkpoints: Sequence[str] = (),
    metrics: Sequence[str] = (),
    protocol: dict | None = None,
) -> dict:
    """Every planned cell under every root, and whether all cells are mutually comparable."""
    if checkpoints and len(checkpoints) != len(roots):
        raise ValueError("pass one --checkpoint per --output-root")
    problems = []
    # Embedded contracts also protect callers who forget --protocol. Historical roots stay readable.
    contracts = {}
    for root in roots:
        for file in root.glob("*/evaluation_run.json"):
            rs = json.loads(file.read_text()).get("research") or {}
            if rs.get("protocol_contract"):
                contracts[content_id("protocol", rs["protocol_contract"])] = rs["protocol_contract"]
    if protocol:
        contracts[protocol["protocol_id"]] = protocol
    if len(contracts) > 1:
        problems.append("evaluation roots contain different protocols")
    protocol = protocol or next(iter(contracts.values()), None)
    if protocol:
        ensure_supported_evaluation(protocol)
        requested = set(planned_cells(seeds, levels, difficulties))
        if requested != protocol_matrix(protocol):
            problems.append("requested evaluation matrix differs from approved protocol")
        ev = protocol["evaluation"]
        seeds, levels, difficulties = ev["seeds"], ev["levels"], ev["difficulties"]
        metrics = list(dict.fromkeys([*metrics, *required_eval_metrics(protocol)]))
    cells = []
    for i, root in enumerate(roots):
        for level, difficulty, seed in planned_cells(seeds, levels, difficulties):
            result = check_eval_cell(
                root / cell_id(seed, level, difficulty), seed=seed, level=level, difficulty=difficulty,
                checkpoint=checkpoints[i] if checkpoints else None,
                expected_episodes=expected_episodes,
                expected_first_episodes=expected_first_episodes, metrics=metrics,
            )  # fmt: skip
            if protocol and result.get("protocol_id") != protocol["protocol_id"]:
                result["problems"].append("evaluation protocol identity missing or mismatched")
                result["ok"] = False
            cells.append({"root": str(root), "level": level, "difficulty": difficulty, "seed": seed, **result})
    done = [c for c in cells if "profile" in c]
    keys = ("profile", "num_envs", "num_episodes", "protocol_id", "episode_sampling")
    protocols = sorted({tuple(str(c.get(k)) for k in keys) for c in done})
    comparable = len(protocols) <= 1
    runtime_differences = runtime_mismatches([c.get("runtime_fingerprint") for c in done])
    return {
        "complete": bool(cells) and not problems and all(c["ok"] for c in cells),
        "problems": problems,
        "comparable": comparable,
        "runtime_mismatches": runtime_differences,
        "protocols": [dict(zip(keys, p, strict=True)) for p in protocols],
        "commits": sorted({str(c.get("commit")) for c in done}),
        "cells": cells,
    }


# ---------------------------------------------------------------------------------------------------
# W&B publication: one eval run per training run, one report run per comparison
# ---------------------------------------------------------------------------------------------------


def _relative(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _finite(metrics: dict) -> dict:
    return {
        k: v
        for k, v in metrics.items()
        if k != "num_episodes" and isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)
    }


def build_eval_publication(  # noqa: C901
    *,
    group: str,
    name: str,
    task: str,
    eval_task: str,
    checkpoint: str,
    output_root: str,
    cells: Sequence[dict],
    rows_by_cell: dict[str, list[dict]],
    aggregate: Callable[[list[dict]], dict],
    levels: Sequence[int] = (),
    difficulties: Sequence[str] = (),
    population: str = "first",
    protocol: dict | None = None,
) -> dict:
    """Everything the W&B evaluation run of one training run will hold. Pure; no W&B access.

    ``population="first"`` scores the first episode of every environment (``episode_id == 0``), the
    unbiased population of a fixed-step evaluation; ``"all"`` uses every recorded episode. Pooled
    metrics are recomputed from the pooled rows with the repository's own ``aggregate`` function, so
    rates keep their episode denominators.
    """
    if population not in ("first", "all"):
        raise ValueError("population must be 'first' or 'all'")
    variant, seed = parse_run_name(name)
    if name.endswith("-smoke"):
        raise ValueError("smoke runs cannot be scored")
    if protocol is None:
        protocol = next(
            (
                c.get("research", {}).get("protocol_contract")
                for c in cells
                if c.get("research", {}).get("protocol_contract")
            ),
            None,
        )
        if protocol is None and any(c.get("protocol_id") for c in cells):
            raise ValueError("new-protocol evidence needs its contract; cannot downgrade to legacy")
    if protocol:
        ensure_supported_evaluation(protocol)
        from research_workflow.verification import protocol_aggregate

        aggregate = protocol_aggregate(aggregate, protocol)
        if population != "first" or any(c.get("protocol_id") != protocol["protocol_id"] for c in cells):
            raise ValueError("mismatched evaluation protocol or episode sampling")
        ev = protocol["evaluation"]
        spec = protocol["variants"].get(variant)
        if (
            group != protocol["group"]
            or not spec
            or task != spec["task"]
            or eval_task != spec["eval_task"]
            or seed not in protocol["seeds"]
        ):
            raise ValueError("publication training identity differs from protocol")
        if Path(checkpoint).name != final_checkpoint_name(protocol["training_budget"]["iterations"]):
            raise ValueError("checkpoint differs from protocol rule")
        actual = [(c["level"], c["difficulty"], c["seed"]) for c in cells]
        if len(actual) != len(set(actual)) or set(actual) != protocol_matrix(protocol):
            raise ValueError("publication requires the complete approved evaluation matrix exactly once")
        if list(levels) != ev["levels"] or list(difficulties) != ev["difficulties"]:
            raise ValueError("publication slices differ from approved protocol")
        attempt = int(re.search(r"-attempt(\d+)$", name).group(1)) if re.search(r"-attempt(\d+)$", name) else 1
        if not 1 <= attempt <= protocol["retry_policy"]["max_attempts"]:
            raise ValueError("publication attempt outside protocol")
        evidence_errors = []
        digest = checkpoint_digest(checkpoint)
        for c in cells:
            if c.get("num_envs") != ev["num_envs"] or c.get("num_episodes") != ev["num_episodes"]:
                evidence_errors.append("evaluation population differs from protocol")
            if c.get("profile") != ev["profile"]:
                evidence_errors.append("evaluation profile differs from protocol")
            rs = c.get("research", {})
            if (
                not rs.get("protocol_contract")
                or content_id("protocol", rs["protocol_contract"]) != protocol["protocol_id"]
            ):
                evidence_errors.append("evaluation protocol contract is missing or mismatched")
            for key, value in {
                "hypothesis_id": protocol["hypothesis_id"],
                "protocol_id": protocol["protocol_id"],
                "protocol_version": protocol["protocol_version"],
                "cell_id": run_name(variant, seed),
                "training_run_name": name,
                "attempt": attempt,
                "job_type": "eval",
            }.items():
                if rs.get(key) != value:
                    evidence_errors.append(f"evaluation training provenance mismatch: {key}")
            if c.get("task") != eval_task:
                evidence_errors.append("evaluation task differs from training variant")
            if not c.get("commit") or c.get("dirty") is not False:
                evidence_errors.append("evaluation needs recorded clean-commit provenance")
            if (
                not digest
                or c.get("checkpoint_sha256") != digest
                or Path(c.get("checkpoint") or "").resolve() != Path(checkpoint).resolve()
            ):
                evidence_errors.append("evaluated checkpoint path/content differs or is unverified")
            rows = rows_by_cell.get(c["cell"], [])
            evidence_errors.extend(scored_row_problems(rows, protocol, c))
        if len({c.get("commit") for c in cells}) != 1:
            evidence_errors.append("evaluation cells have different source commits")
        if evidence_errors:
            raise ValueError("; ".join(dict.fromkeys(evidence_errors)))
    if not cells:
        raise ValueError("cannot publish an empty evaluation")

    def scored(cell: str) -> list[dict]:
        return [r for r in rows_by_cell[cell] if population == "all" or r.get("episode_id") == 0]

    def checked_aggregate(rows: list[dict]) -> dict:
        values = aggregate(rows)
        if protocol:
            errors = finite_metric_problems(values, required_eval_metrics(protocol))
            if errors:
                raise ValueError("non-finite or missing required scored metrics: " + "; ".join(errors))
        return _finite(values)

    table = []
    for c in cells:
        rows = scored(c["cell"])
        table.append({"cell": c["cell"], "level": c["level"], "difficulty": c["difficulty"], "seed": c["seed"],
                      "episodes": len(rows), **checked_aggregate(rows)})  # fmt: skip
    summary: dict = {}

    def pool(prefix: str, selected: Sequence[dict]) -> None:
        rows = [r for c in selected for r in scored(c["cell"])]
        summary[f"{prefix}/episodes"] = len(rows)
        summary.update({f"{prefix}/{k}": v for k, v in checked_aggregate(rows).items()})

    pool("eval", cells)
    for level in levels:
        pool(f"eval/L{level}", [c for c in cells if c["level"] == level])
    for difficulty in difficulties:
        pool(f"eval/{difficulty}", [c for c in cells if c["difficulty"] == difficulty])

    ckpt = Path(checkpoint)
    num_envs, num_episodes = cells[0].get("num_envs"), cells[0].get("num_episodes")
    seeds = sorted({c["seed"] for c in cells})
    slug = protocol_slug(seeds, levels, difficulties, num_envs, num_episodes)
    videos = {}
    for c in cells:
        video = ckpt.parent / "videos" / "play" / f"{name}__{c['cell']}-step-0.mp4"
        if video.is_file():
            videos[f"video/{c['cell']}"] = str(video)
    dirty = any(c.get("dirty") for c in cells)
    tags = identity_tags(variant, seed, task) + [
        check_tag(f"ckpt:{ckpt.stem}"),
        check_tag(f"protocol:{protocol['protocol_id'] if protocol else slug}"),
    ]
    config = {
        "research": {
            "comparison": group,
            "variant": variant,
            "seed": seed,
            "task": task,
            "eval_task": eval_task,
            "checkpoint": _relative(ckpt),
            "train_run_id": None,  # resolved from W&B when publishing
            "commits": sorted({str(c.get("commit")) for c in cells}),
            "dirty": dirty,
            "protocol": {
                "slug": slug,
                "seeds": seeds,
                "levels": list(levels),
                "difficulties": list(difficulties),
                "num_envs": num_envs,
                "num_episodes": num_episodes,
                "population": population,
            },  # fmt: skip
            "output_root": _relative(output_root),
        }
    }
    # The scored episodes themselves, so custom report plots can be rebuilt from W&B alone.
    episodes = []
    for c in cells:
        for row in scored(c["cell"]):
            record = {**row, "cell": c["cell"]}
            if protocol and protocol["evaluation"]["paired_factors"]:
                record["eval_unit_id"] = paired_unit_id(
                    row,
                    {"seed": c["seed"], "level": c["level"], "difficulty": c["difficulty"]},
                    protocol["evaluation"]["paired_factors"],
                )
            episodes.append(record)
    if protocol:
        config["research"].update(
            hypothesis_id=protocol["hypothesis_id"],
            protocol_id=protocol["protocol_id"],
            protocol_version=protocol["protocol_version"],
            protocol_contract=protocol,
            cell_id=run_name(variant, seed),
            attempt=int(re.search(r"-attempt(\d+)$", name).group(1)) if re.search(r"-attempt(\d+)$", name) else 1,
            runtime_fingerprints=[c.get("runtime_fingerprint") for c in cells],
            checkpoint_sha256=digest,
        )
        config["research"]["protocol"]["episode_sampling"] = protocol["evaluation"]["episode_sampling"]
        tags += [
            f"attempt:{config['research']['attempt']}",
            f"cell:{run_name(variant, seed)}",
            f"hypothesis:{protocol['hypothesis_id']}",
            f"protocol_version:{protocol['protocol_version']}",
        ]
    return {
        "name": f"{name}__eval", "job_type": "eval", "group": group, "tags": tags + (["dirty"] if dirty else []),
        "config": config, "summary": summary, "table": table, "videos": videos, "episodes": episodes,
    }  # fmt: skip


def _table_value(value):
    """W&B tables hold scalars; anything structured is stored as JSON text."""
    if value is None or isinstance(value, str | bool | int | float):
        return value
    if hasattr(value, "item") and getattr(value, "ndim", 1) == 0:
        return value.item()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return json.dumps(value, default=str)


def episode_table(wandb, episodes: Sequence[dict]):
    columns = list(dict.fromkeys(k for row in episodes for k in row))
    return wandb.Table(columns=columns, data=[[_table_value(row.get(k)) for k in columns] for row in episodes])


def match_runs(runs: Iterable, display_name: str, job_types: Iterable[str]) -> list:
    """Runs called ``display_name`` (or rsl_rl's ``<timestamp>_<display_name>``) with these job types."""
    types = set(job_types)
    return [r for r in runs if r.job_type in types and (r.name == display_name or r.name.endswith(f"_{display_name}"))]


def _wandb_runs(api, project: str, group: str) -> list:
    entity = os.environ.get("WANDB_USERNAME")
    return list(api.runs(f"{entity}/{project}" if entity else project, filters={"group": group}))


def _previous(runs: Iterable, display_name: str, job_type: str, supersede: bool) -> list:
    previous = [r for r in match_runs(runs, display_name, {job_type}) if "status:superseded" not in r.tags]
    if previous and not supersede:
        raise SystemExit(
            f"{display_name} is already published ({', '.join(r.id for r in previous)}); pass --supersede to tag "
            "the old run status:superseded and publish a new one"
        )
    return previous


def _mark_superseded(previous: Iterable) -> None:
    for run in previous:
        run.tags = sorted(set(run.tags) | {"status:superseded"})
        run.update()


def publish_eval(payload: dict, project: str, *, train_run_id: str | None = None, supersede: bool = False,
                 legacy_project: bool = False, wandb_module=None) -> str:  # fmt: skip
    project = check_project(project, legacy=legacy_project)
    pub_meta = payload["config"]["research"]
    if pub_meta.get("protocol_id"):
        contract = pub_meta.get("protocol_contract") or {}
        if (
            project != contract.get("project")
            or payload["group"] != contract.get("group")
            or content_id("protocol", contract) != pub_meta["protocol_id"]
        ):
            raise SystemExit("publication protocol/project/group mismatch")
    wandb = wandb_module or __import__("wandb")
    runs = _wandb_runs(wandb.Api(), check_project(project, legacy=legacy_project), payload["group"])
    if train_run_id is None:
        training = match_runs(runs, payload["name"].removesuffix("__eval"), {"train"})
        if len(training) != 1:
            raise SystemExit(f"found {len(training)} training runs for {payload['name']}; pass --train-run-id")
        train_run_id = training[0].id
    linked = [r for r in runs if r.id == train_run_id and r.job_type == "train" and "status:smoke" not in r.tags]
    if len(linked) != 1:
        raise SystemExit("scored evaluation must reference a scientific training run, never smoke")
    training_meta = runner_config(linked[0].config).get("research") or linked[0].config.get("research") or {}
    pub_meta = payload["config"]["research"]
    if training_meta.get("protocol_id") != pub_meta.get("protocol_id"):
        raise SystemExit("training/evaluation protocol identity mismatch")
    if not match_runs(linked, payload["name"].removesuffix("__eval"), {"train"}):
        raise SystemExit("training/evaluation run name (variant, seed, attempt) mismatch")
    if pub_meta.get("protocol_id"):
        if getattr(linked[0], "state", None) != "finished" or {"status:invalid", "status:failed"} & set(linked[0].tags):
            raise SystemExit("scored evaluation requires a finished valid training attempt")
        for key in (
            "variant",
            "seed",
            "cell_id",
            "attempt",
            "task",
            "hypothesis_id",
            "protocol_id",
            "protocol_version",
        ):
            if training_meta.get(key) != pub_meta.get(key) or pub_meta.get(key) is None:
                raise SystemExit(f"training/evaluation provenance mismatch: {key}")
        commit = training_meta.get("commit") or (training_meta.get("runtime_fingerprint") or {}).get("commit")
        if not commit or pub_meta.get("commits") != [commit] or training_meta.get("dirty") is not False:
            raise SystemExit("training/evaluation clean source commit mismatch or missing provenance")
        checkpoint = Path(pub_meta["checkpoint"]).resolve()
        recorded_dir = training_meta.get("run_dir")
        if not recorded_dir or checkpoint.parent != Path(recorded_dir).resolve():
            raise SystemExit("checkpoint does not belong to linked training run directory")
        sidecar = checkpoint.parent / "research_provenance.json"
        provenance = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
        for key in ("cell_id", "attempt", "protocol_id", "hypothesis_id"):
            if provenance.get(key) != training_meta.get(key):
                raise SystemExit(f"checkpoint training provenance mismatch: {key}")
        if (provenance.get("runtime_fingerprint") or {}).get("commit") != commit:
            raise SystemExit("checkpoint training commit mismatch")
        if not pub_meta.get("checkpoint_sha256") or checkpoint_digest(checkpoint) != pub_meta["checkpoint_sha256"]:
            raise SystemExit("evaluated checkpoint content changed or is unverified")
    payload["config"]["research"]["train_run_id"] = train_run_id
    previous = _previous(runs, payload["name"], "eval", supersede)
    run = wandb.init(
        project=project, entity=os.environ.get("WANDB_USERNAME"), group=payload["group"],
        job_type=payload["job_type"], name=payload["name"], tags=payload["tags"], config=payload["config"],
        notes=f"Evaluation of training run {train_run_id}; cells under {payload['config']['research']['output_root']}",
    )  # fmt: skip
    columns = list(dict.fromkeys(k for row in payload["table"] for k in row))
    log = {"eval/cells": wandb.Table(columns=columns, data=[[row.get(k) for k in columns] for row in payload["table"]])}
    log.update({k: wandb.Video(v, format="mp4", caption="scored") for k, v in payload["videos"].items()})
    run.log(log)
    artifact = wandb.Artifact(f"eval-episodes-{payload['name'].removesuffix('__eval')}", type="eval-episodes")
    artifact.add(episode_table(wandb, payload["episodes"]), "episodes")
    run.log_artifact(artifact)
    run.summary.update(payload["summary"])
    url = run.url
    run.finish()
    _mark_superseded(previous)
    return url


# Custom plots: code that reads W&B, never local files, so a report can be rebuilt from W&B alone.
FORBIDDEN_CALLS = {
    "open", "read_csv", "read_parquet", "read_json", "read_pickle", "read_table", "read_feather", "read_hdf",
    "read_excel", "load", "loadtxt", "genfromtxt", "fromfile", "listdir", "scandir", "walk", "glob", "rglob",
    "iterdir", "read_text", "read_bytes", "download", "system", "popen", "Popen",
}  # fmt: skip
FORBIDDEN_IMPORTS = {"mbrl", "isaaclab", "glob", "shutil", "subprocess", "h5py", "sqlite3"}
LOCAL_MARKERS = (
    "logs/",
    "reports/",
    "outputs/",
    "data/",
    ".parquet",
    ".jsonl",
    ".csv",
    ".h5",
    ".hdf5",
    ".pt",
)


def plot_source_problems(source: str, *, entry_point: bool = True) -> list[str]:
    """Static guard for analysis code: W&B API reads only.

    A report plot script must also define the ``make_figures`` entry point; notebook cells (``entry_point=False``)
    are free-form.
    """
    import ast

    tree = ast.parse(source)
    problems = []
    if entry_point and not any(isinstance(n, ast.FunctionDef) and n.name == "make_figures" for n in tree.body):
        problems.append("define make_figures(api, project, group, entity) -> {name: figure}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
            if name in FORBIDDEN_CALLS:
                problems.append(f"line {node.lineno}: {name}() reads or runs local resources; use the W&B API")
        elif isinstance(node, ast.Import | ast.ImportFrom):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            problems += [f"line {node.lineno}: import {m} is not allowed" for m in modules
                         if m.split(".")[0] in FORBIDDEN_IMPORTS]  # fmt: skip
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if any(marker in node.value for marker in LOCAL_MARKERS):
                problems.append(f"line {node.lineno}: {node.value!r} looks like a local data path")
    return problems


def run_plot_script(script: Path, api, *, project: str, group: str, entity: str | None) -> dict:
    """Run ``make_figures`` from a copy of the script inside an empty directory."""
    import importlib.util
    import tempfile

    source = script.read_text()
    problems = plot_source_problems(source)
    if problems:
        raise SystemExit("plot script rejected:\n" + "\n".join(problems))
    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="report-plots-") as tmp:
        copy = Path(tmp) / "plots.py"
        copy.write_text(source)
        os.chdir(tmp)
        try:
            spec = importlib.util.spec_from_file_location("report_plots", copy)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            figures = module.make_figures(api=api, project=project, group=group, entity=entity)
        finally:
            os.chdir(previous)
    if not isinstance(figures, dict) or not all(
        isinstance(k, str) and NAME_RE.fullmatch(k) and (hasattr(v, "savefig") or hasattr(v, "to_plotly_json"))
        for k, v in figures.items()
    ):
        raise SystemExit("make_figures must return {slug-name: matplotlib or plotly figure}")
    return figures


# display settings; every other magic can reach the shell or disk
ALLOWED_MAGICS = ("%matplotlib", "%config")
ANALYSIS_DRAFT = re.compile(r"<<<ANALYSIS_DRAFT\n(.*?)\nANALYSIS_DRAFT>>>", re.DOTALL)


def notebook_source_problems(nb, *, group: str, project: str) -> list[str]:
    """Guard for an analysis notebook: the plot-script rules on every code cell, plus a fixed target.

    The ``GROUP`` and ``PROJECT`` constants must match the command line, so a notebook copied from another
    comparison cannot quietly analyse (and log into) the wrong one.
    """
    import ast

    problems: list[str] = []
    constants: dict[str, set] = {"GROUP": set(), "PROJECT": set()}
    cells = [c for c in nb.cells if c.cell_type == "code"]
    if not cells:
        problems.append("the notebook has no code cells")
    for number, cell in enumerate(cells, 1):
        lines = []
        for line in cell.source.splitlines():
            stripped = line.lstrip()
            if stripped.startswith(("!", "%")):
                if not stripped.startswith(ALLOWED_MAGICS):
                    problems.append(f"code cell {number}: {stripped.split()[0]!r} runs outside the W&B API")
                line = line[: len(line) - len(stripped)] + "pass"
            lines.append(line)
        try:
            found = plot_source_problems("\n".join(lines), entry_point=False)
            tree = ast.parse("\n".join(lines))
        except SyntaxError as exc:
            problems.append(f"code cell {number}: syntax error, line {exc.lineno}: {exc.msg}")
            continue
        problems += [f"code cell {number}, {p}" for p in found]
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in constants:
                        constants[target.id].add(node.value.value)
    for name, expected in (("GROUP", group), ("PROJECT", project)):
        if constants[name] != {expected}:
            problems.append(f"{name} must be assigned exactly {expected!r} (found {sorted(map(str, constants[name]))})")
    return problems


def execute_notebook(nb, *, env: dict[str, str], timeout: int):
    """Run every cell in a fresh kernel started in an empty directory; ``env`` is visible to the kernel only."""
    import tempfile

    from nbclient import NotebookClient

    saved = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    try:
        if env.get("RESEARCH_ANALYSIS_LOG") == "0":
            import nbformat

            nb.cells.insert(
                0,
                nbformat.v4.new_code_cell(
                    "from research_workflow.reporting_guard import install_read_only_wandb\ninstall_read_only_wandb()"
                ),
            )
        with tempfile.TemporaryDirectory(prefix="analysis-") as tmp:
            NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": tmp}}).execute()
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return nb


def analysis_draft(nb) -> str | None:
    """The Analysis draft the notebook printed between its ``ANALYSIS_DRAFT`` markers."""
    for cell in nb.cells:
        for output in cell.get("outputs", []):
            match = (
                ANALYSIS_DRAFT.search(str(output.get("text", ""))) if output.get("output_type") == "stream" else None
            )
            if match:
                return match.group(1).strip()
    return None


def notebook_evidence(nb) -> dict | None:
    for cell in nb.cells:
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                match = re.search(
                    r"<<<SCIENTIFIC_EVIDENCE\n(.*?)\nSCIENTIFIC_EVIDENCE>>>",
                    str(output.get("text", "")),
                    re.DOTALL,
                )
                if match:
                    return validate_evidence(json.loads(match.group(1)))
    return None


def analysis_code_fingerprint(notebook: Path) -> dict:
    """Hash the actual notebook and trusted helper bytes, including uncommitted edits."""
    import hashlib

    source_root = Path(__file__).resolve().parents[1]
    sources = {"notebook": notebook, "research_runs": Path(__file__)}
    sources.update({p.stem: p for p in (source_root / "src" / "research_workflow").glob("*.py")})
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sorted(sources.items())}


def run_analysis(
    notebook: Path, project: str, group: str, *, draft_out: Path | None = None, dry_run: bool = False,
    timeout: int = 1800, legacy_project: bool = False, wandb_module=None, executor: Callable | None = None,
) -> dict:  # fmt: skip
    """Check, execute and publish a comparison's analysis notebook as an ``analysis`` run.

    The notebook logs its own run (figures and tables under ``analysis/fig/*`` and ``analysis/table/*``) only
    when this function runs it, with the source cells guarded and the kernel in an empty directory. Afterwards
    the executed notebook, outputs included, is attached to that run as the ``analysis-notebook-<group>`` code
    artifact and the Analysis draft is written for ``publish-report --analysis``. ``dry_run`` executes with
    logging off and writes nothing.
    """
    import tempfile
    import uuid

    import nbformat

    nb = nbformat.read(notebook, as_version=4)
    problems = notebook_source_problems(nb, group=group, project=project)
    if problems:
        raise SystemExit("analysis notebook rejected:\n" + "\n".join(problems))
    project = check_project(project, legacy=legacy_project)
    wandb = wandb_module or __import__("wandb")
    api = wandb.Api()
    entity = os.environ.get("WANDB_USERNAME") or getattr(api, "default_entity", None)
    previous = [r for r in match_runs(_wandb_runs(api, project, group), f"{group}__analysis", {"analysis"})
                if "status:superseded" not in r.tags]  # fmt: skip
    run_id = uuid.uuid4().hex[:8]
    source = git_source()
    env = {
        "RESEARCH_ANALYSIS_LOG": "0" if dry_run else "1",
        "RESEARCH_SOURCE": json.dumps(source),
        "RESEARCH_HYPOTHESIS": "null",
        "RESEARCH_ANALYSIS_CODE": json.dumps(analysis_code_fingerprint(notebook)),
        **({"WANDB_MODE": "disabled", "WANDB_RUN_ID": ""} if dry_run else {}),
        "PYTHONPATH": source_env()["PYTHONPATH"],
    }
    hypothesis_path = ROOT / "research" / group / "hypothesis.yaml"
    if hypothesis_path.exists():
        env["RESEARCH_HYPOTHESIS"] = json.dumps(read_contract(hypothesis_path, "hypothesis"))
    if not dry_run:
        env["WANDB_RUN_ID"] = run_id
    try:
        executed = (executor or execute_notebook)(nb, env=env, timeout=timeout)
    except Exception as exc:  # CellExecutionError carries the failing cell and traceback
        raise SystemExit(f"analysis notebook failed; nothing was superseded or attached:\n{exc}") from exc
    draft = analysis_draft(executed)
    if draft is None:
        raise SystemExit("the notebook printed no Analysis draft (the template's draft cell prints it)")
    evidence = notebook_evidence(executed)
    if evidence and (evidence["project"] != project or evidence["group"] != group):
        raise SystemExit("notebook scientific evidence belongs to another comparison")
    result = {
        "evidence": evidence,
        "evidence_path": None,
        "run_id": None if dry_run else run_id,
        "draft": draft,
        "draft_path": None,
        "dirty": source["dirty"],
    }
    if dry_run:
        return result
    check = api.run(f"{entity}/{project}/{run_id}" if entity else f"{project}/{run_id}")
    if check.job_type != "analysis" or check.group != group:
        raise SystemExit("the notebook did not log an `analysis` run in this group; keep its final cell")
    with tempfile.TemporaryDirectory(prefix="analysis-attach-") as tmp:
        attached = Path(tmp) / "analysis.ipynb"
        nbformat.write(executed, attached)
        resumed = wandb.init(project=project, entity=entity, id=run_id, resume="must")
        artifact = wandb.Artifact(f"analysis-notebook-{group}", type="code")
        artifact.add_file(str(attached), name="analysis.ipynb")
        resumed.log_artifact(artifact)
        if evidence:
            resumed.summary["analysis/scientific_evidence"] = evidence
            evidence_artifact = wandb.Artifact(f"scientific-evidence-{group}", type="scientific-evidence")
            with evidence_artifact.new_file("evidence.json", mode="w") as stream:
                json.dump(evidence, stream, indent=2, allow_nan=False)
            resumed.log_artifact(evidence_artifact)
        resumed.finish()
    _mark_superseded(previous)
    draft_out = draft_out or ROOT / "logs" / "queue" / group / "analysis_draft.md"
    draft_out.parent.mkdir(parents=True, exist_ok=True)
    draft_out.write_text(draft + "\n")
    result["draft_path"] = str(draft_out)
    if evidence:
        evidence_out = draft_out.with_name("scientific_evidence.json")
        evidence_out.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
        result["evidence_path"] = str(evidence_out)
    return result


def to_media(wandb, figure):
    if hasattr(figure, "savefig"):
        return wandb.Image(figure)
    if hasattr(figure, "to_plotly_json"):
        return wandb.Plotly(figure)
    raise SystemExit(f"unsupported figure type {type(figure).__name__}; return matplotlib or plotly figures")


def _heading(block) -> str | None:
    if type(block).__name__ in {"H1", "H2", "H3"}:
        text = getattr(block, "text", "")
        return "".join(str(getattr(t, "text", t)) for t in text) if isinstance(text, list) else str(text)
    if getattr(block, "type", None) == "heading":
        return "".join(
            c.get("text", "") if isinstance(c, dict) else str(getattr(c, "text", ""))
            for c in getattr(block, "children", [])
        )
    return None


def human_sections(blocks: Iterable) -> dict[str, list]:
    """Preserve nested headings, collapsible anchors and unknown legacy block types."""
    kept: dict[str, list] = {}
    current = None
    anchor_level = 1
    boundaries = set(SECTIONS) | {"Overview", "Training", "Evaluation", "Provenance"}
    for block in blocks:
        heading = _heading(block)
        name = type(block).__name__
        level = int(name[1]) if name in {"H1", "H2", "H3"} else getattr(block, "level", None)
        if heading in HUMAN_SECTIONS:
            current = heading
            anchor_level = level or 1
            kept.setdefault(current, [])
            if (
                getattr(block, "collapsed_blocks", None) is not None
                or getattr(block, "collapsedChildren", None) is not None
            ):
                kept[current].append(block)
        elif heading in boundaries and (
            level == 1
            or (heading == "Provenance" and current == "Decision" and level is not None and level <= anchor_level)
        ):
            current = None
        elif current:
            kept[current].append(block)
    return kept


def report_inventory(runs: Iterable, *, resolver=None) -> dict:
    """What a comparison's report draws on: current train and eval runs, their keys and provenance."""
    runs = list(runs)
    resolution = (resolver or MetricResolver(runs)).resolve()
    current = [r for r in runs if "status:superseded" not in r.tags and "status:smoke" not in r.tags]
    train = sorted((r for r in current if r.job_type == "train"), key=lambda r: r.name)
    evals = sorted((r for r in current if r.job_type == "eval"), key=lambda r: r.name)
    eval_keys = sorted({k for r in evals for k, v in r.summary.items()
                        if k.startswith("eval/") and isinstance(v, int | float) and not isinstance(v, bool)})  # fmt: skip
    videos = sorted({k for r in evals for k in r.summary.keys() if k.startswith("video/")})
    analyses = [r for r in current if r.job_type == "analysis" and getattr(r, "state", "finished") == "finished"]
    latest = max(analyses, key=lambda r: str(getattr(r, "created_at", "")), default=None)
    analysis = latest and {
        "id": latest.id,
        "name": latest.name,
        "figures": sorted(k for k in latest.summary.keys() if k.startswith("analysis/fig/")),
        "tables": sorted(k for k in latest.summary.keys() if k.startswith("analysis/table/")),
    }  # fmt: skip
    research = [dict(runner_config(r.config).get("research") or r.config.get("research") or {}) for r in train]
    return {
        "metric_resolution": resolution,
        "train": [r.name for r in train],
        "eval": [r.name for r in evals],
        "eval_keys": eval_keys,
        "videos": videos,
        "analysis": analysis or None,
        "hypothesis_ids": sorted({x["hypothesis_id"] for x in research if x.get("hypothesis_id")}),
        "protocol_ids": sorted({x["protocol_id"] for x in research if x.get("protocol_id")}),
        "runtime_mismatches": runtime_mismatches([x.get("runtime_fingerprint") for x in research]),
        "legacy_runs": [r.name for r, x in zip(train, research, strict=True) if not x.get("protocol_id")],
        "launch_tags": sorted({x["launch_tag"] for x in research if x.get("launch_tag")}),
        "commits": sorted({x["commit"] for x in research if x.get("commit")}),
        "branches": sorted({x["branch"] for x in research if x.get("branch")}),
        "eval_commits": sorted({c for r in evals for c in (r.config.get("research") or {}).get("commits", [])}),
    }


def report_metric_title(metric: str, context: str = "") -> str:
    """Label known objectives without assigning a preference to diagnostic metrics."""
    spec = CATALOG.get(metric, CATALOG.get("eval/" + metric, descriptor(metric)))
    return metric_title({**spec, "name": metric + context})


def build_report_blocks(
    wr, *, entity: str, project: str, group: str, inventory: dict, intro: str, train_metrics: Sequence[str],
    eval_metrics: Sequence[str], report_run_id: str | None, custom_keys: Sequence[str], preserved: dict,
    analysis: str | None = None, decision: str | None = None, plot_artifact: str | None = None,
    train_groupby: Sequence[str] | None = None,
    dashboard: dict | None = None,
    evidence: dict | None = None, review: dict | None = None, review_approval: dict | None = None,
    evidence_only: str | None = None,
) -> list:  # fmt: skip
    """The report: generated sections from W&B data, plus the researcher-owned Analysis/Decision.

    ``train_groupby`` is the W&B config key(s) that identify a training run's variant. Runs launched
    by ``train-job`` carry ``runner_cfg.research.variant``; older runs need a key their configs share.
    """

    def runset(name: str, names: Sequence[str], groupby: Sequence[str] = ()):
        quoted = ", ".join(repr(n) for n in names)
        return wr.Runset(entity=entity, project=project, name=name, filters=f"Metric('Name') in [{quoted}]",
                         groupby=list(groupby))  # fmt: skip

    def panel_grid(*, runsets, panels):
        # W&B report grids span 24 units: two panels of width 12 per row.
        for index, panel in enumerate(panels):
            panel.layout = wr.Layout(x=(index % 2) * 12, y=(index // 2) * 6, w=12, h=6)
        return wr.PanelGrid(runsets=runsets, panels=panels)

    train_runs = runset("Training runs", inventory["train"])
    eval_runs = runset("Evaluation runs", inventory["eval"])
    plan = inventory.get("metric_resolution", {"panels": [], "metrics": [], "errors": [], "warnings": []})

    def metric_grids(scope):
        grids = []
        for panel in plan["panels"]:
            if panel["scope"] != scope or not panel["run_ids"]:
                continue
            # Run IDs are immutable; names can be shared by distinct attempts.
            ids = ", ".join(repr(i) for i in panel["run_ids"])
            selected = wr.Runset(
                entity=entity,
                project=project,
                name=f"{panel['name']} · {panel['role']} · {panel['cohort'][:10]}",
                filters=f"Metric('ID') in [{ids}]",
                groupby=panel["groupby"],
            )
            options = (
                {"groupby_aggfunc": "mean", "groupby_rangefunc": "stderr"}
                if panel["aggregation"] == "mean_seed"
                else {}
            )
            title = panel["title"] + (" · failed/incomplete attempt" if panel["role"] == "diagnostic_attempt" else "")
            if scope == "train":
                plot = wr.LinePlot(
                    title=title,
                    x="Step" if panel["axis"] == "_step" else panel["axis"],
                    y=[panel["name"]],
                    **options,
                    **LINE_PLOT_STYLE,
                )
            else:
                plot = wr.BarPlot(title=title, metrics=[panel["name"]], **options)
            grids.append(panel_grid(runsets=[selected], panels=[plot]))
        return grids

    interpretation = render_review(review, evidence, review_approval)
    blocks = [wr.H1(text=SECTIONS[0]), wr.MarkdownBlock(text=interpretation["executive"])]
    if evidence_only:
        blocks.append(wr.MarkdownBlock(text="Explicit evidence-only fallback: " + evidence_only))
    if "Progress" in preserved:
        blocks += [wr.H2(text="Progress"), *preserved["Progress"]]
    blocks += [
        wr.H1(text=SECTIONS[1]),
        wr.MarkdownBlock(text=intro.strip()),
        wr.MarkdownBlock(text=evidence_markdown(evidence, "hypothesis")),
        wr.H1(text=SECTIONS[2]),
        wr.MarkdownBlock(text=evidence_markdown(evidence, "design")),
    ]
    blocks += [
        wr.H2(text="Researcher dashboard"),
        wr.MarkdownBlock(
            text=(
                dashboard_markdown(dashboard)
                if dashboard
                else "Attended mode (default). Historical approvals remain in the researcher's conversation."
            )
        ),
    ]
    blocks.append(wr.MarkdownBlock(text=coverage_markdown(plan)))
    if plan["errors"] or plan["warnings"]:
        blocks.append(wr.MarkdownBlock(text="Preflight: " + "\n".join(plan["errors"] + plan["warnings"])))
    if inventory["train"]:
        blocks.append(panel_grid(runsets=[train_runs], panels=[wr.RunComparer(diff_only="split")]))
    training_start = len(blocks)
    blocks += [
        wr.H1(text=SECTIONS[4]),
        wr.MarkdownBlock(text=evidence_markdown(evidence, "diagnostics")),
        wr.MarkdownBlock(
            text=(
                "Only observed numeric metrics are plotted. Compatible successful training seeds may be grouped; all"
                " other curves are per-run diagnostics. Smoothing affects display only."
            )
        ),
    ]
    blocks.extend(metric_grids("train"))
    training_blocks = blocks[training_start:]
    del blocks[training_start:]
    blocks.append(wr.H1(text=SECTIONS[3]))
    blocks.append(wr.MarkdownBlock(text=evidence_markdown(evidence, "statistics")))
    if inventory["eval"]:
        blocks.extend(metric_grids("eval"))
    else:
        blocks.append(wr.MarkdownBlock(text="_No published evaluation yet._"))
    blocks.append(wr.H2(text="Videos"))
    if inventory["videos"]:
        blocks.append(wr.MarkdownBlock(text="Scored videos, recorded by the evaluation that produced the numbers."))
        blocks.append(panel_grid(runsets=[eval_runs], panels=[wr.MediaBrowser(title="Scored videos",
                                                                              media_keys=inventory["videos"])]))  # fmt: skip
    else:
        blocks.append(wr.MarkdownBlock(text="_No scored videos._"))
    if custom_keys and report_run_id:
        report_run = wr.Runset(
            entity=entity, project=project, name="Report run", filters=f"Metric('ID') == '{report_run_id}'"
        )
        blocks += [wr.H2(text="Custom plots"), wr.MarkdownBlock(text=(
            f"Built from the W&B API only by the attached code artifact `{plot_artifact}`."))]  # fmt: skip
        blocks.append(panel_grid(runsets=[report_run], panels=[wr.MediaBrowser(media_keys=list(custom_keys))]))
    kept = inventory.get("analysis")
    if kept and (kept["figures"] or kept["tables"]):
        analysis_run = wr.Runset(
            entity=entity, project=project, name="Analysis run", filters=f"Metric('ID') == '{kept['id']}'"
        )
        blocks += [wr.H2(text="Analysis figures"), wr.MarkdownBlock(text=(
            f"Figures and tables kept by the analysis notebook run `{kept['name']}`. These retained outputs are diagnostic until their compatibility is verified in current scientific evidence. Each was computed from the W&B "
            f"API by the executed notebook attached to that run as the code artifact `analysis-notebook-{group}`."
        ))]  # fmt: skip
        if kept["figures"]:
            blocks.append(panel_grid(runsets=[analysis_run], panels=[wr.MediaBrowser(media_keys=kept["figures"])]))
        if kept["tables"]:
            blocks.append(panel_grid(runsets=[analysis_run],
                                     panels=[wr.WeavePanelSummaryTable(table_name=t) for t in kept["tables"]]))  # fmt: skip
    blocks.extend(training_blocks)
    blocks += [
        wr.H1(text=SECTIONS[5]),
        wr.MarkdownBlock(text=evidence_markdown(evidence, "cost")),
        wr.H1(text=SECTIONS[6]),
        wr.MarkdownBlock(text=interpretation["interpretation"]),
        wr.MarkdownBlock(text=evidence_markdown(evidence, "limitations")),
    ]
    blocks.append(wr.H2(text="Analysis"))
    blocks += preserved.get("Analysis", [wr.MarkdownBlock(text=(analysis or "_Analysis pending._").strip())])
    blocks += [
        wr.H1(text=SECTIONS[7]),
        wr.MarkdownBlock(text=interpretation["next"]),
        wr.H1(text=SECTIONS[8]),
        wr.H2(text="Decision"),
    ]
    if decision:
        blocks.append(wr.MarkdownBlock(text=decision.strip()))
    else:
        blocks += preserved.get("Decision", [wr.MarkdownBlock(text="_Pending the researcher's decision._")])
    lines = [
        f"- Hypothesis IDs: {inventory.get('hypothesis_ids', [])}; protocol IDs: {inventory.get('protocol_ids', [])}",
        (
            f"- Runtime fingerprint mismatches: {inventory.get('runtime_mismatches', [])}; legacy workflow:"
            f" {inventory.get('legacy_runs', [])}"
        ),
        f"- Comparison `{group}`; Git branch(es) {', '.join(f'`{b}`' for b in inventory['branches']) or '—'}",
        f"- Launch tags: {', '.join(f'`{t}`' for t in inventory['launch_tags']) or '—'}",
        (
            f"- Training commits: {', '.join(f'`{c[:12]}`' for c in inventory['commits']) or '—'}; "
            f"evaluation commits: {', '.join(f'`{c[:12]}`' for c in inventory['eval_commits']) or '—'}"
        ),
        (
            f"- {len(inventory['train'])} training runs, {len(inventory['eval'])} evaluation runs (superseded runs"
            " excluded)"
        ),
        (
            f"- Generated by `scripts/research_runs.py publish-report`; custom plot code: `{plot_artifact or 'none'}`; "
            f"analysis notebook run: `{kept['name'] if kept else 'none'}`"
        ),
    ]
    blocks += [wr.H2(text="Provenance"), wr.MarkdownBlock(text="\n".join(lines))]
    return [block for block in blocks if type(block).__name__ != "PanelGrid" or block.panels]


def publish_report(  # noqa: C901
    project: str, group: str, *, title: str | None = None, intro: str = "", analysis: str | None = None,
    decision: str | None = None, plots: Path | None = None, train_metrics: Sequence[str] | None = None,
    eval_metrics: Sequence[str] | None = None, dry_run: bool = False,
    train_groupby: Sequence[str] | None = None, legacy_project: bool = False,
    wandb_module=None, wr_module=None, review: dict | None = None,
    review_approval: dict | None = None, evidence_only: str | None = None,
) -> str:  # fmt: skip
    """Build or refresh the comparison's W&B Report; its ``report`` run holds custom plots and their code."""
    wandb = wandb_module or __import__("wandb")
    if wr_module is None:
        import wandb_workspaces.reports.v2 as wr_module
    wr = wr_module
    api = wandb.Api()
    entity = os.environ.get("WANDB_USERNAME") or getattr(api, "default_entity", None)
    runs = _wandb_runs(api, check_project(project, legacy=legacy_project), group)
    resolver = MetricResolver(runs, train_metrics=train_metrics, eval_metrics=eval_metrics, groupby=train_groupby)
    inventory = report_inventory(runs, resolver=resolver)
    previous = [r for r in match_runs(runs, f"{group}__report", {"report"}) if "status:superseded" not in r.tags]
    old_url = previous[-1].summary.get("report_url") if previous else None
    dashboard = decision_dashboard(ROOT, ROOT / "research" / group)
    if dashboard is None and previous:
        snapshot = previous[-1].config.get("research", {}).get("decisions")
        dashboard = {**snapshot, "snapshot": True} if snapshot else None
    if not inventory["train"] and dashboard is None:
        raise SystemExit(f"no training runs or workflow mandate in group {group}")
    analysis_run = next((r for r in runs if inventory.get("analysis") and r.id == inventory["analysis"]["id"]), None)
    evidence = analysis_run.summary.get("analysis/scientific_evidence") if analysis_run else None
    current_evidence = build_evidence(
        runs, project=project, group=group, entity=entity, histories=False, resolver=resolver
    )
    evidence = validate_evidence(dict(evidence)) if evidence else current_evidence
    if evidence["project"] != project or evidence["group"] != group:
        raise ValueError("analysis evidence belongs to another comparison")
    if evidence["runs"] != current_evidence["runs"]:
        evidence = build_evidence(
            runs,
            project=project,
            group=group,
            entity=entity,
            histories=False,
            resolver=resolver,
            problems=["Analysis snapshot is stale: runs, configurations or summaries changed; rerun run-analysis."],
        )
    elif evidence.get("metric_resolution") != resolver.resolve():
        evidence = build_evidence(
            runs,
            project=project,
            group=group,
            entity=entity,
            resolver=resolver,
            tables=evidence.get("tables"),
            baseline=evidence.get("baseline"),
            histories=False,
            hypothesis=evidence.get("hypothesis"),
            problems=[
                *evidence.get("limitations", []),
                (
                    "Analysis metric resolution is stale; comparisons rechecked against current numeric observations"
                    " and context. Rerun run-analysis."
                ),
            ],
        )
    evidence["metric_resolution"] = resolver.resolve()
    evidence["evidence_id"] = identifier("evidence", evidence)
    hypothesis_path = ROOT / "research" / group / "hypothesis.yaml"
    hypothesis = read_contract(hypothesis_path, "hypothesis") if hypothesis_path.exists() else None
    rendering = {}
    for key, value in (("intro", intro), ("analysis", analysis), ("decision", decision)):
        if value:
            rendering[key] = value
    if plots:
        import hashlib

        rendering["plots_sha256"] = hashlib.sha256(plots.read_bytes()).hexdigest()
    if train_metrics is not None:
        rendering["train_metrics"] = list(train_metrics)
    if eval_metrics is not None:
        rendering["eval_metrics"] = list(eval_metrics)
    if train_groupby is not None:
        rendering["train_groupby"] = list(train_groupby)
    local_notebook = ROOT / "research" / group / "analysis.ipynb"
    code = analysis_code_fingerprint(local_notebook) if local_notebook.exists() else None
    evidence = bind_publication_evidence(evidence, analysis_run, hypothesis, rendering=rendering, analysis_code=code)
    # Never silently carry an approved interpretation over changed computed evidence.
    if review is None and previous and not evidence_only:
        saved_review = previous[-1].config.get("research", {}).get("scientific_review")
        if saved_review and saved_review.get("evidence_id") == evidence["evidence_id"]:
            review = saved_review
            if review_approval is None:
                review_approval = previous[-1].config.get("research", {}).get("review_approval")
    if review is not None:
        review = validate_review(review, evidence)
    if review_approval is not None:
        if review is None:
            raise ValueError("review approval requires a structured review")
        validate_approval(review_approval, review)
    if review_approval and review_approval["actor"] == "researcher":
        if not dashboard or dashboard.get("snapshot") or dashboard["workflow"]["mode"] != "unattended":
            raise ValueError("researcher-agent approval requires explicitly delegated unattended mode")
    if evidence_only is not None and not evidence_only.strip():
        raise ValueError("evidence-only publication requires a nonempty reason")
    if evidence_only and review_approval:
        raise ValueError("choose approved review or evidence-only publication, not both")
    preflight_errors = inventory["metric_resolution"]["errors"]
    publication_allowed = bool(review_approval or evidence_only) and not preflight_errors
    if preflight_errors and not dry_run:
        raise ValueError("report preflight failed: " + "; ".join(preflight_errors))
    if not dry_run and not publication_allowed:
        raise ValueError("publication requires an approved scientific review or explicit --evidence-only reason")
    if decision:
        gate_d = next((row for row in (dashboard or {}).get("rows", []) if row.get("gate") == "D"), None)
        if (
            not gate_d
            or (dashboard or {}).get("snapshot")
            or gate_d.get("stale")
            or gate_d.get("verdict") not in {"keep", "reject", "modify", "extend"}
        ):
            raise ValueError("a new Decision requires a current Gate D receipt; existing researcher text is preserved")
    rendered_review = render_review(review, evidence, review_approval)
    if dry_run and plots:
        problems = plot_source_problems(plots.read_text())
        if problems:
            raise SystemExit("plot script rejected:\n" + "\n".join(problems))
    # Dry-run must not execute arbitrary custom code in the authenticated host process.
    figures = run_plot_script(plots, api, project=project, group=group, entity=entity) if plots and not dry_run else {}
    custom_keys = [f"custom/{k}" for k in figures]
    report = wr.Report.from_url(old_url) if old_url else wr.Report(project=project, entity=entity)
    preserved = human_sections(report.blocks) if old_url else {}
    preview_blocks = build_report_blocks(
        wr,
        entity=entity,
        project=project,
        group=group,
        inventory=inventory,
        intro=intro,
        train_metrics=train_metrics,
        eval_metrics=eval_metrics,
        report_run_id=None,
        custom_keys=[],
        preserved=preserved,
        analysis=analysis,
        decision=decision,
        train_groupby=train_groupby,
        dashboard=dashboard,
        evidence=evidence,
        review=review,
        review_approval=review_approval,
        evidence_only=evidence_only,
    )
    if dry_run:
        print(
            json.dumps(
                {
                    **inventory,
                    "custom": custom_keys,
                    "refresh": old_url,
                    "decisions": dashboard,
                    "scientific_evidence": evidence,
                    "scientific_review": rendered_review,
                    "sections": list(SECTIONS),
                    "evidence_only": evidence_only,
                    "publication_allowed": publication_allowed,
                    "plots_executed": False,
                    "preserved_sections": sorted(preserved),
                    "preview": [getattr(b, "text", "[interactive panels]") for b in preview_blocks],
                },
                indent=2,
            )
        )
        return old_url or ""
    run = wandb.init(project=project, entity=entity, group=group, job_type="report", name=f"{group}__report",
                     config={"research": {"comparison": group, "source": git_source(),
                                          "plots": _relative(plots) if plots else None,
                                          "decisions": dashboard, "scientific_review": review,
                                          "review_approval": review_approval, "evidence_only": evidence_only}})  # fmt: skip
    run.summary["scientific/evidence_id"] = evidence["evidence_id"]
    run.summary["scientific/review_status"] = "approved" if review_approval else "pending"
    run.summary["scientific/evidence_only_reason"] = evidence_only
    if review:
        artifact = wandb.Artifact(f"scientific-review-{group}", type="scientific-review")
        with artifact.new_file("review.json", mode="w") as stream:
            json.dump({"review": review, "approval": review_approval}, stream, indent=2)
        run.log_artifact(artifact)
    if dashboard is not None:
        artifact = wandb.Artifact(f"research-decisions-{group}", type="research-decisions")
        with artifact.new_file("decisions.json", mode="w") as stream:
            json.dump(dashboard, stream, indent=2)
        run.log_artifact(artifact)
    plot_artifact = None
    if figures:
        run.log({key: to_media(wandb, fig) for key, fig in zip(custom_keys, figures.values(), strict=True)})
        code = wandb.Artifact(f"report-code-{group}", type="code")
        code.add_file(str(plots), name="plots.py")
        run.log_artifact(code)
        plot_artifact = f"report-code-{group}"
    report.title = title or group
    report.description = (
        f"Research comparison {group}: generated from W&B runs; Analysis and Decision are researcher-owned."
    )
    report.blocks = build_report_blocks(
        wr, entity=entity, project=project, group=group, inventory=inventory, intro=intro,
        train_metrics=train_metrics, eval_metrics=eval_metrics, report_run_id=run.id, custom_keys=custom_keys,
        preserved=preserved, analysis=analysis, decision=decision, plot_artifact=plot_artifact,
        train_groupby=train_groupby, dashboard=dashboard, evidence=evidence,
        review=review, review_approval=review_approval, evidence_only=evidence_only,
    )  # fmt: skip
    report.save()
    run.summary["report_url"] = report.url
    run.finish()
    _mark_superseded(previous)
    return report.url


def select_runs(runs: Iterable, *, names: Sequence[str] = (), variants: Sequence[str] = (),
                job_types: Sequence[str] = ()) -> list:  # fmt: skip
    """Runs of one group chosen by run name (``<variant>-seed<seed>``, any job type) or variant tag."""
    chosen = []
    for run in runs:
        if job_types and run.job_type not in job_types:
            continue
        base = run.name.removesuffix("__eval")
        by_name = any(base == n or base.endswith(f"_{n}") for n in names)
        by_variant = any(f"variant:{v}" in run.tags for v in variants)
        if (not names and not variants) or by_name or by_variant:
            chosen.append(run)
    return chosen


def retag(runs: Iterable, add: Sequence[str], remove: Sequence[str], *, dry_run: bool = False) -> list[tuple]:
    """Apply lifecycle/decision tags. Identity tags are fixed at launch and are never edited here."""
    for tag in add:
        check_tag(tag)
        if tag.partition(":")[0] not in LATER_TAG_KEYS:
            raise ValueError(f"{tag!r}: only {sorted(LATER_TAG_KEYS)} tags are added after a run")
    for tag in remove:
        if tag.partition(":")[0] not in LATER_TAG_KEYS:
            raise ValueError(f"{tag!r}: only {sorted(LATER_TAG_KEYS)} tags can be removed")
    changes = []
    for run in runs:
        new = sorted((set(run.tags) | set(add)) - set(remove))
        if new != sorted(run.tags):
            changes.append((run.name, sorted(run.tags), new))
            if not dry_run:
                run.tags = new
                run.update()
    return changes


# ---------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------


def _print_json_lines(items: Iterable[dict]) -> None:
    for item in items:
        print(json.dumps(item))


def main(argv: list[str] | None = None) -> int:  # noqa: C901
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "world-model":
        from world_modeling.campaign import main as world_model_campaign

        world_model_campaign(argv[1:])
        return 0
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    vc = sub.add_parser("validate-contract", help="validate a hypothesis, protocol or decision; print canonical ID")
    vc.add_argument("--kind", choices=("hypothesis", "protocol", "decision"), required=True)
    vc.add_argument("--file", type=Path, required=True)
    vc.add_argument("--hypothesis", type=Path)

    vw = sub.add_parser("validate-workflow", help="validate attended/unattended authority and mandate")
    vw.add_argument("--file", type=Path, required=True)
    gs = sub.add_parser("gate-subjects", help="print the exact artifacts for independent gate review")
    gs.add_argument("--group", required=True)
    gs.add_argument("--gate", choices=("A", "B", "C", "D"), required=True)
    rg = sub.add_parser("record-gate", help="record an actual user/researcher decision, not infer approval")
    rg.add_argument("--group", required=True)
    rg.add_argument("--file", type=Path, required=True, help="YAML decision returned by the gate owner")

    t = sub.add_parser("train-job", help="print one training job")
    t.add_argument("--task", required=True)
    t.add_argument("--variant", required=True)
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--iterations", type=int, required=True)
    t.add_argument("--project", required=True, help="the W&B project, e.g. rwm-explore")
    t.add_argument("--group", required=True, help="the comparison slug")
    t.add_argument("--smoke", action="store_true", help="short test run: job_type=smoke")
    t.add_argument("--num-envs", type=int)
    t.add_argument("--note")
    t.add_argument("--gpu", type=int, help="pin to one GPU (rarely needed)")
    t.add_argument("--python", help="interpreter (default: the one running this script)")
    t.add_argument("extra", nargs="*", help="after '--': Hydra overrides / extra flags, passed verbatim")

    e = sub.add_parser("eval-jobs", help="print one evaluation job per cell")
    e.add_argument("--task", required=True, help="the registered *eval* task")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--name", required=True, help="the training run name, <variant>-seed<seed>")
    e.add_argument("--seeds", type=int, nargs="+", required=True)
    e.add_argument("--levels", type=int, nargs="*", default=[])
    e.add_argument("--difficulties", nargs="*", default=[])
    e.add_argument("--output-root", required=True, help="cells are written to <output-root>/<cell>")
    e.add_argument("--num-envs", type=int)
    e.add_argument("--num-episodes", type=int)
    e.add_argument("--python")
    e.add_argument("extra", nargs="*", help="after '--': extra play.py flags / overrides, passed verbatim")

    q = sub.add_parser("queue", help="run jobs on a GPU pool")
    q.add_argument("--jobs", type=Path, required=True)
    q.add_argument("--gpus", type=int, nargs="+", required=True)
    q.add_argument("--log-dir", type=Path, required=True)
    q.add_argument("--group", required=True, help="the comparison; launches only from dev/agent/<group>")
    q.add_argument("--allow-shared-gpu", action="store_true", help="start jobs on GPUs other processes use")
    q.add_argument("--no-push", action="store_true", help="tag the launch locally only (no network)")
    q.add_argument("--dry-run", action="store_true", help="print the first wave and launch checks, then exit")
    q.add_argument("--poll", type=float, default=10.0)

    s = sub.add_parser("status", help="show launched jobs")
    s.add_argument("--log-dir", type=Path, required=True)

    k = sub.add_parser("stop", help="stop launched jobs whose identity still matches")
    k.add_argument("--log-dir", type=Path, required=True)
    k.add_argument("--name", nargs="*", help="default: every live job")

    c = sub.add_parser("check-train", help="verify a finished training run")
    where = c.add_mutually_exclusive_group(required=True)
    where.add_argument("--run-dir", type=Path)
    where.add_argument("--log-root", type=Path, help="logs/rsl_rl/<experiment>; with --name")
    c.add_argument("--name", help="run name <variant>-seed<seed>")
    c.add_argument("--iterations", type=int, required=True)
    c.add_argument("--log", type=Path, help="the queue log of this run")

    v = sub.add_parser("check-eval", help="verify evaluation cells and their comparability")
    v.add_argument("--output-root", type=Path, nargs="+", required=True, help="one per compared run")
    v.add_argument("--checkpoint", nargs="*", default=[], help="one per --output-root")
    v.add_argument("--seeds", type=int, nargs="+", required=True)
    v.add_argument("--levels", type=int, nargs="*", default=[])
    v.add_argument("--difficulties", nargs="*", default=[])
    v.add_argument("--expected-episodes", type=int, help="exact row count (only for exact-count protocols)")
    v.add_argument("--expected-first-episodes", type=int, help="rows with episode_id 0; normally num_envs")
    v.add_argument("--metrics", nargs="*", default=[])

    pe = sub.add_parser("publish-eval", help="log one training run's checked evaluation as one W&B eval run")
    pe.add_argument("--project", required=True)
    pe.add_argument("--group", required=True)
    pe.add_argument("--name", required=True, help="the training run name, <variant>-seed<seed>")
    pe.add_argument("--task", required=True, help="the registered training task")
    pe.add_argument("--eval-task", required=True)
    pe.add_argument("--checkpoint", required=True)
    pe.add_argument("--output-root", type=Path, required=True)
    pe.add_argument("--seeds", type=int, nargs="+", required=True)
    pe.add_argument("--levels", type=int, nargs="*", default=[])
    pe.add_argument("--difficulties", nargs="*", default=[])
    pe.add_argument("--expected-first-episodes", type=int, required=True, help="normally num_envs")
    pe.add_argument("--population", choices=("first", "all"), default="first")
    pe.add_argument("--train-run-id", help="W&B id of the training run, if it cannot be found by name")
    pe.add_argument("--supersede", action="store_true", help="tag an earlier publication status:superseded")
    pe.add_argument("--dry-run", action="store_true", help="print what would be logged, without W&B")
    pe.add_argument("--allow-legacy-project", action="store_true", help=f"publish into {FALLBACK_PROJECT!r} (old runs)")

    ra = sub.add_parser("run-analysis", help="guard, execute and log the comparison's analysis notebook")
    ra.add_argument("--project", required=True)
    ra.add_argument("--group", required=True)
    ra.add_argument("--notebook", type=Path, help="default: research/<group>/analysis.ipynb")
    ra.add_argument("--draft-out", type=Path, help="default: logs/queue/<group>/analysis_draft.md")
    ra.add_argument("--timeout", type=int, default=1800, help="seconds allowed per cell")
    ra.add_argument("--dry-run", action="store_true", help="check and execute with logging off; write nothing")
    ra.add_argument("--allow-legacy-project", action="store_true", help=f"analyse {FALLBACK_PROJECT!r} (old runs)")

    sr = sub.add_parser(
        "scientific-review",
        help="prepare or validate a structured review, without model calls or publication",
    )
    sr.add_argument("--evidence", type=Path, required=True)
    sr.add_argument("--review", type=Path)
    sr.add_argument("--approval", type=Path)
    sr.add_argument("--out", type=Path, help="optional JSON output; otherwise print")

    pr = sub.add_parser("publish-report", help="build or refresh the comparison's W&B Report")
    pr.add_argument("--project", required=True)
    pr.add_argument("--group", required=True)
    pr.add_argument("--title")
    pr.add_argument("--intro", type=Path, help="Markdown: hypothesis, variants, protocol (Overview section)")
    pr.add_argument("--analysis", type=Path, help="Markdown draft; used only until the report has an Analysis")
    pr.add_argument(
        "--decision", type=Path, help="Markdown; replaces the Decision section after a current Gate D receipt"
    )
    pr.add_argument("--review", type=Path, help="versioned scientific review JSON bound to computed evidence")
    pr.add_argument("--review-approval", type=Path, help="actual researcher approval JSON bound to review_id")
    pr.add_argument("--evidence-only", metavar="REASON", help="explicit fallback when scientific review is unavailable")
    pr.add_argument("--plots", type=Path, help="research/<group>/plots.py defining make_figures(...)")
    pr.add_argument(
        "--train-metrics",
        nargs="+",
        default=None,
        help="exact required keys; default resolves protocol, profile, then observed history",
    )
    pr.add_argument("--eval-metrics", nargs="+", default=None)
    pr.add_argument(
        "--dry-run",
        action="store_true",
        help="preview report and statically check plots; no publication writes",
    )
    pr.add_argument(
        "--train-groupby",
        nargs="+",
        default=None,
        help="config key(s) identifying a training run's variant (older runs: e.g. policy fields)",
    )
    pr.add_argument("--allow-legacy-project", action="store_true", help=f"report on {FALLBACK_PROJECT!r} (old runs)")

    wt = sub.add_parser("worktree", help="create dev/agent/<group> in its own worktree")
    wt.add_argument("--group", required=True)
    wt.add_argument(
        "--base",
        default=None,
        help=(
            "branch/commit to start from (default: the repository's default branch; "
            "a previous agent branch for follow-ups)"
        ),
    )
    wt.add_argument("--path", type=Path, help="default: ../<repo>-agent-<group>")

    tg = sub.add_parser("tag", help="add/remove status:* and decision:* tags on a comparison's W&B runs")
    tg.add_argument("--project", required=True)
    tg.add_argument("--group", required=True)
    tg.add_argument("--names", nargs="*", default=[], help="run names <variant>-seed<seed>")
    tg.add_argument("--variants", nargs="*", default=[])
    tg.add_argument("--job-types", nargs="*", default=[], choices=JOB_TYPES)
    tg.add_argument("--add", nargs="*", default=[])
    tg.add_argument("--remove", nargs="*", default=[])
    tg.add_argument("--dry-run", action="store_true")
    tg.add_argument("--allow-legacy-project", action="store_true", help=f"tag runs in {FALLBACK_PROJECT!r} (old runs)")

    for command_parser in (t, e, pe, v):
        command_parser.add_argument("--protocol", type=Path, help="approved research/<group>/protocol.yaml")
    t.add_argument("--attempt", type=int, default=1)
    t.add_argument("--retry-reason")
    t.add_argument("--original-settings-id")
    t.add_argument("--original-provenance", help="original run research_provenance.json")
    q.add_argument(
        "--smoke-first",
        action="store_true",
        help="run approved technical smoke, check it, then automatically run full jobs",
    )
    q.add_argument(
        "--legacy-workflow",
        action="store_true",
        help="explicit continuation of a historical group without contracts",
    )
    args = parser.parse_args(argv)
    if args.command in {"validate-workflow", "gate-subjects", "record-gate"}:
        import yaml

        if args.command == "validate-workflow":
            result = validate_workflow(yaml.safe_load(args.file.read_text()))
        else:
            if not NAME_RE.fullmatch(args.group):
                raise SystemExit("invalid comparison group")
            directory = ROOT / "research" / args.group
            if args.command == "gate-subjects":
                workflow = read_workflow(directory)
                result = {
                    "workflow_id": workflow["workflow_id"],
                    "subjects": subjects(ROOT, directory, args.gate),
                }
            else:
                result = record_gate(ROOT, directory, yaml.safe_load(args.file.read_text()))
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "validate-contract":
        document = (
            load_protocol(args.file, args.hypothesis)
            if args.kind == "protocol"
            else read_contract(args.file, args.kind)
        )
        print(json.dumps(document, indent=2))
        return 0
    if args.command == "train-job":
        job = train_job(
            task=args.task, variant=args.variant, seed=args.seed, iterations=args.iterations,
            project=args.project, group=args.group, extra=args.extra, num_envs=args.num_envs,
            note=args.note, python=args.python, gpu=args.gpu, smoke=args.smoke,
            dirty=bool(git_source()["dirty"]), protocol=load_protocol(args.protocol) if args.protocol else None,
            attempt=args.attempt, retry_reason=args.retry_reason, original_settings_id=args.original_settings_id, original_provenance=args.original_provenance,
        )  # fmt: skip
        _fail_fast_on_stale_retry(job)
        _print_json_lines([job])
        return 0
    if args.command == "eval-jobs":
        jobs = eval_jobs(
            task=args.task, checkpoint=args.checkpoint, name=args.name, seeds=args.seeds,
            output_root=args.output_root, levels=args.levels, difficulties=args.difficulties,
            num_envs=args.num_envs, num_episodes=args.num_episodes, extra=args.extra, python=args.python,
            protocol=load_protocol(args.protocol) if args.protocol else None,
        )  # fmt: skip
        if problems := unsupported_cell_factor_problems(jobs):
            raise SystemExit("\n".join(problems))
        _print_json_lines(jobs)
        return 0
    if args.command == "queue":
        jobs = load_jobs(args.jobs)
        source = git_source()
        protocol = None
        problems = launch_problems(args.group, source) + unsupported_cell_factor_problems(jobs)
        contracts = ROOT / "research" / args.group
        if args.legacy_workflow and (
            (contracts / "hypothesis.yaml").exists()
            or (contracts / "protocol.yaml").exists()
            or any(j.get("research", {}).get("protocol_id") for j in jobs)
        ):
            problems.append("legacy-workflow cannot bypass a scientific-contract comparison")
        if (contracts / "hypothesis.yaml").exists() or not args.legacy_workflow:
            try:
                protocol = load_protocol(contracts / "protocol.yaml")
                problems += protocol_job_problems(jobs, protocol)
                problems += launch_approval_problems(ROOT, contracts, protocol, push=not args.no_push)
                if list(args.gpus) != protocol["compute_pool"]:
                    problems.append("GPU pool differs from protocol")
            except (ValueError, OSError) as error:
                problems.append(str(error))
        others = sorted(job_groups(jobs) - {args.group})
        if others:
            problems.append(f"jobs log to other W&B groups {others}")
        env = source_env()
        interpreters = {j["argv"][0] for j in jobs if Path(j["argv"][0]).name.startswith("python")}
        problems += check_imports(interpreters, env)
        scientific_jobs = [j for j in jobs if j.get("research", {}).get("job_type") == "train"]
        mandatory_smoke = bool(scientific_jobs and (protocol or {}).get("smoke_gate", {}).get("enabled"))
        if mandatory_smoke:
            if len(scientific_jobs) != len(jobs):
                problems.append("smoke-gated queues must contain only scientific training jobs")
            if not protocol["smoke_gate"]["auto_full"]:
                problems.append(
                    "protocol requires smoke then researcher approval: amend auto_full at Gate C before full launch"
                )
        if args.smoke_first:
            if args.legacy_workflow or not (protocol or {}).get("smoke_gate", {}).get("auto_full"):
                problems.append("smoke-first requires approved protocol smoke auto_full")
            if any(j.get("research", {}).get("job_type") != "train" for j in jobs):
                problems.append("smoke-first accepts only scientific training jobs")
        from research_workflow.verification import verify_retry

        for job in jobs:
            if job.get("research", {}).get("job_type") in {"train", "smoke"}:
                try:
                    verify_retry(job["research"], {"commit": source.get("commit")}, source_only=True)
                except (ValueError, OSError) as error:
                    problems.append(str(error))
        if args.dry_run:
            for job, gpu in assign(jobs, args.gpus):
                print(f"gpu {gpu}: {job['name']}")
            print(f"{max(0, len(jobs) - len(args.gpus))} more job(s) queue until a GPU frees up")
            print("launch checks: " + ("ok" if not problems else "; ".join(problems)))
            return 0
        if problems:
            raise SystemExit("not launching:\n- " + "\n- ".join(problems))
        kind = (
            "train"
            if any(TRAIN_SCRIPT in j["argv"] or j.get("research", {}).get("job_type") == "train" for j in jobs)
            else "eval"
        )
        tag = pin_launch(args.group, kind, push=not args.no_push)
        print(f"launch pinned as {tag}", flush=True)
        if mandatory_smoke or args.smoke_first:
            if args.legacy_workflow:
                raise SystemExit("smoke gating requires a protocol")
            gate = protocol["smoke_gate"]
            if not gate["enabled"] or not gate["auto_full"]:
                raise SystemExit("protocol must approve smoke then automatic full launch")
            smokes = []
            for job in jobs:
                meta = job["research"]
                variant, seed = parse_run_name(meta["cell_id"])
                spec = protocol["variants"][variant]
                smokes.append(
                    train_job(
                        task=spec["task"],
                        variant=variant,
                        seed=seed,
                        iterations=gate["iterations"],
                        project=protocol["project"],
                        group=args.group,
                        extra=spec["overrides"],
                        num_envs=protocol["training_budget"]["num_envs"],
                        protocol=protocol,
                        smoke=True,
                        attempt=meta["attempt"],
                        retry_reason=meta.get("retry_reason"),
                        original_settings_id=meta["settings_id"] if meta["attempt"] > 1 else None,
                        original_provenance=meta.get("original_provenance"),
                        gpu=job["gpu"],
                        python=job["argv"][0],
                    )
                )
            smoke_dir = args.log_dir / "smoke"
            if run_queue(
                smokes,
                args.gpus,
                smoke_dir,
                allow_shared=args.allow_shared_gpu,
                poll=args.poll,
                env=env,
                launch_tag=tag,
            ):
                raise SystemExit("technical smoke failed; full experiment not launched")
            verdict = check_smoke_gate(smokes, protocol, smoke_dir, commit=source["commit"])
            if not verdict["passed"]:
                raise SystemExit(f"smoke not verified: {verdict}; full experiment not launched")
            current = git_source()
            if launch_problems(args.group, current) or current["commit"] != source["commit"]:
                raise SystemExit("source changed during smoke; full experiment not launched")
        return run_queue(jobs, args.gpus, args.log_dir, allow_shared=args.allow_shared_gpu, poll=args.poll,
                         env=env, launch_tag=tag)  # fmt: skip
    if args.command == "status":
        for row in job_table(args.log_dir):
            state = "running" if row["alive"] else f"exit {row.get('returncode', '?')}"
            flag = "  TRACEBACK in log" if row["traceback"] else ""
            print(f"{row['name']:<40} gpu={row['gpu']}  {state}{flag}")
        return 0
    if args.command == "stop":
        for row in job_table(args.log_dir):
            if (args.name and row["name"] not in args.name) or (not args.name and not row["alive"]):
                continue
            result = stop_process(row["identity"])
            _event(args.log_dir, event="stop", name=row["name"], result=result)
            print(f"{row['name']}: {result}")
        return 0
    if args.command == "check-train":
        problems: list[str] = []
        run_dir = args.run_dir
        if run_dir is None:
            if not args.name:
                parser.error("--log-root needs --name")
            run_dir, problems = find_run_dir(args.log_root, args.name)
        verdict = check_train(run_dir, args.iterations, args.log) if run_dir else {"complete": False}
        verdict["problems"] = problems + verdict.get("problems", [])
        print(json.dumps(verdict, indent=2))
        return 0 if verdict["complete"] and not verdict.get("unverified") else (2 if verdict["complete"] else 1)
    if args.command == "check-eval":
        verdict = check_eval(
            args.output_root, seeds=args.seeds, levels=args.levels, difficulties=args.difficulties,
            expected_episodes=args.expected_episodes, expected_first_episodes=args.expected_first_episodes,
            checkpoints=args.checkpoint, metrics=args.metrics,
            protocol=load_protocol(args.protocol) if args.protocol else None,
        )  # fmt: skip
        for cell in verdict["cells"]:
            status = "ok" if cell["ok"] else "; ".join(cell["problems"])
            print(
                f"{cell['root']}/{cell['cell']:<16} {cell.get('episodes', '-'):>6} "
                f" {cell.get('profile', '-'):<15} {status}"
            )
        print(
            json.dumps(
                {
                    k: verdict[k]
                    for k in (
                        "complete",
                        "comparable",
                        "problems",
                        "protocols",
                        "commits",
                        "runtime_mismatches",
                    )
                },
                indent=2,
            )
        )
        return 0 if verdict["complete"] and verdict["comparable"] else 1
    if args.command == "publish-eval":
        protocol = load_protocol(args.protocol) if args.protocol else None
        if protocol is None:
            for file in args.output_root.glob("*/evaluation_run.json"):
                embedded = (json.loads(file.read_text()).get("research") or {}).get("protocol_contract")
                if embedded:
                    protocol = embedded
                    break
        if protocol and (
            args.project != protocol["project"] or args.expected_first_episodes != protocol["evaluation"]["num_envs"]
        ):
            raise SystemExit("publication project/population differs from approved protocol")
        verdict = check_eval(
            [args.output_root], seeds=args.seeds, levels=args.levels, difficulties=args.difficulties,
            expected_first_episodes=args.expected_first_episodes, checkpoints=[
                args.checkpoint], protocol=protocol,
        )  # fmt: skip
        if not (verdict["complete"] and verdict["comparable"]):
            for problem in verdict["problems"]:
                print(problem, file=sys.stderr)
            for cell in verdict["cells"]:
                if not cell["ok"]:
                    print(f"{cell['cell']}: {'; '.join(cell['problems'])}", file=sys.stderr)
            raise SystemExit("evaluation is incomplete or mixed; only checked evaluations are published")
        # the repository's metric definitions
        from research_workflow.eval_results import aggregate_metrics

        payload = build_eval_publication(
            group=args.group, name=args.name, task=args.task, eval_task=args.eval_task,
            checkpoint=args.checkpoint, output_root=str(args.output_root), cells=verdict["cells"],
            rows_by_cell={c["cell"]: _rows(Path(c["path"]))
                          for c in verdict["cells"]},
            aggregate=aggregate_metrics, levels=args.levels, difficulties=args.difficulties,
            population=args.population, protocol=protocol,
        )  # fmt: skip
        if args.dry_run:
            print(json.dumps(payload, indent=2, default=str))
            return 0
        print(publish_eval(payload, args.project, train_run_id=args.train_run_id, supersede=args.supersede,
                           legacy_project=args.allow_legacy_project))  # fmt: skip
        return 0
    if args.command == "run-analysis":
        result = run_analysis(
            args.notebook or Path("research") / args.group / "analysis.ipynb", args.project, args.group,
            draft_out=args.draft_out, dry_run=args.dry_run, timeout=args.timeout,
            legacy_project=args.allow_legacy_project,
        )  # fmt: skip
        if result["dirty"]:
            print("warning: working tree is dirty; commit the notebook so the branch matches the attached copy",
                  file=sys.stderr)  # fmt: skip
        print(json.dumps({k: v for k, v in result.items() if k not in {"draft", "evidence"}}, indent=2))
        return 0
    if args.command == "scientific-review":
        evidence = validate_evidence(json.loads(args.evidence.read_text()))
        review = (
            validate_review(json.loads(args.review.read_text()), evidence) if args.review else review_template(evidence)
        )
        if not args.review:
            print(
                "No scientific review generated: an agent must author and validate this scaffold before approval.",
                file=sys.stderr,
            )
        if args.approval:
            if not args.review:
                raise ValueError("approval requires --review")
            validate_approval(json.loads(args.approval.read_text()), review)
        rendered = json.dumps(review, indent=2, allow_nan=False) + "\n"
        if args.out:
            args.out.write_text(rendered)
        else:
            print(rendered)
        return 0
    if args.command == "publish-report":

        def text(path: Path | None) -> str | None:
            return path.read_text() if path else None

        url = publish_report(
            args.project, args.group, title=args.title, intro=text(args.intro) or "", analysis=text(args.analysis),
            decision=text(args.decision), plots=args.plots, train_metrics=args.train_metrics,
            eval_metrics=args.eval_metrics, dry_run=args.dry_run, train_groupby=args.train_groupby,
            legacy_project=args.allow_legacy_project,
            review=json.loads(args.review.read_text()) if args.review else None,
            review_approval=json.loads(args.review_approval.read_text()) if args.review_approval else None,
            evidence_only=args.evidence_only,
        )  # fmt: skip
        if not args.dry_run:
            print(url)
        return 0
    if args.command == "worktree":
        path = create_worktree(args.group, base=args.base, path=args.path)
        print(f"{agent_branch(args.group)} checked out at {path}; logs/ is shared with the main checkout.")
        print(f"Work, commit and launch from there: cd {path} && python scripts/research_runs.py …")
        return 0
    if args.command == "tag":
        import wandb

        project = check_project(args.project, legacy=args.allow_legacy_project)
        runs = select_runs(_wandb_runs(wandb.Api(), project, args.group),
                           names=args.names, variants=args.variants, job_types=args.job_types)  # fmt: skip
        for name, old, new in retag(runs, args.add, args.remove, dry_run=args.dry_run):
            print(f"{name}: {old} -> {new}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())

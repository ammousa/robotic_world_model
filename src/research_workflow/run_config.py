"""What the training/evaluation scripts record about a run for the research workflow (see AGENTS.md)."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence

from . import launch_metadata


def git_state(path: str | None = None) -> dict[str, str | bool | None]:
    """The commit, branch and dirty flag of the repository containing ``path`` (default: the cwd)."""
    cwd = os.path.dirname(os.path.abspath(path)) if path else os.getcwd()

    def run(args: list[str]) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10, check=False)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    status = run(["status", "--porcelain"])
    return {
        "commit": run(["rev-parse", "HEAD"]),
        "branch": run(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(status) if status is not None else None,
    }


def research_config(
    tags: Sequence[str], *, group: str | None, job_type: str | None, run_name: str | None, task: str | None,
    source_path: str | None = None,
) -> dict:  # fmt: skip
    """The ``research`` block a tagged run carries in its W&B config.

    Identity comes from the ``key:value`` launch tags (``variant``, ``seed``, ``task``), so W&B reports
    can group and filter on ``config.runner_cfg.research.*``. The Git branch/commit identify the source;
    the launch tag is set by ``scripts/research_runs.py queue`` through ``RESEARCH_LAUNCH_TAG``.
    """
    identity = dict(tag.split(":", 1) for tag in tags if ":" in tag)
    if "seed" in identity and identity["seed"].isdigit():
        identity["seed"] = int(identity["seed"])
    git = git_state(source_path)
    return {
        **{k: identity[k] for k in ("variant", "seed", "task") if k in identity},
        "task": identity.get("task", task),
        "comparison": group,
        "job_type": job_type,
        "run_name": run_name,
        "branch": git["branch"],
        "commit": git["commit"],
        "dirty": git["dirty"],
        "launch_tag": os.environ.get("RESEARCH_LAUNCH_TAG"),
        **launch_metadata(),
    }

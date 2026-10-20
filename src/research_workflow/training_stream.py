"""Per-iteration finiteness stream that ``research_runs.py check-train`` and the smoke gate read.

``attach_metric_stream`` wraps ``runner.log`` so every logged iteration appends one JSON line to
``<log_dir>/locomotion_metrics.jsonl`` and stops training on a non-finite loss, reward or action std.
Fresh runs only: the checks expect one record per iteration, ``0 .. max_iterations - 1``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

STREAM_FILENAME = "locomotion_metrics.jsonl"


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def attach_metric_stream(runner, log_dir: str | Path) -> Path:
    """Record each iteration's finiteness checks; raise ``FloatingPointError`` on a bad one."""
    stream = Path(log_dir) / STREAM_FILENAME
    stream.parent.mkdir(parents=True, exist_ok=True)
    original_log = runner.log

    def log(locs, *args, **kwargs):
        record = {"iteration": int(locs["it"])}
        losses = {k: _number(v) for k, v in (locs.get("loss_dict") or {}).items()}
        rewards = [float(r) for r in locs.get("rewbuffer", [])]
        std = runner.alg.policy.action_std.detach()
        finite = (
            all(v is None or math.isfinite(v) for v in losses.values())
            and all(math.isfinite(r) for r in rewards)
            and bool(std.isfinite().all())
            and bool((std > 0).all())
        )
        if not finite:
            raise FloatingPointError(
                f"non-finite loss/reward or non-positive action std at iteration {record['iteration']}"
            )
        record.update(
            {f"Loss/{k}": v for k, v in losses.items() if v is not None},
            **{"Policy/std_mean": std.mean().item(), "Checks/finite_rewards_std": 1.0},
        )
        if rewards:
            record["Train/mean_reward"] = sum(rewards) / len(rewards)
        with stream.open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        return original_log(locs, *args, **kwargs)

    runner.log = log
    return stream

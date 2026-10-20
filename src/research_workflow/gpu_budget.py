"""Conservative allocated GPU time from the existing append-only queue event logs."""

import json
from datetime import datetime, timezone
from pathlib import Path


def used_seconds(log_root, gpu, now=None):
    now = now or datetime.now(timezone.utc)
    total = 0.0
    for path in Path(log_root).rglob("events.jsonl"):
        starts = {}
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if event.get("gpu") != gpu:
                continue
            if event.get("event") == "start":
                if event["name"] in starts:
                    raise ValueError("duplicate start in GPU budget history")
                starts[event["name"]] = datetime.fromisoformat(event["at"])
            elif event.get("event") == "exit":
                start = starts.pop(event["name"], None)
                if start is None:
                    raise ValueError("GPU exit without a matching start")
                # Events have whole-second timestamps. Add two seconds per job
                # rather than undercounting initialization or shutdown rounding.
                total += max(0, (datetime.fromisoformat(event["at"]) - start).total_seconds()) + 2
        # An interrupted/orphaned job is charged until now; never invent zero cost.
        total += sum(max(0, (now - start).total_seconds()) + 2 for start in starts.values())
    return total


def timeout_argv(argv, log_root, gpu, hours_per_gpu, max_job_seconds):
    remaining = int(hours_per_gpu * 3600 - used_seconds(log_root, gpu) - 5)
    seconds = min(remaining, max_job_seconds)
    if seconds <= 0:
        raise ValueError(f"GPU {gpu} cumulative allocated-time budget exhausted")
    return ["timeout", "--signal=TERM", "--kill-after=1s", f"{seconds}s", *argv]

"""Allocated GPU time fails closed across collection, pilots and evaluation."""

import json
from datetime import datetime, timezone

import pytest

from research_workflow.gpu_budget import timeout_argv, used_seconds


def test_budget_sums_all_existing_queue_logs_and_orphaned_jobs(tmp_path):
    start = {"event": "start", "name": "collect", "gpu": 0, "at": "2026-10-20T00:00:00+00:00"}
    end = {"event": "exit", "name": "collect", "gpu": 0, "at": "2026-10-20T01:00:00+00:00"}
    folder = tmp_path / "collection"
    folder.mkdir()
    (folder / "events.jsonl").write_text(json.dumps(start) + "\n" + json.dumps(end) + "\n")
    assert used_seconds(tmp_path, 0) == 3602
    assert used_seconds(tmp_path, 1) == 0
    running = tmp_path / "train"
    running.mkdir()
    (running / "events.jsonl").write_text(
        json.dumps({**start, "name": "train", "at": "2026-10-20T02:00:00+00:00"}) + "\n"
    )
    now = datetime(2026, 10, 20, 3, tzinfo=timezone.utc)
    assert used_seconds(tmp_path, 0, now) == 7204
    with pytest.raises(ValueError, match="exhausted"):
        timeout_argv(["python", "train.py"], tmp_path, 0, 0.1, 7200)


def test_timeout_wraps_actual_process_group_and_preserves_command(tmp_path):
    result = timeout_argv(["python", "train.py", "--seed", "11"], tmp_path, 0, 12, 7200)
    assert result == ["timeout", "--signal=TERM", "--kill-after=1s", "7200s", "python", "train.py", "--seed", "11"]

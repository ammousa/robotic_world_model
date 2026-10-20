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


def test_world_model_reservation_includes_stochastic_and_replay_populations():
    from research_workflow.governance import reservations

    protocol = {
        "domain": "world_model",
        "training_budget": {"iterations": 20000},
        "smoke_gate": {"enabled": True, "iterations": 1000},
        "variants": {"B0": {}, "B2": {}},
        "seeds": [11, 22],
        "retry_policy": {"max_attempts": 2},
        "evaluation": {"seeds": [4300, 4400], "levels": [], "difficulties": [], "num_envs": 256, "num_episodes": 1},
        "world_model": {"reserved_evaluation_units": 200000},
    }
    training, evaluation = reservations(protocol)
    assert training == 168000
    assert evaluation == 200000
    protocol["world_model"]["reserved_evaluation_units"] = 1
    import pytest

    with pytest.raises(ValueError, match="reservation"):
        reservations(protocol)

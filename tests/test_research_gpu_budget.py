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


def test_carried_global_budget_is_enforced_by_equal_device_shares(tmp_path):
    from research_workflow.gpu_budget import campaign_hours_per_gpu

    limits = {
        "max_GPU_hours_per_device": 12.0,
        "max_total_GPU_hours": 24.0,
        "prior_comparison_GPU_hours": 9.7347368975,
        "remaining_total_GPU_hours": 14.2652631025,
        "prior_device_GPU_hours": {
            "0": 0.15472222222222223,
            "1": 0.12333333333333334,
            "2": 4.696403564166667,
            "3": 4.760277777777778,
        },
    }
    hours = campaign_hours_per_gpu(limits, [0, 1])
    assert hours == pytest.approx(7.13263155125)
    assert 2 * hours + limits["prior_comparison_GPU_hours"] <= 24.0
    for gpu in [0, 1]:
        assert hours + limits["prior_device_GPU_hours"][str(gpu)] <= 12.0
        command = timeout_argv(["python", "job.py"], tmp_path, gpu, hours, 100000)
        assert int(command[3][:-1]) < hours * 3600
    assert campaign_hours_per_gpu({"max_GPU_hours_per_device": 12}, [2, 3]) == 12
    with pytest.raises(ValueError, match="reconcile"):
        campaign_hours_per_gpu({**limits, "remaining_total_GPU_hours": 24}, [0, 1])
    with pytest.raises(ValueError, match="invalid"):
        campaign_hours_per_gpu({**limits, "max_GPU_hours_per_device": float("nan")}, [0, 1])

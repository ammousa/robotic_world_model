"""Regression checks for checkpoint, metric, retry, runtime and dataset boundaries."""

from __future__ import annotations

import copy
import functools
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import test_research_contracts as contracts_tests
from test_research_contracts import job
from test_research_runs import FakeRun, _records, _run_dir, _run_template, _template_world, rr

from research_workflow import runtime_fingerprint, runtime_mismatches, validate_contract
from research_workflow.verification import config_hash, protocol_aggregate, verify_retry


@pytest.fixture
def contracts():
    return contracts_tests.contracts.__wrapped__()


@pytest.mark.parametrize("kind", ["empty", "corrupt", "no_model", "no_weights", "nan", "optimizer_nan"])
def test_unusable_checkpoint_blocks_completion(tmp_path, kind):
    import torch

    run = _run_dir(tmp_path, records=_records(3))
    path = run / "model_3.pt"
    if kind in {"empty", "corrupt"}:
        path.write_bytes(b"" if kind == "empty" else b"not a checkpoint")
    else:
        data = {"policy_state_dict": {"weight": torch.ones(2)}}
        if kind == "no_model":
            data = {"iteration": 2}
        elif kind == "no_weights":
            data = {"policy_state_dict": {"weight": "fake"}}
        elif kind == "nan":
            data["policy_state_dict"]["weight"][0] = float("nan")
        else:
            data["optimizer_state_dict"] = {"state": {0: {"exp_avg": torch.tensor(float("nan"))}}}
        torch.save(data, path)
    verdict = rr.check_train(run, 3)
    assert not verdict["complete"] and any("checkpoint" in p for p in verdict["problems"])


def test_checkpoint_validation_does_not_allocate_gpu(tmp_path, monkeypatch):
    import torch

    run = _run_dir(tmp_path, records=_records(3))
    load = Mock(wraps=torch.load)
    monkeypatch.setattr(torch, "load", load)
    assert rr.check_train(run, 3)["complete"]
    assert load.call_args.kwargs == {"map_location": "cpu", "weights_only": True}


@pytest.mark.parametrize("reduction,expected", [("mean", 2.0), ("rms", 5**0.5), ("rate", 0.5)])
def test_custom_metrics_publish_with_declared_reducers(contracts, reduction, expected):
    _, protocol, _ = contracts
    protocol["evaluation"]["metric_reducers"] = {"custom": {"column": "value", "reduction": reduction}}
    values = [0, 1] if reduction == "rate" else [1, 3]
    aggregate = protocol_aggregate(lambda rows: {"standard": 7}, protocol)
    result = aggregate([{"value": v} for v in values])
    assert result["standard"] == 7 and result["custom"] == pytest.approx(expected)
    with pytest.raises(ValueError, match="custom metric"):
        aggregate([{"value": float("nan")}])


@pytest.mark.parametrize("kind", ["missing", "bad_reduction", "override_standard", "conflicting_column"])
def test_unsupported_metric_protocol_rejected_before_launch(contracts, kind):
    hypothesis, protocol, _ = contracts
    protocol.pop("protocol_id")
    reducers = protocol["evaluation"]["metric_reducers"]
    if kind == "missing":
        reducers.pop("preference_variance")
    elif kind == "bad_reduction":
        reducers["preference_variance"]["reduction"] = "best_episode"
    elif kind == "override_standard":
        reducers["fall_rate"] = {"column": "fell", "reduction": "mean"}
    else:
        protocol["evaluation"]["metric_columns"] = {"preference_variance": "another"}
    with pytest.raises(ValueError):
        validate_contract("protocol", protocol, hypothesis)


def original_run(tmp_path, first, config=None):
    config = config or {"agent": {"run_name": "router-seed1", "device": "cuda:0", "learning_rate": 0.01}}
    fp = {"commit": "abc", "config_hash_version": 2, "resolved_config_hash": config_hash(config)}
    path = tmp_path / "research_provenance.json"
    path.write_text(json.dumps({**first["research"], "runtime_fingerprint": fp, "run_dir": str(tmp_path)}))
    return path, fp


def test_retry_verifies_actual_source_and_resolved_config(contracts, tmp_path):
    _, p, _ = contracts
    path, fp = original_run(tmp_path, job(p))
    retry = job(p, attempt=2, retry_reason="vm_died", original_provenance=str(path))
    assert retry["research"]["settings_identical"] is None
    assert rr.protocol_job_problems([retry], p) == []
    assert verify_retry(retry["research"], fp)["settings_identical"] is True
    with pytest.raises(ValueError, match="source commit"):
        verify_retry(retry["research"], {**fp, "commit": "changed"}, source_only=True)
    with pytest.raises(ValueError, match="resolved configuration"):
        verify_retry(retry["research"], {**fp, "resolved_config_hash": "changed"})
    path.write_text(path.read_text().replace('"abc"', '"new"'))
    with pytest.raises(ValueError, match="provenance changed"):
        verify_retry(retry["research"], fp)


def test_unreferenced_retry_cannot_launch(contracts, tmp_path, monkeypatch):
    _, p, _ = contracts
    retry = job(p, attempt=2, retry_reason="vm_died", original_settings_id=job(p)["research"]["settings_id"])
    monkeypatch.setattr(rr, "git_source", lambda: {"commit": "abc", "branch": "dev/agent/promo-router", "dirty": False})
    spawn = Mock()
    monkeypatch.setattr(rr.subprocess, "Popen", spawn)
    with pytest.raises(SystemExit, match="original-provenance"):
        rr.run_queue([retry], [0], tmp_path / "queue", allow_shared=True)
    spawn.assert_not_called()


def stable_callable(value=1):
    return value


def test_runtime_hash_is_stable_and_retries_ignore_launch_placement(tmp_path):
    first = {
        "agent": {"run_name": "a", "device": "cuda:0", "lr": 0.01},
        "function": functools.partial(stable_callable, value=2),
        "indices": slice(None),
    }
    moved = copy.deepcopy(first)
    moved["agent"].update(run_name="a-attempt2", device="cuda:3")
    assert config_hash(first) == config_hash(moved)
    moved["agent"]["lr"] = 0.1
    assert config_hash(first) != config_hash(moved)
    fp = runtime_fingerprint(tmp_path, resolved_config={"unsupported": object()})
    assert fp["resolved_config_hash"] is None and fp["resolved_config_hash_error"]
    assert fp["isaaclab_commit"] is None


def test_assets_compare_by_content_across_checkout_paths(tmp_path, monkeypatch):
    monkeypatch.setattr("research_workflow.subprocess.run", Mock(side_effect=FileNotFoundError))
    paths = [tmp_path / "one" / "robot.usd", tmp_path / "two" / "robot.usd"]
    for path in paths:
        path.parent.mkdir()
        path.write_bytes(b"robot version 1")
    a, b = [runtime_fingerprint(path.parent, asset_paths=[path]) for path in paths]
    assert "asset_hashes" not in runtime_mismatches([a, b])
    paths[1].write_bytes(b"robot version 2")
    assert "asset_hashes" in runtime_mismatches([a, runtime_fingerprint(paths[1].parent, asset_paths=[paths[1]])])


def test_runtime_files_hashed_without_read_bytes(tmp_path, monkeypatch):
    (tmp_path / "environment.yml").write_text("dependencies: []")
    monkeypatch.setattr(Path, "read_bytes", Mock(side_effect=AssertionError("unbounded read")))
    assert runtime_fingerprint(tmp_path)["environment_hashes"]["environment.yml"]


def test_selected_seed_costs_separate_from_retry_spend(monkeypatch, contracts):
    import pandas as pd
    from test_research_workflow_safety import contracted_template_world

    runs, episodes = contracted_template_world(contracts)
    runs.append(
        FakeRun(
            "mlp-seed1-attempt2",
            "train",
            ["status:superseded"],
            state="finished",
            config={"research": {"variant": "mlp", "seed": 1, "attempt": 2}},
            summary={"_runtime": 360000},
            history=lambda **kw: pd.DataFrame(),
        )
    )
    runs.append(
        FakeRun(
            "mlp-seed2-attempt2",
            "train",
            state="crashed",
            config={"research": {"variant": "mlp", "seed": 2, "attempt": 2}},
            summary={},
            history=lambda **kw: pd.DataFrame(),
        )
    )
    scope, _ = _run_template(monkeypatch, runs, episodes, log=False)
    assert scope["cost_by_variant"].loc["mlp", "wall_time_h_mean"] == 2.5
    costs = scope["TABLES"]["cost_all_attempt_spend"].set_index("variant")
    assert costs.loc["mlp", "wall_time_h_known_subtotal"] == 105
    assert costs.loc["mlp", "wall_time_h_unknown_attempts"] == 1
    assert pd.isna(costs.loc["mlp", "wall_time_h_total"])
    assert len(scope["TABLES"]["cost_selected_seeds"]) == 4
    assert len(scope["TABLES"]["cost_attempts"]) == 6


def test_publish_eval_cli_uses_custom_reducers(contracts, tmp_path, monkeypatch, capsys):
    from test_research_contracts import contracted_publication

    _, protocol, _ = contracts
    args = contracted_publication(protocol, tmp_path)
    standard_defaults = {
        "fell": False,
        "time_seconds": 20.0,
        "episode_length": 1000,
        "distance_travelled": 10.0,
        "path_length": 11.0,
        "terrain_difficulty": 0.0,
        "static_friction": 0.7,
        "dynamic_friction": 0.7,
        "added_mass_kg": 0.0,
        "max_push_delta_velocity_mps": 0.0,
        "num_pushes": 0,
    }
    for rows in args["rows_by_cell"].values():
        for row in rows:
            row.update(standard_defaults)
    for cell in args["cells"]:
        cell["path"] = cell["cell"]
    monkeypatch.setattr(rr, "load_protocol", lambda path: protocol)
    monkeypatch.setattr(
        rr, "check_eval", lambda *a, **kw: {"complete": True, "comparable": True, "cells": args["cells"]}
    )
    monkeypatch.setattr(rr, "_rows", lambda path: args["rows_by_cell"][path.name])
    cli = [
        "publish-eval",
        "--protocol",
        "unused",
        "--project",
        protocol["project"],
        "--group",
        protocol["group"],
        "--name",
        args["name"],
        "--task",
        args["task"],
        "--eval-task",
        args["eval_task"],
        "--checkpoint",
        args["checkpoint"],
        "--output-root",
        args["output_root"],
        "--seeds",
        "41",
        "42",
        "--levels",
        "0",
        "4",
        "8",
        "--expected-first-episodes",
        "150",
        "--dry-run",
    ]
    assert rr.main(cli) == 0
    payload = json.loads(capsys.readouterr().out)
    for metric in protocol["evaluation"]["metric_reducers"]:
        assert payload["summary"]["eval/" + metric] == pytest.approx(0.2)


def test_checkpoint_failure_keeps_smoke_gate_closed(contracts, tmp_path):
    _, protocol, _ = contracts
    protocol = copy.deepcopy(protocol)
    protocol["smoke_gate"]["iterations"] = 3
    protocol["smoke_gate"]["log_roots"]["router"] = str(tmp_path / "logs")
    smoke = job(protocol, smoke=True, iterations=3)
    run = _run_dir(tmp_path, records=_records(3))
    destination = run.with_name("2026_router-seed1-smoke")
    run.rename(destination)
    (destination / "model_3.pt").write_bytes(b"")
    (destination / "research_provenance.json").write_text(
        json.dumps({**smoke["research"], "runtime_fingerprint": {"commit": "abc"}})
    )
    queue = tmp_path / "queue"
    queue.mkdir()
    (queue / "router-seed1-smoke.log").write_text("technical process finished")
    verdict = rr.check_smoke_gate([smoke], protocol, queue, commit="abc")
    assert not verdict["passed"]


def test_custom_diagnostic_survives_notebook_paired_analysis(contracts, monkeypatch):
    from test_research_workflow_safety import contracted_template_world

    runs, artifacts = contracted_template_world(contracts)
    artifacts["eval-episodes-mlp-seed1"]["preference_variance"] = 0.5
    scope, _ = _run_template(monkeypatch, runs, artifacts, log=False)
    paired = scope["paired"]
    selected = paired[(paired["metric"] == "preference_variance_episode_value") & (paired["train_seed"] == 1)]
    assert selected["mean_difference"].iloc[0] == pytest.approx(0.3)
    assert "preference_variance" in scope["per_seed"]


def test_callable_hash_stable_across_processes():
    import subprocess
    import sys

    command = [
        sys.executable,
        "-c",
        "import math; from research_workflow.verification import config_hash; print(config_hash({'fn': math.sin}))",
    ]
    outputs = [subprocess.check_output(command, env=rr.source_env(), text=True).strip() for _ in range(2)]
    assert outputs[0] == outputs[1] and len(outputs[0]) == 64


def test_missing_or_nonfinite_costs_remain_unknown(monkeypatch):
    import pandas as pd

    runs, artifacts = _template_world()
    training = next(r for r in runs if r.job_type == "train")
    training.summary["_runtime"] = None
    training.summary["research/cost"] = {"gpu_hours": float("inf")}
    scope, _ = _run_template(monkeypatch, runs, artifacts, log=False)
    totals = scope["TABLES"]["cost_all_attempt_spend"].set_index("variant")
    assert pd.isna(totals.loc["mlp", "wall_time_h_total"])
    assert totals.loc["mlp", "wall_time_h_unknown_attempts"] == 1
    assert pd.isna(totals.loc["mlp", "gpu_hours_total"])


def test_unbounded_config_limits_have_stable_identity():
    assert config_hash({"limit": float("inf")}) == config_hash({"limit": float("inf")})
    assert config_hash({"limit": float("inf")}) != config_hash({"limit": float("-inf")})
    with pytest.raises(ValueError, match="NaN"):
        config_hash({"learning_rate": float("nan")})


def test_callable_arguments_remain_scientific_settings():
    a = functools.partial(stable_callable, run_name="method_a")
    b = functools.partial(stable_callable, run_name="method_b")
    assert config_hash({"function": a}) != config_hash({"function": b})


@pytest.mark.parametrize("profile,expected", [("general", 0.25), ("forward_course", 0.75)])
def test_profile_survival_publication_and_row_validation_share_definitions(contracts, profile, expected):
    from research_workflow.verification import declared_metric_reducers

    _, protocol, _ = contracts
    ev = protocol["evaluation"]
    ev.update(
        profile=profile,
        primary_metric={"name": "survival_rate"},
        secondary_metrics=[],
        required_metrics=[],
        num_envs=4,
        paired_factors=[],
        metric_reducers={},
    )
    rows = [{"episode_id": 0, "env_id": i, "fell": i == 0, "timed_out": i == 1} for i in range(4)]
    aggregate = protocol_aggregate(lambda rows: {"survival_rate": 0.25, "retained": 9}, protocol)
    assert aggregate(rows) == {"survival_rate": expected, "retained": 9}
    assert rr.scored_row_problems(rows, protocol, {}) == []
    column = declared_metric_reducers(ev)["survival_rate"]["column"]
    rows[0][column] = 2
    assert any("non-binary" in p for p in rr.scored_row_problems(rows, protocol, {}))
    if profile == "forward_course":
        with pytest.raises(ValueError, match="non-binary"):
            aggregate(rows)


@pytest.mark.parametrize("error_type", [ValueError, OSError])
def test_dispatch_retry_failure_drains_active_jobs_and_records_queue_end(tmp_path, monkeypatch, error_type):
    import sys

    import research_workflow.verification as verification

    calls = 0

    def verify(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:  # Two preflight checks, first dispatch, then late second dispatch.
            raise error_type("late retry provenance mismatch")

    monkeypatch.setattr(verification, "verify_retry", verify)
    monkeypatch.setattr(rr, "git_source", lambda: {"commit": "abc", "dirty": False})
    jobs = [
        {
            "name": name,
            "argv": [sys.executable, "-c", "import time; time.sleep(0.1)"],
            "gpu": None,
            "research": {"job_type": "train"},
        }
        for name in ("first", "blocked")
    ]
    with pytest.raises(SystemExit, match="late retry provenance mismatch"):
        rr.run_queue(jobs, [0, 1], tmp_path, allow_shared=True, poll=0.01)
    events = rr.read_events(tmp_path)
    assert [e["name"] for e in events if e["event"] == "start"] == ["first"]
    assert [e["returncode"] for e in events if e["event"] == "exit"] == [0]
    assert any(e["event"] == "dispatch_blocked" and e["name"] == "blocked" for e in events)
    assert events[-1]["event"] == "queue_end"
    assert events[-1]["error"] == "late retry provenance mismatch"


def test_forward_course_publication_corrects_general_survival(contracts, tmp_path):
    from test_research_contracts import contracted_publication

    from research_workflow import content_id
    from research_workflow.eval_results import aggregate_metrics

    _, protocol, _ = contracts
    protocol["evaluation"]["profile"] = "forward_course"
    protocol["evaluation"]["required_metrics"] = ["survival_rate"]
    protocol["protocol_id"] = content_id("protocol", protocol)
    args = contracted_publication(protocol, tmp_path)
    args["aggregate"] = aggregate_metrics
    from research_workflow.verification import STANDARD_REDUCERS

    for rows in args["rows_by_cell"].values():
        for row in rows:
            for spec in STANDARD_REDUCERS.values():
                row.setdefault(spec["column"], 0.0)
            row.update(fell=False, timed_out=False, success=True)
    payload = rr.build_eval_publication(**args)
    assert payload["summary"]["eval/fall_rate"] == 0
    assert payload["summary"]["eval/survival_rate"] == 1

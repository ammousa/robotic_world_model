"""Regression tests for selected/final checkpoint publication verification."""

import json
import numpy as np
import torch
from types import SimpleNamespace

import pytest

from research_workflow import paired_unit_id
from world_modeling import results as candidate
from world_modeling.core import file_digest
from world_modeling.evaluation import HORIZONS


@pytest.fixture
def cell(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "metadata.json").write_text("{}")
    source = {"commit": "commit-main", "rsl_rl_commit": "commit-rsl", "isaaclab_commit": "commit-isaac"}
    checkpoint = tmp_path / "model.pt"
    torch.save({"variant": "B0", "seed": 11, "wandb_run_id": "training-id", "runtime_fingerprint": source}, checkpoint)
    protocol = {
        "protocol_id": "protocol-test",
        "world_model": {"policy_sha256": "policy-hash", "dataset": str(dataset)},
        "evaluation": {
            "paired_factors": [
                "training_seed",
                "evaluation_partition",
                "episode_id",
                "fork_step",
                "command_schedule_id",
            ],
            "artifact_metrics": {
                "timing": {
                    "source": "provenance.json",
                    "selection": {"mode": "A", "partition": "test_id"},
                    "columns": ["policy_plus_model_latency_p95_ms"],
                }
            },
        },
    }
    destination = tmp_path / "cell"
    destination.mkdir()
    (destination / "predictions.npz").write_bytes(b"immutable-test-prediction")
    info = {
        "checkpoint_sha256": file_digest(checkpoint),
        "protocol_id": protocol["protocol_id"],
        "variant": "B0",
        "training_seed": 11,
        "partition": "test_id",
        "mode": "A",
        "population": 1,
        "episode_ids": [7],
        "horizons": HORIZONS,
        "policy_sha256": "policy-hash",
        "dataset_sha256": file_digest(dataset / "metadata.json"),
        "train_run_id": "training-id",
        "prediction_sha256": file_digest(destination / "predictions.npz"),
        "runtime_fingerprint": source,
        "timing": {"policy_plus_model_latency_p95_ms": 3.0},
    }
    rows = []
    for h in HORIZONS:
        row = {
            "variant": "B0",
            "training_seed": 11,
            "evaluation_partition": "test_id",
            "mode": "A",
            "episode_id": 7,
            "fork_step": 281,
            "command_schedule_id": "schedule-test",
            "horizon_steps": h,
            "initialization_valid": True,
            "stable_duration_s": 30.0,
            "unstable": False,
            "reference_eligible": True,
            "prediction_finite": True,
            "failure_aware_score": 1.0,
            "physical_trajectory_nrmse": 1.0,
        }
        row["eval_unit_id"] = paired_unit_id(row, {}, protocol["evaluation"]["paired_factors"])
        rows.append(row)

    def write():
        (destination / "provenance.json").write_text(json.dumps(info))
        (destination / "scored_rows.json").write_text(json.dumps(rows))

    def verify(role):
        write()
        return candidate.verify_cell(destination, checkpoint, protocol, "test_id", "A", [7], checkpoint_role=role)

    return info, rows, verify


def test_selected_missing_timing_still_rejected(cell):
    info, _, verify = cell
    del info["timing"]
    with pytest.raises(ValueError, match="timing evidence is absent"):
        verify("selected")


def test_final_missing_timing_accepted_without_inventing_it(cell):
    info, rows, verify = cell
    del info["timing"]
    actual_rows, actual_info = verify("final")
    assert actual_rows == rows
    assert "timing" not in actual_info


def test_selected_timing_accepted_unchanged(cell):
    info, rows, verify = cell
    actual_rows, actual_info = verify("selected")
    assert actual_info == info and actual_rows == rows


@pytest.mark.parametrize("role", ["", "best", None])
def test_unknown_role_rejected(cell, role):
    with pytest.raises(ValueError, match="checkpoint_role must be"):
        cell[2](role)


@pytest.mark.parametrize("role", ["selected", "final"])
def test_identity_checks_preserved_for_both_roles(cell, role):
    info, _, verify = cell
    info["checkpoint_sha256"] = "wrong-digest"
    with pytest.raises(ValueError, match="identity mismatch"):
        verify(role)


@pytest.mark.parametrize("role", ["selected", "final"])
def test_terminal_censoring_preserved_for_both_roles(cell, role):
    _, rows, verify = cell
    rows[0]["reference_eligible"] = False
    with pytest.raises(ValueError, match="uncensored physical score"):
        verify(role)


@pytest.mark.parametrize("role", ["selected", "final"])
def test_failure_population_preserved_for_both_roles(cell, role):
    _, rows, verify = cell
    rows[0]["failure_aware_score"] = None
    with pytest.raises(ValueError, match="failure was dropped"):
        verify(role)


def test_actual_publisher_tags_validate_and_preserve_full_checkpoint_digest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    checkpoint = tmp_path / "model_20000.pt"
    torch.save({"variant": "B0", "seed": 11, "wandb_run_id": "train-unit-test"}, checkpoint)
    full_digest = candidate.file_digest(checkpoint)
    group = "publication-boundary-test"
    protocol = {
        "group": group,
        "project": "rwm-smoke",
        "compute_pool": [2, 3],
        "protocol_id": "p-" + "a" * 24,
        "hypothesis_id": "h-" + "b" * 24,
        "world_model": {"dataset": "dataset", "pilot_seed": 11},
    }
    train = SimpleNamespace(
        state="finished",
        job_type="train",
        group=group,
        id="train-unit-test",
        config={"runner_cfg": {"research": {"protocol_id": protocol["protocol_id"], "variant": "B0", "seed": 11}}},
    )
    directory = tmp_path / "run"
    destination = directory / "world_model_eval"
    for partition in ("test_id", "test_command_shift"):
        for mode in ("A", "B", "B_shadow", "A-final"):
            (destination / f"{partition}-{mode}").mkdir(parents=True)
    rows = [{"horizon_steps": 100, "reference_eligible": True, "failure_aware_score": 1.0, "episode_id": 7}]
    (destination / "test_id-A/scored_rows.json").write_text(json.dumps(rows))
    streams = [{"sample_stream": s, "episode_id": 7, "horizon_steps": h} for s in range(6100, 6116) for h in HORIZONS]
    (destination / "test_id-A/stochastic_rows.json").write_text(json.dumps(streams))
    monkeypatch.setattr(
        candidate.wandb, "Api", lambda: SimpleNamespace(default_entity="unit-test", run=lambda _: train)
    )
    monkeypatch.setattr(candidate, "used_seconds", lambda *args: 0.0)
    monkeypatch.setattr(candidate, "read_partition", lambda *args: ({"episode_id": np.asarray([7])}, {}))
    monkeypatch.setattr(
        candidate, "verify_cell", lambda *args, **kwargs: (rows, {"dataset_sha256": "dataset-unit-test", "timing": {}})
    )
    monkeypatch.setattr(candidate, "summarize", lambda _: [])
    monkeypatch.setattr(candidate, "verify_videos", lambda *args: [])
    monkeypatch.setattr(candidate, "equality_controls", lambda *args: {})
    captured = {}

    class BoundaryReached(Exception):
        pass

    def capture_init(**kwargs):
        # Real SDK validation: the former full-SHA tag fails here before the
        # sentinel. No SDK initialization, API mutation or artifact write occurs.
        settings = candidate.wandb.Settings(run_tags=kwargs["tags"])
        captured["tags"] = settings.run_tags
        captured["research"] = kwargs["config"]["research"]
        raise BoundaryReached

    monkeypatch.setattr(candidate.wandb, "init", capture_init)
    with pytest.raises(BoundaryReached):
        candidate.publish(protocol, [("B0", 11, directory, checkpoint, checkpoint)])
    assert "ckpt:" + full_digest[:12] in captured["tags"]
    assert all(len(tag) <= 64 for tag in captured["tags"])
    assert captured["research"]["checkpoint_sha256"] == full_digest and len(full_digest) == 64

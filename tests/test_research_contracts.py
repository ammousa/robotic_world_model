"""Scientific boundaries without simulation, network or GPU jobs."""

from __future__ import annotations

import copy
import json
import re
import yaml
from pathlib import Path

import pytest
from test_research_runs import _aggregate, _cell, _rows_for, rr

from research_workflow import (
    content_id,
    ensure_supported_evaluation,
    paired_unit_id,
    runtime_fingerprint,
    runtime_mismatches,
    validate_contract,
)

EXAMPLES = Path(__file__).with_name("research_contract_examples.yaml")
REFERENCE = (
    Path(__file__).resolve().parents[1] / ".agents/skills/research-experiment/references/scientific-contracts.md"
)


@pytest.fixture
def contracts():
    h, p, d = yaml.safe_load_all(EXAMPLES.read_text())
    h = validate_contract("hypothesis", h)
    p["hypothesis_id"] = h["hypothesis_id"]
    p = validate_contract("protocol", p, h)
    return h, p, d


def test_examples_validate_and_ids_are_stable(contracts):
    h, p, d = contracts
    assert validate_contract("decision", d) == d
    assert content_id("hypothesis", dict(reversed(list(h.items())))) == h["hypothesis_id"]
    assert content_id("protocol", yaml.safe_load(yaml.safe_dump(p))) == p["protocol_id"]
    amended = {**p, "seeds": [1, 2, 3, 4], "protocol_version": 2}
    assert content_id("protocol", amended) != p["protocol_id"]
    with pytest.raises(ValueError, match="stale"):
        validate_contract("protocol", amended, h)


def test_missing_scientific_fields_and_reference_mismatch(contracts):
    h, p, _ = contracts
    with pytest.raises(ValueError, match="claim"):
        validate_contract("hypothesis", {k: v for k, v in h.items() if k != "claim"})
    with pytest.raises(ValueError, match="different hypothesis"):
        validate_contract("protocol", {**p, "hypothesis_id": "h-other"}, h)
    broken = copy.deepcopy(p)
    broken["evaluation"]["paired_factors"] = ["seed"]
    with pytest.raises(ValueError, match="differs from hypothesis"):
        validate_contract("protocol", broken, h)


def job(p, **overrides):
    settings = dict(
        task=p["variants"]["router"]["task"],
        variant="router",
        seed=1,
        iterations=3000,
        num_envs=2048,
        project=p["project"],
        group=p["group"],
        extra=p["variants"]["router"]["overrides"],
        protocol=p,
    )
    return rr.train_job(**{**settings, **overrides})


def test_run_metadata_retries_and_exact_protocol(contracts, tmp_path):
    _, p, _ = contracts
    first = job(p)
    assert first["research"]["protocol_id"] == p["protocol_id"]
    assert first["research"]["hypothesis_id"] == p["hypothesis_id"]
    retry = job(p, attempt=2, retry_reason="vm_died", original_settings_id=first["research"]["settings_id"])
    assert retry["name"] != first["name"]
    assert retry["research"]["cell_id"] == first["research"]["cell_id"]
    assert retry["research"]["settings_identical"] is None
    assert rr.parse_run_name(retry["name"]) == ("router", 1)
    with pytest.raises(ValueError, match="identical"):
        job(p, attempt=2, retry_reason="NaNs", original_settings_id="different")
    with pytest.raises(ValueError, match="differs"):
        job(p, iterations=2000)
    assert rr.protocol_job_problems([first, retry], p) == []
    altered = copy.deepcopy(first)
    altered["argv"].append("agent.algorithm.learning_rate=0.1")
    assert rr.protocol_job_problems([altered], p)
    queue_file = tmp_path / "jobs.jsonl"
    queue_file.write_text(json.dumps(retry) + "\n")
    assert rr.load_jobs(queue_file) == [retry]


def test_smoke_is_separate_and_never_scored(contracts):
    _, p, _ = contracts
    smoke = job(p, smoke=True, iterations=5)
    assert smoke["name"].endswith("-smoke") and "status:smoke" in smoke["argv"]
    assert smoke["research"]["job_type"] == "smoke"
    assert rr.protocol_job_problems([smoke], p) == []
    with pytest.raises(ValueError, match="cannot be scored"):
        rr.eval_jobs(task="e", checkpoint="c", name=smoke["name"], seeds=[41], output_root="o")


def test_fixed_k_declared_but_not_silently_executed(contracts):
    h, p, _ = contracts
    changed = copy.deepcopy(p)
    changed.pop("protocol_id")
    changed["evaluation"].update(episode_sampling="fixed_k_episodes_per_env", k=3)
    validate_contract("protocol", changed, h)
    with pytest.raises(ValueError, match="not supported"):
        ensure_supported_evaluation(changed)


def test_sampling_and_protocol_mismatch_rejected(tmp_path):
    for name, pid in (("a", "p-one"), ("b", "p-two")):
        _cell(tmp_path / name, 41, 0, research={"protocol_id": pid})
    result = rr.check_eval([tmp_path / "a", tmp_path / "b"], seeds=[41], levels=[0])
    assert not result["comparable"]


def contracted_publication(p, tmp_path, *, name="router-seed1"):
    """Full authentic-looking local evidence, without simulator or W&B access."""
    variant, _ = rr.parse_run_name(name)
    checkpoint = tmp_path / name / "model_3000.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"checkpoint bytes")
    jobs = rr.eval_jobs(
        task=p["variants"][variant]["eval_task"],
        checkpoint=str(checkpoint),
        name=name,
        seeds=p["evaluation"]["seeds"],
        levels=p["evaluation"]["levels"],
        difficulties=p["evaluation"]["difficulties"],
        output_root="o",
        num_envs=p["evaluation"]["num_envs"],
        num_episodes=1,
        extra=p["variants"][variant]["eval_overrides"],
        protocol=p,
    )
    cells, rows = [], {}
    for (level, difficulty, seed), j in zip(
        rr.planned_cells(p["evaluation"]["seeds"], p["evaluation"]["levels"], p["evaluation"]["difficulties"]),
        jobs,
        strict=True,
    ):
        cell = rr.cell_id(seed, level, difficulty)
        cells.append(
            {
                "cell": cell,
                "level": level,
                "difficulty": difficulty,
                "seed": seed,
                "num_envs": p["evaluation"]["num_envs"],
                "num_episodes": 1,
                "commit": "abc",
                "dirty": False,
                "profile": p["evaluation"]["profile"],
                "task": p["variants"][variant]["eval_task"],
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": rr.checkpoint_digest(checkpoint),
                "research": j["research"],
                "protocol_id": p["protocol_id"],
            }
        )
        rows[cell] = [
            {
                **r,
                "lin_vel_tracking_error": 0.2,
                "ang_vel_tracking_error": 0.3,
                "success": True,
                "timed_out": True,
            }
            for r in _rows_for(p["evaluation"]["num_envs"])
        ]

    for records in rows.values():
        for row in records:
            row.update({m: 0.2 for m in rr.required_eval_metrics(p)})

    def aggregate(rows):
        return {**_aggregate(rows), **{m: 0.2 for m in rr.required_eval_metrics(p)}}

    return dict(
        group=p["group"],
        name=name,
        task=p["variants"][variant]["task"],
        eval_task=p["variants"][variant]["eval_task"],
        checkpoint=str(checkpoint),
        output_root="o",
        cells=cells,
        rows_by_cell=rows,
        aggregate=aggregate,
        protocol=p,
        levels=p["evaluation"]["levels"],
        difficulties=p["evaluation"]["difficulties"],
    )


def test_paired_id_survives_publication(contracts, tmp_path):
    _, p, _ = contracts
    result = rr.build_eval_publication(**contracted_publication(p, tmp_path))
    ids = [r["eval_unit_id"] for r in result["episodes"]]
    assert len(set(ids)) == len(ids) == 900
    assert ids[0] == paired_unit_id(
        {"env_id": 0, "episode_id": 0}, {"seed": 41, "level": 0}, p["evaluation"]["paired_factors"]
    )
    with pytest.raises(ValueError, match="missing"):
        paired_unit_id({}, {}, ["command_trajectory"])


def test_runtime_fingerprint_on_unsupported_host(monkeypatch, tmp_path):
    import research_workflow as rw

    def unavailable(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(rw.subprocess, "run", unavailable)
    fp = runtime_fingerprint(tmp_path, {"task": "x"})
    assert fp["gpu_driver"] is None and fp["commit"] is None and fp["resolved_config_hash"]
    assert runtime_mismatches([fp, {**fp, "python": "different"}]) == ["python"]
    assert runtime_mismatches([fp, {}]) == []


def test_legacy_cells_remain_readable(tmp_path):
    _cell(tmp_path, 41, 0)
    verdict = rr.check_eval([tmp_path], seeds=[41], levels=[0])
    assert verdict["complete"] and verdict["comparable"]
    assert verdict["cells"][0]["protocol_id"] is None


def test_smoke_gate_requires_checkpoint_finite_metrics_and_clean_log(contracts, tmp_path):
    from test_research_runs import _records, _run_dir

    _, p, _ = contracts
    p = copy.deepcopy(p)
    p["smoke_gate"]["iterations"] = 3
    p["smoke_gate"]["log_roots"]["router"] = str(tmp_path / "logs")
    run = _run_dir(tmp_path, records=_records(3))
    run.rename(run.with_name("2026_router-seed1-smoke"))
    logs = tmp_path / "queue"
    logs.mkdir()
    (logs / "router-seed1-smoke.log").write_text("technical start and finish")
    jobs = [job(p, smoke=True, iterations=3)]
    run = tmp_path / "logs" / "2026_router-seed1-smoke"
    (run / "research_provenance.json").write_text(
        json.dumps({**jobs[0]["research"], "runtime_fingerprint": {"commit": "abc"}})
    )
    assert rr.check_smoke_gate(jobs, p, logs)["passed"]
    (logs / "router-seed1-smoke.log").write_text("Traceback (most recent call last)")
    assert not rr.check_smoke_gate(jobs, p, logs)["passed"]
    (logs / "router-seed1-smoke.log").write_text("technical start and finish")
    (tmp_path / "logs" / "2026_router-seed1-smoke" / "locomotion_metrics.jsonl").unlink()
    assert not rr.check_smoke_gate(jobs, p, logs)["passed"]


def test_pairing_missing_duplicate_population_invalidates_cell(contracts, tmp_path):
    _, p, _ = contracts
    for level, _, seed in rr.planned_cells(p["evaluation"]["seeds"], p["evaluation"]["levels"]):
        _cell(
            tmp_path,
            seed,
            level,
            n=150,
            task=p["variants"]["router"]["eval_task"],
            research={"protocol_id": p["protocol_id"], "protocol_contract": p},
        )
    assert rr.check_eval([tmp_path], seeds=[41, 42], levels=[0, 4, 8])["complete"]
    cell = tmp_path / "L0-seed41"
    rows = json.loads((cell / "evaluation_results.json").read_text())
    rows[1]["env_id"] = 0
    (cell / "evaluation_results.json").write_text(json.dumps(rows))
    result = rr.check_eval([tmp_path], seeds=[41, 42], levels=[0, 4, 8])
    assert not result["complete"]
    assert "duplicate paired evaluation unit" in result["cells"][0]["problems"]


def test_template_reports_actual_paired_effects(monkeypatch, contracts):
    from test_research_runs import _run_template
    from test_research_workflow_safety import contracted_template_world

    runs, artifacts = contracted_template_world(contracts)
    scope, _ = _run_template(monkeypatch, runs, artifacts, log=False)
    paired = scope["paired"]
    assert set(paired["metric"]) >= {
        "success_rate",
        "fall_rate",
        "lin_vel_episode_rmse",
        "ang_vel_episode_rmse",
    }
    assert set(paired["units"]) == {80}
    assert paired.loc[paired["metric"] == "lin_vel_episode_rmse", "mean_difference"].tolist() == pytest.approx(
        [0.1, 0.1]
    )
    assert not scope["PROBLEMS"]


@pytest.mark.parametrize("passed", [False, True])
def test_queue_automatically_runs_full_only_after_approved_smoke(contracts, tmp_path, monkeypatch, passed):
    _, p, _ = contracts
    jobs = [job(p)]
    jobs_file = tmp_path / "jobs.jsonl"
    jobs_file.write_text(json.dumps(jobs[0]) + "\n")
    monkeypatch.setattr(
        rr, "git_source", lambda: {"branch": "dev/agent/" + p["group"], "commit": "abc", "dirty": False}
    )
    monkeypatch.setattr(rr, "load_protocol", lambda path: p)
    monkeypatch.setattr(rr, "check_imports", lambda *args: [])
    monkeypatch.setattr(rr, "pin_launch", lambda *args, **kwargs: "agent/test/train-1")
    calls = []

    def queue(jobs, *args, **kwargs):
        calls.append(jobs)
        return 0

    monkeypatch.setattr(rr, "run_queue", queue)
    monkeypatch.setattr(rr, "check_smoke_gate", lambda *args, **kwargs: {"passed": passed})
    argv = [
        "queue",
        "--jobs",
        str(jobs_file),
        "--gpus",
        "0",
        "1",
        "--group",
        p["group"],
        "--log-dir",
        str(tmp_path / "queue"),
        "--smoke-first",
        "--no-push",
    ]
    if passed:
        assert rr.main(argv) == 0
        assert len(calls) == 2 and calls[1] == jobs
    else:
        with pytest.raises(SystemExit, match="full experiment not launched"):
            rr.main(argv)
        assert len(calls) == 1
    assert calls[0][0]["research"]["job_type"] == "smoke"


def test_reference_examples_validate():
    """The documented examples must stay valid for this repository (no levels or difficulties)."""
    examples = [yaml.safe_load(e) for e in re.findall(r"```yaml\n(.*?)```", REFERENCE.read_text(), re.S)]
    h, p, d = examples
    h = validate_contract("hypothesis", h)
    p["hypothesis_id"] = h["hypothesis_id"]
    p = validate_contract("protocol", p, h)
    assert validate_contract("decision", d) == d
    assert p["evaluation"]["levels"] == [] and p["evaluation"]["difficulties"] == []
    assert p["evaluation"]["profile"] == "general" and p["checkpoint_rule"] == "final"

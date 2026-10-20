"""Adversarial workflow boundaries: no simulator, GPU launches or W&B writes."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import test_research_contracts as contracts_tests
from test_research_contracts import contracted_publication, job
from test_research_runs import (
    FakeRun,
    FakeWandb,
    _AttachedEpisodes,
    _cell,
    _records,
    _run_dir,
    _run_template,
    _template_world,
    rr,
)

from research_workflow import content_id, validate_contract


@pytest.fixture
def contracts():
    return contracts_tests.contracts.__wrapped__()


def mock_queue(monkeypatch, p, tmp_path, jobs=None):
    jobs = jobs or [job(p)]
    jobs_file = tmp_path / "jobs.jsonl"
    jobs_file.write_text("".join(json.dumps(j) + "\n" for j in jobs))
    source = {"branch": "dev/agent/" + p["group"], "commit": "abc", "dirty": False}
    monkeypatch.setattr(rr, "git_source", lambda: dict(source))
    monkeypatch.setattr(rr, "load_protocol", lambda path: p)
    monkeypatch.setattr(rr, "check_imports", lambda *args: [])
    calls, pinned = [], []
    monkeypatch.setattr(rr, "pin_launch", lambda *a, **kw: pinned.append(a) or "mock-tag")
    monkeypatch.setattr(rr, "run_queue", lambda j, *a, **kw: calls.append(j) or 0)
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
        "--no-push",
    ]
    return argv, calls, pinned, source


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("passed", [False, True])
def test_smoke_is_enforced_even_without_flag(contracts, tmp_path, monkeypatch, explicit, passed):
    _, p, _ = contracts
    argv, calls, _, _ = mock_queue(monkeypatch, p, tmp_path)
    verdict_calls = []

    def verdict(*args, **kwargs):
        verdict_calls.append(kwargs)
        return {"passed": passed}

    monkeypatch.setattr(rr, "check_smoke_gate", verdict)
    if explicit:
        argv.append("--smoke-first")
    if passed:
        assert rr.main(argv) == 0
        assert [c[0]["research"]["job_type"] for c in calls] == ["smoke", "train"]
    else:
        with pytest.raises(SystemExit, match="full experiment not launched"):
            rr.main(argv)
        assert [c[0]["research"]["job_type"] for c in calls] == ["smoke"]
    assert verdict_calls == [{"commit": "abc"}]


@pytest.mark.parametrize("mode", ["no_auto_full", "mixed_jobs", "disabled_with_flag"])
def test_invalid_smoke_plans_stop_before_pinning(contracts, tmp_path, monkeypatch, mode):
    _, p, _ = contracts
    jobs = [job(p)]
    if mode == "no_auto_full":
        p["smoke_gate"]["auto_full"] = False
    elif mode == "mixed_jobs":
        jobs.append(job(p, smoke=True, iterations=5))
    else:
        p["smoke_gate"] = {"enabled": False, "auto_full": False}
    # Recompute the approved ID and corresponding job metadata after the mocked amendment.
    p["protocol_id"] = content_id("protocol", p)
    jobs = [job(p), *([job(p, smoke=True, iterations=5)] if mode == "mixed_jobs" else [])]
    argv, calls, pinned, _ = mock_queue(monkeypatch, p, tmp_path, jobs)
    if mode == "disabled_with_flag":
        argv.append("--smoke-first")
    with pytest.raises(SystemExit, match="not launching"):
        rr.main(argv)
    assert not calls and not pinned


def test_smoke_preserves_retry_identity_gpu_and_interpreter(contracts, tmp_path, monkeypatch):
    _, p, _ = contracts
    first = job(p)
    original = tmp_path / "original_provenance.json"
    original.write_text(
        json.dumps(
            {
                **first["research"],
                "runtime_fingerprint": {
                    "commit": "abc",
                    "config_hash_version": 2,
                    "resolved_config_hash": "config-hash",
                },
            }
        )
    )
    retry = job(
        p,
        attempt=2,
        retry_reason="vm_died",
        original_settings_id=first["research"]["settings_id"],
        original_provenance=str(original),
        gpu=1,
        python="python3",
    )
    argv, calls, _, _ = mock_queue(monkeypatch, p, tmp_path, [retry])
    monkeypatch.setattr(rr, "check_smoke_gate", lambda *a, **kw: {"passed": True})
    assert rr.main(argv) == 0
    smoke = calls[0][0]
    assert smoke["name"] == "router-seed1-attempt2-smoke"
    assert smoke["research"]["attempt"] == 2 and smoke["gpu"] == 1 and smoke["argv"][0] == "python3"


@pytest.mark.parametrize("failure", ["process", "dirty", "commit"])
def test_smoke_runtime_failure_or_source_change_blocks_full(contracts, tmp_path, monkeypatch, failure):
    _, p, _ = contracts
    argv, calls, _, source = mock_queue(monkeypatch, p, tmp_path)

    def smoke_jobs(jobs, *args, **kwargs):
        calls.append(jobs)
        if failure == "process":
            return 1
        source.update(dirty=failure == "dirty", commit="changed" if failure == "commit" else "abc")
        return 0

    monkeypatch.setattr(rr, "run_queue", smoke_jobs)
    monkeypatch.setattr(rr, "check_smoke_gate", lambda *a, **kw: {"passed": True})
    with pytest.raises(SystemExit, match="full experiment not launched"):
        rr.main(argv)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "field", ["protocol_id", "hypothesis_id", "cell_id", "attempt", "job_type", "settings_id", "commit"]
)
def test_smoke_cannot_reuse_other_attempt_or_commit(contracts, tmp_path, field):
    _, p, _ = contracts
    p["smoke_gate"]["iterations"] = 3
    p["smoke_gate"]["log_roots"]["router"] = str(tmp_path / "logs")
    smoke = job(p, smoke=True, iterations=3)
    run = _run_dir(tmp_path, records=_records(3))
    run = run.rename(run.with_name("2026_" + smoke["name"]))
    provenance = {**smoke["research"], "runtime_fingerprint": {"commit": "abc"}}
    if field == "commit":
        provenance["runtime_fingerprint"][field] = "different"
    else:
        provenance[field] = "different"
    (run / "research_provenance.json").write_text(json.dumps(provenance))
    log = tmp_path / "queue"
    log.mkdir()
    (log / f"{smoke['name']}.log").write_text("finished")
    assert not rr.check_smoke_gate([smoke], p, log, commit="abc")["passed"]


def write_evaluations(root, p):
    for level, _, seed in rr.planned_cells(p["evaluation"]["seeds"], p["evaluation"]["levels"]):
        _cell(
            root,
            seed,
            level,
            n=p["evaluation"]["num_envs"],
            task=p["variants"]["router"]["eval_task"],
            research={"protocol_id": p["protocol_id"], "protocol_contract": p},
        )


@pytest.mark.parametrize("explicit", [False, True])
def test_partial_evaluation_never_becomes_complete(contracts, tmp_path, explicit):
    _, p, _ = contracts
    _cell(
        tmp_path,
        41,
        0,
        n=150,
        task=p["variants"]["router"]["eval_task"],
        research={"protocol_id": p["protocol_id"], "protocol_contract": p},
    )
    result = rr.check_eval([tmp_path], seeds=[41], levels=[0], protocol=p if explicit else None)
    assert not result["complete"] and len(result["cells"]) == 6
    assert "requested evaluation matrix differs" in result["problems"][0]


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), True])
def test_declared_metrics_are_required_without_cli_metric_list(contracts, tmp_path, value):
    _, p, _ = contracts
    write_evaluations(tmp_path, p)
    file = tmp_path / "L0-seed41/evaluation_summary.json"
    summary = json.loads(file.read_text())
    summary["metrics"][p["evaluation"]["primary_metric"]["name"]] = value
    file.write_text(json.dumps(summary))
    result = rr.check_eval([tmp_path], seeds=[41, 42], levels=[0, 4, 8])
    assert not result["complete"]
    assert any("metric lin_vel_tracking_rmse" in problem for problem in result["cells"][0]["problems"])


@pytest.mark.parametrize(
    "column,value", [("lin_vel_tracking_error", float("nan")), ("lin_vel_tracking_error", -1), ("seed", 99)]
)
def test_invalid_scored_rows_cannot_hide_behind_finite_summary(contracts, tmp_path, column, value):
    _, p, _ = contracts
    write_evaluations(tmp_path, p)
    file = tmp_path / "L0-seed41/evaluation_results.json"
    rows = json.loads(file.read_text())
    rows[0][column] = value
    file.write_text(json.dumps(rows))
    result = rr.check_eval([tmp_path], seeds=[41, 42], levels=[0, 4, 8])
    assert not result["complete"]


@pytest.mark.parametrize(
    "bad",
    [
        "missing",
        "duplicate",
        "population",
        "wrong_task",
        "dirty",
        "attempt",
        "hash",
        "profile",
        "env_id",
        "pairing",
    ],
)
def test_publication_independently_checks_evidence(contracts, tmp_path, bad):
    _, p, _ = contracts
    args = contracted_publication(p, tmp_path)
    cell = args["cells"][0]
    if bad == "missing":
        args["cells"].pop()
    elif bad == "duplicate":
        args["cells"].append(copy.deepcopy(cell))
    elif bad == "population":
        cell["num_envs"] = 4
    elif bad == "wrong_task":
        args["eval_task"] = "wrong-eval-task"
    elif bad == "dirty":
        cell["dirty"] = True
    elif bad == "attempt":
        cell["research"]["attempt"] = 2
    elif bad == "hash":
        Path(args["checkpoint"]).write_bytes(b"different bytes")
    elif bad == "profile":
        cell["profile"] = "forward_course"
    elif bad == "env_id":
        args["rows_by_cell"][cell["cell"]][0]["env_id"] = 999
    else:
        args["rows_by_cell"][cell["cell"]][0]["env_id"] = 1
    with pytest.raises(ValueError):
        rr.build_eval_publication(**args)


@pytest.mark.parametrize("metric", ["primary", "secondary"])
@pytest.mark.parametrize("value", [None, float("nan"), float("inf")])
def test_missing_scored_metric_is_not_silently_dropped(contracts, tmp_path, metric, value):
    _, p, _ = contracts
    args = contracted_publication(p, tmp_path)
    key = p["evaluation"]["primary_metric"]["name"] if metric == "primary" else p["evaluation"]["secondary_metrics"][0]
    aggregate = args["aggregate"]
    args["aggregate"] = lambda rows: {**aggregate(rows), key: value}
    if metric == "secondary":
        column = p["evaluation"]["metric_reducers"][key]["column"]
        next(iter(args["rows_by_cell"].values()))[0][column] = value
    with pytest.raises(ValueError, match="required scored metrics|scored column"):
        rr.build_eval_publication(**args)


def valid_training_link(p, tmp_path, *, name="router-seed1", nested=True):
    args = contracted_publication(p, tmp_path, name=name)
    payload = rr.build_eval_publication(**args)
    meta = {
        **job(p)["research"],
        "commit": "abc",
        "dirty": False,
        "run_dir": str(Path(args["checkpoint"]).parent),
        "runtime_fingerprint": {"commit": "abc"},
    }
    if "attempt2" in name:
        meta["attempt"] = 2
    (Path(args["checkpoint"]).parent / "research_provenance.json").write_text(json.dumps(meta))
    config = {"train_cfg": {"research": meta}} if nested else {"research": meta}
    train = FakeRun("2026_" + name, "train", run_id="train1", state="finished", config=config)
    return payload, train, meta


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("name", ["router-seed1", "router-seed1-attempt2"])
def test_publication_links_exact_verified_training_attempt(contracts, tmp_path, nested, name):
    _, p, _ = contracts
    payload, train, _ = valid_training_link(p, tmp_path, name=name, nested=nested)
    fake = FakeWandb([train])
    assert rr.publish_eval(payload, p["project"], train_run_id="train1", wandb_module=fake)
    assert fake.inits[0]["config"]["research"]["train_run_id"] == "train1"
    assert fake.artifacts[0].name == "eval-episodes-" + name


@pytest.mark.parametrize(
    "bad",
    [
        "name",
        "seed",
        "variant",
        "attempt",
        "commit",
        "dirty",
        "run_dir",
        "sidecar",
        "bytes",
        "state",
        "smoke",
    ],
)
def test_wrong_training_link_stops_before_wandb_init(contracts, tmp_path, bad):
    _, p, _ = contracts
    payload, train, meta = valid_training_link(p, tmp_path)
    if bad == "name":
        train.name = "2026_fixed-seed3"
    elif bad in ("seed", "variant", "attempt", "commit", "dirty", "run_dir"):
        meta[bad] = {
            "seed": 3,
            "variant": "fixed",
            "attempt": 2,
            "commit": "other",
            "dirty": True,
            "run_dir": "other",
        }[bad]
    elif bad == "sidecar":
        (Path(meta["run_dir"]) / "research_provenance.json").unlink()
    elif bad == "bytes":
        Path(payload["config"]["research"]["checkpoint"]).write_bytes(b"replaced")
    elif bad == "state":
        train.state = "failed"
    else:
        train.tags = ["status:smoke"]
    fake = FakeWandb([train])
    with pytest.raises(SystemExit):
        rr.publish_eval(payload, p["project"], train_run_id="train1", wandb_module=fake)
    assert not fake.inits and not fake.artifacts


def paired_template_world():
    runs, artifacts = _template_world()
    for frame in artifacts.values():
        frame["eval_unit_id"] = [f"{cell}/env{i % 40}" for i, cell in enumerate(frame["cell"])]
    return runs, artifacts


def test_template_uses_retry_artifact_attached_to_exact_eval_run(monkeypatch):
    runs, artifacts = paired_template_world()
    original = next(r for r in runs if r.name == "2026_mlp-seed1")
    retry = copy.copy(original)
    retry.name, retry.id = "2026_mlp-seed1-attempt2", "retry2"
    runs.append(retry)
    original.state = "failed"
    eval_run = next(r for r in runs if r.name == "mlp-seed1__eval")
    eval_run.name = "mlp-seed1-attempt2__eval"
    chosen = artifacts["eval-episodes-mlp-seed1"].copy()
    chosen["lin_vel_tracking_error"] = 0.9
    eval_run.logged_artifacts = lambda: [_AttachedEpisodes("eval-episodes-mlp-seed1-attempt2:v7", chosen)]
    # Mutable aliases deliberately disagree; the notebook must never consult them.
    artifacts["eval-episodes-mlp-seed1"]["lin_vel_tracking_error"] = 99
    scope, _ = _run_template(monkeypatch, runs, artifacts, log=False)
    actual = scope["per_seed"].query("variant == 'mlp' and train_seed == 1 and slice == 'ALL'")
    assert actual["lin_vel_rmse"].iloc[0] == pytest.approx(0.9)
    assert len(scope["TABLES"]["attempts"]) == 5 and len(scope["inventory"]) == 4
    assert any(r["artifact"].endswith("attempt2:v7") for r in scope["EVIDENCE_ARTIFACTS"])


@pytest.mark.parametrize("bad", ["missing", "multiple", "alias", "wrong_attempt", "nan", "negative"])
def test_template_rejects_unverified_artifacts(monkeypatch, bad):
    runs, artifacts = paired_template_world()
    eval_run = next(r for r in runs if r.job_type == "eval")
    attached = list(eval_run.logged_artifacts())
    if bad == "missing":
        attached = []
    elif bad == "multiple":
        attached *= 2
    elif bad == "alias":
        attached[0].name = attached[0].name.replace(":v0", ":latest")
    elif bad == "wrong_attempt":
        attached[0].name = "eval-episodes-lstm-seed1-attempt2:v0"
    elif bad == "nan":
        attached[0].frame.loc[0, "lin_vel_tracking_error"] = float("nan")
    else:
        attached[0].frame.loc[0, "lin_vel_tracking_error"] = -1
    eval_run.logged_artifacts = lambda: attached
    with pytest.raises(ValueError):
        _run_template(monkeypatch, runs, artifacts, log=False)


def test_paired_episode_rmse_is_distinct_from_pooled_rmse(monkeypatch, contracts):
    runs, artifacts = contracted_template_world(contracts)
    for key, frame in artifacts.items():
        frame["lin_vel_tracking_error"] = [0.0, 2.0] * 40 if "mlp" in key else 1.0
    scope, _ = _run_template(monkeypatch, runs, artifacts, log=False)
    paired = scope["paired"].query("metric == 'lin_vel_episode_rmse'")
    assert paired["mean_difference"].tolist() == pytest.approx([0, 0])
    assert paired["aggregate_difference"].tolist() == pytest.approx([2**0.5 - 1] * 2)
    assert set(paired["aggregate_metric"]) == {"lin_vel_rmse"}
    assert paired["fraction_positive"].tolist() == [0.5, 0.5]
    assert paired["fraction_negative"].tolist() == [0.5, 0.5]


def test_incomplete_pairing_produces_no_paired_statistics(monkeypatch):
    runs, artifacts = paired_template_world()
    frame = artifacts["eval-episodes-mlp-seed1"]
    frame.loc[0, "eval_unit_id"] = "unmatched"
    scope, _ = _run_template(monkeypatch, runs, artifacts, log=False)
    assert scope["paired"].empty
    assert any("pairing incomplete" in problem for problem in scope["PROBLEMS"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("levels", [0, 0]),
        ("difficulties", ["hard", "hard"]),
        ("required_metrics", [""]),
        ("metric_columns", {"m": 1}),
    ],
)
def test_schema_rejects_ambiguous_evaluation_evidence(contracts, field, value):
    h, p, _ = contracts
    p["evaluation"][field] = value
    p.pop("protocol_id")
    with pytest.raises(ValueError):
        validate_contract("protocol", p, h)


def test_smoke_only_protocol_allows_smoke_but_no_full_launch(contracts, tmp_path, monkeypatch):
    _, p, _ = contracts
    p["smoke_gate"]["auto_full"] = False
    p["protocol_id"] = content_id("protocol", p)
    smoke = job(p, smoke=True, iterations=5)
    argv, calls, _, _ = mock_queue(monkeypatch, p, tmp_path, [smoke])
    assert rr.main(argv) == 0
    assert calls == [[smoke]]


def test_source_change_while_waiting_blocks_pending_launch(tmp_path, monkeypatch):
    from unittest.mock import Mock

    clean = {"commit": "abc", "branch": "dev/agent/group", "dirty": False}
    sources = iter([clean, {**clean, "dirty": True}])
    monkeypatch.setattr(rr, "git_source", lambda: next(sources))
    spawn = Mock()
    monkeypatch.setattr(rr.subprocess, "Popen", spawn)
    with pytest.raises(SystemExit, match="source changed while queued"):
        rr.run_queue(
            [{"name": "x", "argv": ["python"], "gpu": None, "research": {"protocol_id": "p-test"}}],
            [0],
            tmp_path,
            occupancy=lambda: set(),
        )
    spawn.assert_not_called()


@pytest.mark.parametrize("kind", ["extra_required", "custom_row", "missing_custom"])
def test_custom_required_metrics_have_finite_scored_inputs(contracts, tmp_path, kind):
    h, p, _ = contracts
    p["evaluation"]["required_metrics"] = ["safety_metric"]
    p["evaluation"]["metric_columns"] = {"safety_metric": "episode_safety"}
    p["evaluation"]["metric_reducers"]["safety_metric"] = {"column": "episode_safety", "reduction": "mean"}
    p.pop("protocol_id")
    p = validate_contract("protocol", p, h)
    args = contracted_publication(p, tmp_path)
    for rows in args["rows_by_cell"].values():
        for row in rows:
            row["episode_safety"] = 1.0
    if kind != "extra_required":
        row = next(iter(args["rows_by_cell"].values()))[0]
        if kind == "custom_row":
            row["episode_safety"] = float("nan")
        else:
            row.pop("episode_safety")
        with pytest.raises(ValueError, match="episode_safety"):
            rr.build_eval_publication(**args)
    else:
        assert rr.build_eval_publication(**args)["summary"]["eval/safety_metric"] == 1.0


def test_explicit_wrong_legacy_training_link_is_also_rejected():
    from test_research_runs import _publication

    fake = FakeWandb([FakeRun("fixed-seed3", "train", run_id="wrong")])
    with pytest.raises(SystemExit, match="run name"):
        rr.publish_eval(_publication(), "rwm-explore", train_run_id="wrong", wandb_module=fake)
    assert not fake.inits


def test_duplicate_evaluated_attempts_are_not_selected_by_score(monkeypatch):
    runs, artifacts = paired_template_world()
    duplicate = copy.copy(next(r for r in runs if r.job_type == "eval"))
    duplicate.id = "another-evaluation"
    runs.append(duplicate)
    with pytest.raises(ValueError, match="multiple evaluated attempts"):
        _run_template(monkeypatch, runs, artifacts, log=False)


def test_template_validates_exact_training_link_for_new_contracts(contracts, tmp_path, monkeypatch):
    _, p, _ = contracts
    runs, artifacts = paired_template_world()
    # Add Unicode to exercise the same canonical JSON encoding as content_id.
    p["assumptions"].append("ASSUMPTION terrain μ")
    p["protocol_id"] = content_id("protocol", p)
    payload, _, _ = valid_training_link(p, tmp_path)
    eval_run = next(r for r in runs if r.job_type == "eval")
    eval_run.config = payload["config"]
    eval_run.config["research"]["train_run_id"] = "does-not-exist"
    with pytest.raises(ValueError, match="exact training attempt is missing"):
        _run_template(monkeypatch, runs, artifacts, log=False)


def contracted_template_world(contracts):
    from research_workflow import paired_unit_id

    _, p, _ = contracts
    p = copy.deepcopy(p)
    p.update(
        group="g",
        seeds=[1, 2],
        training_budget={"iterations": 100, "num_envs": 2048},
        baseline_variants=["lstm"],
    )
    p["variants"] = {"lstm": p["variants"]["fixed"], "mlp": p["variants"]["router"]}
    p["evaluation"].update(seeds=[41], levels=[0, 5], num_envs=40)
    p["reporting"] = {
        "compatibility": {"dataset": "matched-rl-fixture"},
        "metrics": [
            {
                "name": "eval/" + name,
                "scope": "eval",
                "unit": unit,
                "direction": None,
                "aggregation": "mean_seed",
                "reduction": "mean",
            }
            for name, unit in [
                ("preference_variance", "squared fraction"),
                ("preference_entropy", "nats"),
                ("inference_latency_ms", "ms"),
            ]
        ],
    }
    p["protocol_id"] = content_id("protocol", p)
    runs, artifacts = _template_world()
    for r in runs:
        rs = (r.config.get("train_cfg") or {}).get("research") or r.config["research"]
        variant, seed = rs["variant"], rs["seed"]
        rs.update(
            protocol_id=p["protocol_id"],
            hypothesis_id=p["hypothesis_id"],
            protocol_version=1,
            protocol_contract=p,
            cell_id=f"{variant}-seed{seed}",
            attempt=1,
            task=p["variants"][variant]["task"],
            commit="abc",
            commits=["abc"],
        )
        if r.job_type == "eval":
            rs["train_run_id"] = next(t.id for t in runs if t.name == f"2026_{variant}-seed{seed}")
    for frame in artifacts.values():
        for spec in p["evaluation"].get("metric_reducers", {}).values():
            frame[spec["column"]] = 0.2
        frame["env_id"] = list(range(40)) * 2
        frame["eval_unit_id"] = [
            paired_unit_id(
                row,
                {"seed": 41, "level": int(row["cell"].split("-")[0][1:])},
                p["evaluation"]["paired_factors"],
            )
            for row in frame.to_dict("records")
        ]
    return runs, artifacts


def test_new_protocol_template_runs_with_verified_artifact_population(contracts, monkeypatch):
    runs, artifacts = contracted_template_world(contracts)
    scope, _ = _run_template(monkeypatch, runs, artifacts, log=False)
    assert not scope["PROBLEMS"]
    assert scope["BASE"] == "lstm" and len(scope["EVIDENCE_ARTIFACTS"]) == 4
    assert len(scope["paired"]) == 14


@pytest.mark.parametrize("bad", ["unit_id", "population", "cell", "protocol"])
def test_new_protocol_template_rejects_corrupt_attached_evidence(contracts, monkeypatch, bad):
    runs, artifacts = contracted_template_world(contracts)
    if bad == "protocol":
        next(r for r in runs if r.job_type == "eval").config["research"]["protocol_contract"]["seeds"] = [9]
    else:
        frame = artifacts["eval-episodes-lstm-seed1"]
        if bad == "unit_id":
            frame.loc[0, "eval_unit_id"] = "fabricated"
        elif bad == "population":
            frame.loc[0, "env_id"] = 1
        else:
            frame.loc[0, "cell"] = "L9-seed41"
    with pytest.raises(ValueError):
        _run_template(monkeypatch, runs, artifacts, log=False)


def test_legacy_flag_cannot_bypass_smoke_or_protocol_checks(contracts, tmp_path, monkeypatch):
    _, p, _ = contracts
    argv, calls, pinned, _ = mock_queue(monkeypatch, p, tmp_path)
    argv.append("--legacy-workflow")
    with pytest.raises(SystemExit, match="legacy-workflow cannot bypass"):
        rr.main(argv)
    assert not calls and not pinned


def test_publication_cannot_downgrade_contract_evidence_to_legacy(contracts, tmp_path):
    _, p, _ = contracts
    args = contracted_publication(p, tmp_path)
    args.pop("protocol")
    args["cells"].pop()
    with pytest.raises(ValueError, match="complete approved evaluation matrix"):
        rr.build_eval_publication(**args)
    for c in args["cells"]:
        c["research"].pop("protocol_contract", None)
    with pytest.raises(ValueError, match="cannot downgrade to legacy"):
        rr.build_eval_publication(**args)


def test_explicit_evaluation_selection_uses_reviewed_ids(monkeypatch):
    runs, artifacts = paired_template_world()
    selected = [r.id for r in runs if r.job_type == "eval"]
    duplicate = copy.copy(next(r for r in runs if r.job_type == "eval"))
    duplicate.id = "other-version"
    runs.append(duplicate)
    scope, _ = _run_template(monkeypatch, runs, artifacts, log=False, eval_run_ids=selected)
    assert len(scope["eval_info"]) == 4
    assert "other-version" not in {r["eval_run_id"] for r in scope["EVIDENCE_ARTIFACTS"]}
    with pytest.raises(ValueError, match="duplicate or unavailable"):
        _run_template(monkeypatch, runs, artifacts, log=False, eval_run_ids=["missing"])

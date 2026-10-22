"""Controlled source fixes retain actual attempt lineage without forging retry provenance."""

import copy
import hashlib
import json
import subprocess

import pytest

from research_workflow.verification import verify_controlled_startup, verify_retry


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def replacement(tmp_path):
    def git(*args):
        return (
            subprocess.check_output(
                ["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", *args],
                cwd=tmp_path,
                stderr=subprocess.PIPE,
            )
            .decode()
            .strip()
        )

    git("init", "-q")
    bound = {}
    for name in ("train.py", "core.py", "dataset.py", "kinematics.py"):
        p = tmp_path / "src/world_modeling" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# fixed numerical source\n")
        bound[str(p.relative_to(tmp_path))] = digest(p)
    git("add", "src")
    git("commit", "-qm", "original")
    old_commit = git("rev-parse", "HEAD")
    (tmp_path / "startup-guard.txt").write_text("controlled infrastructure repair\n")
    git("add", "startup-guard.txt")
    git("commit", "-qm", "new guard")
    new_commit = git("rev-parse", "HEAD")
    old = {
        "domain": "world_model",
        "group": "startup",
        "project": "rwm-explore",
        "protocol_id": "p-old",
        "protocol_version": 3,
        "hypothesis_id": "h-fixed",
        "seeds": [33],
        "variants": {"B0": {"task": "Task", "eval_task": "Play", "overrides": [], "eval_overrides": []}},
        "training_budget": {"iterations": 20, "num_envs": 4},
        "smoke_gate": {"enabled": True, "iterations": 2},
        "retry_policy": {"max_attempts": 2, "allowed_reasons": ["vm_died"]},
        "evaluation": {"primary_metric": {"name": "lin_vel_tracking_rmse"}, "secondary_metrics": []},
        "world_model": {"algorithm_commit": "unchanged-algorithm", "training_context_admission": {"old": True}},
    }
    identity = {"cell_id": "B0-seed33", "variant": "B0", "seed": 33, "task": "Task", "hypothesis_id": "h-fixed"}
    q = tmp_path / "logs/queue/startup"
    q.mkdir(parents=True)
    name = "B0-seed33-smoke"
    events = [
        {
            "event": "start",
            "name": name,
            "source": {"commit": old_commit, "dirty": False},
            "research": {
                **identity,
                "attempt": 1,
                "job_type": "smoke",
                "protocol_id": "p-old",
                "protocol_contract": old,
            },
        },
        {"event": "isolation_failed", "reason": "startup heartbeat missing"},
        {"event": "exit", "name": name, "returncode": -15},
        {"event": "queue_end", "failures": 1},
    ]
    ledger = q / "old-events.jsonl"
    ledger.write_text("".join(json.dumps(x) + "\n" for x in events))
    child = q / (name + ".log")
    child.write_text("")
    audit = q / "failure.json"
    audit.write_text(
        json.dumps(
            {
                "source_commit": old_commit,
                "original_protocol": "p-old",
                "exit_codes": {name: -15},
                "child_log_bytes": {name: 0},
                "training_output_directories": {name: []},
                "no_automatic_retry": True,
                "files": {"child": {"path": str(child.relative_to(tmp_path)), "sha256": digest(child)}},
            }
        )
    )
    declared = {
        "kind": "changed_source_pretraining_startup_replacement",
        "attempt": 2,
        "original_attempt": 1,
        "original_job_name": name,
        "original_protocol_id": "p-old",
        "original_source_commit": old_commit,
        "closed_queue": str(ledger.relative_to(tmp_path)),
        "closed_queue_sha256": digest(ledger),
        "failure_audit": str(audit.relative_to(tmp_path)),
        "failure_audit_sha256": digest(audit),
        "ordinary_retry": False,
        "settings_identical": False,
    }
    p = copy.deepcopy(old)
    p.update(protocol_id="p-new", protocol_version=4)
    p["world_model"].update(
        startup_correction={"unchanged_model_source_sha256": bound},
        controlled_startup_replacements={"B0-seed33": declared},
        pilot_queue_directory="smoke-new",
        pilot_trace_filename="new.jsonl.gz",
        training_context_admission={"fixed": True},
    )
    meta = {
        **identity,
        "attempt": 2,
        "job_type": "smoke",
        "protocol_id": "p-new",
        "settings_identical": False,
        "controlled_startup_replacement": declared,
        "protocol_contract": p,
    }
    return tmp_path, meta, {"commit": new_commit}, ledger, audit


def test_reviewed_source_change_uses_actual_queue_lineage_without_fake_original_provenance(replacement):
    root, meta, fp, _, _ = replacement
    result = verify_controlled_startup(meta, fp, root=root)
    assert result["controlled_startup_replacement_validated"] is True
    assert result["settings_identical"] is False
    assert "original_provenance" not in result
    assert meta["controlled_startup_replacement"]["original_attempt"] == 1


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m.update(job_type="train"),
        lambda m: m.update(attempt=1),
        lambda m: m.update(attempt=3),
        lambda m: m.update(original_provenance="fabricated.json"),
        lambda m: m.update(settings_identical=True),
        lambda m: m.update(seed=44),
        lambda m: m["protocol_contract"]["training_budget"].update(iterations=21),
    ],
)
def test_unapproved_attempts_or_scientific_changes_cannot_use_startup_boundary(replacement, change):
    root, meta, fp, _, _ = replacement
    change(meta)
    with pytest.raises(ValueError):
        verify_controlled_startup(meta, fp, root=root)


def test_old_source_is_not_mislabeled_as_changed_source(replacement):
    root, meta, _, _, _ = replacement
    with pytest.raises(ValueError, match="new source commit"):
        verify_controlled_startup(
            meta, {"commit": meta["controlled_startup_replacement"]["original_source_commit"]}, root=root
        )


def test_closed_evidence_tamper_fails(replacement):
    root, meta, fp, ledger, _ = replacement
    ledger.write_text(ledger.read_text() + "{}\n")
    with pytest.raises(ValueError, match="evidence changed"):
        verify_controlled_startup(meta, fp, root=root)


def test_claiming_new_model_bytes_unchanged_is_rejected_against_original_git_commit(replacement):
    root, meta, fp, _, _ = replacement
    path = root / "src/world_modeling/core.py"
    path.write_text("# actually changed model\n")
    meta["protocol_contract"]["world_model"]["startup_correction"]["unchanged_model_source_sha256"][
        "src/world_modeling/core.py"
    ] = digest(path)
    with pytest.raises(ValueError, match="numerical source changed"):
        verify_controlled_startup(meta, fp, root=root)


def test_job_constructor_marks_only_declared_pilot_attempt_two_and_preserves_full_attempt_one(replacement):
    from test_research_runs import rr

    _, meta, _, _, _ = replacement
    p = meta["protocol_contract"]
    common = {
        "task": "Task",
        "variant": "B0",
        "seed": 33,
        "project": "rwm-explore",
        "group": "startup",
        "extra": [],
        "num_envs": 4,
        "protocol": p,
    }
    pilot = rr.train_job(**common, iterations=2, smoke=True)
    assert pilot["name"] == "B0-seed33-attempt2-smoke"
    assert pilot["research"]["attempt"] == 2
    assert pilot["research"]["settings_identical"] is False
    assert not rr.protocol_job_problems([pilot], p)
    full = rr.train_job(**common, iterations=20)
    assert full["name"] == "B0-seed33"
    assert full["research"]["attempt"] == 1
    with pytest.raises(ValueError, match="declared attempt"):
        rr.train_job(**common, iterations=2, smoke=True, attempt=3)


def test_ordinary_attempt_two_still_requires_actual_original_provenance():
    with pytest.raises(ValueError, match="original-provenance"):
        verify_retry({"attempt": 2}, {"commit": "new"})


def test_actual_original_output_cannot_be_hidden_by_stale_audit_fields(replacement):
    root, meta, fp, _, _ = replacement
    old_output = root / "logs/rsl_rl/anymal_d_world_model/2026-01-01_B0-seed33-smoke"
    old_output.mkdir(parents=True)
    with pytest.raises(ValueError, match="original startup has run output"):
        verify_controlled_startup(meta, fp, root=root)


@pytest.fixture
def third_replacement(replacement):
    root, previous_meta, previous_fp, _, _ = replacement
    q = root / "logs/queue/startup"
    name = "B0-seed33-attempt2-smoke"
    ledger = q / "second-events.jsonl"
    events = [
        {
            "event": "start",
            "name": name,
            "source": {"commit": previous_fp["commit"], "dirty": False},
            "research": previous_meta,
        },
        {"event": "isolation_failed", "reason": "missed process observation interval"},
        {"event": "exit", "name": name, "returncode": -15},
        {"event": "queue_end", "failures": 1},
    ]
    ledger.write_text("".join(json.dumps(e) + "\n" for e in events))
    child = q / (name + ".log")
    child.write_text("")
    audit = q / "second-failure.json"
    audit.write_text(
        json.dumps(
            {
                "source_commit": previous_fp["commit"],
                "protocol_id": "p-new",
                "exit_codes": {name: -15},
                "child_log_bytes": {name: 0},
                "training_output_directories": {name: []},
                "files": {"child": {"path": str(child.relative_to(root)), "sha256": digest(child)}},
            }
        )
    )
    meta = copy.deepcopy(previous_meta)
    protocol = meta["protocol_contract"]
    protocol.update(protocol_id="p-third", protocol_version=5)
    protocol["retry_policy"]["max_attempts"] = 3
    protocol["world_model"]["startup_correction"]["user_restart"] = {
        "actual_user_instruction": "Start it now on gpu 2 and 3",
        "maximum_ordinary_attempts": 2,
    }
    declared = protocol["world_model"]["controlled_startup_replacements"]["B0-seed33"]
    declared.update(
        attempt=3,
        user_authorized_additional_start=True,
        previous_attempt_record={
            "attempt": 2,
            "job_name": name,
            "source_commit": previous_fp["commit"],
            "protocol_id": "p-new",
            "closed_queue": str(ledger.relative_to(root)),
            "closed_queue_sha256": digest(ledger),
            "failure_audit": str(audit.relative_to(root)),
            "failure_audit_sha256": digest(audit),
        },
    )
    meta.update(attempt=3, protocol_id="p-third", controlled_startup_replacement=declared)
    (root / "user-restart.txt").write_text("approved additional startup only\n")
    subprocess.check_call(["git", "add", "user-restart.txt"], cwd=root)
    subprocess.check_call(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "reviewed third source",
        ],
        cwd=root,
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip()
    return root, meta, {"commit": commit}, ledger


def test_user_authorized_third_start_retains_both_actual_failed_lineages(third_replacement):
    root, meta, fingerprint, _ = third_replacement
    assert verify_controlled_startup(meta, fingerprint, root=root)["controlled_startup_replacement_validated"]
    previous = meta["controlled_startup_replacement"]["previous_attempt_record"]
    with pytest.raises(ValueError, match="new reviewed source"):
        verify_controlled_startup(meta, {"commit": previous["source_commit"]}, root=root)


@pytest.mark.parametrize("mutation", ["authority", "previous_attempt", "closed_evidence"])
def test_third_start_cannot_skip_user_authority_or_second_attempt(third_replacement, mutation):
    root, meta, fp, ledger = third_replacement
    if mutation == "authority":
        meta["protocol_contract"]["world_model"]["startup_correction"]["user_restart"].clear()
    elif mutation == "previous_attempt":
        meta["controlled_startup_replacement"]["previous_attempt_record"]["attempt"] = 1
    else:
        ledger.write_text(ledger.read_text() + "\n")
    with pytest.raises(ValueError):
        verify_controlled_startup(meta, fp, root=root)


def test_ordinary_retry_cannot_use_exceptional_third_start(third_replacement):
    _, meta, fp, _ = third_replacement
    meta.pop("controlled_startup_replacement")
    with pytest.raises(ValueError, match="ordinary retry exceeds"):
        verify_retry(meta, fp)

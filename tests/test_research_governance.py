"""Gate authority, artifact freshness, reserved budgets and dashboards; no external writes."""

from __future__ import annotations

import copy
import io
import json
import subprocess
import yaml
from contextlib import contextmanager

import pytest
import test_research_contracts as contract_tests
from test_research_runs import FakeArtifact, FakeReport, FakeRun, FakeWandb, rr

from research_workflow import validate_contract
from research_workflow.governance import (
    check_mandate,
    dashboard_markdown,
    decision_dashboard,
    implementation_id,
    launch_approval_problems,
    read_receipts,
    read_workflow,
    record_gate,
    reservations,
    subjects,
    validate_workflow,
)


def write(path, document):
    path.write_text(yaml.safe_dump(document, sort_keys=False))


@pytest.fixture
def comparison(tmp_path):
    h, p, d = copy.deepcopy(contract_tests.contracts.__wrapped__())
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    directory = tmp_path / "research" / h["group"]
    directory.mkdir(parents=True)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "model.py").write_text("width = 64\n")
    write(directory / "hypothesis.yaml", h)
    write(directory / "protocol.yaml", p)
    write(directory / "implementation.yaml", {"variants": p["variants"]})
    d["based_on_protocol"] = p["protocol_id"]
    write(directory / "decision.yaml", d)
    workflow = {
        "schema_version": 1,
        "group": h["group"],
        "mode": "unattended",
        "mandate": {
            "authorized_by": "user",
            "objective": h["claim"],
            "allowed_changes": "Router only.",
            "stop_when": "Decision or budget exhausted.",
            "limits": {
                "training_unit": "iterations",
                "max_variants": 2,
                "max_training_seeds": 3,
                "max_attempts": 2,
                "max_followups": 1,
                "max_total_training_units": 80000,
                "max_total_evaluation_units": 30000,
                "compute_pool": [0, 1],
            },
            "permissions": {"commit": True, "push": True, "finalize": True},
        },
    }
    write(directory / "workflow.yaml", workflow)
    return tmp_path, directory, p


def proposal(root, directory, gate, verdict="approve", **changes):
    workflow = read_workflow(directory)
    return {
        "gate": gate,
        "actor": "researcher" if workflow["mode"] == "unattended" else "user",
        "verdict": verdict,
        "workflow_id": workflow["workflow_id"],
        "subjects": subjects(root, directory, gate),
        "reason": "Matched comparison with sufficient verification.",
        "risk": "Low seed count.",
        "next_action": "Proceed to the next approved stage.",
        "evidence": ["https://wandb.ai/team/project/reports/evidence"],
        **changes,
    }


def approve_abc(root, directory):
    for gate in "ABC":
        record_gate(root, directory, proposal(root, directory, gate))


def test_default_attended_and_explicit_delegation(comparison):
    root, directory, _ = comparison
    assert validate_workflow({"schema_version": 1, "group": "g"})["mode"] == "attended"
    with pytest.raises(ValueError, match="authorized"):
        validate_workflow({"schema_version": 1, "group": "g", "mode": "unattended"})
    write(directory / "workflow.yaml", {"schema_version": 1, "group": directory.name})
    with pytest.raises(ValueError, match="actor: user"):
        record_gate(root, directory, proposal(root, directory, "A", actor="researcher"))
    record_gate(root, directory, proposal(root, directory, "A"))
    assert read_receipts(directory)[0]["actor"] == "user"


def test_sequence_and_exact_reviewed_subjects(comparison):
    root, directory, _ = comparison
    with pytest.raises(ValueError, match="Gate A"):
        record_gate(root, directory, proposal(root, directory, "B"))
    with pytest.raises(ValueError, match="exact reviewed"):
        record_gate(root, directory, proposal(root, directory, "A", subjects={"hypothesis_id": "h-old"}))
    approve_abc(root, directory)
    assert launch_approval_problems(root, directory, comparison[2], push=True) == []
    old = proposal(root, directory, "B")
    (root / "src/model.py").write_text("width = 128\n")
    assert "Gate B approval is stale" in launch_approval_problems(root, directory, comparison[2], push=True)[0]
    with pytest.raises(ValueError, match="exact reviewed"):
        record_gate(root, directory, old)


def test_hash_sees_untracked_deleted_and_resolved_overrides(comparison):
    root, directory, _ = comparison
    initial = implementation_id(root, directory)
    subprocess.run(["git", "add", "src/model.py"], cwd=root, check=True)
    assert implementation_id(root, directory) == initial  # staging is not scientific change
    (root / "src/model.py").unlink()
    assert implementation_id(root, directory) != initial
    (root / "src/model.py").write_text("width = 64\n")
    (root / "src/extra.py").write_text("new = True\n")
    assert implementation_id(root, directory) != initial
    (root / "src/extra.py").unlink()
    settings = yaml.safe_load((directory / "implementation.yaml").read_text())
    settings["variants"]["router"]["overrides"].append("agent.policy.actor.hidden_dims=[32]")
    write(directory / "implementation.yaml", settings)
    assert implementation_id(root, directory) != initial
    with pytest.raises(ValueError, match="variants differ"):
        subjects(root, directory, "C")


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_total_training_units", 1),
        ("max_total_evaluation_units", 1),
        ("max_attempts", 1),
        ("max_variants", 1),
        ("max_training_seeds", 1),
        ("compute_pool", [0]),
        ("training_unit", "epochs"),
    ],
)
def test_mandate_blocks_excess_before_approval(comparison, field, value):
    _, directory, p = comparison
    workflow = yaml.safe_load((directory / "workflow.yaml").read_text())
    workflow["mandate"]["limits"][field] = value
    workflow = validate_workflow(workflow)
    with pytest.raises(ValueError, match="exceeds|differs"):
        check_mandate(p, workflow, [], implementation="i-source")


def test_reservations_include_retries_smokes_and_all_evaluation_cells(comparison):
    root, directory, p = comparison
    assert reservations(p) == (36060, 10800)
    mixed = copy.deepcopy(p)
    mixed["evaluation"]["difficulties"] = ["easy", "hard"]
    assert reservations(mixed)[1] == 18000  # levels plus difficulties, not their product
    approve_abc(root, directory)
    records = read_receipts(directory)
    check_mandate(
        p, read_workflow(directory), records, implementation=implementation_id(root, directory)
    )  # same ID is reserved once
    p2 = copy.deepcopy(p)
    p2.pop("protocol_id")
    p2["protocol_version"] += 1
    h = yaml.safe_load((directory / "hypothesis.yaml").read_text())
    p2 = validate_contract("protocol", p2, h)
    check_mandate(p2, read_workflow(directory), records, implementation=implementation_id(root, directory))
    p3 = {**p2, "protocol_id": "p-new"}
    records.append({**records[-1], "subjects": {**records[-1]["subjects"], "protocol_id": p2["protocol_id"]}})
    with pytest.raises(ValueError, match="follow-up"):
        check_mandate(p3, read_workflow(directory), records, implementation=implementation_id(root, directory))


def test_rejection_supersedes_approval_and_mandate_changes_invalidate_it(comparison):
    root, directory, p = comparison
    approve_abc(root, directory)
    record_gate(root, directory, proposal(root, directory, "B", "reject", reason="Export roundtrip is broken."))
    assert "Gate B approval missing or rejected" in launch_approval_problems(root, directory, p, push=True)[0]
    record_gate(root, directory, proposal(root, directory, "B"))
    assert "Gate C approval is stale" in launch_approval_problems(root, directory, p, push=True)[0]
    workflow = yaml.safe_load((directory / "workflow.yaml").read_text())
    workflow["mandate"]["permissions"]["push"] = False
    workflow["mandate"]["permissions"]["commit"] = False
    write(directory / "workflow.yaml", workflow)
    assert "Gate A approval is stale" in launch_approval_problems(root, directory, p, push=True)[0]
    approve_abc(root, directory)
    assert launch_approval_problems(root, directory, p, push=False) == []
    assert "outside the unattended mandate" in launch_approval_problems(root, directory, p, push=True)[0]


def test_decision_and_concise_dashboard_preserve_history(comparison):
    root, directory, p = comparison
    with pytest.raises(ValueError, match="Gate A"):
        record_gate(root, directory, proposal(root, directory, "D", "reject"))
    approve_abc(root, directory)
    decision = yaml.safe_load((directory / "decision.yaml").read_text())
    record_gate(root, directory, proposal(root, directory, "D", decision["decision"]))
    dashboard = decision_dashboard(root, directory)
    assert [r["verdict"] for r in dashboard["rows"]] == [
        "approve",
        "approve",
        "approve",
        decision["decision"],
    ]
    assert len(dashboard["history"]) == 4
    markdown = dashboard_markdown(dashboard)
    assert "| Gate | Decision |" in markdown and "https://wandb.ai/" in markdown
    with pytest.raises(ValueError, match="40 words"):
        record_gate(root, directory, proposal(root, directory, "A", reason="word " * 41))
    record_gate(root, directory, proposal(root, directory, "C", "reject", reason="Needs a smaller compute budget."))
    assert len(decision_dashboard(root, directory)["history"]) == 5
    assert "missing or rejected" in launch_approval_problems(root, directory, p, push=True)[0]


def test_report_before_training_and_snapshot_refresh(comparison, monkeypatch):
    wr = pytest.importorskip("wandb_workspaces.reports.v2")
    root, directory, _ = comparison
    record_gate(root, directory, proposal(root, directory, "A"))
    monkeypatch.setattr(rr, "ROOT", root)
    fake_wr = type("WR", (), {name: getattr(wr, name) for name in dir(wr) if not name.startswith("_")})
    saved = FakeReport(blocks=[wr.H1(text="Analysis"), wr.P(text="Human revision")])
    fake_wr.Report = lambda **kw: saved

    class Artifact(FakeArtifact):
        @contextmanager
        def new_file(self, name, mode):
            stream = io.StringIO()
            yield stream
            self.entries[name] = stream.getvalue()

    fake = FakeWandb([])
    fake.Artifact = Artifact
    url = rr.publish_report(
        "rwm-explore",
        directory.name,
        wandb_module=fake,
        wr_module=fake_wr,
        evidence_only="Gate dashboard update",
    )
    assert url == saved.url
    dashboard = fake.inits[0]["config"]["research"]["decisions"]
    assert dashboard["rows"][0]["actor"] == "researcher"
    assert json.loads(fake.artifacts[0].entries["decisions.json"])["history"][0]["gate"] == "A"
    texts = [b.text for b in saved.blocks if isinstance(b, wr.MarkdownBlock)]
    assert any("| A | approve |" in text for text in texts)
    previous = FakeRun(
        directory.name + "__report",
        "report",
        summary={"report_url": url},
        config={"research": {"decisions": dashboard}},
    )
    (directory / "workflow.yaml").unlink()
    fake2 = FakeWandb([previous])
    fake2.Artifact = Artifact
    fake_wr.Report = FakeReport
    FakeReport.loaded = {url: saved}
    saved.blocks += [wr.H1(text="Analysis"), wr.P(text="My later revision")]
    rr.publish_report(
        "rwm-explore",
        directory.name,
        wandb_module=fake2,
        wr_module=fake_wr,
        evidence_only="Historical dashboard snapshot",
    )
    snapshot = fake2.inits[0]["config"]["research"]["decisions"]
    assert snapshot["snapshot"] and snapshot["history"] == dashboard["history"]
    assert "Last published snapshot" in dashboard_markdown(snapshot)
    assert rr.human_sections(saved.blocks)["Analysis"][-1].text == "My later revision"


def test_legacy_comparisons_keep_existing_approval_behavior(tmp_path):
    assert launch_approval_problems(tmp_path, tmp_path, {}, push=True) == []
    assert decision_dashboard(tmp_path, tmp_path) is None


def test_renewed_gate_reopens_downstream_and_removed_workflow_cannot_bypass(comparison):
    root, directory, p = comparison
    approve_abc(root, directory)
    record_gate(root, directory, proposal(root, directory, "A"))
    assert "Gate B approval is stale" in launch_approval_problems(root, directory, p, push=True)[0]
    dashboard = decision_dashboard(root, directory)
    assert dashboard["rows"][1]["stale"] and dashboard["rows"][2]["stale"]
    (directory / "workflow.yaml").unlink()
    assert "workflow.yaml missing" in launch_approval_problems(root, directory, p, push=True)[0]


def test_source_reruns_reserve_again_without_resetting_budget(comparison):
    root, directory, p = comparison
    approve_abc(root, directory)
    workflow = read_workflow(directory)
    records = read_receipts(directory)
    workflow["mandate"]["limits"]["max_followups"] = 0
    (root / "src/model.py").write_text("width = 128\n")
    with pytest.raises(ValueError, match="follow-up"):
        check_mandate(p, workflow, records, implementation=implementation_id(root, directory))


def test_queue_enforces_approval_before_pinning_or_dispatch(comparison, monkeypatch):
    from test_research_workflow_safety import mock_queue

    root, directory, p = comparison
    monkeypatch.setattr(rr, "ROOT", root)
    argv, calls, pinned, _ = mock_queue(monkeypatch, p, root)
    with pytest.raises(SystemExit, match="Gate A approval missing"):
        rr.main(argv)
    assert not calls and not pinned
    approve_abc(root, directory)
    monkeypatch.setattr(rr, "check_smoke_gate", lambda *a, **kw: {"passed": True})
    assert rr.main(argv) == 0
    assert calls and pinned


def test_gate_cli_records_the_reviewed_response(comparison, monkeypatch, capsys):
    root, directory, _ = comparison
    monkeypatch.setattr(rr, "ROOT", root)
    assert rr.main(["gate-subjects", "--group", directory.name, "--gate", "A"]) == 0
    review = json.loads(capsys.readouterr().out)
    response = proposal(root, directory, "A")
    assert response["subjects"] == review["subjects"]
    path = root / "gate.yaml"
    write(path, response)
    assert rr.main(["record-gate", "--group", directory.name, "--file", str(path)]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["actor"] == "researcher" and receipt["subjects"] == review["subjects"]
    assert read_receipts(directory) == [receipt]


def test_user_can_override_final_decision_without_reopening_unchanged_science(comparison):
    root, directory, p = comparison
    approve_abc(root, directory)
    decision = yaml.safe_load((directory / "decision.yaml").read_text())
    record_gate(root, directory, proposal(root, directory, "D", decision["decision"]))
    decision.update(decision="reject", chosen_variants=[], rejected_variants=list(p["variants"]))
    write(directory / "decision.yaml", decision)
    response = proposal(root, directory, "D", "reject", actor="user")
    with pytest.raises(ValueError, match="actor: researcher"):
        record_gate(root, directory, response)
    receipt = record_gate(root, directory, {**response, "user_override": True})
    assert receipt["actor"] == "user" and receipt["user_override"]
    dashboard = decision_dashboard(root, directory)
    assert len(dashboard["history"]) == 5
    assert "reject · user override" in dashboard_markdown(dashboard)
    assert launch_approval_problems(root, directory, p, push=True) == []

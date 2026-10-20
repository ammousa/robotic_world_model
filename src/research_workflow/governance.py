"""Research authority and immutable gate receipts; Git owns decisions, W&B shows them."""

from __future__ import annotations

import hashlib
import html
import subprocess
import yaml
from datetime import datetime, timezone
from pathlib import Path

from . import content_id, load_protocol, read_contract

GATES = ("A", "B", "C", "D")
VERDICTS = {
    "A": {"approve", "reject"},
    "B": {"approve", "reject"},
    "C": {"approve", "reject"},
    "D": {"keep", "reject", "modify", "extend"},
}


def validate_workflow(document: dict) -> dict:
    if (
        not isinstance(document, dict)
        or type(document.get("schema_version")) is not int
        or document.get("schema_version") != 1
    ):
        raise ValueError("workflow requires schema_version: 1")
    if not isinstance(document.get("group"), str) or not document["group"]:
        raise ValueError("workflow requires a comparison group")
    mode = document.get("mode", "attended")
    if mode not in {"attended", "unattended"}:
        raise ValueError("workflow mode must be attended or unattended")
    doc = {**document, "mode": mode}
    if mode == "unattended":
        mandate = doc.get("mandate")
        if not isinstance(mandate, dict) or mandate.get("authorized_by") != "user":
            raise ValueError("unattended mandate must be explicitly authorized by the user")
        for field in ("objective", "allowed_changes", "stop_when"):
            if not isinstance(mandate.get(field), str) or not mandate[field].strip():
                raise ValueError(f"mandate requires {field}")
        limits = mandate.get("limits", {})
        if not isinstance(limits, dict):
            raise ValueError("mandate.limits must be a mapping")
        for field in (
            "max_variants",
            "max_training_seeds",
            "max_attempts",
            "max_total_training_units",
            "max_total_evaluation_units",
        ):
            if type(limits.get(field)) is not int or limits[field] < 1:
                raise ValueError(f"mandate.limits.{field} must be a positive integer")
        if type(limits.get("max_followups")) is not int or limits["max_followups"] < 0:
            raise ValueError("mandate.limits.max_followups must be a nonnegative integer")
        if limits.get("training_unit") not in {"iterations", "epochs"}:
            raise ValueError("mandate requires training_unit: iterations or epochs")
        pool = limits.get("compute_pool")
        if not isinstance(pool, list) or not pool or any(type(g) is not int or g < 0 for g in pool):
            raise ValueError("mandate requires an explicit compute_pool")
        permissions = mandate.get("permissions", {})
        if not isinstance(permissions, dict):
            raise ValueError("mandate.permissions must be a mapping")
        for field in ("commit", "push", "finalize"):
            if type(permissions.get(field)) is not bool:
                raise ValueError(f"mandate.permissions.{field} must be explicit boolean")
    identity = content_id("workflow", doc)
    if doc.get("workflow_id", identity) != identity:
        raise ValueError("stale workflow_id")
    return {**doc, "workflow_id": identity}


def read_workflow(directory: Path) -> dict:
    path = directory / "workflow.yaml"
    workflow = validate_workflow(yaml.safe_load(path.read_text()))
    if workflow["group"] != directory.name:
        raise ValueError("workflow group differs from directory")
    return workflow


def implementation_id(root: Path, directory: Path) -> str:
    """Hash executable source, including unstaged/deleted/untracked files and reviewed overrides."""
    paths = (
        subprocess.check_output(
            [
                "git",
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
                "--",
                "src",
                "source",
                "configs",
                "scripts",
                "tests",
                "pyproject.toml",
                "Makefile",
                "environment.yml",
            ],
            cwd=root,
        )
        .decode()
        .split("\0")
    )
    files = {}
    for name in sorted(set(paths) - {""}):
        path = root / name
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    settings = directory / "implementation.yaml"
    if not settings.is_file():
        raise ValueError("Gate B requires implementation.yaml with the exact reviewed variants")
    document = yaml.safe_load(settings.read_text())
    if not isinstance(document, dict) or not isinstance(document.get("variants"), dict):
        raise ValueError("implementation.yaml requires variants")
    return content_id("implementation", {"files": files, "settings": document})


def subjects(root: Path, directory: Path, gate: str) -> dict:
    h = read_contract(directory / "hypothesis.yaml", "hypothesis")
    if h["group"] != directory.name:
        raise ValueError("hypothesis group differs from workflow directory")
    result = {"hypothesis_id": h["hypothesis_id"]}
    if gate in {"B", "C", "D"}:
        result["implementation_id"] = implementation_id(root, directory)
    if gate in {"C", "D"}:
        protocol = load_protocol(directory / "protocol.yaml")
        settings = yaml.safe_load((directory / "implementation.yaml").read_text())
        if settings["variants"] != protocol["variants"]:
            raise ValueError("protocol variants differ from Gate B reviewed implementation.yaml")
        result["protocol_id"] = protocol["protocol_id"]
    if gate != "A":
        records = read_receipts(directory)
        result["prior_receipts"] = {
            previous: next((r["receipt_id"] for r in reversed(records) if r["gate"] == previous), None)
            for previous in GATES[: GATES.index(gate)]
        }
    if gate == "D":
        decision = read_contract(directory / "decision.yaml", "decision")
        if decision["based_on_protocol"] != result["protocol_id"]:
            raise ValueError("decision refers to a different protocol")
        result["decision_id"] = content_id("decision", decision)
    return result


def read_receipts(directory: Path) -> list[dict]:
    path = directory / "gate_decisions.yaml"
    records = yaml.safe_load(path.read_text()) if path.exists() else []
    if not isinstance(records, list):
        raise ValueError("gate_decisions.yaml must be a list")
    for record in records:
        if not isinstance(record, dict) or record.get("gate") not in GATES:
            raise ValueError("invalid gate receipt")
        gate = record["gate"]
        if record.get("mode") not in {"attended", "unattended"}:
            raise ValueError("invalid receipt mode")
        if gate == "C" and record.get("verdict") == "approve":
            for key in ("reserved_training_units", "reserved_evaluation_units"):
                if type(record.get(key)) is not int or record[key] < 1:
                    raise ValueError(f"Gate C receipt requires {key}")
        if record.get("verdict") not in VERDICTS[gate]:
            raise ValueError("invalid gate verdict")
        expected = "researcher" if record.get("mode") == "unattended" else "user"
        user_override = (
            expected == "researcher" and record.get("actor") == "user" and record.get("user_override") is True
        )
        if record.get("actor") != expected and not user_override:
            raise ValueError("gate receipt has the wrong decision maker")
        for field in ("reason", "risk", "next_action", "evidence", "subjects", "at", "workflow_id"):
            if not record.get(field):
                raise ValueError(f"gate receipt requires {field}")
        if record.get("receipt_id") != content_id("receipt", record):
            raise ValueError("stale receipt_id")
    return records


def require_approval(root: Path, directory: Path, workflow: dict, gate: str) -> None:
    records = read_receipts(directory)
    latest = next((r for r in reversed(records) if r["gate"] == gate), None)
    if not latest or latest["verdict"] != "approve":
        raise ValueError(f"Gate {gate} approval missing or rejected")
    if latest["workflow_id"] != workflow["workflow_id"] or latest["subjects"] != subjects(root, directory, gate):
        raise ValueError(f"Gate {gate} approval is stale; request renewed review")


def check_mandate(protocol: dict, workflow: dict, records: list[dict], *, implementation: str) -> None:
    if protocol["group"] != workflow["group"]:
        raise ValueError("protocol and workflow groups differ")
    if workflow["mode"] != "unattended":
        return
    limits = workflow["mandate"]["limits"]
    if set(protocol["compute_pool"]) - set(limits["compute_pool"]):
        raise ValueError("protocol exceeds mandated compute pool")
    if len(protocol["variants"]) > limits["max_variants"] or len(protocol["seeds"]) > limits["max_training_seeds"]:
        raise ValueError("protocol exceeds mandated variants or training seeds")
    attempts = protocol["retry_policy"]["max_attempts"]
    if attempts > limits["max_attempts"]:
        raise ValueError("protocol exceeds mandated retry limit")
    if limits["training_unit"] != "iterations":
        raise ValueError("protocol training unit differs from mandate")
    reservation, evaluation = reservations(protocol)
    approved = {
        (r["subjects"]["protocol_id"], r["subjects"]["implementation_id"]): r["reserved_training_units"]
        for r in records
        if r["gate"] == "C" and r["verdict"] == "approve"
    }
    approved[(protocol["protocol_id"], implementation)] = reservation
    if len(approved) > limits["max_followups"] + 1:
        raise ValueError("protocol exceeds mandated follow-up limit")
    evaluations = {
        (r["subjects"]["protocol_id"], r["subjects"]["implementation_id"]): r["reserved_evaluation_units"]
        for r in records
        if r["gate"] == "C" and r["verdict"] == "approve"
    }
    evaluations[(protocol["protocol_id"], implementation)] = evaluation
    if sum(evaluations.values()) > limits["max_total_evaluation_units"]:
        raise ValueError("protocol exceeds total reserved evaluation units")
    if sum(approved.values()) > limits["max_total_training_units"]:
        raise ValueError("protocol exceeds total reserved training units (including smoke and retries)")


def reservations(protocol: dict) -> tuple[int, int]:
    budget = protocol["training_budget"]
    smoke = protocol["smoke_gate"]
    units = budget["iterations"] + (smoke.get("iterations", 0) if smoke["enabled"] else 0)
    runs = len(protocol["variants"]) * len(protocol["seeds"]) * protocol["retry_policy"]["max_attempts"]
    evaluation = protocol["evaluation"]
    cells = len(evaluation["seeds"]) * max(1, len(evaluation["levels"]) + len(evaluation["difficulties"]))
    cells *= evaluation["num_envs"] * evaluation["num_episodes"]
    evaluation_units = cells * runs
    if protocol.get("domain") == "world_model":
        declared = protocol["world_model"].get("reserved_evaluation_units")
        if type(declared) is not int or declared < evaluation_units:
            raise ValueError("world-model reservation must cover all additional diagnostic/replay/validation units")
        evaluation_units = declared
    return units * runs, evaluation_units


def record_gate(root: Path, directory: Path, proposal: dict) -> dict:
    if not isinstance(proposal, dict):
        raise ValueError("gate proposal must be a mapping")
    workflow = read_workflow(directory)
    gate = proposal.get("gate")
    if gate not in GATES or proposal.get("verdict") not in VERDICTS[gate]:
        raise ValueError("invalid gate or verdict")
    actor = "researcher" if workflow["mode"] == "unattended" else "user"
    user_override = actor == "researcher" and proposal.get("actor") == "user" and proposal.get("user_override") is True
    if proposal.get("actor") != actor and not user_override:
        raise ValueError(f"{workflow['mode']} decisions require actor: {actor}")
    for field, words in (("reason", 40), ("risk", 25), ("next_action", 25)):
        text = proposal.get(field)
        if not isinstance(text, str) or not text.strip() or len(text.split()) > words:
            raise ValueError(f"{field} must be nonempty and at most {words} words")
    evidence = proposal.get("evidence")
    if (
        not isinstance(evidence, list)
        or not 1 <= len(evidence) <= 3
        or any(not isinstance(e, str) or not e.strip() for e in evidence)
    ):
        raise ValueError("provide 1–3 evidence references")
    records = read_receipts(directory)
    current = subjects(root, directory, gate)
    if proposal.get("subjects") != current or proposal.get("workflow_id") != workflow["workflow_id"]:
        raise ValueError("proposal must name the current workflow and exact reviewed subjects")
    if gate == "D" or proposal["verdict"] != "reject":
        for previous in GATES[: GATES.index(gate)]:
            require_approval(root, directory, workflow, previous)
    record = {
        k: proposal[k]
        for k in (
            "gate",
            "actor",
            "verdict",
            "reason",
            "risk",
            "next_action",
            "evidence",
            "subjects",
            "workflow_id",
        )
    }
    if gate == "C" and proposal["verdict"] == "approve":
        protocol = load_protocol(directory / "protocol.yaml")
        check_mandate(protocol, workflow, records, implementation=current["implementation_id"])
        record["reserved_training_units"], record["reserved_evaluation_units"] = reservations(protocol)
    if gate == "D":
        decision = read_contract(directory / "decision.yaml", "decision")
        if proposal["verdict"] != decision["decision"]:
            raise ValueError("Gate D verdict differs from decision.yaml")
    if user_override:
        record["user_override"] = True
    record.update(mode=workflow["mode"], at=datetime.now(timezone.utc).isoformat())
    record["receipt_id"] = content_id("receipt", record)
    (directory / "gate_decisions.yaml").write_text(yaml.safe_dump([*records, record], sort_keys=False))
    return record


def launch_approval_problems(root: Path, directory: Path, protocol: dict, *, push: bool) -> list[str]:
    if not (directory / "workflow.yaml").exists():
        if (directory / "gate_decisions.yaml").exists():
            return ["workflow.yaml missing for a comparison with gate receipts"]
        return []  # Historical comparisons retain their conversation-based approvals.
    try:
        workflow = read_workflow(directory)
        for gate in ("A", "B", "C"):
            require_approval(root, directory, workflow, gate)
        check_mandate(protocol, workflow, read_receipts(directory), implementation=implementation_id(root, directory))
        if workflow["mode"] == "unattended":
            permissions = workflow["mandate"]["permissions"]
            if push and not permissions["push"]:
                raise ValueError("launch push is outside the unattended mandate")
    except (ValueError, OSError, KeyError, TypeError) as error:
        return [str(error)]
    return []


def decision_dashboard(root: Path, directory: Path) -> dict | None:
    if not (directory / "workflow.yaml").exists():
        return None
    workflow = read_workflow(directory)
    records = read_receipts(directory)
    rows = []
    for gate in GATES:
        record = next((r for r in reversed(records) if r["gate"] == gate), None)
        row = {
            "gate": gate,
            "verdict": "pending",
            "actor": "researcher" if workflow["mode"] == "unattended" else "user",
        }
        if record:
            row.update(record)
            try:
                stale = record["workflow_id"] != workflow["workflow_id"] or record["subjects"] != subjects(
                    root, directory, gate
                )
            except (ValueError, OSError):
                stale = True
            row["stale"] = stale
        rows.append(row)
    return {"workflow": workflow, "rows": rows, "history": records}


def dashboard_markdown(dashboard: dict) -> str:
    def cell(value):
        return html.escape(str(value)).replace("|", "&#124;").replace("\n", " ")

    workflow = dashboard["workflow"]
    lines = [
        f"Mode: **{cell(workflow['mode'])}**. Decision maker:"
        f" **{'researcher' if workflow['mode'] == 'unattended' else 'user'}**."
    ]
    if dashboard.get("snapshot"):
        lines.append("Last published snapshot; current source and approval freshness were not rechecked.")
    if workflow["mode"] == "unattended":
        lines.append(f"Objective: {cell(workflow['mandate']['objective'])}")
    lines += [
        "",
        "| Gate | Decision | Why | Evidence | Limitation | Next action |",
        "|---|---|---|---|---|---|",
    ]
    for row in dashboard["rows"]:
        verdict = row["verdict"] + (" · stale" if row.get("stale") else "")
        if row.get("user_override"):
            verdict += " · user override"
        evidence = "; ".join(row.get("evidence", []))
        lines.append(
            "| "
            + " | ".join(
                cell(x)
                for x in (
                    row["gate"],
                    verdict,
                    row.get("reason", "—"),
                    evidence or "—",
                    row.get("risk", "—"),
                    row.get("next_action", "Await review"),
                )
            )
            + " |"
        )
    lines.append("\nFull gate history and exact artifact IDs are attached to the report run as `research-decisions`.")
    return "\n".join(lines)

"""Replay the unmodified captured failure; /proc identities below are replay fixtures."""

import copy
import gzip
import hashlib
import json
import yaml
from pathlib import Path

import pytest

from research_workflow.gpu_admission import AdmissionError, TrainingAdmission


@pytest.fixture
def captured():
    root = Path(__file__).resolve().parents[1]
    path = root / "tests/research_gpu_admission_trace.jsonl.gz"
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == "dcdb899386f6be825f1befcfc3e1118345784ea91a9daa992bee83b2d7bf28d8"
    )
    with gzip.open(path, "rt") as stream:
        rows = [json.loads(line) for line in stream]
    settings = yaml.safe_load((root / "research/rwm-discrete-flow/protocol.v6.yaml").read_text())["world_model"][
        "training_context_admission"
    ]
    identities = {int(pid): record["identity"] for pid, record in settings["devices"]["3"]["passive_processes"].items()}
    # Captured snapshots have real PID values but do not contain /proc child identity.
    # These explicit synthetic records exercise identity checks; they are not measurements.
    for pid in (2049295, 2049296):
        identities[pid] = {
            "pid": pid,
            "start_ticks": pid,
            "pgid": 2049295,
            "uid": 0,
            "cmdline": ["CPU-replay-identity-fixture"],
            "cwd": "/replay",
        }
    guard = TrainingAdmission(settings, [3], probe=object(), identity=identities.get)
    owner = copy.deepcopy(identities[2049295])
    resets = []
    for index, row in enumerate(rows[:-1]):
        if index == 1:
            # The real controller reset after the Git/network pin, before dispatch.
            assert row["monotonic"] - rows[0]["monotonic"] > 0.15
            guard.previous = None
            resets.append(index)
        guard.validate(row, [owner])
        guard.previous = row
    assert resets == [1]
    return guard, copy.deepcopy(rows[-1]), identities, owner


def test_complete_captured_trace_passes_without_modifying_a_raw_sample(captured):
    guard, final, _, owner = captured
    before = copy.deepcopy(final)
    guard.validate(final, [owner])
    assert final == before
    assert len(guard.owned_event_history["3"]) > 1
    assert guard.settings["maximum_poll_gap_seconds"] == 0.15
    assert guard.settings["maximum_sample_gap_seconds"] == 0.35
    assert guard.settings["maximum_sample_age_seconds"] == 0.6


def test_only_unchanged_tiny_alias_can_reuse_verified_event_history(captured):
    guard, final, _, owner = captured
    guard.validate(final, [owner])
    assert guard._attributed_origin("3", "1", 1792668237298153, 18) == 1792668237298151
    guard.owned_event_history.clear()
    assert guard._attributed_origin("3", "1", 1792668237298153, 18) is None


@pytest.mark.parametrize("shift,value", [(0, 99), (100000, 99), (3, 18)])
def test_changed_or_out_of_bound_historical_sample_is_not_attributed_from_cache(captured, shift, value):
    guard, final, _, owner = captured
    row = next(row for row in final["devices"]["3"]["aggregate"]["1"] if row[0] == 1792668237298152)
    row[0] = 1792668237298151 + shift
    row[1] = value
    assert guard.owned_event_history["3"]
    with pytest.raises(AdmissionError, match="owned-device heartbeat"):
        guard.validate(final, [owner])


@pytest.mark.parametrize(
    "violation",
    [
        "foreign",
        "identity",
        "missing",
        "stale_owned",
        "old_burst",
        "memory",
        "unknown",
        "sensor",
        "gap",
        "stale_aggregate",
        "poll",
    ],
)
def test_genuine_violations_still_stop_captured_replay(captured, violation):
    guard, final, identities, owner = captured
    device = final["devices"]["3"]
    if violation == "foreign":
        device["events"].append(
            {"pid": 2018122, "stamp": device["events"][-1]["stamp"], "sm": 1, "mem": 0, "enc": 0, "dec": 0}
        )
    elif violation == "identity":
        identities[2049296] = {**identities[2049296], "start_ticks": -1}
    elif violation == "missing":
        device.update(events=[], process_event_status=6)
    elif violation == "stale_owned":
        for event in device["events"]:
            if event["pid"] == 2049296:
                event["stamp"] -= 700000
    elif violation == "old_burst":
        device["aggregate"]["1"][0][1] = 75
    elif violation == "memory":
        device["processes"]["2018122"] += 1
    elif violation == "unknown":
        device["processes"]["99999999"] = 1
    elif violation == "sensor":
        device["process_event_status"] = 999
    elif violation == "gap":
        device["aggregate"]["1"].pop(-3)
    elif violation == "stale_aggregate":
        device["aggregate"]["1"] = device["aggregate"]["1"][:-4]
    else:
        final["monotonic"] += 0.2
    before = copy.deepcopy(guard.owned_event_history)
    with pytest.raises(AdmissionError):
        guard.validate(final, [owner])
    assert guard.owned_event_history == before  # failed checks cannot certify events


def test_retiming_is_bounded_anchored_value_sensitive_and_unambiguous(captured):
    guard, _, _, _ = captured
    guard.attributed_samples = {"3": {"1": {100: (100, 4)}}}
    guard.owned_event_history = {"3": {"synthetic": {"identity": guard.identity(2049296), "event": {"stamp": 100}}}}
    assert guard._attributed_origin("3", "1", 102, 4) == 100
    assert guard._attributed_origin("3", "1", 102, 5) is None
    assert guard._attributed_origin("3", "1", 103, 4) is None
    guard.attributed_samples["3"]["1"] = {102: (100, 4)}
    assert guard._attributed_origin("3", "1", 103, 4) is None  # no cumulative drift
    guard.attributed_samples["3"]["1"] = {100: (100, 4), 102: (102, 4)}
    with pytest.raises(AdmissionError, match="ambiguous"):
        guard._attributed_origin("3", "1", 101, 4)

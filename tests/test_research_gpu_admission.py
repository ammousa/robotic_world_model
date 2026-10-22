"""Inactive contexts may be admitted only with validated continuous activity evidence."""

import copy
import json
import sys

import pytest

from research_workflow.gpu_admission import AdmissionError, TrainingAdmission


def fixture():
    identities = {
        11: {
            "pid": 11,
            "start_ticks": 101,
            "pgid": 11,
            "uid": 9,
            "cmdline": ["train", "--device", "cuda:0"],
            "cwd": "/other",
        },
        12: {
            "pid": 12,
            "start_ticks": 102,
            "pgid": 12,
            "uid": 9,
            "cmdline": ["train", "--device", "cuda:1"],
            "cwd": "/other",
        },
        90: {"pid": 90, "start_ticks": 190, "pgid": 90, "uid": 9, "cmdline": ["owned"], "cwd": "/ours"},
        91: {"pid": 91, "start_ticks": 191, "pgid": 90, "uid": 9, "cmdline": ["owned-child"], "cwd": "/ours"},
    }
    cfg = {
        "scope": "training_and_pilots_only",
        "poll_seconds": 0.05,
        "maximum_poll_gap_seconds": 0.15,
        "maximum_sample_gap_seconds": 0.35,
        "maximum_sample_age_seconds": 0.6,
        "quiet_window_seconds": 10,
        "maximum_quiet_wait_seconds": 60,
        "devices": {},
        "positive_controls": {},
    }
    snapshot = {"wall_us": 20000000, "monotonic": 20.0, "devices": {}}
    for gpu in (0, 1, 2, 3):
        event = {
            "pid": 11 if gpu % 2 == 0 else 12,
            "stamp": 19900000,
            "sm": 50 if gpu < 2 else 0,
            "mem": 30 if gpu < 2 else 0,
            "enc": 0,
            "dec": 0,
        }
        device = {
            "uuid": f"GPU-{gpu}",
            "processes": {"11": 209, "12": 209},
            "process_event_status": 0,
            "events": [event],
            "aggregate": {str(k): [[t, 0] for t in range(7900000, 20000000, 200000)] for k in (1, 2)},
        }
        snapshot["devices"][str(gpu)] = device
        if gpu < 2:
            cfg["positive_controls"][str(gpu)] = {"uuid": device["uuid"], "identity": identities[11 + gpu]}
        else:
            cfg["devices"][str(gpu)] = {
                "uuid": device["uuid"],
                "passive_processes": {
                    str(pid): {"identity": identities[pid], "maximum_memory_bytes": 209} for pid in (11, 12)
                },
            }
    guard = TrainingAdmission(cfg, [2, 3], probe=object(), identity=identities.get)
    return guard, snapshot, identities


def test_documented_absent_events_remain_absent_with_fresh_sensor_and_positive_controls():
    guard, snapshot, _ = fixture()
    snapshot["devices"]["2"]["events"] = []
    snapshot["devices"]["2"]["process_event_status"] = 6
    guard.validate(snapshot, [])
    assert snapshot["devices"]["2"]["events"] == []
    assert "sm" not in snapshot["devices"]["2"]["processes"]


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda d: d["events"][0].update(sm=1), "foreign GPU activity"),
        (lambda d: d["events"][0].update(mem=1), "foreign GPU activity"),
        (lambda d: d["events"][0].update(enc=1), "foreign GPU activity"),
        (lambda d: d["events"][0].update(dec=1), "foreign GPU activity"),
        (lambda d: d["events"][0].update(sm=0xFFFFFFFF), "unavailable process"),
        (lambda d: d["processes"].update({"99": 209}), "unknown GPU process"),
        (lambda d: d["events"][0].update(pid=99), "unknown process activity"),
        (lambda d: d["processes"].update({"11": 210}), "memory grew"),
        (lambda d: d.update(process_event_status=999), "sensor failed"),
        (lambda d: d.update(uuid="GPU-replaced"), "GPU identity changed"),
        (lambda d: d["events"][0].update(stamp=18000000), "stale process"),
        (lambda d: d["aggregate"]["1"].pop(-3), "sample coverage"),
        (lambda d: d["aggregate"]["1"].append([21000000, 0]), "sample coverage"),
        (lambda d: d["aggregate"]["1"][-1].__setitem__(1, 101), "unavailable aggregate"),
    ],
)
def test_foreign_activity_and_unknown_evidence_fail_closed(mutation, reason):
    guard, snapshot, _ = fixture()
    mutation(snapshot["devices"]["2"])
    with pytest.raises(AdmissionError, match=reason):
        guard.validate(snapshot, [])


def test_pid_reuse_or_changed_command_is_not_authorized():
    guard, snapshot, identities = fixture()
    identities[11] = {**identities[11], "start_ticks": 900}
    with pytest.raises(AdmissionError, match="identity changed"):
        guard.validate(snapshot, [])


def test_owned_activity_allowed_but_foreign_activity_still_detected():
    guard, snapshot, identities = fixture()
    guard.validate(snapshot, [])
    guard.previous = copy.deepcopy(snapshot)
    snapshot["monotonic"] += 0.05
    snapshot["wall_us"] += 50000
    device = snapshot["devices"]["2"]
    device["processes"]["91"] = 20000
    device["events"].append({"pid": 91, "stamp": 19900000, "sm": 99, "mem": 90, "enc": 0, "dec": 0})
    device["aggregate"]["1"][-1][1] = 99
    guard.validate(snapshot, [identities[90]])
    device["events"][0]["sm"] = 1
    with pytest.raises(AdmissionError, match="foreign GPU activity"):
        guard.validate(snapshot, [identities[90]])


def test_missed_poll_or_sensor_interval_and_stale_control_fail_closed():
    guard, snapshot, _ = fixture()
    guard.previous = copy.deepcopy(snapshot)
    snapshot["monotonic"] += 0.2
    with pytest.raises(AdmissionError, match="missed process observation"):
        guard.validate(snapshot, [])
    snapshot["monotonic"] -= 0.2
    snapshot["devices"]["0"]["events"][0]["stamp"] = 17000000
    with pytest.raises(AdmissionError, match="control sensor stale"):
        guard.validate(snapshot, [])


def test_queue_stops_only_owned_process_group_on_guard_failure(tmp_path):
    from test_research_runs import rr

    class Guard:
        settings = {"poll_seconds": 0.01}

        def __init__(self):
            self.seen = []

        def check(self, owned):
            self.seen.extend(owned)
            if owned:
                raise AdmissionError("synthetic foreign activity")
            return set()

    guard = Guard()
    jobs = [
        {
            "name": "owned",
            "gpu": None,
            "argv": [sys.executable, "-c", "import time; time.sleep(30)"],
            "research": {"job_type": "smoke"},
        }
    ]
    with pytest.raises(SystemExit, match="synthetic foreign activity"):
        rr.run_queue(jobs, [2], tmp_path, training_admission=guard)
    rows = rr.read_events(tmp_path)
    started = [row for row in rows if row["event"] == "start"]
    stopped = [row for row in rows if row["event"] == "owned_isolation_stop"]
    assert len(started) == len(stopped) == 1
    assert stopped[0]["name"] == "owned"
    assert guard.seen == [started[0]["identity"]]
    assert not rr.matches(started[0]["identity"])
    assert any(row["event"] == "exit" and row["returncode"] != 0 for row in rows)
    assert json.loads((tmp_path / "events.jsonl").read_text().splitlines()[-1])["event"] == "queue_end"


def test_guard_cannot_be_used_for_evaluation_or_to_skip_foreign_occupancy(tmp_path):
    from test_research_runs import rr

    guard, _, _ = fixture()
    jobs = [{"name": "eval", "research": {"job_type": "eval"}}]
    with pytest.raises(ValueError, match="training/pilot-only"):
        rr.run_queue(jobs, [2, 3], tmp_path, training_admission=guard)
    with pytest.raises(ValueError, match="cannot allow shared"):
        rr.run_queue([], [2, 3], tmp_path, training_admission=guard, allow_shared=True)
    assert not (tmp_path / "events.jsonl").exists()


def test_departed_context_retires_without_stopping_training():
    guard, snapshot, identities = fixture()
    guard.validate(snapshot, [])
    guard.previous = copy.deepcopy(snapshot)
    snapshot["monotonic"] += 0.05
    snapshot["wall_us"] += 50000
    for gpu in (2, 3):
        snapshot["devices"][str(gpu)]["processes"].pop("11")
    identities.pop(11)
    guard.validate(snapshot, [])


def test_expired_owned_event_is_distinct_from_unknown_current_process():
    guard, snapshot, identities = fixture()
    guard.validate(snapshot, [])
    guard.previous = copy.deepcopy(snapshot)
    snapshot["monotonic"] += 0.05
    snapshot["wall_us"] += 50000
    device = snapshot["devices"]["2"]
    device["processes"]["91"] = 20000
    device["events"].append({"pid": 91, "stamp": 19900000, "sm": 99, "mem": 90, "enc": 0, "dec": 0})
    guard.validate(snapshot, [identities[90]])
    device["processes"].pop("91")
    identities.pop(91)
    guard.validate(snapshot, [])
    device["processes"]["91"] = 20000
    with pytest.raises(AdmissionError, match="unknown GPU process identity"):
        guard.validate(snapshot, [])


def test_sensor_gap_is_not_equated_with_fast_polling():
    guard, snapshot, _ = fixture()
    guard.previous = copy.deepcopy(snapshot)
    snapshot["monotonic"] += 0.05  # Fast polling cannot repair a lost driver sample.
    snapshot["wall_us"] += 600000
    for gpu in (0, 1, 2, 3):
        snapshot["devices"][str(gpu)]["events"][0]["stamp"] += 600000
        for samples in snapshot["devices"][str(gpu)]["aggregate"].values():
            samples.append([20500000, 0])
    with pytest.raises(AdmissionError, match="sample coverage"):
        guard.validate(snapshot, [])


@pytest.mark.parametrize("status,events", [(6, []), (0, [])])
def test_missing_process_attribution_cannot_pass_on_owned_busy_device(status, events):
    guard, snapshot, identities = fixture()
    guard.validate(snapshot, [])
    guard.previous = copy.deepcopy(snapshot)
    snapshot["monotonic"] += 0.05
    snapshot["wall_us"] += 50000
    device = snapshot["devices"]["2"]
    device["processes"]["91"] = 20000
    device["aggregate"]["1"][-1][1] = 99
    device["aggregate"]["2"][-1][1] = 90
    device.update(process_event_status=status, events=events)
    with pytest.raises(AdmissionError, match="busy device attribution missing"):
        guard.validate(snapshot, [identities[90]])


def test_fresh_foreign_control_cannot_hide_missing_or_stale_owned_device_row():
    guard, snapshot, identities = fixture()
    guard.validate(snapshot, [])
    guard.previous = copy.deepcopy(snapshot)
    snapshot["monotonic"] += 0.05
    snapshot["wall_us"] += 50000
    device = snapshot["devices"]["2"]
    device["processes"]["91"] = 20000
    device["aggregate"]["1"][-1][1] = 99
    with pytest.raises(AdmissionError, match="missing or stale owned-device heartbeat"):
        guard.validate(snapshot, [identities[90]])
    device["events"].append({"pid": 91, "stamp": 19000000, "sm": 99, "mem": 90, "enc": 0, "dec": 0})
    with pytest.raises(AdmissionError, match="missing or stale owned-device heartbeat"):
        guard.validate(snapshot, [identities[90]])
    device["events"][-1]["stamp"] = 19900000
    guard.validate(snapshot, [identities[90]])


def test_absent_events_do_not_attribute_an_active_device_even_without_owned_context():
    guard, snapshot, _ = fixture()
    guard.validate(snapshot, [])
    guard.previous = copy.deepcopy(snapshot)
    snapshot["monotonic"] += 0.05
    snapshot["wall_us"] += 50000
    device = snapshot["devices"]["2"]
    device["aggregate"]["1"][-1][1] = 99
    device.update(process_event_status=6, events=[])
    with pytest.raises(AdmissionError, match="busy device attribution missing"):
        guard.validate(snapshot, [])


def test_transient_new_activity_followed_by_zero_still_requires_attribution():
    guard, snapshot, identities = fixture()
    guard.validate(snapshot, [])
    guard.previous = copy.deepcopy(snapshot)
    snapshot["monotonic"] += 0.05
    snapshot["wall_us"] = 20050000
    device = snapshot["devices"]["2"]
    device["aggregate"]["1"].extend([[19950000, 67], [20000000, 0]])
    device.update(process_event_status=6, events=[])
    with pytest.raises(AdmissionError, match="busy device attribution missing"):
        guard.validate(snapshot, [identities[90]])
    # A zero row at the end cannot erase an un-attributed active interval.
    device["aggregate"]["1"][-2][1] = 0
    guard.validate(snapshot, [identities[90]])


def test_old_attributed_owned_history_does_not_require_new_activity_heartbeat():
    guard, snapshot, identities = fixture()
    guard.validate(snapshot, [])
    guard.previous = copy.deepcopy(snapshot)
    snapshot["monotonic"] += 0.05
    snapshot["wall_us"] += 50000
    device = snapshot["devices"]["2"]
    device["processes"]["91"] = 20000
    device["events"].append({"pid": 91, "stamp": 19900000, "sm": 99, "mem": 90, "enc": 0, "dec": 0})
    device["aggregate"]["1"][-1][1] = 99
    guard.validate(snapshot, [identities[90]])
    guard.previous = copy.deepcopy(snapshot)
    device["processes"].pop("91")
    identities.pop(91)
    snapshot["monotonic"] += 0.05
    snapshot["wall_us"] += 50000
    device.update(process_event_status=6, events=[])
    guard.validate(snapshot, [])

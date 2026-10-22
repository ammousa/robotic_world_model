"""Startup telemetry checks cover each dispatch without relaxing the cadence limit."""

import sys
import time

import pytest

from research_workflow.gpu_admission import AdmissionError


def dispatch_fixture(monkeypatch, delay):
    from test_research_runs import rr

    original = rr.subprocess.Popen
    child = "import time; time.sleep(0.4)"

    def delayed(argv, *args, **kwargs):
        if len(argv) > 2 and argv[1:3] == ["-c", child]:
            time.sleep(delay)
        return original(argv, *args, **kwargs)

    monkeypatch.setattr(rr.subprocess, "Popen", delayed)

    class Guard:
        settings = {"poll_seconds": 0.005}

        def __init__(self):
            self.previous = None
            self.gaps = []

        def check(self, owned):
            current = time.monotonic()
            if self.previous is not None:
                gap = current - self.previous
                self.gaps.append(gap)
                if gap > 0.15:
                    raise AdmissionError("missed process observation interval")
            self.previous = current
            return set()

    jobs = [
        {"name": name, "gpu": None, "argv": [sys.executable, "-c", child], "research": {"job_type": "smoke"}}
        for name in ["first", "second"]
    ]
    return rr, Guard(), jobs


def test_each_dispatch_is_observed_instead_of_accumulating_two_startups(tmp_path, monkeypatch):
    rr, guard, jobs = dispatch_fixture(monkeypatch, 0.08)
    assert rr.run_queue(jobs, [2, 3], tmp_path, training_admission=guard) == 0
    starts = [e for e in rr.read_events(tmp_path) if e["event"] == "start"]
    assert len(starts) == 2
    assert max(guard.gaps) <= 0.15


def test_one_overlong_dispatch_still_stops_owned_and_blocks_second_job(tmp_path, monkeypatch):
    rr, guard, jobs = dispatch_fixture(monkeypatch, 0.18)
    with pytest.raises(SystemExit, match="missed process observation interval"):
        rr.run_queue(jobs, [2, 3], tmp_path, training_admission=guard)
    events = rr.read_events(tmp_path)
    starts = [e for e in events if e["event"] == "start"]
    assert len(starts) == 1
    assert any(e["event"] == "owned_isolation_stop" for e in events)
    assert events[-1]["event"] == "queue_end"
    assert not rr.matches(starts[0]["identity"])

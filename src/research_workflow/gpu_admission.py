"""Opt-in training admission for identity-bound, demonstrably inactive GPU contexts.

NVML is read only. Missing process events remain absent, not numeric zeros.
The stock context-free queue remains the default, including every evaluation.
"""

import ctypes as c
import gzip
import json
import os
import time
from pathlib import Path


class AdmissionError(RuntimeError):
    """Telemetry or isolation evidence is insufficient; never start/continue work."""


class _Value(c.Union):
    _fields_ = [("d", c.c_double), ("ui", c.c_uint), ("ul", c.c_ulong), ("ull", c.c_ulonglong)]


class _Sample(c.Structure):
    _fields_ = [("stamp", c.c_ulonglong), ("value", _Value)]


class _Process(c.Structure):
    _fields_ = [
        ("pid", c.c_uint),
        ("memory", c.c_ulonglong),
        ("gpu_instance", c.c_uint),
        ("compute_instance", c.c_uint),
    ]


class _Utilization(c.Structure):
    _fields_ = [
        ("pid", c.c_uint),
        ("stamp", c.c_ulonglong),
        ("sm", c.c_uint),
        ("mem", c.c_uint),
        ("enc", c.c_uint),
        ("dec", c.c_uint),
    ]


def process_identity(pid):
    """Identity and declared primary are bound by the full command and working directory."""
    root = Path("/proc") / str(pid)
    try:
        stat = (root / "stat").read_text().rsplit(")", 1)[1].split()
        return {
            "pid": pid,
            "start_ticks": int(stat[19]),
            "pgid": os.getpgid(pid),
            "uid": root.stat().st_uid,
            "cmdline": [x.decode() for x in (root / "cmdline").read_bytes().split(b"\0")[:-1]],
            "cwd": os.readlink(root / "cwd"),
        }
    except (OSError, ValueError, IndexError):
        return None


class NVML:
    """Small ABI adapter matching NVIDIA nvml.h; no CUDA context or global driver changes."""

    def __init__(self):
        self.lib = c.CDLL("libnvidia-ml.so.1")
        self._ok(self.lib.nvmlInit_v2())
        self.lib.nvmlDeviceGetProcessUtilization.argtypes = [
            c.c_void_p,
            c.POINTER(_Utilization),
            c.POINTER(c.c_uint),
            c.c_ulonglong,
        ]
        self.lib.nvmlDeviceGetSamples.argtypes = [
            c.c_void_p,
            c.c_int,
            c.c_ulonglong,
            c.POINTER(c.c_int),
            c.POINTER(c.c_uint),
            c.POINTER(_Sample),
        ]
        for kind in ("Compute", "Graphics"):
            getattr(self.lib, f"nvmlDeviceGet{kind}RunningProcesses_v2").argtypes = [
                c.c_void_p,
                c.POINTER(c.c_uint),
                c.POINTER(_Process),
            ]

    @staticmethod
    def _ok(code):
        if code != 0:
            raise AdmissionError(f"NVML query failed, status={code}")

    def snapshot(self, devices):
        result = {"wall_us": time.time_ns() // 1000, "monotonic": time.monotonic(), "devices": {}}
        for gpu in devices:
            handle = c.c_void_p()
            self._ok(self.lib.nvmlDeviceGetHandleByIndex_v2(gpu, c.byref(handle)))
            uuid = c.create_string_buffer(96)
            self._ok(self.lib.nvmlDeviceGetUUID(handle, uuid, len(uuid)))
            processes = {}
            for kind in ("Compute", "Graphics"):
                count = c.c_uint(4096)
                buffer = (_Process * count.value)()
                self._ok(getattr(self.lib, f"nvmlDeviceGet{kind}RunningProcesses_v2")(handle, c.byref(count), buffer))
                for item in buffer[: count.value]:
                    if item.memory == (1 << 64) - 1:
                        raise AdmissionError("GPU process memory is unavailable")
                    processes[str(item.pid)] = max(processes.get(str(item.pid), 0), item.memory)
            count = c.c_uint(4096)
            buffer = (_Utilization * count.value)()
            code = self.lib.nvmlDeviceGetProcessUtilization(handle, buffer, c.byref(count), 0)
            # NOT_FOUND is the documented no-event result, never an invented zero sample.
            if code not in (0, 6):
                self._ok(code)
            events = (
                [{name: getattr(item, name) for name, _ in _Utilization._fields_} for item in buffer[: count.value]]
                if code == 0
                else []
            )
            aggregate = {}
            for kind in (1, 2):  # NVML_GPU_UTILIZATION_SAMPLES, NVML_MEMORY_UTILIZATION_SAMPLES
                count = c.c_uint(4096)
                samples = (_Sample * count.value)()
                value_type = c.c_int()
                self._ok(self.lib.nvmlDeviceGetSamples(handle, kind, 0, c.byref(value_type), c.byref(count), samples))
                if value_type.value != 1 or count.value == 0:
                    raise AdmissionError("missing or unexpected NVML aggregate sample type")
                aggregate[str(kind)] = [[item.stamp, item.value.ui] for item in samples[: count.value]]
            result["devices"][str(gpu)] = {
                "uuid": uuid.value.decode(),
                "processes": processes,
                "process_event_status": code,
                "events": events,
                "aggregate": aggregate,
            }
        result["wall_us"] = time.time_ns() // 1000
        result["monotonic"] = time.monotonic()
        return result


class TrainingAdmission:
    """Validate each observed sample interval; queue owns all signalling and job identities."""

    def __init__(self, settings, pool, *, probe=None, identity=process_identity, trace=None):
        self.settings = settings
        self.pool = list(pool)
        self.probe = probe or NVML()
        self.identity = identity
        self.trace = trace
        self.previous = None
        self.known_owned = {}
        if settings["scope"] != "training_and_pilots_only" or set(settings["devices"]) != {str(g) for g in pool}:
            raise ValueError("passive-context admission is restricted to the declared training pool")
        if settings["poll_seconds"] > 0.05 or settings["maximum_sample_gap_seconds"] > 0.35:
            raise ValueError("admission observation cadence is too slow")

    def check(self, owned):
        snapshot = self.probe.snapshot([*self.pool, *map(int, self.settings["positive_controls"])])
        if self.trace:
            if str(self.trace).endswith(".gz"):
                stream = gzip.open(self.trace, "at", compresslevel=1)
            else:
                stream = Path(self.trace).open("a")
            with stream:
                stream.write(json.dumps(snapshot) + "\n")
        self.validate(snapshot, owned)
        self.previous = snapshot
        return set()

    def wait_for_quiet(self):
        deadline = time.monotonic() + self.settings["maximum_quiet_wait_seconds"]
        while True:
            try:
                self.check([])
                self.previous = None  # Queue preflight may itself take longer than one sensor period.
                return
            except AdmissionError as error:
                if str(error) != "GPU is not inactive before training" or time.monotonic() >= deadline:
                    raise
                time.sleep(self.settings["poll_seconds"])

    def validate(self, snapshot, owned):
        cfg = self.settings
        wall = snapshot["wall_us"]
        max_gap = cfg["maximum_sample_gap_seconds"] * 1e6
        max_age = cfg["maximum_sample_age_seconds"] * 1e6
        if self.previous and snapshot["monotonic"] - self.previous["monotonic"] > cfg["maximum_poll_gap_seconds"]:
            raise AdmissionError("missed process observation interval")
        live_owned = {}
        for record in owned:
            actual = self.identity(record["pid"])
            if actual is None:
                continue  # A leader can exit between the queue poll and this read-only check.
            if any(actual.get(key) != value for key, value in record.items()):
                raise AdmissionError("owned leader identity changed")
            live_owned[record["pgid"]] = actual
        self._validate_controls(snapshot, wall, max_age)
        for gpu in self.pool:
            key = str(gpu)
            device = snapshot["devices"][key]
            expected = cfg["devices"][key]
            if device["uuid"] != expected["uuid"]:
                raise AdmissionError("GPU identity changed")
            foreign = {}
            owned_pids = set()
            for raw_pid, memory in device["processes"].items():
                actual = self.identity(int(raw_pid))
                if not actual:
                    raise AdmissionError("unknown GPU process identity")
                if actual["pgid"] in live_owned:
                    self.known_owned[int(raw_pid)] = actual
                    owned_pids.add(int(raw_pid))
                    continue
                binding = expected["passive_processes"].get(raw_pid)
                if not binding or actual != binding["identity"]:
                    raise AdmissionError("unknown or changed foreign process")
                if memory > binding["maximum_memory_bytes"]:
                    raise AdmissionError("foreign context memory grew")
                foreign[int(raw_pid)] = actual
            self._validate_aggregate(snapshot, key, wall, max_age, max_gap)
            if device["process_event_status"] not in (0, 6):
                raise AdmissionError("process activity sensor failed")
            events = device["events"]
            self._validate_owned_heartbeat(key, device, owned_pids, wall, max_age, max_gap)
            if events and not 0 <= wall - max(row["stamp"] for row in events) <= max_age:
                raise AdmissionError("stale process activity sensor")
            if self.previous and events:
                prior_events = self.previous["devices"][key]["events"]
                if (
                    prior_events
                    and max(row["stamp"] for row in events) - max(row["stamp"] for row in prior_events) > max_gap
                ):
                    raise AdmissionError("missed process sensor interval")
            for event in events:
                values = [event[name] for name in ("sm", "mem", "enc", "dec")]
                if any(not 0 <= value <= 100 for value in values):
                    raise AdmissionError("unavailable process activity")
                actual = self.identity(event["pid"])
                if event["pid"] in self.known_owned and str(event["pid"]) not in device["processes"]:
                    if actual is None or actual == self.known_owned[event["pid"]]:
                        continue  # Last utilization sample of an already reaped owned child.
                if not actual:
                    retired = expected["passive_processes"].get(str(event["pid"]))
                    if retired and str(event["pid"]) not in device["processes"] and not any(values):
                        continue  # Last observed zero sample after an allowed context departed.
                    raise AdmissionError("unknown process activity")
                if actual["pgid"] in live_owned:
                    continue
                if event["pid"] not in foreign or actual != foreign[event["pid"]]:
                    raise AdmissionError("unknown foreign activity")
                if any(values):
                    raise AdmissionError("foreign GPU activity detected")
            # An absent context event is not assigned a zero utilization. Only the
            # documented supported no-event semantics plus fresh aggregate coverage
            # permit continuing; all raw absences/statuses are retained in the trace.

    def _validate_controls(self, snapshot, wall, max_age):
        cfg = self.settings
        for key, binding in cfg["positive_controls"].items():
            pid = str(binding["identity"]["pid"])
            if not any(pid in snapshot["devices"][str(gpu)]["processes"] for gpu in self.pool):
                continue  # Departed contexts need no exemption or positive control.
            control = snapshot["devices"][key]
            if control["uuid"] != binding["uuid"] or self.identity(binding["identity"]["pid"]) != binding["identity"]:
                raise AdmissionError("positive-control identity changed")
            events = control["events"]
            if control["process_event_status"] != 0 or not events:
                raise AdmissionError("positive-control activity sensor unavailable")
            if not any(
                row["pid"] == binding["identity"]["pid"] and 0 <= wall - row["stamp"] <= max_age for row in events
            ):
                raise AdmissionError("positive-control sensor stale or missing")
            if not self.previous and not any(
                row["pid"] == binding["identity"]["pid"] and row["sm"] > 0 for row in events
            ):
                raise AdmissionError("positive control has no observed activity")

    def _validate_aggregate(self, snapshot, key, wall, max_age, max_gap):
        cfg = self.settings
        device = snapshot["devices"][key]
        if set(device["aggregate"]) != {"1", "2"}:
            raise AdmissionError("missing aggregate sensor")
        for kind, samples in device["aggregate"].items():
            stamps = [row[0] for row in samples]
            if not stamps or stamps != sorted(set(stamps)) or any(b - a > max_gap for a, b in zip(stamps, stamps[1:])):
                raise AdmissionError("missing aggregate sample coverage")
            if not 0 <= wall - stamps[-1] <= max_age:
                raise AdmissionError("stale aggregate sensor")
            if any(not 0 <= value <= 100 for _, value in samples):
                raise AdmissionError("unavailable aggregate value")
            if not self.previous and stamps[-1] - stamps[0] < cfg["quiet_window_seconds"] * 1e6:
                raise AdmissionError("insufficient quiet-window history")
            if self.previous:
                prior = self.previous["devices"][key]["aggregate"][kind][-1][0]
                newer = [stamp for stamp in stamps if stamp > prior]
                if newer and newer[0] - prior > max_gap:
                    raise AdmissionError("aggregate buffer lost intervals")
            if not self.previous and any(value for _, value in samples):
                raise AdmissionError("GPU is not inactive before training")

    def _validate_owned_heartbeat(self, key, device, owned_pids, wall, max_age, max_gap):
        active_stamps = []
        for kind, samples in device["aggregate"].items():
            previous = dict(self.previous["devices"][key]["aggregate"][kind]) if self.previous else {}
            prior = max(previous, default=-1)
            # A revised active reading is new information even if its timestamp
            # was seen before; unchanged attributed history does not re-trigger.
            active_stamps.extend(
                stamp for stamp, value in samples if value > 0 and (stamp > prior or previous.get(stamp) != value)
            )
        if not active_stamps and not owned_pids:
            return  # Retained, already covered owned history is not new activity.
        events = device["events"]
        if device["process_event_status"] != 0 or not events:
            raise AdmissionError("busy device attribution missing")
        required_stamps = active_stamps or [max(samples[-1][0] for samples in device["aggregate"].values())]
        heartbeats = []
        for event in events:
            pid = event["pid"]
            current = self.identity(pid)
            retired = (
                pid in self.known_owned
                and str(pid) not in device["processes"]
                and (current is None or current == self.known_owned[pid])
            )
            if pid not in owned_pids and not retired:
                continue
            if 0 <= wall - event["stamp"] <= max_age:
                heartbeats.append(event["stamp"])
        if not all(any(abs(stamp - heartbeat) <= max_gap for heartbeat in heartbeats) for stamp in required_stamps):
            raise AdmissionError("missing or stale owned-device heartbeat")

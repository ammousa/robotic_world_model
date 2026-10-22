"""Small CPU-only launch and evidence checks; no simulator or orchestration."""

from __future__ import annotations

import dataclasses
import enum
import functools
import hashlib
import inspect
import json
import math
import subprocess
from pathlib import Path

# rsl_rl stores the runner config in W&B as ``train_cfg`` (rsl_rl >= 3) or ``runner_cfg`` (the rsl_rl_rwm
# fork this repository trains with); the ``research`` block rides inside it.
RUNNER_CONFIG_KEYS = ("train_cfg", "runner_cfg")


def runner_config(config) -> dict:
    """The W&B config's runner block, whichever key this rsl_rl version stores it under."""
    return next((config[k] for k in RUNNER_CONFIG_KEYS if isinstance(config.get(k), dict)), {})


def canonical_config(value, _path=()):
    """Serialize resolved settings without address-bearing repr or launch-only fields."""
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            raise ValueError("NaN config value")
        if math.isinf(value):
            return {"float": "+inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, enum.Enum):
        return canonical_config(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, slice):
        return {"slice": canonical_config([value.start, value.stop, value.step])}
    if type(value).__module__ == "torch" and type(value).__name__ in {"dtype", "device"}:
        return {type(value).__name__: str(value)}
    if isinstance(value, functools.partial):
        return {
            "partial": canonical_config(value.func),
            "args": canonical_config(value.args, (*_path, "partial_args")),
            "keywords": canonical_config(value.keywords or {}, (*_path, "partial_keywords")),
        }
    if inspect.ismethod(value) and not inspect.isclass(value.__self__):
        raise ValueError("instance-bound callable needs an explicit stable config representation")
    if inspect.isfunction(value) or inspect.isclass(value) or inspect.ismethod(value) or inspect.isbuiltin(value):
        if "<locals>" in value.__qualname__ or value.__name__ == "<lambda>":
            raise ValueError("local/lambda callable has no stable config identity")
        return {"callable": f"{value.__module__}.{value.__qualname__}"}
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("config mapping keys must be strings")
            # These affect launch placement/identity, not the scientific settings.
            if key in {"run_name", "log_dir", "log_root_path", "research"} and _path in {
                (),
                ("agent",),
                ("env",),
                ("train_cfg",),
                ("runner_cfg",),
            }:
                continue
            if key == "wandb" and _path == () and isinstance(item, dict):
                item = {k: v for k, v in item.items() if k not in {"label", "tags"}}
            if key == "device" and isinstance(item, str) and item.startswith("cuda:"):
                item = "cuda"
            result[key] = canonical_config(item, (*_path, key))
        return result
    if isinstance(value, list | tuple):
        return [canonical_config(v, (*_path, "[]")) for v in value]
    if dataclasses.is_dataclass(value):
        return canonical_config({f.name: getattr(value, f.name) for f in dataclasses.fields(value)}, _path)
    if hasattr(value, "to_dict"):
        return canonical_config(value.to_dict(), _path)
    if type(value).__module__.startswith("numpy") and hasattr(value, "tolist"):
        return canonical_config(value.tolist())
    raise ValueError(f"unsupported config type: {type(value).__module__}.{type(value).__qualname__}")


def config_hash(value):
    return hashlib.sha256(
        json.dumps(canonical_config(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def checkpoint_problems(path):
    """Load tensors safely on CPU; never retry with unrestricted pickle loading."""
    try:
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            return ["checkpoint is not a model-state mapping"]
        states = [
            checkpoint[k]
            for k in (
                "model_state_dict",
                "actor_state_dict",
                "policy_state_dict",
                "state_dict",
                "system_dynamics_state_dict",
            )
            if k in checkpoint
        ]
        if not states or not all(isinstance(s, dict) and s for s in states):
            return ["checkpoint has no nonempty recognized model state"]
        if not any(isinstance(t, torch.Tensor) and t.numel() for s in states for t in s.values()):
            return ["checkpoint model state contains no weights"]

        def finite(value):
            if isinstance(value, torch.Tensor):
                return not (value.is_floating_point() or value.is_complex()) or bool(torch.isfinite(value).all())
            if isinstance(value, dict):
                return all(finite(v) for v in value.values())
            if isinstance(value, list | tuple):
                return all(finite(v) for v in value)
            return not isinstance(value, float) or math.isfinite(value)

        return [] if finite(checkpoint) else ["checkpoint contains non-finite state"]
    except Exception as error:
        return [f"checkpoint CPU validation failed: {type(error).__name__}: {error}"]


# Column/reduction definitions of the standard evaluation metrics. ``eval_results.aggregate_metrics``
# computes the evaluation summaries from these same definitions, so the two cannot drift apart.
STANDARD_REDUCERS = {
    "lin_vel_tracking_rmse": {"column": "lin_vel_tracking_error", "reduction": "rms"},
    "ang_vel_tracking_rmse": {"column": "ang_vel_tracking_error", "reduction": "rms"},
    "success_rate": {"column": "success", "reduction": "rate"},
    "fall_rate": {"column": "fell", "reduction": "rate"},
    "timeout_rate": {"column": "timed_out", "reduction": "rate"},
    "survival_rate": {"column": "timed_out", "reduction": "rate"},
    **{
        name: {"column": column, "reduction": "mean"}
        for name, column in {
            "mean_episode_return": "episode_return",
            "mean_episode_duration_s": "time_seconds",
            "mean_episode_length_steps": "episode_length",
            "mean_distance_travelled_m": "distance_travelled",
            "mean_path_length_m": "path_length",
            "realized_terrain_difficulty_mean": "terrain_difficulty",
            "realized_static_friction_mean": "static_friction",
            "realized_dynamic_friction_mean": "dynamic_friction",
            "realized_added_mass_kg_mean": "added_mass_kg",
            "realized_max_push_delta_velocity_mps_mean": "max_push_delta_velocity_mps",
            "realized_num_pushes_mean": "num_pushes",
        }.items()
    },
}
STANDARD_METRICS = set(STANDARD_REDUCERS)


def metric_reducer_specs(evaluation):
    """Shared profile-aware standard and custom metric definitions."""
    specs = {**STANDARD_REDUCERS, **evaluation.get("metric_reducers", {})}
    if evaluation.get("profile") == "forward_course":
        specs["survival_rate"] = {"column": "fell", "reduction": "rate", "complement": True}
    return specs


def required_metric_keys(evaluation):
    """Primary + secondary + required metric keys, in declaration order, deduplicated."""
    keys = list(
        dict.fromkeys(
            [
                evaluation["primary_metric"]["name"],
                *evaluation["secondary_metrics"],
                *evaluation.get("required_metrics", []),
            ]
        )
    )
    if evaluation.get("profile") == "world_model":
        keys = [key for key in keys if key not in evaluation.get("artifact_metrics", {})]
    return keys


def declared_metric_reducers(evaluation):
    """Select the protocol's declared evidence from the shared definitions."""
    specs = metric_reducer_specs(evaluation)
    return {key: specs[key] for key in required_metric_keys(evaluation)}


def validate_reducers(evaluation):
    artifacts = evaluation.get("artifact_metrics", {})
    if artifacts:
        if evaluation.get("profile") != "world_model" or not isinstance(artifacts, dict):
            raise ValueError("artifact metrics require the explicit world-model profile")
        declared = {evaluation["primary_metric"]["name"], *evaluation["secondary_metrics"]}
        for name, spec in artifacts.items():
            if name not in declared or not isinstance(spec, dict) or set(spec) != {"source", "columns", "selection"}:
                raise ValueError("artifact metric needs its exact source, columns and selection")
            if (
                spec["source"] not in {"scored_rows.json", "provenance.json"}
                or not isinstance(spec["columns"], list)
                or not spec["columns"]
            ):
                raise ValueError("unsupported world-model artifact metric")
            if not all(isinstance(column, str) and column for column in spec["columns"]) or not isinstance(
                spec["selection"], dict
            ):
                raise ValueError("invalid world-model artifact metric columns/selection")
    reducers = evaluation.get("metric_reducers", {})
    if not isinstance(reducers, dict):
        raise ValueError("metric_reducers must be a mapping")
    for metric, spec in reducers.items():
        if (
            not isinstance(metric, str)
            or not metric
            or metric in STANDARD_METRICS
            or not isinstance(spec, dict)
            or set(spec) != {"column", "reduction"}
        ):
            raise ValueError("custom metric reducer requires column/reduction and cannot override standard metrics")
        if (
            not isinstance(spec["column"], str)
            or not spec["column"]
            or not isinstance(spec["reduction"], str)
            or spec["reduction"] not in {"mean", "rms", "rate"}
        ):
            raise ValueError("unsupported custom metric reducer")
        if evaluation.get("metric_columns", {}).get(metric, spec["column"]) != spec["column"]:
            raise ValueError("metric_columns differs from metric_reducers")
    for metric, column in evaluation.get("metric_columns", {}).items():
        if metric in STANDARD_METRICS:
            expected = metric_reducer_specs(evaluation)[metric]["column"]
            if column != expected:
                raise ValueError("metric_columns cannot change standard metric definitions")
    metrics = required_metric_keys(evaluation)
    if set(metrics) - STANDARD_METRICS - set(reducers):
        raise ValueError(
            f"unsupported metrics require metric_reducers: {sorted(set(metrics) - STANDARD_METRICS - set(reducers))}"
        )


def protocol_aggregate(base, protocol):
    """Keep repository summaries and apply declared profile/custom definitions."""
    evaluation = (protocol or {}).get("evaluation", {})
    reducers = dict(evaluation.get("metric_reducers", {}))
    if evaluation.get("profile") == "forward_course":
        reducers["survival_rate"] = metric_reducer_specs(evaluation)["survival_rate"]

    def aggregate(rows):
        result = base(rows)
        for metric, spec in reducers.items():
            values = [r.get(spec["column"]) for r in rows]
            if not values or any(not isinstance(v, int | float) or not math.isfinite(v) for v in values):
                raise ValueError(f"missing/non-finite custom metric or profile metric {metric}")
            if spec["reduction"] == "rate" and any(v not in (0, 1) for v in values):
                raise ValueError(f"non-binary custom rate {metric}")
            result[metric] = (
                math.sqrt(sum(v * v for v in values) / len(values))
                if spec["reduction"] == "rms"
                else sum(values) / len(values)
            )
            if spec.get("complement"):
                result[metric] = 1 - result[metric]
        return result

    return aggregate


def retry_reference(metadata, path):
    """Bind a retry to the actual original run's immutable provenance artifact."""
    raw = Path(path).read_bytes()
    original = json.loads(raw)
    for key in ("cell_id", "protocol_id", "hypothesis_id", "settings_id"):
        if original.get(key) != metadata.get(key):
            raise ValueError(f"original provenance differs in {key}")
    if type(original.get("attempt")) is not int or original["attempt"] >= metadata["attempt"]:
        raise ValueError("original attempt must precede retry")
    if original.get("job_type") != metadata.get("job_type") and not (
        metadata.get("job_type") == "smoke" and original.get("job_type") == "train"
    ):
        raise ValueError("original provenance differs in job_type")
    fp = original.get("runtime_fingerprint", {})
    if not fp.get("commit") or not fp.get("resolved_config_hash") or fp.get("config_hash_version") != 2:
        raise ValueError("original provenance needs commit and stable resolved config hash v2")
    return {
        "original_provenance": str(Path(path).resolve()),
        "original_provenance_sha256": hashlib.sha256(raw).hexdigest(),
        "original_attempt": original["attempt"],
        "original_run_dir": original.get("run_dir"),
        "original_commit": fp["commit"],
        "original_config_hash": fp["resolved_config_hash"],
    }


@functools.lru_cache(maxsize=256)
def _committed_source_digest(root, commit, name):
    try:
        raw = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=root, stderr=subprocess.PIPE)
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("original startup numerical source is unavailable") from error
    return hashlib.sha256(raw).hexdigest()


def _startup_evidence(root, group, name, digest):
    path = (root / name).resolve()
    if not path.is_relative_to((root / "logs/queue" / group).resolve()):
        raise ValueError("startup evidence must remain in its comparison queue")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("controlled startup evidence changed")
    return raw


def _startup_numeric_contract(protocol):
    # Only admission and launch bookkeeping may differ after this technical failure.
    result = json.loads(json.dumps(protocol))
    for name in ("protocol_id", "protocol_version"):
        result.pop(name, None)
    for name in (
        "training_context_admission",
        "startup_correction",
        "controlled_startup_replacements",
        "pilot_queue_directory",
        "pilot_trace_filename",
    ):
        result["world_model"].pop(name, None)
    if "retry_policy" in result:
        result["retry_policy"].pop("max_attempts", None)  # Technical attempt authority, not numerical science.
    return result


def verify_controlled_startup(metadata, fingerprint, *, root=None):
    """A reviewed changed-source startup replacement is never an identical retry.

    This narrow path requires actual closed queue evidence when startup produced no
    run provenance. Ordinary training and identical-infrastructure retry rules stay
    in verify_retry. No historical provenance or optimizer-step count is invented.
    """
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    protocol = metadata.get("protocol_contract", {})
    cell = metadata.get("cell_id")
    declared = protocol.get("world_model", {}).get("controlled_startup_replacements", {}).get(cell)
    if (
        not declared
        or protocol.get("domain") != "world_model"
        or metadata.get("job_type") != "smoke"
        or metadata.get("attempt") != declared.get("attempt")
        or metadata.get("controlled_startup_replacement") != declared
        or not declared
        or declared.get("attempt") not in (2, 3)
        or declared.get("original_attempt") != 1
        or declared.get("ordinary_retry") is not False
        or declared.get("settings_identical") is not False
        or metadata.get("settings_identical") is not False
        or metadata.get("original_provenance")
    ):
        raise ValueError("invalid controlled startup replacement")
    if not fingerprint.get("commit") or fingerprint["commit"] == declared["original_source_commit"]:
        raise ValueError("controlled startup replacement requires its reviewed new source commit")
    group = protocol["group"]
    raw = _startup_evidence(root, group, declared["closed_queue"], declared["closed_queue_sha256"])
    events = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    name = declared["original_job_name"]
    starts = [event for event in events if event.get("event") == "start" and event.get("name") == name]
    exits = [event for event in events if event.get("event") == "exit" and event.get("name") == name]
    if (
        len(starts) != 1
        or len(exits) != 1
        or exits[0].get("returncode") != -15
        or not events
        or events[-1].get("event") != "queue_end"
        or not any(event.get("event") == "isolation_failed" for event in events)
    ):
        raise ValueError("startup replacement requires one closed failed original attempt")
    original = starts[0]
    old = original["research"]
    if (
        name != cell + "-smoke"
        or old.get("attempt") != 1
        or old.get("job_type") != "smoke"
        or any(old.get(key) != metadata.get(key) for key in ("cell_id", "variant", "seed", "task", "hypothesis_id"))
        or original["source"].get("dirty") is not False
        or original["source"].get("commit") != declared["original_source_commit"]
        or old.get("protocol_id") != declared["original_protocol_id"]
        or protocol.get("protocol_id") == old.get("protocol_id")
    ):
        raise ValueError("controlled startup lineage differs from original")
    if _startup_numeric_contract(protocol) != _startup_numeric_contract(old["protocol_contract"]):
        raise ValueError("controlled startup replacement changes scientific settings")
    audit = json.loads(_startup_evidence(root, group, declared["failure_audit"], declared["failure_audit_sha256"]))
    if (
        audit.get("original_protocol") != declared["original_protocol_id"]
        or audit.get("source_commit") != declared["original_source_commit"]
        or audit.get("exit_codes", {}).get(name) != -15
        or audit.get("child_log_bytes", {}).get(name) != 0
        or audit.get("training_output_directories", {}).get(name) != []
        or audit.get("no_automatic_retry") is not True
    ):
        raise ValueError("startup failure evidence does not support this replacement")
    for entry in audit["files"].values():
        _startup_evidence(root, group, entry["path"], entry["sha256"])
    child_log = (root / declared["closed_queue"]).parent / (name + ".log")
    old_outputs = root / "logs/rsl_rl/anymal_d_world_model"
    if child_log.stat().st_size != 0 or any(old_outputs.glob("*_" + name)):
        raise ValueError("original startup has run output; cannot use the pre-provenance replacement boundary")
    unchanged = protocol["world_model"]["startup_correction"]["unchanged_model_source_sha256"]
    required = {"src/world_modeling/" + name for name in ("train.py", "core.py", "dataset.py", "kinematics.py")}
    if not required.issubset(unchanged):
        raise ValueError("startup replacement lacks numerical-source binding")
    for name, digest in unchanged.items():
        path = (root / name).resolve()
        if not path.is_relative_to((root / "src/world_modeling").resolve()) or path.name == "campaign.py":
            raise ValueError("invalid startup numerical-source path")
        if (
            _committed_source_digest(str(root), declared["original_source_commit"], name) != digest
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise ValueError("startup replacement numerical source changed")
    if declared["attempt"] == 3:
        _verify_additional_startup(metadata, fingerprint, root, declared)
    return {**metadata, "settings_identical": False, "controlled_startup_replacement_validated": True}


def _verify_additional_startup(metadata, fingerprint, root, declared):
    protocol = metadata["protocol_contract"]
    authority = protocol["world_model"]["startup_correction"].get("user_restart", {})
    previous = declared.get("previous_attempt_record", {})
    if (
        declared.get("user_authorized_additional_start") is not True
        or authority.get("actual_user_instruction") != "Start it now on gpu 2 and 3"
        or authority.get("maximum_ordinary_attempts") != 2
        or previous.get("attempt") != 2
        or fingerprint["commit"] == previous.get("source_commit")
    ):
        raise ValueError("third startup requires explicit user restart and a new reviewed source")
    raw = _startup_evidence(root, protocol["group"], previous["closed_queue"], previous["closed_queue_sha256"])
    events = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    starts = [e for e in events if e.get("event") == "start" and e.get("name") == previous["job_name"]]
    exits = [e for e in events if e.get("event") == "exit" and e.get("name") == previous["job_name"]]
    if (
        len(starts) != 1
        or len(exits) != 1
        or exits[0].get("returncode") != -15
        or events[-1].get("event") != "queue_end"
    ):
        raise ValueError("third startup requires one closed failed second attempt")
    old = starts[0]["research"]
    if (
        old.get("attempt") != 2
        or previous["job_name"] != metadata["cell_id"] + "-attempt2-smoke"
        or old.get("protocol_id") != previous["protocol_id"]
        or starts[0]["source"].get("commit") != previous["source_commit"]
        or starts[0]["source"].get("dirty") is not False
        or _startup_numeric_contract(protocol) != _startup_numeric_contract(old["protocol_contract"])
    ):
        raise ValueError("second startup lineage or numerical settings differ")
    verify_controlled_startup(old, {"commit": previous["source_commit"]}, root=root)
    audit = json.loads(
        _startup_evidence(root, protocol["group"], previous["failure_audit"], previous["failure_audit_sha256"])
    )
    name = previous["job_name"]
    if (
        audit.get("source_commit") != previous["source_commit"]
        or audit.get("protocol_id") != previous["protocol_id"]
        or audit.get("exit_codes", {}).get(name) != -15
        or audit.get("child_log_bytes", {}).get(name) != 0
        or audit.get("training_output_directories", {}).get(name) != []
        or (root / previous["closed_queue"]).parent.joinpath(name + ".log").stat().st_size != 0
        or any((root / "logs/rsl_rl/anymal_d_world_model").glob("*_" + name))
    ):
        raise ValueError("second startup has incompatible failure evidence or run output")
    for entry in audit["files"].values():
        _startup_evidence(root, protocol["group"], entry["path"], entry["sha256"])


def verify_retry(metadata, fingerprint, *, source_only=False):
    if metadata.get("controlled_startup_replacement"):
        return verify_controlled_startup(metadata, fingerprint)
    if metadata.get("attempt", 1) <= 1:
        return metadata
    ordinary_max = (
        metadata.get("protocol_contract", {})
        .get("world_model", {})
        .get("startup_correction", {})
        .get("user_restart", {})
        .get("maximum_ordinary_attempts")
    )
    if ordinary_max is not None and metadata["attempt"] > ordinary_max:
        raise ValueError("ordinary retry exceeds unchanged attempt limit")
    path = metadata.get("original_provenance")
    if not path:
        raise ValueError("retry requires --original-provenance from the original run")
    reference = retry_reference(metadata, path)
    if any(metadata.get(k) != v for k, v in reference.items()):
        raise ValueError("original retry provenance changed")
    if fingerprint.get("commit") != reference["original_commit"]:
        raise ValueError("retry source commit differs from original; amend protocol and re-review")
    source_only = source_only or metadata.get("job_type") == "smoke"
    if not source_only and (
        fingerprint.get("config_hash_version") != 2
        or fingerprint.get("resolved_config_hash") != reference["original_config_hash"]
    ):
        raise ValueError("retry resolved configuration differs from original")
    return {**metadata, "settings_identical": True if not source_only else None}

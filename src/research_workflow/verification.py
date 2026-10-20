"""Small CPU-only launch and evidence checks; no simulator or orchestration."""

from __future__ import annotations

import dataclasses
import enum
import functools
import hashlib
import inspect
import json
import math
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


def verify_retry(metadata, fingerprint, *, source_only=False):
    if metadata.get("attempt", 1) <= 1:
        return metadata
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

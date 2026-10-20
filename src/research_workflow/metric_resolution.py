"""Shared, read-only metric preflight for reports, diagnostics and scientific comparisons.

Profiles are explicit preferences and semantic declarations, never proof of observations
or comparability. Unknown discovered metrics are per-run diagnostics without a direction.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from numbers import Real

from .verification import config_hash, metric_reducer_specs, runner_config


def finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def research(run):
    cfg = run.config
    return runner_config(cfg).get("research") or cfg.get("research") or {}


def successful(run):
    return getattr(run, "state", None) == "finished" and not set(run.tags) & {
        "status:failed",
        "status:invalid",
        "status:superseded",
        "status:smoke",
    }


def descriptor(
    name,
    *,
    scope="train",
    direction=None,
    unit=None,
    aggregation="per_run",
    reduction="logged_scalar",
    comparison="evaluation",
    axis=None,
    required=False,
):
    return dict(
        name=name,
        scope=scope,
        direction=direction,
        unit=unit,
        aggregation=aggregation,
        reduction=reduction,
        comparison=comparison,
        axis=axis,
        required=required,
    )


# Explicit domain registrations. No substring/name heuristic supplies scientific semantics.
RL_METRICS = (
    descriptor("Train/mean_reward", direction="higher", unit="episode return", aggregation="mean_seed"),
    descriptor("Train/mean_episode_length", direction=None, unit="steps", aggregation="mean_seed"),
    descriptor("Episode_Termination/base_contact", direction=None, unit="termination statistic"),
    descriptor("Metrics/base_velocity/error_vel_xy", direction="lower", unit="m/s", aggregation="mean_seed"),
    descriptor("Loss/value", direction=None, unit="configured value-loss units", comparison="loss"),
    descriptor("Policy/mean_std", direction=None, unit="action distribution units"),
)
EVAL_METRICS = (
    descriptor(
        "eval/success_rate",
        scope="eval",
        direction="higher",
        unit="fraction",
        reduction="rate",
        aggregation="mean_seed",
    ),
    descriptor(
        "eval/fall_rate",
        scope="eval",
        direction="lower",
        unit="fraction",
        reduction="rate",
        aggregation="mean_seed",
    ),
    descriptor(
        "eval/lin_vel_tracking_rmse",
        scope="eval",
        direction="lower",
        unit="m/s",
        reduction="rms",
        aggregation="mean_seed",
    ),
)
CATALOG = {d["name"]: d for d in (*RL_METRICS, *EVAL_METRICS)}
# Historical scored-table names, registered explicitly rather than guessed from suffixes.
CATALOG["eval/lin_vel_rmse"] = {**CATALOG["eval/lin_vel_tracking_rmse"], "name": "eval/lin_vel_rmse"}
CATALOG["eval/ang_vel_rmse"] = descriptor(
    "eval/ang_vel_rmse",
    scope="eval",
    direction="lower",
    unit="rad/s",
    reduction="rms",
    aggregation="mean_seed",
)


def validate_descriptor(spec):
    if not isinstance(spec, dict) or not isinstance(spec.get("name"), str) or not spec["name"]:
        raise ValueError("reporting metric needs an exact nonempty name")
    result = {**descriptor(spec["name"]), **spec}
    if result["scope"] not in {"train", "eval"} or result["direction"] not in {None, "higher", "lower"}:
        raise ValueError("invalid metric scope/direction")
    if result["aggregation"] not in {"per_run", "mean_seed"} or result["comparison"] not in {
        "loss",
        "evaluation",
    }:
        raise ValueError("invalid metric aggregation/comparison semantics")
    if type(result["required"]) is not bool or (result["axis"] is not None and not isinstance(result["axis"], str)):
        raise ValueError("invalid metric required/axis declaration")
    for field in ("unit", "reduction"):
        if result[field] is not None and (not isinstance(result[field], str) or not result[field]):
            raise ValueError("metric unit/reduction must be nonempty text or null")
    if result["aggregation"] == "mean_seed" and (
        not result["unit"] or result["reduction"] == "logged_scalar" and result["scope"] == "eval"
    ):
        raise ValueError("mean_seed aggregation requires declared units and an evaluation reduction")
    return result


def metric_title(spec):
    direction = {"higher": "↑ better", "lower": "↓ better"}.get(spec.get("direction"))
    return spec["name"] + (f" ({direction})" if direction else "")


def _get(cfg, path):
    value = cfg
    for key in path.removeprefix("config.").split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


class MetricResolver:
    """One plan and one observation cache shared by all reporting consumers.

    Numeric history is sampled, bounded and joint with its proposed x axis. Coverage
    means at least one finite observed pair, not a completeness/whole-stream guarantee.
    Scientific comparison eligibility is separate from plot availability.
    """

    def __init__(self, runs, *, train_metrics=None, eval_metrics=None, groupby=None, samples=128):
        self.runs = sorted(
            (
                r
                for r in runs
                if r.job_type in {"train", "eval"} and not set(r.tags) & {"status:superseded", "status:smoke"}
            ),
            key=lambda r: r.id,
        )
        self.requested = {"train": train_metrics, "eval": eval_metrics}
        self.groupby = groupby
        self.samples = samples
        self.cache = {}
        self.errors, self.warnings = [], []
        self._plan = None
        self._specs = {}
        self.by_id = {r.id: r for r in self.runs}

    def profile(self, run):
        cfg, rs = run.config, research(run)
        domain = (rs.get("protocol_contract") or {}).get("domain")
        if domain == "world_model":
            return "world_model"
        if domain == "rl" or runner_config(cfg):
            return "rl"
        return "unknown"

    def identity(self, run):
        cfg = run.config
        reporting = (research(run).get("protocol_contract") or {}).get("reporting", {})
        paths = self.groupby if self.groupby is not None and run.job_type == "train" else reporting.get("groupby")
        if isinstance(paths, dict):
            paths = paths.get(run.job_type)
        if isinstance(paths, str):
            paths = [paths]
        if paths:
            values = [_get(cfg, p) for p in paths]
            if any(
                not isinstance(v, (str, int, float, bool)) or (isinstance(v, (int, float)) and not math.isfinite(v))
                for v in values
            ):
                self.errors.append(f"{run.id}: grouping keys absent/non-scalar: {paths}")
                return run.name, []
            return " / ".join(map(str, values)), list(paths)
        for path in (
            "config.train_cfg.research.variant",
            "config.runner_cfg.research.variant",
            "config.research.variant",
        ):
            value = _get(cfg, path)
            if isinstance(value, (str, int)) and not isinstance(value, bool):
                return str(value), [path]
        tags = sorted(t.removeprefix("variant:") for t in run.tags if t.startswith("variant:"))
        self.warnings.append(f"{run.id}: no validated variant config; per-run grouping")
        return tags[0] if len(tags) == 1 else run.name, []

    def history_keys(self, run):
        value = getattr(run, "history_keys", None)
        if value is None:
            try:
                value = getattr(run, "_attrs", {}).get("historyKeys", {})
            except PermissionError:
                value = {}
        if isinstance(value, str):
            value = json.loads(value)
        if isinstance(value, dict):
            return value.get("keys", value)
        return {}

    def observe(self, run, spec):
        key, scope = spec["name"], spec["scope"]
        cache_key = (run.id, key, spec.get("axis"))
        if cache_key in self.cache:
            return self.cache[cache_key]
        if scope == "eval":
            value = run.summary.get(key)
            result = {
                "axis": None,
                "rows": [{key: float(value)}] if finite(value) else [],
                "status": "observed summary",
            }
        else:
            preferred = spec.get("axis")
            axes = [preferred] if preferred else ["_step"]
            result = {"axis": None, "rows": [], "status": "no finite numeric observations"}
            for axis in axes:
                try:
                    if not callable(getattr(run, "history", None)):
                        break
                    frame = run.history(keys=[axis, key], samples=self.samples, pandas=True)
                    rows = frame.to_dict("records") if hasattr(frame, "to_dict") else list(frame)
                    valid = [
                        {axis: float(r[axis]), key: float(r[key])}
                        for r in rows
                        if finite(r.get(key)) and finite(r.get(axis))
                    ]
                    # Old test/legacy APIs may omit _step but expose row order. Do not invent it.
                    if valid:
                        result = {
                            "axis": axis,
                            "rows": sorted(valid, key=lambda r: r[axis]),
                            "status": "sampled finite history",
                            "rejected_points": len(rows) - len(valid),
                        }
                        break
                except Exception as exc:
                    result["status"] = "history unavailable: " + type(exc).__name__
        self.cache[cache_key] = result
        return result

    def compatibility(self, run, spec):
        """Exact declared/scored context; unverified identities never become aggregate cohorts."""
        cfg, rs = run.config, research(run)
        contract = rs.get("protocol_contract") or {}
        declared = contract.get("reporting", {}).get("compatibility", {})
        # Compare each run's own definition, never silently reuse another protocol's semantics.
        own_spec = next(
            (
                validate_descriptor(d)
                for d in contract.get("reporting", {}).get("metrics", [])
                if d.get("name") == spec["name"] and d.get("scope", "train") == spec["scope"]
            ),
            None,
        )
        primary = contract.get("evaluation", {}).get("primary_metric", {})
        if own_spec is None and spec["scope"] == "eval" and primary.get("name") == spec["name"].removeprefix("eval/"):
            own_spec = {
                **spec,
                "direction": primary.get("direction"),
                "unit": primary.get("unit"),
                "reducer_spec": metric_reducer_specs(contract["evaluation"]).get(primary["name"]),
            }
        spec = own_spec or spec
        dataset = declared.get("dataset") or {
            k: cfg[k]
            for k in (
                "data",
                "dataset_name",
                "dataset_task",
                "dataset_scenario",
                "dataset_version",
                "dataset_source_runs",
                "normalization_id",
                "state_keys",
            )
            if k in cfg
        }
        if not dataset:
            dataset = contract.get("evaluation", {}).get("dataset") or rs.get("dataset_id")
        if not dataset and self.profile(run) == "rl":
            dataset = rs.get("task")
        evaluation = (
            declared.get("evaluation") or contract.get("evaluation") or rs.get("protocol") or cfg.get("evaluation")
        )
        # Reuse the existing resolved-config canonicalizer; never erase objective differences.
        loss = (
            declared.get("loss_definition")
            or {k: cfg[k] for k in ("training", "model", "optimizer", "normalization_id", "state_keys") if k in cfg}
            or runner_config(cfg).get("algorithm")
        )
        parent = self.by_id.get(rs.get("train_run_id")) if run.job_type == "eval" else None
        parent_context = self.compatibility(parent, spec) if parent is not None else None
        signature = {
            "protocol_id": rs.get("protocol_id"),
            "dataset": {
                "declared": dataset,
                "verified_manifest": (rs.get("dataset_verification") or {}).get("manifest_sha256"),
                "resolved": {
                    k: cfg[k] for k in ("data", "dataset_version", "normalization_id", "state_keys") if k in cfg
                },
            },
            "evaluation": {"declared": evaluation, "resolved": cfg.get("evaluation")},
            "budget": {
                "declared": contract.get("training_budget"),
                "resolved": {k: cfg[k] for k in ("epochs", "max_iterations") if k in cfg},
                "train_cfg_iterations": runner_config(cfg).get("max_iterations"),
            },
            "semantics": spec,
        }
        if parent_context:
            signature["training_context"] = parent_context["signature"]
        if parent_context and not dataset:
            signature["dataset"] = parent_context["signature"]["dataset"]
            dataset = signature["dataset"]
        if spec["comparison"] == "loss":
            signature["loss_definition"] = {
                "declared": loss,
                "resolved": {
                    k: cfg[k] for k in ("model", "training", "optimizer", "normalization_id", "state_keys") if k in cfg
                },
            }
        missing = [
            k
            for k, v in (
                ("protocol_id", rs.get("protocol_id")),
                ("dataset", dataset),
                ("evaluation", evaluation),
            )
            if not v
        ]
        if parent_context and not parent_context["verified"]:
            missing.append("verified training context")
        if spec["comparison"] == "loss" and not loss:
            missing.append("loss_definition")
        if run.job_type == "eval":
            parent = self.by_id.get(rs.get("train_run_id"))
            if parent is None or not successful(parent):
                missing.append("successful training attempt link")
            elif any(research(parent).get(k) != rs.get(k) for k in ("protocol_id", "variant", "seed")):
                missing.append("matching parent protocol/variant/seed")

        return {
            "id": config_hash(signature),
            "verified": not missing,
            "missing": missing,
            "signature": signature,
        }

    def _candidates(self, scope):
        selected = {}
        scope_runs = [r for r in self.runs if r.job_type == scope]
        for run in self.runs:
            contract = research(run).get("protocol_contract") or {}
            reporting = contract.get("reporting", {})
            for raw in reporting.get("metrics", []):
                spec = validate_descriptor(raw)
                if spec["scope"] != scope:
                    continue
                previous = selected.get(spec["name"])
                if previous and previous[0] != spec:
                    self.errors.append(f"conflicting protocol metric definitions: {spec['name']}")
                selected[spec["name"]] = (spec, "protocol")
            if run.job_type == scope == "eval" and contract.get("evaluation", {}).get("primary_metric"):
                ev = contract["evaluation"]
                primary = ev["primary_metric"]
                for name in [
                    primary["name"],
                    *ev.get("secondary_metrics", []),
                    *ev.get("required_metrics", []),
                ]:
                    if self.profile(run) == "rl":
                        key = name if name.startswith("eval/") else "eval/" + name
                    else:
                        # A scientific scored-table requirement is not necessarily a
                        # W&B scalar key. Only use exact profile or explicit mappings.
                        key = name if name in selected else None
                        if key is None:
                            self.warnings.append(
                                f"{name}: scored evidence requirement has no registered scalar mapping; declare"
                                " reporting.metrics for panels"
                            )
                            continue
                    reducer = metric_reducer_specs(ev).get(name, {})
                    known = CATALOG.get(key, descriptor(key, scope="eval"))
                    spec = {
                        **known,
                        "required": True,
                        "reduction": reducer.get("reduction", known["reduction"]),
                    }
                    if name == primary["name"]:
                        spec.update(direction=primary.get("direction"), unit=primary.get("unit"))
                    selected.setdefault(key, (validate_descriptor(spec), "protocol"))
        for run in scope_runs:
            profile = self.profile(run)
            preferences = (RL_METRICS if scope == "train" else EVAL_METRICS) if profile == "rl" else ()
            for spec in preferences:
                selected.setdefault(spec["name"], (dict(spec), "profile"))
        if scope == "eval":
            # Preserve RL level/difficulty panels, but only for declared cells with
            # actual scalar observations. Reuse the parent's explicit semantics.
            base_specs = list(selected.items())
            for run in scope_runs:
                ev = (research(run).get("protocol_contract") or {}).get("evaluation", {})
                cells = [f"L{level}" for level in ev.get("levels", [])] + list(ev.get("difficulties", []))
                for name, (spec, origin) in base_specs:
                    if not name.startswith("eval/"):
                        continue
                    for cell in cells:
                        key = f"eval/{cell}/{name.removeprefix('eval/')}"
                        if finite(run.summary.get(key)):
                            selected.setdefault(key, ({**spec, "name": key, "required": False}, origin + "_slice"))
        requested = self.requested[scope]
        if requested is not None:
            required = {k: v for k, v in selected.items() if v[0]["required"]}
            for name in requested:
                key = (
                    name
                    if scope == "train"
                    or name in selected
                    or name in CATALOG
                    or any(name in r.summary for r in scope_runs)
                    or name.startswith("eval/")
                    else "eval/" + name
                )
                spec, origin = selected.get(key, (dict(CATALOG.get(key, descriptor(key, scope=scope))), "explicit"))
                required[key] = ({**spec, "required": True}, origin if origin == "protocol" else "explicit")
            return required
        # Discovery is fallback, not an unbounded dump of every counter/hash into the report.
        if not any(self.observe(r, spec)["rows"] for spec, _ in selected.values() for r in scope_runs):
            for run in scope_runs:
                available = self.history_keys(run) if scope == "train" else run.summary
                for key, value in sorted(available.items()):
                    if key.startswith("_") or key in {"epoch", "Step"}:
                        continue
                    numeric = (
                        finite(value)
                        if scope == "eval"
                        else isinstance(value, dict)
                        and any(
                            t.get("type") == "number" and t.get("count", 0) > 0 for t in value.get("typeCounts", [])
                        )
                    )
                    if numeric:
                        selected.setdefault(key, (dict(CATALOG.get(key, descriptor(key, scope=scope))), "discovery"))
        return selected

    def resolve(self):
        if self._plan is not None:
            return self._plan
        identities = {r.id: self.identity(r) for r in self.runs}
        metrics, panels, omitted = [], [], []
        for scope in ("train", "eval"):
            scope_runs = [r for r in self.runs if r.job_type == scope]
            for key, (spec, origin) in self._candidates(scope).items():
                self._specs[key] = spec
                seen, coverage, groups = (
                    [],
                    defaultdict(
                        lambda: {
                            "observed": 0,
                            "recorded": 0,
                            "total": 0,
                            "successful_observed": 0,
                            "successful_total": 0,
                        }
                    ),
                    defaultdict(list),
                )
                missing = []
                for run in scope_runs:
                    variant, grouping = identities[run.id]
                    cov = coverage[variant]
                    cov["total"] += 1
                    cov["successful_total"] += int(successful(run))
                    metadata = self.history_keys(run).get(key, {}) if scope == "train" else {}
                    recorded = (
                        key in run.summary
                        if scope == "eval"
                        else any(t.get("count", 0) > 0 for t in metadata.get("typeCounts", []))
                    )
                    observation = self.observe(run, spec)
                    cov["recorded"] += int(recorded or bool(observation["rows"]))
                    if not observation["rows"]:
                        missing.append({"run_id": run.id, "reason": observation["status"]})
                        continue
                    seen.append(run.id)
                    cov["observed"] += 1
                    cov["successful_observed"] += int(successful(run))
                    context = self.compatibility(run, spec)
                    role = "successful" if successful(run) else "diagnostic_attempt"
                    aggregate = (
                        spec["aggregation"] == "mean_seed"
                        and context["verified"]
                        and role == "successful"
                        and bool(grouping)
                    )
                    group = (context["id"], observation["axis"], tuple(grouping), role, aggregate)
                    groups[group].append(run.id)
                if not seen:
                    if spec["required"] and scope_runs:
                        self.errors.append(f"required metric {key}: no finite numeric observations")
                    omitted.append(
                        {
                            **spec,
                            "source": origin,
                            "total": len(scope_runs),
                            "observed": 0,
                            "missing": missing,
                        }
                    )
                    continue
                if spec["required"]:
                    required_missing = [m for m in missing if successful(self.by_id[m["run_id"]])]
                    if required_missing:
                        self.errors.append(
                            f"required metric {key}: missing on successful runs"
                            f" {[m['run_id'] for m in required_missing]}"
                        )
                if missing:
                    self.warnings.append(
                        f"{key}: finite numeric coverage {len(seen)}/{len(scope_runs)}; see per-run missing reasons"
                    )
                metric = {
                    **spec,
                    "source": origin,
                    "observed": len(seen),
                    "recorded": sum(c["recorded"] for c in coverage.values()),
                    "total": len(scope_runs),
                    "run_ids": seen,
                    "coverage_by_variant": dict(coverage),
                    "missing": missing,
                    "observation_fingerprints": {
                        i: hashlib.sha256(
                            json.dumps(self.observe(self.by_id[i], spec), sort_keys=True).encode()
                        ).hexdigest()
                        for i in seen
                    },
                }
                metrics.append(metric)
                for (cohort, axis, grouping, role, aggregate), ids in groups.items():
                    seed_units = [(identities[i][0], research(self.by_id[i]).get("seed")) for i in ids]
                    if aggregate and (
                        any(seed is None for _, seed in seed_units) or len(seed_units) != len(set(seed_units))
                    ):
                        aggregate = False
                        self.warnings.append(f"{key}: missing/duplicate independent training seed; per-run display")
                    panels.append(
                        {
                            "name": key,
                            "scope": scope,
                            "title": metric_title(spec),
                            "axis": axis,
                            "run_ids": ids,
                            "run_names": [self.by_id[i].name for i in ids],
                            "groupby": list(grouping) if aggregate else [],
                            "variant_keys": list(grouping),
                            "aggregation": "mean_seed" if aggregate else "per_run",
                            "role": role,
                            "cohort": cohort,
                            "semantics": spec,
                            "compatibility": self.compatibility(self.by_id[ids[0]], spec),
                        }
                    )
        self._plan = {
            "schema_version": 1,
            "metrics": metrics,
            "omitted_metrics": omitted,
            "panels": panels,
            "identities": {k: {"variant": v[0], "groupby": v[1]} for k, v in identities.items()},
            "attempts": {
                "successful": [r.id for r in self.runs if successful(r)],
                "diagnostic": [r.id for r in self.runs if not successful(r)],
            },
            "errors": sorted(set(self.errors)),
            "warnings": sorted(set(self.warnings)),
            "numeric_validation": (
                f"at least one finite axis/value pair in at most {self.samples} sampled rows; summary scalars for eval"
            ),
        }
        return self._plan

    def comparison(self, variant, baseline, seeds, metric, *, train_ids=None):
        """Shared eligibility for derived scored-table comparisons, not only native panels."""
        identities = {r.id: self.identity(r)[0] for r in self.runs}
        self.resolve()
        key = metric if metric in self._specs or metric in CATALOG or metric.startswith("eval/") else "eval/" + metric
        spec = dict(self._specs.get(key, CATALOG.get(key, descriptor(key, scope="eval"))))
        seeds = list(seeds)
        if len(seeds) != len(set(seeds)):
            return {
                "eligible_seeds": [],
                "blocked_seeds": seeds,
                "status": "invalid_due_to_protocol_or_provenance",
            }
        eligible, blocked, cohorts = [], [], set()
        for seed in seeds:
            pair = []
            for name in (variant, baseline):
                candidates = [
                    r
                    for r in self.runs
                    if r.job_type == "train"
                    and (train_ids is None or r.id in train_ids)
                    and identities[r.id] == name
                    and research(r).get("seed") == seed
                    and successful(r)
                ]
                if len(candidates) != 1:
                    pair = []
                    break
                pair.append(candidates[0])
            contexts = [self.compatibility(r, spec) for r in pair]
            if any(r.job_type == "eval" for r in self.runs):
                scored = [
                    [
                        e
                        for e in self.runs
                        if e.job_type == "eval" and research(e).get("train_run_id") == r.id and successful(e)
                    ]
                    for r in pair
                ]
                contexts = [self.compatibility(e[0], spec) for e in scored] if all(len(e) == 1 for e in scored) else []
            if (
                len(contexts) == 2
                and spec.get("unit") is not None
                and spec["aggregation"] == "mean_seed"
                and (spec["scope"] != "eval" or spec["reduction"] != "logged_scalar")
                and all(c["verified"] for c in contexts)
                and contexts[0]["id"] == contexts[1]["id"]
            ):
                eligible.append(seed)
                cohorts.add(contexts[0]["id"])
            else:
                blocked.append(seed)
        if len(cohorts) > 1:
            blocked, eligible = list(seeds), []
        return {
            "eligible_seeds": eligible,
            "blocked_seeds": blocked,
            "status": "compatible" if eligible and not blocked else "invalid_due_to_protocol_or_provenance",
        }


def coverage_markdown(plan):
    lines = [
        "Validated metric coverage across all attempts; failed/incomplete attempts are diagnostic only.",
        "",
        (
            "| Metric | Recorded/all | Finite observed/all | Variant | Finite observed/all | Successful finite/all |"
            " Direction | Units |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    for metric in plan["metrics"]:
        for variant, cov in metric["coverage_by_variant"].items():
            lines.append(
                f"| {metric['name']} | {metric['recorded']}/{metric['total']} |"
                f" {metric['observed']}/{metric['total']} | {variant} | {cov['observed']}/{cov['total']} |"
                f" {cov['successful_observed']}/{cov['successful_total']} | {metric['direction'] or 'unspecified'} |"
                f" {metric['unit'] or 'unspecified'} |"
            )
    if not plan["metrics"]:
        lines.append("No validated numeric metrics are available; no metric panels are generated.")
    return "\n".join(lines)

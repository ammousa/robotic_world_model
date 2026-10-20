"""Deterministic review evidence and an optional, separately approved interpretation.

No model calls or experiment execution. Evidence is a versioned projection of W&B and
verified notebook tables; review prose cannot supply its own measured numbers.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from statistics import mean, stdev

from .metric_resolution import MetricResolver
from .verification import runner_config

SCHEMA_VERSION = 1
SECTIONS = (
    "Executive Summary & Recommendation",
    "Research Questions & Hypotheses",
    "Experimental Design & Reproducibility",
    "Comparative Results & Statistical Evidence",
    "Training Dynamics & Failure Analysis",
    "Performance vs Computational Cost",
    "Scientific Interpretation & Limitations",
    "Prioritised Next Experiments",
    "Researcher Decision & Rationale",
)
TOKEN = re.compile(r"\{\{([^{}]+)\}\}")


def _mapping(value):
    # W&B SummarySubDict is dict-like without registering as Mapping; inspect the
    # type because its missing-attribute lookup raises KeyError.
    return isinstance(value, dict) or (
        callable(getattr(type(value), "items", None)) and callable(getattr(type(value), "keys", None))
    )


def clean(value):
    """JSON values only; missing/non-finite measurements are null, never zero."""
    if _mapping(value):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, "item"):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def identifier(prefix, value):
    data = {k: v for k, v in value.items() if k != prefix + "_id"}
    return (
        prefix
        + "-"
        + hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()[
            :24
        ]
    )


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def seed_interval(values, *, valid=True):
    """Exploratory Student-t interval across independent training-seed differences.

    Five seeds is a conservative display minimum, not a guarantee of power or normality.
    Never resample episodes as independent training replicates or emit a pilot CI.
    """
    values = list(values)
    if not all(_finite(v) for v in values):
        return {"n": len(values), "mean": None, "sd": None, "ci95": None, "status": "non-finite evidence"}
    n = len(values)
    avg = mean(values) if n else None
    sd = stdev(values) if n > 1 else None
    result = {"n": n, "mean": avg, "sd": sd, "ci95": None, "status": "low-N; descriptive only"}
    if n >= 5 and valid:
        from scipy.stats import t

        margin = float(t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
        result.update(
            ci95=[avg - margin, avg + margin],
            status=(
                "exploratory paired training-seed t interval; assumes independent seeds and approximately normal"
                " differences; no multiplicity correction"
            ),
        )
    elif not valid:
        result["status"] = "provisional; comparability or provenance problems; no interval"
    return result


def training_diagnostics(run, keys=None, *, resolver=None):
    """Bounded sampled histories: descriptive flags, never a causal collapse diagnosis."""
    resolver = resolver or MetricResolver([run], train_metrics=keys, samples=512)
    plan = resolver.resolve()
    specs = [m for m in plan["metrics"] if m["scope"] == "train" and run.id in m["run_ids"]]
    result = {
        "run_id": run.id,
        "sampling": f"W&B sampled history, at most {resolver.samples} rows per metric",
        "metrics": {},
    }
    for spec in specs:
        key = spec["name"]
        try:
            observation = resolver.observe(run, spec)
            raw = [row[key] for row in observation["rows"]]
            values = [float(x) for x in raw if _finite(x)]
            if len(values) < 8:
                result["metrics"][key] = {
                    "status": "insufficient finite sampled history",
                    "finite_points": len(values),
                }
                continue
            tail = values[-max(4, len(values) // 5) :]
            differences = [b - a for a, b in zip(values, values[1:], strict=False)]
            scale = max(abs(mean(values)), stdev(values), 1e-12)
            reversals = sum(a * b < 0 for a, b in zip(differences, differences[1:], strict=False)) / max(
                1, len(differences) - 1
            )
            med = sorted(values)[len(values) // 2]
            mad = sorted(abs(v - med) for v in values)[len(values) // 2]
            outliers = sum(abs(v - med) > 6 * mad for v in values) if mad else None
            result["metrics"][key] = {
                "status": "descriptive sampled diagnostics",
                "finite_points": len(values),
                "rejected_points": observation.get("rejected_points", 0),
                "last": values[-1],
                "tail_mean": mean(tail),
                "tail_sd": stdev(tail),
                "tail_drift": tail[-1] - tail[0],
                "plateau_candidate": abs(tail[-1] - tail[0]) < 0.02 * scale,
                "oscillation_candidate": reversals > 0.7 and stdev(tail) > 0.1 * scale,
                "outliers_mad6": outliers,
                "collapse_candidate": key == "Policy/mean_std" and values[-1] < 0.1 * max(values),
            }
        except Exception as exc:  # Reporting remains possible for partially logged histories.
            result["metrics"][key] = {"status": "unavailable", "reason": type(exc).__name__}
    return result


def configuration_differences(records):
    """Expose resolved differences for scientific review without guessing which are confounders."""
    flattened = {}
    for record in records:
        if record["job_type"] != "train" or "status:superseded" in record["tags"]:
            continue
        fields = {}

        def walk(value, path, fields=fields):
            if isinstance(value, dict):
                for key, item in value.items():
                    walk(item, path + "/" + key)
            else:
                fields[path] = value

        walk(record["config"], "")
        flattened[record["id"]] = fields
    paths = sorted({path for fields in flattened.values() for path in fields})
    return [
        {"path": path, "runs": {run: fields.get(path) for run, fields in flattened.items()}}
        for path in paths
        if len({json.dumps(fields.get(path), sort_keys=True) for fields in flattened.values()}) > 1
    ]


def cost_effects(tables):
    """Metric changes and resource changes for the same selected training seeds, never all retries."""
    costs = {(r["variant"], r["train_seed"]): r for r in tables.get("cost_selected_seeds", [])}
    rows = []
    for effect in tables.get("effects", []):
        if effect.get("slice") != "ALL":
            continue
        variant, baseline = effect["variant"], effect["baseline"]
        for (name, seed), mine in costs.items():
            base = costs.get((baseline, seed))
            if name != variant or not base:
                continue
            row = {
                "variant": variant,
                "baseline": baseline,
                "train_seed": seed,
                "metric": effect["metric"],
                "metric_delta": None,
            }
            scored = [r for r in tables.get("per_seed", []) if r.get("slice") == "ALL" and r.get("train_seed") == seed]
            a = next((r.get(effect["metric"]) for r in scored if r["variant"] == variant), None)
            b = next((r.get(effect["metric"]) for r in scored if r["variant"] == baseline), None)
            row["metric_delta"] = a - b if _finite(a) and _finite(b) else None
            for resource in (
                "wall_time_h",
                "gpu_hours",
                "parameters",
                "inference_latency_ms",
                "environment_steps",
            ):
                a, b = mine.get(resource), base.get(resource)
                row[resource + "_delta"] = a - b if _finite(a) and _finite(b) else None
            rows.append(row)
    return rows


def run_snapshot(run, *, project, entity=None, analysis=False):
    """Fingerprint all observable evidence, including media/table refs and artifact versions.

    No artifact payloads are downloaded. An unreadable artifact listing fails closed;
    legacy/test objects without the capability explicitly record it as unavailable.
    """
    summary = dict(run.summary)
    if analysis:
        summary = {
            k: v
            for k, v in summary.items()
            if k not in {"analysis/scientific_evidence", "_runtime", "_step", "_timestamp", "_wandb.runtime"}
        }

    def fingerprint_value(value):
        if _mapping(value):
            return {str(k): fingerprint_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [fingerprint_value(v) for v in value]
        if isinstance(value, float) and not math.isfinite(value):
            return {"nonfinite": str(value)}
        return clean(value)

    artifacts = {}
    for method in ("logged_artifacts", "used_artifacts"):
        getter = getattr(run, method, None)
        if not callable(getter):
            artifacts[method] = None
            continue
        try:
            artifacts[method] = sorted(
                (
                    {
                        "name": getattr(a, "name", None),
                        "version": getattr(a, "version", None),
                        "digest": getattr(a, "digest", None),
                        "type": getattr(a, "type", None),
                    }
                    for a in getter()
                ),
                key=lambda a: json.dumps(a, sort_keys=True),
            )
        except Exception as exc:
            raise ValueError("cannot fingerprint " + method + " for " + run.id) from exc
    metadata = runner_config(run.config).get("research") or run.config.get("research") or {}
    return clean(
        {
            "id": run.id,
            "group": getattr(run, "group", None),
            "git_commit": getattr(run, "commit", None),
            "name": run.name,
            "job_type": run.job_type,
            "state": getattr(run, "state", "unknown"),
            "url": getattr(run, "url", None) or (
                f"https://wandb.ai/{entity}/{project}/runs/{run.id}" if entity else None
            ),
            "tags": sorted(run.tags),
            "config": dict(run.config),
            "research": metadata,
            "summary": summary,
            "summary_fingerprint": identifier("summary", fingerprint_value(summary)),
            "artifacts": artifacts,
        }
    )


def bind_publication_evidence(evidence, analysis_run=None, hypothesis=None, *, rendering=None, analysis_code=None):
    """Bind review to the selected analysis, its code/tables, and the current Git hypothesis.

    Obtain this final review input from publish-report --dry-run. A run-analysis
    sidecar alone cannot know the identities of artifacts that have not been uploaded.
    """
    validate_evidence(evidence)
    result = dict(evidence)
    result["publication_inputs"] = {
        "analysis": run_snapshot(analysis_run, project=evidence["project"], analysis=True) if analysis_run else None,
        "hypothesis": hypothesis,
        "rendering": rendering or {},
        "analysis_code": analysis_code,
    }
    result["evidence_id"] = identifier("evidence", result)
    return result


def build_evidence(
    runs,
    *,
    project,
    group,
    entity=None,
    tables=None,
    problems=(),
    baseline=None,
    histories=True,
    hypothesis=None,
    resolver=None,
):
    """Use verified notebook tables when available; summaries alone never become scored evidence."""
    runs = list(runs)
    resolver = resolver or MetricResolver(runs)
    resolution = resolver.resolve()
    records, diagnostics, issues = [], [], [*problems, *resolution["errors"]]
    inventory = (tables or {}).get("inventory", [])
    inventory = inventory.to_dict("records") if hasattr(inventory, "to_dict") else inventory
    selected_train_ids = {row.get("run_id") for row in inventory}
    for run in sorted(runs, key=lambda r: (r.job_type, r.id)):
        if run.job_type not in {"train", "eval"} or "status:smoke" in run.tags:
            continue
        metadata = runner_config(run.config).get("research") or run.config.get("research") or {}
        records.append(run_snapshot(run, project=project, entity=entity))
        if run.job_type == "train" and histories:
            diagnostics.append(training_diagnostics(run, resolver=resolver))
        selected = (
            not selected_train_ids
            or run.id in selected_train_ids
            or (run.job_type == "eval" and metadata.get("train_run_id") in selected_train_ids)
        )
        if selected and (
            getattr(run, "state", "unknown") != "finished" or {"status:invalid", "status:failed"} & set(run.tags)
        ):
            issues.append("unfinished/failed/invalid attempt: " + run.id)
    data = {k: clean(v.to_dict("records") if hasattr(v, "to_dict") else v) for k, v in (tables or {}).items()}
    if not data.get("per_seed"):
        issues.append("Verified scored notebook tables unavailable; summary metrics are exploratory only.")
    statistics = []
    for row in data.get("effects", []):
        seeds = [
            r.get("train_seed")
            for r in data.get("per_seed", [])
            if r.get("variant") == row.get("variant") and r.get("slice") == row.get("slice")
        ]
        eligibility = resolver.comparison(
            row.get("variant"),
            row.get("baseline"),
            seeds,
            row.get("metric", ""),
            train_ids=selected_train_ids or None,
        )
        metric = row.get("metric")
        scored = data.get("per_seed", [])
        expected = []
        for seed in seeds:
            pair = [
                [
                    r.get(metric)
                    for r in scored
                    if r.get("variant") == name and r.get("train_seed") == seed and r.get("slice") == row.get("slice")
                ]
                for name in (row.get("variant"), row.get("baseline"))
            ]
            if not all(len(values) == 1 and _finite(values[0]) for values in pair):
                break
            expected.append(pair[0][0] - pair[1][0])
        supplied = row.get("seed_effects", [])
        if (
            len(expected) != len(seeds)
            or len(supplied) != len(seeds)
            or not all(
                _finite(a) and math.isclose(a, b, abs_tol=1e-9, rel_tol=1e-9)
                for a, b in zip(supplied, expected, strict=False)
            )
        ):
            eligibility = {
                **eligibility,
                "status": "invalid_due_to_protocol_or_provenance",
                "reason": "effects do not match unique finite scored training-seed outcomes",
            }
        stat = {
            **row,
            **seed_interval(row.get("seed_effects", []), valid=not issues),
            "compatibility": eligibility,
        }
        if eligibility["status"] != "compatible":
            stat.update(
                mean=None,
                sd=None,
                ci95=None,
                delta=None,
                seed_effects=[],
                status="invalid_due_to_protocol_or_provenance",
            )
            issues.append(
                f"Incompatible or unverified comparison: {row.get('variant')} vs {row.get('baseline')} on"
                f" {row.get('metric')}"
            )
        statistics.append(stat)
    invalid = {
        (r["variant"], r["baseline"], r["metric"], r.get("slice"))
        for r in statistics
        if r["status"] == "invalid_due_to_protocol_or_provenance"
    }
    if invalid:
        data["incompatible_effects_diagnostic_only"] = [
            r for r in data.get("effects", []) if (r["variant"], r["baseline"], r["metric"], r.get("slice")) in invalid
        ]
        data["effects"] = [
            r
            for r in data.get("effects", [])
            if (r["variant"], r["baseline"], r["metric"], r.get("slice")) not in invalid
        ]
    paired_valid, paired_diagnostic = [], []
    for row in data.get("paired_effects", []):
        eligible = resolver.comparison(
            row.get("variant"),
            row.get("baseline"),
            [row.get("train_seed")],
            row.get("aggregate_metric", row.get("metric", "")),
            train_ids=selected_train_ids or None,
        )
        (paired_valid if eligible["status"] == "compatible" else paired_diagnostic).append(row)
    if paired_diagnostic:
        data["paired_effects"] = paired_valid
        data["incompatible_paired_effects_diagnostic_only"] = paired_diagnostic
    contracts = []
    for record in records:
        contract = record["research"].get("protocol_contract")
        if contract and contract not in contracts:
            contracts.append(contract)
    if hypothesis:
        from research_workflow import validate_contract

        hypothesis = validate_contract("hypothesis", hypothesis)
        if any(c.get("hypothesis_id") != hypothesis["hypothesis_id"] for c in contracts):
            issues.append("Hypothesis contract does not match the evaluated protocol.")
    if not contracts:
        issues.append("Legacy workflow: hypothesis, controls and acceptance criteria unavailable in run contracts.")
    result = clean(
        {
            "schema_version": SCHEMA_VERSION,
            "project": project,
            "group": group,
            "baseline": baseline,
            "runs": records,
            "contracts": contracts,
            "hypothesis": hypothesis,
            "tables": data,
            "statistics": statistics,
            "training_diagnostics": diagnostics,
            "metric_resolution": resolution,
            "configuration_differences": configuration_differences(records),
            "performance_cost": cost_effects(data),
            "failure_slices": [r for r in data.get("per_seed", []) if r.get("slice") != "ALL"],
            "limitations": sorted(set(issues)),
            "interpretation": "Deterministic evidence only; no automatic winner or causal conclusion.",
        }
    )
    # Recompute interval eligibility after all limitations (including missing contracts) are known.
    if result["limitations"]:
        for row in result["statistics"]:
            if row["status"] == "invalid_due_to_protocol_or_provenance":
                continue
            row.update(seed_interval(row.get("seed_effects", []), valid=False))
    result["evidence_id"] = identifier("evidence", result)
    return result


def validate_evidence(evidence):
    if (
        not isinstance(evidence, dict)
        or type(evidence.get("schema_version")) is not int
        or evidence["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError("unsupported evidence schema_version")
    for key in (
        "project",
        "group",
        "runs",
        "tables",
        "statistics",
        "limitations",
        "contracts",
        "training_diagnostics",
    ):
        if key not in evidence:
            raise ValueError("missing evidence field: " + key)
    for key in ("runs", "statistics", "limitations", "contracts", "training_diagnostics"):
        if not isinstance(evidence[key], list):
            raise ValueError("evidence field must be a list: " + key)
    if not isinstance(evidence["tables"], dict):
        raise ValueError("evidence tables must be a mapping")
    if evidence.get("evidence_id") != identifier("evidence", evidence):
        raise ValueError("evidence content hash mismatch")
    return evidence


def facts(evidence):
    """JSON-pointer-like references; every empirical value has an exact, verifiable address."""
    out = {}

    def walk(value, path):
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, path + "/" + key.replace("~", "~0").replace("/", "~1"))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, path + "/" + str(index))
        elif value is not None:
            out[path] = value

    walk(evidence, "")
    return out


def review_template(evidence):
    validate_evidence(evidence)
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_id": evidence["evidence_id"],
        "author": "AI",
        "generation": {"kind": "template", "agent": None, "evidence_id": evidence["evidence_id"]},
        "verdict": "inconclusive",
        "recommendation": "defer",
        "executive": [],
        "interpretation": [],
        "limitations": [],
        "criteria": [],
        "next_experiments": [],
    }


def validate_review(review, evidence):
    """Measured quantities must be inserted from evidence tokens, never free-typed prose.

    Citations establish traceability, not entailment: researcher review remains mandatory.
    Proposed experiment designs may contain numbers, explicitly labelled as proposals.
    """
    validate_evidence(evidence)
    if (
        not isinstance(review, dict)
        or type(review.get("schema_version")) is not int
        or review["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError("unsupported review schema_version")
    if review.get("evidence_id") != evidence["evidence_id"]:
        raise ValueError("review refers to stale or different evidence")
    if review.get("author") not in {"AI", "researcher"}:
        raise ValueError("review author must be AI or researcher")
    if review.get("verdict") not in {
        "supported",
        "not_supported",
        "inconclusive",
        "invalid_due_to_protocol_or_provenance",
    }:
        raise ValueError("invalid hypothesis verdict")
    generation = review.get("generation")
    if not isinstance(generation, dict) or generation.get("kind") not in {"agent", "researcher"}:
        raise ValueError("a template is not an agent-generated scientific review")
    if (
        generation.get("evidence_id") != evidence["evidence_id"]
        or not isinstance(generation.get("agent"), str)
        or not generation["agent"].strip()
    ):
        raise ValueError("review generation must identify the authoring agent and exact evidence")
    if review["author"] == "AI" and generation["kind"] != "agent":
        raise ValueError("AI review requires agent generation attribution")
    if review.get("recommendation") not in {"keep", "reject", "modify", "extend", "defer"}:
        raise ValueError("review requires a recommendation, separate from the actual Gate D decision")
    insufficient = not evidence["statistics"] or any(row["n"] < 2 for row in evidence["statistics"])
    if (evidence["limitations"] or insufficient) and review["verdict"] in {"supported", "not_supported"}:
        raise ValueError("evidence limitations require an inconclusive or invalid verdict")
    available = facts(evidence)

    def claim(item):
        if not isinstance(item, dict) or item.get("kind") not in {
            "measured",
            "interpretation",
            "causal_hypothesis",
        }:
            raise ValueError("claim kind must distinguish measurement, interpretation and causal hypothesis")
        text, refs = item.get("text"), item.get("evidence")
        if not isinstance(text, str) or not text.strip() or not isinstance(refs, list) or not refs:
            raise ValueError("every claim requires text and evidence references")
        tokens = TOKEN.findall(text)
        if any(not isinstance(ref, str) or ref not in available for ref in refs + tokens) or not set(tokens) <= set(
            refs
        ):
            raise ValueError("unknown or uncited evidence reference")
        if re.search(r"\d", TOKEN.sub("", text)):
            raise ValueError("quantitative prose must use {{/evidence/path}} tokens")
        return item

    for section in ("executive", "interpretation", "limitations"):
        if not isinstance(review.get(section), list) or not review[section]:
            raise ValueError("review needs nonempty " + section)
        for item in review[section]:
            claim(item)
    if not isinstance(review.get("criteria"), list) or not review["criteria"]:
        raise ValueError("review requires explicit acceptance-criterion assessments")
    for item in review["criteria"]:
        claim(item)
        if item.get("classification") not in {
            "met",
            "not_met",
            "inconclusive",
            "invalid_due_to_protocol_or_provenance",
        }:
            raise ValueError("invalid criterion classification")
    criterion_paths = {
        path
        for path in available
        if path.startswith("/contracts/")
        and any(
            "/" + category + "/" in path
            for category in ("support_criteria", "contradiction_criteria", "inconclusive_criteria")
        )
    }
    cited_criteria = {ref for item in review["criteria"] for ref in item["evidence"]}
    if criterion_paths - cited_criteria:
        raise ValueError("review must assess every predeclared acceptance criterion")
    experiments = review.get("next_experiments")
    if not isinstance(experiments, list) or not 0 <= len(experiments) <= 3:
        raise ValueError("propose at most three next experiments")
    for experiment in experiments:
        for key in (
            "hypothesis",
            "modification",
            "controls",
            "discriminating_outcomes",
            "metrics",
            "acceptance_criteria",
            "cost_estimate",
        ):
            if not isinstance(experiment.get(key), str) or not experiment[key].strip():
                raise ValueError("next experiment requires " + key)
        claim(experiment.get("rationale"))
    result = dict(review)
    result.pop("review_id", None)
    result["review_id"] = identifier("review", result)
    if review.get("review_id", result["review_id"]) != result["review_id"]:
        raise ValueError("review content hash mismatch")
    return result


def validate_approval(approval, review):
    if (
        not isinstance(approval, dict)
        or type(approval.get("schema_version")) is not int
        or approval["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError("unsupported review approval schema_version")
    if (
        approval.get("review_id") != review["review_id"]
        or approval.get("actor") not in {"user", "researcher"}
        or approval.get("approved") is not True
    ):
        raise ValueError("approval must bind this exact review and record the active researcher's actual approval")
    return approval


def render_review(review, evidence, approval=None):
    """Pure rendering; no invented approval and no unapproved interpretation in the report."""
    if review is None:
        return {
            "executive": "Scientific interpretation pending researcher review. Evidence only.",
            "interpretation": "Scientific interpretation pending; no reviewed causal explanation.",
            "next": "Follow-up experiments pending scientific review; none authorized.",
        }
    review = validate_review(review, evidence)
    if approval is None:
        pending = render_review(None, evidence)
        pending["executive"] += " A structured review exists but approval is pending."
        return pending
    validate_approval(approval, review)
    available = facts(evidence)

    def render(item):
        text = TOKEN.sub(lambda m: str(available[m[1]]), item["text"])
        refs = ", ".join("`" + r + "`" for r in item["evidence"])
        return f"- **{item['kind']}**: {text} (evidence `{evidence['evidence_id']}`: {refs})"

    label = (
        f"{review['author']}-authored scientific interpretation; approved by {approval['actor']}. Recommendation only;"
        " Gate D decides."
    )
    executive = (
        label
        + "\n\nHypothesis verdict: **"
        + review["verdict"]
        + "**; proposed decision: **"
        + review["recommendation"]
        + "**\n\n"
        + "\n".join(map(render, review["executive"]))
    )
    interpretation = label + "\n\n" + "\n".join(map(render, review["interpretation"]))
    interpretation += "\n\nLimitations\n\n" + "\n".join(map(render, review["limitations"]))
    interpretation += "\n\nAcceptance criteria\n\n" + "\n".join(
        render(i) + " — " + i["classification"] for i in review["criteria"]
    )
    next_steps = ["Proposed experiments, in priority order. These proposals authorize no compute."]
    for index, experiment in enumerate(review["next_experiments"], 1):
        next_steps.append(f"\n### Priority {index}\n" + render(experiment["rationale"]))
        next_steps.extend(
            f"- **{key.replace('_', ' ')} (proposal):** {value}"
            for key, value in experiment.items()
            if key != "rationale"
        )
    return {"executive": executive, "interpretation": interpretation, "next": "\n".join(next_steps)}


def evidence_markdown(evidence, section):
    """Concise deterministic main narrative, with full tables retained as supporting evidence."""
    if not evidence:
        return "Structured evidence unavailable for this legacy analysis. Scientific interpretation pending."
    validate_evidence(evidence)
    prefix = f"Automatically computed evidence `{evidence['evidence_id']}`.\n\n"
    tables = evidence["tables"]
    if section == "design":
        lines = [
            f"- [{r['name']}]({r['url']}) · `{r['id']}` · {r['state']} · protocol"
            f" `{r['research'].get('protocol_id', 'unavailable')}` · commit"
            f" `{r['research'].get('commit', 'unavailable')}` · checkpoint"
            f" `{r['research'].get('checkpoint', 'unavailable')}`"
            for r in evidence["runs"]
        ]
        return prefix + "\n".join(lines)
    if section == "hypothesis":
        hypothesis = evidence.get("hypothesis") or {}
        question = "\n".join(
            f"**{k.replace('_', ' ')}:** {hypothesis.get(k, 'unavailable')}"
            for k in ("claim", "mechanism", "held_fixed", "changes")
        )
        return (
            prefix
            + question
            + "\n\n"
            + f"Baseline: `{evidence.get('baseline') or 'unavailable'}`.\n\n"
            + "\n".join(
                "```json\n"
                + json.dumps(
                    {
                        k: c.get(k)
                        for k in (
                            "hypothesis_id",
                            "baseline_variants",
                            "training_budget",
                            "support_criteria",
                            "contradiction_criteria",
                            "inconclusive_criteria",
                            "evaluation",
                        )
                    },
                    indent=2,
                )
                + "\n```"
                for c in evidence["contracts"]
            )
            + "\nHypothesis claim/mechanism and controlled variables require the referenced approved hypothesis; not"
            " reconstructed from scores."
        )
    if section == "statistics":
        lines = [
            f"- `{r.get('variant')}` vs `{r.get('baseline')}` · {r.get('metric')} · {r.get('slice')}: seed differences"
            f" {r.get('seed_effects')}; mean {r['mean']}; SD {r['sd']}; CI {r['ci95']}; {r['status']}"
            for r in evidence["statistics"]
        ]
        return prefix + "\n".join(lines or ["No verified seed differences available."])
    if section == "limitations":
        return (
            prefix
            + "\n".join("- " + p for p in evidence["limitations"])
            + "\nExploratory intervals are not confirmatory tests or practical-significance decisions."
        )
    if section == "diagnostics":
        rows = []
        for run in evidence["training_diagnostics"]:
            for metric, values in run["metrics"].items():
                flags = [
                    k for k in ("plateau_candidate", "oscillation_candidate", "collapse_candidate") if values.get(k)
                ]
                rows.append(
                    f"| `{run['run_id']}` | {metric} | {values.get('finite_points', 'unknown')} |"
                    f" {values.get('tail_mean', 'unknown')} | {', '.join(flags) or values['status']} |"
                )
        return (
            prefix
            + "Sampled history flags require inspection; plateau does not prove convergence, and oscillation/outliers"
            " do not establish failure causes. Sample efficiency needs matched environment-step histories and a"
            " predeclared target. Weak configurations are exposed in the per-seed slice tables, without pooling"
            " levels.\n\n| Run | Metric | Finite sampled points | Tail mean | Diagnostic |\n|---|---|---|---|---|\n"
            + "\n".join(rows[:18])
            + "\nFull diagnostics and failure slices remain in structured evidence."
        )
    if section == "cost":
        rows = []
        for r in tables.get("cost_all_attempt_spend", []):
            rows.append(
                f"| {r['variant']} | {r.get('attempts')} | {r.get('wall_time_h_known_subtotal')} |"
                f" {r.get('wall_time_h_unknown_attempts')} | {r.get('gpu_hours_known_subtotal')} |"
                f" {r.get('gpu_hours_unknown_attempts')} |"
            )
        return (
            prefix
            + "Measured resources only; unavailable costs remain null. Selected-attempt performance/resource"
            " differences are in `performance_cost`; retry spend is separate. A metric improvement does not imply a"
            " better cost trade-off.\n\n| Variant | Attempts | Known wall hours | Unknown attempts | Known GPU-hours"
            " | Unknown attempts |\n|---|---|---|---|---|---|\n"
            + "\n".join(rows)
            + "\nParameter count, latency, memory and pretraining cost are retained when logged; missing values are"
            " unknown."
        )
    raise ValueError("unknown evidence section")

"""Small scientific contracts and best-effort runtime provenance; no runtime database."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import yaml
from pathlib import Path


def content_id(kind: str, document: dict) -> str:
    """Hash semantic YAML content, independent of mapping order, formatting or its own ID."""
    data = {k: v for k, v in document.items() if k != f"{kind}_id"}
    return (
        kind[0]
        + "-"
        + hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()[
            :24
        ]
    )


def _required(doc: dict, fields: str) -> None:
    if not isinstance(doc, dict):
        raise ValueError("expected a mapping")
    missing = [k for k in fields.split() if k not in doc or doc[k] is None]
    if missing:
        raise ValueError(f"missing fields: {', '.join(missing)}")


def _positive(value, field: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field} must be a positive integer")


def validate_contract(kind: str, doc: dict, hypothesis: dict | None = None) -> dict:
    """Validate v1 contracts. Human approval is deliberately not inferred from file contents."""
    if not isinstance(doc, dict) or type(doc.get("schema_version")) is not int or doc.get("schema_version") != 1:
        raise ValueError("expected mapping with schema_version: 1")
    if kind == "hypothesis":
        _required(
            doc,
            "group hypothesis_version title claim mechanism changes held_fixed inputs "
            "gradient_ownership baselines primary_metric secondary_metrics constraints "
            "evaluation_unit paired_factors anticipated_confounders support_if contradict_if "
            "inconclusive_if checkpoint_export_impact",
        )
        for field in (
            "changes",
            "held_fixed",
            "baselines",
            "secondary_metrics",
            "constraints",
            "paired_factors",
            "anticipated_confounders",
            "support_if",
            "contradict_if",
            "inconclusive_if",
        ):
            if not isinstance(doc[field], list):
                raise ValueError(f"{field} must be a list")
        if not doc["baselines"] or not doc["support_if"] or not doc["contradict_if"] or not doc["inconclusive_if"]:
            raise ValueError("baseline and falsification criteria cannot be empty")
        _positive(doc["hypothesis_version"], "hypothesis_version")
        if not isinstance(doc["inputs"], dict):
            raise ValueError("inputs must be a mapping")
        _required(doc["inputs"], "deployable privileged")
        for key in ("claim", "mechanism", "group", "title", "evaluation_unit"):
            if not isinstance(doc[key], str) or not doc[key].strip():
                raise ValueError(f"{key} must be nonempty text")
        if not isinstance(doc["primary_metric"], dict):
            raise ValueError("primary_metric must define name, direction and unit")
        _required(doc["primary_metric"], "name direction unit")
        if doc["primary_metric"]["direction"] not in ("higher", "lower"):
            raise ValueError("primary metric direction must be higher or lower")
    elif kind == "protocol":
        _required(
            doc,
            "group hypothesis_id protocol_version project variants seeds training_budget "
            "compute_pool baseline_variants checkpoint_rule evaluation cost_accounting smoke_gate "
            "retry_policy support_criteria contradiction_criteria inconclusive_criteria assumptions",
        )
        _positive(doc["protocol_version"], "protocol_version")
        if hypothesis is None:
            raise ValueError("protocol validation requires its hypothesis")
        h = validate_contract("hypothesis", hypothesis)
        if doc["hypothesis_id"] != h["hypothesis_id"] or doc["group"] != h["group"]:
            raise ValueError("protocol references a different hypothesis/group")
        _required(doc["evaluation"], "primary_metric secondary_metrics constraints paired_factors")
        for key in ("secondary_metrics", "required_metrics"):
            metrics = doc["evaluation"].get(key, [])
            if not isinstance(metrics, list) or any(not isinstance(m, str) or not m.strip() for m in metrics):
                raise ValueError(f"evaluation.{key} must contain exact nonempty metric keys")
        columns = doc["evaluation"].get("metric_columns", {})
        if not isinstance(columns, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) or not v for k, v in columns.items()
        ):
            raise ValueError("evaluation.metric_columns must map metric keys to scored row columns")
        for field in (
            "seeds",
            "compute_pool",
            "baseline_variants",
            "cost_accounting",
            "support_criteria",
            "contradiction_criteria",
            "inconclusive_criteria",
            "assumptions",
        ):
            if not isinstance(doc[field], list):
                raise ValueError(f"{field} must be a list")
        domain = doc.get("domain", "rl")
        if domain not in {"rl", "world_model"}:
            raise ValueError("domain must be rl or the explicit world_model adapter")
        if domain == "world_model" and not isinstance(doc.get("world_model"), dict):
            raise ValueError("world_model domain requires its executable adapter settings")
        if doc["checkpoint_rule"] != "final" and not (
            domain == "world_model" and doc["checkpoint_rule"] == "validation_min"
        ):
            raise ValueError("tooling supports checkpoint_rule: final; other rules need an explicit adapter")
        for category, source in (
            ("support_criteria", "support_if"),
            ("contradiction_criteria", "contradict_if"),
            ("inconclusive_criteria", "inconclusive_if"),
        ):
            if doc[category] != h[source]:
                raise ValueError(f"{category} differs from hypothesis")
        if (
            not doc["compute_pool"]
            or any(type(g) is not int or g < 0 for g in doc["compute_pool"])
            or len(set(doc["compute_pool"])) != len(doc["compute_pool"])
        ):
            raise ValueError("compute_pool must contain distinct nonnegative GPU indices")
        for key in ("primary_metric", "secondary_metrics", "constraints", "paired_factors"):
            if doc["evaluation"].get(key) != h[key]:
                raise ValueError(f"evaluation.{key} differs from hypothesis; return to frame")
        if doc.get("minimum_meaningful_effect") != h.get("minimum_meaningful_effect"):
            raise ValueError("minimum_meaningful_effect differs from hypothesis")
        if not isinstance(doc["variants"], dict) or not doc["variants"]:
            raise ValueError("variants must be a nonempty mapping")
        for variant, spec in doc["variants"].items():
            _required(spec, "task eval_task overrides eval_overrides")
            for key in ("overrides", "eval_overrides"):
                if not isinstance(spec[key], list) or not all(isinstance(s, str) for s in spec[key]):
                    raise ValueError(f"{variant}.{key} must be an exact ordered argument list")
        if not doc["baseline_variants"] or set(doc["baseline_variants"]) - set(doc["variants"]):
            raise ValueError("baseline_variants must reference variants")
        if (
            not doc["seeds"]
            or any(type(s) is not int or s < 0 for s in doc["seeds"])
            or len(set(doc["seeds"])) != len(doc["seeds"])
        ):
            raise ValueError("seeds must be distinct nonnegative integers")
        _required(doc["training_budget"], "iterations num_envs")
        for key, value in doc["training_budget"].items():
            _positive(value, f"training_budget.{key}")
        ev = doc["evaluation"]
        from .verification import validate_reducers

        validate_reducers(ev)
        _required(
            ev,
            "population seeds scenarios levels difficulties num_envs num_episodes "
            "episode_sampling paired_factors primary_metric secondary_metrics constraints profile",
        )
        for field in ("seeds", "levels", "difficulties", "scenarios", "paired_factors"):
            if not isinstance(ev[field], list):
                raise ValueError(f"evaluation.{field} must be a list")
        if (
            not ev["seeds"]
            or any(type(seed) is not int or seed < 0 for seed in ev["seeds"])
            or len(set(ev["seeds"])) != len(ev["seeds"])
        ):
            raise ValueError("evaluation seeds must be distinct nonnegative integers")
        for field in ("levels", "difficulties"):
            check = (lambda x: type(x) is int and x >= 0) if field == "levels" else (lambda x: isinstance(x, str) and x)
            if any(not check(item) for item in ev[field]) or len(set(ev[field])) != len(ev[field]):
                raise ValueError(f"evaluation {field} must be distinct nonnegative integers/nonempty names")
        if ev["profile"] not in ("general", "forward_course", "world_model"):
            raise ValueError("unsupported evaluation profile")
        _positive(ev["num_envs"], "evaluation.num_envs")
        _positive(ev["num_episodes"], "evaluation.num_episodes")
        sampling = ev["episode_sampling"]
        if sampling not in ("first_episode_per_env", "fixed_k_episodes_per_env"):
            raise ValueError("unknown episode_sampling policy")
        if sampling == "fixed_k_episodes_per_env":
            _positive(ev.get("k"), "evaluation.k")
        _required(doc["smoke_gate"], "enabled auto_full")
        if any(type(doc["smoke_gate"][k]) is not bool for k in ("enabled", "auto_full")):
            raise ValueError("smoke enabled/auto_full must be booleans")
        if doc["smoke_gate"]["auto_full"] and not doc["smoke_gate"]["enabled"]:
            raise ValueError("auto_full requires an enabled smoke gate")
        if doc["smoke_gate"]["enabled"]:
            _required(doc["smoke_gate"], "log_roots")
            if set(doc["smoke_gate"]["log_roots"]) != set(doc["variants"]):
                raise ValueError("smoke log_roots must cover every variant")
            _positive(doc["smoke_gate"].get("iterations"), "smoke_gate.iterations")
        _required(doc["retry_policy"], "max_attempts allowed_reasons selection")
        if not isinstance(doc["retry_policy"]["allowed_reasons"], list) or set(
            doc["retry_policy"]["allowed_reasons"]
        ) - {"vm_died", "infrastructure_interruption", "scheduler_preemption"}:
            raise ValueError("retry reasons must describe infrastructure failure")
        _positive(doc["retry_policy"]["max_attempts"], "retry_policy.max_attempts")
        if doc["retry_policy"]["selection"] != "first_valid_identical_attempt":
            raise ValueError("retry selection must be first_valid_identical_attempt")
    elif kind == "decision":
        _required(
            doc,
            "group decision chosen_variants rejected_variants status based_on_protocol based_on_report researcher_note",
        )
        if doc["decision"] not in ("keep", "reject", "modify", "extend"):
            raise ValueError("unknown researcher decision")
        if doc["status"] not in ("supported", "not_supported", "inconclusive", "invalid_experiment"):
            raise ValueError("unknown research outcome")
        if set(doc["chosen_variants"]) & set(doc["rejected_variants"]):
            raise ValueError("chosen and rejected variants overlap")
        return dict(doc)
    else:
        raise ValueError(f"unknown contract kind: {kind}")
    identity = content_id(kind, doc)
    if doc.get(f"{kind}_id", identity) != identity:
        raise ValueError(f"stale {kind}_id: expected {identity}")
    return {**doc, f"{kind}_id": identity}


def read_contract(path: Path, kind: str, hypothesis: dict | None = None) -> dict:
    return validate_contract(kind, yaml.safe_load(path.read_text()), hypothesis)


def load_protocol(path: Path, hypothesis_path: Path | None = None) -> dict:
    return read_contract(
        path, "protocol", read_contract(hypothesis_path or path.with_name("hypothesis.yaml"), "hypothesis")
    )


def ensure_supported_evaluation(protocol: dict) -> None:
    from .verification import validate_reducers

    if protocol.get("domain") == "world_model":
        raise ValueError("world_model evidence requires world_modeling.campaign, not the RL episode evaluator")
    validate_reducers(protocol["evaluation"])
    if protocol["evaluation"]["episode_sampling"] != "first_episode_per_env":
        raise ValueError("fixed_k_episodes_per_env is declared but not supported by the current evaluator")
    if protocol["evaluation"]["scenarios"]:
        raise ValueError(
            "scenario scheduling requires a task-specific evaluator; generic eval-jobs supports levels/difficulties"
        )


def paired_unit_id(row: dict, context: dict, factors: list[str]) -> str:
    values = {**context, **row}
    if any(f not in values or values[f] is None for f in factors):
        raise ValueError("paired factor missing from evaluation evidence")
    return json.dumps({f: values[f] for f in factors}, sort_keys=True, separators=(",", ":"))


def checkpoint_digest(path: str | Path) -> str | None:
    """Hash checkpoint bytes in bounded memory without importing its tensors."""
    try:
        with Path(path).open("rb") as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            return digest.hexdigest()
    except OSError:
        return None


def runtime_fingerprint(root: Path | None = None, resolved_config=None, asset_paths=()) -> dict:
    """No simulator import or GPU allocation; unavailable metadata is explicit null."""
    root = root or Path.cwd()

    def query(argv):
        try:
            result = subprocess.run(argv, cwd=root, capture_output=True, text=True, timeout=5, check=False)
            return result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    versions = {}
    for package in ("torch", "isaacsim", "isaaclab", "rsl-rl-lib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    torch = sys.modules.get("torch")

    digest = checkpoint_digest
    from .verification import config_hash

    config_error = None
    try:
        resolved_hash = config_hash(resolved_config) if resolved_config is not None else None
    except (ValueError, TypeError, RecursionError) as error:
        resolved_hash, config_error = None, str(error)

    locks = {
        p.name: digest(p)
        for p in (
            root / "environment.yml",
            root / "uv.lock",
            root / "requirements.txt",
            root / "pyproject.toml",
        )
        if p.is_file()
    }

    def source_commit(path):
        """HEAD of an external source checkout, flagged when its working tree is dirty."""
        if path is None or not (path / ".git").exists():
            return None
        commit = query(["git", "-C", str(path), "rev-parse", "HEAD"])
        dirty = query(["git", "-C", str(path), "status", "--porcelain"])
        return f"{commit}-dirty" if commit and dirty else commit

    return {
        "schema_version": 2,
        "config_hash_version": 2,
        "resolved_config_hash_error": config_error,
        "commit": query(["git", "rev-parse", "HEAD"]),
        "launch_tag": os.environ.get("RESEARCH_LAUNCH_TAG"),
        "python": platform.python_version(),
        "packages": versions,
        "isaaclab_commit": source_commit(isaaclab_path(root)),
        "rsl_rl_commit": source_commit(
            Path(os.environ["RSL_RL_REPOSITORY"]) if os.environ.get("RSL_RL_REPOSITORY") else root.parent / "rsl_rl_rwm"
        ),
        "cuda_runtime": getattr(getattr(torch, "version", None), "cuda", None),
        "gpu_driver": query(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
                *(["--id", os.environ["RESEARCH_GPU"]] if "RESEARCH_GPU" in os.environ else []),
            ]
        ),
        "environment_hashes": locks,
        # A sorted list (not a positional/path-keyed mapping) so two runs with the same asset
        # content compare equal regardless of enumeration order or absolute checkout path.
        "asset_hashes": sorted(
            (digest(Path(path)) for path in asset_paths if path), key=lambda value: (value is None, value)
        ),
        "resolved_config_hash": resolved_hash,
    }


def isaaclab_path(root: Path) -> Path | None:
    """The Isaac Lab checkout this repository trains against: ``ISAACLAB_PATH``, else ``../IsaacLab``."""
    for candidate in (os.environ.get("ISAACLAB_PATH"), root.parent / "IsaacLab", root / "external/IsaacLab"):
        if candidate and (Path(candidate) / ".git").exists():
            return Path(candidate)
    return None


def launch_metadata() -> dict:
    """Queue-to-run boundary; manual/legacy launches remain readable."""
    return json.loads(os.environ.get("RESEARCH_METADATA", "{}"))


def runtime_mismatches(fingerprints: list[dict]) -> list[str]:
    """Report recorded material differences; absent fields remain unknown."""
    fields = (
        "python",
        "packages",
        "isaaclab_commit",
        "rsl_rl_commit",
        "cuda_runtime",
        "gpu_driver",
        "environment_hashes",
        "asset_hashes",
    )
    return [
        field
        for field in fields
        if len({json.dumps(fp[field], sort_keys=True) for fp in fingerprints if fp and fp.get(field) is not None}) > 1
    ]


def final_checkpoint_name(iterations: int) -> str:
    """The checkpoint a fresh ``iterations``-iteration run ends with.

    The rsl_rl_rwm runners save the final model as ``model_<last iteration + 1>.pt``, i.e.
    ``model_<iterations>.pt`` (rsl_rl >= 3 names it ``model_<iterations - 1>.pt``).
    """
    return f"model_{iterations}.pt"


def robot_asset_paths(env_cfg) -> list[str]:
    """Reuse resolved spawn references; remote or absent assets remain unknown."""
    spawn = getattr(getattr(getattr(env_cfg, "scene", None), "robot", None), "spawn", None)
    return [str(value) for field in ("asset_path", "usd_path") if (value := getattr(spawn, field, None))]

"""Local, machine-readable artifacts of one policy evaluation (see ``play.py --eval_output_dir``).

Writes three files per evaluation cell:

* ``evaluation_results.parquet``: one row per finished episode (``evaluation_results.json`` when
  pandas/pyarrow are unavailable);
* ``evaluation_summary.json``: aggregate metrics plus the provenance block;
* ``evaluation_run.json``: the provenance block alone, so a driver can read it without the metrics.

Every row carries its own ``seed`` and ``checkpoint`` columns so cells can be concatenated afterwards.
The aggregate metrics come from ``verification.STANDARD_REDUCERS``, the same definitions
``scripts/research_runs.py check-eval`` and ``publish-eval`` apply to the scored rows.
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Any

from .verification import STANDARD_REDUCERS

logger = logging.getLogger(__name__)

__all__ = ["aggregate_metrics", "write_evaluation_artifacts"]

RESULTS_BASENAME = "evaluation_results"
SUMMARY_FILENAME = "evaluation_summary.json"
RUN_FILENAME = "evaluation_run.json"


def _finite(values: list[Any]) -> list[float]:
    return [float(v) for v in values if v is not None and not math.isnan(float(v))]


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-episode rows into summary metrics; each episode contributes equally.

    A metric is reported only when every row carries its column, so a task that does not measure
    something (say terrain friction) never reports a made-up value. Tracking errors aggregate as a
    root-mean-square over per-episode RMSEs.

    Args:
        rows: One record per finished episode.

    Returns:
        ``{"num_episodes": n, <metric>: value, ...}``; only ``num_episodes`` when ``rows`` is empty.
    """
    n = len(rows)
    if n == 0:
        logger.warning("No completed episodes to aggregate; the evaluation run was too short.")
        return {"num_episodes": 0}
    metrics: dict[str, Any] = {"num_episodes": n}
    for name, spec in STANDARD_REDUCERS.items():
        column = spec["column"]
        if any(column not in row for row in rows):
            continue
        values = _finite([row[column] for row in rows])
        if not values:
            metrics[name] = float("nan")
        elif spec["reduction"] == "rms":
            metrics[name] = math.sqrt(sum(v * v for v in values) / len(values))
        else:  # "mean" and "rate" (a mean of 0/1 outcomes)
            metrics[name] = sum(values) / len(values)
    return metrics


def _write_rows(rows: list[dict[str, Any]], output_dir: str) -> str:
    """Write per-episode rows as Parquet, falling back to JSON."""
    parquet_path = os.path.join(output_dir, f"{RESULTS_BASENAME}.parquet")
    try:
        import pandas as pd

        pd.DataFrame(rows).to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception as exc:  # pandas/pyarrow missing, or an unserialisable column
        json_path = os.path.join(output_dir, f"{RESULTS_BASENAME}.json")
        logger.warning("Could not write Parquet (%s: %s); writing %s instead.", type(exc).__name__, exc, json_path)
        with open(json_path, "w") as handle:
            json.dump(rows, handle, indent=2)
        return json_path


def write_evaluation_artifacts(rows: list[dict[str, Any]], output_dir: str, run_info: dict[str, Any]) -> dict[str, str]:
    """Write the per-episode table, the aggregate summary and the run provenance.

    Args:
        rows: Per-episode records.
        output_dir: Directory to write into; created if missing.
        run_info: Provenance for this run (seed, task, checkpoint, git commit, ...), stored verbatim in
            both JSON files.

    Returns:
        Mapping of artifact name to the path written.
    """
    os.makedirs(output_dir, exist_ok=True)
    results_path = _write_rows(rows, output_dir)
    metrics = aggregate_metrics(rows)

    summary_path = os.path.join(output_dir, SUMMARY_FILENAME)
    with open(summary_path, "w") as handle:
        json.dump({"run": run_info, "metrics": metrics}, handle, indent=2)

    run_path = os.path.join(output_dir, RUN_FILENAME)
    with open(run_path, "w") as handle:
        json.dump(run_info, handle, indent=2)

    return {"results": results_path, "summary": summary_path, "run": run_path}

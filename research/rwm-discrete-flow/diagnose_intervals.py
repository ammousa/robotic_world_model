"""Existing-data interval diagnostics; no model construction, inference or simulator.

All historical tests are exploratory. True-velocity and fixed-prediction readouts
are diagnostics, never forecasts with a changed feedback dynamics model.
"""

import argparse
import hashlib
import json
import matplotlib
from pathlib import Path

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

HORIZONS = (1, 10, 50, 100, 250, 500, 1500)
FORK = 281
EXPECTED_MANIFEST = "4272832c98da50d7e78c4dfc3fee9bbc4ca477c2e1bf6326194dbf8ca0d0e494"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def stats(values):
    values = np.asarray(values, dtype=np.float64).ravel()
    assert len(values) and np.isfinite(values).all()
    quantiles = np.quantile(np.abs(values), [0.5, 0.9, 0.95, 0.99])
    return dict(
        n=int(len(values)),
        mean=float(values.mean()),
        rms=float(np.sqrt(np.mean(values**2))),
        mae=float(np.abs(values).mean()),
        p50_abs=float(quantiles[0]),
        p90_abs=float(quantiles[1]),
        p95_abs=float(quantiles[2]),
        p99_abs=float(quantiles[3]),
        max_abs=float(np.abs(values).max()),
    )


def relationship(x, y):
    x, y = np.asarray(x, np.float64).ravel(), np.asarray(y, np.float64).ravel()
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    a, b = x - x.mean(), y - y.mean()
    covariance = float(np.mean(a * b))
    variance = float(np.mean(a * a))
    denominator = np.sqrt(np.mean(a * a) * np.mean(b * b))
    return dict(
        n=int(len(x)),
        covariance=covariance,
        correlation=float(covariance / denominator) if denominator else None,
        slope_y_on_x=float(covariance / variance) if variance else None,
        mean_x=float(x.mean()),
        mean_y=float(y.mean()),
    )


def summary_pair(euler, trapezoid, context):
    rows = []
    for method, error in [("endpoint", euler), ("trapezoid", trapezoid)]:
        rows.append({**context, "quadrature": method, **stats(error)})
    return rows


def oracle_prefix(euler, trapezoid, length, dt, partition):
    outputs = []
    for method, defect in [("endpoint", euler), ("trapezoid", trapezoid)]:
        accumulated = np.cumsum(defect[:, FORK : FORK + 1500], axis=1)
        for horizon in HORIZONS:
            eligible = length >= FORK + horizon
            errors = accumulated[eligible, :horizon]
            per_episode = np.sqrt(np.mean(errors**2, axis=(1, 2)))
            outputs.append(
                dict(
                    partition=partition,
                    quadrature=method,
                    horizon=horizon,
                    seconds=horizon * dt,
                    eligible=int(eligible.sum()),
                    planned=len(length),
                    mean_episode_prefix_rmse_rad=float(per_episode.mean()),
                    p95_episode_prefix_rmse_rad=float(np.quantile(per_episode, 0.95)),
                    end_state_rms_rad=float(np.sqrt(np.mean(errors[:, -1] ** 2))),
                    end_signed_mean_rad=float(errors[:, -1].mean()),
                )
            )
    return outputs


def dataset_diagnostics(root, metadata, output):
    distributions, strata, rollouts, relations, histograms = [], [], [], [], []
    inputs = {}
    dt = metadata["dt"]
    legs = ["LF", "LH", "RF", "RH"]
    joint_leg = np.array([legs.index(name[:2]) for name in metadata["joint_names"]])
    foot_leg = np.array([legs.index(name[:2]) for name in metadata["foot_names"]])
    assert np.array_equal(foot_leg, np.arange(4))
    velocity_bins = [0, 0.5, 1, 2, 4, 8, 16, np.inf]
    acceleration_bins = [0, 5, 25, 50, 100, 250, 500, np.inf]
    histogram_edges = np.r_[0, np.geomspace(1e-8, 2, 120), np.inf]
    for partition, info in metadata["partitions"].items():
        path = root / info["file"]
        assert digest(path) == info["sha256"]
        inputs[partition] = {"path": str(path), "sha256": info["sha256"]}
        with np.load(path) as saved:
            state, length = saved["state"], saved["length"]
            command, contact = saved["command"], saved["contact"]
        valid = np.arange(state.shape[1] - 1)[None] < length[:, None]
        q0, q1 = state[:, :-1, 9:21].astype(np.float64), state[:, 1:, 9:21].astype(np.float64)
        v0, v1 = state[:, :-1, 21:33].astype(np.float64), state[:, 1:, 21:33].astype(np.float64)
        displacement = q1 - q0
        euler = dt * v1 - displacement
        trapezoid = 0.5 * dt * (v0 + v1) - displacement
        assert np.max(np.abs((euler - trapezoid - 0.5 * dt * (v1 - v0))[valid])) < 1e-14
        # An interval-average label is an algebraic observation, not an independently measured velocity.
        average = displacement / dt
        assert np.max(np.abs((q0 + dt * average - q1)[valid])) < 1e-14
        e, t, speed, acceleration = euler[valid], trapezoid[valid], np.abs(v1[valid]), np.abs((v1 - v0)[valid] / dt)
        distributions += summary_pair(e, t, {"partition": partition, "group": "all", "name": "all"})
        distributions += summary_pair(
            e / dt, t / dt, {"partition": partition, "group": "endpoint_vs_interval_velocity_rad_s", "name": "all"}
        )
        for joint, name in enumerate(metadata["joint_names"]):
            distributions += summary_pair(
                e[:, joint], t[:, joint], {"partition": partition, "group": "joint", "name": name}
            )
        for leg in legs:
            indices = [i for i, name in enumerate(metadata["joint_names"]) if name.startswith(leg)]
            distributions += summary_pair(
                e[:, indices], t[:, indices], {"partition": partition, "group": "leg", "name": leg}
            )
        relations.append(
            {
                "partition": partition,
                "x": "half_dt_endpoint_velocity_change_rad",
                "y": "endpoint_defect_rad",
                **relationship(0.5 * dt * (v1 - v0)[valid], e),
            }
        )
        change = np.zeros(command.shape[:2], bool)
        change[:, 1:] = np.max(np.abs(command[:, 1:] - command[:, :-1]), -1) > 1e-7
        recent = np.zeros(valid.shape, bool)
        for lag in range(5):
            recent[:, lag:] |= change[:, : valid.shape[1] - lag]
        axes = np.argmax(np.abs(command[:, :-1]), -1)
        axes[np.max(np.abs(command[:, :-1]), -1) < 0.1] = 3
        groups = [("recent_command_change_100ms", recent[valid]), ("otherwise_command", ~recent[valid])]
        groups += [
            ("command_" + name, axes[valid] == index)
            for index, name in enumerate(["forward_dominant", "lateral_dominant", "yaw_dominant", "near_zero"])
        ]
        for name, mask in groups:
            if mask.any():
                distributions += summary_pair(
                    e[mask], t[mask], {"partition": partition, "group": "command", "name": name}
                )
        feet0, feet1 = contact[:, :-1, 4:8][valid], contact[:, 1:, 4:8][valid]
        c0, c1 = feet0[:, joint_leg] > 0.5, feet1[:, joint_leg] > 0.5
        contact_masks = {
            "persistent_no_label": ~c0 & ~c1,
            "label_onset": ~c0 & c1,
            "label_offset": c0 & ~c1,
            "persistent_label": c0 & c1,
        }
        total_e, total_t = np.sum(e**2), np.sum(t**2)
        for name, mask in contact_masks.items():
            if mask.any():
                distributions += summary_pair(
                    e[mask], t[mask], {"partition": partition, "group": "foot_contact_proxy", "name": name}
                )
                strata.append(
                    dict(
                        partition=partition,
                        group="foot_contact_proxy",
                        name=name,
                        observations=int(mask.sum()),
                        fraction=float(mask.mean()),
                        endpoint_energy_share=float(np.sum(e[mask] ** 2) / total_e),
                        trapezoid_energy_share=float(np.sum(t[mask] ** 2) / total_t),
                    )
                )
        for variable, values, bins in [
            ("abs_endpoint_velocity_rad_s", speed, velocity_bins),
            ("abs_interval_acceleration_proxy_rad_s2", acceleration, acceleration_bins),
        ]:
            for low, high in zip(bins[:-1], bins[1:]):
                mask = (values >= low) & (values < high)
                if mask.any():
                    strata.append(
                        dict(
                            partition=partition,
                            group=variable,
                            name=f"[{low},{high})",
                            observations=int(mask.sum()),
                            fraction=float(mask.mean()),
                            endpoint_rms_rad=float(np.sqrt(np.mean(e[mask] ** 2))),
                            trapezoid_rms_rad=float(np.sqrt(np.mean(t[mask] ** 2))),
                            endpoint_energy_share=float(np.sum(e[mask] ** 2) / total_e),
                            trapezoid_energy_share=float(np.sum(t[mask] ** 2) / total_t),
                        )
                    )
        for method, values in [("endpoint", e), ("trapezoid", t)]:
            counts, _ = np.histogram(np.abs(values), bins=histogram_edges)
            histograms.append({"partition": partition, "quadrature": method, "counts": counts.tolist()})
        rollouts += oracle_prefix(euler, trapezoid, length, dt, partition)
        print(
            "DATASET_DIAGNOSED",
            partition,
            int(valid.sum()),
            "Euler",
            np.sqrt(np.mean(e**2)),
            "trapezoid",
            np.sqrt(np.mean(t**2)),
            flush=True,
        )
        del state, q0, q1, v0, v1, displacement, average, euler, trapezoid, e, t, speed, acceleration
    tables = dict(
        distributions=distributions,
        strata=strata,
        true_velocity_readouts=rollouts,
        endpoint_defect_relationship=relations,
    )
    (output / "label_histograms.json").write_text(
        json.dumps(
            {"edges": [None if np.isinf(x) else float(x) for x in histogram_edges], "rows": histograms}, indent=2
        )
    )
    return tables, inputs


def model_diagnostics(root, dataset, metadata, evidence):
    rows, correlations, inputs = [], [], []
    dt = metadata["dt"]
    for partition in ["test_id", "test_command_shift"]:
        with np.load(dataset / metadata["partitions"][partition]["file"]) as saved:
            q_initial = saved["state"][:, FORK, 9:21].astype(np.float64)
            v_initial = saved["state"][:, FORK, 21:33].astype(np.float64)
            ids = saved["episode_id"]
        for cell in evidence["tables"]["inventory"]:
            variant, seed = cell["variant"], cell["train_seed"]
            directories = list(root.glob(f"*_{variant}-seed{seed}"))
            assert len(directories) == 1, (variant, seed, directories)
            for mode in ["A", "B_shadow"]:
                path = directories[0] / "world_model_eval" / f"{partition}-{mode}" / "predictions.npz"
                provenance = json.loads(path.with_name("provenance.json").read_text())
                assert provenance["checkpoint_sha256"] == cell["checkpoint_sha256"]
                assert provenance["train_run_id"] == cell["run_id"]
                assert provenance["protocol_id"] == cell["protocol_id"]
                assert provenance["variant"] == variant and provenance["training_seed"] == seed
                assert provenance["dataset_sha256"] == EXPECTED_MANIFEST
                assert provenance["partition"] == partition and provenance["mode"] == mode
                assert digest(path) == provenance["prediction_sha256"]
                inputs.append(
                    dict(
                        path=str(path),
                        sha256=provenance["prediction_sha256"],
                        checkpoint_sha256=cell["checkpoint_sha256"],
                        eval_run_id=cell["eval_run_id"],
                    )
                )
                with np.load(path) as saved:
                    predicted, truth = saved["state"], saved["true_state"]
                    valid, initialized = saved["reference_valid"], saved["initialization_valid"]
                    assert np.array_equal(saved["episode_id"], ids)
                qp, qt = predicted[..., 9:21].astype(np.float64), truth[..., 9:21].astype(np.float64)
                vp, vt = predicted[..., 21:33].astype(np.float64), truth[..., 21:33].astype(np.float64)
                error = qp - qt
                if mode == "A":
                    qprev_p = np.concatenate((q_initial[:, None], qp[:, :-1]), axis=1)
                    qprev_t = np.concatenate((q_initial[:, None], qt[:, :-1]), axis=1)
                    vprev_p = np.concatenate((v_initial[:, None], vp[:, :-1]), axis=1)
                    initial_error = np.zeros_like(q_initial)
                    first = 0
                else:
                    # Saved shadow arrays omit the original fork. Anchor at the first fully recorded future state.
                    qprev_p = np.concatenate((qp[:, :1], qp[:, :-1]), axis=1)
                    qprev_t = np.concatenate((qt[:, :1], qt[:, :-1]), axis=1)
                    vprev_p = np.concatenate((vp[:, :1], vp[:, :-1]), axis=1)
                    initial_error = error[:, 0]
                    first = 1
                quadrature = dt * vt - (qt - qprev_t)
                velocity = dt * (vp - vt)
                reconstruction = qp - qprev_p - dt * vp
                increments_t = 0.5 * dt * (vprev_p + vp)
                increments_e = dt * vp
                if first:
                    quadrature[:, 0] = velocity[:, 0] = reconstruction[:, 0] = 0
                    increments_t[:, 0] = increments_e[:, 0] = 0
                g, b, c = (np.cumsum(values, axis=1) for values in [quadrature, velocity, reconstruction])
                predicted_e = (q_initial if not first else qp[:, 0])[:, None] + np.cumsum(increments_e, axis=1)
                predicted_t = (q_initial if not first else qp[:, 0])[:, None] + np.cumsum(increments_t, axis=1)
                finite = np.isfinite(predicted).all(-1) & np.isfinite(truth).all(-1)
                check = valid & finite & initialized[:, None]
                assert np.max(np.abs((g + b + c + initial_error[:, None] - error)[check])) < 1e-8
                for horizon in HORIZONS:
                    eligible = valid[:, :horizon].all(1) & finite[:, :horizon].all(1) & initialized
                    if not eligible.any():
                        continue
                    actual = error[eligible, :horizon]
                    pg, pb, pc = (value[eligible, :horizon] for value in [g, b, c])
                    p0 = initial_error[eligible, None]
                    total = pg + pb + pc + p0
                    assert np.max(np.abs(total - actual)) < 1e-8
                    per_episode = np.sqrt(np.mean(actual**2, axis=(1, 2)))
                    context = dict(
                        partition=partition,
                        mode=mode,
                        variant=variant,
                        train_seed=seed,
                        horizon=horizon,
                        seconds=dt * horizon,
                        eligible=int(eligible.sum()),
                        initialized=int(initialized.sum()),
                        planned=len(initialized),
                        decomposition_start_forecast_step=first + 1,
                    )
                    row = dict(
                        **context,
                        joint_prefix_rmse_rad=float(per_episode.mean()),
                        joint_prefix_rmse_p95_rad=float(np.quantile(per_episode, 0.95)),
                        joint_velocity_prefix_rmse_rad_s=float(np.sqrt(np.mean((vp - vt)[eligible, :horizon] ** 2))),
                        quadrature_component_mse_rad2=float(np.mean(pg**2)),
                        velocity_component_mse_rad2=float(np.mean(pb**2)),
                        quadrature_velocity_cross_rad2=float(np.mean(2 * pg * pb)),
                        reconstruction_component_mse_rad2=float(np.mean(pc**2)),
                        actual_pooled_mse_rad2=float(np.mean(actual**2)),
                        complete_decomposition_mse_rad2=float(np.mean(total**2)),
                        euler_fixed_prediction_readout_rmse_rad=float(
                            np.sqrt(np.mean((predicted_e - qt)[eligible, :horizon] ** 2))
                        ),
                        trapezoid_fixed_prediction_readout_rmse_rad=float(
                            np.sqrt(np.mean((predicted_t - qt)[eligible, :horizon] ** 2))
                        ),
                        euler_fixed_prediction_readout_mean_episode_prefix_rmse_rad=float(
                            np.sqrt(np.mean((predicted_e - qt)[eligible, :horizon] ** 2, axis=(1, 2))).mean()
                        ),
                        trapezoid_fixed_prediction_readout_mean_episode_prefix_rmse_rad=float(
                            np.sqrt(np.mean((predicted_t - qt)[eligible, :horizon] ** 2, axis=(1, 2))).mean()
                        ),
                        local_integration_defect_rms_rad=(
                            float(np.sqrt(np.mean(reconstruction[eligible, first:horizon] ** 2)))
                            if horizon > first
                            else None
                        ),
                    )
                    rows.append(row)
                    if horizon > first:
                        segment = (eligible, slice(first, horizon))
                        compensation = -quadrature[segment] / dt
                        actual_velocity_error = (vp - vt)[segment]
                        correlations.append(
                            {
                                **context,
                                "relationship": "velocity_error_vs_required_endpoint_compensation",
                                **relationship(compensation, actual_velocity_error),
                            }
                        )
                        correlations.append(
                            {
                                **context,
                                "relationship": "velocity_error_vs_local_joint_displacement_error",
                                **relationship(actual_velocity_error, (qp - qprev_p - qt + qprev_t)[segment] / dt),
                            }
                        )
                print("PREDICTIONS_DIAGNOSED", partition, variant, seed, mode, flush=True)
    return dict(model_decomposition=rows, prediction_error_relationships=correlations), inputs


def figures(tables, output):
    labels = pd.DataFrame(tables["distributions"])
    selected = labels[(labels.partition == "train") & (labels.group == "joint")]
    pivot = selected.pivot(index="name", columns="quadrature", values="rms")
    axis = pivot.plot.bar(figsize=(10, 4), ylabel="One-interval joint discrepancy RMS (rad)", rot=45)
    axis.set_title("True recorded velocities; control-rate quadrature, training data")
    axis.figure.tight_layout()
    axis.figure.savefig(output / "joint_quadrature.png", dpi=160)
    plt.close(axis.figure)
    oracle = pd.DataFrame(tables["true_velocity_readouts"])
    fig, ax = plt.subplots(figsize=(7, 4))
    for method, block in oracle[oracle.partition == "train"].groupby("quadrature"):
        ax.loglog(block.seconds, block.mean_episode_prefix_rmse_rad, marker="o", label=method)
    ax.set(
        xlabel="Horizon (s)",
        ylabel="Mean episode prefix joint RMSE (rad)",
        title="True-velocity readout; not autonomous model prediction",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "true_velocity_accumulation.png", dpi=160)
    plt.close(fig)
    model = pd.DataFrame(tables["model_decomposition"])
    selected = model[(model.partition == "test_id") & (model["mode"] == "A") & (model.horizon == 100)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    selected.pivot(index="variant", columns="train_seed", values="joint_prefix_rmse_rad").plot.bar(ax=axes[0], rot=20)
    axes[0].set(ylabel="Mean episode prefix joint RMSE (rad)", title="Existing runs: recorded-action joint accuracy")
    for seed, block in model[(model.partition == "test_id") & (model["mode"] == "A") & (model.variant == "B2")].groupby(
        "train_seed"
    ):
        axes[1].loglog(block.seconds, block.joint_prefix_rmse_rad, marker="o", label=f"B2 seed {seed} actual")
        axes[1].loglog(
            block.seconds,
            block.trapezoid_fixed_prediction_readout_mean_episode_prefix_rmse_rad,
            "--",
            label=f"seed {seed} trapezoid readout only",
        )
    axes[1].set(
        xlabel="Horizon (s)",
        ylabel="Mean episode prefix joint RMSE (rad)",
        title="Fixed predictions; no changed model feedback",
    )
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "model_accuracy_readout.png", dpi=160)
    plt.close(fig)
    selected = model[
        (model.partition == "test_id")
        & (model["mode"] == "A")
        & (model.horizon == 100)
        & model.variant.isin(["B2", "E3_direct_velocity"])
    ]
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(selected))
    width = 0.24
    for offset, key in [
        (-1, "quadrature_component_mse_rad2"),
        (0, "velocity_component_mse_rad2"),
        (1, "quadrature_velocity_cross_rad2"),
    ]:
        ax.bar(x + offset * width, selected[key], width=width, label=key)
    ax.set_xticks(x, [f"{row.variant}\nseed {row.train_seed}" for row in selected.itertuples()])
    ax.set(
        ylabel="Signed contribution to joint MSE (rad²)", title="Large components can cancel; not a causal attribution"
    )
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "decomposition_cancellation.png", dpi=160)
    plt.close(fig)


def distribution_figure(histogram, output):
    fig, axis = plt.subplots(figsize=(7, 4))
    upper_edges = np.asarray(histogram["edges"][1:-1], dtype=np.float64)
    for row in histogram["rows"]:
        if row["partition"] != "train":
            continue
        counts = np.asarray(row["counts"])
        cumulative = counts.cumsum() / counts.sum()
        axis.semilogx(upper_edges, cumulative[:-1], label=row["quadrature"])
    axis.set(
        xlabel="Absolute one-interval joint discrepancy (rad)",
        ylabel="Cumulative fraction of joint intervals",
        title="Recorded true velocities: training-label error distribution",
        xlim=(1e-6, 1),
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "joint_error_distribution.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    assert digest(args.dataset_dir / "metadata.json") == EXPECTED_MANIFEST
    metadata = json.loads((args.dataset_dir / "metadata.json").read_text())
    evidence = json.loads(args.evidence.read_text())
    assert evidence["group"] == "rwm-physical-consistency" and len(evidence["tables"]["inventory"]) == 8
    tables, data_inputs = dataset_diagnostics(args.dataset_dir, metadata, args.output)
    model_tables, prediction_inputs = model_diagnostics(args.runs_root, args.dataset_dir, metadata, evidence)
    tables.update(model_tables)
    for name, rows in tables.items():
        pd.DataFrame(rows).to_parquet(args.output / (name + ".parquet"), index=False)
    record = dict(
        classification="Exploratory existing-evidence diagnosis, no new model result or autonomous forecast",
        code_sha256=digest(__file__),
        dataset_manifest_sha256=EXPECTED_MANIFEST,
        original_evidence_id=evidence["evidence_id"],
        data_inputs=data_inputs,
        prediction_inputs=prediction_inputs,
        units="physical radians and seconds; q values are constant-default offsets, which cancel in displacement",
        endpoint_error_sign="dt*v_next - (q_next-q_current)",
        acceleration_definition=(
            "finite-difference interval endpoint velocity change / dt, not measured instantaneous acceleration"
        ),
        contact_definition="thresholded maximum force over sensor history; coarse label-transition proxies only",
        physics_substep_series_available=False,
        model_error_definition="along existing free-running trajectories; not teacher-forced local transition error",
        decomposition=(
            "q_error = initial_error + cumulative(true_endpoint_quadrature + dt*velocity_error +"
            " model_reconstruction_defect)"
        ),
        shadow_anchor=(
            "first forecast sample because saved shadow arrays omit the original fork; first increment omitted"
        ),
        oracle_average_velocity=(
            "derived algebraically from q displacement; exact reconstruction is a tautological check, not prediction"
            " evidence"
        ),
        fixed_prediction_readout=(
            "same saved velocity sequence with no feedback changes; cannot establish a new quadrature model result"
        ),
        independent_training_seeds=2,
        inferential_statistics="none; joint/time/episode observations are not independent training replications",
        model_constructions=0,
        neural_forward_calls=0,
        physics_steps=0,
        GPU_hours=0,
        tables=tables,
    )
    (args.output / "diagnosis.json").write_text(json.dumps(record, indent=2, allow_nan=False))
    figures(tables, args.output)
    distribution_figure(json.loads((args.output / "label_histograms.json").read_text()), args.output)
    print("EXISTING_INTERVAL_DIAGNOSIS_COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()

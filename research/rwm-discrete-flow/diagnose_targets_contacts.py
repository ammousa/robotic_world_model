"""Training-target conditioning and exploratory contact strata of saved predictions."""

import argparse
import json
import matplotlib
from pathlib import Path

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from diagnose_intervals import EXPECTED_MANIFEST, FORK, digest, relationship


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    assert digest(args.dataset_dir / "metadata.json") == EXPECTED_MANIFEST
    meta = json.loads((args.dataset_dir / "metadata.json").read_text())
    dt = meta["dt"]
    info = meta["partitions"]["train"]
    assert digest(args.dataset_dir / info["file"]) == info["sha256"]
    with np.load(args.dataset_dir / info["file"]) as saved:
        state, length = saved["state"], saved["length"]
    mean, scale = np.array(meta["state_mean"]), np.array(meta["state_std"]) + 1e-8
    indices = np.r_[np.arange(6), np.arange(21, 33)]
    sums = np.zeros((6, len(indices)), np.float64)
    n = 0
    for start in range(0, len(state), 32):
        block = state[start : start + 32]
        valid = np.arange(block.shape[1] - 1)[None] < length[start : start + 32, None]
        current = (block[:, :-1][valid][:, indices].astype(np.float64) - mean[indices]) / scale[indices]
        nxt = (block[:, 1:][valid][:, indices].astype(np.float64) - mean[indices]) / scale[indices]
        residual = nxt - current
        sums += np.stack(
            [
                current.sum(0),
                nxt.sum(0),
                (current**2).sum(0),
                (nxt**2).sum(0),
                (current * nxt).sum(0),
                (residual**2).sum(0),
            ]
        )
        n += len(current)
    names = ["vx", "vy", "vz", "wx", "wy", "wz"] + ["qdot_" + name for name in meta["joint_names"]]
    moments = sums / n
    vx = moments[2] - moments[0] ** 2
    vy = moments[3] - moments[1] ** 2
    covariance = moments[4] - moments[0] * moments[1]
    residual_variance = moments[5] - (moments[1] - moments[0]) ** 2
    targets = [
        dict(
            name=name,
            n=n,
            direct_target_mean=float(moments[1, i]),
            direct_target_std=float(np.sqrt(vy[i])),
            residual_target_mean=float(moments[1, i] - moments[0, i]),
            residual_target_std=float(np.sqrt(residual_variance[i])),
            residual_to_direct_variance_ratio=float(residual_variance[i] / vy[i]),
            endpoint_lag1_correlation=float(covariance[i] / np.sqrt(vx[i] * vy[i])),
        )
        for i, name in enumerate(names)
    ]
    del state
    with np.load(args.dataset_dir / "test_id.npz") as saved:
        q0 = saved["state"][:, FORK, 9:21].astype(np.float64)
        c0 = saved["contact"][:, FORK, 4:8] > 0.5
    inventory = json.loads(args.evidence.read_text())["tables"]["inventory"]
    rows, inputs, plot_audit = [], [], None
    legs = ["LF", "LH", "RF", "RH"]
    joint_leg = np.array([legs.index(name[:2]) for name in meta["joint_names"]])
    for cell in inventory:
        directories = list(args.runs_root.glob(f"*_{cell['variant']}-seed{cell['train_seed']}"))
        assert len(directories) == 1
        path = directories[0] / "world_model_eval/test_id-A/predictions.npz"
        provenance = json.loads(path.with_name("provenance.json").read_text())
        assert digest(path) == provenance["prediction_sha256"]
        assert provenance["checkpoint_sha256"] == cell["checkpoint_sha256"]
        with np.load(path) as saved:
            predicted, truth = saved["state"][:, :100], saved["true_state"][:, :100]
            eligible = saved["reference_valid"][:, :100].all(1) & saved["initialization_valid"]
            eligible &= np.isfinite(predicted).all((1, 2)) & np.isfinite(truth).all((1, 2))
            contact = saved["true_contact"][:, :100, 4:8] > 0.5
        previous_p = np.concatenate((q0[:, None], predicted[:, :-1, 9:21]), 1)
        previous_t = np.concatenate((q0[:, None], truth[:, :-1, 9:21]), 1)
        local_displacement_error = (predicted[..., 9:21].astype(np.float64) - previous_p) - (
            truth[..., 9:21].astype(np.float64) - previous_t
        )
        velocity_error = predicted[..., 21:33].astype(np.float64) - truth[..., 21:33].astype(np.float64)
        required_compensation = (truth[..., 9:21] - previous_t) / dt - truth[..., 21:33]
        contact_previous = np.concatenate((c0[:, None], contact[:, :-1]), 1)
        old, new = contact_previous[..., joint_leg], contact[..., joint_leg]
        speed = np.abs(truth[..., 21:33])
        groups = {
            "persistent_no_label": ~old & ~new,
            "label_onset": ~old & new,
            "label_offset": old & ~new,
            "persistent_label": old & new,
        }
        total = float(np.sum(local_displacement_error[eligible] ** 2))
        for name, mask in groups.items():
            mask &= eligible[:, None, None]
            for low, high in [(-np.inf, np.inf), (0, 1), (1, 2), (2, 4), (4, 8), (8, np.inf)]:
                selected = mask & (speed >= low) & (speed < high)
                if not selected.any():
                    continue
                disp, vel = local_displacement_error[selected], velocity_error[selected]
                rows.append(
                    dict(
                        variant=cell["variant"],
                        train_seed=cell["train_seed"],
                        mode="A",
                        partition="test_id",
                        horizon_steps=100,
                        reference_eligible_episodes=int(eligible.sum()),
                        label=name,
                        speed_bin=f"[{low},{high})",
                        n_joint_intervals=int(selected.sum()),
                        selected_fraction=float(selected.sum() / (eligible.sum() * 100 * 12)),
                        displacement_error_rms_rad=float(np.sqrt(np.mean(disp**2))),
                        displacement_error_energy_share=float(np.sum(disp**2) / total),
                        endpoint_velocity_error_rms_rad_s=float(np.sqrt(np.mean(vel**2))),
                        endpoint_velocity_error_mean_rad_s=float(vel.mean()),
                    )
                )
        if cell["variant"] == "B2" and cell["train_seed"] == 11:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            vx, vy = velocity_error[eligible].ravel(), local_displacement_error[eligible].ravel() / dt
            compensation = required_compensation[eligible].ravel()
            axes[0].hist2d(compensation, vx, bins=100, range=[[-2, 2], [-2, 2]], cmap="magma")
            axes[0].set(
                xlabel="Required endpoint compensation (rad/s)", ylabel="Predicted endpoint velocity error (rad/s)"
            )
            axes[1].hist2d(vx, vy, bins=100, range=[[-2, 2], [-2, 2]], cmap="magma")
            axes[1].set(xlabel="Endpoint velocity error (rad/s)", ylabel="Local displacement error / dt (rad/s)")
            fig.suptitle("B2 seed 11: existing free-running recorded-action paths, first 2 s")
            fig.tight_layout()
            fig.savefig(args.output / "velocity_displacement_relationship.png", dpi=160)
            plt.close(fig)
            plot_audit = dict(
                episodes=int(eligible.sum()),
                points=len(vx),
                points_outside_compensation_panel=int(((np.abs(compensation) > 2) | (np.abs(vx) > 2)).sum()),
                points_outside_displacement_panel=int(((np.abs(vy) > 2) | (np.abs(vx) > 2)).sum()),
                correlation=relationship(compensation, vx),
            )
        inputs.append(dict(path=str(path), sha256=provenance["prediction_sha256"]))
    frame = pd.DataFrame(targets)
    axis = frame.set_index("name")[["direct_target_std", "residual_target_std"]].plot.bar(figsize=(12, 4), rot=60)
    axis.set(
        ylabel="Target std in the unchanged normalized state units",
        title="Training-only target conditioning; no causal performance inference",
    )
    axis.figure.tight_layout()
    axis.figure.savefig(args.output / "velocity_target_conditioning.png", dpi=160)
    plt.close(axis.figure)
    record = dict(
        classification="Exploratory existing-data diagnostic; no model inference",
        code_sha256=digest(__file__),
        dataset_manifest_sha256=EXPECTED_MANIFEST,
        target_partition="train",
        model_partition="historical test_id, descriptive only",
        model_path_caveat=(
            "State errors along free-running paths; contact association is not teacher-forced causal transition"
            " attribution."
        ),
        contact_caveat=(
            "Foot labels are temporally aggregated force-history thresholds, not measured instantaneous contact events."
        ),
        target_conditioning=targets,
        contact_local_errors=rows,
        plot_audit=plot_audit,
        inputs=inputs,
        neural_forward_calls=0,
        model_constructions=0,
        physics_steps=0,
        GPU_hours=0,
    )
    (args.output / "supplement.json").write_text(json.dumps(record, indent=2, allow_nan=False))
    pd.DataFrame(targets).to_parquet(args.output / "target_conditioning.parquet", index=False)
    pd.DataFrame(rows).to_parquet(args.output / "contact_local_errors.parquet", index=False)
    print("TARGET_CONTACT_DIAGNOSIS_COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()

"""Rebuild exploratory diagnosis figures exclusively from immutable W&B tables."""

import matplotlib.pyplot as plt
import numpy as np


def make_figures(api, project, group, entity):
    if group != "rwm-discrete-flow":
        raise ValueError("This diagnosis belongs to rwm-discrete-flow")
    artifact = api.artifact(f"{entity}/{project}/rwm-discrete-flow-diagnostic-tables:v0")
    tables = {
        name: artifact.get(name).get_dataframe()
        for name in ("distributions", "true_velocity_readouts", "model_decomposition", "target_conditioning")
    }
    figures = {}
    labels = tables["distributions"]
    chosen = labels[(labels.partition == "train") & (labels.group == "joint")]
    axis = chosen.pivot(index="name", columns="quadrature", values="rms").plot.bar(figsize=(10, 4), rot=45)
    axis.set(
        ylabel="One-interval discrepancy RMS (rad)", title="Training labels: endpoint versus trapezoidal quadrature"
    )
    figures["joint_quadrature"] = axis.figure
    oracle = tables["true_velocity_readouts"]
    figure, axis = plt.subplots(figsize=(7, 4))
    for method, block in oracle[oracle.partition == "train"].groupby("quadrature"):
        axis.loglog(block.seconds, block.mean_episode_prefix_rmse_rad, marker="o", label=method)
    axis.set(
        xlabel="Horizon (s)",
        ylabel="Mean episode prefix joint RMSE (rad)",
        title="True recorded velocities; readout, not autonomous prediction",
    )
    axis.legend()
    figures["true_velocity_accumulation"] = figure
    models = tables["model_decomposition"]
    selected = models[(models.partition == "test_id") & (models["mode"] == "A") & (models.horizon == 100)]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    selected.pivot(index="variant", columns="train_seed", values="joint_prefix_rmse_rad").plot.bar(ax=axes[0], rot=20)
    axes[0].set(ylabel="Mean episode prefix joint RMSE (rad)", title="Historical joint accuracy; exploratory only")
    for seed, block in models[
        (models.partition == "test_id") & (models["mode"] == "A") & (models.variant == "B2")
    ].groupby("train_seed"):
        axes[1].loglog(block.seconds, block.joint_prefix_rmse_rad, marker="o", label=f"B2 seed {seed}")
        axes[1].loglog(
            block.seconds,
            block.trapezoid_fixed_prediction_readout_mean_episode_prefix_rmse_rad,
            "--",
            label=f"seed {seed}: trapezoid readout",
        )
    axes[1].set(
        xlabel="Horizon (s)",
        ylabel="Mean episode prefix joint RMSE (rad)",
        title="Fixed predictions; model feedback unchanged",
    )
    axes[1].legend(fontsize=7)
    figures["model_accuracy_readout"] = figure
    selected = selected[selected.variant.isin(["B2", "E3_direct_velocity"])]
    figure, axis = plt.subplots(figsize=(9, 4))
    x = np.arange(len(selected))
    for offset, column in [
        (-1, "quadrature_component_mse_rad2"),
        (0, "velocity_component_mse_rad2"),
        (1, "quadrature_velocity_cross_rad2"),
    ]:
        axis.bar(x + offset * 0.24, selected[column], width=0.24, label=column)
    axis.set_xticks(x, [f"{row.variant}\nseed {row.train_seed}" for row in selected.itertuples()])
    axis.set(ylabel="Signed contribution to joint MSE (rad²)", title="Error cancellation; not a causal attribution")
    axis.legend(fontsize=7)
    figures["decomposition_cancellation"] = figure
    targets = tables["target_conditioning"]
    axis = targets.set_index("name")["residual_to_direct_variance_ratio"].plot.bar(figsize=(10, 4))
    axis.set(ylabel="Residual/direct target variance", title="Training velocity targets under common normalization")
    figures["velocity_target_conditioning"] = axis.figure
    for figure in figures.values():
        figure.tight_layout()
    return figures

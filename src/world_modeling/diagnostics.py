"""Descriptive behavior and geometry diagnostics; no state correction or time warping."""

import torch

from rsl_rl.modules.kinematic_dynamics import orientation_from_gravity, projected_gravity, quat_multiply, quat_rotate

from .kinematics import foot_positions


def gait_diagnostics(predicted, truth, dt, names):
    indices = [i for i, name in enumerate(names) if name.endswith("HFE")]
    if len(indices) != 4 or predicted.shape[1] < 100:
        return {}
    x, y = predicted[..., 9:21][..., indices], truth[..., 9:21][..., indices]
    x, y = x - x.mean(1, keepdim=True), y - y.mean(1, keepdim=True)
    frequencies = torch.fft.rfftfreq(x.shape[1], dt, device=x.device)
    band = (frequencies >= 0.5) & (frequencies <= 6)
    # Exact same frequency bins, no phase alignment or time warping.
    fx, fy = torch.fft.rfft(x, dim=1)[:, band], torch.fft.rfft(y, dim=1)[:, band]
    bins = frequencies[band]
    target_index = fy.abs().argmax(1, keepdim=True)
    phase = torch.angle(fx.gather(1, target_index) * fy.gather(1, target_index).conj()).abs().mean((1, 2))
    return {
        "gait_frequency_Hz": bins[fx.abs().argmax(1)].mean(-1),
        "gait_frequency_error_Hz": (bins[fx.abs().argmax(1)] - bins[fy.abs().argmax(1)]).abs().mean(-1),
        "per_leg_phase_error_rad": phase,
        "joint_amplitude_error_rad": ((2 * x.square().mean(1)).sqrt() - (2 * y.square().mean(1)).sqrt()).abs().mean(-1),
    }


def trajectory_diagnostics(prediction, reference, errors, metadata):
    state = prediction["state"]
    device, dt = state.device, metadata["dt"]
    q = errors["orientation"]
    basis = torch.eye(3, device=device).expand(*q.shape[:-1], 3, 3)
    rotation = quat_rotate(q[..., None, :].expand(-1, -1, 3, -1), basis).transpose(-1, -2)
    residual = rotation.transpose(-1, -2) @ rotation - torch.eye(3, device=device)
    result = {
        "rotation_orthogonality_error": residual.square().sum((-1, -2)).sqrt(),
        "rotation_determinant_error": (torch.linalg.det(rotation) - 1).abs(),
    }
    yaw = torch.atan2(
        2 * (q[..., 0] * q[..., 3] + q[..., 1] * q[..., 2]), 1 - 2 * (q[..., 2].square() + q[..., 3].square())
    )
    yaw_q = torch.zeros_like(q)
    yaw_q[..., 0], yaw_q[..., 3] = (yaw / 2).cos(), (yaw / 2).sin()
    tilt_q = quat_multiply(yaw_q, orientation_from_gravity(state[..., 6:9]))
    from rsl_rl.modules.kinematic_dynamics import quat_inverse

    relative = quat_multiply(quat_inverse(tilt_q), errors["true_orientation"])
    result["gravity_readout_orientation_error_rad"] = 2 * torch.atan2(
        relative[..., 1:].norm(dim=-1), relative[..., 0].abs()
    )
    gravity, true_gravity = projected_gravity(q), projected_gravity(errors["true_orientation"])
    for name, component in (("roll", 0), ("pitch", 1)):

        def angle(g):
            if component == 0:
                return torch.atan2(-g[..., 1], -g[..., 2])
            return torch.atan2(g[..., 0], g[..., 1:].norm(dim=-1))

        difference = angle(gravity) - angle(true_gravity)
        result[f"{name}_error_rad"] = torch.atan2(difference.sin(), difference.cos()).abs()
    defaults = state.new_tensor(metadata["default_joint_position"])
    offset = state.new_tensor(metadata["com_offset"])
    feet_local = foot_positions(state[..., 9:21] + defaults, metadata["geometry"])
    relative_feet = feet_local - offset
    feet_world = errors["position"][..., None, :] + quat_rotate(q[..., None, :].expand(-1, -1, 4, -1), relative_feet)
    initial = state.new_tensor(reference["state"][:, 281])
    initial_q = orientation_from_gravity(initial[:, 6:9])
    initial_feet = foot_positions(initial[:, 9:21] + defaults, metadata["geometry"])
    initial_feet = quat_rotate(initial_q[:, None].expand(-1, 4, -1), initial_feet - offset)
    feet_previous = torch.cat((initial_feet[:, None], feet_world[:, :-1]), 1)
    speed = ((feet_world - feet_previous) / dt).norm(dim=-1)
    height = state.new_tensor(reference["com_position"][:, 281, 2])
    result["nominal_plane_foot_origin_penetration_m"] = torch.relu(-feet_world[..., 2] - height[:, None, None]).mean(-1)
    support = prediction["contact"][..., 4:8] >= 0.5
    result["predicted_support_foot_speed_m_s"] = (speed * support).sum(-1) / support.sum(-1).clamp_min(1)
    result["predicted_support_present"] = support.any(-1)
    link = errors["position"] - quat_rotate(q, offset.expand_as(errors["position"]))
    true_link = errors["true_position"] - quat_rotate(errors["true_orientation"], offset.expand_as(errors["position"]))
    result["link_translation_squared_error_m2"] = (link - true_link).square().sum(-1)
    return result

"""Original RWM construction, frozen policy boundary, and matched trajectory rollouts."""

import copy
import hashlib
import json
import numpy as np
import torch
from pathlib import Path
from torch import nn

from rsl_rl.modules.kinematic_dynamics import (
    KinematicSystemDynamics,
    integrate_orientation,
    orientation_from_gravity,
    quat_inverse,
    quat_multiply,
    quat_rotate,
)
from rsl_rl.modules.system_dynamics import SystemDynamicsEnsemble

ARCHITECTURE = {
    "type": "rnn",
    "rnn_type": "gru",
    "rnn_num_layers": 2,
    "rnn_hidden_size": 256,
    "state_mean_shape": [128],
    "state_logstd_shape": [128],
    "extension_shape": [128],
    "contact_shape": [128],
    "termination_shape": [128],
}
VARIANTS = {
    "B0": None,
    "B1": "orientation",
    "B2": "joints",
    "D1": "direct_independent",
    "E3_direct_velocity": "absolute_velocity",
}


def file_digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


class FrozenActor(nn.Module):
    """Load the existing ELU actor without constructing an optimizer or a simulator."""

    def __init__(self, checkpoint, device="cpu"):
        super().__init__()
        data = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]
        weights = {name.removeprefix("actor."): value for name, value in data.items() if name.startswith("actor.")}
        layers = []
        keys = sorted(int(k.split(".")[0]) for k in weights if k.endswith(".weight"))
        for index in keys:
            weight = weights[f"{index}.weight"]
            layers.append(nn.Linear(weight.shape[1], weight.shape[0]))
            if index != keys[-1]:
                layers.append(nn.ELU())
        self.actor = nn.Sequential(*layers)
        self.actor.load_state_dict(weights)
        self.to(device).eval().requires_grad_(False)

    def forward(self, state, command, previous_action):
        return self.actor(torch.cat((state[:, :9], command, state[:, 9:33], previous_action), -1))


def build_model(variant, metadata, seed, device="cpu"):
    if variant not in VARIANTS:
        raise ValueError("unknown dynamics variant")
    torch.manual_seed(seed)
    kwargs = dict(
        state_dim=45,
        action_dim=12,
        extension_dim=0,
        contact_dim=8,
        termination_dim=1,
        device=device,
        ensemble_size=1,
        history_horizon=32,
        architecture_config=copy.deepcopy(ARCHITECTURE),
        freeze_auxiliary=False,
    )
    baseline = SystemDynamicsEnsemble(**kwargs)
    if variant == "B0":
        return baseline
    result = KinematicSystemDynamics(
        **kwargs,
        state_mean=metadata["state_mean"],
        state_std=metadata["state_std"],
        transition=VARIANTS[variant],
        dt=metadata["dt"],
    )
    # Same shared weights, including the 128-unit hidden head layers. Select output
    # rows rather than re-randomizing them when dependent coordinates are removed.
    initial = baseline.state_dict()
    selected = result.state_dict()
    indices = result.independent_indices
    for name, value in initial.items():
        if value.shape != selected[name].shape:
            value = value[..., indices] if value.ndim == 2 and value.shape[0] == 1 else value[indices]
        selected[name] = value
    result.load_state_dict(selected)
    return result


def load_model(checkpoint, device="cpu"):
    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(data["variant"], data["metadata"], data["seed"], device)
    model.load_state_dict(data["system_dynamics_state_dict"])
    return model.eval(), data


def read_partition(dataset, partition):
    root = Path(dataset)
    metadata = json.loads((root / "metadata.json").read_text())
    entry = metadata["partitions"][partition]
    path = root / entry["file"]
    if file_digest(path) != entry["sha256"]:
        raise ValueError("dataset file hash mismatch")
    with np.load(path) as data:
        arrays = {name: data[name] for name in data.files}
    return arrays, metadata


class DatasetReplay:
    """Uniform episode then uniform valid start; adapts departure actions to RWM."""

    def __init__(self, arrays, metadata, seed, device):
        self.arrays, self.metadata, self.device = arrays, metadata, device
        self.rng = np.random.default_rng(seed)
        self.mean = torch.tensor(metadata["state_mean"], device=device)
        self.scale = torch.tensor(metadata["state_std"], device=device) + 1.0e-8

    def mini_batch_generator(self, sequence_length, num_mini_batch, mini_batch_size):
        valid = np.flatnonzero(self.arrays["length"] + 1 >= sequence_length)
        if len(valid) == 0:
            raise ValueError("no complete training windows")
        for _ in range(num_mini_batch):
            episodes = self.rng.choice(valid, mini_batch_size)
            starts = np.array([self.rng.integers(self.arrays["length"][e] + 2 - sequence_length) for e in episodes])
            time = starts[:, None] + np.arange(sequence_length)
            states = torch.as_tensor(self.arrays["state"][episodes[:, None], time], device=self.device)
            # At stored state t, arrival action is departure u[t-1]. Index 0 is
            # unused by the loss. Never shift the applied departure action twice.
            actions = torch.as_tensor(
                self.arrays["action"][episodes[:, None], np.maximum(time - 1, 0)], device=self.device
            )
            contacts = torch.as_tensor(self.arrays["contact"][episodes[:, None], time], device=self.device)
            termination = torch.as_tensor(self.arrays["base_contact"][episodes[:, None], time], device=self.device)
            yield (states - self.mean) / self.scale, actions, None, contacts, termination


@torch.inference_mode()
def predict(model, arrays, metadata, actor=None, steps=1500, fork=281, sample_seed=None):
    device = next(model.parameters()).device
    state = torch.as_tensor(arrays["state"], device=device)
    recorded = torch.as_tensor(arrays["action"], device=device)
    commands = torch.as_tensor(arrays["command"], device=device)
    mean = torch.tensor(metadata["state_mean"], device=device)
    scale = torch.tensor(metadata["state_std"], device=device) + 1.0e-8
    current = (state[:, fork - 31 : fork + 1] - mean) / scale
    actions = recorded[:, fork - 31 : fork + 1].clone()
    previous_action = recorded[:, fork - 1]
    model.reset()
    if sample_seed is not None:
        torch.manual_seed(sample_seed)
    predictions, chosen_actions, contacts, terminations = [], [], [], []
    for step in range(steps):
        physical = current[:, -1] * scale + mean
        applied = (
            recorded[:, fork + step] if actor is None else actor(physical, commands[:, fork + step], previous_action)
        )
        actions[:, -1] = applied
        if sample_seed is None:
            output, _, _, _, contact, termination = model(current, actions)
        else:
            features = model.state_base(current, actions)
            independent, std = model.state_heads[0](features, current)
            noise = torch.randn_like(physical)
            if isinstance(model, KinematicSystemDynamics):
                orientation = model.orientation
                if orientation is None:
                    orientation = orientation_from_gravity(physical[:, 6:9])
                output, model.orientation = model.reconstruct(
                    independent + std * noise[:, model.independent_indices], current[:, -1], orientation
                )
            else:
                output = independent + std * noise
            _, contact, termination = model.auxiliary_heads[0](model.auxiliary_base(current, actions), current)
        predictions.append(output * scale + mean)
        chosen_actions.append(applied)
        contacts.append(torch.sigmoid(contact))
        terminations.append(torch.sigmoid(termination))
        previous_action = applied
        current, actions = output.unsqueeze(1), applied.unsqueeze(1).clone()
    return {
        "state": torch.stack(predictions, 1),
        "action": torch.stack(chosen_actions, 1),
        "contact": torch.stack(contacts, 1),
        "base_contact": torch.stack(terminations, 1),
    }


def pose_from_states(initial_gravity, predicted, dt):
    orientation = orientation_from_gravity(initial_gravity)
    position = torch.zeros_like(initial_gravity)
    positions, orientations = [], []
    for step in range(predicted.shape[1]):
        orientation = integrate_orientation(orientation, predicted[:, step, 3:6], dt)
        position = position + dt * quat_rotate(orientation, predicted[:, step, :3])
        positions.append(position)
        orientations.append(orientation)
    return torch.stack(positions, 1), torch.stack(orientations, 1)


def physical_errors(predicted, arrays, metadata, fork=281):
    """Per-step physical blocks; no post-initial pose fitting or time warping."""
    device = predicted.device
    steps = predicted.shape[1]
    truth = torch.as_tensor(arrays["state"][:, fork + 1 : fork + 1 + steps], device=device)
    initial = torch.as_tensor(arrays["state"][:, fork], device=device)
    position, orientation = pose_from_states(initial[:, 6:9], predicted, metadata["dt"])
    root_q = torch.as_tensor(arrays["quaternion"], device=device)
    root_p = torch.as_tensor(arrays["com_position"], device=device)
    alignment = quat_multiply(orientation_from_gravity(initial[:, 6:9]), quat_inverse(root_q[:, fork]))
    align = alignment[:, None].expand(-1, steps, -1)
    true_orientation = quat_multiply(align, root_q[:, fork + 1 : fork + 1 + steps])
    true_position = quat_rotate(align, root_p[:, fork + 1 : fork + 1 + steps] - root_p[:, fork : fork + 1])
    relative = quat_multiply(quat_inverse(orientation), true_orientation)
    geodesic = 2 * torch.atan2(torch.linalg.vector_norm(relative[..., 1:], dim=-1), relative[..., 0].abs())
    difference = predicted - truth
    blocks = torch.stack(
        (
            (position - true_position).square().sum(-1),
            geodesic.square(),
            difference[..., :3].square().sum(-1),
            difference[..., 3:6].square().sum(-1),
            difference[..., 9:21].square().mean(-1) / 0.5**2,
            difference[..., 21:33].square().mean(-1) / 5.0**2,
        ),
        -1,
    )
    valid = torch.arange(steps, device=device)[None] < torch.as_tensor(arrays["length"] - fork, device=device)[:, None]
    return {
        "blocks": blocks,
        "valid": valid,
        "difference": difference,
        "position": position,
        "orientation": orientation,
        "true_position": true_position,
        "true_orientation": true_orientation,
        "geodesic": geodesic,
        "prediction_finite": (
            torch.isfinite(predicted).all(-1) & torch.isfinite(orientation).all(-1) & torch.isfinite(position).all(-1)
        ),
    }


def score_at_horizon(errors, horizon):
    squared = errors["blocks"][:, :horizon].mean(-1)
    eligible = errors["valid"][:, :horizon].all(-1)
    finite = torch.isfinite(squared).all(-1)
    if "prediction_finite" in errors:
        finite = finite & errors["prediction_finite"][:, :horizon].all(-1)
    per_episode = squared.mean(-1).sqrt()
    capped = torch.nan_to_num(per_episode, nan=10.0, posinf=10.0, neginf=10.0).clamp(max=10)
    capped = torch.where(finite, capped, torch.full_like(capped, 10.0))
    return {"eligible": eligible, "finite": finite, "score": per_episode, "capped_score": capped}

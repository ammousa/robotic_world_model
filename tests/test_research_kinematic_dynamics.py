"""Physical identities, temporal boundaries and gradient ownership; no simulator."""

import io
import numpy as np
import torch

import pytest
from rsl_rl.modules.kinematic_dynamics import (
    integrate_orientation,
    orientation_from_gravity,
    projected_gravity,
    quat_rotate,
)

from world_modeling.core import DatasetReplay, build_model, physical_errors, score_at_horizon


@pytest.fixture
def metadata():
    return {"state_mean": [0.0] * 6 + [0.0, 0.0, -1.0] + [0.0] * 36, "state_std": [1.0] * 45, "dt": 0.02}


def test_body_orientation_update_and_zero_gradient():
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    omega = torch.zeros(1, 3, requires_grad=True)
    result = integrate_orientation(q, omega, 0.02)
    result.sum().backward()
    assert torch.isfinite(omega.grad).all()
    for _ in range(500):
        q = integrate_orientation(q, torch.tensor([[0.0, 0.0, 1.0]]), 0.02)
    assert torch.allclose(q.norm(dim=-1), torch.ones(1), atol=1.0e-6)
    expected = torch.tensor([[np.cos(10.0), np.sin(10.0), 0.0]], dtype=torch.float32)
    assert torch.allclose(quat_rotate(q, torch.tensor([[1.0, 0.0, 0.0]])), expected, atol=2.0e-5)


def test_gravity_initializes_tilt_without_yaw():
    gravity = torch.tensor([[0.3, 0.4, -np.sqrt(0.75)]], dtype=torch.float32)
    q = orientation_from_gravity(gravity)
    assert torch.allclose(projected_gravity(q), gravity, atol=1.0e-6)
    w, x, y, z = q.unbind(-1)
    assert torch.allclose(2 * (w * z + x * y), torch.zeros(1), atol=1.0e-6)


@pytest.mark.parametrize("variant", ["B1", "B2", "D1", "E3_direct_velocity"])
def test_reconstruction_gradients_and_independent_outputs(variant, metadata):
    model = build_model(variant, metadata, 11)
    state = torch.zeros(2, 45)
    independent = torch.zeros(2, len(model.independent_indices), requires_grad=True)
    q = orientation_from_gravity(model.physical(state)[:, 6:9])
    result, q_next = model.reconstruct(independent, state, q)
    physical = model.physical(result)
    assert torch.allclose(physical[:, 6:9], projected_gravity(q_next))
    assert torch.allclose(physical[:, 6:9].norm(dim=-1), torch.ones(2))
    if variant in {"B2", "E3_direct_velocity"}:
        assert torch.allclose(physical[:, 9:21], model.physical(state)[:, 9:21] + 0.02 * physical[:, 21:33])
        assert len(model.independent_indices) == 30
    else:
        assert len(model.independent_indices) == 42
    result.sum().backward()
    assert torch.isfinite(independent.grad).all()
    assert independent.grad.abs().sum() > 0


@pytest.mark.parametrize("variant", ["B2", "D1"])
def test_forecast_backprop_checkpoint_and_row_reset(metadata, variant):
    torch.set_num_threads(1)
    model = build_model(variant, metadata, 11)
    state = torch.zeros(2, 40, 45)
    action = torch.zeros(2, 40, 12)
    model.reset()
    loss = model.compute_state_loss(model.state_heads[0], state, action)[0]
    loss.backward()
    assert torch.isfinite(loss)
    assert model.state_base.memory.rnn.weight_hh_l0.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.auxiliary_base.parameters())
    model.reset()
    with torch.no_grad():
        first = model(state[:, :32], action[:, :32])[0]
        model.reset_partial(torch.tensor([0]))
        assert model.orientation_reset.tolist() == [True, False]
        assert torch.isfinite(model.orientation[1]).all()
        second = model(first[:, None], action[:, :1])[0]
        assert torch.isfinite(second).all()
        model.orientation[1] = float("nan")
        failed = model(first[:, None], action[:, :1])[0]
        assert not torch.isfinite(failed[1]).all()
        assert torch.isfinite(failed[0]).all()
    saved = io.BytesIO()
    torch.save(model.state_dict(), saved)
    saved.seek(0)
    loaded = build_model(variant, metadata, 22)
    loaded.load_state_dict(torch.load(saved, weights_only=True))
    assert loaded.orientation is None
    model.reset()
    assert torch.allclose(model(state[:, :32], action[:, :32])[0], loaded(state[:, :32], action[:, :32])[0])


def test_velocity_acceleration_is_an_exact_reparameterization():
    residual = torch.tensor([[0.02, -0.08, 0.0]])
    dt = 0.02
    direct = torch.tensor([[0.4, 0.5, -0.3]]) + residual
    acceleration = residual / dt
    assert torch.allclose(direct, torch.tensor([[0.4, 0.5, -0.3]]) + dt * acceleration)


def test_departure_actions_are_shifted_once_and_windows_do_not_cross(metadata):
    states = np.zeros((2, 45, 45), dtype=np.float32)
    states[1] += 1000
    actions = np.tile(np.arange(44, dtype=np.float32)[None, :, None], (2, 1, 12))
    states[:, :, 0] += np.arange(45)
    arrays = {
        "state": states,
        "action": actions,
        "length": np.array([44, 44]),
        "contact": np.zeros((2, 45, 8), np.float32),
        "base_contact": np.zeros((2, 45, 1), np.float32),
    }
    buffer = DatasetReplay(arrays, metadata, 11, "cpu")
    batch = next(buffer.mini_batch_generator(40, 1, 16))
    actual_states, arrival = batch[:2]
    # Loss uses state index 31 paired with arrival-action index 32 = departure 31.
    assert torch.allclose(actual_states[:, 31, 0] % 1000, arrival[:, 32, 0])
    assert torch.all(torch.diff(actual_states[:, :, 0], dim=1) == 1)


def test_aligned_pose_scoring_and_invalid_penalty(metadata):
    states = np.zeros((2, 111, 45), np.float32)
    states[..., 8] = -1
    q = np.zeros((2, 111, 4), np.float32)
    q[..., 0] = 1
    arrays = {
        "state": states,
        "quaternion": q,
        "com_position": np.zeros((2, 111, 3), np.float32),
        "length": np.array([110, 110]),
    }
    prediction = torch.from_numpy(states[:, 11:].copy())
    errors = physical_errors(prediction, arrays, metadata, fork=10)
    assert errors["blocks"].max() == 0
    prediction[0, 50, 0] = float("nan")
    scored = score_at_horizon(physical_errors(prediction, arrays, metadata, fork=10), 100)
    assert scored["eligible"].all()
    assert not scored["finite"][0]
    assert scored["capped_score"][0] == 10
    assert scored["capped_score"][1] == 0


def simple_geometry():
    return {
        "root": "base",
        "feet": [f"foot{i}" for i in range(4)],
        "joints": [
            {
                "parent": "base",
                "child": f"foot{i}",
                "position0": [1.0, 0.0, -0.5],
                "position1": [0.0, 0.0, 0.0],
                "rotation0": [1.0, 0.0, 0.0, 0.0],
                "rotation1": [1.0, 0.0, 0.0, 0.0],
                "axis": "Z",
                "index": i,
            }
            for i in range(4)
        ],
    }


def test_forward_kinematics_uses_joint_frames_without_dynamics():
    from world_modeling.kinematics import foot_positions

    angles = torch.zeros(2, 12)
    angles[1, 0] = torch.pi / 2
    result = foot_positions(angles, simple_geometry())
    # A child at the joint origin stays there under rotation. Moving its local
    # frame origin exercises the child-frame correction rather than a link-length heuristic.
    geometry = simple_geometry()
    geometry["joints"][0]["position1"] = [1.0, 0.0, 0.0]
    moved = foot_positions(angles, geometry)
    assert torch.allclose(result[0, 0], torch.tensor([1.0, 0.0, -0.5]))
    assert torch.allclose(moved[1, 0], torch.tensor([1.0, -1.0, -0.5]), atol=1e-6)


def test_invalid_histories_do_not_contribute_to_stability(metadata):
    from world_modeling.evaluation import scored_rows

    metadata.update(
        default_joint_position=[0.0] * 12,
        joint_limits=[[-3.0, 3.0]] * 12,
        joint_names=[f"joint{i}" for i in range(12)],
        com_offset=[0.0, 0.0, 0.0],
        geometry=simple_geometry(),
    )
    state = np.zeros((2, 2001, 45), np.float32)
    state[..., 8] = -1
    quaternion = np.zeros((2, 2001, 4), np.float32)
    quaternion[..., 0] = 1
    reference = {
        "state": state,
        "quaternion": quaternion,
        "com_position": np.zeros((2, 2001, 3), np.float32),
        "command": np.zeros((2, 2001, 3), np.float32),
        "length": np.array([100, 2000]),
        "episode_id": np.arange(2),
        "contact": np.zeros((2, 2001, 8), np.float32),
    }
    prediction = {"state": torch.from_numpy(state[:, 282:1782].copy()), "contact": torch.zeros(2, 1500, 8)}
    prediction["state"][1, 0, 0] = float("nan")
    rows, _ = scored_rows(prediction, reference, metadata, "test_id", "A", "B0", 11)
    first = [row for row in rows if row["episode_id"] == 0]
    assert all(row["unstable"] is None and row["stable_duration_s"] is None for row in first)
    second = [row for row in rows if row["episode_id"] == 1]
    assert all(row["unstable"] and row["failure_aware_score"] == 10 for row in second)


def test_terminal_recorder_copies_targets_and_true_reset_flags(monkeypatch):
    import importlib.util
    import sys
    import types
    from pathlib import Path

    stub = types.ModuleType("isaaclab.managers.recorder_manager")
    stub.RecorderTerm = object
    monkeypatch.setitem(sys.modules, "isaaclab.managers.recorder_manager", stub)
    path = Path(__file__).parents[1] / "source/mbrl/mbrl/mbrl/envs/world_model_recorder.py"
    spec = importlib.util.spec_from_file_location("snapshot_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    targets = torch.ones(2, 12)
    module.system_snapshot = lambda _: {"state": torch.ones(2, 45)}
    recorder = module.WorldModelSnapshotRecorder()
    recorder._env = types.SimpleNamespace(
        action_manager=types.SimpleNamespace(get_term=lambda _: types.SimpleNamespace(processed_actions=targets)),
        reset_terminated=torch.tensor([True, False]),
        reset_time_outs=torch.tensor([False, True]),
    )
    recorder.record_post_step()
    targets.zero_()
    recorder._env.reset_terminated.zero_()
    assert torch.all(recorder.applied_targets == 1)
    assert recorder.terminated.tolist() == [True, False]
    assert recorder.truncated.tolist() == [False, True]


def test_visual_pose_readout_has_no_feedback_and_gravity_nan_is_a_failure(metadata):
    from world_modeling.core import pose_from_states

    state = torch.zeros(1, 10, 45)
    state[..., 8] = -1
    before = state.clone()
    pose_from_states(state[:, 0, 6:9], state, 0.02)
    assert torch.equal(state, before)
    state[0, 5, 6] = float("nan")
    errors = {
        "blocks": torch.zeros(1, 10, 6),
        "valid": torch.ones(1, 10, dtype=torch.bool),
        "prediction_finite": torch.isfinite(state).all(-1),
    }
    score = score_at_horizon(errors, 10)
    assert not score["finite"].item()
    assert score["capped_score"].item() == 10


def test_mechanics_identity_normalizes_only_execution_placement():
    from research_workflow.verification import canonical_config

    first = {"sim": {"device": "cuda:0", "dt": 0.005}, "actuator": {"friction": 0.6}}
    second = {"sim": {"device": "cuda:1", "dt": 0.005}, "actuator": {"friction": 0.6}}
    assert canonical_config(first) == canonical_config(second)
    second["actuator"]["friction"] = 0.7
    assert canonical_config(first) != canonical_config(second)


def test_replay_coverage_requires_bound_failure_outcomes(tmp_path):
    import json

    from world_modeling.core import file_digest
    from world_modeling.results import verify_videos

    cell = tmp_path / "test_id-A"
    cell.mkdir()
    video = tmp_path / "videos"
    video.mkdir()
    metadata = tmp_path / "metadata.json"
    metadata.write_text("{}")
    prediction = cell / "predictions.npz"
    np.savez(
        prediction,
        episode_id=np.array([0]),
        initialization_valid=np.array([False]),
        reference_valid=np.zeros((1, 10), bool),
        prediction_finite=np.zeros((1, 10), bool),
    )
    with pytest.raises(ValueError, match="missing required"):
        verify_videos(tmp_path, metadata, [(0, "A")])
    status = {
        "kind": "illustrative_kinematic_replay_of_scored_arrays",
        "episode_id": 0,
        "frames": 0,
        "end_reason": "invalid_initialization",
        "physics_steps": 0,
        "prediction_sha256": file_digest(prediction),
        "dataset_sha256": file_digest(metadata),
        "video_sha256": None,
    }
    sidecar = video / "A-episode0.json"
    sidecar.write_text(json.dumps(status))
    assert len(verify_videos(tmp_path, metadata, [(0, "A")])) == 1
    status["prediction_sha256"] = "different-candidate"
    sidecar.write_text(json.dumps(status))
    with pytest.raises(ValueError, match="provenance"):
        verify_videos(tmp_path, metadata, [(0, "A")])


def test_world_model_report_profile_does_not_inherit_rl_metrics():
    from types import SimpleNamespace

    from research_workflow.metric_resolution import MetricResolver

    run = SimpleNamespace(
        id="wm",
        name="B0-seed11",
        job_type="train",
        tags=[],
        config={"runner_cfg": {"research": {"protocol_contract": {"domain": "world_model"}}}},
    )
    assert MetricResolver([run]).profile(run) == "world_model"


def test_factorial_heads_share_initial_tensors_and_change_only_velocity_skip(metadata):
    torch.set_num_threads(1)
    models = {variant: build_model(variant, metadata, 33) for variant in ("B0", "B1", "B2", "D1", "E3_direct_velocity")}
    original = models["B0"].state_dict()
    for variant, model in models.items():
        if variant == "B0":
            continue
        for name, tensor in original.items():
            expected = tensor
            actual = model.state_dict()[name]
            if expected.shape != actual.shape:
                indices = model.independent_indices
                expected = (
                    expected[..., indices] if expected.ndim == 2 and expected.shape[0] == 1 else expected[indices]
                )
            assert torch.equal(actual, expected), (variant, name)
    state = torch.randn(2, 32, 45)
    features = torch.randn(2, 256)
    for residual, direct in (("B1", "D1"), ("B2", "E3_direct_velocity")):
        r, d = models[residual], models[direct]
        assert r.independent_indices == d.independent_indices
        r_mean, r_std = r.state_heads[0](features, state)
        d_mean, d_std = d.state_heads[0](features, state)
        skip = torch.zeros_like(d_mean)
        for coordinate, index in enumerate(d.independent_indices):
            if index < 6 or 21 <= index < 33:
                skip[:, coordinate] = state[:, -1, index]
        assert torch.allclose(r_mean - skip, d_mean, atol=1e-7)
        assert torch.equal(r_std, d_std)
    assert sum(p.numel() for p in models["D1"].parameters()) == 1417009
    assert sum(p.numel() for p in models["B2"].parameters()) == 1413889


def test_direct_independent_joint_gradients_and_full_noise_coupling(metadata, monkeypatch):
    torch.set_num_threads(1)
    state = torch.randn(2, 40, 45) * 0.1
    action = torch.randn(2, 40, 12)
    samples = {}
    randn_like = torch.randn_like
    for variant in ("B0", "B1", "B2", "D1", "E3_direct_velocity"):
        model = build_model(variant, metadata, 33)
        draws = []

        def capture(value, **kwargs):
            result = randn_like(value, **kwargs)
            draws.append(result.detach().clone())
            return result

        with monkeypatch.context() as context:
            context.setattr(torch, "randn_like", capture)
            torch.manual_seed(100)
            model.reset()
            loss = model.compute_state_loss(model.state_heads[0], state, action)[0]
        samples[variant] = draws
        assert torch.isfinite(loss)
        assert len(draws) == 16 and all(draw.shape == (2, 45) for draw in draws)
    assert all(torch.equal(a, b) for draws in samples.values() for a, b in zip(draws, samples["B0"]))
    model = build_model("D1", metadata, 33)
    independent = torch.randn(2, 42, requires_grad=True)
    previous = state[:, 0]
    orientation = orientation_from_gravity(model.physical(previous)[:, 6:9])
    result, _ = model.reconstruct(independent, previous, orientation)
    derivative = torch.autograd.grad(result[:, 9:21].sum(), independent)[0]
    for column, index in enumerate(model.independent_indices):
        expected = 1.0 if 9 <= index < 21 else 0.0
        assert torch.allclose(derivative[:, column], torch.full((2,), expected))
    assert torch.equal(result[:, 9:21], independent[:, 6:18])

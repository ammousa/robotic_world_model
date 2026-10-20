"""Physical queue indices remain valid without allocating any CUDA context."""

import numpy as np
import torch
from types import SimpleNamespace

import pytest


def test_preflight_preserves_physical_ordinals_despite_inherited_visibility(tmp_path, monkeypatch):
    from world_modeling import campaign

    protocol = {
        "domain": "world_model",
        "group": "example",
        "compute_pool": [2, 3],
        "seeds": [11, 22],
        "world_model": {
            "algorithm_repository": "algorithm",
            "algorithm_commit": "approved",
            "push_remote": "research",
            "policy": "policy.pt",
            "policy_sha256": "policy-hash",
            "max_job_seconds": 7200,
        },
    }
    workflow = {
        "mandate": {
            "limits": {
                "compute_pool": [2, 3],
                "allowed_training_seed_values": [11, 22],
                "max_GPU_hours_per_device": 11.86097222222222,
            }
        }
    }
    rr = SimpleNamespace(
        launch_problems=lambda *a: [], git_source=lambda: {}, source_env=lambda base: base, check_imports=lambda *a: []
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    monkeypatch.setattr(campaign, "load_protocol", lambda *a: protocol)
    monkeypatch.setattr(campaign, "read_workflow", lambda *a: workflow)
    monkeypatch.setattr(campaign, "queue_module", lambda: rr)
    monkeypatch.setattr(campaign, "launch_approval_problems", lambda *a, **k: [])
    monkeypatch.setattr(
        campaign.subprocess, "check_output", lambda argv, **k: "approved" if "rev-parse" in argv else ""
    )
    monkeypatch.setattr(campaign, "file_digest", lambda *a: "policy-hash")
    _, _, env, budget = campaign.preflight("protocol.yaml")
    assert "CUDA_VISIBLE_DEVICES" not in env
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert budget["hours_per_gpu"] == workflow["mandate"]["limits"]["max_GPU_hours_per_device"]


@pytest.mark.parametrize("entry", ["train", "evaluation"])
@pytest.mark.parametrize("device", ["cpu", "cuda:2", "cuda:3"])
def test_neural_entry_selects_requested_current_device_before_model(entry, device, monkeypatch):
    from world_modeling import evaluation, train

    chosen = []
    monkeypatch.setattr(torch.cuda, "set_device", chosen.append)
    monkeypatch.setattr(torch, "set_num_threads", lambda *a: None)
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda *a: None)

    class ReachedModel(Exception):
        pass

    def model_boundary(*args):
        assert chosen == ([device] if device.startswith("cuda:") else [])
        raise ReachedModel()

    if entry == "evaluation":
        monkeypatch.setattr(evaluation, "load_model", model_boundary)
        arguments = [
            "--dataset",
            "dataset",
            "--checkpoint",
            "model.pt",
            "--policy",
            "policy.pt",
            "--device",
            device,
            "--output",
            "output",
            "--partition",
            "test_id",
            "--mode",
            "A",
        ]
        function = evaluation.main
    else:
        monkeypatch.setattr(train, "build_model", model_boundary)
        monkeypatch.setattr(
            train,
            "launch_metadata",
            lambda: {"variant": "B0", "protocol_contract": {"domain": "world_model", "seeds": [11, 22]}},
        )
        monkeypatch.setattr(train, "read_partition", lambda *a: ({}, {"policy_sha256": "policy-hash"}))
        monkeypatch.setattr(train, "file_digest", lambda *a: "policy-hash")
        arguments = [
            "--task",
            "task",
            "--device",
            device,
            "--logger",
            "wandb",
            "--log_project_name",
            "rwm-explore",
            "--group",
            "example",
            "--job_type",
            "smoke",
            "--run_name",
            "B0-seed11",
            "--world_model_dataset",
            "dataset",
            "--world_model_variant",
            "B0",
            "--world_model_policy",
            "policy.pt",
            "--seed",
            "11",
            "--max_iterations",
            "1000",
            "--num_envs",
            "256",
            f"agent.device={device}",
        ]
        function = train.main
    with pytest.raises(ReachedModel):
        function(arguments)


def test_timing_synchronizes_model_device_instead_of_default_cuda(monkeypatch):
    from world_modeling import evaluation

    device = torch.device("cuda:3")
    synchronized = []
    memory_devices = []
    tensor = torch.tensor
    monkeypatch.setattr(torch, "tensor", lambda *a, **k: tensor(*a, **{**k, "device": "cpu"}))
    monkeypatch.setattr(torch.cuda, "synchronize", synchronized.append)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", memory_devices.append)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda d: memory_devices.append(d) or 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda d: "mock-device")

    class Model:
        def parameters(self):
            return iter([SimpleNamespace(device=device, dtype=torch.float32)])

        def reset(self):
            pass

        def __call__(self, *args):
            pass

    arrays = {"state": np.zeros((1, 282, 45)), "command": np.zeros((1, 282, 3)), "action": np.zeros((1, 282, 12))}
    metadata = {"state_mean": [0.0] * 45, "state_std": [1.0] * 45}
    result = evaluation.timing(Model(), lambda state, *a: state[:, :12], arrays, metadata)
    assert synchronized and set(synchronized) == {device}
    assert memory_devices and set(memory_devices) == {device}
    assert result["peak_allocated_memory_bytes"] == 1

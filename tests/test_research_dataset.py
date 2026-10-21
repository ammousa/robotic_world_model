"""Collection population aliases and immutable CPU-only assembly boundaries."""

import copy
import json
import numpy as np

import pytest

from world_modeling.core import file_digest
from world_modeling.dataset import assemble


@pytest.fixture
def collection(tmp_path):
    identity = [
        "policy_sha256",
        "dt",
        "state_mean",
        "state_std",
        "joint_names",
        "default_joint_position",
        "joint_limits",
        "com_offset",
        "geometry",
        "foot_names",
        "actuator_weight_sha256",
        "body_mass_kg",
        "body_inertia",
        "shape_material_properties",
        "body_com_positions",
    ]
    directory = tmp_path / "collection_parts"
    directory.mkdir()
    parts, records, specs = [], {}, {}
    for partition in ("train", "validation", "test_id", "test_command_shift"):
        n = 1 if partition == "validation" else 2
        spec = {
            "partition": partition,
            "seed": n,
            "command_seed": n + 10,
            "num_envs": n,
            "episode_offset": 0,
        }
        path = directory / f"collect-{partition}-0.npz"
        arrays = {
            key: np.zeros((n, 2, width), dtype=np.float32)
            for key, width in [
                ("state", 45),
                ("action", 12),
                ("command", 3),
                ("quaternion", 4),
                ("com_position", 3),
            ]
        }
        arrays.update(
            length=np.full(n, 2),
            episode_id=np.array([f"{partition}-{i}" for i in range(n)]),
        )
        np.savez_compressed(path, **arrays)
        info = {key: "synthetic-constant" for key in identity}
        info.update(
            sha256=file_digest(path),
            spec=spec,
            resolved_config={
                "seed": n,
                "scene": {"num_envs": n, "terrain": {"num_envs": n, "friction": 0.8}},
                "sim": {"dt": 0.02},
            },
            runtime_fingerprint={
                "commit": "collection-source",
                "rsl_rl_commit": "algorithm-source",
            },
        )
        path.with_suffix(".json").write_text(json.dumps(info))
        parts.append(path)
        records[partition] = copy.deepcopy(info)
        specs[partition] = {**spec, "batches": 1}
    return parts, records, specs


def test_population_alias_exclusion_preserves_arrays_and_original_records(collection, tmp_path):
    parts, records, _ = collection
    result = assemble(parts, tmp_path / "dataset")
    for path in parts:
        partition = records[path.stem.removeprefix("collect-").removesuffix("-0")]["spec"]["partition"]
        assert result["collection_records"][partition] == [records[partition]]
        assert result["partitions"][partition]["episodes"] == records[partition]["spec"]["num_envs"]
        with np.load(tmp_path / "dataset" / (partition + ".npz")) as saved, np.load(path) as raw:
            for key in raw:
                np.testing.assert_array_equal(saved[key], raw[key])
    with pytest.raises(FileExistsError):
        assemble(parts, tmp_path / "dataset")


@pytest.mark.parametrize(
    "change, expected",
    [
        ("friction", "resolved mechanics"),
        ("dt", "resolved mechanics"),
        ("num_envs", "terrain population"),
    ],
)
def test_assembly_still_rejects_mechanics_and_inconsistent_aliases(collection, tmp_path, change, expected):
    parts, records, _ = collection
    changed = copy.deepcopy(records["validation"])
    if change == "dt":
        changed["resolved_config"]["sim"]["dt"] = 0.01
    else:
        changed["resolved_config"]["scene"]["terrain"][change] = 3
    next(p for p in parts if "validation" in p.name).with_suffix(".json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match=expected):
        assemble(parts, tmp_path / "dataset")
    assert not (tmp_path / "dataset" / "metadata.json").exists()


@pytest.mark.parametrize("tamper", [None, "spec", "hash", "source"])
def test_resume_validates_raw_provenance_before_assembly_without_queue(collection, tmp_path, monkeypatch, tamper):
    from world_modeling import campaign, dataset

    parts, records, specs = collection
    reuse = {
        "source_commit": "collection-source",
        "files": {
            p.name: {
                "sha256": file_digest(p),
                "metadata_sha256": file_digest(p.with_suffix(".json")),
            }
            for p in parts
        },
    }
    protocol = {
        "world_model": {
            "collection": specs,
            "algorithm_commit": "algorithm-source",
            "collection_reuse": reuse,
            "dataset": str(tmp_path / "dataset"),
        }
    }
    if tamper:
        info = records["validation"]
        if tamper == "spec":
            info["spec"]["command_seed"] += 1
        elif tamper == "source":
            info["runtime_fingerprint"]["commit"] = "other-source"
        else:
            info["resolved_config"]["sim"]["dt"] = 0.01
        p = next(p for p in parts if "validation" in p.name).with_suffix(".json")
        p.write_text(json.dumps(info))
        if tamper == "source":
            reuse["files"][p.with_suffix(".npz").name]["metadata_sha256"] = file_digest(p)

    class ReachedAssembly(Exception):
        pass

    def assembly_only(*args):
        raise ReachedAssembly()

    monkeypatch.setattr(dataset, "assemble", assembly_only)
    with pytest.raises(ValueError if tamper else ReachedAssembly):
        campaign.assemble_dataset(protocol, None, {}, {"log_root": tmp_path})


def test_followup_reuses_training_validation_exact_bytes_and_rejects_parent_tamper(collection, tmp_path):
    parts, _, _ = collection
    parent_root = tmp_path / "parent"
    parent = assemble(parts, parent_root)
    reuse = {
        "dataset": parent_root,
        "metadata_sha256": file_digest(parent_root / "metadata.json"),
        "artifact": "entity/project/parent:v0",
        "artifact_digest": "pinned-digest",
        "partitions": {name: parent["partitions"][name] for name in ("train", "validation")},
    }
    fresh = [p for p in parts if "test_" in p.name]
    result = assemble(fresh, tmp_path / "followup", reuse=reuse)
    for name in ("train", "validation"):
        assert (parent_root / f"{name}.npz").read_bytes() == (tmp_path / "followup" / f"{name}.npz").read_bytes()
        assert result["collection_records"][name] == parent["collection_records"][name]
        assert result["partitions"][name] == parent["partitions"][name]
    assert result["parent_dataset"]["artifact"] == reuse["artifact"]
    assert "dataset" not in result["parent_dataset"]
    with (parent_root / "train.npz").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="parent partition bytes"):
        assemble(fresh, tmp_path / "rejected", reuse=reuse)
    assert not (tmp_path / "rejected/metadata.json").exists()


def test_followup_requires_same_parent_mechanics(collection, tmp_path):
    parts, _, _ = collection
    parent_root = tmp_path / "parent"
    parent = assemble(parts, parent_root)
    parent["collection_records"]["train"][0]["resolved_config"]["sim"]["dt"] = 0.01
    (parent_root / "metadata.json").write_text(json.dumps(parent))
    reuse = {
        "dataset": parent_root,
        "metadata_sha256": file_digest(parent_root / "metadata.json"),
        "partitions": {name: parent["partitions"][name] for name in ("train", "validation")},
    }
    with pytest.raises(ValueError, match="parent resolved mechanics"):
        assemble([p for p in parts if "test_" in p.name], tmp_path / "rejected", reuse=reuse)


def test_followup_render_uses_protocol_pilot_and_fresh_simulator_seed(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from world_modeling import campaign

    cells = []
    for variant in ("B0", "D1"):
        for seed in (33, 44):
            directory = tmp_path / f"{variant}-seed{seed}"
            cells.append((variant, seed, directory, None, None))
    baseline = cells[0][2] / "world_model_eval/test_id-A"
    baseline.mkdir(parents=True)
    (baseline / "scored_rows.json").write_text(
        json.dumps([{"horizon_steps": 100, "reference_eligible": True, "failure_aware_score": 1, "episode_id": 9}])
    )
    protocol = {
        "group": "test-group",
        "protocol_id": "p-test",
        "compute_pool": [0, 1],
        "variants": {name: {"eval_task": "Play"} for name in ("B0", "D1")},
        "world_model": {
            "pilot_seed": 33,
            "collection": {"test_id": {"seed": 7301}},
            "policy": "policy.pt",
            "dataset": "dataset",
        },
    }
    jobs = []

    def capture(submitted, *args, **kwargs):
        jobs.extend(submitted)
        return 0

    monkeypatch.setattr(campaign, "trained_cells", lambda *args: iter(cells))
    rr = SimpleNamespace(PLAY_SCRIPT="play.py", pin_launch=lambda *args: "tag", run_queue=capture)
    campaign.render(protocol, rr, {}, {"log_root": tmp_path})
    assert len(jobs) == 22
    for job in jobs:
        argv = job["argv"]
        assert argv[argv.index("--seed") + 1] == "7301"
        if "seed44" in job["name"]:
            assert argv[argv.index("--world_model_episode") + 1] == "0"
        else:
            assert "seed33" in job["name"]


def test_full_factorial_score_grid_has_unique_queue_identities(tmp_path):
    from types import SimpleNamespace

    from world_modeling.campaign import score_job

    protocol = {"protocol_id": "p-test", "world_model": {"dataset": "dataset", "policy": "policy.pt"}}
    jobs = []
    for variant in ("B0", "B1", "B2", "D1", "E3_direct_velocity"):
        for seed in (33, 44):
            directory = tmp_path / f"{variant}-seed{seed}"
            for partition in ("test_id", "test_command_shift"):
                for mode in ("A", "B", "B_shadow", "A-final"):
                    checkpoint = directory / ("model_20000.pt" if mode == "A-final" else "model_1000.pt")
                    output = directory / "world_model_eval" / f"{partition}-{mode}"
                    jobs.append(
                        score_job(protocol, SimpleNamespace(PLAY_SCRIPT="play.py"), checkpoint, partition, mode, output)
                    )
    assert len(jobs) == 80
    assert len({job["name"] for job in jobs}) == 80
    assert all(job["research"]["protocol_id"] == "p-test" for job in jobs)


def test_queue_rejects_duplicate_names_before_files_or_dispatch(tmp_path, monkeypatch):
    from test_research_runs import rr

    def forbidden(*args, **kwargs):
        pytest.fail("duplicate jobs must fail before any queue event or subprocess")

    monkeypatch.setattr(rr, "_event", forbidden)
    monkeypatch.setattr(rr.subprocess, "Popen", forbidden)
    destination = tmp_path / "not-created"
    job = {"name": "same-cell", "gpu": None, "argv": ["python", "play.py"]}
    with pytest.raises(ValueError, match="duplicate queue job names"):
        rr.run_queue([job, dict(job)], [0, 1], destination)
    assert not destination.exists()

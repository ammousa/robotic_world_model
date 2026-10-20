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

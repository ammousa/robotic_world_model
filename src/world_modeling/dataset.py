"""Assemble once-collected, immutable episode partitions with content verification."""

import argparse
import copy
import json
import numpy as np
from pathlib import Path

from .core import file_digest


def assemble(parts, output):
    output = Path(output)
    if (output / "metadata.json").exists():
        raise FileExistsError("dataset manifest already exists")
    output.mkdir(parents=True, exist_ok=True)
    grouped, metadata, mechanics = {}, None, None
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
    for path in sorted(map(Path, parts)):
        info = json.loads(path.with_suffix(".json").read_text())
        if file_digest(path) != info["sha256"]:
            raise ValueError("collected partition digest mismatch")
        values = {key: info[key] for key in identity}
        if metadata is not None and metadata != values:
            raise ValueError("collection mechanics/normalization/policy differs across partitions")
        metadata = values
        resolved = copy.deepcopy(info["resolved_config"])
        resolved.pop("seed", None)
        population = resolved["scene"].pop("num_envs")
        if resolved["scene"]["terrain"].pop("num_envs") != population:
            raise ValueError("terrain population differs from declared scene population")
        if mechanics is not None and resolved != mechanics:
            raise ValueError("resolved mechanics/actuators/materials differ across collection batches")
        mechanics = resolved
        grouped.setdefault(info["spec"]["partition"], []).append((path, info))
    if set(grouped) != {"train", "validation", "test_id", "test_command_shift"}:
        raise ValueError("all four declared partitions are required")
    manifest = {**metadata, "schema_version": 1, "partitions": {}, "collection_specs": {}, "collection_records": {}}
    for name, entries in grouped.items():
        chunks = [dict(np.load(path)) for path, _ in entries]
        arrays = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
        if len(set(arrays["episode_id"].tolist())) != len(arrays["episode_id"]):
            raise ValueError("duplicate episode IDs")
        for key in ("state", "action", "command", "quaternion", "com_position"):
            if not np.isfinite(arrays[key]).all():
                raise ValueError(f"nonfinite simulator recording: {key}")
        target = output / f"{name}.npz"
        if target.exists():
            raise FileExistsError(target)
        np.savez_compressed(target, **arrays)
        manifest["partitions"][name] = {
            "file": target.name,
            "sha256": file_digest(target),
            "episodes": len(arrays["length"]),
            "transitions": int(arrays["length"].sum()),
        }
        manifest["collection_specs"][name] = [info["spec"] for _, info in entries]
        manifest["collection_records"][name] = [info for _, info in entries]
    (output / "metadata.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    assemble(arguments.parts, arguments.output)

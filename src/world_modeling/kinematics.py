"""Forward kinematics from the resolved USD joint frames, without a dynamics solver."""

import torch

from rsl_rl.modules.kinematic_dynamics import quat_inverse, quat_multiply, quat_rotate


def extract_geometry(robot):
    """Read fixed geometry once from the live asset, not a second robot description."""
    from pxr import Usd, UsdPhysics

    import isaaclab.sim as sim_utils

    root = sim_utils.find_first_matching_prim(robot.cfg.prim_path)
    joints = []
    for prim in Usd.PrimRange(root):
        if not (prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.FixedJoint)):
            continue
        joint = UsdPhysics.Joint(prim)
        parent, child = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        if len(parent) != 1 or len(child) != 1:
            continue

        def quaternion(value):
            return [float(value.GetReal()), *map(float, value.GetImaginary())]

        name = prim.GetName()
        revolute = prim.IsA(UsdPhysics.RevoluteJoint)
        if revolute and name not in robot.joint_names:
            raise ValueError(f"unrepresented articulated joint: {name}")
        joints.append(
            {
                "name": name,
                "parent": parent[0].name,
                "child": child[0].name,
                "position0": list(map(float, joint.GetLocalPos0Attr().Get())),
                "position1": list(map(float, joint.GetLocalPos1Attr().Get())),
                "rotation0": quaternion(joint.GetLocalRot0Attr().Get()),
                "rotation1": quaternion(joint.GetLocalRot1Attr().Get()),
                "axis": str(UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get()) if revolute else None,
                "index": robot.joint_names.index(name) if revolute else None,
            }
        )
    feet = [name for name in robot.body_names if name.endswith("FOOT")]
    if len(feet) != 4:
        raise ValueError("expected four resolved foot links")
    return {"root": robot.body_names[0], "feet": feet, "joints": sorted(joints, key=lambda value: value["name"])}


def foot_positions(joint_angles, geometry):
    """Absolute joint angles -> four foot-link origins in the root-link frame."""
    zeros = torch.zeros_like(joint_angles[..., :3])
    identity = torch.zeros_like(joint_angles[..., :4])
    identity[..., 0] = 1
    poses = {geometry["root"]: (zeros, identity)}
    pending = list(geometry["joints"])
    while pending:
        progress = False
        for entry in pending[:]:
            if entry["parent"] not in poses:
                continue
            parent_p, parent_q = poses[entry["parent"]]

            def constant(name):
                return joint_angles.new_tensor(entry[name]).expand(*joint_angles.shape[:-1], -1)

            increment = identity.clone()
            if entry["index"] is not None:
                angle = joint_angles[..., entry["index"]] / 2
                increment[..., 0] = angle.cos()
                increment[..., "XYZ".index(entry["axis"]) + 1] = angle.sin()
            q = quat_multiply(quat_multiply(parent_q, constant("rotation0")), increment)
            q = quat_multiply(q, quat_inverse(constant("rotation1")))
            p = parent_p + quat_rotate(parent_q, constant("position0")) - quat_rotate(q, constant("position1"))
            poses[entry["child"]] = (p, q)
            pending.remove(entry)
            progress = True
        if not progress:
            break
    if any(name not in poses for name in geometry["feet"]):
        raise ValueError("resolved USD joint tree cannot reach all four feet")
    return torch.stack([poses[name][0] for name in geometry["feet"]], -2)

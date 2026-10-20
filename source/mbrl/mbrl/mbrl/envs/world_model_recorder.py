"""Noise-free post-step dynamics snapshots taken before automatic episode resets."""

import torch

from isaaclab.managers.recorder_manager import RecorderTerm


def system_snapshot(env):
    robot = env.scene["robot"]
    data = robot.data
    sensor = env.scene.sensors["contact_forces"]
    forces = sensor.data.net_forces_w_history.norm(dim=-1).amax(dim=1)
    thigh = sensor.find_bodies(".*THIGH")[0]
    feet = sensor.find_bodies(".*FOOT")[0]
    base = sensor.find_bodies("base")[0]
    return {
        "state": torch.cat(
            (
                data.root_lin_vel_b,
                data.root_ang_vel_b,
                data.projected_gravity_b,
                data.joint_pos - data.default_joint_pos,
                data.joint_vel - data.default_joint_vel,
                data.applied_torque,
            ),
            -1,
        ).clone(),
        "quaternion": data.root_link_quat_w.clone(),
        "com_position": data.root_com_pos_w.clone(),
        "link_position": data.root_link_pos_w.clone(),
        "contact": torch.cat((forces[:, thigh] > 1.0, forces[:, feet] > 1.0), -1).float(),
        "base_contact": (forces[:, base] > 1.0).float(),
        "foot_position": data.body_link_pos_w[:, [robot.body_names.index(sensor.body_names[i]) for i in feet]].clone(),
    }


class WorldModelSnapshotRecorder(RecorderTerm):
    """Expose one copied terminal-safe snapshot; no manager-owned history or export."""

    last = None

    def record_post_step(self):
        self.last = system_snapshot(self._env)
        self.applied_targets = self._env.action_manager.get_term("joint_pos").processed_actions.clone()
        self.terminated = self._env.reset_terminated.clone()
        self.truncated = self._env.reset_time_outs.clone()
        return None, None

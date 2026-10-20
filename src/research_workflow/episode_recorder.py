"""Per-episode rows for one evaluation cell, recorded while ``play.py --eval_output_dir`` steps the policy.

One row per finished episode, from the environment's own signals. The columns are the ones
``verification.STANDARD_REDUCERS`` aggregates; a column the task cannot provide (no velocity command, say)
is left out rather than faked, and a protocol that declares it as evidence fails loudly at ``check-eval``.

* ``fell``: the episode ended on a termination other than the time limit; ``timed_out``: it ran to the limit;
  ``success``: it ran to the limit without falling.
* ``lin_vel_tracking_error`` / ``ang_vel_tracking_error``: per-episode RMSE of the measured base velocity
  (xy and yaw rate, body frame) against the commanded one. The step on which an episode ends is skipped,
  because the environment has already reset by the time the state is read.
* ``distance_travelled`` / ``path_length``: planar displacement from the start and planar distance covered,
  relative to the environment origin.
"""

from __future__ import annotations

import torch


class EpisodeRecorder:
    """Accumulate per-step state of every environment and emit a row whenever an episode ends."""

    def __init__(self, env, *, seed: int, command_name: str = "base_velocity", asset_name: str = "robot"):
        self.env = env
        self.seed = seed
        self.rows: list[dict] = []
        n, device = env.num_envs, env.device
        self.asset = env.scene[asset_name]
        self.command_name = command_name
        self.has_command = command_name in env.command_manager.active_terms
        self.completed = torch.zeros(n, dtype=torch.long, device=device)
        self.returns = torch.zeros(n, device=device)
        self.steps = torch.zeros(n, device=device)
        self.valid_steps = torch.zeros(n, device=device)
        self.lin_sq = torch.zeros(n, device=device)
        self.ang_sq = torch.zeros(n, device=device)
        self.path = torch.zeros(n, device=device)
        self.start = self._planar_position().clone()
        self.last = self.start.clone()

    def _planar_position(self) -> torch.Tensor:
        return (self.asset.data.root_pos_w - self.env.scene.env_origins)[:, :2]

    def step(self, rewards: torch.Tensor, dones: torch.Tensor, time_outs: torch.Tensor) -> None:
        """Record one environment step; ``dones``/``time_outs`` are the tensors ``env.step`` returned."""
        done = dones.bool()
        alive = ~done
        self.returns += rewards
        self.steps += 1
        position = self._planar_position()
        self.path += torch.where(alive, (position - self.last).norm(dim=-1), torch.zeros_like(self.path))
        if self.has_command:
            command = self.env.command_manager.get_command(self.command_name)
            data = self.asset.data
            lin = (command[:, :2] - data.root_lin_vel_b[:, :2]).square().sum(-1)
            ang = (command[:, 2] - data.root_ang_vel_b[:, 2]).square()
            self.lin_sq += torch.where(alive, lin, torch.zeros_like(lin))
            self.ang_sq += torch.where(alive, ang, torch.zeros_like(ang))
        self.valid_steps += alive.float()
        # ``last`` keeps a finished episode's final pre-reset position until the flush below
        self.last = torch.where(alive.unsqueeze(-1), position, self.last)
        if done.any():
            self._flush(done, time_outs.bool(), dt=self.env.step_dt)
            self.last = torch.where(done.unsqueeze(-1), position, self.last)
            self.start = torch.where(done.unsqueeze(-1), position, self.start)

    def _flush(self, done, time_outs, *, dt) -> None:
        for env_id in done.nonzero(as_tuple=False).flatten().tolist():
            timed_out = bool(time_outs[env_id])
            steps = float(self.steps[env_id])
            valid = max(float(self.valid_steps[env_id]), 1.0)
            row = {
                "env_id": env_id,
                "episode_id": int(self.completed[env_id]),
                "seed": self.seed,
                "episode_return": float(self.returns[env_id]),
                "episode_length": steps,
                "time_seconds": steps * dt,
                "fell": not timed_out,
                "timed_out": timed_out,
                "success": timed_out,
                "distance_travelled": float((self.last[env_id] - self.start[env_id]).norm()),
                "path_length": float(self.path[env_id]),
            }
            if self.has_command:
                row["lin_vel_tracking_error"] = float(self.lin_sq[env_id] / valid) ** 0.5
                row["ang_vel_tracking_error"] = float(self.ang_sq[env_id] / valid) ** 0.5
            self.rows.append(row)
            self.completed[env_id] += 1
        for buffer in (self.returns, self.steps, self.valid_steps, self.lin_sq, self.ang_sq, self.path):
            buffer[done] = 0.0

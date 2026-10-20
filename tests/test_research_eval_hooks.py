"""The training/evaluation hooks that adapt this repository's rsl_rl scripts to the research workflow.

Simulator-free: a small fake environment stands in for Isaac Lab.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import torch
from types import SimpleNamespace

import pytest
from test_research_runs import rr

from research_workflow import final_checkpoint_name
from research_workflow.episode_recorder import EpisodeRecorder
from research_workflow.eval_results import aggregate_metrics, write_evaluation_artifacts
from research_workflow.training_stream import STREAM_FILENAME, attach_metric_stream
from research_workflow.verification import STANDARD_REDUCERS, runner_config


class FakeEnv:
    """Two environments with scripted base state; ``command`` is 1 m/s forward, no yaw."""

    step_dt = 0.5
    device = "cpu"
    num_envs = 2

    def __init__(self, with_command=True):
        self.data = SimpleNamespace(
            root_pos_w=torch.zeros(2, 3),
            root_lin_vel_b=torch.zeros(2, 3),
            root_ang_vel_b=torch.zeros(2, 3),
        )
        self.scene = _Scene(self.data)
        self.command_manager = SimpleNamespace(
            active_terms=["base_velocity"] if with_command else [],
            get_command=lambda name: torch.tensor([[1.0, 0.0, 0.0]] * 2),
        )

    def set(self, x, vx):
        self.data.root_pos_w = torch.tensor([[x[0], 0.0, 0.0], [x[1], 0.0, 0.0]])
        self.data.root_lin_vel_b = torch.tensor([[vx[0], 0.0, 0.0], [vx[1], 0.0, 0.0]])


class _Scene:
    def __init__(self, data):
        self.env_origins = torch.zeros(2, 3)
        self._asset = SimpleNamespace(data=data)

    def __getitem__(self, name):
        return self._asset


def test_recorder_emits_one_row_per_finished_episode():
    env = FakeEnv()
    recorder = EpisodeRecorder(env, seed=7)
    one = torch.ones(2)
    # step 1: both alive, env 0 at 0.5 m/s (error 0.5), env 1 tracks exactly
    env.set(x=(0.5, 0.5), vx=(0.5, 1.0))
    recorder.step(one, torch.tensor([False, False]), torch.tensor([False, False]))
    # step 2: env 1 falls (done, no timeout); its post-reset state is ignored
    env.set(x=(1.0, 0.0), vx=(0.5, 0.0))
    recorder.step(one, torch.tensor([False, True]), torch.tensor([False, False]))
    # step 3: env 0 times out
    env.set(x=(0.0, 0.5), vx=(0.0, 1.0))
    recorder.step(one, torch.tensor([True, False]), torch.tensor([True, False]))
    assert [(r["env_id"], r["episode_id"]) for r in recorder.rows] == [(1, 0), (0, 0)]
    fell, timed_out = recorder.rows
    assert fell["fell"] and not fell["timed_out"] and not fell["success"]
    assert fell["episode_length"] == 2 and fell["time_seconds"] == 1.0 and fell["episode_return"] == 2.0
    assert fell["lin_vel_tracking_error"] == pytest.approx(0.0) and fell["distance_travelled"] == pytest.approx(0.5)
    assert timed_out["timed_out"] and timed_out["success"] and not timed_out["fell"]
    assert timed_out["episode_length"] == 3 and timed_out["seed"] == 7
    # two valid steps with error 0.5 each: RMSE 0.5; displacement and path are the pre-reset 1.0 m
    assert timed_out["lin_vel_tracking_error"] == pytest.approx(0.5)
    assert timed_out["distance_travelled"] == pytest.approx(1.0) and timed_out["path_length"] == pytest.approx(1.0)
    # the reset environment starts its next episode from scratch
    env.set(x=(0.5, 1.0), vx=(1.0, 1.0))
    recorder.step(one, torch.tensor([False, True]), torch.tensor([False, True]))
    assert recorder.rows[-1]["env_id"] == 1 and recorder.rows[-1]["episode_id"] == 1
    assert recorder.rows[-1]["episode_return"] == 2.0


def test_recorder_omits_tracking_columns_without_a_velocity_command():
    env = FakeEnv(with_command=False)
    recorder = EpisodeRecorder(env, seed=0)
    recorder.step(torch.ones(2), torch.tensor([True, False]), torch.tensor([True, False]))
    assert recorder.rows and "lin_vel_tracking_error" not in recorder.rows[0]


def test_aggregate_uses_the_standard_reducers_and_skips_missing_columns():
    rows = [
        {"fell": False, "timed_out": True, "success": True, "episode_return": 4.0, "lin_vel_tracking_error": 3.0},
        {"fell": True, "timed_out": False, "success": False, "episode_return": 2.0, "lin_vel_tracking_error": 4.0},
    ]
    metrics = aggregate_metrics(rows)
    assert metrics["num_episodes"] == 2 and metrics["fall_rate"] == 0.5 and metrics["survival_rate"] == 0.5
    assert metrics["mean_episode_return"] == 3.0
    assert metrics["lin_vel_tracking_rmse"] == pytest.approx(math.sqrt((9 + 16) / 2))
    assert "ang_vel_tracking_rmse" not in metrics and "realized_static_friction_mean" not in metrics
    assert aggregate_metrics([]) == {"num_episodes": 0}
    assert set(metrics) - {"num_episodes"} <= set(STANDARD_REDUCERS)


def test_evaluation_artifacts_roundtrip_through_check_eval(tmp_path):
    rows = [
        {"env_id": i, "episode_id": 0, "seed": 41, "fell": False, "timed_out": True, "success": True,
         "episode_return": 1.0, "lin_vel_tracking_error": 0.1}
        for i in range(3)
    ]  # fmt: skip
    info = {"task": "t", "seed": 41, "num_envs": 3, "num_episodes_requested_per_env": 1,
            "checkpoint": "ckpt.pt", "git": {"commit": "abc", "dirty": False}}  # fmt: skip
    paths = write_evaluation_artifacts(rows, str(tmp_path / "cell"), info)
    assert set(paths) == {"results", "summary", "run"}
    summary = json.loads((tmp_path / "cell" / "evaluation_summary.json").read_text())
    assert summary["run"] == info and summary["metrics"]["fall_rate"] == 0
    cell = rr.check_eval_cell(tmp_path / "cell", seed=41, expected_episodes=3, expected_first_episodes=3)
    assert cell["ok"], cell["problems"]


class FakeRunner:
    def __init__(self, std=0.5):
        self.alg = SimpleNamespace(policy=SimpleNamespace(action_std=torch.full((2, 3), std)))
        self.logged = []

    def log(self, locs, *args, **kwargs):
        self.logged.append(locs["it"])


def test_metric_stream_records_every_iteration_and_stops_on_nonfinite(tmp_path):
    runner = FakeRunner()
    stream = attach_metric_stream(runner, tmp_path)
    for it in range(3):
        runner.log({"it": it, "loss_dict": {"value_function": 0.1}, "rewbuffer": [1.0, 3.0]})
    records = [json.loads(line) for line in stream.read_text().splitlines()]
    assert stream.name == STREAM_FILENAME and runner.logged == [0, 1, 2]
    assert [r["iteration"] for r in records] == [0, 1, 2]
    assert all(r["Checks/finite_rewards_std"] == 1.0 and r["Train/mean_reward"] == 2.0 for r in records)
    with pytest.raises(FloatingPointError):
        runner.log({"it": 3, "loss_dict": {"value_function": float("nan")}, "rewbuffer": []})
    bad = FakeRunner(std=0.0)
    attach_metric_stream(bad, tmp_path / "bad")
    with pytest.raises(FloatingPointError):
        bad.log({"it": 0, "loss_dict": {}, "rewbuffer": []})
    assert bad.logged == []


def test_metric_stream_satisfies_check_train(tmp_path):
    run = tmp_path / "2026-10-20_00-00-00_router-seed1"
    (run / "params").mkdir(parents=True)
    (run / "params" / "agent.yaml").write_text("{}")
    runner = FakeRunner()
    attach_metric_stream(runner, run)
    for it in range(3):
        runner.log({"it": it, "loss_dict": {}, "rewbuffer": []})
    torch.save({"model_state_dict": {"weight": torch.ones(2)}}, run / final_checkpoint_name(3))
    verdict = rr.check_train(run, 3)
    assert verdict["complete"], verdict["problems"]


def test_final_checkpoint_is_the_one_rsl_rl_rwm_saves_last():
    assert final_checkpoint_name(850) == "model_850.pt"


def test_runner_config_accepts_both_rsl_rl_layouts():
    assert runner_config({"runner_cfg": {"research": {"variant": "a"}}}) == {"research": {"variant": "a"}}
    assert runner_config({"train_cfg": {"max_iterations": 3}}) == {"max_iterations": 3}
    assert runner_config({"env_cfg": {}}) == {}


def test_levels_and_difficulties_are_refused_before_launch():
    jobs = rr.eval_jobs(task="t", checkpoint="ckpt.pt", name="a-seed1", seeds=[1], output_root="out", levels=[0])
    problems = rr.unsupported_cell_factor_problems(jobs)
    assert len(problems) == 1 and "--force_terrain_level" in problems[0]
    plain = rr.eval_jobs(task="t", checkpoint="ckpt.pt", name="a-seed1", seeds=[1, 2], output_root="out")
    assert rr.unsupported_cell_factor_problems(plain) == []
    assert all(rr.PLAY_SCRIPT in job["argv"] for job in plain)


def test_default_branch_follows_the_repository(tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q", "-b", "master")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "init")
    assert rr.default_branch(tmp_path) == "master"
    git("branch", "main")
    assert rr.default_branch(tmp_path) == "main"


def test_children_import_mbrl_and_the_workflow_from_this_checkout(tmp_path):
    root = tmp_path / "checkout"
    for package in ("src/research_workflow", "source/mbrl/mbrl"):
        (root / package).mkdir(parents=True)
        (root / package / "__init__.py").write_text("")
    env = rr.source_env(root, base={"PYTHONPATH": ""})
    assert rr.check_imports([sys.executable], env, root) == []
    elsewhere = rr.source_env(tmp_path / "other", base={"PYTHONPATH": ""})
    assert rr.check_imports([sys.executable], elsewhere, root)

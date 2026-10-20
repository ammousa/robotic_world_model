# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path

# The scored neural rollout loads neither Isaac Sim nor a physics world.
if "--world_model_score" in sys.argv[1:]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    from world_modeling.evaluation import main as score_dynamics

    score_dynamics([value for value in sys.argv[1:] if value != "--world_model_score"])
    raise SystemExit(0)

from isaaclab.app import AppLauncher

# the research workflow package (src/research_workflow) is plain Python: import it from this checkout
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--world_model_output", help="Record terminal-safe dynamics transitions, without policy updates.")
parser.add_argument("--world_model_config", help="Exact collection partition/seed/command specification JSON.")
parser.add_argument("--world_model_shadow", help="Replay autonomous departure actions from a saved prediction NPZ.")
parser.add_argument(
    "--world_model_replay", help="Kinematic playback from immutable scored predictions; no dynamics stepping."
)
parser.add_argument("--world_model_metadata", help="Scored dataset metadata for kinematic playback.")
parser.add_argument("--world_model_video", help="Output MP4 for kinematic playback.")
parser.add_argument("--world_model_episode", type=int, default=0)
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--eval_output_dir",
    type=str,
    default=None,
    help=(
        "Scored evaluation: simulate --num_episodes episodes per environment and write per-episode rows, "
        "a summary and the run provenance here (see scripts/research_runs.py eval-jobs)."
    ),
)
parser.add_argument(
    "--num_episodes", type=int, default=1, help="Episodes per environment to simulate with --eval_output_dir."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video or args_cli.world_model_replay:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import json
import os
import time
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import (  # noqa: F401
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)
import mbrl.tasks  # noqa: F401


def evaluate(env, env_cfg, agent_cfg, policy, obs, checkpoint, checkpoint_sha256, task_name):
    """Simulate ``--num_episodes`` episodes per environment and write the scored evaluation artifacts."""
    from research_workflow import launch_metadata, robot_asset_paths, runtime_fingerprint
    from research_workflow.episode_recorder import EpisodeRecorder
    from research_workflow.eval_results import aggregate_metrics, write_evaluation_artifacts
    from research_workflow.run_config import git_state

    base = env.unwrapped
    recorder = EpisodeRecorder(base, seed=agent_cfg.seed)
    num_steps = int(args_cli.num_episodes * base.max_episode_length)
    print(f"[INFO]: Simulating {args_cli.num_episodes} episode(s) per environment ({num_steps} steps).")
    for _ in range(num_steps):
        with torch.inference_mode():
            obs, rewards, dones, infos = env.step(policy(obs))
        recorder.step(rewards, dones, infos.get("time_outs", torch.zeros_like(dones)))
    run_info = {
        "task": task_name,
        "seed": agent_cfg.seed,
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "num_envs": env_cfg.scene.num_envs,
        "num_episodes_requested_per_env": args_cli.num_episodes,
        "steps_simulated": num_steps,
        "control_dt": base.step_dt,
        "physics_dt": env_cfg.sim.dt,
        "decimation": env_cfg.decimation,
        "episode_length_s": env_cfg.episode_length_s,
        "forced_terrain_level": None,
        "requested_difficulty": None,
        "git": git_state(__file__),
        "research": launch_metadata(),
        "runtime_fingerprint": runtime_fingerprint(
            resolved_config={"agent": agent_cfg.to_dict(), "env": env_cfg.to_dict()},
            asset_paths=robot_asset_paths(env_cfg),
        ),
    }
    paths = write_evaluation_artifacts(recorder.rows, args_cli.eval_output_dir, run_info)
    print("-------------- Per-episode aggregate metrics --------------")
    print(f"seed: {agent_cfg.seed}")
    print(json.dumps(aggregate_metrics(recorder.rows), indent=2))
    for name, path in paths.items():
        print(f"[INFO] Wrote {name}: {path}")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    evaluating = args_cli.eval_output_dir is not None

    if args_cli.world_model_output:
        if not args_cli.world_model_config or evaluating or args_cli.video:
            raise ValueError("world-model collection needs a config and a separate, unscored Play invocation")
        from world_modeling.collection import configure

        configure(env_cfg)

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video or args_cli.world_model_replay else None
    )

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": (
                os.path.join(args_cli.eval_output_dir, "videos")
                if evaluating
                else os.path.join(log_dir, "videos", "play")
            ),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            # --note names the scored video of an evaluation cell
            **({"name_prefix": args_cli.note} if args_cli.note else {}),
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if args_cli.world_model_replay:
        from world_modeling.visualization import replay

        replay(
            env,
            args_cli.world_model_replay,
            args_cli.world_model_metadata,
            args_cli.world_model_episode,
            args_cli.world_model_video,
        )
        env.close()
        return

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    from research_workflow import checkpoint_digest

    evaluated_checkpoint_sha256 = checkpoint_digest(resume_path)
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)
    if checkpoint_digest(resume_path) != evaluated_checkpoint_sha256:
        raise RuntimeError("checkpoint changed while loading the evaluation policy")

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    if args_cli.world_model_output:
        from world_modeling.collection import collect

        collect(
            env,
            policy,
            resume_path,
            evaluated_checkpoint_sha256,
            args_cli.world_model_config,
            args_cli.world_model_output,
            shadow=args_cli.world_model_shadow,
        )
        env.close()
        return
    if evaluating:
        evaluate(env, env_cfg, agent_cfg, policy, obs, resume_path, evaluated_checkpoint_sha256, task_name)
        env.close()
        return

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

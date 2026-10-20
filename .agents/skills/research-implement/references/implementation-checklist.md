# Implementation checklist

Paths are relative to the repository root unless they start with `../`. Orientation is in
[`README.md`](../../../../README.md) ("Code Structure"); the workflow rules are in
[`AGENTS.md`](../../../../AGENTS.md).

## Where code lives

| Part | Location |
|---|---|
| Tasks, environment configs, dynamics-model setup, agent configs | `source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/anymal_d/` (`__init__.py` registers task IDs, `flat_env_cfg.py`, `agents/rsl_rl_ppo_cfg.py`, `envs/`) |
| Reusable MDP terms and environment base classes | `source/mbrl/mbrl/envs/` (`mdp/observations.py`, `mdp/events.py`, `mdp/commands/`, `manager_based_mbrl_env.py`) and `tasks/.../velocity/mdp/` |
| Config schemas (`RslRlSystemDynamicsCfg`, `RslRlMbrlPpoAlgorithmCfg`, `RslRlMbrlImaginationCfg`) | `source/mbrl/mbrl/rl/rsl_rl/rl_cfg.py` |
| PPO and model-based PPO algorithm, dynamics model, runners, storage | **the sibling repository `../rsl_rl_rwm`** (`rsl_rl/algorithms/mbpo_ppo.py`, `rsl_rl/modules/system_dynamics.py` and `architectures/`, `rsl_rl/runners/mbpo_on_policy_runner.py`, `rsl_rl/storage/`) |
| Entry points | `scripts/reinforcement_learning/rsl_rl/{train,play,visualize}.py`; simulator-free offline policy training in `scripts/reinforcement_learning/model_based/` |
| Research workflow tooling | `scripts/research_runs.py`, `src/research_workflow/` |

`../rsl_rl_rwm` is a **separate Git repository** that this checkout imports. A change there is
invisible to `git diff` here. Treat it as part of the implementation: commit it on a branch of its
own, name that commit in the Gate B summary, and review it with the same care. The runtime
fingerprint records its HEAD (`rsl_rl_commit`, suffixed `-dirty` when uncommitted); a `-dirty`
fingerprint cannot be reproduced.

## Prefer configuration before source

Most variants are overrides of an existing task, for example `agent.system_dynamics.ensemble_size=5`,
`agent.algorithm.<field>=…`, `agent.imagination.<field>=…` or `env.<field>=…`. Keep the exact
overrides in the protocol and verify the resolved settings (`params/agent.yaml`, `params/env.yaml`
of a smoke run, or the config dump). Do not add a task ID, a config class or a module for a one-off
sweep. Use the layer map only for behavior that existing configuration cannot express.

## Layer map

| Change | Extend | Nearest examples |
|---|---|---|
| Dynamics model: architecture, ensemble, history or forecast horizon | `RslRlSystemDynamicsCfg` values in the agent cfg; model code in `../rsl_rl_rwm` `modules/system_dynamics.py` / `modules/architectures/` | `ensemble_size`, `architecture_config` (GRU), `history_horizon` |
| Model inputs and outputs | the observation groups of `ObservationsCfg_PRETRAIN` in `flat_env_cfg.py` (`SystemStateCfg`, `SystemActionCfg`, `SystemExtensionCfg`, `SystemContactCfg`, `SystemTerminationCfg`) | state/extension/contact/termination heads |
| Model-based update: loss terms, imagination use, uncertainty penalty | `MBPOPPO` in `../rsl_rl_rwm` `algorithms/mbpo_ppo.py`; its schema in `RslRlMbrlPpoAlgorithmCfg` | `uncertainty_penalty_weight`, `system_dynamics_mini_batch_size` |
| Imagination loop, logging, checkpoints | `MBPOOnPolicyRunner` in `../rsl_rl_rwm` `runners/mbpo_on_policy_runner.py` | imagination rollouts, `System Dynamics/*` metrics |
| Policy observations and rewards built from model outputs | `envs/anymal_d_manager_based_mbrl_env.py` on top of `mbrl/envs/manager_based_mbrl_env.py` | `get_imagination_observation`, `imagination_step` |
| New observed signal, reward, event or command | a term in `source/mbrl/mbrl/envs/mdp/` or `tasks/.../velocity/mdp/`, wired in `flat_env_cfg.py` | `mdp/observations.py` |
| Physics, reward weights, curriculum | the env cfg classes in `flat_env_cfg.py` | `AnymalDFlatEnvCfg_INIT` / `_PRETRAIN` / `_FINETUNE` |
| New task ID | `gym.register` in `config/anymal_d/__init__.py`, reusing an existing env cfg and agent cfg | `…-Pretrain-v0`, `…-Finetune-v0` |
| Simulator-free offline training | `scripts/reinforcement_learning/model_based/` (`envs/`, `configs/`) | not launchable through `research_runs.py` |

A **new model behavior** extends the smallest layer that owns it. A new dynamics architecture is a
module plus a config value, not a new runner. A new loss is a `MBPOPPO` hook, not a copied
`update()`.

## Checklist

- [ ] **Ownership.** The dynamics model trains on its own prediction losses; the policy and value
      function train on PPO. Say which losses may reach each network and enforce it (`.detach()`,
      separate optimizers). Imagined transitions never feed the dynamics model's targets.
- [ ] **Observations.** Term order, history orientation and normalization are part of checkpoint
      compatibility; keep them unless the change is the experiment. Privileged or critic-only
      signals must never reach the actor or exports. The model's state/extension/contact/termination
      heads must stay consistent with the environment's observation groups.
- [ ] **Resets.** Nothing crosses an episode boundary: history windows, the dynamics model's hidden
      state (`system_dynamics.reset`), imagination states and auxiliary targets. Timeouts and
      terminations are handled as the existing storage does.
- [ ] **Imagination validity.** Rollout length, the valid-environment mask and the uncertainty penalty
      are explicit scientific settings, never silent defaults.
- [ ] **Configs.** Add the schema field before setting it in an agent cfg. Every config instance owns
      independent nested objects (the existing configs are class-level instances; do not mutate
      them in `__post_init__`). Overrides reach them as `agent.*` / `env.*`.
- [ ] **Checkpoints.** The run keeps saving `model_<iteration>.pt` plus the dynamics checkpoint; a
      resume (`--resume`, `--load_run`, `--checkpoint`) and `--system_dynamics_load_path` still work.
      Existing checkpoints of other variants still load, or the break is documented.
- [ ] **Evaluation compatibility.** Scored evaluation uses `play.py` with the stock
      `Isaac-Velocity-Flat-Anymal-D-Play-v0` task and its `OnPolicyRunner`, loading the policy from
      the checkpoint. A variant that changes the **policy network** needs its own Play registration
      (same agent cfg, evaluation env cfg without corruption or randomization) in the same change.
      Never evaluate with a training task.
- [ ] **Memory.** A larger ensemble or history can exceed GPU memory (the 5-member ensemble needed
      `agent.algorithm.system_dynamics_mini_batch_size=1500` on a 22 GB L4). Such differences are part
      of the variant and must be stated in the protocol.
- [ ] **Recorded signals.** A metric the protocol will score as evidence must be a column of the
      evaluation rows (`src/research_workflow/episode_recorder.py`) or a W&B scalar; declaring a
      metric nothing records fails at `check-eval`.
- [ ] **Docs.** If the change is user-facing, update `README.md`.

## Tests

The upstream repository keeps no test suite and ignores `tests/`; the workflow's own tests are the
exception (`tests/test_research_*.py`, run with `python -m pytest tests/test_research_*.py -q`;
plain Python, no simulator). Add a behavior test beside them for new simulator-free logic
(configuration resolution, row or metric computation, checkpoint round trips on CPU). Assertions that
only import a module, or that are weakened to pass, do not count. Run `pre-commit run --files <changed
files>` before handing the diff over.

Simulator checks run only when the change needs live evidence: the existing `exp/` runner with
`SMOKE=1` (a couple of iterations per stage) or one short `train.py --max_iterations 2 --headless`.
Judge them by their written output, not the exit code: Isaac can exit 0 after an exception. Never
start a full training to "check" a change.

## Anti-patterns (reject in your own diff)

- A copied or overridden `update()`, or a new runner for one method.
- A new environment or task for a model-only change.
- A loss formula inside the dynamics model, or networks constructed inside the loss.
- Experiment-specific switches inside MDP terms, `play.py` or the evaluation recorder.
- Privileged inputs reaching the actor or exports.
- Edits to `../rsl_rl_rwm` left uncommitted or unmentioned in the review.
- Reusable code under `logs/`, `outputs/` or `exp/results*`.
- Tests that only import, or that weaken existing assertions to pass.

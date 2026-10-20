---
name: research-implement
description: >-
  Implement one research idea in this repository — a new policy/network, encoder, auxiliary loss,
  observation, system-dynamics (world model) change, MBPO/PPO algorithm change or UQ method — in the
  correct architectural layer, test its behavior, explain it, and stop for researcher review. Use when the researcher asks
  to "try", "add", "implement" or "prototype" a method in `source/mbrl/mbrl` or the sibling
  `../rsl_rl_rwm` checkout, for example "make the system-dynamics ensemble 5 members for uncertainty
  estimation" or "add an auxiliary prediction loss to the MBPO update". Do not
  use for launching, monitoring or evaluating experiments, comparing runs, or cleaning up after a
  decision; those are later workflow stages.
license: MIT
metadata:
  version: 0.1.0
---

# Research Implement

**Decision authority:** read [research modes](../../../docs/research_modes.md). Attended is the
default (user decisions); explicitly authorized unattended mode routes Gates A–D to a separate
`researcher` via `research-decide`. Stage approval stops remain real gates; the orchestrator
obtains the active researcher's decision and continues approved work. Never self-approve.
Record actual gate receipts and refresh the concise W&B Researcher dashboard after each gate.

This skill covers the **implement** stage of `frame → implement → code-review → experiment → evaluate → review → researcher-decision → finalize` from [`AGENTS.md`](../../../AGENTS.md). You make the approved idea executable using existing configuration first, or the smallest correct
source change when necessary, verify its behavior, and then stop. The researcher reviews before anything is trained.

**Hard limits.** Never start training or evaluation runs, create W&B runs, schedule GPUs, compare
candidates or pick a winner. Do not add experiment-management code.

**Branch.** Work on the comparison's own branch. Agree a group slug with the researcher; it names
both the branch and the W&B group. From the main checkout, run
`python scripts/research_runs.py worktree --group <group> [--base <branch>]`; `--base` defaults to
the repository default branch (`master`), or a previous agent branch for a follow-up. Implement in the printed worktree. Commit there
when the implementation is ready for review: commits on `dev/agent/*` are pre-authorized, while
the default branch (`master`) and other branches are not, and there is no rebasing or force-pushing. The researcher reviews
with `git diff <base>...dev/agent/<group>`.

## 1. Consume the scientific contract

Normally consume the Gate A-approved `research/<group>/hypothesis.yaml` and its content ID.
For a legacy comparison, state `legacy workflow` and reconstruct scientific intent from its record.
Realize the approved mechanism with the fewest necessary changes. A configuration-only experiment
is a complete implementation stage; zero model-source changes is a valid result.

**Scientific choices** (frozen versus joint model training, imagination horizon, uncertainty use, privileged inputs,
gradient ownership) belong in `research-frame`. If unresolved or implementation broadens the claim,
consolidate the questions and return to framing for approval. **Engineering choices** (module
filename, helper structure, fixtures) are the agent's responsibility; resolve them without a gate.

## 2. Inspect before editing

First check whether existing tasks, presets and supported Hydra/CLI overrides already express the
approved change. Prefer, in order:

1. Existing task/config with exact overrides recorded in the protocol.
2. A small reusable configuration extension when an override cannot express the mechanism.
3. The smallest source change at the existing architectural extension point.

Do not add a preset class, task ID, schema field, wrapper, runner or helper just to name a one-off
variant. A larger dynamics ensemble or a different hidden size normally uses the existing tasks and
Hydra overrides (for example `agent.system_dynamics.ensemble_size=5`, `agent.algorithm.*`).
Add a named preset only when reuse, required validation or an explicitly chosen canonical interface
justifies it; explain why overrides are insufficient. Do not generalize for hypothetical future use.


Find the nearest existing implementation. Use the layer map in
[references/implementation-checklist.md](references/implementation-checklist.md) and identify:

- the policy/module or system-dynamics extension point;
- the algorithm/loss extension point;
- the rl cfg schema and the agent preset;
- the task registration (training task and the `eval_task` it is scored with);
- observation/environment requirements;
- the tests to copy;
- checkpoint and export implications.

Where the code lives: tasks, rewards, observations, env cfgs and rl cfgs are in `source/mbrl/mbrl/`
(`tasks/manager_based/locomotion/velocity/config/anymal_d/{flat_env_cfg.py,agents/rsl_rl_ppo_cfg.py,envs/}`,
`mbrl/envs/`, `rl/rsl_rl/rl_cfg.py`). The PPO/MBPO algorithms, the system-dynamics model and the
runners live in the sibling repository `../rsl_rl_rwm` (`rsl_rl/algorithms/mbpo_ppo.py`,
`rsl_rl/modules/system_dynamics.py`, `rsl_rl/runners/mbpo_on_policy_runner.py`). Read the
architecture docs listed in `AGENTS.md` for the layers you will touch.

A **model-only** experiment extends the smallest layer (rl cfg, system-dynamics cfg, an env cfg
term, or the `rsl_rl_rwm` algorithm) and prefers Hydra/CLI overrides. Changes in `../rsl_rl_rwm` are
**outside this git repository**: commit them there, record that sibling commit (the runtime
fingerprint already stores `rsl_rl_commit`, suffixed `-dirty` if uncommitted), and review that diff
too at Gate B.

Choose the smallest existing extension point. Never:

- copy a PPO update loop or override `update()`;
- add a runner for a method;
- duplicate an environment for a model-only change;
- put a loss formula in a policy, or have an algorithm construct or own networks;
- put experiment-specific logic in MDP terms;
- put reusable code under `logs/`, `reports/` or `outputs/`.

## 3. Plan

Before substantial edits, post a short plan:

```text
Hypothesis:
Existing implementation to extend:
Files expected to change:
Files deliberately not changed:
Compatibility risks:
Tests to add/run:
```

Then proceed, unless the researcher asked for plan-only mode. If implementation reveals the plan
was wrong in a way that changes the method, stop and say so.

## 4. Implement

**A new training task names the evaluation task it is scored with.** Evaluation uses Isaac Lab's
play task (for ANYmal-D flat, `Isaac-Velocity-Flat-Anymal-D-Play-v0`, run by `play.py` with
`--eval_output_dir`); there is no train/eval naming rule or task registry here. If a new task needs
its own evaluation configuration, add it and list both IDs in the plan's "Files expected to change"
and in the final Changes table; see [the checklist](references/implementation-checklist.md).

Follow `AGENTS.md` and the checklist for the system you are changing:

- Policies own networks and forwards. Algorithms own losses. The `rsl_rl_rwm` algorithms (`PPO`,
  `MBPOPPO`) expose no auxiliary-loss hooks: extend the class you need by subclassing or a small,
  reviewed change in `../rsl_rl_rwm`, and never copy `update()`.
- Keep intended gradient paths, and block the unintended ones explicitly.
- Assign new parameters to the intended optimizer partition.
- Only when new configurable behavior requires it, extend the rl cfg schema
  (`mbrl/rl/rsl_rl/rl_cfg.py`). Only when a named preset is justified, add it to
  `agents/rsl_rl_ppo_cfg.py`. Every preset instance must own its nested configs.
- Reuse the existing environment unless the signals or physics change. Keep observation term
  order, history orientation and reset semantics.
- Keep old checkpoints loadable unless you break compatibility deliberately and document it. When
  deployable inputs change, update the exports.

## 5. Verify behavior

- For configuration-only experiments, compose and inspect the resolved configs, verify that
  overrides reach the intended components, and run applicable forward/checkpoint/export checks.
  Reuse existing checks; add tests only for new behavior or a meaningful coverage gap.
- Test what the method claims, not that it imports.
- Copy the nearest existing test. The checklist lists which behaviors to test for which kind of
  change.
- Run the new and related tests first, then `pre-commit run --files <changed files>` and the affected tests (the workflow's own tests: `python -m pytest tests/test_research_*.py -q`, plain Python without the simulator; Isaac Lab-dependent code runs in the conda env `rwm`).
- Run a simulator smoke script only if the change needs live evidence, for example a new
  observation term or reset behavior. Say why you ran it.
- **Never start training to validate an implementation.**
- Report pre-existing failures as pre-existing, and show that they fail without your change.

### Deployment smoke (no training)

For deployable policy/network or checkpoint changes, instantiate and forward; save/load a
checkpoint; export/load through an existing export path; and check inference shapes, finite outputs
and agreement where applicable. For persistent recurrence, verify hidden-state initialization,
shape, done-mask reset and episode isolation. Report unsupported exports explicitly.

## 6. Stop for review

Gate B is the active researcher's code review of `git diff <merge-base>...dev/agent/<group>`
(plus the `../rsl_rl_rwm` diff when it changed):
intended inputs; privileged leakage; hypothesis gradient flow; unintended baseline changes;
checkpoint/export compatibility; and whether the implementation broadened the hypothesis.
For configuration-only changes, present the existing tasks, exact proposed overrides, resolved
configuration differences and verification evidence at Gate B; a new model-source diff is not
required. Check that only the approved fields change, including actor/critic scope and evaluation
architecture. Write `implementation.yaml` with the exact proposed variants/overrides for Gate B.
Stop until the active researcher approves. A meaningful implementation/configuration change
requires renewed review. Gate C still locks exact settings and compute budget.


End with exactly these sections:

```markdown
### Implementation summary
Hypothesis implemented; architectural approach; important design decisions (mark assumptions).

### Changes
| File | Why changed |
|---|---|

### Verification
- Tests added:
- Tests run: (command → pass / fail / skipped, with counts)
- Simulator smoke checks: (none, or which and why)

### Scientific/compatibility notes
- Observation changes:
- Gradient ownership:
- Checkpoint compatibility:
- Deployment/export impact:
- Privileged/train-only inputs:
- Unresolved risks:

### Suggested experiment (proposal only)
- Baseline:
- Variants (2–5), each with the one variable it isolates:
- Budget:
- Seeds (training / evaluation):
- Evaluation protocol:
- Primary metrics:
Label every value that is an assumption rather than part of the request.

Implementation is ready for review. No training experiment has been launched.
```

The suggested experiment is a proposal; exact protocol construction belongs to research-experiment.

## References

- [references/implementation-checklist.md](references/implementation-checklist.md): layer map,
  per-change checklists, test templates and anti-patterns.

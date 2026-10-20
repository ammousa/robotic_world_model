---
name: research-experiment
description: >-
  Turn a natural-language request to train variants in this repository into an explicit protocol,
  stop for approval, then launch the runs on a GPU pool, monitor them and verify completion. Use when
  the researcher asks to "run", "test", "train", "launch" or "compare" variants or seeds, for example
  "use 4 GPUs to test these 4 variants for 3000 iterations". Do not use to implement a method
  (research-implement), evaluate checkpoints (research-evaluate), analyze results (research-review)
  or clean up after a decision (research-finalize).
license: MIT
metadata:
  version: 0.1.0
---

# Research Experiment

**Decision authority:** read [research modes](../../../docs/research_modes.md). Attended is the
default (user decisions); explicitly authorized unattended mode routes Gates A–D to a separate
`researcher` via `research-decide`. Stage approval stops remain real gates; the orchestrator
obtains the active researcher's decision and continues approved work. Never self-approve.
Record actual gate receipts and refresh the concise W&B Researcher dashboard after each gate.

This skill covers the **experiment** stage of `frame → implement → code-review → experiment → evaluate → review → researcher-decision → finalize` ([`AGENTS.md`](../../../AGENTS.md)). Conventions, commands, launch safety and completion
checks live in [references/experiment-checklist.md](references/experiment-checklist.md). Follow them
exactly.

**Never** evaluate, rank or pick a winner in this skill. Commit and push only on the comparison's
`dev/agent/<group>` branch, never elsewhere. Do
not create manifests, databases or status files beyond the queue's runtime scratch.

## 1. Consume approved science and reviewed code

Require the approved hypothesis, Gate B-reviewed implementation (source or configuration-only)
and clean comparison branch. Prefer existing tasks with exact per-variant Hydra/CLI
overrides; protocol construction must not introduce preset classes or task IDs merely to label
variants. Preserve the reviewed override intent, and verify resolved training/evaluation settings.
Read [scientific-contracts.md](references/scientific-contracts.md). Resolve the exact executable
`research/<group>/protocol.yaml`: variants and ordered overrides, seeds, budget/pool, baselines,
checkpoint rule, full evaluation population/profile/sampling/paired factors, meaningful effect,
cost accounting, smoke gate, retry policy and predeclared criteria. Every unspecified choice is
**ASSUMPTION** in the protocol and proposal. Validate its hypothesis reference and content ID.
Keep prior executed versions as `protocol.v<N>.yaml`; never overwrite history's meaning.

## 2. Parse the request

Resolve each item from the request, the task registrations and agent configs
(`source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/anymal_d/`), the preset and the code. Mark everything the request did not state as **ASSUMPTION**.

| Item | Where it comes from |
|---|---|
| Baseline | the current method / registered task the variants change |
| Variants | one slug each, with its **exact** overrides (`agent.policy.*`, `agent.algorithm.*`, `agent.system_dynamics.*`, `env.*`) or a registered task ID |
| Training seeds | request; else ASSUMPTION (one seed; state it limits conclusions) |
| Budget | iterations (and `num_envs` if changed); identical for all variants unless the request says otherwise |
| GPUs | the requested pool; no pins unless asked |
| W&B | one stable project per experiment family (for example `rwm-explore`; smoke runs use `rwm-smoke`; never `isaaclab`) and one group slug for the comparison; job type `train`, or `smoke` for a short test run (`--smoke`) |
| Checkpoint rule | final `model_<iterations>.pt` (rsl_rl_rwm saves last+1) unless stated |
| Evaluation population | seeds / `num_envs` / episodes (profile `general`; `evaluation.levels` and `evaluation.difficulties` stay `[]`), recorded now so training and evaluation agree; executed later by research-evaluate |
| Metrics | the primary and secondary metrics, with the evaluation profile they come from |
| Videos | default: one scored video per training run at the first declared evaluation seed; specify the selected cell and camera settings, or an explicit override (see research-evaluate) |

**Check that every override exists before proposing it.** Read the schema or preset field it sets.
The helpers only check form, and Hydra can reject a key, or silently add it, at launch. If a
variant needs code that does not exist yet, stop: that is `research-implement`.

Each variant gives a `task` (training task id) and an `eval_task` (evaluation task id). For the
ANYmal-D flat reference pipeline the training tasks are
`Template-Isaac-Velocity-Flat-Anymal-D-Init-v0` (model-free PPO),
`...-Pretrain-v0` (RWM pretraining; optional `agent.system_dynamics.ensemble_size=5` for RWM-U) and
`...-Finetune-v0` (online model-based PPO). The evaluation task is Isaac Lab's
`Isaac-Velocity-Flat-Anymal-D-Play-v0`; `play.py` with `--eval_output_dir` writes the scored cell.
A variant that loads a checkpoint from another run (for example Finetune from a Pretrain run) is
resumed/chained: the local metric stream needs a fresh run, so set `smoke_gate.enabled: false` for it.

For custom scored metrics, declare explicit equal-unit `metric_reducers` before requesting approval.

## 2. Show the protocol and stop

```markdown
| Variant | Change isolated | Seed(s) | Budget | GPU |
|---|---|---|---|---|
| rwm-u-ens5 | `agent.system_dynamics.ensemble_size=5` | 1 | 3000 it × 4096 envs | pool |
```

Then list:

- evaluation seeds, `num_envs` and episodes;
- the checkpoint rule;
- the metrics, with their profile;
- the W&B project, group, job type, and the tags each run will carry (`variant:`, `seed:`, `task:`, `dirty`);
- the run names;
- the expected commands (the `train-job` output, or one representative job plus the pattern);
- the queue command;
- every ASSUMPTION;
- anything that makes the comparison unfair, such as unequal budgets or RWM pretraining cost.

**Gate C: display the YAML, IDs, exact commands and GPU budget; stop for approval.**
After approval commit the protocol into the clean launch source. Approval may cover “smoke first,
automatically full if technical checks pass”; `queue` enforces the enabled gate even without
`--smoke-first` and needs no second approval after passing. A smoke-only protocol blocks full jobs
until Gate C approves a committed amendment with `auto_full: true`. It checks startup, CPU-loadable
finite model/optimizer checkpoints, finite complete metrics and exception-free logs. Smoke is
`job_type=smoke`, tagged `status:smoke`, and never scientific evidence. Missing local metrics require
manual technical verification; automation stops rather than inferring success.
Scientific changes (architecture, loss, hyperparameters, data or evaluation) invalidate approval:
return to frame/implement/protocol as appropriate, re-review changed code and obtain Gate C again.

**Stop and wait for approval.** Do not launch until the researcher approves, and re-show the
protocol if anything changes.

## 3. Launch (after approval)

1. **Source state.** Work in the comparison's worktree on `dev/agent/<group>`; create it with
   `research_runs.py worktree` if the implementation stage did not. Commit everything the runs need
   there. `queue --group <group>` refuses any other branch, a dirty tree, or an interpreter that
   would import another checkout's code. It then tags the launch `agent/<group>/train-<n>` and
   pushes branch and tag before starting anything.
2. **Jobs.** Write `logs/queue/<group>/jobs.jsonl` with `research_runs.py train-job`, one line per
   variant × seed.
3. **Dry run.** Run `queue … --group <group> --dry-run`. Check the first wave against the approved
   protocol, and check that `launch checks: ok` is printed.
4. **Start.** Start the queue detached: `nohup … queue … > logs/queue/<group>/queue.log 2>&1 &`, or
   your harness's background facility. It assigns free pool GPUs, queues the rest, and writes one
   log per run.

## 4. Monitor

- **Start-up.** Within a few minutes each started job should have a run directory under
  `logs/rsl_rl/<experiment>/`, a W&B run in the group, and a log that is advancing. Report any
  that does not.
- **Status.** Check with `research_runs.py status --log-dir logs/queue/<group>` and the per-run
  logs. Poll at intervals that match the run length; do not spin.
- **Early failure.** Report it with the log excerpt. Keep the failed W&B run.
- **Never restart silently.** A retry keeps cell identity, uses `-attempt<N>` and a new log dir, and
  records the reason plus original-run provenance (`--original-provenance`). Source commit and stable
  resolved config must match; settings-hash equality alone is insufficient. Infrastructure failure
  permits only approved identical retries; algorithm NaNs are a scientific outcome. Changed learning
  rate is a new protocol, not a retry. Select the first valid identical attempt, never the prettiest.
  A retry with different settings needs a new approval.
- **Stopping.** Stop only with `research_runs.py stop`, and only when asked or when a run is
  clearly broken.

## 5. Verify completion and report

Run `check-train` for every run. Exit code 2 means verify the W&B history yourself. Then report:

```markdown
| Run | W&B run | GPU | Checkpoint | Complete | Problems |
|---|---|---|---|---|---|
```

Also give the branch, the launch tag and commit, the queue log dir, and any retries or failures.
End with:

> Training runs are finished (or: N of M complete). No evaluation has been launched.

Evaluation proceeds as a separate `research-evaluate` stage when covered by the approved protocol and GPU budget; otherwise propose its budget at Gate C. Do not ask again for already-approved evaluation.

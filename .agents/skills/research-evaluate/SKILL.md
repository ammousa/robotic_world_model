---
name: research-evaluate
description: >-
  Evaluate completed training runs of one research comparison in this repository on an identical,
  reproducible protocol: resolve runs, commits and checkpoints, run every evaluation cell through
  play.py on a GPU pool, record comparable scored videos, verify every
  cell, and report a completion table. Use when the researcher asks to "evaluate", "score", "test on
  seeds" or "record videos" for trained runs. Do not use to train (research-experiment), to
  interpret or rank results (research-review), or to clean up code (research-finalize).
license: MIT
metadata:
  version: 0.1.0
---

# Research Evaluate

**Decision authority:** read [research modes](../../../docs/research_modes.md). Attended is the
default (user decisions); explicitly authorized unattended mode routes Gates A–D to a separate
`researcher` via `research-decide`. Stage approval stops remain real gates; the orchestrator
obtains the active researcher's decision and continues approved work. Never self-approve.
Record actual gate receipts and refresh the concise W&B Researcher dashboard after each gate.

This skill covers the **evaluate** stage of `frame → implement → code-review → experiment → evaluate → review → researcher-decision → finalize` ([`AGENTS.md`](../../../AGENTS.md)). Naming, commands and the fairness rules are in the
shared [experiment checklist](../research-experiment/references/experiment-checklist.md).

This skill produces evidence; it does not interpret it. Never rank, recommend or pick a winner here.
Run it from the comparison's worktree on `dev/agent/<group>`, so evaluation uses the code that
trained. If the evaluation needs a code change, such as a missing evaluation configuration, commit it there
first. The evaluation queue is `queue --group <group>` like training, and is pinned as
`agent/<group>/eval-<n>`.

## 1. Consume the approved protocol

Load the committed protocol and hypothesis IDs from each training run. Pass `--protocol` to
`eval-jobs`, `check-eval` and `publish-eval`; reject identity, profile or sampling mismatches.
Publication requires the complete approved matrix and finite primary/secondary/required metrics,
including scored row inputs. Verify the exact training variant, seed, attempt, clean commit and
checkpoint SHA-256 captured at policy load; an explicit W&B run ID never overrides those checks.
No new scientific choice is made here. A bug returns to implement, renewed Gate B review and rerunning affected
experiments; old invalid evidence stays in W&B.

## 2. Resolve what is being evaluated

For every run in the comparison, establish the following:

- **The runs.** Use the W&B group (runs named `<timestamp>_<variant>-seed<seed>`) and the training
  queue's `logs/queue/<group>/events.jsonl`. Each run directory comes from
  `research_runs.py check-train --log-root … --name …`. Include only runs whose training
  `check-train` passed, and list the others as not evaluable.
- **The commit.** Take it from the W&B run or the queue's `source` event. Different commits across
  compared runs is a finding to report, not something to ignore.
- **The checkpoint.** Apply the protocol's rule, normally final `model_<iterations>.pt` (rsl_rl_rwm saves last+1), the
  same for every run.
- **The evaluation task.** The protocol variant's `eval_task`, plus any evaluation overrides the
  variant needs, for example a checkpoint path. For the ANYmal-D flat reference pipeline it is Isaac
  Lab's `Isaac-Velocity-Flat-Anymal-D-Play-v0` (the training tasks are the `...-Init-v0`,
  `...-Pretrain-v0` and `...-Finetune-v0` variants); `play.py --eval_output_dir` writes the scored
  cell. Never evaluate with the training task. If the evaluation task is missing or wrong, stop and
  fix it through `research-implement`; do not improvise evaluation settings with overrides.
- **The population.** Seeds, `num_envs` and `num_episodes`. This repository has no terrain levels or
  difficulty presets: `evaluation.levels` and `evaluation.difficulties` stay `[]`, and
  `eval-jobs`/`queue` refuse `--levels`/`--difficulties` (`play.py` has no `--force_terrain_level` or
  `--difficulty`). The profile is `general`.
- **Metrics and videos.** The required metrics, with the profile that defines each, and whether
  videos are required (default: one scored video per training run, using the first declared
  evaluation seed). Standard metrics (`success_rate`, `fall_rate`, `timeout_rate`, `survival_rate`,
  `lin_vel_tracking_rmse`, `ang_vel_tracking_rmse`, `mean_episode_return`, `mean_episode_duration`,
  `mean_episode_length`, `mean_distance_travelled_m`, `mean_path_length_m`) are computed from the
  `episode_recorder` rows (`env_id`, `episode_id`, `seed`, `episode_return`, `episode_length`,
  `time_seconds`, `fell`, `timed_out`, `success`, `distance_travelled`, `path_length`, plus
  `lin_vel_tracking_error`/`ang_vel_tracking_error` when the task has a `base_velocity` command).
  The `realized_*` terrain/friction/mass/push metrics are not available here and fail loudly if
  declared as evidence; a custom diagnostic needs `evaluation.metric_reducers` plus a recorder
  column.

If the request leaves any of this open, say what you will use (ASSUMPTION) before running. Stop only
if the choice changes what is compared.

## 2. Enforce comparability before running

Every run gets identical settings, and only the cell varies:

- the same seeds, environments and episodes;
- the same episode policy;
- the same checkpoint rule;
- the same evaluation profile and metric definitions.

`eval-jobs` builds every cell from one set of arguments, so matching arguments means matching
protocol. **Refuse to compare runs whose evaluation profiles differ.** `survival_rate` differs
between the `general` and `forward_course` profiles; see the checklist.

## 3. Run the evaluation matrix

```bash
G=<group>; Q=logs/queue/$G-eval; mkdir -p $Q
python scripts/research_runs.py eval-jobs --task <eval-task> --checkpoint <run_dir>/model_3000.pt \
    --name <variant>-seed<seed> --seeds 41 42 43 --num-envs 150 --num-episodes 1 \
    --output-root <run_dir>/eval/${G}_model_3000 -- <eval overrides> >> $Q/jobs.jsonl
# ...once per run, then:
python scripts/research_runs.py queue --jobs $Q/jobs.jsonl --gpus <pool> --log-dir $Q --dry-run
nohup python scripts/research_runs.py queue --jobs $Q/jobs.jsonl --gpus <pool> --log-dir $Q > $Q/queue.log 2>&1 &
```

The source-state and one-job-per-GPU rules apply as in training. **Cells create no W&B runs**: they
write local artifacts and scored videos only. Each training run's evaluation reaches W&B as one run,
published after it is checked (step 5).

**The scored population follows the protocol episode-sampling policy.** The current evaluator supports `first_episode_per_env` (`episode_id == 0`); fixed-k is declared but rejected until supported. A
fixed-step evaluation replaces early failures with new episodes, so later episodes over-represent
failures. Check each cell with `--expected-first-episodes <num_envs>`. Never pool replacement episodes for new comparisons. Validate one scored episode per declared environment.

Never replace a missing or failed scored cell with a later run or a different checkpoint. Rerun that
exact cell (a new eval log dir, the same arguments) or report it as missing. Pre-reset terminal
semantics come from the recorders; do not post-process outcomes.

## 4. Videos

- **Default selection.** Record exactly one scored video per training run, from the cell of the
  first declared evaluation seed (there is no level or difficulty axis here). Record this selection
  and the camera settings in the protocol before Gate C. An explicit approved video policy overrides
  this default. This applies to simulator evaluations that support recording.
- **Selected cell only.** Pass `-- --video 1 --robot_idx 0` only for the selected cell; leave video
  flags off all other cells. `eval-jobs` forwards extra flags to every cell, so generate the full
  matrix normally and add these flags to the selected job before queueing. Keep the full scoring
  matrix and its scientific overrides unchanged. Do not pass video flags to the whole matrix.
- **Scored videos.** Use the same robots,
  view and settings for every variant. `publish-eval` attaches them to the evaluation run with the
  caption "scored".
- **File names.** `--note` is the video file prefix, so each cell's video is
  `<run_dir>/videos/play/<run>__<cell>-step-0.mp4`.
- **Replays.** A video requested after scoring is a **replay**:
  - give it its own `--output-root …_replay` and label it "replay" everywhere;
  - on W&B it is a separate `video` run tagged `replay` (see the checklist);
  - never let its episodes count toward scores, and never publish it with `publish-eval`.

## 5. Verify and report

```bash
python scripts/research_runs.py check-eval --output-root <root-A> <root-B> ... --checkpoint <ckpt-A> <ckpt-B> ... \
    --seeds 41 42 43 --expected-first-episodes 150 --metrics success_rate survival_rate ...
```

It checks every planned cell under every root:

- the artifacts exist;
- seed and checkpoint provenance match;
- episode counts match;
- the required metrics are finite;
- all cells share one profile and protocol (`comparable`).

Report:

```markdown
| Run | Cells ok / planned | Missing or failed cells | Profile | Commit | Videos |
|---|---|---|---|---|---|
```

Then publish each checked training run's evaluation as **one** W&B run:

```bash
python scripts/research_runs.py publish-eval --project <project> --group <group> --name <variant>-seed<seed> \
    --task <train-task> --eval-task <eval-task> --checkpoint <ckpt> --output-root <root> \
    --seeds … --expected-first-episodes <num_envs> --dry-run   # review, then without --dry-run
```

It refuses incomplete or mixed evaluations. The run is `<variant>-seed<seed>__eval`, with
`job_type=eval` and tags `variant:`, `seed:`, `task:`, `ckpt:` and `protocol:`. It holds a cell
table, metrics pooled over the first episodes (each with
`episodes`), the scored videos, and a `research` config linking the training run's W&B id.
Republishing needs `--supersede`.

Also give the `check-eval` summary (`complete`, `comparable`, `protocols`, `commits`), the
published evaluation runs, and where
the artifacts are. End with:

> Evaluation finished: N of M cells complete, comparable: yes/no. No conclusions drawn.

Evidence interpretation is a separate `research-review` stage within the authorized comparison. Never combine scoring with deciding what to keep.

## Paired evidence

The protocol declares factors actually controlled by the task profile: seed, environment identity,
command trajectory, disturbance schedule and duration where reproducible.
`eval_unit_id` is derived from configured factor values in the scored row/context and preserved in
W&B episode artifacts. Equal seeds alone do not prove matched random trajectories after resets.
Missing factors or duplicate units invalidate paired analysis; report unverified dimensions.

# Experiment checklist

Shared by `research-experiment`, `research-evaluate`, `research-review` and `research-finalize`.
Git identifies the code, W&B holds the runs and the reports, and
[`scripts/research_runs.py`](../../../../scripts/research_runs.py) removes the repeated manual work.
There is no runtime experiment database or local report. Scientific contracts and the documented state machine live in Git; read [scientific-contracts.md](scientific-contracts.md).

## Git structure (always)

| Rule | How |
|---|---|
| One branch per comparison | `dev/agent/<group>`, the same slug as the W&B group. `research_runs.py worktree --group <group> [--base <branch>]` creates it in its own worktree `../robotic_world_model-agent-<group>` next to this checkout, sharing the main checkout's `logs/` by symlink. `--base` defaults to the repository default branch (`master`). Untracked or LFS-smudged assets such as `assets/models/pretrain_rnn_ens.pt` are **not** in a worktree: copy or link what a run needs before launching |
| Follow-ups | a new group and a new branch created from the previous one (`--base dev/agent/<previous>`) |
| Commits | agents may commit on `dev/agent/*` without asking each time. Never on the default branch (`master`) or any other branch |
| Launch source | `queue --group <group>` launches only from a **clean** commit of `dev/agent/<group>`, run in that worktree. Jobs import the extension package `mbrl` (and `research_workflow`) from the worktree (`PYTHONPATH`), and the queue checks this before launching |
| Launch pin | each launch tags HEAD `agent/<group>/train-<n>` or `eval-<n>` and pushes branch and tag to `origin`. If the push fails, nothing launches. Training runs record branch, commit and launch tag in the run config's `research` block (`config.runner_cfg.research` with the rsl_rl_rwm fork; newer rsl_rl uses `config.train_cfg.research`; the tooling reads both) |
| History | never rebase, amend, force-push or delete an agent branch or launch tag: W&B runs point at those commits. Merging into the default branch (`master`) is the researcher's action |
| Comparison-specific files | `research/<group>/` on the agent branch (for example `analysis.ipynb`, `plots.py`); `research-finalize` retains scientific contracts, version history, analysis code and the minimal decision |

## W&B structure (always)

| Layer | Meaning | Convention |
|---|---|---|
| **Project** | stable home of one experiment family, never per experiment | `rwm-explore` (smoke runs: `rwm-smoke`). Always explicit; the helpers refuse rsl_rl's fallback `isaaclab` |
| **Group** | one research comparison | short slug: `history-lstm-full`. Training, evaluation and report runs of a comparison share project and group |
| **job_type** | what the run is | `train` (counts toward results), `smoke` (short test, `--smoke`), `eval` (one per training run, from `publish-eval`), `video` (illustrative re-render, never scored), `analysis` (the comparison's analysis notebook, from `run-analysis`), `report` (one per comparison, from `publish-report`) |
| **Tags** | filtering; editable after the run | the vocabulary below, enforced by the helpers |
| **Config** | exact identity; never edited | training: the run's own configs and command line, plus the `research` block of the runner config (variant, seed, task, comparison, branch, commit, launch tag); evaluation: a `research` block (comparison, variant, seed, task, eval task, checkpoint, `train_run_id`, commits, dirty, protocol, output root) |

**Tag vocabulary.** Tags are `key:value` with a fixed key set, or one of two flags.

| Tag | Set by | On |
|---|---|---|
| `variant:<variant>` `seed:<n>` `task:<train-task-id>` | `train-job`, `publish-eval` (at launch) | train/smoke and eval runs |
| `ckpt:<model_N>` `protocol:<slug>` | `publish-eval` | eval runs. The slug is the population: `s41-42-43_e260x1` |
| `dirty` | the helpers, when the source was uncommitted (should not happen: launches need a clean agent branch) | any |
| `replay` | you, on `video` runs | illustrative videos |
| `status:smoke` / `status:final` / `status:superseded` | `research_runs.py tag`, later; `publish-*` marks a replaced publication superseded | any |
| `decision:chosen` / `decision:rejected` | `research_runs.py tag`, by `research-finalize` after the researcher decides | the variant's train and eval runs |

Never invent other tags. Identity tags are not edited after launch.

**Names.**

| Thing | Convention |
|---|---|
| Variant | slug describing the one change: `temporal128-p0.1` |
| Training run | `--run_name <variant>-seed<seed>`. rsl_rl (the rsl_rl_rwm fork) names the W&B run after its log directory, so it appears as `<timestamp>_<variant>-seed<seed>` |
| Evaluation cell | `<variant>-seed<seed>__<cell>`, cell = `seed41` (the generic matrix machinery also names level/difficulty cells `L4-seed41` / `hard-seed41`, but this repository has no terrain levels or difficulties). Local only: cells write `<run_dir>/eval/<group>_<checkpoint-stem>/<cell>/` and create **no** W&B run |
| Evaluation run | `<variant>-seed<seed>__eval`: one per training run and checkpoint |
| Report | one W&B Report per comparison, titled after the group. Its `report` run `<group>__report` holds the custom plots (`custom/<key>`), the `report-code-<group>` code artifact and the report URL |
| Retry | a new run name `<variant>-seed<seed>-attempt2` (stable cell, unique attempt, reason, original-run provenance, verified source/config identity) in a new queue log dir. The failed run stays in W&B, untouched |
| Queue scratch | `logs/queue/<group>/` (training), `logs/queue/<group>-eval/` (evaluation). Git-ignored runtime scratch; not a record. Report inputs (`intro.md`, the analysis draft) also live here; they are inputs, not reports |

For every run, W&B must contain the commit, the task or method, the variant, the seed, the exact
overrides, the budget, the group, the checkpoint used, the evaluation protocol, the metrics, and
video or artifact references. Training runs get these from `train.py` (command line, configs,
commit) plus the identity tags. Evaluation runs get them from `publish-eval`'s `research` config,
tags and cell table.

## Commands

Always build jobs with the helpers, so naming, tags and managed flags cannot drift:

```bash
P=rwm-explore; G=rwm-ensemble-size
python scripts/research_runs.py worktree --group $G            # once, from the main checkout
cd ../robotic_world_model-agent-$G; Q=logs/queue/$G; mkdir -p $Q   # everything below runs in the worktree
# implement and commit on dev/agent/$G, then: training
python scripts/research_runs.py train-job --task <train-task> --variant <variant> --seed 1 \
    --iterations 1500 --project $P --group $G -- <exact overrides...> >> $Q/jobs.jsonl
python scripts/research_runs.py queue --jobs $Q/jobs.jsonl --gpus 0 1 2 3 --log-dir $Q --group $G --dry-run
nohup python scripts/research_runs.py queue --jobs $Q/jobs.jsonl --gpus 0 1 2 3 --log-dir $Q --group $G \
    > $Q/queue.log 2>&1 &
python scripts/research_runs.py status --log-dir $Q
python scripts/research_runs.py check-train --log-root logs/rsl_rl/<experiment> --name <variant>-seed1 \
    --iterations 1500 --log $Q/<variant>-seed1.log
# evaluation (cells are local; one W&B eval run per training run after the check)
python scripts/research_runs.py eval-jobs --task <eval-task> --checkpoint <run_dir>/model_1500.pt \
    --name <variant>-seed1 --seeds 41 42 43 \
    --num-envs 260 --num-episodes 1 --output-root <run_dir>/eval/${G}_model_1500 >> $Q-eval/jobs.jsonl
python scripts/research_runs.py check-eval --output-root <root…> --checkpoint <ckpt…> --seeds 41 42 43 \
    --expected-first-episodes 260 --metrics success_rate …
python scripts/research_runs.py publish-eval --project $P --group $G --name <variant>-seed1 \
    --task <train-task> --eval-task <eval-task> --checkpoint <run_dir>/model_1500.pt \
    --output-root <run_dir>/eval/${G}_model_1500 --seeds 41 42 43 \
    --expected-first-episodes 260 [--dry-run]
# the eval queue launches the same way: queue --jobs $Q-eval/jobs.jsonl --log-dir $Q-eval --group $G …
# (`--levels` / `--difficulties` exist in the generic matrix machinery, but eval-jobs/queue refuse them here:
#  play.py has no --force_terrain_level / --difficulty and the protocol keeps evaluation.levels/difficulties `[]`)
# analysis notebook (research-review), then the report (W&B Report; custom plots optional) and lifecycle
python scripts/research_runs.py run-analysis --project $P --group $G \
    --notebook research/$G/analysis.ipynb [--draft-out $Q/analysis_draft.md] [--dry-run]
python scripts/research_runs.py publish-report --project $P --group $G --intro $Q/intro.md \
    --analysis $Q/analysis_draft.md [--plots research/$G/plots.py] [--dry-run]
python scripts/research_runs.py publish-report --project $P --group $G --decision $Q/decision.md   # finalize
python scripts/research_runs.py tag --project $P --group $G --variants <variant> --add decision:chosen [--dry-run]
```

- Everything after `--` is passed through verbatim. The helpers refuse extras that would override
  a setting they own: seed, device, budget, run name, W&B placement, tags, output dirs.
- `publish-eval` publishes only an evaluation that `check-eval` passes.
  - **Metrics:** recomputed from the first episode of every environment with the repository's
    `aggregate_metrics` (computed from the `episode_recorder` columns), pooled over all cells, each
    with its `episodes` denominator. The `realized_*` terrain/friction/mass/push metrics are not
    available here; declaring them as evidence fails loudly, and a custom diagnostic needs
    `evaluation.metric_reducers` plus a recorder column.
  - **Attachments:** the scored videos found for the cells, and the scored episodes themselves as
    the `eval-episodes-<variant>-seed<seed>` artifact, so custom plots can be rebuilt from W&B alone.
  - **Republishing** needs `--supersede`, which tags the old run `status:superseded`; nothing is
    deleted.
- `run-analysis` checks every code cell of the analysis notebook with the custom-plot rules,
  re-executes it from scratch in an empty directory, and only then lets it log. It supersedes the
  previous `analysis` run, attaches the executed notebook as the `analysis-notebook-<group>` code
  artifact and writes the Analysis draft. `--dry-run` executes with logging off.
- `publish-report` builds or refreshes the comparison's W&B Report from W&B data. Refreshing keeps
  the report URL and the human-owned Analysis and Decision sections; see the report section below.
- A job outside the helpers' templates can be written by hand as `{"name", "argv", "gpu": null}`, as long as
  `argv` places the device with `{gpu}` (for example `device=cuda:{gpu}`).
- Run everything with the interpreter of the Isaac Lab conda env `rwm`. `argv[0]` is the interpreter running
  the helper; do not hard-code one.

## Launch safety

- **Approval first.** Show the protocol, then stop. Launch only after the researcher approves.
- **Clean agent branch.** `queue --group <group>` refuses any branch other than `dev/agent/<group>`,
  a dirty tree, jobs that log to another group, and an interpreter that would import another
  checkout's `mbrl`/`research_workflow`. Commit on the agent branch first; there is no override.
- **One heavy job per GPU.** The queue treats `--gpus` as a pool, starts the next job when a pool
  GPU is free of our jobs *and* of anyone else's compute processes, and queues the rest. Use
  `--allow-shared-gpu` only when explicitly asked. Pin (`gpu` / `--gpu`) only when asked.
- **No silent reruns.** A log dir refuses job names it already launched. Changing settings means a
  new protocol and approval.
- **Stopping.** Stop only through `research_runs.py stop`. It signals a process group only if pid,
  start time, uid and command line all still match what the queue recorded. Never `kill` by PID
  or name.
- **Budget.** Never start long training to "check" something. A short launch for a smoke test is
  itself a run to report.

## Completion is evidence, not an exit code

Isaac processes can exit 0 after an exception, so a run is complete only when `check-train` says so:

- `params/agent.yaml` exists;
- `model_<iterations>.pt` exists (fresh runs; rsl_rl_rwm saves last+1, so 1500 iterations end at `model_1500.pt`);
- the local metric stream (`locomotion_metrics.jsonl`, written for every run by `train.py`) has
  every iteration, is finite, and its `Checks/finite_rewards_std` is 1. The stream requires a
  **fresh** run (records 0..N-1), so the smoke gate does not support resumed or chained variants
  (for example Finetune loading a Pretrain checkpoint): set `smoke_gate.enabled: false` for those;
- the run's log has no traceback.

Exit code 2 means the runner writes no local metric stream. Verify the run's W&B history is
finite and complete for the budget, and say that you did.

`research_runs.py` covers the simulator-based pipeline (`train.py` / `play.py`). Offline,
simulator-free policy training (`scripts/reinforcement_learning/model_based/train.py`) is not
covered by it.

Early-failure signs:

- a traceback or `Error` in the log;
- no run directory, or no W&B run, a few minutes after start;
- a metric stream that stops growing;
- a queue `exit` event with a nonzero return code.

## Fair evaluation

Compared runs must share all of the following, and `check-eval` verifies what it can:

- the checkpoint rule: final (`model_<iterations>.pt`) unless the protocol says otherwise, and
  the same for every run;
- the evaluation task and profile;
- the seeds;
- `num_envs` and `num_episodes`;
- the episode-selection policy;
- the metric definitions.

`check-eval` classifies each cell as a `general` or `forward_course` profile. **`survival_rate`
means `timeout_rate` in general evaluations but `1 − failure_rate` in forward-course ones. Never
compare a same-named metric across profiles.** This repository's `play.py` evaluation is the `general` profile.

Other rules:

- A missing or failed scored cell is reported as missing. It is never replaced by a later or
  different run.
- **Videos.**
  - Default to one scored video per training run, at the first declared evaluation seed (this
    repository has no terrain levels). Use `--video 1 --robot_idx 0` on that cell only;
    `eval-jobs` extra flags apply to every cell, so add recording flags only to the selected job.
    Declare the selected cell and camera settings before Gate C; explicit protocol choices override
    this default. See `research-evaluate` for the full selection rule.
  - Scored videos come from the scoring run itself (`--video N --robot_idx …` passed through), with
    the same robots, view and settings for every variant.
  - Videos rendered later are **replays**: kept in their own output root, never presented as the
    scored episodes. If a replay goes to W&B, it is its own `play.py` run with `--logger wandb`,
    `--job_type video`, `--run_name <variant>-seed<seed>__replay-<cell>` and
    `--wandb_tags variant:<v> seed:<n> replay` in the comparison's project and group.
- **Budgets.** Report training budgets and extra costs such as RWM pretraining next to results.
  A common-budget comparison and a latest-capability comparison are different questions.

## Reports: one W&B Report per comparison (always)

There are no local Markdown reports. `publish-report` builds the report from W&B data with native,
interactive panels:

| Section | Content |
|---|---|
| Overview | your intro (hypothesis, variants and what each isolates, budget, protocol); a run comparer of the training configs |
| Training | line panels of the training metrics, mean ± standard error over seeds, grouped by the `research.variant` config key; always random point sampling and time-weighted exponential smoothing 0.95 (`LINE_PLOT_STYLE`) |
| Evaluation | bar panels of `eval/<metric>`, grouped by variant, plus the `eval/cells` tables |
| Videos | the scored videos of the evaluation runs |
| Analysis figures | the figures and tables kept by `research/<group>/analysis.ipynb`, from its latest `analysis` run, with the executed notebook attached to that run; see [analysis-notebook.md](../../research-review/references/analysis-notebook.md) |
| Custom plots | only if needed: figures from `research/<group>/plots.py`, built from the W&B API; see [report-plots.md](../../research-review/references/report-plots.md) |
| Analysis | the agent's draft on first publication; afterwards **owned by the researcher** and preserved on every refresh |
| Decision | `_Pending_` until `research-finalize` replaces it with `--decision` |
| Provenance | branch, launch tags, commits, run counts, generator and plot-code artifact |

Superseded and smoke runs are excluded. Refreshing (running `publish-report` again) keeps the
report's URL, replaces only the generated sections, and supersedes the previous `report` run.

## New comparisons (v1)

Follow Gate A/B/C/D in AGENTS.md and the [contract reference](scientific-contracts.md).
Pass `--protocol research/$G/protocol.yaml` to train-job, eval-jobs and publish-eval. New queues
require protocol-bound jobs. Old examples above remain useful for `legacy workflow` groups;
`queue --legacy-workflow` is only for their explicit continuation. Runtime fingerprint differences
must be reviewed; seed equality alone is not proof of matched commands/disturbances. Sampling is
part of protocol identity; current first-episode scoring is preserved and fixed-k execution rejected.
Smoke is technical evidence only. Retries never change settings or conceal scientific failures.

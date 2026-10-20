# Agent instructions

Repository-wide rules for coding agents. These are operational rules; the reasoning lives in the
linked documents. Read the relevant one before changing that layer.

| Before touching… | Read |
|---|---|
| anything | [`README.md`](README.md) ("Code Structure"), the `exp/PLAN.md` run notes if you launch jobs |
| tasks, environment or agent configs, dynamics-model setup | `source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/anymal_d/` |
| the algorithm, dynamics model or runners | the sibling repository `../rsl_rl_rwm` (see below) |
| the research workflow itself | [`docs/research_modes.md`](docs/research_modes.md), [`docs/research_reporting.md`](docs/research_reporting.md) |

The working tree, including staged, unstaged and untracked files, is authoritative. Never discard,
stash, overwrite or "clean up" work you did not create. In particular `exp/` (the run pipeline and its
results) and the modified `assets/models/pretrain_rnn_ens.pt` (the real file behind a Git LFS pointer)
belong to the user.

## Architecture

- **Extend the smallest appropriate layer.** A model-only experiment is a config value or a
  policy/dynamics module, not a new environment, task or runner.
- **Algorithms live in `../rsl_rl_rwm`**, a separate Git repository (PPO, `MBPOPPO`, the system
  dynamics model, `MBPOOnPolicyRunner`). Extensions reuse the existing update lifecycle; never copy
  `update()` or add a runner for one method. A change there is not in this repository's diff: commit
  it on its own branch, report its commit, and review it at Gate B. The runtime fingerprint records its
  HEAD (`-dirty` when uncommitted).
- **MDP terms** stay reusable, in `source/mbrl/mbrl/envs/mdp/` or `tasks/.../velocity/mdp/`, never in
  workflow scripts.
- **Environment changes** only for real environment, observation or physics differences. Model inputs
  and outputs are the observation groups of `ObservationsCfg_PRETRAIN`.
- **Evaluation uses the stock `Isaac-Velocity-Flat-Anymal-D-Play-v0` task** through `play.py` (an
  `OnPolicyRunner` loads the policy). A variant that changes the policy network needs its own Play
  registration in the same change.
- Every config instance owns **independent nested config objects**.
- **Preserve** observation term order, history orientation, action dimensions, normalization,
  episode/reset and terminal semantics, physical units, checkpoint compatibility where claimed, and
  export inputs. These break experiments without any error.
- **Compatibility and migration logic stays at boundaries**: config/CLI translation and checkpoint
  loading. Core code does not carry it.
- Do not hard-code machine paths or interpreters.

## Research workflow

`frame → implement → code-review → experiment → evaluate → review → researcher-decision → finalize`

Before code, turn a rough idea into a falsifiable scientific contract. Before GPU use, turn that
contract into an exact executable protocol. Agent skills cover frame, implement, experiment
(protocol and execution), evaluate, evidence review and finalize; decision gates use the active
researcher (the user by default).

| Stage / gate | Required result |
|---|---|
| Frame · **Gate A ⏸** | `research-frame` proposes `research/<group>/hypothesis.yaml`: claim, mechanism, inputs, gradients, baselines, outcomes, constraints and falsification. Researcher approves the question. |
| Implement | Smallest architectural change, behavior tests, instantiate/forward/checkpoint/inference smoke; recurrent initialization/shape/reset tests for the dynamics model. No scientific training. |
| Code review · **Gate B ⏸** | Researcher reviews `git diff <merge-base>...dev/agent/<group>` plus any `../rsl_rl_rwm` commit: inputs, privileged leakage, gradients, unchanged baseline, checkpoint/export compatibility, broadened hypothesis. |
| Experiment · **Gate C ⏸** | Exact `protocol.yaml`: overrides, seeds, budget/pool, checkpoint, matched evaluation/sampling, costs, criteria, retry and smoke policy. Mark unspecified values ASSUMPTION. Approve and commit before execution. |
| Execute | Clean-commit launch, provenance, monitoring, technical smoke then already-approved full runs. No second gate for a passing smoke when auto-full was approved. |
| Evaluate | Every cell identically; finite metrics, checkpoints, equal population contribution, verified paired IDs, scored videos. |
| Review | Raw training-seed effects, paired differences, aggregate descriptive effects, low-N limits, practical significance and constraints; classify predeclared criteria, never pick a winner. |
| Researcher decision · **Gate D ⏸** | Keep/reject/modify/extend; supported, not supported, inconclusive and invalid are legitimate evidence outcomes. Record minimal `decision.yaml` and rich W&B narrative. |
| Finalize | Apply only the explicit decision; keep scientific contracts/protocol versions/analysis code. Validate and leave cleanup uncommitted for review; never merge. |

One comparison has one `dev/agent/<group>` branch (created with `research_runs.py worktree`) and one
matching W&B group. Agents may commit/push there when authorized by the task, never elsewhere;
never rebase, force-push, delete branches/tags or merge. Explicit session restrictions take priority.

**Controlled backward transitions:** review → experiment adds seeds via an archived protocol version
and renewed Gate C; review → frame changes the claim via a new hypothesis version/comparison and
Gate A; evaluate → implement fixes a bug, reopens Gate B and reruns affected experiments; smoke →
implement fixes a technical bug, tests and re-reviews the diff before a new clean launch. Preserve
old invalid/negative runs. Never silently mutate an executed experiment's meaning.

Prefer existing configuration and supported Hydra/CLI overrides before changing source. A
configuration-only experiment completes implementation by verifying resolved settings and
applicable behavior, with no new model code required. Do not add presets, task IDs, runners or
helpers merely to name variants. Add source/config abstractions only for behavior overrides cannot
express, concrete reuse or required validation. Gate B reviews exact overrides, resolved differences
and verification evidence when there is no model-source diff; Gate C still approves and commits the
exact protocol and budget. Reuse the existing training tasks and the Play task and preserve their architecture.

Scientific choices require their gate; engineering filenames/helpers/fixtures do not. Consolidate
necessary scientific questions. Everything within approved settings can be automated.

### Attended and unattended modes

Default to **attended**: the user owns Gates A–D. Explicit **unattended** mode delegates those
choices to a separate agent named `researcher`, within a user-authorized objective, scope, compute
limits, follow-up/retry limits and commit/push/finalize permissions. Read
[`docs/research_modes.md`](docs/research_modes.md) before starting or switching modes; the
comparison's `workflow.yaml` records authority. Never infer unattended permission from silence,
a config file or missing approvals. New comparisons create this file before framing.

In unattended mode invoke `researcher` independently at each gate, using the project custom agent
or a fresh generic subagent supplied its full instructions. The worker submits actual artifacts
and verification; it does not self-approve. `research-decide` handles gate decisions. Existing
skills still stop their stage for approval, but the orchestrator obtains the agent's decision and
continues authorized stages without asking the user again. If delegation is unavailable or the
mandate is exceeded, stop dependent work and report the precise blocker.

Append actual gate decisions with `record-gate`; approvals bind current contracts, executable
source and reviewed overrides. `queue` verifies these receipts for mode-configured comparisons.
Publish concise decisions after each gate in the W&B Report's **Researcher dashboard**: decision,
reason, evidence, limitation and next action. Preserve full history and exact IDs in Git and the
report artifact. The dashboard supports later user revision without rewriting previous decisions.
Human Analysis/Decision text survives refresh. No second runtime state system is introduced.
Unattended decisions cannot authorize merging or history rewriting. Finalization stays uncommitted.

### Worked example: dynamics-model ensemble (RWM versus RWM-U)

1. `research-frame` defines the claim (a 5-member ensemble yields a policy with lower tracking error and
   no more falls), the baseline (a single model), the primary effect, the fall constraint and the
   confounders (the ensemble needs a smaller dynamics mini-batch); shows the contract; **Gate A**.
2. `research-implement` finds that Hydra overrides (`agent.system_dynamics.ensemble_size=5`,
   `agent.algorithm.system_dynamics_mini_batch_size=1500`) express it, so no source changes; it
   verifies the resolved settings; **Gate B** reviews the overrides.
3. `research-experiment` resolves exact variants, seeds, budget and matched evaluation, records the
   unverified random schedules, locks the protocol; **Gate C** approves smoke → automatic full runs.
4. `research-evaluate` scores every cell under the same declared sampling and protocol.
5. `research-review` reports met/not_met/inconclusive/invalid criteria, seed and paired effects,
   meaningful effect and constraints, and costs. No winner.
6. The researcher decides keep/reject/modify/extend (**Gate D**) and records the decision.
7. `research-finalize` applies that decision and leaves cleanup uncommitted. This example is not a
   generic configuration; all scientific values require their actual contract.

## Experiments: Git + W&B

There is no local experiment framework. Git identifies the code and W&B holds the runs. Launch the
repository's real entry points directly (`scripts/reinforcement_learning/rsl_rl/train.py` and
`play.py`). `scripts/research_runs.py` builds those commands
with the conventions below, runs them on a GPU pool and checks completion. The details are in
`.agents/skills/research-experiment/references/experiment-checklist.md`.

**W&B structure** (always; details in the checklist)

- **Project:** one stable project per experiment family: `rwm-explore` (`rwm-smoke` for pipeline
  smoke tests outside a comparison). Never rsl_rl's fallback `isaaclab`.
- **Group:** one per research comparison, a short slug, e.g. `rwm-ensemble`.
- **Job type:**
  - `train`, or `smoke` for short test runs;
  - `eval`: one run per training run, `<variant>-seed<seed>__eval`, published after the cells
    check out;
  - `video`: illustrative replays only;
  - `analysis`: one current run per comparison, `<group>__analysis`, from `research_runs.py
    run-analysis`. It holds the figures and tables the analysis notebook kept and the executed notebook;
  - `report`: one per comparison, `<group>__report`. It carries custom plots and their code, and
    belongs to the comparison's W&B Report.

  Evaluation cells write local artifacts only.
- **Training runs** are named `<variant>-seed<seed>`. rsl_rl prefixes the W&B name with the
  log-directory timestamp.
- **Tags** come from a fixed vocabulary:
  - at launch: `variant:` `seed:` `task:`, plus `hypothesis:` `protocol:` `protocol_version:` `cell:` `attempt:` for new runs and `ckpt:` on evaluation runs, plus `dirty`
    or `replay` flags;
  - later, with `research_runs.py tag`: `status:smoke|final|superseded|failed|invalid|inconclusive|supported|not_supported` and
    `decision:chosen|rejected`. Only `research-finalize` sets decision tags, after the researcher
    decides.

**What W&B must contain, or let someone reconstruct, for every run:**

- Git commit
- task/method and variant
- seed
- exact CLI and config overrides
- training budget
- W&B group
- checkpoint used
- evaluation protocol
- metrics
- video and artifact references, where relevant

For RL training, `research_runs.py train-job` sets `--logger wandb --log_project_name … --group …
--job_type train --run_name … --wandb_tags …`, which covers most of this. `train.py` stores the run's
identity and runtime fingerprint under the W&B config's `runner_cfg.research` and in
`<run_dir>/research_provenance.json`, and writes `locomotion_metrics.jsonl` (the finite loss, reward and
action-std stream the smoke gate reads). W&B records the commit automatically; check that it actually
appears on the run.
Do not copy this information into a second local state system.

**Source state**

- Every run launches from a **clean commit of `dev/agent/<group>`**, in that branch's worktree.
  `research_runs.py queue --group <group>` refuses anything else. It tags each launch
  `agent/<group>/<train|eval>-<n>`, pushes branch and tag, and runs jobs against the worktree's own
  `src` and `source/mbrl`. A worktree contains tracked files only: untracked or LFS-smudged assets
  (for example `assets/models/pretrain_rnn_ens.pt`) and `exp/` stay behind.
- `../rsl_rl_rwm` is shared by every worktree and is not part of the clean-commit check; the
  fingerprint records its HEAD. Do not change it while a comparison runs.
- Never rewrite or delete agent branches or launch tags: W&B runs refer to their commits.

**Reports**

- The report of a comparison is its **W&B Report**, built and refreshed by
  `research_runs.py publish-report` from W&B data with interactive native panels. There are no local
  Markdown reports; historical `reports/` stay untouched.
- Custom plots, when native panels cannot show something, come from `research/<group>/plots.py`.
  It reads data only through the W&B API and is attached to the report as a code artifact. See
  `.agents/skills/research-review/references/report-plots.md`.
- Every comparison is analysed in `research/<group>/analysis.ipynb`, copied from
  `.agents/skills/research-review/references/comparison_template.ipynb`. It reads W&B only, and
  `research_runs.py run-analysis` guards, executes and logs it as the `analysis` run. The report shows
  its figures ("Analysis figures"), and the first Analysis draft comes from its output. See
  `.agents/skills/research-review/references/analysis-notebook.md`.
- The report's Analysis and Decision sections belong to the researcher and survive every refresh.

**Comparability**

- Compared variants use the same budget, seeds and evaluation protocol. Any deliberate difference
  must be stated, including extra costs such as teacher pretraining.
- Videos recorded by the scoring evaluation are the scored evidence. Anything re-rendered later is
  illustrative and must be labelled as such.
- Failed and negative runs stay in W&B; never delete or rename them to tidy up a comparison.

**Where outputs go**

- Checkpoints, logs, evaluation tables and videos stay where the scripts already write them
  (`logs/rsl_rl/<experiment_name>/<timestamp>_<run name>/`, whose final checkpoint is `model_<iterations>.pt`).
- Reusable code never lives in ignored `reports/`, `logs/` or `outputs/` directories. Historical
  reports stay untouched.
- Do not hard-code machine paths or interpreters.

The W&B Report's Overview section holds the protocol summary (variants, seeds, budget,
evaluation). The committed hypothesis and approved protocol are scientific source records; mutable Report text is not protocol identity. See `.agents/skills/research-experiment/references/scientific-contracts.md`. Missing contracts on historical runs mean `legacy workflow`, not corruption.

## Where things live

| Tool | Holds |
|---|---|
| **Git** | source, configs, tests, reusable tooling, hypothesis, approved protocol versions, analysis code, minimal decision and provenance schema; one `dev/agent/<group>` branch and launch tags per comparison |
| **W&B** | runs, configs, metrics, groups, checkpoints/artifacts, videos, runtime fingerprints, analysis outputs, status/decision tags and Report |
| **Agent skills** | the workflow: interpret, implement, launch, monitor, evaluate, compare, finalize, all through the real scripts |
| **Obsidian** | informal thinking, brainstorming, hypotheses before formalization, interpretation notes and broader narrative (never runtime state) |
| **Confluence** | mature shared docs: architecture, final methods, deployment guidance (not experiment logs) |

## Validation

Run `python -m pytest tests/test_research_*.py -q` (plain Python: needs pyyaml, pandas, pyarrow, torch,
wandb, wandb-workspaces, nbformat, nbclient, ipykernel and matplotlib; no simulator) and
`pre-commit run --files <changed files>`. Isaac Lab-dependent code runs in the conda env `rwm`. Run
simulator smoke runs only when the change needs live evidence (`SMOKE=1 bash exp/run_all.sh` or one short
`train.py --max_iterations 2 --headless`). Never start training to "check" a change.

## Known limits

- `scripts/reinforcement_learning/model_based/train.py` (simulator-free offline policy training) is not
  launched, verified or published by `research_runs.py`.
- The smoke gate and `check-train` need a **fresh** run (one stream record per iteration,
  `0 … max_iterations - 1`). Resumed or chained variants (Finetune loading a Pretrain checkpoint) must
  declare `smoke_gate.enabled: false`.
- The final checkpoint of a fresh N-iteration run is `model_<N>.pt` (rsl_rl_rwm saves last + 1).
- There are no terrain levels or difficulty presets: `evaluation.levels` and `evaluation.difficulties`
  stay empty, and the `realized_*` standard metrics are not recorded.
- `tests/` is gitignored upstream; only `tests/test_research_*.py` and its fixture are tracked.
- Scored evaluation loads the checkpoint into the Play task's `OnPolicyRunner`; the dynamics model is not
  loaded, which is what `exp/eval_policy.py` does too.

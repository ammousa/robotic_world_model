# Scientific contracts (schema v1)

These are scientific source artifacts, not a runtime database. YAML mappings are hashed as canonical
JSON (sorted keys, ordered lists preserved), excluding their own ID; IDs are `h-`/`p-` plus 24 hex
characters. `validate-contract` prints the validated document with its computed ID; copy the
hypothesis ID into the protocol. An optional stored ID must match. Approval comes from the active
researcher (user by default; a separately delegated agent in
explicit unattended mode), never because YAML says “approved”. See
[research modes](../../../../docs/research_modes.md) for artifact-bound gate receipts and the
concise W&B decision dashboard. Commit the approved files before launch.
Archive executed amendments as `hypothesis.v1.yaml` / `protocol.v1.yaml`; preserve Git history and
prior run configs. Use a new group when the scientific question changes substantially.

## Hypothesis example

```yaml
schema_version: 1
group: rwm-ensemble
hypothesis_version: 1
title: Dynamics-model ensemble in RWM pretraining
claim: A 5-member dynamics ensemble yields a policy with lower velocity-tracking error than a single dynamics model, without more falls.
mechanism: Ensemble disagreement exposes model error, so the model-based update exploits the learned dynamics less.
changes: [agent.system_dynamics.ensemble_size 1 to 5]
held_fixed: [task, rewards, PPO hyperparameters, training budget, evaluation task and population]
inputs:
  deployable: [command, proprioception]
  privileged: []
gradient_ownership: The policy trains on its PPO objective; the dynamics model trains on its own prediction losses; no gradient crosses.
baselines: [single model]
primary_metric: {name: lin_vel_tracking_rmse, direction: lower, unit: m/s}
secondary_metrics: [fall_rate, mean_episode_return]
constraints: [fall rate increase at most 0.01]
minimum_meaningful_effect: {absolute: 0.02, unit: m/s}
evaluation_unit: One environment's first episode in a declared evaluation cell.
paired_factors: [seed, env_id, episode_id]
anticipated_confounders: [ensemble needs a smaller dynamics mini-batch, random streams diverge after reset]
support_if: [tracking reduction at least 0.02 m/s and fall constraint met]
contradict_if: [tracking worsens or fall constraint fails]
inconclusive_if: [low-N seed variability exceeds effect, incomplete matched evidence]
checkpoint_export_impact: None; the policy checkpoint and exports keep their inputs. The dynamics checkpoint gains ensemble members.
notes: ASSUMPTION thresholds and the smaller dynamics mini-batch for the ensemble; Gate A must approve them.
```

## Protocol example

```yaml
schema_version: 1
group: rwm-ensemble
hypothesis_id: h-<computed-by-validate-contract>
protocol_version: 1
project: rwm-explore
variants:
  single:
    task: Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0
    eval_task: Isaac-Velocity-Flat-Anymal-D-Play-v0
    overrides: []
    eval_overrides: []
  ensemble:
    task: Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0
    eval_task: Isaac-Velocity-Flat-Anymal-D-Play-v0
    overrides: [agent.system_dynamics.ensemble_size=5, agent.algorithm.system_dynamics_mini_batch_size=1500]
    eval_overrides: []
seeds: [1, 2, 3]
training_budget: {iterations: 600, num_envs: 4096}
compute_pool: [0, 1]
baseline_variants: [single]
checkpoint_rule: final
evaluation:
  population: First episode of each environment per evaluation seed.
  seeds: [41, 42]
  scenarios: []
  levels: []
  difficulties: []
  num_envs: 256
  num_episodes: 1
  episode_sampling: first_episode_per_env
  paired_factors: [seed, env_id, episode_id]
  primary_metric: {name: lin_vel_tracking_rmse, direction: lower, unit: m/s}
  secondary_metrics: [fall_rate, mean_episode_return]
  constraints: [fall rate increase at most 0.01]
  profile: general
minimum_meaningful_effect: {absolute: 0.02, unit: m/s}
cost_accounting: [environment steps, updates, wall time, GPU-hours]
smoke_gate:
  enabled: true
  iterations: 2
  auto_full: true
  log_roots: {single: logs/rsl_rl/anymal_d_flat, ensemble: logs/rsl_rl/anymal_d_flat}
retry_policy:
  max_attempts: 2
  allowed_reasons: [vm_died, infrastructure_interruption]
  selection: first_valid_identical_attempt
support_criteria: [tracking reduction at least 0.02 m/s and fall constraint met]
contradiction_criteria: [tracking worsens or fall constraint fails]
inconclusive_criteria: [low-N seed variability exceeds effect, incomplete matched evidence]
assumptions: [ASSUMPTION seeds and budget, ASSUMPTION seed-level pairing only; random schedules unverified]
```

These task names and overrides are illustrative; inspect the registered tasks and the agent configs
before proposing executable commands. The generic builder supports `rsl_rl` `train.py`/`play.py`
runs with final checkpoints (`model_<iterations>.pt`: the rsl_rl_rwm runners save the last
iteration plus one). This repository has no terrain levels or difficulty presets: leave
`evaluation.levels` and `evaluation.difficulties` empty (`eval-jobs` and `queue` refuse them), so
each evaluation cell is one evaluation seed. Scenarios that need a different entry point, such as
the simulator-free offline training in `scripts/reinforcement_learning/model_based/train.py`, are not
covered; do not pretend the generic builder supports them. `fixed_k_episodes_per_env` with positive
`k` is a valid scientific declaration but currently rejected for execution/publication. Never pool
arbitrary resets. Resumed or chained variants (for example Finetune loading a Pretrain checkpoint) do
not satisfy the fresh-run metric stream, so declare `smoke_gate.enabled: false` for them.

## Decision example

```yaml
schema_version: 1
group: promo-router
decision: extend
chosen_variants: []
rejected_variants: []
status: inconclusive
based_on_protocol: p-<executed-protocol-id>
based_on_report: https://wandb.ai/<entity>/<project>/reports/<report>
researcher_note: Researcher requested additional seeds before deciding what to keep.
```

Decision values: `keep|reject|modify|extend`; evidence status:
`supported|not_supported|inconclusive|invalid_experiment`. W&B Report holds the narrative.

## CLI and enforcement

- `validate-contract --kind hypothesis|protocol|decision --file PATH [--hypothesis PATH]` validates
  without GPU/network; protocol defaults to sibling `hypothesis.yaml`.
- `train-job --protocol PATH` checks exact ordered overrides, task, seed, budget, environment count,
  project and group. Jobs carry IDs, full protocol, cell, attempt and settings hash into run configs.
- `--attempt N --retry-reason vm_died --original-provenance ORIGINAL/research_provenance.json`
  creates a unique attempt name and binds the original cell/attempt, file digest, commit and stable
  resolved config hash. `--original-settings-id` remains readable for old planning scripts but is
  insufficient to launch a retry. Queue checks the original source before pinning and before each
  launch; the entry point checks resolved settings before training. `settings_identical` remains
  null until verified. GPU index, run labels/tags and output paths are excluded from scientific config
  identity; hyperparameters, architecture, data and losses remain included. Original hash v2 is
  required: missing/unsupported historical config identity requires a newly approved protocol,
  never an assumed-identical retry. A retry smoke is technical evidence, not proof of full-config
  equality. Retries require an approved infrastructure reason. NaNs are a scientific failure;
  changing a learning rate requires an amended approved protocol. Failed attempts stay visible.
- `eval-jobs --protocol PATH` validates evaluation settings and final checkpoint rule.
- `queue` automatically runs technical smoke before scientific training when `smoke_gate.enabled`
  and `auto_full` are true; omitting `--smoke-first` cannot bypass it. The flag remains an explicit
  assertion of that approved plan. Checkpoints, finite complete metrics, exception-free logs and
  matching protocol/cell/attempt/source provenance are required. Checkpoints must load safely on CPU
  (`weights_only=True`), contain nonempty model weights and have finite model/optimizer state;
  zero-byte, corrupt and non-finite checkpoints cannot open the gate. Retry smokes retain the attempt,
  GPU pin and interpreter. Failed or unverified smoke stops. Source changes during smoke or while
  jobs wait block pending launches. A smoke-only protocol (`auto_full: false`) permits smoke jobs,
  but full jobs require renewed Gate C approval and a committed protocol amendment enabling
  automatic full launch. No second approval is needed under an already approved automatic plan.
- `queue` requires new contracts and matching GPU pool/metadata; `--legacy-workflow` explicitly
  continues historical groups without contracts. The legacy flag cannot bypass new-contract jobs
  or a group with scientific source artifacts. Clean commit/branch/import/tag rules still apply.
- `check-eval [--protocol PATH]` verifies the full approved evaluation-seed matrix (× levels/difficulties where a repository defines them);
  embedded contracts enforce this even without the option. A requested subset is incomplete.
  `publish-eval --protocol PATH` independently requires every cell exactly once, the declared
  population/profile, clean commit and exact variant/training-seed/attempt/task identity. Primary
  and secondary metrics are exact keys, required and finite in summaries and recomputed scored
  aggregates. Optional `evaluation.required_metrics` adds constraint metrics without parsing
  free-text criteria. Custom scored metrics require `evaluation.metric_reducers`, each mapping an
  exact metric key to `{column: episode_column, reduction: mean|rms|rate}`. Reducers weight every
  declared evaluation unit equally; rate inputs must be binary. Unsupported reducers and missing
  custom declarations fail protocol validation before GPU use. `metric_columns` remains an optional
  consistent column alias, not a substitute for aggregation. Standard definitions cannot be
  overridden. The publisher and notebook apply the declared reducers to the attached scored rows. A diagnostic must actually be collected before it can
  be declared as scored evidence; the generic evaluator does not invent router diagnostics.
  Each environment ID `0..num_envs-1` contributes exactly once. Paired IDs use configured factors;
  missing factors, row/cell inconsistencies or duplicate units invalidate evidence.
- `publish-eval` links the exact finished, valid W&B training attempt, verifies its clean source
  commit, original checkpoint directory and training provenance sidecar, and compares checkpoint
  SHA-256 with bytes captured when the policy was loaded. Wrong explicit `--train-run-id` cannot
  override these checks. New-contract evidence with missing checkpoint provenance must be
  re-evaluated; it is not silently downgraded to legacy. Historical runs without contracts remain
  readable under legacy checks. Copied checkpoints without the original run-directory provenance
  are not supported by this local-checkpoint publisher.
- Analysis reads the immutable scored artifact attached to the chosen evaluation run via
  `run.logged_artifacts()`, including retry suffixes; never reconstruct an unversioned/latest alias.
  Keep all attempts visible, join on training seed and verified unit ID, require complete matches,
  and never infer pairing from order. Multiple evaluated attempts/versions require explicit
  reviewed `EVAL_RUN_IDS`, consistent with the retry policy; there is no automatic best-attempt selection.
- `check-train` exposes `research_provenance.json`; missing metadata means `legacy workflow`.

Runtime fingerprints are best-effort: package versions, Python, CUDA if torch is loaded, GPU/driver,
Isaac Lab and rsl_rl_rwm checkout commits (`-dirty` when uncommitted), environment file hashes and resolved config hash. Full configs stay in
W&B. Fingerprint schema/hash version 2 uses stable callable names, partial arguments and recursive
config values; unsupported objects produce a null hash with an error, never address-bearing repr.
Streaming file hashes avoid loading assets/lockfiles into memory. Asset comparison keys use stable
slots in the resolved asset list, not absolute checkout paths. An absent Isaac Lab or rsl_rl_rwm checkout is
recorded as unknown, never as the parent repository's commit. Missing fields are unknown, never equality evidence. Existing asset provenance remains the
asset authority. Fingerprints are run artifacts/metadata, not a dependency-management service.

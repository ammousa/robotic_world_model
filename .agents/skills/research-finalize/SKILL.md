---
name: research-finalize
description: >-
  After the researcher explicitly chooses which research candidate to keep, turn the experimental
  working tree into clean repository code: keep the chosen implementation and config as canonical,
  remove code, presets, task IDs and scripts that existed only for rejected variants or one-off runs,
  keep reusable tooling, update tests and docs, validate, and stop before committing. Use when the
  researcher says "keep X", "finalize X" or "clean up, we chose X". Do not use without an explicit
  decision, and never commit or push unless asked.
license: MIT
metadata:
  version: 0.1.0
---

# Research Finalize

**Decision authority:** read [research modes](../../../docs/research_modes.md). Attended is the
default (user decisions); explicitly authorized unattended mode routes Gates A–D to a separate
`researcher` via `research-decide`. Stage approval stops remain real gates; the orchestrator
obtains the active researcher's decision and continues approved work. Never self-approve.
Record actual gate receipts and refresh the concise W&B Researcher dashboard after each gate.

This skill covers the **finalize** stage of `frame → implement → code-review → experiment → evaluate → review → researcher-decision → finalize` ([`AGENTS.md`](../../../AGENTS.md)). It needs the researcher's explicit decision, for
example "Keep rwm-u-ens5. Remove the other variants." If the decision is ambiguous about what
survives, ask.

## 1. Inventory

Work in the comparison's worktree on `dev/agent/<group>`. The baseline is the branch's merge base
with the default branch (`master`) (`git merge-base master dev/agent/<group>`). Classify every change since it, using
`git diff <baseline>...`, `git status`, and a repository-wide search for each variant name:

| Class | Examples | Fate |
|---|---|---|
| Chosen implementation | policy/module/algorithm code, schema fields, the chosen preset and its task registration | keep; make canonical |
| Rejected-variant code | presets, task registrations, config branches, flags used only by rejected variants | remove |
| One-off experiment code | launch/eval snippets, hard-coded seeds/paths, debugging switches, temporary prints | remove |
| Reusable tooling | generic helpers, tests, checks that serve future work | keep (in tracked source/`scripts/`, never under `logs/`, `reports/`, `outputs/`) |
| Scientific record | `research/<group>/` contracts, protocol versions, analysis code and decision | retain the small scientific record; W&B holds rich outputs |
| Docs/tests | method docs, task lists, registry expectations, tests of removed code | update |

Show this inventory before deleting anything that is not obviously experiment-only.

## 2. Clean up

- **Canonical winner.** Keep it under a clear, permanent name. If the task ID or preset was named
  after the experiment, propose the permanent name. Renaming a task ID others may already use is
  the researcher's call.
- **Remove rejected variants completely:**
  - presets and exports in `agents/rsl_rl_ppo_cfg.py` and the task registrations in the anymal_d
    `__init__.py` (under `source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/`). Keep an
    env cfg or evaluation variant if a surviving task still uses it;
  - schema fields that only they used;
  - dead branches;
  - tests that only covered them.
- **Remove temporary code:**
  - debugging switches and temporary logging;
  - stale comments and commands;
  - workstation paths and interpreters.
- **Keep generic tooling**, and move any reusable code found in ignored output directories into
  tracked source.
- **Leave history alone.** Keep historical W&B runs, reports and logs as they are. Never delete or
  rename them. If docs referred to rejected variants, record briefly what was tried and link to
  the W&B group rather than deleting the history.
- **Record the decision in W&B and Git.** Preview with `--dry-run`, then apply:
  - `research_runs.py tag --project <p> --group <g> --variants <chosen> --add decision:chosen status:final`
  - `--variants <others…> --add decision:rejected` for the rest.

  This touches the variants' `train` and `eval` runs; identity tags and run names are never edited.
- **Record the decision in the W&B Report.** Run
  `research_runs.py publish-report --project <p> --group <g> --decision <file>`, where the file
  states the chosen candidate, the reason, and what was rejected. This replaces only the Decision
  section; the researcher's Analysis is kept.
- **Keep checkpoints loadable.** Checkpoints of the chosen variant must still load. If cleanup
  renames or moves keys, add a boundary migration at the load site or state the break. If the cleanup touched
  `../rsl_rl_rwm`, that sibling repository needs its own cleanup and commit.

## 3. Validate

Run these in order. None of them starts training.

1. The targeted tests for the chosen method, then the workflow tests
   (`python -m pytest tests/test_research_*.py -q`, plain Python, no simulator) and any affected
   tests. Isaac Lab-dependent code runs in the conda env `rwm`.
2. `pre-commit run --files <changed files>`.
3. No dead task IDs, imports or names: search for each removed variant and task ID.
4. The task registrations for the IDs you touched still import and register (the training and
   evaluation task ids named in the protocol).
5. The relevant CLIs still compose, for example `scripts/reinforcement_learning/rsl_rl/train.py --help`.
6. Where relevant, a saved checkpoint of the chosen run still loads (and exports, if the method
   has an export path). Run a simulator smoke test only if the cleanup touched live environment
   behaviour.

Report pre-existing failures as pre-existing.

## 4. Stop before committing the cleanup

Report:

```markdown
### Decision applied
### Retained
| File | Why |
### Removed
| File / symbol | Why |
### Reusable tooling kept
### Tests and checks
(command → pass / fail / skipped)
### Compatibility
checkpoints, task IDs, exports, CLI overrides
### Working tree
`git status --short` and `git diff --stat`
```

End with:

> Cleanup is ready for review on `dev/agent/<group>`. It is not committed, and nothing was merged into the default branch (`master`).

After an explicit separate cleanup-commit approval (from the user, or researcher within an
explicit mandate granting cleanup-commit authority), commit on the agent branch and push only
when authorized. Generic unattended finalize permission leaves cleanup uncommitted. Merging into
the default branch (`master`) stays the user's action in both modes.

## Gate D decision record

Only the researcher decides keep, reject, modify or extend. On explicit decision write minimal
`research/<group>/decision.yaml`: decision, chosen/rejected variants, evidence status, protocol ID,
report link and researcher note. Validate it; never invent the note or decision. Keep hypothesis,
protocol versions and analysis code. Negative/inconclusive decisions can retain no candidate;
cleanup only what the researcher authorized. Leave cleanup and decision artifact uncommitted for
review. Do not merge, rewrite branches or remove historical runs.

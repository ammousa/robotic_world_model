---
name: research-review
description: >-
  Analyze the evaluated runs of one research comparison in this repository and help the researcher
  decide: verify provenance and protocol, compare variants per seed and in aggregate with their
  denominators, surface trade-offs, confounders and cost, and name candidates supported by specific
  criteria, without choosing, integrating or launching anything. Use when the researcher asks to
  "review", "compare", "summarize" or "interpret" results. Do not use to train, evaluate or change
  code.
license: MIT
metadata:
  version: 0.3.0
---

# Research Review

**Decision authority:** read [research modes](../../../docs/research_modes.md). Attended is the
default (user decisions); explicitly authorized unattended mode routes Gates A–D to a separate
`researcher` via `research-decide`. Stage approval stops remain real gates; the orchestrator
obtains the active researcher's decision and continues approved work. Never self-approve.
Record actual gate receipts and refresh the concise W&B Researcher dashboard after each gate.

This skill covers the analysis that precedes the researcher's **decide** stage
([`AGENTS.md`](../../../AGENTS.md)). It changes no source, launches nothing and picks no winner; its
only writes are the comparison's analysis notebook and publishing its report (section 4). The
fairness rules are in the shared
[experiment checklist](../research-experiment/references/experiment-checklist.md).

## 1. Gather the evidence

Do this in the comparison's analysis notebook (section 4, step 1): the numbers you will report
come from cells that read W&B, not from values typed by hand.

- **W&B runs.** Read the group's runs with the W&B tools available to you, or its API, by job type:
  - `train` (never `smoke`) for budgets, configs, commits and curves;
  - `eval` for the scored results: the `eval/*` summary with denominators, the `eval/cells` table,
    and the `research` config.

  Skip runs tagged `status:superseded`. A `video` run tagged `replay` is illustrative only.
- **Local artifacts.** For each evaluation cell, read `evaluation_summary.json`,
  `evaluation_run.json` and `evaluation_results.{parquet,json}` under the output roots
  `research-evaluate` reported.
- **Training evidence.** Read the training checks (`check-train`), plus the queue event logs for
  failures and retries.

## 2. Verify before comparing

- **Protocol.** Re-run `research_runs.py check-eval` on the output roots, or reuse its verdict.
  Report incomplete cells, and exclude them rather than filling them.
- **Source.** Check that the commits agree, or state which runs used which.
- **Budget and configuration.** Diff the runs' W&B configs, not only the declared overrides. Flag
  every difference other than the variable each variant was meant to isolate.
- **Metric definitions.** Use the same profile everywhere. `survival_rate` differs between the
  `general` and `forward_course` profiles.

If verification fails, the review says so first, and every comparison it makes is labelled
provisional.

## 3. Analyze

Keep two questions separate when they both arise:

1. **Common budget.** Every variant at the same transitions, iterations or epochs.
2. **Latest capability.** Each variant's best available checkpoint, with its budget stated.

Then, for each variant:

- Report per seed first, then the aggregate. Use the declared equal-contribution episode sampling, per cell.
- Keep the denominators, for example "112/150 first episodes" rather than only "74.7 %".
- Break results down by scenario where effects differ (this repository has no terrain levels or
  difficulties). Never pool OOD scenarios into one number without also showing them separately.
- Show the spread across seeds. Do not claim a difference smaller than that spread, and say
  plainly when there is only one training seed.
- State cost: training budget, wall time if known, and extra costs such as RWM pretraining.

Look at videos when qualitative behaviour matters, or when a metric is surprising: sample frames
with `ffmpeg` and read them. Always say whether a video is **scored** or a **replay**.

## 4. Report

```markdown
### Comparison
| Variant | Success | Survival | Tracking | Cost | Notes |
|---|---|---|---|---|---|
(values with denominators and seed spread; the profile and population stated above the table)

### Seed effects, paired effects and hypothesis criteria
### Trade-offs
### Confounders / limitations
### Candidate(s) worth considering
Each with the specific criteria that support it and what it trades away. Several candidates, or none,
is a valid answer. No forced single winner, and no ranking beyond what the evidence supports. A recommendation remains advice until Gate D.

### Suggested next check
Only if the current evidence cannot support a decision, and as a proposal, never launched.
```

### Publishing the report (always a W&B Report)
The report is the comparison's **W&B Report**. Never write a local Markdown report.

1. **Analysis notebook (required).** Copy
   [references/comparison_template.ipynb](references/comparison_template.ipynb) to
   `research/<group>/analysis.ipynb` on the agent branch and follow
   [references/analysis-notebook.md](references/analysis-notebook.md). Set `GROUP` and `PROJECT`, run it
   interactively, add the comparison's own questions, and write your judgment in `NOTES` (no numbers:
   they come from the cells). Read `PROBLEMS` first; it is where section 2 shows up. Commit the
   notebook without outputs.
2. **Run it.** `research_runs.py run-analysis --project <project> --group <group> --notebook
   research/<group>/analysis.ipynb` (with `--dry-run` first) guards the code cells, executes the
   notebook from scratch, logs the `analysis` run with the executed notebook attached, and writes the
   Analysis draft to `logs/queue/<group>/analysis_draft.md`.
3. **Intro.** Write `logs/queue/<group>/intro.md`: hypothesis, the variants and the variable each
   isolates, budget, protocol. It and the draft are inputs to the report, not reports.
4. **Custom plots, only if the notebook's figures are not enough.** `research/<group>/plots.py`, following
   [references/report-plots.md](references/report-plots.md).
5. **Scientific review (required).** Follow [references/scientific-review.md](references/scientific-review.md).
   Prepare the final publication evidence through `publish-report --dry-run` with exact intended
   inputs. The active agent must author the review from that evidence; the `scientific-review`
   command only prepares a scaffold or validates supplied prose. Record generation attribution,
   validate the completed review, and obtain the active
   researcher's approval of its exact ID. Separate measured results, interpretation and causal
   hypotheses. Recommend a decision with evidence and limitations; do not make Gate D's choice.
   Propose up to three prioritised, discriminating experiments with exact changes, controls,
   metrics, acceptance criteria and estimated cost or unknown. If unavailable, explicitly use
   `--evidence-only "<reason>"` and leave interpretation pending; never fabricate approval.
6. **Check.** `research_runs.py publish-report --project <project> --group <group> --intro … --analysis …
   [--review … --review-approval … | --evidence-only "<reason>"]
   [--plots research/<group>/plots.py] [--train-metrics …] [--eval-metrics …] --dry-run` lists the runs,
   panels and the analysis run's figures without publication. Custom plot code is checked but not executed.
7. **Publish.** Run the same command without `--dry-run`, and link the report URL in your answer.
   The report uses nine scientific-review sections, with an **Analysis figures** subsection from the latest `analysis` run. Refreshing later keeps
   the URL and the researcher-edited Analysis and Decision sections. After the researcher edits the
   analysis, or new runs arrive, re-run step 2 and refresh.

Do not tag decisions here.

End with:

> Evidence review is complete; Gate D belongs to the active researcher. No model source changed and no experiment was launched; analysis code and the W&B report were updated.

Cleanup starts only after the researcher states a decision, via `research-finalize`.

## Evidence and outcomes

Never apply the old “every seed on one side” robustness rule. Show raw per-training-seed outcomes,
then matched differences on the same verified `eval_unit_id`, stratified by training seed/scenario.
Report mean/median effects, seed variation, directional proportions and ties, denominators, and an
uncertainty interval only with a justified sampling unit and enough independent replicates. Do not
count environments as independent training seeds. With few training seeds label **low-N**, show raw
effects and avoid precise asymptotic claims. Compare against predeclared minimum meaningful effect
and every constraint. Inspect mechanism diagnostics (e.g. dynamics-model prediction error or ensemble disagreement).

Classify each predeclared criterion `met`, `not_met`, `inconclusive`, or
`invalid_due_to_protocol_or_provenance`, with evidence. Supported, not supported, inconclusive and
invalid experiment are all legitimate evidence outcomes, not keep/reject decisions. Never choose a
winner. Keep failed/negative evidence visible. Additional seeds require a protocol amendment and
Gate C; a changed claim returns to frame/Gate A. No follow-up is launched during review.

Compare runtime fingerprints (Python/packages, simulator/Lab, CUDA/driver/GPU, lock hashes and
assets where available), resolved configs and commits; flag material differences before inference.
Expose environment steps, updates, wall time, GPU-hours, parameter count, inference latency/memory,
and pretraining costs where available; mark unavailable quantities rather than inventing
numbers. Per-seed cost tables/plots use only the selected training attempt for each scored cell.
Report total comparison spend separately across every attempt, including failed and superseded
runs. Show known subtotals and unknown-attempt counts; missing cost is not zero. Same iterations
need not mean same cost. Custom diagnostics use protocol-declared reducers and paired unit values.

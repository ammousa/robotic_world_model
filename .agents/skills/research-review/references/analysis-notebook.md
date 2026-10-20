# The comparison's analysis notebook

Every comparison is analysed in **one notebook**, `research/<group>/analysis.ipynb` on the agent
branch. It is the working surface for [`research-review`](../SKILL.md): each number in the report's
Analysis comes from a cell that computed it from W&B, and the exact notebook stays attached to the
report for anyone to re-run.

## Start from the template

Copy [comparison_template.ipynb](comparison_template.ipynb), set `GROUP` and `PROJECT` in the
parameters cell, and run it. The template applies the workflow's conventions, so analyses stay
consistent across comparisons:

| Block | What it does |
|---|---|
| Provenance | loads the group's runs, skips `status:superseded`, takes the variant from `config.runner_cfg.research` / `config.train_cfg.research` / `config.research`, and lists budget, commit, seed, checkpoint, protocol and population differences in `PROBLEMS` |
| Scored episodes | reads immutable artifacts attached to the selected evaluation runs, verifies protocol population and paired IDs, and computes metrics per training seed, slice and pooled, with counts |
| Effects | raw per-seed effects; mean/median matched episode differences, directional fractions and separately labelled pooled differences; low-N, no automatic interval or winner rule |
| Cost | training wall time from the runtime, plus the `Perf/*` history |
| Videos | counts the scored videos per evaluation run (it does not judge them) |
| Analysis draft | renders the `### Comparison` … `### Suggested next check` Markdown of the review skill from the tables above, with your judgment in `NOTES` |
| Keep | logs the kept figures and tables, only under `run-analysis` |

Adapt it freely for the comparison: older runs without a `research` block need their own
`train_variant_seed`, and other task profiles need other `METRICS`. Keep a figure or table with
`keep_fig(slug, fig)` / `keep_table(slug, df)`.

## Contract

- **W&B is the only data source.** The same rules as [report-plots.md](report-plots.md): read through
  `wandb.Api()` (runs, `run.history`, `api.artifact(...).get(...).get_dataframe()`); never local
  files, `logs/`, `data/`, `mbrl`/`isaaclab`, `subprocess`, shell or file magics, `download`.
  `%matplotlib` and `%config` are the only magics allowed.
- **Judgment goes in `NOTES`, numbers do not.** The draft cell refuses percentages, fractions and
  decimals in `NOTES`; refer to a table or figure instead. This keeps every number traceable to a cell.
- **Interactive runs write nothing to W&B.** Explore freely; logging happens only under `run-analysis`.
- **Commit the notebook without outputs** (clear them first) on the agent branch. The executed copy,
  outputs included, is attached to W&B.

## Publish

```bash
python scripts/research_runs.py run-analysis --project <project> --group <group> \
    --notebook research/<group>/analysis.ipynb --dry-run     # guard + execute with logging off
python scripts/research_runs.py run-analysis --project <project> --group <group> \
    --notebook research/<group>/analysis.ipynb               # same, then log and attach
```

`run-analysis`:

1. applies the report-plot guard to every code cell and requires `GROUP` and `PROJECT` to equal the
   command line (a notebook copied from another comparison cannot log into the wrong one);
2. executes the notebook from scratch in a fresh kernel in an empty directory;
3. through the notebook's last cell, logs an `analysis` run `<group>__analysis` in the group with the
   kept figures (`analysis/fig/*`), tables (`analysis/table/*`) and draft, tagged `dirty` if the tree is;
4. attaches the executed notebook to that run as the `analysis-notebook-<group>` code artifact and tags
   the previous `analysis` run `status:superseded`;
5. writes the Analysis draft to `logs/queue/<group>/analysis_draft.md` (or `--draft-out`), the input of
   `publish-report --analysis`.

A failing notebook supersedes and attaches nothing. `publish-report` then adds an **Analysis figures**
section for the latest current `analysis` run; the Analysis section stays the researcher's after the
first publication.

## Attempts and immutable evidence

The `attempts` table retains every training attempt and its runtime cost. Scientific tables use only
training attempts explicitly linked by selected evaluation runs. `BASELINE` defaults to the single
approved protocol baseline; multiple baselines require an explicit choice. If multiple evaluations
exist for one training cell, set `EVAL_RUN_IDS` to the reviewed run IDs under the approved retry policy;
never choose by score. The notebook refuses ambiguous selections and mismatched identities.

Each evaluation must have exactly one matching `eval-episodes` artifact attached by
`run.logged_artifacts()`. Its immutable `:vN` version and digest are recorded in `evidence_artifacts`
and the analysis run config. This applies to historical evaluations too: missing attached evidence
stops rather than following a mutable `:latest` alias. Re-publishing cannot change a selected run's
input artifact version. Contracted artifacts must contain the full evaluation cell/population matrix,
first episodes only, finite inputs and correctly derived paired IDs.

Tracking rows contain per-episode RMSE. `lin_vel_episode_rmse` / `ang_vel_episode_rmse` in the
`paired_effects` table describe matched episode-RMSE differences (variant minus baseline). Their
mean/median and directional fractions refer to individual episodes. `aggregate_metric` and
`aggregate_difference` separately report the difference of pooled RMSEs over those same units.
The latter is not the mean of episode differences; compare the predeclared practical threshold to
the metric it names. Incomplete or duplicate joins produce no paired statistics. No winner is selected.


## Metric reducers and cost populations

The template includes protocol-declared custom metrics in per-seed and matched-unit tables using
`evaluation.metric_reducers`; declared primary metrics appear in the draft. Missing/non-finite scored
columns stop analysis. Episode values and aggregate RMS effects remain separate estimands.

`cost_selected_seeds` and the cost plots describe the exact selected attempt per scored seed.
`cost_attempts` retains all train attempts, including failed and superseded runs. `cost_all_attempt_spend`
reports their known wall-time/GPU-hour/step/update/pretraining subtotals, unknown-attempt counts and
an overall total only when every attempt has a recorded value. Unknown spend never becomes zero.
Selected cost summaries include counts so sparse measurements are visible.

## Scientific review hand-off

The template also emits a versioned scientific evidence record. The CLI logs it and writes
`scientific_evidence.json` beside the draft. The approved Git hypothesis, when available, is
supplied at the CLI boundary; scientific measurements still come from W&B. This record adds
exploratory training-seed intervals where eligible, bounded sampled training diagnostics,
resolved configuration differences and matched performance/resource changes. It does not generate
interpretations or choose a method. Continue through
[scientific-review.md](scientific-review.md) before report publication, or explicitly record the
evidence-only fallback. Existing notebooks without the new evidence cell still run; their reports
mark structured evidence/interpretation unavailable until upgraded.

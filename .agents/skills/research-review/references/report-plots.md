# Custom report plots

The comparison's analysis notebook ([analysis-notebook.md](analysis-notebook.md)) is the default place
for figures. Use a `plots.py` only for a figure that should be rebuilt on every `publish-report`,
under the same rules.

The W&B Report is built from native, interactive panels. A custom plot is only for something those
panels cannot show, such as a per-seed distribution of episode outcomes or a cost-versus-success
scatter with seed spread.

## Contract

- **One script per comparison.** It is committed on the comparison's agent branch at
  `research/<group>/plots.py`.
- **One entry point.**

  ```python
  def make_figures(api, project, group, entity) -> dict[str, Figure]
  ```

  `api` is a `wandb.Api()`. Keys are slugs; values are matplotlib or plotly figures.
- **W&B is the only data source.** Read runs, histories, summaries, logged tables and artifacts
  through `api`. Never read `logs/`, `reports/`, `outputs/` or `data/`, local Parquet/JSON/CSV
  files, or downloaded artifact files. Don't import `mbrl` or `isaaclab`.

  `publish-report` enforces this two ways:
  - it rejects scripts that call `open`, `read_*`, `load`, `glob`, `download` and similar, import
    `mbrl`/`isaaclab`/`subprocess`/`shutil`/`glob`, or mention local data paths;
  - it runs the script from a copy inside an empty temporary directory.
- **Attached code.** The figures are logged to the comparison's `report` run as `custom/<key>`. The
  script is attached to that run as the `report-code-<group>` code artifact, and the report links it.
- **Rebuilding from the attached code.** Anyone can regenerate the figures from that artifact with
  W&B access alone. They do not need the repository's local outputs.

## Where the data is

| Data | How to read it |
|---|---|
| Training runs | `api.runs(f"{entity}/{project}", filters={"group": group, "jobType": "train"})`. Skip runs whose tags contain `status:superseded` |
| Training curves | `run.history(keys=[...], samples=…)` (sampled), or `run.scan_history(keys=[...])` (every row) |
| Run identity | `(run.config.get("runner_cfg") or run.config.get("train_cfg"))["research"]` (rsl_rl_rwm stores the runner config under `runner_cfg`, newer rsl_rl under `train_cfg`; variant, seed, task, branch, commit, launch tag); tags `variant:` / `seed:` |
| Evaluation runs | `filters={"group": group, "jobType": "eval"}`; `run.config["research"]` holds protocol and `train_run_id` |
| Pooled evaluation metrics | `run.summary["eval/<metric>"]`, each with `…/episodes` (slice keys such as `eval/L4/<metric>` exist only when the protocol declares levels, which this repository does not) |
| Scored episodes | The exact version from the chosen evaluation run's `run.logged_artifacts()`, then `artifact.get("episodes").get_dataframe()`. One row per declared scored episode, with `cell` and `eval_unit_id` when paired; use the exact attempt artifact, not an unexamined `latest` |

## Minimal example

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_figures(api, project, group, entity):
    evals = [r for r in api.runs(f"{entity}/{project}", filters={"group": group, "jobType": "eval"})
             if "status:superseded" not in r.tags]
    fig, ax = plt.subplots(figsize=(6, 3))
    for run in evals:
        research = run.config["research"]
        name = run.name.removesuffix("__eval")
        artifacts = [a for a in run.logged_artifacts() if a.type == "eval-episodes"
                     and a.name.rsplit("/", 1)[-1].split(":")[0] == f"eval-episodes-{name}"]
        if len(artifacts) != 1:
            raise ValueError(f"missing or ambiguous scored evidence for {run.name}")
        rows = artifacts[0].get("episodes").get_dataframe()
        falls = rows.groupby("cell")["fell"].mean()
        ax.plot(falls.index, falls.values, marker="o", label=name)
    ax.set_ylabel("fall rate (first episodes)")
    ax.legend(fontsize=7)
    return {"fall_rate_per_cell": fig}
```

Check a script with `publish-report … --plots research/<group>/plots.py --dry-run`. That statically checks the
script and lists the report inputs; it does not execute custom Python in the authenticated host.
Actual figure generation runs before publication writes, once the exact source is covered by
review approval or an explicit evidence-only fallback. The temporary directory is not a security
sandbox.

## Protocol-aware evidence

Read `research.protocol_id`, `hypothesis_id`, `protocol_version`, `cell_id`, `attempt` and runtime
fingerprints. Never combine protocol versions or sampling policies. Match units only on verified
`eval_unit_id` and training seed; verify complete, unique joins before reporting paired effects.
No “every seed on one side” winner rule. Report raw seed effects, directional fractions, meaningful
effect/constraints and low-N limitations. Select attempts explicitly under the approved retry policy;
never take a prettier attempt or a mutable latest artifact without checking identity.

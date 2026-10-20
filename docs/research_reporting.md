# Scientific research reporting

*Written in the isaac_wm repository; Go1/Go2W references are historical. The `robot_wm` domain, `make lint`/`make test`, the local `logs/queue/scientific-review-demo/` and `logs/metric-resolution/` previews and `tests/fixtures/` do not exist here: use `python -m pytest tests/test_research_*.py -q` and `pre-commit run --files <changed files>`.*

The research workflow keeps deterministic measurement separate from scientific interpretation and
the active researcher's decision. Training and experiment execution are unchanged.

## Audit and implementation choices

| Existing capability | Reused implementation | Upgrade |
|---|---|---|
| W&B Report creation and refresh | `scripts/research_runs.py`: `publish_report`, `build_report_blocks` | Nine scientific-review sections; native interactive panels, figures, stable URLs |
| Researcher edits | `human_sections`; legacy Analysis/Decision sections | Protected subsections preserve blocks, nested headings and empty sections across migration and repeated refresh |
| Verified scored episodes | Comparison notebook: exact training attempts, immutable artifacts, first episodes, matched unit IDs, finite reducers | Versioned evidence projection includes those tables rather than reimplementing their reducers |
| Raw seed effects and paired episode differences | Notebook `effects`, `paired_effects` | Training-seed variability and eligible exploratory intervals; no environment-level pseudoreplication |
| Cost accounting | Selected attempts and all-attempt spend notebook tables | Matched performance/resource deltas; missing GPU-hours/parameters/latency stay unknown |
| Hypothesis criteria | Protocol contracts and notebook criteria table | Structured reviewer assesses every predeclared criterion with evidence citations |
| Gate decisions | `src/research_workflow/governance.py` and `research-decide` | Dashboard retained; review approval never becomes Gate D or compute authorization |
| Custom figures and analysis artifacts | `plots.py`, `run-analysis`, notebook code artifacts | Existing entry points/media keys retained; evidence and review artifacts added |

Previously, no versioned review schema or evidence-bound narrative approval existed. Training
patterns were shown as plots without a reusable diagnostic record; statistical intervals and
explicit main sections for interpretation, limitations and proposed follow-ups were missing.

The new `src/research_workflow/scientific_review.py` owns pure evidence construction, sampled
history diagnostics, statistical summaries, version/hash validation and review rendering. There
are no LLM dependencies or hidden model calls. A reviewing agent uses `research-review` to interpret
the actual evidence, with the active researcher approving the exact narrative. Machine checks
establish traceability and content identity, not logical entailment or causal validity.

## Interfaces

Existing `run-analysis --project … --group … --notebook …` and
`publish-report --project … --group … --intro … --analysis …` interfaces remain supported.
The template emits a scientific evidence marker; the runner attaches the evidence and writes
`scientific_evidence.json` beside the Analysis draft. Legacy notebooks still run without it,
with an explicit structured-evidence/pending-interpretation message in the report.

`scientific-review --evidence … [--review …] [--approval …] [--out …]` prepares an incomplete
review skeleton or validates a completed review. Each empirical claim references evidence leaf
paths. Numeric prose uses evidence substitution tokens. Proposed experiment quantities remain
explicitly labelled proposals. The review includes a hypothesis verdict, advisory decision,
criterion assessments and at most three prioritised follow-ups.

`publish-report --review … --review-approval … --dry-run` constructs a read-only preview of the
layout and preservation boundaries. Actual approval binds the exact review hash. Approved reviews
are reusable only for unchanged evidence; changed run/configuration/summary snapshots require
reanalysis and a newly approved narrative. Agent approvals require a current, explicitly delegated
unattended workflow, not a historical dashboard snapshot. `--evidence-only "reason"` documents the
fallback when review is unavailable. No review is silently promoted to approved.

The nine main sections are executive summary, research questions, experimental design,
comparative/statistical results, training dynamics/failure analysis, performance/cost,
interpretation/limitations, prioritised next experiments and researcher decision/rationale.
Generated content is labelled; protected Analysis/Decision subsections remain researcher-owned.
The renderer rebuilds sections instead of appending. Supporting tables and figures retain their
existing W&B media keys.

Detailed schema, uncertainty assumptions, approval format and workflow instructions live in
[the scientific-review skill reference](../.agents/skills/research-review/references/scientific-review.md).

## Validation and demonstration

Automated fixtures cover replicated comparisons, single-seed pilots, incomplete experiments,
missing histories, sampled oscillation/collapse flags, cost unknowns, review citations and hashes,
stale evidence, actual approval boundaries, URL refresh, nested human edits, repeated publication
and unchanged command interfaces. Intervals use training-seed differences and are checked against
an independent Student-t calculation. Those review scenarios are synthetic; the metric-resolution
regression fixture below additionally uses an actual Go2W API extract.

Local generated previews were under `logs/queue/scientific-review-demo/` in that repository (ignored scratch, not
scientific records). `index.html` links the three synthetic scenarios; their JSON includes the
structured evidence and an unapproved review proposal.

A read-only preview of the existing `go1-locomotion/unattended-actor-width-pilot` W&B group found
four training runs, four evaluations, an analysis run, an existing Report URL and protected
Analysis/Decision text. The preview preserved that URL and those sections without saving.
Strict dry-run reanalysis of this historical group stopped on a training/evaluation source-commit
mismatch. This upgrade does not reinterpret or repair historical evidence automatically. The
existing analysis tables remain visible in the Report preview, but are not relabelled as newly
verified scientific evidence. No live Report, run, tag or artifact was written during validation.

Repository-wide `make lint` currently reports 41 unrelated pre-existing violations.
`make test` stops at collection because `tests/test_marker_video_wrapper.py` imports the missing
`ilab.envs.wrappers.marker_video_wrapper`, a documented repository drift. The 247 reporting/workflow regression tests pass, as do changed-file lint/format checks; these failures are not hidden by changing
unrelated files or weakening checks.

## Limits

- Confidence intervals are exploratory, assume independent training seeds and approximately
  normal differences, require at least five pairs, and are suppressed by reported provenance or
  comparability problems. Five is a display minimum, not a power claim. Multiple comparisons are
  not corrected; statistical exclusion of zero does not establish practical value.
- Sampled histories can miss spikes and cannot prove convergence or establish failure causes.
  Sample efficiency is unavailable without matched environment-step histories and a declared target.
- Missing controls, failed scored-artifact validation or incomplete evidence are not fabricated.
  A pending evidence-only report remains possible when strict notebook analysis fails.
- Citation checks cannot detect every false paraphrase, spelled-out quantity or causal assertion;
  actual researcher review remains necessary. The CLI never supplies approval on its own.
- Only protected Analysis/Decision subsections preserve human edits; other sections are regenerated.
  W&B report runs/artifact versions may be created on actual publication, while report URL and
  section structure stay stable. No training, push or live publication is part of this change.

## Publication-boundary audit

Confirmed defects and focused repairs:

- Publication previously allowed missing review approval without an explicit fallback. Actual
  publication now fails before writes unless approval is valid or a nonblank evidence-only reason
  is supplied; the reason is recorded. Previews remain available while policy is blocked.
- Numeric-only run summaries omitted statuses, media/table bindings and artifact changes. Full
  summaries and logged/used artifact versions/digests now participate. The final dry-run evidence
  also binds the selected analysis/code/tables, current hypothesis and report prose/plot inputs.
  Notebook/helper bytes identify uncommitted executable changes. Artifact enumeration failure
  fails closed; unavailable legacy capabilities remain explicit, not silently asserted as verified.
- Sorting sections by every H1 could discard researcher subheadings or entire sections. Generated
  sections now retain their own order without reinterpreting human blocks. Analysis, Decision and
  Progress, nested headings, collapsible raw legacy anchors and intentional empty sections survive.
- Dry-run previously executed arbitrary plot code with a writable API. Report previews now only
  statically inspect custom source. Notebook dry-runs execute in a disposable kernel with W&B SDK
  writes and ordinary API mutation capabilities blocked; inherited run identity/hypothesis is cleared.
- The CLI prepared a scaffold and did not generate scientific interpretation. Scaffold provenance
  is now explicit and cannot validate as a completed review. The active agent authors the review;
  the CLI validates and renders it. A portable, substantive agent-authored example based on the
  saved Go2W follow-up evidence was in
  `tests/fixtures/scientific_review/agent_authored_heightmap_review.json` (isaac_wm only). It is unapproved, marks its
  historical/excerpt scope, preserves the published negative outcome and proposes only independent
  verification of saved arrays. No live report is changed by this audit.

The preview's `scientific_evidence` is the final review input. Using only the earlier run-analysis
sidecar cannot bind artifact identities that did not yet exist. Changing rendering inputs after
review invalidates that approval. Approved publication and evidence-only fallback are exclusive.

Remaining limits: authorship and approval records are attributed, not cryptographically
authenticated. Validation checks traceability, not entailment or sound causal inference. Python
capability guards are for trusted code; arbitrary network calls, private SDK entry points or
intentional guard bypass require a real external sandbox to prevent. Section ownership is limited
to the named protected anchors; reserved generated boundary headings are ambiguous if reused as
peer headings inside researcher text. Other generated sections are regenerated. Transitive code
outside the notebook/trusted workflow helpers and dependency-runtime records is not fully sealed.
Direct W&B/GraphQL edits outside these commands are outside repository policy enforcement.

Audit validation: **290 relevant tests passed**, covering every `test_research_*.py` module,
scientific-review/publication boundaries and the existing report tests. Changed-file lint and
format checks, `git diff --check` and skill validation passed. Repository-wide `make lint` still
reports the 41 inherited issues; `make test` still stops at the documented missing marker-video
module. No training, live report updates, commits or pushes were performed during this audit.

## Shared metric resolution and preflight

`src/research_workflow/metric_resolution.py` supplies the same `MetricResolver` plan and
observation cache to `report_inventory`, native panels, training diagnostics, notebook comparisons
and scientific evidence. Selection follows explicit protocol `reporting.metrics`, registered domain
preferences, then numeric W&B history discovery when preferences have no observed data. Existing
CLI metric flags request required keys; omitted flags select automatically. There is no RL metric
fallback for world-model runs. Existing RL registrations remain available when actually logged.

For example, an optional block in a protocol can declare:

```yaml
reporting:
  groupby: [config.research.variant]
  metrics:
    - name: train/loss
      scope: train
      required: true
      axis: epoch
      direction: lower
      unit: normalized squared state error
      reduction: logged_scalar
      aggregation: per_run
      comparison: loss
    - name: val/id/rollout_auc
      scope: train
      required: false
      axis: epoch
      direction: lower
      unit: declared RMSE-horizon AUC
      reduction: logged_scalar
      aggregation: mean_seed
      comparison: evaluation
  compatibility:
    dataset: immutable dataset and split identity
    loss_definition: exact objective, targets, normalization and weights
    evaluation: exact scorer and sampling settings
```

These are examples of declarations, not assertions that every loss is lower-is-better or has those
units. Unknown discovered metrics have unspecified semantics and per-run display. Explicit
registrations supply semantics; spelling/substrings never supply directions or units. The existing
profile-aware scored reducers remain authoritative, including the two `survival_rate` definitions.

Preflight validates grouping values and finite joint axis/value observations. An explicit axis
never silently changes to row order or a different counter. Coverage records both logged-key
availability and finite, plottable observations, by variant and successful attempt. Booleans,
numeric strings and nonfinite values are not numeric evidence. Missing required metrics block
publication before writes; optional missing metrics are omitted. The JSON preview lists omitted
keys, missing run IDs/reasons, exact panel IDs, axes, grouping, semantics, compatibility cohorts,
attempt roles, coverage, observation fingerprints, errors and warnings.

Only finished, nonfailed/noninvalid/nonsuperseded attempts can enter seed aggregates. Duplicate or
missing seed identities disable aggregation. Protocol identity, declared and resolved dataset,
normalization, evaluation settings and (for objectives) loss definitions partition cohorts. Missing
context permits diagnostic plots, not scientific contrasts. Actual scoring attempts must link to
successful training runs with matching identity. Incompatible raw effects are retained under
`incompatible_effects_diagnostic_only`; their scientific means, differences and intervals are
suppressed. Seed effects cannot pool multiple compatibility cohorts. Failed attempts retain their
own curves and all-attempt spend accounting. Historical custom analysis figures are retained as
outputs requiring compatibility verification, rather than being certified by panel availability.

The primary read-only validation used `oxdynamics/robot-wm/go2w-terrain-forecast` and preserved the
existing report URL plus Analysis, Decision and Progress. Its 35 training attempts recorded
`train/loss` in **32/35** cases, but seven reference runs logged only `NaN`: finite loss coverage is
**25/35**. Both counts are displayed. Validation RMSE and rollout AUC have finite coverage **32/35**.
No empty or nonexistent reward panels are emitted. The regression fixture is an actual API extract
with first/last sampled observations and relevant config fields; geometry arrays and narrative
contract fields are omitted. The full local dry-run preview was
`logs/metric-resolution/dry-run-preview.json` (isaac_wm). No report was saved or pushed.

Remaining limits: history coverage is bounded sampling, not proof of complete streams or
convergence. Metadata identities and declarations cannot prove the physical equivalence of two
datasets or scorers; missing context fails conservatively. Custom figures/code are not parsed to
validate their own arbitrary comparisons. The scored-episode notebook retains its RL episode
schema; world-model analysis must supply its own verified scored tables while reusing this resolver.
Recorded-key counts do not imply finite observations. Previously approved evidence must be reviewed
again when resolution, observations, semantics or source inputs change.

Metric-resolution validation: **318 relevant tests passed** across all research workflow modules,
scientific review, publication boundaries, metric resolution and the existing report suite.
Changed-file lint/format and `git diff --check` passed. Repository-wide lint and test collection
retain the documented 41 lint issues and missing marker-video module. The live report changed
during read-only validation; the preview was regenerated from its latest saved view
(`2026-10-19T09:39:05Z`). No live publication calls or Git pushes were made.

### Authorized live correction

On 2026-10-19 the user subsequently authorized updating the existing terrain-forecast report.
The live SDK revealed two additional boundary cases: nested `SummarySubDict` values are dict-like
but raise `KeyError` on missing attributes, and world-model scored requirements do not automatically
map to RL-style `eval/…` scalar keys. Summary normalization now retains nested values/nonfinite
fingerprint distinctions, and the explicit robot-wm scalar profile maps the terrain scorer's actual
keys. Unmapped scored requirements stay scientific evidence requirements; explicitly required
report scalar keys still fail closed. Both cases have regression tests.

The existing report `VmlldzoxODEzODg4OA==` was refreshed through `publish_report` with an explicit
metric-correction/evidence-only reason. No new scientific verdict was approved. Saved content was
read back and checked: **64 populated metric panels**, zero obsolete reward panels, epoch axes,
validated `config.research.variant` identities, loss coverage 32/35 recorded and 25/35 finite, and
unchanged Progress/Analysis/Decision content. The existing URL was retained. **320 relevant tests
passed**; changed-source lint and diff checks passed. No Git push was performed. Local receipts were
under `logs/metric-resolution/publish/` (isaac_wm), including `saved-report-verification.json`.

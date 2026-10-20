# Structured scientific review

Scientific review is required after deterministic analysis and before publication. It is a proposal
for the active researcher, not Gate D and not authorization to run follow-ups. The CLI makes no LLM
calls: the reviewing agent supplies interpretation, deterministic Python supplies measurements.

## Evidence and calculations

The template emits a versioned, content-addressed `scientific_evidence.json`. `run-analysis` writes
it beside the draft and attaches it to the analysis run as `scientific-evidence-<group>`. It includes
exact run IDs/URLs, full resolved configs, checkpoints/commits/protocols, immutable scored-artifact
references, verified per-seed/paired tables, criteria, selected-attempt costs and all-attempt spend.
The approved Git hypothesis is supplied by the CLI boundary when present; interactive use without
it marks the claim unavailable. Empirical values still come exclusively from W&B.

`src/research_workflow/scientific_review.py` computes descriptive effect sizes and variability.
An exploratory 95% Student-t interval uses differences across independent training seeds, never
individual environments as training replicates. It is displayed only with at least five seed
pairs and no reported comparability/provenance problems. This display minimum is not a power
calculation. Check independence, approximate normality, matching, missing seeds and selection bias;
no multiplicity correction or confirmatory significance claim is supplied. Pilots retain raw
outcomes without intervals. Practical importance comes from the approved meaningful-effect
threshold and every constraint, not an interval excluding zero.

Histories are bounded W&B samples. Plateau, oscillation, dispersion/outlier and policy-standard-
deviation collapse indicators are inspection prompts, not convergence certificates or causal
findings. Sample efficiency requires matched environment-step histories and a predeclared target;
when these are missing, say unavailable. Compare metric differences with measured wall time,
GPU-hours, parameter count and inference resources on the same selected seeds. Retry/pretraining
costs remain separate; null is unknown, never zero.

## Prepare and critically review

First run `publish-report` with the intended intro, Analysis draft, metric overrides and plot source,
plus `--dry-run`. This is the final evidence preparation step: it binds the selected analysis/code
and table/artifact identities, current Git hypothesis and report inputs. Save the preview's
`scientific_evidence` object as `publication_evidence.json`; use that file below. The run-analysis
sidecar precedes uploaded notebook/artifact identities and is not the final publication fingerprint.
For example, extract the object at the CLI boundary with
`python -c 'import json; from pathlib import Path; p=Path("logs/queue/<group>"); d=json.loads((p/"preview.json").read_text()); (p/"publication_evidence.json").write_text(json.dumps(d["scientific_evidence"], indent=2))'`.
No data files are read from inside the analysis notebook.

```bash
python scripts/research_runs.py scientific-review \
  --evidence logs/queue/<group>/publication_evidence.json \
  --out logs/queue/<group>/scientific_review.json
```

This prepares an intentionally incomplete schema-v1 skeleton marked `generation.kind: template`.
**The CLI does not generate an interpretation.** The active reviewing agent must now read the actual
evidence and author the executive verdict, interpretation, limitations, criterion assessments and
follow-ups. Do not copy generic scaffold prose or treat template preparation as review completion.
Record `generation: {kind: agent, agent: <actual author>, evidence_id: <exact input ID>}` (or
`kind: researcher` for researcher-authored text). Validation rejects template provenance even if
its section lists are filled. Attribution is an audit record, not cryptographic proof of authorship.
Fill it after inspecting the actual
contracts, tables, videos where relevant, and resolved configuration differences. Evaluate the
claim, baseline, held-fixed controls, all acceptance criteria and mechanism diagnostics. Distinguish
unexpected outcomes, weak slices, insufficient replication and alternative explanations. Recommend
keep/reject/modify/extend only as advice; no automatic winner or Gate D receipt is produced.

Required review fields:

- `schema_version: 1`, `evidence_id`, `author: AI|researcher`, `verdict`:
  `supported|not_supported|inconclusive|invalid_due_to_protocol_or_provenance`;
  `recommendation: keep|reject|modify|extend|defer` (advice only).
- Nonempty `executive`, `interpretation`, `limitations` lists of claims. Each claim has `kind`:
  `measured|interpretation|causal_hypothesis`, `text`, and nonempty `evidence` references.
- Nonempty `criteria`: claims plus `classification`:
  `met|not_met|inconclusive|invalid_due_to_protocol_or_provenance`. Cite the predeclared criterion
  and the computed result; unavailable criteria stay explicitly inconclusive.
- `next_experiments`: zero to three proposals in priority order, each containing `hypothesis`,
  `modification`, `controls`, `discriminating_outcomes`, `metrics`, `acceptance_criteria`,
  `cost_estimate`, and an evidence-grounded claim in `rationale`. Designs may include proposed
  quantities; label computational estimates and their assumptions, or say unknown. Do not launch.

Evidence references are JSON-pointer-style leaf paths, with `/` escaped as `~1` and `~` as `~0`.
For example a measured claim can use:

```json
{
  "kind": "measured",
  "text": "Mean seed difference {{/statistics/0/mean}}.",
  "evidence": ["/statistics/0/mean"]
}
```

Measured numbers in prose must use tokens, not typed digits. The renderer substitutes exact
computed values. Causal explanations remain `causal_hypothesis` unless experimental controls
establish them. References prove traceability, not that prose logically follows from them:
researcher review must check entailment, worded quantities, generalisation and causal claims.

Validate the completed review; output includes its content-addressed `review_id`:

```bash
python scripts/research_runs.py scientific-review \
  --evidence logs/queue/<group>/publication_evidence.json \
  --review logs/queue/<group>/scientific_review.json
```

## Approval and publication

Obtain the active researcher's actual approval of this exact review. Record it in JSON:
`{"schema_version": 1, "review_id": "review-…", "actor": "user", "approved": true}`.
Do not create approval from silence or invent a response. `actor: researcher` is allowed only in
explicitly delegated unattended mode. This approval accepts the narrative, not a Gate D choice;
`research-decide` continues to record the real scientific decision.

Pass `--review <json> --review-approval <json>` to `publish-report`, preview first with `--dry-run`.
Preview loads the existing report read-only, constructs the new layout and lists the preserved
sections. Verify URL and human text before publication. Existing legacy Analysis/Decision blocks,
including Progress, nested headings, collapsible legacy anchors and deliberately empty sections, migrate to protected subsections.
Generated interpretation never replaces these subsections. Use these protected areas for researcher
conclusions; other sections are regenerated. Repeated refreshes rebuild sections rather than append.

The report records the review and approval as an artifact and reuses an unchanged approved review
on refresh. Full scalar/nested summaries (including boolean status, table/media hashes and paths), logged/used
artifact versions and digests, final report text/plots/metric choices, selected analysis inputs,
actual notebook/helper bytes where locally available, and the current Git hypothesis participate
in approval identity. Failed artifact enumeration fails closed. Analysis runtime counters are
excluded because they are transport metadata; training resource measurements remain included.
Changed computed evidence or changed run/configuration/summary snapshots invalidate
reuse; rerun analysis and obtain approval of the revised review. Unsupported schema versions,
unknown citations and invalid hashes fail before publication. Publication refuses a missing approval
unless a nonblank `--evidence-only` reason is supplied, and records that reason. A preview can run
while blocked; `publication_allowed` reports policy readiness. Do not combine approval and fallback.
Changing any fingerprinted render input after review requires a new review/approval.
Writing new Decision text additionally requires a current, non-stale Gate D receipt; a fallback
reason does not authorize inventing a researcher decision. Existing Decision text is preserved.

`publish-report --dry-run` statically checks but **does not execute custom plot code**. Its preview
records `plots_executed: false`; validate plot output during authorized publication, before W&B writes.
`run-analysis --dry-run` still computes evidence in a disposable kernel, with SDK initialization and
ordinary mutating API capabilities blocked, disabled logging and cleared inherited identity. These
are guards for trusted workflow code, not an adversarial Python/network sandbox.

## Evidence-only fallback

If review or approval is unavailable, explicitly use `--evidence-only "<reason>"`. The report
still renders all nine sections and marks interpretation and follow-ups pending. The existing
`publish-report --intro … --analysis …` options remain valid for previews; actual unreviewed
publication now requires the explicit fallback reason. Normal skill use must document it. Missing or invalid scored artifacts still
stop the strict notebook: publish the pending evidence-only view rather than inventing rows or
weakening provenance checks. Legacy notebooks remain supported without structured evidence.

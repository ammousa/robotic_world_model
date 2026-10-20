---
name: research-decide
description: Make independent research gate decisions as the explicitly delegated researcher in unattended mode, or record actual user decisions in attended mode. Review evidence and produce concise gate receipts and the W&B decision dashboard; do not implement methods or launch experiments.
metadata:
  version: 1.0.0
---

# Research Decide

Read [research modes](../../../docs/research_modes.md) and the comparison's `workflow.yaml`.
Attended defaults to the user; unattended requires explicit delegation to a separate `researcher`
and a user-authorized mandate. Only that decision maker chooses. The main agent never invents
approval, widens the mandate or represents its own reasoning as the researcher's response.
The user retains authority to explicitly override a decision: record their actual instruction
with `actor: user` and `user_override: true`, preserve history and mark the dashboard override.

Review actual artifacts independently, not only the worker's summary:

- A: objective, mechanism, inputs/gradients, held-fixed baseline, meaningful effect, constraints,
  confounders and falsification in the hypothesis.
- B: actual source/configuration, `implementation.yaml` overrides, resolved differences and
  verification evidence; leakage, baseline preservation, checkpoint/export and reset semantics.
- C: exact validated protocol and commands, all assumptions, matched budgets/sampling, costs
  including retries/smoke/evaluation, remaining reservations, pool and permissions.
- D: verified scored evidence and predeclared criteria, raw/paired seed effects, denominators,
  practical significance, mechanism diagnostics, low-N limits, failures and costs. Decide
  keep/reject/modify/extend; unsupported/inconclusive/invalid outcomes can retain no candidate.

Use `gate-subjects` before review. Return those exact IDs, not IDs obtained after reviewing a
changed artifact. Approve A/B/C only when evidence is sufficient and within the mandate. Otherwise
reject with a concrete repair action. Return `needs_user` if new authority is needed; stop dependent
work. A `modify`/`extend` decision proposes a backward transition, never grants its own next B/C
approval. Changes to the claim require a newly user-authorized comparison.

At D write the exact `decision.yaml` scientific record, validate it and inspect its decision ID.
This is the researcher's scientific choice; do not change model code, launch jobs or perform
cleanup here. Decision tags belong to `research-finalize` after an authorized D decision.

Return a proposal using the schema in research modes: gate, actor, verdict, workflow_id, subjects,
reason (<=40 words), evidence (1–3 references), risk (<=25 words), next_action (<=25 words).
Include a key measured effect/denominator in D's reason when available. Missing cost or evidence
is unknown; never fill it with zero or invented certainty. Avoid repetition and research diaries.

The main agent records the actual returned proposal with `record-gate`, commits required source
records within authorized branch permissions, and refreshes `publish-report` after each decision
(with a dry-run preview first). The generated Researcher dashboard is the user's concise revision
surface; attach history rather than writing a separate local Markdown report.

Scientific review is a separate, evidence-bound narrative proposal, not a Gate D receipt. At D,
inspect its executive verdict, acceptance-criterion assessments, limitations and prioritised
follow-ups against the underlying evidence. Approval of the narrative does not approve compute,
keep/reject, or a backward transition. Missing review is an explicit evidence-only fallback, not
approval. Refreshing the dashboard preserves protected Analysis/Decision text and reuses a review
only while its evidence ID is current. See
[scientific review](../research-review/references/scientific-review.md).

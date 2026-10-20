---
name: research-frame
description: Turn a rough research idea into a falsifiable hypothesis contract before implementation. Use for scientific framing of a new comparison; do not implement, launch, evaluate or choose a method.
license: MIT
metadata:
  version: 1.0.0
---

# Research Frame

**Decision authority:** read [research modes](../../../docs/research_modes.md). Attended is the
default (user decisions); explicitly authorized unattended mode routes Gates A–D to a separate
`researcher` via `research-decide`. Stage approval stops remain real gates; the orchestrator
obtains the active researcher's decision and continues approved work. Never self-approve.
Record actual gate receipts and refresh the concise W&B Researcher dashboard after each gate.

Gate A asks: **Is this the question the researcher wants to test?** Read `AGENTS.md`, the nearest
method and the shared [contract reference](../research-experiment/references/scientific-contracts.md).
Do not interpret “try a method” as immediate permission to implement an ambiguous scientific idea.

Identify the claim and proposed mechanism; changes and held-fixed behavior; deployment-visible and
privileged inputs; gradient ownership; checkpoint/export impact; appropriate baselines; primary
outcome, secondary diagnostics and failure constraints; evaluation unit and paired factors;
minimum meaningful effect; anticipated confounders; and support, contradiction and inconclusive
criteria. Missing meaningful-effect thresholds may remain explicitly unspecified with a reason.
Do not decide whether the idea is good or bad.

Propose defaults and label them **ASSUMPTION**. Ask only if a missing choice changes scientific
meaning: frozen versus joint model training, imagination horizon or uncertainty use, target objective or success definition.
Consolidate necessary questions. Resolve filenames, helpers and test fixtures independently.

Choose a comparison group (one `dev/agent/<group>` branch and matching W&B group). Write a concise
`research/<group>/hypothesis.yaml`, validate with `research_runs.py validate-contract --kind hypothesis
--file …`, display the complete proposed contract and its content ID, and **stop for researcher
approval**. A file or validation success is not approval. Preserve the actual approval in the conversation and a gate receipt;
commit the approved contract on the comparison branch before implementation launch provenance is
needed. This stage changes no model source, launches no training and creates no W&B runs.

A changed claim after execution requires a new hypothesis version (archive the old file) or a new
comparison, then Gate A again. Existing groups without contracts are labelled `legacy workflow`.

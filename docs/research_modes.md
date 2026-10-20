# Attended and unattended research

Both modes use the same stages, scripts, Git branches, W&B evidence and Gates A–D.
**Attended is the default:** the user makes decisions. Explicit **unattended** mode delegates
those decisions to an independent agent named `researcher`, within a user-authorized mandate.
An absent workflow file retains historical conversation-based approvals; new comparisons create
`research/<group>/workflow.yaml` before framing. A missing `mode` means attended, never unattended.

## Authority and handoff

| Gate | Researcher examines | Decision |
|---|---|---|
| A | Hypothesis, falsification, objective and scope | approve / reject |
| B | Actual source, exact overrides, resolved settings, behavior tests, leakage and compatibility | approve / reject |
| C | Exact protocol/commands, comparability, compute reservations and permissions | approve / reject |
| D | Verified scored evidence, predeclared criteria, uncertainty and costs | keep / reject / modify / extend |

The main agent performs the existing skills and submits evidence. In attended mode it waits for
an actual user decision. In unattended mode it invokes a **separate** `researcher` agent at each
gate and waits for that agent's explicit decision, then continues the already-authorized workflow.
Configure the project role in `.claude/agents/researcher.md` (Claude Code) or `.codex/agents/researcher.toml` (Codex). If the host exposes generic subagent
tools instead of role selection, spawn `researcher` with fresh context and the full role instructions,
mandate, user objective, gate and artifact locations. Preserve its identity between gates when
possible, but always supply current artifact IDs. Never impersonate this agent in the main thread.
If separate agents are unavailable, report the blocker; do not silently self-approve.

Read `research-decide` for decision making. Implementation/evaluation/review skills keep their
existing responsibilities. Evidence review classifies outcomes; `researcher` subsequently makes
Gate D's decision. Changes reopen the corresponding gates in both modes. A failed smoke returns
to implementation and renewed B/C approval; passing an approved auto-full smoke needs no new gate.

The user must explicitly authorize unattended mode and its mandate in the session. The main agent
records that authorization; a file containing `authorized_by: user` is attribution, not proof of
consent. Neither an agent nor a receipt-writing CLI can create user authorization. Gate receipts
likewise record actual decisions; `actor` is attribution, not a cryptographic authentication system.

## Workflow files

Minimal attended configuration:

```yaml
schema_version: 1
group: my-comparison
mode: attended
```

Unattended example (all scientific limits below are illustrative, not defaults):

```yaml
schema_version: 1
group: my-comparison
mode: unattended
mandate:
  authorized_by: user
  objective: Test whether reduced actor width preserves tracking at lower inference cost.
  allowed_changes: Actor widths only; preserve observations, rewards, critic and evaluation.
  stop_when: A defensible decision is reached, the limits are exhausted, or scope changes are needed.
  limits:
    training_unit: iterations
    max_variants: 2
    max_training_seeds: 3
    max_attempts: 2
    max_followups: 0
    max_total_training_units: 36060
    max_total_evaluation_units: 10800
    compute_pool: [0, 1]
  permissions:
    commit: true
    push: true
    finalize: true
```

Use `training_unit: iterations` (the unit this repository's RL runs are budgeted in). Evaluation
units are configured episodes (population identity still requires scientific review). Units are
reservations, not GPU-hours. The researcher must also review environment count,
wall-time/GPU-hour estimates, evaluation scope and cost uncertainty at C; do not present this counter
as a hard monetary or wall-clock ceiling. For a hard GPU-hour limit use an external job scheduler's
runtime limit and include it in the mandate/protocol. Unknown actual cost must remain unknown.

Reservation is `(full units + smoke units) × variants × training seeds × max attempts`.
Evaluation reservation includes every training attempt, evaluation seed and declared evaluation
cell (this repository declares no levels or difficulties), and the declared episode population.
Each approved protocol/implementation pair reserves separately, including amendments and code
fixes requiring reruns; reapproving the same pair does not charge twice. Reservations are never automatically refunded after a failure or rejection.
`max_followups` limits additional approved protocol/implementation pairs within this comparison. A changed
claim/new group needs new user authorization; it cannot reset the budget by changing group names.
Missing mandate values require one consolidated user clarification, not agent-invented authority.
Permissions constrain existing branch rules; merging, rewriting history and deleting historical
runs remain forbidden. Finalize permission permits applying the explicit D decision and leaves
cleanup uncommitted for review. Committing finalized cleanup needs a separate explicit approval.

## Review artifacts and gate receipts

Before B, write `implementation.yaml` with a `variants` mapping identical to the proposed
protocol's `variants`: task, eval_task, ordered overrides and eval_overrides. Include resolved
configuration/verification evidence in the handoff. This file is scientific source, not runtime
state. B hashes executable source/configuration/tests and these settings, including unstaged,
untracked and deleted files; commits alone do not invalidate it. Analysis/report outputs do not
change this implementation identity.

Validate mode and inspect the IDs **before independent review**:

```bash
python scripts/research_runs.py validate-workflow --file research/<group>/workflow.yaml
python scripts/research_runs.py gate-subjects --group <group> --gate A
```

The actual gate owner returns this proposal (B/C/D use the complete `subjects` output):

```yaml
gate: A
actor: researcher
verdict: approve
workflow_id: w-<from-validation>
subjects:
  hypothesis_id: h-<reviewed-id>
reason: The claim isolates actor width and has a matched baseline and explicit falsification.
evidence: [research/my-comparison/hypothesis.yaml]
risk: Low seed count limits precision.
next_action: Implement the approved widths and submit verification at B.
```

Use `actor: user` for an actual attended decision. Save the returned proposal under the existing
`logs/queue/<group>/` scratch directory, then record it:

```bash
python scripts/research_runs.py record-gate --group <group> --file logs/queue/<group>/gate.yaml
```

This appends a bounded summary and exact reviewed IDs to `gate_decisions.yaml`; preserve its
history. Receipts also bind the prior gate receipts; renewing or rejecting an earlier gate reopens
downstream approval even when source bytes are unchanged. A/B/C approve/reject; D keep/reject/modify/extend, matching validated `decision.yaml`.
A D decision always requires current A/B/C approvals. A reject at A/B/C may be recorded without
approval of preceding stages so a blocker remains visible. `needs_user` is a handoff outcome:
record a rejected gate where artifacts exist, state the requested mandate change, and stop.

`queue` checks current A/B/C approvals, reviewed overrides, mandate budgets, pool and push
permissions before pinning or dispatching any job for comparisons with `workflow.yaml`.
Commit scientific files and receipts before clean-source launch. Legacy comparisons are readable
without these new files; never infer an unattended authorization from historical records.

## Dashboard for later revision

After each gate, preview then refresh the comparison's existing W&B report:

```bash
python scripts/research_runs.py publish-report --project <project> --group <group> --dry-run
python scripts/research_runs.py publish-report --project <project> --group <group>
```

This is authorized reporting in unattended mode; no separate publication question is needed.
The **Researcher dashboard** appears near the top, even before training. It shows four latest-gate
rows: decision, reason (40 words maximum), 1–3 evidence references, limitation (25 words) and next
action (25 words). Pending and stale approvals are explicit. Keep reasons specific: cite the
observed effect and denominator at D when evidence supports one; avoid chronological narratives.
Full timestamps, actors, reviewed IDs and prior decisions are attached as a `research-decisions`
artifact and retained in the report run's config. Human Analysis/Decision edits survive refreshes.
A refresh without the local workflow uses the previous report's saved dashboard; it cannot verify
freshness against missing local source. Treat that copy as the last published snapshot.

Later user revision is a new decision, not an edit to a prior receipt. The user can explicitly
supersede a researcher decision while staying in unattended mode: record the actual user response
with `actor: user` and `user_override: true`. Show the override in the dashboard; preserve the
original receipt. This field records a real user instruction, never agent-invented authority.
A D override on unchanged evidence needs no renewed A/B/C review. Renewing earlier gates still
reopens downstream approval. Changed scope/budgets require an amended user-authorized mandate;
switching mode changes the workflow ID and requires renewed approval.
No experiment, chosen/rejected tagging or cleanup runs merely because the dashboard was opened.

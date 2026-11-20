# Implementation plans

One file per non-trivial change: `YYYY-MM-DD-<topic>.md`, written before the implementation and
kept as the record of what was planned. Layout:

```markdown
# <topic>
Status: Planned | In progress | Done (PR #N) | Dropped
Refs: pull requests, sections of docs/protocol.md and docs/configuration.md

## Context      — the problem and the request
## Approach     — the chosen design (alternatives only if rejected for a recorded reason)
## Invariants   — which rules of .claude/rules/fep-invariants.md are touched, and how
## Verification — CPU tests, GPU tests, runs, and the evidence expected
```

A plan may be written in English or Japanese; the plans so far are in Japanese. The `Status:`
line of each plan is the one record of its state. What outlives the plan goes to
`docs/protocol.md` or `docs/configuration.md` and to `PROGRESS.md`.

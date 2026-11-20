# Progress log rules

How `PROGRESS.md` (repository root) is maintained. Shared by Claude Code (`CLAUDE.md`) and Codex
(`AGENTS.md`). This file is loaded every session: keep it, and `PROGRESS.md`, short. Detail
belongs in `docs/protocol.md`, `docs/configuration.md`, `docs/plans/` and
`docs/progress/archive/`.

## When to update PROGRESS.md (before committing)

1. A default, a method or a check changes (and with it `docs/protocol.md` or
   `docs/configuration.md`), including what a review changed.
2. A plan in `docs/plans/` starts, completes or is dropped, or a validation or benchmark run
   gives a result.
3. A significant implementation lands (a new input route, estimator, restraint, sampling
   mechanism, backend or example).
4. A blocker appears or is resolved.

Routine refactors, typo fixes and small edits of documents do NOT need an entry.

## Structure: exactly three sections

- `## Product`: the target product, as the plans approved it. Update only when that changes.
  25 lines or fewer.
- `## Current Status`: a snapshot, OVERWRITTEN in place: the state of the code, the tests, the
  examples and benchmarks with their latest results, open items and blockers. No history.
  80 lines or fewer.
- `## Change Log`: newest first. Entry format:

```markdown
### YYYY-MM-DD — [design|progress|amendment] short headline
- What: what was decided / changed / completed (compressed; Refs hold the detail)
- Why: the reason (REQUIRED for design changes and amendments)
- Refs: plans, pull requests, commits, sections of docs/protocol.md or docs/configuration.md
```

Entries are 10 lines or fewer. Only recent entries stay here; older ones move to
`docs/progress/archive/change-log.md`.

## Size budget (enforced)

`PROGRESS.md`: 200 lines or fewer, 10 Change Log entries or fewer, exactly the three sections
above. The SessionStart hook (`.claude/settings.json` runs `scripts/check_progress_size.py`)
prints a warning when the budget is exceeded, and the `harness` job of CI fails on it: then
follow `docs/progress/archiving.md`, as its own commit, before other work touches
`PROGRESS.md`.

## Hard rules

- NEVER rewrite or delete past Change Log entries, here or in the archive. Moving them verbatim
  to the archive (`docs/progress/archiving.md`) is the ONLY allowed move; a correction is a new
  entry that names the old one.
- Keep `Current Status` consistent with `docs/protocol.md`, `docs/configuration.md` and the
  `Status:` lines of `docs/plans/`; if they diverge, fix them in the same commit.
- A result is given with its uncertainty, the setting it was run with and where it is recorded,
  and without a verdict of cuFEP's own (`.claude/rules/fep-invariants.md`). No speed or run time
  of the cuPotential plugin.
- State what was verified and how. "Not run" and "not verified here" are status values: write
  them rather than leaving a gap.
- Write entries in English.

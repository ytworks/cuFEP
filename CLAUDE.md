# cuFEP

cuFEP computes absolute binding free energies (ABFE) on OpenMM. From a protein PDB and a ligand
SDF with the ligand placed in the pocket it prepares the system (water, or a lipid bilayer when
`membrane.enabled` is set), equilibrates, chooses Boresch restraints, runs double decoupling with
Hamiltonian replica exchange and REST2 on all GPUs, and analyses every leg with MBAR. NVIDIA's
cuPotential plugin for OpenMM is the default compute backend; stock OpenMM is the second backend.
Package: `src/cufep/`.

## Session start

ALWAYS read `PROGRESS.md` first: it is the cross-session memory of the product, the current
status, recent changes and open items. Then read the plan in `docs/plans/` that the work belongs
to, if there is one.

## Execution: Docker only

@.claude/rules/docker-execution.md

## Progress log rules

@.claude/rules/progress-log.md

## Free-energy invariants and project rules

@.claude/rules/fep-invariants.md

## Test policy

Apply these rules whenever a change adds, removes, moves or reclassifies behaviour or tests.

1. Keep only tests that protect current behaviour from a concrete regression: necessary and
   sufficient, not maximal. Do not mirror constants or restate a check that the code already
   makes when it reads a configuration.
2. Test observable results: free energies against synthetic truth and against pymbar, energies
   of the alchemical and REST2 states against setting each state, refusals that happen before
   anything is written, the same directory state through every entry point (`cufep run`,
   `cufep analyze`, `cufep benchmark`), and restart and interruption at every write step.
3. `pytest` (the default selection, `tests/unit`) must pass on a CPU-only machine with stock
   OpenMM: that is what the `unit` job of CI runs. `tests/gpu` (marks `gpu`, `cupotential`)
   needs a GPU and runs on a GPU machine, not in CI. `physical` marks long physical validation
   runs. A change that touches what `tests/gpu` covers runs it before the push.
4. A verdict depends only on the code under test: never on wall-clock time, speed ratios or
   runner load. Use fixed seeds and platforms that give the same numbers in every run (the
   Reference platform, or the CPU platform with one thread and deterministic forces, as the
   comparison of the backends in `tests/gpu/test_gpu.py` does: MD on a GPU and the CPU platform
   with several threads are not reproducible to the bit). Timing is recorded as a diagnostic
   (`timing.json`) and decides nothing.
5. Statistical checks use fixed seeds and explicit tolerances taken from measured scatter.
6. A regression test for a bug must fail on the code before the fix; check that once. A finding
   of a review is reproduced in the code first and its reproduction, with its numbers, becomes
   the test.
7. The numbers of the analysis are pinned (`tests/unit/data/analysis_golden.json`). A change of
   those numbers needs a new `ANALYSIS_VERSION` in `src/cufep/analysis/mbar.py` and new golden
   numbers (`python tests/unit/test_sampling.py --record-golden`, inside the image).
8. When a feature is deleted, delete its tests and helpers with it; file moves are not coverage.

## Where things live

- Package `src/cufep/`: `config/` (settings schema and defaults, values chosen from the input),
  `prep/` (protein, ligand, force field, membrane and its orientation, solvation, crystal water),
  `alchemy/` (alchemical system, Boresch restraints, lambda schedule and adaptive lambda
  scheduling, long-range dispersion), `sampling/` (pre-equilibration, Hamiltonian replica
  exchange, states, sample storage, GPU workers), `analysis/` (MBAR, uncertainty, checks,
  figures), `backends/` (cuPotential and stock OpenMM), `benchmarks/` (sets, runner, statistics),
  `scheduler.py` and `mps.py` (GPUs and NVIDIA MPS), `workflow.py` and `run.py` (the stages),
  `cli.py` (the `cufep` command), `validate.py` (energies of the two backends), `envcheck.py`
- Specification: `docs/protocol.md` (methods, checks, comparison with other programs);
  every setting and its basis: `docs/configuration.md` (a unit test keeps its block identical to
  the defaults of the code); licences: `docs/LICENSING.md`; first risk checks: `docs/M0_REPORT.md`
- Plans: `docs/plans/YYYY-MM-DD-<topic>.md` (`docs/plans/README.md`)
- Examples: `examples/<name>/`; benchmark sets: `benchmarks/<set>/` (inputs from the Schrödinger
  public binding free energy benchmark, MIT licence, `benchmarks/THIRD_PARTY/`)
- Tests: `tests/unit/` (CPU), `tests/gpu/` (GPU); shared fixtures: `tests/conftest.py`
- Investigation scripts: `spikes/` (their output goes to `spikes/out/`, untracked); run
  directories: `runs/` (untracked)
- The official image: `docker/Dockerfile`, `docker/environment.yml`, `docker/entrypoint.sh`;
  wrapper: `scripts/in_docker.sh`
- Archived progress history: `docs/progress/archive/`

## Dev commands

Every command runs in the official image (rule above); write `scripts/in_docker.sh` in front.

- CPU: `scripts/in_docker.sh python -m pytest -q` (the unit tests, as CI runs them),
  `scripts/in_docker.sh python scripts/check_progress_size.py`,
  `scripts/in_docker.sh python scripts/check_readme_pair.py` (the checks of the `harness` job).
- GPU: `scripts/in_docker.sh --gpus 0 python -m pytest tests/gpu -q`;
  `scripts/in_docker.sh --gpus 0 cufep check-env`.
- A long run uses an image built from a commit, so that later edits of the tree cannot change
  it: `IMAGE=cufep:<commit> spikes/run_frozen.sh run config.yaml -o runs/<name>`.
- Before a commit: run the commands above and read the exit status of the command itself. A pipe
  (`pytest | tail`) reports the status of its last stage and hides a failure.
- Commit in small steps and push each one; the pull request is the unit of review. Read the
  reviews of the pull request before each step and after each push.

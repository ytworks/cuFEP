# Contributing to cuFEP

Thank you for your interest in cuFEP. Bug reports, questions and pull requests are welcome on
[GitHub](https://github.com/ytworks/cuFEP). This page explains how the project is developed; the
same rules are written for coding agents in [CLAUDE.md](CLAUDE.md) and `.claude/rules/`.

## Reporting a problem

Open an issue with:

- the cuFEP commit or image you used and the output of `cufep check-env`;
- the configuration (`resolved_config.yaml` of the run) and, if you can share them, the inputs;
- the command, what you expected and what happened, with the end of the log.

Do not post timings measured with the cuPotential plugin: NVIDIA's licence restricts publishing
performance results of its products ([docs/LICENSING.md](docs/LICENSING.md)).

## Development environment

Everything runs in the official Docker image; no Python environment is needed on the host.

```bash
docker build -f docker/Dockerfile -t cufep:dev .   # or: scripts/in_docker.sh --build cufep --help
scripts/in_docker.sh cufep --help
```

`scripts/in_docker.sh` mounts the repository at `/work`, runs the source of the working tree
(`PYTHONPATH=/work/src`) as your user, and passes only the GPUs you name with `--gpus`. See
[.claude/rules/docker-execution.md](.claude/rules/docker-execution.md) for its options.

## Tests

```bash
scripts/in_docker.sh python -m pytest -q                       # unit tests (CPU, stock OpenMM)
scripts/in_docker.sh --gpus 0 python -m pytest tests/gpu -q    # GPU tests (CUDA, both backends)
scripts/lint.sh                                                # ruff, as in CI
scripts/in_docker.sh python scripts/check_progress_size.py     # harness checks, as in CI
scripts/in_docker.sh python scripts/check_readme_pair.py
scripts/in_docker.sh python scripts/check_names.py
```

CI runs the unit tests, the harness checks and ruff on every pull request; it has no GPU, so run the
GPU tests yourself when a change touches what they cover. Read the exit status of the command
itself: a pipe such as `pytest | tail` hides a failure.

The test policy is in [CLAUDE.md](CLAUDE.md#test-policy). In short: test observable results
(free energies against synthetic truth and pymbar, energies against setting each state,
refusals before anything is written), keep verdicts deterministic (fixed seeds, never
wall-clock time), and make a regression test fail on the code before the fix.

## Rules of the project

The full list is [.claude/rules/fep-invariants.md](.claude/rules/fep-invariants.md). The ones a
contribution meets most often:

- **Only published criteria decide.** A threshold that stops, refuses or warns needs a
  published source, quoted in [docs/configuration.md](docs/configuration.md). Everything else
  is reported as a number.
- **Defaults and terms follow established practice and the literature**; give the source of a
  new default in [docs/configuration.md](docs/configuration.md). One calculation per ligand is
  the default.
- **Every setting has a recommended default** with its basis in
  [docs/configuration.md](docs/configuration.md); a unit test keeps the block of all defaults
  there identical to the code.
- **The numbers of the analysis are pinned**: a change of them needs a new `ANALYSIS_VERSION`
  and new golden numbers.
- **Fix a problem for its whole class**, through one shared function, not only the case that
  was reported.

## Pull requests

- Write a plan in [docs/plans/](docs/plans/README.md) first for a non-trivial change.
- Keep commits small; each one should pass the tests. Describe in the pull request what was
  verified and how ("not run" is an acceptable answer, a gap is not).
- Update [docs/protocol.md](docs/protocol.md) and [docs/configuration.md](docs/configuration.md)
  when a method or a default changes, and [PROGRESS.md](PROGRESS.md) as
  [.claude/rules/progress-log.md](.claude/rules/progress-log.md) describes.
- The user documentation is in English and Japanese: a change of `README.md` is made in
  `README.ja.md` too (`scripts/check_readme_pair.py` checks that the two keep the same
  headings, commands and links).
- The README and this page do not describe a design decision of cuFEP as following or copying a
  commercial FEP product. Other programs are named only to compare accuracy or to cite a source.

## Licence

By contributing you agree that your contribution is licensed under the
[Apache License 2.0](LICENSE).

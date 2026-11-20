# Docker execution rule

Everything except CI is executed and tested in the official image, as in
[fmeta](https://github.com/ytworks/fmeta).

## The rule

- `cufep` commands, simulations, `pytest`, the scripts in `scripts/` and `spikes/`, the GPU tests
  and the benchmarks run in a container of the official image, through `scripts/in_docker.sh`
  (`spikes/run.sh` calls it; `spikes/run_frozen.sh` runs the code built into an image of a
  commit, through the entrypoint of the image, for runs whose result matters).
- On the host: `git`, `gh`, `docker`, file editing and read-only state checks (`nvidia-smi`,
  `docker logs`, `docker ps`). No Python environment on the host (no conda, venv or
  `pip install`), and the project's Python is never run there.
- Exactly three exceptions:
  1. CI (`.github/workflows/`). The `unit` job builds the conda environment of
     `docker/environment.yml` on the runner (stock OpenMM, no GPU, no plugin); the `harness` job
     runs the standard-library checks of `scripts/` with the runner's Python.
  2. The SessionStart hook (`scripts/check_progress_size.py`, standard library only), which must
     also work in a session that has no Docker.
  3. ruff runs in its own official image, pinned in `scripts/lint.sh`, locally and in CI; it reads
     the code and needs nothing of the cuFEP image.
- A session without Docker does not fall back to the host: leave the verification to CI and
  report "not verified here".

## The image

- One image (`docker/Dockerfile`): the conda environment of `docker/environment.yml` on NVIDIA's
  CUDA base image, NVIDIA's cuPotential plugin from its release channel (pinned by version and
  checksum), and cuFEP. It holds `pytest`, so the image that is tested is the image that is used.
- `--build-arg WITH_CUPOTENTIAL=0` builds an image with stock OpenMM only, which contains no
  NVIDIA proprietary software.
- The image holds NVIDIA's software under NVIDIA's licences (`/licenses` in the image,
  `docs/LICENSING.md`). No speed or run time measured with the cuPotential plugin goes into a
  tracked file, a commit message or any text on GitHub: the licence restricts disclosing
  performance results without NVIDIA's written permission. Accuracy results are published.
- An image is not pushed to a registry until the licence question of redistributing the runtime
  is settled (`docs/LICENSING.md`).

## Usage

```bash
scripts/in_docker.sh python -m pytest -q                     # unit tests, no GPU
scripts/in_docker.sh --gpus 0 python -m pytest tests/gpu -q  # GPU tests on host GPU 0
scripts/in_docker.sh --gpus 0,1 cufep run examples/kinase_tyk2/config.yaml -o runs/tyk2
scripts/in_docker.sh --gpus all --detach --name tyk2 cufep run examples/kinase_tyk2/config.yaml
scripts/in_docker.sh --build cufep --help                    # rebuild the image first
```

- The repository is mounted at `/work` and its source is the one that runs
  (`PYTHONPATH=/work/src`, `CUFEP_DATA=/work`); the process runs as the calling user, so the
  files it writes belong to that user.
- `--gpus` takes host GPU numbers and passes only those (none when omitted). Inside the container
  they are numbered from 0 again: `--gpus 6,7` outside is `0,1` inside, and that is what
  `resources.gpus` and `--gpus` of `cufep run` see. `--docker-gpus` passes a value to
  `docker run --gpus` as it is (a count, `all`, `"device=..."`); `spikes/run.sh` uses it for
  `$GPUS`, with `--workdir /work`, so its GPUs and its working directory are those it had before
  the wrapper existed.
- An image is built only with `--build`; a missing image stops the wrapper, so a tag such as
  `cufep:<commit>` never names a tree it was not built from.
- `--shm-size=16g` is set: replica coordinates are shared between worker processes in memory, and
  Docker's default of 64 MB is too small.
- cuFEP starts and stops its own NVIDIA MPS daemon for the workers that share a GPU
  (`src/cufep/mps.py`); nothing has to be started on the host.
- Long runs use `--detach`: follow with `docker logs -f NAME`, stop with `docker stop NAME`,
  continue by running the same command again (a run resumes from its checkpoints). A run whose
  result matters uses an image built from a commit (`spikes/run_frozen.sh`), so that edits of
  the tree while it runs cannot change it.

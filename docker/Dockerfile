# syntax=docker/dockerfile:1
#
# Official cuFEP runtime image.
#
#   docker build -f docker/Dockerfile -t cufep:latest .
#   docker run --rm --gpus all cufep:latest check-env
#
# Build args:
#   WITH_CUPOTENTIAL=1   install NVIDIA's cuPotential OpenMM plugin (default).
#                        Set to 0 for a stock-OpenMM-only image ("-stock" variant)
#                        that contains no NVIDIA proprietary runtime.
#
# The cuPotential runtime (libcupo) is downloaded from NVIDIA's GitHub release at
# build time and is governed by the NVIDIA Software License Agreement; its licence
# texts are copied to /licenses. See docs/LICENSING.md before publishing an image.

ARG CUDA_VERSION=13.4.3
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-base-ubuntu24.04

ARG WITH_CUPOTENTIAL=1
ARG CUPOTENTIAL_VERSION=0.3.5
ARG CUPOTENTIAL_CHANNEL_SHA256=5689ab3d002bb9b068c802225a810f5342e93f6b9f65ce6b4cbfcb722c7a4e87

ENV DEBIAN_FRONTEND=noninteractive \
    MAMBA_ROOT_PREFIX=/opt/conda \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update -qq \
 && apt-get install -y --no-install-recommends ca-certificates curl bzip2 \
 && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://micro.mamba.pm/api/micromamba/linux-64/latest \
  | tar -xj -C /usr/local bin/micromamba

# --- scientific stack (conda-forge, one OpenMM only) -------------------------
COPY docker/environment.yml /tmp/environment.yml
RUN micromamba create -y -n cufep -f /tmp/environment.yml \
 && micromamba clean -a -y

# --- cuPotential plugin from NVIDIA's release channel -------------------------
RUN mkdir -p /licenses \
 && if [ "${WITH_CUPOTENTIAL}" = "1" ]; then \
      set -e; \
      cd /tmp; \
      base=https://github.com/NVIDIA/cuPotential; \
      ch=nv-openmm-plugin-conda-channel-${CUPOTENTIAL_VERSION}; \
      curl -fsSL -o ${ch}.tar.gz ${base}/releases/download/v${CUPOTENTIAL_VERSION}/${ch}.tar.gz; \
      echo "${CUPOTENTIAL_CHANNEL_SHA256}  ${ch}.tar.gz" | sha256sum -c -; \
      tar -xzf ${ch}.tar.gz; \
      micromamba install -y -n cufep -c /tmp/${ch} -c conda-forge \
          "nv-openmm-plugin=${CUPOTENTIAL_VERSION}"; \
      for f in LICENSE LICENSE-apache.txt LICENSE-cupo.txt NOTICE THIRD-PARTY.txt; do \
        curl -fsSL -o /licenses/cuPotential-${f} \
          https://raw.githubusercontent.com/NVIDIA/cuPotential/v${CUPOTENTIAL_VERSION}/${f}; \
      done; \
      rm -rf /tmp/${ch} /tmp/${ch}.tar.gz; \
      micromamba clean -a -y; \
    fi

# --- cuFEP itself -------------------------------------------------------------
COPY pyproject.toml README.md LICENSE /opt/cufep/
COPY src /opt/cufep/src
COPY benchmarks /opt/cufep/benchmarks
COPY examples /opt/cufep/examples
RUN micromamba run -n cufep pip install --no-deps /opt/cufep \
 && cp /opt/cufep/LICENSE /licenses/cuFEP-LICENSE

# A world-writable home so the image works with `--user $(id -u):$(id -g)`.
RUN mkdir -p /home/cufep /work && chmod 1777 /home/cufep /work
ENV HOME=/home/cufep \
    CUFEP_DATA=/opt/cufep \
    PATH=/opt/conda/envs/cufep/bin:${PATH} \
    CONDA_PREFIX=/opt/conda/envs/cufep \
    OPENMM_PLUGIN_DIR=/opt/conda/envs/cufep/lib/plugins \
    MPLBACKEND=Agg

COPY docker/entrypoint.sh /usr/local/bin/cufep-entrypoint
RUN chmod 755 /usr/local/bin/cufep-entrypoint

USER 1000:1000
WORKDIR /work
ENTRYPOINT ["/usr/local/bin/cufep-entrypoint"]
CMD ["--help"]

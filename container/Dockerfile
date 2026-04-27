# syntax=docker/dockerfile:1
# Prometheus CPU image — ubuntu:22.04 (glibc 2.35, kernel ≥ 4.18).
#
# Build:
#   docker build -f container/Dockerfile -t prometheus:cpu .
#
# Run:
#   docker run --rm -it prometheus:cpu
#
# The full simulation stack is baked in:
#   PROPOSAL, LeptonInjector (vendored), fennel (vendored),
#   PPC CPU binary, and prometheus itself.

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: system packages + micromamba environment
# ─────────────────────────────────────────────────────────────────────────────
FROM ubuntu:22.04 AS base

LABEL org.opencontainers.image.title="Prometheus"
LABEL org.opencontainers.image.description="Neutrino telescope simulation (CPU)"
LABEL org.opencontainers.image.source="https://github.com/Harvard-Neutrino/prometheus"
LABEL org.opencontainers.image.licenses="LGPL-2.1"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        cmake \
        curl \
        ca-certificates \
        pkg-config \
        libhdf5-serial-dev \
        libboost-all-dev \
        libgsl-dev \
        libsuitesparse-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install micromamba
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
        | tar -xvj -C /usr/local bin/micromamba

ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/prometheus/bin:/usr/local/bin:${PATH}

# Copy only the environment spec first for better layer caching
COPY environment.yml /tmp/environment.yml
RUN micromamba create -y -n prometheus -f /tmp/environment.yml \
    && micromamba clean -afy

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: install PROPOSAL and pre-build its interpolation tables
# ─────────────────────────────────────────────────────────────────────────────
FROM base AS proposal

COPY requirements.txt /tmp/requirements.txt
RUN /opt/conda/envs/prometheus/bin/pip install --no-cache-dir -r /tmp/requirements.txt

# Build PROPOSAL tables at image-build time so the first simulation run
# does not pay the ~2 minute table-generation penalty.
COPY container/build_proposal_tables.py /tmp/build_proposal_tables.py
RUN LD_LIBRARY_PATH=/opt/conda/envs/prometheus/lib \
    /opt/conda/envs/prometheus/bin/python /tmp/build_proposal_tables.py

# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: build and install LeptonInjector from vendored source
# ─────────────────────────────────────────────────────────────────────────────
FROM proposal AS leptoninjector

COPY resources/LeptonInjector /opt/LI/source
WORKDIR /opt/LI

RUN mkdir build install && cd build \
    && cmake \
        -DCMAKE_INSTALL_PREFIX=/opt/LI/install \
        -DPython_EXECUTABLE=/opt/conda/envs/prometheus/bin/python \
        -DCMAKE_PREFIX_PATH=/opt/conda/envs/prometheus \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    /opt/LI/source \
    && make -j"$(nproc)" \
    && make install

ENV PYTHONPATH=/opt/LI/install/lib/python3.12/site-packages
ENV LD_LIBRARY_PATH=/opt/LI/install/lib:/opt/conda/envs/prometheus/lib

# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: build PPC CPU binary from vendored source
# ─────────────────────────────────────────────────────────────────────────────
FROM leptoninjector AS ppc

COPY resources/PPC_executables/PPC /opt/PPC
WORKDIR /opt/PPC
RUN make cpu
ENV PPC=/opt/PPC/ppc

# ─────────────────────────────────────────────────────────────────────────────
# Stage 5: install prometheus + fennel (both vendored)
# ─────────────────────────────────────────────────────────────────────────────
FROM ppc AS prometheus_install

COPY resources/fennel /opt/fennel
RUN /opt/conda/envs/prometheus/bin/pip install --no-cache-dir /opt/fennel

# Copy full repo and install prometheus in editable mode so examples work
COPY . /opt/prometheus
WORKDIR /opt/prometheus

RUN /opt/conda/envs/prometheus/bin/pip install --no-cache-dir pytest \
    && /opt/conda/envs/prometheus/bin/pip install --no-cache-dir -e .

# ─────────────────────────────────────────────────────────────────────────────
# Stage 6: final image — drop build artefacts, set environment
# ─────────────────────────────────────────────────────────────────────────────
FROM prometheus_install AS final

# Persist all necessary environment variables into the image so every shell
# (interactive or exec) has a working Python + LI + PPC environment.
ENV PATH=/opt/conda/envs/prometheus/bin:/opt/PPC:/usr/local/bin:${PATH}
ENV PYTHONPATH=/opt/LI/install/lib/python3.12/site-packages:${PYTHONPATH:-}
ENV LD_LIBRARY_PATH=/opt/LI/install/lib:/opt/conda/envs/prometheus/lib:${LD_LIBRARY_PATH:-}
ENV PPC=/opt/PPC/ppc
ENV EARTH_PARAMS=/opt/LI/source/resources/earthparams/

WORKDIR /opt/prometheus

# Smoke-test the full stack at image-build time: import PROPOSAL, LI, prometheus.
RUN python -c "import proposal; import LeptonInjector; import prometheus; print('Image OK')"

CMD ["bash"]

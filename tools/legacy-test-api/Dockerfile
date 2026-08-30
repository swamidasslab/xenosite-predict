# Python 2.7 + OpenBabel 2.4 + TensorFlow 1.15 for legacy /predict parity tests.
# TF 1.15 py27 is conda-only. Split installs + pinned protobuf match xenosite-legacy/env.lock.
FROM --platform=linux/amd64 mambaorg/micromamba:1.1.0

USER root
WORKDIR /build
ARG MAMBA_DOCKERFILE_ACTIVATE=1

RUN --mount=type=cache,target=/opt/conda/pkgs \
    micromamba install --name base --yes -c defaults python=2.7 pip setuptools \
    && micromamba clean --all --yes

RUN --mount=type=cache,target=/opt/conda/pkgs \
    micromamba install --name base --yes -c conda-forge \
      numpy scipy pandas openbabel=2.4 rdkit mock \
    && micromamba clean --all --yes

RUN --mount=type=cache,target=/opt/conda/pkgs \
    micromamba install --name base --yes -c defaults \
      tensorflow=1.15 protobuf=3.11.2 \
    && micromamba clean --all --yes

# openopt unpickles epoxidation/quinone/… .model files (see xenosite-legacy/requirements.txt).
RUN --mount=type=cache,target=/opt/conda/pkgs \
    micromamba install --name base --yes -c conda-forge setproctitle \
    && /opt/conda/bin/pip install --no-cache-dir openopt \
    && micromamba clean --all --yes

ENV TF_CPP_MIN_LOG_LEVEL=1
ENV KMP_AFFINITY=novverbose
ENV PORT=8099
ENV PYTHONPATH=/src
ENV XENOSITE_LEGACY_DATA=/src

COPY server.py /opt/legacy-test-api/server.py
EXPOSE 8099
CMD ["python", "/opt/legacy-test-api/server.py"]

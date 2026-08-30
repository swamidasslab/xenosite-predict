# Slim python2.7 + OpenBabel oracle for RDKit-vs-OB dumps and NN /predict.
# linux/amd64. Does not require the WashU registry. Not the full TF image.
FROM --platform=linux/amd64 mambaorg/micromamba:1.1.0

USER root
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN micromamba install --name base --yes -c conda-forge -c defaults \
      python=2.7 numpy scipy pandas openbabel=2.4 rdkit setuptools pip \
    && micromamba clean --all --yes

COPY server.py /opt/legacy-test-api/server.py
ENV PORT=8099
ENV PYTHONPATH=/src
EXPOSE 8099
CMD ["python", "/opt/legacy-test-api/server.py"]

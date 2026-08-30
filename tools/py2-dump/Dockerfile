# Python 2.7 dump image: unpickle numpy-NN graphs without the WashU registry,
# and dump OpenBabel 2.4 features via Debian's python-openbabel.
# linux/amd64 for manylinux numpy/scipy wheels (host may be arm64).
FROM --platform=linux/amd64 python:2.7-slim

RUN pip install --no-cache-dir 'numpy==1.16.6' 'scipy==1.2.3'

# Buster is archived. python-openbabel 2.4.1 and python-rdkit install into
# Debian /usr/bin/python (not the image's /usr/local CPython). Dumps must run
# with --entrypoint /usr/bin/python. python-rdkit is for phase1 Bond_and_LonePair
# (GetNOuterElecs); libridass __init__ files are still not imported.
RUN printf 'deb http://archive.debian.org/debian buster main\n' > /etc/apt/sources.list \
 && printf 'deb http://archive.debian.org/debian-security buster/updates main\n' >> /etc/apt/sources.list \
 && printf 'Acquire::Check-Valid-Until "false";\n' > /etc/apt/apt.conf.d/99archive \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      python-openbabel python-rdkit python-pandas python-numpy python-setuptools openbabel \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY dump_nn_py2.py /work/dump_nn_py2.py
COPY dump_ob_features.py /work/dump_ob_features.py
ENTRYPOINT ["python", "/work/dump_nn_py2.py"]

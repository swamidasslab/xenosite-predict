# Python 2.7 dump image: unpickle numpy-NN graphs without the WashU registry.
# linux/amd64 for manylinux numpy/scipy wheels (host may be arm64).
FROM --platform=linux/amd64 python:2.7-slim

RUN pip install --no-cache-dir 'numpy==1.16.6' 'scipy==1.2.3'

WORKDIR /work
COPY dump_nn_py2.py /work/dump_nn_py2.py
ENTRYPOINT ["python", "/work/dump_nn_py2.py"]

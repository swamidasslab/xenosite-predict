# uv has no first-class task runner; this Makefile is the equivalent.
# Tools live under tools/ and are NOT installed with the wheel.

PYTHON ?= uv run python
PYTEST ?= uv run pytest
CONVERT ?= uv run --group convert python
IMAGE ?= dockerreg01.accounts.ad.wustl.edu/swamidass/xenosite-legacy:api
LEGACY_COMPOSE ?= tools/legacy-test-api/compose.yml
TARBALL ?= ../xenosite-legacy/data/xenosite_legacy_data_trimmed.tgz

.PHONY: extract-weights convert-onnx convert-onnx-$(MODEL) test test-golden test-live \
	legacy-test-api legacy-test-api-down py2-dump-image dump-ob dump-ob dump-ob-features \
	capture-suite-onnx gather-golden drift-report help

help:
	@echo "extract-weights     copy pickles/TSV/source from $(IMAGE) into weights/legacy/"
	@echo "convert-onnx        pickle → safetensors → ONNX (all models)"
	@echo "convert-onnx MODEL=epoxidation"
	@echo "test                unit tests, Docker-free (-n auto via pyproject.toml)"
	@echo "test-golden         golden_descriptor_suite ONNX parity (-n auto)"
	@echo "test-live           pytest -m live (skips if Docker/image/weights missing)"
	@echo "py2-dump-image      build python:2.7-slim dump image (numpy + OpenBabel 2.4 + RDKit)"
	@echo "dump-ob             fill descriptor suite incrementally (skip dumps already present)"
	@echo "capture-suite-onnx  cache ONNX scores (CAPTURE_WORKERS=8 default; CAPTURE_MODEL/SMILES to filter)"
	@echo "gather-golden        regather failing suite rows from legacy-test-api (GATHER_WORKERS=8)"
	@echo "drift-report        classify ONNX vs golden from cache (DRIFT_WORKERS=8 default)"
	@echo "legacy-test-api     build/run derived test image"
	@echo "legacy-test-api-down"

extract-weights:
	$(PYTHON) tools/extract_weights.py --image $(IMAGE) --tarball $(TARBALL) --out weights/legacy

convert-onnx:
	$(CONVERT) tools/convert_onnx.py --src weights/legacy --out weights/onnx $(if $(MODEL),--model $(MODEL),)

test:
	$(PYTEST) -m "not live"

test-golden:
	$(PYTEST) tests/test_golden_suite.py

test-live:
	$(PYTEST) -m live

legacy-test-api:
	docker compose -f $(LEGACY_COMPOSE) up --build -d

legacy-test-api-down:
	docker compose -f $(LEGACY_COMPOSE) down

py2-dump-image:
	docker build --platform linux/amd64 -t xenosite-predict-py2:dump tools/py2-dump

SMILES ?= O=C(C)Oc1ccccc1C(=O)O
MODEL ?= epoxidation

# Optional filters for capture-suite-onnx (empty = full golden suite)
CAPTURE_MODEL ?=
CAPTURE_SMILES ?=
CAPTURE_WORKERS ?= 8
DRIFT_WORKERS ?= 8
GATHER_WORKERS ?= 8

dump-ob-features:
	$(PYTHON) tools/dump_ob.py --smiles '$(SMILES)' --model $(MODEL)

dump-ob:
	$(PYTHON) tools/dump_ob.py --suite

capture-suite-onnx:
	$(PYTHON) tools/capture_suite_onnx.py \
	  --workers $(CAPTURE_WORKERS) \
	  $(if $(CAPTURE_MODEL),--model $(CAPTURE_MODEL),) \
	  $(if $(CAPTURE_SMILES),--smiles '$(CAPTURE_SMILES)',) \
	  $(if $(FORCE),--force,)

gather-golden:
	$(PYTHON) tools/gather_golden_suite.py \
	  --workers $(GATHER_WORKERS) \
	  --failing-only \
	  --force

drift-report:
	-$(PYTHON) tools/report_suite_drift.py \
	  --workers $(DRIFT_WORKERS) \
	  $(if $(CAPTURE_MODEL),--model $(CAPTURE_MODEL),) \
	  $(if $(REFRESH),--refresh,)

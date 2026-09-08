# uv has no first-class task runner; this Makefile is the equivalent.
# Tools live under tools/ and are NOT installed with the wheel.

PYTHON ?= uv run python
PYTEST ?= uv run pytest
TOWNCRIER ?= uv run towncrier
CONVERT ?= uv run --group convert python
VERSION ?= $(shell $(PYTHON) -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")
IMAGE ?= dockerreg01.accounts.ad.wustl.edu/swamidass/xenosite-legacy:api
LEGACY_COMPOSE ?= tools/legacy-test-api/compose.yml
LEGACY_REPLICAS ?= 24
TARBALL ?= ../xenosite-legacy/data/xenosite_legacy_data_trimmed.tgz
ONNX_DIR ?= weights/onnx/v0
ONNX_TARBALL ?= weights/xenosite_onnx_v0.tgz

.PHONY: extract-weights convert-onnx convert-onnx-$(MODEL) pack-onnx extract-onnx download-onnx test test-golden test-live \
	legacy-test-api legacy-test-api-down py2-dump-image dump-ob dump-ob dump-ob-features \
	capture-suite-onnx gather-golden drift-report drift-descriptors help \
	regather-ob-dumps regather-golden-onnx changelog changelog-create changelog-release

help:
	@echo "extract-weights     copy pickles/TSV/source from $(IMAGE) into weights/legacy/"
	@echo "convert-onnx        pickle → safetensors → ONNX (all models)"
	@echo "convert-onnx MODEL=epoxidation"
	@echo "pack-onnx           tarball of *.onnx + *.meta.json (no _dump) → $(ONNX_TARBALL)"
	@echo "extract-onnx        unpack $(ONNX_TARBALL) into $(ONNX_DIR)/"
	@echo "download-onnx       fetch $$XENOSITE_ONNX_URL into $(ONNX_DIR)/"
	@echo "test                unit tests, Docker-free (-n auto via pyproject.toml)"
	@echo "test-golden         golden_descriptor_suite ONNX parity (-n auto)"
	@echo "test-live           pytest -m live (skips if Docker/image/weights missing)"
	@echo "py2-dump-image      build python:2.7-slim dump image (numpy + OpenBabel 2.4 + RDKit)"
	@echo "dump-ob             fill descriptor suite incrementally (skip dumps already present)"
	@echo "capture-suite-onnx  cache ONNX scores (CAPTURE_WORKERS=24 default; CAPTURE_MODEL/SMILES to filter)"
	@echo "gather-golden        regather failing suite rows from legacy-test-api (GATHER_WORKERS=24)"
	@echo "regather-ob-dumps    refresh quinone rows in ob_dumps from py3 legacy OMP port"
	@echo "regather-golden-onnx refresh golden scores from ONNX + GOLDEN_PARAMETER"
	@echo "drift-report        classify ONNX vs golden from cache (DRIFT_WORKERS=24 default)"
	@echo "drift-descriptors   cross-tab descriptor vs score drift for one model"
	@echo "legacy-test-api     nginx LB + cache, scale API with LEGACY_REPLICAS=24"
	@echo "legacy-test-api-down"
	@echo "changelog           preview CHANGELOG.md from changelog.d/ (towncrier --draft)"
	@echo "changelog-create    add a fragment: TYPE=added NAME=slug MSG='...'"
	@echo "changelog-release    fold fragments into CHANGELOG.md locally (optional; tags do this)"

extract-weights:
	$(PYTHON) tools/extract_weights.py --image $(IMAGE) --tarball $(TARBALL) --out weights/legacy

convert-onnx:
	$(CONVERT) tools/convert_onnx.py --src weights/legacy --out $(ONNX_DIR) $(if $(MODEL),--model $(MODEL),)

pack-onnx:
	$(PYTHON) tools/pack_onnx.py --src $(ONNX_DIR) --out $(ONNX_TARBALL)

extract-onnx:
	$(PYTHON) tools/pack_onnx.py --extract --src $(ONNX_DIR) --out $(ONNX_TARBALL)

download-onnx:
	$(PYTHON) -m xenosite.predict download --dest $(ONNX_DIR)

test:
	$(PYTEST) -m "not live"

test-golden:
	$(PYTEST) tests/test_golden_suite.py

test-live:
	$(PYTEST) -m live

legacy-test-api:
	docker compose -f $(LEGACY_COMPOSE) up --build -d --scale legacy-test-api=$(LEGACY_REPLICAS)

legacy-test-api-down:
	docker compose -f $(LEGACY_COMPOSE) down

py2-dump-image:
	docker build --platform linux/amd64 -t xenosite-predict-py2:dump tools/py2-dump

SMILES ?= O=C(C)Oc1ccccc1C(=O)O
MODEL ?= epoxidation

# Optional filters for capture-suite-onnx (empty = full golden suite)
CAPTURE_MODEL ?=
CAPTURE_SMILES ?=
CAPTURE_WORKERS ?= 24
DRIFT_WORKERS ?= 24
GATHER_WORKERS ?= 24
GATHER_MODEL ?=

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
	  --force \
	  $(if $(GATHER_MODEL),--models $(GATHER_MODEL),)

drift-report:
	-$(PYTHON) tools/report_suite_drift.py \
	  --workers $(DRIFT_WORKERS) \
	  $(if $(CAPTURE_MODEL),--model $(CAPTURE_MODEL),) \
	  $(if $(REFRESH),--refresh,)

DRIFT_MODEL ?= quinone

drift-descriptors:
	$(PYTHON) tools/analyze_descriptor_score_drift.py \
	  --model $(DRIFT_MODEL) \
	  --workers $(DRIFT_WORKERS)

regather-ob-dumps:
	$(PYTHON) tools/regather_internal_ob_dumps.py --models quinone

regather-golden-onnx:
	$(PYTHON) tools/regather_golden_from_onnx.py \
	  --workers $(GATHER_WORKERS) \
	  --force \
	  --include-smoke \
	  --models epoxidation,quinone,reactivity,ugt,ndealk,isozyme \
	  $(if $(GATHER_MODEL),--models $(GATHER_MODEL),)

# TYPE=added|changed|fixed|removed|deprecated|security
# NAME=slug without an issue ticket; MSG='user-facing sentence.'
# For a GitHub issue, use: uv run towncrier create --no-edit -c "..." 123.fixed.md
TYPE ?= changed
NAME ?= change
MSG ?=

changelog:
	$(TOWNCRIER) build --draft --version $(VERSION)

changelog-create:
	@test -n "$(MSG)" || { echo "usage: make changelog-create TYPE=added NAME=slug MSG='...'" >&2; exit 1; }
	$(TOWNCRIER) create --no-edit -c "$(MSG)" +$(NAME).$(TYPE).md

changelog-release:
	$(TOWNCRIER) build --yes --version $(VERSION)

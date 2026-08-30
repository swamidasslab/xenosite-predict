# xenosite.predict

Python 3.11+ RDKit + ONNX predictors for XenoSite. Dist name **xenosite-predict**; import **`xenosite.predict`** (PEP 420 namespace). This repo is nested under `xenosite-api` and gitignored there; origin is [github.com/swamidasslab/xenosite-predict](https://github.com/swamidasslab/xenosite-predict).

Publish **sdist only** (no wheel): OpenBabel **2.4.x** is a system/conda pin, not a pip package, and ONNX weights stay local (`weights/`, gitignored). `make test` is Docker-free and stays green without weights or OpenBabel (feature tests skip). `make test-live` skips if Docker, the legacy image, or ONNX files are missing. Do not commit model weights, pickles, or extracted `libridass/` trees.

This package is **not** wired into `xenosite-api` yet.

## User API

```python
from xenosite.predict import predict, list_models

mol = predict("O=C(C)Oc1ccccc1C(=O)O", model="epoxidation")
mol = predict("O=C(C)Oc1ccccc1C(=O)O", models=["epoxidation", ("ugt", "0")])
mol = predict(mol, models=["quinone"])  # append
list_models()  # what this process can actually run (backend-aware)
```

- **One molecule at a time** (no batch API).
- **Parse once** when several models run. Canonical SMILES is **non-isomeric** (`isomericSmiles=False`).
- **`models=`** is a name (default version `"0"`) or `(name, version)` pairs. Do not pass one version string for a whole list.
- **Indices** are 0-based RDKit atom/bond indices. Scores are floats (`atol=1e-4` in tests).
- **Name lookup is omitted.** Pass SMILES, not drug names.
- Import does **not** open ONNX, HTTP, or OpenBabel. Load on first use of that `(model, version)`. Callers never import `openbabel` / `pybel`.

### `predict(inp, model=..., models=..., backend=..., backends=..., env=...)`

| Arg | Meaning |
|---|---|
| `inp` | SMILES or an existing `Molecule` (results append) |
| `model` | Single name; ignored if `models` is set |
| `models` | `str` or `(name, version)` iterable |
| `backend` | Pin the whole call: `"onnx"`, `"http"`, `"legacy"`, a URL, or a backend object |
| `backends` | Per-`(name, version)` override (ONNX epoxidation + HTTP bioactivation) |
| `env` | Picker mapping; `None` uses `os.environ`. Tests clear `XENOSITE_*` |

### Return type (`Molecule`)

Ported from `xenosite-api` `types.py`: `smiles`, `atoms`, `bonds`, `results`. Result variants: `MolBondResult`, `MolAtomResult`, `MolAtomPairResult`, `AtomResult`, `BondResult`, `AtomBondResult`. Each result has `model` and `version`.

### `list_models()`

Returns dicts `{name, version, available, backend, reason, heads, two_stage, pipeline}` for **this process**, not a fictional union of every backend.

### Errors

`InvalidMolecule`, `UnknownModel`, `BackendNotConfigured`, `WeightsNotFound`, `ModelNotAvailable`, `OpenBabelNotAvailable`.

## Backends

Picker (explicit env wins; first match):

1. `XENOSITE_BACKEND` is an `http://` / `https://` URL → **HTTP** against that deployed **xenosite-api**. Optional `XENOSITE_API_KEY` as Bearer.
2. Else `XENOSITE_MODELS_WEIGHTS` → local **ONNX** directory.
3. Else auto-detect `./weights/onnx` if `*.onnx` exist → local ONNX.
4. Else raise `BackendNotConfigured`.

Live parity compares **ONNX vs the legacy test-API**, not vs production HTTP. Tests must pass `backend=` and must not inherit a developer shell (`XENOSITE_*` are cleared in `conftest.py`).

| Backend | Role |
|---|---|
| ONNX | Converted numpy-NN heads under `weights/onnx/<model>/<head>.onnx` |
| HTTP | `GET {origin}/v0/<model>?smiles=` (xenosite-api) |
| Legacy | Derived Docker test API (`POST /predict/<model>`, `POST /nn/<model>/<head>`) |

Per-model override: `predict(..., backends={("bioactivation", "0"): "http"})`.

## Built-in models (version `"0"`)

| Name | User results | Notes |
|---|---|---|
| `epoxidation` | `MolBondResult` | Two-stage: bond ONNX then mol ONNX (Top-N site scores). Averages two atom orderings. |
| `quinone` | `MolAtomPairResult` | Atom → pair → mol. Includes null-pair molecule `O=C(Br)C(F)(F)F`. |
| `reactivity` | four `MolAtomResult` (`reactivity.gsh` / `.protein` / `.cyanide` / `.dna`) | Two-stage atom then mol. |
| `ugt` | `AtomResult` | Internal OpenBabel topological + mol descriptors. No MOPAC/SmartCYP on the inference path. |
| `ndealk` | `BondResult` (HLM slice) | Same ONNX as isozyme. Check `CCCC1CCCNC1C=O` for off-by-1. |
| `isozyme` | ten `BondResult` (`isozyme.3a4`, … `isozyme.hlm`) | Production Flask uses **ndealk1** for `metabolism1`, not the MOPAC metabolism predictor. |
| `phase1` | five `AtomBondResult` | TF `molecularNN`. No TF at runtime; ONNX convert is stop-if-fails. |
| `bioactivation` | `MolAtomResult` + metabolites | **Pipeline last** (enumeration + other models), not a single ONNX. |

## Makefile (tools are not in the sdist)

```
make extract-weights          # Docker image or fallback tarball → weights/legacy/
make convert-onnx             # pickle → ONNX; MODEL=epoxidation for one model
make test                     # pytest -m "not live"  (no Docker; skips without OB)
make test-live                # pytest -m live; fixture skips if Docker/image missing
make py2-dump-image           # python:2.7-slim + numpy + Debian OpenBabel 2.4
make dump-ob                  # OpenBabel feature dump via that image (no WashU)
make legacy-test-api          # build/run derived test image
make legacy-test-api-down
```

Convert deps: `uv run --group convert`. Installed runtime: rdkit, numpy, onnxruntime, httpx, pydantic. **Internal** OpenBabel 2.4.x is required to run ONNX feature graphs (not part of the public API; not on PyPI for 3.x). No TensorFlow, pandas, or pickle at inference.

```
conda install -c conda-forge openbabel=2.4
```

OpenBabel 3.x will not match the trained nets. Do not vendor OpenBabel sources (GPL).

Populate pickles from `dockerreg01.accounts.ad.wustl.edu/swamidass/xenosite-legacy:api` (needs registry login) or the sibling tarball `xenosite-legacy/data/xenosite_legacy_data_trimmed.tgz`. `make convert-onnx` unpickles in a public **python:2.7-slim** dump image (`tools/py2-dump/`), not the WashU API image.

The same dump image is the OpenBabel **feature oracle**: Debian Buster `python-openbabel` 2.4.1 from archive.debian.org, running as `/usr/bin/python` (the image's `/usr/local` CPython cannot load the multiarch SWIG module). `make dump-ob` feeds an RDKit molblock so 1-based OB indices align with 0-based RDKit, and dumps BondTD/AtomTD/UGT/Heuristic rows from sibling `xenosite-legacy/src`. Dumps are regenerable and gitignored.

Public parse/canonicalize stays RDKit. Feature graphs call OpenBabel 2.4 internally. `tests/test_ob_features.py` compares host OpenBabel rows to those dumps at atol `1e-4` / rtol `0` (skip without OB or dumps). Frontend golden score tests stay xfailed until that comparison is clean and ONNX is fed from those rows. Random-vector tests prove ONNX == the pickled numpy NN only.

## Layout

```
src/xenosite/predict/   # user API (installed)
tools/                  # extract, convert, legacy-test-api (not in the wheel)
weights/                # local only — README + .gitignore committed
tests/                  # unit + @pytest.mark.live
docs/vendored-diffs.md  # NN/feature hashes, MOPAC/SmartCYP gate
```

## Development

```
uv sync --group dev
make test
```

Vendored-tree comparison (sibling checkout, not committed):

```
uv run python tools/compare_vendored.py --root ../xenosite-legacy/src/libridass
```

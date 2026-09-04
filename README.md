# xenosite.predict

Python 3.11+ RDKit + ONNX predictors for XenoSite. Dist name **xenosite-predict**; import **`xenosite.predict`** (PEP 420 namespace). Checkout as a sibling of `xenosite-api` and `xenosite-legacy`; origin is [github.com/swamidasslab/xenosite-predict](https://github.com/swamidasslab/xenosite-predict).

Publish **sdist only** (no wheel): ONNX weights stay local (`weights/`, gitignored). OpenBabel comes from PyPI (`uv add openbabel`, currently 3.2.x wheels). `make test` is Docker-free. Feature tests compare the installed OpenBabel to the committed `tests/fixtures/ob_dumps.json.gz` (Git LFS). `make test-live` skips if Docker, the legacy image, or ONNX files are missing. Do not commit model weights, pickles, or extracted `libridass/` trees.

This package is **not** wired into `xenosite-api` yet.

## User API

```python
from xenosite.predict import predict, predict_many, apredict, apredict_many, list_models

mol = predict("O=C(C)Oc1ccccc1C(=O)O", model="epoxidation")  # version "1"
mol = predict("O=C(C)Oc1ccccc1C(=O)O", models=["epoxidation", ("ugt", "1")])
mol = predict("O=C(C)Oc1ccccc1C(=O)O", models=[("epoxidation", "0")])  # v0 / legacy params
mol = predict(mol, models=["quinone"])  # append
list_models()  # what this process can actually run (backend-aware)

# Many molecules (process pool for ONNX; sync API)
mols = predict_many(smiles_list, model="ugt", workers=4)

# Async (event-loop friendly; same workers under the hood)
mol = await apredict(smi, model="ugt")
mols = await apredict_many(smiles_list, model="ugt", workers=4)
mols = await asyncio.gather(*[apredict(s, model="ugt") for s in smiles_list])
```

- **One molecule at a time** for ``predict`` / ``apredict`` (no multi-mol batch inside a single call).
- **Many molecules:** ``predict_many`` / ``apredict_many`` run each input independently in parallel.
- **Parse once** when several models run on one molecule. Canonical SMILES is **non-isomeric** (`isomericSmiles=False`). Atom/bond indices and score arrays use **canonical SMILES atom order**, not the input order.
- **`detailed=True`** adds atom/bond properties (`z`, `chrg`, `impHs`, `cipRank`, bond `order`) and `atoms.reordered` (original input atom indices in canonical SMILES order).
- **`models=`** is a name (default scoring version `"1"`) or `(name, version)` pairs. **`"1"`** uses updated scoring parameters (HTTP `/v1`). **`"0"`** uses legacy parameters that match golden fixtures and HTTP `/v0`. Do not pass one version string for a whole list.
- **Indices** are 0-based RDKit atom/bond indices. Scores are floats (`atol=1e-4` in tests).
- **Name lookup is omitted.** Pass SMILES, not drug names.
- Import does **not** open ONNX, HTTP, or OpenBabel. Load on first use of that `(model, version)`. Callers never import `openbabel` / `pybel`.
- First `predict()` downloads ONNX weights when `XENOSITE_ONNX_URL` is set and none are cached (an **INFO** line reports when they are found or downloaded). No separate `download_weights()` call is required.
- **Workers:** ONNX batch/async paths use a process pool (descriptor generation is CPU-bound; threads do not help). Set ``workers=`` or ``XENOSITE_WORKERS``. ``XENOSITE_ORT_INTRA_OP`` caps ORT threads per process under concurrency. ORT profiling is off (no ``:mem:.sess`` dumps). Set ``XENOSITE_ORT_PROFILE`` to a real file path to write an ORT profile.
- **Scoring versions:** `predict(..., models=[("epoxidation", "0")])` is v0/legacy parameters; omit the version or pass `"1"` for the updated mapping. Same ONNX weights. **Score impact summary:** [`docs/legacy-vs-principled.md`](docs/legacy-vs-principled.md#expected-score-impact-production-vs-legacy). Walkthrough: `tests/v0_legacy/test_legacy_vs_principled_guide.py`. `_parameter` overlays individual flags for tests.

### `predict_many` / `apredict` / `apredict_many`

| Helper | Meaning |
|---|---|
| `predict_many(inputs, …, workers=…)` | Sync batch: one molecule per input, process pool for ONNX |
| `apredict(inp, …)` | Async single molecule (offloads to the shared pool) |
| `apredict_many(inputs, …)` | Async batch (same workers as `predict_many`) |

`workers` defaults to CPU count (`XENOSITE_WORKERS` overrides). New models reuse the existing `predict` / runner path — no per-model async code.

### `predict(inp, model=..., models=..., backend=..., backends=..., env=..., detailed=...)`

| Arg | Meaning |
|---|---|
| `inp` | SMILES or an existing `Molecule` (results append) |
| `model` | Single name; ignored if `models` is set |
| `models` | `str` or `(name, version)` iterable. `"1"` (default) = updated params; `"0"` = legacy / `/v0` |
| `backend` | Pin the whole call: `"onnx"`, `"http"`, `"legacy"`, a URL, or a backend object |
| `backends` | Per-`(name, version)` override (ONNX epoxidation + HTTP bioactivation) |
| `env` | Picker mapping; `None` uses `os.environ`. Tests clear `XENOSITE_*` |
| `detailed` | When `True`, fill atom/bond properties and `atoms.reordered` (input → canonical map) |

### Return type (`Molecule`)

Ported from `xenosite-api` `types.py`: `smiles`, `atoms`, `bonds`, `results`. Result variants: `MolBondResult`, `MolAtomResult`, `MolAtomPairResult`, `AtomResult`, `BondResult`, `AtomBondResult`. Each result has `model` and `model_version` (`"0"` or `"1"`, the scoring generation).

### `list_models()`

Returns dicts `{name, version, available, backend, reason, heads, two_stage, pipeline}` for **this process**, not a fictional union of every backend.

### ONNX weights

ONNX graphs are not in the sdist. Set `XENOSITE_ONNX_URL` to an https tarball
or a local `.tgz` path (the URL is not stored in this repo). The first
`predict()` (or `list_models()`) downloads into `$XDG_CACHE_HOME/xenosite/onnx/v0`
(or `~/.cache/xenosite/onnx/v0`, or `XENOSITE_MODELS_WEIGHTS` if set) and prints
an INFO line when weights are found or downloaded. Download logs and errors
never echo the URL (so a private weight location does not leak via stderr or
tracebacks). Tests that pass `env={}` never fetch. `python -m xenosite.predict download` and `make download-onnx`
are optional pre-fetch helpers.

### Errors

`InvalidMolecule`, `UnknownModel`, `BackendNotConfigured`, `WeightsNotFound`, `WeightsDownloadError`, `ModelNotAvailable`, `OpenBabelNotAvailable`.

## Backends

Picker (explicit env wins; first match):

1. `XENOSITE_BACKEND` is an `http://` / `https://` URL → **HTTP** against that deployed **xenosite-api**. Optional `XENOSITE_API_KEY` as Bearer.
2. Else `XENOSITE_MODELS_WEIGHTS` → local **ONNX** directory.
3. Else auto-detect `./weights/onnx/v0` (or a flat `./weights/onnx` tree) → local ONNX.
4. Else user cache (`$XDG_CACHE_HOME/xenosite/onnx/v0`) if `*.onnx` exist.
5. Else, when `XENOSITE_ONNX_URL` is set in the process env, download that archive into the cache (INFO on found/download).
6. Else raise `BackendNotConfigured`.

Live parity compares **ONNX vs the legacy test-API**, not vs production HTTP. Tests must pass `backend=` and must not inherit a developer shell (`XENOSITE_*` are cleared in `conftest.py`).

| Backend | Role |
|---|---|
| ONNX | Converted numpy-NN heads under `weights/onnx/v0/<model>/<head>.onnx` |
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
| `phase1` | five `AtomBondResult` | TF `molecularNN` → ONNX (`site` + `mol`). Bond_and_LonePair descriptors + topology-group pooling. |
| `bioactivation` | `MolAtomResult` + metabolites | **Pipeline last** (enumeration + other models), not a single ONNX. |

## Makefile (tools are not in the sdist)

```
make extract-weights          # Docker image or fallback tarball → weights/legacy/
make convert-onnx             # pickle → ONNX; MODEL=epoxidation for one model
make pack-onnx                # weights/xenosite_onnx_v0.tgz (runtime graphs, no _dump)
make extract-onnx             # unpack that tarball into weights/onnx/v0/
make download-onnx            # fetch $XENOSITE_ONNX_URL into weights/onnx/v0/
make test                     # pytest -m "not live"  (no Docker)
make test-live                # pytest -m live; fixture skips if Docker/image missing
make py2-dump-image           # python:2.7-slim + numpy + Debian OpenBabel 2.4
make dump-ob                  # OpenBabel feature dump via that image (no WashU)
make legacy-test-api          # build/run derived test image
make legacy-test-api-down
```

Convert deps: `uv run --group convert`. Installed runtime: rdkit, openbabel (PyPI 3.2.x), numpy, onnxruntime, httpx, pydantic. OpenBabel is **internal** (not part of the public API). No TensorFlow, pandas, or pickle at inference.

The dump image remains the OpenBabel **2.4.1** feature oracle. Host inference uses the PyPI **3.2.x** wheel; `tests/test_ob_features.py` reports 3.x vs 2.4 drift at atol `1e-4` / rtol `0`. Do not vendor OpenBabel sources (GPL).

Populate pickles from `dockerreg01.accounts.ad.wustl.edu/swamidass/xenosite-legacy:api` (needs registry login) or the sibling tarball `xenosite-legacy/data/xenosite_legacy_data_trimmed.tgz`. `make convert-onnx` unpickles in a public **python:2.7-slim** dump image (`tools/py2-dump/`), not the WashU API image.

The same dump image is the OpenBabel **feature oracle**: Debian Buster `python-openbabel` 2.4.1 and `python-rdkit` from archive.debian.org, running as `/usr/bin/python` (the image's `/usr/local` CPython cannot load the multiarch SWIG module). `make dump-ob` feeds an RDKit molblock so 1-based OB indices align with 0-based RDKit, and dumps BondTD/AtomTD/UGT/Heuristic/Bond_and_LonePair rows from sibling `xenosite-legacy/src`. It is **idempotent**: molecule/model pairs already in the suite are skipped, and the JSON is checkpointed after each chunk. The gzipped suite `tests/fixtures/ob_dumps.json.gz` is committed via Git LFS so dump tests run without Docker; uncompressed JSON stays gitignored. Clone with Git LFS (`git lfs pull`).

Public parse/canonicalize stays RDKit. Feature graphs call OpenBabel internally (PyPI 3.2.x). `tests/test_ob_features.py` compares host OpenBabel 3.2 rows to 2.4 dumps at atol `1e-4` / rtol `0`. Missing dumps fail. Hypothesis draws random finite matrices for ONNX heads (`test_onnx_random_matrix_finite`) and live `/nn` vs ONNX (`test_random_vector_nn`). The convert dump `tests/fixtures/random_vectors.json` is the Python-2 regression (ONNX == pickled numpy NN).

## Layout

```
src/xenosite/predict/   # user API (installed)
tools/                  # extract, convert, legacy-test-api (not in the wheel)
weights/                # local only — README + .gitignore committed
tests/                  # unit + @pytest.mark.live
docs/vendored-diffs.md  # NN/feature hashes, MOPAC/SmartCYP gate
docs/legacy-vs-principled.md  # production defaults vs golden legacy modes
```

## Development

```
uv sync --group dev
make test
```

### Publishing to PyPI (trusted publishing)

No long-lived PyPI tokens. Releases use GitHub OIDC via `.github/workflows/publish.yml`.

1. On PyPI, add a **pending** trusted publisher (project not published yet) at
   [pypi.org/manage/account/publishing](https://pypi.org/manage/account/publishing/):
   - Project: `xenosite-predict`
   - Owner: `swamidasslab`
   - Repo: `xenosite-predict`
   - Workflow: `publish.yml`
   - Environment: `pypi`
2. In GitHub → Settings → Environments, create `pypi` (add required reviewers if you want a human gate).
3. Merge the workflow, then either push a tag `v0.2.0` or run **Publish** manually.
4. The first successful publish creates the PyPI project; later releases reuse the same publisher.

Do **not** commit `XENOSITE_ONNX_URL`, API keys, or weight hostnames. Keep those in local env / deployment secrets only.

Vendored-tree comparison (sibling checkout, not committed):

```
uv run python tools/compare_vendored.py --root ../xenosite-legacy/src/libridass
```

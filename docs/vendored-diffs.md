# Vendored tree diffs and MOPAC / SmartCYP gate

Compared against sibling [`xenosite-legacy/src/libridass`](../../xenosite-legacy/src/libridass) (not committed here). Refresh with:

```
uv run python tools/compare_vendored.py --root ../xenosite-legacy/src/libridass
```

Hashes are SHA-256 prefixes (16 hex chars) of the files as they existed when this document was written. Feature/NN code is **not** merged across models unless a variant is proven identical.

## NN packages

| Tree | `layers.py` | Notes |
|---|---|---|
| `epoxidation1/NN` | `3ed3e2c68458b944` | Full package (13 `.py` files) **identical** to ugt and ndealk |
| `ugt1/NN` | `3ed3e2c68458b944` | Identical to epoxidation NN tree |
| `ndealk1/NN` | `3ed3e2c68458b944` | Identical to epoxidation NN tree |
| `quinone1/NN` | `7c3a9de8cf327626` | Same `layers.py` as reactivity; extra `model/grouped.py` vs reactivity |
| `reactivity1/NN` | `7c3a9de8cf327626` | Same `layers.py` as quinone |
| `metabolism1/xenosite/NN` | `00289cd104eea6f7` | Distinct (third variant) |

**Three `layers.py` variants**, as expected. Converter walks the same layer types (`WindowedInputLayer`, `FullyConnectedLayer`, `LogisticLayer`, `CrossEntropyError`, `GaussianError`) on all of them; do not assume weight layouts are interchangeable without a dump.

`logARD.py`, `prior.py`, and `train.py` match across the numpy-NN trees (`c1fb9dd8a5a0e76e` / `409248cf8d9e1c94`). `model/models.py` matches for epo/ugt/ndealk/quinone/reactivity (`794969f74016c491`); metabolism’s copy differs.

**Merge decision:** share one ONNX converter. Do **not** share pickled weight files. Epoxidation/ugt/ndealk NN *source* is identical; feature graphs are not.

## Feature / descriptor trees

| Tree | File | Hash | Merge? |
|---|---|---|---|
| epoxidation `topological_descriptors/bond.py` | `a0f2451f59cd03ba` | **No** — differs from quinone |
| quinone `topological_descriptors/bond.py` | `6f80fe7536dca36b` | **No** |
| epoxidation `atom.py` | `2b9790f123cc56f1` | **No** |
| quinone `atom.py` | `f0e2090376351702` | **No** |
| reactivity `atom.py` | `69365b4465461865` | **No** — third atom copy |
| epoxidation `molecule.py` | `e320e3f1168adaa4` | **No** vs quinone `c7e7d024499c7f80` |
| epoxidation `atom_pair.py` | `0ae5e64a0ee9e218` | **No** vs quinone `f6bfe226f7baeac2` |
| ugt `xenosite/descriptor/topo.py` | `d601a80a969433fc` | Distinct UGT graph (67 atom columns + mol) |
| ndealk `scripts/bond_desc.py` | `8935509911b58116` | Own bond descriptors |
| ndealk `scripts/Heuristic_desc.py` | `aa7c314f86aaba2b` | N-dealk SMARTS heuristics |
| metabolism `descriptor/topo.py` | `b6a728a8ce747865` | Same hash as `top.py` (duplicate file) |
| metabolism `descriptor/mopac.py` | `4dc946d0943c9541` | MOPAC only on unused metabolism predictor |

Shared helpers in this package (types, canonicalize, adapters, float compare, ONNX session, backend picker, two-stage Top-N assembly) are **not** vendored copies.

Quinone **pair-head TSV** is only four columns after atom scores are joined (`Atom1_Pred`, `Atom2_Pred`, `AtomPair__Distance`, `AtomPair__Distance_Is_Odd`), not a full copy of atom descriptors.

Inference feature graphs live under `src/xenosite/predict/features/` and call **OpenBabel 2.4 internally**. The public API stays RDKit mols and 0-based indices. There is no RDKit chemistry dual path.

**OpenBabel oracle is `xenosite-predict-py2:dump`**, not the WashU image. Debian Buster `python-openbabel` 2.4.1 and `python-rdkit` are installed from `archive.debian.org` onto `/usr/bin/python` (the image’s `/usr/local` CPython cannot load the multiarch SWIG module). RDKit is for phase1 `Bond_and_LonePairTD` (`GetNOuterElecs`). `make dump-ob` writes `tests/fixtures/ob_dumps.json` and `ob_dumps.json.gz` incrementally (skips molecule/model pairs already present; `--force` to redo). RDKit molblock so 1-based OB indices align with 0-based RDKit. The gzipped suite is committed (Git LFS). Compare overlapping columns at atol `1e-4`, **rtol=0**. Do not loosen atol.

If a host-OB vs dump test fails, **do not ship that model** until fixed or an explicit exception is recorded here.

## Isozyme vs ndealk

Production Flask (`docker/legacy-backend/website.py`) sets:

```
"ndealk1": ndealk1.PyMolPredictor(),
"metabolism1": ndealk1.PyMolPredictor(),  # linked with ndealk1
```

So **isozyme (API `/v0/isozyme`) and ndealk share the same ndealk1 net**. User API still exposes two names: `ndealk` = HLM slice, `isozyme.*` = all ten heads. The unused `metabolism1.predictor.PyMolPredictor` (MOPAC + SmartCYP + kNN) is **not** the production path.

## MOPAC / SmartCYP inference gate (early)

Traced each model’s **inference** `PyMolPredictor.predict` / Flask wiring. “Yes” blocks shipping until the plan is revised.

| User-API model | MOPAC on inference? | SmartCYP on inference? | Internal OB features |
|---|---|---|---|
| epoxidation | **no** | **no** | BondTD (two atom orderings) |
| quinone | **no** | **no** | AtomTD (full EDG/EWG/OMP) |
| reactivity | **no** | **no** | AtomTD reduced set |
| ugt | **no** | **no** | `SmartCYPDescriptors` exists in `topo.py` `__main__` only; predictor uses `TopologicalDescriptors` + `MoleculeDescriptors` |
| ndealk | **no** | **no** | BondTD (`BondDesc__`) + Heuristic, join by unordered atom pair |
| isozyme | **no** (Flask uses ndealk1) | **no** | same as ndealk |
| metabolism1.predictor (unused) | **yes** | **yes** | **blocked** — not ported |
| phase1 | **no** in predictor.py | **no** | TF molecularNN → ONNX site+mol (no TF at runtime). SMILES needs Bond_and_LonePair (not ported). |
| bioactivation | **no** directly | **no** | pipeline; depends on phase1/APMP + reactivity etc. |

## Two-stage heads

Epoxidation, quinone, and reactivity **mol** heads take Top-N **site** scores as features. Conversion emits two (or more) ONNX graphs; inference runs site then mol. Random-vector tests must cover both stages.

## Phase1 / bioactivation

Phase1 is TensorFlow `molecularNN`. Bioactivation enumerates metabolites then scores paths. Neither is a drop-in numpy-NN export. ONNX convert **stops and documents** rather than adding TF to the installed package. Bioactivation ONNX mol/path heads without metabolite generation will not match; port last.

## Weights

Feature **names/order** live in ``name_tables.py`` after `make convert-onnx` reads TSV headers. Weight tensors stay in gitignored `weights/`.

### Extract and ONNX convert (2026-08-29)

- WashU registry requires `docker login dockerreg01.accounts.ad.wustl.edu` (DNS works; no basic auth in this environment).
- Fallback tarball contains pickles. Conversion uses a public `python:2.7-slim` (linux/amd64) dump image plus sibling `NN/` sources (OpenOpt stubbed). **Not** the WashU image.
- Numpy-NN heads converted and random-vector parity vs dumped py2 `model.output` holds at atol `1e-4` (typically `~1e-7` float32): epoxidation bond/mol, quinone atom/pair/mol, reactivity atom (AbutLayer) / mol, ugt atom, ndealk bond (10 isozyme heads).
- **Phase1 / bioactivation:** Phase1 TF1 pickles convert on the host to `weights/onnx/v0/phase1/{site,mol}.onnx` (windowed MLP; no TensorFlow at runtime). Bioactivation is still a metabolite pipeline, not one graph. HTTP/legacy backends still apply.
- Feature-name tables are inlined in `name_tables.py` (from TSV headers). N-dealk has no training TSV in the tarball; the aspirin OpenBabel dump supplied 386 ndealk bond columns (`Heuristic` + `BondDesc`).
- Host OpenBabel vs dump tests use `xenosite-predict-py2:dump` (Debian `python-openbabel` 2.4.1 and `python-rdkit` from archive.debian.org + sibling `xenosite-legacy/src`). Not the WashU API image and not the micromamba test-API. `make dump-ob` writes `tests/fixtures/ob_dumps.json` (gitignored) and `ob_dumps.json.gz` (committed via Git LFS). Compare rows by **atom identity** in the dump index (`1.5.10` = mol.atom1.atom2, 1-based), not OpenBabel bond-iterator order. Use **rtol=0**. Default atol is `1e-4`. Keep pybel's read-time Gasteiger charges; calling `OBChargeModel` `gasteiger` on 3.2 equalizes nitro oxygens and misses the dump by ~0.45.
- BondTD `NRings` counts **DFS back-edge cycles** (legacy `UndirectedGraph.cycles`), not OpenBabel SSSR. SSSR is still used for `MolGraph.cycles()`, AtomTD/UGT ring sizes, and quinone aromatic rings. Fusion atoms sit in extra perimeter cycles the dump counted; SSSR dropped them.
- Quinone ortho/meta/para has two modes via ``quinone_omp_mode`` on ``predict(..., _parameter=)``:
  **legacy** (golden tests): one BFS shortest path per pair, neighbors in sorted atom-index order (deterministic port of vendored ``shortest_path``; fixes CPython 2.7 ``set`` hash tie-break). **principled** (API default): mean of the OMP ring indicator over ``all_shortest_paths`` (same [0, 1] scale as legacy; identical when only one shortest path exists). The py2 dump oracle used hash-ordered BFS; quinone rows that disagree only on ``Ortho_``/``Meta_``/``Para_`` against that oracle are xfailed in ``test_ob_features`` (tier-1 oracle drift, not chemistry).

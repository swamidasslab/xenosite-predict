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

RDKit ports live under `src/xenosite/predict/features/`. They are **not** assumed equal to OpenBabel until `@pytest.mark.live` `test_rdkit_vs_ob_*` passes. Known likely drift (document, do not hide in `atol`):

- Gasteiger vs OpenBabel partial charges
- `pybel.calcdesc()` vs RDKit Crippen / TPSA / HBA1 vs HBA2
- `IsRotor`, `HasAlphaBetaUnsat`, explicit-H counts
- Periodic-table corrected radii

If a live RDKit vs OB test fails, **do not ship that model on RDKit** until fixed or an explicit exception is recorded here.

## Isozyme vs ndealk

Production Flask (`docker/legacy-backend/website.py`) sets:

```
"ndealk1": ndealk1.PyMolPredictor(),
"metabolism1": ndealk1.PyMolPredictor(),  # linked with ndealk1
```

So **isozyme (API `/v0/isozyme`) and ndealk share the same ndealk1 net**. User API still exposes two names: `ndealk` = HLM slice, `isozyme.*` = all ten heads. The unused `metabolism1.predictor.PyMolPredictor` (MOPAC + SmartCYP + kNN) is **not** the production path.

## MOPAC / SmartCYP inference gate (early)

Traced each model’s **inference** `PyMolPredictor.predict` / Flask wiring. “Yes” blocks RDKit-only until the plan is revised.

| User-API model | MOPAC on inference? | SmartCYP on inference? | RDKit-only |
|---|---|---|---|
| epoxidation | **no** | **no** | allowed (verify vs OB dumps) |
| quinone | **no** | **no** | allowed (verify vs OB dumps) |
| reactivity | **no** | **no** | allowed (verify vs OB dumps) |
| ugt | **no** | **no** | `SmartCYPDescriptors` exists in `topo.py` `__main__` only; predictor uses `TopologicalDescriptors` + `MoleculeDescriptors` |
| ndealk | **no** | **no** | bond_desc + Heuristic SMARTS |
| isozyme | **no** (Flask uses ndealk1) | **no** | same as ndealk |
| metabolism1.predictor (unused) | **yes** | **yes** | **blocked** — not ported |
| phase1 | **no** in predictor.py | **no** | TF molecularNN; convert is stop-if-fails |
| bioactivation | **no** directly | **no** | pipeline; depends on phase1/APMP + reactivity etc. |

## Two-stage heads

Epoxidation, quinone, and reactivity **mol** heads take Top-N **site** scores as features. Conversion emits two (or more) ONNX graphs; inference runs site then mol. Random-vector tests must cover both stages.

## Phase1 / bioactivation

Phase1 is TensorFlow `molecularNN`. Bioactivation enumerates metabolites then scores paths. Neither is a drop-in numpy-NN export. ONNX convert **stops and documents** rather than adding TF to the installed package. Bioactivation ONNX mol/path heads without metabolite generation will not match; port last.

## Weights

Feature **names/order** JSON may be committed next to Python modules after `make convert-onnx` reads TSV headers. Weight tensors stay in gitignored `weights/`.

### Extract (2026-08-29)

- WashU registry `dockerreg01.accounts.ad.wustl.edu` did not resolve (no VPN/DNS). No `xenosite-legacy:api` image locally.
- Fallback tarball `xenosite_legacy_data_trimmed.tgz` **does** contain pickles and TSV headers (epoxidation, quinone, reactivity, ugt, ndealk, phase1, bioactivation). Sibling `libridass` sources were copied for diffs only.
- ONNX export still needs Python 2.7 unpickling inside the legacy image. **No ONNX files were written** (not faked).
- Feature-name JSON for epo/quinone/reactivity/ugt was committed from those TSV headers. N-dealk has no training TSV in the tarball (`xval.tsv` only).

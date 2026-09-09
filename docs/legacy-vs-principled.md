# Legacy vs principled prediction modes

XenoSite ONNX predictors reproduce scores from the original TensorFlow 1 /
legacy-test-api stack closely enough for regression testing, but **scoring
version `"1"` (the `predict()` default, HTTP `/v1`) differs** from version
`"0"` in a few places where the legacy behavior was accidental, tie-breaking,
or tied to OpenBabel internals rather than chemical symmetry.

This document explains those differences, how to request each version, and
which models are affected.

## Scoring versions

| Version | How to call | Parameter defaults | Matches |
|---|---|---|---|
| `"1"` (default) | `predict(smi, model="epoxidation")` or `models=[("epoxidation", "1")]` | updated / principled / RDKit | HTTP `/v1` |
| `"0"` | `predict(smi, models=[("epoxidation", "0")])` | legacy site/OMP/symmetry/NRings | HTTP `/v0`, golden fixtures, legacy-test-api |

Same ONNX weights in both cases. `Result.model_version` is `"0"` or `"1"` to match.

Implementation: version `"1"` is `xenosite.predict.v1` (principled defaults). Version `"0"` wraps those runners (`xenosite.predict.v1.legacy.LegacyRunner`). Importing `xenosite.predict.v0` warns; `predict(..., models=[(name, "0")])` does not. Importing `xenosite.predict.v2` raises `NotImplementedError` (RDKit-first / multitask, not implemented).

`_parameter` overlays individual flags (tests and ablations only).

```python
from xenosite.predict import predict

# v1 — updated mapping (default)
mol = predict("c1ccc2ccccc2c1", models=["quinone"])

# v0 — legacy mapping
mol = predict("c1ccc2ccccc2c1", models=[("quinone", "0")])
```

## Quick reference

| `_parameter` key | Version `"1"` (default) | Version `"0"` / golden | Affects |
|---|---|---|---|
| `ndealk_site_mode` | `"principled"` | `"legacy"` | `ndealk`, `isozyme` |
| `quinone_omp_mode` | `"principled"` (max/any over tied shortest paths) | `"legacy"` (single sorted-BFS path) | `quinone` |
| `symmetry_group_mode` | `"rdkit"` | `"openbabel"` | `ndealk`, `isozyme`, `epoxidation` |
| `bond_nrings_mode` | `"principled"` | `"legacy"` | `epoxidation`, `ndealk`, `isozyme` |

Bundles live in `xenosite.predict.scoring` (`V0_PARAMETER`, `V1_PARAMETER`).
Test helpers in `tests/support.py`:

- `PRINCIPLED_PARAMETER` — explicit v1 flags (same scores as omitting `_parameter`).
- `GOLDEN_PARAMETER` — v0 flags; `test_golden*.py` still pass this on version `"1"` as an overlay.
- `golden_predict_kwargs(model)` — returns `{"_parameter": GOLDEN_PARAMETER}` for score models.

The public equivalent of `GOLDEN_PARAMETER` is `models=[(name, "0")]`.

## Why two modes exist

**Golden fixtures** (`tests/fixtures/golden_smiles.json`,
`golden_descriptor_suite.json`) were captured from the derived legacy-test-api
(Docker image running pickled TF1 graphs + OpenBabel 2.4 features). Those captures
encode historical quirks: one BFS shortest path for quinone OMP descriptors,
per-row ndealk site keys with orphan indices, and OpenBabel GID bond classes.
Request scoring version `"0"` (or overlay `GOLDEN_PARAMETER`) to match them.

**Version `"1"` / production ONNX** keeps the same neural-network weights and
feature columns but fixes grouping and descriptor tie-breaking so scores respect
RDKit topological symmetry and stable graph-theoretic definitions. ONNX outputs
are unchanged; only **how row scores map onto atom/bond vectors** differs in the
flagged code paths.

Golden tests therefore pass `GOLDEN_PARAMETER` (or `models=[(name, "0")]`) so
bitwise parity with fixtures holds. Application code should call `predict()` at
version `"1"` (the default) unless reproducing legacy numbers intentionally.

## Expected score impact (production vs legacy)

These numbers compare **version `"1"` (`predict()` default)** vs **version `"0"`
/ `GOLDEN_PARAMETER`** on the
327-molecule golden suite (`PARITY_ATOL = 0.005`, max absolute delta across
mol/atom/bond/pair heads). Same ONNX weights in both cases — differences are
descriptor tie-breaking and **how row scores map onto bond vectors**, not model
retraining.

| Model | Within atol | Median max Δ | P95 max Δ | Worst max Δ | Primary cause |
|---|---:|---:|---:|---:|---|
| `reactivity` | 100% | 0 | 0 | 0 | flags are no-ops |
| `ugt` | 100% | 0 | 0 | 0 | flags are no-ops |
| `phase1` | 100% | 0 | 0 | 0 | separate pipeline |
| `epoxidation` | ~83% | 0 | ~0.018 | ~0.039 | `bond_nrings_mode` (DFS vs RDKit ring counts on fused PAHs) |
| `quinone` | ~89% | ~0.008 | ~0.04 | ~0.64 | `quinone_omp_mode` (path tie-breaking for ortho/meta/para) |
| `ndealk` / `isozyme` | ~77% | ~0.0005 | ~0.34 | ~0.96 | `symmetry_group_mode` + `ndealk_site_mode` (RDKit pooling vs per-row legacy sites) |

Regenerate with `uv run python tools/study_legacy_vs_default.py --json docs/legacy_vs_default_parity.json`.
Flag-level attribution: `uv run python tools/study_fix_attribution.py`.

### What changes in practice

**Epoxidation** — Usually a small shift on polycyclic aromatics. Legacy DFS
back-edge ring counts can differ on fusion atoms; production uses RDKit
`NumAtomRings` per bond endpoint. Mol-level scores typically move by ≤4% on
outliers.

**Quinone** — OMP ortho/meta/para descriptors feed atom → pair → mol heads.
Production enumerates **all** tied shortest paths and sets the indicator to **1
if any path qualifies** (binary, deterministic). Legacy follows **one** sorted-BFS
path; when ties disagree, atom features (and mol scores) can diverge. Highly
hydroxylated / fused aromatics see the largest gaps. Pass `quinone_omp_mode="legacy"`
to match golden captures exactly. The older fractional average is available as
`quinone_omp_mode="mean"`.

**N-dealkylation / isozyme** — Largest production-vs-legacy gaps. Legacy emits
**one site score per BondTD row** and **does not broadcast** to RDKit-symmetric
sibling bonds — symmetric partners often stay at zero. Production **collapses**
rows per symmetry class, then **pools** the class mean onto **every** sibling,
which can activate bonds legacy left at zero (worst-case bond score delta ≈0.96).
This is post-ONNX score mapping, not different neural-network weights. To
approximate legacy site collapse without sibling broadcast, use
`ndealk_site_mode="principled"` and `symmetry_group_mode="openbabel"` (see
`tests/v0_legacy/test_legacy_vs_principled_guide.py` chapter 3).

**Unaffected models** — `reactivity`, `ugt`, and `phase1` scores are identical
with or without `GOLDEN_PARAMETER`.

### When to use version `"0"`

| Situation | Recommendation |
|---|---|
| New rankings, UI, or chemistry-facing APIs | Default version `"1"` (omit the version) |
| HTTP `/v1` | `models=[(name, "1")]` or omit version |
| HTTP `/v0` / historical clients | `models=[(name, "0")]` |
| Diffing against committed golden JSON | `models=[(name, "0")]` or `golden_predict_kwargs(model)` |
| Matching legacy-test-api Docker output | Version `"0"` or full `GOLDEN_PARAMETER` |
| Quinone only: closer to golden without full bundle | `quinone_omp_mode="legacy"` overlay |
| Ndealk: principled dedup but no sibling broadcast | `ndealk_site_mode="principled"`, `symmetry_group_mode="openbabel"` |

```python
from tests.support import GOLDEN_PARAMETER, golden_predict_kwargs

# Public v0 (legacy params, Result.model_version == "0")
predict(smiles, models=[("ndealk", "0")])

# Overlay on v1 (golden tests that still pass _parameter)
predict(smiles, models=["ndealk"], _parameter=GOLDEN_PARAMETER)
predict(smiles, **golden_predict_kwargs("quinone"))
```

## `ndealk_site_mode` (ndealk / isozyme)

Bond models run one ONNX score per BondTD feature row, then collapse rows into a
per-bond vector.

### Principled (default)

1. Assign each row a **symmetry class** (RDKit bond key by default; see
   `symmetry_group_mode` below).
2. Keep **one representative row per class** when building the site map
   (`ndealk_site_from_row_scores(..., mode="principled")`).
3. When `symmetry_group_mode="rdkit"`, **pool** scores within the class: bonds
   that received different ONNX values (different descriptors in the same RDKit
   class) are set to the **mean of active scores**; when only one bond is active
   after dedup, all siblings receive that score (`NdealkFamily.symmetrize_bond_scores`).

Symmetric bonds therefore share one score. Fewer distinct site keys appear than
BondTD rows because duplicates are deduplicated before mapping to RDKit bonds.

### Legacy (golden)

1. Emit a site key **per BondTD row**, mirroring `prediction_df_to_dict` in the
   legacy frontend.
2. For duplicate OpenBabel topo-GID classes, sometimes use synthetic orphan keys
   (`max(site)+1`) so the resulting bond vector matches historical captures even
   when the frozenset does not map cleanly to an RDKit bond index.
3. **No RDKit symmetry pooling** after mapping — scores are **not** copied to
   symmetric sibling bonds. Within a RDKit bond symmetry class you typically see
   **one non-zero site** (or zero), with siblings left at zero.

Golden ndealk/isozyme bond vectors match legacy-test-api because of this path.
Production principled + RDKit pooling can **increase** the number of active bonds
and assign the **same** pooled score to all siblings in a class.

**Example SMILES where modes diverge:**  
`COc1ccc2nc(C)cc(NCCCN3CCOCC3)c2c1` (see `tests/test_legacy_vs_principled_guide.py`).

## `quinone_omp_mode` (quinone)

Quinone atom descriptors include ortho / meta / para (OMP) features computed from
shortest paths between atom pairs on an aromatic ring graph.

### Legacy

- Use **one** BFS shortest path between each atom pair (`MolGraph.shortest_path`).
- OMP ring indicator is `1.0` if **any** path qualifies.

When multiple equally short paths exist (common in fused aromatics like naphthalene),
legacy arbitrarily picks the path visit order from BFS neighbor sorting.

### Principled (default)

- Enumerate **all** shortest paths (`MolGraph.all_shortest_paths`).
- OMP indicator is **1.0 if any tied shortest path qualifies**, else 0.0
  (deterministic max over per-path `{0,1}` indicators; binary like legacy).

When only one shortest path exists, principled equals legacy. When several ties
exist, principled avoids arbitrary BFS tie-break: if any equal-length path
qualifies, the feature is on.

### Mean (optional: ``quinone_omp_mode="mean"``)

- Same path set as principled, but indicator is the **mean** of per-path values
  (fractional in `(0, 1)` when paths disagree). Closer to legacy on ~45% of golden
  molecules vs ~89% for principled max; use only if fractional OMP is desired.

**Example SMILES where modes diverge:**  
`c1ccc2ccccc2c1` (naphthalene).

Atom → pair → mol pipeline is otherwise identical; only OMP columns in the atom
feature matrix change.

## `symmetry_group_mode` (ndealk, isozyme, epoxidation)

Controls which bond equivalence classes are used for deduplication and score pooling.

### RDKit (default)

Classes come from RDKit `CanonicalRankAtoms` endpoint ranks plus bond order
(`rdkit_bond_symmetry_key` in `symmetry.py`). Grouping is independent of OpenBabel
GID and aligns with the RDKit atom/bond indices returned in `Molecule`.

**Ndealk / isozyme (principled site mode):** dedupe rows by class, then pool
within the class on the bond vector. When several bonds in the class carry
different raw scores (different BondTD descriptors), the pooled value is their
**mean**; when only one bond is active after site dedup, every sibling receives
that score.

**Epoxidation:** after averaging dual atom-ordering ONNX outputs and mapping to
the bond vector, pool within RDKit classes. Fused polycyclics (pyrene,
dibenzofuran, …) often yield slightly different scores per symmetric bond because
descriptor rows differ; pooling averages them so the class is internally consistent.

### OpenBabel (golden)

Classes use sorted OpenBabel **GID** ranks at bond endpoints (`ndealk_row_topo_gid_pair`),
matching legacy BondTD `BT` grouping.

**Ndealk principled + openbabel:** still dedupes by GID class but **does not**
RDKit-pool; sibling bonds in a coarse RDKit class may stay at zero while one
member carries the score (see `assert_bond_scores_openbabel_principled` in
`tests/rdkit_equiv.py`).

**Epoxidation + openbabel:** no pooling step; bond vectors match golden
fixtures that predate symmetry pooling.

## `bond_nrings_mode` (epoxidation / ndealk / isozyme)

BondTD exposes `Atom1_NRings` / `Atom2_NRings`: the number of rings containing
each **bond endpoint atom** (not the bond itself).

### Legacy (golden / ob dumps)

Uses `MolGraph.dfs_cycles()` (DFS back-edge cycles). Fusion atoms in polycyclic
aromatics can sit in extra perimeter cycles, so symmetric atoms get different
counts — this matches the OpenBabel 2.4 dump oracle.

### Principled (default)

Uses RDKit `RingInfo.NumAtomRings` for each BondTD endpoint (Atom1/Atom2 order
from `_index`). Counts are aromatization/SSSR-based and invariant within
directed OpenBabel bond classes `(GID(Atom1), GID(Atom2), bond order)`.

`tests/test_principled_descriptor_symmetry.py` asserts identical ONNX bond-head
columns within those directed classes. `tests/test_ob_features.py` keeps legacy
DFS counts for dump parity.

## Models not affected by these flags

| Model | Notes |
|---|---|
| `reactivity` | No site/OMP/symmetry flags; `GOLDEN_PARAMETER` is a no-op on scores. |
| `ugt` | Same. |
| `phase1` | Separate descriptor pipeline; not part of the three-flag bundle. |
| `bioactivation` | Pipeline model, not a single ONNX head. |

Epoxidation is affected by `symmetry_group_mode` and `bond_nrings_mode`.
Ndealk/isozyme also use `ndealk_site_mode`.

## Testing layout

| File | Role |
|---|---|
| `tests/test_golden.py`, `test_golden_suite.py` | Fixture parity at `PARITY_ATOL` with `GOLDEN_PARAMETER`. |
| `tests/test_onnx_principled.py` | Concise regressions: default == principled, default ≠ legacy on known molecules. |
| `tests/test_equiv_groups.py` | Global RDKit symmetry invariants on `ob_dumps` (production path). |
| `tests/test_principled_descriptor_symmetry.py` | Principled atom/bond descriptor identity within symmetry classes. |
| `tests/v0_legacy/test_legacy_vs_principled_guide.py` | **Readable walkthrough** of each flag with commented examples (human-first). |
| `tests/v0_legacy/test_bond_nrings_ablation.py` | Epoxidation NRings semantics and single-flag score attribution. |
| `tests/v0_legacy/test_quinone_omp_ablation.py` | Quinone OMP path aggregation vs legacy. |
| `tests/v0_legacy/test_ndealk_principled_ablation.py` | Ndealk site collapse + RDKit pooling vs legacy. |

## Choosing a mode

| Goal | Call |
|---|---|
| New application / updated chemistry | `predict(smiles, models=[...])` (version `"1"`) |
| HTTP `/v1` | version `"1"` (default) |
| HTTP `/v0` / golden JSON / legacy-test-api | `predict(..., models=[(name, "0")])` |
| Debug one flag | Pass only that key in `_parameter`; unspecified keys keep that version's defaults |

Do **not** regather golden fixtures when changing production defaults if golden
tests still pass `GOLDEN_PARAMETER` (legacy site/OMP/symmetry/NRings modes).

## Implementation map

| Concern | Primary module |
|---|---|
| Flag resolution | `symmetry.resolve_symmetry_group_mode`, runner `_ndealk_site_mode`, `_quinone_omp_mode` |
| RDKit score pooling | `symmetry.apply_bond_symmetry`, `BaseRunner.symmetrize_bond_scores` |
| Bond NRings | `features/bond.py` → `BondTD.add_nrings` |
| Directed OB bond class | `symmetry.directed_ob_bond_symmetry_key` |
| Ndealk site collapse | `features/bond.py` → `ndealk_site_from_row_scores` |
| Quinone OMP paths | `features/atom.py` → `_paths_for_omp`, `_omp_paths`; `features/molgraph.py` |
| Public API docs | `api.py` → `predict(..., models=[(name, version)])`; `scoring.py` |

# Legacy vs principled prediction modes

XenoSite ONNX predictors reproduce scores from the original TensorFlow 1 /
legacy-test-api stack closely enough for regression testing, but **production
`predict()` defaults deliberately differ** in a few places where the legacy
behavior was accidental, tie-breaking, or tied to OpenBabel internals rather than
chemical symmetry.

This document explains those differences, how to opt into legacy behavior, and
which models are affected.

## Quick reference

| `_parameter` key | Production default | Golden / parity value | Affects |
|---|---|---|---|
| `ndealk_site_mode` | `"principled"` | `"legacy"` | `ndealk`, `isozyme` |
| `quinone_omp_mode` | `"principled"` | `"legacy"` | `quinone` |
| `symmetry_group_mode` | `"rdkit"` | `"openbabel"` | `ndealk`, `isozyme`, `epoxidation` |

Pass options on the internal `_parameter` mapping (not exposed on the public HTTP
API today):

```python
from xenosite.predict import predict

# Production (defaults — no _parameter needed)
mol = predict("c1ccc2ccccc2c1", models=["quinone"])

# Legacy parity (golden tests, diffing against legacy-test-api captures)
mol = predict(
    "c1ccc2ccccc2c1",
    models=["quinone"],
    _parameter={
        "ndealk_site_mode": "legacy",
        "quinone_omp_mode": "legacy",
        "symmetry_group_mode": "openbabel",
    },
)
```

Test helpers in `tests/support.py` define the merged bundles:

- `PRINCIPLED_PARAMETER` — explicit production defaults (same as omitting `_parameter`).
- `GOLDEN_PARAMETER` — legacy site/OMP/symmetry modes used by `test_golden*.py`.
- `golden_predict_kwargs(model)` — returns `{"_parameter": GOLDEN_PARAMETER}` for score models.

## Why two modes exist

**Golden fixtures** (`tests/fixtures/golden_smiles.json`,
`golden_descriptor_suite.json`) were captured from the derived legacy-test-api
(Docker image running pickled TF1 graphs + OpenBabel 2.4 features). Those captures
encode historical quirks: one BFS shortest path for quinone OMP descriptors,
per-row ndealk site keys with orphan indices, and OpenBabel GID bond classes.

**Production ONNX** keeps the same neural-network weights and feature columns but
fixes grouping and descriptor tie-breaking so scores respect RDKit topological
symmetry and stable graph-theoretic definitions. ONNX outputs are unchanged; only
**how row scores map onto atom/bond vectors** differs in the flagged code paths.

Golden tests therefore pass `GOLDEN_PARAMETER` so bitwise parity with fixtures
holds. Application code should call `predict()` without `_parameter` unless
reproducing legacy numbers intentionally.

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
3. No RDKit symmetry pooling after mapping.

Golden ndealk/isozyme bond vectors match legacy-test-api because of this path.
Production principled mode can assign non-zero scores to fewer bonds on molecules
where legacy kept duplicate keys.

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
- OMP indicator is the **mean** of per-path `{0,1}` values.

When only one shortest path exists, principled equals legacy. When several ties
exist, principled averages ambiguity instead of picking one BFS branch.

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

## Models not affected by these flags

| Model | Notes |
|---|---|
| `reactivity` | No site/OMP/symmetry flags; `GOLDEN_PARAMETER` is a no-op on scores. |
| `ugt` | Same. |
| `phase1` | Separate descriptor pipeline; not part of the three-flag bundle. |
| `bioactivation` | Pipeline model, not a single ONNX head. |

Epoxidation is affected only by `symmetry_group_mode` (not ndealk/quinone flags).

## Testing layout

| File | Role |
|---|---|
| `tests/test_golden.py`, `test_golden_suite.py` | Fixture parity at `PARITY_ATOL` with `GOLDEN_PARAMETER`. |
| `tests/test_onnx_principled.py` | Concise regressions: default == principled, default ≠ legacy on known molecules. |
| `tests/test_equiv_groups.py` | Global RDKit symmetry invariants on `ob_dumps` (production path). |
| `tests/test_legacy_vs_principled_guide.py` | **Readable walkthrough** of each flag with commented examples (human-first). |

## Choosing a mode

| Goal | Call |
|---|---|
| New application / principled chemistry | `predict(smiles, models=[...])` |
| Match committed golden JSON | `predict(..., **golden_predict_kwargs(model))` |
| Match legacy-test-api Docker output | Full `GOLDEN_PARAMETER` on score models |
| Debug one flag | Pass only that key in `_parameter`; unspecified keys keep production defaults |

Do **not** regather golden fixtures when changing production defaults if golden
tests still pass `symmetry_group_mode=openbabel` and legacy site/OMP modes —
epoxidation golden rows were captured without RDKit pooling.

## Implementation map

| Concern | Primary module |
|---|---|
| Flag resolution | `symmetry.resolve_symmetry_group_mode`, runner `_ndealk_site_mode`, `_quinone_omp_mode` |
| RDKit score pooling | `symmetry.apply_bond_symmetry`, `BaseRunner.symmetrize_bond_scores` |
| Ndealk site collapse | `features/bond.py` → `ndealk_site_from_row_scores` |
| Quinone OMP paths | `features/atom.py` → `_paths_for_omp`, `_omp_paths`; `features/molgraph.py` |
| Public API docs | `api.py` → `predict(..., _parameter=...)` docstring |

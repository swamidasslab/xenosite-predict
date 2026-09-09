# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This project uses [towncrier](https://towncrier.readthedocs.io/) for the next
release; fragments live in [`changelog.d/`](changelog.d/).

<!-- towncrier release notes start -->

## [0.3.3.dev12+g191c176f7.d20260908](https://github.com/swamidasslab/xenosite-predict/releases/tag/v0.3.3.dev12+g191c176f7.d20260908) - 2026-09-08

### Added

- Attach DNA and cyanide star adducts from a no-thiol glutathionation ruleset.
- Keep a Changelog with [towncrier](https://towncrier.readthedocs.io/). Package version comes from git tags ([hatch-vcs](https://github.com/ofek/hatch-vcs)); pushing a `v*` tag also compiles `changelog.d/` into `CHANGELOG.md` and opens a GitHub Release.
- Label conjugation dummy atoms in CXSMILES.

### Changed

- Always score on canonical detailed topology; `canonicalize` and `detailed` only change what is returned. Add reorder/strip helpers for presentation atom order.
- Default ONNX to many worker processes × one ORT intra-op thread (`workers=min(cpu_count, 32)`, `XENOSITE_ORT_INTRA_OP=1`).
- Make the HTTP backend async with msgpack responses, per-origin concurrency, backoff on 429/503/timeouts, and optional Bearer auth (`XENOSITE_API_KEY`).

### Fixed

- Reuse a single HTTP client across `predict_many` instead of opening a new event loop per molecule.


## [0.3.2](https://github.com/swamidasslab/xenosite-predict/releases/tag/v0.3.2) - 2026-09-04

### Added

- Keep in-process RDKit mols on `Molecule.rdkit` / `Metabolite.rdkit` when `rdkit=True`. No extra parse; omitted from JSON.

## [0.3.1](https://github.com/swamidasslab/xenosite-predict/releases/tag/v0.3.1) - 2026-09-04

### Added

- `detailed=True` keeps atom/bond properties (`z`, `chrg`, `impHs`, `cipRank`, bond `order`) and `atoms.reordered` (input atom indices in canonical SMILES order).

### Changed

- Atom/bond indices and score arrays use canonical SMILES atom order (`isomericSmiles=False`), not the input order.
- Disable ORT profiling unless `XENOSITE_ORT_PROFILE` names a real file (no `:mem:.sess` dumps).

## [0.3.0](https://github.com/swamidasslab/xenosite-predict/releases/tag/v0.3.0) - 2026-09-04

### Changed

- Rename `Result.version` to `model_version`.
- Scoring version `"0"` uses legacy parameters (HTTP `/v0`); `"1"` (default) uses updated params (HTTP `/v1`). Same ONNX weights.
- Emit UGT, GSH, and protein metabolites as star adducts.

## [0.2.2](https://github.com/swamidasslab/xenosite-predict/releases/tag/v0.2.2) - 2026-09-03

### Changed

- Emit each hydrolysis/UO cleavage fragment as its own metabolite instead of dropping extra fragments.

## [0.2.1](https://github.com/swamidasslab/xenosite-predict/releases/tag/v0.2.1) - 2026-09-03

### Added

- `predict_many`, `apredict`, and `apredict_many` (process pool for ONNX).
- Download ONNX weights from `XENOSITE_ONNX_URL` on first `predict()` / `list_models()`.
- GitHub Actions trusted publishing to PyPI (`publish.yml`). Download logs never echo the weight URL.

### Changed

- Scope runtime graphs under `weights/onnx/v0` and `xenosite_onnx_v0.tgz`.
- Publish both sdist and wheel on the `v*` tag workflow.

### Fixed

- Wheel build by inlining ONNX feature-name tables into the package.

## [0.2](https://github.com/swamidasslab/xenosite-predict/releases/tag/v0.2) - 2026-09-01

### Added

- Forest metabolites: `xenosite-forest` dependency, `predict(metabolites=)`, `predict(mapped_smiles=)`, and `map_idx` on enumerated structures.
- `_private` integration helpers for xenosite-api metabolite attachment, including model filter and forest support discovery.
- `pack-onnx` / `extract-onnx` for runtime graph tarballs.
- `bond_nrings_mode` (principled RDKit `RingInfo` vs legacy OpenBabel NRings) and `symmetry_group_mode` (RDKit vs OpenBabel bond classes).
- Document production vs legacy `_parameter` modes and expected score impact (`docs/legacy-vs-principled.md`).

### Changed

- Isolate legacy predictors and parity tests under `v0_legacy` with public API shims.
- Pool RDKit symmetry-class scores by mean in production runners.
- Use max-over-paths OMP as the principled quinone default.
- Max-merge opposite BondTD directions for ndealk and phase1.
- Keep ob-dump and golden paths on legacy bond NRings mode so fixtures stay comparable.

## [0.1.0](https://github.com/swamidasslab/xenosite-predict/releases/tag/v0.1.0) - 2026-09-01

### Added

- Public `xenosite.predict` API (`predict`, `list_models`) with RDKit parse/canonicalize and ONNX model runners.
- Built-in models: epoxidation, quinone, reactivity, ugt, ndealk, isozyme, phase1, bioactivation.
- Internal OpenBabel feature graphs (BondTD, AtomTD, UGT, Heuristic, Bond_and_LonePair) via the PyPI OpenBabel wheel; dump oracle for OpenBabel 2.4.
- Convert numpy-NN pickles to ONNX (and phase1 TF1 pickles to `site`/`mol` ONNX) without TensorFlow at inference.
- Extract-weights, convert-onnx, pack/extract tooling, and a derived legacy test-API for live parity.
- Golden descriptor suite, cached ONNX capture, and offline drift reporting vs legacy scores.
- `_parameter` overlays for principled vs legacy descriptor/scoring modes (quinone OMP, ndealk site mapping, bond NRings).

### Changed

- Wire inference to internal OpenBabel features and drop the RDKit chemistry path for those descriptors.
- Two-stage mol heads for suite models (epoxidation, quinone, reactivity).
- Principled quinone OMP uses mean over shortest paths; legacy path remains available via `_parameter`.

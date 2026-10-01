# Lab log

## 2026-09-30

- Tried migrating metabolite attach to `xenosite-forest` 0.10.1 (`resolve("xf:…")` + `ForestMol` / `metabolize`). Postponed: `ForestMol(rdkit_mol)` in `xenosite.forest.mol` does `Chem.MolToSmiles(spec)` (canonical) then re-parses, so heavy-atom order / site frame no longer match the predict RDKit mol. Need a library fix that ingests RDKit (or order-preserving SMILES) without reindexing before predict can switch.

## 2026-09-20

- Capped `xenosite-forest` at `>=0.6.0,<0.7`. Forest 0.7.2 epoxidation rewrites sulfonamide, sulfate, thiazole, and nitro groups into charged forms and drops some legacy arene oxides. Stay on 0.6 until that is fixed.

## 2026-09-13

- N-dealkylation: xenosite UI filtered non-N forest products. Upstream: forest `NDealkylation`/`ND` (post-filter + no-N short-circuit); predict maps `ndealk`/`isozyme.*` → `ND`, zeros non-N bond scores, skips ONNX when molecule has no nitrogen.

# Lab log

## 2026-09-20

- Capped `xenosite-forest` at `>=0.6.0,<0.7`. Forest 0.7.2 epoxidation rewrites sulfonamide, sulfate, thiazole, and nitro groups into charged forms and drops some legacy arene oxides. Stay on 0.6 until that is fixed.

## 2026-09-13

- N-dealkylation: xenosite UI filtered non-N forest products. Upstream: forest `NDealkylation`/`ND` (post-filter + no-N short-circuit); predict maps `ndealk`/`isozyme.*` → `ND`, zeros non-N bond scores, skips ONNX when molecule has no nitrogen.

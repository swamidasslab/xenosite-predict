# Lab log

## 2026-09-13

- N-dealkylation: xenosite UI filtered non-N forest products. Upstream: forest `NDealkylation`/`ND` (post-filter + no-N short-circuit); predict maps `ndealk`/`isozyme.*` → `ND`, zeros non-N bond scores, skips ONNX when molecule has no nitrogen.

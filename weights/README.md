# Local weights (not in git)

This directory holds **local-only** artifacts. Nothing here except this README and `.gitignore` is committed.

## Layout

```
weights/
  onnx/<model>/<head>.onnx     # converted inference graphs
  legacy/                      # extracted pickles, TSV headers, vendored NN/descriptor trees
```

## How to populate

From the package root:

```
make extract-weights    # copy from xenosite-legacy:api (or fallback tarball)
make convert-onnx       # pickle → safetensors → ONNX (needs the extract)
```

The Docker image is `dockerreg01.accounts.ad.wustl.edu/swamidass/xenosite-legacy:api`.
If the registry is unreachable, `make extract-weights` tries
`../xenosite-legacy/data/xenosite_legacy_data_trimmed.tgz` (sibling checkout).

ONNX files are **not** distributed with the wheel. `make test` stays green without them;
`make test-live` and ONNX tests skip when weights or Docker are missing.

Do not commit `.onnx`, `.safetensors`, pickles, or extracted `libridass/` / `NN/` / descriptor trees.

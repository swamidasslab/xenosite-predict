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
make pack-onnx          # weights/xenosite_onnx.tgz  (*.onnx + *.meta.json, no _dump)
make extract-onnx       # unpack that tarball into weights/onnx/
make download-onnx      # fetch $XENOSITE_ONNX_URL into weights/onnx/
```

Installed-package users should set `XENOSITE_ONNX_URL`. The first `predict()`
downloads into the user cache (`$XDG_CACHE_HOME/xenosite/onnx`) and prints an
INFO line when weights are found or downloaded. No manual download call is
required. `make download-onnx` pre-fetches into this checkout directory.

`make pack-onnx` is the runtime-weight tarball: `epoxidation/bond.onnx` and friends.
It omits `_dump/` (Python-2 pickle dump intermediates). Feature-name JSON stays in
the Python package. Inference only needs the `.onnx` files; `.meta.json` is packed
for tests/debugging.

The Docker image is `dockerreg01.accounts.ad.wustl.edu/swamidass/xenosite-legacy:api`.
If the registry is unreachable, `make extract-weights` tries
`../xenosite-legacy/data/xenosite_legacy_data_trimmed.tgz` (sibling checkout).

ONNX files are **not** distributed with the wheel. `make test` stays green without them;
`make test-live` and ONNX tests skip when weights or Docker are missing.

Do not commit `.onnx`, `.safetensors`, pickles, or extracted `libridass/` / `NN/` / descriptor trees.

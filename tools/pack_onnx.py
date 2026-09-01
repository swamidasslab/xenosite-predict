#!/usr/bin/env python3
"""Pack or unpack the ONNX files needed to run predictors.

Runtime uses ``weights/onnx/<model>/<head>.onnx``. Convert also writes
``*.meta.json`` next to each graph (I/O dims for tests); those are packed too.
``_dump/`` (pickle dump intermediates) is excluded.

Unpack into ``weights/onnx`` (or point ``XENOSITE_MODELS_WEIGHTS`` at the
extracted tree). Feature-name JSON lives in the Python package, not here.
"""

from __future__ import annotations

import argparse
import io
import tarfile
from pathlib import Path

DEFAULT_SRC = Path("weights/onnx")
DEFAULT_OUT = Path("weights/xenosite_onnx.tgz")
README = """XenoSite ONNX weights (inference graphs).

Layout: <model>/<head>.onnx  (+ optional <head>.meta.json)

Unpack into weights/onnx, or set XENOSITE_MODELS_WEIGHTS to this directory.
Feature names ship in the xenosite-predict package, not this tarball.
"""


def _is_pack_file(path: Path, src: Path) -> bool:
    rel = path.relative_to(src)
    if "_dump" in rel.parts or any(part.startswith(".") for part in rel.parts):
        return False
    if path.name.startswith("._") or path.name == ".DS_Store":
        return False
    return path.suffix == ".onnx" or path.name.endswith(".meta.json")


def collect(src: Path) -> list[Path]:
    if not src.is_dir():
        raise SystemExit(f"missing ONNX dir: {src}")
    files = [p for p in src.rglob("*") if p.is_file() and _is_pack_file(p, src)]
    files.sort()
    if not any(p.suffix == ".onnx" for p in files):
        raise SystemExit(f"no .onnx files under {src} (run make convert-onnx)")
    return files


def pack(src: Path, out: Path) -> None:
    files = collect(src)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tf:
        readme = tarfile.TarInfo("README.txt")
        data = README.encode("utf-8")
        readme.size = len(data)
        readme.mtime = 0
        readme.uid = readme.gid = 0
        readme.uname = readme.gname = ""
        tf.addfile(readme, fileobj=io.BytesIO(data))
        for path in files:
            info = tf.gettarinfo(path, arcname=str(path.relative_to(src)))
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with path.open("rb") as fh:
                tf.addfile(info, fh)
    n_onnx = sum(1 for p in files if p.suffix == ".onnx")
    print(
        f"wrote {out} ({n_onnx} onnx, {len(files) - n_onnx} meta, "
        f"{out.stat().st_size} bytes)"
    )


def extract(tarball: Path, dest: Path) -> None:
    from xenosite.predict.weights import extract_onnx_archive

    try:
        n = extract_onnx_archive(tarball, dest)
    except Exception as exc:
        raise SystemExit(str(exc)) from exc
    print(f"extracted {n} files from {tarball} -> {dest}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=DEFAULT_SRC, help="ONNX tree to pack")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="tarball path")
    p.add_argument(
        "--extract",
        action="store_true",
        help="unpack --out into --src (default pack)",
    )
    args = p.parse_args(argv)
    if args.extract:
        extract(args.out, args.src)
    else:
        pack(args.src, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


"""``python -m xenosite.predict download`` — fetch ONNX weights via env URL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .errors import WeightsDownloadError
from .weights import ENV_ONNX_URL, download_weights


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m xenosite.predict",
        description=(
            "Download ONNX inference graphs. The archive URL comes from "
            f"{ENV_ONNX_URL} unless --url is passed."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    dl = sub.add_parser("download", help="download and extract ONNX weights")
    dl.add_argument(
        "--url",
        default=None,
        help=f"tarball URL or local path (default: ${ENV_ONNX_URL})",
    )
    dl.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="extract directory (default: cache or $XENOSITE_MODELS_WEIGHTS)",
    )
    dl.add_argument(
        "--force",
        action="store_true",
        help="re-download even if *.onnx files are already present",
    )
    args = parser.parse_args(argv)
    if args.cmd == "download":
        try:
            dest = download_weights(url=args.url, dest=args.dest, force=args.force)
        except WeightsDownloadError as exc:
            print(exc, file=sys.stderr)
            return 1
        print(dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

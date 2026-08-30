#!/usr/bin/env python3
"""Copy pickles, TSV headers, and vendored source from the legacy image.

Writes to gitignored ``weights/legacy/``. Does not copy into ``src/``.
Falls back to a sibling tarball when Docker/registry is unavailable.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

DEFAULT_IMAGE = "dockerreg01.accounts.ad.wustl.edu/swamidass/xenosite-legacy:api"
DEFAULT_TARBALL = Path("../xenosite-legacy/data/xenosite_legacy_data_trimmed.tgz")
SIBLING_SRC = Path("../xenosite-legacy/src/libridass")

# Paths inside the image / extracted tree that conversion needs
WANT_GLOBS = (
    "**/models/**",
    "**/*.model",
    "**/*.pyp",
    "**/*.tsv",
    "**/NN/**/*.py",
    "**/topological_descriptors/**/*.py",
    "**/descriptor/**/*.py",
    "**/scripts/*desc*.py",
)


def docker_available() -> bool:
    try:
        subprocess.run(
            ["docker", "info"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def image_present(image: str) -> bool:
    r = subprocess.run(
        ["docker", "image", "inspect", image],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return r.returncode == 0


def extract_from_docker(image: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    cid = subprocess.check_output(
        ["docker", "create", image], text=True
    ).strip()
    try:
        # Typical install location of libridass inside the api image
        for src in (
            "/usr/local/lib/python2.7/site-packages/libridass",
            "/usr/lib/python2.7/site-packages/libridass",
            "/app/libridass",
            "/opt/libridass",
        ):
            r = subprocess.run(
                ["docker", "cp", f"{cid}:{src}", str(dest / "libridass")],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if r.returncode == 0:
                print(f"copied {src} -> {dest / 'libridass'}")
                return
        raise SystemExit(
            "docker cp could not find libridass in the image; "
            "inspect the image layout and update tools/extract_weights.py"
        )
    finally:
        subprocess.run(["docker", "rm", cid], stdout=subprocess.DEVNULL)


def extract_from_tarball(tarball: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:*") as tf:
        tf.extractall(dest / "tarball")
    print(f"extracted {tarball} -> {dest / 'tarball'}")


def copy_sibling_src(src: Path, dest: Path) -> None:
    """Reference copy of Python sources for the comparison script (not models)."""
    target = dest / "sibling_libridass"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(
        src,
        target,
        ignore=shutil.ignore_patterns("*.pyc", "__pycache__", "*.model", "*.pyp"),
    )
    print(f"copied sibling sources {src} -> {target}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--tarball", type=Path, default=DEFAULT_TARBALL)
    p.add_argument("--out", type=Path, default=Path("weights/legacy"))
    p.add_argument("--sibling", type=Path, default=SIBLING_SRC)
    args = p.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    marker = args.out / "README.txt"
    marker.write_text(
        "Gitignored extract. Do not commit this tree.\n"
        f"image={args.image}\n",
        encoding="utf-8",
    )

    pulled = False
    if docker_available() and image_present(args.image):
        extract_from_docker(args.image, args.out)
        pulled = True
    elif docker_available():
        print(f"image {args.image} not present locally; skip docker cp", file=sys.stderr)
    else:
        print("docker not available; skip image extract", file=sys.stderr)

    if args.tarball.is_file():
        extract_from_tarball(args.tarball, args.out)
        pulled = True
    else:
        print(f"tarball not found: {args.tarball}", file=sys.stderr)

    if args.sibling.is_dir():
        copy_sibling_src(args.sibling, args.out)

    if not pulled:
        print(
            "No pickles/TSV copied (Docker/image and tarball both missing). "
            "ONNX conversion and live tests will skip.",
            file=sys.stderr,
        )
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

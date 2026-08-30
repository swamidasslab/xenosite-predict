#!/usr/bin/env python3
"""Host driver: dump OpenBabel features via xenosite-predict-py2:dump.

Uses Debian python-openbabel 2.4 from archive.debian.org inside the dump
image. Does not need the WashU registry. Mounts sibling xenosite-legacy/src.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DUMP_IMAGE = os.environ.get("XENOSITE_PY2_DUMP_IMAGE", "xenosite-predict-py2:dump")
DUMP_DOCKERFILE = ROOT / "tools" / "py2-dump" / "Dockerfile"
HELPER = ROOT / "tools" / "py2-dump" / "dump_ob_features.py"
# Nested repo sits at …/xenosite/xenosite-api/xenosite-predict; sibling ML tree is
# …/xenosite/xenosite-legacy/src (libridass + xenosite.finger).
DEFAULT_SRC = ROOT.parent.parent / "xenosite-legacy" / "src"
MODELS = ("epoxidation", "quinone", "reactivity", "ugt", "ndealk")
GOLDEN = ROOT / "tests" / "fixtures" / "golden_smiles.json"
DESCRIPTOR = ROOT / "tests" / "fixtures" / "descriptor_smiles.json"
# Extra molecules not in the golden score fixture (ndealk C–N off-by-1 case).
EXTRA_SMILES = ("CCCC1CCCNC1C=O",)
ASPIRIN_OUT = ROOT / "tests" / "fixtures" / "ob_dump_aspirin.json"
SUITE_OUT = ROOT / "tests" / "fixtures" / "ob_dumps.json"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def ensure_dump_image() -> None:
    inspect = subprocess.run(
        ["docker", "image", "inspect", DUMP_IMAGE],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if inspect.returncode == 0:
        return
    print(f"building {DUMP_IMAGE} from {DUMP_DOCKERFILE}", file=sys.stderr)
    subprocess.run(
        [
            "docker",
            "build",
            "--platform",
            "linux/amd64",
            "-t",
            DUMP_IMAGE,
            str(DUMP_DOCKERFILE.parent),
        ],
        check=True,
    )


def collect_dump_smiles() -> list[str]:
    """Golden unique SMILES, extras, then the committed descriptor suite."""
    out: list[str] = []
    seen: set[str] = set()

    def add(s: str) -> None:
        if s and s not in seen:
            seen.add(s)
            out.append(s)

    if GOLDEN.is_file():
        for row in json.loads(GOLDEN.read_text(encoding="utf-8")):
            add(row.get("smiles") or "")
    for s in EXTRA_SMILES:
        add(s)
    if DESCRIPTOR.is_file():
        payload = json.loads(DESCRIPTOR.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("smiles") or payload.get("molecules") or []
        for s in payload:
            add(str(s))
    return out


def _rdkit_sdf(smiles: str) -> tuple[str, str]:
    """Canonical non-isomeric molblock so OpenBabel atom order matches RDKit."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse {smiles!r}")
    canonical = Chem.MolToSmiles(mol, isomericSmiles=False)
    mol = Chem.MolFromSmiles(canonical)
    if mol is None:
        raise ValueError(f"RDKit could not reparse {canonical!r}")
    return canonical, Chem.MolToMolBlock(mol)


def dump_one(smiles: str, model: str, src: Path, *, sdf_text: str | None = None) -> dict:
    import tempfile

    ensure_dump_image()
    extra_v: list[str] = []
    extra_args: list[str] = ["--smiles", smiles]
    tmp = None
    if sdf_text:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".sdf", delete=False)
        tmp.write(sdf_text)
        tmp.close()
        extra_v = ["-v", f"{tmp.name}:/work/mol.sdf:ro"]
        extra_args += ["--sdf", "/work/mol.sdf"]
    cmd = [
        "docker",
        "run",
        "--rm",
        "--platform",
        "linux/amd64",
        "--entrypoint",
        "/usr/bin/python",
        "-v",
        f"{src.resolve()}:/src:ro",
        "-v",
        f"{HELPER.resolve()}:/work/dump_ob_features.py:ro",
        *extra_v,
        "-e",
        "PYTHONPATH=/src",
        DUMP_IMAGE,
        "/work/dump_ob_features.py",
        *extra_args,
        "--model",
        model,
        "--src",
        "/src",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    finally:
        if tmp is not None:
            Path(tmp.name).unlink(missing_ok=True)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "")[-2000:]
        raise RuntimeError(f"dump {model} failed (exit {proc.returncode}):\n{err}")
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError(f"empty dump for {model}: {(proc.stderr or '')[-500:]}")
    return json.loads(lines[-1])


def dump_suite(smiles_list: list[str], src: Path) -> list[dict]:
    """Dump every model for each SMILES in one container (RDKit molblock SDFs)."""
    ensure_dump_image()
    td = tempfile.mkdtemp(prefix="ob-dump-")
    try:
        manifest = []
        for i, smi in enumerate(smiles_list):
            canonical, sdf_text = _rdkit_sdf(smi)
            sdf_name = f"{i:03d}.sdf"
            (Path(td) / sdf_name).write_text(sdf_text)
            manifest.append(
                {
                    "smiles": canonical,
                    "input_smiles": smi,
                    "sdf": f"/work/batch/{sdf_name}",
                }
            )
            print(f"queued {i + 1}/{len(smiles_list)} {canonical}", file=sys.stderr)
        (Path(td) / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        cmd = [
            "docker",
            "run",
            "--rm",
            "--platform",
            "linux/amd64",
            "--entrypoint",
            "/usr/bin/python",
            "-v",
            f"{src.resolve()}:/src:ro",
            "-v",
            f"{HELPER.resolve()}:/work/dump_ob_features.py:ro",
            "-v",
            f"{td}:/work/batch",
            "-e",
            "PYTHONPATH=/src",
            DUMP_IMAGE,
            "/work/dump_ob_features.py",
            "--batch",
            "/work/batch/manifest.json",
            "--src",
            "/src",
            "--out",
            "/work/batch/out.json",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        err_tail = (proc.stderr or proc.stdout or "")[-4000:]
        if proc.returncode != 0 and not (Path(td) / "out.json").is_file():
            raise RuntimeError(f"batch dump failed (exit {proc.returncode}):\n{err_tail}")
        if proc.stderr:
            sys.stderr.write(proc.stderr)
        data = json.loads((Path(td) / "out.json").read_text(encoding="utf-8"))
        errors = data.get("errors") or []
        if errors:
            detail = "\n".join(
                f"  {e.get('smiles')}: {e.get('error')}" for e in errors[:40]
            )
            print(
                f"batch dump failed for {len(errors)} molecule(s):\n{detail}",
                file=sys.stderr,
            )
        return list(data.get("molecules") or []), errors
    finally:
        shutil.rmtree(td, ignore_errors=True)


def dump_molecule(smiles: str, src: Path, *, from_smiles: bool = False) -> dict:
    """Dump every model for one SMILES in a single container run."""
    canonical = smiles
    sdf_text = None
    if not from_smiles:
        canonical, sdf_text = _rdkit_sdf(smiles)
    raw = dump_one(canonical, "all", src, sdf_text=sdf_text)
    models = raw.get("models") or {}
    return {
        "smiles": canonical,
        "input_smiles": smiles,
        "via_sdf": sdf_text is not None,
        "models": models,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smiles", default=None, help="one SMILES; ignored with --suite")
    p.add_argument("--model", default=None, help="one model; default all in one container")
    p.add_argument("--src", type=Path, default=DEFAULT_SRC)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--suite",
        action="store_true",
        help=f"Dump golden SMILES + extras to {SUITE_OUT.name} and {SUITE_OUT.name}.gz",
    )
    p.add_argument(
        "--from-smiles",
        action="store_true",
        help="Parse SMILES in OpenBabel (skip RDKit SDF). Default is RDKit molblock so indices match.",
    )
    args = p.parse_args(argv)
    if not args.src.is_dir():
        print(f"legacy src missing: {args.src}", file=sys.stderr)
        return 1
    if args.suite:
        smiles = collect_dump_smiles()
        print(f"dumping suite n={len(smiles)} …", file=sys.stderr)
        errors: list[dict] = []
        if args.from_smiles:
            molecules = [
                dump_molecule(smi, args.src, from_smiles=True) for smi in smiles
            ]
        else:
            molecules, errors = dump_suite(smiles, args.src)
        suite = {"molecules": molecules}
        dest = args.out or SUITE_OUT
        dest.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(suite)
        dest.write_text(text, encoding="utf-8")
        gz = dest.with_name(dest.name + ".gz")
        with gzip.open(gz, "wt", encoding="utf-8") as fh:
            fh.write(text)
        print("wrote", dest, "and", gz, "n=", len(molecules), file=sys.stderr)
        aspirin = next((m for m in molecules if m["smiles"] == ASPIRIN), None)
        if aspirin is not None:
            ASPIRIN_OUT.write_text(json.dumps(aspirin, indent=2) + "\n", encoding="utf-8")
            print("wrote", ASPIRIN_OUT, file=sys.stderr)
        if errors:
            return 1
        return 0

    smiles = args.smiles or "O=C(C)Oc1ccccc1C(=O)O"
    if args.model:
        canonical = smiles
        sdf_text = None
        if not args.from_smiles:
            canonical, sdf_text = _rdkit_sdf(smiles)
        print(f"dumping {args.model} …", file=sys.stderr)
        payload = dump_one(canonical, args.model, args.src, sdf_text=sdf_text)
        out = {
            "smiles": canonical,
            "input_smiles": smiles,
            "via_sdf": sdf_text is not None,
            "models": {args.model: payload},
        }
    else:
        print(f"dumping all models for {smiles} …", file=sys.stderr)
        out = dump_molecule(smiles, args.src, from_smiles=args.from_smiles)
    text = json.dumps(out, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print("wrote", args.out, file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

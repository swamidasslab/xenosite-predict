#!/usr/bin/env python3
"""Host driver: dump OpenBabel features via xenosite-predict-py2:dump.

Uses Debian python-openbabel 2.4 and python-rdkit from archive.debian.org
inside the dump image. Does not need the WashU registry. Mounts sibling
xenosite-legacy/src.
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
MODELS = ("epoxidation", "quinone", "reactivity", "ugt", "ndealk", "phase1")
GOLDEN = ROOT / "tests" / "fixtures" / "golden_smiles.json"
DESCRIPTOR = ROOT / "tests" / "fixtures" / "descriptor_smiles.json"
# Extra molecules not in the golden score fixture (ndealk C–N off-by-1 case).
EXTRA_SMILES = ("CCCC1CCCNC1C=O",)
ASPIRIN_OUT = ROOT / "tests" / "fixtures" / "ob_dump_aspirin.json"
SUITE_OUT = ROOT / "tests" / "fixtures" / "ob_dumps.json"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
DEFAULT_CHUNK = 25


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


def _lfs_pointer(path: Path) -> bool:
    if not path.is_file():
        return False
    return path.read_bytes()[:80].startswith(b"version https://git-lfs.github.com/spec/v1")


def _model_complete(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    cols = payload.get("columns")
    rows = payload.get("rows")
    return isinstance(cols, list) and isinstance(rows, list) and len(cols) > 0


def _index_molecules(molecules: list[dict]) -> dict[str, dict]:
    by: dict[str, dict] = {}
    for rec in molecules:
        for key in (rec.get("smiles"), rec.get("input_smiles")):
            if key:
                by[str(key)] = rec
    return by


def load_existing_suite(dest: Path) -> list[dict]:
    """Uncompressed JSON if present, else gzip. LFS pointers count as missing."""
    gz = dest.with_name(dest.name + ".gz")
    path: Path | None = None
    if dest.is_file() and not _lfs_pointer(dest):
        path = dest
    elif gz.is_file() and not _lfs_pointer(gz):
        path = gz
    if path is None:
        return []
    if path.name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            data = json.load(fh)
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    mols = data.get("molecules") if isinstance(data, dict) else data
    return list(mols or [])


def write_suite(dest: Path, molecules: list[dict]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps({"molecules": molecules})
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(dest)
    gz = dest.with_name(dest.name + ".gz")
    gztmp = Path(str(gz) + ".tmp")
    with gzip.open(gztmp, "wt", encoding="utf-8") as fh:
        fh.write(text)
    gztmp.replace(gz)
    aspirin = next((m for m in molecules if m.get("smiles") == ASPIRIN), None)
    if aspirin is not None:
        ASPIRIN_OUT.write_text(json.dumps(aspirin, indent=2) + "\n", encoding="utf-8")


def merge_dumped(molecules: list[dict], dumped: list[dict]) -> list[dict]:
    """Update existing records in place; append molecules the suite did not have."""
    by = _index_molecules(molecules)
    for fresh in dumped:
        smi = str(fresh.get("smiles") or "")
        rec = by.get(smi) or by.get(str(fresh.get("input_smiles") or ""))
        models = fresh.get("models") or {}
        if rec is None:
            rec = {
                "smiles": fresh.get("smiles"),
                "input_smiles": fresh.get("input_smiles") or fresh.get("smiles"),
                "via_sdf": fresh.get("via_sdf", True),
                "models": {},
            }
            molecules.append(rec)
            for key in (rec.get("smiles"), rec.get("input_smiles")):
                if key:
                    by[str(key)] = rec
        rec.setdefault("models", {})
        rec["models"].update(models)
        if "via_sdf" in fresh:
            rec["via_sdf"] = fresh["via_sdf"]
    return molecules


def plan_jobs(
    smiles_list: list[str],
    molecules: list[dict],
    wanted: tuple[str, ...] | list[str],
    *,
    force: bool,
) -> list[dict]:
    """Jobs for molecule/model pairs the suite does not already have."""
    by = _index_molecules(molecules)
    jobs: list[dict] = []
    for smi in smiles_list:
        rec = by.get(smi)
        sdf_text = None
        canonical = smi
        if rec is None:
            canonical, sdf_text = _rdkit_sdf(smi)
            rec = by.get(canonical)
        else:
            canonical = rec.get("smiles") or smi
        have = (rec or {}).get("models") or {}
        if force:
            missing = list(wanted)
        else:
            missing = [m for m in wanted if not _model_complete(have.get(m))]
        if not missing:
            continue
        if sdf_text is None:
            canonical, sdf_text = _rdkit_sdf(smi)
        jobs.append(
            {
                "smiles": canonical,
                "input_smiles": smi,
                "sdf_text": sdf_text,
                "models": missing,
            }
        )
    return jobs


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


def dump_batch(jobs: list[dict], src: Path) -> tuple[list[dict], list[dict], bool]:
    """Dump the given jobs in one container. Each job may list a subset of models.

    The third return value is True if the container was interrupted; partial
    ``out.json`` is still returned so the host can checkpoint.
    """
    if not jobs:
        return [], [], False
    ensure_dump_image()
    td = tempfile.mkdtemp(prefix="ob-dump-")
    outp = Path(td) / "out.json"

    def _read_out() -> tuple[list[dict], list[dict]]:
        if not outp.is_file():
            return [], []
        data = json.loads(outp.read_text(encoding="utf-8"))
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

    try:
        manifest = []
        for i, job in enumerate(jobs):
            sdf_name = f"{i:03d}.sdf"
            (Path(td) / sdf_name).write_text(job["sdf_text"])
            manifest.append(
                {
                    "smiles": job["smiles"],
                    "input_smiles": job.get("input_smiles") or job["smiles"],
                    "sdf": f"/work/batch/{sdf_name}",
                    "models": list(job.get("models") or MODELS),
                }
            )
            print(
                f"queued {i + 1}/{len(jobs)} {job['smiles']} models={manifest[-1]['models']}",
                file=sys.stderr,
            )
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
        interrupted = False
        proc = None
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
        except KeyboardInterrupt:
            interrupted = True
        if proc is not None and proc.stderr:
            sys.stderr.write(proc.stderr)
        dumped, errors = _read_out()
        if interrupted:
            return dumped, errors, True
        if not outp.is_file():
            err_tail = ((proc.stderr if proc else "") or (proc.stdout if proc else "") or "")[
                -4000:
            ]
            code = proc.returncode if proc is not None else "interrupt"
            raise RuntimeError(f"batch dump failed (exit {code}):\n{err_tail}")
        return dumped, errors, False
    finally:
        shutil.rmtree(td, ignore_errors=True)


def dump_suite(smiles_list: list[str], src: Path) -> tuple[list[dict], list[dict]]:
    """Dump every model for each SMILES (full redo; does not read the suite file)."""
    jobs = plan_jobs(smiles_list, [], MODELS, force=True)
    dumped, errors, _interrupted = dump_batch(jobs, src)
    return dumped, errors


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
    p.add_argument("--model", default=None, help="one model; with --suite, only fill that model")
    p.add_argument("--src", type=Path, default=DEFAULT_SRC)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--suite",
        action="store_true",
        help=f"Fill {SUITE_OUT.name} incrementally (skip molecule/model pairs already dumped)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-dump even when the suite already has that molecule/model",
    )
    p.add_argument(
        "--chunk",
        type=int,
        default=DEFAULT_CHUNK,
        help="Molecules per docker run before writing the suite (default %(default)s)",
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
        dest = args.out or SUITE_OUT
        wanted: tuple[str, ...] = (args.model,) if args.model else MODELS
        if args.model and args.model not in MODELS and args.model != "isozyme":
            print(f"unknown model {args.model}", file=sys.stderr)
            return 1
        if args.model == "isozyme":
            wanted = ("ndealk",)
        molecules = load_existing_suite(dest)
        jobs = plan_jobs(smiles, molecules, wanted, force=args.force)
        print(
            f"suite n={len(smiles)} have={len(molecules)} todo={len(jobs)} models={list(wanted)}",
            file=sys.stderr,
        )
        if not jobs:
            gz = dest.with_name(dest.name + ".gz")
            need_write = (
                bool(molecules)
                and (
                    not dest.is_file()
                    or _lfs_pointer(dest)
                    or not gz.is_file()
                    or _lfs_pointer(gz)
                )
            )
            if need_write:
                write_suite(dest, molecules)
                print("wrote", dest, "and", gz, "n=", len(molecules), file=sys.stderr)
            else:
                print("suite already complete", file=sys.stderr)
            return 0
        errors: list[dict] = []
        chunk = max(1, args.chunk)
        for i in range(0, len(jobs), chunk):
            batch = jobs[i : i + chunk]
            print(
                f"dumping chunk {i // chunk + 1}/{(len(jobs) + chunk - 1) // chunk} "
                f"({len(batch)} molecules) …",
                file=sys.stderr,
            )
            dumped, batch_errors, interrupted = dump_batch(batch, args.src)
            merge_dumped(molecules, dumped)
            errors.extend(batch_errors)
            write_suite(dest, molecules)
            print(
                "wrote",
                dest,
                "n=",
                len(molecules),
                "chunk_ok=",
                len(dumped),
                file=sys.stderr,
            )
            if interrupted:
                print("interrupted; suite checkpointed", dest, file=sys.stderr)
                return 1
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

# coding: utf-8
"""Dump OpenBabel feature rows from vendored libridass (Python 2.7 + pybel).

Does not import libridass.utils / RDKit. Plant namespace packages so
epoxidation1/quinone1/ugt1/ndealk1/reactivity1 __init__ files (which pull
RDKit) are never executed.

Usage (Debian /usr/bin/python inside xenosite-predict-py2:dump):

    /usr/bin/python dump_ob_features.py --smiles 'CC(=O)Oc1ccccc1C(=O)O' \\
        --model epoxidation --src /src
"""
from __future__ import print_function

import argparse
import json
import os
import sys
import types


def _stub_confargparse():
    sys.modules.setdefault("confargparse", types.ModuleType("confargparse"))


def ns_pkg(name, path):
    """Register a namespace package without running its __init__.py."""
    mod = types.ModuleType(name)
    mod.__path__ = [path]
    mod.__file__ = os.path.join(path, "__init__.py")
    sys.modules[name] = mod
    parent, _, child = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], child, mod)
    return mod


def plant_libridass(src):
    lib = os.path.join(src, "libridass")
    if lib not in sys.path:
        sys.path.insert(0, src)
    import libridass  # empty __init__; noqa: F401

    for pkg, rel in (
        ("libridass.epoxidation1", "epoxidation1"),
        ("libridass.epoxidation1.topological_descriptors", "epoxidation1/topological_descriptors"),
        ("libridass.quinone1", "quinone1"),
        ("libridass.quinone1.topological_descriptors", "quinone1/topological_descriptors"),
        ("libridass.reactivity1", "reactivity1"),
        ("libridass.reactivity1.topological_descriptors", "reactivity1/topological_descriptors"),
        ("libridass.ndealk1", "ndealk1"),
        ("libridass.ndealk1.scripts", "ndealk1/scripts"),
        ("libridass.ugt1", "ugt1"),
    ):
        ns_pkg(pkg, os.path.join(lib, rel.replace("/", os.sep)))
    # ndealk BondTD.lone_pairs uses pkg_resources.resource_filename(
    # 'topological_descriptors', 'NOuterElecs.pickle'). That package's
    # __init__ imports RDKit; register it empty so only the pickle is used.
    topo = os.path.join(src, "topological_descriptors")
    if os.path.isdir(topo):
        ns_pkg("topological_descriptors", topo)


def _cell(v):
    try:
        import pandas as pd
        if v is None or (isinstance(v, float) and v != v):
            return 0.0
        if pd.isna(v):
            return 0.0
    except Exception:
        pass
    if hasattr(v, "item"):
        try:
            v = v.item()
        except Exception:
            pass
    try:
        int_types = (int, long)  # noqa: F821  (Python 2)
    except NameError:
        int_types = (int,)
    if isinstance(v, int_types + (float, str, bool)) or v is None:
        return v
    return str(v)


def df_to_dump(df):
    df = df.fillna(0)
    cols = [str(c) for c in df.columns.tolist()]
    rows = []
    for rec in df.itertuples(index=False):
        rows.append([_cell(v) for v in rec])
    index = []
    for ix in df.index.tolist():
        if isinstance(ix, tuple):
            index.append(".".join(str(x) for x in ix))
        else:
            index.append(str(ix))
    return {"columns": cols, "rows": rows, "index": index}


def read_pymol(smiles, sdf_path=None):
    import pybel

    if sdf_path:
        mols = list(pybel.readfile("sdf", sdf_path))
        if not mols:
            raise ValueError("no molecule in SDF %s" % sdf_path)
        return mols[0]
    return pybel.readstring("smi", smiles)


def dump_epoxidation(mol):
    from libridass.epoxidation1.topological_descriptors.bond import BondTD

    B = BondTD(mol, original_atom_ordering=True)
    if B.broken:
        raise ValueError("BondTD marked molecule broken")
    return df_to_dump(B.run_bond_level())


def dump_quinone(mol):
    from libridass.quinone1.topological_descriptors.atom import AtomTD

    A = AtomTD(mol, molnum=1)
    if A.broken:
        raise ValueError("AtomTD marked molecule broken")
    return df_to_dump(A.run_atom_level())


def dump_reactivity(mol):
    from libridass.reactivity1.topological_descriptors.atom import AtomTD

    targets = (
        "MULTITARGET_Cyanide__MULTITARGET_DNA__MULTITARGET_GSH__MULTITARGET_Protein".split(
            "__"
        )
    )
    A = AtomTD(mol, input_target=targets, molnum=1)
    if A.broken:
        raise ValueError("AtomTD marked molecule broken")
    return df_to_dump(A.run_atom_level())


def dump_ugt(mol):
    from libridass.ugt1.xenosite.descriptor.topo import (
        MoleculeDescriptors,
        TopologicalDescriptors,
    )

    atom = TopologicalDescriptors()([mol])
    mold = MoleculeDescriptors()([mol])
    import pandas as pd

    df = pd.concat([atom, mold], join="inner", axis=1)
    return df_to_dump(df)


def _unordered_bond_key(ix):
    """Map ``mol.a.b`` to ``mol.min.max`` so Heuristic and BondTD rows join."""
    parts = [int(x) for x in str(ix).split(".")]
    if len(parts) < 3:
        return str(ix)
    moln, a, b = parts[0], parts[-2], parts[-1]
    lo, hi = (a, b) if a <= b else (b, a)
    return "%d.%d.%d" % (moln, lo, hi)


def dump_ndealk(mol):
    from libridass.ndealk1.scripts import Heuristic_desc, bond_desc

    B = bond_desc.BondTD(mol)
    if B.broken:
        raise ValueError("BondTD marked molecule broken")
    B_desc = B.run_bond_level()
    H = Heuristic_desc.Heuristic(mol).run()
    import pandas as pd

    # Heuristic index is begin/end (C–N forced C then N). BondTD is C–N or
    # min idx. Concat on raw index outer-joins and invents extra rows.
    H2 = H.copy()
    H2.index = [_unordered_bond_key(i) for i in H.index]
    orig_index = list(B_desc.index)
    B2 = B_desc.copy()
    B2.index = [_unordered_bond_key(i) for i in orig_index]
    joined = pd.concat([H2.reindex(B2.index), B2], axis=1)
    joined.index = orig_index
    return df_to_dump(joined)


DUMPERS = {
    "epoxidation": dump_epoxidation,
    "quinone": dump_quinone,
    "reactivity": dump_reactivity,
    "ugt": dump_ugt,
    "ndealk": dump_ndealk,
    "isozyme": dump_ndealk,
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smiles", default="")
    p.add_argument(
        "--sdf",
        default="",
        help="SDF path (RDKit MolBlock); preferred so atom indices match RDKit",
    )
    p.add_argument(
        "--model",
        default="all",
        help="one model, or 'all' (default) for every dumper except isozyme",
    )
    p.add_argument(
        "--src",
        default=os.environ.get("XENOSITE_LEGACY_SRC")
        or os.environ.get("LEGACY_SRC", "/src"),
    )
    p.add_argument("--out", default="-")
    args = p.parse_args(argv)
    if not args.smiles and not args.sdf:
        raise SystemExit("need --smiles or --sdf")

    _stub_confargparse()
    plant_libridass(args.src)
    mol = read_pymol(args.smiles, sdf_path=args.sdf or None)
    if args.model == "all":
        models = {}
        for name in ("epoxidation", "quinone", "reactivity", "ugt", "ndealk"):
            models[name] = DUMPERS[name](mol)
        payload = {
            "smiles": args.smiles,
            "model": "all",
            "from_sdf": bool(args.sdf),
            "models": models,
        }
    else:
        if args.model not in DUMPERS:
            raise SystemExit("unknown model %s" % args.model)
        payload = DUMPERS[args.model](mol)
        payload["smiles"] = args.smiles
        payload["model"] = args.model
        payload["from_sdf"] = bool(args.sdf)
    text = json.dumps(payload)
    if args.out == "-":
        sys.stdout.write(text)
        sys.stdout.write("\n")
    else:
        with open(args.out, "w") as fh:
            fh.write(text)
            fh.write("\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        sys.stderr.write("%s: %s\n" % (type(exc).__name__, exc))
        raise

# coding: utf-8
"""Dump OpenBabel feature rows from vendored libridass (Python 2.7 + pybel).

Does not import libridass.utils. Plant namespace packages so
epoxidation1/quinone1/ugt1/ndealk1/reactivity1 __init__ files are never
executed. Debian ``python-rdkit`` is available for Bond_and_LonePairTD.

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


class _Phase1Options(object):
    verbose = False
    add_weight = False
    several_target_columns = True
    input_target = (
        "StableOxygenation__UnstableOxygenation__Dehydrogenation__"
        "Reduction__Hydrolysis"
    )
    mask_hydrogen = False
    title = ""


def _phase1_prep(mol):
    """Match libridass.phase1.predictor.PyMolPredictor.read (add H, heavy then light)."""
    import openbabel
    import pybel

    mol.removeh()
    mol.addh()
    mol.convertdbonds()
    heavy = [x.idx for x in mol.atoms if not x.OBAtom.IsHydrogen()]
    light = [x.idx for x in mol.atoms if x.OBAtom.IsHydrogen()]
    mol.OBMol.RenumberAtoms(heavy + light)
    openbabel.obErrorLog.SetOutputLevel(0)
    mol = pybel.readstring("sdf", mol.write("sdf"))
    openbabel.obErrorLog.SetOutputLevel(1)
    return mol


def dump_phase1(mol):
    """Bond_and_LonePair + Possible_Sites into the same dump payload as other models."""
    from topological_descriptors.bond_and_lone_pair import Bond_and_LonePairTD
    from topological_descriptors.possible_site import Possible_Sites
    import pandas as pd

    mol = _phase1_prep(mol)
    options = _Phase1Options()
    B = Bond_and_LonePairTD(mol, options, molnum=1)
    if B.broken:
        raise ValueError("Bond_and_LonePairTD marked molecule broken")
    B_desc = B.run_bond_level()
    if B.broken or B_desc is None:
        raise ValueError("Bond_and_LonePairTD failed run_bond_level")
    PS = Possible_Sites(mol, options, molnum=1)
    PS_desc = PS.run()
    quinone = [
        "Quinone_Dehydrogenation",
        "Imine_Dehydrogenation",
        "QuinoneImine_Dehydrogenation",
        "QuinoneMethide_Dehydrogenation",
        "ImineMethide_Dehydrogenation",
    ]
    ester = ["Ester_Hydrolysis", "PEster_Hydrolysis", "HalogenEster_Hydrolysis"]
    PS_desc["Quinone_Dehydrogenation"] = PS_desc.apply(
        lambda x: max(x[quinone]), axis=1
    )
    PS_desc["Ester_Hydrolysis"] = PS_desc.apply(lambda x: max(x[ester]), axis=1)
    PS_desc = PS_desc.drop(
        ["Atom1_Index", "Atom2_Index"] + quinone[1:] + ester[1:],
        axis=1,
    )
    joined = pd.concat([B_desc, PS_desc], axis=1)
    return df_to_dump(joined)


def dump_ndealk(mol):
    from libridass.ndealk1.scripts import Heuristic_desc, bond_desc
    import pandas as pd

    B = bond_desc.BondTD(mol)
    if B.broken:
        raise ValueError("BondTD marked molecule broken")
    B_desc = B.run_bond_level()
    H = Heuristic_desc.Heuristic(mol).run()
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
    "phase1": dump_phase1,
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
    p.add_argument(
        "--batch",
        default="",
        help="JSON list of {smiles, sdf, input_smiles}; dump every job into --out",
    )
    p.add_argument("--out", default="-")
    args = p.parse_args(argv)
    if not args.batch and not args.smiles and not args.sdf:
        raise SystemExit("need --smiles, --sdf, or --batch")

    _stub_confargparse()
    plant_libridass(args.src)

    def dump_named_models(mol, names):
        models = {}
        errors = []
        for name in names:
            try:
                models[name] = DUMPERS[name](mol)
            except Exception as exc:
                errors.append("%s: %s: %s" % (name, type(exc).__name__, exc))
                sys.stderr.write("FAIL model %s: %s\n" % (name, exc))
        return models, errors

    def names_for_job(job):
        requested = None
        if job is not None:
            requested = job.get("models")
        if requested:
            names = list(requested)
        elif args.model not in ("all", "", None) and args.model in DUMPERS:
            names = [args.model]
        else:
            names = ["epoxidation", "quinone", "reactivity", "ugt", "ndealk", "phase1"]
        unknown = [n for n in names if n not in DUMPERS]
        if unknown:
            raise SystemExit("unknown model %s" % unknown[0])
        return names

    def write_out(payload):
        text = json.dumps(payload)
        if args.out == "-":
            sys.stdout.write(text)
            sys.stdout.write("\n")
        else:
            with open(args.out, "w") as fh:
                fh.write(text)
                fh.write("\n")

    if args.batch:
        jobs = json.load(open(args.batch))
        molecules = []
        errors = []
        n = len(jobs)
        for i, job in enumerate(jobs):
            smi = job.get("smiles") or ""
            sys.stderr.write("dumping %d/%d %s %s\n" % (i + 1, n, smi, job.get("models") or "all"))
            try:
                mol = read_pymol(smi, sdf_path=job.get("sdf") or None)
                dumped, merr = dump_named_models(mol, names_for_job(job))
                if dumped:
                    molecules.append(
                        {
                            "smiles": smi,
                            "input_smiles": job.get("input_smiles") or smi,
                            "via_sdf": bool(job.get("sdf")),
                            "models": dumped,
                        }
                    )
                if merr:
                    errors.append({"smiles": smi, "error": "; ".join(merr)})
            except Exception as exc:
                errors.append({"smiles": smi, "error": "%s: %s" % (type(exc).__name__, exc)})
                sys.stderr.write("FAIL %s: %s\n" % (smi, exc))
            write_out({"molecules": molecules, "errors": errors})
        return 1 if errors else 0

    mol = read_pymol(args.smiles, sdf_path=args.sdf or None)
    dumped, merr = dump_named_models(mol, names_for_job(None))
    if merr and args.model not in ("all", "", None):
        raise SystemExit(merr[0])
    if args.model == "all":
        payload = {
            "smiles": args.smiles,
            "model": "all",
            "from_sdf": bool(args.sdf),
            "models": dumped,
        }
    else:
        payload = dumped[args.model]
        payload["smiles"] = args.smiles
        payload["model"] = args.model
        payload["from_sdf"] = bool(args.sdf)
    write_out(payload)
    return 1 if merr else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        sys.stderr.write("%s: %s\n" % (type(exc).__name__, exc))
        raise

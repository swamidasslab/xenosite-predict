#!/usr/bin/env python3
"""Freeze bioactivation path/mol parity: TSV + styrene doctest X, py2 Y.

Writes ``tests/fixtures/bioactivation_parity.json``. Requires Docker image
``xenosite-predict-py2:dump`` (see ``tools/py2-dump/``).

  uv run python tools/gather_bioactivation_parity.py
"""

from __future__ import annotations

import csv
import io
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PARITY_OUT = ROOT / "tests" / "fixtures" / "bioactivation_parity.json"
PATH_TSV = ROOT / "weights" / "legacy" / "bioactivation1" / "data" / "path.tsv"
MOL_TSV = ROOT / "weights" / "legacy" / "bioactivation1" / "data" / "mol.tsv"
SIBLING = ROOT / "weights" / "legacy" / "sibling_libridass" / "bioactivation1"
DUMP_HELPER = ROOT / "tools" / "py2-dump" / "dump_nn_py2.py"
DUMP_IMAGE = "xenosite-predict-py2:dump"
N_TSV = 128

# Sibling doctest CSV anchors (path_descriptors / mol_descriptors).
STYRENE_PATH_CSV = """\
ID,Indicator__Epoxidation,Indicator__Furan,Indicator__NitrogenReduction,Indicator__QuinoneFormation,Indicator__SulfurOxidation,MolDesc__HBA1,MolDesc__HBA2,MolDesc__HBD,MolDesc__MR,MolDesc__MW,MolDesc__NumFragments,MolDesc__NumRings,MolDesc__TPSA,MolDesc__abonds,MolDesc__bonds,MolDesc__dbonds,MolDesc__heavy_atoms,MolDesc__hydrogens,MolDesc__logP,MolDesc__sbonds,MolDesc__tbonds,Score__Formation,Score__GSH_molecule_Bioactivation,Score__GSH_molecule_Reactivity_Delta,Score__GSH_ranked_atom_Bioactivation,Score__GSH_ranked_atom_Reactivity_Delta,Score__GSH_topological_atom_Bioactivation,Score__GSH_topological_atom_Reactivity_Delta,Score__Protein_molecule_Bioactivation,Score__Protein_molecule_Reactivity_Delta,Score__Protein_ranked_atom_Bioactivation,Score__Protein_ranked_atom_Reactivity_Delta,Score__Protein_topological_atom_Bioactivation,Score__Protein_topological_atom_Reactivity_Delta,TARGET
1.8.7_Epoxidation_C1=CC=C(C2CO2)C=C1,1,0,0,0,0,8.0,0.0,0.0,36.533,104.14912,1.0,1.0,0.0,6.0,16.0,1.0,8.0,8.0,2.3296,9.0,0.0,0.97819,0.07304,0.07466,0.13265,0.13561,0.56925,0.58194,0.28973,0.29619,0.28555,0.29191,0.28555,0.29191,0
1.1.2_QuinoneFormation_C=CC1=CC=CC(=O)C1=O,0,0,0,1,0,8.0,0.0,0.0,36.533,104.14912,1.0,1.0,0.0,6.0,16.0,1.0,8.0,8.0,2.3296,9.0,0.0,0.29604,0.0527,0.17802,-0.05851,-0.19764,0.13277,0.4485,0.02944,0.09945,0.04817,0.1627,0.17261,0.58307,0
1.1.4_QuinoneFormation_C=CC1=CC(=O)C=CC1=O,0,0,0,1,0,8.0,0.0,0.0,36.533,104.14912,1.0,1.0,0.0,6.0,16.0,1.0,8.0,8.0,2.3296,9.0,0.0,0.22707,0.04971,0.2189,-0.04488,-0.19764,0.10182,0.44842,0.03941,0.17356,0.03693,0.16265,0.13239,0.58301,0
1.3.4_Epoxidation_C=CC1=CC2OC2C=C1,1,0,0,0,0,8.0,0.0,0.0,36.533,104.14912,1.0,1.0,0.0,6.0,16.0,1.0,8.0,8.0,2.3296,9.0,0.0,0.43302,0.03903,0.09015,-0.06568,-0.15169,0.20381,0.47067,-0.02378,-0.05491,-0.03132,-0.07232,0.1023,0.23625,0
1.1.2_Epoxidation_C=CC1=CC=CC2OC12,1,0,0,0,0,8.0,0.0,0.0,36.533,104.14912,1.0,1.0,0.0,6.0,16.0,1.0,8.0,8.0,2.3296,9.0,0.0,0.32394,0.03234,0.09984,-0.03767,-0.1163,0.1916,0.59147,-0.00797,-0.0246,-0.02387,-0.07368,0.08301,0.25624,0
1.2.3_QuinoneFormation_C=CC1=CC(=O)C(=O)C=C1,0,0,0,1,0,8.0,0.0,0.0,36.533,104.14912,1.0,1.0,0.0,6.0,16.0,1.0,8.0,8.0,2.3296,9.0,0.0,0.06963,0.00986,0.14166,-0.01235,-0.17736,0.02766,0.39724,0.00234,0.03359,-0.00234,-0.03367,0.02706,0.38858,0
1.5.6_Epoxidation_C=CC12C=CC=CC1O2,1,0,0,0,0,8.0,0.0,0.0,36.533,104.14912,1.0,1.0,0.0,6.0,16.0,1.0,8.0,8.0,2.3296,9.0,0.0,0.01474,0.00153,0.10367,-0.00171,-0.11623,0.00834,0.5654,-0.00144,-0.09796,-0.00245,-0.16595,0.00344,0.23329,0
"""

STYRENE_MOL_CSV = """\
,MolDesc__HBA1,MolDesc__HBA2,MolDesc__HBD,MolDesc__MR,MolDesc__MW,MolDesc__NumRings,MolDesc__TPSA,MolDesc__abonds,MolDesc__bonds,MolDesc__dbonds,MolDesc__heavy_atoms,MolDesc__hydrogens,MolDesc__logP,MolDesc__sbonds,MolDesc__tbonds,PBS_1__logit,PBS_2__logit,PBS_3__logit,PBS_4__logit,PBS_5__logit,TARGET
1.1.1.,8.0,0.0,0.0,36.533,104.14912,1.0,0.0,6.0,16.0,1.0,8.0,8.0,2.3296,9.0,0.0,0.87431,-1.35741,-1.97952,-1.99987,-2.48289,0.0
"""


def _names(head: str) -> tuple[str, ...]:
    from xenosite.predict.v0_legacy.features import load_names

    return load_names("bioactivation", head)


def _tsv_rows(tsv: Path, names: tuple[str, ...], n: int) -> tuple[list[str], list[list[float]]]:
    ids: list[str] = []
    rows: list[list[float]] = []
    with tsv.open(newline="") as f:
        for i, row in enumerate(csv.DictReader(f, delimiter="\t")):
            ids.append(row.get("ID", ""))
            rows.append([float(row[c]) for c in names])
            if i + 1 >= n:
                break
    return ids, rows


def _csv_rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def _py2_y(pyp: str, x: list[list[float]]) -> list[float]:
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        x_path = tdp / "x.json"
        y_path = tdp / "y.json"
        x_path.write_text(json.dumps(x))
        cmd = [
            "docker",
            "run",
            "--rm",
            "--platform=linux/amd64",
            "--entrypoint",
            "python",
            "-v",
            f"{SIBLING}:/pkg:ro",
            "-v",
            f"{DUMP_HELPER}:/work/dump_nn_py2.py:ro",
            "-v",
            f"{tdp}:/x",
            DUMP_IMAGE,
            "/work/dump_nn_py2.py",
            f"/pkg/code/{pyp}",
            "/pkg",
            "/tmp/m.json",
            "/tmp/w.bin",
            "/x/x.json",
            "/x/y.json",
        ]
        subprocess.run(cmd, check=True)
        y = json.loads(y_path.read_text())["y"]
    arr = np.asarray(y, dtype=np.float64).reshape(-1)
    return [float(v) for v in arr]


def main() -> None:
    path_names = _names("path")
    mol_names = _names("mol")
    fixture: dict = {
        "notes": {
            "source": (
                "py2 dump via xenosite-predict-py2:dump on path.pyp/mol.pyp; "
                "x from training TSV or sibling doctest feature CSV"
            ),
            "atol": 1e-4,
            "TARGET": "training label — not used for parity",
        }
    }

    path_ids, path_x = _tsv_rows(PATH_TSV, path_names, N_TSV)
    fixture["path_tsv"] = {
        "head": "path",
        "names": list(path_names),
        "ids": path_ids,
        "x": path_x,
        "y": _py2_y("path.pyp", path_x),
    }

    mol_ids, mol_x = _tsv_rows(MOL_TSV, mol_names, N_TSV)
    fixture["mol_tsv"] = {
        "head": "mol",
        "names": list(mol_names),
        "ids": mol_ids,
        "x": mol_x,
        "y": _py2_y("mol.pyp", mol_x),
    }

    path_full = _csv_rows(STYRENE_PATH_CSV)
    styrene_path_x = [[float(r[c]) for c in path_names] for r in path_full]
    fixture["styrene_path"] = {
        "head": "path",
        "names": list(path_names),
        "ids": [r["ID"] for r in path_full],
        "x": styrene_path_x,
        "y": _py2_y("path.pyp", styrene_path_x),
        "descriptor_rows": path_full,
    }

    mol_full = _csv_rows(STYRENE_MOL_CSV)
    styrene_mol_x = [[float(mol_full[0][c]) for c in mol_names]]
    fixture["styrene_mol"] = {
        "head": "mol",
        "names": list(mol_names),
        "ids": ["1.1.1."],
        "x": styrene_mol_x,
        "y": _py2_y("mol.pyp", styrene_mol_x),
        "descriptor_rows": mol_full,
    }

    PARITY_OUT.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"wrote {PARITY_OUT}")


if __name__ == "__main__":
    main()

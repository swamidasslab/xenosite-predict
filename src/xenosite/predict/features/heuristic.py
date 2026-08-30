"""OpenBabel n-dealk Heuristic columns. Pandas-free.

Port of ``libridass.ndealk1.scripts.Heuristic_desc.Heuristic``. Bond order is
C then N when the bond is C–N, otherwise begin/end. Join onto BondTD by
unordered atom pair (Heuristic index is not BondTD's min-idx order).
"""

from __future__ import annotations

from . import _ob


SMARTS = {
    "methyl": "[$([CH3])]",
    "short_chain": (
        "[$([CX4H2][CH3]),$([CX4H1]([CH3])[CH3]),"
        "$([CX4H2][CH2][CH2][CH3]),$([CX4H2][CH2][CH3])]"
    ),
    "ring_c": "[$([C;R])]",
    "ring_n": "[$([N;R])]",
    "aro_c": "[$([c;R])]",
    "aro_n": "[$([n;R])]",
}


def _hits(pymol, smarts: str) -> set[int]:
    ob, _ = _ob.load()
    sp = ob.OBSmartsPattern()
    sp.Init(smarts)
    sp.Match(pymol.OBMol)
    return {x for y in sp.GetMapList() for x in y}


def heuristic_rows(pymol) -> list[dict]:
    ob, _ = _ob.load()
    heavy = [
        b
        for b in ob.OBMolBondIter(pymol.OBMol)
        if 1 not in (b.GetBeginAtom().GetAtomicNum(), b.GetEndAtom().GetAtomicNum())
    ]
    methyl = _hits(pymol, SMARTS["methyl"])
    short_chain = _hits(pymol, SMARTS["short_chain"])
    ring_c = _hits(pymol, SMARTS["ring_c"])
    ring_n = _hits(pymol, SMARTS["ring_n"])
    aro_c = _hits(pymol, SMARTS["aro_c"])
    aro_n = _hits(pymol, SMARTS["aro_n"])
    rows = []
    for b in heavy:
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        is_nc = 0
        ismethyl = isshort = in_nr = in_ar = 0
        if a1.IsCarbon() and a2.IsNitrogen():
            first, second, is_nc = a1, a2, 1
        elif a2.IsCarbon() and a1.IsNitrogen():
            first, second, is_nc = a2, a1, 1
        else:
            first, second = a1, a2
        if is_nc:
            if first.GetIdx() in methyl:
                ismethyl, is_nc = 1, 0
            if first.GetIdx() in short_chain:
                isshort, is_nc = 1, 0
            if first.GetIdx() in ring_c and second.GetIdx() in ring_n:
                in_nr, is_nc = 1, 0
            if first.GetIdx() in aro_c and second.GetIdx() in aro_n:
                in_ar, is_nc = 1, 0
        i, j = first.GetIdx(), second.GetIdx()
        rows.append(
            {
                "_atoms": (_ob.ob_idx_to_rdkit(i), _ob.ob_idx_to_rdkit(j)),
                "_index": f"1.{i}.{j}",
                "otherN_C": float(is_nc),
                "methyl": float(ismethyl),
                "short_chain": float(isshort),
                "in_ring_non_aromatic": float(in_nr),
                "in_ring_aromatic": float(in_ar),
            }
        )
    return rows

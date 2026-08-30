"""Internal OpenBabel 2.4 access. Not part of the public ``xenosite.predict`` API.

Importing this module does not load OpenBabel. Bindings load on first use.
Pin OpenBabel **2.4.x** (the stack the nets were trained on), not 3.x.
"""

from __future__ import annotations

from typing import Any

from ..errors import OpenBabelNotAvailable

_CACHE: tuple[Any, Any] | None = None


def load() -> tuple[Any, Any]:
    """Return ``(openbabel, pybel)`` using the 2.4 or 3.x import layout."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    try:
        import openbabel as ob  # type: ignore[import-untyped]
        try:
            import pybel  # type: ignore[import-untyped]
        except ImportError:
            from openbabel import pybel  # type: ignore[import-untyped]
        _CACHE = (ob, pybel)
        return _CACHE
    except ImportError:
        pass
    try:
        from openbabel import openbabel as ob  # type: ignore[import-untyped]
        from openbabel import pybel  # type: ignore[import-untyped]
        _CACHE = (ob, pybel)
        return _CACHE
    except ImportError as exc:
        raise OpenBabelNotAvailable(
            "OpenBabel 2.4 is required for XenoSite descriptors (internal only). "
            "Install a system or conda package, e.g. `conda install -c conda-forge openbabel=2.4`. "
            "OpenBabel 3.x will not match the trained nets. See the README."
        ) from exc


def installed() -> bool:
    """True if OpenBabel bindings are importable. Does not load them."""
    import importlib.util

    return (
        importlib.util.find_spec("openbabel") is not None
        or importlib.util.find_spec("pybel") is not None
    )


def available() -> bool:
    try:
        load()
        return True
    except OpenBabelNotAvailable:
        return False


def from_rdkit_mol(rdkit_mol) -> Any:
    """Parse an RDKit mol via molblock so OpenBabel atom order matches RDKit.

    OpenBabel ``GetIdx()`` is 1-based; RDKit is 0-based on the same order.
    """
    from rdkit import Chem

    sdf = Chem.MolToMolBlock(rdkit_mol)
    _ob, pybel = load()
    mol = pybel.readstring("mol", sdf)
    _assign_charges(mol)
    return mol


def _assign_charges(pymol) -> None:
    obmol = pymol.OBMol
    if hasattr(obmol, "AssignPartialCharges"):
        try:
            obmol.AssignPartialCharges()
            return
        except Exception:
            pass
    ob, _pybel = load()
    finder = getattr(getattr(ob, "OBChargeModel", None), "FindType", None)
    if finder is None:
        return
    model = finder("gasteiger")
    if model is not None:
        model.ComputeCharges(obmol)


def ob_idx_to_rdkit(ob_idx: int) -> int:
    """1-based OpenBabel atom index → 0-based RDKit (same molblock order)."""
    return int(ob_idx) - 1

"""Internal OpenBabel access. Not part of the public ``xenosite.predict`` API.

Importing this module does not load OpenBabel. Bindings load on first use.
PyPI ships OpenBabel **3.2.x** wheels (``uv add openbabel``). The dump oracle
is Debian OpenBabel **2.4.1**; ``GetHyb()`` is wrapped to that 2.4 behavior
(halogens 0, aromatic sulfur 3). Dump-vs-feature tests catch remaining 3.x drift.
"""

from __future__ import annotations

from typing import Any

from ..errors import OpenBabelNotAvailable

_CACHE: tuple[Any, Any] | None = None


def load() -> tuple[Any, Any]:
    """Return ``(openbabel, pybel)``. Prefer the 3.x layout (PyPI wheels)."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    try:
        from openbabel import openbabel as ob  # type: ignore[import-untyped]
        from openbabel import pybel  # type: ignore[import-untyped]
        return _remember(ob, pybel)
    except ImportError:
        pass
    try:
        import openbabel as ob  # type: ignore[import-untyped]
        try:
            import pybel  # type: ignore[import-untyped]
        except ImportError:
            from openbabel import pybel  # type: ignore[import-untyped]
        return _remember(ob, pybel)
    except ImportError as err:
        raise OpenBabelNotAvailable(
            "OpenBabel is required for XenoSite descriptors (internal only). "
            "Install the PyPI wheel with `uv add openbabel` (3.2.x). "
            "See the README."
        ) from err


def _remember(ob: Any, pybel: Any) -> tuple[Any, Any]:
    global _CACHE
    _patch_legacy_api(ob)
    _CACHE = (ob, pybel)
    return _CACHE


# OpenBabel 2.4 left these unhybridized (GetHyb() == 0). 3.x assigns sp (1).
_HALOGEN_Z = frozenset({9, 17, 35, 53, 85})


def _patch_legacy_api(ob: Any) -> None:
    """Restore OpenBabel 2.4 method names and halogen hybridization."""
    if getattr(ob, "_xenosite_legacy_api", False):
        return
    ob._xenosite_legacy_api = True

    atom = ob.OBAtom
    _add(atom, "IsHydrogen", lambda self: self.GetAtomicNum() == 1)
    _add(atom, "IsCarbon", lambda self: self.GetAtomicNum() == 6)
    _add(atom, "IsNitrogen", lambda self: self.GetAtomicNum() == 7)
    _add(atom, "IsOxygen", lambda self: self.GetAtomicNum() == 8)
    _add(atom, "IsSulfur", lambda self: self.GetAtomicNum() == 16)
    _add(atom, "IsPhosphorus", lambda self: self.GetAtomicNum() == 15)
    _add(atom, "IsHalogen", lambda self: self.GetAtomicNum() in _HALOGEN_Z)
    _add(atom, "ImplicitHydrogenCount", lambda self: int(self.GetImplicitHCount()))
    _wrap_get_hyb(atom)

    bond = ob.OBBond
    _add(bond, "IsSingle", lambda self: (not self.IsAromatic()) and self.GetBondOrder() == 1)
    _add(bond, "IsDouble", lambda self: (not self.IsAromatic()) and self.GetBondOrder() == 2)
    _add(bond, "IsTriple", lambda self: (not self.IsAromatic()) and self.GetBondOrder() == 3)


def _wrap_get_hyb(atom_cls: Any) -> None:
    """Match OpenBabel 2.4 ``GetHyb()`` on the 3.2 wheel.

    2.4 left H/F/Cl/Br/I unhybridized (0). 3.x reports them as sp (1).
    2.4 counted aromatic sulfur as sp3; 3.x types it ``S2`` / hyb 2.
    """
    native = atom_cls.GetHyb

    def GetHyb(self) -> int:
        z = int(self.GetAtomicNum())
        if z == 1 or z in _HALOGEN_Z:
            return 0
        if z == 16 and self.IsAromatic():
            return 3
        return int(native(self))

    atom_cls.GetHyb = GetHyb


def _add(cls: Any, name: str, fn) -> None:
    if not hasattr(cls, name):
        setattr(cls, name, fn)


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
    # pybel assigns Gasteiger on read (same as OpenBabel 2.4 / the dump).
    # OBChargeModel.FindType("gasteiger") in 3.2 overwrites those and
    # equalizes nitro oxygens (~0.45 off the dump). Do not recompute.
    return mol


def ob_idx_to_rdkit(ob_idx: int) -> int:
    """1-based OpenBabel atom index → 0-based RDKit (same molblock order)."""
    return int(ob_idx) - 1


def element_table():
    """Periodic-table helpers with the OpenBabel 2.4 ``OBElementTable`` methods.

    OpenBabel 3.x dropped ``OBElementTable``; the same getters live as module
    functions. ``CorrectedBondRad`` / ``CorrectedVdwRad`` are not wrapped in
    the 3.2 PyPI wheel, so those two use the 2.4 hybridization scale
    (sp ``*0.90``, sp2 ``*0.95``, else covalent/vdW radius).
    """
    ob, _pybel = load()
    table = getattr(ob, "OBElementTable", None)
    if table is not None:
        return table()
    return _ModuleElementTable(ob)


class _ModuleElementTable:
    __slots__ = ("_ob",)

    def __init__(self, ob):
        self._ob = ob

    def GetSymbol(self, z: int) -> str:
        return self._ob.GetSymbol(int(z))

    def GetMass(self, z: int) -> float:
        return float(self._ob.GetMass(int(z)))

    def GetExactMass(self, z: int) -> float:
        return float(self._ob.GetExactMass(int(z)))

    def GetMaxBonds(self, z: int) -> int:
        return int(self._ob.GetMaxBonds(int(z)))

    def GetElectroNeg(self, z: int) -> float:
        return float(self._ob.GetElectroNeg(int(z)))

    def GetElectronAffinity(self, z: int) -> float:
        return float(self._ob.GetElectronAffinity(int(z)))

    def GetIonization(self, z: int) -> float:
        return float(self._ob.GetIonization(int(z)))

    def GetCovalentRad(self, z: int) -> float:
        return float(self._ob.GetCovalentRad(int(z)))

    def GetVdwRad(self, z: int) -> float:
        return float(self._ob.GetVdwRad(int(z)))

    def CorrectedBondRad(self, z: int, hyb: int) -> float:
        rad = self.GetCovalentRad(z)
        if int(hyb) == 2:
            return rad * 0.95
        if int(hyb) == 1:
            return rad * 0.90
        return rad

    def CorrectedVdwRad(self, z: int, hyb: int) -> float:
        rad = self.GetVdwRad(z)
        if int(hyb) == 2:
            return rad * 0.95
        if int(hyb) == 1:
            return rad * 0.90
        return rad


def ob_numbering_mode():
    """Configured heavy-atom numbering policy (see ``xenosite.predict.numbering``)."""
    from ..numbering import numbering_mode

    return numbering_mode()


def ob_numbering_is_gapped(obmol: Any) -> bool:
    """True when this molecule's heavy-atom ``GetIdx()`` values are gapped."""
    from ..numbering import ObNumberingMode, detect_raw_numbering_mode

    return detect_raw_numbering_mode(obmol) == ObNumberingMode.RAW

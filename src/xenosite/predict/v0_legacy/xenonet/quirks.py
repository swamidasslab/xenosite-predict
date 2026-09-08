"""v0 quirks vs Python-2 XenoNet 1.0 (internal; no public contract).

Scoring ``"0"`` matches the shipped plugin as closely as tests allow.
Scoring ``"1"`` only changes documented defects below.

Copied on purpose (v0)
----------------------
- Unpooled Class-row lookup (``get_prob_for_one_site``), including the in-place
  ``a.b`` / ``b.a`` swap on a missed bond row.
- Missing site → edge weight 0.
- ``EpoxideOpening`` weight is 1.0.
- Multi-site reactions multiply site scores (``rxn_prob`` starts at -1).
- Beam keeps the top ``beam_width`` children by weight; untargeted search
  commits a path at ``depth_limit`` or when a parent has no children.
- Likelihoods: root 1.0, max over multi-edges, drop 1-step cycles, then
  ``parent_like * (w / sum_outgoing(parent))``.
- ``trim_weights(0)`` drops non-positive edges by keeping only paths whose
  weights are all > 0.

v1 fixes
--------
- **Pooled atom/bond lookup.** ``predict()`` max-pools Bond_and_LonePair rows
  onto atoms/bonds. v0 walks raw rows; several rows can share an atom and
  disagree with the pooled vector. ``scoring="1"`` uses the pooled map.

Deliberate v0 deviations (tests / determinism)
---------------------------------------------
- Heap tie-break is ``(neg_weight, smiles, site)``. Legacy compared the RDKit
  mol object, which is unordered. Golden fixtures use a large ``beam_width``
  so ties do not drop children; v0 still uses a stable key so tests do not flake.
- ``max_time`` is not a golden cutoff (partial graphs are not reproducible).
- Quinone-weighted edges, conjugation, biomolecule reactivity, and the
  bioactivation PBS/MBS pipeline are not in this engine.
"""

from __future__ import annotations

from ._unvalidated import warn_unvalidated

warn_unvalidated()

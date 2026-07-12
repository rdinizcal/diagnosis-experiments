"""Runtime for sample-aligned time-window quantization (Sprint 7, Feature 2).

The :class:`Quantizer` turns a GA candidate into a *canonical* verdict-cache key
in which every quantizable time token is replaced by its sample-class index
``floor(value / period)``.  Distinct raw bounds that fall inside one inter-sample
interval collapse onto one key, so the second and later members of a class are
served from cache instead of being re-solved.  The formula sent to the solver on
a miss is always the original, un-canonicalized one -- only the key changes.

Two safety nets bracket the static quantizability gate (:mod:`diagnosis.lang.quantize`):

* a **runtime validator** that, every ``validate_every_n_hits`` canonical hits on a
  class, actually re-solves the new raw value and compares verdicts; a mismatch
  permanently disables quantization for the offending position, evicts that
  class's cached entries, and logs a witness (this mirrors the monotonicity
  guard of the interval-inference layer);
* a **vacuity guard** that flags a candidate whose two mutable bounds define an
  empty window ``[a, b]`` with ``a >= b`` (the ForAll is then vacuously SATISFIED)
  so such rows never silently feed the decision tree as ordinary SATs.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Set

from .lang.ast import Formula
from .lang.quantize import bound_role, quantizability, window_variable

_TIMESTAMP_RE = re.compile(r"timestamps\[\s*\d+\s*\]\s*==\s*([-+]?[0-9]*\.?[0-9]+)")


def trace_period_from_property(path: str) -> Optional[float]:
    """Recover the trace timestamp spacing from a ThEodorE property file.

    Returns the common consecutive gap between the ``timestamps[k] == v``
    assertions, or ``None`` when there are too few timestamps or the spacing is
    irregular (either case leaves the period cross-check unsatisfiable, so the
    position is conservatively not quantizable).
    """
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            text = fh.read()
    except OSError:
        return None
    values = [float(m) for m in _TIMESTAMP_RE.findall(text)]
    if len(values) < 2:
        return None
    gaps = {round(b - a, 9) for a, b in zip(values, values[1:])}
    gaps.discard(0.0)
    if len(gaps) != 1:
        return None
    gap = next(iter(gaps))
    return gap if gap > 0 else None


def class_midpoint_value(cls: int, period: float) -> float:
    """A representative raw value for a sample class (its midpoint)."""
    return (cls + 0.5) * period


def bisect_probe_class(sat_classes, unsat_classes, orient: int):
    """Bisection step over a discrete ``(highest-UNSAT, lowest-SAT)`` class bracket.

    ``orient`` is ``+1`` for an INCREASING position (SAT is the upper class
    half-line) and ``-1`` for DECREASING.  Returns ``(probe_class, boundary_class)``:

    * ``probe_class`` is the midpoint class to solve next while the bracket is
      wider than one class -- this is the bisection that provably terminates in
      ``ceil(log2(#classes))`` real solves on the finite ordered class set;
    * ``boundary_class`` is set once the SAT and UNSAT classes are adjacent: the
      breakpoint sits at that class's sample timestamp and shrinking stops.

    Exactly one of the two is non-``None`` when a proper bracket exists; both are
    ``None`` when there is not yet one SAT and one UNSAT observation (or they are
    inconsistent/overlapping).
    """
    if not sat_classes or not unsat_classes:
        return None, None
    sat_min_o = min(orient * c for c in sat_classes)
    unsat_max_o = max(orient * c for c in unsat_classes)
    gap = sat_min_o - unsat_max_o
    if gap <= 0:
        return None, None
    if gap == 1:
        return None, max(orient * unsat_max_o, orient * sat_min_o)
    mid_o = (unsat_max_o + sat_min_o) // 2
    return orient * mid_o, None


class Quantizer:
    """Per-run canonical-key, validator, and vacuity state for time bounds."""

    def __init__(
        self,
        seed_ast: Optional[Formula],
        cfg,
        trace_period: Optional[float] = None,
    ) -> None:
        self.cfg = cfg
        self.enabled = bool(getattr(cfg, "enabled", False)) and seed_ast is not None
        self.validate_every = int(getattr(cfg, "validate_every_n_hits", 0))
        self.trace_period = trace_period

        # {position: period} for the statically quantizable positions, plus the
        # per-variable window pairs used by the vacuity guard.
        self.positions: Dict[int, float] = {}
        self.window_pairs: List[Dict[str, int]] = []
        self.disabled: Set[int] = set()
        if self.enabled:
            self._analyse(seed_ast)
        self.enabled = self.enabled and bool(self.positions)

        # Validator bookkeeping.
        self._hits_by_key: Dict[str, int] = {}
        self._raw_by_key: Dict[str, Set[str]] = {}
        self.exact_hits = 0
        self.canonical_hits = 0
        self.validations = 0
        self.violations = 0
        self.vacuous_candidates = 0
        self.witnesses: List[dict] = []
        # Filled by the class-index bracket layer (Feature 3).
        self.boundary_at: Dict[str, float] = {}

    # -- static setup -------------------------------------------------------
    def _analyse(self, seed_ast: Formula) -> None:
        from .lang.polarity import numeric_positions

        override = getattr(self.cfg, "period", None)
        force = bool(getattr(self.cfg, "force_period", False))
        by_var: Dict[str, Dict[str, int]] = {}
        for p in numeric_positions(seed_ast):
            info = quantizability(seed_ast, p, self.trace_period, override, force)
            if not info.quantizable or info.period is None:
                continue
            self.positions[p] = info.period
            var = window_variable(seed_ast, p)
            role = bound_role(seed_ast, p)
            if var is not None and role in ("lower", "upper"):
                by_var.setdefault(var, {})[role] = p
        for roles in by_var.values():
            if "lower" in roles and "upper" in roles:
                self.window_pairs.append(roles)

    def _active_positions(self) -> Dict[int, float]:
        return {p: period for p, period in self.positions.items() if p not in self.disabled}

    # -- canonicalization ---------------------------------------------------
    def canonical_expr(self, cand_ast: Optional[Formula], raw_expr: str) -> str:
        """Return the canonical cache expression for ``cand_ast``.

        Every active quantizable position is replaced by its integer class index
        ``floor(value / period)``; non-quantizable (and disabled) positions keep
        their raw value.  When quantization is off or no position is active this
        is exactly ``raw_expr``, so the verdict cache behaves bit-for-bit as
        before.
        """
        if not self.enabled or cand_ast is None:
            return raw_expr
        active = self._active_positions()
        if not active:
            return raw_expr
        from .inference import numeric_value_map, set_numeric_at_position
        from .lang.python_printer import formula_to_python_expr

        values = numeric_value_map(cand_ast)
        node = cand_ast
        for p, period in active.items():
            if p in values:
                cls = math.floor(values[p] / period)
                node = set_numeric_at_position(node, p, cls)
        return formula_to_python_expr(node)

    # -- vacuity ------------------------------------------------------------
    def vacuous_flag(self, cand_ast: Optional[Formula]) -> bool:
        """Whether ``cand_ast`` has an empty ``[lower, upper]`` window (a >= b).

        Increments the ``vacuous_candidates`` counter on a positive result; the
        GA marks the ARFF row so the vacuously-SATISFIED verdict does not feed the
        tree as an ordinary SAT.
        """
        if not self.enabled or cand_ast is None or not self.window_pairs:
            return False
        from .inference import numeric_value_map

        values = numeric_value_map(cand_ast)
        for pair in self.window_pairs:
            lo, hi = values.get(pair["lower"]), values.get(pair["upper"])
            if lo is not None and hi is not None and lo >= hi:
                self.vacuous_candidates += 1
                return True
        return False

    # -- validator ----------------------------------------------------------
    def on_lookup(self, canonical: str, raw_expr: str, hit: bool) -> bool:
        """Record a cache lookup; return True if a validation double-solve is due.

        Distinguishes *exact* hits (this raw value was solved before) from
        *canonical* hits (a new raw value collapsed onto an already-solved class).
        A validation is scheduled only on canonical hits, every
        ``validate_every_n_hits`` of them per class.
        """
        if not self.enabled or canonical == raw_expr:
            return False
        seen = self._raw_by_key.setdefault(canonical, set())
        if not hit:
            seen.add(raw_expr)
            return False
        if raw_expr in seen:
            self.exact_hits += 1
            return False
        self.canonical_hits += 1
        seen.add(raw_expr)
        self._hits_by_key[canonical] = self._hits_by_key.get(canonical, 0) + 1
        due = self.validate_every > 0 and self._hits_by_key[canonical] % self.validate_every == 0
        return due

    def record_validation(
        self,
        cand_ast: Optional[Formula],
        canonical: str,
        cached_verdict: str,
        real_verdict: str,
    ) -> List[str]:
        """Compare a re-solved verdict to the cached one; act on a mismatch.

        On disagreement the offending quantizable positions are permanently
        disabled, a witness is logged, and the raw expressions that collapsed onto
        ``canonical`` are returned so the caller can purge them from the cache.
        Returns an empty list when the verdicts agree.
        """
        self.validations += 1
        if real_verdict == cached_verdict:
            return []
        self.violations += 1
        from .inference import numeric_value_map

        values = numeric_value_map(cand_ast) if cand_ast is not None else {}
        active = self._active_positions()
        witness = {
            "positions": sorted(active),
            "values": {str(p): values.get(p) for p in active},
            "classes": {
                str(p): math.floor(values[p] / period)
                for p, period in active.items()
                if p in values
            },
            "cached_verdict": cached_verdict,
            "real_verdict": real_verdict,
        }
        self.witnesses.append(witness)
        self.disabled.update(active)
        purge = list(self._raw_by_key.get(canonical, set())) + [canonical]
        return purge

    # -- reporting ----------------------------------------------------------
    def report(self) -> dict:
        out = {
            "time_quantization": True,
            "quantized_positions": {str(p): period for p, period in self.positions.items()},
            "quant_hits_exact": self.exact_hits,
            "quant_hits_canonical": self.canonical_hits,
            "quant_validations": self.validations,
            "quantization_violations": self.violations,
            "vacuous_candidates": self.vacuous_candidates,
        }
        if self.disabled:
            out["quant_disabled_positions"] = sorted(self.disabled)
        if self.witnesses:
            out["quantization_violation_witnesses"] = self.witnesses
        if self.boundary_at:
            out["boundary_at"] = dict(self.boundary_at)
        return out

"""Banded interval inference for monotone numeric threshold positions.

When a candidate differs from the seed at exactly one numeric position ``p``
whose polarity (``diagnosis.lang.polarity``) is proven INCREASING or DECREASING,
the trace-check verdict is monotone in the constant at ``p``.  This module keeps,
per position, three regions on the real line:

* ``sat_region``   -- values proven SATISFIED (a half-line, given the direction)
* ``unsat_region`` -- values proven VIOLATED (the opposite half-line)
* ``unknown_band`` -- values where the solver returned UNDECIDED

and answers, for a candidate value ``v``: can the verdict be *inferred* (no
solver call), should it be solved with the low tier only (region memory), or is
it on the unresolved frontier and must be solved.

Soundness is enforced by two independent mechanisms:

1. **Static gate** -- inference is only ever attempted for positions whose
   polarity is provably monotone (Asarin/Donze/Maler/Nickovic, RV 2011).
2. **Runtime guard** -- after *every* real solve, the new verdict is checked for
   consistency with the accumulated regions.  A SATISFIED value on the VIOLATED
   side (or vice versa) is a monotonicity violation: inference for that position
   is permanently disabled, its regions discarded, and a witness recorded.

In addition, ``empirical_validation_k`` real, consistent solves are required at a
position before any verdict is returned by inference, so a single lucky proof
never drives the search on its own.  The one-dimensional frontier helpers are
kept for reporting and adaptive-range bookkeeping.
"""

from __future__ import annotations

import json
import math
import random
import threading
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .lang.ast import (
    And,
    ArithOp,
    BoolConst,
    Exists,
    ForAll,
    Formula,
    FuncCall,
    Implies,
    IntConst,
    Not,
    Or,
    RealConst,
    RelOp,
    Subscript,
    Var,
)
from .lang.polarity import Monotonicity, numeric_positions, polarity

# --- Verdict vocabulary (madeit strings used across the GA) ---------------
MADEIT_SAT = "True"
MADEIT_VIOLATED = "False"
MADEIT_UNDECIDED = "Unknown"
MADEIT_PROBLEM = "Problem"

# Small epsilon for float boundary inclusion.
_EPS = 1e-12


# ==========================================================================
# Single-mutation detection on the AST diff
# ==========================================================================


class _Structural(Exception):
    """Raised internally when two ASTs differ in structure (not just a number)."""


def ast_single_numeric_diff(
    seed: Formula, cand: Formula
) -> Optional[Tuple[int, float]]:
    """Return ``(position, new_value)`` iff ``cand`` differs from ``seed`` at
    exactly one numeric constant and is otherwise structurally identical.

    ``position`` is the shared preorder index (same numbering as
    ``allowed_positions`` / the ARFF layout).  Any structural, operator, name,
    boolean, or opaque-term difference -- or more than one numeric change (e.g.
    a two-position crossover offspring) -- yields ``None`` so those candidates go
    to the solver untouched.
    """
    if seed is None or cand is None:
        return None

    num_diffs: List[Tuple[int, float]] = []
    counter = [0]

    def walk(a: Formula, b: Formula) -> None:
        idx = counter[0]
        counter[0] += 1
        if type(a) is not type(b):
            raise _Structural
        if isinstance(a, (IntConst, RealConst)):
            if a.value != b.value:
                num_diffs.append((idx, float(b.value)))
            return
        if isinstance(a, BoolConst):
            if a.value != b.value:
                raise _Structural
            return
        if isinstance(a, Var):
            if a.name != b.name:
                raise _Structural
            return
        if isinstance(a, (Subscript, FuncCall)):
            # Opaque leaves (matches the non-descending preorder used elsewhere).
            if str(a) != str(b):
                raise _Structural
            return
        if isinstance(a, RelOp):
            if a.op != b.op:
                raise _Structural
            walk(a.left, b.left)
            walk(a.right, b.right)
            return
        if isinstance(a, ArithOp):
            if a.op != b.op:
                raise _Structural
            walk(a.left, b.left)
            walk(a.right, b.right)
            return
        if isinstance(a, Not):
            walk(a.arg, b.arg)
            return
        if isinstance(a, Implies):
            walk(a.left, b.left)
            walk(a.right, b.right)
            return
        if isinstance(a, (And, Or)):
            if len(a.args) != len(b.args):
                raise _Structural
            for x, y in zip(a.args, b.args):
                walk(x, y)
            return
        if isinstance(a, (ForAll, Exists)):
            if list(a.vars) != list(b.vars):
                raise _Structural
            walk(a.body, b.body)
            return
        # Unknown node type: compare structurally by string.
        if str(a) != str(b):
            raise _Structural

    try:
        walk(seed, cand)
    except _Structural:
        return None

    if len(num_diffs) == 1:
        return num_diffs[0]
    return None


def ast_numeric_diffs(
    seed: Formula, cand: Formula
) -> Optional[List[Tuple[int, float]]]:
    """Return **all** numeric-constant differences between ``seed`` and ``cand``.

    Returns the list ``[(position, new_value), ...]`` (possibly empty if the two
    formulas are identical), or ``None`` if the two ASTs differ structurally / in
    any non-numeric leaf (operator, name, boolean, opaque term, arity). This is
    the multi-knob generalization of :func:`ast_single_numeric_diff`; the
    k-D inference layer uses it to gather the full tuple of changed knobs.
    """
    if seed is None or cand is None:
        return None

    num_diffs: List[Tuple[int, float]] = []
    counter = [0]

    def walk(a: Formula, b: Formula) -> None:
        idx = counter[0]
        counter[0] += 1
        if type(a) is not type(b):
            raise _Structural
        if isinstance(a, (IntConst, RealConst)):
            if a.value != b.value:
                num_diffs.append((idx, float(b.value)))
            return
        if isinstance(a, BoolConst):
            if a.value != b.value:
                raise _Structural
            return
        if isinstance(a, Var):
            if a.name != b.name:
                raise _Structural
            return
        if isinstance(a, (Subscript, FuncCall)):
            if str(a) != str(b):
                raise _Structural
            return
        if isinstance(a, (RelOp, ArithOp)):
            if a.op != b.op:
                raise _Structural
            walk(a.left, b.left)
            walk(a.right, b.right)
            return
        if isinstance(a, Not):
            walk(a.arg, b.arg)
            return
        if isinstance(a, Implies):
            walk(a.left, b.left)
            walk(a.right, b.right)
            return
        if isinstance(a, (And, Or)):
            if len(a.args) != len(b.args):
                raise _Structural
            for x, y in zip(a.args, b.args):
                walk(x, y)
            return
        if isinstance(a, (ForAll, Exists)):
            if list(a.vars) != list(b.vars):
                raise _Structural
            walk(a.body, b.body)
            return
        if str(a) != str(b):
            raise _Structural

    try:
        walk(seed, cand)
    except _Structural:
        return None
    return num_diffs


def numeric_value_map(formula: Formula) -> Dict[int, float]:
    """Return ``{preorder_position: value}`` for every numeric constant.

    Uses the same non-descending preorder as ``allowed_positions`` / the ARFF
    layout, so keys line up with mutation positions.
    """
    out: Dict[int, float] = {}
    counter = [0]

    def walk(n: Formula) -> None:
        idx = counter[0]
        counter[0] += 1
        if isinstance(n, (IntConst, RealConst)):
            out[idx] = float(n.value)
            return
        if isinstance(n, (Var, BoolConst, Subscript, FuncCall)):
            return
        if isinstance(n, (RelOp, ArithOp)):
            walk(n.left)
            walk(n.right)
            return
        if isinstance(n, Not):
            walk(n.arg)
            return
        if isinstance(n, Implies):
            walk(n.left)
            walk(n.right)
            return
        if isinstance(n, (And, Or)):
            for a in n.args:
                walk(a)
            return
        if isinstance(n, (ForAll, Exists)):
            walk(n.body)
            return

    walk(formula)
    return out


def set_numeric_at_position(formula: Formula, position: int, value: float) -> Formula:
    """Return a copy of ``formula`` with the numeric constant at ``position`` set.

    Preserves Int vs Real kind. If ``position`` is not a numeric constant the
    formula is returned unchanged.
    """
    counter = [0]

    def rebuild(node: Formula) -> Formula:
        idx = counter[0]
        counter[0] += 1
        if isinstance(node, IntConst):
            return IntConst(int(round(value))) if idx == position else node
        if isinstance(node, RealConst):
            return RealConst(float(value)) if idx == position else node
        if isinstance(node, (Var, BoolConst, Subscript, FuncCall)):
            return node
        if isinstance(node, RelOp):
            return RelOp(node.op, rebuild(node.left), rebuild(node.right))
        if isinstance(node, ArithOp):
            return ArithOp(node.op, rebuild(node.left), rebuild(node.right))
        if isinstance(node, Not):
            return Not(rebuild(node.arg))
        if isinstance(node, Implies):
            return Implies(left=rebuild(node.left), right=rebuild(node.right))
        if isinstance(node, And):
            return And([rebuild(a) for a in node.args])
        if isinstance(node, Or):
            return Or([rebuild(a) for a in node.args])
        if isinstance(node, ForAll):
            return ForAll(vars=list(node.vars), body=rebuild(node.body))
        if isinstance(node, Exists):
            return Exists(vars=list(node.vars), body=rebuild(node.body))
        return node

    return rebuild(formula)


# ==========================================================================
# Per-position region state + runtime guard
# ==========================================================================


class Decision(Enum):
    INFER_SAT = "infer_sat"
    INFER_VIOLATED = "infer_violated"
    INFER_UNDECIDED = "infer_undecided"
    SOLVE_LOW_ONLY = "solve_low_only"   # region memory: v in a confirmed band
    SOLVE_NORMAL = "solve_normal"


class PositionRegion:
    """Monotone SAT/UNSAT half-lines and the UNDECIDED band for one position.

    All reasoning is done in an *oriented* coordinate ``w = orient(v)`` chosen so
    that SATISFIED always lies at high ``w`` (INCREASING: ``w = v``; DECREASING:
    ``w = -v``).  The monotonicity invariant is then simply
    ``max(unsat) < min(sat)``.
    """

    def __init__(self, direction: Monotonicity) -> None:
        assert direction in (Monotonicity.INCREASING, Monotonicity.DECREASING)
        self.direction = direction
        self.sat_values: List[float] = []
        self.unsat_values: List[float] = []
        self.unknown_values: List[float] = []
        self.consistent_solves = 0
        self.disabled = False
        self.witness: Optional[dict] = None

    # -- orientation helpers -------------------------------------------------
    def _o(self, v: float) -> float:
        return v if self.direction is Monotonicity.INCREASING else -v

    @property
    def _sat_min_o(self) -> Optional[float]:
        return min(self._o(v) for v in self.sat_values) if self.sat_values else None

    @property
    def _unsat_max_o(self) -> Optional[float]:
        return max(self._o(v) for v in self.unsat_values) if self.unsat_values else None

    # -- queries -------------------------------------------------------------
    def _unknown_bracketed(self, v: float) -> bool:
        """True iff >= 2 UNDECIDED observations bracket ``v`` (confirmed band)."""
        if len(self.unknown_values) < 2:
            return False
        w = self._o(v)
        wos = [self._o(u) for u in self.unknown_values]
        return any(u <= w + _EPS for u in wos) and any(u >= w - _EPS for u in wos)

    def _unknown_near(self, v: float) -> bool:
        """True iff ``v`` lies within the span of observed UNDECIDED values."""
        if not self.unknown_values:
            return False
        w = self._o(v)
        wos = [self._o(u) for u in self.unknown_values]
        return (min(wos) - _EPS) <= w <= (max(wos) + _EPS)

    def decide(self, v: float, k: int) -> Decision:
        """Classify a candidate value ``v`` at this position."""
        w = self._o(v)
        if self.consistent_solves >= k and not self.disabled:
            smin = self._sat_min_o
            umax = self._unsat_max_o
            if smin is not None and w >= smin - _EPS:
                return Decision.INFER_SAT
            if umax is not None and w <= umax + _EPS:
                return Decision.INFER_VIOLATED
            if self._unknown_bracketed(v):
                return Decision.INFER_UNDECIDED
        if self._unknown_near(v):
            return Decision.SOLVE_LOW_ONLY
        return Decision.SOLVE_NORMAL

    # -- updates + runtime guard --------------------------------------------
    def update(self, v: float, madeit: str) -> Optional[dict]:
        """Record a real solve. Return a witness dict on monotonicity violation.

        On violation the region is disabled and its half-lines discarded so no
        further verdicts are inferred for this position.
        """
        if self.disabled:
            return None
        w = self._o(v)

        if madeit == MADEIT_SAT:
            # Violation if a strictly-larger-in-verdict value was VIOLATED.
            umax = self._unsat_max_o
            if umax is not None and w <= umax - _EPS:
                return self._violate(v, madeit, "SATISFIED inside VIOLATED region")
            self.sat_values.append(v)
            self.consistent_solves += 1
        elif madeit == MADEIT_VIOLATED:
            smin = self._sat_min_o
            if smin is not None and w >= smin + _EPS:
                return self._violate(v, madeit, "VIOLATED inside SATISFIED region")
            self.unsat_values.append(v)
            self.consistent_solves += 1
        elif madeit == MADEIT_UNDECIDED:
            self.unknown_values.append(v)
            # UNDECIDED never contradicts a monotone half-line; it is a band.
            self.consistent_solves += 1
        else:
            # Problem / error: do not update regions.
            return None
        return None

    def _violate(self, v: float, madeit: str, note: str) -> dict:
        witness = {
            "note": note,
            "value": v,
            "verdict": madeit,
            "sat_values": list(self.sat_values),
            "unsat_values": list(self.unsat_values),
        }
        self.disabled = True
        self.witness = witness
        # Discard the (now-untrusted) half-lines.
        self.sat_values = []
        self.unsat_values = []
        self.unknown_values = []
        return witness

    # -- frontier helper -----------------------------------------------------
    def frontier_midpoint(self, min_gap: float) -> Optional[float]:
        """Midpoint of the unresolved SAT/UNSAT frontier, or None if too narrow.

        Requires both a SATISFIED and a VIOLATED observation (a proper bracket).
        """
        smin = self._sat_min_o
        umax = self._unsat_max_o
        if smin is None or umax is None:
            return None
        lo, hi = umax, smin  # oriented frontier is (umax, smin)
        if hi - lo <= max(min_gap, _EPS):
            return None
        mid_o = 0.5 * (lo + hi)
        return mid_o if self.direction is Monotonicity.INCREASING else -mid_o

    # -- persistence ---------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "direction": self.direction.value,
            "sat_values": list(self.sat_values),
            "unsat_values": list(self.unsat_values),
            "unknown_values": list(self.unknown_values),
            "consistent_solves": self.consistent_solves,
            "disabled": self.disabled,
            "witness": self.witness,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PositionRegion":
        r = cls(Monotonicity(d["direction"]))
        r.sat_values = list(d.get("sat_values", []))
        r.unsat_values = list(d.get("unsat_values", []))
        r.unknown_values = list(d.get("unknown_values", []))
        r.consistent_solves = int(d.get("consistent_solves", 0))
        r.disabled = bool(d.get("disabled", False))
        r.witness = d.get("witness")
        return r


class MonotoneRegion:
    """k-D monotone SAT up-set / VIOLATED down-set over a fixed set of knobs.

    Generalises :class:`PositionRegion` to several simultaneously-varying numeric
    knobs.  A candidate is the vector ``vec`` of its values at the monotone
    positions (in ``positions`` order).  Reasoning is done in *oriented*
    coordinates ``w_i = sign_i * vec_i`` (``sign = +1`` for INCREASING, ``-1`` for
    DECREASING), so parametric-STL per-coordinate monotonicity makes SATISFIED an
    **up-set** and VIOLATED a **down-set** in the product (Pareto) order:

    * infer SATISFIED if ``w`` dominates some known-SAT vector (``w >= s``);
    * infer VIOLATED if ``w`` is dominated by some known-VIOLATED vector
      (``w <= u``);
    * otherwise solve.

    Runtime guard: a violation is a SAT vector that is dominated by a VIOLATED
    vector (``s <= u``) -- monotonicity would force that VIOLATED point to be
    SATISFIED.  On violation the region is disabled and its points discarded.
    For a single knob (k=1) this is exactly the sat/unsat half-line model.
    """

    def __init__(self, positions: List[int], directions: List[Monotonicity]) -> None:
        assert len(positions) == len(directions) and positions
        assert all(d in (Monotonicity.INCREASING, Monotonicity.DECREASING) for d in directions)
        self.positions = list(positions)
        self.signs = [1 if d is Monotonicity.INCREASING else -1 for d in directions]
        self.sat: List[List[float]] = []      # oriented vectors, SATISFIED
        self.unsat: List[List[float]] = []    # oriented vectors, VIOLATED
        self.unknown: List[List[float]] = []  # oriented vectors, UNDECIDED
        self.consistent_solves = 0
        self.disabled = False
        self.witness: Optional[dict] = None

    # -- orientation ---------------------------------------------------------
    def _o(self, vec: List[float]) -> List[float]:
        return [s * float(v) for s, v in zip(self.signs, vec)]

    def _raw(self, w: List[float]) -> List[float]:
        return [s * x for s, x in zip(self.signs, w)]

    @staticmethod
    def _dominates(a: List[float], b: List[float]) -> bool:
        """``a >= b`` in every coordinate (with tolerance)."""
        return all(ai >= bi - _EPS for ai, bi in zip(a, b))

    # -- queries -------------------------------------------------------------
    def _in_unknown_box(self, w: List[float]) -> bool:
        if not self.unknown:
            return False
        for i in range(len(w)):
            col = [u[i] for u in self.unknown]
            if not (min(col) - _EPS <= w[i] <= max(col) + _EPS):
                return False
        return True

    def _unknown_bracketed(self, w: List[float]) -> bool:
        if len(self.unknown) < 2:
            return False
        return (
            any(self._dominates(w, u) for u in self.unknown)
            and any(self._dominates(u, w) for u in self.unknown)
        )

    def decide(self, vec: List[float], k: int) -> Decision:
        w = self._o(vec)
        if self.consistent_solves >= k and not self.disabled:
            if any(self._dominates(w, s) for s in self.sat):
                return Decision.INFER_SAT
            if any(self._dominates(u, w) for u in self.unsat):
                return Decision.INFER_VIOLATED
            if self._unknown_bracketed(w):
                return Decision.INFER_UNDECIDED
        if self._in_unknown_box(w):
            return Decision.SOLVE_LOW_ONLY
        return Decision.SOLVE_NORMAL

    # -- updates + runtime guard --------------------------------------------
    def update(self, vec: List[float], madeit: str) -> Optional[dict]:
        if self.disabled:
            return None
        w = self._o(vec)
        if madeit == MADEIT_SAT:
            for u in self.unsat:
                if self._dominates(u, w):  # a VIOLATED point dominates this SAT one
                    return self._violate(vec, madeit, "SATISFIED dominated by a VIOLATED point")
            self.sat.append(w)
            self.consistent_solves += 1
        elif madeit == MADEIT_VIOLATED:
            for s in self.sat:
                if self._dominates(w, s):  # this VIOLATED point dominates a SAT one
                    return self._violate(vec, madeit, "VIOLATED dominates a SATISFIED point")
            self.unsat.append(w)
            self.consistent_solves += 1
        elif madeit == MADEIT_UNDECIDED:
            self.unknown.append(w)
            self.consistent_solves += 1
        return None

    def _violate(self, vec: List[float], madeit: str, note: str) -> dict:
        witness = {
            "note": note,
            "positions": list(self.positions),
            "value": list(vec),
            "verdict": madeit,
            "sat": [self._raw(s) for s in self.sat],
            "unsat": [self._raw(u) for u in self.unsat],
        }
        self.disabled = True
        self.witness = witness
        self.sat = []
        self.unsat = []
        self.unknown = []
        return witness

    # -- frontier helper (single-axis) --------------------------------------
    def frontier_midpoint_axis(self, axis: int, min_gap: float) -> Optional[float]:
        if not self.sat or not self.unsat:
            return None
        smin = min(s[axis] for s in self.sat)
        umax = max(u[axis] for u in self.unsat)
        lo, hi = umax, smin
        if hi - lo <= max(min_gap, _EPS):
            return None
        mid_o = 0.5 * (lo + hi)
        return self.signs[axis] * mid_o

    # -- accessors + persistence --------------------------------------------
    def sat_raw(self) -> List[List[float]]:
        return [self._raw(s) for s in self.sat]

    def unsat_raw(self) -> List[List[float]]:
        return [self._raw(u) for u in self.unsat]

    def unknown_raw(self) -> List[List[float]]:
        return [self._raw(u) for u in self.unknown]

    def to_dict(self) -> dict:
        return {
            "positions": list(self.positions),
            "directions": ["INCREASING" if s > 0 else "DECREASING" for s in self.signs],
            "sat": self.sat_raw(),
            "unsat": self.unsat_raw(),
            "unknown": self.unknown_raw(),
            "consistent_solves": self.consistent_solves,
            "disabled": self.disabled,
            "witness": self.witness,
        }


# ==========================================================================
# Evaluation plan + controller
# ==========================================================================


class Plan:
    """The evaluation plan for one candidate, produced by the controller.

    * ``inferred`` -- a madeit string if the verdict was inferred without any
      solver call, else ``None``.
    * ``timeouts`` -- ordered per-solve z3 timeouts (seconds) to attempt; the GA
      stops at the first non-UNDECIDED verdict (two-tier escalation).
    * ``position`` / ``value`` -- the single monotone mutation, if any, so the
      controller can update its regions after the real solve.
    * ``include_in_arff`` -- whether this candidate feeds the ARFF dataset
      (guide mode excludes inferred rows).
    * ``inferred_flag`` -- whether the recorded verdict came from inference.
    """

    __slots__ = (
        "inferred",
        "err",
        "timeouts",
        "position",
        "value",
        "vector",
        "positions",
        "include_in_arff",
        "inferred_flag",
        "decision",
    )

    def __init__(
        self,
        inferred: Optional[str] = None,
        err: str = "",
        timeouts: Optional[List[int]] = None,
        position: Optional[int] = None,
        value: Optional[float] = None,
        vector: Optional[List[float]] = None,
        positions: Optional[List[int]] = None,
        include_in_arff: bool = True,
        inferred_flag: bool = False,
        decision: Optional[Decision] = None,
    ) -> None:
        self.inferred = inferred
        self.err = err
        self.timeouts = timeouts or []
        self.position = position    # single-knob position (k=1), else None
        self.value = value          # single-knob value (k=1), else None
        self.vector = vector        # full monotone-knob vector (k-D), or None
        self.positions = positions  # coordinate order for ``vector``
        self.include_in_arff = include_in_arff
        self.inferred_flag = inferred_flag
        self.decision = decision


class HeuristicsController:
    """Owns polarity gating, region state, two-tier policy, and counters.

    A single controller is created per GA run.  It is thread-safe (region state
    is guarded by a lock) so it composes with parallel evaluation, though region
    updates are then order-dependent.
    """

    def __init__(
        self,
        seed_ast: Optional[Formula],
        heuristics_cfg,
        trace_check_timeout_sec: int,
        run_dir: Optional[str] = None,
        trace_period: Optional[float] = None,
    ) -> None:
        self.ii = heuristics_cfg.interval_inference
        self.tt = heuristics_cfg.two_tier_timeout
        self.ar = heuristics_cfg.adaptive_range
        # Sample-aligned time quantization (Sprint 7) is orthogonal to the plan
        # path: it only rewrites the verdict-cache key, so it is NOT folded into
        # ``any_on`` (that would reroute quant-only runs through the plan path).
        from .quantization import Quantizer

        self.quant = Quantizer(
            seed_ast, heuristics_cfg.time_quantization, trace_period
        )
        self.quant_on = self.quant.enabled
        self.trace_timeout = int(trace_check_timeout_sec)
        self.seed_ast = seed_ast
        self.mode = self.ii.mode
        self.two_tier = bool(self.tt.enabled)
        self.adaptive_on = bool(self.ar.enabled)
        self.inference_on = bool(self.ii.enabled) and seed_ast is not None
        self.any_on = self.two_tier or self.inference_on or self.adaptive_on

        # Polarity of every numeric position (for reporting), and the subset that
        # is monotone -- the knobs the k-D region reasons over.
        self._all_directions: Dict[int, Monotonicity] = {}
        self._M: List[int] = []
        self._M_dirs: List[Monotonicity] = []
        self._seed_vals: Dict[int, float] = {}
        if seed_ast is not None:
            self._seed_vals = numeric_value_map(seed_ast)
            for p in numeric_positions(seed_ast):
                # With quantization on, sample-aligned window bounds are promoted
                # from UNKNOWN to a definite (floor-monotone) direction so they can
                # act as class-index adaptive knobs; off, this is unchanged.
                d = polarity(seed_ast, p, quantize=self.quant_on)
                self._all_directions[p] = d
                if d in (Monotonicity.INCREASING, Monotonicity.DECREASING):
                    self._M.append(p)
                    self._M_dirs.append(d)
        self._M_set = set(self._M)
        # One joint monotone region over all monotone knobs (k-D). None if there
        # are no monotone knobs (nothing is inferable).
        self.region: Optional[MonotoneRegion] = (
            MonotoneRegion(self._M, self._M_dirs) if (self.any_on and self._M) else None
        )
        self._lock = threading.Lock()
        self.state_path = Path(run_dir) / "inference_state.json" if run_dir else None
        # Sidecar rows for label mode: (arrf_row, madeit, inferred_bool).
        self.inferred_sidecar: List[Tuple[str, str, bool]] = []

        self.counters: Dict[str, int] = {
            "inferred_satisfied": 0,
            "inferred_violated": 0,
            "inferred_undecided": 0,
            "real_solves": 0,
            "region_memory_skips": 0,
            "tier1_decided": 0,
            "tier2_decided": 0,
            "tier2_undecided": 0,
            "monotonicity_violations": 0,
        }
        self.violations: List[dict] = []
        self.adaptive_observations: Dict[int, Dict[str, List[float]]] = {}
        self.endpoint_verdicts: List[dict] = []
        self.one_class_findings: List[dict] = []
        self.widenings: List[dict] = []
        self.adaptive_bands: Dict[int, dict] = {}
        self.adaptive_counters: Dict[str, int] = {
            "adaptive_draws_bracket": 0,
            "adaptive_draws_exploration": 0,
            "adaptive_draws_full_fallback": 0,
            "adaptive_endpoint_probes": 0,
        }

    # -- detection -----------------------------------------------------------
    def _detect(
        self, cand_ast: Optional[Formula]
    ) -> Optional[Tuple[List[float], Optional[Tuple[int, float]]]]:
        """Return ``(vector, single)`` for a candidate that only moves monotone knobs.

        ``vector`` is the candidate's value at every monotone knob (in ``self._M``
        order; seed value where unchanged). ``single`` is ``(position, value)``
        when exactly one knob changed (for reporting/adaptive range), else ``None``.

        Returns ``None`` -- i.e. "solve, do not infer" -- if the candidate differs
        structurally, differs at a non-monotone numeric knob, or there are no
        monotone knobs at all.
        """
        if self.region is None or cand_ast is None or self.seed_ast is None:
            return None
        diffs = ast_numeric_diffs(self.seed_ast, cand_ast)
        if diffs is None:
            return None
        # Every changed numeric position must be a monotone knob; a change at an
        # UNKNOWN-polarity numeric position breaks monotonicity of the whole
        # candidate, so it must be solved.
        for p, _v in diffs:
            if p not in self._M_set:
                return None
        changed = dict(diffs)
        vector = [changed.get(p, self._seed_vals.get(p, 0.0)) for p in self._M]
        single = diffs[0] if len(diffs) == 1 else None
        return vector, single

    def _tier_timeouts(self, region_memory: bool) -> List[int]:
        if not self.two_tier:
            return [self.trace_timeout]
        if region_memory:
            return [self.tt.low_sec]
        return [self.tt.low_sec, self.tt.high_sec]

    # -- public API ---------------------------------------------------------
    def plan(self, cand_ast: Optional[Formula]) -> Plan:
        """Decide how to evaluate a candidate. Never solves; only plans."""
        if not self.any_on:
            return Plan(timeouts=[self.trace_timeout])

        detected = self._detect(cand_ast)
        if detected is None:
            # Structural change or a non-monotone knob moved: solve, no bookkeeping.
            return Plan(timeouts=self._tier_timeouts(region_memory=False))

        vector, single = detected
        with self._lock:
            region = self.region
            if self.inference_on and not region.disabled:
                decision = region.decide(vector, self.ii.empirical_validation_k)
                if decision is Decision.INFER_SAT:
                    self.counters["inferred_satisfied"] += 1
                    return self._inferred_plan(MADEIT_SAT, vector, single, decision)
                if decision is Decision.INFER_VIOLATED:
                    self.counters["inferred_violated"] += 1
                    return self._inferred_plan(MADEIT_VIOLATED, vector, single, decision)
                if decision is Decision.INFER_UNDECIDED:
                    self.counters["inferred_undecided"] += 1
                    return self._inferred_plan(MADEIT_UNDECIDED, vector, single, decision)
                region_memory = decision is Decision.SOLVE_LOW_ONLY
            else:
                # inference off but two-tier on: use region memory if a band exists.
                region_memory = region.decide(vector, self.ii.empirical_validation_k) is Decision.SOLVE_LOW_ONLY
                decision = Decision.SOLVE_LOW_ONLY if region_memory else Decision.SOLVE_NORMAL

        if region_memory and self.two_tier:
            self.counters["region_memory_skips"] += 1
        return Plan(
            timeouts=self._tier_timeouts(region_memory=region_memory),
            vector=vector,
            positions=list(self._M),
            position=single[0] if single else None,
            value=single[1] if single else None,
            decision=decision,
        )

    def _inferred_plan(
        self, madeit: str, vector: List[float], single: Optional[Tuple[int, float]], decision: Decision
    ) -> Plan:
        err = "REQUIREMENT UNDECIDED" if madeit == MADEIT_UNDECIDED else ""
        return Plan(
            inferred=madeit,
            err=err,
            timeouts=[],
            vector=vector,
            positions=list(self._M),
            position=single[0] if single else None,
            value=single[1] if single else None,
            include_in_arff=(self.mode == "label"),
            inferred_flag=True,
            decision=decision,
        )

    def record_solve(self, plan: Plan, tier_verdicts: List[str], final_madeit: str) -> None:
        """Update counters + region after the GA ran ``plan.timeouts`` in order.

        ``tier_verdicts`` is the madeit produced at each attempted tier.
        """
        with self._lock:
            self.counters["real_solves"] += 1
            if self.two_tier and tier_verdicts:
                # tier 1 == low_sec (unless single-tier trace fallback)
                if len(plan.timeouts) >= 1 and plan.timeouts[0] == self.tt.low_sec:
                    if tier_verdicts[0] != MADEIT_UNDECIDED:
                        self.counters["tier1_decided"] += 1
                    elif len(tier_verdicts) >= 2:
                        if tier_verdicts[1] != MADEIT_UNDECIDED:
                            self.counters["tier2_decided"] += 1
                        else:
                            self.counters["tier2_undecided"] += 1

            if plan.vector is not None and (self.inference_on or self.adaptive_on) and self.region is not None:
                witness = self.region.update(plan.vector, final_madeit)
                if witness is not None:
                    self.counters["monotonicity_violations"] += 1
                    self.violations.append(witness)
            if self.adaptive_on and plan.position is not None and plan.value is not None:
                self._record_adaptive_observation(plan.position, plan.value, final_madeit)

    def record_arff_row(self, arrf_row: str, madeit: str, inferred: bool) -> None:
        """Record a label-mode sidecar row (schema-stable provenance of inference)."""
        if self.mode == "label":
            self.inferred_sidecar.append((arrf_row, madeit, inferred))

    # -- adaptive mutation range --------------------------------------------
    def numeric_range_positions(self, allowed_changes: Dict[int, Dict[str, object]]) -> List[int]:
        if not self.adaptive_on:
            return []
        return [
            p for p in self._M
            if isinstance(allowed_changes.get(p, {}).get("numeric"), (list, tuple))
        ]

    def record_endpoint_probe(self, position: int, value: float, madeit: str, role: str) -> None:
        if not self.adaptive_on:
            return
        with self._lock:
            self.adaptive_counters["adaptive_endpoint_probes"] += 1
            self.endpoint_verdicts.append({
                "position": position,
                "value": float(value),
                "role": role,
                "verdict": madeit,
            })
            self._record_adaptive_observation(position, value, madeit)

    def record_one_class_space(
        self,
        position: int,
        bounds: Tuple[float, float],
        madeit: str,
        witnesses: List[dict],
        action: str,
    ) -> dict:
        finding = {
            "position": position,
            "range": [float(bounds[0]), float(bounds[1])],
            "class": madeit,
            "witnesses": witnesses,
            "action": action,
        }
        with self._lock:
            self.one_class_findings.append(finding)
        return finding

    def record_widening(
        self,
        position: int,
        old_bounds: Tuple[float, float],
        new_bounds: Tuple[float, float],
        factor: float,
    ) -> None:
        with self._lock:
            self.widenings.append({
                "position": position,
                "old_range": [float(old_bounds[0]), float(old_bounds[1])],
                "new_range": [float(new_bounds[0]), float(new_bounds[1])],
                "factor": float(factor),
            })

    def _record_adaptive_observation(self, position: int, value: float, madeit: str) -> None:
        obs = self.adaptive_observations.setdefault(
            position, {"sat": [], "unsat": [], "unknown": []}
        )
        if madeit == MADEIT_SAT:
            obs["sat"].append(float(value))
        elif madeit == MADEIT_VIOLATED:
            obs["unsat"].append(float(value))
        elif madeit == MADEIT_UNDECIDED:
            obs["unknown"].append(float(value))
        self._maybe_record_unknown_band(position)

    def _orientation(self, position: int) -> Optional[int]:
        direction = self._all_directions.get(position)
        if direction is Monotonicity.INCREASING:
            return 1
        if direction is Monotonicity.DECREASING:
            return -1
        return None

    def _adaptive_bracket(self, position: int) -> Optional[Tuple[float, float]]:
        orient = self._orientation(position)
        if orient is None:
            return None
        obs = self.adaptive_observations.get(position)
        if not obs or not obs["sat"] or not obs["unsat"]:
            return None
        sat_min = min(orient * v for v in obs["sat"])
        unsat_max = max(orient * v for v in obs["unsat"])
        if sat_min - unsat_max <= max(self.ii.min_gap, _EPS):
            return None
        lo_o, hi_o = unsat_max, sat_min
        a, b = orient * lo_o, orient * hi_o
        return (min(a, b), max(a, b))

    def _maybe_record_unknown_band(self, position: int) -> None:
        if position in self.adaptive_bands:
            return
        orient = self._orientation(position)
        obs = self.adaptive_observations.get(position)
        if orient is None or not obs or len(obs["unknown"]) < 2:
            return
        unknown_o = sorted(orient * v for v in obs["unknown"])
        band_lo_o, band_hi_o = unknown_o[0], unknown_o[-1]
        if obs["sat"] and obs["unsat"]:
            sat_min = min(orient * v for v in obs["sat"])
            unsat_max = max(orient * v for v in obs["unsat"])
            if not (unsat_max <= band_lo_o + _EPS and band_hi_o <= sat_min + _EPS):
                return
        lo, hi = orient * band_lo_o, orient * band_hi_o
        lo, hi = min(lo, hi), max(lo, hi)
        self.adaptive_bands[position] = {
            "position": position,
            "interval": [lo, hi],
            "width": hi - lo,
        }

    def adapt_candidate(
        self,
        cand_ast: Optional[Formula],
        allowed_changes: Dict[int, Dict[str, object]],
        rng: Optional[random.Random] = None,
    ) -> Optional[Formula]:
        """Resample a single monotone numeric mutation from its current bracket.

        Returns a new AST when adaptive sampling applies, otherwise ``None``.
        Brackets are learned only from single-mutation solves/endpoints. If the
        runtime guard disables the shared monotone region, this reverts to the
        full configured range immediately.
        """
        if not self.adaptive_on or cand_ast is None:
            return None
        if rng is None:
            rng = random
        detected = self._detect(cand_ast)
        if detected is None:
            return None
        _vector, single = detected
        if single is None:
            return None
        position, _value = single
        bounds = allowed_changes.get(position, {}).get("numeric")
        if not (isinstance(bounds, (list, tuple)) and len(bounds) == 2):
            return None
        full_lo, full_hi = float(bounds[0]), float(bounds[1])
        if full_lo > full_hi:
            full_lo, full_hi = full_hi, full_lo

        # Combination (adaptive_range + time_quantization): a quantizable monotone
        # bound brackets on CLASS INDICES, not raw floats, and bisects to a
        # sample-aligned boundary. Falls through to the float bracket otherwise.
        if (
            self.quant_on
            and position in self.quant.positions
            and self._orientation(position) is not None
        ):
            return self._adapt_class_index(cand_ast, position, full_lo, full_hi, rng)

        with self._lock:
            guard_disabled = self.region is not None and self.region.disabled
            banded = position in self.adaptive_bands
            bracket = None if (guard_disabled or banded) else self._adaptive_bracket(position)
            explore = rng.random() < float(self.ar.exploration_fraction)
            if bracket is not None and not explore:
                lo, hi = bracket
                self.adaptive_counters["adaptive_draws_bracket"] += 1
            else:
                lo, hi = full_lo, full_hi
                if bracket is None:
                    self.adaptive_counters["adaptive_draws_full_fallback"] += 1
                else:
                    self.adaptive_counters["adaptive_draws_exploration"] += 1
        return set_numeric_at_position(cand_ast, position, rng.uniform(lo, hi))

    def _adapt_class_index(
        self,
        cand_ast: Formula,
        position: int,
        full_lo: float,
        full_hi: float,
        rng: random.Random,
    ) -> Optional[Formula]:
        """Resample a quantizable monotone bound on sample-class indices.

        The bracket lives on class indices (highest-known-UNSAT vs lowest-known-SAT
        class); ``min_gap`` for these positions is automatically one class, so the
        configured ``min_gap`` is ignored.  Exploitation draws take the bracket
        midpoint (bisection -> ``ceil(log2(#classes))`` real solves);
        ``exploration_fraction`` draws are uniform over the full class range.  The
        emitted mutation carries the class midpoint as a real time value -- only
        search and caching think in classes.  On convergence (a one-class bracket)
        the sample-timestamp boundary is recorded and shrinking stops.
        """
        from .quantization import bisect_probe_class, class_midpoint_value

        period = self.quant.positions[position]
        orient = self._orientation(position)
        lo_cls = math.floor(min(full_lo, full_hi) / period)
        hi_cls = math.floor(max(full_lo, full_hi) / period)
        with self._lock:
            obs = self.adaptive_observations.get(position)
            sat_cls = {math.floor(v / period) for v in obs["sat"]} if obs else set()
            unsat_cls = {math.floor(v / period) for v in obs["unsat"]} if obs else set()
            probe, boundary = bisect_probe_class(sat_cls, unsat_cls, orient)
            if boundary is not None:
                self.quant.boundary_at[str(position)] = float(boundary) * period
            explore = rng.random() < float(self.ar.exploration_fraction)
            if probe is not None and not explore:
                cls = probe
                self.adaptive_counters["adaptive_draws_bracket"] += 1
            else:
                cls = rng.randint(lo_cls, hi_cls)
                if sat_cls and unsat_cls:
                    self.adaptive_counters["adaptive_draws_exploration"] += 1
                else:
                    self.adaptive_counters["adaptive_draws_full_fallback"] += 1
            cls = max(lo_cls, min(hi_cls, cls))
        return set_numeric_at_position(cand_ast, position, class_midpoint_value(cls, period))

    # -- time quantization (Sprint 7) ---------------------------------------
    def quant_cache_expr(self, cand_ast, raw_expr: str) -> str:
        """Canonical verdict-cache expression (raw string when quantization off)."""
        return self.quant.canonical_expr(cand_ast, raw_expr)

    def quant_knob(self, position: int) -> bool:
        """Whether ``position`` is a quantization-managed (class-index) knob."""
        return self.quant_on and position in self.quant.positions

    def quant_vacuous(self, cand_ast) -> bool:
        """Flag + count a candidate whose mutable window is empty (a >= b)."""
        return self.quant.vacuous_flag(cand_ast)

    def quant_on_lookup(self, canonical: str, raw_expr: str, hit: bool) -> bool:
        """Record a cache lookup; return True when a validation double-solve is due."""
        return self.quant.on_lookup(canonical, raw_expr, hit)

    def quant_validate(self, cand_ast, canonical: str, cached: str, real: str) -> List[str]:
        """Compare a re-solved verdict to the cached one; return keys to purge."""
        return self.quant.record_validation(cand_ast, canonical, cached, real)

    # -- reporting / persistence --------------------------------------------
    def report(self) -> dict:
        out = dict(self.counters)
        if self.quant_on:
            out.update(self.quant.report())
        out["inference_mode"] = self.mode
        out["two_tier_timeout"] = self.two_tier
        out["interval_inference"] = self.inference_on
        out["adaptive_range"] = self.adaptive_on
        out["monotone_knobs"] = list(self._M)
        if self.adaptive_on:
            out.update(self.adaptive_counters)
            out["adaptive_observations"] = self.adaptive_observations
            out["adaptive_endpoint_verdicts"] = self.endpoint_verdicts
            if self.one_class_findings:
                out["one_class_space"] = self.one_class_findings
            if self.widenings:
                out["adaptive_widenings"] = self.widenings
            if self.adaptive_bands:
                out["adaptive_unknown_bands"] = list(self.adaptive_bands.values())
        if self.violations:
            out["monotonicity_violation_witnesses"] = self.violations
        directions = {str(p): d.value for p, d in self._all_directions.items()}
        if directions:
            out["position_directions"] = directions
        return out

    def _legacy_regions(self) -> Optional[dict]:
        """1-D per-position view of the region for single-knob backward-compat."""
        if self.region is None or len(self._M) != 1:
            return None
        p = self._M[0]
        d = self.region.to_dict()
        return {
            str(p): {
                "direction": self._all_directions[p].value,
                "sat_values": [vec[0] for vec in d["sat"]],
                "unsat_values": [vec[0] for vec in d["unsat"]],
                "unknown_values": [vec[0] for vec in d["unknown"]],
                "consistent_solves": d["consistent_solves"],
                "disabled": d["disabled"],
                "witness": d["witness"],
            }
        }

    def persist(self) -> None:
        if self.state_path is None:
            return
        data = {
            "directions": {str(p): d.value for p, d in self._all_directions.items()},
            "monotone_knobs": list(self._M),
            "region": self.region.to_dict() if self.region is not None else None,
            "counters": self.counters,
            "violations": self.violations,
            "adaptive_observations": self.adaptive_observations,
            "adaptive_endpoint_verdicts": self.endpoint_verdicts,
            "one_class_space": self.one_class_findings,
            "adaptive_widenings": self.widenings,
            "adaptive_unknown_bands": list(self.adaptive_bands.values()),
        }
        legacy = self._legacy_regions()
        if legacy is not None:
            data["regions"] = legacy  # 1-D compatibility for existing tooling
        try:
            self.state_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            pass

    def write_sidecar(self, path: str) -> Optional[str]:
        """Write the label-mode inferred-provenance sidecar CSV; return its path."""
        if self.mode != "label" or not self.inferred_sidecar:
            return None
        out = Path(path) / "inferred_labels.csv"
        lines = ["inferred,verdict,row"]
        for row, madeit, inferred in self.inferred_sidecar:
            lines.append(f"{int(bool(inferred))},{madeit.upper()},{row}")
        try:
            out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError:
            return None
        return str(out)

"""Static quantizability analysis for sample-aligned time-window bounds.

Traces are sampled at a fixed period (10,000 time units in the AT/CC encodings),
and signals are accessed through a floor-indexed subscript
``v_speed[ToInt((t - offset) / PERIOD)]``.  A mutated *time-window bound* ``B`` in
a comparison such as ``t <= B`` therefore affects the trace-check verdict only
through ``floor(B / PERIOD)``: every ``B`` inside one inter-sample interval yields
a logically equivalent formula.  Such a numeric position is *quantizable*.

This module decides, purely from the formula structure, whether a numeric
position is a quantizable time bound and, if so, recovers its sampling PERIOD
from the index expression (cross-checked against the trace timestamp spacing).
It is conservative in the same sense as :mod:`diagnosis.lang.polarity`:
``quantizable = False`` is always the safe answer, so any unclassifiable path
disables quantization for that position rather than guessing.

Two structural shapes are recognised, mirroring the two admissible paths from the
constant to a signal access:

* (a) the constant is a *direct* operand of an ordering comparison against the
  quantified time variable (``t <= B``, AT1-style);
* (b) the constant feeds that comparison through *additive/affine* arithmetic
  (``t2 <= timestamps[i] + N``, AT53-style ``i2t(s) + N``).

Both are quantizable iff the quantified variable of the window comparison is used
*only* through floor-indexed subscripts (never a dense-time/interpolated signal
access).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .ast import (
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
from .polarity import Monotonicity

# Ordering comparisons are the only monotone-classifiable relops (shared with
# :mod:`diagnosis.lang.polarity`).
_ORDERING_OPS = ("<", "<=", ">", ">=")
# Floor-index markers: an index expression tagged with one of these maps dense
# time onto a sample class via truncation, so the window bound is quantizable.
_FLOOR_MARKERS = ("ToInt", "t2i", "floor")
# Divisor literal in an affine index ``(t - offset) / PERIOD``.
_DIVISOR_RE = re.compile(r"/\s*([0-9]+(?:\.[0-9]+)?)")
# A signal-like Var carried across the GA internal round-trip renders the whole
# ``base[index]`` access into a single Var name; this splits it back apart.
_SUBSCRIPT_VAR_RE = re.compile(r"^([A-Za-z_]\w*)\[(.+)\]$")
# A quantified variable declaration ``Real('t')`` / ``Int('t')`` carried as a
# single Var name across the GA internal round-trip.
_DECL_VAR_RE = re.compile(r"^(?:Real|Int|Bool)\(\s*'([^']+)'\s*\)$")
_EPS = 1e-9


@dataclass(frozen=True)
class QuantInfo:
    """Result of :func:`quantizability` for one numeric position.

    ``quantizable`` is the gate; ``period`` is the sampling period recovered from
    the floor index (``None`` when not quantizable); ``reason`` is a
    human-readable explanation for the CLI and logs.
    """

    quantizable: bool
    period: Optional[float]
    reason: str


class _QRecord:
    """Per-position facts from a single affine-aware preorder traversal.

    The preorder index is identical to
    :func:`diagnosis.lang.polarity._collect_records` (and therefore to
    ``mutation.api._walk`` / the ARFF layout): ForAll/Exists descend into the
    body, And/Or into args, Not/Implies-antecedent flip the sign, and
    Subscript/FuncCall are opaque leaves.  ``coeff_sign`` additionally tracks the
    sign of the constant's contribution to the affine RelOp operand that encloses
    it (``+1``/``-1``, or ``None`` when a non-affine node such as ``*``/``/``
    breaks affinity).
    """

    __slots__ = ("node", "sign", "relop_op", "side", "sibling", "coeff_sign")

    def __init__(self, node, sign, relop_op, side, sibling, coeff_sign) -> None:
        self.node = node
        self.sign = sign
        self.relop_op = relop_op
        self.side = side
        self.sibling = sibling
        self.coeff_sign = coeff_sign


def _collect_quant_records(formula: Formula) -> Dict[int, _QRecord]:
    """Assign preorder indices and gather sign/relop/affine context per node."""
    records: Dict[int, _QRecord] = {}
    counter = 0

    def visit(node, sign, relop_op, side, sibling, coeff_sign) -> None:
        nonlocal counter
        records[counter] = _QRecord(node, sign, relop_op, side, sibling, coeff_sign)
        counter += 1

        if isinstance(node, (ForAll, Exists)):
            visit(node.body, sign, None, None, None, None)
        elif isinstance(node, (And, Or)):
            for arg in node.args:
                visit(arg, sign, None, None, None, None)
        elif isinstance(node, Not):
            visit(node.arg, -sign, None, None, None, None)
        elif isinstance(node, Implies):
            visit(node.left, -sign, None, None, None, None)
            visit(node.right, sign, None, None, None, None)
        elif isinstance(node, RelOp):
            # Direct operands start a fresh affine chain (coeff +1).
            visit(node.left, sign, node.op, "left", node.right, 1)
            visit(node.right, sign, node.op, "right", node.left, 1)
        elif isinstance(node, ArithOp):
            # Preserve the enclosing RelOp context across additive arithmetic so
            # a bound offset ``... + N`` stays classifiable; ``-`` flips the
            # right operand's coefficient; ``*``/``/`` break affinity (coeff None).
            if node.op == "+":
                lc = rc = coeff_sign
            elif node.op == "-":
                lc = coeff_sign
                rc = None if coeff_sign is None else -coeff_sign
            else:
                lc = rc = None
            visit(node.left, sign, relop_op, side, sibling, lc)
            visit(node.right, sign, relop_op, side, sibling, rc)
        # Var / Subscript / FuncCall / constants are leaves.

    visit(formula, 1, None, None, None, None)
    return records


def _stringify_subscripts(formula: Formula) -> List[Tuple[str, str]]:
    """Return ``(base, index_string)`` for every signal-like subscript access.

    Handles both the parsed :class:`Subscript` form and the GA round-trip form
    where ``base[index]`` is carried inside a single :class:`Var` name.
    """
    out: List[Tuple[str, str]] = []

    def index_str(node: Formula) -> str:
        if isinstance(node, Var):
            return node.name
        if isinstance(node, (IntConst, RealConst, BoolConst)):
            return str(node)
        if isinstance(node, FuncCall):
            return f"{node.func}(" + ",".join(index_str(a) for a in node.args) + ")"
        if isinstance(node, ArithOp):
            return f"({index_str(node.left)}{node.op}{index_str(node.right)})"
        if isinstance(node, Subscript):
            return f"{index_str(node.base)}[{index_str(node.index)}]"
        return str(node)

    def walk(node: Formula) -> None:
        if isinstance(node, Subscript):
            base = node.base.name if isinstance(node.base, Var) else str(node.base)
            out.append((base, index_str(node.index)))
            return  # opaque leaf, like the preorder analysers
        if isinstance(node, Var):
            m = _SUBSCRIPT_VAR_RE.match(node.name)
            if m:
                out.append((m.group(1), m.group(2)))
            return
        if isinstance(node, (RelOp, ArithOp, Implies)):
            walk(node.left)
            walk(node.right)
        elif isinstance(node, Not):
            walk(node.arg)
        elif isinstance(node, (And, Or)):
            for a in node.args:
                walk(a)
        elif isinstance(node, (ForAll, Exists)):
            walk(node.body)
        elif isinstance(node, FuncCall):
            for a in node.args:
                walk(a)

    walk(formula)
    return out


def _is_floor_index(index_string: str) -> bool:
    return any(marker in index_string for marker in _FLOOR_MARKERS)


def detect_period(formula: Formula) -> Optional[float]:
    """Recover the sampling period from floor-indexed signal accesses.

    Returns the common divisor of every ``ToInt(... / PERIOD)`` signal index, or
    ``None`` when there is no floor index or the divisors disagree (ambiguous
    encoding -> conservatively not quantizable).
    """
    periods: set[float] = set()
    for _base, idx in _stringify_subscripts(formula):
        if not _is_floor_index(idx):
            continue
        matches = _DIVISOR_RE.findall(idx)
        divisors = {float(m) for m in matches if float(m) != 0.0}
        if len(divisors) != 1:
            return None
        periods.add(next(iter(divisors)))
    if len(periods) != 1:
        return None
    period = next(iter(periods))
    return period if period > 0 else None


def _var_name_of(node: Formula) -> Optional[str]:
    """Return the bare variable name a node denotes, else ``None``.

    Handles the two forms a quantified variable takes across the pipeline: a bare
    ``Var('t')`` (constructed/round-trip ASTs) and the ThEodorE declaration form
    ``Real('t')`` / ``Int('t')`` (a :class:`FuncCall` wrapping a quoted-name
    :class:`Var`).  Signal accesses (``Var`` names carrying ``base[index]``) and
    non-variable terms return ``None``.
    """
    if isinstance(node, Var):
        if _SUBSCRIPT_VAR_RE.match(node.name):
            return None
        decl = _DECL_VAR_RE.match(node.name)
        if decl:  # round-trip form ``Real('t')`` carried as one Var name
            return decl.group(1)
        return node.name.strip("'\"")
    if isinstance(node, FuncCall) and node.func in ("Real", "Int", "Bool") and len(node.args) == 1:
        return _var_name_of(node.args[0])
    return None


def _time_var_floor_only(
    formula: Formula, var: str, records: Dict[int, _QRecord]
) -> Tuple[bool, str]:
    """Whether quantified ``var`` reaches signals only through floor indices.

    ``var`` must (i) appear inside at least one floor index, (ii) never appear in
    a non-floor (dense/interpolated) signal index, and (iii) occur as a bare term
    only as an operand of ordering comparisons (its window bounds).  Occurrences
    buried inside subscript indices are opaque leaves and are inspected via the
    stringified index expressions rather than the preorder records.
    """
    token = re.compile(r"\b" + re.escape(var) + r"\b")
    found_floor = False
    for base, idx in _stringify_subscripts(formula):
        if base == "timestamps":
            continue  # time axis, not a dense signal reading
        if not token.search(idx):
            continue
        if _is_floor_index(idx):
            found_floor = True
        else:
            return False, f"variable {var!r} reaches signal {base!r} through a dense (non-floor) index"
    if not found_floor:
        return False, f"variable {var!r} is not used in any floor-indexed subscript"
    for rec in records.values():
        if _var_name_of(rec.node) == var:
            if rec.relop_op not in _ORDERING_OPS:
                return False, f"variable {var!r} occurs outside an ordering window comparison"
    return True, "floor-indexed monotone time variable"


def _window_var(rec: _QRecord) -> Optional[str]:
    """The quantified time variable a bound at ``rec`` is compared against."""
    return _var_name_of(rec.sibling)


def quantizability(
    formula: Formula,
    position: int,
    trace_period: Optional[float] = None,
    period_override: Optional[float] = None,
    force: bool = False,
) -> QuantInfo:
    """Return the :class:`QuantInfo` for the numeric constant at ``position``.

    ``position`` is a preorder node index (as in ``allowed_positions``).  A
    position is quantizable iff it is (directly or through additive/affine
    arithmetic) an operand of an ordering comparison whose other operand is a
    quantified time variable used only through floor-indexed subscripts, and a
    single sampling period can be recovered.

    ``trace_period`` (when given) is cross-checked against the recovered index
    divisor; a mismatch disables quantization.  ``period_override`` forces the
    period for exotic encodings but must still pass the cross-check unless
    ``force`` is set (validation-only escape hatch).  ``UNKNOWN``-style
    conservatism: any failure returns ``quantizable=False`` with a reason.
    """
    records = _collect_quant_records(formula)
    rec = records.get(position)
    if rec is None:
        return QuantInfo(False, None, f"position {position} is not reachable")
    if not isinstance(rec.node, (IntConst, RealConst)):
        return QuantInfo(False, None, "position is not a numeric constant")
    if rec.relop_op is None:
        return QuantInfo(False, None, "constant does not feed an ordering comparison")
    if rec.relop_op not in _ORDERING_OPS:
        return QuantInfo(False, None, f"non-ordering operator {rec.relop_op!r}")
    if rec.coeff_sign is None:
        return QuantInfo(False, None, "constant reaches the comparison through non-affine arithmetic")

    var = _window_var(rec)
    if var is None:
        return QuantInfo(
            False, None, "comparison is against a signal/term, not a quantified time variable"
        )
    ok, why = _time_var_floor_only(formula, var, records)
    if not ok:
        return QuantInfo(False, None, why)

    index_period = detect_period(formula)
    if index_period is None:
        return QuantInfo(False, None, "no single floor-index divisor (period) could be recovered")

    period = index_period
    if period_override is not None:
        if not force and abs(period_override - index_period) > _EPS:
            return QuantInfo(
                False,
                None,
                f"period override {period_override} disagrees with index divisor {index_period}",
            )
        period = period_override
    if trace_period is not None and abs(period - trace_period) > _EPS:
        return QuantInfo(
            False,
            None,
            f"index divisor {period} disagrees with trace timestamp spacing {trace_period}",
        )
    return QuantInfo(True, period, f"sample-aligned time bound on {var!r} (period={period})")


def quantized_direction(
    formula: Formula, position: int, info: Optional[QuantInfo] = None
) -> Tuple[Monotonicity, str]:
    """Direction of the SATISFIED verdict in a quantizable window bound.

    The floor of a monotone index map is monotone non-decreasing, so a quantized
    time bound orients the verdict exactly like a signal threshold would: apply
    the ordering base rule (``<``/``<=`` -> ``+1``, ``>``/``>=`` -> ``-1``),
    mirror it when the constant is on the left, then fold in the accumulated
    polarity sign and the affine coefficient sign.  Returns ``UNKNOWN`` if the
    position is not quantizable.
    """
    if info is None:
        info = quantizability(formula, position)
    if not info.quantizable:
        return Monotonicity.UNKNOWN, info.reason
    rec = _collect_quant_records(formula)[position]
    base = 1 if rec.relop_op in ("<", "<=") else -1
    if rec.side == "left":
        base = -base
    final = base * rec.sign * rec.coeff_sign
    direction = Monotonicity.INCREASING if final > 0 else Monotonicity.DECREASING
    return direction, f"quantizable {info.reason}; monotone floor index -> {direction.value}"


def window_variable(formula: Formula, position: int) -> Optional[str]:
    """The quantified time variable the bound at ``position`` compares against."""
    rec = _collect_quant_records(formula).get(position)
    return _window_var(rec) if rec is not None else None


def bound_role(formula: Formula, position: int) -> Optional[str]:
    """Classify a window bound as ``"lower"`` or ``"upper"`` (else ``None``).

    ``t <= B`` / ``B >= t`` make ``B`` an *upper* bound of the window; ``B <= t``
    / ``t >= B`` make it a *lower* bound.  Used by the vacuity guard to decide
    when a candidate's ``[lower, upper]`` window is empty.
    """
    rec = _collect_quant_records(formula).get(position)
    if rec is None or rec.relop_op not in _ORDERING_OPS:
        return None
    on_right = rec.side == "right"
    lt = rec.relop_op in ("<", "<=")
    # var <op> B (B on right, <) -> upper ;  B <op> var (B on left, <) -> lower.
    if (on_right and lt) or (not on_right and not lt):
        return "upper"
    return "lower"


def quantizable_positions(
    formula: Formula,
    positions: Optional[Iterable[int]] = None,
    trace_period: Optional[float] = None,
    period_override: Optional[float] = None,
    force: bool = False,
) -> Dict[int, QuantInfo]:
    """Return ``{position: QuantInfo}`` for the requested (or all numeric) positions."""
    from .polarity import numeric_positions

    targets = list(positions) if positions is not None else numeric_positions(formula)
    return {
        p: quantizability(formula, p, trace_period, period_override, force)
        for p in sorted(set(targets))
    }

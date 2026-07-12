"""Static monotonicity (polarity) analysis for numeric threshold parameters.

The genetic search mutates a single numeric constant at a fixed AST *position*
(a preorder node index, identical to the numbering produced by
``internal_encoder._collect_positions`` and ``mutation.api._walk``).  For many
requirements the *verdict* of the trace check is monotone in that constant: as a
speed limit ``N`` in ``v_speed < N`` grows, the property can only become "more
satisfied".  Proving that direction statically lets the inference layer
(``diagnosis.inference``) replace solver calls by half-line lookups.

This module derives, purely from the formula structure, whether the SATISFIED
verdict is non-decreasing (``INCREASING``) or non-increasing (``DECREASING``) in
the constant at a position, or whether it cannot be classified (``UNKNOWN``).

``UNKNOWN`` is deliberately conservative: it means "inference disabled for this
position", never a guess.  A ``DECREASING``/``INCREASING`` result is only ever a
*gate* for inference; the runtime consistency guard in ``diagnosis.inference`` is
the actual soundness net.

Semantics of the verdict.  ``load_formula_from_property`` strips the outer
negation, so the analysed AST is the (non-negated) requirement ``phi``; the GA
asserts ``Not(phi)`` and reads UNSAT as SATISFIED.  SATISFIED therefore means
"phi is valid over the trace", so monotonicity of the SATISFIED verdict is
exactly monotonicity of the validity of ``phi`` in the constant -- which is what
the structural rules below compute.

Literature basis.  The polarity/monotonicity rule is the parametric-STL
monotonicity of Asarin, Donze, Maler and Nickovic, "Parametric Identification of
Temporal Properties" (RV 2011); the downstream validity-domain bisection it
enables is Jin, Donze, Deshmukh and Seshia, "Mining Requirements from
Closed-Loop Control Models" (HSCC'13 / IEEE TCAD 2015).  (References stated from
memory; verify page/theorem numbers before citing in the paper.)
"""

from __future__ import annotations

from enum import Enum
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


class Monotonicity(Enum):
    """Direction of the SATISFIED verdict as a function of a numeric constant."""

    INCREASING = "INCREASING"
    DECREASING = "DECREASING"
    UNKNOWN = "UNKNOWN"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


# Half-line orderings: for an INCREASING position the SATISFIED region is the
# upper half-line ``[cut, +inf)``; for DECREASING it is ``(-inf, cut]``.  These
# constants are re-exported for the inference layer so both modules agree.
_ORDERING_OPS = ("<", "<=", ">", ">=")
_EQUALITY_OPS = ("==", "!=")


class _Record:
    """Per-position analysis facts gathered in a single preorder traversal."""

    __slots__ = ("node", "sign", "relop_op", "side", "sibling")

    def __init__(
        self,
        node: Formula,
        sign: int,
        relop_op: Optional[str],
        side: Optional[str],
        sibling: Optional[Formula],
    ) -> None:
        self.node = node
        self.sign = sign            # accumulated +1 / -1 polarity flag
        self.relop_op = relop_op    # op of the enclosing RelOp, if a direct operand
        self.side = side            # "left" | "right" position of the constant
        self.sibling = sibling      # the other RelOp operand


def _contains_signal(node: Formula) -> bool:
    """Whether a subtree contains a signal access (a ``Subscript``).

    Threshold comparisons (``v_speed[...] < N``) always reference a signal via a
    subscript; temporal-domain bounds (``0 <= Real('t')``) never do.  Requiring a
    signal on the *other* side of the RelOp is the structural gate that keeps the
    analysis conservative about time-window bounds -- the prompt mandates
    UNKNOWN for those even though a lone ``forall`` interval bound is technically
    monotone, because in nested-quantifier requirements the same bound is not.
    """
    stack: List[Formula] = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, Subscript):
            return True
        # After the internal round-trip used by the GA, a signal access
        # ``v_speed[ToInt(...)]`` is carried as a Var whose name contains the
        # subscript brackets; temporal bound terms (``Real('t')``, ``t``) do not.
        if isinstance(cur, Var) and "[" in cur.name and "]" in cur.name:
            return True
        if isinstance(cur, (RelOp, ArithOp)):
            stack.append(cur.left)
            stack.append(cur.right)
        elif isinstance(cur, Not):
            stack.append(cur.arg)
        elif isinstance(cur, (And, Or)):
            stack.extend(cur.args)
        elif isinstance(cur, Implies):
            stack.append(cur.left)
            stack.append(cur.right)
        elif isinstance(cur, (ForAll, Exists)):
            stack.append(cur.body)
        elif isinstance(cur, FuncCall):
            stack.extend(cur.args)
    return False


def _collect_records(formula: Formula) -> Dict[int, _Record]:
    """Assign preorder indices and gather sign + RelOp context per node.

    The traversal order is identical to
    ``internal_encoder._collect_positions`` and ``mutation.api._walk`` so the
    indices coincide with the positions used in configs and the ARFF layout.
    Subscript/FuncCall subtrees are treated as opaque leaves (no recursion),
    matching those two functions, so a constant buried in a ``ToInt(...)`` index
    expression never receives an addressable position.
    """
    records: Dict[int, _Record] = {}
    counter = 0

    def visit(node: Formula, sign: int, relop_op, side, sibling) -> None:
        nonlocal counter
        records[counter] = _Record(node, sign, relop_op, side, sibling)
        counter += 1

        if isinstance(node, (ForAll, Exists)):
            visit(node.body, sign, None, None, None)
        elif isinstance(node, (And, Or)):
            for arg in node.args:
                visit(arg, sign, None, None, None)
        elif isinstance(node, Not):
            visit(node.arg, -sign, None, None, None)
        elif isinstance(node, Implies):
            # Implies(a, b) == Or(Not(a), b): the antecedent flips polarity.
            visit(node.left, -sign, None, None, None)
            visit(node.right, sign, None, None, None)
        elif isinstance(node, RelOp):
            # Record RelOp context for direct operands so a numeric leaf knows
            # its threshold role; sign is preserved across the comparison.
            visit(node.left, sign, node.op, "left", node.right)
            visit(node.right, sign, node.op, "right", node.left)
        elif isinstance(node, ArithOp):
            # Constants under arithmetic are not clean thresholds -> no RelOp
            # context is propagated, which yields UNKNOWN downstream.
            visit(node.left, sign, None, None, None)
            visit(node.right, sign, None, None, None)
        # Var / Subscript / FuncCall / constants: leaves.

    visit(formula, 1, None, None, None)
    return records


def _classify(rec: _Record) -> Tuple[Monotonicity, str]:
    node = rec.node
    if not isinstance(node, (IntConst, RealConst)):
        return Monotonicity.UNKNOWN, "position is not a numeric constant"

    if rec.relop_op is None:
        return (
            Monotonicity.UNKNOWN,
            "constant is not a direct operand of a comparison "
            "(e.g. nested under arithmetic) - not a classifiable threshold",
        )

    if rec.relop_op in _EQUALITY_OPS:
        return Monotonicity.UNKNOWN, f"equality operator {rec.relop_op!r} is not monotone"

    if rec.relop_op not in _ORDERING_OPS:
        return Monotonicity.UNKNOWN, f"unsupported relational operator {rec.relop_op!r}"

    if rec.sibling is None or not _contains_signal(rec.sibling):
        return (
            Monotonicity.UNKNOWN,
            "compared against a temporal/parameter-free term (no signal); "
            "treated as a time-window bound",
        )

    # Base direction of SATISFIED w.r.t. the constant, constant on the right:
    #   expr < N / expr <= N  -> non-decreasing (+1)
    #   expr > N / expr >= N  -> non-increasing (-1)
    # If the constant is on the left, the comparison is mirrored -> invert.
    if rec.relop_op in ("<", "<="):
        base = 1
    else:  # ">", ">="
        base = -1
    if rec.side == "left":
        base = -base

    final = base * rec.sign
    direction = Monotonicity.INCREASING if final > 0 else Monotonicity.DECREASING

    flip = "flipped" if rec.sign < 0 else "preserved"
    reason = (
        f"threshold {rec.side} of '{rec.relop_op}' against a signal term; "
        f"polarity flag {flip} ({'+' if rec.sign > 0 else '-'}) -> {direction.value}"
    )
    return direction, reason


def polarity(formula: Formula, position: int, quantize: bool = False) -> Monotonicity:
    """Return the monotonicity of the SATISFIED verdict in the constant at ``position``.

    ``position`` is a preorder node index (as in ``allowed_positions``).  Returns
    ``UNKNOWN`` whenever the direction cannot be soundly proven, including when
    the position is unreachable or is not a numeric constant.

    ``quantize`` is opt-in and defaults to ``False`` so the result is bit-for-bit
    identical to the pre-Sprint-7 analysis.  When ``True``, a window-bound
    constant that the structural rules would leave ``UNKNOWN`` is promoted to a
    definite direction if it is a sample-aligned (floor-indexed) time bound --
    see :mod:`diagnosis.lang.quantize`.
    """
    return polarity_with_reason(formula, position, quantize)[0]


def polarity_with_reason(
    formula: Formula, position: int, quantize: bool = False
) -> Tuple[Monotonicity, str]:
    """Like :func:`polarity` but also return a human-readable explanation."""
    records = _collect_records(formula)
    rec = records.get(position)
    if rec is None:
        return (
            Monotonicity.UNKNOWN,
            f"position {position} is not reachable "
            "(outside the mutable AST, e.g. inside an index expression)",
        )
    direction, reason = _classify(rec)
    if quantize and direction is Monotonicity.UNKNOWN:
        # Lazy import avoids a module-load cycle (quantize imports Monotonicity).
        from .quantize import quantized_direction

        q_dir, q_reason = quantized_direction(formula, position)
        if q_dir is not Monotonicity.UNKNOWN:
            return q_dir, q_reason
    return direction, reason


def parameter_polarity(formula: Formula, positions: Iterable[int]) -> Monotonicity:
    """Combine the polarity of several positions bound to one logical parameter.

    A parameter that occurs with conflicting directions (or any UNKNOWN
    occurrence) is not monotone overall and yields ``UNKNOWN``.  With the tool's
    one-position-per-parameter configs this reduces to :func:`polarity`, but it
    makes the "same parameter reachable with both signs" case explicit.
    """
    seen: Optional[Monotonicity] = None
    any_position = False
    for pos in positions:
        any_position = True
        d = polarity(formula, pos)
        if d is Monotonicity.UNKNOWN:
            return Monotonicity.UNKNOWN
        if seen is None:
            seen = d
        elif seen is not d:
            return Monotonicity.UNKNOWN
    if not any_position or seen is None:
        return Monotonicity.UNKNOWN
    return seen


def numeric_positions(formula: Formula) -> List[int]:
    """Return the sorted preorder indices of all numeric constants."""
    records = _collect_records(formula)
    return sorted(
        idx for idx, rec in records.items()
        if isinstance(rec.node, (IntConst, RealConst))
    )


def explain(
    formula: Formula,
    positions: Optional[Iterable[int]] = None,
    quantize: bool = False,
) -> Dict[int, Tuple[Monotonicity, str]]:
    """Return ``{position: (direction, reason)}`` for the requested positions.

    If ``positions`` is ``None`` every numeric-constant position is described.
    ``quantize`` is threaded through to :func:`polarity_with_reason` (opt-in;
    default ``False`` preserves the pre-Sprint-7 output).
    """
    records = _collect_records(formula)
    if positions is None:
        targets: Iterable[int] = [
            idx for idx, rec in records.items()
            if isinstance(rec.node, (IntConst, RealConst))
        ]
    else:
        targets = positions
    out: Dict[int, Tuple[Monotonicity, str]] = {}
    for pos in sorted(set(targets)):
        rec = records.get(pos)
        if rec is None:
            out[pos] = (
                Monotonicity.UNKNOWN,
                f"position {pos} is not reachable",
            )
        else:
            out[pos] = polarity_with_reason(formula, pos, quantize)
    return out

"""Frozen grammar compilation diagnostics recorded with a training point."""

from json import dumps
from types import MappingProxyType
from typing import TYPE_CHECKING, final

from lofi._frozen import FrozenValue

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi._types import ReportValue
    from lofi.formalism import Grammar
    from lofi.judge import ClosureReport
    from lofi.terms import Term, TermCodec, TermSyntax

__all__ = ["CompileReport"]


def _json(value: object, /) -> str:
    """Return stable compact JSON for one structured measurement."""
    return dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _operators(terms: tuple[Term, ...], /) -> frozenset[str]:
    """Return every operator reached by a tuple of category terms."""
    reached: set[str] = set()
    pending = list(terms)

    while pending:
        term = pending.pop()
        reached.add(term.op.name)
        pending.extend(term.args)

    return frozenset(reached)


def _depth(term: Term, /) -> int:
    """Return the number of occupied levels in one term."""
    return 1 + max((_depth(argument) for argument in term.args), default=0)


@final
class CompileReport(FrozenValue):
    """Stable measurements and diagnostics of one compiled grammar."""

    _noun = "compile report"
    __slots__ = ("_measurements",)

    def __init__(
        self,
        grammar: Grammar,
        closure: ClosureReport[Term],
        syntax: TermSyntax,
        codec: TermCodec,
        /,
        *,
        category_sort: str,
    ) -> None:
        reached_names = _operators(closure.reached)
        reached_arities = frozenset(
            grammar.signature.by_name(name).arity for name in reached_names
        )
        category_operators = tuple(
            operator
            for operator in grammar.signature.operators
            if operator.sort == category_sort
        )
        diagnostics = tuple(
            f"E310: operator {operator.name!r} declares unreached arity "
            f"{operator.arity}"
            for operator in category_operators
            if operator.arity not in reached_arities
        )
        goal_depths = tuple(_depth(target) for target in grammar.targets.categories)
        capacity = grammar.capacity
        measurements: dict[str, ReportValue] = {
            "operator-codes": _json(
                {
                    operator.name: codec.index.index(operator)
                    for operator in grammar.signature.operators
                }
            ),
            "congruence": grammar.matching.name,
            "closure-category-set": closure.category_set,
            "closure-saturated": closure.saturated,
            "closure-steps": closure.steps,
            "closure-budget": closure.budget,
            "closure-reached": len(closure.reached),
            "closure-never-applicable": _json(closure.never_applicable),
            "closure-escaping": _json(
                tuple(
                    syntax.render(grammar.store, category)
                    for category in closure.escaping
                )
            ),
            "rule-directions": _json(
                {rule.name: sorted(rule.directions) for rule in grammar.rules}
            ),
            "codec-layout": codec.layout.name,
            "codec-branching": max(
                2,
                max(
                    (grammar.signature.by_name(name).arity for name in reached_names),
                    default=0,
                ),
            ),
            "codec-slots": codec.capacity,
            "goal-depths": _json(goal_depths),
            "mixed-goal-depths": len(set(goal_depths)) > 1,
            "diagnostics": _json(diagnostics),
            "capacity-hypotheses": _json(
                {
                    "result": capacity.result,
                    "theorem_applicable": capacity.theorem_applicable,
                    "rule_family": capacity.rule_family,
                    "restricted_rules": capacity.restricted_rules,
                    "properly_includes_context_free": (
                        capacity.properly_includes_context_free
                    ),
                    "parikh_context_free_sublanguage": (
                        capacity.parikh_context_free_sublanguage
                    ),
                    "citation": capacity.citation,
                    "reason": capacity.reason,
                }
            ),
        }
        object.__setattr__(self, "_measurements", MappingProxyType(measurements))

    def measurements(self) -> Mapping[str, ReportValue]:
        """Return the metrics-boundary projection of this report."""
        return self._measurements

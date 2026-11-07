"""Rule-law cases for generalized substitution."""

from random import Random
from typing import TYPE_CHECKING, Final

from hypothesis import strategies as st

from lofi.judge.laws import (
    RuleFixture,
    RuleLawCase,
    RuleLawCases,
    RuleTotalityCase,
    check_rule_laws,
)
from lofi.spec._rule import SubstitutionSpec
from lofi.spec._rule_laws import LawGrammar

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi._types import JsonValue
    from lofi.judge import Rule
    from lofi.spec._rule_laws import LawView, SlashRole
    from lofi.terms import Functors, OpName, Term

__all__ = ["provide_rule_laws"]

_R3_DRAWS: Final = 3_000
_R3_SEED: Final = 0
_ARGUMENTS: Final = ("z0", "z1")
_VIEWS: Final[tuple[LawView, ...]] = ("direct", "swapped", "undirected")
_ROLES: Final[tuple[SlashRole, ...]] = ("forward", "backward")


def _spine(
    grammar: LawGrammar,
    functors: Functors[Term],
    foot: Term,
    slashes: tuple[OpName, ...],
) -> Term:
    """Construct a result spine, its slashes from the outside inward."""
    current = foot

    for slash, argument in reversed(
        tuple(zip(slashes, _ARGUMENTS[: len(slashes)], strict=True))
    ):
        current = functors.make(slash, current, grammar.atom(argument))

    return current


def _terms(
    grammar: LawGrammar,
    role: SlashRole,
    order: int,
    *,
    crossed: bool,
) -> tuple[Term, ...]:
    """Return the conclusion and premises of one licensed substitution step."""
    functors, slash = grammar.functor(role)
    deepest = functors.opposite(slash) if crossed else slash
    slashes = (functors.opposite(deepest),) * (order - 1) + (deepest,)
    x = grammar.atom("x")
    y = grammar.atom("y")
    functor_term = _spine(grammar, functors, functors.make(slash, x, y), slashes)
    container = _spine(grammar, functors, y, slashes)
    output = _spine(grammar, functors, x, slashes)
    inputs = (
        (functor_term, container) if role == "forward" else (container, functor_term)
    )

    return (output, *inputs)


class _Rebuild:
    """Rebuild a substitution rule from its declaration."""

    __slots__ = ("_grammar",)

    def __init__(self, grammar: LawGrammar, /) -> None:
        self._grammar = grammar

    def __call__(self, spec: Mapping[str, JsonValue], /) -> Rule[Term]:
        """Compile the grammar the declaration names.

        Parameters
        ----------
        spec : Mapping[str, JsonValue]
            Rule declaration.

        Raises
        ------
        KeyError
            If a required field or the slash's view is missing.
        TypeError
            If a field has the wrong type.
        """
        order = spec["order"]
        slash = spec["slash"]
        crossed = spec["crossed"]

        if (
            isinstance(order, bool)
            or not isinstance(order, int | None)
            or not isinstance(slash, str)
            or not isinstance(crossed, bool)
        ):
            msg = f"invalid substitution declaration: {spec!r}"
            raise TypeError(msg)

        return self._grammar.rule(
            SubstitutionSpec(
                view=self._grammar.view_of(slash), order=order, crossed=crossed
            )
        )


def _ordinary_case(
    grammar: LawGrammar,
    role: SlashRole,
    order: int,
    *,
    crossed: bool,
) -> RuleLawCase:
    """Build one ordinary substitution case."""
    rule = grammar.rule(
        SubstitutionSpec(view=grammar.view_name(role), order=order, crossed=crossed)
    )
    terms = st.sampled_from(_terms(grammar, role, order, crossed=crossed))
    fixture = RuleFixture(rule, terms, _Rebuild(grammar), 200)
    variant = "crossed" if crossed else "harmonic"
    name = f"substitution/{grammar.view}/{role}/{order}/{variant}"

    def check() -> None:
        """Run every law for the captured substitution fixture."""
        check_rule_laws(fixture)

    return RuleLawCase(name, rule.name, check)


def _totality_case(grammar: LawGrammar, role: SlashRole) -> RuleTotalityCase:
    """Build one seeded totality case over the direct arrangement."""
    rule = grammar.rule(
        SubstitutionSpec(view=grammar.view_name(role), order=1, crossed=False)
    )
    functors, _ = grammar.functor("forward")
    x = grammar.atom("x")
    y = grammar.atom("y")
    pool = (
        x,
        y,
        *(grammar.atom(argument) for argument in _ARGUMENTS),
        grammar.atom("other"),
        functors.make(functors.forward_slash, x, y),
        functors.make(functors.backward_slash, x, y),
    )

    def check() -> int:
        """Run the captured seed-0 totality draws."""
        random = Random(_R3_SEED)
        calls = 0

        for _ in range(_R3_DRAWS):
            inputs = (random.choice(pool), random.choice(pool))
            output = random.choice(pool)
            rule.admits(inputs, output)
            calls += 1

        return calls

    return RuleTotalityCase(f"r3/substitution/{role}/1/harmonic", rule.name, check)


def provide_rule_laws() -> RuleLawCases:
    """Return ordinary and totality generalized-substitution cases."""
    grammars = {view: LawGrammar(view) for view in _VIEWS}
    ordinary = tuple(
        _ordinary_case(grammars[view], role, order, crossed=crossed)
        for view in _VIEWS
        for role in _ROLES
        for order in (1, 2)
        for crossed in (False, True)
    )
    totality = tuple(_totality_case(grammars["direct"], role) for role in _ROLES)

    return RuleLawCases("substitution", ordinary, totality)

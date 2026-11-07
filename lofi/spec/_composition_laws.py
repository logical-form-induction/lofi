"""Rule-law cases for application and generalized composition."""

from random import Random
from typing import TYPE_CHECKING, Final

from hypothesis import strategies as st

from lofi.judge import COMPOSITION_RESTRICTIONS
from lofi.judge.laws import (
    RuleFixture,
    RuleLawCase,
    RuleLawCases,
    RuleTotalityCase,
    check_rule_laws,
)
from lofi.spec._rule import ApplicationSpec, CompositionSpec
from lofi.spec._rule_laws import LawGrammar

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi._types import JsonValue
    from lofi.judge import CompositionRestriction, Rule
    from lofi.spec._rule import RuleSpec
    from lofi.spec._rule_laws import LawView, SlashRole
    from lofi.terms import Functors, OpName, Term

__all__ = ["provide_rule_laws"]

_R3_DRAWS: Final = 3_000
_R3_SEED: Final = 0
_UNBOUNDED_WITNESS: Final = 2
_ARGUMENTS: Final = ("z0", "z1")
_VIEWS: Final[tuple[LawView, ...]] = ("direct", "swapped", "undirected")
_ROLES: Final[tuple[SlashRole, ...]] = ("forward", "backward")


def _declaration(
    grammar: LawGrammar,
    role: SlashRole,
    order: int | None,
    direction: CompositionRestriction,
    *,
    crossed: bool,
) -> RuleSpec:
    """Return the declaration of one application or composition rule."""
    view = grammar.view_name(role)

    if order == 0:
        return ApplicationSpec(view=view)

    return CompositionSpec(view=view, order=order, direction=direction, crossed=crossed)


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


def _spine_slashes(
    functors: Functors[Term],
    slash: OpName,
    order: int,
    direction: CompositionRestriction,
    *,
    crossed: bool,
) -> tuple[OpName, ...]:
    """Return a spine the direction restriction licenses and a stricter one refuses."""
    if order == 0:
        return ()

    named = functors.opposite(slash) if crossed else slash
    deepest = named if direction != "unrestricted" else functors.opposite(named)
    shallow = deepest if direction == "steedman" else functors.opposite(deepest)

    return (*(shallow for _ in range(order - 1)), deepest)


def _terms(
    grammar: LawGrammar,
    role: SlashRole,
    order: int,
    direction: CompositionRestriction,
    *,
    crossed: bool,
) -> tuple[Term, ...]:
    """Return the conclusion and premises of one licensed composition step."""
    functors, slash = grammar.functor(role)
    slashes = _spine_slashes(functors, slash, order, direction, crossed=crossed)
    x = grammar.atom("x")
    y = grammar.atom("y")
    principal = functors.make(slash, x, y)
    container = _spine(grammar, functors, y, slashes)
    output = _spine(grammar, functors, x, slashes)
    inputs = (principal, container) if role == "forward" else (container, principal)

    return (output, *inputs)


class _Rebuild:
    """Rebuild an application or composition rule from its declaration."""

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
        direction = next(
            (known for known in COMPOSITION_RESTRICTIONS if known == spec["direction"]),
            None,
        )

        if (
            isinstance(order, bool)
            or not isinstance(order, int | None)
            or not isinstance(slash, str)
            or not isinstance(crossed, bool)
            or direction is None
        ):
            msg = f"invalid composition declaration: {spec!r}"
            raise TypeError(msg)

        view = self._grammar.view_of(slash)

        return self._grammar.rule(
            ApplicationSpec(view=view)
            if order == 0
            else CompositionSpec(
                view=view, order=order, direction=direction, crossed=crossed
            )
        )


def _witness_order(order: int | None) -> int:
    """Return the order whose witnesses exercise a declaration."""
    return _UNBOUNDED_WITNESS if order is None else order


def _declared(order: int | None) -> str:
    """Return the declaration's name segment."""
    return "any" if order is None else str(order)


def _application_case(grammar: LawGrammar, role: SlashRole) -> RuleLawCase:
    """Build one ordinary application case."""
    rule = grammar.rule(ApplicationSpec(view=grammar.view_name(role)))
    terms = st.sampled_from(_terms(grammar, role, 0, "unrestricted", crossed=False))
    fixture = RuleFixture(rule, terms, _Rebuild(grammar), 200)

    def check() -> None:
        """Run every law for the captured application fixture."""
        check_rule_laws(fixture)

    return RuleLawCase(f"application/{grammar.view}/{role}", rule.name, check)


def _ordinary_case(
    grammar: LawGrammar,
    role: SlashRole,
    order: int | None,
    direction: CompositionRestriction,
    *,
    crossed: bool,
) -> RuleLawCase:
    """Build one ordinary composition case."""
    rule = grammar.rule(_declaration(grammar, role, order, direction, crossed=crossed))
    terms = st.sampled_from(
        _terms(grammar, role, _witness_order(order), direction, crossed=crossed)
    )
    fixture = RuleFixture(rule, terms, _Rebuild(grammar), 200)
    variant = "crossed" if crossed else "harmonic"
    name = f"composition/{grammar.view}/{role}/{_declared(order)}/{variant}/{direction}"

    def check() -> None:
        """Run every law for the captured composition fixture."""
        check_rule_laws(fixture)

    return RuleLawCase(name, rule.name, check)


def _totality_case(
    grammar: LawGrammar,
    role: SlashRole,
    order: int | None,
    direction: CompositionRestriction,
    *,
    crossed: bool,
) -> RuleTotalityCase:
    """Build one seeded totality case over the direct arrangement."""
    rule = grammar.rule(_declaration(grammar, role, order, direction, crossed=crossed))
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
    variant = "crossed" if crossed else "harmonic"
    name = (
        f"r3/application/{role}"
        if order == 0
        else f"r3/composition/{role}/{_declared(order)}/{variant}/{direction}"
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

    return RuleTotalityCase(name, rule.name, check)


def provide_rule_laws() -> RuleLawCases:
    """Return ordinary and totality application and composition cases."""
    grammars = {view: LawGrammar(view) for view in _VIEWS}
    applications = tuple(
        _application_case(grammars[view], role) for view in _VIEWS for role in _ROLES
    )
    compositions = tuple(
        _ordinary_case(grammars[view], role, order, direction, crossed=crossed)
        for view in _VIEWS
        for role in _ROLES
        for order in (2, None)
        for direction in COMPOSITION_RESTRICTIONS
        for crossed in (False, True)
    )
    direct = grammars["direct"]
    totality = (
        *(
            _totality_case(direct, role, 0, "unrestricted", crossed=False)
            for role in _ROLES
        ),
        *(
            _totality_case(direct, role, order, direction, crossed=crossed)
            for role in _ROLES
            for order in (1, 2, None)
            for direction in COMPOSITION_RESTRICTIONS
            for crossed in (False, True)
        ),
    )

    return RuleLawCases("composition", (*applications, *compositions), totality)

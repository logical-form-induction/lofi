"""Small generated categorial grammars for chart differentials.

Every generated grammar is a declaration over one featured CCG signature,
compiled through the native grammar compiler, so the rules a differential
checks are the rules a practitioner's grammar would run. All compiles share
the signature's term store, so a category built here is a premise of every
generated rule.
"""

from typing import TYPE_CHECKING, Final, NamedTuple

from hypothesis import strategies as st
from tests.support.rules import ccg_signature, declared_rules

from lofi.judge import COMPOSITION_RESTRICTIONS, RuleSet
from lofi.spec import (
    ApplicationSpec,
    CompositionSpec,
    ConstantPatternSpec,
    FixedOperatorSpec,
    NodePatternSpec,
    PatternRuleSpec,
    RuleSetSpec,
    SubstitutionSpec,
    TypeRaisingSpec,
)
from lofi.spec.grammar import (
    EisnerNormalFormSpec,
    FeatureErasureMatchingSpec,
    HockenmaierBiskNormalFormSpec,
    IdentityMatchingSpec,
    NoNormalFormSpec,
    OperatorSpec,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.spec import PatternSpec, RuleSpec
    from lofi.spec.grammar import AdoptedRules, MatchingSpec, NormalFormSpec
    from lofi.terms import Functors, Term

__all__ = [
    "CHART_CASES",
    "FEATURED_SIGNATURE",
    "FEATURED_STORE",
    "FUNCTORS",
    "ChartCase",
    "category",
    "compiled",
    "exact_pattern",
]


FEATURED_SIGNATURE: Final = ccg_signature(
    "a",
    "b",
    "c",
    "t",
    operators=(
        OperatorSpec(name="a[f1]", sort="Cat", erases_to="a"),
        OperatorSpec(name="a[f2]", sort="Cat", erases_to="a"),
    ),
    name="chart-reference",
)
"""Feature-bearing category signature used only by the differential suite."""


def compiled(
    rules: Sequence[RuleSpec],
    /,
    *,
    matching: MatchingSpec | None = None,
    normal_form: NormalFormSpec | None = None,
    name: str = "generated-chart",
) -> AdoptedRules:
    """Compile a rule inventory over the featured signature.

    Parameters
    ----------
    rules : Sequence[RuleSpec]
        The rule declarations.
    matching : MatchingSpec | None
        The premise congruence, identity unless given.
    normal_form : NormalFormSpec | None
        The normal form, none unless given.
    name : str
        The rule-set name.
    """
    return declared_rules(
        FEATURED_SIGNATURE,
        RuleSetSpec(name=name, rules=tuple(rules)),
        matching=matching,
        normal_form=normal_form,
    )


_ANCHOR: Final = compiled((ApplicationSpec(view="/"),), name="anchor")
FEATURED_STORE: Final = _ANCHOR.context.store
"""Interning store for the generated categories."""

FUNCTORS: Final[Functors[Term]] = _ANCHOR.context.functors["/"][0]
"""The declared slash functor view shared by generated rules."""


def category(name: str, *arguments: Term) -> Term:
    """Build one category: an atom by name, or a functor by its slash."""
    if arguments:
        result, argument = arguments
        return FUNCTORS.make(name, result, argument)

    return _ANCHOR.context.syntax.parse(FEATURED_STORE, name)


ATOMS: Final = tuple(category(name) for name in ("a", "a[f1]", "a[f2]", "b", "c", "t"))
"""Atomic categories from which generated functors are built."""


def exact_pattern(term: Term) -> PatternSpec:
    """Return the ground pattern declaration naming one exact category."""
    if FUNCTORS.is_functor(term):
        return NodePatternSpec(
            operator=FixedOperatorSpec(name=FUNCTORS.slash(term)),
            arguments=tuple(exact_pattern(child) for child in term.args),
        )

    return ConstantPatternSpec(name=_ANCHOR.context.syntax.render(FEATURED_STORE, term))


class ChartCase(NamedTuple):
    """One generated grammar and its finite lexical rows."""

    sentence: tuple[str, ...]
    rows: tuple[tuple[int, Term], ...]
    rules: RuleSet[Term]
    unary_depth: int


def _functor(parts: tuple[str, Term, Term]) -> Term:
    """Build the functor category of a slash, a result and an argument."""
    slash, result, argument = parts

    return category(slash, result, argument)


def _categories(depth: int) -> st.SearchStrategy[Term]:
    """Draw feature-bearing categories up to one small slash depth."""
    atoms = st.sampled_from(ATOMS)

    if depth == 0:
        return atoms

    smaller = _categories(depth - 1)

    return st.one_of(
        atoms,
        st.tuples(st.sampled_from(("/", "\\")), smaller, smaller).map(_functor),
    )


@st.composite
def _case(draw: st.DrawFn) -> ChartCase:
    """Draw one bounded grammar from the shipped CCG rule families."""
    matching: MatchingSpec = (
        FeatureErasureMatchingSpec() if draw(st.booleans()) else IdentityMatchingSpec()
    )
    direction = draw(st.sampled_from(COMPOSITION_RESTRICTIONS))
    normal_form: NormalFormSpec = draw(
        st.sampled_from(
            (
                NoNormalFormSpec(),
                EisnerNormalFormSpec(),
                HockenmaierBiskNormalFormSpec(),
            )
        )
    )
    views = st.sampled_from(("/", "\\"))
    rules: list[RuleSpec] = [ApplicationSpec(view=view) for view in ("/", "\\")]

    if draw(st.booleans()):
        rules.append(
            CompositionSpec(
                view=draw(views),
                order=1,
                direction=direction,
                crossed=draw(st.booleans()),
            )
        )

    if draw(st.booleans()):
        rules.append(
            SubstitutionSpec(view=draw(views), order=1, crossed=draw(st.booleans()))
        )

    unary_depth = draw(st.integers(0, 2))

    if unary_depth and draw(st.booleans()):
        rules.append(TypeRaisingSpec(view=draw(views), targets=("t",), arguments=None))

    if unary_depth:
        coercions = (("a[f1]", "b"), ("b", "c"))
        count = draw(st.integers(0, min(2, unary_depth + 1)))
        rules.extend(
            PatternRuleSpec(
                name=f"coercion-{position}",
                premises=(ConstantPatternSpec(name=premise),),
                conclusion=ConstantPatternSpec(name=output),
                head_policy="pass-through",
            )
            for position, (premise, output) in enumerate(coercions[:count])
        )

    length = draw(st.integers(2, 4))
    rows: list[tuple[int, Term]] = []

    for position in range(length):
        readings = draw(
            st.lists(
                _categories(2),
                min_size=1,
                max_size=2,
                unique=True,
            )
        )
        rows.extend((position, reading) for reading in readings)

    return ChartCase(
        tuple(f"w{position}" for position in range(length)),
        tuple(rows),
        compiled(rules, matching=matching, normal_form=normal_form).rule_set,
        unary_depth,
    )


CHART_CASES: Final = _case()
"""Generated bounded grammars covering every composition restriction."""

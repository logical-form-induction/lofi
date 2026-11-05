"""Constraint sources over derived, lexical, semantic and declared evidence."""

from collections import Counter
from math import exp
from random import Random
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple, override

from lofi._checks import ensure_finite, ensure_fraction, ensure_name
from lofi._frozen import FrozenValue
from lofi.data._constraint import (
    SpanConstraint,
    SpanConstraints,
    TypeScope,
    occurrences,
)
from lofi.data._report import LoadReport
from lofi.data._selection import CorpusProfile, ScopeKind, SelectionReport
from lofi.data._supervision import Example, GoldDerivation
from lofi.data._values import TokenSpan

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Mapping, Sequence

    from lofi._types import ItemId
    from lofi.data._selection import Form, SelectionPolicy
    from lofi.data._supervision import Supervision
    from lofi.data._values import Item
    from lofi.lexicon import LexicalSource

__all__ = [
    "NO_GOLD_DERIVATION",
    "ConstraintPattern",
    "ConstraintSelection",
    "DeclaredSource",
    "GoldDerivationSource",
    "GoldLexiconSource",
    "LogicalFormSource",
    "PinnedConstraintSource",
    "ScoreQuantile",
]

NO_GOLD_DERIVATION: Final = "no_gold_derivation"
"""The skip reason for an item whose supervision is not a gold derivation."""


class ConstraintSelection[C: Hashable, L: Hashable](NamedTuple):
    """A constrained corpus beside the budget the selection achieved on it.

    Parameters
    ----------
    examples : tuple[Example[SpanConstraints[L]], ...]
        The constrained examples, in corpus order.
    report : LoadReport
        What the pass read, emitted and set aside.
    budget : SelectionReport[C]
        How many types the selection pinned, how many tokens those cover, what
        share of the corpus's token mass that is, and the same by category.
    """

    examples: tuple[Example[SpanConstraints[L]], ...]
    report: LoadReport
    budget: SelectionReport[C]


class _Occurrence[L: Hashable](NamedTuple):
    """One constraint drawn at one span of one item."""

    index: int
    constraint: SpanConstraint[L]


class ConstraintPattern[L: Hashable](FrozenValue):
    """A declared surface pattern and the labels its spans may bear.

    A pattern member is either one exact surface word or ``None`` for a
    one-token wildcard. A pattern of ``None`` rather than a tuple denotes the
    whole sentence. Matches are returned in start order and may overlap.

    Parameters
    ----------
    pattern : tuple[str | None, ...] | None
        Surface matchers, or ``None`` for the sentence top.
    permitted : frozenset[L]
        Labels admitted at every matching span.

    Raises
    ------
    ValueError
        If a surface pattern or permitted set is empty, or a word is blank.
    """

    __slots__ = ("_pattern", "_permitted")

    _noun = "constraint pattern"

    def __init__(
        self,
        pattern: tuple[str | None, ...] | None,
        permitted: frozenset[L],
    ) -> None:
        if pattern == ():
            msg = "a constraint pattern with no matcher covers no span"
            raise ValueError(msg)

        checked = (
            None
            if pattern is None
            else tuple(
                None
                if member is None
                else ensure_name("a constraint pattern word", member)
                for member in pattern
            )
        )

        if not permitted:
            msg = "a constraint pattern permits no label"
            raise ValueError(msg)

        object.__setattr__(self, "_pattern", checked)
        object.__setattr__(self, "_permitted", permitted)

    @property
    def pattern(self) -> tuple[str | None, ...] | None:
        """Return the surface matchers, or ``None`` for the sentence top."""
        return self._pattern

    @property
    def permitted(self) -> frozenset[L]:
        """Return the labels admitted at every matching span."""
        return self._permitted

    def spans(self, item: Item, /) -> tuple[TokenSpan, ...]:
        """Return the spans this pattern covers in one item."""
        surfaces = tuple(token.surface for token in item.tokens)
        pattern = self._pattern

        if pattern is None:
            return () if not surfaces else (TokenSpan(0, len(surfaces)),)

        width = len(pattern)

        return tuple(
            TokenSpan(start, start + width)
            for start in range(len(surfaces) - width + 1)
            if all(
                matcher is None or matcher == surfaces[start + offset]
                for offset, matcher in enumerate(pattern)
            )
        )

    def _identity(self) -> tuple[tuple[str | None, ...] | None, frozenset[L]]:
        """Return the fields this pattern is equal on."""
        return self._pattern, self._permitted

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another pattern covers and permits the same values."""
        if type(other) is not ConstraintPattern:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the surface matchers and labels."""
        return hash((ConstraintPattern, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this pattern."""
        return f"ConstraintPattern({self._pattern!r}, {self._permitted!r})"


class ScoreQuantile(FrozenValue):
    """Select corpus items in the highest declared score quantile.

    The quantile denotes the share below the selected tail. The selected tail
    contains the rounded complement of the scored items, never fewer than one,
    and includes every item tied at its boundary.

    Parameters
    ----------
    quantile : float
        Share of scored items below the selected tail.
    scores : Mapping[ItemId, float]
        Finite normalized score by item.

    Raises
    ------
    ValueError
        If the quantile is not a fraction or no score is declared.
    """

    __slots__ = ("_quantile", "_scores")

    _noun = "score quantile"

    def __init__(self, quantile: float, scores: Mapping[ItemId, float]) -> None:
        if not scores:
            msg = "an score quantile has no scored item"
            raise ValueError(msg)

        object.__setattr__(
            self,
            "_quantile",
            ensure_fraction("an score selection quantile", quantile),
        )
        object.__setattr__(
            self,
            "_scores",
            MappingProxyType(
                {
                    item_id: ensure_finite(f"the score of item {item_id!r}", score)
                    for item_id, score in scores.items()
                }
            ),
        )

    @property
    def quantile(self) -> float:
        """Return the share of scored items below the selected tail."""
        return self._quantile

    @property
    def scores(self) -> Mapping[ItemId, float]:
        """Return normalized score by item."""
        return self._scores

    def selected(self, items: Sequence[Item], /) -> frozenset[ItemId]:
        """Return the corpus items in the declared upper tail.

        Raises
        ------
        ValueError
            If a score names an item outside the corpus.
        """
        held = {item.item_id for item in items}
        unknown = sorted(set(self._scores) - held)

        if unknown:
            msg = f"scores are declared for items outside the corpus: {unknown}"
            raise ValueError(msg)

        ordered = sorted(self._scores.values(), reverse=True)
        count = max(1, round((1.0 - self._quantile) * len(ordered)))
        threshold = ordered[count - 1]

        return frozenset(
            item_id for item_id, score in self._scores.items() if score >= threshold
        )

    def _identity(self) -> tuple[float, frozenset[tuple[ItemId, float]]]:
        """Return the fields this selection is equal on."""
        return self._quantile, frozenset(self._scores.items())

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another selection names the same upper tail."""
        if type(other) is not ScoreQuantile:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the quantile and item scores."""
        return hash((ScoreQuantile, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this selection."""
        return f"ScoreQuantile({self._quantile!r}, {dict(self._scores)!r})"


def _form(item: Item, span: TokenSpan) -> Form:
    """Return the surfaces of the tokens a span covers.

    Parameters
    ----------
    item : Item
        The item the span is read in.
    span : TokenSpan
        The positions to read.
    """
    return tuple(token.surface for token in item.tokens[span.start : span.end])


def _frequencies(items: Sequence[Item], forms: Iterable[Form]) -> Mapping[Form, int]:
    """Return how often each form occurs across the items.

    Parameters
    ----------
    items : Sequence[Item]
        The corpus.
    forms : Iterable[Form]
        The forms to count.

    Returns
    -------
    Mapping[Form, int]
        Occurrences of each form, overlapping matches included.
    """
    wanted = frozenset(forms)
    widths = sorted({len(form) for form in wanted})
    counts: Counter[Form] = Counter()

    for item in items:
        surfaces = tuple(token.surface for token in item.tokens)

        for width in widths:
            for start in range(len(surfaces) - width + 1):
                candidate = surfaces[start : start + width]

                if candidate in wanted:
                    counts[candidate] += 1

    return MappingProxyType({form: counts[form] for form in wanted})


def _covered[C: Hashable, L: Hashable](
    count: int, policy: SelectionPolicy[C, L]
) -> frozenset[int]:
    """Return the positions the policy's coverage keeps out of a count.

    Parameters
    ----------
    count : int
        How many candidates there are.
    policy : SelectionPolicy[C, L]
        The coverage fraction and the seed to draw under.

    Returns
    -------
    frozenset[int]
        The coverage fraction of the count, a half rounded to even, drawn
        without replacement under the seed.
    """
    kept = round(policy.coverage * count)

    return frozenset(Random(policy.seed).sample(range(count), kept))


def _token_level[C: Hashable, L: Hashable](
    items: Sequence[Item],
    occurrences: Sequence[_Occurrence[L]],
    policy: SelectionPolicy[C, L],
    categories: Mapping[Form, frozenset[C]],
) -> tuple[dict[int, list[SpanConstraint[L]]], SelectionReport[C]]:
    """Return the occurrences the policy admits, by item index, and its budget.

    Parameters
    ----------
    items : Sequence[Item]
        The corpus.
    occurrences : Sequence[_Occurrence[L]]
        Every candidate constraint, in corpus order.
    policy : SelectionPolicy[C, L]
        Where in the corpus the span's form must sit and the coverage to keep.
    categories : Mapping[Form, frozenset[C]]
        The categories each form was observed under.

    Returns
    -------
    tuple[dict[int, list[SpanConstraint[L]]], SelectionReport[C]]
        The selected constraints by item index, and the budget they achieved,
        whose covered tokens are the distinct spans they bind.
    """
    forms = tuple(
        _form(items[occurrence.index], occurrence.constraint.span)
        for occurrence in occurrences
    )
    profile = _profile(_frequencies(items, forms), categories)
    pinned = policy.selected(profile)
    admitted = tuple(
        (occurrence, form)
        for occurrence, form in zip(occurrences, forms, strict=True)
        if form in pinned
    )
    kept = _covered(len(admitted), policy)
    selected: dict[int, list[SpanConstraint[L]]] = {}
    covered: Counter[Form] = Counter()
    bound: set[tuple[int, TokenSpan]] = set()

    for position, (occurrence, form) in enumerate(admitted):
        if position not in kept:
            continue

        selected.setdefault(occurrence.index, []).append(occurrence.constraint)
        at = (occurrence.index, occurrence.constraint.span)

        if at not in bound:
            bound.add(at)
            covered[form] += 1

    return selected, policy.report(profile, covered)


def _profile[C: Hashable](
    counts: Mapping[Form, int], categories: Mapping[Form, frozenset[C]]
) -> CorpusProfile[C]:
    """Return the profile of the counted forms, uncategorized where unobserved.

    Parameters
    ----------
    counts : Mapping[Form, int]
        How often each candidate form occurs.
    categories : Mapping[Form, frozenset[C]]
        The categories each form was observed under.
    """
    return CorpusProfile(
        counts, {form: categories.get(form, frozenset()) for form in counts}
    )


def _type_level[C: Hashable, L: Hashable](
    items: Sequence[Item],
    permitted: Mapping[Form, frozenset[L]],
    policy: SelectionPolicy[C, L],
    categories: Mapping[Form, frozenset[C]],
) -> tuple[dict[int, list[SpanConstraint[L]]], SelectionReport[C]]:
    """Return the forms the policy admits, anchored in every item holding each.

    Parameters
    ----------
    items : Sequence[Item]
        The corpus.
    permitted : Mapping[Form, frozenset[L]]
        The labels each form may bear.
    policy : SelectionPolicy[C, L]
        Where in the corpus a form must sit and the coverage to keep.
    categories : Mapping[Form, frozenset[C]]
        The categories each form was observed under.

    Returns
    -------
    tuple[dict[int, list[SpanConstraint[L]]], SelectionReport[C]]
        One type-level constraint per admitted form occurrence, keyed by item index,
        and the budget those forms achieved.
    """
    profile = _profile(_frequencies(items, permitted), categories)
    admitted = tuple(sorted(policy.selected(profile)))
    kept = _covered(len(admitted), policy)
    selected: dict[int, list[SpanConstraint[L]]] = {}
    covered: dict[Form, int] = {}

    for position, form in enumerate(admitted):
        if position not in kept or profile.counts[form] == 0:
            continue

        covered[form] = profile.counts[form]
        scope = TypeScope(form)

        for index, item in enumerate(items):
            starts = occurrences(form, item.tokens)

            for start in starts:
                anchor = TokenSpan(start, start + len(form))
                selected.setdefault(index, []).append(
                    SpanConstraint(anchor, permitted[form], scope)
                )

    return selected, policy.report(profile, covered)


def _assembled[C: Hashable, L: Hashable](
    examples: Sequence[Example[Supervision[C]]],
    readable: Sequence[int],
    selected: Mapping[int, Sequence[SpanConstraint[L]]],
    skipped: Mapping[str, int],
) -> tuple[tuple[Example[SpanConstraints[L]], ...], LoadReport]:
    """Return the constrained examples and the report accounting for every item.

    Parameters
    ----------
    examples : Sequence[Example[Supervision[C]]]
        The corpus, in item order.
    readable : Sequence[int]
        The indices of the examples a source could read, ascending.
    selected : Mapping[int, Sequence[SpanConstraint[L]]]
        The constraints emitted for each readable index.
    skipped : Mapping[str, int]
        How many examples were set aside, by reason.
    """
    emitted = tuple(
        Example(
            examples[index].item,
            SpanConstraints(
                tuple(selected.get(index, ())), len(examples[index].item.tokens)
            ),
        )
        for index in readable
    )

    return emitted, LoadReport(len(examples), len(emitted), skipped)


class GoldDerivationSource[C: Hashable, L: Hashable]:
    """Draws constraints from the constituents of gold derivations.

    Token-level policies emit each constituent occurrence with its label.
    Type-level policies pool labels by form and bind the resolved set at every
    occurrence.

    Parameters
    ----------
    policy : SelectionPolicy[C, L]
        Which occurrences or forms are emitted, and how ambiguity resolves.
    """

    __slots__ = ("_policy",)

    def __init__(self, policy: SelectionPolicy[C, L]) -> None:
        self._policy = policy

    @property
    def scope(self) -> ScopeKind:
        """Return the scope the policy's rule binds at."""
        return self._policy.scope

    @property
    def policy(self) -> SelectionPolicy[C, L]:
        """Return which constraints are emitted from the derivations."""
        return self._policy

    def select(
        self, examples: Iterable[Example[Supervision[C]]], /
    ) -> ConstraintSelection[C, L]:
        """Return the constrained corpus, the report and the achieved budget.

        Parameters
        ----------
        examples : Iterable[Example[Supervision[C]]]
            The corpus, in item order.

        Returns
        -------
        ConstraintSelection[C, L]
            One constrained example per gold-derivation item, a report skipping
            every item whose supervision is not a gold derivation, and what the
            selection pinned against the constituents it could have drawn.
        """
        held = tuple(examples)
        items = tuple(example.item for example in held)
        readable: list[int] = []
        skipped: Counter[str] = Counter()
        nodes: list[tuple[int, TokenSpan, C]] = []

        for index, example in enumerate(held):
            supervision = example.supervision

            if not isinstance(supervision, GoldDerivation):
                skipped[NO_GOLD_DERIVATION] += 1

                continue

            readable.append(index)
            derivation = supervision.derivation

            for node_id in sorted(derivation.nodes):
                node = derivation.node(node_id)
                nodes.append((index, node.span, node.label))

        rule = self._policy.ambiguity
        masses: dict[Form, dict[C, float]] = {}

        for index, span, label in nodes:
            observed = masses.setdefault(_form(items[index], span), {})
            observed[label] = observed.get(label, 0.0) + 1.0

        categories = {form: frozenset(observed) for form, observed in masses.items()}

        if rule.scope is ScopeKind.OCCURRENCE:
            occurrences = tuple(
                _Occurrence(index, SpanConstraint(span, rule.permitted({label: 1.0})))
                for index, span, label in nodes
            )
            selected, budget = _token_level(
                items, occurrences, self._policy, categories
            )

        else:
            selected, budget = _type_level(
                items,
                {form: rule.permitted(observed) for form, observed in masses.items()},
                self._policy,
                categories,
            )

        emitted, report = _assembled(held, readable, selected, skipped)

        return ConstraintSelection(emitted, report, budget)

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this source."""
        return f"GoldDerivationSource({self._policy!r})"


class GoldLexiconSource[C: Hashable, L: Hashable]:
    """Draw constraints from a gold lexicon and its observed derivations.

    Every position a lexicon admits entries at is one occurrence of that token's
    form, with each entry's category weighted by the exponential of the weight
    the lexicon reports for it. The categories observed for one form are pooled
    across the corpus and a form-level rule binds the resolved set at every
    occurrence. A token-level rule instead reads the lexical leaves of each gold
    derivation and binds that leaf's category only at its observed occurrence.

    Parameters
    ----------
    lexicon : LexicalSource[C, float]
        The entries a form may bear, with their log weights.
    policy : SelectionPolicy[C, L]
        Which forms are emitted, and how a form with several entries resolves.

    """

    __slots__ = ("_lexicon", "_policy")

    def __init__(
        self, lexicon: LexicalSource[C, float], policy: SelectionPolicy[C, L]
    ) -> None:
        self._lexicon = lexicon
        self._policy = policy

    @property
    def scope(self) -> ScopeKind:
        """Return the scope the policy's rule binds at."""
        return self._policy.scope

    @property
    def policy(self) -> SelectionPolicy[C, L]:
        """Return which forms are emitted and how their entries resolve."""
        return self._policy

    @property
    def lexicon(self) -> LexicalSource[C, float]:
        """Return the entries the constraints are drawn from."""
        return self._lexicon

    def select(
        self, examples: Iterable[Example[Supervision[C]]], /
    ) -> ConstraintSelection[C, L]:
        """Return the constrained corpus, the report and the achieved budget.

        Parameters
        ----------
        examples : Iterable[Example[Supervision[C]]]
            The corpus, in item order.

        Returns
        -------
        ConstraintSelection[C, L]
            One constrained example per item, every item read, and what the
            selection pinned against the forms the lexicon admits entries at.
        """
        held = tuple(examples)
        items = tuple(example.item for example in held)
        masses: dict[Form, dict[C, float]] = {}

        for item in items:
            surfaces = tuple(token.surface for token in item.tokens)

            for position, entry, weight in self._lexicon.axioms(surfaces):
                observed = masses.setdefault((surfaces[position],), {})
                observed[entry.category] = observed.get(entry.category, 0.0) + exp(
                    weight
                )

        rule = self._policy.ambiguity
        categories = {form: frozenset(observed) for form, observed in masses.items()}

        if rule.scope is ScopeKind.OCCURRENCE:
            readable: list[int] = []
            skipped: Counter[str] = Counter()
            occurrences: list[_Occurrence[L]] = []

            for index, example in enumerate(held):
                supervision = example.supervision

                if not isinstance(supervision, GoldDerivation):
                    skipped[NO_GOLD_DERIVATION] += 1
                    continue

                readable.append(index)
                derivation = supervision.derivation

                for node_id in sorted(derivation.nodes):
                    node = derivation.node(node_id)

                    if node.is_lexical:
                        occurrences.append(
                            _Occurrence(
                                index,
                                SpanConstraint(
                                    node.span, rule.permitted({node.label: 1.0})
                                ),
                            )
                        )

            selected, budget = _token_level(
                items, occurrences, self._policy, categories
            )
            emitted, report = _assembled(held, readable, selected, skipped)

        else:
            selected, budget = _type_level(
                items,
                {form: rule.permitted(observed) for form, observed in masses.items()},
                self._policy,
                categories,
            )
            emitted, report = _assembled(held, range(len(held)), selected, {})

        return ConstraintSelection(emitted, report, budget)

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this source."""
        return f"GoldLexiconSource({self._lexicon!r}, {self._policy!r})"


class LogicalFormSource[C: Hashable, L: Hashable]:
    """Pins the logical form of a lexical form at every occurrence of it.

    A logical-form clamp permits one meaning on the span it names, so the
    assignment gives each form the one meaning it bears rather than a set for
    the policy's rule to resolve. The logical-constant classes that meaning
    belongs to are what a category class selects on and what the achieved
    budget breaks down by, so a policy naming the quantifiers pins the forms
    whose meaning is built on one and a policy naming the two-place predicates
    pins those.

    Parameters
    ----------
    assignment : Mapping[Form, L]
        The logical form each lexical form bears.
    classes : Mapping[Form, frozenset[C]]
        The logical-constant classes each of those meanings belongs to.
    policy : SelectionPolicy[C, L]
        Which forms are pinned, and the coverage to keep.

    Raises
    ------
    ValueError
        If the policy's rule binds at the token level, since a logical form is
        pinned on a lexical form rather than on one occurrence of it, if the
        assignment and the classes name different forms, if a form has no
        words, or if a form's meaning belongs to no class.
    """

    __slots__ = ("_assignment", "_classes", "_policy")

    def __init__(
        self,
        assignment: Mapping[Form, L],
        classes: Mapping[Form, frozenset[C]],
        policy: SelectionPolicy[C, L],
    ) -> None:
        if policy.scope is not ScopeKind.FORM:
            msg = (
                "a logical form is pinned on a lexical form, so its clamp binds "
                f"at the type level, not under {policy.ambiguity!r}"
            )
            raise ValueError(msg)

        if set(assignment) != set(classes):
            named = sorted(set(assignment) ^ set(classes))
            msg = (
                "a logical-form source assigns and classifies the same forms, "
                f"and these are named on one side only: {named}"
            )
            raise ValueError(msg)

        for form, held in classes.items():
            if not form:
                msg = "a logical form is assigned to a form of no words"
                raise ValueError(msg)

            if not held:
                msg = f"the form {list(form)} bears a meaning of no class"
                raise ValueError(msg)

        self._assignment = MappingProxyType(dict(assignment))
        self._classes = MappingProxyType(dict(classes))
        self._policy = policy

    @property
    def scope(self) -> ScopeKind:
        """Return the type-level scope."""
        return ScopeKind.FORM

    @property
    def policy(self) -> SelectionPolicy[C, L]:
        """Return which forms have their logical form pinned."""
        return self._policy

    @property
    def assignment(self) -> Mapping[Form, L]:
        """Return the logical form each lexical form bears."""
        return self._assignment

    @property
    def classes(self) -> Mapping[Form, frozenset[C]]:
        """Return the logical-constant classes each meaning belongs to."""
        return self._classes

    def select(
        self, examples: Iterable[Example[Supervision[C]]], /
    ) -> ConstraintSelection[C, L]:
        """Return the constrained corpus, the report and the achieved budget.

        Parameters
        ----------
        examples : Iterable[Example[Supervision[C]]]
            The corpus, in item order.

        Returns
        -------
        ConstraintSelection[C, L]
            One constrained example per item, every item read, and what the
            selection pinned against the forms the assignment gives a meaning,
            broken down by logical-constant class.
        """
        held = tuple(examples)
        items = tuple(example.item for example in held)
        selected, budget = _type_level(
            items,
            {form: frozenset({meaning}) for form, meaning in self._assignment.items()},
            self._policy,
            self._classes,
        )
        emitted, report = _assembled(held, range(len(held)), selected, {})

        return ConstraintSelection(emitted, report, budget)

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this source."""
        return (
            f"LogicalFormSource({dict(self._assignment)!r}, "
            f"{dict(self._classes)!r}, {self._policy!r})"
        )


class DeclaredSource[C: Hashable, L: Hashable]:
    """Emits constraints an analyst wrote down.

    A token-level source maps corpus items to span constraints. A type-level
    source maps forms to labels and binds them at every occurrence. Each source
    accepts exactly one representation, and its policy must use that scope.

    Parameters
    ----------
    policy : SelectionPolicy[C, L]
        The band a declared form must fall in, the coverage to keep, and the
        scope the list is written at.
    by_item : Mapping[ItemId, tuple[SpanConstraint[L], ...]]
        Token-level constraints keyed by the item they constrain.
    by_form : Mapping[Form, frozenset[L]]
        The labels each form may bear.

    Raises
    ------
    ValueError
        If both lists or neither is given, if the policy's scope is not the list's,
        if a token-level constraint names a form, or if a permitted set is empty.
    """

    __slots__ = ("_by_form", "_by_item", "_policy")

    def __init__(
        self,
        policy: SelectionPolicy[C, L],
        *,
        by_item: Mapping[ItemId, tuple[SpanConstraint[L], ...]] = MappingProxyType({}),
        by_form: Mapping[Form, frozenset[L]] = MappingProxyType({}),
    ) -> None:
        if bool(by_item) == bool(by_form):
            msg = (
                "a declared source holds one list: constraints by item or "
                "labels by form"
            )
            raise ValueError(msg)

        scope = ScopeKind.OCCURRENCE if by_item else ScopeKind.FORM

        if policy.scope is not scope:
            msg = (
                f"a list declared at the {scope.value} level is selected under "
                f"{policy.ambiguity!r}, which binds at the {policy.scope.value} level"
            )
            raise ValueError(msg)

        for item_id, constraints in by_item.items():
            for constraint in constraints:
                if constraint.scope.form is not None:
                    msg = (
                        f"a constraint declared for item {item_id!r} binds the form "
                        f"{list(constraint.scope.form)}; a form is declared by form"
                    )
                    raise ValueError(msg)

        for form, labels in by_form.items():
            if not form:
                msg = "a declared form has no words"
                raise ValueError(msg)

            if not labels:
                msg = f"the form {list(form)} is declared to bear no label"
                raise ValueError(msg)

        self._policy = policy
        self._by_item = MappingProxyType(
            {item_id: tuple(constraints) for item_id, constraints in by_item.items()}
        )
        self._by_form = MappingProxyType(dict(by_form))

    @property
    def scope(self) -> ScopeKind:
        """Return the scope the list is written at."""
        return self._policy.scope

    @property
    def policy(self) -> SelectionPolicy[C, L]:
        """Return the band, the coverage and the scope the list is selected under."""
        return self._policy

    @property
    def by_item(self) -> Mapping[ItemId, tuple[SpanConstraint[L], ...]]:
        """Return the token-level constraints declared for each item."""
        return self._by_item

    @property
    def by_form(self) -> Mapping[Form, frozenset[L]]:
        """Return the labels declared for each form."""
        return self._by_form

    def select(
        self, examples: Iterable[Example[Supervision[C]]], /
    ) -> ConstraintSelection[C, L]:
        """Return the constrained corpus, the report and the achieved budget.

        A declared list names the labels a form may bear rather than the
        categories the corpus was observed under, so the budget's per-category
        breakdown is empty and a category class selects nothing from it.

        Parameters
        ----------
        examples : Iterable[Example[Supervision[C]]]
            The corpus, in item order.

        Returns
        -------
        ConstraintSelection[C, L]
            One constrained example per item, every item read, and what the
            selection pinned against the list it was drawn from.

        Raises
        ------
        ValueError
            If a constraint is declared for an item the corpus does not hold.
        """
        held = tuple(examples)
        items = tuple(example.item for example in held)
        uncategorized: Mapping[Form, frozenset[C]] = MappingProxyType({})

        if self._by_item:
            index_of = {item.item_id: index for index, item in enumerate(items)}
            unknown = sorted(set(self._by_item) - set(index_of))

            if unknown:
                msg = (
                    "constraints are declared for items the corpus does not "
                    f"hold: {unknown}"
                )
                raise ValueError(msg)

            occurrences = tuple(
                _Occurrence(index_of[item_id], constraint)
                for item_id, constraints in self._by_item.items()
                for constraint in constraints
            )
            selected, budget = _token_level(
                items, occurrences, self._policy, uncategorized
            )

        else:
            selected, budget = _type_level(
                items, self._by_form, self._policy, uncategorized
            )

        emitted, report = _assembled(held, range(len(held)), selected, {})

        return ConstraintSelection(emitted, report, budget)

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this source."""
        return (
            f"DeclaredSource({self._policy!r}, by_item={dict(self._by_item)!r}, "
            f"by_form={dict(self._by_form)!r})"
        )


class PinnedConstraintSource[C: Hashable, L: Hashable]:
    """Emit occurrence-level constraints from declared surface patterns.

    One treatment covers lexical items, multiword or wildcard span patterns,
    and the sentence top. An optional score quantile restricts the
    occurrences that receive constraints without removing any item from the
    emitted corpus.

    Parameters
    ----------
    policy : SelectionPolicy[C, L]
        Which admitted occurrences are emitted.
    patterns : Sequence[ConstraintPattern[L]]
        Surface patterns and their permitted labels, in declaration order.
    score_quantile : ScoreQuantile | None
        Optional upper-tail selection over corpus items.

    Raises
    ------
    ValueError
        If no pattern is declared or the policy does not bind occurrences.
    """

    __slots__ = ("_patterns", "_policy", "_score_quantile")

    def __init__(
        self,
        policy: SelectionPolicy[C, L],
        patterns: Sequence[ConstraintPattern[L]],
        *,
        score_quantile: ScoreQuantile | None,
    ) -> None:
        held = tuple(patterns)

        if not held:
            msg = "a pinned constraint source declares at least one pattern"
            raise ValueError(msg)

        if policy.scope is not ScopeKind.OCCURRENCE:
            msg = (
                "declared surface constraints bind individual occurrences, "
                f"not under {policy.ambiguity!r}"
            )
            raise ValueError(msg)

        self._policy = policy
        self._patterns = held
        self._score_quantile = score_quantile

    @property
    def scope(self) -> ScopeKind:
        """Return the occurrence-level scope."""
        return ScopeKind.OCCURRENCE

    @property
    def policy(self) -> SelectionPolicy[C, L]:
        """Return which admitted occurrences are emitted."""
        return self._policy

    @property
    def patterns(self) -> tuple[ConstraintPattern[L], ...]:
        """Return the surface patterns in declaration order."""
        return self._patterns

    @property
    def score_quantile(self) -> ScoreQuantile | None:
        """Return the upper-tail restriction, if one was declared."""
        return self._score_quantile

    def select(
        self, examples: Iterable[Example[Supervision[C]]], /
    ) -> ConstraintSelection[C, L]:
        """Return the constrained corpus and its achieved selection budget."""
        held = tuple(examples)
        items = tuple(example.item for example in held)
        admitted = (
            frozenset(item.item_id for item in items)
            if self._score_quantile is None
            else self._score_quantile.selected(items)
        )
        candidates = tuple(
            _Occurrence(index, SpanConstraint(span, pattern.permitted))
            for index, item in enumerate(items)
            for pattern in self._patterns
            for span in pattern.spans(item)
        )
        admitted_candidates = tuple(
            occurrence
            for occurrence in candidates
            if items[occurrence.index].item_id in admitted
        )
        categories: Mapping[Form, frozenset[C]] = MappingProxyType({})
        selected, _ = _token_level(items, admitted_candidates, self._policy, categories)
        candidate_forms = tuple(
            _form(items[occurrence.index], occurrence.constraint.span)
            for occurrence in candidates
        )
        profile = _profile(_frequencies(items, candidate_forms), categories)
        covered: Counter[Form] = Counter()
        bound: set[tuple[int, TokenSpan]] = set()

        for index, constraints in selected.items():
            for constraint in constraints:
                at = (index, constraint.span)

                if at in bound:
                    continue

                bound.add(at)
                covered[_form(items[index], constraint.span)] += 1

        budget = self._policy.report(profile, covered)
        emitted, report = _assembled(held, range(len(held)), selected, {})

        return ConstraintSelection(emitted, report, budget)

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this source."""
        tail = (
            ""
            if self._score_quantile is None
            else f", score_quantile={self._score_quantile!r}"
        )

        return f"PinnedConstraintSource({self._policy!r}, {self._patterns!r}{tail})"

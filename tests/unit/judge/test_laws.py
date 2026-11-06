from importlib.metadata import EntryPoint, EntryPoints
from itertools import product
from typing import TYPE_CHECKING, Final, NamedTuple, override

import pytest
from hypothesis import strategies as st

from lofi._checks import LawViolationError
from lofi.judge import NormalFormRole
from lofi.judge._ports import (
    HOLE,
    Direction,
    NormalForm,
    NormalFormDecisionTable,
    Rule,
    Slot,
)
from lofi.judge.laws import (
    NORMAL_FORM_LAWS,
    RULE_LAW_FAMILY,
    RULE_LAWS,
    NormalFormFixture,
    RuleFixture,
    RuleLawCase,
    RuleLawCases,
    RuleLawProvider,
    RuleTotalityCase,
    adjointness,
    admits_is_total,
    check_normal_form_laws,
    check_registered_rule_laws,
    check_rule_law_cases,
    check_rule_laws,
    conditional_completeness,
    discover_rule_law_cases,
    filtering_preserves_goal_terms,
    non_mutation,
    soundness,
    spec_round_trip,
    stability,
    tagging_is_pure,
    tags_are_declared,
)
from lofi.registry import Registry

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from hypothesis.strategies import SearchStrategy

    from lofi._types import JsonValue
    from lofi.judge._index import RuleProbe
    from lofi.judge._values import RuleRestriction

_ALL: Final[frozenset[Direction]] = frozenset({"forward", "check", "residual"})
_EXAMPLES: Final = 400


def _pass_rule_laws() -> None:
    """Do nothing."""


def _one_totality_call() -> int:
    """Report one test admission call."""
    return 1


def _no_totality_calls() -> int:
    """Report no test admission calls."""
    return 0


def _registered_rule_law_points() -> EntryPoints:
    """Return the built-in rule-law entry points in reverse order."""
    declarations = (
        ("composition", "lofi.spec._composition_laws:provide_rule_laws"),
        ("substitution", "lofi.spec._substitution_laws:provide_rule_laws"),
        ("raising", "lofi.spec._raising_laws:provide_rule_laws"),
        ("pattern", "lofi.spec._pattern_laws:provide_rule_laws"),
    )

    return EntryPoints(
        EntryPoint(name=name, value=value, group="lofi.rule_laws")
        for name, value in reversed(declarations)
    )


class Atom(NamedTuple):
    """Atomic category with an optional feature."""

    name: str
    feature: str = ""


type Cat = tuple[Atom, ...]
"""Result atom followed by its arguments."""


class Congruence:
    """Category congruence with optional feature erasure.

    Parameters
    ----------
    erases_features
        Whether equivalence ignores category features.
    """

    def __init__(self, *, erases_features: bool) -> None:
        """Fix which distinctions the congruence keeps."""
        self._erases_features = erases_features

    @property
    def name(self) -> str:
        """Name the congruence."""
        return "feature-erasure" if self._erases_features else "identity"

    @property
    def has_key(self) -> bool:
        """Report that classes have representatives."""
        return True

    def key(self, term: Cat, /) -> Cat:
        """Return the class representative."""
        if not self._erases_features:
            return term

        return tuple(Atom(atom.name) for atom in term)

    def equivalent(self, left: Cat, right: Cat, /) -> bool:
        """Compare representatives."""
        return self.key(left) == self.key(right)


class ForwardApplication:
    """A forward-application rule.

    Parameters
    ----------
    matching
        The congruence used to compare categories.
    """

    def __init__(self, matching: Congruence) -> None:
        """Bind the rule to one congruence."""
        self._matching = matching

    @property
    def name(self) -> str:
        """Name the rule."""
        return "forward-application"

    @property
    def arity(self) -> int:
        """Take a functor and an argument."""
        return 2

    @property
    def directions(self) -> frozenset[Direction]:
        """Compute in all three directions."""
        return _ALL

    @property
    def normal_form_role(self) -> NormalFormRole:
        return NormalFormRole()

    @property
    def spine_order(self) -> int:
        return 0

    @property
    def residual_slots(self) -> frozenset[int]:
        """Solve either premise."""
        return frozenset({0, 1})

    @property
    def forward_is_complete(self) -> bool:
        """Enumerate every conclusion the rule admits."""
        return True

    @property
    def matching(self) -> Congruence:
        """Read categories under the injected congruence."""
        return self._matching

    @property
    def restriction(self) -> RuleRestriction | None:
        """Declare no rule-instance restriction."""
        return None

    @property
    def probe(self) -> RuleProbe | None:
        """Decline premise-head indexing."""
        return None

    def apply(self, inputs: tuple[Cat, ...], /) -> Iterator[Cat]:
        """Consume the functor's first argument."""
        if len(inputs) != self.arity:
            return

        functor, argument = inputs[0], inputs[1]

        if len(functor) < 2 or len(argument) != 1:
            return

        if not self._matching.equivalent((functor[1],), argument):
            return

        yield (functor[0], *functor[2:])

    def admits(self, inputs: tuple[Cat, ...], output: Cat, /) -> bool:
        """Decide the relation independently of forward output."""
        if len(inputs) != self.arity:
            return False

        functor, argument = inputs[0], inputs[1]

        if len(functor) < 2 or len(argument) != 1:
            return False

        if not self._matching.equivalent((functor[1],), argument):
            return False

        return self._matching.equivalent((functor[0], *functor[2:]), output)

    def residuate(self, output: Cat, known: tuple[Slot[Cat], ...], /) -> Iterator[Cat]:
        """Rebuild either missing premise."""
        if len(known) != self.arity:
            return

        first, second = known[0], known[1]

        if first is HOLE and isinstance(second, tuple):
            if len(second) == 1:
                yield (output[0], second[0], *output[1:])

        elif (
            second is HOLE
            and isinstance(first, tuple)
            and len(first) >= self.arity
            and self._matching.equivalent((first[0], *first[2:]), output)
        ):
            yield (first[1],)

    def to_spec(self) -> Mapping[str, JsonValue]:
        """Return the declaration this rule was built from."""
        return {"rule": self.name, "matching": self._matching.name}


class Unsound(ForwardApplication):
    """Rule with unsound forward enumeration."""

    @override
    def apply(self, inputs: tuple[Cat, ...], /) -> Iterator[Cat]:
        """Add a conclusion the rule does not license."""
        yield from super().apply(inputs)
        yield (Atom("Z"),)


class OverAdmitting(ForwardApplication):
    """Complete rule that admits unenumerated conclusions."""

    @override
    def admits(self, inputs: tuple[Cat, ...], output: Cat, /) -> bool:
        """Accept anything of the right arity."""
        del output

        return len(inputs) == self.arity


class RaisingAdmits(ForwardApplication):
    """Rule whose check direction raises on invalid arity."""

    @override
    def admits(self, inputs: tuple[Cat, ...], output: Cat, /) -> bool:
        """Raise on a premise tuple of the wrong length."""
        if len(inputs) != self.arity:
            message = f"expected {self.arity} premises, got {len(inputs)}"

            raise ValueError(message)

        return super().admits(inputs, output)


class Unresiduable(ForwardApplication):
    """Rule that omits a valid functor residual."""

    @override
    def residuate(self, output: Cat, known: tuple[Slot[Cat], ...], /) -> Iterator[Cat]:
        """Answer only the argument slot."""
        if known and known[0] is HOLE:
            return iter(())

        return super().residuate(output, known)


class UndeclaredRightResidual(ForwardApplication):
    """Rule that computes a residual outside its declaration."""

    @property
    @override
    def residual_slots(self) -> frozenset[int]:
        """Declare only the left premise slot."""
        return frozenset({0})

    @override
    def residuate(self, output: Cat, known: tuple[Slot[Cat], ...], /) -> Iterator[Cat]:
        """Answer every right-slot query despite the declaration."""
        if len(known) == self.arity and known[1] is HOLE:
            yield output

        yield from super().residuate(output, known)


class FeatureSensitive(ForwardApplication):
    """Rule inconsistent with its feature-erasing congruence."""

    @override
    def apply(self, inputs: tuple[Cat, ...], /) -> Iterator[Cat]:
        """Apply only to an exact argument match."""
        if len(inputs) != self.arity:
            return

        functor, argument = inputs[0], inputs[1]

        if len(functor) >= 2 and len(argument) == 1 and functor[1] == argument[0]:
            yield (functor[0], *functor[2:])


class Cell:
    """A mutable test term.

    Parameters
    ----------
    value
        The stored integer.
    """

    __slots__ = ("value",)

    def __init__(self, value: int) -> None:
        """Hold a value a rule could change."""
        self.value = value

    @override
    def __eq__(self, other: object) -> bool:
        """Compare by value."""
        return isinstance(other, Cell) and other.value == self.value

    @override
    def __hash__(self) -> int:
        """Hash by value, so a change to it is visible."""
        return hash(self.value)

    @override
    def __repr__(self) -> str:
        """Render for a counterexample."""
        return f"Cell({self.value})"


class CellIdentity:
    """Cell-value congruence."""

    @property
    def name(self) -> str:
        """Name the congruence."""
        return "identity"

    @property
    def has_key(self) -> bool:
        """Report that classes have representatives."""
        return True

    def key(self, term: Cell, /) -> Cell:
        """Return the term, which is its own representative."""
        return term

    def equivalent(self, left: Cell, right: Cell, /) -> bool:
        """Compare cells."""
        return left == right


class Bump:
    """Rule that mutates its premise."""

    @property
    def name(self) -> str:
        """Name the rule."""
        return "bump"

    @property
    def arity(self) -> int:
        """Take one premise."""
        return 1

    @property
    def directions(self) -> frozenset[Direction]:
        """Compute forward and check."""
        return frozenset({"forward", "check"})

    @property
    def normal_form_role(self) -> NormalFormRole:
        return NormalFormRole()

    @property
    def spine_order(self) -> int:
        return 0

    @property
    def residual_slots(self) -> frozenset[int]:
        """Declare no computable residual."""
        return frozenset()

    @property
    def forward_is_complete(self) -> bool:
        """Enumerate every conclusion the rule admits."""
        return True

    @property
    def matching(self) -> CellIdentity:
        """Read premises under equality."""
        return CellIdentity()

    @property
    def restriction(self) -> RuleRestriction | None:
        """Declare no rule-instance restriction."""
        return None

    @property
    def probe(self) -> RuleProbe | None:
        """Decline premise-head indexing."""
        return None

    def apply(self, inputs: tuple[Cell, ...], /) -> Iterator[Cell]:
        """Increment the premise and return a copy."""
        if len(inputs) != 1:
            return

        inputs[0].value += 1

        yield Cell(inputs[0].value)

    def admits(self, inputs: tuple[Cell, ...], output: Cell, /) -> bool:
        """Decide whether the conclusion is the premise's successor."""
        return len(inputs) == 1 and output == Cell(inputs[0].value + 1)

    def residuate(
        self, output: Cell, known: tuple[Slot[Cell], ...], /
    ) -> Iterator[Cell]:
        """Yield no residuals."""
        del output, known

        return iter(())

    def to_spec(self) -> Mapping[str, JsonValue]:
        """Return the declaration this rule was built from."""
        return {"rule": self.name}


def rebuild(spec: Mapping[str, JsonValue]) -> ForwardApplication:
    """Build a forward-application rule from its declaration.

    Parameters
    ----------
    spec
        The rule declaration returned by ``to_spec``.
    """
    return ForwardApplication(
        Congruence(erases_features=spec["matching"] == "feature-erasure")
    )


def rebuild_forgetting_the_congruence(
    _spec: Mapping[str, JsonValue],
) -> ForwardApplication:
    return ForwardApplication(Congruence(erases_features=False))


def rebuild_a_cell_rule(_spec: Mapping[str, JsonValue]) -> Bump:
    return Bump()


_ATOMS = st.builds(Atom, st.sampled_from(("S", "NP")), st.sampled_from(("", "dcl")))
_PLAIN_ATOMS = st.builds(Atom, st.sampled_from(("S", "NP")), st.just(""))
_CATS = st.lists(_ATOMS, min_size=1, max_size=2).map(tuple)
_PLAIN_CATS = st.lists(_PLAIN_ATOMS, min_size=1, max_size=2).map(tuple)
_ATOMIC_CATS = st.lists(_PLAIN_ATOMS, min_size=1, max_size=1).map(tuple)


def exact_fixture(rule: Rule[Cat]) -> RuleFixture[Cat]:
    """Return a fixture over featureless categories.

    Parameters
    ----------
    rule
        The rule under test.
    """
    return RuleFixture(rule, _PLAIN_CATS, rebuild, _EXAMPLES)


def erasing_fixture(rule: Rule[Cat]) -> RuleFixture[Cat]:
    """Return a fixture over categories with features.

    Parameters
    ----------
    rule
        The rule under test.
    """
    return RuleFixture(rule, _CATS, rebuild, _EXAMPLES)


@pytest.mark.parametrize("law", RULE_LAWS, ids=lambda law: law.__name__)
def test_each_law_collects_and_runs_on_its_own(
    law: Callable[[RuleFixture[Cat]], None],
) -> None:
    law(exact_fixture(ForwardApplication(Congruence(erases_features=False))))


def test_the_entry_point_reports_the_law_that_failed() -> None:
    with pytest.raises(LawViolationError, match="soundness"):
        check_rule_laws(exact_fixture(Unsound(Congruence(erases_features=False))))


def test_an_honest_rule_holds_to_every_law() -> None:
    check_rule_laws(
        exact_fixture(ForwardApplication(Congruence(erases_features=False)))
    )


def test_an_honest_rule_holds_to_every_law_under_a_coarser_congruence() -> None:
    check_rule_laws(
        erasing_fixture(ForwardApplication(Congruence(erases_features=True)))
    )


def test_soundness_catches_a_conclusion_the_rule_will_not_admit() -> None:
    with pytest.raises(LawViolationError, match="soundness"):
        soundness(exact_fixture(Unsound(Congruence(erases_features=False))))


def test_soundness_refuses_to_pass_on_a_rule_it_never_fired() -> None:
    rule = ForwardApplication(Congruence(erases_features=False))
    never_fires = RuleFixture(rule, _ATOMIC_CATS, rebuild, _EXAMPLES)

    with pytest.raises(LawViolationError, match="never exercised"):
        soundness(never_fires)


def test_completeness_catches_a_rule_admitting_more_than_it_enumerates() -> None:
    with pytest.raises(LawViolationError, match="conditional completeness"):
        conditional_completeness(
            exact_fixture(OverAdmitting(Congruence(erases_features=False)))
        )


def test_completeness_is_skipped_by_a_rule_that_declares_forward_incomplete() -> None:
    class Partial(ForwardApplication):
        """Incomplete-forward rule."""

        @property
        @override
        def forward_is_complete(self) -> bool:
            """Report the forward direction as incomplete."""
            return False

    conditional_completeness(exact_fixture(Partial(Congruence(erases_features=False))))


def test_totality_catches_a_rule_that_raises_instead_of_answering() -> None:
    with pytest.raises(LawViolationError, match="totality"):
        admits_is_total(exact_fixture(RaisingAdmits(Congruence(erases_features=False))))


def test_adjointness_catches_a_slot_residuation_forgets() -> None:
    with pytest.raises(LawViolationError, match="adjointness"):
        adjointness(exact_fixture(Unresiduable(Congruence(erases_features=False))))


def test_adjointness_catches_an_undeclared_residual() -> None:
    rule = UndeclaredRightResidual(Congruence(erases_features=False))

    with pytest.raises(LawViolationError, match="undeclared premise slot 1"):
        adjointness(exact_fixture(rule))


def test_adjointness_catches_an_out_of_range_residual_slot() -> None:
    class OutOfRange(ForwardApplication):
        """Rule with an invalid residual-slot declaration."""

        @property
        @override
        def residual_slots(self) -> frozenset[int]:
            """Declare a nonexistent premise slot."""
            return frozenset({2})

    with pytest.raises(LawViolationError, match="out-of-range"):
        adjointness(exact_fixture(OutOfRange(Congruence(erases_features=False))))


def test_adjointness_catches_direction_and_slot_disagreement() -> None:
    class MissingDirection(ForwardApplication):
        """Rule with residual slots but no residual direction."""

        @property
        @override
        def directions(self) -> frozenset[Direction]:
            """Declare only forward and checking support."""
            return frozenset({"forward", "check"})

    with pytest.raises(LawViolationError, match="residual direction False"):
        adjointness(exact_fixture(MissingDirection(Congruence(erases_features=False))))


def test_non_mutation_catches_a_rule_that_changes_its_premise() -> None:
    mutating = RuleFixture(Bump(), st.builds(Cell, st.integers()), rebuild_a_cell_rule)

    with pytest.raises(LawViolationError, match="non-mutation"):
        non_mutation(mutating)


def test_stability_catches_a_rule_that_sees_past_its_congruence() -> None:
    with pytest.raises(LawViolationError, match="stability"):
        stability(erasing_fixture(FeatureSensitive(Congruence(erases_features=True))))


def test_the_round_trip_catches_a_declaration_the_rebuild_drops() -> None:
    rule = ForwardApplication(Congruence(erases_features=True))
    lossy = RuleFixture(rule, _CATS, rebuild_forgetting_the_congruence, _EXAMPLES)

    with pytest.raises(LawViolationError, match="specification round trip"):
        spec_round_trip(lossy)


def test_a_rule_declaring_no_residual_direction_skips_adjointness() -> None:
    mutating = RuleFixture(Bump(), st.builds(Cell, st.integers()), rebuild_a_cell_rule)

    adjointness(mutating)


class Clone(ForwardApplication):
    """Renamed copy of forward application."""

    @property
    @override
    def name(self) -> str:
        """Name the rule."""
        return "application-again"


class Everything:
    """Permissive filter with distinct lexical and derived tags."""

    @property
    def name(self) -> str:
        """Name the filter."""
        return "everything"

    @property
    def alphabet(self) -> frozenset[str]:
        """Use one tag for a lexical item and one for a derived item."""
        return frozenset({"lex", "derived"})

    def lexical_tag(self) -> str:
        """Tag an item straight from the lexicon."""
        return "lex"

    def decisions(self, rules: Sequence[Rule[Cat]], /) -> NormalFormDecisionTable:
        """Defer every finite tag state to exact step admission."""
        return tuple(
            dict.fromkeys(product(self.alphabet, repeat=rule.arity), "premises")
            for rule in rules
        )

    def tag(self, rule: Rule[Cat], children: tuple[str, ...], /) -> str:
        """Tag an item a rule built."""
        del rule, children

        return "derived"

    def admits(
        self,
        rule: Rule[Cat],
        children: tuple[str, ...],
        inputs: tuple[Cat, ...],
        output: Cat,
        /,
    ) -> bool:
        """Keep the step."""
        del rule, children, inputs, output

        return True


class LexicalFunctors(Everything):
    """Filter that requires lexical functors."""

    @property
    @override
    def name(self) -> str:
        """Name the filter."""
        return "lexical-functors"

    @override
    def admits(
        self,
        rule: Rule[Cat],
        children: tuple[str, ...],
        inputs: tuple[Cat, ...],
        output: Cat,
        /,
    ) -> bool:
        """Reject a step with a derived functor."""
        del rule, inputs, output

        return bool(children) and children[0] == self.lexical_tag()


class OneRule(Everything):
    """Filter that accepts one of two equivalent rules."""

    @property
    @override
    def name(self) -> str:
        """Name the filter."""
        return "one-rule"

    @override
    def admits(
        self,
        rule: Rule[Cat],
        children: tuple[str, ...],
        inputs: tuple[Cat, ...],
        output: Cat,
        /,
    ) -> bool:
        """Reject steps from the second rule."""
        del children, inputs, output

        return rule.name == "forward-application"


class Untagged(Everything):
    """Filter with an undeclared derived-item tag."""

    @override
    def tag(self, rule: Rule[Cat], children: tuple[str, ...], /) -> str:
        """Return a tag from outside the alphabet."""
        del rule, children

        return "deep"


class StrayLexicalTag(Everything):
    """Filter with an undeclared lexical-item tag."""

    @override
    def lexical_tag(self) -> str:
        """Return a tag from outside the alphabet."""
        return "word"


class NoTags(Everything):
    """Normal form with an empty tag alphabet."""

    @property
    @override
    def alphabet(self) -> frozenset[str]:
        """Declare an empty alphabet."""
        return frozenset()


class Counting(Everything):
    """Filter with stateful derived-item tags."""

    def __init__(self) -> None:
        """Start the counter."""
        self._calls = 0

    @override
    def tag(self, rule: Rule[Cat], children: tuple[str, ...], /) -> str:
        """Alternate between the two declared tags."""
        del rule, children
        self._calls += 1

        return "lex" if self._calls % 2 else "derived"


class Forgetful(Everything):
    """A filter that returns its previous result.

    Parameters
    ----------
    answer
        The result returned before the first update.
    """

    def __init__(self, *, answer: bool) -> None:
        """Hold an answer over from before the first question."""
        self._held = answer

    @override
    def admits(
        self,
        rule: Rule[Cat],
        children: tuple[str, ...],
        inputs: tuple[Cat, ...],
        output: Cat,
        /,
    ) -> bool:
        """Return the previous answer and save the next one."""
        del rule, inputs, output
        held = self._held
        self._held = bool(children) and children[0] == self.lexical_tag()

        return held


class Drifting(Everything):
    """Filter with a stateful lexical tag."""

    def __init__(self) -> None:
        """Start the counter."""
        self._calls = 0

    @override
    def lexical_tag(self) -> str:
        """Alternate between the two declared tags."""
        self._calls += 1

        return "lex" if self._calls % 2 else "derived"


_TOY_RULES: Final[tuple[Rule[Cat], ...]] = (
    ForwardApplication(Congruence(erases_features=False)),
    Clone(Congruence(erases_features=False)),
)

_LEXICON: Final[Mapping[str, tuple[Cat, ...]]] = {
    "f": ((Atom("S"), Atom("NP"), Atom("NP")),),
    "g": ((Atom("NP"), Atom("NP")),),
    "a": ((Atom("NP"),),),
}

_TOKENS: Final = st.lists(
    st.sampled_from(sorted(_LEXICON)), min_size=1, max_size=4
).map(tuple)

_UNKNOWN_TOKENS: Final = st.lists(st.just("z"), min_size=1, max_size=3).map(tuple)

_SENTENCE_EXAMPLES: Final = 200


def _reach(
    normal_form: NormalForm[Cat] | None, tokens: tuple[str, ...]
) -> frozenset[Cat]:
    """Return categories spanning a sentence in the toy grammar.

    Parameters
    ----------
    normal_form
        The filter, or ``None`` to admit every derivation.
    tokens
        The sentence to parse.
    """
    lexical = "" if normal_form is None else normal_form.lexical_tag()
    cells: dict[tuple[int, int], set[tuple[Cat, str]]] = {
        (i, i + 1): {(cat, lexical) for cat in _LEXICON.get(token, ())}
        for i, token in enumerate(tokens)
    }

    for width in range(2, len(tokens) + 1):
        for i in range(len(tokens) - width + 1):
            j = i + width
            here: set[tuple[Cat, str]] = set()

            for k in range(i + 1, j):
                for left, left_tag in cells[i, k]:
                    for right, right_tag in cells[k, j]:
                        children = (left_tag, right_tag)
                        premises = (left, right)

                        for rule in _TOY_RULES:
                            for output in rule.apply(premises):
                                if normal_form is None:
                                    here.add((output, ""))

                                elif normal_form.admits(
                                    rule, children, premises, output
                                ):
                                    here.add((output, normal_form.tag(rule, children)))

            cells[i, j] = here

    return frozenset(cat for cat, _ in cells[0, len(tokens)])


def filter_fixture(
    normal_form: NormalForm[Cat],
    tokens: SearchStrategy[tuple[str, ...]] = _TOKENS,
) -> NormalFormFixture[Cat]:
    """Return a normal-form fixture over the toy grammar.

    Parameters
    ----------
    normal_form
        The filter under test.
    tokens
        The sentence strategy used by the laws.
    """
    return NormalFormFixture(
        normal_form, _TOY_RULES, _PLAIN_CATS, tokens, _reach, _SENTENCE_EXAMPLES
    )


@pytest.mark.parametrize("law", NORMAL_FORM_LAWS, ids=lambda law: law.__name__)
def test_each_normal_form_law_collects_and_runs_on_its_own(
    law: Callable[[NormalFormFixture[Cat]], None],
) -> None:
    law(filter_fixture(Everything()))


def test_a_filter_that_keeps_every_step_holds_to_every_law() -> None:
    check_normal_form_laws(filter_fixture(Everything()))


def test_a_filter_that_cuts_only_redundant_steps_holds_to_every_law() -> None:
    check_normal_form_laws(filter_fixture(OneRule()))


def test_the_normal_form_entry_point_reports_the_law_that_failed() -> None:
    with pytest.raises(LawViolationError, match="goal completeness"):
        check_normal_form_laws(filter_fixture(LexicalFunctors()))


def test_the_alphabet_law_catches_a_tag_the_filter_did_not_declare() -> None:
    with pytest.raises(LawViolationError, match="declared tags"):
        tags_are_declared(filter_fixture(Untagged()))


def test_the_alphabet_law_catches_an_undeclared_lexical_tag() -> None:
    with pytest.raises(LawViolationError, match="declared tags"):
        tags_are_declared(filter_fixture(StrayLexicalTag()))


def test_the_alphabet_law_catches_a_filter_with_no_tags_at_all() -> None:
    with pytest.raises(LawViolationError, match="empty tag alphabet"):
        tags_are_declared(filter_fixture(NoTags()))


def test_a_law_refuses_a_fixture_naming_no_rules() -> None:
    ruleless = NormalFormFixture(
        Everything(), (), _PLAIN_CATS, _TOKENS, _reach, _SENTENCE_EXAMPLES
    )

    with pytest.raises(LawViolationError, match="no rules"):
        tags_are_declared(ruleless)


def test_purity_catches_a_filter_tagging_from_a_counter() -> None:
    with pytest.raises(LawViolationError, match="purity"):
        tagging_is_pure(filter_fixture(Counting()))


def test_purity_catches_a_filter_that_answers_the_previous_question() -> None:
    with pytest.raises(LawViolationError, match="purity"):
        tagging_is_pure(filter_fixture(Forgetful(answer=True)))


def test_purity_catches_a_lexical_tag_that_changes_under_the_filter() -> None:
    with pytest.raises(LawViolationError, match="purity"):
        tagging_is_pure(filter_fixture(Drifting()))


def test_the_differential_law_catches_a_filter_that_drops_a_reading() -> None:
    with pytest.raises(LawViolationError, match="goal completeness"):
        filtering_preserves_goal_terms(filter_fixture(LexicalFunctors()))


def test_the_differential_law_refuses_to_pass_on_sentences_with_no_reading() -> None:
    with pytest.raises(LawViolationError, match="never exercised"):
        filtering_preserves_goal_terms(filter_fixture(Everything(), _UNKNOWN_TOKENS))


def test_provider_validation_rejects_duplicate_provider_names() -> None:
    case = RuleLawCase("ordinary/one", "rule", _pass_rule_laws)
    providers = (
        RuleLawCases("duplicate", (case,)),
        RuleLawCases(
            "duplicate",
            (RuleLawCase("ordinary/two", "rule", _pass_rule_laws),),
        ),
    )

    with pytest.raises(ValueError, match="duplicate rule-law provider"):
        check_rule_law_cases(providers)


def test_provider_validation_rejects_duplicate_case_names() -> None:
    providers = (
        RuleLawCases(
            "first",
            (RuleLawCase("ordinary/shared", "one", _pass_rule_laws),),
        ),
        RuleLawCases(
            "second",
            (RuleLawCase("ordinary/shared", "two", _pass_rule_laws),),
        ),
    )

    with pytest.raises(ValueError, match="duplicate rule-law case"):
        check_rule_law_cases(providers)


def test_provider_validation_rejects_an_empty_ordinary_contribution() -> None:
    providers = (RuleLawCases("empty", ()),)

    with pytest.raises(ValueError, match="contributes no ordinary cases"):
        check_rule_law_cases(providers)


def test_provider_validation_rejects_an_r3_case_that_makes_no_calls() -> None:
    providers = (
        RuleLawCases(
            "provider",
            (RuleLawCase("ordinary/one", "rule", _pass_rule_laws),),
            (RuleTotalityCase("r3/one", "rule", _no_totality_calls),),
        ),
    )

    with pytest.raises(ValueError, match="made no admission calls"):
        check_rule_law_cases(providers)


def test_a_rule_law_report_names_every_coverage_channel() -> None:
    providers = (
        RuleLawCases(
            "provider",
            (RuleLawCase("ordinary/one", "ordinary-rule", _pass_rule_laws),),
            (RuleTotalityCase("r3/one", "totality-rule", _one_totality_call),),
        ),
    )
    report = check_rule_law_cases(providers)

    assert report.providers == ("provider",)
    assert report.ordinary_cases == ("ordinary/one",)
    assert report.totality_cases == ("r3/one",)
    assert report.runtime_rules == (
        "ordinary/one:ordinary-rule",
        "r3/one:totality-rule",
    )
    assert report.totality_calls == 1
    assert report.covered == (
        "provider:provider",
        "case:ordinary/one",
        "case:r3/one",
        "rule:ordinary/one:ordinary-rule",
        "rule:r3/one:totality-rule",
    )


@pytest.mark.conformance
def test_registered_providers_run_the_ordinary_and_r3_matrices() -> None:
    report = check_registered_rule_laws(
        Registry[RuleLawProvider](RULE_LAW_FAMILY, source=_registered_rule_law_points)
    )

    assert report.providers == (
        "composition",
        "pattern",
        "raising",
        "substitution",
    )
    assert len(report.ordinary_cases) == 116
    assert len(report.totality_cases) == 41
    assert len(report.runtime_rules) == 157
    assert report.totality_calls == 123_000
    assert report.covered


def test_installed_metadata_discovers_every_builtin_rule_law_provider() -> None:
    providers = discover_rule_law_cases()

    assert tuple(provider.provider for provider in providers) == (
        "composition",
        "pattern",
        "raising",
        "substitution",
    )
    assert tuple(len(provider.ordinary) for provider in providers) == (
        78,
        2,
        12,
        24,
    )
    assert tuple(len(provider.totality) for provider in providers) == (38, 0, 1, 2)

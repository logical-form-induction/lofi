from copy import copy, deepcopy
from pickle import dumps, loads
from typing import TYPE_CHECKING, Final

import pytest
from tests.support.rules import ccg_signature, declared_rules

from lofi.data import (
    Derivation,
    DerivationNode,
    LabelBlocker,
    ReachabilityReport,
    ReachableDerivation,
    RuleBlocker,
    TokenSpan,
    UnrecoverableStep,
    reachability_report,
)
from lofi.judge import NoNormalForm, NormalFormRole, RuleSet
from lofi.spec import ApplicationSpec, CompositionSpec, RuleSetSpec

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from lofi._types import JsonValue
    from lofi.judge import Direction, RuleProbe, RuleRestriction, Slot
    from lofi.terms import Term


class Exact:
    """Integer-identity matching."""

    __slots__ = ()

    @property
    def name(self) -> str:
        """Return the matching name."""
        return "exact"

    @property
    def has_key(self) -> bool:
        """Report that representatives are available."""
        return True

    def key(self, term: int, /) -> int:
        """Return the integer representative."""
        return term

    def equivalent(self, left: int, right: int, /) -> bool:
        """Compare integers exactly."""
        return left == right


_MATCHING: Final = Exact()


class Labels:
    """A finite integer label inventory."""

    __slots__ = ("_labels",)

    def __init__(self, labels: frozenset[int]) -> None:
        self._labels = labels

    def contains(self, label: int, /) -> bool:
        """Return whether the inventory contains a label."""
        return label in self._labels


_INTEGER_LABELS: Final = Labels(frozenset(range(10)))


class TrackingLabels:
    """A label inventory that records membership checks."""

    __slots__ = ("_labels", "checked")

    def __init__(self, labels: frozenset[int]) -> None:
        self._labels = labels
        self.checked: list[int] = []

    def contains(self, label: int, /) -> bool:
        """Record and test one label."""
        self.checked.append(label)
        return label in self._labels


class Relation:
    """A check-only integer relation."""

    __slots__ = ("_arity", "_holds", "_name")

    def __init__(
        self,
        name: str,
        arity: int,
        holds: Callable[[tuple[int, ...], int], bool],
    ) -> None:
        self._name = name
        self._arity = arity
        self._holds = holds

    @property
    def name(self) -> str:
        """Return the rule name."""
        return self._name

    @property
    def arity(self) -> int:
        """Return the premise count."""
        return self._arity

    @property
    def directions(self) -> frozenset[Direction]:
        """Return the check direction."""
        return frozenset({"check"})

    @property
    def normal_form_role(self) -> NormalFormRole:
        return NormalFormRole()

    @property
    def spine_order(self) -> int:
        return 0

    @property
    def residual_slots(self) -> frozenset[int]:
        """Return no residual slots."""
        return frozenset()

    @property
    def forward_is_complete(self) -> bool:
        """Report incomplete forward enumeration."""
        return False

    @property
    def matching(self) -> Exact:
        """Return exact integer matching."""
        return _MATCHING

    @property
    def restriction(self) -> RuleRestriction | None:
        """Return no rule restriction."""
        return None

    @property
    def probe(self) -> RuleProbe | None:
        """Return no head probe."""
        return None

    def apply(self, inputs: tuple[int, ...], /) -> Iterator[int]:
        """Yield no forward conclusions."""
        del inputs

        return iter(())

    def admits(self, inputs: tuple[int, ...], output: int, /) -> bool:
        """Return whether the declared relation holds."""
        return self._holds(inputs, output)

    def residuate(
        self,
        output: int,
        known: tuple[Slot[int], ...],
        /,
    ) -> Iterator[int]:
        """Yield no residual premises."""
        del output, known

        return iter(())

    def to_spec(self) -> Mapping[str, JsonValue]:
        """Return the rule declaration."""
        return {"rule": self._name}


def _ruleset(*rules: Relation) -> RuleSet[int]:
    """Build an integer rule set."""
    return RuleSet("integer", rules, _MATCHING, NoNormalForm())


def _local_derivation(output: int, inputs: tuple[int, ...]) -> Derivation[int]:
    """Build a one-step derivation."""
    root_span = TokenSpan(0, len(inputs))
    nodes = {
        0: DerivationNode(root_span, output, children=tuple(range(1, len(inputs) + 1)))
    }
    nodes.update(
        {
            position: DerivationNode(
                TokenSpan(position - 1, position),
                value,
            )
            for position, value in enumerate(inputs, start=1)
        }
    )

    return Derivation(0, nodes)


def _leaf_derivation(value: int) -> Derivation[int]:
    """Build a lexical derivation."""
    return Derivation(0, {0: DerivationNode(TokenSpan(0, 1), value)})


def _nested_failures() -> Derivation[int]:
    """Build a derivation with two distinct failed steps."""
    return Derivation(
        0,
        {
            0: DerivationNode(TokenSpan(0, 3), 9, children=(1, 4)),
            1: DerivationNode(TokenSpan(0, 2), 5, children=(2, 3)),
            2: DerivationNode(TokenSpan(0, 1), 1),
            3: DerivationNode(TokenSpan(1, 2), 2),
            4: DerivationNode(TokenSpan(2, 3), 4),
        },
    )


def _out_of_order_derivation() -> Derivation[int]:
    """Build a nested derivation whose ids and mapping order differ from preorder."""
    return Derivation(
        40,
        {
            13: DerivationNode(TokenSpan(1, 2), 2),
            2: DerivationNode(TokenSpan(2, 3), 4),
            40: DerivationNode(
                TokenSpan(0, 3),
                9,
                rule="outer",
                children=(7, 2),
            ),
            11: DerivationNode(TokenSpan(0, 1), 1),
            7: DerivationNode(
                TokenSpan(0, 2),
                5,
                rule="inner",
                children=(11, 13),
            ),
        },
    )


def test_empty_and_leaf_only_inputs_have_optional_zero_denominator_rates() -> None:
    empty = reachability_report(_INTEGER_LABELS, _ruleset(), iter(()))
    leaf = reachability_report(
        _INTEGER_LABELS,
        _ruleset(),
        iter((_leaf_derivation(1),)),
    )

    assert empty == ReachabilityReport(0, 0, 0, 0)
    assert empty.step_reachability is None
    assert empty.derivation_reachability is None
    assert leaf == ReachabilityReport(0, 0, 1, 1, outcomes=(ReachableDerivation(),))
    assert leaf.step_reachability is None
    assert leaf.derivation_reachability == 1.0


def test_unary_binary_and_ternary_steps_consume_a_generator_once() -> None:
    rules = _ruleset(
        Relation("unary", 1, lambda inputs, output: (inputs, output) == ((1,), 2)),
        Relation(
            "binary",
            2,
            lambda inputs, output: (inputs, output) == ((1, 2), 3),
        ),
        Relation(
            "ternary",
            3,
            lambda inputs, output: (inputs, output) == ((1, 2, 3), 6),
        ),
    )
    visited: list[int] = []

    def derivations() -> Iterator[Derivation[int]]:
        """Yield each fixture once."""
        for position, derivation in enumerate(
            (
                _leaf_derivation(1),
                _local_derivation(2, (1,)),
                _local_derivation(3, (1, 2)),
                _local_derivation(6, (1, 2, 3)),
            )
        ):
            visited.append(position)
            yield derivation

    report = reachability_report(_INTEGER_LABELS, rules, derivations())

    assert visited == [0, 1, 2, 3]
    assert report == ReachabilityReport[int](
        3,
        3,
        4,
        4,
        outcomes=tuple(ReachableDerivation() for _ in range(4)),
    )
    assert report.step_reachability == 1.0
    assert report.derivation_reachability == 1.0


def test_ambiguous_recovery_counts_as_one_reachable_step() -> None:
    rules = _ruleset(
        Relation("first", 1, lambda inputs, output: (inputs, output) == ((2,), 4)),
        Relation("second", 1, lambda inputs, output: (inputs, output) == ((2,), 4)),
    )

    report = reachability_report(
        _INTEGER_LABELS,
        rules,
        (_local_derivation(4, (2,)),),
    )

    assert report == ReachabilityReport(
        1,
        1,
        1,
        1,
        outcomes=(ReachableDerivation(),),
    )
    assert report.unrecoverable_steps == ()


def test_every_failed_step_is_checked_in_preorder() -> None:
    report = reachability_report(
        _INTEGER_LABELS,
        _ruleset(),
        (_nested_failures(),),
    )

    assert report == ReachabilityReport[int](
        2,
        0,
        1,
        0,
        (
            UnrecoverableStep((5, 4), 9, 1),
            UnrecoverableStep((1, 2), 5, 1),
        ),
        (RuleBlocker(0, (5, 4), 9),),
    )


def test_repeated_failed_shapes_are_aggregated_at_first_observation() -> None:
    repeated = _local_derivation(8, (3, 4))
    other = _local_derivation(7, (3, 4))
    report = reachability_report(
        _INTEGER_LABELS,
        _ruleset(),
        (repeated, other, repeated),
    )

    assert report.unrecoverable_steps == (
        UnrecoverableStep((3, 4), 8, 2),
        UnrecoverableStep((3, 4), 7, 1),
    )
    assert report.checked_steps == 3
    assert report.reachable_steps == 0
    assert report.checked_derivations == 3
    assert report.reachable_derivations == 0


def test_label_precedence_keeps_all_rule_failures_and_checks_leaf_labels() -> None:
    labels = TrackingLabels(frozenset({1, 5, 9}))
    report = reachability_report(labels, _ruleset(), (_out_of_order_derivation(),))

    assert labels.checked == [9, 5, 1, 2, 4]
    assert report.outcomes == (LabelBlocker(13, 2),)
    assert report.label_blocked_derivations == 1
    assert report.rule_blocked_derivations == 0
    assert report.unrecoverable_steps == (
        UnrecoverableStep((5, 4), 9, 1),
        UnrecoverableStep((1, 2), 5, 1),
    )


def test_first_failed_nonlexical_node_is_the_rule_blocker() -> None:
    report = reachability_report(
        _INTEGER_LABELS,
        _ruleset(),
        (_out_of_order_derivation(),),
    )

    assert report.outcomes == (RuleBlocker(40, (5, 4), 9, "outer"),)
    assert report.label_blocked_derivations == 0
    assert report.rule_blocked_derivations == 1
    assert report.checked_derivations == (
        report.reachable_derivations
        + report.label_blocked_derivations
        + report.rule_blocked_derivations
    )


def test_outcomes_preserve_derivation_input_order() -> None:
    report = reachability_report(
        Labels(frozenset({1, 2, 4, 5, 9})),
        _ruleset(),
        (
            _leaf_derivation(1),
            _out_of_order_derivation(),
            _leaf_derivation(3),
        ),
    )

    assert report.outcomes == (
        ReachableDerivation(),
        RuleBlocker(40, (5, 4), 9, "outer"),
        LabelBlocker(0, 3),
    )


def test_inventory_supersets_do_not_reduce_reachability() -> None:
    unary = Relation(
        "unary",
        1,
        lambda inputs, output: (inputs, output) == ((1,), 2),
    )
    binary = Relation(
        "binary",
        2,
        lambda inputs, output: (inputs, output) == ((1, 2), 3),
    )
    derivations = (_local_derivation(2, (1,)), _local_derivation(3, (1, 2)))

    subset = reachability_report(_INTEGER_LABELS, _ruleset(unary), derivations)
    superset = reachability_report(
        _INTEGER_LABELS, _ruleset(unary, binary), derivations
    )

    assert subset.reachable_steps <= superset.reachable_steps
    assert subset.reachable_derivations <= superset.reachable_derivations
    assert (subset.reachable_steps, subset.reachable_derivations) == (1, 1)
    assert (superset.reachable_steps, superset.reachable_derivations) == (2, 2)


_CCG_RULES: Final = declared_rules(
    ccg_signature("s", "np", "n", name="reachability-ccg"),
    RuleSetSpec(
        name="reachability-ccg",
        rules=(
            ApplicationSpec(view="/"),
            ApplicationSpec(view="\\"),
            CompositionSpec(direction="unrestricted", order=1, view="/"),
            CompositionSpec(direction="unrestricted", order=1, view="\\"),
        ),
    ),
)
_CCG_STORE: Final = _CCG_RULES.context.store


def _category(text: str) -> Term:
    """Parse one category of the compiled CCG signature."""
    return _CCG_RULES.context.syntax.parse(_CCG_STORE, text)


def test_removing_order_one_ccg_composition_records_its_exact_shape() -> None:
    full = _CCG_RULES.rule_set
    removed_name = "composition:/:1:harmonic:unrestricted"
    reduced = RuleSet(
        "without-forward-order-one",
        tuple(rule for rule in full if rule.name != removed_name),
        full.matching,
        full.normal_form,
        full.restriction_mode,
    )
    left = _category("s/np")
    right = _category("np/n")
    output = _category("s/n")
    derivation = Derivation(
        0,
        {
            0: DerivationNode(TokenSpan(0, 2), output, children=(1, 2)),
            1: DerivationNode(TokenSpan(0, 1), left),
            2: DerivationNode(TokenSpan(1, 2), right),
        },
    )

    full_report = reachability_report(_CCG_STORE, full, (derivation,))
    reduced_report = reachability_report(_CCG_STORE, reduced, (derivation,))

    assert full_report == ReachabilityReport(
        1,
        1,
        1,
        1,
        outcomes=(ReachableDerivation(),),
    )
    assert reduced_report == ReachabilityReport(
        1,
        0,
        1,
        0,
        (UnrecoverableStep((left, right), output, 1),),
        (RuleBlocker(0, (left, right), output),),
    )


class DomainError(ValueError):
    """A domain error raised by a rule."""

    def __init__(self) -> None:
        super().__init__("outside domain")


class DecodeError(ValueError):
    """A decode error raised by an input stream."""

    def __init__(self) -> None:
        super().__init__("undecoded input")


def test_domain_and_decode_errors_propagate() -> None:
    def reject_domain(_inputs: tuple[int, ...], _output: int) -> bool:
        """Raise the test domain error."""
        raise DomainError

    def broken_derivations() -> Iterator[Derivation[int]]:
        """Raise the test decode error."""
        raise DecodeError
        yield _leaf_derivation(1)

    with pytest.raises(DomainError, match="outside domain"):
        reachability_report(
            _INTEGER_LABELS,
            _ruleset(Relation("domain", 1, reject_domain)),
            (_local_derivation(2, (1,)),),
        )

    with pytest.raises(DecodeError, match="undecoded input"):
        reachability_report(_INTEGER_LABELS, _ruleset(), broken_derivations())


def test_report_values_are_frozen_hashable_copyable_and_pickleable() -> None:
    failure = UnrecoverableStep((1, 2), 4, 2)
    report = ReachabilityReport(
        3,
        1,
        2,
        0,
        (failure,),
        (RuleBlocker(0, (1, 2), 4), RuleBlocker(1, (1, 2), 4)),
    )

    for value, field in ((failure, "_output"), (report, "_checked_steps")):
        assert copy(value) is value
        assert deepcopy(value) is value
        assert loads(dumps(value)) == value
        assert hash(loads(dumps(value))) == hash(value)

        with pytest.raises(AttributeError, match="immutable"):
            setattr(value, field, 9)

        with pytest.raises(AttributeError, match="immutable"):
            delattr(value, field)


def test_report_values_reject_inconsistent_counts_and_duplicate_shapes() -> None:
    failure = UnrecoverableStep((1,), 2, 1)

    with pytest.raises(ValueError, match="reachable steps exceed"):
        ReachabilityReport(0, 1, 0, 0)

    with pytest.raises(ValueError, match="checked steps require"):
        ReachabilityReport(1, 1, 0, 0)

    with pytest.raises(ValueError, match="do not account"):
        ReachabilityReport(1, 0, 1, 0)

    with pytest.raises(ValueError, match="failed steps require"):
        ReachabilityReport(
            1,
            0,
            1,
            1,
            (failure,),
            (ReachableDerivation(),),
        )

    with pytest.raises(ValueError, match="repeats an unrecoverable shape"):
        ReachabilityReport(
            2,
            0,
            1,
            0,
            (failure, failure),
            (RuleBlocker(0, (1,), 2),),
        )

    with pytest.raises(ValueError, match="no occurrences"):
        UnrecoverableStep((1,), 2, 0)

from typing import TYPE_CHECKING, Final, override

import pytest
from hypothesis import strategies as st

from lofi._checks import LawViolationError
from lofi._types import ItemId
from lofi.eval._alignment import Alignment, Contingency
from lofi.eval._policies import ManyToOne, OneToOne
from lofi.eval._ports import TIE_BREAKS, AlignmentPolicy, Metric, TieBreak
from lofi.eval._values import ConventionVector, MetricResult, Resampling
from lofi.eval.laws import (
    AlignmentPolicyFixture,
    MetricFixture,
    check_alignment_policy_laws,
    check_metric_laws,
    compute_does_not_consume_the_metric,
    compute_is_order_invariant_when_declared,
    empty_metric_has_no_value,
    fit_is_deterministic,
    fit_is_injective_when_declared,
    fit_is_insensitive_to_accumulation_order,
    fit_records_the_counts_it_read,
    fit_refuses_the_test_split,
    per_item_keys_are_the_update_keys,
    reset_restores_the_constructed_state,
    update_does_not_mutate_its_arguments,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from hypothesis.strategies import SearchStrategy

type Spans = Sequence[str]

_ITEMS: Final = Resampling("item")

_CONVENTIONS: Final = ConventionVector(
    punctuation="strip",
    unary_chains="collapse",
    trivial_spans="drop",
    minimum_length=3,
    aggregation="corpus",
    coverage="zero",
    self_arcs="forbid",
    gold_derivation="canonical",
)


def _f1(counts: tuple[int, int, int]) -> float:
    true_positive, false_positive, false_negative = counts
    denominator = 2 * true_positive + false_positive + false_negative

    if denominator == 0:
        return 1.0

    return 2 * true_positive / denominator


def _result(counts: tuple[int, int, int], n: int) -> MetricResult:
    true_positive, false_positive, false_negative = counts

    return MetricResult(
        _f1(counts),
        n,
        _ITEMS,
        components={
            "tp": float(true_positive),
            "fp": float(false_positive),
            "fn": float(false_negative),
        },
    )


class SpanOverlap:
    """Micro-averaged F1 over proposed spans."""

    def __init__(self, conventions: ConventionVector) -> None:
        self._conventions = conventions
        self._items: dict[ItemId, tuple[int, int, int]] = {}

    @property
    def name(self) -> str:
        """Return the metric name."""
        return "span-overlap"

    @property
    def conventions(self) -> ConventionVector:
        """Return the scoring conventions."""
        return self._conventions

    @property
    def order_invariant(self) -> bool:
        """Return whether pooled counts are order-independent."""
        return True

    def reset(self) -> None:
        """Forget every item accumulated so far."""
        self._items = {}

    def update(self, pred: Spans, gold: Spans, *, key: ItemId) -> None:
        """Accumulate one item's true and false predictions."""
        predicted, wanted = set(pred), set(gold)
        seen = self._items.get(key, (0, 0, 0))
        hit = len(predicted & wanted)
        self._items[key] = (
            seen[0] + hit,
            seen[1] + len(predicted) - hit,
            seen[2] + len(wanted) - hit,
        )

    def compute(self) -> MetricResult:
        """Return the F1 of the pooled counts, absent over no items."""
        if not self._items:
            return MetricResult(None, 0, _ITEMS)

        pooled = tuple(sum(part) for part in zip(*self._items.values(), strict=True))

        return _result((pooled[0], pooled[1], pooled[2]), len(self._items))

    def per_item(self) -> Mapping[ItemId, MetricResult]:
        """Return each item's own F1, keyed by the key it was updated with."""
        return {key: _result(counts, 1) for key, counts in self._items.items()}


class NeverForgets(SpanOverlap):
    """A metric whose reset does nothing."""

    @override
    def reset(self) -> None:
        """Leave the accumulator unchanged."""


class InventsAnItem(SpanOverlap):
    """A metric with an extra item result."""

    @override
    def per_item(self) -> Mapping[ItemId, MetricResult]:
        """Return item results plus a synthetic item."""
        return {**super().per_item(), ItemId("ghost"): _result((0, 0, 0), 1)}


class ReportsTheFirstItem(SpanOverlap):
    """A metric that reports only its first item."""

    @override
    def compute(self) -> MetricResult:
        """Return the first item's result."""
        decomposed = self.per_item()

        if not decomposed:
            return MetricResult(None, 0, _ITEMS)

        return decomposed[next(iter(decomposed))]


class SortsItsInput(SpanOverlap):
    """A metric that mutates its input."""

    @override
    def update(self, pred: Spans, gold: Spans, *, key: ItemId) -> None:
        """Sort the prediction before scoring it."""
        if isinstance(pred, list):
            pred.sort()

        super().update(pred, gold, key=key)


class ScoresNothingAsZero(SpanOverlap):
    """A metric that reports zero for no items."""

    @override
    def compute(self) -> MetricResult:
        """Return zero when no items were scored."""
        if not self._items:
            return MetricResult(0.0, 1, _ITEMS)

        return super().compute()


class EmptiesOnCompute(SpanOverlap):
    """A metric consumed by computation."""

    @override
    def compute(self) -> MetricResult:
        """Return the result and clear the accumulator."""
        computed = super().compute()
        self._items = {}

        return computed


_KEYS: Final = ("i0", "i1", "i2")
_LABELS: Final = ("a", "b", "c")

_EXAMPLES: SearchStrategy[tuple[str, Spans, Spans]] = st.tuples(
    st.sampled_from(_KEYS),
    st.lists(st.sampled_from(_LABELS), max_size=3, unique=True).map(tuple),
    st.lists(st.sampled_from(_LABELS), max_size=3, unique=True).map(tuple),
)

_SCORED: SearchStrategy[tuple[str, Spans, Spans]] = st.sampled_from(
    [("i0", ("a",), ("a",)), ("i1", ("b",), ("c",))]
)


def _mutable_example() -> tuple[str, Spans, Spans]:
    return ("i0", ["c", "a"], ["a"])


def _fixture(
    build: Callable[[], Metric[Spans, Spans]],
    examples: SearchStrategy[tuple[str, Spans, Spans]] = _EXAMPLES,
) -> MetricFixture[Spans, Spans]:
    return MetricFixture(build, examples, max_examples=50)


def _conforming() -> Metric[Spans, Spans]:
    return SpanOverlap(_CONVENTIONS)


def test_a_conforming_metric_passes_every_law() -> None:
    check_metric_laws(_fixture(_conforming))


def test_a_metric_that_never_resets_is_caught() -> None:
    with pytest.raises(LawViolationError, match="reset"):
        reset_restores_the_constructed_state(
            _fixture(lambda: NeverForgets(_CONVENTIONS))
        )


def test_a_metric_that_invents_an_item_is_caught() -> None:
    with pytest.raises(LawViolationError, match="per-item keys"):
        per_item_keys_are_the_update_keys(_fixture(lambda: InventsAnItem(_CONVENTIONS)))


def test_a_metric_whose_value_follows_the_order_is_caught() -> None:
    with pytest.raises(LawViolationError, match="order invariance"):
        compute_is_order_invariant_when_declared(
            _fixture(lambda: ReportsTheFirstItem(_CONVENTIONS), _SCORED)
        )


def test_a_metric_that_normalizes_its_argument_is_caught() -> None:
    with pytest.raises(LawViolationError, match="non-mutation"):
        update_does_not_mutate_its_arguments(
            _fixture(lambda: SortsItsInput(_CONVENTIONS), st.builds(_mutable_example))
        )


def test_a_metric_that_scores_no_data_as_zero_is_caught() -> None:
    with pytest.raises(LawViolationError, match="the empty metric"):
        empty_metric_has_no_value(_fixture(lambda: ScoresNothingAsZero(_CONVENTIONS)))


def test_a_metric_that_empties_itself_on_computing_is_caught() -> None:
    with pytest.raises(LawViolationError, match="not consumption"):
        compute_does_not_consume_the_metric(
            _fixture(lambda: EmptiesOnCompute(_CONVENTIONS))
        )


def test_an_order_dependent_metric_is_excused_the_order_law() -> None:
    class Sequential(ReportsTheFirstItem):
        """A declared order-dependent metric."""

        @property
        @override
        def order_invariant(self) -> bool:
            """Return false."""
            return False

    compute_is_order_invariant_when_declared(
        _fixture(lambda: Sequential(_CONVENTIONS), _SCORED)
    )


class Alternates(ManyToOne):
    """A policy that changes the gold label on every fit."""

    def __init__(self, *, tie_break: TieBreak) -> None:
        super().__init__(tie_break=tie_break)
        self._fits = 0

    @override
    def fit(self, contingency: Contingency, /) -> Alignment:
        """Assign each induced label to the next gold label."""
        self._fits += 1
        gold = contingency.gold_labels[self._fits % len(contingency.gold_labels)]

        return Alignment(dict.fromkeys(contingency.induced_labels, gold), contingency)


class FirstSeenWins(ManyToOne):
    """A policy that selects the first listed gold label."""

    @override
    def fit(self, contingency: Contingency, /) -> Alignment:
        """Assign each induced label to its first listed gold label."""
        mapping: dict[str, str] = {}

        for induced, gold in contingency.counts:
            mapping.setdefault(induced, gold)

        return Alignment(mapping, contingency)


class ClaimsInjectivity(ManyToOne):
    """A policy that incorrectly claims injectivity."""

    @property
    @override
    def injective(self) -> bool:
        """Return true."""
        return True


class RelabelsTheSplit(ManyToOne):
    """A policy that relabels the fitted split."""

    @override
    def fit(self, contingency: Contingency, /) -> Alignment:
        """Fit after relabeling the split."""
        return super().fit(Contingency(contingency.counts, split="dev"))


class TrimsTheCounts(ManyToOne):
    """A policy that drops fit data."""

    @override
    def fit(self, contingency: Contingency, /) -> Alignment:
        """Fit on only the first table row."""
        first = next(iter(contingency.counts))

        return super().fit(
            Contingency({first: contingency.counts[first]}, split=contingency.split)
        )


_TABLES: SearchStrategy[Contingency] = st.dictionaries(
    st.tuples(st.sampled_from(("u0", "u1")), st.sampled_from(("NP", "VP"))),
    st.integers(min_value=0, max_value=3),
    min_size=4,
    max_size=4,
).map(lambda counts: Contingency(counts, split="dev"))

_COLLIDING: SearchStrategy[Contingency] = st.just(
    Contingency(
        {("u0", "NP"): 3, ("u0", "VP"): 1, ("u1", "NP"): 4, ("u1", "VP"): 2},
        split="dev",
    )
)

_SKEWED: SearchStrategy[Contingency] = st.just(
    Contingency(
        {("u0", "NP"): 1, ("u0", "VP"): 5, ("u1", "VP"): 1, ("u1", "NP"): 5},
        split="dev",
    )
)


def _policy_fixture(
    policy: AlignmentPolicy, contingencies: SearchStrategy[Contingency] = _TABLES
) -> AlignmentPolicyFixture:
    return AlignmentPolicyFixture(policy, contingencies, max_examples=25)


@pytest.mark.parametrize("tie_break", TIE_BREAKS)
def test_a_conforming_many_to_one_policy_passes_every_law(tie_break: TieBreak) -> None:
    check_alignment_policy_laws(_policy_fixture(ManyToOne(tie_break=tie_break)))


@pytest.mark.parametrize("tie_break", TIE_BREAKS)
def test_a_conforming_one_to_one_policy_passes_every_law(tie_break: TieBreak) -> None:
    check_alignment_policy_laws(_policy_fixture(OneToOne(tie_break=tie_break)))


def test_each_declared_tie_break_settles_a_tie_its_own_way() -> None:
    tied = Contingency(
        {("u0", "NP"): 2, ("u0", "VP"): 2, ("u1", "NP"): 0, ("u1", "VP"): 9},
        split="dev",
    )
    least = ManyToOne(tie_break="least-gold-label").fit(tied)
    marginal = ManyToOne(tie_break="largest-gold-marginal").fit(tied)

    assert least.apply("u0") == "NP"
    assert marginal.apply("u0") == "VP"


def test_a_one_to_one_fit_leaves_a_label_unassigned_when_gold_runs_out() -> None:
    crowded = Contingency(
        {("u0", "NP"): 5, ("u1", "NP"): 4, ("u2", "NP"): 3}, split="dev"
    )
    fitted = OneToOne(tie_break="least-gold-label").fit(crowded)

    assert fitted.apply("u0") == "NP"
    assert fitted.apply("u1") is None
    assert fitted.apply("u2") is None
    assert fitted.is_injective


def test_a_policy_that_answers_differently_each_time_is_caught() -> None:
    with pytest.raises(LawViolationError, match="determinism"):
        fit_is_deterministic(
            _policy_fixture(Alternates(tie_break="least-gold-label"), _COLLIDING)
        )


def test_a_policy_reading_the_accumulation_order_is_caught() -> None:
    with pytest.raises(LawViolationError, match="accumulation order"):
        fit_is_insensitive_to_accumulation_order(
            _policy_fixture(FirstSeenWins(tie_break="least-gold-label"), _SKEWED)
        )


def test_a_policy_claiming_injectivity_it_lacks_is_caught() -> None:
    with pytest.raises(LawViolationError, match="injectivity"):
        fit_is_injective_when_declared(
            _policy_fixture(ClaimsInjectivity(tie_break="least-gold-label"), _COLLIDING)
        )


def test_a_policy_that_renames_the_split_it_fitted_on_is_caught() -> None:
    with pytest.raises(LawViolationError, match="the test split"):
        fit_refuses_the_test_split(
            _policy_fixture(RelabelsTheSplit(tie_break="least-gold-label"), _COLLIDING)
        )


def test_a_policy_recording_less_than_it_read_is_caught() -> None:
    with pytest.raises(LawViolationError, match="recorded counts"):
        fit_records_the_counts_it_read(
            _policy_fixture(TrimsTheCounts(tie_break="least-gold-label"), _COLLIDING)
        )


def test_a_many_to_one_policy_is_excused_the_injectivity_law() -> None:
    fit_is_injective_when_declared(
        _policy_fixture(ManyToOne(tie_break="least-gold-label"), _COLLIDING)
    )

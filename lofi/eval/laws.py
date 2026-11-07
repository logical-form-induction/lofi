"""Evaluation conformance laws."""

from typing import TYPE_CHECKING, Final, NamedTuple

from hypothesis import given, settings
from hypothesis import strategies as st

from lofi._checks import fail_law
from lofi._types import ItemId
from lofi.eval._alignment import Contingency, SplitLeakError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from hypothesis.strategies import SearchStrategy
    from xarray import DataTree

    from lofi.eval._ports import AlignmentPolicy, Metric
    from lofi.eval._values import MetricResult
    from lofi.eval.refit import Refit, RefitCell

__all__ = [
    "ALIGNMENT_POLICY_LAWS",
    "METRIC_LAWS",
    "REFIT_LAWS",
    "AlignmentPolicyFixture",
    "MetricFixture",
    "RefitFixture",
    "a_refit_answers_every_cell",
    "a_refit_is_deterministic",
    "check_alignment_policy_laws",
    "check_metric_laws",
    "check_refit_laws",
    "compute_does_not_consume_the_metric",
    "compute_is_order_invariant_when_declared",
    "empty_metric_has_no_value",
    "every_fit_scores_the_same_observations",
    "fit_is_deterministic",
    "fit_is_injective_when_declared",
    "fit_is_insensitive_to_accumulation_order",
    "fit_records_the_counts_it_read",
    "fit_refuses_the_test_split",
    "per_item_keys_are_the_update_keys",
    "reset_restores_the_constructed_state",
    "update_does_not_mutate_its_arguments",
]

_FITS: Final = 20
"""The fit count for the determinism law."""


class MetricFixture[Pred, Gold](NamedTuple):
    """Stores inputs for a metric law.

    Parameters
    ----------
    build : Callable[[], Metric[Pred, Gold]]
        A factory for fresh metric instances.
    examples : SearchStrategy[tuple[str, Pred, Gold]]
        A strategy over item keys, predictions, and gold values.
    max_examples : int
        The maximum generated examples per law.
    """

    build: Callable[[], Metric[Pred, Gold]]
    examples: SearchStrategy[tuple[str, Pred, Gold]]
    max_examples: int = 100


class AlignmentPolicyFixture(NamedTuple):
    """Stores inputs for an alignment-policy law.

    Parameters
    ----------
    policy : AlignmentPolicy
        The alignment policy under test.
    contingencies : SearchStrategy[Contingency]
        A strategy over contingency tables.
    max_examples : int
        The maximum generated examples per law.
    """

    policy: AlignmentPolicy
    contingencies: SearchStrategy[Contingency]
    max_examples: int = 100


def _batch[Pred, Gold](
    fixture: MetricFixture[Pred, Gold],
) -> SearchStrategy[list[tuple[str, Pred, Gold]]]:
    """Return a strategy over batches of updates.

    Parameters
    ----------
    fixture : MetricFixture[Pred, Gold]
        The metric law fixture.
    """
    return st.lists(fixture.examples, min_size=1, max_size=6)


def _metric_budget[Pred, Gold](fixture: MetricFixture[Pred, Gold]) -> settings:
    """Return the Hypothesis settings for metric laws.

    Parameters
    ----------
    fixture : MetricFixture[Pred, Gold]
        The metric law fixture.
    """
    # allow slow metric implementations
    return settings(max_examples=fixture.max_examples, deadline=None)


def _policy_budget(fixture: AlignmentPolicyFixture) -> settings:
    """Return the Hypothesis settings for alignment laws.

    Parameters
    ----------
    fixture : AlignmentPolicyFixture
        The alignment-policy fixture.
    """
    return settings(max_examples=fixture.max_examples, deadline=None)


def _updated[Pred, Gold](
    fixture: MetricFixture[Pred, Gold], batch: Sequence[tuple[str, Pred, Gold]]
) -> Metric[Pred, Gold]:
    """Return a fresh metric after applying a batch of updates.

    Parameters
    ----------
    fixture : MetricFixture[Pred, Gold]
        The metric law fixture.
    batch : Sequence[tuple[str, Pred, Gold]]
        The updates to apply.
    """
    metric = fixture.build()

    for key, pred, gold in batch:
        metric.update(pred, gold, key=ItemId(key))

    return metric


def _state[Pred, Gold](
    metric: Metric[Pred, Gold],
) -> tuple[MetricResult, dict[ItemId, MetricResult]]:
    """Return all published metric state.

    Parameters
    ----------
    metric : Metric[Pred, Gold]
        The metric to inspect.
    """
    return (metric.compute(), dict(metric.per_item()))


def _unchanged[T](value: T) -> tuple[str, int | None]:
    """Return a mutation-detection witness.

    Parameters
    ----------
    value : T
        The value to represent and hash.
    """
    try:
        return (repr(value), hash(value))

    except TypeError:
        return (repr(value), None)


def empty_metric_has_no_value[Pred, Gold](
    fixture: MetricFixture[Pred, Gold],
) -> None:
    """Check that an empty metric reports no value or items.

    Parameters
    ----------
    fixture : MetricFixture[Pred, Gold]
        The metric law fixture.

    Raises
    ------
    LawViolationError
        If a fresh metric reports a value, count, or item.
    """
    metric = fixture.build()
    result = metric.compute()

    if result.value is not None or result.n != 0:
        detail = (
            f"{metric.name} over no updates reports {result.value!r} "
            f"over {result.n!r} units"
        )

        fail_law(detail, law="the empty metric")

    if dict(metric.per_item()):
        detail = f"{metric.name} over no updates reports {dict(metric.per_item())!r}"

        fail_law(detail, law="the empty metric")


def reset_restores_the_constructed_state[Pred, Gold](
    fixture: MetricFixture[Pred, Gold],
) -> None:
    """Check that reset restores the constructed metric state.

    Parameters
    ----------
    fixture : MetricFixture[Pred, Gold]
        The metric law fixture.

    Raises
    ------
    LawViolationError
        If reset does not restore a generated metric state.
    """
    fresh = _state(fixture.build())

    @given(_batch(fixture))
    @_metric_budget(fixture)
    def law(batch: list[tuple[str, Pred, Gold]]) -> None:
        metric = _updated(fixture, batch)
        metric.reset()

        if _state(metric) != fresh:
            detail = (
                f"{metric.name} reset after {len(batch)} updates reports "
                f"{_state(metric)!r}, a fresh one {fresh!r}"
            )

            fail_law(detail, law="reset")

    law()


def per_item_keys_are_the_update_keys[Pred, Gold](
    fixture: MetricFixture[Pred, Gold],
) -> None:
    """Check that per-item results use exactly the update keys.

    Parameters
    ----------
    fixture : MetricFixture[Pred, Gold]
        The metric law fixture.

    Raises
    ------
    LawViolationError
        If generated update and result keys differ.
    """

    @given(_batch(fixture))
    @_metric_budget(fixture)
    def law(batch: list[tuple[str, Pred, Gold]]) -> None:
        metric = _updated(fixture, batch)
        wanted = {ItemId(key) for key, _, _ in batch}
        got = set(metric.per_item())

        if got != wanted:
            detail = (
                f"{metric.name} was updated with {sorted(wanted)!r} and "
                f"decomposes over {sorted(got)!r}"
            )

            fail_law(detail, law="per-item keys")

    law()


def compute_is_order_invariant_when_declared[Pred, Gold](
    fixture: MetricFixture[Pred, Gold],
) -> None:
    """Check a metric's declared update-order invariance.

    Parameters
    ----------
    fixture : MetricFixture[Pred, Gold]
        The metric law fixture.

    Raises
    ------
    LawViolationError
        If two generated update orders produce different results.
    """
    if not fixture.build().order_invariant:
        return

    permuted = _batch(fixture).flatmap(
        lambda batch: st.tuples(st.just(batch), st.permutations(batch))
    )

    @given(permuted)
    @_metric_budget(fixture)
    def law(
        pair: tuple[list[tuple[str, Pred, Gold]], list[tuple[str, Pred, Gold]]],
    ) -> None:
        batch, shuffled = pair
        first = _updated(fixture, batch).compute()
        second = _updated(fixture, shuffled).compute()

        if first != second:
            detail = (
                f"{fixture.build().name} declares itself order-invariant but "
                f"computes {first!r} in one order and {second!r} in another"
            )

            fail_law(detail, law="order invariance")

    law()


def update_does_not_mutate_its_arguments[Pred, Gold](
    fixture: MetricFixture[Pred, Gold],
) -> None:
    """Check that update does not mutate its arguments.

    Parameters
    ----------
    fixture : MetricFixture[Pred, Gold]
        The metric law fixture.

    Raises
    ------
    LawViolationError
        If an update changes a generated prediction or gold value.
    """

    @given(_batch(fixture))
    @_metric_budget(fixture)
    def law(batch: list[tuple[str, Pred, Gold]]) -> None:
        metric = fixture.build()

        for key, pred, gold in batch:
            before = (_unchanged(pred), _unchanged(gold))
            metric.update(pred, gold, key=ItemId(key))

            if (_unchanged(pred), _unchanged(gold)) != before:
                detail = f"{metric.name} changed {pred!r} or {gold!r} under {key!r}"

                fail_law(detail, law="non-mutation")

    law()


def compute_does_not_consume_the_metric[Pred, Gold](
    fixture: MetricFixture[Pred, Gold],
) -> None:
    """Check that compute does not consume metric state.

    Parameters
    ----------
    fixture : MetricFixture[Pred, Gold]
        The metric law fixture.

    Raises
    ------
    LawViolationError
        If repeated computation changes published state.
    """

    @given(_batch(fixture))
    @_metric_budget(fixture)
    def law(batch: list[tuple[str, Pred, Gold]]) -> None:
        metric = _updated(fixture, batch)
        first = _state(metric)

        if _state(metric) != first:
            detail = (
                f"{metric.name} reports {first!r} and then {_state(metric)!r} "
                f"with no update in between"
            )

            fail_law(detail, law="computation is not consumption")

    law()


def fit_is_deterministic(fixture: AlignmentPolicyFixture) -> None:
    """Check that repeated fits return one alignment.

    Parameters
    ----------
    fixture : AlignmentPolicyFixture
        The alignment-policy fixture.

    Raises
    ------
    LawViolationError
        If repeated fits of a generated table differ.
    """
    policy = fixture.policy

    @given(fixture.contingencies)
    @_policy_budget(fixture)
    def law(contingency: Contingency) -> None:
        texts = {policy.fit(contingency).serialize() for _ in range(_FITS)}

        if len(texts) != 1:
            detail = (
                f"{policy.name} fitted {contingency!r} {_FITS} times and "
                f"produced {len(texts)} different alignments"
            )

            fail_law(detail, law="determinism")

    law()


def fit_is_injective_when_declared(fixture: AlignmentPolicyFixture) -> None:
    """Check a policy's declared injectivity.

    Parameters
    ----------
    fixture : AlignmentPolicyFixture
        The alignment-policy fixture.

    Raises
    ------
    LawViolationError
        If a generated fit violates declared injectivity.
    """
    policy = fixture.policy

    if not policy.injective:
        return

    @given(fixture.contingencies)
    @_policy_budget(fixture)
    def law(contingency: Contingency) -> None:
        alignment = policy.fit(contingency)

        if not alignment.is_injective:
            detail = (
                f"{policy.name} declares itself injective and fitted "
                f"{dict(alignment.mapping)!r}"
            )

            fail_law(detail, law="injectivity")

    law()


def fit_refuses_the_test_split(fixture: AlignmentPolicyFixture) -> None:
    """Check that fitting refuses the test split.

    Parameters
    ----------
    fixture : AlignmentPolicyFixture
        The alignment-policy fixture.

    Raises
    ------
    LawViolationError
        If the policy fits a generated test contingency.
    """
    policy = fixture.policy

    @given(fixture.contingencies)
    @_policy_budget(fixture)
    def law(contingency: Contingency) -> None:
        on_test = Contingency(contingency.counts, split="test")

        try:
            fitted = policy.fit(on_test)

        except SplitLeakError:
            return

        detail = f"{policy.name} fitted {dict(fitted.mapping)!r} on the test split"

        fail_law(detail, law="the test split")

    law()


def fit_records_the_counts_it_read(fixture: AlignmentPolicyFixture) -> None:
    """Check that a fit records its source contingency.

    Parameters
    ----------
    fixture : AlignmentPolicyFixture
        The alignment-policy fixture.

    Raises
    ------
    LawViolationError
        If a fit records different counts or a different split.
    """
    policy = fixture.policy

    @given(fixture.contingencies)
    @_policy_budget(fixture)
    def law(contingency: Contingency) -> None:
        alignment = policy.fit(contingency)

        if alignment.contingency != contingency:
            detail = (
                f"{policy.name} fitted on {contingency!r} and recorded "
                f"{alignment.contingency!r}"
            )

            fail_law(detail, law="recorded counts")

        if alignment.split != contingency.split:
            detail = (
                f"{policy.name} fitted on the {contingency.split!r} split and "
                f"recorded the {alignment.split!r} one"
            )

            fail_law(detail, law="recorded counts")

    law()


def fit_is_insensitive_to_accumulation_order(fixture: AlignmentPolicyFixture) -> None:
    """Check that contingency accumulation order does not affect a fit.

    Parameters
    ----------
    fixture : AlignmentPolicyFixture
        The alignment-policy fixture.

    Raises
    ------
    LawViolationError
        If reordered equal counts produce different fits.
    """
    policy = fixture.policy

    @given(fixture.contingencies)
    @_policy_budget(fixture)
    def law(contingency: Contingency) -> None:
        reordered = Contingency(
            dict(reversed(list(contingency.counts.items()))),
            split=contingency.split,
        )
        first = policy.fit(contingency).serialize()
        second = policy.fit(reordered).serialize()

        if first != second:
            detail = (
                f"{policy.name} fits {first} from {contingency!r} and {second} "
                f"from the same counts accumulated in the opposite order"
            )

            fail_law(detail, law="accumulation order")

    law()


METRIC_LAWS: Final = (
    empty_metric_has_no_value,
    reset_restores_the_constructed_state,
    per_item_keys_are_the_update_keys,
    compute_is_order_invariant_when_declared,
    update_does_not_mutate_its_arguments,
    compute_does_not_consume_the_metric,
)
"""Metric laws in conformance order."""

ALIGNMENT_POLICY_LAWS: Final = (
    fit_is_deterministic,
    fit_is_insensitive_to_accumulation_order,
    fit_is_injective_when_declared,
    fit_refuses_the_test_split,
    fit_records_the_counts_it_read,
)
"""Alignment-policy laws in conformance order."""


def check_metric_laws[Pred, Gold](fixture: MetricFixture[Pred, Gold]) -> None:
    """Run every metric law against one metric.

    Parameters
    ----------
    fixture : MetricFixture[Pred, Gold]
        The metric law fixture.

    Raises
    ------
    LawViolationError
        If the metric violates a law.
    """
    for law in METRIC_LAWS:
        law(fixture)


def check_alignment_policy_laws(fixture: AlignmentPolicyFixture) -> None:
    """Run every alignment-policy law against one policy.

    Parameters
    ----------
    fixture : AlignmentPolicyFixture
        The alignment-policy fixture.

    Raises
    ------
    LawViolationError
        If the policy violates a law.
    """
    for law in ALIGNMENT_POLICY_LAWS:
        law(fixture)


class RefitFixture(NamedTuple):
    """Stores inputs for a refit law.

    Parameters
    ----------
    refit : Refit
        The refit under test.
    cells : tuple[RefitCell, ...]
        The configurations the laws fit it under.
    densities : Callable[[DataTree], tuple[float, ...]]
        What reads a fit's leave-one-out density at each observation.
    """

    refit: Refit
    cells: tuple[RefitCell, ...]
    densities: Callable[[DataTree], tuple[float, ...]]


def a_refit_answers_every_cell(fixture: RefitFixture) -> None:
    """Check that every configuration is fit and scored.

    Parameters
    ----------
    fixture : RefitFixture
        The refit fixture.

    Raises
    ------
    LawViolationError
        If the fixture names no cell, or a cell produces a fit that scores no
        observation, which leaves the cell with no row to report.
    """
    if not fixture.cells:
        fail_law("the fixture names no cell to fit", law="answers_every_cell")

    for cell in fixture.cells:
        scored = fixture.densities(fixture.refit(cell))

        if not scored:
            detail = f"the fit of {cell!r} scored no observation"

            fail_law(detail, law="answers_every_cell")


def a_refit_is_deterministic(fixture: RefitFixture) -> None:
    """Check that fitting one configuration twice gives one answer.

    Parameters
    ----------
    fixture : RefitFixture
        The refit fixture.

    Raises
    ------
    LawViolationError
        If two fits of one cell score the same observations differently, which
        makes a difference between two cells unreadable.
    """
    for cell in fixture.cells:
        first = fixture.densities(fixture.refit(cell))
        second = fixture.densities(fixture.refit(cell))

        if first != second:
            detail = f"two fits of {cell!r} scored differently"

            fail_law(detail, law="deterministic")


def every_fit_scores_the_same_observations(fixture: RefitFixture) -> None:
    """Check that all configurations are scored on one set of observations.

    Parameters
    ----------
    fixture : RefitFixture
        The refit fixture.

    Raises
    ------
    LawViolationError
        If two cells score different numbers of observations, since a difference
        in expected log predictive density between fits scored on different data
        measures the data rather than the signature.
    """
    counts = {
        cell: len(fixture.densities(fixture.refit(cell))) for cell in fixture.cells
    }

    if len(set(counts.values())) > 1:
        detail = f"the cells scored different observation counts: {counts!r}"

        fail_law(detail, law="same_observations")


REFIT_LAWS: Final = (
    a_refit_answers_every_cell,
    a_refit_is_deterministic,
    every_fit_scores_the_same_observations,
)
"""Refit laws in conformance order."""


def check_refit_laws(fixture: RefitFixture) -> None:
    """Run every refit law against one refit.

    Parameters
    ----------
    fixture : RefitFixture
        The refit fixture.

    Raises
    ------
    LawViolationError
        If the refit violates a law.
    """
    for law in REFIT_LAWS:
        law(fixture)

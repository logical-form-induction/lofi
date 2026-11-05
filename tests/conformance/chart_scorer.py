from math import isclose
from typing import TYPE_CHECKING, Final, NamedTuple, NoReturn

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from tests.conformance._sites import SiteMapSuite, SiteObservation

from lofi._checks import LawViolationError
from lofi.nn._values import LogDistribution

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from torch import Tensor

    from lofi.nn._values import ChartReadouts, Encoded
    from lofi.nn.ports import ChartScorer, InstrumentScope, Reading
    from lofi.nn.sites import Site

__all__ = [
    "CHART_INSTRUMENTATION_LAWS",
    "CHART_SCORER_LAWS",
    "ChartInstrumentationConformance",
    "ChartInstrumentationSuite",
    "ChartScorerConformance",
    "ChartScorerSuite",
    "published",
]

_NORMALIZATION_ATOL: Final = 1e-5


def _fail(detail: str, *, law: str) -> NoReturn:
    """Raise a law violation, naming the law and what went wrong.

    Parameters
    ----------
    detail : str
        The counterexample, in that law's own terms.
    law : str
        The law's short name.

    Raises
    ------
    LawViolationError
        Always.
    """
    message = f"{law}: {detail}"

    raise LawViolationError(message)


def published(readouts: ChartReadouts) -> Mapping[str, LogDistribution | Encoded]:
    """Return the readouts a pass filled, under the name each has on the record.

    Parameters
    ----------
    readouts : ChartReadouts
        What a pass returned.

    Returns
    -------
    collections.abc.Mapping
        Each present readout under its name, in declaration order.
    """
    every = (
        ("inside", readouts.inside),
        ("outside", readouts.outside),
        ("alpha", readouts.alpha),
        ("mu", readouts.mu),
        ("log_z", readouts.log_z),
        (
            "actions",
            None if readouts.actions is None else readouts.actions.scores,
        ),
        ("identity", readouts.identity),
        ("cells", readouts.cells),
        ("outside_cells", readouts.outside_cells),
        ("spans", readouts.spans),
        ("entropy", readouts.entropy),
        ("mbr", readouts.mbr),
    )

    return {name: value for name, value in every if value is not None}


def _padded(mask: Tensor, ndim: int) -> Tensor:
    """Return a mask over the positions a tensor of some rank does not hold data at.

    Parameters
    ----------
    mask : Tensor
        True where a position holds data, over the outermost axes.
    ndim : int
        The rank of the tensor the mask covers.

    Returns
    -------
    torch.Tensor
        A boolean tensor of that rank, broadcastable against it, true where the position
        is padding.
    """
    return ~mask.reshape(tuple(mask.shape) + (1,) * (ndim - mask.ndim))


class ChartScorerConformance[T: ChartScorer](NamedTuple):
    """Inputs for the chart-scorer conformance laws.

    Parameters
    ----------
    build : Callable[[], T]
        Return the scorer under test.
    leaves : Callable[[int, int], Encoded]
        Return the leaves of a batch of a given size and sentence length, over the axes
        ``("item", "word", "width")`` and masked over the ragged part.
    axes : Mapping[str, tuple[str, ...]]
        The axes expected on each published readout, under the name that
        readout has on the record.
    split_axis : str
        The axis of the split posterior the mass sums over.
    ragged_axis : str
        The axis of a readout that runs along the ragged part of the batch, so that a
        readout with it must have a mask.
    wire : Callable[[T, str], None]
        Wire a loss that consumes one named readout.
    sites : SiteObservation[T]
        How the site map law reaches this scorer's interior.
    items : int
        How many items the laws run the scorer over.
    words : int
        How many words each of those items holds.
    """

    build: Callable[[], T]
    leaves: Callable[[int, int], Encoded]
    axes: Mapping[str, tuple[str, ...]]
    split_axis: str
    ragged_axis: str
    wire: Callable[[T, str], None]
    sites: SiteObservation[T]
    items: int = 2
    words: int = 5


class ChartInstrumentationConformance[
    T: ChartScorer,
    ReadingT: Reading,
](NamedTuple):
    """Inputs for exact chart-readout observation laws.

    Parameters
    ----------
    build : Callable[[], T]
        Return the scorer under test, ready to run.
    leaves : Callable[[int, int], Encoded]
        Return a batch of pooled leaves.
    scope : Callable[[T, frozenset[Site]], InstrumentScope[ReadingT]]
        Open an observe-only scope over selected chart sites.
    items : int
        How many items the laws score.
    words : int
        How many words the longest item holds.
    """

    build: Callable[[], T]
    leaves: Callable[[int, int], Encoded]
    scope: Callable[[T, frozenset[Site]], InstrumentScope[ReadingT]]
    items: int = 2
    words: int = 5


def _same_readouts(left: ChartReadouts, right: ChartReadouts) -> bool:
    """Return whether two records hold bit-identical values, axes and masks."""
    left_values = published(left)
    right_values = published(right)

    if tuple(left_values) != tuple(right_values):
        return False

    for name, first in left_values.items():
        second = right_values[name]

        if first.axes != second.axes or not torch.equal(first.tensor, second.tensor):
            return False

        first_mask = first.mask
        second_mask = second.mask

        if (first_mask is None) != (second_mask is None):
            return False

        if (
            first_mask is not None
            and second_mask is not None
            and not torch.equal(first_mask, second_mask)
        ):
            return False

    return True


class ChartInstrumentationSuite:
    """Reusable fidelity and purity laws for chart instrumentation scopes."""

    pytestmark = pytest.mark.conformance

    def test_a_scope_reads_the_exact_objects_the_chart_pass_returns[
        T: ChartScorer,
        ReadingT: Reading,
    ](
        self,
        chart_instrumentation: ChartInstrumentationConformance[T, ReadingT],
    ) -> None:
        """Check selection order and exact readout identity against one pass."""
        subject = chart_instrumentation.build()
        declared = subject.site_map()
        watched = frozenset(declared)

        with chart_instrumentation.scope(subject, watched) as scope:
            readouts = subject.readouts(
                chart_instrumentation.leaves(
                    chart_instrumentation.items, chart_instrumentation.words
                )
            )

        readings = scope.readings

        if tuple(readings) != tuple(declared):
            _fail(
                f"the site map orders {list(map(repr, declared))}, but the scope "
                f"ordered {list(map(repr, readings))}",
                law="a_scope_reads_the_exact_objects_the_chart_pass_returns",
            )

        filled = published(readouts)

        for site, spec in declared.items():
            name = str(site.key[0])
            expected = filled.get(name)

            reading = readings[site]

            if reading.site != site:
                _fail(
                    f"the reading filed under {site!r} names {reading.site!r}",
                    law="a_scope_reads_the_exact_objects_the_chart_pass_returns",
                )

            if reading.axes != spec.axes:
                _fail(
                    f"{site!r} was read over {reading.axes} and declared over "
                    f"{spec.axes}",
                    law="a_scope_reads_the_exact_objects_the_chart_pass_returns",
                )

            if expected is None:
                if reading.tensor.ndim != spec.rank:
                    _fail(
                        f"{site!r} was read at rank {reading.tensor.ndim} and "
                        f"declared at rank {spec.rank}",
                        law="a_scope_reads_the_exact_objects_the_chart_pass_returns",
                    )

                mask = reading.mask

                if mask is not None and tuple(mask.shape) != tuple(
                    reading.tensor.shape[: mask.ndim]
                ):
                    _fail(
                        f"{site!r} has tensor shape {tuple(reading.tensor.shape)} "
                        f"and incompatible mask shape {tuple(mask.shape)}",
                        law="a_scope_reads_the_exact_objects_the_chart_pass_returns",
                    )

                continue

            if reading.axes != expected.axes:
                _fail(
                    f"{site!r} was read over {reading.axes} and returned over "
                    f"{expected.axes}",
                    law="a_scope_reads_the_exact_objects_the_chart_pass_returns",
                )

            if reading.tensor is not expected.tensor:
                _fail(
                    f"{site!r} was copied or recomputed instead of observing the "
                    "returned tensor",
                    law="a_scope_reads_the_exact_objects_the_chart_pass_returns",
                )

            if reading.mask is not expected.mask:
                _fail(
                    f"{site!r} was given a different mask from the returned value",
                    law="a_scope_reads_the_exact_objects_the_chart_pass_returns",
                )

    def test_observing_leaves_the_chart_pass_bit_identical[
        T: ChartScorer,
        ReadingT: Reading,
    ](
        self,
        chart_instrumentation: ChartInstrumentationConformance[T, ReadingT],
    ) -> None:
        """Check that observation changes no returned chart value or mask."""
        subject = chart_instrumentation.build()
        leaves = chart_instrumentation.leaves(
            chart_instrumentation.items, chart_instrumentation.words
        )
        watched = frozenset(subject.site_map())
        before = subject.readouts(leaves)

        with chart_instrumentation.scope(subject, watched):
            observed = subject.readouts(leaves)

        after = subject.readouts(leaves)

        if not _same_readouts(before, observed):
            _fail(
                "the pass changed while its readouts were observed",
                law="observing_leaves_the_chart_pass_bit_identical",
            )

        if not _same_readouts(before, after):
            _fail(
                "the pass changed after the observation scope closed",
                law="observing_leaves_the_chart_pass_bit_identical",
            )


class ChartScorerSuite(SiteMapSuite):
    pytestmark = pytest.mark.conformance

    def test_split_weights_normalize_per_span[T: ChartScorer](
        self,
        conformance: ChartScorerConformance[T],
    ) -> None:
        """Check that a cell no span occupies is marked and is not zero.

        Parameters
        ----------
        conformance : ChartScorerConformance[T]
            What the laws run against.

        Raises
        ------
        LawViolationError
            If a log-space readout running along the ragged axis has no mask, or
            holds an exact zero at a position its mask excludes.
        """
        subject = conformance.build()
        readouts = subject.readouts(
            conformance.leaves(conformance.items, conformance.words)
        )
        alpha = readouts.alpha

        if alpha is None:
            return

        if alpha.normalized_over != conformance.split_axis:
            _fail(
                f"the split posterior normalizes over {alpha.normalized_over!r} "
                f"where the splits run along {conformance.split_axis!r}",
                law="split_weights_normalize_per_span",
            )

        _check_mass(alpha, conformance.split_axis)

    def test_padding_in_a_log_chart_is_never_an_exact_zero[T: ChartScorer](
        self,
        conformance: ChartScorerConformance[T],
    ) -> None:
        """Check that what a pass filled is exactly what the scorer publishes.

        Parameters
        ----------
        conformance : ChartScorerConformance[T]
            What the laws run against.

        Raises
        ------
        LawViolationError
            If a published readout came back absent, if an unpublished one came back
            present, or if a present one has axes other than the ones it was
            expected to.
        """
        subject = conformance.build()
        readouts = subject.readouts(
            conformance.leaves(conformance.items, conformance.words)
        )

        for name, readout in published(readouts).items():
            if not isinstance(readout, LogDistribution):
                continue

            if conformance.ragged_axis not in readout.axes:
                continue

            if readout.mask is None:
                _fail(
                    f"{name} runs along {conformance.ragged_axis!r} and has "
                    "no mask, so nothing separates a cell from padding",
                    law="padding_in_a_log_chart_is_never_an_exact_zero",
                )

            zeros = readout.tensor.eq(0.0) & _padded(readout.mask, readout.tensor.ndim)

            if bool(zeros.any()):
                _fail(
                    f"{name} holds an exact zero at {int(zeros.sum())} padded "
                    "positions, which read as probability one",
                    law="padding_in_a_log_chart_is_never_an_exact_zero",
                )

    def test_every_declared_readout_is_present_and_the_rest_absent[T: ChartScorer](
        self,
        conformance: ChartScorerConformance[T],
    ) -> None:
        """Check that a pass returns exactly the declared readouts and axes.

        Parameters
        ----------
        conformance : ChartScorerConformance[T]
            The scorer fixture under test.

        Raises
        ------
        LawViolationError
            If readout presence or axes differ from the scorer's declaration.
        """
        subject = conformance.build()
        readouts = subject.readouts(
            conformance.leaves(conformance.items, conformance.words)
        )
        filled = published(readouts)

        if frozenset(filled) != subject.publishes:
            _fail(
                f"the scorer publishes {sorted(subject.publishes)} and the pass "
                f"filled {sorted(filled)}",
                law="every_declared_readout_is_present_and_the_rest_absent",
            )

        for name, readout in filled.items():
            expected = conformance.axes.get(name)

            if expected is not None and readout.axes != expected:
                _fail(
                    f"{name} was expected over {expected} and came back over "
                    f"{readout.axes}",
                    law="every_declared_readout_is_present_and_the_rest_absent",
                )

    def test_a_readout_no_scorer_publishes_is_refused_at_wiring[T: ChartScorer](
        self,
        conformance: ChartScorerConformance[T],
    ) -> None:
        """Check that a loss over an unpublished readout is rejected before a pass.

        Parameters
        ----------
        conformance : ChartScorerConformance[T]
            What the laws run against.

        Raises
        ------
        LawViolationError
            If a loss naming a readout the scorer does not publish is wired, or if one
            naming a readout it does publish is rejected.
        """
        subject = conformance.build()
        absent = sorted(frozenset(conformance.axes) - subject.publishes)

        if not absent:
            _fail(
                f"the scorer publishes every readout the fixture names "
                f"({sorted(subject.publishes)}), so nothing is left to refuse",
                law="a_readout_no_scorer_publishes_is_refused_at_wiring",
            )

        try:
            conformance.wire(subject, absent[0])

        except ValueError:
            pass

        else:
            _fail(
                f"a loss over {absent[0]!r} wired against a scorer publishing "
                f"{sorted(subject.publishes)}",
                law="a_readout_no_scorer_publishes_is_refused_at_wiring",
            )

        for name in sorted(subject.publishes):
            conformance.wire(subject, name)

    @settings(
        max_examples=25,
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.differing_executors,
        ],
    )
    @given(
        items=st.integers(min_value=1, max_value=4),
        words=st.integers(min_value=2, max_value=12),
    )
    def test_split_weights_normalize_over_any_batch[T: ChartScorer](
        self,
        conformance: ChartScorerConformance[T],
        items: int,
        words: int,
    ) -> None:
        """Check split normalization over generated batch dimensions.

        Parameters
        ----------
        conformance : ChartScorerConformance[T]
            The chart scorer and its test adapters.
        items : int
            The generated batch size.
        words : int
            The generated item length.

        Raises
        ------
        LawViolationError
            If a live span's split mass differs from one by more than 1e-5.
        """
        readouts = conformance.build().readouts(conformance.leaves(items, words))
        alpha = readouts.alpha

        if alpha is not None:
            _check_mass(alpha, conformance.split_axis)


def _check_mass(alpha: LogDistribution, split_axis: str) -> None:
    """Raise where a live span's split mass is not one.

    Parameters
    ----------
    alpha : LogDistribution
        The split posterior.
    split_axis : str
        The axis its mass sums over.

    Raises
    ------
    LawViolationError
        If the mass at a live span differs from one by more than 1e-5.
    """
    dim = alpha.axis(split_axis)
    total = alpha.tensor.logsumexp(dim=dim).exp()
    mask = alpha.mask

    if mask is not None:
        live = mask.any(dim=dim) if mask.ndim > dim else mask
        total = total.masked_fill(~live.reshape(total.shape), 1.0)

    worst = float(total.detach().sub(1.0).abs().max()) if total.numel() else 0.0

    if not isclose(worst, 0.0, abs_tol=_NORMALIZATION_ATOL):
        _fail(
            f"a span's splits sum to {1.0 + worst}, which is {worst} away from one",
            law="split_weights_normalize_per_span",
        )


CHART_SCORER_LAWS: Final = (
    ChartScorerSuite.test_split_weights_normalize_per_span,
    ChartScorerSuite.test_padding_in_a_log_chart_is_never_an_exact_zero,
    ChartScorerSuite.test_every_declared_readout_is_present_and_the_rest_absent,
    ChartScorerSuite.test_a_readout_no_scorer_publishes_is_refused_at_wiring,
    ChartScorerSuite.test_the_site_map_matches_what_instrumentation_yields,
)

CHART_INSTRUMENTATION_LAWS: Final = (
    ChartInstrumentationSuite.test_a_scope_reads_the_exact_objects_the_chart_pass_returns,
    ChartInstrumentationSuite.test_observing_leaves_the_chart_pass_bit_identical,
)

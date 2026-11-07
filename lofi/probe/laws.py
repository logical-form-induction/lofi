"""Probing conformance laws."""

from math import isclose, isfinite
from typing import TYPE_CHECKING, Final, NamedTuple

import torch
from hypothesis import given, settings

from lofi._checks import (
    ensure_finite,
    fail_law,
)
from lofi.nn.sites import Role
from lofi.probe._diagnose import DiagnosticContext
from lofi.probe._marginals import SpanAggregate, SpanWeighting
from lofi.probe._selectivity import ControlTask
from lofi.probe._values import Observation, Trace

if TYPE_CHECKING:
    from collections.abc import Mapping

    from hypothesis.strategies import SearchStrategy
    from torch import Tensor

    from lofi.nn.sites import Site
    from lofi.probe._diagnose import DiagnosticResult
    from lofi.probe._ports import Classifier, Diagnostic
    from lofi.probe._selectivity import SpanInventory

__all__ = [
    "CLASSIFIER_LAWS",
    "DIAGNOSTIC_LAWS",
    "SCALINGS",
    "ClassifierFixture",
    "DiagnosticFixture",
    "Readings",
    "a_reading_of_any_magnitude_measures_to_a_number",
    "a_result_carries_the_seed_its_control_labels_were_drawn_under",
    "a_result_reports_selectivity_as_the_gap_between_its_accuracies",
    "an_aggregate_over_spans_is_reported_under_both_weightings",
    "both_tasks_are_fitted_at_the_capacity_the_result_reports",
    "check_classifier_laws",
    "check_diagnostic_laws",
    "every_score_lies_in_the_unit_interval",
    "scaling_the_readings_leaves_every_score_in_range",
]

type Readings = Mapping[Site, Tensor]
"""Specifies what one scope read, keyed by the point each reading came from."""

SCALINGS: Final[tuple[float, ...]] = (1e-6, 1e6)
"""Scale factors applied before repeated law measurements."""

_AGREEMENT: Final[float] = 1e-6
"""How far two means of one quantity may sit apart and still be the same mean."""


class DiagnosticFixture(NamedTuple):
    """Inputs for diagnostic-law checks.

    Parameters
    ----------
    diagnostic : Diagnostic
        The diagnostic under test, built.
    context : DiagnosticContext
        What the components under study published.
    readings : SearchStrategy[Readings]
        A strategy over what a scope read, keyed by the point each came from.
    max_examples : int
        The example budget each law gives Hypothesis.
    """

    diagnostic: Diagnostic
    context: DiagnosticContext
    readings: SearchStrategy[Readings]
    max_examples: int = 25


def _budget(fixture: DiagnosticFixture) -> settings:
    """Return the Hypothesis settings one law runs under.

    Parameters
    ----------
    fixture : DiagnosticFixture
        The diagnostic under test and its strategy.

    Returns
    -------
    settings
        The fixture's own budget, with no per-example deadline.
    """
    return settings(max_examples=fixture.max_examples, deadline=None)


def _trace(fixture: DiagnosticFixture, readings: Readings, scale: float) -> Trace:
    """Return one scope of scaled diagnostic readings.

    Parameters
    ----------
    fixture : DiagnosticFixture
        The diagnostic under test and what its context published.
    readings : Readings
        What was read at each point.
    scale : float
        What every reading is multiplied by first.

    Returns
    -------
    Trace
        A closed trace holding one observation per point, over the axes the context
        declares for it.
    """
    trace = Trace()

    for site, tensor in readings.items():
        axes = fixture.context.spec_at(site).axes

        trace.record(Observation(site, tensor * scale, axes=axes))

    trace.close()

    return trace


def _measured(
    fixture: DiagnosticFixture,
    readings: Readings,
    scale: float,
    *,
    law: str,
    context: DiagnosticContext | None = None,
) -> DiagnosticResult:
    """Return what the diagnostic measured over scaled readings.

    Parameters
    ----------
    fixture : DiagnosticFixture
        The diagnostic under test.
    readings : Readings
        What was read at each point.
    scale : float
        What every reading is multiplied by first.
    law : str
        The law's short name, for the message.
    context : DiagnosticContext | None
        What the diagnostic runs under instead of the fixture's own context, or None
        to run under the fixture's.

    Returns
    -------
    DiagnosticResult
        What it measured.

    Raises
    ------
    LawViolationError
        If measurement raises on an infinity or NaN.
    """
    trace = _trace(fixture, readings, scale)

    try:
        return fixture.diagnostic.run(
            trace, context=fixture.context if context is None else context
        )

    except (ArithmeticError, ValueError) as error:
        detail = (
            f"{fixture.diagnostic.tag} measured readings scaled by {scale:g} "
            f"and raised {error!r}"
        )

        fail_law(detail, law=law)


def _scored(
    result: DiagnosticResult, declared: frozenset[str]
) -> tuple[tuple[str, Site, float], ...]:
    """Return every number a result reports under a declared score name.

    Parameters
    ----------
    result : DiagnosticResult
        What the diagnostic measured.
    declared : frozenset[str]
        The names the diagnostic calls scores.

    Returns
    -------
    tuple[tuple[str, Site, float], ...]
        Each declared name beside the point it was reported at and the number, with
        both halves of an aggregation reported separately.
    """
    single = tuple(
        (measurement.name, measurement.site, measurement.value)
        for measurement in result.measurements
        if measurement.name in declared
    )
    paired = tuple(
        (aggregation.name, aggregation.site, value)
        for aggregation in result.aggregations
        if aggregation.name in declared
        for value in (aggregation.aggregate.weighted, aggregation.aggregate.uniform)
    )

    return single + paired


def every_score_lies_in_the_unit_interval(fixture: DiagnosticFixture) -> None:
    """Check that every number a diagnostic calls a score is one.

    Parameters
    ----------
    fixture : DiagnosticFixture
        The diagnostic under test and its strategy.

    Raises
    ------
    LawViolationError
        When a declared score falls outside the closed unit interval.
    """
    declared = fixture.diagnostic.scores

    @given(fixture.readings)
    @_budget(fixture)
    def law(readings: Readings) -> None:
        result = _measured(fixture, readings, 1.0, law="score range")

        for name, site, value in _scored(result, declared):
            if not 0.0 <= value <= 1.0:
                detail = (
                    f"{fixture.diagnostic.tag} scored {name!r} at {site!r} as {value!r}"
                )

                fail_law(detail, law="score range")

    law()


def scaling_the_readings_leaves_every_score_in_range(
    fixture: DiagnosticFixture,
) -> None:
    """Check that a score stays in range when the readings it came from do not.

    Parameters
    ----------
    fixture : DiagnosticFixture
        The diagnostic under test and its strategy.

    Raises
    ------
    LawViolationError
        When a declared score falls outside the closed unit interval under readings
        scaled by a millionth or by a million.
    """
    declared = fixture.diagnostic.scores

    @given(fixture.readings)
    @_budget(fixture)
    def law(readings: Readings) -> None:
        for scale in SCALINGS:
            result = _measured(fixture, readings, scale, law="scale invariance")

            for name, site, value in _scored(result, declared):
                if not 0.0 <= value <= 1.0:
                    detail = (
                        f"{fixture.diagnostic.tag} scored {name!r} at {site!r} as "
                        f"{value!r} under readings scaled by {scale:g}"
                    )

                    fail_law(detail, law="scale invariance")

    law()


def a_reading_of_any_magnitude_measures_to_a_number(
    fixture: DiagnosticFixture,
) -> None:
    """Check diagnostic outputs across reading magnitudes.

    Parameters
    ----------
    fixture : DiagnosticFixture
        The diagnostic under test and its strategy.

    Raises
    ------
    LawViolationError
        When measuring raises under readings of either extreme, when a number that came
        back is not finite, or when a result is reported under a tag other than the
        diagnostic's own.
    """
    scales = (1.0, *SCALINGS)

    @given(fixture.readings)
    @_budget(fixture)
    def law(readings: Readings) -> None:
        for scale in scales:
            result = _measured(fixture, readings, scale, law="finite measurement")

            if result.tag != fixture.diagnostic.tag:
                detail = (
                    f"{fixture.diagnostic.tag} reported its result under "
                    f"{result.tag!r} at a scaling of {scale:g}"
                )

                fail_law(detail, law="finite measurement")

            for measurement in result.measurements:
                if not isfinite(measurement.value):
                    detail = (
                        f"{fixture.diagnostic.tag} measured {measurement.name!r} "
                        f"at {measurement.site!r} as {measurement.value!r} under "
                        f"readings scaled by {scale:g}"
                    )

                    fail_law(detail, law="finite measurement")

    law()


def _reweighted(context: DiagnosticContext, weights: Tensor) -> DiagnosticContext:
    """Return a context publishing the same points under another weighting.

    Parameters
    ----------
    context : DiagnosticContext
        The context whose points and weighting axes are kept.
    weights : Tensor
        The weights to hold instead, of the weighting's own shape.

    Returns
    -------
    DiagnosticContext
        The same points, weighted by the given tensor.

    Raises
    ------
    ValueError
        If the context holds no weighting to take the axes from.
    """
    weighting = context.weighting

    if weighting is None:
        msg = "a context without a weighting cannot be reweighted"
        raise ValueError(msg)

    return DiagnosticContext(
        context.published,
        groupings=context.groupings,
        relations=context.relations,
        span_index=context.span_index,
        weighting=SpanWeighting(weights, axes=weighting.axes),
    )


def _weightings(weighting: SpanWeighting) -> tuple[Tensor, Tensor, Tensor]:
    """Return two different non-uniform weightings and the uniform one.

    Parameters
    ----------
    weighting : SpanWeighting
        The weighting whose shape and dtype the three take.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        A ramp rising along the flattened positions, the same ramp falling, and all
        ones, each of the weighting's shape.
    """
    shape = tuple(weighting.tensor.shape)
    rising = torch.arange(1, weighting.tensor.numel() + 1, dtype=torch.float64)

    return (
        rising.reshape(shape),
        rising.flip(0).reshape(shape),
        torch.ones(shape, dtype=torch.float64),
    )


def _marginal_readings(
    context: DiagnosticContext, readings: Readings, *, law: str
) -> tuple[Readings, Readings, Readings]:
    """Return readings under rising, falling and uniform constituent marginals.

    Parameters
    ----------
    context : DiagnosticContext
        The context declaring the marginal sites.
    readings : Readings
        One generated reading at each observed site.
    law : str
        The law's short name, for a missing-reading message.

    Returns
    -------
    tuple[Readings, Readings, Readings]
        Three copies differing only at constituent-marginal sites.

    Raises
    ------
    LawViolationError
        If a required marginal site has no generated reading or holds no position.
    """
    varied = (dict(readings), dict(readings), dict(readings))
    sites = context.sites_for(Role.CONSTITUENT_MARGINAL)

    for site in sites:
        tensor = readings.get(site)

        if tensor is None:
            detail = f"no generated reading supplies required marginal site {site!r}"
            fail_law(detail, law=law)

        if tensor.numel() == 0:
            detail = f"the generated marginal reading at {site!r} holds no position"
            fail_law(detail, law=law)

        rising = torch.arange(
            1,
            tensor.numel() + 1,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        rising = rising / rising[-1]
        replacements = (
            rising.reshape(tensor.shape),
            rising.flip(0).reshape(tensor.shape),
            torch.ones_like(tensor),
        )

        for changed, replacement in zip(varied, replacements, strict=True):
            changed[site] = replacement

    return varied


def _weighted_results(
    fixture: DiagnosticFixture, readings: Readings, *, law: str
) -> tuple[DiagnosticResult, DiagnosticResult, DiagnosticResult] | None:
    """Return results under three weightings, or None where no weighting exists.

    Parameters
    ----------
    fixture : DiagnosticFixture
        The diagnostic under test and its context.
    readings : Readings
        One generated reading at each observed site.
    law : str
        The law's short name.

    Returns
    -------
    tuple[DiagnosticResult, DiagnosticResult, DiagnosticResult] | None
        Results under rising, falling and uniform weighting, or None when the
        diagnostic has neither a marginal observation nor a context weighting.
    """
    if Role.CONSTITUENT_MARGINAL in fixture.diagnostic.requires:
        rising, falling, uniform = _marginal_readings(
            fixture.context, readings, law=law
        )

        return (
            _measured(fixture, rising, 1.0, law=law),
            _measured(fixture, falling, 1.0, law=law),
            _measured(fixture, uniform, 1.0, law=law),
        )

    weighting = fixture.context.weighting

    if weighting is None:
        return None

    rising, falling, uniform = _weightings(weighting)

    return (
        _measured(
            fixture,
            readings,
            1.0,
            law=law,
            context=_reweighted(fixture.context, rising),
        ),
        _measured(
            fixture,
            readings,
            1.0,
            law=law,
            context=_reweighted(fixture.context, falling),
        ),
        _measured(
            fixture,
            readings,
            1.0,
            law=law,
            context=_reweighted(fixture.context, uniform),
        ),
    )


def _entry(result: DiagnosticResult, site: Site, name: str, *, law: str) -> float:
    """Return one single number a result reports, failing the law where it is absent.

    Parameters
    ----------
    result : DiagnosticResult
        What the diagnostic measured.
    site : Site
        The point the number was computed from.
    name : str
        What the number is.
    law : str
        The law's short name, for the message.

    Raises
    ------
    LawViolationError
        If the result holds no such measurement, since what a diagnostic reports then
        depends on the weighting it ran under.
    """
    try:
        return result.value(site, name)

    except KeyError:
        detail = f"{result.tag} measured {name!r} at {site!r} under one weighting only"

        fail_law(detail, law=law)


def _paired(
    result: DiagnosticResult, site: Site, name: str, *, law: str
) -> SpanAggregate:
    """Return one aggregate a result reports, failing the law where it is absent.

    Parameters
    ----------
    result : DiagnosticResult
        What the diagnostic measured.
    site : Site
        The point the quantity was computed from.
    name : str
        What the quantity is.
    law : str
        The law's short name, for the message.

    Raises
    ------
    LawViolationError
        If the result holds no such aggregation, since what a diagnostic reports then
        depends on the weighting it ran under.
    """
    try:
        return result.aggregate(site, name)

    except KeyError:
        detail = (
            f"{result.tag} aggregated {name!r} at {site!r} under one weighting only"
        )

        fail_law(detail, law=law)


def an_aggregate_over_spans_is_reported_under_both_weightings(
    fixture: DiagnosticFixture,
) -> None:
    """Check that what is aggregated over spans is aggregated both ways, genuinely.

    Parameters
    ----------
    fixture : DiagnosticFixture
        The diagnostic under test and its strategy. Its context's weighting fixes the
        axes and shape the law's own weightings take.

    Raises
    ------
    LawViolationError
        When a diagnostic aggregates over spans although its context holds no
        weighting; when an aggregation's uniform mean or span count moves with the
        weighting, or its weighted mean under the uniform weighting is not the
        uniform mean; when a single measurement moves with the weighting, which
        makes it a weighted aggregate reported without its uniform half; or when an
        aggregation's weighted mean never moves with the weighting over every
        example, which makes it the uniform mean reported twice.
    """
    law_name = "both weightings"
    moved: dict[tuple[Site, str], bool] = {}

    @given(fixture.readings)
    @_budget(fixture)
    def law(readings: Readings) -> None:
        results = _weighted_results(fixture, readings, law=law_name)

        if results is None:
            result = _measured(fixture, readings, 1.0, law=law_name)

            if result.aggregations:
                first = result.aggregations[0]
                detail = (
                    f"{fixture.diagnostic.tag} aggregated {first.name!r} over spans "
                    f"at {first.site!r} under no weighting"
                )

                fail_law(detail, law=law_name)

            return

        rising, falling, uniform = results

        for measurement in rising.measurements:
            others = tuple(
                _entry(other, measurement.site, measurement.name, law=law_name)
                for other in (falling, uniform)
            )

            if not all(isclose(measurement.value, value) for value in others):
                detail = (
                    f"{fixture.diagnostic.tag} measured {measurement.name!r} at "
                    f"{measurement.site!r} as a single number that moves with the "
                    "weighting, so it is a weighted aggregate without its uniform half"
                )

                fail_law(detail, law=law_name)

        for aggregation in rising.aggregations:
            key = (aggregation.site, aggregation.name)
            under_rising = aggregation.aggregate
            under_falling = _paired(falling, *key, law=law_name)
            under_uniform = _paired(uniform, *key, law=law_name)
            steady = (
                under_rising.count == under_falling.count == under_uniform.count
                and isclose(under_rising.uniform, under_falling.uniform)
                and isclose(under_rising.uniform, under_uniform.uniform)
            )

            if not steady:
                detail = (
                    f"{fixture.diagnostic.tag} reported the uniform half of "
                    f"{aggregation.name!r} at {aggregation.site!r} as moving with "
                    "the weighting"
                )

                fail_law(detail, law=law_name)

            if not isclose(
                under_uniform.weighted, under_uniform.uniform, rel_tol=_AGREEMENT
            ):
                detail = (
                    f"{fixture.diagnostic.tag} weighted {aggregation.name!r} at "
                    f"{aggregation.site!r} under equal weights as "
                    f"{under_uniform.weighted!r} against a uniform mean of "
                    f"{under_uniform.uniform!r}"
                )

                fail_law(detail, law=law_name)

            moved[key] = moved.get(key, False) or not isclose(
                under_rising.weighted, under_falling.weighted, rel_tol=_AGREEMENT
            )

    law()

    for (site, name), witnessed in moved.items():
        if not witnessed:
            detail = (
                f"{fixture.diagnostic.tag} weighted {name!r} at {site!r} the same "
                "under every weighting over every example, so the weighted half is "
                "the uniform mean reported twice"
            )

            fail_law(detail, law=law_name)


DIAGNOSTIC_LAWS: Final = (
    every_score_lies_in_the_unit_interval,
    scaling_the_readings_leaves_every_score_in_range,
    a_reading_of_any_magnitude_measures_to_a_number,
    an_aggregate_over_spans_is_reported_under_both_weightings,
)
"""Diagnostic laws in conformance order."""


def check_diagnostic_laws(fixture: DiagnosticFixture) -> None:
    """Run every diagnostic law against one diagnostic.

    Parameters
    ----------
    fixture : DiagnosticFixture
        The diagnostic under test and its strategy.

    Raises
    ------
    LawViolationError
        On the first law the diagnostic fails.
    """
    for law in DIAGNOSTIC_LAWS:
        law(fixture)


class ClassifierFixture(NamedTuple):
    """Inputs for classifier-probe law checks.

    Parameters
    ----------
    probe : Classifier
        The probe under test, holding the capacity and the control task it runs.
    fitting : SpanInventory
        The spans the probe fits on.
    held_out : SpanInventory
        The spans it is scored on, held out from the ones fitting saw.
    """

    probe: Classifier
    fitting: SpanInventory
    held_out: SpanInventory


def _number(report: object, name: str, *, law: str) -> float:
    """Return one number a probe report carries, failing the law where it has none.

    Parameters
    ----------
    report : object
        What the probe returned.
    name : str
        The field to read.
    law : str
        The law's short name, for the message.

    Returns
    -------
    float
        The number the field holds.

    Raises
    ------
    LawViolationError
        If the report carries no such field, or carries one that is not a finite
        number, so that what the probe found cannot be read off it.
    """
    try:
        return ensure_finite(f"a probe report's {name}", getattr(report, name))

    except AttributeError, TypeError, ValueError:
        detail = f"{type(report).__name__} reports no {name} a law can read"

        fail_law(detail, law=law)


def a_result_reports_selectivity_as_the_gap_between_its_accuracies(
    fixture: ClassifierFixture,
) -> None:
    """Check that a probe reports selectivity, and reports it as the gap.

    Parameters
    ----------
    fixture : ClassifierFixture
        The probe under test and the spans it runs on.

    Raises
    ------
    LawViolationError
        When a result carries no selectivity, when either accuracy is not a
        proportion, or when the selectivity is not the real-task accuracy less the
        control-task accuracy.
    """
    law = "selectivity"
    report = fixture.probe.run(fixture.fitting, fixture.held_out)
    reported = _number(report, "selectivity", law=law)
    accuracy = _number(report, "accuracy", law=law)
    control = _number(report, "control_accuracy", law=law)

    for name, value in (("accuracy", accuracy), ("control_accuracy", control)):
        if not 0.0 <= value <= 1.0:
            detail = f"a probe reported {name} as {value!r}, which is no proportion"

            fail_law(detail, law=law)

    if not isclose(reported, accuracy - control, abs_tol=_AGREEMENT):
        detail = (
            f"a probe reported a selectivity of {reported!r} against an accuracy of "
            f"{accuracy!r} on the real task and {control!r} on the control task"
        )

        fail_law(detail, law=law)


def a_result_carries_the_seed_its_control_labels_were_drawn_under(
    fixture: ClassifierFixture,
) -> None:
    """Check that the control-task seed is reported and reaches the control task only.

    Parameters
    ----------
    fixture : ClassifierFixture
        The probe under test and the spans it runs on.

    Raises
    ------
    LawViolationError
        When a result carries a seed other than the one its control task draws
        under, or when the real task's accuracy moves with that seed, which is a
        probe whose real fit saw the control labels.
    """
    law = "control seed"
    probe = fixture.probe
    drawn = probe.control.seed
    report = probe.run(fixture.fitting, fixture.held_out)

    if getattr(report, "control_seed", None) != drawn:
        detail = (
            f"a probe drawing control labels under {drawn!r} reported the seed "
            f"{getattr(report, 'control_seed', None)!r}"
        )

        fail_law(detail, law=law)

    again = probe.under(ControlTask(seed=drawn + 1)).run(
        fixture.fitting, fixture.held_out
    )

    if getattr(again, "control_seed", None) != drawn + 1:
        detail = (
            f"a probe drawing control labels under {drawn + 1!r} reported the seed "
            f"{getattr(again, 'control_seed', None)!r}"
        )

        fail_law(detail, law=law)

    moved = _number(again, "accuracy", law=law)
    first = _number(report, "accuracy", law=law)

    if not isclose(first, moved, abs_tol=_AGREEMENT):
        detail = (
            f"a probe reached {first!r} on the real task under control seed "
            f"{drawn!r} and {moved!r} under {drawn + 1!r}, so its real fit reads "
            "the control labels"
        )

        fail_law(detail, law=law)


def both_tasks_are_fitted_at_the_capacity_the_result_reports(
    fixture: ClassifierFixture,
) -> None:
    """Check that the real task and the control task are fitted alike.

    Parameters
    ----------
    fixture : ClassifierFixture
        The probe under test and the spans it runs on.

    Raises
    ------
    LawViolationError
        When a result reports a capacity or a weight count other than the declared
        one, or when a probe run on spans the control task has already labeled
        reports a selectivity away from zero, which is two fits of unequal capacity
        on one set of labels.
    """
    law = "matched capacity"
    probe = fixture.probe
    declared = probe.capacity
    report = probe.run(fixture.fitting, fixture.held_out)
    wanted = declared.parameters(fixture.fitting.features, fixture.fitting.classes)

    if getattr(report, "capacity", None) != declared:
        detail = (
            f"a probe fitting at {declared!r} reported the capacity "
            f"{getattr(report, 'capacity', None)!r}"
        )

        fail_law(detail, law=law)

    if getattr(report, "parameters", None) != wanted:
        detail = (
            f"a probe fitting {wanted!r} weights at {declared!r} reported "
            f"{getattr(report, 'parameters', None)!r} of them"
        )

        fail_law(detail, law=law)

    controlled = probe.run(
        probe.control.over(fixture.fitting), probe.control.over(fixture.held_out)
    )
    gap = _number(controlled, "selectivity", law=law)

    if not isclose(gap, 0.0, abs_tol=_AGREEMENT):
        detail = (
            f"a probe run on the control task's own labels reported a selectivity "
            f"of {gap!r}, so the two tasks are not fitted at one capacity"
        )

        fail_law(detail, law=law)


CLASSIFIER_LAWS: Final = (
    a_result_reports_selectivity_as_the_gap_between_its_accuracies,
    a_result_carries_the_seed_its_control_labels_were_drawn_under,
    both_tasks_are_fitted_at_the_capacity_the_result_reports,
)
"""Classifier-probe laws in conformance order."""


def check_classifier_laws(fixture: ClassifierFixture) -> None:
    """Run every classifier-probe law against one probe.

    Parameters
    ----------
    fixture : ClassifierFixture
        The probe under test and the spans it runs on.

    Raises
    ------
    LawViolationError
        On the first law the probe fails.
    """
    for law in CLASSIFIER_LAWS:
        law(fixture)

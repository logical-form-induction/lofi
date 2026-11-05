from types import MappingProxyType
from typing import ClassVar, Literal

import didactic.api as dx
import pytest

from lofi import judgment
from lofi.judgment import (
    _acquisition,
    _active,
    _attributes,
    _design,
    _items,
    _nuts,
    _ports,
    _predicted,
    _program,
    _results,
    _sources,
    _spec,
    _values,
    ablation,
    agreement,
    backends,
    bead,
    bounded,
    calibration,
    choice,
    compiler,
    crossed,
    cumulative,
    dynamic_range,
    effects,
    fitting,
    grouped,
    guides,
    heldout,
    identification,
    linear,
    linking,
    marginal,
    mass,
    metric,
    morcela,
    objective,
    ordinal,
    ordinal_bounds,
    pooled,
    prior_predictive,
    rank,
    ratings,
    relations,
    simulate,
    splits,
    targets,
)

MODULES = (
    _acquisition,
    _active,
    _attributes,
    _design,
    _items,
    _nuts,
    _ports,
    _predicted,
    _program,
    _results,
    _sources,
    _spec,
    _values,
    ablation,
    agreement,
    backends,
    bead,
    bounded,
    calibration,
    choice,
    compiler,
    crossed,
    cumulative,
    dynamic_range,
    effects,
    fitting,
    guides,
    grouped,
    heldout,
    identification,
    linking,
    marginal,
    linear,
    mass,
    metric,
    morcela,
    objective,
    ordinal,
    ordinal_bounds,
    pooled,
    prior_predictive,
    rank,
    ratings,
    relations,
    simulate,
    splits,
    targets,
)


class Term(judgment.Predictor, extra="forbid"):
    kind: Literal["init_readout"] = "init_readout"
    readout: str


def test_the_package_re_exports_the_family_its_priors_and_the_estimator() -> None:
    published = set()

    for module in MODULES:
        published |= set(module.__all__)

    assert set(judgment.__all__) == published

    for module in MODULES:
        for name in module.__all__:
            assert getattr(judgment, name) is getattr(module, name)


def test_the_package_exposes_the_calibration_treatment() -> None:
    assert judgment.CalibrationReplicate is calibration.CalibrationReplicate
    assert judgment.CalibrationReport is calibration.CalibrationReport
    assert judgment.CoverageSummary is calibration.CoverageSummary
    assert judgment.CutpointRecovery is calibration.CutpointRecovery
    assert judgment.SbcBlockSummary is calibration.SbcBlockSummary
    assert judgment.calibration_report is calibration.calibration_report
    assert (
        judgment.prior_predictive_calibration
        is prior_predictive.prior_predictive_calibration
    )
    assert judgment.CalibrationCampaign is prior_predictive.CalibrationCampaign
    assert judgment.NutsDiagnostics is prior_predictive.NutsDiagnostics
    assert judgment.ReplicateRecord is prior_predictive.ReplicateRecord


def test_a_family_built_from_the_package_namespace_emits_its_program() -> None:
    family = judgment.CumulativeFamily(5, base=judgment.DEFAULT_BASE_PRIOR)
    steps = family.steps(predictor="eta", response="y", plate="obs")

    assert family.categories == 5
    assert family.thresholds == 4
    assert family.increment is judgment.DEFAULT_INCREMENT_PRIOR
    assert any(
        isinstance(step, judgment.Bind) and step.site == family.cutpoint_site
        for step in steps
    )


def test_a_specification_built_from_the_package_namespace_refuses_a_dead_prior() -> (
    None
):
    family = judgment.Cumulative(discrimination=True)
    priors = judgment.PriorBlock(thresholds=judgment.ThresholdPriors())

    assert family.draws_discrimination
    assert priors.thresholds is not None

    with pytest.raises(dx.ValidationError):
        judgment.ResponseSpec(
            family=family,
            levels=7,
            objective=judgment.ConditionalLikelihood(),
            predictor=(Term(readout="p_gram"),),
            identification=judgment.FixedCutpointVariance(value=1.0),
            priors=priors,
        )


def test_a_ceiling_computed_from_the_package_namespace_carries_its_interval() -> None:
    items = [[float(index % 7), float((index * 3) % 7)] for index in range(60)]
    ceiling = judgment.split_half_ceiling(items)
    lower, upper = ceiling.interval

    assert isinstance(ceiling, judgment.AgreementCeiling)
    assert lower <= ceiling.value <= upper
    assert ceiling.items == 60
    assert ceiling.responses == 120


class BoundRows:
    """A source of ordinal rows for the public bound functions."""

    response_levels = 3
    factors: tuple[str, ...] = ()
    skips: ClassVar[MappingProxyType[str, int]] = MappingProxyType({})

    def __init__(self, responses: tuple[tuple[str, str, int], ...]) -> None:
        self._rows = tuple(
            judgment.JudgmentRow(
                participant,
                item,
                response,
                judgment.RowColumns({}, {}),
            )
            for participant, item, response in responses
        )

    def rows(self) -> tuple[judgment.JudgmentRow, ...]:
        """Return the declared rows."""
        return self._rows


def test_ordinal_bounds_construct_and_score_through_the_package_namespace() -> None:
    fitting = BoundRows(
        (
            ("fit-0", "fit-a", 0),
            ("fit-1", "fit-b", 1),
            ("fit-2", "fit-c", 1),
        )
    )
    scored = BoundRows(
        (
            ("score-0", "item", 0),
            ("score-1", "item", 1),
        )
    )

    majority = judgment.majority_class_bound(fitting, scored)
    oracle = judgment.modal_response_oracle(scored, scored)

    assert isinstance(majority, judgment.OrdinalBoundResult)
    assert isinstance(majority.predictions[0], judgment.OrdinalBoundPrediction)
    assert majority.kind is judgment.OrdinalBoundKind.MAJORITY_CLASS
    assert majority.predictions[0].distribution == (1 / 3, 2 / 3, 0.0)
    assert majority.predictions[0].predicted_level == 1
    assert majority.accuracy == 0.5
    assert oracle.kind is judgment.OrdinalBoundKind.MODAL_RESPONSE_ORACLE
    assert oracle.predictions[0].distribution == (0.5, 0.5, 0.0)
    assert oracle.predictions[0].predicted_level == 0

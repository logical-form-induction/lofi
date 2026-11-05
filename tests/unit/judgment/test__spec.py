import json
import math
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, cast

import didactic.api as dx
import pytest

from lofi.judgment._design import POSITION
from lofi.judgment._spec import (
    AgreementBin,
    AuxiliaryParameter,
    Bounded,
    BoundedResponseSpec,
    Choice,
    ColumnInventory,
    ColumnLabels,
    ColumnLevel,
    ColumnLift,
    ComparisonSpec,
    ConditionalLikelihood,
    ConstraintCost,
    CorrelatedRandomSlope,
    Cumulative,
    DesignProvenance,
    DynamicRangeCriterion,
    DynamicRangeReport,
    Elicitation,
    ElicitationMethod,
    FactorTerm,
    FitArtifact,
    FitDiagnostics,
    FitSchedule,
    FitSummary,
    FixedCutpointVariance,
    FixedDiscrimination,
    IdentityLinkOnLogit,
    InferenceMethod,
    InteractionTerm,
    JointTypeTensor,
    JudgmentFitCalibration,
    JudgmentFitManifest,
    JudgmentFitSpec,
    LexicalPossession,
    LexicalPrimitiveInventory,
    LikelihoodObjective,
    LinkingSpec,
    LinkTerm,
    MapInference,
    Measurement,
    MemorizedCells,
    Metric,
    ModelComparisonCriterion,
    ModelComparisonSpec,
    ModelWeighting,
    NutsInference,
    OffsetTerm,
    ParameterFormula,
    PinnedPrefix,
    PossessionAndClasses,
    PosteriorHandle,
    Predictor,
    PriorBlock,
    PriorSpec,
    RandomIntercept,
    RandomSlope,
    RankConsistent,
    RankConsistentScheme,
    ResponseFamily,
    ResponsePrediction,
    ResponseSpec,
    RowClasses,
    ScaleIdentification,
    ScaleNormalization,
    ScaleProfile,
    TableAblationSpec,
    TableColumns,
    TableRows,
    ThresholdPriors,
    TruncatedMarginalLikelihood,
    VariationalInference,
    decode_fit_manifest,
    encode_fit_manifest,
    spec_digest,
)
from lofi.judgment.cumulative import DEFAULT_BASE_PRIOR, DEFAULT_INCREMENT_PRIOR
from lofi.spec import HeldOutParticipantPolicy, SeedDerivationRecord, point_root
from lofi.spec.optimizers.adam import AdamOptimizerSpec

if TYPE_CHECKING:
    from lofi._types import JsonValue

DECLARED_MODELS = (
    PriorSpec,
    ThresholdPriors,
    PriorBlock,
    LikelihoodObjective,
    ConditionalLikelihood,
    TruncatedMarginalLikelihood,
    ResponseFamily,
    Cumulative,
    Bounded,
    RankConsistent,
    Metric,
    LinkingSpec,
    Predictor,
    LinkTerm,
    FactorTerm,
    InteractionTerm,
    CorrelatedRandomSlope,
    RandomIntercept,
    RandomSlope,
    OffsetTerm,
    ParameterFormula,
    PinnedPrefix,
    ScaleIdentification,
    FixedCutpointVariance,
    IdentityLinkOnLogit,
    FixedDiscrimination,
    ResponseSpec,
    BoundedResponseSpec,
    ComparisonSpec,
    ModelComparisonSpec,
    FitSchedule,
    FitArtifact,
    Measurement,
    ScaleProfile,
    ConstraintCost,
    ResponsePrediction,
    PosteriorHandle,
    FitDiagnostics,
    FitSummary,
    DynamicRangeCriterion,
    DynamicRangeReport,
    ColumnLevel,
    ColumnInventory,
    LexicalPrimitiveInventory,
    ColumnLift,
    MemorizedCells,
    ColumnLabels,
    JointTypeTensor,
    LexicalPossession,
    RowClasses,
    PossessionAndClasses,
    TableAblationSpec,
    JudgmentFitCalibration,
    NutsInference,
    MapInference,
    VariationalInference,
    Elicitation,
    JudgmentFitSpec,
    JudgmentFitManifest,
)


def bounded_link(family: str = "logit") -> LinkTerm:
    return LinkTerm(
        readout="p_gram",
        linking=LinkingSpec(
            family=family,
            columns={"complement": "p_not_gram"},
        ),
    )


def test_a_fixed_bounded_response_declares_no_ordinal_levels_or_dead_prior() -> None:
    declared = BoundedResponseSpec(
        family=Bounded(scale=0.2),
        predictor=(bounded_link(),),
        identification=IdentityLinkOnLogit(),
    )

    assert declared.family.scale == 0.2
    assert declared.normalization is ScaleNormalization.UNIT_INTERVAL
    assert declared.priors.scale is None
    assert "levels" not in declared.model_dump()


def test_a_drawn_bounded_scale_has_one_strictly_positive_prior() -> None:
    prior = PriorSpec(distribution="LogNormal", parameters=(-1.0, 0.5))
    declared = BoundedResponseSpec(
        family=Bounded(scale=None),
        predictor=(bounded_link(),),
        identification=IdentityLinkOnLogit(),
        priors=PriorBlock(scale=prior),
    )

    assert declared.family.draws_scale
    assert declared.priors.scale == prior


@pytest.mark.parametrize("scale", [0.0, -0.1, float("nan"), float("inf")])
def test_a_bounded_family_refuses_a_bad_fixed_scale(scale: float) -> None:
    with pytest.raises(dx.ValidationError, match="finite positive"):
        Bounded(scale=scale)


@pytest.mark.parametrize(
    ("prior", "message"),
    [
        (None, "LogNormal"),
        (PriorSpec(distribution="HalfNormal", parameters=(1.0,)), "LogNormal"),
        (
            PriorSpec(distribution="LogNormal", parameters=(0.0, 0.0)),
            "positive spread",
        ),
    ],
)
def test_a_drawn_bounded_scale_refuses_a_nonpositive_prior(
    prior: PriorSpec | None,
    message: str,
) -> None:
    with pytest.raises(dx.ValidationError, match=message):
        BoundedResponseSpec(
            family=Bounded(scale=None),
            predictor=(bounded_link(),),
            identification=IdentityLinkOnLogit(),
            priors=PriorBlock(scale=prior),
        )


def test_a_bounded_response_refuses_an_unsupported_link_by_name() -> None:
    with pytest.raises(dx.ValidationError, match="'probability'"):
        BoundedResponseSpec(
            family=Bounded(scale=0.2),
            predictor=(bounded_link("probability"),),
            identification=IdentityLinkOnLogit(),
        )


def test_a_bounded_response_requires_identity_on_logit() -> None:
    with pytest.raises(dx.ValidationError, match="identity_link_on_logit"):
        BoundedResponseSpec(
            family=Bounded(scale=0.2),
            predictor=(bounded_link(),),
            identification=FixedDiscrimination(value=1.0),
        )


def test_a_bounded_response_refuses_an_ordinal_declaration() -> None:
    with pytest.raises(dx.ValidationError, match="BoundedResponseSpec"):
        ResponseSpec(
            family=Bounded(scale=0.2),
            levels=2,
            objective=ConditionalLikelihood(),
            predictor=(bounded_link(),),
            identification=IdentityLinkOnLogit(),
        )


LADDER: Final = (
    Path(__file__).resolve().parents[3] / "tests" / "data" / "judgment" / "ladder"
)
PACKAGE: Final = Path(__file__).resolve().parents[3] / "lofi"


LOGIT = LinkingSpec(family="logit", columns={"complement": "p_gram_complement"})
PROBABILITY = LinkingSpec(family="probability")
LINEAR = LinkingSpec(family="linear")
MORCELA = LinkingSpec(
    family="morcela",
    columns={"length": "length", "frequency": "unigram_log_probability"},
)
P_GRAM = LinkTerm(readout="p_gram", linking=LOGIT)
ROOT_ENTROPY = LinkTerm(readout="root_entropy", linking=LINEAR)
PAIR = ("p_gram", "p_gram_complement")
LM = LinkTerm(readout="lm_log_probability", linking=MORCELA)
VERB = FactorTerm(name="verb")
FRAME = FactorTerm(name="frame")
VERB_BY_FRAME = InteractionTerm(left=VERB, right=FRAME)
PARTICIPANT = RandomIntercept(group="participant")
SLOPE = RandomSlope(group="participant", term=P_GRAM)
CORRELATED_SLOPE = CorrelatedRandomSlope(group="participant", term=P_GRAM)
LENGTH = OffsetTerm(readout="length")
TERMS = (
    P_GRAM,
    VERB,
    VERB_BY_FRAME,
    PARTICIPANT,
    SLOPE,
    CORRELATED_SLOPE,
    LENGTH,
)
CUMULATIVE = Cumulative()
PINNED = FixedDiscrimination(value=1.0)
IDENTITY = IdentityLinkOnLogit()
DISCRIMINATION_PRIOR = PriorSpec(distribution="LogNormal", parameters=(0.0, 0.5))
THRESHOLD_PRIORS = PriorBlock(thresholds=ThresholdPriors())
NO_PRIORS = PriorBlock()
FAMILIES_AND_WHAT_THEY_DRAW = (
    (Cumulative(), True, False),
    (Cumulative(discrimination=True), True, True),
    (RankConsistent(scheme=RankConsistentScheme.CORAL), False, False),
    (RankConsistent(scheme=RankConsistentScheme.CORN), False, False),
    (Metric(), False, False),
)


class ReadoutTerm(Predictor, extra="forbid"):
    kind: Literal["test_readout"] = "test_readout"
    readout: str


def spec_for(
    family: ResponseFamily,
    *,
    levels: int = 7,
    predictor: tuple[Predictor, ...] = (PARTICIPANT, P_GRAM),
    identification: ScaleIdentification = PINNED,
    priors: PriorBlock = NO_PRIORS,
    normalization: ScaleNormalization = ScaleNormalization.IDENTITY,
) -> ResponseSpec:
    return ResponseSpec(
        family=family,
        levels=levels,
        objective=ConditionalLikelihood(),
        predictor=predictor,
        identification=identification,
        priors=priors,
        normalization=normalization,
    )


def cumulative_spec() -> ResponseSpec:
    return spec_for(Cumulative(), priors=THRESHOLD_PRIORS)


def comparison_for(
    held_out_unit: str,
    *,
    response: ResponseSpec | BoundedResponseSpec | None = None,
    readout_units: dict[str, str] | None = None,
    splitter: str = "kfold",
    folds: int | None = 3,
) -> ComparisonSpec:
    declared = (
        response if response is not None else spec_for(Metric(), predictor=(P_GRAM,))
    )
    grains = (
        readout_units
        if readout_units is not None
        else dict.fromkeys(declared.readouts, "item")
    )

    return ComparisonSpec(
        held_out_unit=held_out_unit,
        readout_units=grains,
        splitter=splitter,
        folds=folds,
        seed=17,
        coverage_threshold=0.25,
        unseen_participant=HeldOutParticipantPolicy.MARGINAL,
        models=ModelComparisonSpec(
            candidates=("declared", "twin"),
            candidate="declared",
            criterion=ModelComparisonCriterion.PSIS_LOO,
            weighting=ModelWeighting.STACKING,
            pareto_fraction=0.1,
            pareto_threshold=0.7,
        ),
    )


def fit_with_comparison(
    response: ResponseSpec | BoundedResponseSpec,
    comparison: ComparisonSpec,
) -> JudgmentFitSpec:
    executable = response

    if isinstance(response, ResponseSpec) and not response.family.programmed:
        executable = response.with_(family=Cumulative(), priors=THRESHOLD_PRIORS)

    return JudgmentFitSpec(
        dataset="fixture",
        response=executable,
        inference=NutsInference(warmup=10, retained=20, chains=2),
        elicitation=Elicitation(
            method=ElicitationMethod.PARTICIPANTS,
            version="1",
        ),
        calibration=None,
        comparison=comparison,
        dynamic_range=None,
        ablation=(),
        interannotator=None,
    )


def test_a_comparison_names_any_design_factor_and_every_execution_setting() -> None:
    comparison = comparison_for("list")

    assert comparison.held_out_unit == "list"
    assert comparison.readout_units == {
        "p_gram": "item",
        "p_gram_complement": "item",
    }
    assert comparison.splitter == "kfold"
    assert comparison.folds == 3
    assert comparison.seed == 17
    assert comparison.models.weighting is ModelWeighting.STACKING


@pytest.mark.parametrize(
    "held_out_unit",
    ["response", "participant", "verb", "frame", "list"],
)
def test_a_non_item_holdout_refuses_the_two_competing_item_terms_by_name(
    held_out_unit: str,
) -> None:
    response = spec_for(Metric(), predictor=(RandomIntercept(group="item"), P_GRAM))

    with pytest.raises(
        dx.ValidationError,
        match=r"RandomIntercept\(kind='random_intercept'.*LinkTerm\(kind='link'",
    ):
        fit_with_comparison(response, comparison_for(held_out_unit, response=response))


def test_item_holdout_admits_the_two_item_terms_together() -> None:
    response = spec_for(Metric(), predictor=(RandomIntercept(group="item"), P_GRAM))

    comparison = comparison_for("item", response=response)

    assert fit_with_comparison(response, comparison).comparison == comparison


@pytest.mark.parametrize(
    "response",
    [
        spec_for(Metric(), predictor=(P_GRAM,)),
        spec_for(Metric(), predictor=(RandomIntercept(group="item"),)),
    ],
)
@pytest.mark.parametrize(
    "held_out_unit", ["response", "item", "participant", "verb", "frame", "list"]
)
def test_every_holdout_admits_a_response_missing_either_competitor(
    response: ResponseSpec, held_out_unit: str
) -> None:
    grains = dict.fromkeys(response.readouts, "item")

    comparison = comparison_for(
        held_out_unit,
        response=response,
        readout_units=grains,
    )

    assert fit_with_comparison(response, comparison).comparison == comparison


def test_a_link_is_item_grain_only_when_every_consumed_readout_says_so() -> None:
    response = spec_for(Metric(), predictor=(RandomIntercept(group="item"), P_GRAM))
    grains = {
        "p_gram": "item",
        "p_gram_complement": "response",
    }

    comparison = comparison_for(
        "participant",
        response=response,
        readout_units=grains,
    )

    assert fit_with_comparison(response, comparison).comparison == comparison


PINNED_P_GRAM = LinkTerm(
    readout=P_GRAM.readout,
    linking=LinkingSpec(
        family=LOGIT.family,
        columns=LOGIT.columns,
        fixed={"slope": 1.0},
    ),
)


@pytest.mark.parametrize(
    "response",
    [
        spec_for(
            Metric(),
            predictor=(
                RandomIntercept(group="item"),
                RandomSlope(group="participant", term=P_GRAM),
            ),
        ),
        spec_for(
            Metric(),
            predictor=(
                RandomIntercept(group="item"),
                CorrelatedRandomSlope(group="participant", term=P_GRAM),
            ),
        ),
        spec_for(
            Metric(),
            predictor=(
                RandomIntercept(group="item"),
                InteractionTerm(left=VERB, right=PINNED_P_GRAM),
            ),
        ),
        ResponseSpec(
            family=Cumulative(),
            levels=7,
            objective=ConditionalLikelihood(),
            predictor=(RandomIntercept(group="item"),),
            identification=FixedCutpointVariance(value=1.0),
            priors=THRESHOLD_PRIORS,
            parameters=(
                ParameterFormula(
                    parameter=AuxiliaryParameter.DISCRIMINATION,
                    predictor=(P_GRAM,),
                ),
            ),
        ),
    ],
    ids=("random-slope", "correlated-slope", "interaction", "auxiliary-formula"),
)
def test_the_interlock_finds_a_link_nested_anywhere_in_the_response(
    response: ResponseSpec,
) -> None:
    with pytest.raises(
        dx.ValidationError,
        match=r"RandomIntercept\(kind='random_intercept'.*LinkTerm\(kind='link'",
    ):
        fit_with_comparison(response, comparison_for("frame", response=response))


@pytest.mark.parametrize(
    ("readout_units", "detail"),
    [
        ({"p_gram": "item"}, "missing.*p_gram_complement"),
        (
            {
                "p_gram": "item",
                "p_gram_complement": "item",
                "unused": "item",
            },
            "extra.*unused",
        ),
    ],
)
def test_a_comparison_refuses_a_missing_or_extra_readout_mapping(
    readout_units: dict[str, str], detail: str
) -> None:
    response = spec_for(Metric(), predictor=(P_GRAM,))
    comparison = comparison_for("item", response=response, readout_units=readout_units)

    with pytest.raises(dx.ValidationError, match=detail):
        fit_with_comparison(response, comparison)


@pytest.mark.parametrize(
    "omitted",
    [
        "held_out_unit",
        "readout_units",
        "splitter",
        "folds",
        "seed",
        "coverage_threshold",
        "unseen_participant",
        "models",
    ],
)
def test_a_comparison_requires_every_declaration(omitted: str) -> None:
    payload = comparison_for("item").model_dump()
    del payload[omitted]

    with pytest.raises(dx.ValidationError, match="required"):
        ComparisonSpec.model_validate(payload)


def test_a_comparison_refuses_a_blank_holdout_factor() -> None:
    payload = comparison_for("item").model_dump()
    payload["held_out_unit"] = " "

    with pytest.raises(dx.ValidationError, match="held-out unit"):
        ComparisonSpec.model_validate(payload)


def test_a_comparison_accepts_a_named_factor_as_a_readout_grain() -> None:
    payload = comparison_for("item").model_dump()
    payload["readout_units"] = {
        "p_gram": "item",
        "p_gram_complement": "sentence",
    }

    assert (
        ComparisonSpec.model_validate(payload).readout_units == payload["readout_units"]
    )


def test_comparison_wire_forms_round_trip_and_distinguish_declared_grains() -> None:
    item = comparison_for("item")
    participant = comparison_for("participant")
    response_readout = comparison_for(
        "item",
        readout_units=dict.fromkeys(PAIR, "response"),
    )
    by_list = comparison_for(
        "list",
        splitter="leave-one-group-out",
        folds=None,
    )
    variants = (item, participant, response_readout, by_list)

    assert len({spec_digest(variant) for variant in variants}) == len(variants)

    for variant in variants:
        restored = ComparisonSpec.model_validate_json(variant.model_dump_json())

        assert restored == variant


@pytest.mark.parametrize("field", ["pareto_fraction", "pareto_threshold"])
def test_a_model_comparison_validates_its_pareto_gate(field: str) -> None:
    payload = comparison_for("item").models.model_dump()
    payload[field] = -0.1

    with pytest.raises(dx.ValidationError, match="Pareto"):
        ModelComparisonSpec.model_validate(payload)


@pytest.mark.parametrize(
    ("candidates", "candidate", "match"),
    [
        (("only",), "only", "at least two"),
        (("same", "same"), "same", "repeats"),
        (("first", "second"), "absent", "not among"),
    ],
)
def test_a_model_comparison_declares_one_member_of_a_complete_candidate_set(
    candidates: tuple[str, ...], candidate: str, match: str
) -> None:
    payload = comparison_for("item").models.model_dump()
    payload["candidates"] = candidates
    payload["candidate"] = candidate

    with pytest.raises(dx.ValidationError, match=match):
        ModelComparisonSpec.model_validate(payload)


def test_likelihood_wire_forms_keep_their_exact_previous_fields() -> None:
    ordinal = spec_for(Metric(), predictor=(P_GRAM,))
    bounded = BoundedResponseSpec(
        family=Bounded(scale=0.2),
        predictor=(bounded_link(),),
        identification=IdentityLinkOnLogit(),
    )
    ordinal_fields = (
        "family",
        "levels",
        "predictor",
        "identification",
        "objective",
        "priors",
        "normalization",
        "parameters",
    )
    bounded_fields = (
        "family",
        "predictor",
        "identification",
        "priors",
        "normalization",
    )
    ordinal_wire = (
        *ordinal_fields,
        "terms",
        "predicted",
        "readouts",
        "factors",
        "groups",
    )
    bounded_wire = (
        *bounded_fields,
        "terms",
        "predicted",
        "parameters",
        "readouts",
        "factors",
        "groups",
    )

    assert tuple(ResponseSpec.__field_specs__) == ordinal_fields
    assert tuple(BoundedResponseSpec.__field_specs__) == bounded_fields
    assert tuple(ordinal.model_dump()) == ordinal_wire
    assert tuple(json.loads(ordinal.model_dump_json())) == ordinal_wire
    assert tuple(bounded.model_dump()) == bounded_wire
    assert tuple(json.loads(bounded.model_dump_json())) == bounded_wire


def test_a_specification_reads_back_every_field_it_was_given() -> None:
    spec = ResponseSpec(
        family=Cumulative(discrimination=True),
        levels=5,
        objective=ConditionalLikelihood(),
        predictor=(P_GRAM, PARTICIPANT),
        identification=FixedCutpointVariance(value=2.0),
        priors=PriorBlock(
            thresholds=ThresholdPriors(), discrimination=DISCRIMINATION_PRIOR
        ),
        normalization=ScaleNormalization.IDENTITY,
    )
    metric = ResponseSpec(
        family=Metric(),
        levels=5,
        objective=ConditionalLikelihood(),
        predictor=(P_GRAM,),
        identification=PINNED,
        normalization=ScaleNormalization.LOG,
    )

    assert spec.family == Cumulative(discrimination=True)
    assert spec.objective == ConditionalLikelihood()
    assert spec.levels == 5
    assert spec.predictor == (P_GRAM, PARTICIPANT)
    assert spec.identification == FixedCutpointVariance(value=2.0)
    assert spec.priors.thresholds == ThresholdPriors()
    assert spec.priors.discrimination == DISCRIMINATION_PRIOR
    assert spec.normalization is ScaleNormalization.IDENTITY
    assert metric.normalization is ScaleNormalization.LOG


def test_a_response_specification_requires_an_explicit_objective() -> None:
    with pytest.raises(dx.ValidationError, match="objective"):
        ResponseSpec.model_validate(
            {
                "family": {"family": "metric"},
                "levels": 7,
                "predictor": [{"kind": "random_intercept", "group": "participant"}],
                "identification": {
                    "constraint": "fixed_discrimination",
                    "value": 1.0,
                },
            }
        )


def test_both_likelihood_treatments_round_trip_through_the_tagged_boundary() -> None:
    conditional = LikelihoodObjective.model_validate({"objective": "conditional"})
    marginal = LikelihoodObjective.model_validate(
        {
            "objective": "truncated_marginal",
            "group": "participant",
            "radius": 4.0,
            "relative_tolerance": 1e-6,
            "absolute_tolerance": 1e-8,
            "quadrature_points": 64,
            "quadrature_seed": 31,
        }
    )

    assert conditional == ConditionalLikelihood()
    assert marginal == TruncatedMarginalLikelihood(
        group="participant",
        radius=4.0,
        relative_tolerance=1e-6,
        absolute_tolerance=1e-8,
        quadrature_points=64,
        quadrature_seed=31,
    )
    assert (
        LikelihoodObjective.model_validate_json(marginal.model_dump_json()) == marginal
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("radius", 0.0),
        ("relative_tolerance", math.inf),
        ("absolute_tolerance", -1.0),
        ("quadrature_points", 0),
        ("quadrature_seed", -1),
    ],
)
def test_a_marginal_objective_refuses_invalid_numerical_controls(
    field: str, value: float
) -> None:
    declaration: dict[str, JsonValue] = {
        "objective": "truncated_marginal",
        "group": "participant",
        "radius": 4.0,
        "relative_tolerance": 1e-6,
        "absolute_tolerance": 1e-8,
        "quadrature_points": 64,
        "quadrature_seed": 31,
    }
    declaration[field] = value

    with pytest.raises(dx.ValidationError):
        LikelihoodObjective.model_validate(declaration)


def test_the_predictor_keeps_its_terms_in_the_order_they_were_declared() -> None:
    forward = spec_for(Metric(), predictor=(P_GRAM, PARTICIPANT))
    backward = spec_for(Metric(), predictor=(PARTICIPANT, P_GRAM))

    assert forward.predictor[0] == P_GRAM
    assert backward.predictor[0] == PARTICIPANT
    assert forward != backward


def test_a_specification_changes_only_by_making_another() -> None:
    spec = cumulative_spec()
    longer = spec_for(Cumulative(), levels=9, priors=THRESHOLD_PRIORS)

    assert longer.levels == 9
    assert spec.levels == 7


@pytest.mark.parametrize("levels", [2, 3, 7, 101])
def test_a_response_of_at_least_two_levels_is_accepted(levels: int) -> None:
    assert spec_for(Metric(), levels=levels).levels == levels


@pytest.mark.parametrize("levels", [-1, 0, 1])
def test_a_response_of_fewer_than_two_levels_is_refused(levels: int) -> None:
    with pytest.raises(dx.ValidationError):
        spec_for(Metric(), levels=levels)


def test_a_predictor_with_no_terms_is_refused() -> None:
    with pytest.raises(dx.ValidationError):
        spec_for(Metric(), predictor=())


def test_a_predictor_of_one_term_is_accepted() -> None:
    assert spec_for(Metric(), predictor=(PARTICIPANT,)).predictor == (PARTICIPANT,)


@pytest.mark.parametrize(
    "omitted", ["family", "levels", "predictor", "identification", "objective"]
)
def test_a_specification_missing_a_required_field_is_refused(omitted: str) -> None:
    supplied: dict[str, JsonValue] = {
        "family": {"family": "metric"},
        "levels": 7,
        "predictor": [{"kind": "random_intercept", "group": "participant"}],
        "identification": {"constraint": "identity_link_on_logit"},
        "objective": {"objective": "conditional"},
    }
    del supplied[omitted]

    with pytest.raises(dx.ValidationError):
        ResponseSpec.model_validate(supplied)


def test_the_five_required_fields_together_are_enough() -> None:
    spec = ResponseSpec.model_validate(
        {
            "family": {"family": "metric"},
            "levels": 7,
            "predictor": [{"kind": "random_intercept", "group": "participant"}],
            "identification": {"constraint": "identity_link_on_logit"},
            "objective": {"objective": "conditional"},
        }
    )

    assert spec.identification == IdentityLinkOnLogit()
    assert spec.priors == NO_PRIORS
    assert spec.normalization is ScaleNormalization.IDENTITY


def test_the_specification_declares_exactly_its_eight_fields() -> None:
    assert tuple(ResponseSpec.__field_specs__) == (
        "family",
        "levels",
        "predictor",
        "identification",
        "objective",
        "priors",
        "normalization",
        "parameters",
    )


def test_an_unknown_key_is_refused_by_every_declaration() -> None:
    payloads: dict[type[dx.Model], dict[str, JsonValue]] = {
        PriorSpec: {"distribution": "Normal", "scale": 1.0},
        ThresholdPriors: {"slope": 1.0},
        PriorBlock: {"coefficient": None},
        Cumulative: {"family": "cumulative", "link": "probit"},
        Metric: {"family": "metric", "residual": "normal"},
        ResponseSpec: {
            "family": {"family": "metric"},
            "levels": 7,
            "predictor": [{"kind": "random_intercept", "group": "participant"}],
            "identification": {"constraint": "identity_link_on_logit"},
            "weight": 1.0,
        },
    }

    for model, payload in payloads.items():
        with pytest.raises(dx.ValidationError):
            model.model_validate(payload)


def test_a_variant_is_refused_a_key_its_sibling_declares() -> None:
    with pytest.raises(dx.ValidationError):
        ResponseFamily.model_validate({"family": "metric", "discrimination": True})


def test_no_field_states_a_bound_as_an_annotation() -> None:
    for model in DECLARED_MODELS:
        for name, spec in model.__field_specs__.items():
            assert spec.axioms == (), f"{model.__name__}.{name}"


def test_a_prior_names_a_distribution_and_defaults_to_no_parameters() -> None:
    prior = PriorSpec(distribution="Exponential")

    assert prior.distribution == "Exponential"
    assert prior.parameters == ()


def test_a_prior_naming_no_distribution_is_refused() -> None:
    with pytest.raises(dx.ValidationError):
        PriorSpec(distribution="", parameters=(0.0, 1.0))


def test_the_threshold_priors_default_to_the_ones_the_family_module_states() -> None:
    priors = ThresholdPriors()

    assert priors.base.distribution == DEFAULT_BASE_PRIOR.family
    assert priors.base.parameters == DEFAULT_BASE_PRIOR.arguments
    assert priors.increment.distribution == DEFAULT_INCREMENT_PRIOR.family
    assert priors.increment.parameters == DEFAULT_INCREMENT_PRIOR.arguments
    assert priors.base != priors.increment


def test_a_stated_threshold_prior_displaces_the_default() -> None:
    stated = PriorSpec(distribution="StudentT", parameters=(3.0, 0.0, 2.0))
    priors = ThresholdPriors(base=stated)

    assert priors.base == stated
    assert priors.increment.parameters == DEFAULT_INCREMENT_PRIOR.arguments


def test_a_family_says_which_parameters_it_draws() -> None:
    for family, thresholds, discrimination in FAMILIES_AND_WHAT_THEY_DRAW:
        assert family.draws_thresholds is thresholds, family
        assert family.draws_discrimination is discrimination, family


def test_a_rank_consistent_family_states_which_scheme_it_ties_with() -> None:
    with pytest.raises(dx.ValidationError):
        ResponseFamily.model_validate({"family": "rank_consistent"})

    coral = RankConsistent(scheme=RankConsistentScheme.CORAL)
    corn = RankConsistent(scheme=RankConsistentScheme.CORN)

    assert coral.scheme is RankConsistentScheme.CORAL
    assert corn.scheme is RankConsistentScheme.CORN
    assert coral != corn


def test_a_family_with_thresholds_needs_priors_over_them() -> None:
    with pytest.raises(dx.ValidationError, match="thresholds"):
        spec_for(Cumulative(), priors=NO_PRIORS)


def test_a_family_without_thresholds_is_refused_priors_over_them() -> None:
    with pytest.raises(dx.ValidationError, match="thresholds"):
        spec_for(Metric(), priors=THRESHOLD_PRIORS)


def test_a_rank_consistent_family_refuses_threshold_priors() -> None:
    with pytest.raises(dx.ValidationError, match="does not draw"):
        spec_for(
            RankConsistent(scheme=RankConsistentScheme.CORN),
            predictor=(P_GRAM,),
            identification=IDENTITY,
            priors=THRESHOLD_PRIORS,
        )


def test_a_rank_consistent_family_constructs_without_threshold_priors() -> None:
    declared = spec_for(
        RankConsistent(scheme=RankConsistentScheme.CORN),
        predictor=(P_GRAM,),
        identification=IDENTITY,
        priors=NO_PRIORS,
    )

    assert declared.priors == NO_PRIORS


@pytest.mark.parametrize(
    "term",
    [VERB, VERB_BY_FRAME, LENGTH, PARTICIPANT, SLOPE, CORRELATED_SLOPE],
)
def test_a_rank_consistent_family_refuses_every_non_scalar_link_term(
    term: Predictor,
) -> None:
    with pytest.raises(dx.ValidationError, match="exactly one scalar link term"):
        spec_for(
            RankConsistent(scheme=RankConsistentScheme.CORAL),
            predictor=(term,),
            identification=IDENTITY,
        )


def test_a_rank_consistent_family_refuses_more_than_one_link_term() -> None:
    with pytest.raises(dx.ValidationError, match="exactly one scalar link term"):
        spec_for(
            RankConsistent(scheme=RankConsistentScheme.CORAL),
            predictor=(P_GRAM, LM),
            identification=IDENTITY,
        )


@pytest.mark.parametrize("identification", [PINNED, FixedCutpointVariance(value=1.0)])
def test_a_rank_consistent_family_refuses_other_scale_identifications(
    identification: ScaleIdentification,
) -> None:
    with pytest.raises(dx.ValidationError, match="identity_link_on_logit"):
        spec_for(
            RankConsistent(scheme=RankConsistentScheme.CORAL),
            predictor=(P_GRAM,),
            identification=identification,
        )


def test_a_rank_consistent_family_refuses_a_non_logit_link() -> None:
    with pytest.raises(dx.ValidationError, match="identified on the logit scale"):
        spec_for(
            RankConsistent(scheme=RankConsistentScheme.CORAL),
            predictor=(LinkTerm(readout="p_gram", linking=PROBABILITY),),
            identification=IDENTITY,
        )


def test_a_family_drawing_discrimination_needs_a_prior_over_it() -> None:
    with pytest.raises(dx.ValidationError, match="discrimination"):
        spec_for(Cumulative(discrimination=True), priors=THRESHOLD_PRIORS)


def test_a_family_fixing_discrimination_is_refused_a_prior_over_it() -> None:
    with pytest.raises(dx.ValidationError, match="discrimination"):
        spec_for(
            Cumulative(),
            priors=PriorBlock(
                thresholds=ThresholdPriors(), discrimination=DISCRIMINATION_PRIOR
            ),
        )


def test_a_family_drawing_discrimination_constructs_with_a_prior_over_it() -> None:
    spec = spec_for(
        Cumulative(discrimination=True),
        identification=FixedCutpointVariance(value=1.0),
        priors=PriorBlock(
            thresholds=ThresholdPriors(), discrimination=DISCRIMINATION_PRIOR
        ),
    )

    assert spec.family.draws_discrimination
    assert spec.priors.discrimination == DISCRIMINATION_PRIOR


def test_a_metric_family_constructs_with_no_priors_of_its_own() -> None:
    assert spec_for(Metric()).priors == NO_PRIORS


def test_the_refusal_names_the_family_whose_declaration_was_wrong() -> None:
    with pytest.raises(dx.ValidationError, match="cumulative"):
        spec_for(Cumulative(), priors=NO_PRIORS)


def test_ridit_scaling_is_refused_beside_a_family_that_fits_thresholds() -> None:
    with pytest.raises(dx.ValidationError, match="twice"):
        spec_for(
            Cumulative(),
            priors=THRESHOLD_PRIORS,
            normalization=ScaleNormalization.RIDIT,
        )


def test_a_rank_consistent_family_refuses_response_normalization() -> None:
    with pytest.raises(dx.ValidationError, match="does not normalize"):
        spec_for(
            RankConsistent(scheme=RankConsistentScheme.CORAL),
            predictor=(P_GRAM,),
            identification=IDENTITY,
            normalization=ScaleNormalization.RIDIT,
        )


def test_the_refusal_of_ridit_names_both_corrections_and_the_family() -> None:
    with pytest.raises(dx.ValidationError) as raised:
        spec_for(
            Cumulative(),
            priors=THRESHOLD_PRIORS,
            normalization=ScaleNormalization.RIDIT,
        )

    message = str(raised.value)

    assert "visibly" in message
    assert "invisibly" in message
    assert "cumulative" in message


def test_ridit_scaling_constructs_beside_a_family_that_fits_no_thresholds() -> None:
    spec = spec_for(Metric(), normalization=ScaleNormalization.RIDIT)

    assert spec.normalization is ScaleNormalization.RIDIT


def test_the_recorded_level_stands_beside_thresholds() -> None:
    spec = spec_for(Cumulative(), priors=THRESHOLD_PRIORS)

    assert spec.normalization is ScaleNormalization.IDENTITY


@pytest.mark.parametrize(
    "normalization", [ScaleNormalization.LOG, ScaleNormalization.UNIT_INTERVAL]
)
def test_a_family_with_a_program_refuses_a_normalization_of_its_level(
    normalization: ScaleNormalization,
) -> None:
    with pytest.raises(dx.ValidationError, match="observes the recorded level"):
        spec_for(Cumulative(), priors=THRESHOLD_PRIORS, normalization=normalization)

    with pytest.raises(dx.ValidationError, match="observes the recorded level"):
        choice_spec(
            CHOICE_LINK,
            POSITION_TERM,
            family=Choice(),
            normalization=normalization,
        )


@pytest.mark.parametrize("normalization", list(ScaleNormalization))
def test_a_normalization_travels_as_the_string_it_declares(
    normalization: ScaleNormalization,
) -> None:
    spec = spec_for(Metric(), normalization=normalization)
    restored = ResponseSpec.model_validate_json(spec.model_dump_json())

    assert f'"{normalization.value}"' in spec.model_dump_json()
    assert restored.normalization is normalization


def test_a_specification_round_trips_through_its_wire_form() -> None:
    spec = ResponseSpec(
        family=Cumulative(discrimination=True),
        levels=5,
        objective=ConditionalLikelihood(),
        predictor=(LM, PARTICIPANT),
        identification=FixedCutpointVariance(value=2.5),
        priors=PriorBlock(
            thresholds=ThresholdPriors(
                base=PriorSpec(distribution="StudentT", parameters=(3.0, 0.0, 2.0))
            ),
            discrimination=DISCRIMINATION_PRIOR,
        ),
    )
    restored = ResponseSpec.model_validate_json(spec.model_dump_json())

    assert restored == spec
    assert restored.family == Cumulative(discrimination=True)
    assert restored.predictor == (LM, PARTICIPANT)
    assert restored.identification == FixedCutpointVariance(value=2.5)
    assert restored.priors.thresholds is not None
    assert restored.priors.thresholds.base.parameters == (3.0, 0.0, 2.0)


def test_two_equal_specifications_collapse_in_a_set() -> None:
    spec = cumulative_spec()
    same = cumulative_spec()
    other = spec_for(Cumulative(), levels=5, priors=THRESHOLD_PRIORS)

    assert {spec, same, other} == {spec, other}


def test_a_family_is_decoded_from_the_root_by_its_declared_tag() -> None:
    payloads: tuple[tuple[str, dict[str, JsonValue]], ...] = (
        ("cumulative", {"family": "cumulative", "discrimination": True}),
        ("rank_consistent", {"family": "rank_consistent", "scheme": "corn"}),
        ("metric", {"family": "metric"}),
    )
    decoded = {tag: ResponseFamily.model_validate(payload) for tag, payload in payloads}

    assert decoded["cumulative"] == Cumulative(discrimination=True)
    assert decoded["rank_consistent"] == RankConsistent(
        scheme=RankConsistentScheme.CORN
    )
    assert decoded["metric"] == Metric()


def test_a_discrimination_group_names_the_factor_the_discrimination_varies_over() -> (
    None
):
    grouped = Cumulative(discrimination=True, discrimination_group="item")
    decoded = ResponseFamily.model_validate(
        {"family": "cumulative", "discrimination": True, "discrimination_group": "item"}
    )

    assert grouped.discrimination_group == "item"
    assert grouped.draws_discrimination
    assert decoded == grouped
    assert Cumulative().discrimination_group is None
    assert grouped != Cumulative(discrimination=True)


def test_a_discrimination_group_without_a_drawn_discrimination_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="holds it at one"):
        Cumulative(discrimination_group="item")


def test_a_discrimination_group_without_a_name_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="names its grouping factor"):
        Cumulative(discrimination=True, discrimination_group="")


def test_a_term_declared_outside_the_module_decodes_through_the_root() -> None:
    term = Predictor.model_validate({"kind": "test_readout", "readout": "p_gram"})

    assert term == ReadoutTerm(readout="p_gram")


def test_an_unregistered_tag_is_refused_by_the_root() -> None:
    with pytest.raises(dx.ValidationError):
        ResponseFamily.model_validate({"family": "beta"})

    with pytest.raises(dx.ValidationError):
        Predictor.model_validate({"kind": "test_absent"})

    with pytest.raises(dx.ValidationError):
        ScaleIdentification.model_validate({"constraint": "free_scale"})


def test_a_linking_names_its_family_and_defaults_to_its_argument_alone() -> None:
    assert PROBABILITY.family == "probability"
    assert PROBABILITY.columns == {}
    assert PROBABILITY.fixed == {}
    assert LOGIT.columns == {"complement": "p_gram_complement"}


def test_a_linking_keeps_its_further_columns_in_role_order() -> None:
    forward = LinkingSpec(family="morcela", columns={"length": "n", "frequency": "f"})
    backward = LinkingSpec(family="morcela", columns={"frequency": "f", "length": "n"})

    assert tuple(forward.columns) == ("frequency", "length")
    assert forward == backward
    assert forward.model_dump_json() == backward.model_dump_json()


@pytest.mark.parametrize(
    ("columns", "detail"),
    [({"": "length"}, "has no role"), ({"length": ""}, "names no readout")],
)
def test_a_linking_column_blank_on_either_side_is_refused(
    columns: dict[str, str], detail: str
) -> None:
    with pytest.raises(dx.ValidationError, match=detail):
        LinkingSpec(family="morcela", columns=columns)


def test_a_linking_naming_no_family_is_refused() -> None:
    with pytest.raises(dx.ValidationError):
        LinkingSpec(family="")


def test_a_linking_keeps_its_pinned_parameters_in_name_order() -> None:
    forward = LinkingSpec(family="morcela", fixed={"length": 1.0, "frequency": 1.0})
    backward = LinkingSpec(family="morcela", fixed={"frequency": 1.0, "length": 1.0})

    assert tuple(forward.fixed) == ("frequency", "length")
    assert forward == backward
    assert hash(forward) == hash(backward)
    assert forward.model_dump_json() == backward.model_dump_json()


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_a_pinned_parameter_that_is_not_finite_is_refused(value: float) -> None:
    with pytest.raises(dx.ValidationError, match="is not finite"):
        LinkingSpec(family="logit", fixed={"slope": value})


def test_a_pinned_parameter_without_a_name_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="has no name"):
        LinkingSpec(family="logit", fixed={"": 1.0})


def test_the_seven_term_kinds_decode_through_the_root_by_their_tags() -> None:
    decoded = {
        term.kind: Predictor.model_validate_json(term.model_dump_json())
        for term in TERMS
    }

    assert tuple(decoded) == (
        "link",
        "factor",
        "interaction",
        "random_intercept",
        "random_slope",
        "correlated_random_slope",
        "offset",
    )
    assert tuple(decoded.values()) == TERMS


def test_a_link_term_names_its_readout_and_its_linking() -> None:
    assert P_GRAM.readout == "p_gram"
    assert P_GRAM.linking == LOGIT
    assert P_GRAM.readouts == PAIR
    assert P_GRAM.factors == ()
    assert P_GRAM.groups == ()
    assert P_GRAM.fixed


def test_a_link_term_reads_its_argument_and_then_what_its_linking_names() -> None:
    assert LM.readouts == ("lm_log_probability", "unigram_log_probability", "length")
    assert RandomSlope(group="participant", term=LM).readouts == LM.readouts


def test_a_link_term_names_a_column_read_under_two_roles_once() -> None:
    twice = LinkTerm(
        readout="lm",
        linking=LinkingSpec(family="f", columns={"a": "lm", "b": "n", "c": "n"}),
    )

    assert twice.readouts == ("lm", "n")


def test_a_link_term_makes_the_log_odds_link_one_declaration() -> None:
    declared = LinkTerm(
        readout="p_gram",
        linking=LinkingSpec(
            family="logit", columns={"complement": "p_gram_complement"}
        ),
    )
    pinned = LinkTerm(
        readout="p_gram",
        linking=LinkingSpec(
            family="logit",
            columns={"complement": "p_gram_complement"},
            fixed={"slope": 1.0},
        ),
    )

    assert declared == P_GRAM
    assert pinned != declared
    assert pinned.linking.fixed == {"slope": 1.0}


def test_a_factor_term_names_its_factor() -> None:
    assert VERB.name == "verb"
    assert VERB.factors == ("verb",)
    assert VERB.readouts == ()
    assert VERB.fixed


def test_an_interaction_reads_what_both_operands_read() -> None:
    mixed = InteractionTerm(left=VERB, right=P_GRAM)

    assert VERB_BY_FRAME.factors == ("verb", "frame")
    assert VERB_BY_FRAME.readouts == ()
    assert mixed.factors == ("verb",)
    assert mixed.readouts == PAIR
    assert mixed.fixed


def test_an_interaction_nests_and_names_each_factor_once() -> None:
    nested = InteractionTerm(
        left=VERB_BY_FRAME, right=InteractionTerm(left=VERB, right=P_GRAM)
    )

    assert nested.factors == ("verb", "frame")
    assert nested.readouts == PAIR


@pytest.mark.parametrize(
    ("operand", "kind"),
    [(PARTICIPANT, "random_intercept"), (SLOPE, "random_slope"), (LENGTH, "offset")],
)
def test_an_interaction_refuses_an_operand_that_is_not_a_fixed_effect(
    operand: Predictor, kind: str
) -> None:
    with pytest.raises(dx.ValidationError, match=f"left operand is the {kind} term"):
        InteractionTerm(left=operand, right=VERB)

    with pytest.raises(dx.ValidationError, match=f"right operand is the {kind} term"):
        InteractionTerm(left=VERB, right=operand)


def test_an_interaction_of_a_term_with_itself_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="with itself"):
        InteractionTerm(left=VERB, right=FactorTerm(name="verb"))


def test_an_interaction_reading_one_readout_through_two_link_terms_is_refused() -> None:
    linear = LinkTerm(readout="score", linking=LinkingSpec(family="linear"))
    pinned = LinkTerm(
        readout="score", linking=LinkingSpec(family="linear", fixed={"slope": 2.0})
    )

    with pytest.raises(dx.ValidationError, match=r"reads \['score'\] through more"):
        InteractionTerm(left=linear, right=pinned)

    with pytest.raises(dx.ValidationError, match=r"reads \['score'\] through more"):
        InteractionTerm(left=InteractionTerm(left=VERB, right=linear), right=pinned)

    assert InteractionTerm(left=linear, right=P_GRAM).readouts == ("score", *PAIR)


@pytest.mark.parametrize(
    "repeated",
    [
        (
            LinkTerm(readout="score", linking=LinkingSpec(family="linear")),
            LinkTerm(readout="score", linking=LinkingSpec(family="probability")),
        ),
        (
            InteractionTerm(
                left=LinkTerm(readout="score", linking=LinkingSpec(family="linear")),
                right=VERB,
            ),
            InteractionTerm(
                left=LinkTerm(
                    readout="score", linking=LinkingSpec(family="probability")
                ),
                right=VERB,
            ),
        ),
    ],
)
def test_two_fixed_effects_over_the_same_columns_are_refused(
    repeated: tuple[Predictor, Predictor],
) -> None:
    linear, probability = repeated
    distinct = InteractionTerm(left=probability, right=FactorTerm(name="frame"))

    with pytest.raises(dx.ValidationError, match="as two fixed effects"):
        ResponseSpec(
            family=Cumulative(),
            levels=5,
            objective=ConditionalLikelihood(),
            predictor=(linear, probability),
            identification=IdentityLinkOnLogit(),
            priors=PriorBlock(thresholds=ThresholdPriors()),
        )

    with pytest.raises(dx.ValidationError, match="as two fixed effects"):
        ParameterFormula(
            parameter=AuxiliaryParameter.DISCRIMINATION, predictor=(linear, probability)
        )

    assert ResponseSpec(
        family=Cumulative(),
        levels=5,
        objective=ConditionalLikelihood(),
        predictor=(linear, distinct),
        identification=IdentityLinkOnLogit(),
        priors=PriorBlock(thresholds=ThresholdPriors()),
    ).predictor == (linear, distinct)


def test_a_random_intercept_names_the_group_it_varies_over() -> None:
    assert PARTICIPANT.group == "participant"
    assert PARTICIPANT.groups == ("participant",)
    assert PARTICIPANT.readouts == ()
    assert not PARTICIPANT.fixed


def test_a_random_slope_varies_a_fixed_effect_over_a_group() -> None:
    assert SLOPE.group == "participant"
    assert SLOPE.term == P_GRAM
    assert SLOPE.groups == ("participant",)
    assert SLOPE.readouts == PAIR
    assert SLOPE.factors == ()
    assert not SLOPE.fixed


def test_a_correlated_random_slope_declares_intercept_then_scalar_slope() -> None:
    decoded = Predictor.model_validate_json(CORRELATED_SLOPE.model_dump_json())

    assert decoded == CORRELATED_SLOPE
    assert CORRELATED_SLOPE.group == "participant"
    assert CORRELATED_SLOPE.term == P_GRAM
    assert CORRELATED_SLOPE.groups == ("participant",)
    assert CORRELATED_SLOPE.readouts == PAIR
    assert CORRELATED_SLOPE.factors == ()
    assert CORRELATED_SLOPE.concentration == 1.0
    assert not CORRELATED_SLOPE.fixed


@pytest.mark.parametrize(
    ("term", "concentration", "message"),
    [
        (VERB, 1.0, "scalar fixed effect"),
        (PARTICIPANT, 1.0, "scalar fixed effect"),
        (P_GRAM, 0.0, "concentration"),
        (P_GRAM, float("inf"), "not finite"),
    ],
)
def test_a_correlated_random_slope_refuses_an_invalid_declaration(
    term: Predictor, concentration: float, message: str
) -> None:
    with pytest.raises(dx.ValidationError, match=message):
        CorrelatedRandomSlope(
            group="participant", term=term, concentration=concentration
        )


@pytest.mark.parametrize(
    ("term", "kind"),
    [(PARTICIPANT, "random_intercept"), (SLOPE, "random_slope"), (LENGTH, "offset")],
)
def test_a_random_slope_refuses_a_term_that_is_not_a_fixed_effect(
    term: Predictor, kind: str
) -> None:
    with pytest.raises(dx.ValidationError, match=f"varies the {kind} term"):
        RandomSlope(group="participant", term=term)


def test_an_offset_term_names_its_readout_and_fits_nothing() -> None:
    assert LENGTH.readout == "length"
    assert LENGTH.readouts == ("length",)
    assert LENGTH.groups == ()
    assert not LENGTH.fixed


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "link", "readout": "", "linking": {"family": "logit"}},
        {"kind": "link", "readout": "p_gram"},
        {"kind": "factor", "name": ""},
        {"kind": "random_intercept", "group": ""},
        {
            "kind": "random_slope",
            "group": "",
            "term": {"kind": "factor", "name": "verb"},
        },
        {"kind": "offset", "readout": ""},
        {"kind": "interaction", "left": {"kind": "factor", "name": "verb"}},
    ],
)
def test_a_term_missing_or_blanking_a_required_field_is_refused(
    payload: dict[str, JsonValue],
) -> None:
    with pytest.raises(dx.ValidationError):
        Predictor.model_validate(payload)


def test_a_specification_reads_every_column_its_terms_read_once_each() -> None:
    spec = spec_for(
        Cumulative(),
        predictor=(
            PARTICIPANT,
            P_GRAM,
            VERB_BY_FRAME,
            SLOPE,
            LM,
            LENGTH,
            RandomIntercept(group="frame"),
        ),
        priors=THRESHOLD_PRIORS,
    )

    assert spec.readouts == (
        *PAIR,
        "lm_log_probability",
        "unigram_log_probability",
        "length",
    )
    assert spec.factors == ("verb", "frame")
    assert spec.groups == ("participant", "frame")


def test_a_specification_declaring_a_term_twice_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="twice"):
        spec_for(
            Metric(),
            predictor=(P_GRAM, VERB, LinkTerm(readout="p_gram", linking=LOGIT)),
        )


def test_the_three_identifications_decode_through_the_root_by_their_tags() -> None:
    variants = (
        FixedCutpointVariance(value=1.0),
        IdentityLinkOnLogit(),
        FixedDiscrimination(value=1.0),
    )
    decoded = tuple(
        ScaleIdentification.model_validate_json(variant.model_dump_json())
        for variant in variants
    )

    assert decoded == variants
    assert tuple(variant.constraint for variant in variants) == (
        "fixed_cutpoint_variance",
        "identity_link_on_logit",
        "fixed_discrimination",
    )


@pytest.mark.parametrize("variant", [FixedCutpointVariance, FixedDiscrimination])
@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf])
def test_a_pinned_scale_value_must_be_positive_and_finite(
    variant: type[FixedCutpointVariance | FixedDiscrimination], value: float
) -> None:
    with pytest.raises(dx.ValidationError):
        variant(value=value)


def test_the_identity_link_on_logit_takes_no_value() -> None:
    with pytest.raises(dx.ValidationError):
        ScaleIdentification.model_validate(
            {"constraint": "identity_link_on_logit", "value": 1.0}
        )


def test_a_specification_over_the_declared_algebra_round_trips() -> None:
    spec = ResponseSpec(
        family=Cumulative(),
        levels=7,
        objective=ConditionalLikelihood(),
        predictor=(PARTICIPANT, P_GRAM, VERB_BY_FRAME, SLOPE, LENGTH),
        identification=FixedCutpointVariance(value=1.0),
        priors=THRESHOLD_PRIORS,
    )
    restored = ResponseSpec.model_validate_json(spec.model_dump_json())

    assert restored == spec
    assert restored.predictor == spec.predictor
    assert spec_digest(restored) == spec_digest(spec)


def test_fixed_discrimination_refuses_a_composite_with_two_free_scales() -> None:
    offending = InteractionTerm(left=VERB, right=P_GRAM)

    with pytest.raises(dx.ValidationError, match=r"predictor term .*scale-degenerate"):
        spec_for(Metric(), predictor=(offending,))


def test_pinning_the_composite_link_slope_identifies_the_predictor() -> None:
    pinned = LinkTerm(
        readout=P_GRAM.readout,
        linking=LinkingSpec(
            family=LOGIT.family,
            columns=LOGIT.columns,
            fixed={"slope": 1.0},
        ),
    )
    identified = InteractionTerm(left=VERB, right=pinned)

    assert spec_for(Metric(), predictor=(identified,)).predictor == (identified,)


def test_scale_degeneracy_is_refused_before_a_backend_is_reached() -> None:
    class SpyBackend:
        def __init__(self) -> None:
            self.reached = False

        def compile(self, spec: ResponseSpec) -> None:
            del spec
            self.reached = True

    backend = SpyBackend()
    payload: dict[str, JsonValue] = {
        "family": {"family": "metric"},
        "levels": 7,
        "predictor": [
            {
                "kind": "interaction",
                "left": {"kind": "factor", "name": "verb"},
                "right": {
                    "kind": "link",
                    "readout": "p_gram",
                    "linking": {
                        "family": "logit",
                        "columns": {"complement": "p_gram_complement"},
                    },
                },
            }
        ],
        "identification": {"constraint": "fixed_discrimination", "value": 1.0},
        "objective": {"objective": "conditional"},
    }

    with pytest.raises(dx.ValidationError, match="scale-degenerate"):
        backend.compile(ResponseSpec.model_validate(payload))

    assert not backend.reached


def test_a_digest_is_a_hex_sha256_that_two_equal_specifications_share() -> None:
    digest = spec_digest(cumulative_spec())

    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")
    assert digest == spec_digest(cumulative_spec())


def test_a_digest_changes_with_any_declared_field() -> None:
    base = spec_for(
        Cumulative(), predictor=(PARTICIPANT, P_GRAM), priors=THRESHOLD_PRIORS
    )
    variants = (
        spec_for(
            Cumulative(),
            levels=5,
            predictor=(PARTICIPANT, P_GRAM),
            priors=THRESHOLD_PRIORS,
        ),
        spec_for(
            Cumulative(), predictor=(P_GRAM, PARTICIPANT), priors=THRESHOLD_PRIORS
        ),
        spec_for(
            Cumulative(),
            predictor=(
                PARTICIPANT,
                LinkTerm(readout="p_gram", linking=LinkingSpec(family="probability")),
            ),
            priors=THRESHOLD_PRIORS,
        ),
        spec_for(
            Cumulative(),
            predictor=(
                PARTICIPANT,
                LinkTerm(
                    readout="p_gram",
                    linking=LinkingSpec(family="logit", columns={"complement": "n"}),
                ),
            ),
            priors=THRESHOLD_PRIORS,
        ),
        spec_for(
            Cumulative(),
            predictor=(
                PARTICIPANT,
                LinkTerm(
                    readout="p_gram",
                    linking=LinkingSpec(
                        family="logit",
                        columns={"complement": "p_gram_complement"},
                        fixed={"slope": 1.0},
                    ),
                ),
            ),
            priors=THRESHOLD_PRIORS,
        ),
        spec_for(
            Cumulative(discrimination=True),
            identification=FixedCutpointVariance(value=1.0),
            predictor=(PARTICIPANT, P_GRAM),
            priors=PriorBlock(
                thresholds=ThresholdPriors(), discrimination=DISCRIMINATION_PRIOR
            ),
        ),
    )
    digests = {spec_digest(base), *(spec_digest(variant) for variant in variants)}

    assert len(digests) == len(variants) + 1


def test_a_digest_ignores_the_order_pinned_parameters_were_given_in() -> None:
    columns = {"length": "n", "frequency": "unigram"}
    forward = LinkTerm(
        readout="lm",
        linking=LinkingSpec(
            family="morcela", columns=columns, fixed={"length": 1.0, "frequency": 2.0}
        ),
    )
    backward = LinkTerm(
        readout="lm",
        linking=LinkingSpec(
            family="morcela", columns=columns, fixed={"frequency": 2.0, "length": 1.0}
        ),
    )

    assert spec_digest(spec_for(Metric(), predictor=(forward,))) == spec_digest(
        spec_for(Metric(), predictor=(backward,))
    )


def ladder_index() -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    index = json.loads((LADDER / "ladder.json").read_text(encoding="utf-8"))

    return tuple(index["rungs"]), {
        rung: tuple(names) for rung, names in index["ablations"].items()
    }


def annotator_model() -> str:
    index = json.loads((LADDER / "ladder.json").read_text(encoding="utf-8"))

    return index["annotator"]


def rung(name: str) -> ResponseSpec:
    return ResponseSpec.model_validate_json(
        (LADDER / f"{name}.json").read_text(encoding="utf-8")
    )


def declared_fields(spec: ResponseSpec) -> dict[str, JsonValue]:
    dumped = json.loads(spec.model_dump_json())
    fields: dict[str, JsonValue] = {
        name: value
        for name, value in dumped.items()
        if name in ResponseSpec.__field_specs__ and name != "predictor"
    }

    for position, term in enumerate(dumped["predictor"]):
        fields[f"predictor[{position}]"] = term

    return fields


def differing_fields(left: ResponseSpec, right: ResponseSpec) -> frozenset[str]:
    first = declared_fields(left)
    second = declared_fields(right)

    return frozenset(
        name
        for name in first.keys() | second.keys()
        if first.get(name) != second.get(name)
    )


def test_the_ladder_index_names_the_rungs_the_ablations_and_the_annotator_model() -> (
    None
):
    rungs, ablations = ladder_index()
    named = (
        frozenset(rungs)
        | frozenset(name for names in ablations.values() for name in names)
        | {annotator_model()}
    )
    fixtures = frozenset(path.stem for path in LADDER.glob("*.json")) - {"ladder"}

    assert len(rungs) == 10
    assert sum(len(names) for names in ablations.values()) == 6
    assert len(named) == 17
    assert named == fixtures
    assert set(ablations) <= set(rungs)
    assert annotator_model() not in rungs


def test_the_ladder_declares_item_holdout_explicitly() -> None:
    index = json.loads((LADDER / "ladder.json").read_text(encoding="utf-8"))

    assert index["held_out_unit"] == "item"


def test_the_seventeen_fixtures_load_into_distinct_specifications() -> None:
    rungs, ablations = ladder_index()
    names = (
        *rungs,
        *(name for names in ablations.values() for name in names),
        annotator_model(),
    )
    specs = {name: rung(name) for name in names}
    digests = {spec_digest(spec) for spec in specs.values()}

    assert len(specs) == 17
    assert len(digests) == 17
    assert len(set(specs.values())) == 17


def test_every_rung_declares_the_identical_head() -> None:
    rungs, _ = ladder_index()
    heads = {
        (spec.family, spec.levels, spec.identification, spec.priors, spec.normalization)
        for spec in map(rung, rungs)
    }

    assert heads == {
        (
            Cumulative(),
            7,
            FixedDiscrimination(value=1.0),
            PriorBlock(thresholds=ThresholdPriors()),
            ScaleNormalization.IDENTITY,
        )
    }


def test_the_two_grouped_morcela_rungs_have_distinct_hashes_and_identical_heads() -> (
    None
):
    by_frame = rung("morcela_by_frame")
    by_verb_by_frame = rung("morcela_by_verb_by_frame")
    head_fields = (
        "family",
        "levels",
        "identification",
        "priors",
        "normalization",
    )

    assert spec_digest(by_frame) != spec_digest(by_verb_by_frame)
    assert tuple(getattr(by_frame, field) for field in head_fields) == tuple(
        getattr(by_verb_by_frame, field) for field in head_fields
    )
    assert by_frame.predictor == RUNG_TERMS["morcela_by_frame"]
    assert by_verb_by_frame.predictor == RUNG_TERMS["morcela_by_verb_by_frame"]


def test_every_rung_differs_from_its_neighbor_in_exactly_one_declared_field() -> None:
    rungs, _ = ladder_index()
    specs = tuple(map(rung, rungs))

    for lower, upper in pairwise(specs):
        differing = differing_fields(lower, upper)

        assert len(differing) == 1, (lower, upper, differing)
        assert next(iter(differing)).startswith("predictor[")


def test_every_rung_shares_the_participant_intercept_the_head_needs() -> None:
    rungs, _ = ladder_index()

    for spec in map(rung, rungs):
        assert spec.predictor[0] == PARTICIPANT


def test_the_first_rung_is_the_intercept_alone_and_the_last_adds_both_groups() -> None:
    rungs, _ = ladder_index()
    first = rung(rungs[0])
    last = rung(rungs[-1])

    assert first.predictor == (PARTICIPANT,)
    assert last.groups == ("participant", "frame", "verb")
    assert last.readouts != ()


RUNG_TERMS: Final[dict[str, tuple[Predictor, ...]]] = {
    "intercept_only": (PARTICIPANT,),
    "frame_only": (PARTICIPANT, FRAME),
    "verb_only": (PARTICIPANT, VERB),
    "verb_by_frame": (PARTICIPANT, VERB_BY_FRAME),
    "pooled_encoder": (
        PARTICIPANT,
        LinkTerm(readout="encoder_pooled", linking=LinkingSpec(family="identity")),
    ),
    "p_gram_only": (PARTICIPANT, P_GRAM),
    "morcela_only": (PARTICIPANT, LM),
    "p_gram_morcela": (PARTICIPANT, LM, P_GRAM),
    "morcela_by_frame": (PARTICIPANT, LM, RandomIntercept(group="frame")),
    "morcela_by_verb_by_frame": (
        PARTICIPANT,
        LM,
        RandomIntercept(group="frame"),
        RandomIntercept(group="verb"),
    ),
}


def test_each_rung_declares_the_predictor_its_name_says() -> None:
    rungs, _ = ladder_index()

    assert set(rungs) == set(RUNG_TERMS)

    for name in rungs:
        assert rung(name).predictor == RUNG_TERMS[name], name


def test_the_morcela_rungs_declare_the_length_and_frequency_columns() -> None:
    rungs, _ = ladder_index()
    morcela = {
        name: spec.readouts
        for name in rungs
        if "lm_log_probability" in (spec := rung(name)).readouts
    }

    assert len(morcela) == 4

    for name, readouts in morcela.items():
        assert {"length", "unigram_log_probability"} <= set(readouts), name


def test_the_link_rungs_declare_the_readout_and_linking_they_are_named_for() -> None:
    rungs, _ = ladder_index()
    links = {
        name: tuple(term for term in rung(name).predictor if isinstance(term, LinkTerm))
        for name in rungs
    }
    linked = {name: terms for name, terms in links.items() if terms}
    families = {term.linking.family for terms in linked.values() for term in terms}

    assert len(linked) == 6
    assert all(term.linking.fixed == {} for terms in linked.values() for term in terms)
    assert families == {"logit", "morcela", "identity"}
    assert any(len(terms) == 2 for terms in linked.values())
    assert linked["p_gram_only"] == (P_GRAM,)


def test_each_ablation_head_changes_one_thing_about_its_rung() -> None:
    _, ablations = ladder_index()
    families = set()
    identified = set()
    linkings = set()

    for name, heads in ablations.items():
        base = rung(name)

        for head_name in heads:
            head = rung(head_name)

            if isinstance(head.family, RankConsistent):
                families.add(head.family.family)

                assert head.predictor == (P_GRAM,)
                assert head.identification == IDENTITY
                assert head.priors == NO_PRIORS
                assert head.groups == ()
                continue

            if head.identification != base.identification:
                identified.add(head_name)
                continue

            differing = differing_fields(base, head)

            assert head.levels == base.levels
            assert head.identification == base.identification
            assert head.normalization == base.normalization

            if "family" in differing:
                families.add(head.family.family)

                assert differing <= {"family", "priors"}
                assert head.predictor == base.predictor
                assert ("priors" in differing) == (not head.family.draws_thresholds)
            else:
                changed = [
                    (position, before, after)
                    for position, (before, after) in enumerate(
                        zip(base.predictor, head.predictor, strict=True)
                    )
                    if before != after
                ]

                assert len(changed) == 1

                position, before, after = changed[0]

                assert differing == {f"predictor[{position}]"}
                assert isinstance(before, LinkTerm)
                assert isinstance(after, LinkTerm)
                assert after.readout == before.readout
                assert head.priors == base.priors
                linkings.add(after.linking.family)

    assert families == {"rank_consistent", "metric"}
    assert identified == {"p_gram_signed", "p_gram_entropy"}
    assert linkings == {"probability"}


def test_the_probability_rung_differs_from_the_logit_rung_in_the_linking_alone() -> (
    None
):
    _, ablations = ladder_index()
    logit = rung("p_gram_only")
    probability = rung("p_gram_probability")

    assert "p_gram_probability" in ablations["p_gram_only"]
    assert differing_fields(logit, probability) == {"predictor[1]"}
    assert logit.predictor[1] == P_GRAM
    assert probability.predictor[1] == LinkTerm(readout="p_gram", linking=PROBABILITY)
    assert probability.readouts == ("p_gram",)
    assert logit.readouts == ("p_gram", "p_gram_complement")


def test_the_signed_scalar_rung_changes_only_the_scale_identification() -> None:
    _, ablations = ladder_index()
    base = rung("p_gram_only")
    signed = rung("p_gram_signed")

    assert "p_gram_signed" in ablations["p_gram_only"]
    assert differing_fields(base, signed) == {"identification"}
    assert signed.predictor == base.predictor
    assert signed.identification == FixedCutpointVariance(value=1.0)
    assert signed.priors == base.priors


def test_the_two_scalar_rung_adds_only_root_entropy_to_the_signed_rung() -> None:
    _, ablations = ladder_index()
    base = rung("p_gram_only")
    signed = rung("p_gram_signed")
    entropy = rung("p_gram_entropy")

    assert "p_gram_entropy" in ablations["p_gram_only"]
    assert differing_fields(base, entropy) == {"identification", "predictor[2]"}
    assert entropy.predictor == (*signed.predictor, ROOT_ENTROPY)
    assert entropy.identification == signed.identification
    assert entropy.priors == signed.priors


def test_the_annotator_model_is_the_ordinal_two_parameter_model() -> None:
    two_parameter = rung(annotator_model())

    assert annotator_model() == "ordinal_2pl"
    assert two_parameter.family == Cumulative(
        discrimination=True, discrimination_group="item"
    )
    assert two_parameter.predictor == (
        PARTICIPANT,
        RandomIntercept(group="item"),
    )
    assert two_parameter.groups == ("participant", "item")
    assert two_parameter.identification == FixedCutpointVariance(value=1.0)
    assert two_parameter.priors.discrimination == DISCRIMINATION_PRIOR
    assert two_parameter.levels == 7


def test_no_rung_has_a_module_of_its_own() -> None:
    rungs, ablations = ladder_index()
    names = (
        frozenset(rungs)
        | frozenset(name for names in ablations.values() for name in names)
        | {annotator_model()}
    )
    stems = {
        path.stem.replace("-", "_")
        for path in PACKAGE.rglob("*.py")
        if "__pycache__" not in path.parts
    }

    assert not names & stems


DISTRIBUTION = (0.1, 0.2, 0.4, 0.2, 0.1)
RECORDED = 2


def prediction(
    *,
    distribution: tuple[float, ...] = DISTRIBUTION,
    response: int = RECORDED,
    convention: HeldOutParticipantPolicy | None = HeldOutParticipantPolicy.MARGINAL,
) -> ResponsePrediction:
    return ResponsePrediction(
        participant="ann",
        item="item-3",
        response=response,
        distribution=distribution,
        convention=convention,
        agreement=AgreementBin.LOW,
    )


def diagnostics(
    method: InferenceMethod = InferenceMethod.NUTS,
    **numbers: float | None,
) -> FitDiagnostics:
    return FitDiagnostics(
        method=method,
        divergences=0,
        max_rhat=1.01,
        min_ess=400.0,
        variance_components={"participant.latent": 0.8, "item.latent": 0.3},
        **numbers,
    )


HANDLE = PosteriorHandle(location="fits/reference.nc", chains=4, draws=1000)
PROFILE = ScaleProfile(
    scales=(0.5, 1.0, 2.0),
    log_posterior=(-3.0, 0.0, -2.0),
    minimum_tail_drop=2.0,
)
COST = ConstraintCost(
    constrained=Measurement(value=-3.0, n=250, unit="item"),
    unconstrained=Measurement(value=-1.5, n=250, unit="item"),
)


def summary(
    *,
    spec_hash: str = "3f2a",
    method: InferenceMethod = InferenceMethod.NUTS,
    posterior: PosteriorHandle | None = HANDLE,
    profile: ScaleProfile = PROFILE,
    cost: ConstraintCost = COST,
) -> FitSummary:
    return FitSummary(
        spec_hash=spec_hash,
        posterior=posterior,
        diagnostics=diagnostics(method),
        identification=PINNED,
        scale_profile=profile,
        constraint_cost=cost,
    )


def measurement(**changes: JsonValue) -> Measurement:
    fields: dict[str, JsonValue] = {
        "value": 0.5,
        "n": 10,
        "unit": "item",
        "lower": 0.4,
        "upper": 0.6,
        "level": 0.9,
    }
    fields.update(changes)

    return Measurement.model_validate(fields)


def test_a_measurement_reads_back_its_value_count_unit_and_interval() -> None:
    measured = measurement()

    assert measured.value == 0.5
    assert measured.n == 10
    assert measured.unit == "item"
    assert (measured.lower, measured.upper) == (0.4, 0.6)
    assert measured.level == 0.9


def test_a_measurement_without_an_interval_has_no_bounds_and_no_level() -> None:
    bare = Measurement(value=0.5, n=10, unit="item")

    assert (bare.lower, bare.upper, bare.level) == (None, None, None)


def test_a_measurement_over_no_units_has_no_value() -> None:
    absent = Measurement(value=None, n=0, unit="item")

    assert absent.value is None

    with pytest.raises(dx.ValidationError, match="over no units reports a value"):
        Measurement(value=0.0, n=0, unit="item")


def test_a_measurement_may_be_undefined_over_some_units() -> None:
    assert Measurement(value=None, n=3, unit="participant").n == 3


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"value": None}, "reports an interval"),
        ({"lower": None}, "interval is incomplete"),
        ({"upper": None}, "interval is incomplete"),
        ({"level": None}, "interval is incomplete"),
        ({"lower": None, "upper": None}, "interval is incomplete"),
        ({"lower": 0.6}, "must bracket"),
        ({"upper": 0.4}, "must bracket"),
        ({"lower": 0.6, "upper": 0.4}, "must bracket"),
        ({"level": 0.0}, "not a proportion"),
        ({"level": 1.0}, "not a proportion"),
        ({"level": -0.5}, "not a proportion"),
        ({"level": math.nan}, "coverage level is not finite"),
        ({"value": math.nan}, "value is not finite"),
        ({"lower": -math.inf}, "lower bound is not finite"),
        ({"upper": math.inf}, "upper bound is not finite"),
        ({"n": -1}, "nonnegative"),
        ({"unit": ""}, "names its unit"),
    ],
)
def test_an_inconsistent_measurement_is_refused(
    changes: dict[str, JsonValue], message: str
) -> None:
    with pytest.raises(dx.ValidationError, match=message):
        measurement(**changes)


def test_a_measurement_needs_its_unit() -> None:
    fields = measurement().model_dump()
    del fields["unit"]

    with pytest.raises(dx.ValidationError, match="required"):
        Measurement.model_validate(fields)


def test_a_measurement_may_sit_on_either_end_of_its_interval() -> None:
    assert measurement(value=0.4).value == 0.4
    assert measurement(value=0.6).value == 0.6


def test_a_measurement_round_trips_through_its_wire_form() -> None:
    measured = measurement()

    assert Measurement.model_validate_json(measured.model_dump_json()) == measured


def test_a_prediction_reads_back_the_distribution_density_and_convention() -> None:
    predicted = prediction()

    assert predicted.participant == "ann"
    assert predicted.item == "item-3"
    assert predicted.response == RECORDED
    assert predicted.distribution == DISTRIBUTION
    assert predicted.log_density == math.log(DISTRIBUTION[RECORDED])
    assert predicted.convention is HeldOutParticipantPolicy.MARGINAL
    assert predicted.agreement is AgreementBin.LOW


def test_a_prediction_for_a_participant_in_the_fit_names_no_convention() -> None:
    assert prediction(convention=None).convention is None

    fields = prediction().model_dump()
    del fields["convention"]

    with pytest.raises(dx.ValidationError, match="required"):
        ResponsePrediction.model_validate(fields)


@pytest.mark.parametrize(
    "policy", [HeldOutParticipantPolicy.MARGINAL, HeldOutParticipantPolicy.PLUG_IN]
)
def test_a_prediction_records_either_held_out_convention(
    policy: HeldOutParticipantPolicy,
) -> None:
    assert prediction(convention=policy).convention is policy


def test_a_prediction_tolerates_rounding_in_its_distribution() -> None:
    nearly = (0.1, 0.2, 0.4, 0.2, 0.1 + 5e-7)

    assert prediction(distribution=nearly).distribution == nearly


def test_the_log_density_is_derived_from_the_recorded_level() -> None:
    for response, probability in enumerate(DISTRIBUTION):
        assert prediction(response=response).log_density == math.log(probability)


def test_a_wire_form_cannot_overstate_the_log_density() -> None:
    fields = {**prediction().model_dump(), "log_density": 0.0}

    assert ResponsePrediction.model_validate(fields).log_density == math.log(
        DISTRIBUTION[RECORDED]
    )


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"distribution": (0.5, 0.5, 0.5)}, "sum to 1.5"),
        ({"distribution": (0.4, 0.4, 0.1)}, "sum to 0.9"),
        ({"distribution": (1.2, -0.2), "response": 0}, "outside the unit interval"),
        (
            {"distribution": (1.0 + 5e-7, 0.0), "response": 0},
            "outside the unit interval",
        ),
        ({"distribution": (0.5, 0.5), "response": 2}, "lies outside a distribution"),
        ({"distribution": (0.5, 0.5, 0.0), "response": 2}, "has probability zero"),
        ({"response": -1}, "nonnegative"),
        ({"distribution": (1.0,), "response": 0}, "at least two"),
    ],
)
def test_an_improper_prediction_is_refused(
    fields: dict[str, JsonValue], message: str
) -> None:
    with pytest.raises(dx.ValidationError, match=message):
        ResponsePrediction.model_validate({**prediction().model_dump(), **fields})


@pytest.mark.parametrize(
    ("blank", "message"),
    [("participant", "names its participant"), ("item", "names its item")],
)
def test_a_prediction_needs_a_participant_and_an_item(blank: str, message: str) -> None:
    fields = prediction().model_dump()
    fields[blank] = ""

    with pytest.raises(dx.ValidationError, match=message):
        ResponsePrediction.model_validate(fields)


def test_a_prediction_round_trips_through_its_wire_form() -> None:
    predicted = prediction(convention=HeldOutParticipantPolicy.PLUG_IN)
    wire = predicted.model_dump_json()

    assert '"plug_in"' in wire
    assert '"low"' in wire
    assert '"log_density"' in wire
    assert ResponsePrediction.model_validate_json(wire) == predicted


def test_a_posterior_handle_names_where_the_draws_are() -> None:
    handle = PosteriorHandle(location="fits/reference.nc", chains=4, draws=1000)

    assert handle.location == "fits/reference.nc"
    assert (handle.chains, handle.draws) == (4, 1000)


@pytest.mark.parametrize(
    "fields",
    [
        {"location": "", "chains": 4, "draws": 1000},
        {"location": "fits/reference.nc", "chains": 0, "draws": 1000},
        {"location": "fits/reference.nc", "chains": 4, "draws": 0},
    ],
)
def test_a_posterior_handle_without_draws_is_refused(
    fields: dict[str, JsonValue],
) -> None:
    with pytest.raises(dx.ValidationError):
        PosteriorHandle.model_validate(fields)


def test_diagnostics_read_back_the_method_and_the_variance_components() -> None:
    fitted = diagnostics()

    assert fitted.method is InferenceMethod.NUTS
    assert fitted.divergences == 0
    assert fitted.max_rhat == 1.01
    assert fitted.min_ess == 400.0
    assert fitted.variance_components == {
        "participant.latent": 0.8,
        "item.latent": 0.3,
    }


def test_diagnostics_default_to_no_numbers_and_no_components() -> None:
    bare = FitDiagnostics(method=InferenceMethod.MAP)

    assert bare.divergences is None
    assert bare.max_rhat is None
    assert bare.min_ess is None
    assert bare.variance_components == {}


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"divergences": -1}, "divergence count is negative"),
        ({"max_rhat": math.nan}, "R-hat is not finite"),
        ({"min_ess": -1.0}, "sample size is negative"),
        ({"min_ess": math.inf}, "sample size is not finite"),
        ({"variance_components": {"item.latent": -0.1}}, "is not a variance"),
        ({"variance_components": {"item.latent": math.nan}}, "is not a variance"),
    ],
)
def test_diagnostics_refuse_a_number_that_cannot_be_one(
    changes: dict[str, JsonValue], message: str
) -> None:
    with pytest.raises(dx.ValidationError, match=message):
        FitDiagnostics.model_validate({**diagnostics().model_dump(), **changes})


def test_a_fit_summary_reads_back_its_handle_diagnostics_and_constraint() -> None:
    fitted = summary()

    assert fitted.spec_hash == "3f2a"
    assert fitted.posterior == PosteriorHandle(
        location="fits/reference.nc", chains=4, draws=1000
    )
    assert fitted.diagnostics.method is InferenceMethod.NUTS
    assert fitted.identification == PINNED
    assert fitted.scale_profile == PROFILE
    assert fitted.constraint_cost.value == 1.5
    assert fitted.constraint_cost.as_measurement().n == 250


@pytest.mark.parametrize("field", ["scale_profile", "constraint_cost"])
def test_every_fit_summary_requires_both_identification_results(field: str) -> None:
    fields = summary().model_dump()
    del fields[field]

    with pytest.raises(dx.ValidationError, match="required"):
        FitSummary.model_validate(fields)


def test_a_point_estimate_has_no_posterior_draws() -> None:
    fitted = summary(method=InferenceMethod.MAP, posterior=None)

    assert fitted.posterior is None

    with pytest.raises(dx.ValidationError, match="point estimate reports a handle"):
        summary(method=InferenceMethod.MAP)


@pytest.mark.parametrize(
    "method",
    [
        InferenceMethod.NUTS,
        InferenceMethod.FULL_RANK,
        InferenceMethod.STRUCTURED,
        InferenceMethod.MEAN_FIELD,
    ],
)
def test_a_sampling_fit_reports_its_posterior_draws(method: InferenceMethod) -> None:
    assert summary(method=method).diagnostics.method is method

    with pytest.raises(dx.ValidationError, match=f"by {method.value} draws"):
        summary(method=method, posterior=None)


@pytest.mark.parametrize("method", tuple(InferenceMethod))
def test_a_ridged_profile_forbids_a_posterior_handle_under_every_method(
    method: InferenceMethod,
) -> None:
    ridged = ScaleProfile(
        scales=(0.5, 1.0, 2.0),
        log_posterior=(-0.5, 0.0, -0.5),
        minimum_tail_drop=1.0,
    )

    assert summary(method=method, posterior=None, profile=ridged).posterior is None

    with pytest.raises(dx.ValidationError, match="ridged scale profile"):
        summary(method=method, posterior=HANDLE, profile=ridged)


def test_a_fit_summary_needs_a_spec_hash() -> None:
    with pytest.raises(dx.ValidationError, match="names its spec"):
        summary(spec_hash="")


def test_a_fit_summary_round_trips_through_its_wire_form() -> None:
    fitted = summary()
    restored = FitSummary.model_validate_json(fitted.model_dump_json())

    assert restored == fitted
    assert restored.identification == PINNED
    assert restored.diagnostics.variance_components == {
        "participant.latent": 0.8,
        "item.latent": 0.3,
    }
    assert restored.scale_profile.lower_tail_drop == 3.0
    assert restored.scale_profile.upper_tail_drop == 2.0
    assert restored.scale_profile.ridge is False
    assert restored.constraint_cost.value == 1.5


REGISTERED: Final = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def criterion(
    *,
    registered_at: str = REGISTERED.isoformat(),
    minimum_interquartile_range: float = 1.0,
    ceiling: float = 10.0,
    maximum_ceiling_fraction: float = 0.9,
) -> DynamicRangeCriterion:
    return DynamicRangeCriterion(
        registered_at=registered_at,
        minimum_interquartile_range=minimum_interquartile_range,
        ceiling=ceiling,
        maximum_ceiling_fraction=maximum_ceiling_fraction,
    )


def report(
    *,
    registered: DynamicRangeCriterion | None = None,
    inventory: str = "ccg-core",
    observed_at: str = (REGISTERED + timedelta(hours=1)).isoformat(),
    items: int = 120,
    quartiles: tuple[float, float, float, float, float] = (-4.0, -1.5, 0.0, 1.5, 4.0),
    ceiling_fraction: float = 0.0,
) -> DynamicRangeReport:
    minimum, lower, median, upper, maximum = quartiles

    return DynamicRangeReport(
        criterion=criterion() if registered is None else registered,
        inventory=inventory,
        observed_at=observed_at,
        items=items,
        minimum=minimum,
        lower_quartile=lower,
        median=median,
        upper_quartile=upper,
        maximum=maximum,
        ceiling_fraction=ceiling_fraction,
    )


def test_a_criterion_reads_back_its_registration_and_bounds() -> None:
    registered = criterion()

    assert registered.registration_time() == REGISTERED
    assert registered.minimum_interquartile_range == 1.0
    assert registered.ceiling == 10.0
    assert registered.maximum_ceiling_fraction == 0.9


def test_a_criterion_a_constant_passes_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="constant"):
        criterion(minimum_interquartile_range=0.0)

    with pytest.raises(dx.ValidationError, match="constant"):
        criterion(minimum_interquartile_range=-1.0)


def test_a_criterion_every_distribution_passes_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="every distribution"):
        criterion(maximum_ceiling_fraction=1.0)

    with pytest.raises(dx.ValidationError, match="proportion"):
        criterion(maximum_ceiling_fraction=-0.1)


def test_a_criterion_with_an_infinite_bound_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="not finite"):
        criterion(minimum_interquartile_range=math.inf)

    with pytest.raises(dx.ValidationError, match="not finite"):
        criterion(ceiling=math.inf)


def test_a_criterion_registered_without_an_offset_or_a_date_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="UTC offset"):
        criterion(registered_at="2026-01-02T03:04:05")

    with pytest.raises(dx.ValidationError, match="ISO 8601"):
        criterion(registered_at="yesterday")


def test_a_criterion_round_trips_through_its_wire_form() -> None:
    registered = criterion()

    assert (
        DynamicRangeCriterion.model_validate_json(registered.model_dump_json())
        == registered
    )


def test_a_report_computes_its_spread_and_its_verdict() -> None:
    passing = report()

    assert passing.interquartile_range == 3.0
    assert passing.failures == ()
    assert passing.passed


def test_a_report_names_the_clause_it_fails() -> None:
    narrow = report(quartiles=(-4.0, -0.25, 0.0, 0.25, 4.0))
    piled = report(ceiling_fraction=0.95)

    assert not narrow.passed
    assert len(narrow.failures) == 1
    assert "interquartile range" in narrow.failures[0]
    assert not piled.passed
    assert len(piled.failures) == 1
    assert "at or above" in piled.failures[0]


def test_a_report_at_the_bounds_passes() -> None:
    at_bounds = report(quartiles=(-1.0, -0.5, 0.0, 0.5, 1.0), ceiling_fraction=0.9)

    assert at_bounds.interquartile_range == 1.0
    assert at_bounds.passed


def test_a_report_with_disordered_quartiles_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="lower quartile"):
        report(quartiles=(0.0, -1.0, 0.0, 1.0, 2.0))

    with pytest.raises(dx.ValidationError, match="median"):
        report(quartiles=(-1.0, 0.5, 0.0, 1.0, 2.0))

    with pytest.raises(dx.ValidationError, match="upper quartile"):
        report(quartiles=(-1.0, -0.5, 1.5, 1.0, 2.0))

    with pytest.raises(dx.ValidationError, match="maximum"):
        report(quartiles=(-1.0, -0.5, 0.0, 1.0, 0.5))


def test_a_report_over_no_items_or_an_unnamed_inventory_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="at least one item"):
        report(items=0)

    with pytest.raises(dx.ValidationError, match="names its inventory"):
        report(inventory="")


def test_a_report_with_an_infinite_bound_or_an_improper_fraction_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="not finite"):
        report(quartiles=(-math.inf, -1.0, 0.0, 1.0, 2.0))

    with pytest.raises(dx.ValidationError, match="proportion"):
        report(ceiling_fraction=1.5)


def test_a_report_observed_before_its_criterion_was_registered_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="registered after"):
        report(observed_at=(REGISTERED - timedelta(seconds=1)).isoformat())

    with pytest.raises(dx.ValidationError, match="UTC offset"):
        report(observed_at="2026-01-02T05:04:05")


def test_a_report_round_trips_through_its_wire_form() -> None:
    reported = report(ceiling_fraction=0.95)
    restored = DynamicRangeReport.model_validate_json(reported.model_dump_json())

    assert restored == reported
    assert restored.failures == reported.failures
    assert not restored.passed


INVENTORY = ColumnInventory(
    primitive_types=("NP", "S_that", "VP_to"),
    null_type="NONE",
    levels=(
        ColumnLevel(label="_ NP", arguments=("NP",)),
        ColumnLevel(label="_ NP S_that", arguments=("NP", "S_that")),
        ColumnLevel(label="_ NP VP_to", arguments=("NP", "VP_to")),
    ),
)
LEXICAL = LexicalPrimitiveInventory(names=("cause", "attitude"))
LIFTS = (
    MemorizedCells(),
    ColumnLabels(depth=1),
    ColumnLabels(depth=2),
    JointTypeTensor(),
    LexicalPossession(lexical_primitives=LEXICAL),
    RowClasses(classes=3),
    PossessionAndClasses(lexical_primitives=LEXICAL, classes=3),
)
TABLE_SCHEDULE = FitSchedule(
    optimizer=AdamOptimizerSpec(learning_rate=0.05, weight_decay=0.005),
    steps=400,
    batch_size=None,
    initial_scale=0.5,
)


def table_spec(lift: ColumnLift) -> TableAblationSpec:
    return TableAblationSpec(
        rows=TableRows(attribute="verb", levels=("think", "tell")),
        columns=TableColumns(attribute="frame", inventory=INVENTORY),
        width=4,
        lift=lift,
        schedule=TABLE_SCHEDULE,
    )


def test_an_inventory_assigns_the_null_type_an_index_no_primitive_type_holds() -> None:
    assert INVENTORY.null_index == len(INVENTORY.primitive_types)
    assert INVENTORY.type_count == len(INVENTORY.primitive_types) + 1
    assert INVENTORY.null_index not in range(len(INVENTORY.primitive_types))


def test_an_inventory_pads_every_column_level_to_the_widest_argument_sequence() -> None:
    assert INVENTORY.arity == 2
    assert INVENTORY.labels == ("_ NP", "_ NP S_that", "_ NP VP_to")


def test_an_inventory_repeating_a_name_or_typing_no_argument_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="declared twice"):
        ColumnInventory(
            primitive_types=("NP", "NP"),
            null_type="NONE",
            levels=(ColumnLevel(label="_ NP", arguments=("NP",)),),
        )

    with pytest.raises(dx.ValidationError, match="repeats a primitive type"):
        ColumnInventory(
            primitive_types=("NP",),
            null_type="NP",
            levels=(ColumnLevel(label="_ NP", arguments=("NP",)),),
        )

    with pytest.raises(dx.ValidationError, match="declared twice"):
        ColumnInventory(
            primitive_types=("NP",),
            null_type="NONE",
            levels=(
                ColumnLevel(label="_ NP", arguments=("NP",)),
                ColumnLevel(label="_ NP", arguments=("NP",)),
            ),
        )

    with pytest.raises(dx.ValidationError, match="does not declare"):
        ColumnInventory(
            primitive_types=("NP",),
            null_type="NONE",
            levels=(ColumnLevel(label="_ S", arguments=("S_that",)),),
        )

    with pytest.raises(dx.ValidationError, match="none decomposes"):
        ColumnInventory(
            primitive_types=("NP",),
            null_type="NONE",
            levels=(ColumnLevel(label="_", arguments=()),),
        )


def test_the_lexical_primitives_are_a_second_inventory_with_its_own_names() -> None:
    collided = LexicalPrimitiveInventory(names=INVENTORY.primitive_types)

    assert LEXICAL.count == 2
    assert collided.count == len(INVENTORY.primitive_types)
    assert frozenset(LexicalPrimitiveInventory.__field_specs__) == {"names"}
    assert "names" not in ColumnInventory.__field_specs__
    assert "primitive_types" not in LexicalPrimitiveInventory.__field_specs__


def test_a_lexical_primitive_inventory_repeating_a_name_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="declared twice"):
        LexicalPrimitiveInventory(names=("cause", "cause"))

    with pytest.raises(dx.ValidationError, match="has no name"):
        LexicalPrimitiveInventory(names=("cause", ""))


def test_only_the_lifts_that_read_an_argument_sequence_say_they_decompose() -> None:
    decomposing = {lift.lift for lift in LIFTS if lift.decomposes}

    assert decomposing == {
        "joint_type",
        "lexical_possession",
        "row_classes",
        "possession_and_classes",
    }


def test_each_lift_declares_the_row_side_latents_its_links_are_chosen_for() -> None:
    mediators = {lift.lift: lift.mediators for lift in LIFTS}

    assert mediators["memorized"] == ()
    assert mediators["joint_type"] == ()
    assert mediators["lexical_possession"] == ("possession",)
    assert mediators["row_classes"] == ("membership",)
    assert mediators["possession_and_classes"] == ("membership", "possession")


def test_a_lift_with_no_layer_or_a_single_row_class_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="has a layer"):
        ColumnLabels(depth=0)

    with pytest.raises(dx.ValidationError, match="two classes"):
        RowClasses(classes=1)

    with pytest.raises(dx.ValidationError, match="two classes"):
        PossessionAndClasses(lexical_primitives=LEXICAL, classes=1)


def test_a_pinned_prefix_names_its_target_mode_and_cardinality() -> None:
    primitive = PinnedPrefix(
        target="semantic_primitives",
        mode="independent_sigmoid",
        cardinality=1,
    )
    cluster = PinnedPrefix(
        target="item_clusters",
        mode="normalized_softmax",
        cardinality=1,
    )

    assert primitive.target == "semantic_primitives"
    assert primitive.mode == "independent_sigmoid"
    assert primitive.cardinality == 1
    assert cluster.target == "item_clusters"
    assert cluster.mode == "normalized_softmax"


def test_a_pinned_prefix_refuses_an_unknown_target_or_mode_by_name() -> None:
    with pytest.raises(dx.ValidationError, match="target"):
        PinnedPrefix.model_validate(
            {
                "target": "frames",
                "mode": "independent_sigmoid",
                "cardinality": 1,
            }
        )

    with pytest.raises(dx.ValidationError, match="mode"):
        PinnedPrefix.model_validate(
            {
                "target": "semantic_primitives",
                "mode": "sigmoid",
                "cardinality": 1,
            }
        )


def test_a_pinned_prefix_refuses_a_mode_belonging_to_the_other_target() -> None:
    with pytest.raises(dx.ValidationError, match=r"target.*requires mode"):
        PinnedPrefix(
            target="semantic_primitives",
            mode="normalized_softmax",
            cardinality=1,
        )

    with pytest.raises(dx.ValidationError, match=r"target.*requires mode"):
        PinnedPrefix(
            target="item_clusters",
            mode="independent_sigmoid",
            cardinality=1,
        )


@pytest.mark.parametrize("cardinality", [-1, True])
def test_a_pinned_prefix_refuses_an_invalid_cardinality_by_name(
    cardinality: int,
) -> None:
    with pytest.raises(dx.ValidationError, match="cardinality"):
        PinnedPrefix(
            target="semantic_primitives",
            mode="independent_sigmoid",
            cardinality=cardinality,
        )


def test_a_lift_routes_each_prefix_to_the_inventory_it_names() -> None:
    primitive = PinnedPrefix(
        target="semantic_primitives",
        mode="independent_sigmoid",
        cardinality=1,
    )
    cluster = PinnedPrefix(
        target="item_clusters",
        mode="normalized_softmax",
        cardinality=1,
    )

    with pytest.raises(dx.ValidationError, match=r"target.*item_clusters"):
        LexicalPossession(lexical_primitives=LEXICAL, prefix=cluster)

    with pytest.raises(dx.ValidationError, match=r"target.*semantic_primitives"):
        RowClasses(classes=3, prefix=primitive)

    with pytest.raises(dx.ValidationError, match=r"target.*semantic_primitives"):
        PossessionAndClasses(
            lexical_primitives=LEXICAL,
            classes=3,
            primitive_prefix=cluster,
            cluster_prefix=cluster,
        )


def test_a_prefix_cardinality_may_fill_but_not_exceed_its_inventory() -> None:
    primitive = PinnedPrefix(
        target="semantic_primitives",
        mode="independent_sigmoid",
        cardinality=LEXICAL.count,
    )
    cluster = PinnedPrefix(
        target="item_clusters",
        mode="normalized_softmax",
        cardinality=3,
    )

    assert LexicalPossession(lexical_primitives=LEXICAL, prefix=primitive).prefix
    assert RowClasses(classes=3, prefix=cluster).prefix

    with pytest.raises(dx.ValidationError, match=r"cardinality.*inventory"):
        LexicalPossession(
            lexical_primitives=LEXICAL,
            prefix=PinnedPrefix(
                target="semantic_primitives",
                mode="independent_sigmoid",
                cardinality=LEXICAL.count + 1,
            ),
        )

    with pytest.raises(dx.ValidationError, match=r"cardinality.*inventory"):
        RowClasses(
            classes=3,
            prefix=PinnedPrefix(
                target="item_clusters",
                mode="normalized_softmax",
                cardinality=4,
            ),
        )


def test_a_comparison_refuses_more_special_rows_than_row_levels() -> None:
    prefix = PinnedPrefix(
        target="semantic_primitives",
        mode="independent_sigmoid",
        cardinality=2,
    )

    with pytest.raises(dx.ValidationError, match=r"cardinality.*row cardinality"):
        TableAblationSpec(
            rows=TableRows(attribute="verb", levels=("think",)),
            columns=TableColumns(attribute="frame", inventory=INVENTORY),
            width=4,
            lift=LexicalPossession(lexical_primitives=LEXICAL, prefix=prefix),
            schedule=TABLE_SCHEDULE,
        )


def test_a_full_cluster_prefix_requires_every_row_level_to_be_special() -> None:
    prefix = PinnedPrefix(
        target="item_clusters",
        mode="normalized_softmax",
        cardinality=2,
    )

    with pytest.raises(dx.ValidationError, match=r"cardinality.*regular rows"):
        TableAblationSpec(
            rows=TableRows(attribute="verb", levels=("think", "tell", "promise")),
            columns=TableColumns(attribute="frame", inventory=INVENTORY),
            width=4,
            lift=RowClasses(classes=2, prefix=prefix),
            schedule=TABLE_SCHEDULE,
        )

    all_special = TableAblationSpec(
        rows=TableRows(attribute="verb", levels=("think", "tell")),
        columns=TableColumns(attribute="frame", inventory=INVENTORY),
        width=4,
        lift=RowClasses(classes=2, prefix=prefix),
        schedule=TABLE_SCHEDULE,
    )

    assert all_special.lift == RowClasses(classes=2, prefix=prefix)


def test_prefix_wire_forms_and_digests_distinguish_each_choice() -> None:
    prefixes = (
        PinnedPrefix(
            target="semantic_primitives",
            mode="independent_sigmoid",
            cardinality=0,
        ),
        PinnedPrefix(
            target="semantic_primitives",
            mode="independent_sigmoid",
            cardinality=1,
        ),
        PinnedPrefix(
            target="item_clusters",
            mode="normalized_softmax",
            cardinality=1,
        ),
    )

    restored = tuple(
        PinnedPrefix.model_validate_json(prefix.model_dump_json())
        for prefix in prefixes
    )

    assert restored == prefixes
    assert len({spec_digest(prefix) for prefix in prefixes}) == len(prefixes)


def test_default_unpinned_lifts_round_trip_with_their_explicit_routes() -> None:
    possession = LexicalPossession(lexical_primitives=LEXICAL)
    classes = RowClasses(classes=3)
    composite = PossessionAndClasses(lexical_primitives=LEXICAL, classes=3)

    assert possession.prefix == PinnedPrefix(
        target="semantic_primitives",
        mode="independent_sigmoid",
        cardinality=0,
    )
    assert classes.prefix == PinnedPrefix(
        target="item_clusters", mode="normalized_softmax", cardinality=0
    )
    assert composite.primitive_prefix == possession.prefix
    assert composite.cluster_prefix == classes.prefix

    for lift in (possession, classes, composite):
        assert ColumnLift.model_validate_json(lift.model_dump_json()) == lift


def test_unpinned_lift_wire_forms_may_omit_their_zero_cardinality_prefixes() -> None:
    wires = (
        json.dumps(
            {
                "lift": "lexical_possession",
                "lexical_primitives": {"names": ["cause", "attitude"]},
            }
        ),
        json.dumps({"lift": "row_classes", "classes": 3}),
        json.dumps(
            {
                "lift": "possession_and_classes",
                "lexical_primitives": {"names": ["cause", "attitude"]},
                "classes": 3,
            }
        ),
    )
    expected = (
        LexicalPossession(lexical_primitives=LEXICAL),
        RowClasses(classes=3),
        PossessionAndClasses(lexical_primitives=LEXICAL, classes=3),
    )

    assert tuple(ColumnLift.model_validate_json(wire) for wire in wires) == expected


def test_composite_prefix_routing_changes_the_canonical_digest() -> None:
    base = PossessionAndClasses(lexical_primitives=LEXICAL, classes=3)
    primitive = PossessionAndClasses(
        lexical_primitives=LEXICAL,
        classes=3,
        primitive_prefix=PinnedPrefix(
            target="semantic_primitives",
            mode="independent_sigmoid",
            cardinality=1,
        ),
    )
    cluster = PossessionAndClasses(
        lexical_primitives=LEXICAL,
        classes=3,
        cluster_prefix=PinnedPrefix(
            target="item_clusters",
            mode="normalized_softmax",
            cardinality=1,
        ),
    )

    variants = (base, primitive, cluster)

    assert len({spec_digest(variant) for variant in variants}) == 3

    for variant in variants:
        restored = ColumnLift.model_validate_json(variant.model_dump_json())

        assert restored == variant
        assert spec_digest(restored) == spec_digest(variant)


def test_every_rung_of_the_comparison_declares_the_one_crossed_term() -> None:
    terms = {table_spec(lift).terms for lift in LIFTS}

    assert len(terms) == 1
    assert terms == {
        (InteractionTerm(left=FactorTerm(name="verb"), right=FactorTerm(name="frame")),)
    }


def test_a_comparison_repeating_a_row_or_crossing_one_attribute_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="declared twice"):
        TableAblationSpec(
            rows=TableRows(attribute="verb", levels=("think", "think")),
            columns=TableColumns(attribute="frame", inventory=INVENTORY),
            width=4,
            lift=JointTypeTensor(),
            schedule=TABLE_SCHEDULE,
        )

    with pytest.raises(dx.ValidationError, match="crosses them"):
        TableAblationSpec(
            rows=TableRows(attribute="cell", levels=("think",)),
            columns=TableColumns(attribute="cell", inventory=INVENTORY),
            width=4,
            lift=JointTypeTensor(),
            schedule=TABLE_SCHEDULE,
        )

    with pytest.raises(dx.ValidationError, match="names its row attribute"):
        TableRows(attribute="", levels=("think",))

    with pytest.raises(dx.ValidationError, match="has a width"):
        TableAblationSpec(
            rows=TableRows(attribute="verb", levels=("think",)),
            columns=TableColumns(attribute="frame", inventory=INVENTORY),
            width=0,
            lift=JointTypeTensor(),
            schedule=TABLE_SCHEDULE,
        )


def test_the_lifts_give_the_comparison_one_fingerprint_apiece() -> None:
    digests = {spec_digest(table_spec(lift)) for lift in LIFTS}

    assert len(digests) == len(LIFTS)


def test_a_comparison_round_trips_through_its_wire_form() -> None:
    for lift in LIFTS:
        declared = table_spec(lift)
        restored = TableAblationSpec.model_validate_json(declared.model_dump_json())

        assert restored == declared
        assert spec_digest(restored) == spec_digest(declared)


CHOICE_LINK: Final = LinkTerm(readout="p_gram", linking=LOGIT)
PINNED_LINK: Final = LinkTerm(
    readout="p_gram",
    linking=LinkingSpec(
        family="logit",
        columns={"complement": "p_gram_complement"},
        fixed={"slope": 1.0},
    ),
)
POSITION_TERM: Final = FactorTerm(name=POSITION)


def choice_spec(
    *terms: Predictor,
    family: Choice,
    levels: int = 2,
    identification: ScaleIdentification | None = None,
    priors: PriorBlock | None = None,
    normalization: ScaleNormalization = ScaleNormalization.IDENTITY,
) -> ResponseSpec:
    return ResponseSpec(
        family=family,
        levels=levels,
        objective=ConditionalLikelihood(),
        predictor=terms,
        identification=IdentityLinkOnLogit()
        if identification is None
        else identification,
        priors=PriorBlock() if priors is None else priors,
        normalization=normalization,
    )


def test_a_choice_family_counts_its_alternatives_and_names_what_it_draws() -> None:
    plain = Choice()
    tied = Choice(ties=True, discrimination=True)

    assert plain.alternatives(4) == 4
    assert plain.draws_thresholds is False
    assert plain.draws_discrimination is False
    assert tied.alternatives(3) == 2
    assert tied.draws_thresholds is True
    assert tied.draws_discrimination is True


def test_a_choice_family_varying_its_discrimination_draws_it() -> None:
    with pytest.raises(dx.ValidationError, match="holds it at one"):
        Choice(discrimination_group="participant")

    with pytest.raises(dx.ValidationError, match="names its grouping factor"):
        Choice(discrimination=True, discrimination_group="")


def test_a_tie_is_recorded_over_three_levels() -> None:
    with pytest.raises(dx.ValidationError, match="three levels they and it make"):
        choice_spec(
            CHOICE_LINK,
            family=Choice(ties=True),
            levels=4,
            priors=PriorBlock(thresholds=ThresholdPriors()),
        )


def test_a_choice_predictor_tells_the_alternatives_apart() -> None:
    with pytest.raises(dx.ValidationError, match="takes one value on all of them"):
        choice_spec(CHOICE_LINK, FactorTerm(name="verb"), family=Choice())

    with pytest.raises(dx.ValidationError, match="takes one value on all of them"):
        choice_spec(CHOICE_LINK, RandomIntercept(group="participant"), family=Choice())

    assert choice_spec(
        CHOICE_LINK,
        POSITION_TERM,
        RandomSlope(group="participant", term=CHOICE_LINK),
        family=Choice(),
    ).factors == (POSITION,)


def test_a_free_slope_and_a_free_bias_need_a_scale_identification() -> None:
    with pytest.raises(
        dx.ValidationError,
        match=r"leaves its slope free.*'position' factor leaves the position bias free",
    ):
        choice_spec(
            CHOICE_LINK,
            POSITION_TERM,
            family=Choice(discrimination=True),
            identification=FixedCutpointVariance(value=1.0),
            priors=PriorBlock(
                discrimination=PriorSpec(
                    distribution="LogNormal", parameters=(0.0, 1.0)
                )
            ),
        )


def test_a_pinned_slope_or_a_pinning_identification_leaves_the_scale_fixed() -> None:
    drawn = PriorBlock(
        discrimination=PriorSpec(distribution="LogNormal", parameters=(0.0, 1.0))
    )
    pinned = choice_spec(
        PINNED_LINK,
        POSITION_TERM,
        family=Choice(discrimination=True),
        identification=FixedCutpointVariance(value=1.0),
        levels=3,
        priors=drawn,
    )
    on_logit = choice_spec(
        CHOICE_LINK,
        POSITION_TERM,
        family=Choice(discrimination=True),
        priors=drawn,
    )

    assert pinned.family.draws_discrimination
    assert on_logit.identification == IdentityLinkOnLogit()


def test_a_choice_specification_round_trips_through_its_wire_form() -> None:
    declared = choice_spec(
        CHOICE_LINK,
        POSITION_TERM,
        family=Choice(
            ties=True, discrimination=True, discrimination_group="participant"
        ),
        levels=3,
        priors=PriorBlock(
            thresholds=ThresholdPriors(),
            discrimination=PriorSpec(distribution="LogNormal", parameters=(0.0, 1.0)),
        ),
    )
    restored = ResponseSpec.model_validate_json(declared.model_dump_json())

    assert restored == declared
    assert spec_digest(restored) == spec_digest(declared)


ATTENTION = FactorTerm(name="attention")
ATTENTION_SLOPE = RandomSlope(group="participant", term=ATTENTION)
DISCRIMINATION_FORMULA = ParameterFormula(
    parameter=AuxiliaryParameter.DISCRIMINATION, predictor=(ATTENTION,)
)
CUTPOINT_VARIANCE = FixedCutpointVariance(value=1.0)


def predicting_spec(
    *,
    family: ResponseFamily = CUMULATIVE,
    identification: ScaleIdentification = CUTPOINT_VARIANCE,
    priors: PriorBlock = THRESHOLD_PRIORS,
    parameters: tuple[ParameterFormula, ...] = (DISCRIMINATION_FORMULA,),
) -> ResponseSpec:
    return ResponseSpec(
        family=family,
        levels=7,
        objective=ConditionalLikelihood(),
        predictor=(P_GRAM,),
        identification=identification,
        priors=priors,
        parameters=parameters,
    )


def test_a_formula_reads_back_the_parameter_and_the_terms_it_was_given() -> None:
    formula = ParameterFormula(
        parameter=AuxiliaryParameter.DISCRIMINATION, predictor=(ATTENTION, VERB)
    )

    assert formula.parameter is AuxiliaryParameter.DISCRIMINATION
    assert formula.predictor == (ATTENTION, VERB)


def test_a_formula_over_no_term_at_all_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="at least one term"):
        ParameterFormula(parameter=AuxiliaryParameter.DISCRIMINATION, predictor=())


def test_a_formula_declaring_one_term_twice_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="twice"):
        ParameterFormula(
            parameter=AuxiliaryParameter.DISCRIMINATION,
            predictor=(ATTENTION, VERB, ATTENTION),
        )


def test_a_family_says_whether_it_has_a_discrimination_to_predict() -> None:
    assert Cumulative().takes_discrimination
    assert Cumulative(discrimination=True).takes_discrimination
    assert Choice().takes_discrimination
    assert not Metric().takes_discrimination
    assert not RankConsistent(scheme=RankConsistentScheme.CORAL).takes_discrimination


def test_a_predicted_parameter_stands_beside_the_latent_response_formula() -> None:
    spec = predicting_spec()

    assert spec.parameters == (DISCRIMINATION_FORMULA,)
    assert spec.predicted == (AuxiliaryParameter.DISCRIMINATION,)
    assert spec.predictor == (P_GRAM,)
    assert spec.terms == (P_GRAM, ATTENTION)


def test_a_specification_declaring_no_formula_predicts_no_parameter() -> None:
    spec = predicting_spec(parameters=())

    assert spec.parameters == ()
    assert spec.predicted == ()
    assert spec.terms == spec.predictor


def test_the_columns_a_specification_reads_cover_every_formula() -> None:
    spec = predicting_spec(
        parameters=(
            ParameterFormula(
                parameter=AuxiliaryParameter.DISCRIMINATION,
                predictor=(ATTENTION, LENGTH, ATTENTION_SLOPE),
            ),
        )
    )

    assert spec.readouts == ("p_gram", "p_gram_complement", "length")
    assert spec.factors == ("attention",)
    assert spec.groups == ("participant",)


def test_a_parameter_carrying_two_formulas_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="two linear predictors"):
        predicting_spec(
            parameters=(
                DISCRIMINATION_FORMULA,
                ParameterFormula(
                    parameter=AuxiliaryParameter.DISCRIMINATION, predictor=(VERB,)
                ),
            )
        )


def test_a_discrimination_formula_over_a_family_that_has_none_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="family has none"):
        predicting_spec(family=Metric(), priors=NO_PRIORS)


def test_a_discrimination_that_is_both_drawn_and_predicted_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="one discrimination and a linear"):
        predicting_spec(
            family=Cumulative(discrimination=True),
            priors=PriorBlock(
                thresholds=ThresholdPriors(), discrimination=DISCRIMINATION_PRIOR
            ),
        )


@pytest.mark.parametrize("identification", [PINNED, IdentityLinkOnLogit()])
def test_a_predicted_discrimination_takes_the_cutpoint_variance_constraint(
    identification: ScaleIdentification,
) -> None:
    with pytest.raises(dx.ValidationError, match="variance of the cutpoints"):
        predicting_spec(identification=identification)


def test_a_specification_carrying_a_formula_survives_its_wire_form() -> None:
    spec = predicting_spec()

    restored = ResponseSpec.model_validate_json(spec.model_dump_json())

    assert restored == spec
    assert restored.parameters == (DISCRIMINATION_FORMULA,)
    assert spec_digest(restored) == spec_digest(spec)


def a_fit_document(objective: JsonValue) -> dict[str, JsonValue]:
    response = json.loads((LADDER / "intercept_only.json").read_text(encoding="utf-8"))
    response["objective"] = objective

    return {
        "dataset": "mega-acceptability-v2",
        "response": {"ResponseDeclaration_responsespec": response},
        "inference": {
            "inference": "nuts",
            "warmup": 500,
            "retained": 1000,
            "chains": 4,
        },
        "elicitation": {"method": "participants", "version": "1"},
        "calibration": {
            "replicates": 100,
            "guide_iterations": 200,
            "guide_learning_rate": 0.05,
        },
        "comparison": None,
        "dynamic_range": None,
        "ablation": [],
        "interannotator": None,
    }


def ordinal_response(document: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Return the ordinal response body of a fit test document."""
    response = document["response"]

    if not isinstance(response, dict):
        msg = "the fit test document has no response union"
        raise TypeError(msg)

    body = response["ResponseDeclaration_responsespec"]

    if not isinstance(body, dict):
        msg = "the fit test document has no ordinal response body"
        raise TypeError(msg)

    return body


def validate_fit_document(document: dict[str, JsonValue]) -> JudgmentFitSpec:
    """Validate a fit document through its JSON boundary."""
    return JudgmentFitSpec.model_validate_json(json.dumps(document))


TRUNCATED_MARGINAL: Final[dict[str, JsonValue]] = {
    "objective": "truncated_marginal",
    "group": "participant",
    "radius": 6.0,
    "relative_tolerance": 1e-6,
    "absolute_tolerance": 1e-8,
    "quadrature_points": 1,
    "quadrature_seed": 0,
}


def test_a_judgment_fit_declares_its_response_inference_and_calibration() -> None:
    fit = validate_fit_document(a_fit_document({"objective": "conditional"}))

    assert fit.dataset == "mega-acceptability-v2"
    assert fit.response.family.programmed
    assert fit.inference == NutsInference(warmup=500, retained=1000, chains=4)
    assert fit.inference.method is InferenceMethod.NUTS
    assert fit.elicitation == Elicitation(
        method=ElicitationMethod.PARTICIPANTS, version="1"
    )
    assert fit.calibration == JudgmentFitCalibration(
        replicates=100, guide_iterations=200, guide_learning_rate=0.05
    )
    assert JudgmentFitSpec.model_validate_json(fit.model_dump_json()) == fit


def test_a_truncated_marginal_objective_is_refused_for_sampled_inference() -> None:
    document = a_fit_document(TRUNCATED_MARGINAL)

    with pytest.raises(dx.ValidationError, match="leaves its posterior unchanged"):
        validate_fit_document(document)

    document["calibration"] = None
    document["inference"] = SVI_INFERENCE

    with pytest.raises(dx.ValidationError, match="leaves its posterior unchanged"):
        validate_fit_document(document)


def test_a_map_fit_is_the_single_truncated_marginal_execution_arm() -> None:
    document = a_fit_document(TRUNCATED_MARGINAL)
    document["calibration"] = None
    document["inference"] = {
        "inference": "map",
        "optimizer": {"kind": "adam", "learning_rate": 0.02},
        "steps": 25,
        "initial_scale": 0.1,
    }

    fit = validate_fit_document(document)

    assert fit.inference == MapInference(
        optimizer=AdamOptimizerSpec(learning_rate=0.02),
        steps=25,
        initial_scale=0.1,
    )
    assert fit.inference.method is InferenceMethod.MAP
    assert JudgmentFitSpec.model_validate_json(fit.model_dump_json()) == fit


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("steps", 0, "not positive"),
        ("initial_scale", 0.0, "not positive"),
        ("initial_scale", math.inf, "not finite"),
    ],
)
def test_a_map_fit_refuses_invalid_numerical_controls(
    field: str, value: float, message: str
) -> None:
    document = a_fit_document({"objective": "conditional"})
    document["calibration"] = None
    document["inference"] = {
        "inference": "map",
        "optimizer": {"kind": "adam"},
        "steps": 5,
        "initial_scale": 0.1,
        field: value,
    }

    with pytest.raises(dx.ValidationError, match=message):
        validate_fit_document(document)


SVI_INFERENCE: Final[dict[str, JsonValue]] = {
    "inference": "svi",
    "guide": "structured",
    "iterations": 50,
    "draws": 20,
    "learning_rate": 0.05,
}


def test_a_variational_fit_declares_its_guide() -> None:
    document = a_fit_document({"objective": "conditional"})
    document["inference"] = SVI_INFERENCE
    document["calibration"] = None

    fit = validate_fit_document(document)

    assert fit.inference == VariationalInference(
        guide=InferenceMethod.STRUCTURED, iterations=50, draws=20, learning_rate=0.05
    )
    assert fit.inference.method is InferenceMethod.STRUCTURED
    assert JudgmentFitSpec.model_validate_json(fit.model_dump_json()) == fit


def test_a_variational_fit_declares_its_learning_rate() -> None:
    document = a_fit_document({"objective": "conditional"})
    document["inference"] = {
        key: value for key, value in SVI_INFERENCE.items() if key != "learning_rate"
    }
    document["calibration"] = None

    with pytest.raises(dx.ValidationError, match="learning_rate"):
        validate_fit_document(document)


@pytest.mark.parametrize("rate", [0.0, -0.05, math.inf, math.nan])
def test_a_variational_fit_refuses_a_rate_that_is_not_positive_and_finite(
    rate: float,
) -> None:
    document = a_fit_document({"objective": "conditional"})
    document["inference"] = {**SVI_INFERENCE, "learning_rate": rate}
    document["calibration"] = None

    with pytest.raises(dx.ValidationError, match="learning rate"):
        validate_fit_document(document)


def test_a_calibration_declares_its_guide_learning_rate() -> None:
    document = a_fit_document({"objective": "conditional"})
    document["calibration"] = {"replicates": 100, "guide_iterations": 200}

    with pytest.raises(dx.ValidationError, match="guide_learning_rate"):
        validate_fit_document(document)


@pytest.mark.parametrize("rate", [0.0, -0.05, math.inf, math.nan])
def test_a_calibration_refuses_a_guide_rate_that_is_not_positive_and_finite(
    rate: float,
) -> None:
    document = a_fit_document({"objective": "conditional"})
    document["calibration"] = {
        "replicates": 100,
        "guide_iterations": 200,
        "guide_learning_rate": rate,
    }

    with pytest.raises(dx.ValidationError, match="learning rate"):
        validate_fit_document(document)


@pytest.mark.parametrize("guide", ["nuts", "map", "full_rank"])
def test_a_variational_fit_refuses_a_method_that_is_not_its_guide(guide: str) -> None:
    document = a_fit_document({"objective": "conditional"})
    document["inference"] = {**SVI_INFERENCE, "guide": guide}
    document["calibration"] = None

    with pytest.raises(dx.ValidationError, match="declares one of the guides"):
        validate_fit_document(document)


def test_calibration_beside_a_variational_fit_is_refused() -> None:
    document = a_fit_document({"objective": "conditional"})
    document["inference"] = SVI_INFERENCE

    with pytest.raises(dx.ValidationError, match="under the fit's NUTS plan"):
        validate_fit_document(document)


def test_calibration_beside_a_map_fit_is_refused() -> None:
    document = a_fit_document({"objective": "conditional"})
    document["inference"] = {
        "inference": "map",
        "optimizer": {"kind": "adam"},
        "steps": 5,
        "initial_scale": 0.1,
    }

    with pytest.raises(dx.ValidationError, match="under the fit's NUTS plan"):
        validate_fit_document(document)


def test_a_changed_elicitation_changes_the_fit_digest() -> None:
    document = a_fit_document({"objective": "conditional"})
    prompted = {
        **document,
        "elicitation": {"method": "model_prompting", "version": "1"},
    }
    revised = {**document, "elicitation": {"method": "participants", "version": "2"}}
    digests = {
        validate_fit_document(cast("dict[str, JsonValue]", each)).model_dump_json()
        for each in (document, prompted, revised)
    }

    assert len(digests) == 3

    with pytest.raises(dx.ValidationError, match="elicitation"):
        validate_fit_document(
            {**document, "elicitation": {"method": "participants", "version": ""}}
        )


@pytest.mark.parametrize(
    ("family", "priors"),
    [
        ({"family": "metric"}, {}),
        ({"family": "rank_consistent", "scheme": "corn"}, {}),
    ],
)
def test_a_family_that_compiles_to_no_program_is_refused_by_a_fit(
    family: dict[str, JsonValue], priors: dict[str, JsonValue]
) -> None:
    document = a_fit_document({"objective": "conditional"})
    response = ordinal_response(document)
    response["family"] = family
    response["priors"] = priors
    response["identification"] = {"constraint": "identity_link_on_logit"}
    response["predictor"] = [
        {
            "kind": "link",
            "readout": "p_gram",
            "linking": {"family": "logit", "columns": {"complement": "p_not"}},
        }
    ]
    ResponseSpec.model_validate(response)

    with pytest.raises(dx.ValidationError, match="compiles to none"):
        validate_fit_document(document)


def test_a_linking_family_that_is_not_installed_is_refused_by_a_fit() -> None:
    document = a_fit_document({"objective": "conditional"})
    response = ordinal_response(document)
    response["predictor"] = [
        *cast("list[JsonValue]", response["predictor"]),
        {"kind": "link", "readout": "score", "linking": {"family": "probit"}},
    ]
    ResponseSpec.model_validate(response)

    with pytest.raises(dx.ValidationError, match=r"declares \['probit'\]"):
        validate_fit_document(document)


@pytest.mark.parametrize(
    ("linking", "message"),
    [
        (LinkingSpec(family="logit"), "reads the role 'complement'"),
        (
            LinkingSpec(family="logit", columns={"complement": "p_gram"}),
            "as both the argument and the complement",
        ),
        (
            LinkingSpec(
                family="logit", columns={"complement": "p_not"}, fixed={"scale": 2.0}
            ),
            "these were pinned",
        ),
    ],
)
def test_an_installed_linking_refuses_a_term_it_cannot_read_at_validation(
    linking: LinkingSpec, message: str
) -> None:
    with pytest.raises(dx.ValidationError, match=message):
        ResponseSpec(
            family=Cumulative(),
            levels=5,
            objective=ConditionalLikelihood(),
            predictor=(LinkTerm(readout="p_gram", linking=linking),),
            identification=FixedDiscrimination(value=1.0),
            priors=PriorBlock(thresholds=ThresholdPriors()),
        )

    with pytest.raises(dx.ValidationError, match=message):
        BoundedResponseSpec(
            family=Bounded(scale=0.2),
            predictor=(LinkTerm(readout="p_gram", linking=linking),),
            identification=IdentityLinkOnLogit(),
        )


def a_linked_response(*terms: Predictor) -> ResponseSpec:
    return ResponseSpec(
        family=Cumulative(),
        levels=5,
        objective=ConditionalLikelihood(),
        predictor=terms,
        identification=FixedDiscrimination(value=1.0),
        priors=PriorBlock(thresholds=ThresholdPriors()),
    )


def test_a_correlated_slope_holds_its_installed_linking_at_a_unit_slope() -> None:
    pinned = LinkTerm(
        readout="p_gram",
        linking=LinkingSpec(
            family="logit", columns={"complement": "p_not"}, fixed={"slope": 2.0}
        ),
    )

    a_linked_response(pinned, RandomSlope(group="participant", term=pinned))

    with pytest.raises(dx.ValidationError, match="holds the slope at one"):
        a_linked_response(
            pinned, CorrelatedRandomSlope(group="participant", term=pinned)
        )


def test_a_judgment_fit_without_calibration_states_it() -> None:
    document = a_fit_document({"objective": "conditional"})
    document["calibration"] = None

    assert validate_fit_document(document).calibration is None

    del document["calibration"]

    with pytest.raises(dx.ValidationError, match="calibration"):
        validate_fit_document(document)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "inference",
            {"inference": "nuts", "warmup": 0, "retained": 1000, "chains": 4},
            "warmup",
        ),
        (
            "inference",
            {"inference": "nuts", "warmup": 500, "retained": 3, "chains": 4},
            "four draws",
        ),
        (
            "inference",
            {"inference": "nuts", "warmup": 500, "retained": 1000, "chains": 1},
            "two chains",
        ),
        (
            "calibration",
            {"replicates": 0, "guide_iterations": 200, "guide_learning_rate": 0.05},
            "replicate",
        ),
        (
            "calibration",
            {"replicates": 100, "guide_iterations": 0, "guide_learning_rate": 0.05},
            "sweep",
        ),
        ("dataset", " ", "dataset"),
        ("response", {"levels": 7}, "response"),
    ],
)
def test_a_malformed_judgment_fit_is_refused(
    field: str, value: JsonValue, message: str
) -> None:
    document = a_fit_document({"objective": "conditional"})
    document[field] = value

    with pytest.raises(dx.ValidationError, match=message):
        validate_fit_document(document)


def test_a_run_document_is_not_a_judgment_fit() -> None:
    with pytest.raises(dx.ValidationError):
        JudgmentFitSpec.model_validate(
            {"model": {"parser": {"kind": "inside-outside"}}, "trainer": {}}
        )


def test_a_judgment_fit_manifest_records_its_fit_and_seed_derivation() -> None:
    fit = validate_fit_document(a_fit_document({"objective": "conditional"}))
    derivation = SeedDerivationRecord(
        experiment_root=7,
        point_id="a",
        seed=0,
        root=point_root(7, "a"),
    )
    design = DesignProvenance(
        tag="ratings",
        checkpoint=f"an-org/a-model@{'a' * 40}",
        checkpoint_dtype="float32",
        unigram_digest="b" * 64,
        corpus_digests={"responses.tsv": "c" * 64},
        seed=None,
        participants=None,
        items=None,
        levels=None,
        population_digest=None,
        producer_run=None,
        producer_point=None,
    )
    manifest = JudgmentFitManifest(
        fit=fit,
        design=design,
        seed_derivation=derivation,
        artifact=FitArtifact(name="posterior.nc", sha256="d" * 64),
        package_versions={"quivers": "0.25.2"},
        git_revision="0123456789abcdef",
    )

    assert JudgmentFitManifest.model_validate_json(manifest.model_dump_json()) == (
        manifest
    )
    assert manifest.design.checkpoint == f"an-org/a-model@{'a' * 40}"
    assert manifest.artifact == FitArtifact(name="posterior.nc", sha256="d" * 64)

    with pytest.raises(dx.ValidationError, match="cannot be empty"):
        manifest.with_(package_versions={})

    with pytest.raises(dx.ValidationError, match="derives as"):
        JudgmentFitManifest.model_validate(
            {
                **manifest.model_dump(),
                "seed_derivation": {
                    **derivation.model_dump(),
                    "root": derivation.root + 1,
                },
            }
        )

    with pytest.raises(dx.ValidationError, match="artifact digest"):
        manifest.with_(artifact=FitArtifact(name="posterior.nc", sha256="short"))

    with pytest.raises(dx.ValidationError, match=r"persists 'posterior\.nc'"):
        manifest.with_(
            artifact=FitArtifact(name="point-estimate.safetensors", sha256="d" * 64)
        )

    map_document = a_fit_document({"objective": "conditional"})
    map_document["calibration"] = None
    map_document["inference"] = {
        "inference": "map",
        "optimizer": {"kind": "adam"},
        "steps": 5,
        "initial_scale": 0.1,
    }
    mapped = manifest.with_(
        fit=validate_fit_document(map_document),
        artifact=FitArtifact(name="point-estimate.safetensors", sha256="e" * 64),
    )

    assert decode_fit_manifest(encode_fit_manifest(mapped)) == mapped


@pytest.mark.parametrize(
    ("field", "value", "detail"),
    [
        ("tag", "", "design tag"),
        ("unigram_digest", "short", "digest"),
        ("corpus_digests", {"responses.tsv": "C" * 64}, "digest"),
        ("checkpoint", "", "checkpoint"),
        ("checkpoint", f"an-org/a-model@{'a' * 40}", "recorded together"),
        ("checkpoint_dtype", "float32", "recorded together"),
        ("participants", 0, "design count"),
        ("items", None, "recorded together"),
        ("population_digest", "short", "digest"),
        ("producer_run", "runs/source", "recorded together"),
    ],
)
def test_a_design_provenance_refuses_a_malformed_identity(
    field: str, value: JsonValue, detail: str
) -> None:
    declared: dict[str, JsonValue] = {
        "tag": "ratings",
        "checkpoint": None,
        "checkpoint_dtype": None,
        "unigram_digest": None,
        "corpus_digests": {},
        "seed": 3,
        "participants": 2,
        "items": 4,
        "levels": 7,
        "population_digest": "d" * 64,
        "producer_run": None,
        "producer_point": None,
    }
    declared[field] = value

    with pytest.raises(dx.ValidationError, match=detail):
        DesignProvenance.model_validate(declared)

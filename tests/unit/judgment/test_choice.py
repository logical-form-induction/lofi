import math
from typing import TYPE_CHECKING, Final

import pytest
import torch
from torch.nn import functional

from lofi.judgment import (
    PLATE,
    POSITION,
    PREDICTOR,
    RESPONSE,
    Apply,
    Binary,
    Bind,
    Choice,
    ChoiceFamily,
    ConditionalLikelihood,
    Cumulative,
    CumulativeFamily,
    DesignCodec,
    DiscriminationGroup,
    Draw,
    FactorTerm,
    Function,
    IdentityLinkOnLogit,
    JudgmentDesign,
    JudgmentRow,
    LinkingSpec,
    LinkTerm,
    Literal,
    Name,
    ObserveChoice,
    ObserveOrdered,
    Operator,
    Prior,
    PriorBlock,
    PriorSpec,
    ResponseCompiler,
    ResponseSpec,
    RowColumns,
    Table,
    ThresholdPriors,
    latent,
    log_likelihood,
    logit_steps,
    probability_steps,
)

if TYPE_CHECKING:
    from torch import Tensor

P_GRAM: Final = "p_gram"
COMPLEMENT: Final = "p_gram_complement"
READOUTS: Final = (P_GRAM, COMPLEMENT)
LOGIT: Final = LinkingSpec(family="logit", columns={"complement": COMPLEMENT})
PROBABILITY: Final = LinkingSpec(family="probability")
LINK: Final = LinkTerm(readout=P_GRAM, linking=LOGIT)
POSITION_TERM: Final = FactorTerm(name=POSITION)
TRIALS: Final = 120


def compiler() -> ResponseCompiler:
    return ResponseCompiler({"logit": logit_steps, "probability": probability_steps})


def mass(trial: int, at: int) -> float:
    return 0.05 + 0.9 * (((trial * 7 + at * 5) % 13) / 13.0)


def columns(masses: tuple[float, ...]) -> RowColumns:
    return RowColumns(
        {},
        {},
        {
            P_GRAM: tuple(math.log(value) for value in masses),
            COMPLEMENT: tuple(math.log1p(-value) for value in masses),
        },
    )


def choice_design(
    trials: int,
    alternatives: int,
    levels: int,
    *,
    participants: int = 5,
) -> JudgmentDesign:
    rows = tuple(
        JudgmentRow(
            f"annotator-{trial % participants}",
            f"trial-{trial}",
            trial % levels,
            columns(tuple(mass(trial, at) for at in range(alternatives))),
            tuple(f"item-{trial}-{at}" for at in range(alternatives)),
        )
        for trial in range(trials)
    )

    return DesignCodec(levels, readouts=READOUTS, alternatives=alternatives).encode(
        rows
    )


def choice_spec(
    *terms: FactorTerm | LinkTerm,
    family: Choice,
    levels: int,
    priors: PriorBlock | None = None,
) -> ResponseSpec:
    return ResponseSpec(
        family=family,
        levels=levels,
        objective=ConditionalLikelihood(),
        predictor=terms,
        identification=IdentityLinkOnLogit(),
        priors=PriorBlock() if priors is None else priors,
    )


def scores(design: JudgmentDesign, alternatives: int, shift: Tensor) -> Tensor:
    """Return the log odds of each alternative plus its position's coefficient."""
    odds = design.readouts[P_GRAM] - design.readouts[COMPLEMENT]
    place = torch.arange(alternatives).repeat(design.n_rows)

    return (odds + shift[place]).reshape(-1, alternatives)


def test_a_family_shows_at_least_two_alternatives() -> None:
    with pytest.raises(ValueError, match="at least 2 alternatives: 1"):
        ChoiceFamily(1)


def test_a_varying_discrimination_is_drawn() -> None:
    with pytest.raises(ValueError, match="does not draw it"):
        ChoiceFamily(
            2,
            discrimination=2.0,
            discrimination_group=DiscriminationGroup("participant", "Participant"),
        )


def test_a_tie_is_recorded_between_two_alternatives_over_three_categories() -> None:
    with pytest.raises(ValueError, match="shows 3 over 4"):
        ChoiceFamily(3, ordered=CumulativeFamily(4))


def test_the_difference_is_scaled_once() -> None:
    with pytest.raises(ValueError, match=r"scales it again by 2\.0"):
        ChoiceFamily(
            2,
            discrimination=1.5,
            ordered=CumulativeFamily(3, discrimination=2.0),
        )


def test_a_family_names_its_sets_its_sites_and_its_columns() -> None:
    plain = ChoiceFamily(3)
    tied = ChoiceFamily(
        2,
        discrimination=Prior("LogNormal", 0.0, 0.5),
        ordered=CumulativeFamily(3),
    )

    assert plain.categories == 3
    assert plain.cutpoint_object is None
    assert plain.contrast_column is None
    assert plain.sites == ()
    assert plain.bound_names == ("choice_logits",)
    assert plain.host_columns == ()
    assert (plain.index_object, plain.scalar_object) == ("Alternative", "Unit")
    assert tied.categories == 3
    assert tied.cutpoint_object == "Cutpoint"
    assert tied.contrast_column == "alternative_contrast"
    assert tied.contrast == (-1.0, 1.0)
    assert tied.sites == (
        "choice_discrimination",
        "cutpoint_base",
        "cutpoint_increments",
    )
    assert tied.host_columns == ("alternative_contrast",)
    assert "cutpoints" in tied.bound_names
    assert tied.scalar_object == "Unit"
    assert repr(plain) == "ChoiceFamily(3, ties=False)"


def test_a_family_refuses_a_name_it_binds_and_an_object_it_declares() -> None:
    family = ChoiceFamily(2)

    with pytest.raises(ValueError, match="a predictor must not be a name"):
        family.steps(predictor="choice_logits", response=RESPONSE, plate=PLATE)

    with pytest.raises(ValueError, match="must differ"):
        family.steps(predictor=RESPONSE, response=RESPONSE, plate=PLATE)

    with pytest.raises(ValueError, match="must not be an object"):
        family.steps(predictor=PREDICTOR, response=RESPONSE, plate="Alternative")


def test_a_family_without_ties_observes_a_multinomial_logit() -> None:
    steps = ChoiceFamily(3).steps(predictor=PREDICTOR, response=RESPONSE, plate=PLATE)

    assert steps[-1] == ObserveChoice(RESPONSE, PLATE, logits="choice_logits")
    assert steps[0] == Bind(
        "choice_logits", Table(PREDICTOR, PLATE, "Alternative", width=3)
    )


def test_a_family_with_ties_observes_the_difference_under_two_cutpoints() -> None:
    steps = ChoiceFamily(2, ordered=CumulativeFamily(3)).steps(
        predictor=PREDICTOR, response=RESPONSE, plate=PLATE
    )
    assert isinstance(steps[-1], ObserveOrdered)
    assert steps[-1].location == "choice_difference"
    assert steps[-1].thresholds == "cutpoints"
    assert (
        Bind(
            "choice_difference",
            Binary(
                Operator.MUL,
                Literal(2.0),
                Apply(Function.MEAN, Name("choice_weighted")),
            ),
        )
        in steps
    )
    assert (
        Draw("cutpoint_increments", ("Cutpoint",), Prior("Normal", 0.0, 1.0)) in steps
    )


def test_two_alternatives_are_chosen_between_by_a_logistic_in_the_difference() -> None:
    design = choice_design(TRIALS, 2, 2)
    spec = choice_spec(LINK, POSITION_TERM, family=Choice(), levels=2)
    bound = compiler().compile(spec).over(design)
    params = torch.tensor([0.4, -0.7], dtype=torch.float64)
    place = scores(design, 2, bound.sites(params)["position_effects"])
    first = torch.sigmoid(place[:, 0] - place[:, 1])
    expected = torch.where(design.responses == 0, first, 1.0 - first)

    assert bound.size == 2
    assert torch.allclose(log_likelihood(bound, params), torch.log(expected))


def test_a_tie_is_scored_by_the_two_cutpoints_over_the_difference() -> None:
    design = choice_design(TRIALS, 2, 3)
    spec = choice_spec(
        LINK,
        POSITION_TERM,
        family=Choice(ties=True),
        levels=3,
        priors=PriorBlock(thresholds=ThresholdPriors()),
    )
    bound = compiler().compile(spec).over(design)
    params = torch.tensor([0.4, -0.7, -0.2, 0.3, 0.1], dtype=torch.float64)
    sites = bound.sites(params)
    place = scores(design, 2, sites["position_effects"])
    difference = place[:, 1] - place[:, 0]
    cutpoints = sites["cutpoint_base"] + torch.cumsum(
        functional.softplus(sites["cutpoint_increments"]), dim=-1
    )
    below = torch.sigmoid(cutpoints[0] - difference)
    within = torch.sigmoid(cutpoints[1] - difference)
    expected = torch.stack((below, within - below, 1.0 - within), dim=1)
    recorded = expected.gather(1, design.responses.unsqueeze(1)).squeeze(1)

    assert cutpoints.numel() == 2
    assert torch.allclose(log_likelihood(bound, params), torch.log(recorded))


def test_three_alternatives_are_chosen_between_by_a_multinomial_logit() -> None:
    design = choice_design(TRIALS, 3, 3)
    spec = choice_spec(LINK, POSITION_TERM, family=Choice(), levels=3)
    bound = compiler().compile(spec).over(design)
    params = torch.tensor([0.4, -0.7, 0.25], dtype=torch.float64)
    place = scores(design, 3, bound.sites(params)["position_effects"])
    weights = torch.exp(place)
    expected = weights / weights.sum(dim=1, keepdim=True)
    recorded = expected.gather(1, design.responses.unsqueeze(1)).squeeze(1)

    assert bound.size == 3
    assert torch.allclose(log_likelihood(bound, params), torch.log(recorded))


def test_a_drawn_discrimination_scales_the_difference() -> None:
    design = choice_design(TRIALS, 2, 2)
    spec = choice_spec(
        LINK,
        POSITION_TERM,
        family=Choice(discrimination=True),
        levels=2,
        priors=PriorBlock(
            discrimination=PriorSpec(distribution="LogNormal", parameters=(0.0, 0.5))
        ),
    )
    bound = compiler().compile(spec).over(design)
    params = torch.tensor([0.4, -0.7, 1.7], dtype=torch.float64)
    sites = bound.sites(params)
    place = scores(design, 2, sites["position_effects"])
    first = torch.sigmoid(sites["choice_discrimination"] * (place[:, 0] - place[:, 1]))
    expected = torch.where(design.responses == 0, first, 1.0 - first)

    assert "choice_discrimination" in sites
    assert torch.allclose(log_likelihood(bound, params), torch.log(expected))


def reversed_pair_design() -> JudgmentDesign:
    """Return two rows showing one pair of items in each presentation order."""
    forward = tuple(mass(3, at) for at in range(2))
    rows = (
        JudgmentRow("annotator-1", "trial-0", 0, columns(forward), ("left", "right")),
        JudgmentRow(
            "annotator-1",
            "trial-1",
            0,
            columns(forward[::-1]),
            ("right", "left"),
        ),
    )

    return DesignCodec(2, readouts=READOUTS, alternatives=2).encode(rows)


def test_the_position_bias_is_identified_by_the_presentation_order() -> None:
    design = reversed_pair_design()
    spec = choice_spec(LINK, POSITION_TERM, family=Choice(), levels=2)
    bound = compiler().compile(spec).over(design)
    params = torch.tensor([0.4, -0.7], dtype=torch.float64)
    chosen = torch.exp(log_likelihood(bound, params))
    odds = torch.log(chosen) - torch.log1p(-chosen)

    assert design.alternatives == (("left", "right"), ("right", "left"))
    assert torch.allclose(odds.mean(), params[0] - params[1], atol=1e-12, rtol=0.0)


def rating_design(trials: int) -> JudgmentDesign:
    rows = tuple(
        JudgmentRow(
            f"annotator-{trial % 5}",
            f"item-{trial}",
            trial % 2,
            RowColumns(
                {},
                {
                    P_GRAM: math.log(mass(trial // 2, trial % 2)),
                    COMPLEMENT: math.log1p(-mass(trial // 2, trial % 2)),
                },
            ),
        )
        for trial in range(trials)
    )

    return DesignCodec(2, readouts=READOUTS).encode(rows)


def rated_differences(linking: LinkingSpec) -> Tensor:
    """Return the rating path's latent of the first item less that of the second."""
    bound = (
        compiler()
        .compile(
            ResponseSpec(
                family=Cumulative(),
                levels=2,
                objective=ConditionalLikelihood(),
                predictor=(LinkTerm(readout=P_GRAM, linking=linking),),
                identification=IdentityLinkOnLogit(),
                priors=PriorBlock(thresholds=ThresholdPriors()),
            )
        )
        .over(rating_design(2 * TRIALS))
    )
    located = latent(bound, torch.tensor([0.2, 0.5], dtype=torch.float64)).location
    paired = located.reshape(-1, 2)

    return paired[:, 0] - paired[:, 1]


def chosen_likelihoods(linking: LinkingSpec, design: JudgmentDesign) -> Tensor:
    """Return the choice path's likelihood of the level each row records."""
    bound = (
        compiler()
        .compile(
            choice_spec(
                LinkTerm(readout=P_GRAM, linking=linking), family=Choice(), levels=2
            )
        )
        .over(design)
    )

    return log_likelihood(bound, torch.tensor([], dtype=torch.float64))


@pytest.mark.parametrize("linking", [LOGIT, PROBABILITY], ids=("logit", "probability"))
def test_the_rating_and_the_choice_paths_read_one_latent(
    linking: LinkingSpec,
) -> None:
    design = choice_design(TRIALS, 2, 2)
    difference = rated_differences(linking)
    first = torch.sigmoid(difference)
    expected = torch.where(design.responses == 0, first, 1.0 - first)

    assert torch.allclose(chosen_likelihoods(linking, design), torch.log(expected))


def test_swapping_the_linking_moves_both_paths() -> None:
    design = choice_design(TRIALS, 2, 2)

    assert not torch.allclose(rated_differences(LOGIT), rated_differences(PROBABILITY))
    assert not torch.allclose(
        chosen_likelihoods(LOGIT, design), chosen_likelihoods(PROBABILITY, design)
    )


def test_a_choice_program_declares_the_set_indexing_the_alternatives() -> None:
    program = compiler().compile(
        choice_spec(LINK, POSITION_TERM, family=Choice(), levels=3)
    )

    bound = program.over(choice_design(30, 3, 3))

    assert program.alternatives == 3
    assert bound.sizes["Alternative"] == 3
    assert program.observation == ObserveChoice(RESPONSE, PLATE, logits="choice_logits")

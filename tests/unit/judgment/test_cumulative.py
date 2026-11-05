from typing import TYPE_CHECKING, NamedTuple, cast

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from quivers.effects import TraceHandler, clamp, run_program
from torch import distributions

from lofi.judgment import (
    Apply,
    Binary,
    Bind,
    BoundProgram,
    ConditionalPlan,
    Draw,
    Function,
    Gather,
    Literal,
    Name,
    ObserveBounded,
    ObserveChoice,
    ObserveOrdered,
    Operator,
    Prior,
    ResponseProgram,
    Table,
)
from lofi.judgment.cumulative import (
    DEFAULT_SPREAD_PRIOR,
    CumulativeFamily,
    DiscriminationGroup,
    PredictedParameter,
    ThresholdGroup,
)
from lofi.judgment.runtime import QuiversRuntime

if TYPE_CHECKING:
    from collections.abc import Callable

    from quivers.effects import Trace

    from lofi.judgment import Step

OBSERVATIONS = 8
RESPONSE = "response"
PLATE = "Resp"
PARTICIPANT = "participant"
PARTICIPANTS = 3
ITEM = "item"
ITEMS = 4
RUNTIME = QuiversRuntime()


def emitted(family: CumulativeFamily, **names: str) -> tuple[Step, ...]:
    return family.steps(**names)


def program_of(
    family: CumulativeFamily,
    given: dict[str, torch.Tensor],
    *,
    observations: int = OBSERVATIONS,
    levels: int = PARTICIPANTS,
) -> BoundProgram:
    sizes = {
        PLATE: observations,
        family.index_object: family.thresholds,
        family.scalar_object: 1,
    }

    if family.group is not None:
        sizes[family.group.index_object] = levels

    if family.discrimination_group is not None:
        sizes[family.discrimination_group.index_object] = ITEMS

    program = ResponseProgram(
        levels=family.categories,
        steps=(
            Draw("slope", (family.scalar_object,), Prior("Normal", 0.0, 1.0)),
            Bind("predictor", Binary(Operator.MUL, Name("slope"), Name("covariate"))),
            *emitted(family, predictor="predictor", response=RESPONSE, plate=PLATE),
        ),
        objects={},
        indices={},
        readouts=(),
        objective=ConditionalPlan(),
        groups=(),
        cutpoint_object=family.index_object,
        threshold_group=None if family.group is None else family.group.index,
        scalar_object=family.scalar_object,
    )

    return BoundProgram(program, sizes=sizes, host_data=given)


def trace_of(
    family: CumulativeFamily,
    pinned: dict[str, torch.Tensor] | None = None,
    *,
    observations: int = OBSERVATIONS,
    supplied: dict[str, torch.Tensor] | None = None,
    levels: int = PARTICIPANTS,
) -> Trace:
    given = {"covariate": torch.linspace(-1.0, 1.0, observations)}

    if family.group is not None:
        given[family.group.index] = torch.arange(observations) % levels

    if family.discrimination_group is not None:
        given[family.discrimination_group.index] = torch.arange(observations) % ITEMS

    given.update(supplied or {})
    program = program_of(family, given, observations=observations, levels=levels)
    handler = TraceHandler()

    with clamp(pinned or {}), handler:
        run_program(RUNTIME.program(program), torch.zeros(1, 1), dict(given))

    return handler.trace


def ordered_logistic_probabilities(
    cutpoints: torch.Tensor, predictor: torch.Tensor
) -> torch.Tensor:
    below = torch.sigmoid(cutpoints - predictor)
    edge = torch.zeros_like(below[:, :1])

    return torch.cat([edge, below, edge + 1.0], dim=-1).diff(dim=-1)


def ordered_logistic_draw(
    cutpoints: torch.Tensor, predictor: torch.Tensor
) -> torch.Tensor:
    probabilities = ordered_logistic_probabilities(cutpoints, predictor)
    generator = torch.Generator().manual_seed(11)

    return torch.multinomial(probabilities, 1, generator=generator).reshape(-1)


@pytest.mark.parametrize("categories", [2, 3, 5, 7])
def test_the_emitted_program_declares_one_cutpoint_per_boundary(
    categories: int,
) -> None:
    family = CumulativeFamily(categories)

    trace = trace_of(family)

    assert family.cutpoint_site in trace.sites
    assert trace.sites[family.cutpoint_site].value.shape[-1] == categories - 1


def test_the_emitted_program_observes_the_response_it_was_given() -> None:
    family = CumulativeFamily(5)

    trace = trace_of(family)

    assert RESPONSE in trace.sites
    assert trace.sites[RESPONSE].value.shape == (OBSERVATIONS,)
    assert int(trace.sites[RESPONSE].value.max()) < family.categories


@pytest.mark.parametrize(
    "discrimination", [None, Prior("LogNormal", 0.0, 0.5)], ids=["fixed", "drawn"]
)
def test_the_family_draws_exactly_the_latent_sites_it_publishes(
    discrimination: Prior | None,
) -> None:
    family = CumulativeFamily(4, discrimination=discrimination)

    trace = trace_of(
        family, supplied={RESPONSE: torch.tensor([0, 1, 2, 3, 1, 2, 3, 0])}
    )

    assert set(trace.latent_sites) == {"slope", *family.sites}
    assert trace.sites[RESPONSE].is_observed


def test_the_family_draws_its_latent_sites_in_the_order_it_publishes_them() -> None:
    family = CumulativeFamily(4, discrimination=Prior("LogNormal", 0.0, 0.5))

    steps = emitted(family, predictor="predictor", response=RESPONSE, plate=PLATE)
    drawn = tuple(step.site for step in steps if isinstance(step, Draw))

    assert drawn == family.sites


def test_a_family_without_discrimination_draws_no_discrimination() -> None:
    family = CumulativeFamily(4)

    trace = trace_of(family)

    assert family.discrimination_site not in family.sites
    assert family.discrimination_site not in trace.sites


def test_the_cutpoints_and_a_covariate_beside_them_all_receive_gradients() -> None:
    family = CumulativeFamily(5)
    base = torch.zeros(1, requires_grad=True)
    increments = torch.zeros(family.thresholds, requires_grad=True)
    covariate = torch.linspace(-1.0, 1.0, OBSERVATIONS).requires_grad_()

    trace = trace_of(
        family,
        {
            "slope": torch.full((1,), 0.8),
            family.base_site: base,
            family.increment_site: increments,
        },
        supplied={
            RESPONSE: torch.tensor([0, 1, 2, 3, 4, 1, 2, 3]),
            "covariate": covariate,
        },
    )
    trace.sites[RESPONSE].log_prob.sum().backward()

    assert base.grad is not None
    assert increments.grad is not None
    assert covariate.grad is not None
    assert bool((base.grad != 0).any())
    assert bool((increments.grad != 0).any())
    assert bool((covariate.grad != 0).any())


def test_a_gradient_fit_recovers_the_cutpoints_the_responses_were_drawn_from() -> None:
    observations = 400
    family = CumulativeFamily(5)
    covariate = torch.linspace(-2.0, 2.0, observations).reshape(observations, 1)
    drawn = torch.tensor([-1.2, -0.4, 0.4, 1.2])
    responses = ordered_logistic_draw(drawn, covariate)
    column = covariate.reshape(-1)

    base = torch.zeros(1, requires_grad=True)
    increments = torch.zeros(family.thresholds, requires_grad=True)

    def fit_step() -> tuple[torch.Tensor, torch.Tensor]:
        trace = trace_of(
            family,
            {
                "slope": torch.ones(1),
                family.base_site: base,
                family.increment_site: increments,
            },
            observations=observations,
            supplied={RESPONSE: responses, "covariate": column},
        )

        return (
            trace.sites[RESPONSE].log_prob.sum(),
            trace.sites[family.cutpoint_site].value,
        )

    before, started_at = fit_step()
    optimizer = torch.optim.Adam([base, increments], lr=0.1)

    for _ in range(300):
        optimizer.zero_grad()
        objective, _ = fit_step()
        (-objective).backward()
        optimizer.step()

    after, ended_at = fit_step()

    assert float(after.detach()) > float(before.detach())
    assert float((started_at.detach() - drawn).abs().max()) > 1.0
    assert float((ended_at.detach() - drawn).abs().max()) < 0.25


CATEGORIES = 6
RAW = st.floats(min_value=-8.0, max_value=8.0, allow_nan=False, allow_infinity=False)


@settings(max_examples=50, deadline=None)
@given(
    base=RAW,
    increments=st.lists(RAW, min_size=CATEGORIES - 1, max_size=CATEGORIES - 1),
)
def test_the_cutpoints_rise_whatever_raw_values_are_drawn(
    base: float, increments: list[float]
) -> None:
    family = CumulativeFamily(CATEGORIES)

    trace = trace_of(
        family,
        {
            family.base_site: torch.full((1,), base),
            family.increment_site: torch.tensor(increments),
        },
    )
    cutpoints = trace.sites[family.cutpoint_site].value

    assert bool((cutpoints.diff(dim=-1) > 0).all())


def test_discrimination_scales_the_predictor_and_the_cutpoints_together() -> None:
    family = CumulativeFamily(5, discrimination=Prior("LogNormal", 0.0, 0.5))
    scale = 2.5

    trace = trace_of(
        family,
        {family.discrimination_site: torch.full((1,), scale)},
    )
    cutpoints = trace.sites[family.cutpoint_site].value

    assert torch.allclose(
        trace.sites[family.scaled_cutpoint_site].value, scale * cutpoints
    )
    assert torch.allclose(
        trace.sites[family.scaled_predictor_site].value,
        scale * trace.sites["predictor"].value,
    )


def test_scaling_by_a_positive_discrimination_keeps_the_cutpoints_rising() -> None:
    family = CumulativeFamily(5, discrimination=Prior("LogNormal", 0.0, 1.0))

    trace = trace_of(family, {family.discrimination_site: torch.full((1,), 0.3)})

    assert bool((trace.sites[family.scaled_cutpoint_site].value.diff(dim=-1) > 0).all())


def test_one_raw_increment_is_drawn_per_threshold() -> None:
    family = CumulativeFamily(6)

    trace = trace_of(family)

    assert trace.sites[family.increment_site].value.shape == (family.thresholds,)


def test_fewer_than_two_categories_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 2 categories"):
        CumulativeFamily(1)


def test_a_predictor_that_cannot_name_a_variable_is_refused() -> None:
    family = CumulativeFamily(4)

    with pytest.raises(ValueError, match="a predictor must be an identifier"):
        family.steps(predictor="eta + 1", response=RESPONSE, plate=PLATE)


def test_a_prior_family_that_cannot_name_a_distribution_is_refused() -> None:
    with pytest.raises(ValueError, match="distribution family must be an identifier"):
        Prior("Normal(0, 1)", 0.0)


def test_a_prior_argument_must_be_finite() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        Prior("Normal", 0.0, float("inf"))


def test_a_prior_holds_its_arguments_as_floating_point_numbers() -> None:
    assert Prior("Normal", 0.0, 5.0).arguments == (0.0, 5.0)
    assert Prior("Uniform").arguments == ()
    assert Prior("Binomial", 3).arguments == (3.0,)
    assert isinstance(Prior("Binomial", 3).arguments[0], float)


def name_bound_by(step: Step) -> str:
    match step:
        case Draw(site=site) | Bind(site=site):
            return site
        case (
            ObserveBounded(response=response)
            | ObserveOrdered(response=response)
            | ObserveChoice(response=response)
        ):
            return response


def names_bound_by(steps: tuple[Step, ...]) -> tuple[str, ...]:
    return tuple(name_bound_by(step) for step in steps)


@pytest.mark.parametrize(
    "discrimination", [None, Prior("LogNormal", 0.0, 0.5)], ids=["fixed", "drawn"]
)
def test_the_family_publishes_exactly_the_names_its_steps_bind(
    discrimination: Prior | None,
) -> None:
    family = CumulativeFamily(5, discrimination=discrimination)

    steps = emitted(family, predictor="predictor", response=RESPONSE, plate=PLATE)

    assert names_bound_by(steps) == (*family.bound_names, RESPONSE)


def test_a_family_without_discrimination_binds_no_scaled_names() -> None:
    family = CumulativeFamily(5)

    assert family.scaled_predictor_site not in family.bound_names
    assert family.scaled_cutpoint_site not in family.bound_names
    assert family.discrimination_site not in family.bound_names
    assert family.cutpoint_site in family.bound_names


def test_a_predictor_naming_something_the_family_binds_is_refused() -> None:
    family = CumulativeFamily(5, discrimination=Prior("LogNormal", 0.0, 0.5))

    for name in family.bound_names:
        with pytest.raises(ValueError, match="a predictor must not be a name"):
            family.steps(predictor=name, response=RESPONSE, plate=PLATE)


def test_a_response_naming_something_the_family_binds_is_refused() -> None:
    family = CumulativeFamily(5, discrimination=Prior("LogNormal", 0.0, 0.5))

    for name in family.bound_names:
        with pytest.raises(ValueError, match="a response must not be a name"):
            family.steps(predictor="predictor", response=name, plate=PLATE)


def test_a_predictor_and_a_response_of_one_name_are_refused() -> None:
    family = CumulativeFamily(5)

    with pytest.raises(ValueError, match="a predictor and a response must differ"):
        family.steps(predictor=RESPONSE, response=RESPONSE, plate=PLATE)


@pytest.mark.parametrize("declared", ["index_object", "scalar_object"])
def test_a_plate_naming_an_object_the_family_declares_is_refused(
    declared: str,
) -> None:
    family = CumulativeFamily(5)
    plate = {"index_object": family.index_object, "scalar_object": family.scalar_object}

    with pytest.raises(ValueError, match="must not be an object the family declares"):
        family.steps(predictor="predictor", response=RESPONSE, plate=plate[declared])


def test_the_observation_reads_the_scaled_predictor_and_the_scaled_cutpoints() -> None:
    family = CumulativeFamily(5, discrimination=Prior("LogNormal", 0.0, 0.5))
    responses = torch.tensor([0, 1, 2, 3, 4, 1, 2, 3])

    trace = trace_of(
        family,
        {
            "slope": torch.ones(1),
            family.base_site: torch.full((1,), -1.0),
            family.increment_site: torch.zeros(family.thresholds),
            family.discrimination_site: torch.full((1,), 2.5),
        },
        supplied={RESPONSE: responses},
    )
    probabilities = ordered_logistic_probabilities(
        trace.sites[family.scaled_cutpoint_site].value,
        trace.sites[family.scaled_predictor_site].value.unsqueeze(-1),
    )
    expected = probabilities.gather(-1, responses.reshape(-1, 1)).log().sum()

    assert torch.allclose(trace.sites[RESPONSE].log_prob.sum(), expected, atol=1e-4)


def test_each_drawn_site_names_the_prior_it_was_given() -> None:
    family = CumulativeFamily(
        4,
        base=Prior("Cauchy", 1.0, 2.0),
        increment=Prior("Exponential", 3.0),
        discrimination=Prior("LogNormal", 0.0, 0.5),
    )

    steps = emitted(family, predictor="predictor", response=RESPONSE, plate=PLATE)
    drawn = {step.site: step.prior for step in steps if isinstance(step, Draw)}

    assert drawn[family.base_site] == family.base
    assert drawn[family.increment_site] == family.increment
    assert drawn[family.discrimination_site] == family.discrimination


def test_each_drawn_site_is_drawn_from_the_prior_it_was_given() -> None:
    group = ThresholdGroup(
        PARTICIPANT,
        "Participant",
        spread=Prior("Normal", 50.0, 0.01),
    )
    family = CumulativeFamily(
        4,
        base=Prior("Normal", 100.0, 0.01),
        increment=Prior("Normal", -100.0, 0.01),
        discrimination=Prior("Normal", 25.0, 0.01),
        group=group,
    )
    located = {
        family.base_site: 100.0,
        family.increment_site: -100.0,
        group.spread_site: 50.0,
        family.discrimination_site: 25.0,
    }

    trace = trace_of(family)

    for site, location in located.items():
        drawn = trace.sites[site].value.detach()

        assert float((drawn - location).abs().max()) < 1.0


def test_a_name_opening_with_a_digit_is_refused() -> None:
    family = CumulativeFamily(4)

    with pytest.raises(ValueError, match="a predictor must be an identifier"):
        family.steps(predictor="1eta", response=RESPONSE, plate=PLATE)

    with pytest.raises(ValueError, match="a response must be an identifier"):
        family.steps(predictor="eta", response="2y", plate=PLATE)

    with pytest.raises(ValueError, match="a response plate must be an identifier"):
        family.steps(predictor="eta", response=RESPONSE, plate="3Resp")

    with pytest.raises(ValueError, match="distribution family must be an identifier"):
        Prior("2Normal", 0.0)


def test_a_plate_that_cannot_name_an_object_is_refused() -> None:
    family = CumulativeFamily(4)

    with pytest.raises(ValueError, match="a response plate must be an identifier"):
        family.steps(predictor="eta", response=RESPONSE, plate="Resp[0]")


@pytest.mark.parametrize("plate", ["Resp", "Row"])
def test_the_observation_indexes_the_response_by_the_plate_it_was_given(
    plate: str,
) -> None:
    family = CumulativeFamily(4, discrimination=Prior("LogNormal", 0.0, 0.5))

    steps = emitted(family, predictor="predictor", response=RESPONSE, plate=plate)

    assert steps[-1] == ObserveOrdered(
        RESPONSE,
        plate,
        location=family.scaled_predictor_site,
        thresholds=family.scaled_cutpoint_site,
    )


def test_a_prior_publishes_the_family_and_the_arguments_it_was_built_from() -> None:
    prior = Prior("Beta", 2.0, 3.5)

    assert prior.family == "Beta"
    assert prior.arguments == (2.0, 3.5)
    assert repr(prior) == "Prior('Beta', 2.0, 3.5)"


def test_a_prior_without_arguments_publishes_none() -> None:
    prior = Prior("Uniform")

    assert prior.family == "Uniform"
    assert prior.arguments == ()
    assert repr(prior) == "Prior('Uniform')"


def grouped_family(
    categories: int = 4,
    *,
    spread: Prior = DEFAULT_SPREAD_PRIOR,
    discrimination: Prior | None = None,
) -> CumulativeFamily:
    group = ThresholdGroup(PARTICIPANT, "Participant", spread=spread)

    return CumulativeFamily(categories, group=group, discrimination=discrimination)


def discriminating_family(
    categories: int = 4, *, group: ThresholdGroup | None = None
) -> CumulativeFamily:
    return CumulativeFamily(
        categories,
        discrimination=Prior("LogNormal", 0.0, 0.5),
        group=group,
        discrimination_group=DiscriminationGroup(ITEM, "Item"),
    )


DISCRIMINATIONS = torch.tensor([0.5, 1.0, 1.5, 2.0])


def discriminating_trace(family: CumulativeFamily) -> Trace:
    return trace_of(
        family,
        {family.discrimination_site: DISCRIMINATIONS},
        supplied={"covariate": torch.linspace(-1.0, 1.0, OBSERVATIONS)},
    )


def test_a_discrimination_group_publishes_its_names() -> None:
    group = DiscriminationGroup(ITEM, "Item")

    assert group.index == ITEM
    assert group.index_object == "Item"
    assert repr(group) == "DiscriminationGroup('item', 'Item')"


def test_a_discrimination_group_index_and_index_object_must_be_identifiers() -> None:
    with pytest.raises(ValueError, match="discrimination group's index must be"):
        DiscriminationGroup("by item", "Item")

    with pytest.raises(ValueError, match="discrimination group's index object must"):
        DiscriminationGroup(ITEM, "Item[0]")


def test_a_discrimination_group_index_and_index_object_must_differ() -> None:
    with pytest.raises(ValueError, match="index and index object must differ"):
        DiscriminationGroup(ITEM, ITEM)


@pytest.mark.parametrize("discrimination", [None, 2.5])
def test_a_discrimination_group_without_a_drawn_discrimination_is_refused(
    discrimination: float | None,
) -> None:
    with pytest.raises(ValueError, match="the family does not draw it"):
        CumulativeFamily(
            4,
            discrimination=discrimination,
            discrimination_group=DiscriminationGroup(ITEM, "Item"),
        )


@pytest.mark.parametrize("declared", ["Cutpoint", "Unit"])
def test_a_discrimination_group_indexed_by_an_object_the_family_declares_is_refused(
    declared: str,
) -> None:
    with pytest.raises(ValueError, match="must not be an object the family declares"):
        CumulativeFamily(
            4,
            discrimination=Prior("LogNormal", 0.0, 0.5),
            discrimination_group=DiscriminationGroup(ITEM, declared),
        )


@pytest.mark.parametrize("index", ["cutpoints", "discrimination_rows"])
def test_a_discrimination_group_indexed_by_a_name_the_family_binds_is_refused(
    index: str,
) -> None:
    with pytest.raises(
        ValueError, match=f"must not name what the family binds: {index!r}"
    ):
        CumulativeFamily(
            4,
            discrimination=Prior("LogNormal", 0.0, 0.5),
            discrimination_group=DiscriminationGroup(index, "Item"),
        )


def test_a_predictor_naming_the_discrimination_group_index_is_refused() -> None:
    family = discriminating_family()

    with pytest.raises(ValueError, match="must not be a name this family binds"):
        family.steps(predictor=ITEM, response=RESPONSE, plate=PLATE)


def test_a_discrimination_group_object_cannot_be_the_plate() -> None:
    family = discriminating_family()

    with pytest.raises(ValueError, match="must not be an object the family declares"):
        family.steps(predictor="predictor", response=RESPONSE, plate="Item")


def test_a_discrimination_over_a_group_is_drawn_per_level_and_scales_each_row() -> None:
    family = discriminating_family(5)
    steps = emitted(family, predictor="predictor", response=RESPONSE, plate=PLATE)

    trace = discriminating_trace(family)
    items = torch.arange(OBSERVATIONS) % ITEMS

    assert steps[-6:-1] == (
        Draw("discrimination", ("Item",), Prior("LogNormal", 0.0, 0.5)),
        Bind("discrimination_rows", Gather("discrimination", ITEM)),
        Bind(
            "discrimination_column",
            Table("discrimination_rows", PLATE, "Unit", width=1),
        ),
        Bind(
            "scaled_predictor",
            Binary(Operator.MUL, Name("discrimination_rows"), Name("predictor")),
        ),
        Bind(
            "scaled_cutpoints",
            Binary(Operator.MUL, Name("discrimination_column"), Name("cutpoints")),
        ),
    )
    assert names_bound_by(steps) == (*family.bound_names, RESPONSE)
    assert family.host_columns == (ITEM,)
    assert family.discrimination_site in family.sites
    assert trace.sites[family.discrimination_site].value.shape == (ITEMS,)
    assert trace.sites[family.discrimination_rows_site].value.shape == (OBSERVATIONS,)
    assert trace.sites[family.discrimination_column_site].value.shape == (
        OBSERVATIONS,
        1,
    )
    assert torch.allclose(
        trace.sites[family.discrimination_rows_site].value, DISCRIMINATIONS[items]
    )
    assert torch.allclose(
        trace.sites[family.scaled_predictor_site].value,
        DISCRIMINATIONS[items] * trace.sites["predictor"].value,
    )
    assert torch.allclose(
        trace.sites[family.scaled_cutpoint_site].value,
        DISCRIMINATIONS[items].unsqueeze(1) * trace.sites[family.cutpoint_site].value,
    )


def test_a_discrimination_group_beside_a_threshold_group_reads_both_indices_once() -> (
    None
):
    beside = discriminating_family(5, group=ThresholdGroup(PARTICIPANT, "Participant"))
    same = CumulativeFamily(
        5,
        discrimination=Prior("LogNormal", 0.0, 0.5),
        group=ThresholdGroup(ITEM, "Item"),
        discrimination_group=DiscriminationGroup(ITEM, "Item"),
    )

    trace = discriminating_trace(beside)

    assert beside.host_columns == (PARTICIPANT, ITEM)
    assert same.host_columns == (ITEM,)
    assert trace.sites[beside.scaled_cutpoint_site].value.shape == (
        OBSERVATIONS,
        beside.thresholds,
    )
    assert names_bound_by(
        emitted(beside, predictor="predictor", response=RESPONSE, plate=PLATE)
    ) == (*beside.bound_names, RESPONSE)


def test_a_threshold_group_publishes_its_names_and_its_spread_prior() -> None:
    group = ThresholdGroup("annotator", "Annotator")

    assert group.index == "annotator"
    assert group.index_object == "Annotator"
    assert group.spread is DEFAULT_SPREAD_PRIOR
    assert group.spread_site == "annotator_cutpoint_spread"
    assert group.offset_site == "annotator_cutpoint_offsets"
    assert group.shift_site == "annotator_cutpoint_shifts"
    assert group.cutpoint_site == "annotator_cutpoints"
    assert repr(group) == (
        "ThresholdGroup('annotator', 'Annotator', spread=Prior('HalfNormal', 1.0))"
    )


def test_a_threshold_group_index_and_index_object_must_be_identifiers() -> None:
    with pytest.raises(ValueError, match="threshold group's index must be"):
        ThresholdGroup("by participant", "Participant")

    with pytest.raises(ValueError, match="threshold group's index object must be"):
        ThresholdGroup(PARTICIPANT, "Participant[0]")


def test_a_threshold_group_index_and_index_object_must_differ() -> None:
    with pytest.raises(ValueError, match="index and index object must differ"):
        ThresholdGroup(PARTICIPANT, PARTICIPANT)


@pytest.mark.parametrize("declared", ["Cutpoint", "Unit"])
def test_a_group_indexed_by_an_object_the_family_declares_is_refused(
    declared: str,
) -> None:
    group = ThresholdGroup(PARTICIPANT, declared)

    with pytest.raises(ValueError, match="must not be an object the family declares"):
        CumulativeFamily(4, group=group)


@pytest.mark.parametrize("index", ["cutpoints", "cutpoint_base", "discrimination"])
def test_a_group_indexed_by_a_name_the_family_binds_is_refused(index: str) -> None:
    group = ThresholdGroup(index, "Participant")

    with pytest.raises(
        ValueError, match=f"must not name what the family binds: \\[{index!r}"
    ):
        CumulativeFamily(4, group=group)


@pytest.mark.parametrize("discrimination", [None, Prior("LogNormal", 0.0, 1.0)])
def test_a_group_whose_cutpoint_table_the_family_binds_is_refused(
    discrimination: Prior | None,
) -> None:
    group = ThresholdGroup("scaled", "Scaled")

    with pytest.raises(ValueError, match="family binds: \\['scaled_cutpoints'\\]"):
        CumulativeFamily(4, group=group, discrimination=discrimination)


def test_a_grouped_family_binds_every_name_once() -> None:
    family = grouped_family(discrimination=Prior("LogNormal", 0.0, 1.0))
    bound = family.bound_names

    assert len(bound) == len(frozenset(bound))
    assert not frozenset(bound) & frozenset(family.host_columns)


def test_a_grouped_family_declares_its_own_objects_and_reads_the_group_index() -> None:
    family = grouped_family()

    assert (family.index_object, family.scalar_object) == ("Cutpoint", "Unit")
    assert family.thresholds == 3
    assert family.host_columns == (PARTICIPANT,)


def test_a_fixed_family_reads_no_host_column() -> None:
    assert CumulativeFamily(4).host_columns == ()


def test_a_grouped_family_draws_one_spread_per_threshold_and_one_offset_per_cell() -> (
    None
):
    family = grouped_family(5)
    group = family.group

    assert group is not None

    trace = trace_of(family)

    assert trace.sites[group.spread_site].value.shape == (family.thresholds,)
    assert trace.sites[group.offset_site].value.numel() == (
        PARTICIPANTS * family.thresholds
    )
    assert trace.sites[group.shift_site].value.shape == (
        PARTICIPANTS,
        family.thresholds,
    )


def test_a_grouped_family_compiles_to_one_row_of_cutpoints_per_level() -> None:
    family = grouped_family(5)
    group = family.group

    assert group is not None

    trace = trace_of(family)

    assert trace.sites[group.cutpoint_site].value.shape == (
        PARTICIPANTS,
        family.thresholds,
    )
    assert trace.sites[family.cutpoint_site].value.shape == (
        OBSERVATIONS,
        family.thresholds,
    )


def test_a_fixed_family_compiles_to_a_single_row_of_cutpoints() -> None:
    family = CumulativeFamily(5)

    trace = trace_of(family)

    assert trace.sites[family.span_site].value.shape == (family.thresholds,)
    assert trace.sites[family.cutpoint_site].value.shape == (family.thresholds,)


def test_each_row_takes_the_cutpoints_of_its_own_level_through_the_index() -> None:
    family = grouped_family(4)
    group = family.group

    assert group is not None

    offsets = torch.tensor([[-2.0, 0.0, 2.0], [0.0, 0.0, 0.0], [2.0, 0.0, -2.0]])
    participants = torch.tensor([2, 0, 1, 1, 0, 2, 2, 0])
    trace = trace_of(
        family,
        {
            family.base_site: torch.zeros(1),
            family.increment_site: torch.zeros(family.thresholds),
            group.spread_site: torch.ones(family.thresholds),
            group.offset_site: offsets.reshape(-1),
        },
        supplied={PARTICIPANT: participants},
    )
    by_level = trace.sites[group.cutpoint_site].value
    by_row = trace.sites[family.cutpoint_site].value

    assert torch.equal(trace.sites[group.shift_site].value, offsets)
    assert torch.equal(by_row, by_level[participants])
    assert not torch.allclose(by_level[0], by_level[2])


def test_a_level_shift_moves_every_cutpoint_of_that_level_from_the_shift_up() -> None:
    family = grouped_family(4)
    group = family.group

    assert group is not None

    offsets = torch.tensor([[0.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    trace = trace_of(
        family,
        {
            family.base_site: torch.zeros(1),
            family.increment_site: torch.zeros(family.thresholds),
            group.spread_site: torch.ones(family.thresholds),
            group.offset_site: offsets.reshape(-1),
        },
        levels=2,
    )
    by_level = trace.sites[group.cutpoint_site].value
    moved = by_level[1] - by_level[0]

    assert float(moved[0]) == pytest.approx(0.0)
    assert float(moved[1]) > 0.0
    assert float(moved[2]) == pytest.approx(float(moved[1]))


def test_the_spread_scales_the_shifts_of_its_own_threshold_alone() -> None:
    family = grouped_family(4)
    group = family.group

    assert group is not None

    offsets = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    def spans_at(spread: torch.Tensor) -> torch.Tensor:
        trace = trace_of(
            family,
            {
                family.base_site: torch.zeros(1),
                family.increment_site: torch.zeros(family.thresholds),
                group.spread_site: spread,
                group.offset_site: offsets.reshape(-1),
            },
            levels=2,
        )

        return trace.sites[family.span_site].value

    narrow = spans_at(torch.tensor([1.0, 1.0, 1.0]))
    widened = spans_at(torch.tensor([1.0, 4.0, 1.0]))
    gap = widened[1] - narrow[1]

    assert torch.allclose(widened[0], narrow[0])
    assert float(gap[0]) == pytest.approx(0.0)
    assert float(gap[1]) > 0.0
    assert float(gap[2]) == pytest.approx(float(gap[1]))


def test_the_grouped_cutpoints_rise_within_every_level() -> None:
    family = grouped_family(6)
    group = family.group

    assert group is not None

    generator = torch.Generator().manual_seed(3)
    trace = trace_of(
        family,
        {
            family.base_site: torch.zeros(1),
            family.increment_site: torch.randn(family.thresholds, generator=generator),
            group.spread_site: torch.full((family.thresholds,), 3.0),
            group.offset_site: torch.randn(5 * family.thresholds, generator=generator),
        },
        observations=10,
        levels=5,
    )

    assert bool((trace.sites[group.cutpoint_site].value.diff(dim=-1) > 0).all())


@pytest.mark.parametrize(
    "discrimination", [None, Prior("LogNormal", 0.0, 0.5)], ids=["fixed", "drawn"]
)
def test_a_grouped_family_publishes_exactly_the_names_its_steps_bind(
    discrimination: Prior | None,
) -> None:
    family = grouped_family(5, discrimination=discrimination)

    steps = emitted(family, predictor="predictor", response=RESPONSE, plate=PLATE)
    drawn = tuple(step.site for step in steps if isinstance(step, Draw))

    assert names_bound_by(steps) == (*family.bound_names, RESPONSE)
    assert drawn == family.sites


def test_a_grouped_family_draws_exactly_the_latent_sites_it_publishes() -> None:
    family = grouped_family(4)

    trace = trace_of(
        family, supplied={RESPONSE: torch.tensor([0, 1, 2, 3, 1, 2, 3, 0])}
    )

    assert set(trace.latent_sites) == {"slope", *family.sites}


def test_the_group_spread_site_names_the_prior_it_was_given() -> None:
    family = grouped_family(4, spread=Prior("Exponential", 2.0))
    group = family.group

    assert group is not None

    steps = emitted(family, predictor="predictor", response=RESPONSE, plate=PLATE)
    drawn = {step.site: step.prior for step in steps if isinstance(step, Draw)}

    assert drawn[group.spread_site] == Prior("Exponential", 2.0)
    assert drawn[group.offset_site] == Prior("Normal", 0.0, 1.0)


def test_a_predictor_naming_the_group_index_is_refused() -> None:
    family = grouped_family(4)

    with pytest.raises(ValueError, match="a predictor must not be a name"):
        family.steps(predictor=PARTICIPANT, response=RESPONSE, plate=PLATE)


def test_a_plate_naming_the_group_object_is_refused() -> None:
    family = grouped_family(4)

    with pytest.raises(ValueError, match="must not be an object the family declares"):
        family.steps(predictor="predictor", response=RESPONSE, plate="Participant")


class Synthetic(NamedTuple):
    responses: torch.Tensor
    covariate: torch.Tensor
    participants: torch.Tensor
    spread: float


def synthetic_judgments(
    family: CumulativeFamily,
    *,
    levels: int,
    per_level: int,
    spread: float,
    seed: int,
) -> Synthetic:
    generator = torch.Generator().manual_seed(seed)
    offsets = torch.randn(levels, family.thresholds, generator=generator)
    increments = 1.0 + spread * offsets
    cutpoints = -1.5 + torch.cumsum(torch.nn.functional.softplus(increments), dim=-1)
    participants = torch.arange(levels).repeat_interleave(per_level)
    covariate = torch.randn(participants.numel(), 1, generator=generator)
    probabilities = ordered_logistic_probabilities(cutpoints[participants], covariate)
    responses = torch.multinomial(probabilities, 1, generator=generator).reshape(-1)

    return Synthetic(responses, covariate, participants, spread)


def distribution_of(prior: Prior) -> distributions.Distribution:
    match prior.family, prior.arguments:
        case "Normal", (location, scale):
            return distributions.Normal(location, scale)
        case "HalfNormal", (scale,):
            return distributions.HalfNormal(scale)

    msg = f"no torch distribution stands in for the prior {prior!r}"

    raise ValueError(msg)


class Laplace(NamedTuple):
    mode: torch.Tensor
    scale: torch.Tensor


def laplace_at_the_mode(
    log_joint: Callable[[torch.Tensor], torch.Tensor], dimension: int
) -> Laplace:
    """Return the joint mode and the Cholesky factor of the curvature there."""
    mode = torch.zeros(dimension, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [mode], lr=0.5, max_iter=200, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = -log_joint(mode)
        loss.backward()

        return loss

    for _ in range(3):
        optimizer.step(closure)

    located = mode.detach()
    hessian = torch.autograd.functional.hessian(lambda z: -log_joint(z), located)

    return Laplace(located, torch.linalg.cholesky(torch.linalg.inv(hessian)))


class Chain(NamedTuple):
    draws: int
    warmup: int
    steps: int
    step_size: float
    seed: int


def hamiltonian_draws(
    log_joint: Callable[[torch.Tensor], torch.Tensor], dimension: int, chain: Chain
) -> torch.Tensor:
    """Return posterior draws by Hamiltonian Monte Carlo from the joint mode.

    The chain moves in coordinates whitened by the Laplace approximation at
    the mode, so one step size serves every dimension.
    """
    laplace = laplace_at_the_mode(log_joint, dimension)
    generator = torch.Generator().manual_seed(chain.seed)
    steps, step_size = chain.steps, chain.step_size

    def evaluate(whitened: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        whitened = whitened.detach().requires_grad_()
        density = log_joint(laplace.mode + laplace.scale @ whitened)
        (gradient,) = torch.autograd.grad(density, whitened)

        return density.detach(), gradient

    def leapfrog(
        position: torch.Tensor, momentum: torch.Tensor, gradient: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        moved = momentum + 0.5 * step_size * gradient
        density = torch.zeros(())

        for step in range(steps):
            position = position + step_size * moved
            density, gradient = evaluate(position)
            half = step_size if step < steps - 1 else 0.5 * step_size
            moved = moved + half * gradient

        return position, moved, density, gradient

    position = torch.zeros(dimension)
    density, gradient = evaluate(position)
    kept = []

    for iteration in range(chain.warmup + chain.draws):
        momentum = torch.randn(dimension, generator=generator)
        energy = 0.5 * (momentum * momentum).sum() - density
        proposed, moved, proposed_density, proposed_gradient = leapfrog(
            position, momentum, gradient
        )
        proposed_energy = 0.5 * (moved * moved).sum() - proposed_density

        if torch.log(torch.rand((), generator=generator)) < energy - proposed_energy:
            position, density, gradient = proposed, proposed_density, proposed_gradient

        if iteration >= chain.warmup:
            kept.append(laplace.mode + laplace.scale @ position)

    return torch.stack(kept)


def test_the_generating_threshold_spread_lies_inside_the_posterior_interval() -> None:
    family = grouped_family(4)
    group = family.group

    assert group is not None

    levels = 16
    data = synthetic_judgments(family, levels=levels, per_level=100, spread=1.0, seed=5)
    observations = data.responses.numel()
    sizes = {
        "slope": 1,
        family.base_site: 1,
        family.increment_site: family.thresholds,
        group.spread_site: family.thresholds,
        group.offset_site: levels * family.thresholds,
    }
    priors = {
        "slope": distributions.Normal(0.0, 1.0),
        family.base_site: distribution_of(family.base),
        family.increment_site: distribution_of(family.increment),
        group.spread_site: distribution_of(group.spread),
        group.offset_site: distributions.Normal(0.0, 1.0),
    }

    def unpack(latent: torch.Tensor) -> dict[str, torch.Tensor]:
        parts = dict(zip(sizes, torch.split(latent, list(sizes.values())), strict=True))
        parts[group.spread_site] = torch.exp(parts[group.spread_site])

        return parts

    def site_of(latent: torch.Tensor, name: str) -> torch.Tensor:
        trace = trace_of(
            family,
            unpack(latent),
            observations=observations,
            supplied={
                RESPONSE: data.responses,
                "covariate": data.covariate,
                PARTICIPANT: data.participants,
            },
            levels=levels,
        )

        return (
            trace.sites[name].value
            if name != RESPONSE
            else (trace.sites[name].log_prob.sum())
        )

    def log_joint(latent: torch.Tensor) -> torch.Tensor:
        parts = unpack(latent)
        log_prior = sum(
            prior.log_prob(parts[name]).sum() for name, prior in priors.items()
        )
        log_prior = log_prior + torch.log(parts[group.spread_site]).sum()

        return site_of(latent, RESPONSE) + log_prior

    chain = Chain(draws=400, warmup=60, steps=5, step_size=0.25, seed=7)
    draws = hamiltonian_draws(log_joint, sum(sizes.values()), chain)
    spreads = torch.stack([site_of(draw, group.spread_site) for draw in draws])
    lower, upper = spreads.quantile(torch.tensor([0.025, 0.975]), dim=0)

    assert spreads.shape == (chain.draws, family.thresholds)
    assert bool(((lower <= data.spread) & (data.spread <= upper)).all())
    assert bool(((data.spread / 2 < lower) & (upper < 2 * data.spread)).all())


def test_a_held_discrimination_scales_by_its_literal_and_draws_nothing() -> None:
    family = CumulativeFamily(5, discrimination=2.5)

    steps = emitted(family, predictor="predictor", response=RESPONSE, plate=PLATE)
    trace = trace_of(family)

    assert family.discrimination == 2.5
    assert family.discrimination_site not in family.sites
    assert family.discrimination_site not in family.bound_names
    assert family.scaled_predictor_site in family.bound_names
    assert steps[-3:] == (
        Bind("scaled_predictor", Binary(Operator.MUL, Literal(2.5), Name("predictor"))),
        Bind("scaled_cutpoints", Binary(Operator.MUL, Literal(2.5), Name("cutpoints"))),
        ObserveOrdered(
            RESPONSE,
            PLATE,
            location="scaled_predictor",
            thresholds="scaled_cutpoints",
        ),
    )
    assert names_bound_by(steps) == (*family.bound_names, RESPONSE)
    assert torch.allclose(
        trace.sites[family.scaled_cutpoint_site].value,
        2.5 * trace.sites[family.cutpoint_site].value,
    )
    assert family.discrimination_site not in trace.sites


@pytest.mark.parametrize("held", [0.0, -1.0, float("inf"), float("nan")])
def test_a_held_discrimination_must_be_a_finite_positive_number(held: float) -> None:
    with pytest.raises(ValueError, match="not a finite positive number"):
        CumulativeFamily(5, discrimination=held)


@pytest.mark.parametrize("held", [True, False])
def test_a_held_discrimination_that_is_a_boolean_is_refused(held: object) -> None:
    with pytest.raises(TypeError, match="not a number"):
        CumulativeFamily(5, discrimination=cast("float", held))


PREDICTED = "log_discrimination"
LOG_DISCRIMINATIONS = torch.log(torch.tensor([0.5, 1.0, 1.5, 2.0, 2.0, 1.5, 1.0, 0.5]))


def predicting_family(categories: int = 4) -> CumulativeFamily:
    return CumulativeFamily(categories, discrimination=PredictedParameter(PREDICTED))


def predicting_trace(family: CumulativeFamily) -> Trace:
    return trace_of(
        family,
        supplied={
            "covariate": torch.linspace(-1.0, 1.0, OBSERVATIONS),
            PREDICTED: LOG_DISCRIMINATIONS,
        },
    )


def test_a_predicted_parameter_publishes_the_variable_it_reads() -> None:
    predicted = PredictedParameter(PREDICTED)

    assert predicted.variable == PREDICTED
    assert repr(predicted) == "PredictedParameter('log_discrimination')"


def test_a_predicted_parameter_reads_a_variable_named_by_an_identifier() -> None:
    with pytest.raises(ValueError, match="predicted parameter's variable must be"):
        PredictedParameter("log discrimination")


def test_a_predicted_discrimination_is_the_exponential_of_its_own_predictor() -> None:
    family = predicting_family(5)

    steps = emitted(family, predictor="predictor", response=RESPONSE, plate=PLATE)

    assert steps[-5:-1] == (
        Bind("discrimination", Apply(Function.EXP, Name(PREDICTED))),
        Bind("discrimination_column", Table("discrimination", PLATE, "Unit", width=1)),
        Bind(
            "scaled_predictor",
            Binary(Operator.MUL, Name("discrimination"), Name("predictor")),
        ),
        Bind(
            "scaled_cutpoints",
            Binary(Operator.MUL, Name("discrimination_column"), Name("cutpoints")),
        ),
    )
    assert names_bound_by(steps) == (*family.bound_names, RESPONSE)
    assert family.discrimination_site not in family.sites
    assert family.discrimination_rows_site not in family.bound_names
    assert family.host_columns == ()


def test_a_predicted_discrimination_scales_each_row_by_its_own_value() -> None:
    family = predicting_family(5)

    trace = predicting_trace(family)
    discriminations = torch.exp(LOG_DISCRIMINATIONS)

    assert torch.allclose(
        trace.sites[family.discrimination_site].value, discriminations
    )
    assert trace.sites[family.discrimination_column_site].value.shape == (
        OBSERVATIONS,
        1,
    )
    assert torch.allclose(
        trace.sites[family.scaled_predictor_site].value,
        discriminations * trace.sites["predictor"].value,
    )
    assert torch.allclose(
        trace.sites[family.scaled_cutpoint_site].value,
        discriminations.unsqueeze(1) * trace.sites[family.cutpoint_site].value,
    )
    assert trace.sites[family.scaled_cutpoint_site].value.shape == (
        OBSERVATIONS,
        family.thresholds,
    )


def test_a_family_holding_its_discrimination_leaves_every_row_the_same() -> None:
    family = CumulativeFamily(5)

    trace = trace_of(
        family, supplied={"covariate": torch.linspace(-1.0, 1.0, OBSERVATIONS)}
    )

    assert family.discrimination is None
    assert family.discrimination_site not in family.bound_names
    assert family.scaled_cutpoint_site not in family.bound_names
    assert trace.sites[family.cutpoint_site].value.shape == (family.thresholds,)


@pytest.mark.parametrize(
    "variable", ["discrimination", "cutpoints", "scaled_predictor"]
)
def test_a_predicted_discrimination_reading_a_family_name_is_refused(
    variable: str,
) -> None:
    with pytest.raises(ValueError, match="must not read what the family binds"):
        CumulativeFamily(4, discrimination=PredictedParameter(variable))


def test_a_predicted_discrimination_beside_a_discrimination_group_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot also vary over"):
        CumulativeFamily(
            4,
            discrimination=PredictedParameter(PREDICTED),
            discrimination_group=DiscriminationGroup(ITEM, "Item"),
        )


def test_a_predictor_naming_the_variable_the_discrimination_follows_is_refused() -> (
    None
):
    family = predicting_family()

    with pytest.raises(ValueError, match="a predictor and the predictor a"):
        family.steps(predictor=PREDICTED, response=RESPONSE, plate=PLATE)


def test_a_response_naming_the_variable_the_discrimination_follows_is_refused() -> None:
    family = predicting_family()

    with pytest.raises(ValueError, match="a response and the predictor a"):
        family.steps(predictor="predictor", response=PREDICTED, plate=PLATE)

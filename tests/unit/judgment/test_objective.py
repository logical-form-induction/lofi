import math
from itertools import pairwise
from typing import TYPE_CHECKING, Final, override

import didactic.api as dx
import pytest
import torch
from tests.conformance.objective import ObjectiveLawSuite

from lofi._types import ReadoutId
from lofi.deduce import SpanIndex
from lofi.judgment import (
    COMPLEMENT,
    LOGIT,
    PROBABILITY,
    SLOPE,
    DesignCodec,
    GroupedJudgmentSet,
    JudgmentLikelihoodObjective,
    JudgmentRow,
    LinkingDomain,
    LinkingSpec,
    LinkTerm,
    LogitLinking,
    OrdinalHead,
    ProbabilityLinking,
    ReadoutColumns,
    RowColumns,
    SpecialCase,
    level_residuals,
    log_category_probabilities,
)
from lofi.nn._values import LogDistribution
from lofi.nn.loss._errors import (
    ImproperMassError,
    UncomputedCellError,
)
from lofi.nn.loss._parser import SPAN_AXES
from lofi.nn.loss._ports import Objective
from lofi.nn.loss._readout import (
    MASS_AXES,
    log_mass,
)
from lofi.nn.loss._suite import ObjectiveSuite
from lofi.nn.loss._values import (
    CellChart,
    ObjectiveOutputs,
    RecordingOutputs,
    SpanReadout,
)
from lofi.nn.loss.laws import (
    ObjectiveFixture,
    the_value_is_monotone_where_the_fixture_says,
)
from lofi.stats import Interval

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from lofi.judgment import Linking
    from lofi.judgment.grouped import _GroupedJudgmentBatch
    from lofi.nn.loss._ports import LossTerm

LOG_MASS: Final = "p_gram"
LOG_COMPLEMENT: Final = "p_gram_complement"
ROOT_MASS: Final = ReadoutId("sentences")
LEVELS: Final = 5
CUTPOINTS: Final = (-2.0, -0.5, 1.0, 2.5)
LENGTH: Final = 3
ITEMS: Final = 3
LENGTHS: Final = (3, 2, 3)
RESPONSES: Final = (
    ("participant", "first", 4),
    ("participant", "first", 2),
    ("participant", "second", 0),
    ("participant", "third", 3),
    ("participant", "second", 1),
)
SPANS: Final = SpanIndex(LENGTH)
FILL: Final = -9.0
IDS: Final = ("first", "second", "third")


type Outputs = ObjectiveOutputs[_GroupedJudgmentBatch]


def logit_term(slope: float | None = None) -> LinkTerm:
    fixed = {} if slope is None else {SLOPE: slope}

    return LinkTerm(
        readout=LOG_MASS,
        linking=LinkingSpec(
            family=LOGIT, columns={COMPLEMENT: LOG_COMPLEMENT}, fixed=fixed
        ),
    )


def logit_linking(
    *, unit_slope: bool = False, slope: float | None = None
) -> LogitLinking:
    return LogitLinking(logit_term(slope), unit_slope=unit_slope)


def probability_linking() -> ProbabilityLinking:
    return ProbabilityLinking(
        LinkTerm(readout=LOG_MASS, linking=LinkingSpec(family=PROBABILITY))
    )


def objective(
    *,
    linking: Linking | None = None,
    levels: int = LEVELS,
    cutpoints: tuple[float, ...] | None = CUTPOINTS,
    parameters: tuple[float, ...] | None = None,
    readout: ReadoutId = ROOT_MASS,
    name: str = "judgment",
) -> JudgmentLikelihoodObjective:
    built = JudgmentLikelihoodObjective(
        readout=readout,
        linking=logit_linking() if linking is None else linking,
        head=OrdinalHead(levels, cutpoints),
        parameters=parameters,
        name=name,
    )

    return built.double()


def live_spans(lengths: tuple[int, ...] = LENGTHS) -> Tensor:
    mask = torch.zeros(len(lengths), SPANS.num_spans, dtype=torch.bool)

    for item, length in enumerate(lengths):
        for start, end in SPANS.spans_by_length():
            if end <= length:
                mask[item, SPANS.span_id(start, end)] = True

    return mask


LIVE: Final = live_spans()


def roots(lengths: tuple[int, ...] = LENGTHS) -> list[int]:
    return [SPANS.span_id(0, length) for length in lengths]


def random_log_masses(seed: int, items: int = ITEMS) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    masses = 0.05 + 0.9 * torch.rand(
        items, SPANS.num_spans, generator=generator, dtype=torch.float64
    )

    return masses.log()


def pair_of(
    log_masses: Tensor,
    *,
    computed: Tensor = LIVE,
    fill: float = FILL,
) -> SpanReadout:
    rest = torch.log(-torch.expm1(log_masses))
    pair = torch.stack((log_masses, rest), dim=-1)
    padded = torch.where(computed.unsqueeze(-1), pair, fill)

    return SpanReadout(
        SPANS,
        LogDistribution(
            padded.detach().requires_grad_(),
            MASS_AXES,
            computed,
            normalized_over="outcome",
        ),
    )


def assemble(
    log_masses: Tensor,
    *,
    lengths: tuple[int, ...] = LENGTHS,
    responses: tuple[tuple[str, str, int], ...] = RESPONSES,
    computed: Tensor = LIVE,
    fill: float = FILL,
    ids: tuple[str, ...] = IDS,
    key: ReadoutId = ROOT_MASS,
) -> Outputs:
    used = frozenset(item_id for _, item_id, _ in responses)
    selected = tuple(
        position for position, item_id in enumerate(ids) if item_id in used
    )
    positions = torch.tensor(selected, dtype=torch.int64)
    selected_ids = tuple(ids[position] for position in selected)
    selected_lengths = tuple(lengths[position] for position in selected)
    selected_log_masses = log_masses.index_select(0, positions)
    selected_computed = computed.index_select(0, positions)
    design = DesignCodec(LEVELS).encode(
        JudgmentRow(participant, item_id, response, RowColumns({}, {}))
        for participant, item_id, response in responses
    )
    grouped = GroupedJudgmentSet(
        design,
        selected_ids,
        tuple(
            tuple(f"word-{position}-{word}" for word in range(length))
            for position, length in zip(selected, selected_lengths, strict=True)
        ),
    )
    batch = grouped.select(tuple(range(grouped.group_count)))

    return ObjectiveOutputs(
        spans=SPANS,
        chart=CellChart(
            LogDistribution(
                torch.where(selected_computed, -1.0, fill),
                SPAN_AXES,
                selected_computed,
            )
        ),
        batch=batch,
        readouts={
            key: pair_of(
                selected_log_masses,
                computed=selected_computed,
                fill=fill,
            )
        },
    )


def bundle(seed: int = 0) -> Outputs:
    return assemble(random_log_masses(seed))


def log_masses_of(outputs: Outputs) -> Tensor:
    return log_mass(outputs.readouts[ROOT_MASS].distribution)


def computed_of(outputs: Outputs) -> Tensor:
    return outputs.readouts[ROOT_MASS].computed


def permuted(outputs: Outputs) -> Outputs:
    order = [2, 0, 1]
    batch = outputs.batch

    return assemble(
        log_masses_of(outputs)[torch.tensor(order)],
        lengths=tuple(batch.lengths[position] for position in order),
        responses=tuple(
            (
                batch.design.participant_levels[int(participant)],
                batch.design.item_levels[int(item)],
                int(response),
            )
            for participant, item, response in reversed(
                tuple(
                    zip(
                        batch.design.participants,
                        batch.design.items,
                        batch.design.responses,
                        strict=True,
                    )
                )
            )
        ),
        computed=computed_of(outputs)[torch.tensor(order)],
        ids=tuple(batch.item_ids[position] for position in order),
    )


def refilled(outputs: Outputs) -> Outputs:
    return assemble(
        log_masses_of(outputs),
        lengths=outputs.batch.lengths,
        responses=tuple(
            (
                outputs.batch.design.participant_levels[int(participant)],
                outputs.batch.design.item_levels[int(item)],
                int(response),
            )
            for participant, item, response in zip(
                outputs.batch.design.participants,
                outputs.batch.design.items,
                outputs.batch.design.responses,
                strict=True,
            )
        ),
        computed=computed_of(outputs),
        fill=FILL / 3.0,
        ids=outputs.batch.item_ids,
    )


def inputs_of(outputs: Outputs) -> tuple[Tensor, ...]:
    return (outputs.readouts[ROOT_MASS].distribution.tensor,)


def scores_of(outputs: Outputs) -> list[float]:
    read = log_masses_of(outputs).detach()

    return [
        math.exp(read[item, span].item())
        for item, span in enumerate(roots(outputs.batch.lengths))
    ]


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def level_probabilities(
    latent: float, cutpoints: tuple[float, ...] = CUTPOINTS
) -> list[float]:
    cumulative = [sigmoid(cutpoint - latent) for cutpoint in cutpoints]

    return [
        cumulative[0],
        *(above - below for below, above in pairwise(cumulative)),
        1.0 - cumulative[-1],
    ]


def latent_of(score: float, *, slope: float = 1.0, through_mass: bool = False) -> float:
    if through_mass:
        return score

    return slope * (math.log(score) - math.log1p(-score))


def by_hand(
    outputs: Outputs,
    *,
    slope: float = 1.0,
    cutpoints: tuple[float, ...] = CUTPOINTS,
    through_mass: bool = False,
) -> float:
    scores = scores_of(outputs)
    read: list[float] = []

    for item, level in zip(
        outputs.batch.response_items,
        outputs.batch.design.responses,
        strict=True,
    ):
        latent = latent_of(scores[item], slope=slope, through_mass=through_mass)
        read.append(-math.log(level_probabilities(latent, cutpoints)[int(level)]))

    return sum(read) / len(read)


def with_score(
    score: float, *, elsewhere: float = 0.5, lengths: tuple[int, ...] = LENGTHS
) -> Tensor:
    held = torch.full((len(lengths), SPANS.num_spans), math.log(elsewhere)).double()

    for item, span in enumerate(roots(lengths)):
        held[item, span] = math.log(score)

    return held


def models_built(call: Callable[[], LossTerm]) -> int:
    counted = 0
    original = dx.Model.__init__

    def counting(model: dx.Model, /, **fields: int) -> None:
        nonlocal counted
        counted += 1

        original(model, **fields)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(dx.Model, "__init__", counting)
        call()

    return counted


TOP: Final[tuple[tuple[str, str, int], ...]] = tuple(
    ("participant", item_id, LEVELS - 1) for item_id in IDS
)


def fixture() -> ObjectiveFixture[_GroupedJudgmentBatch]:
    return ObjectiveFixture(
        objective=objective(),
        outputs=bundle,
        inputs=inputs_of,
        unweighted=by_hand,
        monotone=lambda: (
            assemble(with_score(0.2), responses=TOP),
            assemble(with_score(0.9), responses=TOP),
        ),
        models_built=models_built,
        permuted=permuted,
        masked=refilled,
        tolerance=1e-9,
    )


class TestJudgmentLikelihoodObjectiveLaws(ObjectiveLawSuite):
    @pytest.fixture
    def conformance(self) -> ObjectiveFixture[_GroupedJudgmentBatch]:
        return fixture()

    @override
    def test_the_value_is_monotone_where_the_fixture_says[BatchT](
        self,
        conformance: ObjectiveFixture[BatchT],
    ) -> None:
        """Check that less root mass under top-level judgments scores above more."""
        the_value_is_monotone_where_the_fixture_says(conformance)


def test_the_objective_answers_the_port_and_declares_what_it_reads() -> None:
    scored = objective()

    assert isinstance(scored, Objective)
    assert scored.name == "judgment"
    assert scored.requires == frozenset({"batch", "readouts"})
    assert scored.readout == ROOT_MASS
    assert scored.levels == LEVELS
    assert objective(name="acceptability").name == "acceptability"
    assert "sentences" in repr(scored)
    assert "LogitLinking" in repr(scored)


def test_the_value_is_the_mean_negative_log_density_of_the_judgments() -> None:
    outputs = bundle()

    term = objective().evaluate(outputs)

    assert term.value.ndim == 0
    assert term.value.item() == pytest.approx(by_hand(outputs), rel=1e-12)
    assert term.value.item() > 0.0
    assert term.count == len(RESPONSES)
    assert term.residual is None


def test_row_and_participant_group_order_do_not_change_the_objective() -> None:
    records = (
        ("participant-b", "first", 4),
        ("participant-a", "second", 1),
        ("participant-b", "third", 2),
        ("participant-a", "first", 3),
    )
    forward = objective().evaluate(assemble(random_log_masses(0), responses=records))
    reversed_rows = objective().evaluate(
        assemble(random_log_masses(0), responses=tuple(reversed(records)))
    )

    assert forward.count == reversed_rows.count
    assert torch.allclose(forward.value, reversed_rows.value)


def test_the_count_is_the_number_of_judgments_not_of_items() -> None:
    twice = (("participant", "first", 1), ("participant", "first", 3))

    term = objective().evaluate(assemble(random_log_masses(0), responses=twice))

    assert term.count == 2


def test_an_item_judged_twice_contributes_each_judgment() -> None:
    once = assemble(random_log_masses(0), responses=(("participant", "first", 1),))
    twice = assemble(
        random_log_masses(0),
        responses=(
            ("participant", "first", 1),
            ("participant", "first", 1),
        ),
    )

    single = objective().evaluate(once)
    doubled = objective().evaluate(twice)

    assert doubled.count == 2
    assert doubled.value.item() == pytest.approx(single.value.item(), rel=1e-12)
    assert doubled.diagnostics["log_density"].item() == pytest.approx(
        -single.value.item(), rel=1e-12
    )


def test_the_mean_predictive_log_density_is_a_detached_diagnostic() -> None:
    outputs = bundle()

    term = objective().evaluate(outputs)

    assert term.diagnostics["log_density"].item() == pytest.approx(
        -by_hand(outputs), rel=1e-12
    )
    assert term.diagnostics["log_density"].grad_fn is None
    assert term.value.grad_fn is not None


def test_the_per_level_residuals_are_observed_minus_predicted_frequencies() -> None:
    outputs = bundle()
    scores = scores_of(outputs)
    response_items = outputs.batch.response_items.tolist()
    responses = outputs.batch.design.responses.tolist()
    predicted = [
        level_probabilities(latent_of(scores[item])) for item in response_items
    ]

    term = objective().evaluate(outputs)

    for level in range(LEVELS):
        observed = sum(1.0 for held in responses if held == level) / len(responses)
        expected = sum(row[level] for row in predicted) / len(responses)
        residual = term.diagnostics[f"level_residual_{level}"]

        assert residual.item() == pytest.approx(observed - expected, rel=1e-9)
        assert residual.grad_fn is None

    summed = sum(
        term.diagnostics[f"level_residual_{level}"].item() for level in range(LEVELS)
    )

    assert summed == pytest.approx(0.0, abs=1e-12)


def test_level_residuals_vanish_where_the_prediction_matches_the_frequencies() -> None:
    log_probabilities = torch.log(
        torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float64)
    )
    responses = torch.tensor([0, 1])

    residuals = level_residuals(log_probabilities, responses)

    assert residuals.shape == (2,)
    assert residuals.abs().max().item() == pytest.approx(0.0, abs=1e-15)


def test_level_residuals_are_positive_where_a_level_is_over_observed() -> None:
    log_probabilities = torch.log(
        torch.tensor([[0.2, 0.8], [0.2, 0.8], [0.2, 0.8]], dtype=torch.float64)
    )
    responses = torch.tensor([0, 0, 1])

    residuals = level_residuals(log_probabilities, responses)

    assert residuals[0].item() == pytest.approx(2.0 / 3.0 - 0.2, rel=1e-12)
    assert residuals[1].item() == pytest.approx(1.0 / 3.0 - 0.8, rel=1e-12)


def test_the_dynamic_range_of_the_score_is_a_diagnostic_over_the_whole_batch() -> None:
    outputs = assemble(
        random_log_masses(3),
        responses=(("participant", "first", 4), ("participant", "third", 1)),
    )
    scores = scores_of(outputs)

    term = objective().evaluate(outputs)

    assert term.count == 2
    assert set(term.diagnostics) == {
        "log_density",
        "lowest",
        "highest",
        "range",
        *(f"level_residual_{level}" for level in range(LEVELS)),
    }
    assert term.diagnostics["lowest"].item() == pytest.approx(min(scores), rel=1e-12)
    assert term.diagnostics["highest"].item() == pytest.approx(max(scores), rel=1e-12)
    assert term.diagnostics["range"].item() == pytest.approx(
        max(scores) - min(scores), rel=1e-12
    )
    assert max(scores) - min(scores) > 0.0
    assert all(diagnostic.grad_fn is None for diagnostic in term.diagnostics.values())


def test_raising_the_root_mass_under_top_level_judgments_lowers_the_value() -> None:
    scored = objective()
    values = [
        scored.evaluate(assemble(with_score(score), responses=TOP)).value.item()
        for score in (0.1, 0.3, 0.5, 0.7, 0.9, 0.99)
    ]

    assert all(later < earlier for earlier, later in pairwise(values))
    assert values[0] == pytest.approx(
        -math.log(level_probabilities(latent_of(0.1))[LEVELS - 1]), rel=1e-12
    )


def test_raising_the_root_mass_under_bottom_level_judgments_raises_the_value() -> None:
    bottom = tuple(("participant", item_id, 0) for item_id in IDS)
    scored = objective()
    values = [
        scored.evaluate(assemble(with_score(score), responses=bottom)).value.item()
        for score in (0.1, 0.5, 0.9)
    ]

    assert all(later > earlier for earlier, later in pairwise(values))


@pytest.mark.parametrize("slope", [0.5, 2.0])
def test_the_linking_parameters_scale_the_log_odds(slope: float) -> None:
    outputs = bundle()

    term = objective(parameters=(slope,)).evaluate(outputs)

    assert term.value.item() == pytest.approx(by_hand(outputs, slope=slope), rel=1e-12)
    assert term.value.item() != pytest.approx(by_hand(outputs))


def test_a_pinned_slope_is_read_from_the_linking_and_fits_nothing() -> None:
    outputs = bundle()
    scored = objective(linking=logit_linking(slope=2.0))

    term = scored.evaluate(outputs)

    assert scored.parameters_of_linking.numel() == 0
    assert term.value.item() == pytest.approx(by_hand(outputs, slope=2.0), rel=1e-12)


def test_the_unit_slope_identification_fits_nothing() -> None:
    outputs = bundle()
    scored = objective(linking=logit_linking(unit_slope=True))

    term = scored.evaluate(outputs)

    assert scored.parameters_of_linking.numel() == 0
    assert term.value.item() == pytest.approx(by_hand(outputs), rel=1e-12)


def test_the_probability_link_reads_the_mass_alone() -> None:
    outputs = bundle()
    scored = objective(linking=probability_linking())

    term = scored.evaluate(outputs)

    assert scored.parameters_of_linking.numel() == 0
    assert term.value.item() == pytest.approx(
        by_hand(outputs, through_mass=True), rel=1e-12
    )


def test_a_fitted_parameter_starts_at_the_declared_closed_form() -> None:
    scored = objective(parameters=None)

    assert scored.parameters_of_linking.tolist() == [1.0]


def test_the_gradient_reaches_the_root_cells_and_the_head_parameters() -> None:
    outputs = assemble(
        random_log_masses(0),
        responses=(("participant", "first", 4), ("participant", "third", 1)),
    )
    scored = objective()

    scored.evaluate(outputs).value.backward()

    gradient = outputs.readouts[ROOT_MASS].distribution.tensor.grad
    touched = torch.zeros(2, SPANS.num_spans, 2, dtype=torch.bool)
    touched[0, SPANS.span_id(0, 3)] = True
    touched[1, SPANS.span_id(0, 3)] = True

    assert gradient is not None
    assert bool((gradient[touched] != 0.0).all())
    assert not bool(gradient[~touched].any())
    assert scored.head.base.grad is not None
    assert scored.head.spans.grad is not None
    assert scored.parameters_of_linking.grad is not None
    assert float(scored.head.base.grad.abs()) > 0.0
    assert float(scored.head.spans.grad.abs().sum()) > 0.0
    assert float(scored.parameters_of_linking.grad.abs().sum()) > 0.0


def test_the_head_parameters_are_the_module_parameters() -> None:
    scored = objective()
    named = dict(scored.named_parameters())

    assert set(named) == {"head.base", "head.spans", "parameters_of_linking"}
    assert named["head.spans"].shape == (LEVELS - 1,)
    assert named["head.base"].shape == ()


def test_the_head_is_the_object_the_objective_was_given() -> None:
    head = OrdinalHead(LEVELS, CUTPOINTS)
    scored = JudgmentLikelihoodObjective(
        readout=ROOT_MASS, linking=logit_linking(), head=head
    )

    assert scored.head is head
    assert scored.levels == LEVELS
    assert "OrdinalHead" in repr(scored)


def test_a_step_of_gradient_descent_on_the_head_lowers_the_value() -> None:
    outputs = bundle()
    scored = objective()
    optimizer = torch.optim.SGD(scored.parameters(), lr=0.1)
    before = scored.evaluate(outputs).value

    before.backward()
    optimizer.step()
    after = scored.evaluate(outputs).value

    assert after.item() < before.item()


def test_the_objective_reads_the_batch_and_the_mass_id_alone() -> None:
    scored = objective()
    outputs = bundle()
    proxy = RecordingOutputs(outputs, objective=scored.name, requires=scored.requires)

    term = scored.evaluate(proxy)

    assert proxy.touched == {"batch", "readouts"}
    assert term.count == len(RESPONSES)


def test_the_value_agrees_with_the_head_used_by_the_response_model() -> None:
    outputs = bundle()
    scored = objective()
    scores = scores_of(outputs)
    latent = torch.tensor(
        [latent_of(scores[IDS.index(item_id)]) for _, item_id, _ in RESPONSES],
        dtype=torch.float64,
    )
    responses = torch.tensor([level for _, _, level in RESPONSES])
    log_probabilities = log_category_probabilities(latent, scored.head.cutpoints())
    reference = -log_probabilities.gather(1, responses.unsqueeze(1)).mean()

    term = scored.evaluate(outputs)

    assert term.value.item() == pytest.approx(reference.item(), rel=1e-12)


def test_a_root_the_readout_never_computed_is_a_named_error() -> None:
    with pytest.raises(UncomputedCellError) as caught:
        objective().evaluate(assemble(random_log_masses(0), lengths=(3, 3, 3)))

    assert caught.value.item == 1
    assert caught.value.span == (0, 3)


def test_an_empty_item_is_refused() -> None:
    with pytest.raises(ValueError, match="has no word"):
        objective().evaluate(assemble(random_log_masses(0), lengths=(3, 0, 3)))


@pytest.mark.parametrize("outcome", [0, 1])
def test_a_log_mass_above_zero_is_refused_rather_than_a_nan(outcome: int) -> None:
    pair = torch.full((1, SPANS.num_spans, 2), -1.0).double()
    pair[0, SPANS.span_id(0, 2), outcome] = 0.1
    live = live_spans((2,))
    outputs: Outputs = ObjectiveOutputs(
        spans=SPANS,
        chart=CellChart(
            LogDistribution(torch.where(live, -1.0, FILL), SPAN_AXES, live)
        ),
        batch=assemble(
            random_log_masses(0),
            responses=(("participant", "second", 2),),
        ).batch,
        readouts={
            ROOT_MASS: SpanReadout(SPANS, LogDistribution(pair, MASS_AXES, live))
        },
    )

    with pytest.raises(ImproperMassError) as caught:
        objective().evaluate(outputs)

    assert caught.value.readout == ROOT_MASS
    assert caught.value.item == 0
    assert caught.value.span == (0, 2)


def test_a_declared_id_the_step_did_not_publish_is_refused() -> None:
    with pytest.raises(ValueError, match="reads the root mass 'sentences'"):
        objective().evaluate(assemble(random_log_masses(0), key=ReadoutId("other")))


def test_a_readout_that_is_not_a_mass_pair_is_refused() -> None:
    outputs: Outputs = ObjectiveOutputs(
        spans=SPANS,
        chart=CellChart(
            LogDistribution(torch.where(LIVE, -1.0, FILL), SPAN_AXES, LIVE)
        ),
        batch=bundle().batch,
        readouts={
            ROOT_MASS: SpanReadout(
                SPANS,
                LogDistribution(torch.full((ITEMS, SPANS.num_spans), -1.0), SPAN_AXES),
            )
        },
    )

    with pytest.raises(ValueError, match="reads a mass pair over"):
        objective().evaluate(outputs)


def test_a_parameter_vector_of_another_length_than_the_linking_fits_is_refused() -> (
    None
):
    with pytest.raises(ValueError, match="parameters"):
        objective(parameters=(1.0, 2.0))

    with pytest.raises(ValueError, match="parameters"):
        objective(linking=probability_linking(), parameters=(1.0,))


class TwoColumnLinking:
    """A linking over its argument and two further columns."""

    def __init__(self, linking: Linking) -> None:
        self._held = linking

    @property
    def domain(self) -> LinkingDomain:
        domain = self._held.domain

        return LinkingDomain(
            domain.argument,
            {**domain.columns, "length": Interval(1.0, 100.0)},
            domain.parameters,
        )

    @property
    def special_cases(self) -> tuple[SpecialCase, ...]:
        return self._held.special_cases

    def contribution(self, columns: ReadoutColumns, parameters: Tensor, /) -> Tensor:
        return self._held.contribution(columns, parameters)


def test_a_linking_reading_more_than_the_mass_pair_is_refused() -> None:
    with pytest.raises(ValueError, match="mass pair"):
        objective(linking=TwoColumnLinking(logit_linking()))


class NoCaseLinking:
    """A linking that fits a parameter and declares no closed form."""

    def __init__(self, linking: Linking) -> None:
        self._held = linking

    @property
    def domain(self) -> LinkingDomain:
        return self._held.domain

    @property
    def special_cases(self) -> tuple[SpecialCase, ...]:
        return ()

    def contribution(self, columns: ReadoutColumns, parameters: Tensor, /) -> Tensor:
        return self._held.contribution(columns, parameters)


def test_a_fitted_parameter_with_no_closed_form_needs_an_explicit_start() -> None:
    with pytest.raises(ValueError, match="parameters"):
        objective(linking=NoCaseLinking(logit_linking()))

    scored = objective(linking=NoCaseLinking(logit_linking()), parameters=(0.5,))

    assert scored.parameters_of_linking.tolist() == [0.5]


def test_an_empty_id_or_name_is_refused() -> None:
    with pytest.raises(ValueError, match="readout id"):
        objective(readout=ReadoutId(""))

    with pytest.raises(ValueError, match="name"):
        objective(name="")


def test_the_term_is_weighted_by_the_suite_beside_another() -> None:
    outputs = bundle()
    plain = objective()
    steep = objective(parameters=(2.0,), name="steep")
    suite = ObjectiveSuite([(0.5, plain), (2.0, steep)])

    result = suite.evaluate(bundle())

    assert result.total.item() == pytest.approx(
        0.5 * plain.evaluate(outputs).value.item()
        + 2.0 * steep.evaluate(outputs).value.item(),
        rel=1e-12,
    )
    assert "judgment/log_density" in result.diagnostics
    assert "steep/level_residual_0" in result.diagnostics
    assert result.terms["judgment"].count == len(RESPONSES)


def test_the_fixtures_transform_the_outputs_they_are_handed() -> None:
    held = random_log_masses(7)
    outputs = assemble(held)

    swapped = permuted(outputs)
    refilled_outputs = refilled(outputs)
    order = [2, 0, 1]

    assert swapped.batch.item_ids == tuple(IDS[position] for position in order)
    assert swapped.batch.lengths == tuple(LENGTHS[position] for position in order)
    assert sorted(
        (swapped.batch.item_ids[item], level)
        for item, level in zip(
            swapped.batch.response_items.tolist(),
            swapped.batch.design.responses.tolist(),
            strict=True,
        )
    ) == sorted((item_id, level) for _, item_id, level in RESPONSES)
    assert torch.equal(log_masses_of(swapped)[LIVE[order]], held[order][LIVE[order]])
    assert torch.equal(log_masses_of(refilled_outputs)[LIVE], held[LIVE])
    assert bool((log_masses_of(refilled_outputs)[~LIVE] == FILL / 3.0).all())
    assert refilled_outputs.batch.item_ids == outputs.batch.item_ids
    assert torch.equal(
        refilled_outputs.batch.design.responses, outputs.batch.design.responses
    )
    assert torch.equal(
        refilled_outputs.batch.response_items, outputs.batch.response_items
    )

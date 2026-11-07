from typing import TYPE_CHECKING, Literal

import pytest
import torch

from lofi.judgment import (
    ConditionalLikelihood,
    Cumulative,
    CumulativeLinkHead,
    CumulativeParameters,
    DesignCodec,
    FixedDiscrimination,
    JudgmentRow,
    Predictor,
    PriorBlock,
    RandomEffectStore,
    ResponseSpec,
    RowColumns,
    RowEffects,
    ThresholdPriors,
    VarianceComponent,
    log_category_probabilities,
    ordered_cutpoints,
)

if TYPE_CHECKING:
    from torch import Tensor

    from lofi.judgment import JudgmentDesign

READOUT = "p_gram"
LEVELS = 7
CUTPOINTS = LEVELS - 1
PARTICIPANTS = ("ann", "bea", "cal")
ITEMS = ("item-0", "item-1")
BASE = -2.0
SPANS = (0.4, 0.1, -0.5, 0.9, 0.3, 0.2)


class Readout(Predictor, extra="forbid"):
    kind: Literal["effects_readout"] = "effects_readout"
    name: str


class ReadoutPredictor:
    def __init__(self, name: str) -> None:
        self._terms = (Readout(name=name),)
        self._name = name

    @property
    def terms(self) -> tuple[Predictor, ...]:
        return self._terms

    def latent_scores(self, design: JudgmentDesign, /) -> Tensor:
        return design.readouts[self._name]


def participant_store(dtype: torch.dtype = torch.float64) -> RandomEffectStore:
    return RandomEffectStore(
        "participant", PARTICIPANTS, cutpoints=CUTPOINTS, dtype=dtype
    )


def item_store() -> RandomEffectStore:
    return RandomEffectStore("item", ITEMS, cutpoints=0, dtype=torch.float64)


def seeded(store: RandomEffectStore, seed: int) -> RandomEffectStore:
    generator = torch.Generator().manual_seed(seed)

    with torch.no_grad():
        for parameter in store.parameters():
            parameter.copy_(
                torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype)
            )

    return store


def head() -> CumulativeLinkHead:
    declared = ResponseSpec(
        family=Cumulative(),
        levels=LEVELS,
        objective=ConditionalLikelihood(),
        predictor=(Readout(name=READOUT),),
        identification=FixedDiscrimination(value=1.0),
        priors=PriorBlock(thresholds=ThresholdPriors()),
    )

    return CumulativeLinkHead(declared, ReadoutPredictor(READOUT))


def parameters(*, requires_grad: bool = False) -> CumulativeParameters:
    return CumulativeParameters(
        torch.tensor(BASE, dtype=torch.float64, requires_grad=requires_grad),
        torch.tensor(SPANS, dtype=torch.float64, requires_grad=requires_grad),
    )


def design_of(
    participants: tuple[str, ...],
    responses: tuple[int, ...],
    scores: tuple[float, ...],
) -> JudgmentDesign:
    codec = DesignCodec(LEVELS, readouts=(READOUT,))
    rows = tuple(
        JudgmentRow(
            participant,
            ITEMS[position % len(ITEMS)],
            response,
            RowColumns(factors={}, readouts={READOUT: score}),
        )
        for position, (participant, response, score) in enumerate(
            zip(participants, responses, scores, strict=True)
        )
    )

    return codec.encode(rows)


def test_a_store_reports_its_group_levels_and_parameters() -> None:
    store = participant_store()

    assert store.group == "participant"
    assert store.levels == PARTICIPANTS
    assert store.cutpoints == CUTPOINTS
    assert store.fallback == len(PARTICIPANTS)
    assert store.parameter_names == (
        "latent",
        *(f"span_{position}" for position in range(1, LEVELS)),
    )


def test_a_store_over_the_latent_alone_names_one_parameter() -> None:
    store = item_store()

    assert store.parameter_names == ("latent",)
    assert store.cutpoints == 0


def test_a_fresh_store_puts_every_group_at_the_population_mean() -> None:
    store = participant_store()
    effects = store.effects(torch.arange(store.fallback + 1))

    assert torch.equal(store.scale, torch.ones(LEVELS, dtype=torch.float64))
    assert torch.equal(
        effects.latent, torch.zeros(store.fallback + 1, dtype=torch.float64)
    )
    assert effects.spans is not None
    assert torch.equal(
        effects.spans,
        torch.zeros(store.fallback + 1, CUTPOINTS, dtype=torch.float64),
    )


def test_an_intercept_is_the_standardized_value_times_the_parameter_scale() -> None:
    store = seeded(participant_store(), 11)
    standardized, log_scale = (parameter.detach() for parameter in store.parameters())
    index = torch.tensor([2, 0, 2, 1])
    effects = store.effects(index)
    expected = standardized[index] * torch.exp(log_scale)

    assert effects.spans is not None
    assert torch.allclose(effects.latent, expected[:, 0], rtol=0.0, atol=1e-12)
    assert torch.allclose(effects.spans, expected[:, 1:], rtol=0.0, atol=1e-12)
    assert torch.allclose(store.scale, torch.exp(log_scale), rtol=0.0, atol=1e-12)


def test_the_intercept_table_ends_with_the_population_mean() -> None:
    store = seeded(participant_store(), 5)
    table = store.intercepts()

    assert table.shape == (len(PARTICIPANTS) + 1, LEVELS)
    assert torch.equal(table[-1], torch.zeros(LEVELS, dtype=torch.float64))
    assert bool(torch.all(table[:-1] != 0.0))


def test_alignment_maps_known_levels_to_their_index_and_unknown_to_the_fallback() -> (
    None
):
    store = participant_store()
    alignment = store.align(("cal", "dee", "ann", "bea", "eve"))

    assert alignment.dtype is torch.int64
    assert alignment.tolist() == [2, 3, 0, 1, 3]


def test_an_unseen_group_is_scored_at_the_population_mean() -> None:
    store = seeded(participant_store(), 3)
    effects = store.effects(store.align(("dee",)))

    assert effects.spans is not None
    assert torch.equal(effects.latent, torch.zeros(1, dtype=torch.float64))
    assert torch.equal(effects.spans, torch.zeros(1, CUTPOINTS, dtype=torch.float64))


def test_an_unseen_group_passes_no_gradient_to_the_store() -> None:
    store = seeded(participant_store(), 3)
    effects = store.effects(store.align(("dee", "eve")))

    assert effects.spans is not None

    (effects.latent.sum() + effects.spans.sum()).backward()

    for parameter in store.parameters():
        assert parameter.grad is not None
        assert bool(torch.all(parameter.grad == 0.0))


def test_a_seen_group_passes_a_gradient_to_its_row_and_to_the_scale() -> None:
    store = seeded(participant_store(), 3)
    standardized, log_scale = store.parameters()
    effects = store.effects(store.align(("bea",)))

    assert effects.spans is not None

    (effects.latent.sum() + effects.spans.sum()).backward()

    assert standardized.grad is not None
    assert log_scale.grad is not None
    assert bool(torch.all(standardized.grad[1] != 0.0))
    assert bool(torch.all(standardized.grad[0] == 0.0))
    assert bool(torch.all(standardized.grad[2] == 0.0))
    assert bool(torch.all(log_scale.grad != 0.0))


def test_a_store_over_the_latent_alone_offsets_no_spans() -> None:
    store = seeded(item_store(), 9)
    effects = store.effects(torch.tensor([1, 0, 1]))

    assert effects.spans is None
    assert effects.rows == 3
    assert bool(torch.all(effects.latent != 0.0))


def test_a_span_component_measures_the_span_offset_and_not_the_cutpoint() -> None:
    store = seeded(participant_store(), 21)
    standardized, log_scale = (parameter.detach() for parameter in store.parameters())
    offsets = standardized * torch.exp(log_scale)
    table = store.intercepts().detach()[:-1]
    given = parameters()
    cutpoints = torch.stack(
        [ordered_cutpoints(given.base, given.spans + row[1:]) for row in table]
    )
    components = store.variance_components()

    for position, component in enumerate(components[1:], start=1):
        assert component.parameter == f"span_{position}"
        assert component.realized == pytest.approx(
            float(torch.var(offsets[:, position], correction=1))
        )
        assert component.realized != pytest.approx(
            float(torch.var(cutpoints[:, position - 1], correction=1))
        )


def test_the_variance_components_square_the_scale_and_sample_the_intercepts() -> None:
    store = seeded(participant_store(), 21)
    standardized, log_scale = (parameter.detach() for parameter in store.parameters())
    scale = torch.exp(log_scale)
    realized = torch.var(standardized * scale, dim=0, correction=1)
    components = store.variance_components()

    assert len(components) == LEVELS

    for position, component in enumerate(components):
        assert isinstance(component, VarianceComponent)
        assert component.group == "participant"
        assert component.parameter == store.parameter_names[position]
        assert component.groups == len(PARTICIPANTS)
        assert component.variance == pytest.approx(float(scale[position] ** 2))
        assert component.realized == pytest.approx(float(realized[position]))


def test_a_single_group_has_no_realized_variance() -> None:
    store = seeded(
        RandomEffectStore("item", ("only",), cutpoints=0, dtype=torch.float64), 2
    )
    (component,) = store.variance_components()

    assert component.groups == 1
    assert component.realized is None
    assert component.variance > 0.0
    assert repr(component) == (
        f"VarianceComponent('item', 'latent', {component.variance!r}, None, 1)"
    )


def test_the_store_checkpoints_its_standardized_values_and_scales() -> None:
    store = participant_store()
    state = store.state_dict()

    assert {tuple(value.shape) for value in state.values()} == {
        (len(PARTICIPANTS), LEVELS),
        (LEVELS,),
    }

    restored = participant_store()
    restored.load_state_dict(seeded(store, 4).state_dict())

    assert torch.equal(restored.intercepts(), store.intercepts())


def test_the_store_keeps_the_dtype_it_was_given() -> None:
    store = participant_store(torch.float32)
    effects = store.effects(torch.tensor([0]))

    assert effects.latent.dtype is torch.float32
    assert store.scale.dtype is torch.float32


@pytest.mark.parametrize(
    ("levels", "cutpoints", "message"),
    [
        ((), CUTPOINTS, "no levels"),
        (("ann", "ann"), CUTPOINTS, "repeats a level"),
        (PARTICIPANTS, -1, "is negative"),
    ],
)
def test_a_store_refuses_an_impossible_declaration(
    levels: tuple[str, ...],
    cutpoints: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RandomEffectStore(
            "participant", levels, cutpoints=cutpoints, dtype=torch.float64
        )


def test_a_store_refuses_a_boolean_cutpoint_count() -> None:
    with pytest.raises(TypeError, match="is not an integer"):
        RandomEffectStore(
            "participant", PARTICIPANTS, cutpoints=True, dtype=torch.float64
        )


def test_a_store_refuses_an_integer_dtype() -> None:
    with pytest.raises(TypeError, match="must be floating point"):
        RandomEffectStore(
            "participant", PARTICIPANTS, cutpoints=CUTPOINTS, dtype=torch.int64
        )


def test_a_store_refuses_a_blank_group_or_level() -> None:
    with pytest.raises(ValueError, match="grouping factor"):
        RandomEffectStore("", PARTICIPANTS, cutpoints=0, dtype=torch.float64)

    with pytest.raises(ValueError, match="level of 'participant'"):
        RandomEffectStore("participant", ("ann", ""), cutpoints=0, dtype=torch.float64)


def test_the_head_scores_each_row_at_its_group_intercepts() -> None:
    store = seeded(participant_store(), 17)
    subject, given = head(), parameters()
    design = design_of(
        ("bea", "ann", "bea", "cal"), (0, 3, 6, 2), (0.5, -1.0, 2.0, 0.0)
    )
    effects = store.effects(store.align(design.participant_levels)[design.participants])
    scored = subject.log_likelihood(design, given, effects=effects)
    table = store.intercepts()
    expected = []

    for position, participant in enumerate(("bea", "ann", "bea", "cal")):
        row = table[PARTICIPANTS.index(participant)]
        latent = design.readouts[READOUT][position : position + 1] + row[0]
        cutpoints = ordered_cutpoints(given.base, given.spans + row[1:])
        expected.append(
            log_category_probabilities(latent, cutpoints)[
                0, int(design.responses[position])
            ]
        )

    assert torch.allclose(scored, torch.stack(expected), rtol=0.0, atol=1e-12)


def test_a_participant_outside_the_fit_is_scored_by_the_population_head() -> None:
    store = seeded(participant_store(), 17)
    subject, given = head(), parameters()
    design = design_of(("dee", "eve"), (1, 5), (0.25, -0.75))
    effects = store.effects(store.align(design.participant_levels)[design.participants])

    assert torch.equal(
        subject.log_likelihood(design, given, effects=effects),
        subject.log_likelihood(design, given),
    )


def test_crossed_stores_add_their_intercepts_row_by_row() -> None:
    by_participant = seeded(participant_store(), 8)
    by_item = seeded(item_store(), 13)
    design = design_of(("ann", "cal", "ann"), (2, 4, 1), (0.0, 1.0, -1.0))
    participant_effects = by_participant.effects(
        by_participant.align(design.participant_levels)[design.participants]
    )
    item_effects = by_item.effects(by_item.align(design.item_levels)[design.items])
    crossed = participant_effects + item_effects

    assert crossed.spans is not None
    assert participant_effects.spans is not None
    assert torch.equal(crossed.latent, participant_effects.latent + item_effects.latent)
    assert torch.equal(crossed.spans, participant_effects.spans)
    assert isinstance(crossed, RowEffects)


def test_the_head_trains_the_store_through_a_batch_of_gathered_rows() -> None:
    generator = torch.Generator().manual_seed(2)
    levels = tuple(f"annotator-{index}" for index in range(727))
    store = seeded(
        RandomEffectStore(
            "participant", levels, cutpoints=CUTPOINTS, dtype=torch.float64
        ),
        6,
    )
    subject, given = head(), parameters(requires_grad=True)
    drawn = torch.randint(0, len(levels), (512,), generator=generator)
    design = design_of(
        tuple(levels[int(index)] for index in drawn),
        tuple(
            int(value)
            for value in torch.randint(0, LEVELS, (512,), generator=generator)
        ),
        tuple(float(value) for value in torch.randn(512, generator=generator)),
    )
    index = store.align(design.participant_levels)[design.participants]
    loss = -subject.log_likelihood(design, given, effects=store.effects(index)).mean()
    loss.backward()
    standardized, log_scale = store.parameters()

    assert bool(torch.isfinite(loss))
    assert standardized.grad is not None
    assert log_scale.grad is not None
    assert int((standardized.grad != 0.0).any(dim=1).sum()) == len(set(drawn.tolist()))
    assert bool(torch.all(log_scale.grad != 0.0))
    assert given.spans.grad is not None

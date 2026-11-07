import math
from itertools import product
from pathlib import Path
from typing import Final, cast

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.conformance._sites import SiteObservation
from tests.conformance.component_plugin import plugin_context
from tests.conformance.subword_pooler import (
    SubwordPoolerConformance,
    SubwordPoolerSuite,
)

from lofi.data import Alignment
from lofi.data.synthetic.training import SyntheticTrainingBatch
from lofi.nn._values import Encoded, TokenEncoding
from lofi.nn.build import build_component
from lofi.nn.sites import (
    Access,
    ComponentId,
    ComponentKind,
    DTypeClass,
    Role,
    Site,
    SiteMap,
    SiteSpec,
)
from lofi.nn.tokens.pool import (
    AttentionSubwordPooler,
    FirstSubwordPooler,
    MeanSubwordPooler,
)
from lofi.probe import ensure_publishes
from lofi.run import (
    RandomStreams,
    SeedPolicy,
    packaged_options,
    plan_training,
    resolve_device_policy,
)
from lofi.spec import RunSpecCodec, config_search_path
from lofi.spec.layers import (
    BUILTIN_ACTIVATIONS,
    BUILTIN_RECURRENT_CELLS,
    Activation,
    FeedForwardLayer,
    FeedForwardLayerSpec,
    RecurrentCell,
)
from lofi.spec.poolers.attention import (
    AttentionSubwordPoolerSpec,
    SubwordContextSettings,
    SubwordContextSpec,
)
from lofi.train import Featurized, ParserFittingPlan, ParserStep, prepare_training

RUN: Final = (
    Path(__file__).resolve().parents[4] / "lofi/configs/run/synthetic-parser.yaml"
)
BUILT_BY_TAG: Final = {
    "attention": AttentionSubwordPooler,
    "first": FirstSubwordPooler,
    "mean": MeanSubwordPooler,
}
ROW_AXES: Final = ("item", "row", "width")
POOLED_AXES: Final = ("item", "word", "width")
POOLER: Final = ComponentId("pooler")
WIDTH: Final = 2

# three words occupy three, one, and three rows.
GOLDEN_ROWS: Final = torch.tensor(
    [[[float(step + 1), -float(step + 1)] for step in range(7)]]
)
GOLDEN_LIVE: Final = torch.ones(1, 7, dtype=torch.bool)
GOLDEN_ALIGNMENT: Final = (Alignment(((0, 3), (3, 4), (4, 7))),)
GOLDEN: Final = TokenEncoding(
    Encoded(GOLDEN_ROWS, ROW_AXES, GOLDEN_LIVE), GOLDEN_ALIGNMENT
)

# one ragged item includes an unaligned word and padding.
RAGGED_ROWS: Final = torch.tensor(
    [
        [[1.0, -1.0], [2.0, -2.0], [3.0, -3.0], [0.0, 0.0]],
        [[7.0, -7.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    ]
)
RAGGED_LIVE: Final = torch.tensor(
    [[True, True, True, False], [True, False, False, False]]
)
RAGGED: Final = TokenEncoding(
    Encoded(RAGGED_ROWS, ROW_AXES, RAGGED_LIVE),
    (Alignment(((0, 2), (2, 3), None)), Alignment(((0, 1),))),
    words=(("multi", "piece", "dropped"), ("kept",)),
)


GRU: Final = SubwordContextSettings("gru", layers=1, width=None, bidirectional=True)


def attention(
    *,
    context: SubwordContextSettings | None,
    scorer: tuple[FeedForwardLayer, ...] = (),
) -> AttentionSubwordPooler:
    return AttentionSubwordPooler(
        WIDTH,
        component=POOLER,
        context=context,
        scorer=scorer,
    )


def every_pooler() -> list[
    FirstSubwordPooler | MeanSubwordPooler | AttentionSubwordPooler
]:
    """Return one pooler of each kind, at the test width."""
    return [
        FirstSubwordPooler(WIDTH, component=POOLER),
        MeanSubwordPooler(WIDTH, component=POOLER),
        attention(context=None),
        attention(context=GRU),
        attention(context=None, scorer=(FeedForwardLayer(2, "tanh", 0.0),)),
    ]


def straight(pooler: AttentionSubwordPooler) -> AttentionSubwordPooler:
    with torch.no_grad():
        for name, parameter in pooler.named_parameters():
            parameter.zero_()

            if name == "_query.0.weight":
                parameter[0, 0] = 1.0

    return pooler


def attention_pooler(
    *, context: SubwordContextSettings | None = None
) -> AttentionSubwordPooler:
    return straight(attention(context=context))


def softmax(scores: list[float]) -> list[float]:
    weights = [math.exp(score) for score in scores]
    total = sum(weights)

    return [weight / total for weight in weights]


def test_the_first_pooler_takes_the_row_of_each_words_first_subword() -> None:
    pooled = FirstSubwordPooler(WIDTH, component=POOLER).pool(GOLDEN)

    assert pooled.axes == POOLED_AXES
    assert pooled.tensor.tolist() == [[[1.0, -1.0], [4.0, -4.0], [5.0, -5.0]]]


def test_the_mean_pooler_averages_the_rows_each_word_reached() -> None:
    pooled = MeanSubwordPooler(WIDTH, component=POOLER).pool(GOLDEN)

    assert pooled.tensor.tolist() == [[[2.0, -2.0], [4.0, -4.0], [6.0, -6.0]]]


def test_the_attention_pooler_weighs_a_word_by_its_own_rows() -> None:
    pooled = attention_pooler().pool(GOLDEN)
    first = softmax([1.0, 2.0, 3.0])
    third = softmax([5.0, 6.0, 7.0])
    expected = [
        sum(weight * (step + 1) for step, weight in enumerate(first)),
        4.0,
        sum(weight * (step + 5) for step, weight in enumerate(third)),
    ]

    assert pooled.tensor[0, :, 0].detach().tolist() == pytest.approx(expected)
    assert pooled.tensor[0, :, 1].detach().tolist() == pytest.approx(
        [-value for value in expected]
    )


def test_a_still_recurrence_leaves_the_attention_pooler_averaging() -> None:
    pooled = attention_pooler(context=GRU).pool(GOLDEN)
    averaged = MeanSubwordPooler(WIDTH, component=POOLER).pool(GOLDEN)

    assert torch.allclose(pooled.tensor, averaged.tensor, atol=1e-6)


def test_the_recurrence_moves_the_weights_off_the_average() -> None:
    torch.manual_seed(0)
    reading = attention(context=GRU)
    averaged = MeanSubwordPooler(WIDTH, component=POOLER).pool(GOLDEN).tensor

    assert not torch.allclose(reading.pool(GOLDEN).tensor, averaged)


GATES: Final = {
    "gru": 3,
    "lstm": 4,
    "elman-tanh": 1,
    "elman-relu": 1,
}


@pytest.mark.parametrize(
    ("cell", "layers", "bidirectional"),
    list(product(BUILTIN_RECURRENT_CELLS, (1, 2), (False, True))),
)
def test_the_declared_recurrence_is_the_one_the_scorer_reads_through(
    cell: RecurrentCell, layers: int, *, bidirectional: bool
) -> None:
    context = SubwordContextSettings(
        cell, layers=layers, width=5, bidirectional=bidirectional
    )
    reading = attention(context=context)
    state = reading.state_dict()
    directions = 2 if bidirectional else 1
    rows = GOLDEN_ROWS.clone().requires_grad_()
    encoding = TokenEncoding(Encoded(rows, ROW_AXES, GOLDEN_LIVE), GOLDEN_ALIGNMENT)

    reading.pool(encoding).tensor.sum().backward()

    assert state["_context.weight_ih_l0"].shape == (GATES[cell] * 5, WIDTH)
    assert state["_context.weight_hh_l0"].shape == (GATES[cell] * 5, 5)
    assert f"_context.weight_ih_l{layers - 1}" in state
    assert f"_context.weight_ih_l{layers}" not in state
    assert ("_context.weight_ih_l0_reverse" in state) is bidirectional
    assert state["_query.0.weight"].shape == (1, 5 * directions)
    assert rows.grad is not None
    assert bool(rows.grad.isfinite().all())


def test_the_two_ungated_cells_differ_only_in_their_nonlinearity() -> None:
    torch.manual_seed(3)
    squashing = attention(
        context=SubwordContextSettings(
            "elman-tanh", layers=1, width=None, bidirectional=False
        )
    )
    rectifying = attention(
        context=SubwordContextSettings(
            "elman-relu", layers=1, width=None, bidirectional=False
        )
    )
    rectifying.load_state_dict(squashing.state_dict())

    assert not torch.allclose(
        squashing.pool(GOLDEN).tensor, rectifying.pool(GOLDEN).tensor
    )


NONLINEARITIES: Final = {
    "tanh": torch.tanh,
    "relu": torch.relu,
    "gelu": torch.nn.functional.gelu,
    "silu": torch.nn.functional.silu,
}


@pytest.mark.parametrize("kind", BUILTIN_ACTIVATIONS)
def test_a_hidden_scorer_layer_applies_its_declared_nonlinearity(
    kind: Activation,
) -> None:
    pooler = attention(context=None, scorer=(FeedForwardLayer(WIDTH, kind, 0.0),))

    with torch.no_grad():
        for parameter in pooler.parameters():
            parameter.zero_()

        pooler.state_dict()["_query.0.weight"].copy_(torch.eye(WIDTH))
        pooler.state_dict()["_query.3.weight"][0, 1] = 1.0

    rows = GOLDEN_ROWS[0, :3]
    weights = NONLINEARITIES[kind](rows[:, 1]).softmax(dim=0)
    expected = (weights.unsqueeze(1) * rows).sum(dim=0)

    assert torch.allclose(pooler.pool(GOLDEN).tensor[0, 0], expected, atol=1e-6)


def test_the_scorer_stacks_one_hidden_layer_per_declared_nonlinearity() -> None:
    hidden = (
        FeedForwardLayer(5, "relu", 0.0),
        FeedForwardLayer(4, "gelu", 0.1),
        FeedForwardLayer(3, "tanh", 0.2),
    )
    pooler = attention(context=GRU, scorer=hidden)
    weights = [
        tuple(value.shape)
        for name, value in pooler.state_dict().items()
        if name.startswith("_query.") and name.endswith(".weight")
    ]

    assert weights == [(5, 2 * WIDTH), (4, 5), (3, 4), (1, 3)]


def test_each_scorer_layer_uses_its_declared_width() -> None:
    hidden = (FeedForwardLayer(5, "relu", 0.0), FeedForwardLayer(3, "tanh", 0.0))
    weights = [
        tuple(value.shape)
        for name, value in attention(context=GRU, scorer=hidden).state_dict().items()
        if name.startswith("_query.") and name.endswith(".weight")
    ]

    assert weights == [(5, 2 * WIDTH), (3, 5), (1, 3)]


def test_the_plugin_builds_the_declared_recurrence_and_scorer() -> None:
    class Rows:
        output_width = WIDTH

    declared = AttentionSubwordPoolerSpec(
        context=SubwordContextSpec(cell="lstm", layers=2, width=5, bidirectional=False),
        scorer=(FeedForwardLayerSpec(width=4, activation="silu", dropout=0.1),),
    )
    built = cast(
        "AttentionSubwordPooler",
        build_component(
            declared, plugin_context(component=POOLER, producers={"encoder": Rows()})
        ),
    )
    state = built.state_dict()

    assert built.output_width == WIDTH
    assert state["_context.weight_ih_l0"].shape == (4 * 5, WIDTH)
    assert state["_context.weight_ih_l1"].shape == (4 * 5, 5)
    assert "_context.weight_ih_l0_reverse" not in state
    assert state["_query.0.weight"].shape == (4, 5)
    assert isinstance(built.get_submodule("_query.1"), torch.nn.SiLU)
    assert state["_query.3.weight"].shape == (1, 4)


def test_the_plugin_scores_rows_as_they_are_without_a_declared_recurrence() -> None:
    class Rows:
        output_width = WIDTH

    built = build_component(
        AttentionSubwordPoolerSpec(),
        plugin_context(component=POOLER, producers={"encoder": Rows()}),
    )
    state = cast("AttentionSubwordPooler", built).state_dict()

    assert set(state) == {"_query.0.weight"}
    assert state["_query.0.weight"].shape == (1, WIDTH)


def test_the_attention_weights_are_a_convex_combination_of_a_words_rows() -> None:
    pooled = attention_pooler().pool(GOLDEN)
    rows = GOLDEN_ROWS[0, :, 0]

    for word, span in enumerate(((0, 3), (3, 4), (4, 7))):
        start, end = span
        value = float(pooled.tensor[0, word, 0].detach())

        assert float(rows[start:end].min()) <= value <= float(rows[start:end].max())


def test_a_word_no_row_encoded_is_masked_off_by_every_pooler() -> None:
    for pooler in every_pooler():
        pooled = pooler.pool(RAGGED)

        assert pooled.mask is not None
        assert pooled.mask.tolist() == [[True, True, False], [True, False, False]]


def test_swapping_the_pooler_moves_nothing_else_a_graph_was_built_from() -> None:
    built = every_pooler()
    pooled = [pooler.pool(RAGGED) for pooler in built]
    reference = pooled[0]

    for pooler, one in zip(built, pooled, strict=True):
        assert pooler.output_width == reference.size("width")
        assert pooler.site_map() == built[0].site_map()
        assert one.axes == reference.axes
        assert one.tensor.shape == reference.tensor.shape
        assert one.mask is not None
        assert reference.mask is not None
        assert torch.equal(one.mask, reference.mask)


def test_the_same_pooler_aggregates_characters_into_words() -> None:
    rows = torch.tensor([[[1.0, -1.0], [2.0, -2.0], [3.0, -3.0], [4.0, -4.0]]])
    characters = TokenEncoding(
        Encoded(rows, ROW_AXES, torch.ones(1, 4, dtype=torch.bool)),
        (Alignment(((0, 2), (2, 4))),),
    )
    pooler = MeanSubwordPooler(WIDTH, component=POOLER)

    assert pooler.pool(characters).tensor.tolist() == [[[1.5, -1.5], [3.5, -3.5]]]
    assert pooler.pool(GOLDEN).size("word") == 3


def test_a_gradient_reaches_the_rows_a_word_was_pooled_from() -> None:
    rows = GOLDEN_ROWS.clone().requires_grad_()
    encoding = TokenEncoding(Encoded(rows, ROW_AXES, GOLDEN_LIVE), GOLDEN_ALIGNMENT)

    attention_pooler().pool(encoding).tensor.sum().backward()

    assert rows.grad is not None
    assert bool(rows.grad.isfinite().all())
    assert bool(rows.grad.abs().sum() > 0.0)


@st.composite
def alignments(draw: st.DrawFn) -> tuple[tuple[tuple[int, int] | None, ...], ...]:
    items = draw(st.integers(min_value=1, max_value=3))
    built: list[tuple[tuple[int, int] | None, ...]] = []

    for _ in range(items):
        cursor = 0
        spans: list[tuple[int, int] | None] = []
        words = draw(st.integers(min_value=1, max_value=4))
        kept = draw(st.integers(min_value=0, max_value=words))

        for _ in range(kept):
            length = draw(st.integers(min_value=1, max_value=3))
            spans.append((cursor, cursor + length))
            cursor += length

        spans.extend([None] * (words - kept))
        built.append(tuple(spans))

    return tuple(built)


def batch(spans: tuple[tuple[tuple[int, int] | None, ...], ...]) -> TokenEncoding:
    lengths = [
        max((end for word in item if word is not None for _, end in (word,)), default=0)
        for item in spans
    ]
    rows = max(max(lengths, default=0), 1)
    live = torch.zeros(len(spans), rows, dtype=torch.bool)

    for item, length in enumerate(lengths):
        live[item, :length] = True

    vectors = torch.arange(
        float(len(spans) * rows * WIDTH), dtype=torch.float32
    ).reshape(len(spans), rows, WIDTH)

    return TokenEncoding(
        Encoded(vectors, ROW_AXES, live),
        tuple(Alignment(item) for item in spans),
        words=tuple(
            tuple(f"word-{word}" for word, _ in enumerate(item)) for item in spans
        ),
    )


@settings(max_examples=25)
@given(spans=alignments())
def test_one_row_comes_back_per_word_of_every_generated_alignment(
    spans: tuple[tuple[tuple[int, int] | None, ...], ...],
) -> None:
    encoding = batch(spans)
    widest = max(len(item) for item in spans)

    for pooler in every_pooler():
        pooled = pooler.pool(encoding)

        assert pooled.size("word") == widest
        assert pooled.size("item") == len(spans)
        assert pooled.mask is not None

        for item, words in enumerate(spans):
            encoded = [word is not None for word in words]
            padded = [False] * (widest - len(words))

            assert pooled.mask[item].tolist() == encoded + padded


def learned_conformance() -> SubwordPoolerConformance[AttentionSubwordPooler]:
    return SubwordPoolerConformance(
        build=lambda: attention_pooler(context=GRU),
        encoding=RAGGED,
        sites=SiteObservation(
            build=lambda: attention_pooler(context=GRU),
            observe=lambda subject: {
                Site(POOLER, Role.WORD_VECTOR): subject.pool(RAGGED)
            },
            read=lambda subject: (subject.pool(RAGGED).tensor,),
        ),
    )


class TestAnAttentionPoolerOverAWordsOwnRows(SubwordPoolerSuite):
    @pytest.fixture
    def conformance(self) -> SubwordPoolerConformance[AttentionSubwordPooler]:
        return learned_conformance()


class TestAMeanPoolerOverAWordsOwnRows(SubwordPoolerSuite):
    @pytest.fixture
    def conformance(self) -> SubwordPoolerConformance[MeanSubwordPooler]:

        def build() -> MeanSubwordPooler:
            return MeanSubwordPooler(WIDTH, component=POOLER)

        return SubwordPoolerConformance(
            build=build,
            encoding=RAGGED,
            sites=SiteObservation(
                build=build,
                observe=lambda subject: {
                    Site(POOLER, Role.WORD_VECTOR): subject.pool(RAGGED)
                },
                read=lambda subject: (subject.pool(RAGGED).tensor,),
            ),
        )


class TestAFirstSubwordPoolerOverAWordsOwnRows(SubwordPoolerSuite):
    @pytest.fixture
    def conformance(self) -> SubwordPoolerConformance[FirstSubwordPooler]:

        def build() -> FirstSubwordPooler:
            return FirstSubwordPooler(WIDTH, component=POOLER)

        return SubwordPoolerConformance(
            build=build,
            encoding=RAGGED,
            sites=SiteObservation(
                build=build,
                observe=lambda subject: {
                    Site(POOLER, Role.WORD_VECTOR): subject.pool(RAGGED)
                },
                read=lambda subject: (subject.pool(RAGGED).tensor,),
            ),
        )


def test_the_published_point_is_the_one_a_pass_fills() -> None:
    published = MeanSubwordPooler(WIDTH, component=POOLER).site_map()
    site = Site(POOLER, Role.WORD_VECTOR)

    assert set(published) == {site}
    assert published.component == POOLER
    assert published.kind is ComponentKind.SUBWORD_POOLER
    assert published[site] == SiteSpec(
        POOLED_AXES, DTypeClass.REAL, differentiable=True, access=Access.OBSERVE
    )
    assert ensure_publishes(published) is published


@pytest.mark.parametrize(
    ("axes", "dtype"),
    [
        (("item", "width", "word"), DTypeClass.REAL),
        (POOLED_AXES, DTypeClass.PROBABILITY),
    ],
)
def test_the_pooler_contract_refuses_a_mutated_real_site(
    axes: tuple[str, ...], dtype: DTypeClass
) -> None:
    published = MeanSubwordPooler(WIDTH, component=POOLER).site_map()
    site = Site(POOLER, Role.WORD_VECTOR)
    mutated = SiteMap(
        published.component,
        published.kind,
        {
            site: SiteSpec(
                axes,
                dtype,
                differentiable=True,
                access=Access.OBSERVE,
            )
        },
    )

    with pytest.raises(ValueError, match=r"runs over|holds"):
        ensure_publishes(mutated)


@pytest.mark.parametrize("option", packaged_options("model.subword_pooler"))
def test_every_packaged_pooler_document_selects_its_pooler_by_tag_and_trains(
    tmp_path: Path, option: str
) -> None:
    document = tmp_path / "run.yaml"
    document.write_text(
        f"defaults:\n  - synthetic-parser\n  - model.subword_pooler: {option}\n",
        encoding="utf-8",
        newline="\n",
    )
    run_spec = RunSpecCodec(search_path=(RUN.parent, *config_search_path(tmp_path)))(
        document, overrides=("trainer.accumulation_steps=1",)
    )
    declared = run_spec.model.subword_pooler
    blueprint = plan_training(run_spec)
    streams = RandomStreams(SeedPolicy(13, samplers=blueprint.sampler_streams))
    streams.apply()
    prepared = prepare_training(blueprint, streams, resolve_device_policy(run_spec))
    plan = cast("ParserFittingPlan", prepared.plan)
    pooler = prepared.model.modules["subword_pooler"]
    batch = plan.batches[0]

    assert isinstance(plan.step, ParserStep)
    assert isinstance(batch, SyntheticTrainingBatch)

    result = plan.step.run(Featurized(batch, 0)).result

    assert declared is not None
    assert declared.kind == option
    assert type(pooler) is BUILT_BY_TAG[option]
    assert result.metrics["optimizer/loss_finite"] == 1.0
    assert all(math.isfinite(value) for value in result.metrics.values())

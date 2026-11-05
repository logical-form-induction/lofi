import math
from itertools import pairwise
from typing import TYPE_CHECKING, Final, override

import didactic.api as dx
import pytest
import torch
from tests.conformance.objective import ObjectiveLawSuite
from torch import nn

from lofi._artifacts import PinnedCheckpoint
from lofi._types import ReadoutId
from lofi.data import Alignment
from lofi.deduce import SpanIndex
from lofi.judgment import (
    PROBABILITY,
    DesignCodec,
    GroupedJudgmentSet,
    JudgmentLikelihoodObjective,
    JudgmentRow,
    LinkingSpec,
    LinkTerm,
    OrdinalHead,
    PooledEncoderObjective,
    ProbabilityLinking,
    RowColumns,
    frozen_encoder_objective,
    log_category_probabilities,
    pooled_rows,
    rank_correlation,
)
from lofi.nn._values import Encoded, LogDistribution, TokenEncoding
from lofi.nn.loss._parser import SPAN_AXES
from lofi.nn.loss._ports import Objective
from lofi.nn.loss._values import CellChart, ObjectiveOutputs, RecordingOutputs
from lofi.nn.loss.laws import (
    ObjectiveFixture,
    the_value_is_monotone_where_the_fixture_says,
)
from lofi.nn.ports import TokenEncoder
from lofi.nn.sites import ComponentId, SiteMap
from lofi.nn.tokens.pretrained import (
    ENCODER_ONLY,
    T5_ENCODER,
    EncoderArchitecturePlugin,
    PretrainedTokenEncoder,
)
from lofi.nn.tokens.window import REPORT, ContextWindow

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from torch import Tensor

    from lofi.judgment.grouped import _GroupedJudgmentBatch
    from lofi.nn.loss._ports import LossTerm
WIDTH: Final = 4
PINNED: Final = PinnedCheckpoint("an-org/an-encoder", "e" * 40)
LEVELS: Final = 5
CUTPOINTS: Final = (-2.0, -0.5, 1.0, 2.5)
WEIGHT: Final = (0.2, 0.4, 0.6, 0.8)
BIAS: Final = 0.25
WORDS: Final = (("ab", "c"), ("dd",), ("e", "fg", "h"))
ITEM_IDS: Final = ("first", "second", "third")
RESPONSES: Final = (
    ("participant", "first", 4),
    ("participant", "first", 2),
    ("participant", "second", 0),
    ("participant", "third", 3),
    ("participant", "second", 1),
)
SPANS: Final = SpanIndex(3)
AXES: Final = ("item", "row", "width")


type Outputs = ObjectiveOutputs[_GroupedJudgmentBatch]


def scale_of(width: int = WIDTH) -> list[float]:
    return torch.linspace(0.5, 1.5, width).tolist()


def code_of(character: str, width: int = WIDTH) -> int:
    return ord(character) % width


class Rows(nn.Module):
    """One row per character, scaled by a parameter, with padding filled."""

    def __init__(self, *, width: int = WIDTH, trainable: bool = False) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale_of(width)))
        self.scale.requires_grad_(trainable)
        self.fill = 0.0
        self.calls = 0

    @property
    def output_width(self) -> int:
        return self.scale.numel()

    def encode_words(self, items: Sequence[Sequence[str]], /) -> TokenEncoding:
        self.calls += 1
        width = self.output_width
        codes = [[code_of(c, width) for word in item for c in word] for item in items]
        longest = max(map(len, codes), default=0)
        index = torch.zeros(len(items), longest, dtype=torch.long)
        mask = torch.zeros(len(items), longest, dtype=torch.bool)
        alignments: list[Alignment] = []

        for item, (words, item_codes) in enumerate(zip(items, codes, strict=True)):
            mask[item, : len(item_codes)] = True
            index[item, : len(item_codes)] = torch.tensor(item_codes, dtype=torch.long)
            spans: list[tuple[int, int] | None] = []
            start = 0

            for word in words:
                spans.append((start, start + len(word)))
                start += len(word)

            alignments.append(Alignment(tuple(spans)))

        basis = nn.functional.one_hot(index, num_classes=width).to(self.scale.dtype)
        rows = torch.where(mask.unsqueeze(-1), basis * self.scale, self.fill)

        return TokenEncoding(Encoded(rows, AXES, mask), tuple(alignments), words=items)

    def site_map(self) -> SiteMap:
        return SiteMap.custom(ComponentId("rows"), {})


class Configured:
    """A configuration naming its width under both architectures' field names."""

    def __init__(self, hidden_size: int) -> None:
        self.hidden_size = hidden_size
        self.d_model = hidden_size
        self.num_hidden_layers = 1
        self.num_layers = 1


class Output:
    def __init__(
        self, last_hidden_state: Tensor, hidden_states: tuple[Tensor, ...] | None
    ) -> None:
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states


class Stack(nn.Module):
    """A pretrained-shaped stack whose rows are one-hot codes times a parameter."""

    def __init__(self, width: int = WIDTH) -> None:
        super().__init__()
        self.config = Configured(width)
        self.scale = nn.Parameter(torch.tensor(scale_of(width)))

    @override
    def forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        return_dict: bool,
        output_hidden_states: bool,
    ) -> Output:
        del return_dict
        basis = nn.functional.one_hot(
            input_ids.remainder(self.config.hidden_size),
            num_classes=self.config.hidden_size,
        ).to(self.scale.dtype)

        rows = basis * self.scale * attention_mask.unsqueeze(-1)
        return Output((rows), (rows * 0.5, rows) if output_hidden_states else None)


class Tokenized:
    def __init__(
        self, input_ids: Tensor, attention_mask: Tensor, offset_mapping: Tensor
    ) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.offset_mapping = offset_mapping


class Tokenizer:
    """One row per non-space character plus a closing special row, padded with zeros."""

    pad_token_id = 0
    model_max_length = 128

    def __call__(
        self,
        texts: Sequence[str],
        *,
        padding: bool,
        return_tensors: str,
        truncation: bool,
        return_offsets_mapping: bool,
    ) -> Tokenized:
        del padding, return_tensors, truncation, return_offsets_mapping
        spans = [
            [(at, at + 1) for at, character in enumerate(text) if character != " "]
            for text in texts
        ]
        longest = max((len(entry) + 1 for entry in spans), default=0)
        input_ids = torch.zeros(len(texts), longest, dtype=torch.long)
        attention = torch.zeros(len(texts), longest, dtype=torch.long)
        offsets = torch.zeros(len(texts), longest, 2, dtype=torch.long)

        for item, (text, entry) in enumerate(zip(texts, spans, strict=True)):
            item_ids = [ord(text[start]) + 2 for start, _ in entry] + [1]
            input_ids[item, : len(item_ids)] = torch.tensor(item_ids)
            attention[item, : len(item_ids)] = 1

            if entry:
                offsets[item, : len(entry)] = torch.tensor(entry)

        return Tokenized(input_ids, attention, offsets)


class Loader:
    """A loading backend that records the checkpoints it is asked for."""

    def __init__(self) -> None:
        self.requested: list[PinnedCheckpoint] = []
        self.loaded: list[tuple[EncoderArchitecturePlugin, torch.dtype]] = []
        self.stack = Stack()

    def load_encoder(
        self,
        checkpoint: PinnedCheckpoint,
        *,
        architecture: EncoderArchitecturePlugin,
        dtype: torch.dtype,
    ) -> Stack:
        self.requested.append(checkpoint)
        self.loaded.append((architecture, dtype))

        return self.stack.to(dtype=dtype)

    def load_tokenizer(self, checkpoint: PinnedCheckpoint) -> Tokenizer:
        self.requested.append(checkpoint)

        return Tokenizer()


def head(cutpoints: tuple[float, ...] | None = CUTPOINTS) -> OrdinalHead:
    return OrdinalHead(LEVELS, cutpoints)


def objective(
    encoder: Rows | None = None,
    *,
    shared: OrdinalHead | None = None,
    weight: tuple[float, ...] = WEIGHT,
    bias: float = BIAS,
    name: str = "encoder",
) -> PooledEncoderObjective:
    built = PooledEncoderObjective(
        Rows() if encoder is None else encoder,
        head() if shared is None else shared,
        name=name,
    )

    with torch.no_grad():
        built.projection.weight.copy_(torch.tensor([weight]))
        built.projection.bias.copy_(torch.tensor([bias]))

    return built


def assemble(
    words: tuple[tuple[str, ...], ...] = WORDS,
    responses: tuple[tuple[str, str, int], ...] = RESPONSES,
    item_ids: tuple[str, ...] = ITEM_IDS,
) -> Outputs:
    used = frozenset(item_id for _, item_id, _ in responses)
    selected = tuple(
        (item_id, item_words)
        for item_id, item_words in zip(item_ids, words, strict=True)
        if item_id in used
    )
    selected_ids = tuple(item_id for item_id, _ in selected)
    selected_words = tuple(item_words for _, item_words in selected)
    design = DesignCodec(LEVELS).encode(
        JudgmentRow(participant, item_id, response, RowColumns({}, {}))
        for participant, item_id, response in responses
    )
    grouped = GroupedJudgmentSet(design, selected_ids, selected_words)
    batch = grouped.select(tuple(range(grouped.group_count)))
    live = torch.ones(len(selected_words), SPANS.num_spans, dtype=torch.bool)

    return ObjectiveOutputs(
        spans=SPANS,
        chart=CellChart(LogDistribution(torch.zeros(live.shape), SPAN_AXES, live)),
        batch=batch,
    )


def bundle() -> Outputs:
    return assemble()


def pooled_by_hand(words: tuple[str, ...]) -> list[float]:
    scale = scale_of()
    codes = [code_of(c) for word in words for c in word]
    summed = [0.0] * WIDTH

    for code in codes:
        summed[code] += scale[code]

    return [value / len(codes) for value in summed]


def latent_by_hand(words: tuple[str, ...]) -> float:
    return BIAS + sum(w * p for w, p in zip(WEIGHT, pooled_by_hand(words), strict=True))


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def level_probabilities(latent: float) -> list[float]:
    cumulative = [sigmoid(cutpoint - latent) for cutpoint in CUTPOINTS]

    return [
        cumulative[0],
        *(above - below for below, above in pairwise(cumulative)),
        1.0 - cumulative[-1],
    ]


def by_hand(outputs: Outputs) -> float:
    batch = outputs.batch
    read = [
        -math.log(level_probabilities(latent_by_hand(batch.words[item]))[level])
        for item, level in zip(
            batch.response_items.tolist(),
            batch.design.responses.tolist(),
            strict=True,
        )
    ]

    return sum(read) / len(read)


def permuted(outputs: Outputs) -> Outputs:
    order = [2, 0, 1]
    batch = outputs.batch

    return assemble(
        tuple(batch.words[position] for position in order),
        tuple(
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
        tuple(batch.item_ids[position] for position in order),
    )


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


BOTTOM: Final = tuple(("participant", item_id, 0) for item_id in ITEM_IDS)
TOP: Final = tuple(("participant", item_id, LEVELS - 1) for item_id in ITEM_IDS)


def fixture() -> ObjectiveFixture[_GroupedJudgmentBatch]:
    encoder = Rows()
    scored = objective(encoder, bias=3.0)

    def refilled(outputs: Outputs) -> Outputs:
        encoder.fill = -7.0

        return outputs

    return ObjectiveFixture(
        objective=scored,
        outputs=bundle,
        inputs=lambda _: (
            scored.projection.weight,
            scored.projection.bias,
            scored.head.base,
            scored.head.spans,
        ),
        unweighted=lambda outputs: by_hand_at(outputs, bias=3.0),
        monotone=lambda: (assemble(responses=BOTTOM), assemble(responses=TOP)),
        models_built=models_built,
        permuted=permuted,
        masked=refilled,
        tolerance=1e-6,
    )


def by_hand_at(outputs: Outputs, *, bias: float) -> float:
    batch = outputs.batch
    read = [
        -math.log(
            level_probabilities(latent_by_hand(batch.words[item]) - BIAS + bias)[level]
        )
        for item, level in zip(
            batch.response_items.tolist(),
            batch.design.responses.tolist(),
            strict=True,
        )
    ]

    return sum(read) / len(read)


class TestPooledEncoderObjectiveLaws(ObjectiveLawSuite):
    @pytest.fixture
    def conformance(self) -> ObjectiveFixture[_GroupedJudgmentBatch]:
        return fixture()

    @override
    def test_the_value_is_monotone_where_the_fixture_says[BatchT](
        self,
        conformance: ObjectiveFixture[BatchT],
    ) -> None:
        """Check that bottom-level judgments of high-scoring items score above top."""
        the_value_is_monotone_where_the_fixture_says(conformance)


def test_the_objective_answers_the_port_and_declares_what_it_reads() -> None:
    scored = objective()

    assert isinstance(scored, Objective)
    assert isinstance(scored.encoder, TokenEncoder)
    assert scored.name == "encoder"
    assert scored.requires == frozenset({"batch"})
    assert scored.levels == LEVELS
    assert objective(name="pooled").name == "pooled"


def test_the_value_is_the_mean_negative_log_density_of_the_judgments() -> None:
    outputs = bundle()

    term = objective().evaluate(outputs)

    assert term.value.ndim == 0
    assert term.value.item() == pytest.approx(by_hand(outputs), rel=1e-6)
    assert term.value.item() > 0.0
    assert term.count == len(RESPONSES)
    assert term.residual is None


def test_the_value_agrees_with_the_head_the_full_model_uses() -> None:
    scored = objective()
    latent = torch.tensor(
        [latent_by_hand(WORDS[ITEM_IDS.index(item_id)]) for _, item_id, _ in RESPONSES]
    )
    responses = torch.tensor([level for _, _, level in RESPONSES])
    log_probabilities = log_category_probabilities(latent, scored.head.cutpoints())
    reference = -log_probabilities.gather(1, responses.unsqueeze(1)).mean()

    term = scored.evaluate(bundle())

    assert term.value.item() == pytest.approx(reference.item(), rel=1e-6)


def test_the_latent_scores_are_the_projection_of_the_pooled_rows() -> None:
    scored = objective()

    latent = scored.latent_scores(WORDS)

    assert latent.shape == (len(WORDS),)
    assert latent.tolist() == pytest.approx(
        [latent_by_hand(words) for words in WORDS], rel=1e-6
    )
    assert latent.grad_fn is not None


def test_the_pooled_rows_are_the_mean_of_each_items_live_rows() -> None:
    encoder = Rows()
    encoder.fill = 9.0

    pooled = pooled_rows(encoder.encode_words(WORDS))

    assert pooled.shape == (len(WORDS), WIDTH)

    for item, words in enumerate(WORDS):
        assert pooled[item].tolist() == pytest.approx(pooled_by_hand(words), rel=1e-6)


def test_rows_without_a_mask_are_all_pooled() -> None:
    rows = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    encoding = TokenEncoding(
        Encoded(rows, AXES, torch.ones(2, 3, dtype=torch.bool)),
        (Alignment(((0, 3),)), Alignment(((0, 3),))),
    )

    assert torch.equal(pooled_rows(encoding), rows.mean(dim=1))


def test_an_item_with_no_word_is_refused_at_the_grouped_boundary() -> None:
    with pytest.raises(ValueError, match="has no word"):
        objective().evaluate(
            assemble(
                (("ab",), (), ("c",)),
                (("participant", "second", 1),),
            )
        )


def test_the_padded_rows_do_not_move_the_value() -> None:
    encoder = Rows()
    scored = objective(encoder)
    before = scored.evaluate(bundle()).value.item()
    encoder.fill = float("nan")

    after = scored.evaluate(bundle()).value.item()

    assert after == pytest.approx(before, rel=1e-9)


def test_the_diagnostics_report_density_accuracy_and_rank_correlation() -> None:
    outputs = bundle()
    latent = [
        latent_by_hand(WORDS[ITEM_IDS.index(item_id)]) for _, item_id, _ in RESPONSES
    ]
    levels = [level for _, _, level in RESPONSES]
    modal = [
        max(range(LEVELS), key=lambda level: level_probabilities(score)[level])
        for score in latent
    ]
    hits = sum(1 for guess, level in zip(modal, levels, strict=True) if guess == level)

    term = objective().evaluate(outputs)

    assert set(term.diagnostics) == {"log_density", "accuracy", "rank_correlation"}
    assert term.diagnostics["log_density"].item() == pytest.approx(
        -by_hand(outputs), rel=1e-6
    )
    assert term.diagnostics["accuracy"].item() == pytest.approx(hits / len(RESPONSES))
    assert term.diagnostics["rank_correlation"].item() == pytest.approx(
        rank_correlation(latent, levels), rel=1e-6
    )

    for diagnostic in term.diagnostics.values():
        assert diagnostic.ndim == 0
        assert diagnostic.grad_fn is None


def test_the_rank_correlation_is_a_gap_where_it_is_undefined() -> None:
    scored = objective()

    two = scored.evaluate(
        assemble(
            responses=(
                ("participant", "first", 4),
                ("participant", "second", 0),
            )
        )
    )
    flat = scored.evaluate(
        assemble(responses=tuple(("participant", item_id, 2) for item_id in ITEM_IDS))
    )
    one_item = scored.evaluate(
        assemble(
            responses=tuple(("participant", "first", level) for level in (1, 2, 3))
        )
    )

    assert "rank_correlation" not in two.diagnostics
    assert "rank_correlation" not in flat.diagnostics
    assert "rank_correlation" not in one_item.diagnostics
    assert {"log_density", "accuracy"} <= set(two.diagnostics)


@pytest.mark.parametrize("loaded", ["frozen", "trainable"])
def test_no_encoder_parameter_takes_a_gradient_and_the_head_does(loaded: str) -> None:
    encoder = Rows(trainable=loaded == "trainable")
    scored = objective(encoder)

    scored.evaluate(bundle()).value.backward()

    assert encoder.scale.grad is None
    assert scored.head.base.grad is not None
    assert scored.head.spans.grad is not None
    assert scored.projection.weight.grad is not None
    assert scored.projection.bias.grad is not None
    assert float(scored.head.base.grad.abs()) > 0.0
    assert float(scored.head.spans.grad.abs().sum()) > 0.0
    assert float(scored.projection.weight.grad.abs().sum()) > 0.0
    assert float(scored.projection.bias.grad.abs()) > 0.0


def test_the_encoder_is_run_once_per_evaluation_outside_the_graph() -> None:
    encoder = Rows(trainable=True)
    scored = objective(encoder)

    scored.evaluate(bundle())
    latent = scored.latent_scores(WORDS)
    reached = torch.autograd.grad(latent.sum(), [encoder.scale], allow_unused=True)

    assert encoder.calls == 2
    assert reached == (None,)
    assert pooled_rows(encoder.encode_words(WORDS)).grad_fn is not None


def test_the_trainable_parameters_are_the_projection_and_the_head() -> None:
    scored = objective(Rows(trainable=False))
    trainable = {
        name for name, parameter in scored.named_parameters() if parameter.requires_grad
    }

    assert trainable == {
        "projection.weight",
        "projection.bias",
        "head.base",
        "head.spans",
    }
    assert "encoder.scale" in dict(scored.named_parameters())


def test_the_head_is_the_object_the_full_model_holds() -> None:
    shared = head()
    full = JudgmentLikelihoodObjective(
        readout=ReadoutId("sentences"),
        linking=ProbabilityLinking(
            LinkTerm(readout="p_gram", linking=LinkingSpec(family=PROBABILITY))
        ),
        head=shared,
    )
    baseline = objective(shared=shared)

    assert baseline.head is full.head
    assert (
        dict(baseline.named_parameters())["head.base"]
        is (dict(full.named_parameters())["head.base"])
    )


def test_a_step_of_gradient_descent_lowers_the_value() -> None:
    outputs = bundle()
    scored = objective()
    optimizer = torch.optim.SGD(
        [parameter for parameter in scored.parameters() if parameter.requires_grad],
        lr=0.05,
    )
    before = scored.evaluate(outputs).value

    before.backward()
    optimizer.step()
    after = scored.evaluate(outputs).value

    assert after.item() < before.item()


def test_the_objective_reads_the_batch_alone() -> None:
    scored = objective()
    proxy = RecordingOutputs(bundle(), objective=scored.name, requires=scored.requires)

    term = scored.evaluate(proxy)

    assert proxy.touched == {"batch"}
    assert term.count == len(RESPONSES)


def test_an_empty_name_is_refused() -> None:
    with pytest.raises(ValueError, match="name"):
        objective(name="")


def test_the_projection_is_as_wide_as_the_encoder() -> None:
    scored = PooledEncoderObjective(Rows(width=6), head())

    assert scored.projection.in_features == 6
    assert scored.projection.out_features == 1
    assert scored.latent_scores((("ab",),)).shape == (1,)


def test_the_fixtures_transform_the_outputs_they_are_handed() -> None:
    outputs = bundle()
    moved = permuted(outputs)

    assert moved.batch.words == (WORDS[2], WORDS[0], WORDS[1])
    assert sorted(moved.batch.design.responses.tolist()) == sorted(
        level for _, _, level in RESPONSES
    )
    assert by_hand(moved) == pytest.approx(by_hand(outputs), rel=1e-12)


def test_the_frozen_encoder_objective_loads_the_named_checkpoint_untrained() -> None:
    loader = Loader()

    scored = frozen_encoder_objective(
        head(),
        checkpoint=PINNED,
        architecture=ENCODER_ONLY,
        dtype=torch.float64,
        loader=loader,
    )
    encoder = scored.encoder

    assert loader.requested == [PINNED] * 2
    assert loader.loaded == [(ENCODER_ONLY, torch.float64)]
    assert isinstance(encoder, PretrainedTokenEncoder)
    assert encoder.trainable is False
    assert encoder.checkpoint == PINNED
    assert encoder.window == ContextWindow(128, overflow=REPORT)
    assert loader.stack.scale.dtype is torch.float64
    assert encoder.output_width == WIDTH
    assert scored.projection.in_features == WIDTH
    assert scored.name == "encoder"
    assert not loader.stack.scale.requires_grad


def test_the_frozen_encoder_objective_runs_the_stack_its_architecture_names() -> None:
    loader = Loader()

    scored = frozen_encoder_objective(
        head(),
        checkpoint=PINNED,
        architecture=T5_ENCODER,
        dtype=torch.float32,
        loader=loader,
    )
    encoder = scored.encoder

    assert loader.loaded == [(T5_ENCODER, torch.float32)]
    assert isinstance(encoder, PretrainedTokenEncoder)
    assert encoder.architecture is T5_ENCODER
    assert encoder.output_width == WIDTH


def test_the_frozen_encoder_objective_names_the_checkpoint_and_the_term() -> None:
    loader = Loader()

    pinned = PinnedCheckpoint("x/y", "c" * 40)
    scored = frozen_encoder_objective(
        head(),
        checkpoint=pinned,
        architecture=ENCODER_ONLY,
        dtype=torch.float32,
        loader=loader,
        name="p",
    )
    encoder = scored.encoder

    assert loader.requested == [pinned, pinned]
    assert scored.name == "p"
    assert isinstance(encoder, PretrainedTokenEncoder)
    assert encoder.checkpoint == pinned


def test_the_frozen_encoder_objective_refuses_an_empty_name_before_loading() -> None:
    loader = Loader()

    with pytest.raises(ValueError, match="name"):
        frozen_encoder_objective(
            head(),
            checkpoint=PINNED,
            architecture=ENCODER_ONLY,
            dtype=torch.float32,
            loader=loader,
            name="",
        )

    assert loader.requested == []


def test_the_frozen_encoder_objective_trains_the_head_and_not_the_stack() -> None:
    loader = Loader()
    shared = head()
    scored = frozen_encoder_objective(
        shared,
        checkpoint=PINNED,
        architecture=ENCODER_ONLY,
        dtype=torch.float32,
        loader=loader,
    )
    latent = scored.latent_scores(WORDS)

    scored.evaluate(bundle()).value.backward()
    base = shared.base.grad
    weight = scored.projection.weight.grad

    assert latent.shape == (len(WORDS),)
    assert loader.stack.scale.grad is None
    assert scored.head is shared
    assert base is not None
    assert weight is not None
    assert float(base.abs()) > 0.0
    assert float(weight.abs().sum()) > 0.0
    assert {
        name for name, parameter in scored.named_parameters() if parameter.requires_grad
    } == {"projection.weight", "projection.bias", "head.base", "head.spans"}

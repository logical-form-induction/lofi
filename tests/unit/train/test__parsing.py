import math
from typing import TYPE_CHECKING, Final, NamedTuple, cast, override

import didactic.api as dx
import pytest
import torch
from tests.conformance.train_step import TrainStepConformance

from lofi._types import ReadoutId
from lofi.deduce import SpanIndex
from lofi.judge import Action, ActionSet
from lofi.nn._values import ChartActionScores, ChartReadouts, Encoded, LogDistribution
from lofi.nn.codec import NeuralCodecView
from lofi.nn.decoders import RecursiveTermDecoder, TreeDecoderSettings
from lofi.nn.loss._errors import MissingOutputError
from lofi.nn.loss._mapping import SOFTMAX
from lofi.nn.loss._parser import (
    SPAN_AXES,
    SPLIT_AXES,
)
from lofi.nn.loss._ports import LossTerm
from lofi.nn.loss._readout import (
    TermsReadout,
)
from lofi.nn.loss._values import OUTSIDE_CHART, ObjectiveOutputs
from lofi.nn.sites import ComponentId
from lofi.nn.tokens.embedder import (
    ReservedTokens,
    StaticTokenEmbedder,
    TokenVocabulary,
)
from lofi.nn.tokens.pool import FirstSubwordPooler
from lofi.nn.tokens.window import REPORT, ContextWindow
from lofi.nn.type_directed import UNRESTRICTED_ADMISSION
from lofi.spec.readouts import DecodedMeaningSettings, ReadoutSource
from lofi.spec.readouts.terms import ReadoutMass
from lofi.terms import (
    ChoiceIndex,
    CompleteTree,
    Operator,
    OperatorIndex,
    Signature,
    TermCodec,
)
from lofi.train import LossAssembly, OptimizationResult, WeightedObjective, edges
from lofi.train._parsing import ChartForward, Featurized, ParserStep
from lofi.train._ports import Forward
from lofi.train._pretraining import Trained
from lofi.train.laws import StepFixture

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from torch import Tensor

    from lofi.nn._values import DerivationScoreChart
    from lofi.nn.loss._ports import PublishedOutputs
    from lofi.nn.ports import ChartScorer, SubwordPooler, TokenEncoder
    from lofi.nn.sites import SiteMap


SIGNATURE: Final = Signature(
    "parsed",
    ("T",),
    (
        Operator("A", "T", family="atom"),
        Operator("B", "T", family="atom"),
        Operator("f", "T", arg_sorts=("T", "T"), family="functor"),
    ),
)
CODEC: Final = TermCodec(
    OperatorIndex(SIGNATURE),
    CompleteTree(SIGNATURE, SIGNATURE.operators, order="df", depth=2),
)
VIEW: Final = NeuralCodecView(CODEC)
CHOICES: Final = ChoiceIndex(SIGNATURE, depth_bounds={"T": 0})
WIDTH: Final = 6
RESERVED: Final = ReservedTokens({"<pad>": 0, "<unk>": 1})
WORDS: Final = ("the", "cat", "sat", "down")
VOCABULARY: Final = TokenVocabulary(WORDS, reserved=RESERVED, unknown="<unk>")
IDENTITY: Final = Action("identity")
ACTIONS: Final = ActionSet((IDENTITY, Action("apply", "application")), IDENTITY)
AVAILABLE: Final = frozenset({"batch", "step"})
EVERY: Final = frozenset(
    {
        "inside",
        "outside",
        "alpha",
        "log_z",
        "actions",
        "identity",
        "cells",
        "outside_cells",
        "spans",
    }
)


class Tokens(NamedTuple):
    words: tuple[tuple[str, ...], ...]


class SumScorer:
    """Scores each span by a learned reading of the mean of its leaves."""

    def __init__(
        self,
        declared: frozenset[str] = EVERY,
        action_set: ActionSet | None = ACTIONS,
    ) -> None:
        self.declared = declared
        self.action_set = action_set
        self.weight = torch.linspace(-1.0, 1.0, WIDTH, dtype=torch.float64)
        self.weight.requires_grad_()
        self.filled: dict[str, bool] = {}
        self.derivations = cast("DerivationScoreChart[object]", object())

    @property
    def publishes(self) -> frozenset[str]:
        return self.declared

    def _fill(self, name: str) -> bool:
        return name in self.declared and self.filled.get(name, True)

    def readouts(self, leaves: Encoded, /) -> ChartReadouts:
        items, n, _ = leaves.tensor.shape
        spans = SpanIndex(n)
        held = (
            torch.ones(items, n, dtype=torch.bool)
            if leaves.mask is None
            else leaves.mask
        )
        lengths = held.sum(dim=1)
        cells = torch.stack(
            [
                leaves.tensor[:, i:j].mean(dim=1)
                for i, j in (spans.span_of(sid) for sid in range(spans.num_spans))
            ],
            dim=1,
        )
        live = torch.stack(
            [lengths >= spans.span_of(sid)[1] for sid in range(spans.num_spans)],
            dim=1,
        )
        inside = (cells @ self.weight).masked_fill(~live, -torch.inf)
        chart = LogDistribution(inside, SPAN_AXES, live)
        split_mask = torch.tensor(spans.split_mask, dtype=torch.bool)
        valid = split_mask.unsqueeze(0) & live.unsqueeze(-1)
        counts = valid.sum(dim=-1, keepdim=True).clamp(min=1).to(inside.dtype)
        alpha = torch.where(valid, -counts.log(), torch.full_like(counts, -torch.inf))
        action_scores = None

        if self._fill("actions"):
            if self.action_set is None:
                msg = "action scores require an action set"
                raise ValueError(msg)

            offsets = torch.arange(
                len(self.action_set.actions), dtype=inside.dtype
            ).reshape(1, 1, 1, -1)
            weighted = inside.unsqueeze(-1).unsqueeze(-1) + offsets
            weighted = weighted.expand(-1, -1, spans.max_splits, -1)
            action_scores = ChartActionScores(
                self.action_set,
                LogDistribution(
                    weighted.masked_fill(~valid.unsqueeze(-1), -torch.inf),
                    (*SPLIT_AXES, "action"),
                    valid,
                ),
            )

        roots = torch.tensor(
            [spans.span_id(0, int(length)) if length else 0 for length in lengths],
            dtype=torch.long,
        )
        log_z = inside.gather(1, roots.unsqueeze(1)).squeeze(1)

        return ChartReadouts(
            inside=chart,
            outside=LogDistribution(-inside, SPAN_AXES, live)
            if self._fill("outside")
            else None,
            alpha=LogDistribution(alpha, SPLIT_AXES, valid, normalized_over="split")
            if self._fill("alpha")
            else None,
            mu=Encoded(live.to(inside.dtype), SPAN_AXES, live)
            if self._fill("mu")
            else None,
            log_z=LogDistribution(log_z, ("item",)) if self._fill("log_z") else None,
            actions=action_scores,
            cells=Encoded(cells, (*SPAN_AXES, "width"), live)
            if self._fill("cells")
            else None,
            outside_cells=Encoded(-cells, (*SPAN_AXES, "width"), live)
            if self._fill("outside_cells")
            else None,
            spans=Encoded(
                torch.cat((cells, -cells), dim=-1),
                (*SPAN_AXES, "span_width"),
                live,
            )
            if self._fill("spans")
            else None,
            derivations=self.derivations if self._fill("derivations") else None,
        )

    def site_map(self) -> SiteMap:
        msg = "no site is published"
        raise NotImplementedError(msg)


class Descent:
    """Plain gradient descent, reporting whether it stepped."""

    def __init__(self, parameters: Sequence[Tensor], rate: float = 0.1) -> None:
        self.parameters = tuple(parameters)
        self.rate = rate
        self.calls = 0

    def step(self, forward: Callable[[], Tensor], /) -> OptimizationResult:
        self.calls += 1

        for parameter in self.parameters:
            parameter.grad = None

        loss = forward()

        if not bool(torch.isfinite(loss)):
            return OptimizationResult(loss_finite=False, stepped=False, scale=None)

        loss.backward()

        with torch.no_grad():
            for parameter in self.parameters:
                if parameter.grad is not None:
                    parameter.sub_(self.rate * parameter.grad)

        return OptimizationResult(loss_finite=True, stepped=True, scale=None)


class Partition:
    """Pushes each item's log partition down, reading only the scores."""

    def __init__(self, name: str = "partition") -> None:
        self._name = name
        self.published: list[ObjectiveOutputs[Tokens]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def requires(self) -> frozenset[str]:
        return frozenset({"scores"})

    def evaluate(self, outputs: PublishedOutputs[Tokens], /) -> LossTerm:
        scores = outputs.scores

        if scores is None:
            msg = "the scores were declared and not published"
            raise ValueError(msg)

        return LossTerm(scores.tensor.mean(), int(scores.tensor.shape[0]))


class Reader:
    """Records the bundle and contributes a zero term over its items."""

    def __init__(self, requires: frozenset[str], name: str = "reader") -> None:
        self._name = name
        self._requires = requires
        self.bundles: list[PublishedOutputs[Tokens]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def requires(self) -> frozenset[str]:
        return self._requires | {"batch"}

    def evaluate(self, outputs: PublishedOutputs[Tokens], /) -> LossTerm:
        self.bundles.append(outputs)

        return LossTerm(
            torch.zeros((), dtype=torch.float64, requires_grad=True),
            len(outputs.batch.words),
        )


def embedder(window: ContextWindow | None = None) -> StaticTokenEmbedder:
    torch.manual_seed(11)

    return StaticTokenEmbedder(
        torch.randn(VOCABULARY.size, WIDTH, dtype=torch.float64),
        vocabulary=VOCABULARY,
        component=ComponentId("embedder"),
        trainable=True,
        window=window,
        load_report=None,
    )


def pooler() -> FirstSubwordPooler:
    return FirstSubwordPooler(WIDTH, component=ComponentId("pooler"))


def readout(
    reads: ReadoutSource = ReadoutSource.INSIDE,
) -> TermsReadout:
    torch.manual_seed(3)
    decoder = RecursiveTermDecoder(
        codec=VIEW,
        choices=CHOICES,
        admission=UNRESTRICTED_ADMISSION,
        component=ComponentId("decoder"),
        settings=TreeDecoderSettings(
            WIDTH if reads is ReadoutSource.INSIDE else 2 * WIDTH,
            4,
        ),
        choice_heads="flat",
    ).to(torch.float64)

    return TermsReadout(
        decoder,
        codec=VIEW,
        index=CHOICES,
        reads=reads,
        decoding=None,
        key=ReadoutId("terms"),
        targets={},
        strategy=ReadoutMass.FACTORIZED,
        mapping=SOFTMAX,
        store=None,
        root_sort="T",
        max_nodes=None,
        clamp=None,
        meanings=DecodedMeaningSettings("decoded"),
        proof_terms=None,
        approximation=None,
        run_approximation=None,
    )


def forward(
    declared: frozenset[str] = EVERY,
    *,
    readouts: Sequence[TermsReadout] = (),
    actions: ActionSet | None = ACTIONS,
    window: ContextWindow | None = None,
) -> ChartForward[Tokens]:
    return ChartForward(
        cast("TokenEncoder", embedder(window)),
        cast("SubwordPooler", pooler()),
        cast("ChartScorer", SumScorer(declared, actions)),
        readouts=readouts,
        actions=actions,
    )


class Parsing:
    """One parser step over the sum scorer, with the parameters it moves."""

    def __init__(self, *, rate: float = 0.1, weight: float = 1.0) -> None:
        self.tokens = embedder()
        self.scorer = SumScorer()
        self.rate = rate
        self.weight = weight
        self.policies: list[Descent] = []

    def forward(self) -> ChartForward[Tokens]:
        return ChartForward(
            cast("TokenEncoder", self.tokens),
            cast("SubwordPooler", pooler()),
            cast("ChartScorer", self.scorer),
            actions=ACTIONS,
        )

    def policy(self) -> Descent:
        built = Descent((*self.tokens.parameters(), self.scorer.weight), self.rate)
        self.policies.append(built)

        return built

    def step(
        self, *, role: str = "parser", name: str | None = None
    ) -> ParserStep[Tokens]:
        partition = Partition()
        assembly = LossAssembly[Tokens](
            (WeightedObjective(partition.name, self.weight, partition),)
        )

        return ParserStep(self.forward(), assembly, self.policy(), role=role, name=name)


class SeenFeaturized(Featurized[Tokens]):
    __slots__ = ("_seen",)

    def __init__(self, batch: Tokens, step: int) -> None:
        super().__init__(batch, step)
        object.__setattr__(self, "_seen", set())

    @property
    @override
    def batch(self) -> Tokens:
        self._seen.add("batch")

        return super().batch

    @property
    @override
    def step(self) -> int:
        self._seen.add("step")

        return super().step

    @property
    def seen(self) -> frozenset[str]:
        return frozenset(self._seen)


ITEMS: Final = (("the", "cat", "sat"), ("down",), ("cat", "sat"))


def featurized(
    items: tuple[tuple[str, ...], ...] = ITEMS, *, step: int = 0
) -> SeenFeaturized:
    return SeenFeaturized(Tokens(items), step)


def reading(
    requires: frozenset[str], built: ChartForward[Tokens] | None = None
) -> tuple[Reader, ParserStep[Tokens]]:
    reader = Reader(requires)
    assembly = LossAssembly[Tokens]((WeightedObjective(reader.name, 1.0, reader),))
    step = ParserStep(
        forward() if built is None else built, assembly, Descent(()), role="parser"
    )

    return reader, step


class TestTheParserStepConforms(TrainStepConformance):
    @pytest.fixture
    def conformance(self) -> StepFixture[SeenFeaturized, Trained]:
        return StepFixture(
            step=Parsing().step(),
            inputs=featurized,
            reads=lambda inputs: inputs.seen,
            idle=lambda: featurized(()),
            variants=(Parsing(rate=0.01, weight=0.5).step(), Parsing(rate=0.5).step()),
            graph=(Parsing().step(),),
            available=AVAILABLE,
        )


def test_a_featurized_batch_is_frozen_and_names_its_position() -> None:
    inputs = Featurized(Tokens(ITEMS), 4)

    assert inputs.batch == Tokens(ITEMS)
    assert inputs.step == 4
    assert repr(inputs).startswith("Featurized(Tokens(")
    assert repr(inputs).endswith("step=4)")

    with pytest.raises(AttributeError, match="immutable"):
        inputs.__setattr__("step", 0)

    with pytest.raises(ValueError, match="step"):
        Featurized(Tokens(ITEMS), -1)


def test_the_chart_forward_is_a_forward_and_publishes_what_the_scorer_declares() -> (
    None
):
    every = forward()
    bare = forward(frozenset({"inside"}), actions=None)
    typed = forward(readouts=(readout(),))

    assert isinstance(every, Forward)
    assert every.publishes == frozenset(
        {
            "spans",
            "chart",
            "batch",
            OUTSIDE_CHART,
            "splits",
            "scores",
            "cells",
            "outside_cells",
            "actions",
        }
    )
    assert bare.publishes == frozenset({"spans", "chart", "batch"})
    assert typed.publishes == every.publishes | {"choice_rows", "readouts"}
    assert "publishes=" in repr(every)
    assert "SumScorer" in repr(every)


def test_the_bundle_holds_each_published_output_over_the_batch_span_index() -> None:
    bundle = forward().publish(Tokens(ITEMS))
    splits = bundle.splits
    scores = bundle.scores
    cells = bundle.cells
    outside_cells = bundle.outside_cells

    assert bundle.spans == SpanIndex(3)
    assert bundle.batch == Tokens(ITEMS)
    assert bundle.chart.inside.axes == SPAN_AXES
    assert bundle.chart.inside.size("span") == SpanIndex(3).num_spans
    assert bundle.chart.outside is not None
    assert splits is not None
    assert splits.items == 3
    assert splits.spans == SpanIndex(3)
    assert scores is not None
    assert scores.axes == ("item",)
    assert cells is not None
    assert cells.axes == (*SPAN_AXES, "width")
    assert outside_cells is not None
    assert outside_cells.axes == (*SPAN_AXES, "width")
    assert tuple(bundle.actions) == ACTIONS.actions
    assert bundle.actions[IDENTITY].axes == SPLIT_AXES
    assert bundle.actions[Action("apply", "application")].axes == SPLIT_AXES
    assert bundle.marginals is None
    assert bundle.readouts == {}
    assert math.isclose(
        scores.tensor[0].item(),
        bundle.chart.inside.tensor[0, SpanIndex(3).span_id(0, 3)].item(),
    )


def test_the_forward_propagates_declared_derivation_scoring() -> None:
    built = forward(EVERY | {"derivations"})
    bundle = built.publish(Tokens(ITEMS))

    assert "derivations" in built.publishes
    assert bundle.derivations is not None
    assert "derivations" in bundle.published


def test_the_bundle_names_the_words_the_token_encoder_left_out_of_each_item() -> None:
    windowed = forward(window=ContextWindow(2, overflow=REPORT))

    assert windowed.publish(Tokens(ITEMS)).truncated == ((2,), (), ())
    assert forward().publish(Tokens(ITEMS)).truncated == ((), (), ())


def test_the_step_reports_how_many_items_the_token_encoder_truncated() -> None:
    _, truncating = reading(
        frozenset({"spans"}),
        forward(window=ContextWindow(1, overflow=REPORT)),
    )
    _, whole = reading(frozenset({"spans"}))

    truncated = truncating.run(featurized()).result.metrics
    complete = whole.run(featurized()).result.metrics

    assert truncated["tokens/truncated_items"] == 2.0
    assert complete["tokens/truncated_items"] == 0.0


def test_the_step_hands_the_objectives_what_the_forward_published() -> None:
    reader, step = reading(frozenset({"spans", "chart", "scores"}))

    step.run(featurized())
    (bundle,) = reader.bundles
    scores = bundle.scores

    assert bundle.spans == SpanIndex(3)
    assert bundle.batch == Tokens(ITEMS)
    assert scores is not None
    assert scores.tensor.shape == (3,)


def test_a_terms_readout_publishes_over_the_cells_under_its_id() -> None:
    bundle = forward(readouts=(readout(),)).publish(Tokens(ITEMS))
    published = bundle.readouts[ReadoutId("terms")]
    choices = bundle.choice_rows[ReadoutId("terms")]

    assert set(bundle.readouts) == {ReadoutId("terms")}
    assert set(bundle.choice_rows) == {ReadoutId("terms")}
    assert published.spans == SpanIndex(3)
    assert published.items == 3
    assert published.axes[:2] == SPAN_AXES
    assert bool(published.computed[0].all())
    assert not bool(published.computed[1, SpanIndex(3).span_id(0, 2)])
    assert choices.axes == (*SPAN_AXES, "slot", "choice")
    assert choices.distribution.normalized_over == "choice"
    assert torch.equal(choices.computed, published.computed)


def test_a_span_readout_publishes_from_the_concatenated_span_representation() -> None:
    bundle = forward(readouts=(readout(ReadoutSource.SPAN),)).publish(Tokens(ITEMS))
    published = bundle.readouts[ReadoutId("terms")]

    assert published.spans == SpanIndex(3)
    assert published.items == 3
    assert published.axes[:2] == SPAN_AXES
    assert bool(published.computed[0].all())


def test_the_forward_publishes_rule_marginals_from_mu_and_splits() -> None:
    built = forward(EVERY | {"mu"})
    bundle = built.publish(Tokens(ITEMS))
    marginals = bundle.marginals
    splits = bundle.splits

    assert "marginals" in built.publishes
    assert marginals is not None
    assert splits is not None
    assert marginals.lengths.tolist() == [3, 1, 2]
    assert torch.equal(marginals.built, splits.valid.any(dim=-1))
    assert torch.allclose(
        marginals.rule.tensor,
        marginals.constituent.tensor.unsqueeze(-1)
        * splits.log_probabilities.tensor.exp().masked_fill(~splits.valid, 0.0),
    )


def test_a_bare_scorer_publishes_neither_actions_nor_splits_nor_scores() -> None:
    bundle = forward(frozenset({"inside"}), actions=None).publish(Tokens(ITEMS))

    assert bundle.chart.outside is None
    assert bundle.splits is None
    assert bundle.scores is None
    assert bundle.cells is None
    assert bundle.outside_cells is None
    assert bundle.actions == {}


def test_reused_outside_vectors_explain_why_the_contextual_output_is_absent() -> None:
    built = forward(EVERY - {"outside_cells"})

    assert "outside_cells" not in built.publishes
    assert "outside_weights" in built.unavailable_outputs["outside_cells"]
    assert built.publish(Tokens(ITEMS)).outside_cells is None


def test_an_action_free_scorer_ignores_the_runs_action_set() -> None:
    built = forward(frozenset({"inside"}), actions=ACTIONS)
    bundle = built.publish(Tokens(ITEMS))

    assert "actions" not in built.publishes
    assert bundle.actions == {}


def test_a_forward_refuses_what_it_could_not_publish() -> None:
    with pytest.raises(ValueError, match="inside pass"):
        forward(frozenset({"cells", "log_z"}), actions=None)

    with pytest.raises(ValueError, match="require chart vectors"):
        forward(frozenset({"inside"}), readouts=(readout(),), actions=None)

    with pytest.raises(ValueError, match="spans"):
        forward(EVERY - {"spans"}, readouts=(readout(ReadoutSource.SPAN),))

    with pytest.raises(ValueError, match="chart-action inventory"):
        forward(frozenset({"inside", "actions", "identity"}), actions=None)

    with pytest.raises(ValueError, match="published once"):
        forward(readouts=(readout(), readout()))


def test_a_scorer_that_declares_a_readout_and_fills_none_is_reported() -> None:
    scorer = SumScorer()
    built = ChartForward[Tokens](
        cast("TokenEncoder", embedder()),
        cast("SubwordPooler", pooler()),
        cast("ChartScorer", scorer),
        readouts=(readout(),),
        actions=ACTIONS,
    )
    scorer.filled["cells"] = False

    with pytest.raises(ValueError, match="declared 'inside' vectors"):
        built.publish(Tokens(ITEMS))


def test_a_scorer_that_declares_contextual_outside_vectors_must_fill_them() -> None:
    scorer = SumScorer()
    built = ChartForward[Tokens](
        cast("TokenEncoder", embedder()),
        cast("SubwordPooler", pooler()),
        cast("ChartScorer", scorer),
        actions=ACTIONS,
    )
    scorer.filled["outside_cells"] = False

    with pytest.raises(ValueError, match="contextual outside vectors"):
        built.publish(Tokens(ITEMS))


def test_a_forward_refuses_an_action_inventory_with_a_mismatched_permutation() -> None:
    permuted = ActionSet(tuple(reversed(ACTIONS.actions)), IDENTITY)
    scorer = SumScorer(action_set=permuted)
    built = ChartForward[Tokens](
        cast("TokenEncoder", embedder()),
        cast("SubwordPooler", pooler()),
        cast("ChartScorer", scorer),
        actions=ACTIONS,
    )

    with pytest.raises(ValueError, match="not the exact declared inventory") as caught:
        built.publish(Tokens(ITEMS))

    assert "('apply', 'identity')" in str(caught.value)
    assert "('identity', 'apply')" in str(caught.value)


def test_an_objective_reading_an_unpublished_output_fails_at_construction() -> None:
    reader = Reader(frozenset({"marginals", "scores"}))
    assembly = LossAssembly[Tokens]((WeightedObjective(reader.name, 1.0, reader),))

    with pytest.raises(MissingOutputError) as caught:
        ParserStep(forward(), assembly, Descent(()), role="parser")

    assert caught.value.objective == "reader"
    assert caught.value.missing == frozenset({"marginals"})
    assert "scores" in caught.value.published


def test_the_parser_step_reports_the_loss_and_names_its_role() -> None:
    built = Parsing()
    step = built.step()

    outcome = step.run(featurized(step=2))
    reported = outcome.result

    assert outcome.outputs == Trained("parser")
    assert reported.items == 3
    assert reported.dropped == 0
    assert reported.step == 3
    assert reported.metrics["optimizer/stepped"] == 1.0
    assert reported.metrics["optimizer/loss_finite"] == 1.0
    assert math.isfinite(reported.metrics["loss/partition"])
    assert step.produces == frozenset({"parser"})
    assert step.requires == AVAILABLE
    assert step.forward.publishes >= {"scores"}
    assert edges((step,), AVAILABLE) == ()


def test_the_parser_step_lowers_the_loss_it_trains() -> None:
    step = Parsing(rate=0.5).step()
    losses = [
        step.run(featurized(step=at)).result.metrics["loss/partition"]
        for at in range(6)
    ]

    assert losses[-1] < losses[0]


def test_a_batch_of_no_items_takes_no_step_and_reports_no_item() -> None:
    built = Parsing()
    outcome = built.step().run(featurized((), step=5))

    assert outcome.outputs == Trained("parser")
    assert outcome.result.items == 0
    assert outcome.result.step == 5
    assert outcome.result.metrics == {}
    assert built.policies[0].calls == 0


def test_a_step_is_named_after_its_role_unless_told_otherwise() -> None:
    built = Parsing()

    assert built.step().name == "train:parser"
    assert built.step(name="fit").name == "fit"
    assert built.step(role="chart").role == "chart"
    assert "role='parser'" in repr(built.step())
    assert "ChartForward" in repr(built.step())

    with pytest.raises(ValueError, match="role"):
        built.step(role="")

    with pytest.raises(ValueError, match="name"):
        built.step(name="")


def models_built(call: Callable[[], None]) -> int:
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


def test_a_hundred_parser_steps_construct_no_boundary_model() -> None:
    step = Parsing().step()

    def run() -> None:
        for at in range(100):
            step.run(featurized(step=at))

    assert models_built(run) == 0

"""The installed coherence declaration binds current represented-space roles."""

import json
import math
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, cast

import pytest
import torch
from didactic.settings import load_document

from lofi._types import ReadoutId
from lofi.deduce import SpanIndex
from lofi.judge import Action, ActionSet
from lofi.nn._values import Encoded, LogDistribution
from lofi.nn.codec import NeuralCodecView
from lofi.nn.loss._mapping import SOFTMAX
from lofi.nn.loss._parser import (
    SPAN_AXES,
    SPLIT_AXES,
    SpanMarginals,
    SplitPosterior,
)
from lofi.nn.loss._readout import (
    CELL_AXES,
    MASS_AXES,
    TermsReadout,
)
from lofi.nn.loss._values import CellChart, ObjectiveOutputs, SpanReadout
from lofi.nn.spaces import CombinatorFamily, Concatenate, Space
from lofi.registry import Registry
from lofi.run import packaged_options, section_root
from lofi.spec import (
    ApproximationSpec,
    DataModelLanguageSpec,
    DataSection,
    DatasetSpec,
    DevicePolicySpec,
    FeaturizerSpec,
    LossSection,
    ModelSection,
    OptimizerSpec,
    ParserSpec,
    ParseSpec,
    ReleaseSpec,
    RunSpec,
    TermDecoderSpec,
    TermEncoderSpec,
    TrainerSpec,
    WeightedLossSpec,
)
from lofi.spec._plugins import load_plugins
from lofi.spec.components.data import DatasetSection
from lofi.spec.decoding import DecodingSettings, ExhaustiveDecodingSpec
from lofi.spec.losses.coherence import CoherenceLossSpec
from lofi.spec.losses.grammaticality import GrammaticalityLossSpec
from lofi.spec.policy import ComputeDtype, MatmulPrecision
from lofi.spec.readouts import DecodedMeaningSettings, ReadoutSource, TermsReadoutSpec
from lofi.spec.readouts.terms import ReadoutMass
from lofi.spec.spaces import CombinatorFamilySpec, Merge, SpaceSpec
from lofi.terms import (
    BatchEncoding,
    ChoiceIndex,
    CompleteTree,
    HoleContext,
    Operator,
    OperatorIndex,
    Signature,
    TermCodec,
    TermStore,
)
from lofi.train import LossPlugin, resolve_loss_section

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.judge import RuleSet
    from lofi.nn.ports import TermEncoder
    from lofi.terms import Term


class FixtureParser(ParserSpec):
    kind: Literal["coherence-fixture-parser"] = "coherence-fixture-parser"

    def chart_outputs(self) -> frozenset[str]:
        """Return the chart values the fixture's readout consumes."""
        return frozenset({"cells"})


class FixtureEncoder(TermEncoderSpec):
    kind: Literal["coherence-fixture-encoder"] = "coherence-fixture-encoder"


class FixtureDecoder(TermDecoderSpec):
    kind: Literal["coherence-fixture-decoder"] = "coherence-fixture-decoder"


class FixtureDataset(DatasetSpec):
    kind: Literal["coherence-fixture-dataset"] = "coherence-fixture-dataset"


class FixtureFeaturizer(FeaturizerSpec):
    kind: Literal["coherence-fixture-featurizer"] = "coherence-fixture-featurizer"


class FixtureTrainer(TrainerSpec):
    kind: Literal["coherence-fixture-trainer"] = "coherence-fixture-trainer"


class FixtureOptimizer(OptimizerSpec):
    kind: Literal["coherence-fixture-optimizer"] = "coherence-fixture-optimizer"


class Batch(NamedTuple):
    """One parser item with an optional grammaticality label."""

    ids: tuple[str, ...]
    lengths: tuple[int, ...]
    labels: tuple[bool | None, ...]
    step: int = 0


class State(NamedTuple):
    """The scalar one frontier decoder conditions its atom choice on."""

    score: torch.Tensor


class Encoder:
    """Minimal codec-bound encoder identity for the represented space."""

    output_width = 2

    @property
    def codec(self) -> NeuralCodecView:
        """Return the represented space's exact codec view."""
        return CODEC


class Decoder:
    """One-step frontier decoder over the fixture's two atoms."""

    def start(self, inputs: Encoded, /) -> State:
        """Reduce one decoder input to its signed atom preference."""
        return State(inputs.tensor.sum())

    def weigh(self, state: State, /, *, context: HoleContext) -> LogDistribution:
        """Return the two legal atom weights at the root hole."""
        del context
        logits = state.score.new_full((CHOICES.width,), -torch.inf)
        logits[CHOICES.index("a")] = state.score
        logits[CHOICES.index("b")] = -state.score
        normalized = logits.log_softmax(0)

        return LogDistribution(
            normalized,
            ("choice",),
            normalized.isfinite(),
            normalized_over="choice",
        )

    def fill(self, state: State, /, *, choices: torch.Tensor) -> State:
        """Return the terminal state after the atom is selected."""
        del choices
        return state

    def admits(self, term: object, /) -> bool:
        """Admit every completed term in the fixture's finite language."""
        del term
        return True

    def marginal_rows(self, inputs: Encoded, /, *, root_sort: str) -> LogDistribution:
        del root_sort
        scores = inputs.tensor.sum(dim=-1)
        logits = scores.new_full((*scores.shape, 1, CHOICES.width), -torch.inf)
        logits[..., 0, CHOICES.index("a")] = scores
        logits[..., 0, CHOICES.index("b")] = -scores
        normalized = logits.log_softmax(dim=-1)

        return LogDistribution(
            normalized,
            ("term", "slot", "choice"),
            normalized.isfinite(),
            normalized_over="choice",
        )

    def rows(self, inputs: Encoded, /, *, terms: BatchEncoding) -> LogDistribution:
        del terms

        return self.marginal_rows(inputs, root_sort="T")


SPANS: Final = SpanIndex(2)
IDENTITY: Final = Action("identity")
APPLY: Final = Action("apply", "application")
ACTIONS: Final = ActionSet((IDENTITY, APPLY), IDENTITY)
ATOM_A: Final = Operator("a", "T", family="primitive")
ATOM_B: Final = Operator("b", "T", family="primitive")
SIGNATURE: Final = Signature("coherence", ("T",), (ATOM_A, ATOM_B)).with_holes()
STORE: Final = TermStore(SIGNATURE)
CHOICES: Final = ChoiceIndex(SIGNATURE, depth_bounds={"T": 1})
CODEC: Final = NeuralCodecView(
    TermCodec(
        OperatorIndex(SIGNATURE),
        CompleteTree(SIGNATURE, (ATOM_A, ATOM_B), order="bf", depth=1),
    )
)
DECODING: Final = DecodingSettings(
    "exhaustive", None, 0.0, 8, None, None, None, None, 1e-6
)
DECODER: Final = Decoder()
ENCODER: Final = cast("TermEncoder", Encoder())


class BinaryRule:
    name = "application"
    arity = 2


SPACE: Final = Space(
    "T",
    codec=CODEC,
    choices=CHOICES,
    store=STORE,
    max_nodes=1,
    encoder=ENCODER,
    identity=DECODER,
    combinators=CombinatorFamily(
        {APPLY: DECODER},
        actions=ACTIONS,
        rules=cast("RuleSet[Term]", (BinaryRule(),)),
        merge=Concatenate(2),
        codec=CODEC.codec,
        encoder=ENCODER,
        identity=DECODER,
        index=CHOICES,
        root_sort="T",
    ),
)
READOUT: Final = TermsReadout(
    DECODER,
    codec=CODEC,
    index=CHOICES,
    reads=ReadoutSource.INSIDE,
    decoding=DECODING,
    key=ReadoutId("category-terms"),
    targets={},
    strategy=ReadoutMass.FACTORIZED,
    mapping=SOFTMAX,
    store=STORE,
    root_sort="T",
    max_nodes=1,
    clamp=None,
    meanings=DecodedMeaningSettings("decoded"),
    proof_terms=None,
    approximation=None,
    run_approximation=None,
)


def run_spec(loss: LossSection) -> RunSpec:
    """Return a complete run declaration using the supplied loss section."""
    return RunSpec(
        approximation=ApproximationSpec(residual_mass=0.0),
        model=ModelSection(
            language=DataModelLanguageSpec(),
            parser=FixtureParser(),
            encoders={"terms": FixtureEncoder()},
            decoders={
                "identity": FixtureDecoder(),
                "application": FixtureDecoder(),
            },
            spaces={
                "categories": SpaceSpec(
                    sort="T",
                    encoder="terms",
                    identity="identity",
                    combinators=CombinatorFamilySpec(
                        merge=Merge.CONCATENATE,
                        decoders={"apply": "application"},
                    ),
                )
            },
            readouts={
                "category-terms": TermsReadoutSpec(
                    space="categories",
                    reads=ReadoutSource.INSIDE,
                    decoding=ExhaustiveDecodingSpec(residual_mass=0.0, cap=8),
                )
            },
        ),
        loss=loss,
        data=DataSection(
            datasets={
                "train": DatasetSection(
                    dataset=FixtureDataset(),
                    featurizer=FixtureFeaturizer(),
                )
            },
        ),
        trainer=FixtureTrainer(),
        release=ReleaseSpec(checkpoint="resume-latest"),
        parse=ParseSpec(retain="forest", reachability="all"),
        optimizer=FixtureOptimizer(),
        device_policy=DevicePolicySpec(
            device="cpu",
            compute_dtype=ComputeDtype.FLOAT32,
            deterministic=True,
            matmul_precision=MatmulPrecision.HIGHEST,
            activation_memory_bytes=8_000_000_000,
        ),
    )


def coherence_section() -> LossSection:
    """Return the packaged coherence loss section."""
    document = load_document(section_root("loss") / "coherence.yaml")

    return LossSection.model_validate_json(json.dumps(document))


def mixed_section() -> LossSection:
    """Return coherence followed by grammaticality under distinct weights."""
    return LossSection(
        terms=(
            WeightedLossSpec(
                objective=CoherenceLossSpec(
                    readout="category-terms",
                    space="categories",
                    warmup=None,
                ),
                weight=0.5,
            ),
            WeightedLossSpec(
                objective=GrammaticalityLossSpec(readout="sentences"),
                weight=2.0,
            ),
        )
    )


def outputs(*, label: bool | None = True) -> ObjectiveOutputs[Batch]:
    """Return one live binary chart with coherence and root-mass publications."""
    live = torch.ones((1, SPANS.num_spans), dtype=torch.bool)
    valid = torch.tensor(SPANS.split_mask, dtype=torch.bool).unsqueeze(0)
    built = torch.zeros_like(live)
    built[:, SPANS.span_id(0, 2)] = True
    cells = Encoded(
        torch.tensor([[[0.2, -0.1], [0.7, 0.3], [0.8, -0.4]]]),
        CELL_AXES,
        live,
    )
    split_values = torch.where(
        valid,
        torch.zeros(valid.shape),
        torch.full(valid.shape, -math.inf),
    )
    marginals = SpanMarginals(
        SPANS,
        Encoded(built.to(torch.float32), SPAN_AXES, built),
        Encoded(valid.to(torch.float32), SPLIT_AXES, valid),
        torch.tensor([2]),
    )
    root_mass = torch.full((1, SPANS.num_spans, 2), -math.inf)
    root_mass[0, SPANS.span_id(0, 2)] = torch.tensor([math.log(0.8), math.log(0.2)])
    readouts = {
        ReadoutId("sentences"): SpanReadout(
            SPANS,
            LogDistribution(
                root_mass,
                MASS_AXES,
                built,
                normalized_over="outcome",
            ),
        )
    }

    return ObjectiveOutputs(
        spans=SPANS,
        chart=CellChart(LogDistribution(torch.full(live.shape, -1.0), SPAN_AXES, live)),
        batch=Batch(("item",), (2,), (label,)),
        splits=SplitPosterior(
            SPANS,
            LogDistribution(
                split_values,
                SPLIT_AXES,
                valid,
                normalized_over="split",
            ),
        ),
        marginals=marginals,
        actions={
            APPLY: LogDistribution(
                split_values.detach().clone().requires_grad_(),
                SPLIT_AXES,
                valid,
            )
        },
        readouts=readouts,
        cells=cells,
    )


def installed_registry() -> Registry[LossPlugin[Batch]]:
    """Return the installed loss registry after loading declaration plugins."""
    load_plugins()

    return Registry("loss")


def test_a_complete_run_selects_packaged_coherence_and_round_trips_json() -> None:
    loss = coherence_section()
    run = run_spec(loss)
    restored = RunSpec.model_validate_json(run.model_dump_json())
    restored_loss = restored.loss

    assert restored_loss is not None
    objective = cast("CoherenceLossSpec", restored_loss.terms[0].objective)

    assert "coherence" in packaged_options("loss")
    assert type(objective) is CoherenceLossSpec
    assert objective.readout == "category-terms"
    assert objective.space == "categories"
    assert restored == run


def test_installed_coherence_and_grammaticality_preserve_suite_semantics() -> None:
    assembly = resolve_loss_section(
        mixed_section(),
        installed_registry(),
        published=frozenset(
            {"spans", "readouts", "cells", "splits", "marginals", "actions", "batch"}
        ),
        readouts={"category-terms": READOUT},
        spaces={"categories": SPACE},
        readout_spaces={"category-terms": SPACE},
        relations={},
        span_decoders={},
        proof_reward=None,
    )
    active = assembly.evaluate(outputs(), step=0)
    partial = assembly.evaluate(outputs(label=None), step=0)

    assert assembly.names == ("coherence", "grammaticality")
    assert assembly.weights == (0.5, 2.0)
    assert [record.name for record in active.terms] == ["coherence", "grammaticality"]
    assert [record.inactive for record in active.terms] == [0, 0]
    assert active.total.item() == pytest.approx(
        0.5 * active.terms[0].term.value.item()
        + 2.0 * active.terms[1].term.value.item()
    )
    assert "loss/coherence" in active.metrics
    assert "loss/coherence/controller_entropy" in active.metrics
    assert "loss/grammaticality" in active.metrics
    assert [record.inactive for record in partial.terms] == [0, 1]
    assert partial.inactive == 1
    assert "loss/coherence" in partial.metrics
    assert "loss/grammaticality" not in partial.metrics


@pytest.mark.parametrize(
    ("readouts", "spaces", "error", "detail"),
    [
        ({}, {"categories": SPACE}, ValueError, "term-readout role 'category-terms'"),
        (
            {"category-terms": SPACE},
            {"categories": SPACE},
            TypeError,
            "term-readout role 'category-terms'",
        ),
        ({"category-terms": READOUT}, {}, ValueError, "space role 'categories'"),
        (
            {"category-terms": READOUT},
            {"categories": READOUT},
            TypeError,
            "space role 'categories'",
        ),
    ],
)
def test_missing_or_wrongly_typed_roles_fail_during_loss_binding_by_name(
    readouts: dict[str, TermsReadout | Space],
    spaces: dict[str, TermsReadout | Space],
    error: type[Exception],
    detail: str,
) -> None:
    with pytest.raises(error, match=detail):
        resolve_loss_section(
            coherence_section(),
            installed_registry(),
            readouts=cast("Mapping[str, TermsReadout]", readouts),
            spaces=cast("Mapping[str, Space]", spaces),
            readout_spaces={"category-terms": SPACE},
            relations={},
            span_decoders={},
            proof_reward=None,
        )

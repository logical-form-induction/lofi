"""The combination matrix over the registered term encoders and decoders."""

import io
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple, Protocol, cast

import didactic.api as dx
import pytest
import torch
from scripts.report_combinations import (
    CONSTRUCTED,
    DECLARED_INCOMPATIBLE,
    count_outcomes,
    main,
    render_report,
)
from tests.conformance.component_plugin import plugin_context
from tests.conftest import counted_constructions
from tests.support.installed import installed
from torch import nn

from lofi.nn.codec import NeuralCodecView
from lofi.nn.decoders import (
    DiffusionTermDecoder,
    HolographicTermDecoder,
    RecursiveTermDecoder,
    TransformerTermDecoder,
)
from lofi.nn.encoders import (
    HrrTermEncoder,
    TransformerTermEncoder,
    TreeRnnEncoder,
)
from lofi.nn.ports import TeacherForcedDecoder, TermEncoder
from lofi.nn.sharing import (
    ModuleSharingProducer,
    StateTensor,
)
from lofi.nn.strategies import score
from lofi.registry import Registry, UnknownTagError
from lofi.run import (
    ExperimentSeed,
    RunStore,
    Selection,
    SweepPoint,
    compose,
    resolved_document,
)
from lofi.run import main as cli_main
from lofi.run.operations import TRAINING_OPERATIONS
from lofi.spec import config_search_path
from lofi.spec.components.model import (
    DataModelLanguageSpec,
    ModelSection,
    TermDecoderSpec,
    TermEncoderSpec,
)
from lofi.spec.decoders.diffusion import DiffusionTermDecoderSpec
from lofi.spec.decoders.hrr import HrrTermDecoderSpec
from lofi.spec.decoders.rnn import RnnTermDecoderSpec
from lofi.spec.decoders.transformer import TransformerTermDecoderSpec
from lofi.spec.encoders.hrr import HrrTermEncoderSpec
from lofi.spec.encoders.rnn import TreeRnnEncoderSpec
from lofi.spec.encoders.transformer import TransformerTermEncoderSpec
from lofi.spec.spaces import SpaceSpec
from lofi.terms import (
    ChoiceIndex,
    CompleteTree,
    Operator,
    OperatorIndex,
    Signature,
    Term,
    TermCodec,
    TermStore,
)
from lofi.train import _assembly as assembly_module
from lofi.train._spaces import SpaceGraph, assemble_space_graph, wire_spaces

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping

    from tests.composability.conftest import (
        ComposabilityHarness,
        HarnessFactory,
        Incompatibility,
        TagPair,
    )

    from lofi.nn.build import BuildContext
    from lofi.nn.spaces import TermMorphism
    from lofi.nn.tokens.pretrained import PretrainedLoader
    from lofi.spec.components.base import ComponentSpec
    from lofi.terms import BatchEncoding, TargetSet

type Producer = ModuleSharingProducer[nn.Parameter, StateTensor]
type Matrix = ComposabilityHarness[Registration, Registration, SpaceGraph]
type Storage = tuple[Hashable, int, int]


class CodecEncoder(TermEncoder, Protocol):
    """A fixture encoder bound to its represented-space codec."""

    @property
    def codec(self) -> NeuralCodecView:
        """Return the codec this encoder represents."""
        ...


ENCODER_FAMILY: Final = "term_encoder"
DECODER_FAMILY: Final = "term_decoder"
RUN: Final = "model"
ENCODER_ROLE: Final = "terms"
DECODER_ROLE: Final = "identity"
SPACE_ROLE: Final = "represented-terms"
ENCODER_PORT: Final = "encoder"
CODEBOOK: Final = "codebook"

WIDTH: Final = 8
HEADS: Final = 2
LAYERS: Final = 1
FEEDFORWARD_WIDTH: Final = 16
NOISE_STEPS: Final = 4
FIRST_MASKING: Final = 0.2
OPTIMIZER_STEPS: Final = 10
LEARNING_RATE: Final = 0.05
SEED: Final = 11

README: Final = Path(__file__).resolve().parents[2] / "README.md"

ATOM: Final = Operator("a", "T", family="primitive")
OTHER_ATOM: Final = Operator("b", "T", family="primitive")
UNARY: Final = Operator("raise", "T", arg_sorts=("T",), family="constructor")
BINARY: Final = Operator("pair", "T", arg_sorts=("T", "T"), family="constructor")
SIGNATURE: Final = Signature(
    "combination-matrix", ("T",), (ATOM, OTHER_ATOM, UNARY, BINARY)
)
CODEC: Final = TermCodec(
    OperatorIndex(SIGNATURE),
    CompleteTree(SIGNATURE, SIGNATURE.operators, order="bf", depth=3),
)
VIEW: Final = NeuralCodecView(CODEC)
CHOICES: Final = ChoiceIndex(SIGNATURE, depth_bounds={"T": CODEC.capacity})


def terms() -> BatchEncoding:
    store = TermStore(SIGNATURE)
    left = store.op(ATOM)
    right = store.op(OTHER_ATOM)

    return CODEC.encode_batch(
        (
            store.op(BINARY, store.op(UNARY, left), right),
            store.op(BINARY, right, left),
        )
    )


TERMS: Final = terms()


class Registration(NamedTuple):
    """A registered component: its declaration and its factory."""

    spec: ComponentSpec

    @property
    def tag(self) -> str:
        return cast("str", self.spec.model_dump()["kind"])


ENCODERS: Final = (
    Registration(HrrTermEncoderSpec(width=WIDTH)),
    Registration(TreeRnnEncoderSpec(width=WIDTH, layers=LAYERS)),
    Registration(
        TransformerTermEncoderSpec(
            width=WIDTH,
            heads=HEADS,
            layers=LAYERS,
            feedforward_width=FEEDFORWARD_WIDTH,
        )
    ),
)
DECODERS: Final = (
    Registration(HrrTermDecoderSpec()),
    Registration(RnnTermDecoderSpec(width=WIDTH)),
    Registration(
        TransformerTermDecoderSpec(
            width=WIDTH,
            heads=HEADS,
            layers=LAYERS,
            feedforward_width=FEEDFORWARD_WIDTH,
            structure="parent-recurrence",
        )
    ),
    Registration(
        DiffusionTermDecoderSpec(
            width=WIDTH,
            layers=LAYERS,
            feedforward_width=FEEDFORWARD_WIDTH,
            noise_steps=NOISE_STEPS,
            first_masking=FIRST_MASKING,
        )
    ),
)

HRR_ENCODER, RNN_ENCODER, TRANSFORMER_ENCODER = (each.tag for each in ENCODERS)
HRR_DECODER, RNN_DECODER, TRANSFORMER_DECODER, DIFFUSION_DECODER = (
    each.tag for each in DECODERS
)

ARCHITECTURES: Final = (
    HrrTermEncoder,
    TreeRnnEncoder,
    TransformerTermEncoder,
    HolographicTermDecoder,
    RecursiveTermDecoder,
    TransformerTermDecoder,
    DiffusionTermDecoder,
)

NO_CODEBOOK: Final = (
    f"selects publication {CODEBOOK!r} with sharing the space producer does not publish"
)

INCOMPATIBLE: Final[Mapping[TagPair, Incompatibility]] = {
    (RNN_ENCODER, HRR_DECODER): (ValueError, NO_CODEBOOK),
    (TRANSFORMER_ENCODER, HRR_DECODER): (ValueError, NO_CODEBOOK),
}

PAIRS: Final = tuple(
    product(
        sorted(each.tag for each in ENCODERS),
        sorted(each.tag for each in DECODERS),
    )
)
COMPATIBLE: Final = tuple(pair for pair in PAIRS if pair not in INCOMPATIBLE)
DECLARED: Final = tuple(pair for pair in PAIRS if pair in INCOMPATIBLE)


def _registry(
    family: str, registrations: tuple[Registration, ...]
) -> Registry[Registration]:
    return installed(
        family, {registration.tag: registration for registration in registrations}
    )


ENCODER_REGISTRY: Final = _registry(ENCODER_FAMILY, ENCODERS)
DECODER_REGISTRY: Final = _registry(DECODER_FAMILY, DECODERS)


def _wire(encoder: Registration, decoder: Registration) -> SpaceGraph:
    torch.manual_seed(SEED)

    model = ModelSection(
        language=DataModelLanguageSpec(),
        encoders={ENCODER_ROLE: cast("TermEncoderSpec", encoder.spec)},
        decoders={DECODER_ROLE: cast("TermDecoderSpec", decoder.spec)},
        spaces={
            SPACE_ROLE: SpaceSpec(sort="T", encoder=ENCODER_ROLE, identity=DECODER_ROLE)
        },
    )

    return wire_spaces(
        model,
        plugin_context(codec=VIEW, component=RUN),
        parser=None,
        pretrained_resources={},
        erasure=None,
    )


def _matrix(
    factory: HarnessFactory,
    incompatibilities: Mapping[TagPair, Incompatibility] = INCOMPATIBLE,
) -> Matrix:
    return factory(
        ENCODER_REGISTRY,
        DECODER_REGISTRY,
        construct=_wire,
        incompatibilities=incompatibilities,
    )


class Pair(NamedTuple):
    graph: SpaceGraph
    encoder: nn.Module
    decoder: nn.Module
    decoder_spec: TermDecoderSpec

    @property
    def encoding(self) -> TermEncoder:
        return cast("TermEncoder", self.encoder)

    @property
    def decoding(self) -> TeacherForcedDecoder:
        return cast("TeacherForcedDecoder", self.decoder)

    def declared(self) -> Producer | None:
        """Return the producer whose publication the decoder declared, if any."""
        declaration = self.decoder_spec.sharing().get(ENCODER_PORT)

        if declaration is None:
            return None

        return cast("Producer", self.encoder)


def _pair(pair: TagPair) -> Pair:
    encoder_tag, decoder_tag = pair
    graph = _wire(ENCODER_REGISTRY.get(encoder_tag), DECODER_REGISTRY.get(decoder_tag))

    components = graph.components

    return Pair(
        graph,
        cast("nn.Module", components[ENCODER_ROLE]),
        cast("nn.Module", components[DECODER_ROLE]),
        cast("TermDecoderSpec", DECODER_REGISTRY.get(decoder_tag).spec),
    )


def _parameter_ids(module: nn.Module) -> frozenset[int]:
    return frozenset(id(parameter) for parameter in module.parameters())


def _storages(module: nn.Module) -> frozenset[Storage]:
    return frozenset(
        (tensor.untyped_storage(), tensor.storage_offset(), tensor.numel())
        for tensor in module.state_dict().values()
    )


def _loss(pair: Pair) -> torch.Tensor:
    weights = score(pair.decoding, pair.encoding.encode(TERMS), TERMS, index=CHOICES)

    return -weights.tensor.mean()


def _optimizer(pair: Pair) -> torch.optim.Optimizer:
    held: dict[int, nn.Parameter] = {}

    for parameter in (*pair.encoder.parameters(), *pair.decoder.parameters()):
        held.setdefault(id(parameter), parameter)

    return torch.optim.SGD(list(held.values()), lr=LEARNING_RATE)


def _failed_pairs(matrix: Matrix, pairs: tuple[TagPair, ...]) -> tuple[TagPair, ...]:
    failed: list[TagPair] = []

    for pair in pairs:
        try:
            matrix.check(pair)

        except AssertionError, ValueError:
            failed.append(pair)

    return tuple(failed)


def _outcomes(matrix: Matrix) -> tuple[str, ...]:
    """Return one outcome per registered pair, constructing every undeclared one."""
    return tuple(c.outcome.value for c in matrix.evaluate())


ENCODER_ROWS: Final = [pytest.param(each, id=each.tag) for each in ENCODERS]
DECODER_ROWS: Final = [pytest.param(each, id=each.tag) for each in DECODERS]
COMPATIBLE_ROWS: Final = [pytest.param(pair, id=":".join(pair)) for pair in COMPATIBLE]
DECLARED_ROWS: Final = [pytest.param(pair, id=":".join(pair)) for pair in DECLARED]


def test_the_matrix_enumerates_every_registered_pair(
    composability_harness: HarnessFactory,
) -> None:
    assert _matrix(composability_harness).pairs() == PAIRS
    assert len(PAIRS) == len(ENCODERS) * len(DECODERS)
    assert set(COMPATIBLE) | set(DECLARED) == set(PAIRS)


def test_the_matrix_rows_cover_every_shipped_variant_of_both_families() -> None:
    assert {each.tag for each in ENCODERS} <= set(TermEncoderSpec.__variants__)
    assert {each.tag for each in DECODERS} <= set(TermDecoderSpec.__variants__)
    assert len({type(each.spec) for each in (*ENCODERS, *DECODERS)}) == len(
        ENCODERS
    ) + len(DECODERS)


@pytest.mark.parametrize("registration", ENCODER_ROWS)
def test_an_encoder_tag_resolves_to_the_variant_its_factory_lowers(
    registration: Registration,
) -> None:
    resolved = TermEncoderSpec.model_validate_json(registration.spec.model_dump_json())

    assert type(resolved) is type(registration.spec)
    assert TermEncoderSpec.__variants__[registration.tag] is type(registration.spec)


@pytest.mark.parametrize("registration", DECODER_ROWS)
def test_a_decoder_tag_resolves_to_the_variant_its_factory_lowers(
    registration: Registration,
) -> None:
    resolved = TermDecoderSpec.model_validate_json(registration.spec.model_dump_json())

    assert type(resolved) is type(registration.spec)
    assert TermDecoderSpec.__variants__[registration.tag] is type(registration.spec)


def test_an_unknown_tag_is_refused_at_the_boundary_with_the_tag_named() -> None:
    with pytest.raises(dx.ValidationError, match="'lstm-term-encoder'"):
        TermEncoderSpec.model_validate({"kind": "lstm-term-encoder", "width": WIDTH})

    registry = _registry("combination_matrix_term_encoder", ENCODERS)

    with pytest.raises(UnknownTagError, match="lstm-term-encoder"):
        registry.get("lstm-term-encoder")


@pytest.mark.parametrize("pair", COMPATIBLE_ROWS)
def test_an_undeclared_pair_constructs_through_the_wiring_boundary(
    composability_harness: HarnessFactory, pair: TagPair
) -> None:
    assert _matrix(composability_harness).check(pair) == CONSTRUCTED


@pytest.mark.parametrize("pair", DECLARED_ROWS)
def test_a_declared_pair_is_refused_at_the_wiring_boundary_with_its_reason(
    composability_harness: HarnessFactory, pair: TagPair
) -> None:
    assert _matrix(composability_harness).check(pair) == DECLARED_INCOMPATIBLE


def test_every_declared_reason_names_a_mechanism_and_no_architecture() -> None:
    identities = {
        *(each.tag for each in ENCODERS),
        *(each.tag for each in DECODERS),
        *(architecture.__name__ for architecture in ARCHITECTURES),
    }

    for _, reason in INCOMPATIBLE.values():
        assert CODEBOOK in reason
        assert all(identity not in reason for identity in identities)


@pytest.mark.parametrize("pair", COMPATIBLE_ROWS)
def test_a_constructed_pair_runs_forward_and_backward_to_a_finite_loss(
    pair: TagPair,
) -> None:
    wired = _pair(pair)
    loss = _loss(wired)
    loss.backward()

    assert loss.ndim == 0
    assert bool(loss.isfinite())

    for module in (wired.encoder, wired.decoder):
        gradients = [
            parameter.grad
            for parameter in module.parameters()
            if parameter.grad is not None
        ]

        assert gradients
        assert all(bool(gradient.isfinite().all()) for gradient in gradients)


@pytest.mark.parametrize("pair", COMPATIBLE_ROWS)
def test_a_constructed_pair_shares_exactly_what_it_declared(pair: TagPair) -> None:
    wired = _pair(pair)
    producer = wired.declared()
    shared_parameters = _parameter_ids(wired.decoder) & _parameter_ids(wired.encoder)
    shared_storages = _storages(wired.decoder) & _storages(wired.encoder)

    if producer is None:
        assert shared_parameters == frozenset()
        assert shared_storages == frozenset()

        return

    published = cast("nn.Module", producer.shared[CODEBOOK].module)

    assert shared_parameters == _parameter_ids(published)
    assert shared_storages == _storages(published)
    assert shared_parameters


@pytest.mark.parametrize("pair", COMPATIBLE_ROWS)
def test_ten_optimizer_steps_construct_no_didactic_model(pair: TagPair) -> None:
    wired = _pair(pair)
    optimizer = _optimizer(wired)
    losses: list[float] = []

    with counted_constructions() as counter:
        for _ in range(OPTIMIZER_STEPS):
            optimizer.zero_grad()
            loss = _loss(wired)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))

    assert counter.count == 0
    assert len(losses) == OPTIMIZER_STEPS
    assert all(torch.isfinite(torch.tensor(losses)).tolist())
    assert losses[-1] < losses[0]


def test_the_crossing_counter_sees_a_declaration_built_on_purpose() -> None:
    with counted_constructions() as counter:
        built = HrrTermEncoderSpec(width=WIDTH)

    assert built.width == WIDTH
    assert counter.count == 1


def test_declaring_a_constructing_pair_incompatible_exposes_exactly_that_pair(
    composability_harness: HarnessFactory,
) -> None:
    dishonest = (HRR_ENCODER, RNN_DECODER)
    matrix = _matrix(
        composability_harness,
        {**INCOMPATIBLE, dishonest: (ValueError, "a reason nothing raises")},
    )

    assert dishonest in COMPATIBLE
    assert _failed_pairs(matrix, COMPATIBLE) == (dishonest,)


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param({}, id="row-removed"),
        pytest.param(
            {DECLARED[0]: (TypeError, NO_CODEBOOK)},
            id="wrong-exception",
        ),
        pytest.param(
            {DECLARED[0]: (ValueError, "another reason")},
            id="wrong-reason",
        ),
    ],
)
def test_misdeclaring_a_genuine_refusal_exposes_exactly_that_pair(
    composability_harness: HarnessFactory,
    mutation: Mapping[TagPair, Incompatibility],
) -> None:
    kept = {pair: row for pair, row in INCOMPATIBLE.items() if pair != DECLARED[0]}
    matrix = _matrix(composability_harness, {**kept, **mutation})

    assert _failed_pairs(matrix, DECLARED) == (DECLARED[0],)


def test_the_readme_quotes_the_generated_count(
    composability_harness: HarnessFactory,
    capsys: pytest.CaptureFixture[str],
) -> None:
    outcomes = _outcomes(_matrix(composability_harness))
    counts = count_outcomes(outcomes)
    line = render_report(counts)
    readme = README.read_text(encoding="utf-8")

    assert counts.total == len(PAIRS)
    assert counts.constructed == len(COMPATIBLE)
    assert counts.declared_incompatible == len(DECLARED)
    assert line in readme
    assert all(reason in readme for _, reason in INCOMPATIBLE.values())
    assert main(outcomes) == 0
    assert capsys.readouterr().out == f"{line}\n"


PACKAGED_RUN: Final = (
    Path(__file__).resolve().parents[2] / "lofi/configs/run/synthetic-parser.yaml"
)


def _lofi_run(root: Path, /) -> tuple[int, str, str]:
    """Run ``lofi run`` on the packaged parser run and return its status and output."""
    stdout = io.StringIO()
    stderr = io.StringIO()
    status = cli_main(
        [
            "run",
            "--config",
            str(PACKAGED_RUN),
            "--experiment-seed",
            "7",
            "--root",
            str(root),
        ],
        training=TRAINING_OPERATIONS,
        environ={},
        stdout=stdout,
        stderr=stderr,
    )

    return status, stdout.getvalue(), stderr.getvalue()


def test_lofi_run_wires_its_graph_through_the_function_the_matrix_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wired: list[tuple[str, ...]] = []

    def observed(
        model: ModelSection,
        run: BuildContext,
        encoders: Mapping[str, CodecEncoder],
        /,
        *,
        parser: object | None,
        pretrained_resources: Mapping[str, PretrainedLoader],
        role_targets: Mapping[str, Mapping[str, TargetSet[Term]]] | None = None,
        erasure: TermMorphism | None,
    ) -> SpaceGraph:
        graph = assemble_space_graph(
            model,
            run,
            encoders,
            parser=parser,
            pretrained_resources=pretrained_resources,
            role_targets=role_targets,
            erasure=erasure,
        )
        wired.append(tuple(graph.components))

        return graph

    monkeypatch.setattr(assembly_module, "assemble_space_graph", observed)
    point = SweepPoint(
        resolved_document(
            compose(
                Selection(
                    config=PACKAGED_RUN,
                    profile=None,
                    overrides=(),
                    overlays=(),
                    search_path=config_search_path(tmp_path),
                )
            )
        )
    )
    directory = RunStore(tmp_path, freeze_id=None, seed=ExperimentSeed(7)).directory(
        point
    )

    status, stdout, stderr = _lofi_run(tmp_path)

    assert status == 0, stderr
    assert stdout == f"{point.point_id} completed {directory}\n"
    assert wired == [
        (
            "terms",
            "identity",
            "application",
            "category-terms",
            "types",
        )
    ]

    status, stdout, stderr = _lofi_run(tmp_path)

    assert status == 0, stderr
    assert stdout == f"{point.point_id} completed {directory}\n"

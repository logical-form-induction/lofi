"""Benchmark every shipped structural encoder and decoder declaration.

One deterministic term batch is encoded by each named term encoder and scored by
each named term decoder. The committed measurement records mean seconds per batch
and the peak tensor bytes allocated by the same callable.
"""

import json
import math
import os
import platform
import subprocess
from functools import partial
from hashlib import sha256
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple, cast

import pytest
import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile

from lofi._types import ComponentId, FrozenJsonValue, JsonValue
from lofi.judge import Action, ActionSet, NoNormalForm, RuleSet
from lofi.nn.build import BuildContext
from lofi.nn.codec import NeuralCodecView
from lofi.nn.policy import DevicePolicy
from lofi.nn.strategies import score
from lofi.nn.tokens.embedder import ReservedTokens, TokenVocabulary
from lofi.spec.language import LanguageCodec, SortBound
from lofi.terms import (
    CompleteTree,
    Identity,
    Operator,
    OperatorIndex,
    Signature,
    TermCodec,
    TermStore,
    Vocabulary,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import didactic.api as dx
    from pytest_benchmark.fixture import BenchmarkFixture

    from lofi.lf import SurfaceSyntax
    from lofi.nn._values import Encoded, LogDistribution
    from lofi.nn.ports import TeacherForcedDecoder, TermEncoder
    from lofi.registry import ComponentPlugin
    from lofi.spec import ConfiguredSpec
    from lofi.spec.language import LanguageAssembly
    from lofi.terms import BatchEncoding


WIDTH: Final = 32
HEADS: Final = 4
LAYERS: Final = 1
FEEDFORWARD_WIDTH: Final = 64
BATCH_SIZE: Final = 32
SEED: Final = 17
ROUNDS: Final = 10
WARMUP_ROUNDS: Final = 2
ITERATIONS: Final = 1

FIXTURE_PATH: Final = Path(__file__).parents[1] / "fixtures" / "nn_structure.json"
PACKAGE: Final = "lofi"
ENCODER_FAMILY: Final = "term_encoder"
DECODER_FAMILY: Final = "term_decoder"
GROUPS: Final[Mapping[str, str]] = MappingProxyType(
    {ENCODER_FAMILY: "lofi.term_encoder", DECODER_FAMILY: "lofi.term_decoder"}
)
"""The entry-point group each measured family's plugins are listed under."""

ENCODER_SETTINGS: Final[Mapping[str, Mapping[str, FrozenJsonValue]]] = MappingProxyType(
    {
        "hrr": MappingProxyType({"width": WIDTH, "magnitude": 1.0, "floor": 1e-3}),
        "linear-rnn": MappingProxyType(
            {
                "width": WIDTH,
                "cell": "gru",
                "layers": LAYERS,
                "direction": "forward",
                "variables": "index",
            }
        ),
        "tree-rnn": MappingProxyType(
            {"width": WIDTH, "layers": LAYERS, "norm_cap": None}
        ),
        "transformer": MappingProxyType(
            {
                "width": WIDTH,
                "heads": HEADS,
                "layers": LAYERS,
                "feedforward_width": FEEDFORWARD_WIDTH,
                "structure": "ancestor-mask",
                "norm_cap": None,
            }
        ),
    }
)

DECODER_SETTINGS: Final[Mapping[str, Mapping[str, FrozenJsonValue]]] = MappingProxyType(
    {
        "diffusion": MappingProxyType(
            {
                "width": WIDTH,
                "layers": LAYERS,
                "feedforward_width": FEEDFORWARD_WIDTH,
                "noise_steps": 4,
                "first_masking": 0.25,
            }
        ),
        "hrr": MappingProxyType({}),
        "linear-rnn": MappingProxyType(
            {
                "width": WIDTH,
                "cell": "gru",
                "layers": LAYERS,
                "choice_heads": "flat",
                "tie": None,
            }
        ),
        "tree-rnn": MappingProxyType({"width": WIDTH}),
        "transformer": MappingProxyType(
            {
                "width": WIDTH,
                "heads": HEADS,
                "layers": LAYERS,
                "feedforward_width": FEEDFORWARD_WIDTH,
                "structure": "parent-recurrence",
            }
        ),
    }
)

DECODER_ENCODERS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "diffusion": "tree-rnn",
        "hrr": "hrr",
        "linear-rnn": "linear-rnn",
        "tree-rnn": "tree-rnn",
        "transformer": "transformer",
    }
)

UNARY: Final = Operator("raise", "T", arg_sorts=("T",), family="constructor")
BINARY: Final = Operator("pair", "T", arg_sorts=("T", "T"), family="constructor")
SIGNATURE: Final = Signature(
    "structure-benchmark",
    ("T",),
    (UNARY, BINARY),
    vocabularies=(Vocabulary("T", ("free", "other")),),
)
CODEC: Final = TermCodec(
    OperatorIndex(SIGNATURE),
    CompleteTree(SIGNATURE, SIGNATURE.operators, order="bf", depth=3),
)
VIEW: Final = NeuralCodecView(CODEC)


class _Language(NamedTuple):
    """The benchmark fixture's learner-language capability."""

    codec: LanguageCodec
    store: TermStore
    syntax: SurfaceSyntax


STORE: Final = TermStore(SIGNATURE)
LANGUAGE: Final = _Language(
    LanguageCodec(
        "structure-benchmark",
        SIGNATURE,
        {
            "T": SortBound(
                "complete-tree", depth=3, nodes=None, order="bf", free_ranks=0
            )
        },
    ),
    STORE,
    cast("SurfaceSyntax", object()),
)
POLICY: Final = DevicePolicy(torch.device("cpu"), torch.float32)
IDENTITY: Final = Action("identity")
ACTIONS: Final = ActionSet((IDENTITY,), IDENTITY)
RULES: Final = RuleSet("structure-benchmark", (), Identity(), NoNormalForm())
CATEGORIES: Final = (STORE.const("T", "free"), STORE.const("T", "other"))
VOCABULARY: Final = TokenVocabulary(
    ("one", "two"),
    reserved=ReservedTokens({"<pad>": 0, "<unk>": 1}),
    unknown="<unk>",
)


def _terms() -> BatchEncoding:
    """Return the deterministic mixed-shape batch every declaration receives."""
    atom = STORE.const("T", "free")
    other = STORE.const("T", "other")
    choices = (
        STORE.op(BINARY, STORE.op(UNARY, atom), other),
        STORE.op(BINARY, other, atom),
        STORE.op(UNARY, STORE.op(BINARY, atom, other)),
        STORE.op(BINARY, STORE.op(BINARY, atom, other), STORE.op(UNARY, atom)),
    )

    return CODEC.encode_batch(
        tuple(choices[index % len(choices)] for index in range(BATCH_SIZE))
    )


TERMS: Final = _terms()


class Declaration(NamedTuple):
    """One entry point shipped by this distribution."""

    family: str
    tag: str
    point: EntryPoint


class Case(NamedTuple):
    """One named declaration and the exact batch operation timed for it."""

    family: str
    declaration: str
    operation: Callable[[], Encoded | LogDistribution]
    axes: tuple[str, ...]

    @property
    def identity(self) -> tuple[str, str]:
        """Return the declaration's family and tag."""
        return (self.family, self.declaration)


class Measurement(NamedTuple):
    """One declaration's measured batch time and peak tensor allocation."""

    family: str
    declaration: str
    seconds_per_batch: float
    peak_bytes: int


def shipping_declarations() -> tuple[Declaration, ...]:
    """Return this distribution's encoder and decoder entry points in stable order."""
    found = (
        Declaration(family, point.name, point)
        for family, group in GROUPS.items()
        for point in entry_points(group=group)
        if point.dist is not None and point.dist.name == PACKAGE
    )

    return tuple(sorted(found, key=lambda declaration: declaration[:2]))


def _build(
    declaration: Declaration,
    settings: Mapping[str, FrozenJsonValue],
    dependencies: Mapping[str, nn.Module],
) -> nn.Module:
    """Construct one declaration through its installed production plugin."""
    plugin = cast(
        "ComponentPlugin[object, BuildContext, nn.Module]", declaration.point.load()
    )
    spec = cast(
        "ConfiguredSpec[object]",
        cast("type[dx.Model]", plugin.spec).model_validate(
            {"kind": declaration.tag, **settings}
        ),
    )
    context = BuildContext(
        component=ComponentId(declaration.tag),
        policy=POLICY,
        language=cast("LanguageAssembly", LANGUAGE),
        codec=VIEW,
        vocabulary=VOCABULARY,
        actions=ACTIONS,
        rules=RULES,
        categories=CATEGORIES,
        producers=dependencies,
        targets={},
        loader=None,
    )

    return plugin.construct(spec, context).eval()


def _encode(encoder: TermEncoder) -> Encoded:
    """Encode the pinned batch without constructing an autograd graph."""
    with torch.no_grad():
        return encoder.encode(TERMS)


def _score(decoder: TeacherForcedDecoder, encoded: Encoded) -> LogDistribution:
    """Score the pinned batch without constructing an autograd graph."""
    with torch.no_grad():
        return score(decoder, encoded, TERMS, index=LANGUAGE.codec.choices)


def cases() -> tuple[Case, ...]:
    """Build one executable case per shipped declaration."""
    torch.manual_seed(SEED)
    declarations = shipping_declarations()
    by_identity = {
        (declaration.family, declaration.tag): declaration
        for declaration in declarations
    }
    encoders = {
        tag: cast(
            "TermEncoder",
            _build(
                by_identity[ENCODER_FAMILY, tag],
                settings,
                MappingProxyType({}),
            ),
        )
        for tag, settings in ENCODER_SETTINGS.items()
    }
    encoded = {tag: _encode(encoder) for tag, encoder in encoders.items()}
    built: list[Case] = [
        Case(
            ENCODER_FAMILY,
            tag,
            partial(_encode, encoder),
            ("term", "width"),
        )
        for tag, encoder in encoders.items()
    ]

    for tag, settings in DECODER_SETTINGS.items():
        encoder_tag = DECODER_ENCODERS[tag]
        decoder = cast(
            "TeacherForcedDecoder",
            _build(
                by_identity[DECODER_FAMILY, tag],
                settings,
                MappingProxyType({"encoder": cast("nn.Module", encoders[encoder_tag])}),
            ),
        )
        built.append(
            Case(
                DECODER_FAMILY,
                tag,
                partial(_score, decoder, encoded[encoder_tag]),
                ("term",),
            )
        )

    return tuple(sorted(built, key=lambda case: case.identity))


@pytest.fixture(scope="module")
def structure_cases() -> tuple[Case, ...]:
    """Return the built declaration cases once for this module."""
    return cases()


def fixture_payload() -> dict[str, JsonValue]:
    """Return the canonical JSON payload whose digest pins the measured fixture."""
    return cast(
        "dict[str, JsonValue]",
        {
            "seed": SEED,
            "batch": {
                "symbols": [list(row) for row in TERMS.symbols],
                "parents": [list(row) for row in TERMS.parents],
                "depths": [list(row) for row in TERMS.depths],
                "mask": [list(row) for row in TERMS.mask],
                "payloads": (
                    None
                    if TERMS.payloads is None
                    else [list(row) for row in TERMS.payloads]
                ),
            },
            "encoder_settings": {
                tag: dict(settings) for tag, settings in ENCODER_SETTINGS.items()
            },
            "decoder_settings": {
                tag: dict(settings) for tag, settings in DECODER_SETTINGS.items()
            },
            "decoder_encoders": dict(DECODER_ENCODERS),
        },
    )


def fixture_digest() -> str:
    """Return the stable digest of the batch and declaration settings."""
    encoded = json.dumps(
        fixture_payload(), ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )

    return sha256(encoded.encode()).hexdigest()


def _hardware_profile() -> dict[str, str | int]:
    """Return the nonsecret hardware identity of the machine running a measurement."""
    identity: dict[str, str | int] = {
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cores": os.cpu_count() or 0,
    }

    if platform.system() != "Darwin":
        return identity

    completed = subprocess.run(
        ("/usr/sbin/system_profiler", "SPHardwareDataType", "-json"),
        check=True,
        capture_output=True,
        encoding="utf-8",
    )
    report = cast("dict[str, list[dict[str, str]]]", json.loads(completed.stdout))
    hardware = report["SPHardwareDataType"][0]
    identity.update(
        {
            "model": hardware["machine_model"],
            "chip": hardware["chip_type"],
            "cores": hardware["number_processors"],
            "memory": hardware["physical_memory"],
        }
    )

    return identity


def load_fixture() -> dict[str, JsonValue]:
    """Return the committed measurement fixture."""
    return cast(
        "dict[str, JsonValue]", json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    )


def measurements() -> tuple[Measurement, ...]:
    """Construct the typed measurement rows recorded by the fixture."""
    rows = cast("list[dict[str, JsonValue]]", load_fixture()["measurements"])

    for row in rows:
        if set(row) != {
            "family",
            "declaration",
            "seconds_per_batch",
            "peak_bytes",
        }:
            message = "each neural structure measurement must name four fields"
            raise ValueError(message)

    return tuple(
        Measurement(
            cast("str", row["family"]),
            cast("str", row["declaration"]),
            cast("float", row["seconds_per_batch"]),
            cast("int", row["peak_bytes"]),
        )
        for row in rows
    )


def _peak_bytes(operation: Callable[[], Encoded | LogDistribution]) -> int:
    """Return peak tensor bytes allocated while one batch operation runs."""
    with profile(activities=(ProfilerActivity.CPU,), profile_memory=True) as measured:
        result = operation()
        del result

    current = 0
    peak = 0

    for event in sorted(measured.events(), key=lambda held: held.time_range.start):
        current += event.self_cpu_memory_usage
        peak = max(peak, current)

    return peak


def test_the_measurement_rows_cover_the_shipping_inventory_exactly_once(
    structure_cases: tuple[Case, ...],
) -> None:
    declarations = shipping_declarations()
    expected = tuple(declaration[:2] for declaration in declarations)
    built = tuple(case.identity for case in structure_cases)
    rows = measurements()
    measured = tuple(row[:2] for row in rows)

    assert len(expected) == len(set(expected))
    assert built == expected
    assert measured == expected
    assert len(measured) == len(set(measured))
    assert set(ENCODER_SETTINGS) == {
        tag for family, tag in expected if family == ENCODER_FAMILY
    }
    assert set(DECODER_SETTINGS) == {
        tag for family, tag in expected if family == DECODER_FAMILY
    }
    assert set(DECODER_ENCODERS) == set(DECODER_SETTINGS)


def test_the_committed_fixture_names_its_digest_hardware_and_real_measurements() -> (
    None
):
    fixture = load_fixture()
    hardware = cast("dict[str, JsonValue]", fixture["hardware"])
    runtime = cast("dict[str, JsonValue]", fixture["runtime"])
    rows = measurements()

    assert fixture["fixture_sha256"] == fixture_digest()
    assert set(hardware) == {
        "system",
        "machine",
        "processor",
        "logical_cores",
        "model",
        "chip",
        "cores",
        "memory",
    }
    assert all(value != "" for value in hardware.values())
    assert set(runtime) == {"python", "torch", "device", "dtype"}
    assert isinstance(runtime["python"], str)
    assert isinstance(runtime["torch"], str)
    assert runtime["python"]
    assert runtime["torch"]
    assert runtime["device"] == "cpu"
    assert runtime["dtype"] == "float32"

    for row in rows:
        assert isinstance(row.family, str)
        assert isinstance(row.declaration, str)
        assert isinstance(row.seconds_per_batch, float)
        assert math.isfinite(row.seconds_per_batch)
        assert row.seconds_per_batch > 0.0
        assert isinstance(row.peak_bytes, int)
        assert not isinstance(row.peak_bytes, bool)
        assert row.peak_bytes > 0


def test_every_untimed_case_executes_the_same_callable_the_benchmark_times(
    structure_cases: tuple[Case, ...],
) -> None:
    for case in structure_cases:
        result = case.operation()

        assert result.axes == case.axes
        assert result.tensor.shape[0] == BATCH_SIZE
        assert bool(torch.isfinite(result.tensor).all())


@pytest.mark.benchmark(group="named-structure-declarations")
@pytest.mark.parametrize(
    "identity",
    tuple(declaration[:2] for declaration in shipping_declarations()),
    ids=tuple("/".join(declaration[:2]) for declaration in shipping_declarations()),
)
def test_declaration_seconds_per_batch_and_peak_memory(
    benchmark: BenchmarkFixture,
    structure_cases: tuple[Case, ...],
    identity: tuple[str, str],
) -> None:
    case = next(held for held in structure_cases if held.identity == identity)
    peak = _peak_bytes(case.operation)
    benchmark.pedantic(
        case.operation,
        rounds=ROUNDS,
        warmup_rounds=WARMUP_ROUNDS,
        iterations=ITERATIONS,
    )
    measured = benchmark.stats

    if measured is None:
        pytest.fail("the declaration ran but pytest-benchmark timed nothing")

    benchmark.extra_info["family"] = case.family
    benchmark.extra_info["declaration"] = case.declaration
    benchmark.extra_info["fixture_sha256"] = fixture_digest()
    benchmark.extra_info["hardware"] = _hardware_profile()
    benchmark.extra_info["runtime"] = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "device": "cpu",
        "dtype": "float32",
    }
    benchmark.extra_info["seconds_per_batch"] = float(measured.stats.mean)
    benchmark.extra_info["peak_bytes"] = peak

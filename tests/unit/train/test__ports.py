import annotationlib
from typing import TYPE_CHECKING, Final, NamedTuple, override

import pytest
import torch
from tests.conformance.test_checkpoint_store import MemoryStore, NullStore
from tests.conformance.test_optimization_policy import fixture_for
from tests.conformance.test_train_step import Declared, Scoring

from lofi.deduce import SpanIndex
from lofi.nn._values import LogDistribution
from lofi.nn.loss._parser import SPAN_AXES
from lofi.nn.loss._values import (
    CellChart,
    ObjectiveOutputs,
)
from lofi.run import RandomStreams, SeedPolicy
from lofi.train._ports import (
    CheckpointStore,
    EvalStep,
    Forward,
    OptimizationPolicy,
    RestorableRandomStreams,
    StepDeclaration,
    TrainStep,
    ValueCodec,
)

if TYPE_CHECKING:
    from types import FunctionType

    from lofi._types import JsonValue
    from lofi.nn.loss._ports import PublishedOutputs

MAPPINGS: Final = ("dict", "Mapping", "MutableMapping", "Any", "object")


def annotations_of(function: FunctionType) -> dict[str, str]:
    return annotationlib.get_annotations(function, format=annotationlib.Format.STRING)


def test_a_training_step_takes_and_returns_declared_values_rather_than_mappings() -> (
    None
):
    written = annotations_of(TrainStep.run)

    assert set(written) == {"inputs", "self", "return"} - {"self"} | {"inputs"}
    assert written["return"] == "StepOutcome[OutputsT, TrainResult]"
    assert not any(mapping in written["inputs"] for mapping in MAPPINGS)
    assert not any(mapping in written["return"] for mapping in MAPPINGS)


def test_an_evaluation_step_reports_an_evaluation_result() -> None:
    written = annotations_of(EvalStep.run)

    assert written["return"] == "StepOutcome[OutputsT, EvalResult]"
    assert not any(mapping in written["inputs"] for mapping in MAPPINGS)


def test_a_store_takes_a_manifest_and_state_keyed_by_the_role() -> None:
    written = annotations_of(CheckpointStore.save)

    assert written["manifest"] == "CheckpointManifest"
    assert written["states"] == "Mapping[str, StateT]"
    assert written["return"] == "CheckpointId"


def test_a_store_compares_the_manifest_it_is_given_against_what_it_holds() -> None:
    written = annotations_of(CheckpointStore.load)

    assert written["expected"] == "CheckpointManifest"
    assert written["return"] == "Checkpoint[StateT]"


def test_a_store_lists_the_handles_it_holds() -> None:
    assert annotations_of(CheckpointStore.list)["return"] == "tuple[CheckpointId, ...]"


def test_random_streams_publish_policy_seed_generator_snapshot_and_restore() -> None:
    assert set(annotations_of(RestorableRandomStreams.sampler_seed)) == {
        "name",
        "return",
    }
    assert (
        annotations_of(RestorableRandomStreams.loader_generator)["return"]
        == "Generator"
    )
    assert (
        annotations_of(RestorableRandomStreams.snapshot)["return"]
        == "Mapping[str, str]"
    )
    assert (
        annotations_of(RestorableRandomStreams.validate)["streams"]
        == "Mapping[str, str]"
    )
    prepared = annotations_of(RestorableRandomStreams.prepare_restore)
    assert prepared["streams"] == "Mapping[str, str]"
    assert prepared["return"] == "Callable[[], None]"
    assert (
        annotations_of(RestorableRandomStreams.restore)["streams"]
        == "Mapping[str, str]"
    )


def test_the_run_implementation_satisfies_the_consumer_owned_stream_port() -> None:
    streams = RandomStreams(SeedPolicy(3, loaders=("fit",)))

    assert isinstance(streams, RestorableRandomStreams)


def test_a_step_declaration_publishes_a_name_and_two_name_sets() -> None:
    scoring = Scoring()

    assert isinstance(scoring, StepDeclaration)
    assert scoring.name == "scoring"
    assert scoring.requires == frozenset({"examples", "weights"})
    assert scoring.produces == frozenset({"scores"})


def test_a_step_that_runs_is_a_declaration_as_well() -> None:
    assert isinstance(Scoring(), TrainStep)


def test_a_declaration_that_does_not_run_is_no_training_step() -> None:
    declaration = Declared("featurizing", frozenset(), frozenset({"weights"}))

    assert isinstance(declaration, StepDeclaration)
    assert not isinstance(declaration, TrainStep)


@pytest.mark.parametrize("store", [MemoryStore(), NullStore()])
def test_a_store_offering_the_three_calls_satisfies_the_port(
    store: MemoryStore | NullStore,
) -> None:
    assert isinstance(store, CheckpointStore)


def test_a_store_missing_one_of_the_three_calls_does_not_satisfy_the_port() -> None:
    class Partial:
        def save(self) -> None: ...

    assert not isinstance(Partial(), CheckpointStore)


def test_an_optimization_policy_owns_one_complete_call() -> None:
    written = annotations_of(OptimizationPolicy.step)

    assert written["forward"] == "Callable[[], LossT]"
    assert written["return"] == "OptimizationResult"


def test_a_policy_that_steps_satisfies_the_optimization_port() -> None:
    policy = fixture_for().build(amp=True, accumulation_steps=1).policy

    assert isinstance(policy, OptimizationPolicy)


def test_a_value_without_a_step_is_no_optimization_policy() -> None:
    assert not isinstance(object(), OptimizationPolicy)


class Words(NamedTuple):
    words: tuple[tuple[str, ...], ...]


class Chartless:
    @property
    def publishes(self) -> frozenset[str]:
        return frozenset({"spans", "chart", "batch"})

    def publish(self, batch: Words, /) -> ObjectiveOutputs[Words]:
        inside = LogDistribution(torch.zeros(len(batch.words), 1), SPAN_AXES)

        return ObjectiveOutputs(
            spans=SpanIndex(1), chart=CellChart(inside), batch=batch
        )


class Unpublished:
    @property
    def publishes(self) -> frozenset[str]:
        return frozenset()


def test_a_forward_publishes_a_bundle_over_a_batch() -> None:
    forward: Forward[Words] = Chartless()
    batch = Words((("a",), ("b",)))

    bundle: PublishedOutputs[Words] = forward.publish(batch)

    assert isinstance(forward, Forward)
    assert not isinstance(Unpublished(), Forward)
    assert bundle.batch == batch
    assert set(forward.publishes) == {"spans", "chart", "batch"}
    assert bundle.chart.inside.tensor.shape == (2, 1)


class Label:
    __slots__ = ("_text",)

    def __init__(self, text: str) -> None:
        self._text = text

    @property
    def text(self) -> str:
        return self._text

    @override
    def __eq__(self, other: object) -> bool:
        if type(other) is not Label:
            return NotImplemented

        return self._text == other._text

    @override
    def __hash__(self) -> int:
        return hash((Label, self._text))


class LabelCodec:
    def encode(self, value: Label, /) -> JsonValue:
        return {"text": value.text}

    def decode(self, data: JsonValue, /) -> Label:
        if not isinstance(data, dict) or not isinstance(data.get("text"), str):
            msg = f"not a label: {data!r}"

            raise TypeError(msg)

        return Label(str(data["text"]))


class Encoder:
    def encode(self, value: Label, /) -> JsonValue:
        return value.text


def test_a_codec_publishes_both_directions() -> None:
    assert isinstance(LabelCodec(), ValueCodec)
    assert not isinstance(Encoder(), ValueCodec)


def test_decoding_an_encoding_returns_an_equal_value() -> None:
    codec: ValueCodec[Label] = LabelCodec()
    label = Label("subject")

    assert codec.decode(codec.encode(label)) == label
    assert codec.encode(label) == codec.encode(Label("subject"))

    with pytest.raises(TypeError, match="not a label"):
        codec.decode(["subject"])

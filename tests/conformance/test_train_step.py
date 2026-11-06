from typing import TYPE_CHECKING, Final, override

import pytest
from tests.conformance.train_step import (
    TRAIN_STEP_CONFORMANCE_LAWS,
    TrainStepConformance,
)

from lofi._checks import LawViolationError
from lofi.train import StepOutcome, TrainResult
from lofi.train.laws import TRAIN_STEP_LAWS, StepFixture

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.train import StepDeclaration

STEP: Final = 12

AVAILABLE: Final = frozenset({"examples", "labels"})


class Batch:
    __slots__ = ("_examples", "_labels", "_seen", "_weights")

    def __init__(
        self,
        examples: tuple[str, ...],
        labels: tuple[int, ...],
        weights: tuple[float, ...],
    ) -> None:
        self._examples = examples
        self._labels = labels
        self._weights = weights
        self._seen: set[str] = set()

    @property
    def examples(self) -> tuple[str, ...]:
        self._seen.add("examples")

        return self._examples

    @property
    def labels(self) -> tuple[int, ...]:
        self._seen.add("labels")

        return self._labels

    @property
    def weights(self) -> tuple[float, ...]:
        self._seen.add("weights")

        return self._weights

    @property
    def seen(self) -> frozenset[str]:
        return frozenset(self._seen)


class Scores:
    __slots__ = ("_values",)

    def __init__(self, values: tuple[float, ...]) -> None:
        self._values = values

    @property
    def values(self) -> tuple[float, ...]:
        return self._values


class Declared:
    __slots__ = ("_name", "_produces", "_requires")

    def __init__(
        self,
        name: str,
        requires: frozenset[str],
        produces: frozenset[str],
    ) -> None:
        self._name = name
        self._requires = requires
        self._produces = produces

    @property
    def name(self) -> str:
        return self._name

    @property
    def requires(self) -> frozenset[str]:
        return self._requires

    @property
    def produces(self) -> frozenset[str]:
        return self._produces


FEATURIZING: Final = Declared(
    "featurizing", frozenset({"examples"}), frozenset({"weights"})
)

REPORTING: Final = Declared("reporting", frozenset({"scores"}), frozenset({"report"}))


class Scoring:
    __slots__ = ("temperature",)

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = temperature

    @property
    def name(self) -> str:
        return "scoring"

    @property
    def requires(self) -> frozenset[str]:
        return frozenset({"examples", "weights"})

    @property
    def produces(self) -> frozenset[str]:
        return frozenset({"scores"})

    def scored(self, inputs: Batch) -> tuple[tuple[float, ...], int]:
        weights = inputs.weights
        values: list[float] = []
        dropped = 0

        for position, example in enumerate(inputs.examples):
            if not example:
                dropped += 1

                continue

            values.append(len(example) * weights[position] / self.temperature)

        return tuple(values), dropped

    def run(self, inputs: Batch, /) -> StepOutcome[Scores, TrainResult]:
        values, dropped = self.scored(inputs)
        mean = sum(values) / len(values) if values else 0.0

        return StepOutcome(
            Scores(values),
            TrainResult(STEP, len(values), dropped, {"score.mean": mean}),
        )


class Moded(Scoring):
    __slots__ = ("verbose",)

    def __init__(self, *, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose

    @override
    def run(self, inputs: Batch, /) -> StepOutcome[Scores, TrainResult]:
        outcome = super().run(inputs)
        reported = dict(outcome.result.metrics)

        if self.verbose:
            reported["score.count"] = float(outcome.result.items)

        return StepOutcome(
            outcome.outputs,
            TrainResult(
                outcome.result.step,
                outcome.result.items,
                outcome.result.dropped,
                reported,
            ),
        )


class Idle(Scoring):
    __slots__ = ()

    @property
    @override
    def requires(self) -> frozenset[str]:
        return frozenset({"examples", "weights", "labels"})


class Shouting(Scoring):
    __slots__ = ()

    @override
    def run(self, inputs: Batch, /) -> StepOutcome[Scores, TrainResult]:
        values, _ = self.scored(inputs)

        if not values:
            msg = "the batch holds no sentence to score"

            raise ValueError(msg)

        return super().run(inputs)


class Peeking(Scoring):
    __slots__ = ()

    @override
    def run(self, inputs: Batch, /) -> StepOutcome[Scores, TrainResult]:
        outcome = super().run(inputs)
        offset = float(sum(inputs.labels))
        reported = {"score.mean": outcome.result.metrics["score.mean"] + offset}

        return StepOutcome(
            outcome.outputs,
            TrainResult(
                outcome.result.step,
                outcome.result.items,
                outcome.result.dropped,
                reported,
            ),
        )


def batch() -> Batch:
    return Batch(("a sentence", "", "another"), (1, 0, 1), (0.5, 0.25, 2.0))


def empty() -> Batch:
    return Batch((), (), ())


def graph_for(step: Scoring) -> Sequence[StepDeclaration]:
    return (FEATURIZING, step, REPORTING)


def fixture_for(
    step: Scoring,
    variants: Sequence[Scoring] | None = None,
) -> StepFixture[Batch, Scores]:
    built = (type(step)(), type(step)()) if variants is None else tuple(variants)

    return StepFixture(
        step=step,
        inputs=batch,
        reads=lambda inputs: inputs.seen,
        idle=empty,
        variants=built,
        graph=graph_for(step),
        available=AVAILABLE,
    )


class TestAStepThatReadsWhatItDeclared(TrainStepConformance):
    @pytest.fixture
    def conformance(self) -> StepFixture[Batch, Scores]:
        return fixture_for(Scoring(), (Scoring(1.0), Scoring(4.0)))


class TestAStepWhoseReportBranchesOnAFlag(TrainStepConformance):
    @pytest.fixture
    def conformance(self) -> StepFixture[Batch, Scores]:
        return fixture_for(
            Moded(verbose=True), (Moded(verbose=False), Moded(verbose=True))
        )

    @pytest.mark.xfail(strict=True, reason="the reported names follow the flag")
    @override
    def test_the_output_schema_does_not_depend_on_a_setting[InputsT, OutputsT](
        self,
        conformance: StepFixture[InputsT, OutputsT],
    ) -> None:
        super().test_the_output_schema_does_not_depend_on_a_setting(conformance)


class TestAStepDeclaringAValueItNeverReads(TrainStepConformance):
    @pytest.fixture
    def conformance(self) -> StepFixture[Batch, Scores]:
        return fixture_for(Idle())

    @pytest.mark.xfail(strict=True, reason="the labels are declared and never read")
    @override
    def test_every_declared_parameter_is_consumed[InputsT, OutputsT](
        self,
        conformance: StepFixture[InputsT, OutputsT],
    ) -> None:
        super().test_every_declared_parameter_is_consumed(conformance)


class TestAStepThatRaisesOnAnEmptyBatch(TrainStepConformance):
    @pytest.fixture
    def conformance(self) -> StepFixture[Batch, Scores]:
        return fixture_for(Shouting())

    @pytest.mark.xfail(strict=True, reason="an empty batch is reported by raising")
    @override
    def test_nothing_raises_to_report_success[InputsT, OutputsT](
        self,
        conformance: StepFixture[InputsT, OutputsT],
    ) -> None:
        super().test_nothing_raises_to_report_success(conformance)


class TestAStepReadingAValueItNeverDeclared(TrainStepConformance):
    @pytest.fixture
    def conformance(self) -> StepFixture[Batch, Scores]:
        return fixture_for(Peeking())

    @pytest.mark.xfail(strict=True, reason="the labels reach the step on no edge")
    @override
    def test_the_edges_are_exactly_the_data_dependencies[InputsT, OutputsT](
        self,
        conformance: StepFixture[InputsT, OutputsT],
    ) -> None:
        super().test_the_edges_are_exactly_the_data_dependencies(conformance)


def test_the_suite_binds_a_method_to_every_training_step_law() -> None:
    bound = {
        method.__name__.removeprefix("test_") for method in TRAIN_STEP_CONFORMANCE_LAWS
    }

    assert bound == {law.__name__ for law in TRAIN_STEP_LAWS}


def test_the_schema_law_refuses_a_fixture_naming_one_variant() -> None:
    single = fixture_for(Scoring(), (Scoring(),))

    with pytest.raises(LawViolationError, match="needs two"):
        TrainStepConformance().test_the_output_schema_does_not_depend_on_a_setting(
            single
        )


def test_a_dropped_sentence_is_counted_rather_than_silent() -> None:
    outcome = Scoring().run(batch())

    assert outcome.result.dropped == 1
    assert outcome.result.items == 2
    assert len(outcome.outputs.values) == 2

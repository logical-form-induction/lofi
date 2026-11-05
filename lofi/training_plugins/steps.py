"""Installed training-task declaration plugins."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol, cast, runtime_checkable

from lofi._frozen import FrozenValue
from lofi.data import Relational, Unlabeled
from lofi.registry import resolve_plugin
from lofi.spec.trainers.steps import (
    AlternatingStepSpec,
    ControllerStepSpec,
    GroundedStepSpec,
    JudgmentStepSpec,
    ParserStepSpec,
    StepSpec,
    SupervisedWarmStartStepSpec,
    TermsStepSpec,
    TokenWarmupStepSpec,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from lofi.data import SupervisionArm

__all__ = [
    "FittingOutcome",
    "FittingTask",
    "OptimizationOutcome",
    "StepBuildContext",
    "StepPlugin",
    "merge_presentations",
    "step_plugin",
    "steps_presentation",
]


class OptimizationOutcome(Protocol):
    """The optimizer facts a staged or alternating fit consumes."""

    @property
    def loss_finite(self) -> bool:
        """Return whether the optimized loss was finite."""

    @property
    def stepped(self) -> bool:
        """Return whether the optimizer took a step."""


class FittingOutcome(Protocol):
    """What one fitted batch reports to a composed trainer."""

    @property
    def result(self) -> OptimizationOutcome | None:
        """Return the optimizer outcome, or None when no item was fitted."""

    @property
    def items(self) -> int:
        """Return how many items the fitted batch held."""

    @property
    def metrics(self) -> Mapping[str, float]:
        """Return the batch metrics under their declared names."""


@runtime_checkable
class FittingTask(Protocol):
    """One named dataset and the operation that fits one of its batches."""

    @property
    def name(self) -> str:
        """Return what the task is called."""

    @property
    def batches(self) -> int:
        """Return how many batches one pass over the task holds."""

    @property
    def optimizer_steps(self) -> int:
        """Return how many optimizer steps one complete pass takes."""

    def fit(self, position: int, step: int, /) -> FittingOutcome:
        """Fit the batch at ``position`` under the current optimizer step."""


class StepBuildContext(Protocol):
    """Build each declared task through its concrete training semantics."""

    def parser(self, declaration: ParserStepSpec, /) -> FittingTask:
        """Build parser loss over sentence batches."""

    def supervised_warm_start(
        self, declaration: SupervisedWarmStartStepSpec, /
    ) -> FittingTask:
        """Build gold-form supervision over sentence batches."""

    def judgment(self, declaration: JudgmentStepSpec, /) -> FittingTask:
        """Build judgment-response grounding over inference pairs."""

    def grounded(self, declaration: GroundedStepSpec, /) -> FittingTask:
        """Build grounded inference over labeled pairs."""

    def alternating(self, declaration: AlternatingStepSpec, /) -> FittingTask:
        """Build one nested alternating task."""

    def token_warmup(self, declaration: TokenWarmupStepSpec, /) -> FittingTask:
        """Build lexical token warm-up over utterance batches."""

    def terms(self, declaration: TermsStepSpec, /) -> FittingTask:
        """Build generated-term autoencoding and combinator decoding."""

    def controller(self, declaration: ControllerStepSpec, /) -> FittingTask:
        """Build labeled parser-controller supervision."""


class StepPlugin[S: StepSpec](FrozenValue):
    """Publish one exact declaration, the arms it presents and its task builder.

    Parameters
    ----------
    spec : type[S]
        The exact declaration variant this entry implements.
    build : Callable[[S, StepBuildContext], FittingTask]
        The builder of the declared task.
    present : Callable[[S], Mapping[str, frozenset[SupervisionArm]]]
        The supervision arms the task presents to each featurized dataset role.
    """

    _noun = "training-step plugin"
    __slots__ = ("_build", "_present", "_spec")

    def __init__(
        self,
        spec: type[S],
        build: Callable[[S, StepBuildContext], FittingTask],
        present: Callable[[S], Mapping[str, frozenset[SupervisionArm]]],
        /,
    ) -> None:
        object.__setattr__(self, "_spec", spec)
        object.__setattr__(self, "_build", build)
        object.__setattr__(self, "_present", present)

    @property
    def spec(self) -> type[S]:
        """Return the exact declaration variant this entry implements."""
        return self._spec

    def presented(self, declaration: S, /) -> Mapping[str, frozenset[SupervisionArm]]:
        """Return the arms the declared task presents to each featurized dataset."""
        return self._present(declaration)

    def prepare(self, declaration: S, context: StepBuildContext, /) -> FittingTask:
        """Build this exact declared task through its installed semantics."""
        if type(declaration) is not self._spec:
            msg = (
                f"the step plugin implements {self._spec.__name__}, not "
                f"{type(declaration).__name__}"
            )
            raise TypeError(msg)

        return self._build(declaration, context)


def step_plugin(declaration: StepSpec, /) -> StepPlugin[StepSpec]:
    """Return the unique installed plugin selected by one step declaration."""
    return cast(
        "StepPlugin[StepSpec]",
        resolve_plugin(
            "step", cast("str", declaration.kind), StepPlugin, type(declaration)
        ),
    )


def merge_presentations(
    presentations: Iterable[Mapping[str, frozenset[SupervisionArm]]], /
) -> Mapping[str, frozenset[SupervisionArm]]:
    """Return every presented arm, collected by featurized dataset role."""
    merged: dict[str, frozenset[SupervisionArm]] = {}

    for presentation in presentations:
        for role, arms in presentation.items():
            merged[role] = merged.get(role, frozenset()) | arms

    return merged


def steps_presentation(
    steps: Iterable[StepSpec], /
) -> Mapping[str, frozenset[SupervisionArm]]:
    """Return the arms a sequence of installed steps presents, by dataset role."""
    return merge_presentations(step_plugin(step).presented(step) for step in steps)


def _sentences(declaration: StepSpec, /) -> Mapping[str, frozenset[SupervisionArm]]:
    """Present the step's dataset as unlabeled sentences."""
    return {declaration.dataset: frozenset({Unlabeled})}


def _pairs(declaration: StepSpec, /) -> Mapping[str, frozenset[SupervisionArm]]:
    """Present the step's dataset as relational inference pairs."""
    return {declaration.dataset: frozenset({Relational})}


def _unfeaturized(declaration: StepSpec, /) -> Mapping[str, frozenset[SupervisionArm]]:
    """Present nothing, for a step that reads its dataset without a featurizer."""
    del declaration
    return {}


def _nested(
    declaration: AlternatingStepSpec, /
) -> Mapping[str, frozenset[SupervisionArm]]:
    """Present what every nested task presents."""
    return steps_presentation(declaration.tasks)


def _parser(declaration: ParserStepSpec, context: StepBuildContext, /) -> FittingTask:
    return context.parser(declaration)


def _supervised_warm_start(
    declaration: SupervisedWarmStartStepSpec, context: StepBuildContext, /
) -> FittingTask:
    return context.supervised_warm_start(declaration)


def _judgment(
    declaration: JudgmentStepSpec, context: StepBuildContext, /
) -> FittingTask:
    return context.judgment(declaration)


def _grounded(
    declaration: GroundedStepSpec, context: StepBuildContext, /
) -> FittingTask:
    return context.grounded(declaration)


def _alternating(
    declaration: AlternatingStepSpec, context: StepBuildContext, /
) -> FittingTask:
    return context.alternating(declaration)


def _token_warmup(
    declaration: TokenWarmupStepSpec, context: StepBuildContext, /
) -> FittingTask:
    return context.token_warmup(declaration)


def _terms(declaration: TermsStepSpec, context: StepBuildContext, /) -> FittingTask:
    return context.terms(declaration)


def _controller(
    declaration: ControllerStepSpec, context: StepBuildContext, /
) -> FittingTask:
    return context.controller(declaration)


ALTERNATING_PLUGIN: Final = StepPlugin(AlternatingStepSpec, _alternating, _nested)
CONTROLLER_PLUGIN: Final = StepPlugin(ControllerStepSpec, _controller, _unfeaturized)
GROUNDED_PLUGIN: Final = StepPlugin(GroundedStepSpec, _grounded, _pairs)
JUDGMENT_PLUGIN: Final = StepPlugin(JudgmentStepSpec, _judgment, _pairs)
PARSER_PLUGIN: Final = StepPlugin(ParserStepSpec, _parser, _sentences)
SUPERVISED_WARM_START_PLUGIN: Final = StepPlugin(
    SupervisedWarmStartStepSpec, _supervised_warm_start, _sentences
)
TOKEN_WARMUP_PLUGIN: Final = StepPlugin(
    TokenWarmupStepSpec, _token_warmup, _unfeaturized
)
TERMS_PLUGIN: Final = StepPlugin(TermsStepSpec, _terms, _unfeaturized)

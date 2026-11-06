"""Declared tasks composed by alternating and staged trainers."""

from types import MappingProxyType
from typing import Literal, Self, override

import didactic.api as dx

from lofi._checks import ensure_name, ensure_positive
from lofi.mixing import MixingSpec
from lofi.spec.components.base import RunRole
from lofi.spec.components.data import TrainingBatchKind
from lofi.spec.trainers.early_stopping import AccuracyStoppingSpec
from lofi.spec.trainers.generation import TermGenerationSpec, terms_metrics

__all__ = [
    "AlternatingStepSpec",
    "ControllerStepSpec",
    "GroundedStepSpec",
    "JudgmentStepSpec",
    "ParserStepSpec",
    "StepSpec",
    "SupervisedWarmStartStepSpec",
    "TermsStepSpec",
    "TokenWarmupStepSpec",
]


def _epoch_sampler_purposes(dataset: str, /) -> frozenset[tuple[str, str]]:
    """Return deterministic split, training and validation streams."""
    return frozenset(
        (dataset, purpose)
        for purpose in (
            "validation-split",
            "training-batches",
            "validation-batches",
        )
    )


class StepSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declare one named objective view within a composed trainer."""

    name: str
    dataset: str
    objectives: tuple[str, ...] = ()

    @dx.validates("dataset", "name")
    @classmethod
    def _name_is_nonempty(cls, value: str) -> str:
        """Validate the task name used in persisted metric keys."""
        return ensure_name("a training-step name", value)

    @dx.validates("objectives")
    @classmethod
    def _objectives_are_named_once(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Validate an optional ordered subset of declared loss objectives."""
        held = tuple(
            ensure_name("a training-step objective", value) for value in values
        )
        repeated = sorted({value for value in held if held.count(value) > 1})

        if repeated:
            msg = f"a training step names objectives more than once: {repeated}"
            raise ValueError(msg)

        return held

    def trains_objectives(self) -> bool:
        """Return whether the step trains a view of the run's loss objectives."""
        return False

    def refuse_objectives(self, declared: frozenset[str], /) -> None:
        """Refuse a step whose ``objectives`` would not select what it trains.

        Parameters
        ----------
        declared : frozenset[str]
            The names the run's loss terms are reported under.

        Raises
        ------
        ValueError
            If the step selects objectives while it trains no loss objective,
            or selects an objective outside ``declared``.
        """
        if self.objectives and not self.trains_objectives():
            msg = (
                f"trainer: training task {self.name!r} of kind "
                f"{self.model_dump()['kind']!r} trains no loss objective, so it "
                f"cannot select {list(self.objectives)}"
            )
            raise ValueError(msg)

        missing = sorted(set(self.objectives) - declared)

        if missing:
            msg = (
                f"trainer: training task {self.name!r} selects absent objectives "
                f"{missing}; declared objectives are {sorted(declared)}"
            )
            raise ValueError(msg)

    def required_batches(self) -> frozenset[TrainingBatchKind]:
        """Return the batch capability this step consumes."""
        return frozenset()

    def requires_featurizer(self) -> bool:
        """Return whether the step lowers its dataset through a featurizer."""
        return False

    def dataset_requirements(
        self,
    ) -> MappingProxyType[str, frozenset[TrainingBatchKind]]:
        """Return required batch capabilities keyed by dataset role."""
        return MappingProxyType({self.dataset: self.required_batches()})

    def featurized_datasets(self) -> frozenset[str]:
        """Return the datasets this task consumes through a featurizer."""
        return frozenset({self.dataset}) if self.requires_featurizer() else frozenset()

    def sampler_purposes(self) -> frozenset[tuple[str, str]]:
        """Return seeded sampler purposes keyed by dataset role."""
        return frozenset()

    def accuracy_target(self) -> AccuracyStoppingSpec | None:
        """Return the held-out accuracy target that ends the step, if declared."""
        return None

    def evaluated_metrics(self) -> frozenset[str]:
        """Return the held-out metrics the step evaluates for its target."""
        return frozenset()

    def consumes_constraints(self) -> bool:
        """Return whether the step reads the run's selected constraints."""
        return False

    def sampler_roles(self) -> frozenset[RunRole]:
        """Return the generated data component roles this step consumes."""
        return frozenset()


class ParserStepSpec(StepSpec, extra="forbid"):
    """Train the declared parser objectives."""

    kind: Literal["parser"] = "parser"

    @override
    def trains_objectives(self) -> bool:
        """Return that the step trains its selected view of the run's loss."""
        return True

    @override
    def required_batches(self) -> frozenset[TrainingBatchKind]:
        """Return sentence batches."""
        return frozenset({TrainingBatchKind.SENTENCES})

    @override
    def requires_featurizer(self) -> bool:
        """Return that parser batches are featurized."""
        return True

    @override
    def sampler_purposes(self) -> frozenset[tuple[str, str]]:
        """Return the three deterministic parser-epoch streams."""
        return _epoch_sampler_purposes(self.dataset)


class SupervisedWarmStartStepSpec(StepSpec, extra="forbid"):
    """Train declared supervised warm-start objectives."""

    kind: Literal["supervised-warm-start"] = "supervised-warm-start"

    @override
    def trains_objectives(self) -> bool:
        """Return that the step trains its selected view of the run's loss."""
        return True

    @override
    def required_batches(self) -> frozenset[TrainingBatchKind]:
        """Return sentence batches carrying gold forms."""
        return frozenset({TrainingBatchKind.SENTENCES})

    @override
    def requires_featurizer(self) -> bool:
        """Return that supervised sentence batches are featurized."""
        return True

    @override
    def sampler_purposes(self) -> frozenset[tuple[str, str]]:
        """Return the three deterministic sentence-epoch streams."""
        return _epoch_sampler_purposes(self.dataset)


class JudgmentStepSpec(StepSpec, extra="forbid"):
    """Train declared judgment objectives."""

    kind: Literal["judgment"] = "judgment"

    @override
    def trains_objectives(self) -> bool:
        """Return that the step trains its selected view of the run's loss."""
        return True

    @override
    def required_batches(self) -> frozenset[TrainingBatchKind]:
        """Return collected inference judgment pairs."""
        return frozenset(
            {TrainingBatchKind.INFERENCE_PAIRS, TrainingBatchKind.JUDGMENTS}
        )

    @override
    def requires_featurizer(self) -> bool:
        """Return that judgment pairs are featurized."""
        return True

    @override
    def sampler_purposes(self) -> frozenset[tuple[str, str]]:
        """Return the three deterministic inference-epoch streams."""
        return _epoch_sampler_purposes(self.dataset)


class GroundedStepSpec(StepSpec, extra="forbid"):
    """Train declared grounded inference objectives."""

    kind: Literal["grounded"] = "grounded"

    @override
    def trains_objectives(self) -> bool:
        """Return that the step trains its selected view of the run's loss."""
        return True

    @override
    def required_batches(self) -> frozenset[TrainingBatchKind]:
        """Return labeled inference pair batches."""
        return frozenset({TrainingBatchKind.INFERENCE_PAIRS})

    @override
    def requires_featurizer(self) -> bool:
        """Return that grounded pairs are featurized."""
        return True

    @override
    def sampler_purposes(self) -> frozenset[tuple[str, str]]:
        """Return the three deterministic inference-epoch streams."""
        return _epoch_sampler_purposes(self.dataset)


class TokenWarmupStepSpec(StepSpec, extra="forbid"):
    """Train declared token warmup objectives."""

    kind: Literal["token-warm-up"] = "token-warm-up"
    space: str

    @dx.validates("space")
    @classmethod
    def _space_is_named(cls, value: str) -> str:
        """Validate the represented space supplying the lexical types."""
        return ensure_name("a token-warm-up space", value)

    @override
    def required_batches(self) -> frozenset[TrainingBatchKind]:
        """Return sentence batches whose tokens have lexical types."""
        return frozenset({TrainingBatchKind.SENTENCES})

    @override
    def requires_featurizer(self) -> bool:
        """Return that token warmup batches are featurized."""
        return True

    @override
    def consumes_constraints(self) -> bool:
        """Return that token warmup reads the selected lexical constraints."""
        return True


class TermsStepSpec(StepSpec, extra="forbid"):
    """Train one represented space on generated terms and combinator pairs.

    The step trains the space's encoder and identity decoder to autoencode
    generated terms and, when it declares ``combinators``, the space's
    combinator decoders on generated rule instances.
    """

    kind: Literal["terms"] = "terms"
    space: str
    combinators: bool
    generation: TermGenerationSpec
    stop: AccuracyStoppingSpec | None = None

    @dx.validates("space")
    @classmethod
    def _space_is_named(cls, value: str) -> str:
        """Validate the represented space whose components are trained."""
        return ensure_name("a terms-step space", value)

    @override
    def required_batches(self) -> frozenset[TrainingBatchKind]:
        """Return a sentence dataset supplying the term language."""
        return frozenset({TrainingBatchKind.SENTENCES})

    @override
    def sampler_purposes(self) -> frozenset[tuple[str, str]]:
        """Return the generated-batch stream and any held-out stream."""
        purposes = {(self.dataset, "terms-batches")}

        if self.stop is not None:
            purposes.add((self.dataset, "terms-held-out-batches"))

        return frozenset(purposes)

    @override
    def accuracy_target(self) -> AccuracyStoppingSpec | None:
        """Return the declared held-out accuracy target."""
        return self.stop

    @override
    def evaluated_metrics(self) -> frozenset[str]:
        """Return the exact-match metrics of the trained objectives."""
        return terms_metrics(((self.space, self.combinators),))

    @override
    def sampler_roles(self) -> frozenset[RunRole]:
        """Return the type sampler and, for combinator pairs, the pair sampler."""
        if not self.combinators:
            return frozenset({RunRole.TYPE_SAMPLER})

        return frozenset({RunRole.PAIR_SAMPLER, RunRole.TYPE_SAMPLER})


class ControllerStepSpec(StepSpec, extra="forbid"):
    """Train declared controller objectives."""

    kind: Literal["controller"] = "controller"
    space: str
    max_depth: int
    stop: AccuracyStoppingSpec | None = None

    @dx.validates("space")
    @classmethod
    def _space_is_named(cls, value: str) -> str:
        """Validate the represented space supplying rule instances."""
        return ensure_name("a controller-supervision space", value)

    @dx.validates("max_depth")
    @classmethod
    def _depth_is_positive(cls, value: int) -> int:
        """Validate the rule instance depth bound."""
        return ensure_positive("a controller-supervision depth", value)

    @override
    def required_batches(self) -> frozenset[TrainingBatchKind]:
        """Return a sentence dataset supplying grammar actions."""
        return frozenset({TrainingBatchKind.SENTENCES})

    @override
    def sampler_purposes(self) -> frozenset[tuple[str, str]]:
        """Return the generated controller-instance stream."""
        purposes = {(self.dataset, "controller-batches")}

        if self.stop is not None:
            purposes.add((self.dataset, "controller-held-out-batches"))

        return frozenset(purposes)

    @override
    def accuracy_target(self) -> AccuracyStoppingSpec | None:
        """Return the declared held-out accuracy target."""
        return self.stop

    @override
    def evaluated_metrics(self) -> frozenset[str]:
        """Return the action accuracy reported under the step's name."""
        return frozenset({f"loss/{self.name}/accuracy"})

    @override
    def sampler_roles(self) -> frozenset[RunRole]:
        """Return the pair and type samplers the generated instances come from."""
        return frozenset({RunRole.PAIR_SAMPLER, RunRole.TYPE_SAMPLER})


class AlternatingStepSpec(StepSpec, extra="forbid"):
    """Compose nested declared tasks under their own mixing policy."""

    kind: Literal["alternating"] = "alternating"
    tasks: tuple[StepSpec, ...]
    mixing: MixingSpec

    @dx.model_validator()
    def _tasks_are_distinct(self) -> Self:
        """Require at least two uniquely named nested tasks."""
        if len(self.tasks) < 2:
            msg = "an alternating step declares at least two nested tasks"
            raise ValueError(msg)

        names = tuple(task.name for task in self.tasks)
        repeated = sorted({name for name in names if names.count(name) > 1})

        if repeated:
            msg = f"an alternating step names tasks more than once: {repeated}"
            raise ValueError(msg)

        return self

    @override
    def refuse_objectives(self, declared: frozenset[str], /) -> None:
        """Refuse objectives on this step, or a nested task's undeclared ones.

        The step trains only its nested tasks, so it selects no objective of
        its own.
        """
        super().refuse_objectives(declared)

        for task in self.tasks:
            task.refuse_objectives(declared)

    @override
    def required_batches(self) -> frozenset[TrainingBatchKind]:
        """Return the union of the nested tasks' batch capabilities."""
        return frozenset(
            kind for task in self.tasks for kind in task.required_batches()
        )

    @override
    def requires_featurizer(self) -> bool:
        """Return whether any nested task lowers featurized batches."""
        return any(task.requires_featurizer() for task in self.tasks)

    @override
    def dataset_requirements(
        self,
    ) -> MappingProxyType[str, frozenset[TrainingBatchKind]]:
        """Return the union of nested requirements by dataset role."""
        required: dict[str, set[TrainingBatchKind]] = {}

        for task in self.tasks:
            for role, kinds in task.dataset_requirements().items():
                required.setdefault(role, set()).update(kinds)

        return MappingProxyType(
            {role: frozenset(kinds) for role, kinds in required.items()}
        )

    @override
    def featurized_datasets(self) -> frozenset[str]:
        """Return every nested dataset consumed through a featurizer."""
        return frozenset(
            role for task in self.tasks for role in task.featurized_datasets()
        )

    @override
    def consumes_constraints(self) -> bool:
        """Return whether any nested task reads the selected constraints."""
        return any(task.consumes_constraints() for task in self.tasks)

    @override
    def sampler_purposes(self) -> frozenset[tuple[str, str]]:
        """Return the union of every nested task's sampler purposes."""
        return frozenset(
            stream for task in self.tasks for stream in task.sampler_purposes()
        )

    @override
    def sampler_roles(self) -> frozenset[RunRole]:
        """Return the union of every nested task's sampler roles."""
        return frozenset(role for task in self.tasks for role in task.sampler_roles())

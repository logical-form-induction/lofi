"""Parser-training plugin."""

from typing import TYPE_CHECKING, Final, Literal, Protocol, runtime_checkable

from lofi.data import Relational, Unlabeled
from lofi.spec.components.data import LearnerLanguageDatasetSpec
from lofi.spec.constraint_policy import AmbiguityMode, PolicyDomain
from lofi.spec.constraint_sources import GoldDerivationConstraintSourceSpec
from lofi.spec.trainers.parser import ParserTrainerSpec
from lofi.training_plugins._plugin import (
    TrainerBuilderTarget,
    TrainerPlugin,
    TrainingLength,
    TrainingUnit,
    training_sampler_stream,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.data import SupervisionArm
    from lofi.spec import RunSpec

__all__ = ["PLUGIN"]


@runtime_checkable
class _ConstraintConsumer(Protocol):
    """Declare that a loss consumes selected dataset constraints."""

    def constraint_route(self) -> Literal["target", "negative-bracket"]:
        """Return the selected-constraint representation the loss consumes."""
        ...


@runtime_checkable
class _ClampedConstraintSource(Protocol):
    """Expose the constituency route selected by a constraint source."""

    def constraint_route(self) -> Literal["target", "negative-bracket"]:
        """Return the representation required by the source's clamp."""
        ...


def _represented_sort(run_spec: RunSpec, dataset: str, domain: PolicyDomain, /) -> str:
    """Return the learner sort a constraint domain labels."""
    bundle = run_spec.data.datasets[dataset]
    language = run_spec.model.language.declared_language()

    if language is None and isinstance(bundle.dataset, LearnerLanguageDatasetSpec):
        language = bundle.dataset.learner_language()

    if language is None:
        msg = (
            f"data dataset {dataset!r} and the model publish no learner language "
            "for constraint planning"
        )
        raise ValueError(msg)

    match domain:
        case PolicyDomain.SYNTACTIC:
            return language.category_sort
        case PolicyDomain.SEMANTIC:
            return language.term_sort


def _plan(
    declaration: ParserTrainerSpec,
    run_spec: RunSpec,
    /,
    *,
    composed_constraint_consumer: bool = False,
) -> tuple[str, ...]:
    """Validate the wiring between the parser run's evaluators, loss and data.

    The run's roles were checked against the declaration when it was composed.
    """
    for role, evaluator in run_spec.evaluators.items():
        for dependency in evaluator.dependencies():
            if (
                dependency.port in ("span-readout", "terms-readout")
                and dependency.role not in run_spec.model.readouts
            ):
                msg = (
                    f"evaluator {role!r} names readout role "
                    f"{dependency.role!r}, which is not declared among "
                    f"{sorted(run_spec.model.readouts)}"
                )
                raise ValueError(msg)

    loss = run_spec.loss
    constraint_dependencies = (
        ()
        if loss is None
        else tuple(
            dependency
            for term in loss.terms
            for dependency in term.objective.dependencies()
            if dependency.port == "constraint-readout"
        )
    )
    bundle = run_spec.data.datasets[declaration.dataset]
    source = bundle.constraint_source
    constraint_routes = (
        frozenset[str]()
        if loss is None
        else frozenset(
            term.objective.constraint_route()
            for term in loss.terms
            if isinstance(term.objective, _ConstraintConsumer)
        )
    )

    constraint_roles = sorted(
        {dependency.role for dependency in constraint_dependencies}
    )

    if len(constraint_roles) > 1:
        msg = (
            "configured parser constraint objectives name more than one "
            f"term-readout role: {constraint_roles}"
        )
        raise ValueError(msg)

    if source is None:
        if constraint_routes or composed_constraint_consumer:
            msg = (
                "configured constraint selection requires both one source and a "
                "parser constraint objective or composed token-warm-up step"
            )
            raise ValueError(msg)
    else:
        if not isinstance(source, _ClampedConstraintSource):
            msg = f"constraint source {source.kind!r} publishes no clamp route"
            raise TypeError(msg)

        expected_route = source.constraint_route()
        if constraint_routes and constraint_routes != {expected_route}:
            msg = (
                f"constraint source {source.kind!r} serves the {expected_route!r} "
                f"route, but its consumers request {sorted(constraint_routes)!r}"
            )
            raise ValueError(msg)

        if not constraint_routes and not composed_constraint_consumer:
            msg = (
                f"constraint source {source.kind!r} requires at least one "
                f"{expected_route!r} constraint consumer"
            )
            raise ValueError(msg)

        if composed_constraint_consumer and expected_route != "target":
            msg = "a token-warm-up step cannot consume negative-bracket constraints"
            raise ValueError(msg)

        if expected_route == "negative-bracket" and isinstance(
            source, GoldDerivationConstraintSourceSpec
        ):
            msg = (
                f"constraint source {source.kind!r} cannot serve the "
                "'negative-bracket' route: a negative bracket excludes its span, "
                "and the source selects every gold constituent, including spans "
                "every derivation builds, so no derivation survives the clamp"
            )
            raise ValueError(msg)

        if constraint_dependencies:
            dependency = constraint_dependencies[0]
            readout = run_spec.model.readouts[dependency.role]
            represented = run_spec.model.spaces[readout.space].sort
            expected = _represented_sort(run_spec, declaration.dataset, source.policy)

            if represented != expected:
                msg = (
                    f"constraint source {source.kind!r} serves the {source.policy} "
                    f"policy over sort {expected!r}, but term-readout role "
                    f"{dependency.role!r} represents {represented!r}"
                )
                raise ValueError(msg)

        elif source.policy is PolicyDomain.SEMANTIC and expected_route == "target":
            msg = (
                "semantic target constraints require a constraint objective "
                "naming a logical-form term readout"
            )
            raise ValueError(msg)

        policy = bundle.constraint_policy.serving(source.policy)

        if policy.ambiguity is AmbiguityMode.WEIGHTED:
            msg = "configured pinning does not admit weighted labels"
            raise ValueError(msg)

    return tuple(
        training_sampler_stream(declaration.dataset, purpose)
        for purpose in (
            "validation-split",
            "training-batches",
            "validation-batches",
        )
    )


def _present(
    declaration: ParserTrainerSpec, run_spec: RunSpec, /
) -> Mapping[str, frozenset[SupervisionArm]]:
    """Return the arm the parser presents to its dataset's featurizer.

    A grounded parser featurizes each premise with its hypothesis as a
    relational pair; an ungrounded parser featurizes each sentence unlabeled.
    """
    arm = Unlabeled if run_spec.model.grounding is None else Relational

    return {declaration.dataset: frozenset({arm})}


def _price(declaration: ParserTrainerSpec, /) -> TrainingLength:
    """Return the declared epochs, which cap any early stop."""
    return TrainingLength(declaration.epochs, TrainingUnit.EPOCHS)


PLUGIN: Final = TrainerPlugin(
    spec=ParserTrainerSpec,
    plan=_plan,
    present=_present,
    build=TrainerBuilderTarget("lofi.train._assembly:_build_parser_training"),
    price=_price,
)
"""The parser trainer's planning, construction and pricing plugin."""

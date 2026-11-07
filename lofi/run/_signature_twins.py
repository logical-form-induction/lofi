"""Post-composition expansion into trainable scrambled-signature twins."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

from lofi.eval._scrambled import sample_exchangeable_signature_twins
from lofi.run._boundary.sweep import SweepPoint
from lofi.run._training import materialize_learner_language, plan_training
from lofi.spec import RunSpec, SignatureAssignmentsSpec
from lofi.spec._signature import SignatureAssignments

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi._types import JsonValue
    from lofi.run._boundary.experiment import SignatureTwinSettings
    from lofi.run._seed import ExperimentSeed, PointSeed
    from lofi.run._training import TrainingBlueprint
    from lofi.spec.components.model import InlineModelLanguageSpec

__all__ = ["signature_twin_augmenter"]


def _assignments(
    declaration: InlineModelLanguageSpec,
    blueprint: TrainingBlueprint,
    /,
) -> SignatureAssignments:
    """Observe codec and action slots from the compiled learner."""
    grammar = blueprint.learner.grammar
    signature = grammar.grammar.signature
    sort = grammar.language.category_sort
    vocabulary = signature.vocabulary(sort)
    declared = declaration.grammar.signature
    primitive_names = frozenset(declared.primitives)
    constructor_names = frozenset(
        constructor.name for constructor in declared.constructors
    )
    return SignatureAssignments(
        tuple(name for name in vocabulary.names if name in primitive_names),
        tuple(
            operator.name
            for operator in signature.operators
            if operator.name in constructor_names
        ),
        tuple(action.name for action in blueprint.actions.composing),
    )


def _constructor_shapes(
    blueprint: TrainingBlueprint,
    names: frozenset[str],
    /,
) -> tuple[str, ...]:
    """Return compiled constructor kinds in codec-slot order."""
    grammar = blueprint.learner.grammar
    signature = grammar.grammar.signature
    return tuple(
        json.dumps(
            (operator.sort, *operator.arg_sorts),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for operator in signature.operators
        if operator.name in names
    )


def _permuted_language(
    declaration: InlineModelLanguageSpec,
    assignments: SignatureAssignments,
    /,
) -> InlineModelLanguageSpec:
    """Return an inline learner declaration reordered by one assignment."""
    grammar = declaration.grammar
    signature = grammar.signature
    constructors = {
        constructor.name: constructor for constructor in signature.constructors
    }

    if set(constructors) != set(assignments.constructors):
        msg = "a signature twin changes the learner constructor inventory"
        raise ValueError(msg)

    return declaration.with_(
        grammar=grammar.with_(
            signature=signature.with_(
                primitives=assignments.primitives,
                constructors=tuple(
                    constructors[name] for name in assignments.constructors
                ),
            )
        )
    )


def _run(
    base: RunSpec,
    language: InlineModelLanguageSpec,
    assignments: SignatureAssignments,
    *,
    twin: str,
) -> RunSpec:
    """Materialize one ordinary run declaration for a fitted assignment."""
    comparison = base.comparison

    if comparison is None:
        msg = "signature twins require source-comparison point metadata"
        raise ValueError(msg)

    return base.with_(
        model=base.model.with_(language=language),
        comparison=comparison.with_(twin=twin),
        signature_assignment=SignatureAssignmentsSpec(
            primitives=assignments.primitives,
            constructors=assignments.constructors,
            combinators=assignments.combinators,
        ),
    )


def _point(base: SweepPoint, run: RunSpec, /) -> SweepPoint:
    """Return one expanded point carrying a materialized run declaration."""
    config = cast("dict[str, JsonValue]", json.loads(run.model_dump_json()))
    return SweepPoint(
        config,
        overrides=base.overrides,
        provenance=base.provenance,
        seed=base.seed,
    )


def _materialize_data_seeds(base: RunSpec, point: PointSeed, /) -> RunSpec:
    """Freeze every derived source seed under the unexpanded base point."""
    datasets = {
        role: section.with_(
            dataset=section.dataset.with_derived_seeds(
                point.data_seed, f"data.datasets.{role}.dataset"
            )
        )
        for role, section in base.data.datasets.items()
    }
    return base.with_(data=base.data.with_(datasets=datasets))


def signature_twin_augmenter(
    settings: SignatureTwinSettings,
    experiment_seed: ExperimentSeed,
    /,
) -> Callable[[SweepPoint], tuple[SweepPoint, ...]]:
    """Return deterministic observed-plus-twin expansion for composed run points."""
    templates: dict[str, tuple[RunSpec, ...]] = {}

    def augment(point: SweepPoint) -> tuple[SweepPoint, ...]:
        base_seed = experiment_seed.point(point.point_id, point.seed)
        declared = _materialize_data_seeds(
            RunSpec.model_validate(point.config), base_seed
        )
        template_key = declared.model_dump_json()
        standing = templates.get(template_key)

        if standing is not None:
            return tuple(_point(point, run) for run in standing)

        planned = plan_training(declared)
        inline = materialize_learner_language(
            declared.model.language, planned.checked_datasets
        )
        observed = _assignments(inline, planned)
        comparison = declared.comparison

        if comparison is None:
            msg = "signature twins require source-comparison point metadata"
            raise ValueError(msg)

        if comparison.condition == settings.primary_condition:
            category_sort = planned.learner.grammar.language.category_sort
            twins = sample_exchangeable_signature_twins(
                observed,
                primitive_shapes=tuple(category_sort for _ in observed.primitives),
                constructor_shapes=_constructor_shapes(
                    planned, frozenset(observed.constructors)
                ),
                count=settings.count,
                seed=settings.seed,
            )
            runs = (
                _run(declared, inline, observed, twin=settings.observed_twin),
                *(
                    _run(
                        declared,
                        _permuted_language(inline, assignment),
                        assignment,
                        twin=f"twin-{position:03d}",
                    )
                    for position, assignment in enumerate(twins, start=1)
                ),
            )
            expected_assignments = (observed, *twins)
        else:
            runs = (_run(declared, inline, observed, twin=settings.observed_twin),)
            expected_assignments = (observed,)

        points = tuple(_point(point, run) for run in runs)

        for run, expected in zip(runs, expected_assignments, strict=True):
            declared_assignment = run.signature_assignment

            if declared_assignment is None or declared_assignment.build() != expected:
                msg = "signature-twin construction changed its compiled assignment"
                raise ValueError(msg)

        templates[template_key] = runs
        return points

    return augment

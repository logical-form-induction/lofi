from typing import TYPE_CHECKING, NamedTuple

import pytest

from lofi.nn.loss._ports import (
    BoundObjective,
    LossComponentBindings,
    Objective,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi.nn.loss._ports import SpanTokenDecoder
    from lofi.nn.loss._readout import TermsReadout
    from lofi.nn.spaces import Space, SpaceRelationBinding

__all__ = [
    "BoundObjectiveFixture",
    "BoundObjectiveLawSuite",
    "LossComponentBindingsFixture",
    "LossComponentBindingsLawSuite",
]


class LossComponentBindingsFixture(NamedTuple):
    """One binding context, the term name it carries and what it resolves by role."""

    context: LossComponentBindings
    name: str
    readout_role: str
    readout: TermsReadout
    space_role: str
    space: Space
    relation_role: str
    relation: SpaceRelationBinding
    span_decoder_role: str
    span_decoder: SpanTokenDecoder


class LossComponentBindingsLawSuite:
    """Reusable laws for model-role bindings supplied to loss recipes."""

    pytestmark = pytest.mark.conformance

    def test_the_context_implements_the_public_binding_capability(
        self,
        binding_conformance: LossComponentBindingsFixture,
    ) -> None:
        """Check that the context is recognizable through the public protocol.

        Parameters
        ----------
        binding_conformance : LossComponentBindingsFixture
            Context and the exact components it resolves.
        """
        assert isinstance(binding_conformance.context, LossComponentBindings)

    def test_the_context_carries_the_name_of_the_term_being_built(
        self,
        binding_conformance: LossComponentBindingsFixture,
    ) -> None:
        """Check that the context reports the term name it was given.

        Parameters
        ----------
        binding_conformance : LossComponentBindingsFixture
            Context and the exact components it resolves.
        """
        assert binding_conformance.context.name == binding_conformance.name

    def test_the_context_resolves_each_exact_component_family_by_role(
        self,
        binding_conformance: LossComponentBindingsFixture,
    ) -> None:
        """Check that each role returns the component supplied under its family.

        Parameters
        ----------
        binding_conformance : LossComponentBindingsFixture
            Context and the exact components it resolves.
        """
        context = binding_conformance.context

        assert (
            context.terms_readout(binding_conformance.readout_role)
            is binding_conformance.readout
        )
        assert (
            context.space(binding_conformance.space_role) is binding_conformance.space
        )
        assert (
            context.readout_space(binding_conformance.readout_role)
            is binding_conformance.space
        )
        assert (
            context.relation(binding_conformance.relation_role)
            is binding_conformance.relation
        )
        assert (
            context.span_decoder(binding_conformance.span_decoder_role)
            is binding_conformance.span_decoder
        )


class BoundObjectiveFixture[ContextT, OutputsT](NamedTuple):
    """One bound-objective recipe, its context, and a context-state predicate."""

    recipe: BoundObjective[ContextT, OutputsT]
    context: ContextT
    unchanged: Callable[[ContextT], bool]


class BoundObjectiveLawSuite:
    """Reusable laws for objectives that bind after component construction."""

    pytestmark = pytest.mark.conformance

    def test_the_recipe_is_bound_but_not_already_an_objective[ContextT, OutputsT](
        self,
        bound_conformance: BoundObjectiveFixture[ContextT, OutputsT],
    ) -> None:
        """Check that evaluation is unavailable until the context is supplied.

        Parameters
        ----------
        bound_conformance : BoundObjectiveFixture[ContextT, OutputsT]
            Recipe, context and state predicate used by the laws.
        """
        assert isinstance(bound_conformance.recipe, BoundObjective)
        assert not isinstance(bound_conformance.recipe, Objective)

    def test_binding_returns_a_fresh_objective_each_time[ContextT, OutputsT](
        self,
        bound_conformance: BoundObjectiveFixture[ContextT, OutputsT],
    ) -> None:
        """Check that each bind produces a separate live objective.

        Parameters
        ----------
        bound_conformance : BoundObjectiveFixture[ContextT, OutputsT]
            Recipe, context and state predicate used by the laws.
        """
        first = bound_conformance.recipe.bind(bound_conformance.context)
        second = bound_conformance.recipe.bind(bound_conformance.context)

        assert isinstance(first, Objective)
        assert isinstance(second, Objective)
        assert first is not second

    def test_binding_preserves_the_declared_name_and_outputs[ContextT, OutputsT](
        self,
        bound_conformance: BoundObjectiveFixture[ContextT, OutputsT],
    ) -> None:
        """Check that binding changes dependencies into references, not metadata.

        Parameters
        ----------
        bound_conformance : BoundObjectiveFixture[ContextT, OutputsT]
            Recipe, context and state predicate used by the laws.
        """
        objective = bound_conformance.recipe.bind(bound_conformance.context)

        assert objective.name == bound_conformance.recipe.name
        assert objective.requires == bound_conformance.recipe.requires

    def test_binding_does_not_mutate_the_supplied_context[ContextT, OutputsT](
        self,
        bound_conformance: BoundObjectiveFixture[ContextT, OutputsT],
    ) -> None:
        """Check that dependencies are read without changing their persistent state.

        Parameters
        ----------
        bound_conformance : BoundObjectiveFixture[ContextT, OutputsT]
            Recipe, context and state predicate used by the laws.
        """
        bound_conformance.recipe.bind(bound_conformance.context)

        assert bound_conformance.unchanged(bound_conformance.context)

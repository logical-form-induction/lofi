"""The root of every configurable component declaration."""

from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

import didactic.api as dx

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi._types import SubmoduleDeclaration
    from lofi.spec._ports import Dependency
    from lofi.spec.outputs import DeclaredOutput

__all__ = ["ComponentSpec", "ConfiguredSpec", "RunRole"]


class RunRole(StrEnum):
    """A structural role a run declares components under, named by its path.

    A single component's role is declared when the component is present, a
    keyed collection's when it holds at least one member, and the loss's when
    the run declares a loss section.
    """

    DATASETS = "data.datasets"
    PAIR_SAMPLER = "data.pair_sampler"
    TYPE_SAMPLER = "data.type_sampler"
    LOSS = "loss"
    CAPACITY = "model.capacity"
    PARSER = "model.parser"
    TOKENS_ENCODER = "model.token_encoder"
    SUBWORD_POOLER = "model.subword_pooler"
    TERM_ENCODERS = "model.encoders"
    CONTROLLER = "model.controller"
    GROUNDING = "model.grounding"
    TERM_DECODERS = "model.decoders"
    READOUTS = "model.readouts"
    SPAN_DECODERS = "model.span_decoders"
    SINKS = "output.sinks"
    PROBES = "output.probes"
    EVALUATORS = "output.evaluators"
    TRAINER = "training.trainer"
    OPTIMIZER = "training.optimizer"
    SCHEDULER = "training.scheduler"


class ComponentSpec(dx.Model, extra="forbid"):
    """Base for configurable component declarations."""

    def dependencies(self) -> tuple[Dependency, ...]:
        """Return dependencies declared by this component.

        Returns
        -------
        tuple[Dependency, ...]
            Dependencies in declaration order.
        """
        return ()

    def sharing(self) -> Mapping[str, SubmoduleDeclaration]:
        """Return the producer submodules this component holds, by input port.

        Returns
        -------
        Mapping[str, SubmoduleDeclaration]
            The publication each port's producer lends this component and how
            the component registers it. A port absent here shares nothing.
        """
        return MappingProxyType({})

    def publications(self) -> frozenset[str]:
        """Return the names of the submodules this component lends its consumers.

        Returns
        -------
        frozenset[str]
            Every name a consumer's sharing declaration may select from this
            component.
        """
        return frozenset()

    def pretrained_weights(self) -> str | None:
        """Return the pinned checkpoint the component's weights are loaded from.

        Returns
        -------
        str | None
            The repository and the revision it is pinned at, joined by ``@``,
            for a component whose weights come through a pretrained loader, so
            a run may hand it a loader of its own, or None for one whose weights
            are initialized.
        """
        return None

    def checkpointed(self) -> bool:
        """Return whether this component contributes independently stored state.

        Returns
        -------
        bool
            Whether model checkpoints and model-only bundles store this
            component under its declared role.
        """
        return True

    @dx.computed
    def declared_output(self) -> DeclaredOutput | None:
        """Return the component's output declaration.

        Returns
        -------
        DeclaredOutput | None
            Output metadata, or ``None`` when the component declares none.
        """
        return None


class ConfiguredSpec[V](Protocol):
    """A declaration a component plugin builds from.

    The declaration's fields hold every default the component has. Its settings
    value restates those fields, resolved, with no default of its own.
    """

    @classmethod
    def family(cls) -> str:
        """Return the family whose entry-point group lists the plugins."""
        ...

    @property
    def kind(self) -> str:
        """Return the tag that selects the plugin within its family."""
        ...

    def settings(self) -> V:
        """Return the resolved settings the plugin's builder takes.

        Returns
        -------
        V
            A frozen value with one field per setting the builder reads.
        """
        ...


def _in_role_order[S: ComponentSpec](roles: dict[str, S]) -> dict[str, S]:
    """Return components ordered by role name.

    Parameters
    ----------
    roles : dict[str, S]
        Components keyed by role.

    Returns
    -------
    dict[str, S]
        New mapping in ascending role order.
    """
    return dict(sorted(roles.items()))

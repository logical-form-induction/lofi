"""Sort-indexed bindings between declared neural components."""

from enum import StrEnum
from typing import TYPE_CHECKING, Literal, NamedTuple, Self

import didactic.api as dx

from lofi._checks import ensure_finite, ensure_name, ensure_positive
from lofi._types import ModuleSharing, SubmoduleDeclaration
from lofi.spec.components.base import ComponentSpec

if TYPE_CHECKING:
    from lofi._types import JsonValue

__all__ = [
    "CombinatorFamilySpec",
    "Merge",
    "SpaceMapForm",
    "SpaceRelation",
    "SpaceRelationSpec",
    "SpaceSpec",
    "TypeSpaceMapSettings",
    "TypeSpaceMapSpec",
]


class Merge(StrEnum):
    """Names how a combinator decoder receives its daughter vectors."""

    CONCATENATE = "concatenate"
    CHART_COMBINE = "chart-combine"


class SpaceRelation(StrEnum):
    """Names how a category space is related to another term space."""

    ERASE_THEN_DECODE = "erase-then-decode"
    CONSISTENCY = "consistency"
    MAPPED = "mapped"


class SpaceMapForm(StrEnum):
    """Names the architecture of a learned map between represented spaces."""

    LINEAR = "linear"
    MULTILAYER = "multilayer"


class TypeSpaceMapSettings(NamedTuple):
    """Resolved architecture of a learned map between two represented spaces."""

    form: SpaceMapForm
    hidden: int | None
    depth: int | None
    dropout: float | None


class TypeSpaceMapSpec(ComponentSpec, extra="forbid"):
    """Declare the learned map used by a mapped represented-space relation."""

    kind: Literal["type-space-map"] = "type-space-map"
    form: SpaceMapForm
    hidden: int | None = None
    depth: int | None = None
    dropout: float | None = None

    @dx.validates("hidden", "depth")
    @classmethod
    def _counts_are_positive(cls, value: int | None) -> int | None:
        """Validate an optional positive network dimension."""
        return None if value is None else ensure_positive("a space-map count", value)

    @dx.validates("dropout")
    @classmethod
    def _dropout_is_a_rate(cls, value: float | None) -> float | None:
        """Validate an optional dropout rate in the half-open unit interval."""
        if value is None:
            return None

        rate = ensure_finite("a space-map dropout rate", value)

        if not 0.0 <= rate < 1.0:
            msg = f"a space-map dropout rate lies in [0, 1), not {value!r}"
            raise ValueError(msg)

        return rate

    @dx.model_validator()
    def _form_owns_its_fields(self) -> Self:
        """Require hidden-layer settings exactly for a multilayer map."""
        layers = (self.hidden, self.depth, self.dropout)

        if self.form is SpaceMapForm.LINEAR:
            if any(value is not None for value in layers):
                msg = "a linear space map declares no hidden-layer settings"
                raise ValueError(msg)
        elif any(value is None for value in layers):
            msg = "a multilayer space map declares hidden, depth and dropout"
            raise ValueError(msg)

        return self

    def settings(self) -> TypeSpaceMapSettings:
        """Return the validated map architecture."""
        return TypeSpaceMapSettings(
            self.form,
            self.hidden,
            self.depth,
            self.dropout,
        )


class SpaceRelationSpec(dx.Model, extra="forbid"):
    """Declares a relation from one represented space to another."""

    space: str
    via: SpaceRelation
    mapping: TypeSpaceMapSpec | None = None

    @dx.validates("space")
    @classmethod
    def _space_is_named(cls, value: str) -> str:
        """Return a nonempty related-space role."""
        return ensure_name("a related space", value)

    @dx.model_validator()
    def _mapped_relation_owns_its_map(self) -> Self:
        """Require one learned map exactly for the mapped relation."""
        if (self.via is SpaceRelation.MAPPED) != (self.mapping is not None):
            msg = "only a mapped space relation declares one mapping"
            raise ValueError(msg)

        return self


class CombinatorFamilySpec(dx.Model, extra="forbid"):
    """Declare unary rules and the decoders for binary composing actions."""

    merge: Merge | None = None
    decoders: dict[str, str]

    @dx.validates("decoders", mode="before")
    @classmethod
    def _bindings_are_named(cls, value: JsonValue) -> dict[str, str]:
        """Return action bindings in canonical action-name order."""
        if not isinstance(value, dict):
            msg = f"combinator decoders are a mapping by action name, not {value!r}"
            raise TypeError(msg)

        bindings: dict[str, str] = {}

        for action, decoder in value.items():
            if not isinstance(action, str):
                msg = f"a combinator action name is a string, not {action!r}"
                raise TypeError(msg)

            if not isinstance(decoder, str):
                msg = f"the decoder for action {action!r} is named, not {decoder!r}"
                raise TypeError(msg)

            bindings[ensure_name("a combinator action", action)] = ensure_name(
                f"the decoder for action {action!r}", decoder
            )

        return dict(sorted(bindings.items()))

    @dx.model_validator()
    def _binary_decoders_own_the_merge(self) -> Self:
        """Require a merge exactly when binary decoder bindings exist."""
        if (self.merge is not None) != bool(self.decoders):
            msg = "a combinator family declares a merge exactly for binary decoders"
            raise ValueError(msg)

        return self

    def chart_sharing(self) -> SubmoduleDeclaration | None:
        """Return the chart publication selected by this merge, if any."""
        if self.merge is not Merge.CHART_COMBINE:
            return None

        return SubmoduleDeclaration(
            "combine", ModuleSharing(in_parameters=False, in_state_dict=False)
        )


class SpaceSpec(dx.Model, extra="forbid"):
    """Bind one represented sort to its encoder and decoders."""

    sort: str
    encoder: str
    identity: str
    combinators: CombinatorFamilySpec | None = None
    relation: SpaceRelationSpec | None = None

    @dx.validates("sort", "encoder", "identity")
    @classmethod
    def _binding_is_named(cls, value: str) -> str:
        """Return a nonempty sort or component role."""
        return ensure_name("a space binding", value)

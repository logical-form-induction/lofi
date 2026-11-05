"""MegaAttitude release metadata."""

from typing import Final, Literal

import didactic.api as dx

__all__ = ["MegaAttitudeRelationSpec", "MegaAttitudeSpec"]


class MegaAttitudeRelationSpec(dx.Model, extra="forbid"):
    """Declare one released response and its inference relation."""

    response: str
    relation: Literal["entails", "contradicts", "neutral"]

    @dx.validates("response", "relation")
    @classmethod
    def _text_is_present(cls, value: str) -> str:
        """Return a nonempty relation-table field."""
        if not value.strip():
            msg = "MegaAttitude relation fields cannot be empty"
            raise ValueError(msg)

        return value


_VERIDICALITY_RELATIONS: Final = (
    MegaAttitudeRelationSpec(response="no", relation="contradicts"),
    MegaAttitudeRelationSpec(response="maybe", relation="neutral"),
    MegaAttitudeRelationSpec(response="yes", relation="entails"),
)


class MegaAttitudeSpec(dx.Model, extra="forbid"):
    """Declare one participant-level MegaAttitude release."""

    source_id: str
    release_id: str
    language: str
    source_method: str
    spdx: str
    eprint_refs: tuple[str, ...] = ()
    veridicality_relations: tuple[MegaAttitudeRelationSpec, ...] = (
        _VERIDICALITY_RELATIONS
    )

    @dx.validates(
        "source_id",
        "release_id",
        "language",
        "source_method",
        "spdx",
    )
    @classmethod
    def _text_is_present(cls, value: str) -> str:
        """Return a nonempty source declaration."""
        if not value.strip():
            msg = "MegaAttitude source metadata cannot be empty"
            raise ValueError(msg)

        return value

    @dx.validates("eprint_refs")
    @classmethod
    def _references_are_present(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Return nonempty publication references."""
        if any(not value.strip() for value in values):
            msg = "MegaAttitude publication references cannot be empty"
            raise ValueError(msg)

        return values

    @dx.validates("veridicality_relations")
    @classmethod
    def _relations_are_bijective(
        cls, values: tuple[MegaAttitudeRelationSpec, ...]
    ) -> tuple[MegaAttitudeRelationSpec, ...]:
        """Return a nonempty one-to-one source-to-inference relation table."""
        sources = tuple(value.response for value in values)
        targets = tuple(value.relation for value in values)
        if (
            not values
            or len(set(sources)) != len(sources)
            or len(set(targets)) != len(targets)
        ):
            msg = "MegaAttitude veridicality relations must be a nonempty bijection"
            raise ValueError(msg)

        return values

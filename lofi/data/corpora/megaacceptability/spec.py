"""English MegaAcceptability codec settings."""

from typing import Self

import didactic.api as dx

__all__ = ["MegaAcceptabilitySpec"]


class MegaAcceptabilitySpec(dx.Model, extra="forbid"):
    """A MegaAcceptability release and its referenced resources."""

    source_id: str
    release_id: str
    spdx: str
    scale_min: int
    scale_max: int
    frame_refs: dict[str, str]
    verb_refs: dict[str, str]
    eprint_refs: tuple[str, ...] = ()

    @dx.validates("source_id", "release_id", "spdx")
    @classmethod
    def _text_is_present(cls, value: str) -> str:
        """Return a nonempty declaration."""
        if not value.strip():
            msg = "MegaAcceptability metadata cannot be empty"
            raise ValueError(msg)

        return value

    @dx.validates("frame_refs", "verb_refs")
    @classmethod
    def _references_are_complete(cls, refs: dict[str, str]) -> dict[str, str]:
        """Return a nonempty reference map without blank keys or values."""
        if not refs or any(
            not key.strip() or not value.strip() for key, value in refs.items()
        ):
            msg = "MegaAcceptability resource references cannot be empty"
            raise ValueError(msg)

        return refs

    @dx.model_validator()
    def _scale_is_ordered(self) -> Self:
        """Return a declaration with ordered scale bounds."""
        if self.scale_min > self.scale_max:
            msg = "MegaAcceptability scale minimum exceeds its maximum"
            raise ValueError(msg)

        return self

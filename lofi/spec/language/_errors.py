"""Refusals raised at the declared-language boundary."""

from typing import TYPE_CHECKING, final

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.typecheck import Constraint

__all__ = ["AnnotationError", "ConstantInferenceError", "LanguageError"]


class LanguageError(ValueError):
    """A language declaration or logical form that cannot be lowered."""


@final
class AnnotationError(LanguageError):
    """A logical form whose declared annotation disagrees with inference."""

    __slots__ = ("residual",)

    def __init__(self, residual: Sequence[Constraint]) -> None:
        """Initialize the refusal from its located annotation residuals."""
        self.residual = tuple(residual)
        locations = ", ".join(repr(item.path) for item in self.residual)
        super().__init__(f"logical-form annotation disagrees at {locations}")


@final
class ConstantInferenceError(LanguageError):
    """A constant whose uses do not determine one principal type scheme."""

    __slots__ = ("constant",)

    def __init__(self, constant: str, reason: str) -> None:
        """Initialize a named refusal for one inferred constant."""
        self.constant = constant
        super().__init__(f"cannot infer the type of constant {constant!r}: {reason}")

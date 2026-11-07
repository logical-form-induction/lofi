"""Lexicon errors."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lofi.lexicon._ports import Clamp

__all__ = ["ExhaustedBeamError", "LexiconError", "UnsatisfiableClampError"]


class LexiconError(Exception):
    """Reports a lexicon error."""


class UnsatisfiableClampError[T](LexiconError):
    """Reports a clamp that admits no entry, or a bracket no derivation omits.

    A clamp that requires a constituent fails where a position offers nothing it
    admits. A negative bracket, which requires none, fails where every derivation
    of the input brackets the span it excludes.

    Parameters
    ----------
    clamp : Clamp[T]
        Unsatisfied clamp.
    position : int
        Token position where the clamp failed.
    form : str
        Surface form at the position.
    offered : tuple[T, ...]
        Categories offered at the position.
    """

    __slots__ = ("clamp", "form", "offered", "position")

    def __init__(
        self,
        *,
        clamp: Clamp[T],
        position: int,
        form: str,
        offered: tuple[T, ...],
    ) -> None:
        self.clamp = clamp
        self.position = position
        self.form = form
        self.offered: tuple[T, ...] = offered

        start, end = clamp.span

        if clamp.require_constituent:
            demand = "forbids" if clamp.negate else "requires"
            wanted = sorted(repr(category) for category in clamp.category)

            if clamp.meaning is not None:
                wanted.append(repr(clamp.meaning))

            named = ", ".join(wanted) or "an entry"
            available = ", ".join(repr(category) for category in offered) or "nothing"
            msg = (
                f"the clamp on the span [{start}, {end}) {demand} {named}, and "
                f"position {position}, {form!r}, offers {available}"
            )

        else:
            msg = (
                f"the negative bracket on the span [{start}, {end}) excludes a span "
                f"that every derivation of the input brackets, at position "
                f"{position}, {form!r}"
            )

        super().__init__(msg)


class ExhaustedBeamError(LexiconError):
    """Reports a widening beyond the end of a beam.

    Parameters
    ----------
    position : int
        Token position whose beam is exhausted.
    width : int
        Number of available candidates.
    """

    __slots__ = ("position", "width")

    def __init__(self, *, position: int, width: int) -> None:
        self.position = position
        self.width = width

        if width == 0:
            admitted = "no candidate at all"

        elif width == 1:
            admitted = "its one candidate"

        else:
            admitted = f"all {width} of its candidates"

        msg = (
            f"position {position} already admits {admitted}, so there is no "
            f"next-best category to widen to"
        )

        super().__init__(msg)

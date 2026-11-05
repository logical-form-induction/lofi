"""Shape and bound guards shared by the arity-generic and scoped modules."""

from typing import TYPE_CHECKING, Final

from lofi._checks import ensure_non_negative

if TYPE_CHECKING:
    from lofi.nn._values import Encoded

__all__ = ["WIDTH_AXIS", "checked_state", "positive"]

WIDTH_AXIS: Final[str] = "width"
"""The feature axis of a decoder state."""


def positive(field: str, value: int) -> int:
    """Return a positive integer.

    Parameters
    ----------
    field : str
        The name to report in a message.
    value : int
        The number to check.

    Raises
    ------
    TypeError
        If the number is not an integer.
    ValueError
        If the number is zero or negative.
    """
    checked = ensure_non_negative(field, value)

    if checked == 0:
        msg = f"{field} is zero, so nothing is built there"
        raise ValueError(msg)

    return checked


def checked_state(state: Encoded, width: int, reader: str, /) -> Encoded:
    """Return a decoder state a module of one feature width may read.

    Parameters
    ----------
    state : Encoded
        The state to check.
    width : int
        The feature width the reader was built at.
    reader : str
        The noun a refusal names the reader by.

    Raises
    ------
    ValueError
        If its innermost axis is not the feature axis, or its feature axis is not the
        reader's own width.
    """
    if not state.axes or state.axes[-1] != WIDTH_AXIS:
        msg = (
            f"a decoder state runs over {WIDTH_AXIS!r} innermost, not over {state.axes}"
        )
        raise ValueError(msg)

    held = state.size(WIDTH_AXIS)

    if held != width:
        msg = f"this {reader} reads states {width} wide, not {held} wide"
        raise ValueError(msg)

    return state

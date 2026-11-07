"""Rule-label recovery from observed steps."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from lofi.judge._ports import Rule

__all__ = ["recover"]


def recover[T](
    rules: Iterable[Rule[T]], inputs: Sequence[T], output: T
) -> tuple[Rule[T], ...]:
    """Return rules that relate the premises to the conclusion.

    Parameters
    ----------
    rules : Iterable[Rule[T]]
        Rule inventory to search.
    inputs : Sequence[T]
        Premises in rule order.
    output : T
        Candidate conclusion.

    Returns
    -------
    tuple[Rule[T], ...]
        Matching rules in inventory order.
    """
    premises = tuple(inputs)

    return tuple(
        rule
        for rule in rules
        if "check" in rule.directions
        and rule.arity == len(premises)
        and rule.admits(premises, output)
    )

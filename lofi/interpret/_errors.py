"""Interpretation errors."""

__all__ = [
    "AmbiguousMeaningError",
    "CategoryWitnessRequiredError",
    "CyclicForestError",
]


class AmbiguousMeaningError(RuntimeError):
    """Report that a scalar reading reached several selected meanings."""

    __slots__ = ("alternatives", "rule")

    def __init__(self, rule: str, alternatives: int, /) -> None:
        self.rule = rule
        self.alternatives = alternatives

        msg = (
            f"rule {rule!r} resolves to {alternatives} selected semantic actions; "
            "a scalar meaning reading requires exactly one"
        )
        super().__init__(msg)


class CategoryWitnessRequiredError(RuntimeError):
    """Report that resolving an action requires concrete step categories."""

    __slots__ = ("rule",)

    def __init__(self, rule: str, /) -> None:
        self.rule = rule

        msg = (
            f"rule {rule!r} has category-conditioned semantic actions; "
            "resolving one requires a concrete derivation-step witness"
        )
        super().__init__(msg)


class CyclicForestError(Exception):
    """Reports a cycle in a packed forest.

    Parameters
    ----------
    item : str
        Rendering of the item reached from itself.
    """

    __slots__ = ("item",)

    def __init__(self, *, item: str) -> None:
        self.item = item

        msg = (
            f"{item} is reached from itself, so the forest beneath it has no "
            f"best-first order"
        )

        super().__init__(msg)

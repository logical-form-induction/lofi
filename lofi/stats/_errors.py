"""Statistical errors."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lofi.stats._values import Dependence

__all__ = ["DependenceError"]


class DependenceError(Exception):
    """Signals an invalid dependence regime for a correction.

    Parameters
    ----------
    correction : str
        The procedure that was asked.
    dependence : Dependence
        The regime it was asked to work under.
    valid_under : frozenset[Dependence]
        The regimes it does cover.
    """

    __slots__ = ("correction", "dependence", "valid_under")

    def __init__(
        self,
        *,
        correction: str,
        dependence: Dependence,
        valid_under: frozenset[Dependence],
    ) -> None:
        self.correction = correction
        self.dependence = dependence
        self.valid_under = valid_under

        covered = ", ".join(sorted(regime.name for regime in valid_under)) or "nothing"
        msg = (
            f"{correction} is valid under {covered} and was applied under "
            f"{dependence.name}"
        )

        super().__init__(msg)

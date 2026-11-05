"""The record of how a point's run seed was derived from its experiment's root.

Every point of an experiment takes the run seed ``derive_seed(root, "point:<id>")``
under the experiment's root seed. A run
manifest and a judgment-fit manifest both record that derivation, so a reader can
recompute the run seed from the root and the point and refuse a record whose seed
the two do not derive.

A data seed, the seed a corpus or a lexicon is drawn under, is declared either as
a fixed integer, so every point of an experiment draws the same data, or as
``derived``, so each point draws its own: ``derive_seed(run_seed, "data:<path>")``
under the point's run seed, named by the dotted path of the seed in the run
document.
"""

from typing import TYPE_CHECKING, Final, Self

import didactic.api as dx

from lofi._checks import ensure_name, ensure_non_negative
from lofi.spec._interpolation import derive_seed

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "DERIVED_SEED",
    "SeedDerivationRecord",
    "UnresolvedSeedError",
    "data_seed",
    "ensure_data_seed",
    "fixed_seed",
    "point_root",
    "resolved_seed",
]

DERIVED_SEED: Final = "derived"
"""The spelling of a data seed derived from the point a run executes for."""


class UnresolvedSeedError(ValueError):
    """Raised when data is drawn under a derived seed no point has resolved."""


def ensure_data_seed(field: str, value: int | str) -> int | str:
    """Return a data seed that is a nonnegative integer or the derived spelling.

    Parameters
    ----------
    field : str
        The name the error message gives the seed.
    value : int | str
        The declared seed.

    Raises
    ------
    ValueError
        If the seed is a negative integer or a string other than ``derived``.
    """
    if isinstance(value, str):
        if value != DERIVED_SEED:
            msg = f"{field} is a nonnegative integer or {DERIVED_SEED!r}, not {value!r}"
            raise ValueError(msg)

        return value

    return ensure_non_negative(field, value)


def fixed_seed(field: str, value: int | str) -> int:
    """Return a data seed that is fixed, refusing one still to be derived.

    Parameters
    ----------
    field : str
        The name the error message gives the seed.
    value : int | str
        A seed :func:`ensure_data_seed` accepted.

    Raises
    ------
    UnresolvedSeedError
        If the seed is derived and no point has resolved it.
    """
    if isinstance(value, str):
        msg = (
            f"{field} is derived from the point a run executes for, and no point "
            "has resolved it"
        )
        raise UnresolvedSeedError(msg)

    return value


def resolved_seed(value: int | str, derive: Callable[[str], int], path: str, /) -> int:
    """Return a fixed data seed as declared, or a derived one as derived for its path.

    Parameters
    ----------
    value : int | str
        A seed :func:`ensure_data_seed` accepted.
    derive : Callable[[str], int]
        The seed a point derives for a dotted path.
    path : str
        The dotted path of the seed in the run document.
    """
    return derive(path) if isinstance(value, str) else value


def data_seed(run_seed: int, path: str, /) -> int:
    """Return the data seed a run seed derives for a dotted path.

    Parameters
    ----------
    run_seed : int
        The point's run seed.
    path : str
        The dotted path of the seed in the run document.
    """
    return derive_seed(run_seed, f"data:{ensure_name('a data-seed path', path)}")


def point_root(experiment_root: int, point_id: str, /) -> int:
    """Return the run seed one point derives under an experiment's root seed.

    Parameters
    ----------
    experiment_root : int
        The experiment's nonnegative root seed.
    point_id : str
        The point's content-addressed identifier.

    Raises
    ------
    ValueError
        If the root is negative or the identifier is empty.
    """
    root = ensure_non_negative("an experiment root seed", experiment_root)
    name = ensure_name("a point identifier", point_id)

    return derive_seed(root, f"point:{name}")


class SeedDerivationRecord(dx.Model, extra="forbid"):
    """Records how a run seed was derived from its experiment's root seed.

    Attributes
    ----------
    experiment_root : int
        The experiment's nonnegative root seed.
    point_id : str
        The point the seed belongs to.
    seed : int
        The nonnegative sweep seed the point was expanded with.
    root : int
        The derived run seed.
    """

    experiment_root: int
    point_id: str
    seed: int
    root: int

    @dx.validates("experiment_root", "seed")
    @classmethod
    def _counts_from_zero(cls, value: int) -> int:
        """Return a nonnegative seed."""
        return ensure_non_negative("a recorded seed", value)

    @dx.validates("point_id")
    @classmethod
    def _point_is_named(cls, point_id: str) -> str:
        """Return a nonempty point identifier."""
        return ensure_name("a point identifier", point_id)

    @dx.model_validator()
    def _root_is_derived(self) -> Self:
        """Refuse a record whose run seed its root and point do not derive.

        Raises
        ------
        ValueError
            If the recorded run seed is not the one the experiment root derives
            for the point.
        """
        derived = point_root(self.experiment_root, self.point_id)

        if self.root != derived:
            msg = (
                f"the seed derivation records the root {self.root}, which the "
                f"experiment root {self.experiment_root} derives as {derived} for "
                f"the point {self.point_id!r}"
            )
            raise ValueError(msg)

        return self

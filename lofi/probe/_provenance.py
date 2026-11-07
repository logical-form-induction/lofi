"""Probing provenance records."""

from typing import TYPE_CHECKING, override

from lofi._checks import ensure_name, ensure_non_negative
from lofi.probe._json import a_count, a_name, an_object, field

if TYPE_CHECKING:
    from lofi._types import JsonValue

__all__ = ["Provenance", "provenance_from_json"]


class Provenance:
    """Records what produced a probing result, over what data, under what seed.

    Parameters
    ----------
    method : str
        The name of the estimator, generator or test that produced the result.
    dataset : str
        The handle of the data it ran over.
    seed : int | None
        The seed used for a draw, or None if no draw occurred.

    Raises
    ------
    TypeError
        If a field is given a value of the wrong type.
    ValueError
        If the method name or the dataset handle is empty, or the seed is negative.
    """

    __slots__ = ("_dataset", "_method", "_seed")

    def __init__(self, method: str, dataset: str, seed: int | None) -> None:
        self._method = ensure_name("a result's method", method)
        self._dataset = ensure_name("a result's dataset", dataset)
        self._seed = (
            None if seed is None else ensure_non_negative("a result's seed", seed)
        )

    @property
    def method(self) -> str:
        """Return the name of what produced the result."""
        return self._method

    @property
    def dataset(self) -> str:
        """Return the handle of the data it ran over."""
        return self._dataset

    @property
    def seed(self) -> int | None:
        """Return the seed a draw inside it ran under, if a draw ran."""
        return self._seed

    def to_json(self) -> dict[str, JsonValue]:
        """Return the three fields as a JSON object.

        Returns
        -------
        dict[str, JsonValue]
            The method, the dataset and the seed, under those names.
        """
        return {"method": self._method, "dataset": self._dataset, "seed": self._seed}

    def _identity(self) -> tuple[str, str, int | None]:
        """Return the three fields a provenance is equal on."""
        return (self._method, self._dataset, self._seed)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value records the same three fields.

        Parameters
        ----------
        other : object
            The value to compare against.
        """
        if type(other) is not Provenance:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the three fields the provenance is equal on."""
        return hash((Provenance, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this provenance."""
        return f"Provenance({self._method!r}, {self._dataset!r}, seed={self._seed!r})"


def provenance_from_json(value: JsonValue) -> Provenance:
    """Return the provenance a JSON object describes.

    Parameters
    ----------
    value : JsonValue
        The object ``Provenance.to_json`` produced.

    Returns
    -------
    Provenance
        The deserialized provenance.

    Raises
    ------
    TypeError
        If the value is not a JSON object, or a field is of the wrong type.
    ValueError
        If a field is missing, empty or negative.
    """
    fields = an_object(value, "provenance")
    seed = field(fields, "seed")

    return Provenance(
        a_name(fields, "method"),
        a_name(fields, "dataset"),
        None if seed is None else a_count(fields, "seed"),
    )

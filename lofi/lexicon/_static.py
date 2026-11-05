"""Static lexical sources."""

from math import exp, inf, log, log1p
from typing import TYPE_CHECKING, override

from lofi._checks import ensure_finite, ensure_name
from lofi.lexicon._ports import Clamp, Entry

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = ["StaticLexicon"]


def _log_add(a: float, b: float) -> float:
    """Return the log-sum-exp of two finite log masses.

    Parameters
    ----------
    a : float
        First log mass.
    b : float
        Second log mass.

    Returns
    -------
    float
        Logarithm of the summed masses.
    """
    if a < b:
        a, b = b, a

    return a + log1p(exp(b - a))


def _log_total(masses: Iterable[float]) -> float:
    """Return the log-sum-exp of finite log masses.

    Parameters
    ----------
    masses : Iterable[float]
        Nonempty log-mass sequence.

    Returns
    -------
    float
        Logarithm of the total mass.
    """
    kept = tuple(masses)
    top = max(kept)

    return top + log(sum(exp(mass - top) for mass in kept))


class StaticLexicon[T]:
    """Lexical entries from a fixed table.

    Parameters
    ----------
    entries : Iterable[Entry[T]]
        Entries used to build the form-indexed table.

    Raises
    ------
    TypeError
        If a form or weight has the wrong type.
    ValueError
        If a form is empty or a weight is not finite.
    """

    __slots__ = ("_forms", "_masses", "_size", "_table")

    def __init__(self, entries: Iterable[Entry[T]]) -> None:
        pool: dict[T, T] = {}
        masses: dict[str, dict[tuple[T, T | None], float]] = {}

        for entry in entries:
            form = ensure_name("entry form", entry.form)
            weight = ensure_finite("entry weight", entry.weight)
            meaning = entry.meaning
            slot = (
                pool.setdefault(entry.category, entry.category),
                None if meaning is None else pool.setdefault(meaning, meaning),
            )
            held = masses.setdefault(form, {})
            standing = held.get(slot)
            held[slot] = weight if standing is None else _log_add(standing, weight)

        table: dict[str, tuple[Entry[T], ...]] = {}
        totals: dict[str, float] = {}

        for form, held in masses.items():
            total = _log_total(held.values())
            totals[form] = total
            table[form] = tuple(
                Entry(form, category, meaning, mass - total)
                for (category, meaning), mass in held.items()
            )

        self._table = table
        self._masses = totals
        self._forms: tuple[str, ...] = tuple(sorted(table))
        self._size = sum(len(found) for found in table.values())

    @property
    def is_re_callable(self) -> bool:
        """Return whether the enumeration may be read more than once, which it may."""
        return True

    @property
    def clamps(self) -> tuple[Clamp[T], ...]:
        """Return no clamps."""
        return ()

    @property
    def forms(self) -> tuple[str, ...]:
        """Return the surface forms the table holds, in ascending order."""
        return self._forms

    def entries(self, form: str) -> tuple[Entry[T], ...]:
        """Return the entries admitted by a form.

        Parameters
        ----------
        form : str
            Surface form to look up.

        Returns
        -------
        tuple[Entry[T], ...]
            Entries with normalized log weights, or an empty tuple if absent.
        """
        return self._table.get(form, ())

    def log_mass(self, form: str) -> float:
        """Return the log of the total weight a form's entries were built with.

        Parameters
        ----------
        form : str
            Surface form.

        Returns
        -------
        float
            Log-sum-exp of the weights the form's entries were built with,
            before they were normalized into a distribution over the form's
            categories, or negative infinity if the form is absent.
        """
        return self._masses.get(form, -inf)

    def log_probability(self, form: str, category: T) -> float:
        """Return the log probability of a category given a form.

        Parameters
        ----------
        form : str
            Surface form.
        category : T
            Category to score.

        Returns
        -------
        float
            Combined log probability, or negative infinity if absent.
        """
        found = tuple(
            entry.weight
            for entry in self._table.get(form, ())
            if entry.category == category
        )

        if not found:
            return -inf

        return _log_total(found)

    def axioms(
        self, tokens: Sequence[str], /
    ) -> tuple[tuple[int, Entry[T], float], ...]:
        """Return every entry admitted by the tokens.

        Parameters
        ----------
        tokens : Sequence[str]
            Surface tokens in position order.

        Returns
        -------
        tuple[tuple[int, Entry[T], float], ...]
            Position, entry, and normalized log weight for each axiom.
        """
        return tuple(
            (position, entry, entry.weight)
            for position, token in enumerate(tokens)
            for entry in self._table.get(token, ())
        )

    def __len__(self) -> int:
        """Return how many entries the table holds."""
        return self._size

    @override
    def __repr__(self) -> str:
        """Return a rendering naming the form count and the entry count."""
        return f"StaticLexicon(forms={len(self._forms)}, entries={self._size})"

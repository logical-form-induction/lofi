"""Protocols the synthetic package requires of what it reads."""

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from lofi.data.synthetic._semantics import Meaning
    from lofi.lexicon import Entry

__all__ = ["SemanticSource", "SourceLexicon"]


class SourceLexicon[T](Protocol):
    """A lexical source whose forms and their categories can be enumerated.

    The bare lexical-source protocol admits entries at the positions of a
    token sequence, which says what a given string may bear and not what the
    lexicon holds. Estimating a joint needs the enumeration this protocol
    adds: the forms, the entries of each, and the mass each form was built
    with before its entries were normalized over it.
    """

    @property
    def forms(self) -> tuple[str, ...]:
        """Return the surface forms the lexicon holds."""

    def entries(self, form: str) -> tuple[Entry[T], ...]:
        """Return the entries a form admits.

        Parameters
        ----------
        form : str
            The surface form to look up.

        Returns
        -------
        tuple[Entry[T], ...]
            One entry per category the form bears, each weighted by the log
            probability of that category given the form, or nothing when the
            lexicon does not hold the form.
        """

    def log_mass(self, form: str) -> float:
        """Return the log of the token mass a form was built with.

        Parameters
        ----------
        form : str
            The surface form to look up.

        Returns
        -------
        float
            The log mass, or negative infinity when the lexicon does not hold
            the form.
        """


class SemanticSource(Protocol):
    """A read lexicon whose items carry logical forms.

    A syntactic source is read form by form, since a form is what bears a
    category. A semantic source is read item by item, since it is the item —
    a form at a category — that bears a logical form, and two items of one
    form may bear logical forms of different classes.
    """

    @property
    def items(self) -> tuple[tuple[str, str], ...]:
        """Return the category text and surface form of every item, in read order."""

    def meaning(self, item: tuple[str, str]) -> Meaning | None:
        """Return the logical form an item bears.

        Parameters
        ----------
        item : tuple[str, str]
            The category text and surface form to look up.

        Returns
        -------
        Meaning | None
            The form, with the class and the logical constant it is built
            on, or nothing when the source holds no such item.
        """

    def log_mass(self, item: tuple[str, str]) -> float:
        """Return the log of the token mass an item was read with.

        Parameters
        ----------
        item : tuple[str, str]
            The category text and surface form to look up.

        Returns
        -------
        float
            The log mass, or negative infinity when the source holds no such
            item.
        """

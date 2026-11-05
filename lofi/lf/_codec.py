"""Term-codec construction for a lowered language."""

from typing import TYPE_CHECKING, Protocol

from lofi.terms import CompleteTree, OperatorIndex, PreorderArityLayout, TermCodec

if TYPE_CHECKING:
    from lofi.terms import TermStore

__all__ = ["term_codec"]


class _Language(Protocol):
    """A lowered language that publishes its one store."""

    @property
    def store(self) -> TermStore:
        """Return the language store."""
        ...


def term_codec(
    language: _Language,
    layout: CompleteTree | PreorderArityLayout,
    /,
) -> TermCodec:
    """Build a term codec over one lowered language's complete signature.

    Parameters
    ----------
    language : _Language
        The one-store language whose operators the codec indexes.
    layout : CompleteTree | PreorderArityLayout
        The caller-owned term layout.

    Returns
    -------
    TermCodec
        A codec over the language signature and supplied layout.
    """
    return TermCodec(OperatorIndex(language.store.signature), layout)

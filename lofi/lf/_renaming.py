"""Renaming a hypothesis's argument constants to a premise's."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Never, final, override

from lofi._checks import ensure_name

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.terms import Term, TermStore

__all__ = ["ArgumentRenaming"]


def _renamed(store: TermStore, term: Term, names: Mapping[str, str]) -> Term:
    """Return a term with its leaf constants renamed.

    Parameters
    ----------
    store : TermStore
        The store the rebuilt nodes are interned in.
    term : Term
        The term to walk.
    names : Mapping[str, str]
        The new name of each renamed constant, keyed by its current name.

    Returns
    -------
    Term
        The same node where nothing beneath it is renamed.
    """
    op = term.op

    if store.signature.is_constant(op) and term.payload is not None:
        target = names.get(store.signature.vocabulary(op.sort).name(term.payload))

        return term if target is None else store.const(op.sort, target)

    if not term.args:
        return term

    arguments = tuple(_renamed(store, argument, names) for argument in term.args)

    if arguments == term.args:
        return term

    return store.op(op, *arguments, payload=term.payload)


@final
class ArgumentRenaming:
    """One way a hypothesis's argument constants correspond to a premise's.

    Each hypothesis constant maps to at most one premise constant. The mapping stores
    names, so it applies to terms from any store.

    Parameters
    ----------
    correspondences : Mapping[str, str]
        The premise constant each hypothesis constant is renamed to, by name.

    Raises
    ------
    TypeError
        If a name is not a string.
    ValueError
        If a name is empty.
    """

    __slots__ = ("_correspondences",)

    def __init__(self, correspondences: Mapping[str, str]) -> None:
        held = {
            ensure_name("a hypothesis constant", source): ensure_name(
                "a premise constant", target
            )
            for source, target in correspondences.items()
        }
        object.__setattr__(self, "_correspondences", MappingProxyType(held))

    @classmethod
    def identity(cls) -> ArgumentRenaming:
        """Return the renaming that leaves every constant as it is."""
        return cls({})

    @property
    def correspondences(self) -> Mapping[str, str]:
        """Return the premise constant each hypothesis constant is renamed to."""
        return self._correspondences

    def apply(self, store: TermStore, term: Term, /) -> Term:
        """Return a hypothesis form with its argument constants renamed.

        Parameters
        ----------
        store : TermStore
            The store the form is interned in, which the renamed form is
            interned in too.
        term : Term
            The hypothesis form.

        Returns
        -------
        Term
            The form with each named constant replaced, or the very same node
            where it names none of them.
        """
        if not self._correspondences:
            return term

        return _renamed(store, term, self._correspondences)

    def __len__(self) -> int:
        """Return how many hypothesis constants are renamed."""
        return len(self._correspondences)

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "an argument renaming is immutable"

        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "an argument renaming is immutable"

        raise AttributeError(msg)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value renames the same constants the same way."""
        if type(other) is not ArgumentRenaming:
            return NotImplemented

        return dict(self._correspondences) == dict(other._correspondences)

    @override
    def __hash__(self) -> int:
        """Return a hash over the correspondences."""
        return hash((ArgumentRenaming, frozenset(self._correspondences.items())))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this renaming."""
        return f"ArgumentRenaming({dict(self._correspondences)!r})"

"""Label membership over a comparison signature."""

from typing import TYPE_CHECKING, Never, Protocol, Self, override

from lofi.terms import TermError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.terms import Signature

__all__ = ["GrammarInventory", "SignatureLabels"]


class _Syntax[Store, Parsed](Protocol):
    """Parse source labels into one store's symbolic values."""

    def parse(self, store: Store, source: str, /) -> Parsed:
        """Return the symbolic value named by source text."""
        ...


class _Rules[Parsed, Recovered](Protocol):
    """Recover rules over parsed symbolic values."""

    def recover(
        self, inputs: Sequence[Parsed], output: Parsed, /
    ) -> tuple[Recovered, ...]:
        """Return rules licensing one exact step."""
        ...


class _Grammar[Parsed, Recovered](Protocol):
    """Expose the compiled rule inventory."""

    @property
    def rules(self) -> _Rules[Parsed, Recovered]:
        """Return the rules that recover exact steps."""
        ...


class _GrammarSource[Store, Parsed, Recovered](Protocol):
    """Expose the syntax, store, and rules that interpret category text."""

    @property
    def syntax(self) -> _Syntax[Store, Parsed]:
        """Return the category syntax."""
        ...

    @property
    def store(self) -> Store:
        """Return the parsed-term store."""
        ...

    @property
    def grammar(self) -> _Grammar[Parsed, Recovered]:
        """Return the compiled grammar kernel."""
        ...


class GrammarInventory[Store, Parsed, Recovered]:
    """Admit and recover category text through one compiled grammar.

    Parameters
    ----------
    grammar : _GrammarSource[Store, Parsed, Recovered]
        The grammar whose syntax and rule inventory interpret the labels.
    """

    __slots__ = ("_grammar",)

    def __init__(self, grammar: _GrammarSource[Store, Parsed, Recovered]) -> None:
        object.__setattr__(self, "_grammar", grammar)

    @property
    def grammar(self) -> _GrammarSource[Store, Parsed, Recovered]:
        """Return the compiled grammar that owns this inventory."""
        return self._grammar

    def contains(self, label: str, /) -> bool:
        """Return whether the grammar syntax parses a category label."""
        try:
            self._grammar.syntax.parse(self._grammar.store, label)
        except ValueError, TermError:
            return False
        return True

    def recover(
        self,
        inputs: Sequence[str],
        output: str,
        /,
    ) -> tuple[Recovered, ...]:
        """Return rules relating the parsed premises and conclusion."""
        grammar = self._grammar
        try:
            held_inputs = tuple(
                grammar.syntax.parse(grammar.store, label) for label in inputs
            )
            held_output = grammar.syntax.parse(grammar.store, output)
        except ValueError, TermError:
            return ()
        return grammar.grammar.rules.recover(held_inputs, held_output)

    def __copy__(self) -> Self:
        """Return this immutable value."""
        return self

    def __deepcopy__(self, memo: dict[int, Self]) -> Self:
        """Return this immutable value."""
        del memo
        return self

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "a grammar inventory is immutable"
        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "a grammar inventory is immutable"
        raise AttributeError(msg)


class SignatureLabels:
    """Label membership over the operators a signature declares.

    Parameters
    ----------
    signature : Signature
        The signature whose operator names are the admitted labels.
    """

    __slots__ = ("_names", "_signature")

    def __init__(self, signature: Signature) -> None:
        object.__setattr__(self, "_signature", signature)
        object.__setattr__(
            self,
            "_names",
            frozenset(operator.name for operator in signature.operators),
        )

    @property
    def signature(self) -> Signature:
        """Return the signature the labels are read against."""
        return self._signature

    def contains(self, label: str, /) -> bool:
        """Return whether the signature declares an operator named by a label.

        Parameters
        ----------
        label : str
            The label to look up.
        """
        return label in self._names

    def __copy__(self) -> Self:
        """Return this immutable value."""
        return self

    def __deepcopy__(self, memo: dict[int, Self]) -> Self:
        """Return this immutable value."""
        del memo
        return self

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "a signature's labels are immutable"
        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "a signature's labels are immutable"
        raise AttributeError(msg)

    @override
    def __reduce__(self) -> tuple[type[Self], tuple[Signature]]:
        """Return the reconstruction call."""
        return type(self), (self._signature,)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value reads labels against the same signature."""
        if type(other) is not SignatureLabels:
            return NotImplemented

        return self._signature == other._signature

    @override
    def __hash__(self) -> int:
        """Return a hash over the signature."""
        return hash((SignatureLabels, self._signature))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds these labels."""
        return f"SignatureLabels({self._signature!r})"

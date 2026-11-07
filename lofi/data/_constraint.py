"""Token- and type-scoped span constraints."""

from typing import TYPE_CHECKING, Final, override

from lofi._checks import ensure_non_negative
from lofi.data._values import TokenSpan

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

    from lofi.data._derivation import Derivation
    from lofi.data._values import Token

__all__ = [
    "ConstraintScope",
    "SpanConstraint",
    "SpanConstraints",
    "TokenScope",
    "TypeScope",
    "occurrences",
]


class TokenScope:
    """Constraint scope over one span occurrence."""

    __slots__ = ()

    @property
    def form(self) -> None:
        """Return None because a token-level scope does not bind a form."""
        return None

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is also the token-level scope."""
        if type(other) is not TokenScope:
            return NotImplemented

        return True

    @override
    def __hash__(self) -> int:
        """Return the shared hash for token-level scopes."""
        return hash(TokenScope)

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this scope."""
        return "TokenScope()"


class TypeScope:
    """Constraint scope over every occurrence of a form.

    Parameters
    ----------
    form : tuple[str, ...]
        Surface words in order.

    Raises
    ------
    ValueError
        If the form is empty.
    """

    __slots__ = ("_form",)

    def __init__(self, form: tuple[str, ...]) -> None:
        if not form:
            msg = "a type-level scope names a form of no words"
            raise ValueError(msg)

        self._form = form

    @property
    def form(self) -> tuple[str, ...]:
        """Return the words the scope binds, in order."""
        return self._form

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value binds the same form."""
        if type(other) is not TypeScope:
            return NotImplemented

        return self._form == other._form

    @override
    def __hash__(self) -> int:
        """Return a hash over the form."""
        return hash((TypeScope, self._form))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this scope."""
        return f"TypeScope({self._form!r})"


type ConstraintScope = TokenScope | TypeScope
"""Constraint scope over one occurrence."""


_TOKEN_LEVEL: Final = TokenScope()
"""The scope for one occurrence without a named form."""


class SpanConstraint[L: Hashable]:
    """The labels permitted on a span.

    Parameters
    ----------
    span : TokenSpan
        Constrained token span.
    permitted : frozenset[L]
        Permitted labels.
    scope : ConstraintScope
        Token-level or form-level scope.

    Raises
    ------
    ValueError
        If the span or label set is empty, or if the form width differs from
        the span width.
    """

    __slots__ = ("_permitted", "_scope", "_span")

    def __init__(
        self,
        span: TokenSpan,
        permitted: frozenset[L],
        scope: ConstraintScope = _TOKEN_LEVEL,
    ) -> None:
        if span.length == 0:
            msg = f"a constraint covers no token: [{span.start}, {span.end})"
            raise ValueError(msg)

        if not permitted:
            msg = (
                f"a constraint on [{span.start}, {span.end}) permits no label, "
                "so no analysis of the span satisfies it"
            )
            raise ValueError(msg)

        form = scope.form

        if form is not None and len(form) != span.length:
            msg = (
                f"a constraint anchored at [{span.start}, {span.end}) binds the "
                f"form {list(form)}, which is {len(form)} words wide"
            )
            raise ValueError(msg)

        self._span = span
        self._permitted = permitted
        self._scope = scope

    @property
    def span(self) -> TokenSpan:
        """Return the token positions the constraint names."""
        return self._span

    @property
    def permitted(self) -> frozenset[L]:
        """Return the labels the span may bear."""
        return self._permitted

    @property
    def scope(self) -> ConstraintScope:
        """Return the constraint's token or form scope."""
        return self._scope

    def at(self, span: TokenSpan) -> SpanConstraint[L]:
        """Return this constraint's token-level instance at one span.

        Parameters
        ----------
        span : TokenSpan
            Span for the instance.

        Returns
        -------
        SpanConstraint[L]
            A token-level constraint with the same permitted labels.
        """
        return SpanConstraint(span, self._permitted, _TOKEN_LEVEL)

    def _identity(self) -> tuple[TokenSpan, frozenset[L], ConstraintScope]:
        """Return the three fields a constraint is equal on."""
        return (self._span, self._permitted, self._scope)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value constrains the same span the same way."""
        if type(other) is not SpanConstraint:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the span, the permitted set and the scope."""
        return hash((SpanConstraint, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this constraint."""
        return f"SpanConstraint({self._span!r}, {self._permitted!r}, {self._scope!r})"


def occurrences(form: tuple[str, ...], tokens: Sequence[Token]) -> tuple[int, ...]:
    """Return every position a form begins at in a token sequence.

    Parameters
    ----------
    form : tuple[str, ...]
        Surface words to find.
    tokens : Sequence[Token]
        The sequence to search.

    Returns
    -------
    tuple[int, ...]
        Matching start positions, including overlapping matches.
    """
    surfaces = tuple(token.surface for token in tokens)

    return tuple(
        start
        for start in range(len(surfaces) - len(form) + 1)
        if surfaces[start : start + len(form)] == form
    )


class SpanConstraints[L: Hashable]:
    """The span constraints for one item.

    Parameters
    ----------
    constraints : tuple[SpanConstraint[L], ...]
        Constraints in source order.
    token_count : int
        Number of tokens in the item.

    Raises
    ------
    TypeError
        If the token count is not an integer.
    ValueError
        If the token count is negative or a constraint exceeds it.
    """

    __slots__ = ("_constraints", "_token_count")

    def __init__(
        self,
        constraints: tuple[SpanConstraint[L], ...],
        token_count: int,
    ) -> None:
        ensure_non_negative("a constrained item's token count", token_count)

        for constraint in constraints:
            span = constraint.span

            if span.end > token_count:
                msg = (
                    f"a constraint names [{span.start}, {span.end}), which runs "
                    f"past the {token_count} tokens of the item it constrains"
                )
                raise ValueError(msg)

        self._constraints = constraints
        self._token_count = token_count

    @classmethod
    def from_positions(cls, permitted: Sequence[frozenset[L]]) -> SpanConstraints[L]:
        """Return one width-one constraint per token.

        Parameters
        ----------
        permitted : Sequence[frozenset[L]]
            Permitted labels by token position.

        Returns
        -------
        SpanConstraints[L]
            One token-level constraint per position.

        Raises
        ------
        ValueError
            If a position permits no label.
        """
        return cls(
            tuple(
                SpanConstraint(TokenSpan(position, position + 1), labels)
                for position, labels in enumerate(permitted)
            ),
            len(permitted),
        )

    @classmethod
    def from_derivation(cls, derivation: Derivation[L]) -> SpanConstraints[L]:
        """Return one constraint per derivation node.

        Parameters
        ----------
        derivation : Derivation[L]
            Gold derivation.

        Returns
        -------
        SpanConstraints[L]
            One constraint per node, in node id order.
        """
        nodes = derivation.nodes

        return cls(
            tuple(
                SpanConstraint(nodes[node_id].span, frozenset({nodes[node_id].label}))
                for node_id in sorted(nodes)
            ),
            derivation.span.end,
        )

    @property
    def constraints(self) -> tuple[SpanConstraint[L], ...]:
        """Return what the corpus constrains, in the order it recorded them."""
        return self._constraints

    @property
    def token_count(self) -> int:
        """Return how many tokens the constrained item holds."""
        return self._token_count

    def instances(self, tokens: Sequence[Token]) -> tuple[SpanConstraint[L], ...]:
        """Return the token-level constraints this arm imposes on a token sequence.

        Parameters
        ----------
        tokens : Sequence[Token]
            Token sequence used to locate form-level constraints.

        Returns
        -------
        tuple[SpanConstraint[L], ...]
            Token-level instances in constraint and position order.
        """
        expanded: list[SpanConstraint[L]] = []

        for constraint in self._constraints:
            form = constraint.scope.form

            if form is None:
                if constraint.span.end <= len(tokens):
                    expanded.append(constraint)
            else:
                expanded.extend(
                    constraint.at(TokenSpan(start, start + len(form)))
                    for start in occurrences(form, tokens)
                )

        return tuple(expanded)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value constrains the same item the same way."""
        if type(other) is not SpanConstraints:
            return NotImplemented

        return (
            self._constraints == other._constraints
            and self._token_count == other._token_count
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the constraints and the item's token count."""
        return hash((SpanConstraints, self._constraints, self._token_count))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this arm."""
        return f"SpanConstraints({self._constraints!r}, {self._token_count!r})"

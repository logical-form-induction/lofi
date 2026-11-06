"""Values used by logical form type checking."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, final, override

from lofi._checks import ensure_name, ensure_non_negative
from lofi._frozen import FrozenValue
from lofi.terms import substitute

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi.terms import OpName, Path, Pattern, Term, TermStore

__all__ = [
    "Constraint",
    "Scheme",
    "TypingDirection",
    "TypingFailureKind",
    "TypingResult",
    "TypingRule",
]


type TypingDirection = Literal["up", "down", "both"]
"""Direction in which a soft typing constraint is applied."""

type TypingFailureKind = Literal["unification", "annotation", "infinite_type"]
"""Reason a typing constraint remains unsatisfied."""


def _path(path: Sequence[int]) -> Path:
    """Return a checked path."""
    return tuple(
        ensure_non_negative(f"path[{at}]", position) for at, position in enumerate(path)
    )


@final
class TypingRule(FrozenValue):
    """Type pattern for one logical form operator."""

    _noun = "typing value"

    __slots__ = (
        "_binds",
        "_direction",
        "_enforced",
        "_op",
        "_premises",
        "_result",
    )

    def __init__(
        self,
        op: OpName,
        binds: Sequence[Pattern | None],
        premises: Sequence[Pattern],
        result: Pattern,
        enforced: Sequence[int] | None = None,
        direction: TypingDirection = "both",
    ) -> None:
        """Initialize the rule."""
        held_binds = tuple(binds)
        held_premises = tuple(premises)

        if len(held_binds) != len(held_premises):
            msg = (
                f"typing rule {op!r} has {len(held_binds)} binder entries for "
                f"{len(held_premises)} premises"
            )
            raise ValueError(msg)

        if direction not in ("up", "down", "both"):
            msg = f"typing rule {op!r} has unknown direction {direction!r}"
            raise ValueError(msg)

        held_enforced = (
            None
            if enforced is None
            else tuple(
                ensure_non_negative(f"enforced[{at}]", position)
                for at, position in enumerate(enforced)
            )
        )

        if held_enforced is not None:
            if len(set(held_enforced)) != len(held_enforced):
                msg = f"typing rule {op!r} enforces a premise more than once"
                raise ValueError(msg)

            if any(position >= len(held_premises) for position in held_enforced):
                msg = (
                    f"typing rule {op!r} enforces a premise outside its "
                    f"{len(held_premises)} positions"
                )
                raise ValueError(msg)

        object.__setattr__(self, "_op", ensure_name("op", op))
        object.__setattr__(self, "_binds", held_binds)
        object.__setattr__(self, "_premises", held_premises)
        object.__setattr__(self, "_result", result)
        object.__setattr__(self, "_enforced", held_enforced)
        object.__setattr__(self, "_direction", direction)

    @property
    def op(self) -> OpName:
        """Return the logical form operator name."""
        return self._op

    @property
    def binds(self) -> tuple[Pattern | None, ...]:
        """Return the type introduced under each premise."""
        return self._binds

    @property
    def premises(self) -> tuple[Pattern, ...]:
        """Return the argument type patterns."""
        return self._premises

    @property
    def result(self) -> Pattern:
        """Return the result type pattern."""
        return self._result

    @property
    def enforced(self) -> tuple[int, ...] | None:
        """Return the premises enforced by the soft rule, or None for all."""
        return self._enforced

    @property
    def direction(self) -> TypingDirection:
        """Return the soft constraint direction."""
        return self._direction

    def _identity(
        self,
    ) -> tuple[
        OpName,
        tuple[Pattern | None, ...],
        tuple[Pattern, ...],
        Pattern,
        tuple[int, ...] | None,
        TypingDirection,
    ]:
        """Return fields in comparison order."""
        return (
            self._op,
            self._binds,
            self._premises,
            self._result,
            self._enforced,
            self._direction,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another rule has the same fields."""
        if type(other) is not TypingRule:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the fields."""
        return hash((TypingRule, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call."""
        return (
            f"TypingRule({self._op!r}, {self._binds!r}, {self._premises!r}, "
            f"{self._result!r}, enforced={self._enforced!r}, "
            f"direction={self._direction!r})"
        )


@final
class Constraint(FrozenValue):
    """Typed equality constraint at a source path."""

    _noun = "typing value"

    __slots__ = ("_kind", "_left", "_path", "_right")

    def __init__(
        self,
        kind: TypingFailureKind,
        left: Term,
        right: Term,
        path: Sequence[int],
    ) -> None:
        """Initialize a located constraint."""
        if kind not in ("unification", "annotation", "infinite_type"):
            msg = f"unknown typing failure kind: {kind!r}"
            raise ValueError(msg)

        object.__setattr__(self, "_kind", kind)
        object.__setattr__(self, "_left", left)
        object.__setattr__(self, "_right", right)
        object.__setattr__(self, "_path", _path(path))

    @property
    def kind(self) -> TypingFailureKind:
        """Return why the constraint may remain unsatisfied."""
        return self._kind

    @property
    def left(self) -> Term:
        """Return the left type."""
        return self._left

    @property
    def right(self) -> Term:
        """Return the right type."""
        return self._right

    @property
    def path(self) -> Path:
        """Return the source node path."""
        return self._path

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another constraint has the same fields."""
        if type(other) is not Constraint:
            return NotImplemented

        return (
            self._kind == other._kind
            and self._left == other._left
            and self._right == other._right
            and self._path == other._path
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the fields."""
        return hash((Constraint, self._kind, self._left, self._right, self._path))

    @override
    def __repr__(self) -> str:
        """Return the constructor call."""
        return (
            f"Constraint({self._kind!r}, {self._left!r}, {self._right!r}, "
            f"{self._path!r})"
        )


@final
class Scheme(FrozenValue):
    """Type term with explicitly quantified free atoms."""

    _noun = "typing value"

    __slots__ = ("_body", "_quantified")

    def __init__(self, body: Term, quantified: Sequence[int] = ()) -> None:
        """Initialize a type scheme."""
        held = tuple(
            ensure_non_negative(f"quantified[{at}]", atom)
            for at, atom in enumerate(quantified)
        )

        if len(set(held)) != len(held):
            msg = "a scheme quantifies a free atom more than once"
            raise ValueError(msg)

        if any(not body.free & (1 << atom) for atom in held):
            msg = "a scheme quantifies an atom absent from its body"
            raise ValueError(msg)

        object.__setattr__(self, "_body", body)
        object.__setattr__(self, "_quantified", held)

    @property
    def body(self) -> Term:
        """Return the principal type term."""
        return self._body

    @property
    def quantified(self) -> tuple[int, ...]:
        """Return the free atom indices bound by the scheme."""
        return self._quantified

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another scheme has the same fields."""
        if type(other) is not Scheme:
            return NotImplemented

        return self._body == other._body and self._quantified == other._quantified

    @override
    def __hash__(self) -> int:
        """Return a hash over the fields."""
        return hash((Scheme, self._body, self._quantified))

    @override
    def __repr__(self) -> str:
        """Return the constructor call."""
        return f"Scheme({self._body!r}, {self._quantified!r})"


def _defaulted(
    term: Term,
    quantified: tuple[int, ...],
    store: TermStore,
    ground: Term,
) -> Term:
    """Return a term with quantified atoms replaced."""
    replaced = term

    for atom in quantified:
        replaced = substitute(store, replaced, atom, ground)

    return replaced


def _ground(store: TermStore, ground: Term) -> Term:
    """Return a closed ground term interned by the supplied store."""
    if not ground.is_closed or ground.free:
        msg = "ground must contain no free or open variables"
        raise ValueError(msg)

    rebuilt: dict[Term, Term] = {}

    def intern(term: Term) -> Term:
        """Intern one ground subtree in the target store."""
        known = rebuilt.get(term)

        if known is not None:
            return known

        store.signature.erase(term.op)
        op = store.signature.by_name(term.op.name)
        arguments = tuple(intern(child) for child in term.args)
        held = store.op(op, *arguments, payload=term.payload)
        rebuilt[term] = held

        return held

    return intern(ground)


@final
class TypingResult(FrozenValue):
    """Total typing judgment for one logical form."""

    _noun = "typing value"

    __slots__ = ("_assignment", "_residual", "_scheme")

    def __init__(
        self,
        assignment: Mapping[Path, Term],
        residual: Sequence[Constraint],
        scheme: Scheme | None,
    ) -> None:
        """Initialize a typing result."""
        checked = {_path(path): term for path, term in assignment.items()}
        object.__setattr__(self, "_assignment", MappingProxyType(checked))
        object.__setattr__(self, "_residual", tuple(residual))
        object.__setattr__(self, "_scheme", scheme)

    @property
    def assignment(self) -> Mapping[Path, Term]:
        """Return inferred types keyed by logical form node path."""
        return self._assignment

    @property
    def residual(self) -> tuple[Constraint, ...]:
        """Return the unsatisfied located constraints."""
        return self._residual

    @property
    def scheme(self) -> Scheme | None:
        """Return the principal root scheme, or None when unavailable."""
        return self._scheme

    @property
    def is_well_typed(self) -> bool:
        """Return whether the residual is empty."""
        return not self._residual

    def defaulted(self, store: TermStore, ground: Term) -> TypingResult:
        """Replace quantified type variables with a ground type."""
        scheme = self._scheme

        if scheme is None or not scheme.quantified:
            return self

        quantified = scheme.quantified
        held_ground = _ground(store, ground)
        assignment = {
            path: _defaulted(term, quantified, store, held_ground)
            for path, term in self._assignment.items()
        }
        body = _defaulted(scheme.body, quantified, store, held_ground)

        return TypingResult(assignment, self._residual, Scheme(body))

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another result has the same fields."""
        if type(other) is not TypingResult:
            return NotImplemented

        return (
            self._assignment == other._assignment
            and self._residual == other._residual
            and self._scheme == other._scheme
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the fields."""
        return hash(
            (
                TypingResult,
                tuple(sorted(self._assignment.items())),
                self._residual,
                self._scheme,
            )
        )

    @override
    def __repr__(self) -> str:
        """Return the constructor call."""
        return (
            f"TypingResult({dict(self._assignment)!r}, {self._residual!r}, "
            f"{self._scheme!r})"
        )

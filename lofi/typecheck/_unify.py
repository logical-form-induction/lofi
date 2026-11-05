"""First-order unification of semantic type terms."""

from types import MappingProxyType
from typing import TYPE_CHECKING, final

from lofi._checks import ensure_non_negative

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.terms import Term, TermStore
    from lofi.typecheck._ports import Substitution

__all__ = ["InfiniteTypeError", "Unifier"]


@final
class InfiniteTypeError(ValueError):
    """An attempted substitution that would make an infinite type."""

    __slots__ = ("term", "variable")

    def __init__(self, variable: Term, term: Term) -> None:
        """Initialize the error from a variable and recursive type."""
        self.variable = variable
        self.term = term
        super().__init__(f"cannot construct an infinite type: {variable!r} in {term!r}")


def _metavariable(store: TermStore, term: Term) -> int | None:
    """Return a metavariable's atom index, if present."""
    if not store.signature.is_free_variable(term.op):
        return None

    payload = term.payload

    if payload is None:
        msg = f"malformed type metavariable: {term!r}"
        raise ValueError(msg)

    return ensure_non_negative("metavariable atom", payload)


@final
class Unifier:
    """First-order unifier bound to one semantic type store."""

    __slots__ = ("_store",)

    def __init__(self, store: TermStore) -> None:
        """Initialize a unifier for a semantic type store."""
        self._store = store

    @property
    def store(self) -> TermStore:
        """Return the semantic type store."""
        return self._store

    def _require(self, field: str, term: Term) -> None:
        """Require a valid term from the bound store's lineage."""
        if not self._store.contains(term):
            msg = f"{field} is outside the unifier's store lineage"
            raise ValueError(msg)

        def validate(node: Term) -> None:
            """Validate metavariables throughout one subtree."""
            _metavariable(self._store, node)

            for child in node.args:
                validate(child)

        validate(term)

    def _walk(
        self,
        term: Term,
        substitution: dict[int, Term],
    ) -> Term:
        """Follow substitutions at a term's root."""
        held = term
        atom = _metavariable(self._store, held)

        while atom is not None and atom in substitution:
            held = substitution[atom]
            atom = _metavariable(self._store, held)

        return held

    def _occurs(
        self,
        atom: int,
        term: Term,
        substitution: dict[int, Term],
    ) -> bool:
        """Return whether an atom occurs after substitution."""
        held = self._walk(term, substitution)
        found = _metavariable(self._store, held)

        if found is not None:
            return found == atom

        return any(self._occurs(atom, child, substitution) for child in held.args)

    def _bind(
        self,
        variable: Term,
        term: Term,
        substitution: dict[int, Term],
    ) -> None:
        """Bind a metavariable after checking for recursion."""
        atom = _metavariable(self._store, variable)

        if atom is None:
            msg = f"not a type metavariable: {variable!r}"
            raise ValueError(msg)

        held = self._walk(term, substitution)

        if variable is held:
            return

        if self._occurs(atom, held, substitution):
            raise InfiniteTypeError(variable, held)

        substitution[atom] = held

    def _solve(
        self,
        left: Term,
        right: Term,
        substitution: dict[int, Term],
    ) -> bool:
        """Unify a pair against call-local state."""
        here = self._walk(left, substitution)
        there = self._walk(right, substitution)

        if here is there:
            return True

        here_atom = _metavariable(self._store, here)

        if here_atom is not None:
            self._bind(here, there, substitution)

            return True

        there_atom = _metavariable(self._store, there)

        if there_atom is not None:
            self._bind(there, here, substitution)

            return True

        if here.op != there.op or here.payload != there.payload:
            return False

        return all(
            self._solve(here_child, there_child, substitution)
            for here_child, there_child in zip(here.args, there.args, strict=True)
        )

    def _rebuild(self, term: Term, arguments: tuple[Term, ...]) -> Term:
        """Rebuild a term when an argument changed."""
        if arguments == term.args:
            return term

        return self._store.op(term.op, *arguments, payload=term.payload)

    def _normalize(
        self,
        term: Term,
        substitution: dict[int, Term],
    ) -> Term:
        """Apply the substitution transitively to a term."""
        atom = _metavariable(self._store, term)

        if atom is not None and atom in substitution:
            return self._normalize(substitution[atom], substitution)

        arguments = tuple(self._normalize(child, substitution) for child in term.args)

        return self._rebuild(term, arguments)

    def _freeze(
        self,
        substitution: dict[int, Term],
    ) -> Substitution:
        """Return the normalized substitution as an immutable mapping."""
        normalized = {
            atom: self._normalize(term, substitution)
            for atom, term in substitution.items()
        }

        return MappingProxyType(normalized)

    def apply(
        self,
        substitution: Mapping[int, Term],
        term: Term,
        /,
    ) -> Term:
        """Apply an atom-indexed substitution simultaneously."""
        self._require("term", term)
        held: dict[int, Term] = {}

        for atom, replacement in substitution.items():
            checked = ensure_non_negative("substitution atom", atom)
            self._require(f"substitution[{checked}]", replacement)
            held[checked] = replacement

        def replace(node: Term) -> Term:
            """Replace metavariables in one subtree."""
            atom = _metavariable(self._store, node)

            if atom is not None and atom in held:
                return held[atom]

            arguments = tuple(replace(child) for child in node.args)

            return self._rebuild(node, arguments)

        return replace(term)

    def unify(self, left: Term, right: Term, /) -> Substitution | None:
        """Return a most-general unifier for compatible terms."""
        self._require("left", left)
        self._require("right", right)
        substitution: dict[int, Term] = {}

        if not self._solve(left, right, substitution):
            return None

        return self._freeze(substitution)

    def subsumes(
        self,
        general: Term,
        specific: Term,
        /,
    ) -> Substitution | None:
        """Match a general type against a fixed specific type."""
        self._require("general", general)
        self._require("specific", specific)
        substitution: dict[int, Term] = {}

        def match(pattern: Term, target: Term) -> bool:
            """Match one pattern subtree against one fixed target."""
            atom = _metavariable(self._store, pattern)

            if atom is not None:
                known = substitution.get(atom)

                if known is None:
                    substitution[atom] = target

                    return True

                return known is target

            if pattern.op != target.op or pattern.payload != target.payload:
                return False

            return all(
                match(pattern_child, target_child)
                for pattern_child, target_child in zip(
                    pattern.args,
                    target.args,
                    strict=True,
                )
            )

        if not match(general, specific):
            return None

        return MappingProxyType(dict(substitution))

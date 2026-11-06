"""Total constraint solving for logical form types."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Never, final, override

from lofi.typecheck._generate import _generate
from lofi.typecheck._unify import InfiniteTypeError, Unifier
from lofi.typecheck._values import Constraint, Scheme, TypingResult

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.terms import OpName, Path, Term, TermStore
    from lofi.typecheck._lexsig import LexicalSignature
    from lofi.typecheck._ports import Substitution
    from lofi.typecheck._values import TypingRule

__all__ = ["TypeChecker"]


def _compose(
    unifier: Unifier,
    substitution: dict[int, Term],
    update: Substitution,
) -> dict[int, Term]:
    """Compose a successful update after a substitution."""
    composed = {
        atom: unifier.apply(update, replacement)
        for atom, replacement in substitution.items()
    }
    composed.update(update)

    return composed


def _normalize_constraint(
    unifier: Unifier,
    substitution: Substitution,
    constraint: Constraint,
) -> Constraint:
    """Apply a substitution to a residual constraint."""
    return Constraint(
        constraint.kind,
        unifier.apply(substitution, constraint.left),
        unifier.apply(substitution, constraint.right),
        constraint.path,
    )


def _quantified(mask: int, free: int) -> tuple[int, ...]:
    """List issued atoms that remain free in the root type."""
    retained = mask & free

    return tuple(
        atom for atom in range(retained.bit_length()) if retained & (1 << atom)
    )


@final
class TypeChecker:
    """Logical form type checker over one many-sorted language store."""

    __slots__ = ("_store", "_unifier")

    def __init__(self, store: TermStore) -> None:
        """Bind the checker to one store containing terms and their types."""
        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_unifier", Unifier(store))

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "a type checker is immutable"

        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "a type checker is immutable"

        raise AttributeError(msg)

    @property
    def store(self) -> TermStore:
        """Return the language store shared by terms and their types."""
        return self._store

    def check(
        self,
        term: Term,
        signature: LexicalSignature,
        rules: Mapping[OpName, TypingRule],
        /,
        *,
        annotations: Mapping[Path, Term] | None = None,
        constraints: Mapping[Path, Term] | None = None,
    ) -> TypingResult:
        """Check a logical form and collect unsatisfied constraints.

        Parameters
        ----------
        term : Term
            The logical form to type.
        signature : LexicalSignature
            The type schemes of the form's constants.
        rules : Mapping[OpName, TypingRule]
            Structural typing rules keyed by operator name.
        annotations : Mapping[Path, Term] | None
            Surface binder or leaf annotations keyed by node path.
        constraints : Mapping[Path, Term] | None
            Additional expected node types keyed by path.
        """
        if not self._store.contains(term):
            msg = "term is outside the checker's language store lineage"
            raise ValueError(msg)

        generated = _generate(
            term,
            self._store,
            signature,
            rules,
            annotations=annotations,
            constraints=constraints,
        )
        substitution: dict[int, Term] = {}
        failed: list[Constraint] = []

        for constraint in generated.constraints:
            left = self._unifier.apply(substitution, constraint.left)
            right = self._unifier.apply(substitution, constraint.right)

            try:
                update = self._unifier.unify(left, right)
            except InfiniteTypeError:
                failed.append(Constraint("infinite_type", left, right, constraint.path))
                continue

            if update is None:
                failed.append(Constraint(constraint.kind, left, right, constraint.path))
                continue

            substitution = _compose(self._unifier, substitution, update)

        assignment: dict[Path, Term] = {
            path: self._unifier.apply(substitution, inferred)
            for path, inferred in generated.assignment.items()
        }
        residual = tuple(
            _normalize_constraint(self._unifier, substitution, constraint)
            for constraint in failed
        )
        root = assignment[()]
        scheme = (
            None
            if any(constraint.path == () for constraint in residual)
            else Scheme(root, _quantified(generated.issued, root.free))
        )

        return TypingResult(MappingProxyType(assignment), residual, scheme)

    def subsumes(
        self,
        general: Term,
        specific: Term,
        /,
    ) -> Substitution | None:
        """Match a general type against a fixed specific type."""
        return self._unifier.subsumes(general, specific)

    def unifies(
        self,
        left: Term,
        right: Term,
        /,
    ) -> Substitution | None:
        """Return a most-general unifier for two compatible types."""
        return self._unifier.unify(left, right)

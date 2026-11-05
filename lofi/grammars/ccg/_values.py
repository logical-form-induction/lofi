"""CCG matching refinements used by generic grammar declarations."""

from hashlib import sha256
from json import dumps
from types import MappingProxyType
from typing import TYPE_CHECKING, final

from lofi._frozen import FrozenValue, Unpicklable
from lofi.terms import RootMatching, SignatureStableMatching

if TYPE_CHECKING:
    from lofi.judge import Conjunctions
    from lofi.terms import RepresentativeKernel, Signature, Term, TermMatching

__all__ = ["marked_refinement"]


def _term_key(term: Term) -> object:
    """Return one JSON-compatible term key."""
    return [term.op.name, term.payload, [_term_key(arg) for arg in term.args]]


@final
class _MarkedMatching(Unpicklable, FrozenValue):
    """A congruence that keeps declared conjunction marks apart."""

    _noun = "CCG matching refinement"
    _because = "retains live term identities"
    __slots__ = ("_base", "_marked", "_name", "_representatives")

    def __init__(
        self, conjunctions: Conjunctions[Term] | None, base: TermMatching
    ) -> None:
        groups: list[tuple[Term, Term]] = []
        marked: dict[Term, Term] = {}
        pairs = () if conjunctions is None else conjunctions.pairs
        for unmarked, marked_term in pairs:
            representative = next(
                (
                    held_marked
                    for held_unmarked, held_marked in groups
                    if base.equivalent(held_unmarked, unmarked)
                ),
                None,
            )
            if representative is None:
                representative = marked_term
                groups.append((unmarked, representative))
            marked[marked_term] = representative

        declaration = [
            [_term_key(unmarked), _term_key(marked_term)]
            for unmarked, marked_term in pairs
        ]
        encoded = dumps(declaration, separators=(",", ":"), sort_keys=True)
        digest = sha256(encoded.encode()).hexdigest()[:16]
        inventory = "none" if conjunctions is None else conjunctions.name
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "_marked", MappingProxyType(marked))
        object.__setattr__(
            self,
            "_representatives",
            base.representatives.refine(tuple(marked.items())),
        )
        object.__setattr__(self, "_name", f"ccg-{base.name}:{inventory}:{digest}")

    @property
    def name(self) -> str:
        """Return the qualified congruence name."""
        return self._name

    @property
    def has_key(self) -> bool:
        """Report whether the base congruence supplies keys."""
        return self._base.has_key

    @property
    def signature_extension_stable(self) -> bool:
        """Report whether the base representative survives signature extension."""
        return (
            isinstance(self._base, SignatureStableMatching)
            and self._base.signature_extension_stable
        )

    @property
    def exact(self) -> bool:
        """Report that the refinement is not declared to be identity."""
        return False

    @property
    def representatives(self) -> RepresentativeKernel:
        """Return the refined native representative kernel."""
        return self._representatives

    def identified_roots(
        self, operator: str, signature: Signature, /
    ) -> frozenset[str] | None:
        """Return the base congruence's roots and those of related marked terms.

        A marked category matches every marked category of its conjunction
        group, whose roots may differ from its own, so the roots of the groups
        holding a marked category rooted at the operator join the bound.
        """
        base = self._base
        bound = (
            base.identified_roots(operator, signature)
            if isinstance(base, RootMatching)
            else None
        )
        if bound is None:
            return None
        groups = {
            representative
            for term, representative in self._marked.items()
            if term.op.name == operator
        }
        return bound | frozenset(
            term.op.name
            for term, representative in self._marked.items()
            if representative in groups
        )

    def key(self, term: Term, /) -> Term:
        """Return the base key or a marked representative."""
        representative = self._marked.get(term)
        return self._base.key(term) if representative is None else representative

    def equivalent(self, left: Term, right: Term, /) -> bool:
        """Compare categories while keeping marked ones apart."""
        left_mark = self._marked.get(left)
        right_mark = self._marked.get(right)
        if left_mark is None and right_mark is None:
            return self._base.equivalent(left, right)
        return left_mark is right_mark


def marked_refinement(
    base: TermMatching, conjunctions: Conjunctions[Term] | None, /
) -> TermMatching:
    """Return a congruence that keeps declared conjunction marks apart."""
    pairs = () if conjunctions is None else conjunctions.pairs
    if not any(
        base.equivalent(unmarked, marked)
        for unmarked, _ in pairs
        for _, marked in pairs
    ):
        return base
    return _MarkedMatching(conjunctions, base)

"""Term congruence relations."""

from typing import TYPE_CHECKING, cast, override

from lofi._native import RepresentativeKernel, Term
from lofi.terms._ports import ExactMatching, RootMatching, TermMatching

if TYPE_CHECKING:
    from lofi.terms._ports import Matching
    from lofi.terms._values import Signature

__all__ = [
    "Alpha",
    "FeatureErasure",
    "Identity",
    "Product",
    "RepresentativeUnavailableError",
    "Subsumption",
]


class RepresentativeUnavailableError(ValueError):
    """A matching relation has no native representative kernel."""


class Identity[T]:
    """Identity matching for terms."""

    __slots__ = ()

    @property
    def name(self) -> str:
        """Return the relation's name, as it appears in a grammar fingerprint."""
        return "identity"

    @property
    def has_key(self) -> bool:
        """Return whether representative keys support chart packing."""
        return True

    @property
    def signature_extension_stable(self) -> bool:
        """Report that identity keys survive signature extension."""
        return True

    @property
    def exact(self) -> bool:
        """Report that identity relates only identical terms."""
        return True

    @property
    def representatives(self) -> RepresentativeKernel:
        """Compile exact native representatives."""
        return RepresentativeKernel.identity()

    def identified_roots(
        self, operator: str, signature: Signature, /
    ) -> frozenset[str]:
        """Return the operator alone, since identity relates only one root."""
        del signature
        return frozenset({operator})

    def key(self, t: T, /) -> T:
        """Return a term as its identity representative.

        Parameters
        ----------
        t : T
            Term to represent.

        Returns
        -------
        T
            The input object.
        """
        return t

    def equivalent(self, a: T, b: T, /) -> bool:
        """Return whether two terms are identical.

        Parameters
        ----------
        a : T
            First term.
        b : T
            Second term.

        Returns
        -------
        bool
            Whether ``a`` and ``b`` are the same object.
        """
        return a is b

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this relation."""
        return "Identity()"


class Alpha[T](Identity[T]):
    """Identity matching named as alpha-equivalence."""

    __slots__ = ()

    @property
    @override
    def name(self) -> str:
        """Return the alpha relation's name."""
        return "alpha"

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this relation."""
        return "Alpha()"


class FeatureErasure:
    """Feature-erasure matching for interned terms."""

    __slots__ = ()

    @property
    def name(self) -> str:
        """Return the relation's name."""
        return "feature-erasure"

    @property
    def has_key(self) -> bool:
        """Return whether representative keys support chart packing."""
        return True

    @property
    def signature_extension_stable(self) -> bool:
        """Report that feature cores survive signature extension."""
        return True

    @property
    def exact(self) -> bool:
        """Report that feature erasure relates feature variants."""
        return False

    @property
    def representatives(self) -> RepresentativeKernel:
        """Compile precomputed feature-erasure representatives."""
        return RepresentativeKernel.core()

    def identified_roots(
        self, operator: str, signature: Signature, /
    ) -> frozenset[str]:
        """Return every operator that erases to the operator's feature core."""
        erased = {
            declared.name: declared.erases_to or declared.name
            for declared in signature.operators
        }
        core = erased.get(operator, operator)
        return frozenset(
            {operator, *(name for name, held in erased.items() if held == core)}
        )

    def key(self, t: Term, /) -> Term:
        """Return the term's precomputed feature-erasure core."""
        return t.core

    def equivalent(self, a: Term, b: Term, /) -> bool:
        """Return whether two terms have the same feature-erasure core."""
        return a.core is b.core

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this relation."""
        return "FeatureErasure()"


class Product[T]:
    """The intersection of two matching relations over one subject type."""

    __slots__ = ("_left", "_representatives", "_right")

    def __init__(self, left: Matching[T], right: Matching[T]) -> None:
        self._left = left
        self._right = right
        self._representatives = (
            RepresentativeKernel.product(left.representatives, right.representatives)
            if isinstance(left, TermMatching) and isinstance(right, TermMatching)
            else None
        )

    @property
    def left(self) -> Matching[T]:
        """Return the first component relation."""
        return self._left

    @property
    def right(self) -> Matching[T]:
        """Return the second component relation."""
        return self._right

    @property
    def name(self) -> str:
        """Return the relation's component-qualified name."""
        return f"product({self._left.name},{self._right.name})"

    @property
    def has_key(self) -> bool:
        """Return whether both component relations compile native keys."""
        return self._representatives is not None

    @property
    def exact(self) -> bool:
        """Report whether either component relates only identical terms."""
        return any(
            isinstance(component, ExactMatching) and component.exact
            for component in (self._left, self._right)
        )

    def identified_roots(
        self, operator: str, signature: Signature, /
    ) -> frozenset[str] | None:
        """Return the roots both component relations identify with the operator.

        A component that bounds no root leaves the other's bound in force.
        """
        bounds = tuple(
            bound
            for component in (self._left, self._right)
            if isinstance(component, RootMatching)
            and (bound := component.identified_roots(operator, signature)) is not None
        )
        if not bounds:
            return None
        return frozenset.intersection(*bounds)

    @property
    def representatives(self) -> RepresentativeKernel:
        """Return the native intersection kernel."""
        if self._representatives is None:
            msg = "product matching has no native representative kernel"
            raise RepresentativeUnavailableError(msg)
        return self._representatives

    def key(self, term: T, /) -> T:
        """Return one deterministic representative of the intersection class."""
        if self._representatives is None or not isinstance(term, Term):
            msg = "product matching has no representative key for this subject"
            raise RepresentativeUnavailableError(msg)
        return cast("T", self._representatives.key(term))

    def equivalent(self, a: T, b: T, /) -> bool:
        """Return whether both component relations equate the subjects."""
        return self._left.equivalent(a, b) and self._right.equivalent(a, b)

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this relation."""
        return f"Product({self._left!r}, {self._right!r})"


class Subsumption:
    """Structural matching under mutual free-variable subsumption."""

    __slots__ = ("_representatives",)

    def __init__(self) -> None:
        self._representatives = RepresentativeKernel.subsumption()

    @property
    def name(self) -> str:
        """Return the relation's name."""
        return "subsumption"

    @property
    def has_key(self) -> bool:
        """Report that structural subsumption has native representatives."""
        return True

    @property
    def exact(self) -> bool:
        """Report that subsumption relates renamed variables."""
        return False

    @property
    def representatives(self) -> RepresentativeKernel:
        """Return the native structural representative kernel."""
        return self._representatives

    def identified_roots(
        self, operator: str, signature: Signature, /
    ) -> frozenset[str]:
        """Return the operator alone, since subsumption renames only variables."""
        del signature
        return frozenset({operator})

    def key(self, term: Term, /) -> Term:
        """Return one representative of the term's variable-sharing class."""
        return self._representatives.key(term)

    def subsumes(self, general: Term, specific: Term, /) -> bool:
        """Return whether free variables in ``general`` cover ``specific``."""
        assignments: dict[tuple[str, int | None], Term] = {}

        def covers(left: Term, right: Term) -> bool:
            if left.op.name.startswith("#fvar["):
                key = (left.op.sort, left.payload)
                prior = assignments.get(key)

                if prior is None:
                    assignments[key] = right
                    return True

                return prior is right

            return (
                left.op is right.op
                and left.payload == right.payload
                and len(left.args) == len(right.args)
                and all(
                    covers(left_arg, right_arg)
                    for left_arg, right_arg in zip(left.args, right.args, strict=True)
                )
            )

        return covers(general, specific)

    def equivalent(self, left: Term, right: Term, /) -> bool:
        """Return whether each term structurally subsumes the other."""
        return self.subsumes(left, right) and self.subsumes(right, left)

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this relation."""
        return "Subsumption()"

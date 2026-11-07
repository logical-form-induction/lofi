"""Formalism declaration and budget values."""

from hashlib import sha256
from typing import TYPE_CHECKING, final, override

from lofi._checks import (
    ensure_name,
    ensure_positive,
)
from lofi._frozen import FrozenValue

if TYPE_CHECKING:
    from collections.abc import Iterable

    from lofi._types import JsonValue
    from lofi.judge import CapacityReport, ClosureReport
    from lofi.lexicon import Clamp
    from lofi.terms import Term

__all__ = ["Budget", "FormalismDeclaration", "TheoryArtifact"]


@final
class TheoryArtifact(FrozenValue):
    """A checked theory's canonical native representation."""

    _noun = "formalism value"

    __slots__ = ("_digest", "_specification", "_version")

    def __init__(self, version: str, specification: bytes, digest: str) -> None:
        held_version = ensure_name("a theory version", version)

        if not isinstance(specification, bytes):
            msg = f"a theory specification is not bytes: {specification!r}"
            raise TypeError(msg)

        if not specification:
            msg = "a theory specification is empty"
            raise ValueError(msg)

        held_digest = ensure_name("a theory digest", digest)
        computed = sha256(specification).hexdigest()

        if held_digest != computed:
            msg = (
                f"theory digest {held_digest!r} does not match canonical "
                f"specification digest {computed!r}"
            )
            raise ValueError(msg)

        object.__setattr__(self, "_version", held_version)
        object.__setattr__(self, "_specification", specification)
        object.__setattr__(self, "_digest", held_digest)

    @classmethod
    def from_specification(
        cls,
        version: str,
        specification: bytes,
    ) -> TheoryArtifact:
        """Build an artifact and compute its digest."""
        return cls(version, specification, sha256(specification).hexdigest())

    @property
    def version(self) -> str:
        """Return the theory format version."""
        return self._version

    @property
    def specification(self) -> bytes:
        """Return the canonical theory specification."""
        return self._specification

    @property
    def digest(self) -> str:
        """Return the canonical specification digest."""
        return self._digest

    @override
    def __eq__(self, other: object) -> bool:
        """Compare every artifact field."""
        if type(other) is not TheoryArtifact:
            return NotImplemented

        return (
            self._version == other._version
            and self._specification == other._specification
            and self._digest == other._digest
        )

    @override
    def __hash__(self) -> int:
        """Hash every artifact field."""
        return hash((TheoryArtifact, self._version, self._specification, self._digest))

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return (
            f"TheoryArtifact({self._version!r}, {self._specification!r}, "
            f"{self._digest!r})"
        )


@final
class Budget(FrozenValue):
    """Run-scoped approximation bounds, as the grammar declaration validated them."""

    _noun = "formalism value"

    __slots__ = (
        "_beam",
        "_beta_fuel",
        "_depth",
        "_mass",
        "_max_category_arity",
        "_max_category_size",
        "_threads",
        "_unary_rounds",
    )

    def __init__(
        self,
        *,
        depth: int,
        mass: float,
        beam: int,
        unary_rounds: int,
        threads: int,
        beta_fuel: int,
        max_category_arity: int | None,
        max_category_size: int | None,
    ) -> None:
        object.__setattr__(self, "_depth", depth)
        object.__setattr__(self, "_mass", mass)
        object.__setattr__(self, "_beam", beam)
        object.__setattr__(self, "_unary_rounds", unary_rounds)
        object.__setattr__(self, "_threads", threads)
        object.__setattr__(self, "_beta_fuel", beta_fuel)
        object.__setattr__(self, "_max_category_arity", max_category_arity)
        object.__setattr__(self, "_max_category_size", max_category_size)

    @property
    def depth(self) -> int:
        """Return the term depth bound."""
        return self._depth

    @property
    def mass(self) -> float:
        """Return the probability mass target."""
        return self._mass

    @property
    def beam(self) -> int:
        """Return the search beam width."""
        return self._beam

    @property
    def unary_rounds(self) -> int:
        """Return the unary closure round bound."""
        return self._unary_rounds

    @property
    def threads(self) -> int:
        """Return the chart cell worker count."""
        return self._threads

    @property
    def beta_fuel(self) -> int:
        """Return the beta reduction step bound."""
        return self._beta_fuel

    @property
    def max_category_arity(self) -> int | None:
        """Return the functor spine arity bound, or no bound."""
        return self._max_category_arity

    @property
    def max_category_size(self) -> int | None:
        """Return the category node-count bound, or no bound."""
        return self._max_category_size

    def to_spec(self) -> dict[str, JsonValue]:
        """Return the canonical budget declaration."""
        return {
            "depth": self._depth,
            "mass": self._mass,
            "beam": self._beam,
            "unary_rounds": self._unary_rounds,
            "threads": self._threads,
            "beta_fuel": self._beta_fuel,
            "max_category_arity": self._max_category_arity,
            "max_category_size": self._max_category_size,
        }

    @override
    def __eq__(self, other: object) -> bool:
        """Compare every budget field."""
        if type(other) is not Budget:
            return NotImplemented

        return self.to_spec() == other.to_spec()

    @override
    def __hash__(self) -> int:
        """Hash every budget field."""
        return hash(
            (
                Budget,
                self._depth,
                self._mass,
                self._beam,
                self._unary_rounds,
                self._threads,
                self._beta_fuel,
                self._max_category_arity,
                self._max_category_size,
            )
        )

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return (
            f"Budget(depth={self._depth!r}, mass={self._mass!r}, "
            f"beam={self._beam!r}, unary_rounds={self._unary_rounds!r}, "
            f"threads={self._threads!r}, "
            f"beta_fuel={self._beta_fuel!r}, "
            f"max_category_arity={self._max_category_arity!r}, "
            f"max_category_size={self._max_category_size!r})"
        )


@final
class FormalismDeclaration(FrozenValue):
    """Native inputs controlled by a grammar declaration."""

    _noun = "formalism value"

    __slots__ = (
        "_budget",
        "_capacity",
        "_clamps",
        "_closure",
        "_fanout",
        "_lexicalized",
        "_theory",
    )

    def __init__(
        self,
        closure: ClosureReport[Term],
        theory: TheoryArtifact,
        fanout: int,
        budget: Budget,
        capacity: CapacityReport,
        *,
        lexicalized: bool,
        clamps: Iterable[Clamp[Term]] = (),
    ) -> None:
        if not isinstance(lexicalized, bool):
            msg = f"lexicalization is not a boolean: {lexicalized!r}"
            raise TypeError(msg)

        object.__setattr__(self, "_closure", closure)
        object.__setattr__(self, "_theory", theory)
        object.__setattr__(
            self, "_fanout", ensure_positive("a formalism's fan-out", fanout)
        )
        object.__setattr__(self, "_lexicalized", lexicalized)
        object.__setattr__(self, "_budget", budget)
        object.__setattr__(self, "_capacity", capacity)
        object.__setattr__(self, "_clamps", tuple(clamps))

    @property
    def closure(self) -> ClosureReport[Term]:
        """Return the native structural closure report."""
        return self._closure

    @property
    def theory(self) -> TheoryArtifact:
        """Return the checked theory artifact."""
        return self._theory

    @property
    def fanout(self) -> int:
        """Return the declared item fan-out."""
        return self._fanout

    @property
    def lexicalized(self) -> bool:
        """Return whether every derivation has a lexical anchor."""
        return self._lexicalized

    @property
    def budget(self) -> Budget:
        """Return the approximation budget."""
        return self._budget

    @property
    def capacity(self) -> CapacityReport:
        """Return the checked language-capacity report."""
        return self._capacity

    @property
    def clamps(self) -> tuple[Clamp[Term], ...]:
        """Return clamps in declaration order."""
        return self._clamps

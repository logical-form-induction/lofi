"""Typed evidence carried by crisp, graded, and abstaining verdicts."""

from typing import TYPE_CHECKING, Final, Protocol, final, override

from lofi._checks import ensure_finite, ensure_name, ensure_non_negative
from lofi._frozen import FrozenValue

if TYPE_CHECKING:
    from lofi.terms import Term

__all__ = [
    "ABSTENTION_CAUSES",
    "Abstain",
    "AbstentionCause",
    "CounterSatisfiable",
    "Countermodel",
    "CrispStructure",
    "Graded",
    "Proof",
    "Proved",
    "Refuted",
    "SearchBound",
    "Verdict",
]

type AbstentionCause = str
"""The stopping condition an abstention names."""

ABSTENTION_CAUSES: Final[tuple[str, ...]] = (
    "timeout",
    "budget-spent",
    "out-of-fragment",
    "gave-up",
    "search-bound",
)
"""Standard causes included in reports even when they do not occur."""


class CrispStructure(Protocol):
    """A finite crisp structure that can check logical-form evidence."""

    @property
    def identity(self) -> str:
        """Return a stable identity for equality and reporting."""
        ...

    @property
    def rendered(self) -> str:
        """Return an inspectable representation of the structure."""
        ...

    def holds(self, formula: Term, /) -> bool:
        """Return whether the closed formula is true in this structure."""
        ...


@final
class Proof(FrozenValue):
    """A typed proof certificate emitted by a declared system."""

    _noun = "proof certificate"
    __slots__ = ("_derivation", "_system", "_unsat_core")

    def __init__(
        self,
        system: str,
        derivation: str,
        unsat_core: tuple[str, ...] = (),
    ) -> None:
        object.__setattr__(self, "_system", ensure_name("a proof system", system))
        object.__setattr__(
            self, "_derivation", ensure_name("a proof derivation", derivation)
        )
        object.__setattr__(
            self,
            "_unsat_core",
            tuple(ensure_name("an unsat-core member", member) for member in unsat_core),
        )

    @property
    def system(self) -> str:
        """Return the system that checked the derivation."""
        return self._system

    @property
    def derivation(self) -> str:
        """Return the system's derivation text."""
        return self._derivation

    @property
    def unsat_core(self) -> tuple[str, ...]:
        """Return the named assumptions the system reports as sufficient."""
        return self._unsat_core

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another certificate carries the same proof."""
        return (
            type(other) is Proof
            and self._system == other._system
            and self._derivation == other._derivation
            and self._unsat_core == other._unsat_core
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the proof evidence."""
        return hash((Proof, self._system, self._derivation, self._unsat_core))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this certificate."""
        return f"Proof({self._system!r}, {self._derivation!r}, {self._unsat_core!r})"


@final
class CounterSatisfiable(FrozenValue):
    """A typed counter-satisfiable status reported by a prover."""

    _noun = "counter-satisfiable certificate"
    __slots__ = ("_detail", "_status", "_system")

    def __init__(self, system: str, status: str, detail: str) -> None:
        object.__setattr__(self, "_system", ensure_name("a refutation system", system))
        object.__setattr__(
            self, "_status", ensure_name("a counter-satisfiable status", status)
        )
        object.__setattr__(
            self, "_detail", ensure_name("counter-satisfiable evidence", detail)
        )

    @property
    def system(self) -> str:
        """Return the system that emitted the status."""
        return self._system

    @property
    def status(self) -> str:
        """Return the system's counter-satisfiable status."""
        return self._status

    @property
    def detail(self) -> str:
        """Return the system's supporting output."""
        return self._detail

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another certificate carries the same status."""
        return (
            type(other) is CounterSatisfiable
            and self._system == other._system
            and self._status == other._status
            and self._detail == other._detail
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the counter-satisfiable evidence."""
        return hash((CounterSatisfiable, self._system, self._status, self._detail))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this certificate."""
        return (
            f"CounterSatisfiable({self._system!r}, {self._status!r}, {self._detail!r})"
        )


@final
class Countermodel(FrozenValue):
    """A structure checked to satisfy every countermodel obligation."""

    _noun = "countermodel certificate"
    __slots__ = ("_negated_goal", "_premises", "_structure", "_theory")

    def __init__(
        self,
        structure: CrispStructure,
        premises: tuple[Term, ...],
        negated_goal: Term,
        /,
        *,
        theory: tuple[Term, ...] = (),
    ) -> None:
        ensure_name("a countermodel structure identity", structure.identity)
        ensure_name("a rendered countermodel", structure.rendered)
        held_premises = tuple(premises)
        held_theory = tuple(theory)

        for position, premise in enumerate(held_premises):
            if not structure.holds(premise):
                msg = f"countermodel premise {position} is false: {premise!r}"
                raise ValueError(msg)

        for position, sentence in enumerate(held_theory):
            if not structure.holds(sentence):
                msg = f"countermodel theory sentence {position} is false: {sentence!r}"
                raise ValueError(msg)

        if not structure.holds(negated_goal):
            msg = f"countermodel negated-goal is false: {negated_goal!r}"
            raise ValueError(msg)

        object.__setattr__(self, "_structure", structure)
        object.__setattr__(self, "_premises", held_premises)
        object.__setattr__(self, "_negated_goal", negated_goal)
        object.__setattr__(self, "_theory", held_theory)

    @property
    def structure(self) -> CrispStructure:
        """Return the checked crisp structure."""
        return self._structure

    @property
    def premises(self) -> tuple[Term, ...]:
        """Return the premises checked true in the structure."""
        return self._premises

    @property
    def negated_goal(self) -> Term:
        """Return the negated goal checked true in the structure."""
        return self._negated_goal

    @property
    def theory(self) -> tuple[Term, ...]:
        """Return the theory sentences checked true in the structure."""
        return self._theory

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another certificate checks the same obligations."""
        return (
            type(other) is Countermodel
            and self._structure.identity == other._structure.identity
            and self._premises == other._premises
            and self._negated_goal == other._negated_goal
            and self._theory == other._theory
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the checked structure and formulas."""
        return hash(
            (
                Countermodel,
                self._structure.identity,
                self._premises,
                self._negated_goal,
                self._theory,
            )
        )

    @override
    def __repr__(self) -> str:
        """Return an inspectable representation of this checked evidence."""
        return (
            "Countermodel("
            f"structure={self._structure.rendered!r}, "
            f"premises={self._premises!r}, "
            f"negated_goal={self._negated_goal!r}, theory={self._theory!r})"
        )


@final
class SearchBound(FrozenValue):
    """The named finite search resource an oracle exhausted."""

    _noun = "search bound"
    __slots__ = ("_kind", "_limit")

    def __init__(self, kind: str, limit: int) -> None:
        object.__setattr__(self, "_kind", ensure_name("a search-bound kind", kind))
        object.__setattr__(
            self, "_limit", ensure_non_negative("a search-bound limit", limit)
        )

    @property
    def kind(self) -> str:
        """Return the bounded search resource."""
        return self._kind

    @property
    def limit(self) -> int:
        """Return the exhausted limit."""
        return self._limit

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another search bound names the same limit."""
        return (
            type(other) is SearchBound
            and self._kind == other._kind
            and self._limit == other._limit
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the exhausted resource and limit."""
        return hash((SearchBound, self._kind, self._limit))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this bound."""
        return f"SearchBound({self._kind!r}, {self._limit!r})"


@final
class Proved(FrozenValue):
    """An entailment established by a typed proof certificate."""

    _noun = "proved verdict"
    __slots__ = ("_proof",)

    def __init__(self, proof: Proof) -> None:
        if not isinstance(proof, Proof):
            msg = f"a proved verdict requires a Proof: {proof!r}"
            raise TypeError(msg)

        object.__setattr__(self, "_proof", proof)

    @property
    def proof(self) -> Proof:
        """Return the checked proof certificate."""
        return self._proof

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another verdict carries the same proof."""
        return type(other) is Proved and self._proof == other._proof

    @override
    def __hash__(self) -> int:
        """Return a hash over the proof."""
        return hash((Proved, self._proof))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this verdict."""
        return f"Proved({self._proof!r})"


@final
class Refuted(FrozenValue):
    """An entailment rejected by typed counterevidence."""

    _noun = "refuted verdict"
    __slots__ = ("_evidence",)

    def __init__(self, evidence: CounterSatisfiable | Countermodel) -> None:
        if not isinstance(evidence, (CounterSatisfiable, Countermodel)):
            msg = (
                "a refuted verdict requires CounterSatisfiable or Countermodel: "
                f"{evidence!r}"
            )
            raise TypeError(msg)

        object.__setattr__(self, "_evidence", evidence)

    @property
    def evidence(self) -> CounterSatisfiable | Countermodel:
        """Return the checked counterevidence."""
        return self._evidence

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another verdict carries the same counterevidence."""
        return type(other) is Refuted and self._evidence == other._evidence

    @override
    def __hash__(self) -> int:
        """Return a hash over the counterevidence."""
        return hash((Refuted, self._evidence))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this verdict."""
        return f"Refuted({self._evidence!r})"


@final
class Graded(FrozenValue):
    """A solver's graded entailment judgment on the unit interval."""

    _noun = "graded verdict"
    __slots__ = ("_degree", "_detail", "_system")

    def __init__(self, system: str, degree: float, detail: str) -> None:
        held = ensure_finite("a graded entailment degree", degree)

        if not 0.0 <= held <= 1.0:
            msg = f"a graded entailment degree lies in [0, 1]: {degree!r}"
            raise ValueError(msg)

        object.__setattr__(self, "_system", ensure_name("a graded system", system))
        object.__setattr__(self, "_degree", held)
        object.__setattr__(self, "_detail", ensure_name("graded evidence", detail))

    @property
    def system(self) -> str:
        """Return the system that emitted the judgment."""
        return self._system

    @property
    def degree(self) -> float:
        """Return the entailment degree on the unit interval."""
        return self._degree

    @property
    def detail(self) -> str:
        """Return the system's supporting output."""
        return self._detail

    @override
    def __eq__(self, other: object) -> bool:
        return type(other) is Graded and (
            self._system,
            self._degree,
            self._detail,
        ) == (other._system, other._degree, other._detail)

    @override
    def __hash__(self) -> int:
        return hash((Graded, self._system, self._degree, self._detail))

    @override
    def __repr__(self) -> str:
        return f"Graded({self._system!r}, {self._degree!r}, {self._detail!r})"


@final
class Abstain(FrozenValue):
    """A question the oracle left undecided.

    Parameters
    ----------
    reason : str
        Why no answer was reached, in the words of the oracle that stopped.
    cause : AbstentionCause
        The stopping condition the reason spells out. A caller that does not
        classify its own stopping condition leaves it at the prover giving up.

    Raises
    ------
    TypeError
        If the reason is not a string.
    ValueError
        If the reason or cause is empty, or the search-bound evidence does not
        agree with the cause.
    """

    _noun = "abstention verdict"
    __slots__ = ("_bound", "_cause", "_reason")

    def __init__(
        self,
        reason: str,
        *,
        cause: AbstentionCause = "gave-up",
        bound: SearchBound | None = None,
    ) -> None:
        held_cause = ensure_name("an abstention cause", cause)

        if held_cause == "search-bound" and bound is None:
            msg = "a search-bound abstention requires its exhausted SearchBound"
            raise ValueError(msg)

        if held_cause != "search-bound" and bound is not None:
            msg = (
                f"only a search-bound abstention carries a SearchBound: {held_cause!r}"
            )
            raise ValueError(msg)

        object.__setattr__(self, "_reason", ensure_name("a reason", reason))
        object.__setattr__(self, "_cause", held_cause)
        object.__setattr__(self, "_bound", bound)

    @property
    def reason(self) -> str:
        """Return why no answer was reached."""
        return self._reason

    @property
    def cause(self) -> AbstentionCause:
        """Return the stopping condition the reason spells out."""
        return self._cause

    @property
    def bound(self) -> SearchBound | None:
        """Return the exhausted search bound, when model search stopped on one."""
        return self._bound

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another verdict abstains on the same reason and cause."""
        return (
            type(other) is Abstain
            and self._reason == other._reason
            and self._cause == other._cause
            and self._bound == other._bound
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the reason and the cause."""
        return hash((Abstain, self._reason, self._cause, self._bound))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this verdict."""
        suffix = "" if self._bound is None else f", bound={self._bound!r}"
        return f"Abstain({self._reason!r}, cause={self._cause!r}{suffix})"


type Verdict = Proved | Refuted | Graded | Abstain
"""The answer to one entailment question."""

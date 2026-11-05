"""Entailment by exhaustive truth assignment over a formula's atoms."""

from itertools import product
from time import monotonic
from typing import TYPE_CHECKING, Never, final, override

from lofi._checks import ensure_finite, ensure_non_negative
from lofi.lf._verdict import (
    Abstain,
    CounterSatisfiable,
    Proof,
    Proved,
    Refuted,
    SearchBound,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from lofi.lf._profile import LogicFragment, LogicProfile, ProverBound
    from lofi.lf._roles import LogicalRole
    from lofi.lf._verdict import Verdict
    from lofi.terms import Term, TermStore

__all__ = ["TruthTableOracle"]

_BINARY_ARITY = 2


def _spine(term: Term, application: str) -> tuple[Term, tuple[Term, ...]]:
    """Unwind an application spine into its head and its arguments.

    Parameters
    ----------
    term : Term
        The term whose application spine to unwind.
    """
    head = term
    arguments: list[Term] = []

    while head.op.name == application and len(head.args) == _BINARY_ARITY:
        head, argument = head.args
        arguments.append(argument)

    arguments.reverse()

    return head, tuple(arguments)


def _outside_atom(term: Term, application: str) -> str | None:
    """Return the first construct keeping a term from being an atom, or None.

    An atom is a symbol applied to atoms, at any depth. The construct named is
    the head of the first application spine whose head is not a symbol, however
    deeply that spine sits under other symbols.

    Parameters
    ----------
    term : Term
        The term to inspect.
    """
    head, arguments = _spine(term, application)

    if not head.op.is_constant:
        return head.op.name

    for argument in arguments:
        outside = _outside_atom(argument, application)

        if outside is not None:
            return outside

    return None


def _constant_role(
    store: TermStore,
    profile: LogicProfile,
    term: Term,
) -> LogicalRole | None:
    """Return the truth-functional role of an applied constant."""
    head, _ = _spine(term, profile.application)

    if not head.op.is_constant:
        return None

    return profile.logical_roles.get(store.payload_name(head))


def _collect(
    term: Term,
    found: dict[Term, None],
    store: TermStore,
    profile: LogicProfile,
) -> str | None:
    """Record a formula's atoms, returning the first construct outside the fragment.

    Parameters
    ----------
    term : Term
        The formula to inspect.
    found : dict[Term, None]
        The atoms recorded so far.
    """
    role = _constant_role(store, profile, term)
    _, arguments = _spine(term, profile.application)

    if role is not None and role.truth_functional and len(arguments) == role.arity:
        for argument in arguments:
            outside = _collect(argument, found, store, profile)

            if outside is not None:
                return outside

        return None

    if role is not None:
        return role.name

    outside = _outside_atom(term, profile.application)

    if outside is None:
        found[term] = None

    return outside


def _holds(
    term: Term,
    assignment: Mapping[Term, bool],
    store: TermStore,
    profile: LogicProfile,
) -> bool:
    """Evaluate a fragment formula under one truth assignment.

    Parameters
    ----------
    term : Term
        The formula to evaluate.
    assignment : Mapping[Term, bool]
        The truth value assigned to each atom.
    """
    role = _constant_role(store, profile, term)
    _, arguments = _spine(term, profile.application)

    if role is not None and role.truth_functional and len(arguments) == role.arity:
        return role.truth(
            tuple(
                _holds(argument, assignment, store, profile) for argument in arguments
            )
        )

    return assignment[term]


def _text(store: TermStore, term: Term, application: str) -> str:
    """Render one atom as its symbol applied to its arguments.

    Parameters
    ----------
    store : TermStore
        The store that names the atom.
    term : Term
        The atom to render.
    """
    head, arguments = _spine(term, application)
    name = store.payload_name(head) if head.payload is not None else head.op.name

    if not arguments:
        return name

    written = ", ".join(_text(store, argument, application) for argument in arguments)

    return f"{name}({written})"


@final
class TruthTableOracle:
    """An entailment oracle for the quantifier-free, variable-free fragment.

    Exhaustive truth assignments produce either a proof or a countermodel. Equality,
    quantifiers, abstractions, and free variables produce abstentions.

    Parameters
    ----------
    store : TermStore
        The store every formula it is asked about is interned in.
    profile : LogicProfile
        The logic profile in force, whose meaning postulates every decision
        assumes.
    bound : ProverBound
        The profiles this oracle serves, checked against the profile here.
    atoms : int
        The largest number of distinct atoms one decision enumerates over, which
        caps a decision at two to that power assignments.

    Raises
    ------
    ProverUnsupportedError
        If the profile is outside the bound.
    TypeError
        If the atom cap is not an integer.
    ValueError
        If the atom cap is below one.
    """

    __slots__ = ("_atoms", "_bound", "_profile", "_store")

    def __init__(
        self,
        store: TermStore,
        profile: LogicProfile,
        /,
        *,
        bound: ProverBound,
        atoms: int,
    ) -> None:
        bound.check(profile)
        held = ensure_non_negative("an atom cap", atoms)

        if held == 0:
            msg = "an atom cap is at least one atom"

            raise ValueError(msg)

        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_profile", profile)
        object.__setattr__(self, "_bound", bound)
        object.__setattr__(self, "_atoms", held)

    @property
    def bound(self) -> ProverBound:
        """Return the profiles this oracle serves."""
        return self._bound

    @property
    def fragment(self) -> LogicFragment:
        """Return the fragment of the profile this oracle was built for."""
        return self._profile.fragment

    @property
    def atom_cap(self) -> int:
        """Return the largest number of atoms one decision may range over."""
        return self._atoms

    def entails(
        self,
        premise: Term,
        hypothesis: Term,
        /,
        *,
        theory: Sequence[Term] = (),
        timeout: float,
    ) -> Verdict:
        """Decide whether a premise, under a theory, entails a hypothesis.

        Parameters
        ----------
        premise : Term
            The closed formula assumed.
        hypothesis : Term
            The closed formula to be decided.
        theory : Sequence[Term]
            Further closed formulas assumed alongside the premise, in addition
            to the meaning postulates the profile declares.
        timeout : float
            The wall clock in seconds one decision may take before abstention.

        Returns
        -------
        Verdict
            A proof, a countermodel, or an abstention naming the wall clock, the
            atom cap, or the construct outside the fragment.

        Raises
        ------
        ValueError
            If a formula is outside the store's lineage, or the timeout is not
            a positive number.
        """
        held = tuple(theory)
        self._require_lineage((premise, hypothesis, *held))
        clock = ensure_finite("a timeout", timeout)

        if clock <= 0.0:
            msg = f"a timeout is a positive number of seconds: {timeout!r}"

            raise ValueError(msg)

        return self._settle(premise, hypothesis, held, clock)

    def _require_lineage(self, forms: Iterable[Term]) -> None:
        """Refuse a formula the oracle's own store never published.

        Parameters
        ----------
        forms : Iterable[Term]
            The formulas to check.
        """
        for at, form in enumerate(forms):
            if not self._store.contains(form):
                msg = f"formula {at} is outside the supplied store lineage"

                raise ValueError(msg)

    def _settle(
        self,
        premise: Term,
        hypothesis: Term,
        theory: tuple[Term, ...],
        clock: float,
    ) -> Verdict:
        """Decide one question by enumerating truth assignments to its atoms.

        Parameters
        ----------
        premise : Term
            The formula assumed.
        hypothesis : Term
            The formula to test.
        theory : tuple[Term, ...]
            The additional formulas assumed.
        clock : float
            The maximum wall-clock duration in seconds.
        """
        assumptions = (*self._profile.theory, *theory, premise)
        found: dict[Term, None] = {}

        for formula in (*assumptions, hypothesis):
            outside = _collect(formula, found, self._store, self._profile)

            if outside is not None:
                msg = f"{outside!r} is outside the propositional fragment"

                return Abstain(msg, cause="out-of-fragment")

        atoms = tuple(found)

        if len(atoms) > self._atoms:
            msg = (
                f"the question mentions {len(atoms)} atoms, above the cap of "
                f"{self._atoms}"
            )

            return Abstain(
                msg,
                cause="search-bound",
                bound=SearchBound("atoms", self._atoms),
            )

        return self._enumerate(assumptions, hypothesis, atoms, clock)

    def _enumerate(
        self,
        assumptions: tuple[Term, ...],
        hypothesis: Term,
        atoms: tuple[Term, ...],
        clock: float,
    ) -> Verdict:
        """Try every assignment, returning the first countermodel or a proof.

        Parameters
        ----------
        assumptions : tuple[Term, ...]
            The formulas assumed.
        hypothesis : Term
            The formula to test.
        atoms : tuple[Term, ...]
            The atoms to assign.
        clock : float
            The maximum wall-clock duration in seconds.
        """
        started = monotonic()
        checked = 0

        for values in product((False, True), repeat=len(atoms)):
            if monotonic() - started > clock:
                msg = (
                    f"the wall clock of {clock} seconds ran out after "
                    f"{checked} assignments"
                )

                return Abstain(msg, cause="timeout")

            assignment = dict(zip(atoms, values, strict=True))
            checked += 1

            if not all(
                _holds(each, assignment, self._store, self._profile)
                for each in assumptions
            ):
                continue

            if not _holds(hypothesis, assignment, self._store, self._profile):
                return Refuted(
                    CounterSatisfiable(
                        "truth-table",
                        "CounterSatisfiable",
                        self._countermodel(assignment),
                    )
                )

        return Proved(
            Proof(
                "truth-table",
                f"no countermodel among {checked} assignments to {len(atoms)} atoms",
            )
        )

    def _countermodel(self, assignment: Mapping[Term, bool]) -> str:
        """Render a falsifying assignment as its atoms and their truth values.

        Parameters
        ----------
        assignment : Mapping[Term, bool]
            The falsifying truth assignment.
        """
        return ", ".join(
            sorted(
                f"{_text(self._store, atom, self._profile.application)}="
                f"{'true' if value else 'false'}"
                for atom, value in assignment.items()
            )
        )

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "a truth-table oracle is immutable"

        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "a truth-table oracle is immutable"

        raise AttributeError(msg)

    @override
    def __repr__(self) -> str:
        """Return the prover name, the fragment, and the atom cap."""
        return (
            f"TruthTableOracle(prover={self._bound.prover!r}, "
            f"fragment={self._profile.fragment!r}, "
            f"atoms={self._atoms})"
        )

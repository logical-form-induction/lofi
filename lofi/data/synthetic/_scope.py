"""The logical form readings of one string, read off its parse forest.

A string's derivations are enumerated from the forest, each read into a
logical form under the declared interpretation, and the forms grouped by
logical equivalence: two forms belong to the same reading when the oracle
proves each from the other under the fragment's ambient theory. Derivations
that differ only in how the same lexical meanings were combined collapse to
one reading, and derivations that fix two operators in opposite orders stay
apart when the ambient theory does not equate them.
"""

from typing import TYPE_CHECKING, Literal, final, override

from lofi.lf import Abstain, Graded, Proved, Refuted
from lofi.lf.provers import Questions

if TYPE_CHECKING:
    from collections.abc import Iterable

    from lofi.data.synthetic._interpretation import InterpretationFunction
    from lofi.deduce import Chart, ItemKey
    from lofi.terms import Term

__all__ = ["Readings", "UndecidedScopePolicy", "readings"]

type UndecidedScopePolicy = Literal["distinct", "equivalent"]
"""Whether an undecided equivalence keeps readings apart or joins them."""


@final
class Readings:
    """The logical forms of one string, one per reading.

    Parameters
    ----------
    forms : tuple[Term, ...]
        One form per reading, in the order the derivations were enumerated.
    derivations : int
        How many derivations the forms were read from.
    undecided : int
        How many equivalence questions the oracle left undecided, each of
        which keeps two forms apart that a decision might have joined.
    """

    __slots__ = ("_derivations", "_forms", "_undecided")

    def __init__(
        self, forms: tuple[Term, ...], *, derivations: int, undecided: int
    ) -> None:
        self._forms = forms
        self._derivations = derivations
        self._undecided = undecided

    @property
    def forms(self) -> tuple[Term, ...]:
        """Return one form per reading, in enumeration order."""
        return self._forms

    @property
    def count(self) -> int:
        """Return how many readings the string has."""
        return len(self._forms)

    @property
    def derivations(self) -> int:
        """Return how many derivations the readings were read from."""
        return self._derivations

    @property
    def undecided(self) -> int:
        """Return how many equivalence questions the oracle left undecided."""
        return self._undecided

    @override
    def __repr__(self) -> str:
        """Return a rendering naming the counts."""
        return (
            f"Readings(count={len(self._forms)}, "
            f"derivations={self._derivations!r}, undecided={self._undecided!r})"
        )


def _joined(
    interpretation: InterpretationFunction,
    left: Term,
    right: Term,
) -> bool | None:
    """Return whether two forms entail each other, or nothing when undecided.

    Parameters
    ----------
    interpretation : InterpretationFunction
        The interpretation whose oracle decides the two questions.
    left : Term
        One form.
    right : Term
        The other form.
    """
    verdicts = interpretation.verdicts(Questions((left, right), (right, left)))

    if all(
        isinstance(verdict, Proved)
        or (isinstance(verdict, Graded) and verdict.degree == 1.0)
        for verdict in verdicts
    ):
        return True

    if any(
        isinstance(verdict, Abstain)
        or (isinstance(verdict, Graded) and 0.0 < verdict.degree < 1.0)
        for verdict in verdicts
    ):
        return None

    if any(
        isinstance(verdict, Refuted)
        or (isinstance(verdict, Graded) and verdict.degree == 0.0)
        for verdict in verdicts
    ):
        return False

    msg = f"an equivalence oracle returned something other than Verdict: {verdicts!r}"
    raise TypeError(msg)


def readings(
    interpretation: InterpretationFunction,
    chart: Chart[Term, float],
    roots: Iterable[ItemKey[Term]],
    *,
    undecided: UndecidedScopePolicy = "distinct",
) -> Readings:
    """Return the logical forms of a parsed string, one per reading.

    The oracle's spent count is returned to zero before the first question,
    so the interpretation's budget bounds the decisions one string takes.

    Parameters
    ----------
    interpretation : InterpretationFunction
        The interpretation the derivations are read under.
    chart : Chart[Term, float]
        The parsed forest.
    roots : Iterable[ItemKey[Term]]
        The goal items whose derivations the readings are read from.
    undecided : UndecidedScopePolicy
        Whether an equivalence the oracle cannot decide keeps two readings
        distinct or counts them as equivalent.

    Raises
    ------
    UninterpretableDerivationError
        If the interpretation assigns a derivation no form.
    IllTypedFormError
        If a form does not type as a formula.
    """
    if undecided not in ("distinct", "equivalent"):
        msg = f"an undecided-scope policy is not declared: {undecided!r}"
        raise ValueError(msg)
    policy = undecided
    distinct: dict[Term, None] = {}
    derivations = 0

    for root in roots:
        for derivation in chart.derivations(root):
            for meaning in interpretation.read(derivation, interpretation):
                derivations += 1
                distinct[meaning] = None

    held: list[Term] = []
    undecided_count = 0

    for form in distinct:
        certified = interpretation.formula(form)
        joined = False

        for representative in held:
            decision = _joined(interpretation, representative, certified)

            if decision is None:
                undecided_count += 1
                if policy == "equivalent":
                    joined = True
                    break
            elif decision:
                joined = True

                break

        if not joined:
            held.append(certified)

    return Readings(tuple(held), derivations=derivations, undecided=undecided_count)

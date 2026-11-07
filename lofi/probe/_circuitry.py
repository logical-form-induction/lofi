"""The three circuitry criteria, as hypothesis tests against random circuits.

The criteria are those of Shi et al. (2024), "Hypothesis Testing the Circuit
Hypothesis in LLMs". Mechanism preservation asks whether the circuit alone
reproduces the behavior. Mechanism localization asks whether what the circuit
leaves out reproduces it. Minimality asks whether any proper subset of the
circuit preserves it. Each is a statistic over a circuit, and each is tested
against the same null, which draws circuits holding as many units at each
published point as the observed one does.

A criterion read off the observed scores alone asks whether the circuit recovers
the metric, which a large enough set of units does by inclusion. Read against
circuits of its own shape, it asks whether the circuit recovers more than those
do, which a set of units picked at random does not. The proper subsets minimality
searches are the ones reached by dropping one unit, so a circuit two of whose
units each do the whole job reads as minimal here.
"""

from enum import Enum
from random import Random
from statistics import fmean
from types import MappingProxyType
from typing import TYPE_CHECKING, override

from lofi._checks import ensure_finite, ensure_non_negative
from lofi.probe._circuits import Circuit, Unit
from lofi.probe._faithfulness import Endpoints, recovery
from lofi.stats import PValue, p_value_against_replicates

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from lofi.nn.ports import Reading
    from lofi.nn.sites import Site
    from lofi.probe._faithfulness import Ablation
    from lofi.stats import NullFamily, Statistic, Tail

__all__ = [
    "CircuitryReport",
    "Criterion",
    "CriterionResult",
    "MatchedRandomCircuits",
    "circuitry",
]


class Criterion(Enum):
    """One of the three criteria a circuit is tested against."""

    #: The circuit alone reproduces the behavior.
    PRESERVATION = "preservation"

    #: What the circuit leaves out does not reproduce the behavior.
    LOCALIZATION = "localization"

    #: Dropping any one unit costs the circuit some of the behavior.
    MINIMALITY = "minimality"

    @property
    def tail(self) -> Tail:
        """Return which direction of the reference distribution counts as extreme."""
        return "less" if self is Criterion.LOCALIZATION else "greater"


def _a_circuit(field: str, circuit: Circuit) -> Circuit:
    """Return a circuit, checked to be one holding at least one unit.

    Parameters
    ----------
    field : str
        What the circuit is, for a message.
    circuit : Circuit
        The value to check.

    Raises
    ------
    TypeError
        If the value is not a circuit.
    ValueError
        If the circuit holds no unit.
    """
    if type(circuit) is not Circuit:
        msg = f"{field} is a Circuit, not {circuit!r}"
        raise TypeError(msg)

    if not circuit.units:
        msg = f"{field} holds no unit"
        raise ValueError(msg)

    return circuit


class MatchedRandomCircuits:
    """Draws circuits shaped like an observed one out of a pool of units.

    A drawn circuit holds as many units at each published point as the observed
    circuit does, taken without replacement from the pool's units at that point,
    so it has the same size and the same spread over points.

    Parameters
    ----------
    pool : Iterable[Unit]
        The units a drawn circuit is built from, which are the candidates at the
        points under study. A circuit tested against this null is drawn from them
        too, and what it leaves out is read within them.
    replicates : int
        How many circuits to draw.
    seed : int
        The seed the draw runs under, so a published p-value can be recomputed.

    Raises
    ------
    TypeError
        If a member of the pool is not a unit, or the budget or the seed is not an
        integer.
    ValueError
        If the pool is empty, if the budget is zero, or if the seed is negative.
    """

    __slots__ = ("_matching", "_pool", "_replicates", "_seed", "_strata")

    def __init__(
        self, pool: Iterable[Unit], /, *, replicates: int = 999, seed: int
    ) -> None:
        self._pool = frozenset(pool)

        for unit in self._pool:
            if type(unit) is not Unit:
                msg = f"a circuit is drawn from units, not {unit!r}"
                raise TypeError(msg)

        if not self._pool:
            msg = f"the null {self.name!r} was given an empty pool to draw from"
            raise ValueError(msg)

        self._replicates = ensure_non_negative("a replicate budget", replicates)
        self._seed = ensure_non_negative("a null's seed", seed)

        if self._replicates == 0:
            msg = f"the null {self.name!r} was given no replicate budget"
            raise ValueError(msg)

        gathered: dict[Site, list[Unit]] = {}

        for unit in sorted(self._pool, key=repr):
            gathered.setdefault(unit.site, []).append(unit)

        self._strata = {site: tuple(members) for site, members in gathered.items()}
        self._matching: Mapping[str, str | int] = MappingProxyType(
            {
                "criterion": "units per site",
                "sites": len(self._strata),
                "pool": len(self._pool),
            }
        )

    @property
    def name(self) -> str:
        """Return the null's name, unique within the report it appears in."""
        return "matched random circuit"

    @property
    def family(self) -> NullFamily:
        """Return how this null draws its reference statistics."""
        return "matched"

    @property
    def replicates(self) -> int | None:
        """Return how many circuits it draws."""
        return self._replicates

    @property
    def seed(self) -> int | None:
        """Return the sampling seed."""
        return self._seed

    @property
    def matching(self) -> Mapping[str, str | int]:
        """Return what a drawn circuit is matched on, as a read-only view."""
        return self._matching

    @property
    def pool(self) -> frozenset[Unit]:
        """Return the units a drawn circuit is built from."""
        return self._pool

    def _units_of(self, circuit: Circuit) -> frozenset[Unit]:
        """Return an observed circuit's units, checked to come out of the pool.

        Parameters
        ----------
        circuit : Circuit
            The observed circuit.

        Raises
        ------
        TypeError
            If the value is not a circuit.
        ValueError
            If the circuit holds no unit, or holds one the pool does not.
        """
        units = _a_circuit("a circuit tested against a null", circuit).units
        outside = units - self._pool

        if outside:
            listed = ", ".join(repr(unit) for unit in sorted(outside, key=repr))
            msg = f"the null {self.name!r} cannot match a circuit holding {listed}"
            raise ValueError(msg)

        return units

    def complement(self, circuit: Circuit, /) -> frozenset[Unit]:
        """Return the pool's units the circuit leaves out.

        Parameters
        ----------
        circuit : Circuit
            The observed circuit.

        Returns
        -------
        frozenset[Unit]
            Every unit of the pool outside the circuit, which is empty when the
            circuit is the whole pool.

        Raises
        ------
        TypeError
            If the value is not a circuit.
        ValueError
            If the circuit holds no unit, or holds one the pool does not.
        """
        return self._pool - self._units_of(circuit)

    def draw(self, circuit: Circuit, /) -> tuple[Circuit, ...]:
        """Draw one circuit per replicate, shaped like the observed one.

        Parameters
        ----------
        circuit : Circuit
            The observed circuit whose shape each draw copies.

        Returns
        -------
        tuple[Circuit, ...]
            One circuit per replicate, in the order drawn, each holding as many
            units at each point as the observed circuit does.

        Raises
        ------
        TypeError
            If the value is not a circuit.
        ValueError
            If the circuit holds no unit, or holds one the pool does not.
        """
        wanted: dict[Site, int] = {}

        for unit in sorted(self._units_of(circuit), key=repr):
            wanted[unit.site] = wanted.get(unit.site, 0) + 1

        rng = Random(self._seed)

        return tuple(
            Circuit(
                unit
                for site, count in wanted.items()
                for unit in rng.sample(self._strata[site], count)
            )
            for _ in range(self._replicates)
        )

    def test(
        self, statistic: Statistic[Circuit], observed: Circuit, /, *, tail: Tail
    ) -> PValue:
        """Return the p-value of the statistic on this circuit under this null.

        Parameters
        ----------
        statistic : Statistic[Circuit]
            The summary evaluated on the observed circuit and on each drawn one.
        observed : Circuit
            The circuit as it came in.
        tail : Tail
            Which direction of the reference distribution counts as extreme.

        Returns
        -------
        PValue
            The tail probability with the null's name, family, replicate count,
            seed and matching criterion.

        Raises
        ------
        TypeError
            If the observed value is not a circuit, or the statistic returns
            something that is not a number.
        ValueError
            If the circuit holds no unit or holds one the pool does not, if the
            statistic returns a nonfinite value, or if the tail is undeclared.
        """
        drawn = self.draw(observed)

        return p_value_against_replicates(
            ensure_finite("an observed statistic", statistic(observed)),
            tuple(ensure_finite("a null replicate", statistic(one)) for one in drawn),
            tail=tail,
            null=self.name,
            family=self.family,
            exact=False,
            seed=self._seed,
            matching=self._matching,
        )

    @override
    def __repr__(self) -> str:
        """Return the pool's size, the replicate budget and the seed."""
        return (
            f"MatchedRandomCircuits(pool={len(self._pool)}, "
            f"replicates={self._replicates!r}, seed={self._seed!r})"
        )


class CriterionResult:
    """Records what one criterion measured and how it stood against the null.

    Parameters
    ----------
    criterion : Criterion
        Which of the three this is.
    statistic : float
        What the criterion measured on the circuit as it came in.
    p_value : PValue
        The probability of a statistic this extreme under the null that ran.

    Raises
    ------
    TypeError
        If the criterion is not one, the statistic is not a number, or the p-value
        is not one.
    ValueError
        If the statistic is not finite, or the p-value was read off a tail other
        than the one this criterion calls extreme.
    """

    __slots__ = ("_criterion", "_p_value", "_statistic")

    def __init__(self, criterion: Criterion, statistic: float, p_value: PValue) -> None:
        if not isinstance(criterion, Criterion):
            msg = f"a result is of a Criterion, not {criterion!r}"
            raise TypeError(msg)

        if type(p_value) is not PValue:
            msg = f"a result records a PValue, not {p_value!r}"
            raise TypeError(msg)

        self._criterion = criterion
        self._statistic = ensure_finite(f"the {criterion.value} statistic", statistic)
        self._p_value = p_value

        if p_value.provenance.tail != criterion.tail:
            msg = (
                f"the {criterion.value} criterion calls the "
                f"{criterion.tail!r} tail extreme and was given a p-value read off "
                f"the {p_value.provenance.tail!r} one"
            )
            raise ValueError(msg)

    @property
    def criterion(self) -> Criterion:
        """Return which of the three this is."""
        return self._criterion

    @property
    def statistic(self) -> float:
        """Return what the criterion measured on the circuit as it came in."""
        return self._statistic

    @property
    def p_value(self) -> PValue:
        """Return the probability of a statistic this extreme under the null."""
        return self._p_value

    def _identity(self) -> tuple[Criterion, float, PValue]:
        """Return the three fields a result is equal on."""
        return (self._criterion, self._statistic, self._p_value)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value records the same three fields."""
        if type(other) is not CriterionResult:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the three fields the result is equal on."""
        return hash((CriterionResult, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the criterion, the statistic and the probability."""
        return (
            f"CriterionResult({self._criterion!r}, {self._statistic!r}, "
            f"{self._p_value!r})"
        )


class CircuitryReport:
    """Records all three criteria of one circuit, each with its own p-value.

    The three stand side by side and nothing here reduces them to one verdict, so
    a circuit the null rejects on one criterion and not on another reads as
    exactly that.

    Parameters
    ----------
    circuit : Circuit
        The circuit the three criteria were measured on.
    preservation : CriterionResult
        How far the circuit alone reproduced the behavior.
    localization : CriterionResult
        How far what the circuit leaves out reproduced the behavior.
    minimality : CriterionResult
        How much dropping the least useful unit cost the circuit.

    Raises
    ------
    TypeError
        If the circuit is not one, or a result is not one.
    ValueError
        If the circuit holds no unit, or a result is filed under a criterion other
        than its own.
    """

    __slots__ = ("_circuit", "_results")

    def __init__(
        self,
        circuit: Circuit,
        /,
        *,
        preservation: CriterionResult,
        localization: CriterionResult,
        minimality: CriterionResult,
    ) -> None:
        self._circuit = _a_circuit("a reported circuit", circuit)
        filed = {
            Criterion.PRESERVATION: preservation,
            Criterion.LOCALIZATION: localization,
            Criterion.MINIMALITY: minimality,
        }

        for criterion, result in filed.items():
            if type(result) is not CriterionResult:
                msg = f"a report files a CriterionResult, not {result!r}"
                raise TypeError(msg)

            if result.criterion is not criterion:
                msg = (
                    f"a {result.criterion.value} result is filed under "
                    f"{criterion.value}"
                )
                raise ValueError(msg)

        self._results: Mapping[Criterion, CriterionResult] = MappingProxyType(filed)

    @property
    def circuit(self) -> Circuit:
        """Return the circuit the three criteria were measured on."""
        return self._circuit

    @property
    def results(self) -> Mapping[Criterion, CriterionResult]:
        """Return all three results, keyed by criterion, as a read-only view."""
        return self._results

    @property
    def preservation(self) -> CriterionResult:
        """Return how far the circuit alone reproduced the behavior."""
        return self._results[Criterion.PRESERVATION]

    @property
    def localization(self) -> CriterionResult:
        """Return how far what the circuit leaves out reproduced the behavior."""
        return self._results[Criterion.LOCALIZATION]

    @property
    def minimality(self) -> CriterionResult:
        """Return how much dropping the least useful unit cost the circuit."""
        return self._results[Criterion.MINIMALITY]

    def _identity(self) -> tuple[Circuit, tuple[CriterionResult, ...]]:
        """Return the two fields a report is equal on, the results in one order."""
        return (self._circuit, tuple(self._results[key] for key in Criterion))

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value reports the same circuit and results."""
        if type(other) is not CircuitryReport:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the circuit and the three results."""
        return hash((CircuitryReport, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the circuit's size and the three probabilities."""
        probabilities = ", ".join(
            f"{key.value}={self._results[key].p_value.value!r}" for key in Criterion
        )

        return f"CircuitryReport(units={len(self._circuit.units)}, {probabilities})"


class _Recovered[ReadingT: Reading]:
    """Scores unit sets between an ablation's endpoints, once each.

    Parameters
    ----------
    ablation : Ablation[ReadingT]
        The metric, the points under study and what replaces a removed entry.
    endpoints : Endpoints
        The clean and corrupt runs the recovery is read between.
    """

    __slots__ = ("_ablation", "_endpoints", "_seen")

    def __init__(self, ablation: Ablation[ReadingT], endpoints: Endpoints, /) -> None:
        self._ablation = ablation
        self._endpoints = endpoints
        self._seen: dict[frozenset[Unit], float] = {}

    def __call__(self, units: frozenset[Unit]) -> float:
        """Return where keeping only these units lands between the endpoints.

        Parameters
        ----------
        units : frozenset[Unit]
            The units whose entries keep their values.

        Returns
        -------
        float
            Zero where the run scores like the corrupt one and one where it
            scores like the clean one.
        """
        scored = self._seen.get(units)

        if scored is None:
            scored = recovery(
                fmean(self._ablation.only(units)),
                corrupt=self._endpoints.corrupt_mean,
                clean=self._endpoints.clean_mean,
            )
            self._seen[units] = scored

        return scored


def circuitry[ReadingT: Reading](
    ablation: Ablation[ReadingT],
    circuit: Circuit,
    /,
    *,
    endpoints: Endpoints,
    null: MatchedRandomCircuits,
) -> CircuitryReport:
    """Test a circuit on all three criteria against matched random circuits.

    Parameters
    ----------
    ablation : Ablation[ReadingT]
        The metric, the points under study and what replaces a removed entry.
    circuit : Circuit
        The circuit to test, drawn from the null's pool.
    endpoints : Endpoints
        The clean and corrupt runs of the same ablation.
    null : MatchedRandomCircuits
        The pool the circuit is read against, and how many circuits to draw from
        it.

    Returns
    -------
    CircuitryReport
        One result per criterion, each with the statistic measured on the circuit
        as it came in and the p-value of that statistic under the null.

    Raises
    ------
    TypeError
        If the circuit, the endpoints or the null is not one.
    ValueError
        If the circuit holds no unit, holds one the pool does not, or names a
        point the ablation does not study, or if a pass cannot be scored.
    """
    tested = _a_circuit("a tested circuit", circuit)

    if type(endpoints) is not Endpoints:
        msg = f"a circuitry test is read between Endpoints, not {endpoints!r}"
        raise TypeError(msg)

    if type(null) is not MatchedRandomCircuits:
        msg = f"a circuitry test runs against MatchedRandomCircuits, not {null!r}"
        raise TypeError(msg)

    recovered = _Recovered(ablation, endpoints)
    pool = null.pool

    def preserved(drawn: Circuit) -> float:
        return recovered(drawn.units)

    def localized(drawn: Circuit) -> float:
        return recovered(pool - drawn.units)

    def minimal(drawn: Circuit) -> float:
        held = drawn.units

        return preserved(drawn) - max(recovered(held - {unit}) for unit in held)

    def result(criterion: Criterion, statistic: Statistic[Circuit]) -> CriterionResult:
        against = null.test(statistic, tested, tail=criterion.tail)

        return CriterionResult(criterion, statistic(tested), against)

    return CircuitryReport(
        tested,
        preservation=result(Criterion.PRESERVATION, preserved),
        localization=result(Criterion.LOCALIZATION, localized),
        minimality=result(Criterion.MINIMALITY, minimal),
    )

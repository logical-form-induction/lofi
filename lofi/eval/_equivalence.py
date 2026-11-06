"""Logical form equivalence over the shared proof service."""

from statistics import fmean
from time import perf_counter
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, final

from lofi.eval._ports import SZS_STATUSES
from lofi.eval._values import ConventionVector, MetricResult, Resampling
from lofi.lf import Graded, Proved, Refuted
from lofi.lf.provers import ProofBudget, ProofService, ProofSite

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi._types import ItemId
    from lofi.eval._ports import SzsStatus
    from lofi.lf import AbstentionCause, Verdict
    from lofi.spec.language import LanguageAssembly
    from lofi.terms import Term

__all__ = [
    "ACCOUNTING_MEASURES",
    "EQUIVALENCE_MEASURES",
    "AccountingMeasure",
    "EquivalenceAccountingMetric",
    "EquivalenceDecision",
    "EquivalenceLedger",
    "EquivalenceMeasure",
    "EquivalenceMetric",
    "EquivalenceScores",
    "equivalence_scores",
    "equivalence_status",
]

type EquivalenceMeasure = Literal["equivalence", "well-formed", "timeout", "proof-time"]
"""A headline number an equivalence metric reports."""

EQUIVALENCE_MEASURES: Final[tuple[EquivalenceMeasure, ...]] = (
    "equivalence",
    "well-formed",
    "timeout",
    "proof-time",
)
"""The four headline numbers, each a separate metric."""

type AccountingMeasure = Literal["proof-calls", "memo-hits"]
"""A proof service count an equivalence evaluation reports beside its rates."""

ACCOUNTING_MEASURES: Final[tuple[AccountingMeasure, ...]] = ("proof-calls", "memo-hits")
"""The uncached prover calls and memo hits the equivalence checks spent."""

_DECIDED: Final = frozenset(("Theorem", "CounterSatisfiable"))
_SITE: Final = ProofSite("evaluation.equivalence")
_STATUS_COMPONENT: Final = {
    "Theorem": "theorem",
    "CounterSatisfiable": "counter-satisfiable",
    "Timeout": "timeout",
    "ResourceOut": "resource-out",
    "GaveUp": "gave-up",
}
_CAUSE_STATUS: Final[dict[AbstentionCause, SzsStatus]] = {
    "timeout": "Timeout",
    "budget-spent": "ResourceOut",
    "out-of-fragment": "GaveUp",
    "gave-up": "GaveUp",
    "search-bound": "ResourceOut",
}


def equivalence_status(
    service: ProofService,
    left: Term,
    right: Term,
    /,
    *,
    budget: ProofBudget,
    theory: tuple[Term, ...],
) -> SzsStatus:
    """Return the SZS status of one service-backed equivalence decision."""
    service.open_step(_SITE)
    verdict: Verdict = service.equivalent(
        left,
        right,
        budget=budget,
        site=_SITE,
        theory=theory,
    )

    if isinstance(verdict, Proved):
        return "Theorem"

    if isinstance(verdict, Refuted):
        return "CounterSatisfiable"

    if isinstance(verdict, Graded):
        if verdict.degree == 1.0:
            return "Theorem"

        if verdict.degree == 0.0:
            return "CounterSatisfiable"

        return "GaveUp"

    return _CAUSE_STATUS.get(verdict.cause, "GaveUp")


class EquivalenceScores(NamedTuple):
    """The outcome of checking one predicted form against its gold form.

    Attributes
    ----------
    parsed : bool
        Whether the predicted text parsed into a closed form.
    typed : bool
        Whether the parsed form type-checked.
    status : SzsStatus | None
        The status of the equivalence check, or None when the form was
        ill-formed and the prover was not asked.
    seconds : float | None
        The wall clock the check took, or None when the prover was not asked.
    """

    parsed: bool
    typed: bool
    status: SzsStatus | None
    seconds: float | None

    @property
    def well_formed(self) -> bool:
        """Return whether the prediction both parsed and type-checked."""
        return self.parsed and self.typed

    @property
    def decided(self) -> bool:
        """Return whether the check answered rather than timing out or giving up."""
        return self.status in _DECIDED

    def components(self) -> dict[str, float]:
        """Return the outcome as one-hot indicators, with the seconds when asked."""
        held = {
            "parsed": float(self.parsed),
            "well-formed": float(self.well_formed),
            "ill-formed": float(not self.well_formed),
        }

        for status in SZS_STATUSES:
            held[_STATUS_COMPONENT[status]] = float(self.status == status)

        if self.decided and self.seconds is not None:
            held["proof-time"] = self.seconds

        return held


def _parsed(language: LanguageAssembly, text: str) -> bool:
    """Return whether text parses and desugars under a declared language.

    Parameters
    ----------
    language : LanguageAssembly
        The language whose generated syntax reads the form.
    text : str
        The predicted logical form text.
    """
    try:
        desugared = language.syntax.desugar(language.syntax.parse(text))
    except ValueError:
        return False

    return desugared.term.is_closed and not desugared.term.free


def equivalence_scores(
    service: ProofService,
    language: LanguageAssembly,
    predicted: str,
    gold: Term,
    /,
    *,
    budget: ProofBudget,
    theory: Sequence[Term] = (),
) -> EquivalenceScores:
    """Check one predicted logical form text against its gold form.

    The text is parsed and desugared into the store, then type-checked. A
    text that does not parse into a closed form, or whose form does not
    type-check, is ill-formed and is not put to the prover. A well-formed
    form is checked for equivalence with the gold form under the wall clock,
    and the time the check took is recorded.

    Parameters
    ----------
    service : ProofService
        The shared service deciding equivalence.
    language : LanguageAssembly
        The one-store language that reads and checks the prediction.
    predicted : str
        The predicted logical form text.
    gold : Term
        The gold form.
    budget : ProofBudget
        The single use-site budget governing both prover decisions.
    theory : Sequence[Term]
        Further closed formulas both directions assume.

    Raises
    ------
    TypeError
        If the predicted text is not a string.
    ValueError
        If the service refuses the gold form.
    """
    if not isinstance(predicted, str):
        msg = f"a predicted logical form is not text: {predicted!r}"
        raise TypeError(msg)

    if not _parsed(language, predicted):
        return EquivalenceScores(parsed=False, typed=False, status=None, seconds=None)

    try:
        term = language.read(predicted)
    except ValueError:
        return EquivalenceScores(parsed=True, typed=False, status=None, seconds=None)

    started = perf_counter()
    status = equivalence_status(
        service,
        term,
        gold,
        budget=budget,
        theory=tuple(theory),
    )
    seconds = perf_counter() - started

    return EquivalenceScores(parsed=True, typed=True, status=status, seconds=seconds)


class EquivalenceDecision(NamedTuple):
    """One item's checked pair, its outcome, and what deciding it spent.

    Attributes
    ----------
    predicted : str
        The predicted logical form text.
    gold : Term
        The gold form.
    scores : EquivalenceScores
        The outcome of the check.
    calls : int
        The uncached prover calls the check dispatched.
    memo_hits : int
        The questions the service answered from its memo.
    """

    predicted: str
    gold: Term
    scores: EquivalenceScores
    calls: int
    memo_hits: int


@final
class EquivalenceLedger:
    """Decide each item's equivalence once for every row that reads it.

    Parameters
    ----------
    service : ProofService
        The shared service deciding equivalence.
    language : LanguageAssembly
        The language predictions are read, typed and interned in, which holds
        the gold forms.
    budget : ProofBudget
        The single use-site budget governing each equivalence check.
    theory : Sequence[Term]
        Further closed formulas every check assumes.

    Raises
    ------
    TypeError
        If the service or budget is not one.
    """

    __slots__ = ("_budget", "_items", "_language", "_service", "_theory")

    def __init__(
        self,
        service: ProofService,
        language: LanguageAssembly,
        /,
        *,
        budget: ProofBudget,
        theory: Sequence[Term] = (),
    ) -> None:
        if not isinstance(service, ProofService):
            msg = f"an equivalence ledger needs a ProofService, not {service!r}"
            raise TypeError(msg)

        if not isinstance(budget, ProofBudget):
            msg = f"an equivalence ledger needs a ProofBudget, not {budget!r}"
            raise TypeError(msg)

        self._service = service
        self._language = language
        self._budget = budget
        self._theory = tuple(theory)
        self._items: dict[ItemId, EquivalenceDecision] = {}

    @property
    def budget(self) -> ProofBudget:
        """Return the single use-site proof budget."""
        return self._budget

    @property
    def service(self) -> ProofService:
        """Return the shared proof service."""
        return self._service

    @property
    def theory(self) -> tuple[Term, ...]:
        """Return the further formulas every check assumes."""
        return self._theory

    def decide(self, pred: str, gold: Term, *, key: ItemId) -> EquivalenceDecision:
        """Return one item's decision, checking it the first time it is asked.

        Parameters
        ----------
        pred : str
            The predicted logical form text.
        gold : Term
            The gold form, interned in the ledger's language.
        key : ItemId
            Item identifier.

        Raises
        ------
        TypeError
            If the prediction is not text.
        ValueError
            If the key was already decided for a different pair, or the service
            refuses the gold form.
        """
        standing = self._items.get(key)

        if standing is not None:
            if (standing.predicted, standing.gold) != (pred, gold):
                msg = f"the item {key!r} was already scored against a different pair"
                raise ValueError(msg)

            return standing

        scores = equivalence_scores(
            self._service,
            self._language,
            pred,
            gold,
            budget=self._budget,
            theory=self._theory,
        )

        if scores.status is None:
            calls, hits = 0, 0
        else:
            spent = self._service.step_report(_SITE)
            calls, hits = spent.calls, spent.memo_hits

        decision = EquivalenceDecision(pred, gold, scores, calls, hits)
        self._items[key] = decision
        return decision


@final
class EquivalenceMetric:
    """One equivalence rate over predicted logical form texts.

    Parameters
    ----------
    ledger : EquivalenceLedger
        The ledger every row of one evaluation decides through.
    measure : EquivalenceMeasure
        The headline number this metric reports.
    conventions : ConventionVector
        The conventions the row is reported under.

    Raises
    ------
    ValueError
        If the measure is unknown.
    """

    __slots__ = ("_conventions", "_items", "_ledger", "_measure", "_resampling")

    def __init__(
        self,
        ledger: EquivalenceLedger,
        /,
        *,
        measure: EquivalenceMeasure,
        conventions: ConventionVector,
    ) -> None:
        if measure not in EQUIVALENCE_MEASURES:
            msg = f"an equivalence measure is not one of {EQUIVALENCE_MEASURES!r}"
            raise ValueError(msg)

        self._ledger = ledger
        self._measure: EquivalenceMeasure = measure
        self._conventions = conventions
        self._resampling = Resampling("item")
        self._items: dict[ItemId, EquivalenceScores] = {}

    @property
    def name(self) -> str:
        """Return the metric name."""
        return f"equivalence-{self._measure}"

    @property
    def measure(self) -> EquivalenceMeasure:
        """Return the headline number."""
        return self._measure

    @property
    def ledger(self) -> EquivalenceLedger:
        """Return the ledger this row decides through."""
        return self._ledger

    @property
    def conventions(self) -> ConventionVector:
        """Return the conventions the row is reported under."""
        return self._conventions

    @property
    def order_invariant(self) -> bool:
        """Return whether update order affects the aggregate."""
        return True

    def reset(self) -> None:
        """Forget every item scored so far."""
        self._items = {}

    def update(self, pred: str, gold: Term, *, key: ItemId) -> None:
        """Check one predicted text against its gold form.

        Parameters
        ----------
        pred : str
            The predicted logical form text.
        gold : Term
            The gold form, interned in the ledger's language.
        key : ItemId
            Item identifier.

        Raises
        ------
        TypeError
            If the prediction is not text.
        ValueError
            If the key was already scored against a different pair, or the
            service refuses the gold form.
        """
        self._items[key] = self._ledger.decide(pred, gold, key=key).scores

    def compute(self) -> MetricResult:
        """Return the headline rate over every item scored so far.

        Returns
        -------
        MetricResult
            The rate over items and every other rate as components, or an
            absent result when no item is scored. The mean proof time is absent
            when no check was decided.
        """
        scored = tuple(self._items.values())

        if not scored:
            return MetricResult(None, 0, self._resampling)

        components = self._aggregate(scored)

        return MetricResult(
            components.get(_MEASURE_COMPONENT[self._measure]),
            len(scored),
            self._resampling,
            components=components,
        )

    def per_item(self) -> Mapping[ItemId, MetricResult]:
        """Return the outcome of each item scored so far.

        Returns
        -------
        Mapping[ItemId, MetricResult]
            One single-item result per key.
        """
        return {
            key: MetricResult(
                scores.components().get(_MEASURE_COMPONENT[self._measure]),
                1,
                self._resampling,
                components=scores.components(),
            )
            for key, scores in self._items.items()
        }

    def _aggregate(self, scored: Sequence[EquivalenceScores]) -> dict[str, float]:
        """Return every rate over scored items.

        Parameters
        ----------
        scored : Sequence[EquivalenceScores]
            The outcomes to aggregate.
        """
        rows = tuple(scores.components() for scores in scored)
        held = {
            name: fmean(row[name] for row in rows)
            for name in rows[0]
            if name != "proof-time"
        }
        times = tuple(
            scores.seconds
            for scores in scored
            if scores.decided and scores.seconds is not None
        )

        if times:
            held["proof-time"] = fmean(times)

        return held


_MEASURE_COMPONENT: Final[dict[EquivalenceMeasure, str]] = {
    "equivalence": "theorem",
    "well-formed": "well-formed",
    "timeout": "timeout",
    "proof-time": "proof-time",
}


@final
class EquivalenceAccountingMetric:
    """The prover calls or memo hits an evaluation's equivalence checks spent.

    The value is the total over scored items; each item's own count is its
    per-item value. An ill-formed prediction is never put to the service and
    spends nothing.

    Parameters
    ----------
    ledger : EquivalenceLedger
        The ledger every row of one evaluation decides through.
    measure : AccountingMeasure
        The count this row reports.
    conventions : ConventionVector
        The conventions the row is reported under.

    Raises
    ------
    ValueError
        If the measure is unknown.
    """

    __slots__ = ("_conventions", "_items", "_ledger", "_measure", "_resampling")

    def __init__(
        self,
        ledger: EquivalenceLedger,
        /,
        *,
        measure: AccountingMeasure,
        conventions: ConventionVector,
    ) -> None:
        if measure not in ACCOUNTING_MEASURES:
            msg = f"an accounting measure is not one of {ACCOUNTING_MEASURES!r}"
            raise ValueError(msg)

        self._ledger = ledger
        self._measure: AccountingMeasure = measure
        self._conventions = conventions
        self._resampling = Resampling("item")
        self._items: dict[ItemId, int] = {}

    @property
    def name(self) -> str:
        """Return the metric name."""
        return f"equivalence-{self._measure}"

    @property
    def conventions(self) -> ConventionVector:
        """Return the conventions the row is reported under."""
        return self._conventions

    @property
    def order_invariant(self) -> bool:
        """Return whether update order affects the aggregate."""
        return True

    def reset(self) -> None:
        """Forget every item scored so far."""
        self._items = {}

    def update(self, pred: str, gold: Term, *, key: ItemId) -> None:
        """Record what deciding one item spent."""
        decision = self._ledger.decide(pred, gold, key=key)
        self._items[key] = (
            decision.calls if self._measure == "proof-calls" else decision.memo_hits
        )

    def compute(self) -> MetricResult:
        """Return the total count over every item scored so far."""
        if not self._items:
            return MetricResult(None, 0, self._resampling)

        return MetricResult(
            float(sum(self._items.values())), len(self._items), self._resampling
        )

    def per_item(self) -> Mapping[ItemId, MetricResult]:
        """Return each scored item's own count."""
        return {
            key: MetricResult(float(count), 1, self._resampling)
            for key, count in self._items.items()
        }

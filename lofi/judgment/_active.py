"""Quantitative active collection of judgments in constrained batches.

Each round fits the model to every judgment collected so far, selects one batch
from the candidates not yet judged, and collects the responses to it. A
collector that cannot yet answer a batch, because its responses are still being
collected elsewhere, returns no responses; the operation then stops with that
round awaiting, and running it again with the same inputs reaches the same
batch and continues once the responses are available.

A run may start from judgments it is given, or from a round 0 whose batch is a
seeded draw of a declared number of candidates from every stratum a declared
expression names, answered by the same collector as every later round.
Collection ends when the declared stopping rule holds, when the remaining
candidates cannot fill another batch, or when the declared number of rounds is
reached, and the run records which.
"""

from collections import defaultdict
from itertools import combinations
from math import isfinite
from random import Random
from statistics import fmean
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, cast, final

from lofi._checks import ensure_name, ensure_non_negative
from lofi._frozen import FrozenValue
from lofi.judgment._acquisition import AcquisitionBatch, PoolExhaustedError
from lofi.judgment._attributes import AttributeExpression, item_key
from lofi.judgment.agreement import fisher_interval, rank_correlation
from lofi.registry import ComponentPlugin, Registry, RegistryError
from lofi.spec.acquisition import (
    ActiveLearningSettings,
    AgreementStopSettings,
    AsymptoteStopSettings,
    RankStopSettings,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path

    from bead.items.item import Item

    from lofi.judgment._acquisition import AcquisitionRule, AcquisitionScore
    from lofi.judgment._sources import BeadResponseEncoding
    from lofi.spec.acquisition import (
        ActiveStopReason,
        CollectedInitialSettings,
        CollectorSettings,
    )

__all__ = [
    "COLLECTOR_FAMILY",
    "ActiveCollector",
    "ActiveFit",
    "ActiveFitter",
    "ActiveJudgment",
    "ActiveLearningResult",
    "ActiveRound",
    "AwaitedBatch",
    "CollectorContext",
    "active_collector_for",
    "collect_initial",
    "draw_initial_batch",
    "run_active_learning",
]

COLLECTOR_FAMILY = "active_collector"
"""The entry-point family the installed collectors register under."""


@final
class ActiveJudgment(FrozenValue):
    """One collected ordinal response with preserved participant and item ids."""

    _noun = "active judgment"
    __slots__ = ("_item_id", "_participant_id", "_response")

    def __init__(self, participant_id: str, item_id: str, response: int) -> None:
        object.__setattr__(
            self,
            "_participant_id",
            ensure_name("an active-judgment participant", participant_id),
        )
        object.__setattr__(
            self, "_item_id", ensure_name("an active-judgment item", item_id)
        )
        ensure_non_negative("an active-judgment response", response)
        object.__setattr__(self, "_response", response)

    @property
    def participant_id(self) -> str:
        """Return the annotator identity."""
        return self._participant_id

    @property
    def item_id(self) -> str:
        """Return the judged item's preserved identity."""
        return self._item_id

    @property
    def response(self) -> int:
        """Return the zero-based ordinal response."""
        return self._response


class ActiveFit(Protocol):
    """One fit's acquisition and validation predictions."""

    def acquisition_scores(
        self, items: Sequence[Item], /
    ) -> tuple[AcquisitionScore, ...]:
        """Return one predicted expectation for every candidate."""
        ...

    @property
    def performance(self) -> Mapping[str, float]:
        """Return current performance by acquisition stratum."""
        ...

    @property
    def validation_ranks(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return paired observed and cross-validated predicted ranks."""
        ...


class ActiveFitter(Protocol):
    """Fit one active model over every judgment collected so far."""

    def fit(self, judgments: Sequence[ActiveJudgment], /) -> ActiveFit:
        """Return a fitted acquisition and validation capability."""
        ...


class ActiveCollector(Protocol):
    """Collect or simulate responses for one exact bead batch."""

    @property
    def sources(self) -> tuple[Path, ...]:
        """Return every declared document the collector read."""
        ...

    def collect(
        self, round_index: int, batch: AcquisitionBatch, /
    ) -> tuple[ActiveJudgment, ...] | None:
        """Return responses to the selected batch, or None while they are awaited."""
        ...


@final
class CollectorContext(FrozenValue):
    """What a run supplies to the collector it builds.

    Parameters
    ----------
    directory : Path
        Directory the run keeps its rounds under.
    root : Path
        Directory the active-learning document's declared paths are relative
        to.
    candidates : Sequence[Item]
        The whole candidate pool, in construction order.
    readouts : Mapping[str, Mapping[str, float]]
        Value of each predictor column the trained point fills, by candidate
        identity.
    levels : int
        Number of ordered response levels of the refitted response model.
    seed : int
        Seed of the collector's own draws.
    encoding : BeadResponseEncoding | None
        The response encoding the document declares for its bead readers, or
        None for a document without one.
    """

    _noun = "collector context"
    __slots__ = (
        "_candidates",
        "_directory",
        "_encoding",
        "_levels",
        "_readouts",
        "_root",
        "_seed",
    )

    def __init__(
        self,
        directory: Path,
        root: Path,
        candidates: Sequence[Item],
        readouts: Mapping[str, Mapping[str, float]],
        *,
        levels: int,
        seed: int,
        encoding: BeadResponseEncoding | None = None,
    ) -> None:
        object.__setattr__(self, "_directory", directory)
        object.__setattr__(self, "_root", root)
        object.__setattr__(self, "_candidates", tuple(candidates))
        object.__setattr__(
            self,
            "_readouts",
            MappingProxyType(
                {
                    item: MappingProxyType(dict(columns))
                    for item, columns in readouts.items()
                }
            ),
        )
        object.__setattr__(self, "_levels", levels)
        object.__setattr__(self, "_seed", seed)
        object.__setattr__(self, "_encoding", encoding)

    @property
    def directory(self) -> Path:
        """Return the directory the run keeps its rounds under."""
        return self._directory

    @property
    def root(self) -> Path:
        """Return the directory the document's declared paths are relative to."""
        return self._root

    @property
    def candidates(self) -> tuple[Item, ...]:
        """Return the candidate pool in construction order."""
        return self._candidates

    @property
    def readouts(self) -> Mapping[str, Mapping[str, float]]:
        """Return each candidate's predictor columns by candidate identity."""
        return self._readouts

    @property
    def levels(self) -> int:
        """Return the number of ordered response levels."""
        return self._levels

    @property
    def seed(self) -> int:
        """Return the seed of the collector's own draws."""
        return self._seed

    @property
    def encoding(self) -> BeadResponseEncoding | None:
        """Return the response encoding the document declares, if any."""
        return self._encoding


def active_collector_for(
    settings: CollectorSettings, context: CollectorContext, /
) -> ActiveCollector:
    """Build the collector installed under the settings' tag.

    Raises
    ------
    UnknownTagError
        If no collector is installed under the tag.
    RegistryError
        If the entry is not a component plugin.
    """
    member = Registry[object](COLLECTOR_FAMILY).get(settings.kind)
    if not isinstance(member, ComponentPlugin):
        msg = f"collector entry {settings.kind!r} names {member!r}, not a plugin"
        raise RegistryError(msg, family=COLLECTOR_FAMILY)
    plugin = cast(
        "ComponentPlugin[CollectorSettings, CollectorContext, ActiveCollector]",
        member,
    )
    return plugin.build(settings, context)


@final
class ActiveRound(FrozenValue):
    """Measurements and stop decision recorded after one acquired batch."""

    _noun = "active-learning round"
    __slots__ = (
        "_agreement",
        "_agreement_interval",
        "_batch",
        "_index",
        "_judged",
        "_rank",
        "_rank_interval",
        "_stop_reason",
    )

    def __init__(
        self,
        index: int,
        batch: AcquisitionBatch,
        agreement: float,
        agreement_interval: tuple[float, float],
        rank: float,
        rank_interval: tuple[float, float],
        stop_reason: ActiveStopReason | None,
        judged: int,
    ) -> None:
        object.__setattr__(self, "_index", index)
        object.__setattr__(self, "_judged", judged)
        object.__setattr__(self, "_batch", batch)
        object.__setattr__(self, "_agreement", agreement)
        object.__setattr__(self, "_agreement_interval", agreement_interval)
        object.__setattr__(self, "_rank", rank)
        object.__setattr__(self, "_rank_interval", rank_interval)
        object.__setattr__(self, "_stop_reason", stop_reason)

    @property
    def index(self) -> int:
        """Return the one-based round index."""
        return self._index

    @property
    def batch(self) -> AcquisitionBatch:
        """Return the sole batch acquired this round."""
        return self._batch

    @property
    def judged(self) -> int:
        """Return the judgments collected by the end of this round."""
        return self._judged

    @property
    def agreement(self) -> float:
        """Return Krippendorff's ordinal alpha."""
        return self._agreement

    @property
    def agreement_interval(self) -> tuple[float, float]:
        """Return the declared bootstrap interval for alpha."""
        return self._agreement_interval

    @property
    def rank_correlation(self) -> float:
        """Return cross-validated Spearman correlation."""
        return self._rank

    @property
    def rank_interval(self) -> tuple[float, float]:
        """Return the declared Fisher interval for rank correlation."""
        return self._rank_interval

    @property
    def stopped(self) -> bool:
        """Return whether collection ended with this round."""
        return self._stop_reason is not None

    @property
    def stop_reason(self) -> ActiveStopReason | None:
        """Return why collection ended with this round, or None if it went on."""
        return self._stop_reason


@final
class AwaitedBatch(FrozenValue):
    """A selected batch whose responses the collector does not yet hold."""

    _noun = "awaited batch"
    __slots__ = ("_batch", "_index")

    def __init__(self, index: int, batch: AcquisitionBatch) -> None:
        object.__setattr__(self, "_index", index)
        object.__setattr__(self, "_batch", batch)

    @property
    def index(self) -> int:
        """Return the round the batch was selected in, 0 for an initial draw."""
        return self._index

    @property
    def batch(self) -> AcquisitionBatch:
        """Return the batch awaiting responses."""
        return self._batch


@final
class ActiveLearningResult(FrozenValue):
    """All acquired responses and round records from one operation."""

    _noun = "active-learning result"
    __slots__ = ("_awaiting", "_judgments", "_rounds", "_stop_reason")

    def __init__(
        self,
        rounds: Iterable[ActiveRound],
        judgments: Iterable[ActiveJudgment],
        *,
        awaiting: AwaitedBatch | None,
        stop_reason: ActiveStopReason | None,
    ) -> None:
        if (awaiting is None) == (stop_reason is None):
            msg = (
                "an active-learning result either awaits a batch or records why "
                f"it stopped, not awaiting={awaiting!r} and "
                f"stop_reason={stop_reason!r}"
            )
            raise ValueError(msg)
        object.__setattr__(self, "_rounds", tuple(rounds))
        object.__setattr__(self, "_judgments", tuple(judgments))
        object.__setattr__(self, "_awaiting", awaiting)
        object.__setattr__(self, "_stop_reason", stop_reason)

    @property
    def awaiting(self) -> AwaitedBatch | None:
        """Return the batch whose responses stopped the run, or None."""
        return self._awaiting

    @property
    def rounds(self) -> tuple[ActiveRound, ...]:
        """Return rounds in execution order."""
        return self._rounds

    @property
    def judgments(self) -> tuple[ActiveJudgment, ...]:
        """Return initial then actively collected responses."""
        return self._judgments

    @property
    def stopped(self) -> bool:
        """Return whether collection ended rather than awaiting a batch."""
        return self._stop_reason is not None

    @property
    def stop_reason(self) -> ActiveStopReason | None:
        """Return why collection ended, or None while a batch is awaited."""
        return self._stop_reason


def _alpha(groups: Sequence[Sequence[int]], /) -> float:
    usable = tuple(tuple(group) for group in groups if len(group) >= 2)
    if not usable:
        msg = "Krippendorff alpha needs a multiply judged item"
        raise ValueError(msg)
    observed = [
        float(left - right) ** 2
        for group in usable
        for left, right in combinations(group, 2)
    ]
    pooled = tuple(response for group in usable for response in group)
    expected = [float(left - right) ** 2 for left, right in combinations(pooled, 2)]
    expected_disagreement = fmean(expected)
    observed_disagreement = fmean(observed)
    if expected_disagreement == 0.0:
        return 1.0
    return 1.0 - observed_disagreement / expected_disagreement


def _quantile(values: Sequence[float], probability: float, /) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _agreement(
    judgments: Sequence[ActiveJudgment],
    *,
    confidence: float,
    samples: int,
    seed: int,
) -> tuple[float, tuple[float, float]]:
    by_item: dict[str, list[int]] = defaultdict(list)
    for judgment in judgments:
        by_item[judgment.item_id].append(judgment.response)
    groups = tuple(tuple(group) for group in by_item.values() if len(group) >= 2)
    estimate = _alpha(groups)
    random = Random(seed)
    replicas: list[float] = []
    for _ in range(samples):
        sampled = tuple(groups[random.randrange(len(groups))] for _ in groups)
        value = _alpha(sampled)
        if isfinite(value):
            replicas.append(value)
    if not replicas:
        return estimate, (estimate, estimate)
    tail = (1.0 - confidence) / 2.0
    return estimate, (_quantile(replicas, tail), _quantile(replicas, 1.0 - tail))


def _rank(fit: ActiveFit, confidence: float, /) -> tuple[float, tuple[float, float]]:
    observed, predicted = fit.validation_ranks
    estimate = rank_correlation(observed, predicted)
    if estimate <= -1.0 or estimate >= 1.0:
        return estimate, (estimate, estimate)
    return estimate, fisher_interval(estimate, len(observed), level=confidence)


def _stop_reason(
    settings: ActiveLearningSettings,
    agreement: float,
    agreement_interval: tuple[float, float],
    rank: float,
    rank_interval: tuple[float, float],
    history: Sequence[float],
    /,
) -> ActiveStopReason | None:
    stopping = settings.stopping
    if isinstance(stopping, AgreementStopSettings):
        value = agreement if stopping.basis == "estimate" else agreement_interval[0]
        return "agreement" if value >= stopping.threshold else None
    if isinstance(stopping, RankStopSettings):
        value = rank if stopping.basis == "estimate" else rank_interval[0]
        return "rank-correlation" if value >= stopping.threshold else None
    if not isinstance(stopping, AsymptoteStopSettings):
        msg = f"unknown stopping settings: {stopping!r}"
        raise TypeError(msg)
    if len(history) < stopping.window:
        return None
    recent = history[-stopping.window :]
    return "asymptote" if max(recent) - min(recent) <= stopping.maximum_gain else None


def draw_initial_batch(
    candidates: Sequence[Item],
    stratify_by: AttributeExpression,
    items_per_stratum: int,
    /,
    *,
    seed: int,
) -> AcquisitionBatch:
    """Draw the declared number of candidates from every stratum of the pool.

    Each stratum, in name order, draws its candidates without replacement from
    one generator seeded by ``seed``; the batch holds every stratum's draw in
    pool order. No prediction exists before the first fit, so the batch
    records no variance and no performance.

    Raises
    ------
    ValueError
        If a stratum holds fewer candidates than are drawn from each stratum.
    """
    by_stratum: dict[str, list[Item]] = defaultdict(list)
    for candidate in candidates:
        by_stratum[stratify_by.level(candidate)].append(candidate)
    random = Random(seed)
    drawn: set[str] = set()
    for stratum in sorted(by_stratum):
        members = by_stratum[stratum]
        if len(members) < items_per_stratum:
            msg = (
                f"stratum {stratum!r} holds {len(members)} candidates, and the "
                f"initial draw takes {items_per_stratum} from every stratum"
            )
            raise ValueError(msg)
        drawn.update(
            item_key(member) for member in random.sample(members, items_per_stratum)
        )
    selected = tuple(item for item in candidates if item_key(item) in drawn)
    shares = {
        stratum: items_per_stratum / len(selected) for stratum in sorted(by_stratum)
    }
    return AcquisitionBatch(selected, shares, None, performance={}, floored=())


def _checked(
    collected: Sequence[ActiveJudgment], batch: AcquisitionBatch, /
) -> tuple[ActiveJudgment, ...]:
    """Return a round's responses after checking they answer its batch.

    Raises
    ------
    ValueError
        If the collector returned no judgment or one for an unselected item.
    """
    if not collected:
        msg = "an active-learning round collected no judgments"
        raise ValueError(msg)
    outside = sorted({judgment.item_id for judgment in collected} - set(batch.item_ids))
    if outside:
        msg = f"an active-learning collector returned unselected items: {outside!r}"
        raise ValueError(msg)
    return tuple(collected)


def collect_initial(
    settings: CollectedInitialSettings,
    candidates: Sequence[Item],
    collector: ActiveCollector,
    /,
    *,
    seed: int,
) -> tuple[ActiveJudgment, ...] | AwaitedBatch:
    """Return the collector's responses to the initial draw, or the awaited draw.

    The draw is round 0's batch: ``settings.items_per_stratum`` candidates from
    every stratum ``settings.stratify_by`` names, drawn under ``seed``, so
    running again with the same inputs draws the same batch.

    Raises
    ------
    ValueError
        If a stratum is too small for the draw, or the collector returns no
        judgment or one for an item outside the draw.
    """
    batch = draw_initial_batch(
        candidates,
        AttributeExpression(settings.stratify_by, role="an initial stratification"),
        settings.items_per_stratum,
        seed=seed,
    )
    collected = collector.collect(0, batch)
    if collected is None:
        return AwaitedBatch(0, batch)
    return _checked(collected, batch)


def _next_batch(
    rule: AcquisitionRule,
    fit: ActiveFit,
    remaining: Sequence[Item],
    previous: Sequence[AcquisitionBatch],
    /,
) -> AcquisitionBatch | None:
    """Return the rule's next batch, or None once the pool cannot fill one.

    ``previous`` holds the batches already acquired, in round order.
    """
    if not remaining:
        return None
    candidates = tuple(remaining)
    try:
        return rule.select(
            candidates,
            fit.acquisition_scores(candidates),
            fit.performance,
            previous=previous,
        )
    except PoolExhaustedError:
        return None


def run_active_learning(
    settings: ActiveLearningSettings,
    candidates: Sequence[Item],
    initial: Sequence[ActiveJudgment],
    fitter: ActiveFitter,
    collector: ActiveCollector,
    rule: AcquisitionRule,
    /,
    *,
    bootstrap_seed: int,
) -> ActiveLearningResult:
    """Fit, acquire one batch, ingest, and refit once per round until stopping.

    The candidates an initial judgment names are already judged, so only the
    others are offered to the first round. Round $r$ resamples agreement
    under seed $s+r$, where $s$ is ``bootstrap_seed``. Collection ends with the
    first round at which
    the stopping rule holds; failing that, with the round after which the
    remaining candidates cannot fill a batch, recorded as ``pool-exhausted``;
    failing that, with round ``max_rounds``, recorded as ``max-rounds``. A
    pool the initial judgments leave unable to fill one batch ends the run
    with no round, as ``pool-exhausted``.

    Raises
    ------
    ValueError
        If an initial judgment names an item outside the pool, or a round's
        collector returns no judgment or one for an item outside its batch.
    """
    pooled = {item_key(item) for item in candidates}
    judgments = list(initial)
    judged = {judgment.item_id for judgment in judgments}
    outside_pool = sorted(judged - pooled)
    if outside_pool:
        msg = f"initial judgments name items outside the pool: {outside_pool!r}"
        raise ValueError(msg)
    remaining = [item for item in candidates if item_key(item) not in judged]
    rounds: list[ActiveRound] = []
    ranks: list[float] = []
    fit = fitter.fit(tuple(judgments))
    batch = _next_batch(rule, fit, remaining, ())
    if batch is None:
        return ActiveLearningResult(
            rounds, judgments, awaiting=None, stop_reason="pool-exhausted"
        )
    round_index = 0
    while True:
        round_index += 1
        collected = collector.collect(round_index, batch)
        if collected is None:
            return ActiveLearningResult(
                rounds,
                judgments,
                awaiting=AwaitedBatch(round_index, batch),
                stop_reason=None,
            )
        judgments.extend(_checked(collected, batch))
        selected = set(batch.item_ids)
        remaining = [item for item in remaining if item_key(item) not in selected]
        fit = fitter.fit(tuple(judgments))
        agreement, agreement_interval = _agreement(
            judgments,
            confidence=settings.confidence,
            samples=settings.bootstrap_samples,
            seed=bootstrap_seed + round_index,
        )
        rank, rank_interval = _rank(fit, settings.confidence)
        ranks.append(rank)
        reason = _stop_reason(
            settings,
            agreement,
            agreement_interval,
            rank,
            rank_interval,
            ranks,
        )
        acquired = batch
        if reason is None and not remaining:
            reason = "pool-exhausted"
        elif reason is None and round_index == settings.max_rounds:
            reason = "max-rounds"
        elif reason is None:
            following = _next_batch(
                rule,
                fit,
                remaining,
                (*(prior.batch for prior in rounds), acquired),
            )
            if following is None:
                reason = "pool-exhausted"
            else:
                batch = following
        rounds.append(
            ActiveRound(
                round_index,
                acquired,
                agreement,
                agreement_interval,
                rank,
                rank_interval,
                reason,
                len(judgments),
            )
        )
        if reason is not None:
            return ActiveLearningResult(
                rounds, judgments, awaiting=None, stop_reason=reason
            )

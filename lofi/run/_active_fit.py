"""The MAP refit an active-learning run scores its candidates by.

Each round refits the response model of the run's judgment-fit document by MAP
to every judgment collected so far. An item's predictor columns are the
readouts a trained point published for it, and its factors are the attributes
the active-learning document declares, read from the item's ``item_metadata``.
The refitted point scores each unjudged candidate by its expected response for
a participant the fit never saw, under the declared unseen-participant rule or,
where none is declared, at the population mean.

Performance is read from the cross-validation the active-learning document
declares. Every fold is refitted by MAP on its retained rows and predicts its
held-out rows, so each judged row receives one cross-validated expected
response. An item's observed
value is the mean of its responses and its predicted value the mean of its
rows' cross-validated expectations; the validation ranks pair the two over
every judged item, and a stratum's performance is their Spearman correlation
over the judged items the acquisition rule allocates to that stratum.
"""

from types import MappingProxyType
from typing import TYPE_CHECKING, final

import torch

from lofi._frozen import FrozenValue
from lofi.judgment import (
    AcquisitionScore,
    DesignCodec,
    JudgmentRow,
    MapFitter,
    RowColumns,
    item_key,
    rank_correlation,
)
from lofi.run.fits import map_fold_probabilities, map_point_objective
from lofi.spec import HeldOutParticipantPolicy, derive_seed

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from bead.items.item import Item
    from torch import Tensor

    from lofi.judgment import (
        ActiveJudgment,
        CoverageGate,
        JudgmentDesign,
        MapInference,
        PointFit,
        PointObjective,
        ResponseProgram,
        Splitter,
    )

__all__ = ["ActiveItemColumns", "MapActiveFit", "MapActiveFitter"]

_UNJUDGED = "unjudged"
"""The stem of the participant every candidate is scored for."""


@final
class ActiveItemColumns(FrozenValue):
    """The stratum, factors and predictor readouts of one pool item.

    Parameters
    ----------
    stratum : str
        The stratum the acquisition rule allocates the item under.
    factors : Mapping[str, str]
        Level of each declared factor.
    readouts : Mapping[str, float]
        Value of each predictor column the trained point filled.
    """

    _noun = "active item columns"
    __slots__ = ("_factors", "_readouts", "_stratum")

    def __init__(
        self, stratum: str, factors: Mapping[str, str], readouts: Mapping[str, float]
    ) -> None:
        object.__setattr__(self, "_stratum", stratum)
        object.__setattr__(self, "_factors", MappingProxyType(dict(factors)))
        object.__setattr__(self, "_readouts", MappingProxyType(dict(readouts)))

    @property
    def stratum(self) -> str:
        """Return the stratum the acquisition rule allocates the item under."""
        return self._stratum

    @property
    def factors(self) -> Mapping[str, str]:
        """Return the level of each declared factor."""
        return self._factors

    @property
    def readouts(self) -> Mapping[str, float]:
        """Return the value of each predictor column."""
        return self._readouts


@final
class MapActiveFit:
    """One round's MAP point, its candidate scores and its validation."""

    __slots__ = ("_performance", "_ranks", "_score")

    def __init__(
        self,
        score: _CandidateScorer,
        ranks: tuple[tuple[float, ...], tuple[float, ...]],
        performance: Mapping[str, float],
    ) -> None:
        self._score = score
        self._ranks = ranks
        self._performance = MappingProxyType(dict(performance))

    def acquisition_scores(
        self, items: Sequence[Item], /
    ) -> tuple[AcquisitionScore, ...]:
        """Return each candidate's expected response for an unseen participant."""
        return self._score(items)

    @property
    def performance(self) -> Mapping[str, float]:
        """Return the cross-validated rank correlation of each judged stratum."""
        return self._performance

    @property
    def validation_ranks(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return the judged items' observed and cross-validated predicted means."""
        return self._ranks


@final
class _CandidateScorer:
    """Score candidates at one MAP point fitted to the collected design."""

    __slots__ = ("_fitted", "_fitter", "_objective", "_retained", "_rows", "_seed")

    def __init__(
        self,
        fitter: MapActiveFitter,
        rows: tuple[JudgmentRow, ...],
        retained: JudgmentDesign,
        objective: PointObjective,
        fitted: PointFit,
        seed: int,
    ) -> None:
        self._fitter = fitter
        self._rows = rows
        self._retained = retained
        self._objective = objective
        self._fitted = fitted
        self._seed = seed

    def __call__(self, items: Sequence[Item], /) -> tuple[AcquisitionScore, ...]:
        """Return each candidate's expected response for an unseen participant."""
        if not items:
            return ()
        seen = frozenset(row.participant for row in self._rows)
        participant = _UNJUDGED
        while participant in seen:
            participant = f"{participant}'"
        rows = tuple(self._fitter.row(participant, item_key(item), 0) for item in items)
        expected = self._fitter.expected(
            self._retained,
            rows,
            self._objective,
            self._fitted,
            self._fitter.scoring_policy,
            seed=self._seed,
            unseen=True,
        )
        return tuple(
            AcquisitionScore(item_key(item), value)
            for item, value in zip(items, expected, strict=True)
        )


@final
class MapActiveFitter:
    """Refit a declared response model by MAP to the judgments collected so far.

    Parameters
    ----------
    program : ResponseProgram
        The compiled response model of the judgment-fit document.
    plan : MapInference
        The document's MAP settings.
    levels : int
        Number of ordered response levels.
    factors : tuple[str, ...]
        The declared factors every row records, in column order.
    columns : tuple[str, ...]
        The predictor columns every row records, in column order.
    items : Mapping[str, ActiveItemColumns]
        Stratum, factors and predictor readouts of every pool item, by item
        identity.
    splitter : Splitter
        The declared cross-validation split.
    gate : CoverageGate
        The coverage gate every fold of the split is admitted through.
    unseen : HeldOutParticipantPolicy | None
        The declared rule for a participant a fit never saw.
    seed : int
        Seed every MAP initialization and prediction draw derives from.
    """

    __slots__ = (
        "_columns",
        "_factors",
        "_gate",
        "_items",
        "_levels",
        "_plan",
        "_program",
        "_seed",
        "_splitter",
        "_unseen",
    )

    def __init__(
        self,
        program: ResponseProgram,
        plan: MapInference,
        *,
        levels: int,
        factors: tuple[str, ...],
        columns: tuple[str, ...],
        items: Mapping[str, ActiveItemColumns],
        splitter: Splitter,
        gate: CoverageGate,
        unseen: HeldOutParticipantPolicy | None,
        seed: int,
    ) -> None:
        held = MappingProxyType(dict(items))
        if not held:
            msg = "an active fitter needs at least one pool item"
            raise ValueError(msg)
        for item, values in held.items():
            if tuple(values.factors) != factors or tuple(values.readouts) != columns:
                msg = (
                    f"the pool item {item!r} records the factors "
                    f"{tuple(values.factors)!r} and the columns "
                    f"{tuple(values.readouts)!r}, not {factors!r} and {columns!r}"
                )
                raise ValueError(msg)
        self._program = program
        self._plan = plan
        self._levels = levels
        self._items = held
        self._factors = factors
        self._columns = columns
        self._splitter = splitter
        self._gate = gate
        self._unseen = unseen
        self._seed = seed

    @property
    def scoring_policy(self) -> HeldOutParticipantPolicy:
        """Return the rule candidates are scored for an unseen participant under."""
        return (
            HeldOutParticipantPolicy.PLUG_IN if self._unseen is None else self._unseen
        )

    def row(self, participant: str, item: str, response: int, /) -> JudgmentRow:
        """Return one design row carrying an item's factors and predictors."""
        columns = self._items.get(item)
        if columns is None:
            msg = f"the item {item!r} is not in the candidate pool"
            raise ValueError(msg)
        return JudgmentRow(
            participant,
            item,
            response,
            RowColumns(dict(columns.factors), dict(columns.readouts)),
        )

    def _codec(self) -> DesignCodec:
        return DesignCodec(self._levels, self._factors, self._columns)

    def _fit(
        self, design: JudgmentDesign, seed: int, /
    ) -> tuple[PointObjective, PointFit]:
        objective, _ = map_point_objective(self._program.over(design))
        return objective, MapFitter(self._plan, seed=seed).fit(objective)

    def expected(
        self,
        retained: JudgmentDesign,
        rows: Sequence[JudgmentRow],
        objective: PointObjective,
        fitted: PointFit,
        policy: HeldOutParticipantPolicy | None,
        *,
        seed: int,
        unseen: bool,
    ) -> tuple[float, ...]:
        """Return the expected response of each row under a fitted MAP point."""
        codec = self._codec()
        probabilities: Tensor = map_fold_probabilities(
            self._program,
            codec,
            retained,
            codec.encode(rows),
            objective,
            fitted,
            policy,
            seed=seed,
            unseen=unseen,
        )
        scale = torch.arange(
            self._levels, dtype=probabilities.dtype, device=probabilities.device
        )
        return tuple(float(value) for value in (probabilities * scale).sum(dim=-1))

    def _cross_validated(
        self, rows: tuple[JudgmentRow, ...], seed: int, /
    ) -> dict[int, float]:
        """Return each row's cross-validated expected response, by row position."""
        positions = {id(row): position for position, row in enumerate(rows)}
        predicted: dict[int, float] = {}
        codec = self._codec()

        for fold in self._splitter.split(rows, self._gate).folds:
            retained = codec.encode(fold.retained)
            fold_seed = derive_seed(seed, f"fold:{fold.index}")
            objective, fitted = self._fit(retained, fold_seed)
            seen = frozenset(row.participant for row in fold.retained)

            for unseen in (False, True):
                selected = tuple(
                    row
                    for row in fold.held_out
                    if (row.participant not in seen) is unseen
                )
                if not selected:
                    continue
                if unseen and fold.policy is None:
                    msg = (
                        f"fold {fold.index} holds out participants its retained "
                        "rows never show, and the cross-validation declares no "
                        "unseen-participant rule to predict them under"
                    )
                    raise ValueError(msg)
                values = self.expected(
                    retained,
                    selected,
                    objective,
                    fitted,
                    fold.policy if unseen else None,
                    seed=derive_seed(fold_seed, f"prediction:{unseen}"),
                    unseen=unseen,
                )
                for row, value in zip(selected, values, strict=True):
                    predicted[positions[id(row)]] = value

        return predicted

    def fit(self, judgments: Sequence[ActiveJudgment], /) -> MapActiveFit:
        """Refit the response model to every judgment and validate it.

        Raises
        ------
        ValueError
            If a judgment names an item outside the pool, a judged row receives
            no cross-validated prediction, or a judged stratum has too few
            distinct items to rank.
        """
        rows = tuple(
            self.row(judgment.participant_id, judgment.item_id, judgment.response)
            for judgment in judgments
        )
        seed = derive_seed(self._seed, f"judgments:{len(rows)}")
        design = self._codec().encode(rows)
        objective, fitted = self._fit(design, seed)
        predicted = self._cross_validated(rows, seed)
        missing = len(rows) - len(predicted)
        if missing:
            msg = f"{missing} judged rows received no cross-validated prediction"
            raise ValueError(msg)

        observed: dict[str, list[float]] = {}
        expected: dict[str, list[float]] = {}
        for position, row in enumerate(rows):
            observed.setdefault(row.item, []).append(float(row.response))
            expected.setdefault(row.item, []).append(predicted[position])
        items = tuple(sorted(observed))
        means = {item: sum(observed[item]) / len(observed[item]) for item in items}
        predictions = {
            item: sum(expected[item]) / len(expected[item]) for item in items
        }

        performance: dict[str, float] = {}
        for stratum in sorted({self._items[item].stratum for item in items}):
            members = tuple(
                item for item in items if self._items[item].stratum == stratum
            )
            try:
                performance[stratum] = rank_correlation(
                    tuple(means[item] for item in members),
                    tuple(predictions[item] for item in members),
                )
            except ValueError as refused:
                msg = (
                    f"the judged items of stratum {stratum!r} cannot be ranked: "
                    f"{refused}"
                )
                raise ValueError(msg) from refused

        return MapActiveFit(
            _CandidateScorer(
                self, rows, design, objective, fitted, derive_seed(seed, "candidates")
            ),
            (
                tuple(means[item] for item in items),
                tuple(predictions[item] for item in items),
            ),
            performance,
        )

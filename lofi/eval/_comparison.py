"""Source selection from held-out fit rows and capacity-matched twins."""

import json
from hashlib import sha256
from math import isfinite
from statistics import fmean
from typing import TYPE_CHECKING, Literal, NamedTuple, Protocol, cast, final

from lofi._checks import ensure_finite, ensure_name, ensure_non_negative
from lofi._frozen import FrozenValue
from lofi.eval._prereg import Preregistration
from lofi.eval._scrambled import (
    SignatureAssignments,
    TwinFit,
    empirical_twin_null,
)
from lofi.eval._values import Comparison
from lofi.stats import (
    Correction,
    Dependence,
    Equivalence,
    EquivalenceResult,
    PValue,
    paired_permutation,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi._types import JsonValue
    from lofi.spec.reducers import SourceComparisonReducerSettings

__all__ = [
    "PointMetadata",
    "Reduction",
    "SourceComparisonObservation",
    "SourceComparisonReducer",
    "reduce_source_comparisons",
]

type ComparisonRole = Literal[
    "candidate", "twin", "reversed-ground-truth", "lexicon-scrambled"
]
type WrongWinnerClass = Literal["source", "simpler", "richer", "differently-directed"]

_ROLES: frozenset[str] = frozenset(
    {"candidate", "twin", "reversed-ground-truth", "lexicon-scrambled"}
)


@final
class PointMetadata(FrozenValue):
    """One expanded point and the measured facts that license its comparison."""

    _noun = "comparison point metadata"
    __slots__ = (
        "_assignments",
        "_config_json",
        "_parameter_count",
        "_point_id",
        "_result_seed",
        "_seed",
        "_signature_fingerprint",
    )

    def __init__(
        self,
        config: Mapping[str, JsonValue],
        *,
        point_id: str,
        seed: int,
        result_seed: int,
        parameter_count: int,
        signature_fingerprint: str,
        assignments: SignatureAssignments | None = None,
    ) -> None:
        if assignments is not None and not isinstance(
            assignments, SignatureAssignments
        ):
            msg = (
                f"comparison point metadata has malformed assignments: {assignments!r}"
            )
            raise TypeError(msg)
        try:
            document = json.dumps(
                dict(config),
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        except (TypeError, ValueError) as error:
            msg = f"comparison point metadata has a non-JSON configuration: {error}"
            raise ValueError(msg) from error

        object.__setattr__(self, "_config_json", document)
        object.__setattr__(self, "_point_id", ensure_name("a point id", point_id))
        object.__setattr__(self, "_seed", ensure_non_negative("a point seed", seed))
        object.__setattr__(
            self,
            "_result_seed",
            ensure_non_negative("a point result seed", result_seed),
        )
        object.__setattr__(
            self,
            "_parameter_count",
            ensure_non_negative("a measured parameter count", parameter_count),
        )
        object.__setattr__(
            self,
            "_signature_fingerprint",
            ensure_name("a signature fingerprint", signature_fingerprint),
        )
        object.__setattr__(self, "_assignments", assignments)

    @property
    def config(self) -> dict[str, JsonValue]:
        """Return a fresh copy of the validated expanded configuration."""
        return cast("dict[str, JsonValue]", json.loads(self._config_json))

    @property
    def point_id(self) -> str:
        """Return the expanded point identifier."""
        return self._point_id

    @property
    def seed(self) -> int:
        """Return the sweep seed."""
        return self._seed

    @property
    def result_seed(self) -> int:
        """Return the derived seed written on the point's result rows."""
        return self._result_seed

    @property
    def parameter_count(self) -> int:
        """Return the measured trainable-model capacity."""
        return self._parameter_count

    @property
    def signature_fingerprint(self) -> str:
        """Return the exact codec-signature fingerprint."""
        return self._signature_fingerprint

    @property
    def assignments(self) -> SignatureAssignments | None:
        """Return the learner's slot assignments, when this point declares them."""
        return self._assignments


class Reduction(NamedTuple):
    """Rows and statistical comparisons emitted by one or more reducers."""

    rows: tuple[Mapping[str, JsonValue], ...]
    comparisons: tuple[Comparison, ...]


@final
class SourceComparisonObservation(FrozenValue):
    """One held-out fit with the point metadata needed for source selection."""

    _noun = "source comparison observation"
    __slots__ = (
        "_arm",
        "_assignments",
        "_candidate",
        "_data_size",
        "_description_length",
        "_direction",
        "_parameter_count",
        "_role",
        "_score",
        "_seed",
        "_signature_digest",
        "_source",
    )

    def __init__(
        self,
        *,
        source: str,
        candidate: str,
        arm: str,
        role: str,
        seed: int,
        data_size: int,
        score: float,
        description_length: float,
        signature_digest: str,
        parameter_count: int,
        direction: str,
        assignments: SignatureAssignments,
    ) -> None:
        if role not in _ROLES:
            msg = f"a source comparison has an unknown role: {role!r}"
            raise ValueError(msg)
        if not isinstance(assignments, SignatureAssignments):
            msg = f"a source comparison's assignments are not one: {assignments!r}"
            raise TypeError(msg)

        object.__setattr__(self, "_source", ensure_name("a source arm", source))
        object.__setattr__(
            self, "_candidate", ensure_name("a candidate arm", candidate)
        )
        object.__setattr__(self, "_arm", ensure_name("a fitted arm", arm))
        object.__setattr__(self, "_role", cast("ComparisonRole", role))
        object.__setattr__(self, "_seed", ensure_non_negative("a seed", seed))
        size = ensure_non_negative("a data size", data_size)
        if size == 0:
            msg = "a source comparison has an empty data set"
            raise ValueError(msg)
        object.__setattr__(self, "_data_size", size)
        object.__setattr__(self, "_score", ensure_finite("a held-out score", score))
        length = ensure_finite("a description length", description_length)
        if length < 0.0:
            msg = f"a description length is negative: {description_length!r}"
            raise ValueError(msg)
        object.__setattr__(self, "_description_length", length)
        object.__setattr__(
            self,
            "_signature_digest",
            ensure_name("a signature digest", signature_digest),
        )
        object.__setattr__(
            self,
            "_parameter_count",
            ensure_non_negative("a measured parameter count", parameter_count),
        )
        object.__setattr__(
            self, "_direction", ensure_name("a direction class", direction)
        )
        object.__setattr__(self, "_assignments", assignments)

    @property
    def source(self) -> str:
        """Return the data-generating arm."""
        return self._source

    @property
    def candidate(self) -> str:
        """Return the candidate whose fit or control this is."""
        return self._candidate

    @property
    def arm(self) -> str:
        """Return the fitted point's distinct arm name."""
        return self._arm

    @property
    def role(self) -> ComparisonRole:
        """Return the fit's role in the comparison."""
        return self._role

    @property
    def seed(self) -> int:
        """Return the fitting seed."""
        return self._seed

    @property
    def data_size(self) -> int:
        """Return the training-data size."""
        return self._data_size

    @property
    def score(self) -> float:
        """Return the held-out predictive score before description length."""
        return self._score

    @property
    def description_length(self) -> float:
        """Return the description length charge in score units."""
        return self._description_length

    @property
    def selection_score(self) -> float:
        """Return held-out fit minus the description length charge."""
        return self._score - self._description_length

    @property
    def signature_digest(self) -> str:
        """Return the exact signature digest licensing the comparison."""
        return self._signature_digest

    @property
    def parameter_count(self) -> int:
        """Return the point's measured parameter count."""
        return self._parameter_count

    @property
    def direction(self) -> str:
        """Return the grammar's declared direction class."""
        return self._direction

    @property
    def assignments(self) -> SignatureAssignments:
        """Return the assignment identity used by the empirical twin null."""
        return self._assignments


def _grouped(
    observations: Sequence[SourceComparisonObservation],
) -> dict[tuple[str, int], tuple[SourceComparisonObservation, ...]]:
    groups: dict[tuple[str, int], list[SourceComparisonObservation]] = {}
    for observation in observations:
        if not isinstance(observation, SourceComparisonObservation):
            msg = f"a source-comparison observation is not one: {observation!r}"
            raise TypeError(msg)
        groups.setdefault((observation.source, observation.data_size), []).append(
            observation
        )
    return {key: tuple(value) for key, value in sorted(groups.items())}


def _assignment_inventory(
    assignments: SignatureAssignments, /
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Return an order-insensitive identity for one signature inventory."""
    return (
        tuple(sorted(assignments.primitives)),
        tuple(sorted(assignments.constructors)),
        tuple(sorted(assignments.combinators)),
    )


def _licensed(observations: Sequence[SourceComparisonObservation]) -> tuple[str, int]:
    by_candidate: dict[
        str, set[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]]
    ] = {}
    for row in observations:
        by_candidate.setdefault(row.candidate, set()).add(
            _assignment_inventory(row.assignments)
        )

    mismatched = tuple(
        candidate
        for candidate, inventories in sorted(by_candidate.items())
        if len(inventories) != 1
    )
    if mismatched:
        msg = (
            "a source comparison crosses signature-family inventories within "
            f"candidates {mismatched!r}"
        )
        raise ValueError(msg)
    capacities = {row.parameter_count for row in observations}
    if len(capacities) != 1:
        msg = (
            "a source comparison lacks exact matched capacity: measured "
            f"{sorted(capacities)!r}"
        )
        raise ValueError(msg)
    inventory = tuple(
        (candidate, next(iter(inventories)))
        for candidate, inventories in sorted(by_candidate.items())
    )
    identity = sha256(
        json.dumps(inventory, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    return identity, next(iter(capacities))


def _by_seed(
    observations: Sequence[SourceComparisonObservation],
    *,
    candidate: str,
    role: ComparisonRole,
) -> dict[int, SourceComparisonObservation]:
    selected: dict[int, SourceComparisonObservation] = {}
    for row in observations:
        if row.candidate != candidate or row.role != role:
            continue
        if row.seed in selected:
            msg = f"the {role} arm {candidate!r} occurs twice at seed {row.seed}"
            raise ValueError(msg)
        selected[row.seed] = row
    return selected


def _winner_class(
    winner: SourceComparisonObservation, source: SourceComparisonObservation
) -> WrongWinnerClass:
    if winner.candidate == source.candidate:
        return "source"
    if winner.description_length < source.description_length:
        return "simpler"
    if winner.description_length > source.description_length:
        return "richer"
    if winner.direction != source.direction:
        return "differently-directed"
    msg = (
        f"the wrong winner {winner.candidate!r} is neither simpler, richer, nor "
        "differently directed from the source"
    )
    raise ValueError(msg)


def reduce_source_comparisons(
    observations: Sequence[SourceComparisonObservation],
    preregistration: Preregistration,
    *,
    alpha: float,
    correction: Correction,
    dependence: Dependence,
    replicates: int,
    seed: int,
    equivalence_replicates: int | None = None,
    equivalence_seed: int | None = None,
    falsifier: str,
    smoke_test: bool,
) -> Reduction:
    """Reduce held-out fits to deltas, tests, verdicts, and winner matrices.

    The candidate contrasts and validity controls at one source and data size
    form one family, which ``correction`` adjusts at ``alpha`` under the
    ``dependence`` regime the experiment declares.
    """
    if not isinstance(preregistration, Preregistration):
        msg = f"a source comparison's preregistration is not one: {preregistration!r}"
        raise TypeError(msg)
    level = ensure_finite("a source-comparison level", alpha)
    if not 0.0 < level < 0.5:
        msg = f"a source-comparison level is not below one half: {alpha!r}"
        raise ValueError(msg)
    if not isinstance(correction, Correction):
        msg = f"a source-comparison correction is not one: {correction!r}"
        raise TypeError(msg)
    if not isinstance(dependence, Dependence):
        msg = f"a source-comparison dependence regime is not one: {dependence!r}"
        raise TypeError(msg)
    permutations = ensure_non_negative("a permutation replicate count", replicates)
    permutation_seed = ensure_non_negative("a permutation seed", seed)
    equivalence_draws = (
        permutations
        if equivalence_replicates is None
        else ensure_non_negative(
            "an equivalence replicate count", equivalence_replicates
        )
    )
    equivalence_draw_seed = (
        permutation_seed
        if equivalence_seed is None
        else ensure_non_negative("an equivalence seed", equivalence_seed)
    )
    if permutations == 0 or equivalence_draws == 0:
        msg = "a source comparison declares no statistical-test replicates"
        raise ValueError(msg)

    rows_out: list[Mapping[str, JsonValue]] = []
    comparisons: list[Comparison] = []
    threshold: dict[str, int] = {}

    for (source_name, data_size), group in _grouped(observations).items():
        signature, capacity = _licensed(group)
        candidates = tuple(
            sorted({row.candidate for row in group if row.role == "candidate"})
        )
        if source_name not in candidates:
            msg = f"the source arm {source_name!r} has no primary fit"
            raise ValueError(msg)
        primary = {
            candidate: _by_seed(group, candidate=candidate, role="candidate")
            for candidate in candidates
        }
        seeds = tuple(sorted(primary[source_name]))
        if not seeds or any(
            tuple(sorted(found)) != seeds for found in primary.values()
        ):
            msg = f"source-comparison candidates at size {data_size} do not share seeds"
            raise ValueError(msg)

        source_scores = tuple(
            primary[source_name][draw].selection_score for draw in seeds
        )
        contrasts: list[tuple[str, ComparisonRole, PValue, float]] = []
        tost: dict[str, EquivalenceResult] = {}

        for candidate in candidates:
            if candidate == source_name:
                continue
            scores = tuple(primary[candidate][draw].selection_score for draw in seeds)
            p_value = paired_permutation(
                source_scores,
                scores,
                replicates=permutations,
                seed=permutation_seed,
                tail="greater",
            )
            effect = fmean(
                left - right for left, right in zip(source_scores, scores, strict=True)
            )
            declared = preregistration.hypothesis(source_name, candidate)
            margin = preregistration.margin(source_name, candidate)
            equivalence = Equivalence(
                margin=margin,
                replicates=equivalence_draws,
                seed=equivalence_draw_seed,
                level=level,
            ).compare(source_scores, scores)
            tost[candidate] = equivalence
            contrasts.append((candidate, "candidate", p_value, effect))
            comparisons.append(
                Comparison(
                    source_name,
                    candidate,
                    statistic=effect,
                    p_value=p_value,
                    effect=effect,
                    falsifier=falsifier,
                    smoke_test=smoke_test,
                    family=declared.family,
                )
            )
            for draw in seeds:
                source_row = primary[source_name][draw]
                candidate_row = primary[candidate][draw]
                rows_out.append(
                    {
                        "reducer": "source-comparison-delta",
                        "source": source_name,
                        "candidate": candidate,
                        "data_size": data_size,
                        "seed": draw,
                        "value": source_row.selection_score
                        - candidate_row.selection_score,
                        "predictive_delta": source_row.score - candidate_row.score,
                        "description_length_delta": source_row.description_length
                        - candidate_row.description_length,
                        "signature_fingerprint": signature,
                        "matched_capacity": capacity,
                    }
                )

        controls: dict[ComparisonRole, dict[int, SourceComparisonObservation]] = {}
        for role in cast(
            "tuple[ComparisonRole, ...]",
            ("reversed-ground-truth", "lexicon-scrambled"),
        ):
            held = _by_seed(group, candidate=source_name, role=role)
            if tuple(sorted(held)) != seeds:
                msg = f"the validity control {role!r} does not cover the source seeds"
                raise ValueError(msg)
            controls[role] = held
            scores = tuple(held[draw].selection_score for draw in seeds)
            p_value = paired_permutation(
                source_scores,
                scores,
                replicates=permutations,
                seed=permutation_seed,
                tail="greater",
            )
            effect = fmean(
                left - right for left, right in zip(source_scores, scores, strict=True)
            )
            contrasts.append((role, role, p_value, effect))

        decisions = correction.apply(
            tuple(contrast[2] for contrast in contrasts),
            level,
            dependence,
        )
        test_passes: dict[str, bool] = {}
        for index, (candidate, role, p_value, effect) in enumerate(contrasts):
            corrected = decisions.adjusted[index]
            rejected = decisions.reject[index]
            test_passes[candidate] = rejected and effect > 0.0
            equivalence = tost.get(candidate)
            rows_out.append(
                {
                    "reducer": "source-comparison-test",
                    "source": source_name,
                    "candidate": candidate,
                    "role": role,
                    "data_size": data_size,
                    "value": effect,
                    "raw_p": p_value.value,
                    "corrected_p": corrected,
                    "correction": decisions.correction,
                    "dependence": decisions.dependence.value,
                    "reject": rejected,
                    "tost_verdict": None
                    if equivalence is None
                    else equivalence.verdict.value,
                    "tost_margin": None if equivalence is None else equivalence.margin,
                    "tost_lower_inside_p": None
                    if equivalence is None
                    else equivalence.lower.inside.value,
                    "tost_upper_inside_p": None
                    if equivalence is None
                    else equivalence.upper.inside.value,
                    "signature_fingerprint": signature,
                    "matched_capacity": capacity,
                }
            )

        twin_passes: dict[str, bool] = {}
        for candidate in candidates:
            per_seed: list[bool] = []
            for draw in seeds:
                observed = primary[candidate][draw]
                twins = tuple(
                    row
                    for row in group
                    if row.candidate == candidate
                    and row.role == "twin"
                    and row.seed == draw
                )
                if len(twins) < 3:
                    msg = (
                        f"candidate {candidate!r} has {len(twins)} twins at seed "
                        f"{draw}, fewer than three"
                    )
                    raise ValueError(msg)
                null = empirical_twin_null(
                    TwinFit(observed.assignments, observed.selection_score),
                    tuple(
                        TwinFit(twin.assignments, twin.selection_score)
                        for twin in twins
                    ),
                    declared_twins=len(twins),
                )
                passed = min(null.deltas) > 0.0 and null.quantile >= 1.0 - level
                per_seed.append(passed)
                rows_out.append(
                    {
                        "reducer": "source-comparison-twin",
                        "source": source_name,
                        "candidate": candidate,
                        "data_size": data_size,
                        "seed": draw,
                        "value": null.quantile,
                        "minimum_delta": min(null.deltas),
                        "twins": len(null.twins),
                        "passed": passed,
                        "signature_fingerprint": signature,
                        "matched_capacity": capacity,
                    }
                )
            twin_passes[candidate] = all(per_seed)

        means = {
            candidate: fmean(row.selection_score for row in primary[candidate].values())
            for candidate in candidates
        }
        ordered = sorted(
            candidates, key=lambda candidate: (-means[candidate], candidate)
        )
        winner = ordered[0]
        source_rank = ordered.index(source_name) + 1
        source_example = primary[source_name][seeds[0]]
        winner_example = primary[winner][seeds[0]]
        wrong_class = _winner_class(winner_example, source_example)
        beats_candidates = all(
            test_passes[candidate]
            for candidate in candidates
            if candidate != source_name
        )
        controls_pass = all(test_passes[role] for role in controls)
        beats_twins = twin_passes[source_name]
        positive = (
            winner == source_name
            and source_rank == 1
            and beats_candidates
            and beats_twins
            and controls_pass
        )
        if positive and source_name not in threshold:
            threshold[source_name] = data_size
        rows_out.extend(
            (
                {
                    "reducer": "source-comparison-verdict",
                    "source": source_name,
                    "data_size": data_size,
                    "winner": winner,
                    "wrong_winner_class": wrong_class,
                    "source_rank": source_rank,
                    "positive": positive,
                    "beats_candidates": beats_candidates,
                    "beats_twins": beats_twins,
                    "validity_controls_pass": controls_pass,
                    "signature_fingerprint": signature,
                    "matched_capacity": capacity,
                },
                {
                    "reducer": "source-by-winner",
                    "source": source_name,
                    "winner": winner,
                    "data_size": data_size,
                    "value": 1,
                    "wrong_winner_class": wrong_class,
                    "signature_fingerprint": signature,
                    "matched_capacity": capacity,
                },
            )
        )

    rows_out.extend(
        {
            "reducer": "source-win-threshold",
            "source": source_name,
            "data_size": threshold.get(source_name),
            "value": threshold.get(source_name),
        }
        for source_name in sorted({row.source for row in observations})
    )
    return Reduction(tuple(rows_out), tuple(comparisons))


def _path(config: Mapping[str, JsonValue], dotted: str, /) -> JsonValue:
    """Return one value reached by a strict dotted path."""
    value: JsonValue = dict(config)
    walked: list[str] = []
    for name in dotted.split("."):
        walked.append(name)
        if not isinstance(value, dict) or name not in value:
            stopped = ".".join(walked)
            msg = f"point configuration has no path {dotted!r} (stopped at {stopped!r})"
            raise ValueError(msg)
        value = value[name]
    return value


def _text_value(config: Mapping[str, JsonValue], path: str, /) -> str:
    value = _path(config, path)
    if not isinstance(value, str) or not value:
        msg = f"point configuration path {path!r} is not named text: {value!r}"
        raise TypeError(msg)
    return value


def _count_value(config: Mapping[str, JsonValue], path: str, /) -> int:
    value = _path(config, path)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        msg = f"point configuration path {path!r} is not a count: {value!r}"
        raise TypeError(msg)
    return value


class _ComparisonContext(Protocol):
    """Expose experiment metadata needed by the source comparison reducer."""

    @property
    def preregistration(self) -> Preregistration: ...

    @property
    def alpha(self) -> float: ...

    @property
    def correction(self) -> Correction: ...

    @property
    def dependence(self) -> Dependence: ...

    def point(self, config_hash: str, seed: int, /) -> PointMetadata: ...


@final
class SourceComparisonReducer:
    """Build source comparison observations from rows and expanded points."""

    __slots__ = ("_context", "_settings")

    def __init__(
        self,
        settings: SourceComparisonReducerSettings,
        context: _ComparisonContext,
    ) -> None:
        self._settings = settings
        self._context = context

    def reduce(self, rows: Sequence[Mapping[str, JsonValue]], /) -> Reduction:
        """Return the declared source comparison report."""
        settings = self._settings
        selected = tuple(row for row in rows if row.get("metric") == settings.metric)
        lengths = tuple(
            row
            for row in rows
            if row.get("metric") == settings.description_length_metric
        )
        if not selected:
            msg = (
                "a source-comparison reducer found no rows for metric "
                f"{settings.metric!r}"
            )
            raise ValueError(msg)

        descriptions: dict[tuple[str, int, str], float] = {}
        for row in lengths:
            config_hash = self._row_text(row, "config_hash")
            row_seed = self._row_count(row, "seed")
            method = self._row_text(row, settings.method_column)
            key = (config_hash, row_seed, method)
            if key in descriptions:
                msg = (
                    f"description length is reported twice for point {config_hash!r}, "
                    f"seed {row_seed}, method {method!r}"
                )
                raise ValueError(msg)
            descriptions[key] = self._row_number(row, settings.value_column)

        by_candidate: dict[str, list[str]] = {}
        raw: list[
            tuple[
                PointMetadata,
                Mapping[str, JsonValue],
                str,
                str,
                str,
                int,
                str,
                str,
                int,
                float,
            ]
        ] = []
        for row in selected:
            config_hash = self._row_text(row, "config_hash")
            row_seed = self._row_count(row, "seed")
            method = self._row_text(row, settings.method_column)
            point = self._context.point(config_hash, row_seed)
            source = _text_value(point.config, settings.source_path)
            candidate = _text_value(point.config, settings.candidate_path)
            data_size = _count_value(point.config, settings.data_size_path)
            condition = _text_value(point.config, settings.condition_path)
            direction = _text_value(point.config, settings.direction_path)
            twin = _text_value(point.config, settings.twin_path)
            units = self._row_count(row, "n")
            if units == 0:
                msg = (
                    f"point {point.point_id!r} reports its primary metric over no "
                    "held-out units"
                )
                raise ValueError(msg)
            key = (config_hash, row_seed, method)
            if key not in descriptions:
                msg = f"point {point.point_id!r} has no matching description-length row"
                raise ValueError(msg)
            charge = descriptions[key] / units
            by_candidate.setdefault(candidate, []).append(twin)
            raw.append(
                (
                    point,
                    row,
                    source,
                    candidate,
                    condition,
                    data_size,
                    direction,
                    twin,
                    point.parameter_count,
                    charge,
                )
            )

        for candidate, twins in by_candidate.items():
            identities = tuple(sorted(set(twins)))
            if settings.observed_twin not in identities:
                msg = (
                    f"candidate {candidate!r} has no observed twin arm "
                    f"{settings.observed_twin!r}"
                )
                raise ValueError(msg)

        observations: list[SourceComparisonObservation] = []
        for (
            point,
            row,
            source,
            candidate,
            condition,
            data_size,
            direction,
            twin,
            parameter_count,
            charge,
        ) in raw:
            role: ComparisonRole
            if condition == settings.primary_condition:
                role = "candidate" if twin == settings.observed_twin else "twin"
            elif condition == settings.reversed_ground_truth_condition:
                role = "reversed-ground-truth"
            elif condition == settings.lexicon_scrambled_condition:
                role = "lexicon-scrambled"
            else:
                msg = (
                    f"point {point.point_id!r} carries undeclared source-comparison "
                    f"condition {condition!r}"
                )
                raise ValueError(msg)
            observations.append(
                SourceComparisonObservation(
                    source=source,
                    candidate=candidate,
                    arm=f"{candidate}:{condition}:{twin}",
                    role=role,
                    seed=point.seed,
                    data_size=data_size,
                    score=self._row_number(row, settings.value_column),
                    description_length=charge,
                    signature_digest=point.signature_fingerprint,
                    parameter_count=parameter_count,
                    direction=direction,
                    assignments=self._point_assignments(point),
                )
            )
        return reduce_source_comparisons(
            observations,
            self._context.preregistration,
            alpha=self._context.alpha,
            correction=self._context.correction,
            dependence=self._context.dependence,
            replicates=settings.permutation.replicates,
            seed=settings.permutation.seed,
            equivalence_replicates=settings.equivalence.replicates,
            equivalence_seed=settings.equivalence.seed,
            falsifier=settings.falsifier,
            smoke_test=settings.smoke_test,
        )

    @staticmethod
    def _point_assignments(point: PointMetadata) -> SignatureAssignments:
        assignments = point.assignments

        if assignments is None:
            msg = (
                f"point {point.point_id!r} declares no signature_assignment for "
                "source comparison"
            )
            raise ValueError(msg)

        return assignments

    @staticmethod
    def _row_text(row: Mapping[str, JsonValue], column: str) -> str:
        value = row.get(column)
        if not isinstance(value, str) or not value:
            msg = (
                f"source-comparison row column {column!r} is not named text: {value!r}"
            )
            raise TypeError(msg)
        return value

    @staticmethod
    def _row_count(row: Mapping[str, JsonValue], column: str) -> int:
        value = row.get(column)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            msg = f"source-comparison row column {column!r} is not a count: {value!r}"
            raise TypeError(msg)
        return value

    @staticmethod
    def _row_number(row: Mapping[str, JsonValue], column: str) -> float:
        value = row.get(column)
        if isinstance(value, bool) or not isinstance(value, int | float):
            msg = f"source-comparison row column {column!r} is not numeric: {value!r}"
            raise TypeError(msg)
        number = float(value)
        if not isfinite(number):
            msg = f"source-comparison row column {column!r} is not finite: {value!r}"
            raise ValueError(msg)
        return number

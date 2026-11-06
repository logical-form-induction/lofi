"""Matched-budget comparison of fragment-named grounding regimes.

One arm is one declared position of the yoked grounding, form-readout, and
objective-signal axis. The reducer reads completed run directories, pairs
outcomes by run seed, and reports every arm's held-out outcomes beside its
optimizer steps and measured prover clocks. Synthetic runs report label
likelihood, label accuracy, and symbolic entailment preservation. Judgment
runs report held-out judgment likelihood through the same reduction route.
"""

import json
from datetime import UTC, datetime
from enum import Enum
from itertools import combinations
from math import fsum
from statistics import fmean
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple, Protocol, cast, override

from lofi._checks import (
    ensure_finite,
    ensure_name,
    ensure_non_negative,
    ensure_positive,
)
from lofi._frozen import FrozenValue
from lofi._types import ItemId
from lofi.eval._boundary import (
    ValidationError,
    decode_manifest,
    decode_metrics,
    decode_object,
    decode_rows,
)
from lofi.eval._prereg import PreregistrationError
from lofi.eval._values import Comparison
from lofi.stats import (
    Bootstrap,
    Equivalence,
    EquivalenceResult,
    Interval,
    Verdict,
    paired_effect_size,
    paired_permutation,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from lofi._types import JsonValue

__all__ = [
    "GROUNDING_FIELD",
    "PRIMARY_LEFT",
    "PRIMARY_RIGHT",
    "READOUT_FIELD",
    "REGIME_FIELDS",
    "RESAMPLING_UNIT",
    "SIGNAL_FIELD",
    "ArmCost",
    "ArmSpecification",
    "BudgetSetting",
    "CurvePoint",
    "EstimatorComparison",
    "EstimatorProtocol",
    "EstimatorReport",
    "EstimatorRow",
    "EstimatorVerdict",
    "GroundedRun",
    "Outcome",
    "RegimeRunDirectory",
    "RunKind",
    "SpecificationMismatchError",
    "UnmatchedBudgetError",
    "estimator_arms",
    "estimator_report",
    "estimator_verdict",
    "paired_contrast",
    "reduce_grounding_runs",
]

GROUNDING_FIELD: Final = "model.grounding"
READOUT_FIELD: Final = "model.readouts.forms.decoding"
SIGNAL_FIELD: Final = "loss.terms"
REGIME_FIELDS: Final = (GROUNDING_FIELD, READOUT_FIELD, SIGNAL_FIELD)
"""The three valid configuration assignments advanced together as one regime.

The loss owner is varied as a unit because declared settings do not address
members of a list by numeric path segments.
"""

PRIMARY_LEFT: Final = "prover-reward"
PRIMARY_RIGHT: Final = "expected-embedding"
RESAMPLING_UNIT: Final = "seed"


class Outcome(Enum):
    """A held-out outcome reduced across matched runs."""

    SYMBOLIC_RATE = "entailment-preserving-rate-at-1"
    LABEL_LIKELIHOOD = "held-out-label-log-likelihood"
    LABEL_ACCURACY = "entailment-label-accuracy"
    JUDGMENT_LIKELIHOOD = "held-out-judgment-log-likelihood"


class RunKind(Enum):
    """The response carried by a regime sweep."""

    SYNTHETIC = "synthetic"
    JUDGMENT = "judgment"

    @property
    def outcomes(self) -> tuple[Outcome, ...]:
        """Return the outcomes every run of this kind reports."""
        if self is RunKind.SYNTHETIC:
            return (
                Outcome.LABEL_LIKELIHOOD,
                Outcome.LABEL_ACCURACY,
                Outcome.SYMBOLIC_RATE,
            )

        return (Outcome.JUDGMENT_LIKELIHOOD,)


class BudgetSetting(Enum):
    """A named optimizer-step budget in a comparison curve."""

    HALF = "half"
    FULL = "full"


class EstimatorVerdict(Enum):
    """The joint reading of the paired test and TOST."""

    LEFT_SEPARATES = "left-separates"
    RIGHT_SEPARATES = "right-separates"
    EQUIVALENT_WITHIN_MARGIN = "equivalent-within-margin"
    INDETERMINATE = "indeterminate"


class SpecificationMismatchError(ValueError):
    """Signals arms that differ outside the declared yoked regime."""


class UnmatchedBudgetError(ValueError):
    """Signals arms compared after different numbers of optimizer steps."""


def _proportion(field: str, value: float) -> float:
    held = ensure_finite(field, value)

    if not 0.0 <= held <= 1.0:
        msg = f"{field} lies outside the unit interval: {value!r}"
        raise ValueError(msg)

    return held


def _canonical(value: JsonValue) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True)


def _moment(field: str, value: datetime) -> datetime:
    if not isinstance(value, datetime):
        msg = f"{field} is not a datetime: {value!r}"
        raise TypeError(msg)

    if value.utcoffset() is None:
        msg = f"{field} names no UTC offset: {value!r}"
        raise ValueError(msg)

    return value


class ArmSpecification(FrozenValue):
    """One fragment-named position of the yoked regime axis."""

    _noun = "arm specification"
    __slots__ = ("_fields", "_name", "_regime")

    def __init__(self, name: str, fields: Mapping[str, JsonValue]) -> None:
        held = dict(fields)
        missing = tuple(field for field in REGIME_FIELDS if field not in held)

        if missing:
            msg = f"the arm {name!r} does not declare regime fields {missing!r}"
            raise ValueError(msg)

        object.__setattr__(self, "_name", ensure_name("an arm fragment", name))
        object.__setattr__(self, "_fields", MappingProxyType(held))
        object.__setattr__(
            self,
            "_regime",
            MappingProxyType({field: held[field] for field in REGIME_FIELDS}),
        )

    @property
    def name(self) -> str:
        """Return the configuration fragment naming this arm."""
        return self._name

    @property
    def fields(self) -> Mapping[str, JsonValue]:
        """Return every flattened run field."""
        return self._fields

    @property
    def regime(self) -> Mapping[str, JsonValue]:
        """Return the three assignments advanced together for this arm."""
        return self._regime

    def to_spec(self) -> dict[str, JsonValue]:
        """Return the arm as report columns."""
        return {
            "arm": self._name,
            "regime": dict(self._regime),
            "fields": dict(self._fields),
        }

    @override
    def __eq__(self, other: object) -> bool:
        if type(other) is not ArmSpecification:
            return NotImplemented
        return self._name == other._name and dict(self._fields) == dict(other._fields)

    @override
    def __hash__(self) -> int:
        fields = tuple(
            sorted((name, _canonical(value)) for name, value in self._fields.items())
        )
        return hash((ArmSpecification, self._name, fields))


def _one_yoked_axis(arms: Sequence[ArmSpecification]) -> None:
    if len(arms) < 2:
        msg = "an estimator comparison needs at least two declared arms"
        raise ValueError(msg)

    first, *rest = arms
    names = frozenset(first.fields)

    for other in rest:
        apart = sorted(names.symmetric_difference(other.fields))

        if apart:
            msg = (
                f"the arms {first.name!r} and {other.name!r} "
                f"declare different fields: {apart}"
            )
            raise SpecificationMismatchError(msg)

    for name in sorted(names - set(REGIME_FIELDS)):
        values = {_canonical(arm.fields[name]) for arm in arms}

        if len(values) > 1:
            msg = f"the arms differ in {name!r} outside the yoked regime fields"
            raise SpecificationMismatchError(msg)

    regimes = [
        tuple(_canonical(arm.fields[field]) for field in REGIME_FIELDS) for arm in arms
    ]

    if len(regimes) != len(set(regimes)):
        msg = "the arms do not declare distinct regime tuples"
        raise SpecificationMismatchError(msg)


def estimator_arms(
    shared: Mapping[str, JsonValue],
    regimes: Mapping[str, Mapping[str, JsonValue]],
) -> tuple[ArmSpecification, ...]:
    """Build fragment-named arms from shared fields and one yoked axis."""
    fixed = sorted(set(shared).intersection(REGIME_FIELDS))

    if fixed:
        msg = f"the shared fields fix regime fields {fixed!r}"
        raise ValueError(msg)

    arms = tuple(
        ArmSpecification(name, {**shared, **dict(regime)})
        for name, regime in regimes.items()
    )
    _one_yoked_axis(arms)
    return arms


class ArmCost(FrozenValue):
    """Matched optimizer steps and measured proof service clocks."""

    _noun = "arm cost"
    __slots__ = ("_optimizer_steps", "_prover_cpu_seconds", "_wall_seconds")

    def __init__(
        self,
        *,
        optimizer_steps: int,
        prover_cpu_seconds: float,
        wall_seconds: float,
    ) -> None:
        cpu = ensure_finite("an arm's prover CPU seconds", prover_cpu_seconds)
        wall = ensure_finite("an arm's wall seconds", wall_seconds)

        if cpu < 0.0:
            msg = f"an arm's prover CPU seconds are negative: {prover_cpu_seconds!r}"
            raise ValueError(msg)
        if wall < 0.0:
            msg = f"an arm's wall seconds are negative: {wall_seconds!r}"
            raise ValueError(msg)

        object.__setattr__(
            self,
            "_optimizer_steps",
            ensure_positive("an arm's optimizer steps", optimizer_steps),
        )
        object.__setattr__(self, "_prover_cpu_seconds", cpu)
        object.__setattr__(self, "_wall_seconds", wall)

    @property
    def optimizer_steps(self) -> int:
        """Return completed optimizer steps."""
        return self._optimizer_steps

    @property
    def prover_cpu_seconds(self) -> float:
        """Return measured prover CPU seconds."""
        return self._prover_cpu_seconds

    @property
    def wall_seconds(self) -> float:
        """Return measured prover wall seconds."""
        return self._wall_seconds

    def to_spec(self) -> dict[str, JsonValue]:
        """Return the cost as report columns."""
        return {
            "optimizer_steps": self._optimizer_steps,
            "prover_cpu_seconds": self._prover_cpu_seconds,
            "wall_seconds": self._wall_seconds,
        }

    def _identity(self) -> tuple[int, float, float]:
        return self._optimizer_steps, self._prover_cpu_seconds, self._wall_seconds

    @override
    def __eq__(self, other: object) -> bool:
        if type(other) is not ArmCost:
            return NotImplemented
        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        return hash((ArmCost, self._identity()))


class GroundedRun(FrozenValue):
    """One arm at one optimizer-step budget with paired held-out outcomes."""

    _noun = "grounded run"
    __slots__ = (
        "_arm",
        "_cost",
        "_items",
        "_kind",
        "_outcomes",
        "_setting",
        "_started_at",
    )

    def __init__(
        self,
        arm: str,
        setting: BudgetSetting,
        *,
        kind: RunKind,
        started_at: datetime,
        cost: ArmCost,
        outcomes: Mapping[Outcome, Mapping[ItemId, float]],
    ) -> None:
        if not isinstance(setting, BudgetSetting):
            msg = f"a run's budget setting is not one: {setting!r}"
            raise TypeError(msg)
        if not isinstance(kind, RunKind):
            msg = f"a run's kind is not one: {kind!r}"
            raise TypeError(msg)
        if not isinstance(cost, ArmCost):
            msg = f"a run's cost is not one: {cost!r}"
            raise TypeError(msg)

        supplied = tuple(outcomes)
        expected = kind.outcomes

        if set(supplied) != set(expected):
            msg = (
                f"a {kind.value} run reports {[one.value for one in supplied]}, "
                f"not {[one.value for one in expected]}"
            )
            raise ValueError(msg)

        lowered: dict[Outcome, Mapping[ItemId, float]] = {}
        items: tuple[ItemId, ...] | None = None

        for outcome in expected:
            values = {
                key: (
                    _proportion(f"the {outcome.value} of {key!r}", value)
                    if outcome in (Outcome.SYMBOLIC_RATE, Outcome.LABEL_ACCURACY)
                    else ensure_finite(f"the {outcome.value} of {key!r}", value)
                )
                for key, value in outcomes[outcome].items()
            }
            ordered = tuple(sorted(values))

            if not ordered:
                msg = f"the arm {arm!r} reports no held-out unit"
                raise ValueError(msg)
            if items is not None and ordered != items:
                msg = f"the arm {arm!r}'s outcomes cover different held-out units"
                raise ValueError(msg)

            items = ordered
            lowered[outcome] = MappingProxyType(values)

        object.__setattr__(self, "_arm", ensure_name("a run's arm", arm))
        object.__setattr__(self, "_setting", setting)
        object.__setattr__(self, "_kind", kind)
        object.__setattr__(self, "_started_at", _moment("a run's start", started_at))
        object.__setattr__(self, "_cost", cost)
        object.__setattr__(self, "_outcomes", MappingProxyType(lowered))
        object.__setattr__(self, "_items", cast("tuple[ItemId, ...]", items))

    @property
    def arm(self) -> str:
        """Return the regime fragment name."""
        return self._arm

    @property
    def setting(self) -> BudgetSetting:
        """Return the named optimizer-step budget."""
        return self._setting

    @property
    def kind(self) -> RunKind:
        """Return whether this run scores synthetic labels or judgments."""
        return self._kind

    @property
    def started_at(self) -> datetime:
        """Return the run's start boundary."""
        return self._started_at

    @property
    def cost(self) -> ArmCost:
        """Return matched steps and measured clocks."""
        return self._cost

    @property
    def items(self) -> tuple[ItemId, ...]:
        """Return paired held-out unit identifiers."""
        return self._items

    @property
    def outcomes(self) -> tuple[Outcome, ...]:
        """Return reported outcomes in task order."""
        return self._kind.outcomes

    def scores(self, outcome: Outcome) -> tuple[float, ...]:
        """Return one outcome in paired unit order."""
        values = self._outcomes.get(outcome)

        if values is None:
            msg = f"the {self._kind.value} run reports no {outcome.value!r}"
            raise ValueError(msg)

        return tuple(values[item] for item in self._items)

    def value(self, outcome: Outcome) -> float:
        """Return the mean of one held-out outcome."""
        return fmean(self.scores(outcome))

    def to_spec(self) -> dict[str, JsonValue]:
        """Return the run as comparison-table columns."""
        return {
            "arm": self._arm,
            "setting": self._setting.value,
            "task": self._kind.value,
            "started_at": self._started_at.isoformat(),
            "cost": self._cost.to_spec(),
            "units": len(self._items),
            **{outcome.value: self.value(outcome) for outcome in self.outcomes},
        }


class EstimatorProtocol(FrozenValue):
    """The preregistered primary contrast and its statistical settings."""

    _noun = "estimator protocol"
    __slots__ = (
        "_alpha",
        "_falsifier",
        "_family",
        "_left",
        "_level",
        "_margin",
        "_outcome",
        "_registered_at",
        "_replicates",
        "_right",
        "_seed",
    )

    def __init__(
        self,
        left: str,
        right: str,
        *,
        outcome: Outcome,
        margin: float,
        falsifier: str,
        family: str,
        registered_at: datetime,
        alpha: float,
        level: float,
        replicates: int,
        seed: int,
    ) -> None:
        left_name = ensure_name("a protocol's left arm", left)
        right_name = ensure_name("a protocol's right arm", right)

        if left_name == right_name:
            msg = f"a protocol names one arm on both sides: {left_name!r}"
            raise ValueError(msg)
        if not isinstance(outcome, Outcome):
            msg = f"a protocol's outcome is not one: {outcome!r}"
            raise TypeError(msg)

        held_margin = ensure_finite("a protocol's margin", margin)
        held_alpha = ensure_finite("a protocol's alpha", alpha)
        held_level = ensure_finite("a protocol's coverage", level)

        if held_margin <= 0.0:
            msg = f"a protocol's margin is not positive: {margin!r}"
            raise ValueError(msg)
        if not 0.0 < held_alpha < 0.5:
            msg = f"a protocol's alpha does not lie below one half: {alpha!r}"
            raise ValueError(msg)
        if not 0.0 < held_level < 1.0:
            msg = f"a protocol's coverage is not a proportion: {level!r}"
            raise ValueError(msg)

        object.__setattr__(self, "_left", left_name)
        object.__setattr__(self, "_right", right_name)
        object.__setattr__(self, "_outcome", outcome)
        object.__setattr__(self, "_margin", held_margin)
        object.__setattr__(
            self, "_falsifier", ensure_name("a protocol's falsifier", falsifier)
        )
        object.__setattr__(self, "_family", ensure_name("a protocol's family", family))
        object.__setattr__(
            self, "_registered_at", _moment("a protocol's registration", registered_at)
        )
        object.__setattr__(self, "_alpha", held_alpha)
        object.__setattr__(self, "_level", held_level)
        object.__setattr__(
            self, "_replicates", ensure_positive("a protocol's replicates", replicates)
        )
        object.__setattr__(
            self, "_seed", ensure_non_negative("a protocol's seed", seed)
        )

    @property
    def left(self) -> str:
        return self._left

    @property
    def right(self) -> str:
        return self._right

    @property
    def outcome(self) -> Outcome:
        return self._outcome

    @property
    def margin(self) -> float:
        return self._margin

    @property
    def falsifier(self) -> str:
        return self._falsifier

    @property
    def family(self) -> str:
        return self._family

    @property
    def registered_at(self) -> datetime:
        return self._registered_at

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def level(self) -> float:
        return self._level

    @property
    def replicates(self) -> int:
        return self._replicates

    @property
    def seed(self) -> int:
        return self._seed

    def registered_before(self, moment: datetime, /) -> bool:
        """Return whether registration strictly preceded a run boundary."""
        return self._registered_at < _moment("a compared instant", moment)

    def to_spec(self) -> dict[str, JsonValue]:
        """Return preregistration columns."""
        return {
            "left": self._left,
            "right": self._right,
            "primary_outcome": self._outcome.value,
            "margin": self._margin,
            "falsifier": self._falsifier,
            "family": self._family,
            "registered_at": self._registered_at.isoformat(),
            "alpha": self._alpha,
            "level": self._level,
            "replicates": self._replicates,
            "seed": self._seed,
        }


class CurvePoint(NamedTuple):
    """One matched optimizer-step setting's two arm outcomes."""

    setting: BudgetSetting
    left: float
    right: float

    def to_spec(self) -> dict[str, JsonValue]:
        return {"setting": self.setting.value, "left": self.left, "right": self.right}


def estimator_verdict(
    separation: Comparison, equivalence: EquivalenceResult
) -> EstimatorVerdict:
    """Combine the paired test with the equivalence verdict."""
    if equivalence.verdict is Verdict.EQUIVALENT_WITHIN_MARGIN:
        return EstimatorVerdict.EQUIVALENT_WITHIN_MARGIN
    if (
        equivalence.verdict is not Verdict.SEPARATED
        or separation.p_value.value > equivalence.alpha
    ):
        return EstimatorVerdict.INDETERMINATE
    return (
        EstimatorVerdict.LEFT_SEPARATES
        if separation.statistic > 0.0
        else EstimatorVerdict.RIGHT_SEPARATES
    )


def paired_contrast(
    left: str,
    right: str,
    one: Sequence[float],
    other: Sequence[float],
    *,
    unit: str,
    groups: Sequence[str] | None,
    falsifier: str,
    family: str,
    replicates: int,
    seed: int,
    level: float,
) -> Comparison:
    """Return a paired permutation test and clustered rank-effect interval."""
    bootstrap = Bootstrap(
        unit=unit, replicates=replicates, seed=seed, level=level, groups=groups
    )
    effect = paired_effect_size(one, other, bootstrap, estimator="paired-cliffs-delta")
    interval = effect.interval
    difference = fsum(
        first - second for first, second in zip(one, other, strict=True)
    ) / len(one)
    return Comparison(
        left,
        right,
        statistic=difference,
        p_value=paired_permutation(
            one,
            other,
            replicates=replicates,
            seed=seed,
            tail="two-sided",
            groups=groups,
        ),
        effect=interval.observed,
        falsifier=falsifier,
        smoke_test=False,
        family=family,
        ci=Interval(interval.bca.low, interval.bca.high, interval.bca.level),
    )


class EstimatorRow(FrozenValue):
    """One paired arm contrast at one matched optimizer-step setting."""

    _noun = "estimator row"
    __slots__ = ("_curve", "_equivalence", "_left", "_primary", "_right", "_separation")

    def __init__(
        self,
        left: GroundedRun,
        right: GroundedRun,
        *,
        separation: Comparison,
        equivalence: EquivalenceResult,
        curve: Sequence[CurvePoint],
        primary: bool,
    ) -> None:
        if left.arm == right.arm:
            msg = f"a row names one arm on both sides: {left.arm!r}"
            raise ValueError(msg)
        if left.setting is not right.setting:
            msg = (
                f"a row compares the {left.setting.value} and "
                f"{right.setting.value} budgets"
            )
            raise ValueError(msg)

        object.__setattr__(self, "_left", left)
        object.__setattr__(self, "_right", right)
        object.__setattr__(self, "_separation", separation)
        object.__setattr__(self, "_equivalence", equivalence)
        object.__setattr__(self, "_curve", tuple(curve))
        object.__setattr__(self, "_primary", primary)

    @property
    def setting(self) -> BudgetSetting:
        return self._left.setting

    @property
    def left(self) -> GroundedRun:
        return self._left

    @property
    def right(self) -> GroundedRun:
        return self._right

    @property
    def separation(self) -> Comparison:
        return self._separation

    @property
    def equivalence(self) -> EquivalenceResult:
        return self._equivalence

    @property
    def curve(self) -> tuple[CurvePoint, ...]:
        return self._curve

    @property
    def primary(self) -> bool:
        return self._primary

    @property
    def verdict(self) -> EstimatorVerdict:
        return estimator_verdict(self._separation, self._equivalence)

    def to_spec(self) -> dict[str, JsonValue]:
        """Return a complete comparison-table row."""
        return {
            "setting": self.setting.value,
            "primary": self._primary,
            "left": self._left.to_spec(),
            "right": self._right.to_spec(),
            "curve": [point.to_spec() for point in self._curve],
            "statistic": self._separation.statistic,
            "p_value": self._separation.p_value.value,
            "effect": self._separation.effect,
            "family": self._separation.family,
            "falsifier": self._separation.falsifier,
            "margin": self._equivalence.margin,
            "observed_difference": self._equivalence.observed,
            "equivalence": self._equivalence.verdict.value,
            "verdict": self.verdict.value,
        }


def _separation(
    protocol: EstimatorProtocol, left: GroundedRun, right: GroundedRun
) -> Comparison:
    outcome = protocol.outcome
    return paired_contrast(
        left.arm,
        right.arm,
        left.scores(outcome),
        right.scores(outcome),
        unit=RESAMPLING_UNIT,
        groups=tuple(str(item) for item in left.items),
        falsifier=protocol.falsifier,
        family=protocol.family,
        replicates=protocol.replicates,
        seed=protocol.seed,
        level=protocol.level,
    )


def _equivalence(
    protocol: EstimatorProtocol, left: GroundedRun, right: GroundedRun
) -> EquivalenceResult:
    return Equivalence(
        margin=protocol.margin,
        replicates=protocol.replicates,
        seed=protocol.seed,
        level=protocol.alpha,
    ).compare(
        left.scores(protocol.outcome),
        right.scores(protocol.outcome),
        groups=tuple(str(item) for item in left.items),
    )


def _contrasts(
    protocol: EstimatorProtocol, arms: Sequence[str]
) -> tuple[tuple[str, str], ...]:
    primary = (protocol.left, protocol.right)
    others = tuple(
        pair for pair in combinations(arms, 2) if frozenset(pair) != frozenset(primary)
    )
    return primary, *others


class EstimatorComparison(FrozenValue):
    """Arbitrary fragment-named arms compared at matched optimizer steps."""

    _noun = "estimator comparison"
    __slots__ = ("_arms", "_protocol", "_rows", "_runs", "_settings")

    def __init__(
        self,
        protocol: EstimatorProtocol,
        arms: Sequence[ArmSpecification],
        runs: Mapping[tuple[str, BudgetSetting], GroundedRun],
    ) -> None:
        if not isinstance(protocol, EstimatorProtocol):
            msg = f"a comparison's protocol is not one: {protocol!r}"
            raise TypeError(msg)

        held_arms = tuple(arms)
        _one_yoked_axis(held_arms)
        arm_names = tuple(arm.name for arm in held_arms)

        if len(set(arm_names)) != len(arm_names):
            msg = f"a comparison repeats an arm name: {arm_names!r}"
            raise ValueError(msg)
        if protocol.left not in arm_names or protocol.right not in arm_names:
            msg = (
                f"the primary contrast {protocol.left!r}/{protocol.right!r} "
                f"is not among {arm_names!r}"
            )
            raise ValueError(msg)

        settings = tuple(
            setting
            for setting in BudgetSetting
            if any(key[1] is setting for key in runs)
        )

        if not settings:
            msg = "an estimator comparison was given no run"
            raise ValueError(msg)

        held_runs: dict[tuple[str, BudgetSetting], GroundedRun] = {}

        for arm in arm_names:
            for setting in settings:
                run = runs.get((arm, setting))

                if run is None:
                    msg = f"the arm {arm!r} has no run at the {setting.value} budget"
                    raise ValueError(msg)
                if run.arm != arm or run.setting is not setting:
                    msg = (
                        f"a run of {run.arm!r} at {run.setting.value} is "
                        f"misfiled under {arm!r} at {setting.value}"
                    )
                    raise ValueError(msg)
                held_runs[arm, setting] = run

        for key in runs:
            if key not in held_runs:
                msg = (
                    f"a run is filed outside the declared arm and budget grid: {key!r}"
                )
                raise ValueError(msg)

        self._validate(protocol, arm_names, settings, held_runs)
        rows = self._build_rows(protocol, arm_names, settings, held_runs)
        object.__setattr__(self, "_protocol", protocol)
        object.__setattr__(self, "_arms", held_arms)
        object.__setattr__(self, "_runs", MappingProxyType(held_runs))
        object.__setattr__(self, "_settings", settings)
        object.__setattr__(self, "_rows", rows)

    @staticmethod
    def _validate(
        protocol: EstimatorProtocol,
        arms: Sequence[str],
        settings: Sequence[BudgetSetting],
        runs: Mapping[tuple[str, BudgetSetting], GroundedRun],
    ) -> None:
        for (arm, setting), run in runs.items():
            if not protocol.registered_before(run.started_at):
                msg = (
                    f"the arm {arm!r}'s {setting.value} run started at "
                    f"{run.started_at.isoformat()}, which registration at "
                    f"{protocol.registered_at.isoformat()} does not precede"
                )
                raise PreregistrationError(msg)

        for setting in settings:
            at_setting = tuple(runs[arm, setting] for arm in arms)
            steps = {run.cost.optimizer_steps for run in at_setting}

            if len(steps) != 1:
                msg = (
                    f"the arms completed {sorted(steps)} optimizer steps at "
                    f"the {setting.value} budget"
                )
                raise UnmatchedBudgetError(msg)
            units = {run.items for run in at_setting}

            if len(units) != 1:
                msg = (
                    "the arms scored different held-out units at "
                    f"the {setting.value} budget"
                )
                raise ValueError(msg)
            kinds = {run.kind for run in at_setting}

            if len(kinds) != 1:
                msg = (
                    "the arms mix synthetic and judgment outcomes at "
                    f"the {setting.value} budget"
                )
                raise ValueError(msg)
            for run in at_setting:
                run.scores(protocol.outcome)

    @staticmethod
    def _build_rows(
        protocol: EstimatorProtocol,
        arms: Sequence[str],
        settings: Sequence[BudgetSetting],
        runs: Mapping[tuple[str, BudgetSetting], GroundedRun],
    ) -> tuple[EstimatorRow, ...]:
        rows = []

        for index, (left_arm, right_arm) in enumerate(_contrasts(protocol, arms)):
            curve = tuple(
                CurvePoint(
                    setting,
                    runs[left_arm, setting].value(protocol.outcome),
                    runs[right_arm, setting].value(protocol.outcome),
                )
                for setting in settings
            )
            for setting in settings:
                left = runs[left_arm, setting]
                right = runs[right_arm, setting]
                rows.append(
                    EstimatorRow(
                        left,
                        right,
                        separation=_separation(protocol, left, right),
                        equivalence=_equivalence(protocol, left, right),
                        curve=curve,
                        primary=index == 0,
                    )
                )

        return tuple(rows)

    @property
    def protocol(self) -> EstimatorProtocol:
        return self._protocol

    @property
    def arms(self) -> tuple[ArmSpecification, ...]:
        return self._arms

    @property
    def runs(self) -> Mapping[tuple[str, BudgetSetting], GroundedRun]:
        return self._runs

    @property
    def rows(self) -> tuple[EstimatorRow, ...]:
        return self._rows

    @property
    def primary(self) -> tuple[EstimatorRow, ...]:
        return tuple(row for row in self._rows if row.primary)

    def verdict(self, setting: BudgetSetting, /) -> EstimatorVerdict:
        """Return the primary verdict at one optimizer-step budget."""
        for row in self.primary:
            if row.setting is setting:
                return row.verdict
        msg = f"the comparison has no primary row at the {setting.value} budget"
        raise ValueError(msg)


class EstimatorReport(FrozenValue):
    """The comparison table produced by the run-directory reducer."""

    _noun = "estimator report"
    __slots__ = ("_arms", "_protocol", "_rows", "_runs")

    def __init__(
        self,
        protocol: EstimatorProtocol,
        arms: Sequence[ArmSpecification],
        runs: Sequence[GroundedRun],
        rows: Sequence[EstimatorRow],
    ) -> None:
        object.__setattr__(self, "_protocol", protocol)
        object.__setattr__(self, "_arms", tuple(arms))
        object.__setattr__(self, "_runs", tuple(runs))
        object.__setattr__(self, "_rows", tuple(rows))

    @property
    def protocol(self) -> EstimatorProtocol:
        return self._protocol

    @property
    def arms(self) -> tuple[ArmSpecification, ...]:
        return self._arms

    @property
    def runs(self) -> tuple[GroundedRun, ...]:
        return self._runs

    @property
    def rows(self) -> tuple[EstimatorRow, ...]:
        return self._rows

    @property
    def primary(self) -> tuple[EstimatorRow, ...]:
        return tuple(row for row in self._rows if row.primary)

    def to_spec(self) -> dict[str, JsonValue]:
        """Return the complete reducer document."""
        return {
            "protocol": self._protocol.to_spec(),
            "arms": [arm.to_spec() for arm in self._arms],
            "runs": [run.to_spec() for run in self._runs],
            "rows": [row.to_spec() for row in self._rows],
        }


def estimator_report(comparison: EstimatorComparison, /) -> EstimatorReport:
    """Return a comparison's unified report."""
    if not isinstance(comparison, EstimatorComparison):
        msg = f"an emitted comparison is not one: {comparison!r}"
        raise TypeError(msg)

    ordered = tuple(
        comparison.runs[arm.name, setting]
        for arm in comparison.arms
        for setting in BudgetSetting
        if (arm.name, setting) in comparison.runs
    )
    return EstimatorReport(
        comparison.protocol, comparison.arms, ordered, comparison.rows
    )


class RegimeRunDirectory(NamedTuple):
    """One completed run directory filed under its fragment and budget."""

    arm: str
    setting: BudgetSetting
    directory: Path


def _document[M](path: Path, decode: Callable[[str], M], /) -> M:
    """Decode one required run document through its boundary model."""
    if not path.is_file():
        msg = f"the run directory lacks {path.name!r}: {path.parent}"
        raise FileNotFoundError(msg)
    try:
        return decode(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValidationError, TypeError) as refusal:
        msg = f"the document {path} does not validate: {refusal}"
        raise TypeError(msg) from refusal


class _Step(Protocol):
    """One training step exposing its measured scalar metrics."""

    @property
    def metrics(self) -> Mapping[str, float]:
        """Return measured scalars by metric name."""
        ...


def _clock(steps: Sequence[_Step], key: str, /) -> float:
    total = 0.0

    for step in steps:
        total += ensure_finite(f"the step metric {key!r}", step.metrics.get(key, 0.0))

    return total


def _outcome(metric: str, /) -> Outcome | None:
    try:
        return Outcome(metric)
    except ValueError:
        return None


class _DirectoryObservation(NamedTuple):
    seed: int
    started_at: datetime
    cost: ArmCost
    kind: RunKind
    outcomes: Mapping[Outcome, float]


def _read_directory(directory: Path, /) -> _DirectoryObservation:
    point_path = directory / "point.json"
    _document(point_path, decode_object)
    manifest_path = directory / "manifest.json"
    manifest = _document(manifest_path, decode_manifest)

    if manifest.status != "completed":
        msg = f"the run directory {directory} is not completed"
        raise ValueError(msg)

    metrics = _document(directory / "metrics.json", decode_metrics)

    if metrics.stage != "closed":
        msg = f"the run directory {directory} has no closed metrics record"
        raise ValueError(msg)

    steps = (
        *(step for epoch in metrics.fitting.epochs for step in epoch.steps),
        *metrics.fitting.steps,
    )
    cost = ArmCost(
        optimizer_steps=metrics.fitting.optimizer_steps,
        prover_cpu_seconds=_clock(steps, "grounded/cpu_seconds"),
        wall_seconds=_clock(steps, "grounded/wall_seconds"),
    )
    rows_document = _document(directory / "rows.json", decode_rows)

    values: dict[Outcome, float] = {}
    kind: RunKind | None = None

    for raw in rows_document.rows:
        if (outcome := _outcome(raw.metric)) is None:
            continue
        if outcome in values:
            msg = f"the run directory {directory} repeats {outcome.value!r}"
            raise ValueError(msg)
        if raw.value is None:
            msg = f"the run directory {directory} leaves {outcome.value!r} undefined"
            raise ValueError(msg)
        values[outcome] = ensure_finite(f"the run outcome {outcome.value!r}", raw.value)
        found = (
            RunKind.JUDGMENT
            if outcome is Outcome.JUDGMENT_LIKELIHOOD
            else RunKind.SYNTHETIC
        )

        if kind is not None and found is not kind:
            msg = f"the run directory {directory} mixes synthetic and judgment outcomes"
            raise ValueError(msg)
        kind = found

    if kind is None:
        msg = f"the run directory {directory} reports no grounding-regime outcome"
        raise ValueError(msg)
    missing = tuple(outcome.value for outcome in kind.outcomes if outcome not in values)

    if missing:
        msg = f"the run directory {directory} lacks outcomes {missing!r}"
        raise ValueError(msg)

    started_at = datetime.fromtimestamp(point_path.stat().st_mtime, tz=UTC)
    return _DirectoryObservation(
        metrics.point.seed,
        started_at,
        cost,
        kind,
        MappingProxyType(values),
    )


def reduce_grounding_runs(
    arms: Sequence[ArmSpecification],
    directories: Sequence[RegimeRunDirectory],
    *,
    protocol: EstimatorProtocol,
) -> EstimatorReport:
    """Reduce completed run directories into matched regime comparisons."""
    held_arms = tuple(arms)
    arm_names = tuple(arm.name for arm in held_arms)
    grouped: dict[tuple[str, BudgetSetting], list[_DirectoryObservation]] = {}

    for filed in directories:
        if filed.arm not in arm_names:
            msg = f"the run directory is filed under undeclared arm {filed.arm!r}"
            raise ValueError(msg)
        grouped.setdefault((filed.arm, filed.setting), []).append(
            _read_directory(filed.directory)
        )

    runs: dict[tuple[str, BudgetSetting], GroundedRun] = {}

    for key, observations in grouped.items():
        arm, setting = key
        ordered = sorted(observations, key=lambda item: item.seed)
        seeds = tuple(item.seed for item in ordered)

        if len(seeds) != len(set(seeds)):
            msg = (
                f"the arm {arm!r} repeats a run seed at "
                f"the {setting.value} budget: {seeds!r}"
            )
            raise ValueError(msg)
        kinds = {item.kind for item in ordered}

        if len(kinds) != 1:
            msg = f"the arm {arm!r} mixes synthetic and judgment runs"
            raise ValueError(msg)

        kind = ordered[0].kind
        outcomes = {
            outcome: {
                ItemId(f"seed-{item.seed}"): item.outcomes[outcome] for item in ordered
            }
            for outcome in kind.outcomes
        }
        runs[key] = GroundedRun(
            arm,
            setting,
            kind=kind,
            started_at=min(item.started_at for item in ordered),
            cost=ArmCost(
                optimizer_steps=sum(item.cost.optimizer_steps for item in ordered),
                prover_cpu_seconds=sum(
                    item.cost.prover_cpu_seconds for item in ordered
                ),
                wall_seconds=sum(item.cost.wall_seconds for item in ordered),
            ),
            outcomes=outcomes,
        )

    return estimator_report(EstimatorComparison(protocol, held_arms, runs))

"""Canonical evaluation rows and reports.

A metric may declare that its value is interpretable only beside a named
control. The requirement travels with the row rather than being inferred from
the metric's spelling, and a report checks it within one complete evaluation
context.

A judgment row is keyed either by one held-out fold of one rung, or by a pooled
judgment naming every fold and rung a cross-fitted statistic pools. A pooled row
reports its first rung and the content digest of its pooled judgment as its
specification hash, and names no single fold.
"""

from hashlib import sha256
from json import dumps
from typing import TYPE_CHECKING, override

from lofi._checks import ensure_name, ensure_non_negative
from lofi._frozen import FrozenValue
from lofi.eval._identity import ReportIdentity
from lofi.eval._residual import (
    RESIDUAL_SOURCES,
    Approximation,
    ResidualSummary,
    residual_summary,
)
from lofi.eval._values import MetricResult
from lofi.spec import HeldOutParticipantPolicy

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from lofi._types import ItemId, JsonValue
    from lofi.eval._alignment import Split
    from lofi.stats import Interval

__all__ = [
    "MissingControlError",
    "PooledJudgment",
    "Report",
    "ReportRow",
    "report",
]


class MissingControlError(ValueError):
    """Signals a metric reported without the control capability it requires."""


def _uncontrolled(
    rows: Sequence[ReportRow],
) -> tuple[tuple[str, str, str], ...]:
    """Return comparison sets missing a control a metric capability requires.

    A comparison set is the rows sharing one freeze, run, grammar, convention,
    suite, task, dataset, split, seed, and metric. Thus a report covering two
    evaluation contexts needs the control in each of them.

    Parameters
    ----------
    rows : Sequence[ReportRow]
        The rows the report would emit.

    Returns
    -------
    tuple[tuple[str, str, str], ...]
        The run digest, metric, and missing method of each uncontrolled set.
    """
    methods: dict[tuple[object, ...], set[str]] = {}

    for entry in rows:
        methods.setdefault(_comparison_key(entry), set()).add(entry.method)

    return tuple(
        dict.fromkeys(
            (entry.config_hash, entry.metric, entry.required_control)
            for entry in rows
            if entry.required_control is not None
            and entry.required_control not in methods[_comparison_key(entry)]
        )
    )


def _comparison_key(row: ReportRow, /) -> tuple[object, ...]:
    """Return the complete evaluation context a control must share."""
    return (
        row.freeze_id,
        row.config_hash,
        row.grammar_fingerprint,
        row.convention_hash,
        row.suite,
        row.task,
        row.dataset,
        row.split,
        row.seed,
        row.metric,
    )


def _counted(summary: ResidualSummary) -> tuple[int, ...]:
    """Return comparable summary fields.

    Parameters
    ----------
    summary : ResidualSummary
        Residual summary.

    Returns
    -------
    tuple[int, ...]
        Measured, not-applicable and per-source counts.
    """
    return (
        summary.measured,
        summary.not_applicable,
        *(summary.by_source.get(source, 0) for source in RESIDUAL_SOURCES),
    )


class PooledJudgment(FrozenValue):
    """The rungs and held-out folds one pooled cross-fitted statistic is read from.

    The first rung is the one the statistic is reported for, and any later rung
    is one it is compared against. The folds form a set, held in ascending order.

    Attributes
    ----------
    rungs : tuple[str, ...]
        Pooled rung names, the reported rung first.
    specification_hashes : tuple[str, ...]
        Judgment specification digest of each rung, aligned with ``rungs``.
    folds : tuple[int, ...]
        Pooled held-out fold indices, ascending.
    rung : str
        The reported rung.
    specification_hash : str
        Content digest of the ordered rungs, their digests and the folds.

    Parameters
    ----------
    rungs : Sequence[str]
        Pooled rung names, the reported rung first.
    specification_hashes : Sequence[str]
        Judgment specification digest of each rung, aligned with ``rungs``.
    folds : Sequence[int]
        Pooled held-out fold indices.

    Raises
    ------
    TypeError
        If a name or a fold has an invalid type.
    ValueError
        If no rung is pooled, a rung repeats, the digests do not align with the
        rungs, a name is empty, a fold is negative or repeats, or fewer than two
        folds are pooled.
    """

    _noun = "pooled judgment"

    __slots__ = ("_digest", "_folds", "_rungs", "_specification_hashes")

    def __init__(
        self,
        *,
        rungs: Sequence[str],
        specification_hashes: Sequence[str],
        folds: Sequence[int],
    ) -> None:
        names = tuple(ensure_name("a pooled rung", rung) for rung in rungs)
        digests = tuple(
            ensure_name("a pooled rung's specification hash", digest)
            for digest in specification_hashes
        )
        indices = tuple(ensure_non_negative("a pooled fold", fold) for fold in folds)

        if not names:
            msg = "a pooled judgment pools no rung"
            raise ValueError(msg)

        if len(digests) != len(names):
            msg = (
                f"a pooled judgment gives {len(digests)} specification hashes "
                f"for {len(names)} rungs"
            )
            raise ValueError(msg)

        if len(frozenset(names)) != len(names):
            msg = f"a pooled judgment repeats a rung: {names!r}"
            raise ValueError(msg)

        if len(frozenset(indices)) != len(indices):
            msg = f"a pooled judgment repeats a fold: {indices!r}"
            raise ValueError(msg)

        if len(indices) < 2:
            msg = f"a pooled judgment pools fewer than two folds: {indices!r}"
            raise ValueError(msg)

        ordered = tuple(sorted(indices))
        preimage: JsonValue = {
            "folds": list(ordered),
            "rungs": [
                {"rung": name, "specification_hash": digest}
                for name, digest in zip(names, digests, strict=True)
            ],
        }
        encoded = dumps(preimage, ensure_ascii=False, separators=(",", ":"))
        object.__setattr__(self, "_rungs", names)
        object.__setattr__(self, "_specification_hashes", digests)
        object.__setattr__(self, "_folds", ordered)
        object.__setattr__(self, "_digest", sha256(encoded.encode("utf-8")).hexdigest())

    @property
    def rungs(self) -> tuple[str, ...]:
        """Return the pooled rungs, the reported rung first."""
        return self._rungs

    @property
    def specification_hashes(self) -> tuple[str, ...]:
        """Return each pooled rung's specification digest."""
        return self._specification_hashes

    @property
    def folds(self) -> tuple[int, ...]:
        """Return the pooled held-out folds in ascending order."""
        return self._folds

    @property
    def rung(self) -> str:
        """Return the rung the statistic is reported for."""
        return self._rungs[0]

    @property
    def specification_hash(self) -> str:
        """Return the content digest of the rungs, their digests and the folds."""
        return self._digest

    def to_spec(self) -> dict[str, JsonValue]:
        """Return the pooled rungs, their digests and the folds.

        Returns
        -------
        dict[str, JsonValue]
            Serialized pooled judgment, with its content digest.
        """
        return {
            "specification_hash": self._digest,
            "folds": list(self._folds),
            "rungs": [
                {"rung": name, "specification_hash": digest}
                for name, digest in zip(
                    self._rungs, self._specification_hashes, strict=True
                )
            ],
        }

    @override
    def __eq__(self, other: object) -> bool:
        """Compare the rungs, their digests and the folds."""
        if type(other) is not PooledJudgment:
            return NotImplemented

        return (self._rungs, self._specification_hashes, self._folds) == (
            other._rungs,
            other._specification_hashes,
            other._folds,
        )

    @override
    def __hash__(self) -> int:
        """Hash the rungs, their digests and the folds."""
        return hash(
            (PooledJudgment, self._rungs, self._specification_hashes, self._folds)
        )

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this pooled judgment."""
        return (
            f"PooledJudgment(rungs={self._rungs!r}, "
            f"specification_hashes={self._specification_hashes!r}, "
            f"folds={self._folds!r})"
        )


class ReportRow(FrozenValue):
    """A published value and its approximation.

    Attributes
    ----------
    identity : ReportIdentity
        Content identity for the freeze, run, grammar, and conventions.
    freeze_id : str
        The freeze the row is reported against.
    method : str
        Source-system name.
    metric : str
        Metric name.
    result : MetricResult
        Metric result.
    approximation : Approximation
        Computation budget and residual.
    is_measured : bool
        Whether every approximation source was measured.
    required_control : str | None
        Method name required beside this metric, if the metric declares one.
    pooled : PooledJudgment | None
        Folds and rungs a pooled cross-fitted statistic is read from, if any.

    Parameters
    ----------
    identity : ReportIdentity
        Content identity for the freeze, run, grammar, and conventions.
    method : str
        Source-system name.
    metric : str
        Metric name.
    result : MetricResult
        Metric result.
    approximation : Approximation
        Computation budget and residual.
    required_control : str | None
        Method name required beside this metric, if any.
    pooled : PooledJudgment | None
        Folds and rungs a pooled cross-fitted statistic is read from, in place
        of ``specification_hash``, ``fold`` and ``rung``.

    Raises
    ------
    TypeError
        If an input has an invalid type.
    ValueError
        If a name is empty, a partial experiment or judgment context is
        supplied, or a held-out fold context is given beside pooled folds.
    """

    _noun = "report row"

    __slots__ = (
        "_approximation",
        "_dataset",
        "_fold",
        "_held_out_convention",
        "_identity_value",
        "_method",
        "_metric",
        "_normal_form",
        "_pooled",
        "_required_control",
        "_result",
        "_rung",
        "_seed",
        "_specification_hash",
        "_split",
        "_suite",
        "_task",
    )

    def __init__(
        self,
        *,
        identity: ReportIdentity,
        method: str,
        metric: str,
        result: MetricResult,
        approximation: Approximation,
        suite: str | None = None,
        task: str | None = None,
        dataset: str | None = None,
        split: Split | None = None,
        seed: int | None = None,
        normal_form: str | None = None,
        specification_hash: str | None = None,
        fold: int | None = None,
        rung: str | None = None,
        held_out_convention: str | None = None,
        required_control: str | None = None,
        pooled: PooledJudgment | None = None,
    ) -> None:
        if not isinstance(identity, ReportIdentity):
            msg = f"a report row's identity is not one: {identity!r}"
            raise TypeError(msg)

        if not isinstance(result, MetricResult):
            msg = f"a report row's result is not one: {result!r}"
            raise TypeError(msg)

        if not isinstance(approximation, Approximation):
            msg = f"a report row's approximation is not one: {approximation!r}"
            raise TypeError(msg)

        context = (suite, task, dataset, split, seed)

        if any(value is None for value in context) and any(
            value is not None for value in context
        ):
            msg = "a report row has incomplete experiment context"
            raise ValueError(msg)

        judgment = (specification_hash, fold, rung)

        if pooled is not None:
            if not isinstance(pooled, PooledJudgment):
                msg = f"a report row's pooled judgment is not one: {pooled!r}"
                raise TypeError(msg)

            if any(value is not None for value in judgment):
                msg = "a report row names both a held-out fold and pooled folds"
                raise ValueError(msg)

            judgment = (pooled.specification_hash, None, pooled.rung)
            specification_hash, rung = pooled.specification_hash, pooled.rung

        elif any(value is None for value in judgment) and any(
            value is not None for value in judgment
        ):
            msg = "a report row has incomplete judgment context"
            raise ValueError(msg)

        if held_out_convention is not None and not any(
            value is not None for value in judgment
        ):
            msg = "a report row has a held-out convention without judgment context"
            raise ValueError(msg)

        if held_out_convention is not None and held_out_convention not in {
            policy.value for policy in HeldOutParticipantPolicy
        }:
            msg = (
                "a report row has an unknown held-out convention: "
                f"{held_out_convention!r}"
            )
            raise ValueError(msg)

        object.__setattr__(self, "_identity_value", identity)
        object.__setattr__(
            self, "_method", ensure_name("a report row's method", method)
        )
        object.__setattr__(
            self, "_metric", ensure_name("a report row's metric", metric)
        )
        object.__setattr__(self, "_result", result)
        object.__setattr__(self, "_approximation", approximation)
        object.__setattr__(
            self,
            "_suite",
            None if suite is None else ensure_name("a row's suite", suite),
        )
        object.__setattr__(
            self, "_task", None if task is None else ensure_name("a row's task", task)
        )
        object.__setattr__(
            self,
            "_dataset",
            None if dataset is None else ensure_name("a row's dataset", dataset),
        )
        object.__setattr__(
            self,
            "_split",
            None if split is None else ensure_name("a row's split", split),
        )
        object.__setattr__(
            self,
            "_seed",
            None if seed is None else ensure_non_negative("a row's seed", seed),
        )
        object.__setattr__(
            self,
            "_normal_form",
            None
            if normal_form is None
            else ensure_name("a row's normal form", normal_form),
        )
        object.__setattr__(
            self,
            "_specification_hash",
            None
            if specification_hash is None
            else ensure_name("a row's specification hash", specification_hash),
        )
        object.__setattr__(
            self,
            "_fold",
            None if fold is None else ensure_non_negative("a row's fold", fold),
        )
        object.__setattr__(
            self, "_rung", None if rung is None else ensure_name("a row's rung", rung)
        )
        object.__setattr__(self, "_held_out_convention", held_out_convention)
        object.__setattr__(self, "_pooled", pooled)
        object.__setattr__(
            self,
            "_required_control",
            None
            if required_control is None
            else ensure_name("a row's required control", required_control),
        )

    @property
    def identity(self) -> ReportIdentity:
        """Return the report identity."""
        return self._identity_value

    @property
    def freeze_id(self) -> str:
        """Return the freeze the row is reported against."""
        return self._identity_value.freeze_id

    @property
    def config_hash(self) -> str:
        """Return the run-configuration digest."""
        return self._identity_value.config_hash

    @property
    def grammar_fingerprint(self) -> str:
        """Return the grammar fingerprint."""
        return self._identity_value.grammar_fingerprint

    @property
    def convention_hash(self) -> str | None:
        """Return the parsing-convention digest, absent for judgment analyses."""
        return self._identity_value.convention_hash

    @property
    def method(self) -> str:
        """Return the source system."""
        return self._method

    @property
    def metric(self) -> str:
        """Return the metric name."""
        return self._metric

    @property
    def suite(self) -> str | None:
        """Return the evaluation suite, when this is an experiment row."""
        return self._suite

    @property
    def task(self) -> str | None:
        """Return the evaluation task, when this is an experiment row."""
        return self._task

    @property
    def dataset(self) -> str | None:
        """Return the dataset, when this is an experiment row."""
        return self._dataset

    @property
    def split(self) -> Split | None:
        """Return the dataset split, when this is an experiment row."""
        return self._split

    @property
    def seed(self) -> int | None:
        """Return the run seed, when this is an experiment row."""
        return self._seed

    @property
    def normal_form(self) -> str | None:
        """Return the normal form, when one applies."""
        return self._normal_form

    @property
    def specification_hash(self) -> str | None:
        """Return the judgment specification digest, when one applies."""
        return self._specification_hash

    @property
    def fold(self) -> int | None:
        """Return the held-out fold, when one applies."""
        return self._fold

    @property
    def rung(self) -> str | None:
        """Return the judgment rung, when one applies."""
        return self._rung

    @property
    def pooled(self) -> PooledJudgment | None:
        """Return the folds and rungs a pooled statistic is read from, if any."""
        return self._pooled

    @property
    def held_out_convention(self) -> str | None:
        """Return the held-out-participant convention, when one applies."""
        return self._held_out_convention

    @property
    def required_control(self) -> str | None:
        """Return the method required beside this metric, when one is required."""
        return self._required_control

    @property
    def result(self) -> MetricResult:
        """Return the metric result."""
        return self._result

    @property
    def value(self) -> float | None:
        """Return the metric value, when defined."""
        return self._result.value

    @property
    def n(self) -> int:
        """Return the number of resampling units."""
        return self._result.n

    @property
    def resampling_unit(self) -> str:
        """Return the unit counted by ``n``."""
        return self._result.resampling.unit

    @property
    def ci(self) -> Interval | None:
        """Return the confidence interval, when one was computed."""
        return self._result.ci

    @property
    def ci_lo(self) -> float | None:
        """Return the lower confidence bound, when one was computed."""
        return None if self._result.ci is None else self._result.ci.low

    @property
    def ci_hi(self) -> float | None:
        """Return the upper confidence bound, when one was computed."""
        return None if self._result.ci is None else self._result.ci.high

    @property
    def ci_level(self) -> float | None:
        """Return confidence coverage, when an interval was computed."""
        return None if self._result.ci is None else self._result.ci.level

    @property
    def approximation(self) -> Approximation:
        """Return the approximation budget and residual."""
        return self._approximation

    @property
    def is_measured(self) -> bool:
        """Return whether every source of approximation was measured for this row."""
        return self._approximation.is_measured

    def to_spec(self) -> dict[str, JsonValue]:
        """Return the row as report columns.

        Returns
        -------
        dict[str, JsonValue]
            Serialized report row.
        """
        interval = self._result.ci

        return {
            **self._identity_value.to_spec(),
            "suite": self._suite,
            "task": self._task,
            "dataset": self._dataset,
            "split": self._split,
            "method": self._method,
            "metric": self._metric,
            "value": self._result.value,
            "n": self._result.n,
            "resampling_unit": self._result.resampling.unit,
            "resampling_seed": self._result.resampling.seed,
            "seed": self._seed,
            "ci_low": None if interval is None else interval.low,
            "ci_high": None if interval is None else interval.high,
            "ci_level": None if interval is None else interval.level,
            "normal_form": self._normal_form,
            "specification_hash": self._specification_hash,
            "fold": self._fold,
            "rung": self._rung,
            "held_out_convention": self._held_out_convention,
            "required_control": self._required_control,
            **self._approximation.to_spec(),
        }

    def _fields(
        self,
    ) -> tuple[object, ...]:
        """Return every field a row is equal on."""
        return (
            self._identity_value,
            self._method,
            self._metric,
            self._result,
            self._approximation,
            self._suite,
            self._task,
            self._dataset,
            self._split,
            self._seed,
            self._normal_form,
            self._specification_hash,
            self._fold,
            self._rung,
            self._held_out_convention,
            self._required_control,
            self._pooled,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Compare every row field."""
        if type(other) is not ReportRow:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Hash every row field."""
        return hash((ReportRow, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this row."""
        fields = (
            ("suite", self._suite),
            ("task", self._task),
            ("dataset", self._dataset),
            ("split", self._split),
            ("seed", self._seed),
            ("normal_form", self._normal_form),
            *(
                (
                    ("specification_hash", self._specification_hash),
                    ("fold", self._fold),
                    ("rung", self._rung),
                )
                if self._pooled is None
                else ()
            ),
            ("held_out_convention", self._held_out_convention),
            ("required_control", self._required_control),
            ("pooled", self._pooled),
        )
        optional = "".join(
            f", {name}={value!r}" for name, value in fields if value is not None
        )
        return (
            f"ReportRow(identity={self._identity_value!r}, method={self._method!r}, "
            f"metric={self._metric!r}, result={self._result!r}, "
            f"approximation={self._approximation!r}{optional})"
        )


class Report(FrozenValue):
    """Report rows and their stated budget costs.

    Attributes
    ----------
    rows : tuple[ReportRow, ...]
        Published rows in input order.
    residuals : ResidualSummary
        Residual-measurement summary.

    Parameters
    ----------
    rows : Sequence[ReportRow]
        Published rows in report order.
    residuals : ResidualSummary
        Residual-measurement summary.

    Raises
    ------
    TypeError
        If a row or residual summary has an invalid type.
    ValueError
        If the supplied summary disagrees with the rows.
    MissingControlError
        If a row's metric requires a control no row of the same run and metric
        carries.
    """

    _noun = "report"

    __slots__ = ("_residuals", "_rows", "_unmatched", "_unpredicted")

    def __init__(
        self,
        rows: Sequence[ReportRow],
        residuals: ResidualSummary,
        *,
        unpredicted: Sequence[ItemId] = (),
        unmatched: Sequence[ItemId] = (),
    ) -> None:
        if not isinstance(residuals, ResidualSummary):
            msg = f"a report's residual summary is not one: {residuals!r}"
            raise TypeError(msg)

        object.__setattr__(self, "_rows", tuple(rows))

        for entry in self._rows:
            if not isinstance(entry, ReportRow):
                msg = f"a report row is not one: {entry!r}"
                raise TypeError(msg)

        uncontrolled = _uncontrolled(self._rows)

        if uncontrolled:
            listed = ", ".join(
                f"{metric!r} under run {run!r} requires {control!r}"
                for run, metric, control in uncontrolled
            )
            msg = (
                f"a metric is reported without its required control in {listed}; "
                "the uncontrolled number would not be interpretable"
            )
            raise MissingControlError(msg)

        object.__setattr__(self, "_residuals", residuals)
        object.__setattr__(self, "_unpredicted", tuple(unpredicted))
        object.__setattr__(self, "_unmatched", tuple(unmatched))

        if residuals.rows != len(self._rows):
            msg = (
                f"a report of {len(self._rows)} rows summarizes "
                f"{residuals.rows} of them"
            )
            raise ValueError(msg)

        derived = residual_summary(entry.approximation for entry in self._rows)

        if _counted(residuals) != _counted(derived):
            msg = (
                f"a report's summary counts {residuals.measured} rows measuring "
                f"every source and {residuals.not_applicable} rows no budget "
                f"applies to, and leaves {_counted(residuals)[2:]!r} unmeasured, "
                f"where its own rows give {derived.measured}, "
                f"{derived.not_applicable} and {_counted(derived)[2:]!r}"
            )
            raise ValueError(msg)

    @property
    def rows(self) -> tuple[ReportRow, ...]:
        """Return the rows, in the order they were handed in."""
        return self._rows

    @property
    def residuals(self) -> ResidualSummary:
        """Return how many rows state what their approximation cost."""
        return self._residuals

    @property
    def unpredicted(self) -> tuple[ItemId, ...]:
        """Return dataset items without predictions."""
        return self._unpredicted

    @property
    def unmatched(self) -> tuple[ItemId, ...]:
        """Return predictions without dataset items."""
        return self._unmatched

    def __len__(self) -> int:
        """Return the number of report rows."""
        return len(self._rows)

    def __iter__(self) -> Iterator[ReportRow]:
        """Iterate over report rows in order."""
        return iter(self._rows)

    def to_spec(self) -> dict[str, JsonValue]:
        """Return the report as rows and a residual summary.

        Returns
        -------
        dict[str, JsonValue]
            Serialized report.
        """
        by_source: dict[str, JsonValue] = {
            source: self._residuals.by_source.get(source, 0)
            for source in RESIDUAL_SOURCES
        }
        residuals: dict[str, JsonValue] = {
            "rows": self._residuals.rows,
            "measured": self._residuals.measured,
            "unmeasured": self._residuals.unmeasured,
            "not_applicable": self._residuals.not_applicable,
            "by_source": by_source,
        }

        return {
            "rows": [entry.to_spec() for entry in self._rows],
            "residuals": residuals,
        }

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value holds the same rows and the same summary."""
        if type(other) is not Report:
            return NotImplemented

        return (
            self._rows == other._rows
            and self._residuals == other._residuals
            and self._unpredicted == other._unpredicted
            and self._unmatched == other._unmatched
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the rows and the summary."""
        return hash(
            (Report, self._rows, self._residuals, self._unpredicted, self._unmatched)
        )

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this report."""
        optional = ""
        if self._unpredicted:
            optional += f", unpredicted={self._unpredicted!r}"
        if self._unmatched:
            optional += f", unmatched={self._unmatched!r}"
        return f"Report({list(self._rows)!r}, {self._residuals!r}{optional})"


def report(
    rows: Iterable[ReportRow],
    *,
    unpredicted: Sequence[ItemId] = (),
    unmatched: Sequence[ItemId] = (),
) -> Report:
    """Return a report with a computed residual summary.

    Parameters
    ----------
    rows : Iterable[ReportRow]
        Published rows in report order.

    Returns
    -------
    Report
        Report and derived residual summary.

    Raises
    ------
    TypeError
        If ``rows`` contains a value other than ``ReportRow``.
    MissingControlError
        If a row's metric requires a control no row of the same run and metric
        carries.
    """
    collected = tuple(rows)

    for entry in collected:
        if not isinstance(entry, ReportRow):
            msg = f"a report row is not one: {entry!r}"
            raise TypeError(msg)

    return Report(
        collected,
        residual_summary(entry.approximation for entry in collected),
        unpredicted=unpredicted,
        unmatched=unmatched,
    )

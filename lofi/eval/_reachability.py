"""Gold-derivation reachability rows per rule inventory, corpus and language.

A row reads one :class:`~lofi.data.ReachabilityReport`, the result of checking a
corpus's decoded gold derivations against one declared inventory, and reports
step and derivation reachability beside a blocking histogram. The histogram has
one cell per distinct canonical blocker and counts each unreachable derivation
once, under the single blocker the report selected for it, so its cells sum to
the unreachable derivation count however many steps of a tree fail.

A cell is one of three kinds. A label cell names a label the inventory does not
admit. A rule cell names a recorded rule the inventory cannot recover. A
construction cell names the exact local shape, premises and conclusion, of a
step that carries no recorded rule name and that no rule in the inventory
recovers.
"""

from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, Protocol, final, override

from lofi._checks import ensure_name, ensure_non_negative
from lofi._frozen import FrozenValue
from lofi.data import (
    LabelBlocker,
    LabelMembership,
    ReachabilityReport,
    RuleBlocker,
    StepRecovery,
    reachability_report,
)
from lofi.data.corpora import GrammarInventory
from lofi.eval._held_out import derivation_dataset
from lofi.eval._values import ConventionVector, MetricResult, Resampling
from lofi.registry import ComponentPlugin
from lofi.spec.evaluators.reachability import (
    ReachabilityEvaluatorSettings,
    ReachabilityEvaluatorSpec,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable, Mapping

    from lofi._types import ItemId, JsonValue
    from lofi.data import Derivation, DerivationOutcome, EvalDataset, Item, RuleName
    from lofi.eval import HeldOutEvaluationContext, HeldOutItems
    from lofi.eval._ports import HeldOutData
    from lofi.nn.loss._values import ObjectiveOutputs

__all__ = [
    "BLOCKER_KINDS",
    "PLUGIN",
    "BlockerCell",
    "BlockerKind",
    "InconsistentReachabilityError",
    "ReachabilityEvaluator",
    "ReachabilityMetric",
    "ReachabilityRow",
    "blocker_cell",
    "reachability_table",
]

type BlockerKind = Literal["label", "rule", "construction"]
"""The kind of canonical blocker a histogram cell counts."""

BLOCKER_KINDS: tuple[BlockerKind, ...] = ("label", "rule", "construction")
"""Every blocker kind, in the order a row reports their totals."""


class InconsistentReachabilityError(ValueError):
    """Signals reachability rows that cannot be reported in one table."""


class BlockerCell[T: Hashable](NamedTuple):
    """The identity of a canonical blocker, without the node it was found at.

    Parameters
    ----------
    kind : BlockerKind
        Which kind of blocker the cell counts.
    label : T | None
        The absent label of a label cell.
    rule : RuleName | None
        The unrecovered recorded rule name of a rule cell.
    inputs : tuple[T, ...]
        The premises of a construction cell, in child order.
    output : T | None
        The conclusion of a construction cell.
    """

    kind: BlockerKind
    label: T | None = None
    rule: RuleName | None = None
    inputs: tuple[T, ...] = ()
    output: T | None = None

    def to_spec(self, render: Callable[[T], str], /) -> dict[str, JsonValue]:
        """Return the cell as JSON, with labels written by a renderer.

        Parameters
        ----------
        render : Callable[[T], str]
            Write one label as text.
        """
        spec: dict[str, JsonValue] = {"kind": self.kind}

        if self.label is not None:
            spec["label"] = render(self.label)

        if self.rule is not None:
            spec["rule"] = self.rule

        if self.output is not None:
            spec["inputs"] = [render(label) for label in self.inputs]
            spec["output"] = render(self.output)

        return spec


def blocker_cell[T: Hashable](
    blocker: LabelBlocker[T] | RuleBlocker[T],
    /,
) -> BlockerCell[T]:
    """Return the histogram cell a canonical blocker is counted under.

    A rule blocker with a recorded rule name is counted under that name, and
    one without is counted under its exact local shape.

    Parameters
    ----------
    blocker : LabelBlocker[T] | RuleBlocker[T]
        The canonical blocker of one unreachable derivation.
    """
    if isinstance(blocker, LabelBlocker):
        return BlockerCell("label", label=blocker.label)

    if blocker.rule is None:
        return BlockerCell("construction", inputs=blocker.inputs, output=blocker.output)

    return BlockerCell("rule", rule=blocker.rule)


def _histogram[T: Hashable](
    outcomes: Iterable[DerivationOutcome[T]],
) -> tuple[tuple[BlockerCell[T], int], ...]:
    """Return blocker cells and their counts, most frequent first.

    Cells with equal counts keep the order in which their first derivation
    appears.

    Parameters
    ----------
    outcomes : Iterable[DerivationOutcome[T]]
        One outcome per checked derivation.
    """
    counts: dict[BlockerCell[T], int] = {}

    for outcome in outcomes:
        if isinstance(outcome, (LabelBlocker, RuleBlocker)):
            cell = blocker_cell(outcome)
            counts[cell] = counts.get(cell, 0) + 1

    return tuple(sorted(counts.items(), key=lambda entry: -entry[1]))


@final
class ReachabilityRow[T: Hashable](FrozenValue):
    """Reachability of one corpus's gold derivations under one inventory.

    Parameters
    ----------
    report : ReachabilityReport[T]
        The outcomes of checking the corpus's decoded derivations against the
        inventory.
    inventory : str
        Name of the declared rule inventory.
    corpus : str
        Name of the corpus the derivations were decoded from.
    language : str
        Language of the derivations.
    training_size : int
        Number of derivations in the corpus's training split for the language.

    Raises
    ------
    TypeError
        If the report is not a reachability report.
    ValueError
        If a name is empty or the training size is negative.
    """

    _noun = "reachability row"

    __slots__ = (
        "_corpus",
        "_histogram",
        "_inventory",
        "_language",
        "_report",
        "_training_size",
    )

    def __init__(
        self,
        report: ReachabilityReport[T],
        *,
        inventory: str,
        corpus: str,
        language: str,
        training_size: int,
    ) -> None:
        if not isinstance(report, ReachabilityReport):
            msg = f"a reachability row needs a reachability report: {report!r}"
            raise TypeError(msg)

        object.__setattr__(self, "_report", report)
        object.__setattr__(
            self, "_inventory", ensure_name("an inventory name", inventory)
        )
        object.__setattr__(self, "_corpus", ensure_name("a corpus name", corpus))
        object.__setattr__(self, "_language", ensure_name("a language", language))
        object.__setattr__(
            self,
            "_training_size",
            ensure_non_negative("a training-set size", training_size),
        )
        object.__setattr__(self, "_histogram", _histogram(report.outcomes))

    @property
    def report(self) -> ReachabilityReport[T]:
        """Return the reachability report the row reads."""
        return self._report

    @property
    def inventory(self) -> str:
        """Return the inventory name."""
        return self._inventory

    @property
    def corpus(self) -> str:
        """Return the corpus name."""
        return self._corpus

    @property
    def language(self) -> str:
        """Return the language of the derivations."""
        return self._language

    @property
    def training_size(self) -> int:
        """Return the training-split size for the corpus and language."""
        return self._training_size

    @property
    def key(self) -> tuple[str, str, str]:
        """Return the inventory, corpus and language the row is reported under."""
        return self._inventory, self._corpus, self._language

    @property
    def checked_steps(self) -> int:
        """Return the number of nonlexical steps checked."""
        return self._report.checked_steps

    @property
    def reachable_steps(self) -> int:
        """Return the number of steps the inventory recovers."""
        return self._report.reachable_steps

    @property
    def checked_derivations(self) -> int:
        """Return the number of decoded gold derivations checked."""
        return self._report.checked_derivations

    @property
    def reachable_derivations(self) -> int:
        """Return the number of wholly reachable derivations."""
        return self._report.reachable_derivations

    @property
    def unreachable_derivations(self) -> int:
        """Return the number of derivations with a canonical blocker."""
        return self._report.checked_derivations - self._report.reachable_derivations

    @property
    def step_reachability(self) -> float | None:
        """Return the reachable share of steps, or None if none were checked."""
        return self._report.step_reachability

    @property
    def derivation_reachability(self) -> float | None:
        """Return the reachable share of derivations, or None if none were checked."""
        return self._report.derivation_reachability

    @property
    def histogram(self) -> tuple[tuple[BlockerCell[T], int], ...]:
        """Return each blocker cell with its derivation count, most frequent first."""
        return self._histogram

    def blocked(self, kind: BlockerKind, /) -> int:
        """Return how many derivations a kind of blocker accounts for.

        Parameters
        ----------
        kind : BlockerKind
            The blocker kind to total.

        Raises
        ------
        ValueError
            If the kind is not a blocker kind.
        """
        if kind not in BLOCKER_KINDS:
            msg = f"an unknown blocker kind was requested: {kind!r}"
            raise ValueError(msg)

        return sum(count for cell, count in self._histogram if cell.kind == kind)

    def to_spec(self, render: Callable[[T], str], /) -> dict[str, JsonValue]:
        """Return the row as JSON, with labels written by a renderer.

        Parameters
        ----------
        render : Callable[[T], str]
            Write one label as text.
        """
        return {
            "inventory": self._inventory,
            "corpus": self._corpus,
            "language": self._language,
            "training_size": self._training_size,
            "checked_steps": self.checked_steps,
            "reachable_steps": self.reachable_steps,
            "checked_derivations": self.checked_derivations,
            "reachable_derivations": self.reachable_derivations,
            "step_reachability": self.step_reachability,
            "derivation_reachability": self.derivation_reachability,
            "blocked": {kind: self.blocked(kind) for kind in BLOCKER_KINDS},
            "histogram": [
                {"cell": cell.to_spec(render), "derivations": count}
                for cell, count in self._histogram
            ],
        }

    def _fields(self) -> tuple[ReachabilityReport[T], str, str, str, int]:
        """Return fields in comparison order."""
        return (
            self._report,
            self._inventory,
            self._corpus,
            self._language,
            self._training_size,
        )

    @override
    def __reduce__(
        self,
    ) -> tuple[
        Callable[[ReachabilityReport[T], str, str, str, int], ReachabilityRow[T]],
        tuple[ReachabilityReport[T], str, str, str, int],
    ]:
        """Return the reconstruction call."""
        return _restore_row, self._fields()

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another row reads the same report under the same key."""
        if type(other) is not ReachabilityRow:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the report and key."""
        return hash((ReachabilityRow, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return (
            f"ReachabilityRow({self._report!r}, inventory={self._inventory!r}, "
            f"corpus={self._corpus!r}, language={self._language!r}, "
            f"training_size={self._training_size!r})"
        )


def _restore_row[T: Hashable](
    report: ReachabilityReport[T],
    inventory: str,
    corpus: str,
    language: str,
    training_size: int,
) -> ReachabilityRow[T]:
    """Rebuild a reachability row from its fields."""
    return ReachabilityRow(
        report,
        inventory=inventory,
        corpus=corpus,
        language=language,
        training_size=training_size,
    )


def reachability_table[T: Hashable](
    rows: Iterable[ReachabilityRow[T]],
    /,
) -> tuple[ReachabilityRow[T], ...]:
    """Return rows that can be compared in one table, in input order.

    Rows for the same corpus and language must agree on the training-set
    size and on the numbers of derivations and steps they checked. Only
    these counts are compared; the identities of the checked derivations
    are not.

    Parameters
    ----------
    rows : Iterable[ReachabilityRow[T]]
        The rows to report.

    Raises
    ------
    InconsistentReachabilityError
        If two rows share an inventory, corpus and language, or if two rows
        for one corpus and language disagree on the training-set size, the
        checked derivations or the checked steps.
    """
    held = tuple(rows)
    keys: set[tuple[str, str, str]] = set()
    material: dict[tuple[str, str], tuple[int, int, int]] = {}

    for row in held:
        if row.key in keys:
            msg = f"two reachability rows are reported under {row.key!r}"
            raise InconsistentReachabilityError(msg)

        keys.add(row.key)
        checked = (row.training_size, row.checked_derivations, row.checked_steps)
        seen = material.setdefault((row.corpus, row.language), checked)

        if seen != checked:
            msg = (
                f"rows for corpus {row.corpus!r} in {row.language!r} disagree on "
                f"training size, checked derivations or checked steps: "
                f"{seen!r} against {checked!r}"
            )
            raise InconsistentReachabilityError(msg)

    return held


type ReachabilityMeasure = Literal[
    "steps", "derivations", "blocked-label", "blocked-rule", "blocked-construction"
]
"""What one reachability row reports off a corpus's reachability row."""

_MEASURES: Final[tuple[ReachabilityMeasure, ...]] = (
    "steps",
    "derivations",
    "blocked-label",
    "blocked-rule",
    "blocked-construction",
)


_BLOCKED: Final[dict[ReachabilityMeasure, BlockerKind]] = {
    "blocked-label": "label",
    "blocked-rule": "rule",
    "blocked-construction": "construction",
}


class _Inventory(LabelMembership[str], StepRecovery[str, object], Protocol):
    """The label membership and step recovery a corpus is checked against."""


def _cell_name(cell: BlockerCell[str], /) -> str:
    """Return a histogram cell as one component name."""
    match cell.kind:
        case "label":
            return f"label:{cell.label}"
        case "rule":
            return f"rule:{cell.rule}"
        case "construction":
            return f"construction:{' '.join(cell.inputs)} -> {cell.output}"


def _value(row: ReachabilityRow[str], measure: ReachabilityMeasure, /) -> MetricResult:
    """Return one measure of a reachability row with its supporting counts."""
    match measure:
        case "steps":
            return MetricResult(
                row.step_reachability,
                row.checked_derivations,
                Resampling("sentence"),
                components={
                    "checked-steps": float(row.checked_steps),
                    "reachable-steps": float(row.reachable_steps),
                },
            )
        case "derivations":
            return MetricResult(
                row.derivation_reachability,
                row.checked_derivations,
                Resampling("sentence"),
                components={
                    "checked-derivations": float(row.checked_derivations),
                    "reachable-derivations": float(row.reachable_derivations),
                    "training-size": float(row.training_size),
                },
            )
        case "blocked-label" | "blocked-rule" | "blocked-construction":
            kind = _BLOCKED[measure]
            return MetricResult(
                float(row.blocked(kind)),
                row.checked_derivations,
                Resampling("sentence"),
                components={
                    _cell_name(cell): float(count)
                    for cell, count in row.histogram
                    if cell.kind == kind
                },
            )


@final
class ReachabilityMetric:
    """One reachability row over the held-out gold derivations scored so far.

    Parameters
    ----------
    inventory : _Inventory
        The label and rule inventory the derivations are checked against.
    measure : ReachabilityMeasure
        What the row reports.
    names : tuple[str, str, str]
        The inventory, corpus and language the row is reported under.
    training_size : int
        The number of derivations in the corpus's training partition.
    conventions : ConventionVector
        The conventions the row is reported under.
    """

    __slots__ = (
        "_conventions",
        "_inventory",
        "_items",
        "_measure",
        "_names",
        "_training_size",
    )

    def __init__(
        self,
        inventory: _Inventory,
        /,
        *,
        measure: ReachabilityMeasure,
        names: tuple[str, str, str],
        training_size: int,
        conventions: ConventionVector,
    ) -> None:
        if measure not in _MEASURES:
            msg = f"a reachability measure is not one of {_MEASURES!r}"
            raise ValueError(msg)

        self._inventory = inventory
        self._measure: ReachabilityMeasure = measure
        self._names = names
        self._training_size = training_size
        self._conventions = conventions
        self._items: dict[ItemId, Derivation[str]] = {}

    @property
    def name(self) -> str:
        """Return the row name."""
        return f"reachability-{self._measure}"

    @property
    def conventions(self) -> ConventionVector:
        """Return the conventions the row is reported under."""
        return self._conventions

    @property
    def order_invariant(self) -> bool:
        """Return that derivation order affects no row value."""
        return True

    def reset(self) -> None:
        """Forget every scored derivation."""
        self._items = {}

    def update(self, pred: None, gold: Derivation[str], *, key: ItemId) -> None:
        """Record one held-out gold derivation."""
        del pred
        held = self._items.get(key)

        if held is not None and held != gold:
            msg = f"reachability already saw another derivation for {key!r}"
            raise ValueError(msg)

        self._items[key] = gold

    def _row(self, derivations: Iterable[Derivation[str]], /) -> ReachabilityRow[str]:
        """Return the reachability row of some derivations."""
        inventory, corpus, language = self._names
        return ReachabilityRow(
            reachability_report(self._inventory, self._inventory, derivations),
            inventory=inventory,
            corpus=corpus,
            language=language,
            training_size=self._training_size,
        )

    def compute(self) -> MetricResult:
        """Return the row over every scored derivation."""
        if not self._items:
            return MetricResult(None, 0, Resampling("sentence"))

        return _value(
            self._row(self._items[key] for key in sorted(self._items)), self._measure
        )

    def per_item(self) -> Mapping[ItemId, MetricResult]:
        """Return the row for each scored derivation alone."""
        return MappingProxyType(
            {
                key: _value(self._row((derivation,)), self._measure)
                for key, derivation in sorted(self._items.items())
            }
        )


class ReachabilityEvaluator:
    """Report gold-derivation reachability under the run grammar's inventory.

    The rows read only the held-out gold derivations, so the parser's
    predictions do not change them.
    """

    __slots__ = ("_conventions", "_metrics")

    def __init__(
        self,
        context: HeldOutEvaluationContext,
        /,
        *,
        inventory: str,
        corpus: str,
        language: str,
        conventions: ConventionVector,
    ) -> None:
        training = context.training_items

        if training is None:
            msg = "reachability rows need the size of the training partition"
            raise ValueError(msg)

        names = (
            ensure_name("an inventory name", inventory),
            ensure_name("a corpus name", corpus),
            ensure_name("a language", language),
        )
        self._conventions = conventions.extended(
            {
                "reachability_inventory": names[0],
                "reachability_corpus": names[1],
                "reachability_language": names[2],
            }
        )
        held = GrammarInventory(context.grammar)
        self._metrics = tuple(
            ReachabilityMetric(
                held,
                measure=measure,
                names=names,
                training_size=training,
                conventions=self._conventions,
            )
            for measure in _MEASURES
        )

    @property
    def conventions(self) -> ConventionVector:
        """Return the conventions every row carries."""
        return self._conventions

    @property
    def metrics(self) -> tuple[ReachabilityMetric, ...]:
        """Return the rows in report order."""
        return self._metrics

    def held_out(
        self, data: HeldOutData, split: str, /
    ) -> EvalDataset[Derivation[str]]:
        """Return a retained partition's derivations as this evaluator reads them."""
        return derivation_dataset(self.annotate, data.derivations(split))

    def annotate(self, item: Item, derivation: Derivation[str], /) -> Derivation[str]:
        """Return the held-out gold derivation itself."""
        del item
        return derivation

    def predict[BatchT: HeldOutItems](
        self, outputs: ObjectiveOutputs[BatchT], /
    ) -> Mapping[ItemId, None]:
        """Return an empty prediction for every published item."""
        return MappingProxyType(dict.fromkeys(outputs.batch.item_ids))

    @override
    def __repr__(self) -> str:
        """Return a summary naming the row count."""
        return f"ReachabilityEvaluator(rows={len(self._metrics)})"


def _build(
    settings: ReachabilityEvaluatorSettings,
    context: HeldOutEvaluationContext,
    /,
) -> ReachabilityEvaluator:
    """Build the declared reachability evaluator in its held-out context."""
    return ReachabilityEvaluator(
        context,
        inventory=settings.inventory,
        corpus=settings.corpus,
        language=settings.language,
        conventions=ConventionVector.from_settings(settings.conventions),
    )


PLUGIN: Final = ComponentPlugin(spec=ReachabilityEvaluatorSpec, build=_build)

"""The external baseline harness and the two arms its evidence selects.

A baseline family is a named comparison class together with the metric its
published numbers are quoted in. A system enters a family through one of two
arms. A system whose predictions were released enters the re-scoring arm and
its row cites the released file; a system whose predictions were not released
is rerun here and its row cites the configuration hash the rerun ran under.
The two arms answer different questions -- one asks what the published
predictions score under these conventions, the other what the published
description reproduces to -- so the table keeps them apart and this module
never puts a number from one beside a number from the other.
"""

from typing import TYPE_CHECKING, NamedTuple, override

from lofi._checks import ensure_name
from lofi.eval._churn import ARMS, ArmMixingError
from lofi.eval._values import ConventionVector

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from lofi._types import ItemId, JsonValue
    from lofi.eval._churn import Arm
    from lofi.eval._ports import Metric
    from lofi.eval._values import MetricResult

__all__ = [
    "BaselineFamily",
    "BaselineHarness",
    "BaselineItem",
    "BaselineRow",
    "BaselineSystem",
    "BaselineTable",
    "FamilyMismatchError",
    "ReleasedFile",
    "RerunConfiguration",
]


class FamilyMismatchError(ValueError):
    """Signals a system offered to a family it does not belong to."""


class ReleasedFile:
    """The released prediction file that puts a system in the re-scoring arm.

    Attributes
    ----------
    path : str
        The released file, as the row cites it.
    digest : str
        The digest of the released file's bytes.
    arm : Arm
        The re-scoring arm.
    cites : str
        The released file.

    Parameters
    ----------
    path : str
        The released file, as the row cites it.
    digest : str
        The digest of the released file's bytes.

    Raises
    ------
    TypeError
        If the path or the digest is not text.
    ValueError
        If the path or the digest is empty.
    """

    __slots__ = ("_digest", "_path")

    def __init__(self, path: str, *, digest: str) -> None:
        self._path = ensure_name("a released prediction file", path)
        self._digest = ensure_name("a released prediction file's digest", digest)

    @property
    def path(self) -> str:
        """Return the released file."""
        return self._path

    @property
    def digest(self) -> str:
        """Return the digest of the released file's bytes."""
        return self._digest

    @property
    def arm(self) -> Arm:
        """Return the arm released predictions put a system in."""
        return "re-scoring"

    @property
    def cites(self) -> str:
        """Return what a row carrying this evidence cites."""
        return self._path

    def to_spec(self) -> dict[str, JsonValue]:
        """Return the arm and the citation as columns.

        Returns
        -------
        dict[str, JsonValue]
            The arm, what the row cites, and its digest.
        """
        return {"arm": self.arm, "cites": self._path, "digest": self._digest}

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value cites the same file at the same digest."""
        if type(other) is not ReleasedFile:
            return NotImplemented

        return self._path == other._path and self._digest == other._digest

    @override
    def __hash__(self) -> int:
        """Return a hash over the path and the digest."""
        return hash((ReleasedFile, self._path, self._digest))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this file."""
        return f"ReleasedFile({self._path!r}, digest={self._digest!r})"


class RerunConfiguration:
    """The configuration that puts a rerun system in the re-running arm.

    Attributes
    ----------
    configuration : str
        The configuration the system was rerun under, as the row cites it.
    config_hash : str
        The hash of that configuration.
    arm : Arm
        The re-running arm.
    cites : str
        The configuration hash.

    Parameters
    ----------
    configuration : str
        The configuration the system was rerun under.
    config_hash : str
        The hash of that configuration.

    Raises
    ------
    TypeError
        If the configuration or its hash is not text.
    ValueError
        If the configuration or its hash is empty.
    """

    __slots__ = ("_config_hash", "_configuration")

    def __init__(self, configuration: str, *, config_hash: str) -> None:
        self._configuration = ensure_name("a re-run configuration", configuration)
        self._config_hash = ensure_name("a re-run configuration's hash", config_hash)

    @property
    def configuration(self) -> str:
        """Return the configuration the system was rerun under."""
        return self._configuration

    @property
    def config_hash(self) -> str:
        """Return the hash of that configuration."""
        return self._config_hash

    @property
    def arm(self) -> Arm:
        """Return the arm a rerun puts a system in."""
        return "re-running"

    @property
    def cites(self) -> str:
        """Return what a row carrying this evidence cites."""
        return self._config_hash

    def to_spec(self) -> dict[str, JsonValue]:
        """Return the arm and the citation as columns.

        Returns
        -------
        dict[str, JsonValue]
            The arm, what the row cites, and the configuration behind it.
        """
        return {
            "arm": self.arm,
            "cites": self._config_hash,
            "digest": self._configuration,
        }

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value names the same configuration and hash."""
        if type(other) is not RerunConfiguration:
            return NotImplemented

        return (
            self._configuration == other._configuration
            and self._config_hash == other._config_hash
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the configuration and its hash."""
        return hash((RerunConfiguration, self._configuration, self._config_hash))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this configuration."""
        return (
            f"RerunConfiguration({self._configuration!r}, "
            f"config_hash={self._config_hash!r})"
        )


class BaselineItem[Pred, Gold](NamedTuple):
    """One item a baseline system predicted and the gold value it is scored on.

    Parameters
    ----------
    item_id : ItemId
        The item identifier.
    pred : Pred
        What the system predicted for the item.
    gold : Gold
        The gold value the prediction is scored against.
    """

    item_id: ItemId
    pred: Pred
    gold: Gold


class BaselineFamily[Pred, Gold]:
    """A named baseline family and the metric its published numbers are quoted in.

    Attributes
    ----------
    name : str
        The family name carried on every row.
    metric : Metric[Pred, Gold]
        The metric each system in the family is scored by.
    conventions : ConventionVector
        The conventions the metric was constructed under.

    Parameters
    ----------
    name : str
        The family name carried on every row.
    metric : Metric[Pred, Gold]
        The metric each system in the family is scored by. It declares the
        family's conventions, so that the vector on a row and the vector the
        number was computed under cannot come apart.

    Raises
    ------
    TypeError
        If the name is not text, or the metric declares no convention vector.
    ValueError
        If the name is empty.
    """

    __slots__ = ("_conventions", "_metric", "_name")

    def __init__(self, name: str, metric: Metric[Pred, Gold]) -> None:
        conventions = metric.conventions

        if not isinstance(conventions, ConventionVector):
            msg = (
                f"the metric of the baseline family {name!r} declares "
                f"{conventions!r} for its conventions, so the family has no "
                f"convention vector to report under"
            )
            raise TypeError(msg)

        self._name = ensure_name("a baseline family", name)
        self._metric = metric
        self._conventions = conventions

    @property
    def name(self) -> str:
        """Return the family name."""
        return self._name

    @property
    def metric(self) -> Metric[Pred, Gold]:
        """Return the metric the family's systems are scored by."""
        return self._metric

    @property
    def conventions(self) -> ConventionVector:
        """Return the conventions the family reports under."""
        return self._conventions

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this family."""
        return f"BaselineFamily({self._name!r}, {self._metric!r})"


class BaselineSystem[Pred, Gold]:
    """One published system, the evidence for it, and the items it predicted.

    Attributes
    ----------
    system : str
        The system name.
    family : str
        The family the system belongs to.
    reason : str
        Why the system's evidence puts it in the arm it is in.
    evidence : ReleasedFile | RerunConfiguration
        The released file or the rerun configuration.
    arm : Arm
        The arm the evidence selects.
    cites : str
        What a row for this system cites.
    items : tuple[BaselineItem[Pred, Gold], ...]
        The predicted items, in the order they were handed in.

    Parameters
    ----------
    system : str
        The system name.
    family : str
        The family the system belongs to.
    reason : str
        Why the system's evidence puts it in the arm it is in.
    items : Sequence[BaselineItem[Pred, Gold]]
        The predicted items.
    released : ReleasedFile | None
        The released prediction file, when the system released one.
    rerun : RerunConfiguration | None
        The configuration the system was rerun under, when it released none.

    Raises
    ------
    TypeError
        If the evidence or an item has an invalid type.
    ValueError
        If a text field is empty, both or neither kind of evidence is given,
        no item is given, or an item identifier is repeated.
    """

    __slots__ = ("_evidence", "_family", "_items", "_reason", "_system")

    def __init__(
        self,
        system: str,
        *,
        family: str,
        reason: str,
        items: Sequence[BaselineItem[Pred, Gold]],
        released: ReleasedFile | None = None,
        rerun: RerunConfiguration | None = None,
    ) -> None:
        if released is not None and not isinstance(released, ReleasedFile):
            msg = f"a system's released predictions are not a file: {released!r}"
            raise TypeError(msg)

        if rerun is not None and not isinstance(rerun, RerunConfiguration):
            msg = f"a system's re-run configuration is not one: {rerun!r}"
            raise TypeError(msg)

        self._system = ensure_name("a baseline system", system)
        self._family = ensure_name("a baseline system's family", family)
        self._reason = ensure_name("a baseline system's reason", reason)

        if released is not None and rerun is not None:
            msg = (
                f"the system {system!r} supplies a released prediction file "
                f"and a re-run configuration, so its arm is undecided"
            )
            raise ValueError(msg)

        held: ReleasedFile | RerunConfiguration | None = (
            released if released is not None else rerun
        )

        if held is None:
            msg = (
                f"the system {system!r} supplies neither a released prediction "
                f"file nor a re-run configuration, so no arm holds it"
            )
            raise ValueError(msg)

        self._evidence = held
        self._items = tuple(items)
        seen: set[str] = set()

        for entry in self._items:
            if not isinstance(entry, BaselineItem):
                msg = f"a baseline item is not one: {entry!r}"
                raise TypeError(msg)

            identifier = ensure_name("a baseline item id", entry.item_id)

            if identifier in seen:
                msg = f"the system {system!r} predicts the item {identifier!r} twice"
                raise ValueError(msg)

            seen.add(identifier)

        if not self._items:
            msg = f"the system {system!r} predicts no item, so it scores nothing"
            raise ValueError(msg)

    @property
    def system(self) -> str:
        """Return the system name."""
        return self._system

    @property
    def family(self) -> str:
        """Return the family the system belongs to."""
        return self._family

    @property
    def reason(self) -> str:
        """Return why the evidence puts the system in the arm it is in."""
        return self._reason

    @property
    def evidence(self) -> ReleasedFile | RerunConfiguration:
        """Return the released file or the rerun configuration."""
        return self._evidence

    @property
    def arm(self) -> Arm:
        """Return the arm the evidence selects."""
        return self._evidence.arm

    @property
    def cites(self) -> str:
        """Return what a row for this system cites."""
        return self._evidence.cites

    @property
    def items(self) -> tuple[BaselineItem[Pred, Gold], ...]:
        """Return the predicted items, in the order they were handed in."""
        return self._items

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this system."""
        keyword = "released" if self.arm == "re-scoring" else "rerun"

        return (
            f"BaselineSystem({self._system!r}, family={self._family!r}, "
            f"reason={self._reason!r}, items={list(self._items)!r}, "
            f"{keyword}={self._evidence!r})"
        )


class BaselineRow(NamedTuple):
    """One family's score for one system, carrying the evidence behind it.

    The arm and the citation are read off the evidence rather than stored
    beside it, so a row cannot record one arm and cite the other.

    Parameters
    ----------
    family : str
        The baseline family.
    system : str
        The system scored.
    metric : str
        The metric the family reports.
    evidence : ReleasedFile | RerunConfiguration
        The released file or the rerun configuration the score came from.
    reason : str
        Why the evidence puts the system in the arm it is in.
    conventions : ConventionVector
        The conventions the family reports under.
    result : MetricResult
        The score the family's metric computed.
    """

    family: str
    system: str
    metric: str
    evidence: ReleasedFile | RerunConfiguration
    reason: str
    conventions: ConventionVector
    result: MetricResult

    @property
    def arm(self) -> Arm:
        """Return the arm the evidence selects."""
        return self.evidence.arm

    @property
    def cites(self) -> str:
        """Return the released file, or the hash the rerun ran under."""
        return self.evidence.cites

    def to_spec(self) -> dict[str, JsonValue]:
        """Return the row as report columns.

        Returns
        -------
        dict[str, JsonValue]
            The flattened baseline row.
        """
        return {
            "family": self.family,
            "system": self.system,
            "metric": self.metric,
            **self.evidence.to_spec(),
            "reason": self.reason,
            **dict(self.conventions.factors),
            "value": self.result.value,
            "n": self.result.n,
            "unit": self.result.resampling.unit,
            "seed": self.result.resampling.seed,
        }


class BaselineTable:
    """The baseline rows, kept in the arm each row's evidence selected.

    Attributes
    ----------
    rescoring : tuple[BaselineRow, ...]
        The rows scored from released predictions.
    rerunning : tuple[BaselineRow, ...]
        The rows scored from predictions this harness re-ran.

    Parameters
    ----------
    rescoring : Sequence[BaselineRow]
        The rows scored from released predictions.
    rerunning : Sequence[BaselineRow]
        The rows scored from predictions this harness re-ran.

    Raises
    ------
    TypeError
        If an entry is not a baseline row.
    ArmMixingError
        If a row is filed under an arm other than the one it records.
    """

    __slots__ = ("_rerunning", "_rescoring")

    def __init__(
        self, rescoring: Sequence[BaselineRow], rerunning: Sequence[BaselineRow]
    ) -> None:
        self._rescoring = tuple(rescoring)
        self._rerunning = tuple(rerunning)

        for arm in ARMS:
            for entry in self.arm(arm):
                if not isinstance(entry, BaselineRow):
                    msg = f"a baseline row is not one: {entry!r}"
                    raise TypeError(msg)

                if entry.arm != arm:
                    msg = (
                        f"the row for {entry.system!r} is filed under the "
                        f"{arm!r} arm and records {entry.arm!r} evidence"
                    )
                    raise ArmMixingError(msg)

    @property
    def rescoring(self) -> tuple[BaselineRow, ...]:
        """Return the rows scored from released predictions, in harness order."""
        return self._rescoring

    @property
    def rerunning(self) -> tuple[BaselineRow, ...]:
        """Return the rows scored from rerun predictions, in harness order."""
        return self._rerunning

    def arm(self, arm: Arm) -> tuple[BaselineRow, ...]:
        """Return one arm's rows.

        Parameters
        ----------
        arm : Arm
            The arm to select.

        Returns
        -------
        tuple[BaselineRow, ...]
            The rows filed under that arm, in harness order.

        Raises
        ------
        ValueError
            If the arm is not a declared one.
        """
        if arm == "re-scoring":
            return self._rescoring

        if arm == "re-running":
            return self._rerunning

        msg = f"a baseline table was asked for an undeclared arm: {arm!r}"
        raise ValueError(msg)

    def families(self) -> tuple[str, ...]:
        """Return the families the table reports, in harness order.

        Returns
        -------
        tuple[str, ...]
            Each family name once, the re-scoring arm's families first.
        """
        ordered: list[str] = []

        for arm in ARMS:
            for entry in self.arm(arm):
                if entry.family not in ordered:
                    ordered.append(entry.family)

        return tuple(ordered)

    def systems(self, arm: Arm) -> tuple[str, ...]:
        """Return the systems one arm reports, in harness order.

        Parameters
        ----------
        arm : Arm
            The arm to inspect.

        Returns
        -------
        tuple[str, ...]
            Each system name once.

        Raises
        ------
        ValueError
            If the arm is not a declared one.
        """
        ordered: list[str] = []

        for entry in self.arm(arm):
            if entry.system not in ordered:
                ordered.append(entry.system)

        return tuple(ordered)

    def to_spec(self) -> dict[str, JsonValue]:
        """Return one list of rows per arm.

        Returns
        -------
        dict[str, JsonValue]
            The serialized arms, which no key combines.
        """
        return {arm: [entry.to_spec() for entry in self.arm(arm)] for arm in ARMS}

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another table holds the same rows in the same arms."""
        if type(other) is not BaselineTable:
            return NotImplemented

        return (
            self._rescoring == other._rescoring and self._rerunning == other._rerunning
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the rows of both arms."""
        return hash((BaselineTable, self._rescoring, self._rerunning))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this table."""
        return f"BaselineTable({list(self._rescoring)!r}, {list(self._rerunning)!r})"


class BaselineHarness:
    """Runs baseline families and files each system's row under its own arm.

    One harness runs every family, so the families share nothing but the row
    shape they emit, and each carries its own metric and its own convention
    vector. A family runs once; a second run of the same name would give one
    family two sets of rows with no column to tell them apart.
    """

    __slots__ = ("_filed", "_ran")

    def __init__(self) -> None:
        self._filed: dict[Arm, list[BaselineRow]] = {arm: [] for arm in ARMS}
        self._ran: list[str] = []

    @property
    def ran(self) -> tuple[str, ...]:
        """Return the families run so far, in the order they were run."""
        return tuple(self._ran)

    def run[Pred, Gold](
        self,
        family: BaselineFamily[Pred, Gold],
        systems: Iterable[BaselineSystem[Pred, Gold]],
    ) -> tuple[BaselineRow, ...]:
        """Score one family's systems and file each row under its arm.

        Parameters
        ----------
        family : BaselineFamily[Pred, Gold]
            The family to run.
        systems : Iterable[BaselineSystem[Pred, Gold]]
            The systems in that family, in report order.

        Returns
        -------
        tuple[BaselineRow, ...]
            This family's rows, one per system, in report order.

        Raises
        ------
        TypeError
            If the family or a system has an invalid type.
        ValueError
            If the family has already run, it is run over no system, or a
            system is named twice.
        FamilyMismatchError
            If a system belongs to another family.
        """
        if not isinstance(family, BaselineFamily):
            msg = f"a baseline family is not one: {family!r}"
            raise TypeError(msg)

        held = tuple(systems)
        seen: set[str] = set()

        for entry in held:
            if not isinstance(entry, BaselineSystem):
                msg = f"a baseline system is not one: {entry!r}"
                raise TypeError(msg)

            if entry.family != family.name:
                msg = (
                    f"the system {entry.system!r} belongs to the "
                    f"{entry.family!r} family and was offered to "
                    f"{family.name!r}"
                )
                raise FamilyMismatchError(msg)

            if entry.system in seen:
                msg = (
                    f"the family {family.name!r} scores the system "
                    f"{entry.system!r} twice"
                )
                raise ValueError(msg)

            seen.add(entry.system)

        if not held:
            msg = f"the family {family.name!r} was run over no system"
            raise ValueError(msg)

        if family.name in self._ran:
            msg = f"the family {family.name!r} has already run"
            raise ValueError(msg)

        self._ran.append(family.name)
        produced = tuple(self._score(family, entry) for entry in held)

        for row in produced:
            self._filed[row.arm].append(row)

        return produced

    def table(self) -> BaselineTable:
        """Return the rows run so far, kept in their arms.

        Returns
        -------
        BaselineTable
            The two arms, separately.
        """
        return BaselineTable(self._filed["re-scoring"], self._filed["re-running"])

    def _score[Pred, Gold](
        self, family: BaselineFamily[Pred, Gold], system: BaselineSystem[Pred, Gold]
    ) -> BaselineRow:
        """Score one system under its family's metric.

        Parameters
        ----------
        family : BaselineFamily[Pred, Gold]
            The family being run.
        system : BaselineSystem[Pred, Gold]
            The system to score.
        """
        metric = family.metric
        metric.reset()

        for entry in system.items:
            metric.update(entry.pred, entry.gold, key=entry.item_id)

        return BaselineRow(
            family.name,
            system.system,
            metric.name,
            system.evidence,
            system.reason,
            family.conventions,
            metric.compute(),
        )

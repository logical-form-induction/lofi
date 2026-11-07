"""Shared judgment source rows and the bead collection boundary."""

from collections import Counter
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, final, override

from lofi._checks import ensure_name
from lofi._frozen import FrozenValue
from lofi.judgment._attributes import NO_FACTORS
from lofi.judgment._design import DesignError, JudgmentRow, RowColumns

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from lofi.judgment._attributes import ItemFactors

__all__ = [
    "AnonymousParticipantPolicy",
    "BeadAnnotationRecord",
    "BeadJudgmentSource",
    "BeadResponseEncoding",
    "JudgmentSkip",
]


class AnonymousParticipantPolicy(StrEnum):
    """Select how a judgment without an attributed participant is read."""

    #: Count the judgment as missing a participant and emit no row.
    SKIP = "skip"

    #: Give the judgment occurrence its own singleton pseudo-participant.
    SINGLETON = "singleton"


class JudgmentSkip(StrEnum):
    """Why a judgment source set a judgment aside, as its skips key it."""

    #: A judgment records no response.
    MISSING_RESPONSE = "missing-response"

    #: A judgment names no participant.
    MISSING_PARTICIPANT = "missing-participant"

    #: A bead collection row belongs to another declared question.
    OTHER_QUESTION = "other-question"


class BeadAnnotationRecord(Protocol):
    """The bead collection fields needed to construct one judgment row."""

    @property
    def annotator_id(self) -> str:
        """Return the participant identifier."""
        ...

    @property
    def item_id(self) -> str:
        """Return the preserved bead item identifier."""
        ...

    @property
    def question_name(self) -> str:
        """Return the response-space name this record answers."""
        ...

    @property
    def response_label(self) -> str:
        """Return the participant-facing response label."""
        ...


class BeadResponseEncoding(Protocol):
    """The bead response-space fields needed by the judgment boundary."""

    @property
    def name(self) -> str:
        """Return the question name."""
        ...

    @property
    def n_levels(self) -> int:
        """Return the number of response levels."""
        ...

    @property
    def labels(self) -> tuple[str, ...]:
        """Return response labels in model-index order."""
        ...


@final
class BeadJudgmentSource(FrozenValue):
    """Read one bead response space as identity-preserving judgment rows.

    A source selects the records answering one ``ResponseEncoding``. Its label
    order is the design's response order. Records for other questions are
    accounted for as skips, while selected records preserve bead's item and
    participant identifiers byte for byte. Each row's factor levels are read
    from the item it names by the declared factors.
    """

    _noun = "bead judgment source"
    __slots__ = ("_factors", "_labels", "_question", "_rows", "_skips")

    def __init__(
        self,
        records: Iterable[BeadAnnotationRecord],
        encoding: BeadResponseEncoding,
        /,
        *,
        factors: ItemFactors = NO_FACTORS,
    ) -> None:
        question = ensure_name("a bead response encoding", encoding.name)
        labels = tuple(
            ensure_name("a bead response label", label) for label in encoding.labels
        )
        if isinstance(encoding.n_levels, bool) or encoding.n_levels != len(labels):
            msg = (
                f"bead response encoding {question!r} declares "
                f"{encoding.n_levels!r} levels for {len(labels)} labels"
            )
            raise DesignError(msg)
        if len(labels) < 2:
            msg = f"bead response encoding {question!r} has fewer than two labels"
            raise DesignError(msg)
        if len(frozenset(labels)) != len(labels):
            msg = f"bead response encoding {question!r} repeats a label: {labels!r}"
            raise DesignError(msg)

        indices = {label: index for index, label in enumerate(labels)}
        rows: list[JudgmentRow] = []
        skips: Counter[str] = Counter()
        for record in records:
            record_question = ensure_name(
                "a bead judgment question", record.question_name
            )
            if record_question != question:
                skips[JudgmentSkip.OTHER_QUESTION] += 1
                continue
            participant = ensure_name(
                "a bead judgment participant", record.annotator_id
            )
            item_id = ensure_name("a bead judgment item", record.item_id)
            label = ensure_name("a bead judgment response", record.response_label)
            response = indices.get(label)
            if response is None:
                msg = (
                    f"bead item {item_id!r} records label {label!r} outside {labels!r}"
                )
                raise DesignError(msg)
            rows.append(
                JudgmentRow(
                    participant,
                    item_id,
                    response,
                    RowColumns(factors.levels(item_id), {}),
                )
            )

        object.__setattr__(self, "_question", question)
        object.__setattr__(self, "_labels", labels)
        object.__setattr__(self, "_factors", factors.names)
        object.__setattr__(self, "_rows", tuple(rows))
        object.__setattr__(self, "_skips", MappingProxyType(dict(skips)))

    @property
    def question(self) -> str:
        """Return the selected bead response-space name."""
        return self._question

    @property
    def labels(self) -> tuple[str, ...]:
        """Return response labels in design-index order."""
        return self._labels

    @property
    def response_levels(self) -> int:
        """Return the number of response levels every row is over."""
        return len(self._labels)

    @property
    def factors(self) -> tuple[str, ...]:
        """Return the item factors every row declares, in column order."""
        return self._factors

    @property
    def skips(self) -> Mapping[str, int]:
        """Return records belonging to other questions by disposition."""
        return self._skips

    def rows(self) -> tuple[JudgmentRow, ...]:
        """Return one row per record answering the selected question."""
        return self._rows

    @override
    def __repr__(self) -> str:
        """Return the selected question and disposition counts."""
        return (
            f"BeadJudgmentSource(question={self._question!r}, "
            f"rows={len(self._rows)}, levels={len(self._labels)}, "
            f"skipped={sum(self._skips.values())})"
        )

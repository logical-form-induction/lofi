"""The bead and simulated collectors of an active learning run.

A simulated collector answers from a rating design over the whole candidate
pool, drawn or read once when the run builds it, so every batch selected again
receives the same responses, each of the design's participants answering a
candidate once per response the design records. Three designs are installed:
``predictor-truth`` draws a known-effects design whose item scores are one
predictor column of the trained point, and ``drawn-effects`` draws one with its
own item scores, both over the pool in order, so candidate ``i`` is the drawn
design's ``i``-th item; ``registered-design`` reads a design registered in the
``lofi.judgment_design`` group and joins each candidate to the design item its
declared join expression names, or to the one sharing its identity. A drawn
design has the declared participants and the refitted response model's levels,
its population cutpoints dividing the declared latent range evenly, and is
drawn from the run's collector seed.

The bead collector fields every batch through bead, reading the responses
through the response encoding the active learning document declares. Round
``r`` lives in ``rounds/<r>/`` under the run's directory; an initial draw is
round 0. The first time a round's batch is
selected the collector writes it there as ``batch.jsonl``, one bead item per
line under the item's preserved identity, and the run stops awaiting its
responses. The responses are ingested by placing the bead annotation records
that answer the batch beside it as ``responses.jsonl``; the run, executed again
from the same inputs, selects the same batch, reads the responses through the
bead judgment source, and continues. A batch selected
again that differs from the one written is refused rather than overwritten.
"""

from statistics import fmean, pstdev
from types import MappingProxyType
from typing import TYPE_CHECKING, final

from bead.items import Item

from lofi.judgment import (
    ActiveJudgment,
    AttributeExpression,
    ReadBoundedDesign,
    ReadDesign,
    item_key,
)
from lofi.judgment.synthetic_designs import KnownEffectsDesign
from lofi.run._active_inputs import read_bead_judgments
from lofi.run._boundary.active import BATCH_NAME, RESPONSES_NAME, round_directory
from lofi.run._manifest import replace_file
from lofi.run.fits import RegisteredDesigns

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from lofi.judgment import (
        AcquisitionBatch,
        BeadResponseEncoding,
        CollectorContext,
        JudgmentDesign,
    )
    from lofi.spec.acquisition import (
        BeadCollectorSettings,
        DrawnEffectsCollectorSettings,
        PredictorTruthCollectorSettings,
        RegisteredDesignCollectorSettings,
    )

__all__ = [
    "BeadCollector",
    "SimulatedCollector",
    "bead_collector",
    "drawn_effects_collector",
    "predictor_truth_collector",
    "registered_design_collector",
]


def _written(path: Path, /) -> tuple[str, ...]:
    """Return the item identities of a batch already written."""
    return tuple(
        str(Item.model_validate_json(stripped).id)
        for line in path.read_text(encoding="utf-8").splitlines()
        if (stripped := line.strip())
    )


@final
class BeadCollector:
    """Collect each acquired batch through bead, stopping until it is answered.

    Parameters
    ----------
    directory : Path
        The run's directory.
    encoding : BeadResponseEncoding
        The question and labels every response answers.
    """

    __slots__ = ("_directory", "_encoding")

    def __init__(self, directory: Path, encoding: BeadResponseEncoding) -> None:
        self._directory = directory
        self._encoding = encoding

    @property
    def sources(self) -> tuple[Path, ...]:
        """Return no document, since the run reads the response encoding."""
        return ()

    def collect(
        self, round_index: int, batch: AcquisitionBatch, /
    ) -> tuple[ActiveJudgment, ...] | None:
        """Return the ingested responses to a batch, or None while they are awaited.

        Raises
        ------
        ValueError
            If the round's written batch holds other items, or its responses
            include records answering another question.
        """
        directory = round_directory(self._directory, round_index)
        written = directory / BATCH_NAME
        items = batch.items

        if not written.is_file():
            directory.mkdir(parents=True, exist_ok=True)
            replace_file(
                written,
                "".join(f"{item.model_dump_json()}\n" for item in items).encode(),
            )
            return None

        recorded = _written(written)
        selected = tuple(str(item.id) for item in items)
        if recorded != selected:
            msg = (
                f"round {round_index}'s written batch holds other items than the "
                f"batch selected again: {written}"
            )
            raise ValueError(msg)

        responses = directory / RESPONSES_NAME
        if not responses.is_file():
            return None

        read = read_bead_judgments(responses, self._encoding)
        if read.skipped:
            msg = (
                f"round {round_index}'s responses hold {read.skipped} records "
                f"answering a question other than {self._encoding.name!r}"
            )
            raise ValueError(msg)
        return read.judgments


def bead_collector(
    settings: BeadCollectorSettings, context: CollectorContext, /
) -> BeadCollector:
    """Build the bead collector of one run.

    Raises
    ------
    ValueError
        If the run supplies no response encoding.
    """
    del settings
    if context.encoding is None:
        msg = "the bead collector reads responses, and the run has no response encoding"
        raise ValueError(msg)
    return BeadCollector(context.directory, context.encoding)


@final
class SimulatedCollector:
    """Answer every batch from a rating design over the whole candidate pool.

    Parameters
    ----------
    answers : Mapping[str, Sequence[ActiveJudgment]]
        Every response to each candidate, by candidate identity.
    """

    __slots__ = ("_answers",)

    def __init__(self, answers: Mapping[str, Sequence[ActiveJudgment]], /) -> None:
        self._answers = MappingProxyType(
            {item: tuple(judgments) for item, judgments in answers.items()}
        )

    @property
    def sources(self) -> tuple[Path, ...]:
        """Return no document, since the design is drawn or registered."""
        return ()

    @classmethod
    def over(
        cls, design: JudgmentDesign, candidates: Sequence[Item], /
    ) -> SimulatedCollector:
        """Answer candidate ``i`` with every response to the design's ``i``-th item.

        Raises
        ------
        ValueError
            If the design rates fewer items than the pool holds.
        """
        order: dict[int, int] = {}
        for item in design.items.tolist():
            order.setdefault(int(item), len(order))
        if len(order) < len(candidates):
            msg = (
                f"the design rates {len(order)} items and the pool holds "
                f"{len(candidates)} candidates"
            )
            raise ValueError(msg)

        answers: dict[str, list[ActiveJudgment]] = {
            item_key(candidate): [] for candidate in candidates
        }
        for participant, item, response in zip(
            design.participants.tolist(),
            design.items.tolist(),
            design.responses.tolist(),
            strict=True,
        ):
            position = order[int(item)]
            if position < len(candidates):
                answers[item_key(candidates[position])].append(
                    ActiveJudgment(
                        design.participant_levels[int(participant)],
                        item_key(candidates[position]),
                        int(response),
                    )
                )
        return cls(answers)

    @classmethod
    def joined(
        cls, design: JudgmentDesign, candidates: Sequence[Item], join: str | None, /
    ) -> SimulatedCollector:
        """Answer each candidate with every response to the design item it joins.

        Raises
        ------
        ValueError
            If the join expression cannot be read from a candidate, or names a
            design item the design does not rate.
        """
        expression = (
            None
            if join is None
            else AttributeExpression(join, role="a registered design's join")
        )
        targets = {
            item_key(candidate): (
                item_key(candidate)
                if expression is None
                else expression.level(candidate)
            )
            for candidate in candidates
        }
        levels = {level: index for index, level in enumerate(design.item_levels)}
        unrated = sorted(target for target in targets.values() if target not in levels)
        if unrated:
            msg = f"the registered design rates no item {unrated[:3]!r}"
            raise ValueError(msg)
        answers: dict[str, list[ActiveJudgment]] = {key: [] for key in targets}
        by_level: dict[int, list[str]] = {}
        for key, target in targets.items():
            by_level.setdefault(levels[target], []).append(key)
        for participant, item, response in zip(
            design.participants.tolist(),
            design.items.tolist(),
            design.responses.tolist(),
            strict=True,
        ):
            for key in by_level.get(int(item), ()):
                answers[key].append(
                    ActiveJudgment(
                        design.participant_levels[int(participant)], key, int(response)
                    )
                )
        return cls(answers)

    def collect(
        self, round_index: int, batch: AcquisitionBatch, /
    ) -> tuple[ActiveJudgment, ...]:
        """Return every response the design records to each batch item."""
        del round_index
        return tuple(
            judgment for item in batch.item_ids for judgment in self._answers[item]
        )


def _scores(
    settings: PredictorTruthCollectorSettings, context: CollectorContext, /
) -> tuple[float, ...]:
    """Return each candidate's true effect from the declared predictor column."""
    values: list[float] = []
    for candidate in context.candidates:
        columns = context.readouts[item_key(candidate)]
        value = columns.get(settings.column)
        if value is None:
            offered = ", ".join(repr(column) for column in sorted(columns))
            msg = (
                f"the simulated truth reads predictor {settings.column!r}, and the "
                f"trained point fills {offered or 'nothing'}"
            )
            raise ValueError(msg)
        values.append(value)
    if settings.scale == "raw":
        return tuple(values)
    center = fmean(values)
    spread = pstdev(values, center)
    if spread == 0.0:
        msg = (
            f"predictor {settings.column!r} is constant over the pool and cannot "
            "be standardized"
        )
        raise ValueError(msg)
    return tuple((value - center) / spread for value in values)


def predictor_truth_collector(
    settings: PredictorTruthCollectorSettings, context: CollectorContext, /
) -> SimulatedCollector:
    """Build the collector whose true item effects are a predictor column."""
    scores = _scores(settings, context)
    design = KnownEffectsDesign(
        settings.kind,
        participants=settings.participants,
        items=len(scores),
        levels=context.levels,
        seed=context.seed,
        latent_range=settings.latent_range,
        scores=scores,
    )
    return SimulatedCollector.over(design(), context.candidates)


def drawn_effects_collector(
    settings: DrawnEffectsCollectorSettings, context: CollectorContext, /
) -> SimulatedCollector:
    """Build the collector over a known-effects design with drawn item scores."""
    design = KnownEffectsDesign(
        settings.kind,
        participants=settings.participants,
        items=len(context.candidates),
        levels=context.levels,
        seed=context.seed,
        latent_range=settings.latent_range,
    )
    return SimulatedCollector.over(design(), context.candidates)


def registered_design_collector(
    settings: RegisteredDesignCollectorSettings, context: CollectorContext, /
) -> SimulatedCollector:
    """Build the collector answering from a registered rating design.

    Raises
    ------
    UnknownTagError
        If no design is registered under the tag.
    ValueError
        If the design is not ordinal, or its levels are not the fit's.
    """
    match RegisteredDesigns().read(settings.design):
        case ReadDesign(design=design):
            pass
        case ReadBoundedDesign():
            msg = (
                f"the registered design {settings.design!r} records bounded "
                "responses, and the refitted response model is ordinal"
            )
            raise ValueError(msg)
    if design.response_levels != context.levels:
        msg = (
            f"the registered design {settings.design!r} has "
            f"{design.response_levels} levels and the fit's response has "
            f"{context.levels}"
        )
        raise ValueError(msg)
    return SimulatedCollector.joined(design, context.candidates, settings.join)

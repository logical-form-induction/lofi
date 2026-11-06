"""The inputs an active learning run reads from its declared documents.

The candidate pool is the bead items a judgment item construction document
declares, built through the installed lexicon generators. Judgments, the
initial ones and every batch a bead collector ingests, are bead annotation
records read through the bead judgment source, so their participant and item
identifiers are preserved and records answering another question are counted
rather than read. The predictors of every pool item are root-span readouts of a
trained point: the point is restored from its run directory, and its forward
pass is published over the words the declared bead tokenizer splits the
element mapped to its one input role into, for every pool item; each declared
column takes the log mass or log complement of one published mass readout over
the whole element.
"""

from typing import TYPE_CHECKING, Final, NamedTuple, cast, final

import torch
from bead.evaluation.reliability import AnnotationRecord
from bead.protocol import ResponseEncoding
from bead.tokenization.config import TokenizerConfig
from bead.tokenization.tokenizers import create_tokenizer

from lofi._frozen import FrozenValue
from lofi._types import ReadoutId
from lofi.judgment import (
    ActiveJudgment,
    BeadJudgmentSource,
    PointInputs,
    construct_items,
    decode_item_construction,
    item_key,
    lexicon_generator_for,
)
from lofi.nn.loss._readout import MASS_AXES, log_complement, log_mass
from lofi.run.hub import restore_trained_point

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from bead.items.item import Item

    from lofi.judgment import (
        BeadResponseEncoding,
        LexiconGenerator,
    )
    from lofi.nn.loss._values import ObjectiveOutputs
    from lofi.spec.acquisition import PointReadoutSettings, TokenizerSettings
    from lofi.spec.lexicon_generators import LexiconGeneratorSpec
    from lofi.train import (
        PreparedTraining,
    )

__all__ = [
    "PARSER_INPUT_ROLES",
    "BeadJudgments",
    "Pool",
    "pool_readouts",
    "read_bead_judgments",
    "read_encoding",
    "read_pool",
]


PARSER_INPUT_ROLES: Final = ("words",)
"""The input roles a parser point publishes: the one word sequence it parses."""


@final
class BeadJudgments(FrozenValue):
    """The judgments one bead annotation document answers a question with."""

    _noun = "bead judgments"
    __slots__ = ("_judgments", "_skipped")

    def __init__(self, judgments: Sequence[ActiveJudgment], skipped: int) -> None:
        object.__setattr__(self, "_judgments", tuple(judgments))
        object.__setattr__(self, "_skipped", skipped)

    @property
    def judgments(self) -> tuple[ActiveJudgment, ...]:
        """Return the judgments in record order."""
        return self._judgments

    @property
    def skipped(self) -> int:
        """Return the records that answer another question."""
        return self._skipped


def read_encoding(path: Path, /) -> ResponseEncoding:
    """Validate a bead response-encoding document."""
    return ResponseEncoding.model_validate_json(path.read_text(encoding="utf-8"))


def read_bead_judgments(path: Path, encoding: BeadResponseEncoding, /) -> BeadJudgments:
    """Read bead annotation records as judgments of the encoded question.

    Raises
    ------
    DesignError
        If a record answering the question records a label the encoding does
        not declare.
    """
    records = tuple(
        AnnotationRecord.model_validate_json(stripped)
        for line in path.read_text(encoding="utf-8").splitlines()
        if (stripped := line.strip())
    )
    source = BeadJudgmentSource(records, encoding)
    return BeadJudgments(
        tuple(
            ActiveJudgment(row.participant, row.item, row.response)
            for row in source.rows()
        ),
        sum(source.skips.values()),
    )


class Pool(NamedTuple):
    """The candidate pool one construction document declares and what it read."""

    items: tuple[Item, ...]
    sources: tuple[Path, ...]


def read_pool(
    path: Path,
    /,
    *,
    generator_for: Callable[
        [LexiconGeneratorSpec], LexiconGenerator
    ] = lexicon_generator_for,
) -> Pool:
    """Construct the pool a judgment item construction document declares.

    The document's files are relative to the directory holding it, and the
    pool records each of them beside the document itself.
    """
    declaration = decode_item_construction(path.read_bytes())
    return Pool(
        construct_items(declaration, path.parent, generator_for=generator_for),
        (path, *(path.parent / declared for declared in declaration.files())),
    )


@final
class _Elements:
    """The mapped element of every pool item as a batch a forward pass reads."""

    __slots__ = ("_item_ids", "_words")

    def __init__(
        self,
        items: Sequence[Item],
        inputs: PointInputs,
        tokenizer: TokenizerSettings,
        /,
    ) -> None:
        split = create_tokenizer(
            TokenizerConfig(
                backend=tokenizer.backend,
                language=tokenizer.language,
                model_name=tokenizer.model_name,
            )
        )
        (role,) = PARSER_INPUT_ROLES
        self._words = tuple(
            split(inputs.texts(item)[role]).token_texts for item in items
        )
        self._item_ids = tuple(item_key(item) for item in items)

    @property
    def words(self) -> tuple[tuple[str, ...], ...]:
        """Return the words of each item's element."""
        return self._words

    @property
    def item_ids(self) -> tuple[str, ...]:
        """Return the identity of each element's item."""
        return self._item_ids


def _published(
    prepared: PreparedTraining,
    batch: _Elements,
    readouts: Mapping[str, PointReadoutSettings],
    /,
) -> dict[str, tuple[float, ...]]:
    """Return each declared column's root-span value for every item."""
    plan = prepared.plan.parser_plan()
    modules = tuple(prepared.model.modules.values())
    training = tuple(module.training for module in modules)

    try:
        for module in modules:
            module.eval()
        publish = cast(
            "Callable[[_Elements], ObjectiveOutputs[_Elements]]",
            plan.forward.publish,
        )
        with torch.no_grad():
            outputs = publish(batch)
    finally:
        for module, mode in zip(modules, training, strict=True):
            module.train(mode)

    columns: dict[str, tuple[float, ...]] = {}
    spans = [(row, 0, len(words)) for row, words in enumerate(batch.words)]
    for column, declared in readouts.items():
        readout = outputs.readouts.get(ReadoutId(declared.readout))
        if readout is None:
            offered = ", ".join(sorted(outputs.readouts)) or "nothing"
            msg = (
                f"predictor {column!r} reads {declared.readout!r}, and the trained "
                f"point publishes {offered}"
            )
            raise ValueError(msg)
        if readout.axes != MASS_AXES:
            msg = (
                f"predictor {column!r} reads a mass readout over {MASS_AXES}, and "
                f"{declared.readout!r} is over {readout.axes}"
            )
            raise ValueError(msg)
        cells = readout.gather(spans)
        values = (
            log_mass(cells)
            if declared.measure == "log-mass"
            else (log_complement(cells))
        )
        columns[column] = tuple(float(value) for value in values.detach().cpu())
    return columns


def pool_readouts(
    run: Path,
    items: Sequence[Item],
    inputs: Mapping[str, str],
    tokenizer: TokenizerSettings,
    readouts: Mapping[str, PointReadoutSettings],
    /,
    *,
    restore: Callable[[Path], PreparedTraining] = restore_trained_point,
) -> dict[str, Mapping[str, float]]:
    """Return the trained point's predictor columns for every pool item.

    Raises
    ------
    ValueError
        If the declared inputs map other roles than the parser point
        publishes, an item renders a mapped element nowhere, the run directory
        records no completed point, the point publishes no readout under a
        declared name, or a declared readout is not a mass readout.
    """
    batch = _Elements(items, PointInputs(inputs, roles=PARSER_INPUT_ROLES), tokenizer)
    values = _published(restore(run), batch, readouts)
    return {
        item: {column: values[column][position] for column in readouts}
        for position, item in enumerate(batch.item_ids)
    }

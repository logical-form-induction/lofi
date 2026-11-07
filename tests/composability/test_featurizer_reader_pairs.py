"""Compatibility checks for concrete corpus readers and featurizers."""

from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple, cast

import pytest
import torch

from lofi.data import (
    CorpusReader,
    Example,
    Featurizer,
    GoldDependencies,
    GoldDerivation,
    GoldGraph,
    Inventory,
    Item,
    ItemHasher,
    ItemId,
    ItemLookup,
    SpanConstraints,
    Supervision,
    SupervisionArm,
)
from lofi.data.corpora.layers import LayersCorpusReader, corpus_at
from lofi.data.featurizers import (
    DeferredSupervisionError,
    FeaturizedBatch,
    featurizer_plugin,
    require_featurizable,
)
from lofi.registry import Registry
from lofi.spec import FeaturizerSpec, load_plugins

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.spec import ConfiguredSpec

TESTS: Final = Path(__file__).resolve().parents[1]
LAYERS_SAMPLE: Final = TESTS / "data/layers_sample/repository"
PAD: Final = "<pad>"

type Reader = CorpusReader[str, None]
type ConcreteReader = LayersCorpusReader
type DataFeaturizer = Featurizer[Supervision[str], FeaturizedBatch[Supervision[str]]]


class TorchCodeTensorizer:
    """A tensorizer for padded integer rows and boolean presence masks."""

    def __call__(
        self,
        rows: Sequence[tuple[int, ...]],
        /,
        *,
        padding: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        width = max((len(row) for row in rows), default=0)
        vectors = torch.full((len(rows), width), padding, dtype=torch.long)
        present = torch.zeros((len(rows), width), dtype=torch.bool)

        for position, row in enumerate(rows):
            if row:
                vectors[position, : len(row)] = torch.tensor(row, dtype=torch.long)
                present[position, : len(row)] = True

        return vectors, present


class ReaderCase(NamedTuple):
    """A named concrete reader and its examples."""

    name: str
    reader: Reader
    items: tuple[Item, ...]
    examples: tuple[Example[Supervision[str]], ...]


class FeaturizerCase(NamedTuple):
    """A named concrete featurizer and the arms its plugin declares."""

    name: str
    featurizer: DataFeaturizer
    accepted: frozenset[SupervisionArm]


class Resources(NamedTuple):
    """Dataset resources supplied to discovered featurizers."""

    hasher: ItemHasher
    tensorizer: TorchCodeTensorizer
    lookup: ItemLookup


def _readers() -> tuple[ReaderCase, ...]:
    layers = LayersCorpusReader(corpus_at(LAYERS_SAMPLE))

    return tuple(_reader_case(reader) for reader in (layers,))


def _reader_case(reader: ConcreteReader) -> ReaderCase:
    typed = cast("Reader", reader)
    items = tuple(item for split in typed.splits() for item in typed.items(split))
    examples = tuple(
        example for split in typed.splits() for example in typed.examples(split)
    )
    return ReaderCase(typed.name, typed, items, examples)


def _unique_items(items: Sequence[Item]) -> tuple[Item, ...]:
    unique: dict[ItemId, Item] = {}

    for item in items:
        existing = unique.get(item.item_id)

        if existing is not None:
            assert existing == item
        else:
            unique[item.item_id] = item

    return tuple(unique.values())


def _hasher(case: ReaderCase) -> ItemHasher:
    words = tuple(
        dict.fromkeys(token.surface for item in case.items for token in item.tokens)
    )
    categories: dict[str, None] = {}

    for example in case.examples:
        supervision = example.supervision

        if isinstance(supervision, SpanConstraints):
            for constraint in supervision.constraints:
                for label in constraint.permitted:
                    categories[label] = None
        elif isinstance(supervision, GoldDerivation):
            for node in supervision.derivation.nodes.values():
                categories[node.label] = None

    return ItemHasher(
        Inventory(f"{case.name}-words", words, (PAD,)),
        Inventory(f"{case.name}-categories", tuple(categories)),
    )


def _featurizer(case: ReaderCase, tag: str) -> FeaturizerCase:
    """Construct one discovered featurizer for a reader's resources."""
    resources = Resources(
        _hasher(case),
        TorchCodeTensorizer(),
        ItemLookup(_unique_items(case.items)),
    )
    spec = FeaturizerSpec.model_validate({"kind": tag})
    plugin = featurizer_plugin(spec)
    configured = cast("ConfiguredSpec[object]", spec)
    component = plugin.construct(configured, resources)

    return FeaturizerCase(tag, cast("DataFeaturizer", component), plugin.accepted(spec))


def _selected_examples(
    examples: Sequence[Example[Supervision[str]]],
    arm: SupervisionArm,
) -> tuple[Example[Supervision[str]], ...]:
    return tuple(example for example in examples if type(example.supervision) is arm)


def test_every_real_reader_featurizer_pair_succeeds_or_refuses_at_setup() -> None:
    load_plugins()
    registry = Registry[object]("featurizer")
    tags = registry.available()
    outcomes: dict[tuple[str, str], str] = {}

    for reader_case in _readers():
        for tag in tags:
            featurizer_case = _featurizer(reader_case, tag)
            featurizer = featurizer_case.featurizer
            key = (reader_case.name, featurizer_case.name)

            shared = reader_case.reader.supervision_kinds() & featurizer_case.accepted

            if not shared:
                outcomes[key] = "refused"
                continue

            for arm in shared:
                examples = _selected_examples(reader_case.examples, arm)
                result = featurizer(examples)

                assert examples
                assert len(result) == len(examples)
                assert len(result.batch) > 0
                assert all(type(value) is arm for value in result.supervision)

            outcomes[key] = "featurized"

    assert tags
    assert len(outcomes) == len(_readers()) * len(tags)
    assert set(outcomes.values()) == {"featurized", "refused"}


def test_real_readers_report_every_deferred_target_boundary() -> None:
    deferred = {
        (case.name, arm)
        for case in _readers()
        for arm in case.reader.supervision_kinds()
        if arm in {GoldDependencies, GoldGraph}
    }

    assert deferred == {
        ("layers", GoldDependencies),
        ("layers", GoldGraph),
    }

    for _, arm in deferred:
        with pytest.raises(DeferredSupervisionError) as raised:
            require_featurizable(arm)

        assert raised.value.arm is arm

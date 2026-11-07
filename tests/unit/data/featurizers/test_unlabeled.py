from typing import TYPE_CHECKING, Final, cast

import pytest
import torch
from tests.conformance.configured_component import (
    ConfiguredComponentConformance,
    ConfiguredComponentFixture,
)
from tests.conformance.featurizer import FeaturizerConformance, FeaturizerSuite
from tests.unit.data.featurizers.conftest import Resources, wired

from lofi.data import (
    CodeTensorizer,
    Example,
    Featurizer,
    Item,
    ItemHasher,
    ItemId,
    Scalar,
    Scale,
    Supervision,
    Unlabeled,
)
from lofi.data._join import ItemLookup, ItemMismatchError
from lofi.data.featurizers.unlabeled import UnlabeledFeaturizer
from lofi.spec.featurizers.unlabeled import UnlabeledFeaturizerSpec

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi.data.featurizers import FeaturizedBatch
    from lofi.spec import ConfiguredSpec

PAD: Final = "<pad>"
SCALE: Final = Scale("ordinal", 1.0, 7.0)


class TestUnlabeledFeaturizer(FeaturizerSuite):
    @pytest.fixture
    def conformance(
        self,
        hasher: ItemHasher,
        tensorize: CodeTensorizer,
        item_factory: Callable[..., Item],
    ) -> FeaturizerConformance:
        first = item_factory("first", "the", "dog")
        second = item_factory("second", "runs")
        examples: tuple[Example[Supervision[str]], ...] = (
            Example(first, Unlabeled()),
            Example(second, Unlabeled()),
        )
        subject = cast(
            "Featurizer[Supervision[str], FeaturizedBatch[Supervision[str]]]",
            wired(
                UnlabeledFeaturizerSpec(),
                Resources(hasher, tensorize, ItemLookup((first, second))),
            ),
        )
        return FeaturizerConformance(
            subject,
            examples,
            tuple(example.supervision for example in examples),
            ((first.item_id,), (second.item_id,)),
            Example(first, Scalar(4.0, SCALE)),
        )


def test_unlabeled_featurization_pads_with_the_reserved_code(
    hasher: ItemHasher,
    tensorize: CodeTensorizer,
    item_factory: Callable[..., Item],
) -> None:
    first = item_factory("first", "the", "dog", "runs")
    second = item_factory("second", "dog", "runs")
    result = UnlabeledFeaturizer(hasher, PAD, tensorize)(
        (Example(first, Unlabeled()), Example(second, Unlabeled()))
    )

    assert result.batch.lengths == (3, 2)
    assert result.batch.item_ids == (first.item_id, second.item_id)
    assert result.batch.mask is not None
    assert result.batch.mask.values.tolist() == [
        [True, True, True],
        [True, True, False],
    ]
    assert result.batch.vectors[1].tolist() == [
        hasher.lexicon.index("dog"),
        hasher.lexicon.index("runs"),
        hasher.lexicon.index(PAD),
    ]
    assert result.batch.alignments[0].rows_for(2) == (2, 3)
    assert result.member_rows == ((0,), (1,))


def test_padding_must_be_a_reserved_inventory_symbol(
    hasher: ItemHasher,
    tensorize: CodeTensorizer,
) -> None:
    with pytest.raises(ValueError, match="not reserved"):
        UnlabeledFeaturizer(hasher, "the", tensorize)


def test_featurization_rejects_one_id_for_different_items(
    hasher: ItemHasher,
    tensorize: CodeTensorizer,
    item_factory: Callable[..., Item],
) -> None:
    examples = (
        Example(item_factory("same", "dog"), Unlabeled()),
        Example(item_factory("same", "runs"), Unlabeled()),
    )

    with pytest.raises(ItemMismatchError) as raised:
        UnlabeledFeaturizer(hasher, PAD, tensorize)(examples)

    assert raised.value.item_id == ItemId("same")


def test_empty_featurization_has_rank_two_inputs(
    hasher: ItemHasher,
    tensorize: CodeTensorizer,
) -> None:
    result = UnlabeledFeaturizer(hasher, PAD, tensorize)(())

    assert len(result) == 0
    assert result.batch.vectors.shape == torch.Size((0, 0))
    assert result.batch.mask is not None
    assert result.batch.mask.values.shape == torch.Size((0, 0))


class TestUnlabeledFeaturizerPlugin(ConfiguredComponentConformance):
    @pytest.fixture
    def configured_component_fixture(
        self,
        hasher: ItemHasher,
        tensorize: CodeTensorizer,
        item_factory: Callable[..., Item],
    ) -> ConfiguredComponentFixture:
        resources = Resources(
            hasher, tensorize, ItemLookup((item_factory("plugin", "dog"),))
        )

        return ConfiguredComponentFixture(
            declaration=cast("ConfiguredSpec[object]", UnlabeledFeaturizerSpec()),
            context=resources,
            expected_type=UnlabeledFeaturizer,
            observe=lambda built: {
                "reads the declared arm": cast("UnlabeledFeaturizer", built).selection
                is Unlabeled
            },
            unchanged=lambda: resources.hasher is hasher,
        )


def test_the_declared_padding_reaches_the_constructor(
    hasher: ItemHasher,
    tensorize: CodeTensorizer,
    item_factory: Callable[..., Item],
) -> None:
    resources = Resources(hasher, tensorize, ItemLookup((item_factory("one", "dog"),)))

    with pytest.raises(ValueError, match="'dog' is not reserved"):
        wired(UnlabeledFeaturizerSpec(padding="dog"), resources)

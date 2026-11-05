import pickle
from typing import TYPE_CHECKING, Final, cast

import pytest
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
    ForcedChoice,
    Item,
    ItemHasher,
    Relational,
    RelationalScalar,
    Scalar,
    Scale,
    Supervision,
    Unlabeled,
)
from lofi.data._join import ItemLookup
from lofi.data.featurizers.pair import PairFeaturizer
from lofi.spec.featurizers.pair import PairFeaturizerSpec

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi.data.featurizers import FeaturizedBatch
    from lofi.spec import ConfiguredSpec

PAD: Final = "<pad>"
SCALE: Final = Scale("ordinal", 1.0, 7.0)


class TestPairFeaturizer(FeaturizerSuite):
    @pytest.fixture
    def conformance(
        self,
        hasher: ItemHasher,
        tensorize: CodeTensorizer,
        item_factory: Callable[..., Item],
    ) -> FeaturizerConformance:
        first = item_factory("first", "the")
        second = item_factory("second", "dog")
        choice = ForcedChoice(
            (first.item_id, second.item_id),
            frozenset({second.item_id}),
        )
        examples: tuple[Example[Supervision[str]], ...] = (Example(first, choice),)
        subject = cast(
            "Featurizer[Supervision[str], FeaturizedBatch[Supervision[str]]]",
            wired(
                PairFeaturizerSpec(),
                Resources(hasher, tensorize, ItemLookup((first, second))),
            ),
        )
        return FeaturizerConformance(
            subject,
            examples,
            tuple(example.supervision for example in examples),
            ((first.item_id, second.item_id),),
            Example(first, Unlabeled()),
        )


def test_pair_featurization_prejoins_each_item_once(
    hasher: ItemHasher,
    tensorize: CodeTensorizer,
    item_factory: Callable[..., Item],
) -> None:
    first = item_factory("first", "the", "dog")
    second = item_factory("second", "dog", "runs")
    examples: tuple[Example[Supervision[str]], ...] = (
        Example(first, Relational(second.item_id, "contrast")),
        Example(second, Relational(first.item_id, "contrast")),
    )
    result = PairFeaturizer(
        hasher, PAD, tensorize, ItemLookup((first, second)), Relational
    )(examples)

    assert result.batch.item_ids == (first.item_id, second.item_id)
    assert result.member_rows == ((0, 1), (1, 0))
    assert len(result.batch) == 2


def test_choice_featurization_preserves_presentation_order(
    hasher: ItemHasher,
    tensorize: CodeTensorizer,
    item_factory: Callable[..., Item],
) -> None:
    first = item_factory("first", "the")
    second = item_factory("second", "dog")
    third = item_factory("third", "runs")
    choice = ForcedChoice(
        (first.item_id, third.item_id, second.item_id),
        frozenset({third.item_id}),
    )
    examples: tuple[Example[Supervision[str]], ...] = (Example(first, choice),)
    result = PairFeaturizer(
        hasher,
        PAD,
        tensorize,
        ItemLookup((first, second, third)),
        ForcedChoice,
    )(examples)

    assert result.batch.item_ids == (first.item_id, third.item_id, second.item_id)
    assert result.member_rows == ((0, 1, 2),)
    assert result.supervision == (choice,)


def test_pair_featurization_supports_graded_relations(
    hasher: ItemHasher,
    tensorize: CodeTensorizer,
    item_factory: Callable[..., Item],
) -> None:
    first = item_factory("first", "the")
    second = item_factory("second", "dog")
    relation = RelationalScalar(second.item_id, "similarity", Scalar(5.0, SCALE))
    examples: tuple[Example[Supervision[str]], ...] = (Example(first, relation),)
    result = PairFeaturizer(
        hasher,
        PAD,
        tensorize,
        ItemLookup((first, second)),
        RelationalScalar,
    )(examples)

    assert result.member_rows == ((0, 1),)
    assert result.supervision == (relation,)


def test_a_pair_featurizer_survives_a_pickle_round_trip(
    hasher: ItemHasher,
    tensorize: CodeTensorizer,
    item_factory: Callable[..., Item],
) -> None:
    first = item_factory("first", "the")
    second = item_factory("second", "dog")
    relation = Relational(second.item_id, "contrast")
    featurizer = PairFeaturizer(
        hasher, PAD, tensorize, ItemLookup((first, second)), Relational
    )

    restored = cast("PairFeaturizer", pickle.loads(pickle.dumps(featurizer)))
    result = restored((Example(first, relation),))

    assert restored.selection is Relational
    assert result.batch.item_ids == (first.item_id, second.item_id)
    assert result.member_rows == ((0, 1),)


class TestPairFeaturizerPlugin(ConfiguredComponentConformance):
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
            declaration=cast(
                "ConfiguredSpec[object]", PairFeaturizerSpec(selection="relational")
            ),
            context=resources,
            expected_type=PairFeaturizer,
            observe=lambda built: {
                "reads the declared arm": cast("PairFeaturizer", built).selection
                is Relational
            },
            unchanged=lambda: resources.hasher is hasher,
        )

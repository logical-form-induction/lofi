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
    Item,
    ItemHasher,
    Scalar,
    Scale,
    Supervision,
    Unlabeled,
)
from lofi.data._join import ItemLookup
from lofi.data.featurizers import MixedSupervisionError
from lofi.data.featurizers.scalar import ScalarFeaturizer
from lofi.spec.featurizers.scalar import ScalarFeaturizerSpec

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi.data.featurizers import FeaturizedBatch
    from lofi.spec import ConfiguredSpec

PAD: Final = "<pad>"
SCALE: Final = Scale("ordinal", 1.0, 7.0)


class TestScalarFeaturizer(FeaturizerSuite):
    @pytest.fixture
    def conformance(
        self,
        hasher: ItemHasher,
        tensorize: CodeTensorizer,
        item_factory: Callable[..., Item],
    ) -> FeaturizerConformance:
        item = item_factory("rating", "dog")
        examples: tuple[Example[Supervision[str]], ...] = (
            Example(item, Scalar(6.0, SCALE)),
        )
        subject = cast(
            "Featurizer[Supervision[str], FeaturizedBatch[Supervision[str]]]",
            wired(
                ScalarFeaturizerSpec(),
                Resources(hasher, tensorize, ItemLookup((item,))),
            ),
        )
        return FeaturizerConformance(
            subject,
            examples,
            tuple(example.supervision for example in examples),
            ((item.item_id,),),
            Example(item, Unlabeled()),
        )


def test_scalar_featurization_preserves_the_hashed_response(
    hasher: ItemHasher,
    tensorize: CodeTensorizer,
    item_factory: Callable[..., Item],
) -> None:
    example = Example(item_factory("rating", "dog"), Scalar(6.0, SCALE))
    result = ScalarFeaturizer(hasher, PAD, tensorize)((example,))

    assert result.hashed[0].response is example.supervision
    assert result.supervision == (example.supervision,)
    assert result.member_rows == ((0,),)


def test_scalar_featurization_rejects_an_unselected_arm(
    hasher: ItemHasher,
    tensorize: CodeTensorizer,
    item_factory: Callable[..., Item],
) -> None:
    example = cast(
        "Example[Scalar]",
        Example(item_factory("wrong", "dog"), Unlabeled()),
    )

    with pytest.raises(MixedSupervisionError) as raised:
        ScalarFeaturizer(hasher, PAD, tensorize)((example,))

    assert raised.value.position == 0
    assert raised.value.selected is Scalar
    assert raised.value.observed is Unlabeled


class TestScalarFeaturizerPlugin(ConfiguredComponentConformance):
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
            declaration=cast("ConfiguredSpec[object]", ScalarFeaturizerSpec()),
            context=resources,
            expected_type=ScalarFeaturizer,
            observe=lambda built: {
                "reads the declared arm": cast("ScalarFeaturizer", built).selection
                is Scalar
            },
            unchanged=lambda: resources.hasher is hasher,
        )

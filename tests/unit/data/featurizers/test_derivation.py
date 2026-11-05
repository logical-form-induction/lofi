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
    Derivation,
    Example,
    Featurizer,
    GoldDerivation,
    Item,
    ItemHasher,
    Supervision,
    Unlabeled,
)
from lofi.data._join import ItemLookup
from lofi.data.featurizers.derivation import DerivationFeaturizer
from lofi.spec.featurizers.derivation import DerivationFeaturizerSpec

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi.data.featurizers import FeaturizedBatch
    from lofi.spec import ConfiguredSpec

PAD: Final = "<pad>"


class TestDerivationFeaturizer(FeaturizerSuite):
    @pytest.fixture
    def conformance(
        self,
        hasher: ItemHasher,
        tensorize: CodeTensorizer,
        item_factory: Callable[..., Item],
        derivation: Derivation[str],
    ) -> FeaturizerConformance:
        item = item_factory("parse", "the", "dog", "runs")
        examples: tuple[Example[Supervision[str]], ...] = (
            Example(item, GoldDerivation(derivation)),
        )
        subject = cast(
            "Featurizer[Supervision[str], FeaturizedBatch[Supervision[str]]]",
            wired(
                DerivationFeaturizerSpec(),
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


def test_derivation_featurization_hashes_lexical_leaf_labels(
    hasher: ItemHasher,
    tensorize: CodeTensorizer,
    item_factory: Callable[..., Item],
    derivation: Derivation[str],
) -> None:
    result = DerivationFeaturizer(hasher, PAD, tensorize)(
        (
            Example(
                item_factory("parse", "the", "dog", "runs"),
                GoldDerivation(derivation),
            ),
        )
    )

    assert result.hashed[0].analyses == (
        frozenset({hasher.categories.index("NP/N")}),
        frozenset({hasher.categories.index("N")}),
        frozenset({hasher.categories.index("S\\NP")}),
    )


class TestDerivationFeaturizerPlugin(ConfiguredComponentConformance):
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
            declaration=cast("ConfiguredSpec[object]", DerivationFeaturizerSpec()),
            context=resources,
            expected_type=DerivationFeaturizer,
            observe=lambda built: {
                "reads the declared arm": cast("DerivationFeaturizer", built).selection
                is GoldDerivation
            },
            unchanged=lambda: resources.hasher is hasher,
        )

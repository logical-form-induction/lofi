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
    SpanConstraint,
    SpanConstraints,
    Supervision,
    TokenSpan,
    TypeScope,
    Unlabeled,
)
from lofi.data._join import ItemLookup
from lofi.data.featurizers.span_constraint import SpanConstraintFeaturizer
from lofi.spec.featurizers.span_constraint import SpanConstraintFeaturizerSpec

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi.data.featurizers import FeaturizedBatch
    from lofi.spec import ConfiguredSpec

PAD: Final = "<pad>"


def _constraints() -> SpanConstraints[str]:
    """Return a type-scoped span constraint."""
    return SpanConstraints(
        (
            SpanConstraint(
                TokenSpan(0, 2),
                frozenset({"NP"}),
                TypeScope(("the", "dog")),
            ),
        ),
        4,
    )


class TestSpanConstraintFeaturizer(FeaturizerSuite):
    @pytest.fixture
    def conformance(
        self,
        hasher: ItemHasher,
        tensorize: CodeTensorizer,
        item_factory: Callable[..., Item],
    ) -> FeaturizerConformance:
        item = item_factory("constraints", "the", "dog", "the", "dog")
        examples: tuple[Example[Supervision[str]], ...] = (
            Example(item, _constraints()),
        )
        expected = SpanConstraints(
            _constraints().instances(item.tokens),
            len(item.tokens),
        )
        subject = cast(
            "Featurizer[Supervision[str], FeaturizedBatch[Supervision[str]]]",
            wired(
                SpanConstraintFeaturizerSpec(),
                Resources(hasher, tensorize, ItemLookup((item,))),
            ),
        )
        return FeaturizerConformance(
            subject,
            examples,
            (expected,),
            ((item.item_id,),),
            Example(item, Unlabeled()),
        )


def test_span_featurization_expands_every_form_occurrence(
    hasher: ItemHasher,
    tensorize: CodeTensorizer,
    item_factory: Callable[..., Item],
) -> None:
    subject = item_factory("constraints", "the", "dog", "the", "dog")
    supervision = _constraints()
    result = SpanConstraintFeaturizer(hasher, PAD, tensorize)(
        (Example(subject, supervision),)
    )

    constraints = result.supervision[0].constraints
    assert tuple(constraint.span for constraint in constraints) == (
        TokenSpan(0, 2),
        TokenSpan(2, 4),
    )
    assert all(constraint.scope.form is None for constraint in constraints)


def test_position_constraints_hash_to_their_permitted_category_codes(
    hasher: ItemHasher,
    tensorize: CodeTensorizer,
    item_factory: Callable[..., Item],
) -> None:
    supervision = SpanConstraints.from_positions(
        (frozenset({"NP/N", "N"}), frozenset({"N"}))
    )
    result = SpanConstraintFeaturizer(hasher, PAD, tensorize)(
        (Example(item_factory("tags", "the", "dog"), supervision),)
    )

    assert result.hashed[0].analyses == (
        frozenset(
            {
                hasher.categories.index("NP/N"),
                hasher.categories.index("N"),
            }
        ),
        frozenset({hasher.categories.index("N")}),
    )


class TestSpanConstraintFeaturizerPlugin(ConfiguredComponentConformance):
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
            declaration=cast("ConfiguredSpec[object]", SpanConstraintFeaturizerSpec()),
            context=resources,
            expected_type=SpanConstraintFeaturizer,
            observe=lambda built: {
                "reads the declared arm": cast(
                    "SpanConstraintFeaturizer", built
                ).selection
                is SpanConstraints
            },
            unchanged=lambda: resources.hasher is hasher,
        )

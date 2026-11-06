from typing import TYPE_CHECKING

import pytest

from lofi.data import (
    Categorical,
    CodeTensorizer,
    Example,
    ForcedChoice,
    GoldDependencies,
    GoldDerivation,
    GoldGraph,
    Item,
    ItemHasher,
    Relational,
    RelationalScalar,
    Scalar,
    SpanConstraints,
    SupervisionArm,
    Unlabeled,
)
from lofi.data.featurizers import (
    DeferredSupervisionError,
    FeaturizedBatch,
    FeaturizerPlugin,
    SupervisionDisposition,
    disposition_for,
    featurizer_plugin,
    require_featurizable,
)
from lofi.data.featurizers.unlabeled import UnlabeledFeaturizer
from lofi.registry import ComponentPlugin, Registry, RegistryError
from lofi.spec import FeaturizerSpec
from lofi.spec.featurizers.unlabeled import UnlabeledFeaturizerSpec

if TYPE_CHECKING:
    from collections.abc import Callable

PAD = "<pad>"


@pytest.mark.parametrize(
    ("arm", "expected"),
    [
        (Unlabeled, SupervisionDisposition.BATCH),
        (Scalar, SupervisionDisposition.HASHED_TARGET),
        (GoldDerivation, SupervisionDisposition.HASHED_TARGET),
        (Relational, SupervisionDisposition.PREJOIN),
        (RelationalScalar, SupervisionDisposition.PREJOIN),
        (ForcedChoice, SupervisionDisposition.PREJOIN),
        (SpanConstraints, SupervisionDisposition.EXPAND),
        (Categorical, SupervisionDisposition.DEFERRED),
        (GoldGraph, SupervisionDisposition.DEFERRED),
        (GoldDependencies, SupervisionDisposition.DEFERRED),
    ],
)
def test_every_supervision_arm_has_a_disposition(
    arm: SupervisionArm,
    expected: SupervisionDisposition,
) -> None:
    assert disposition_for(arm) is expected


@pytest.mark.parametrize("arm", [Categorical, GoldGraph, GoldDependencies])
def test_undeclared_target_representations_are_deferred(
    arm: SupervisionArm,
) -> None:
    with pytest.raises(DeferredSupervisionError) as raised:
        require_featurizable(arm)

    assert raised.value.arm is arm


def test_featurized_batch_refuses_an_invalid_member_row(
    hasher: ItemHasher,
    tensorize: CodeTensorizer,
    item_factory: Callable[..., Item],
) -> None:
    subject = item_factory("one", "dog")
    result = UnlabeledFeaturizer(hasher, PAD, tensorize)(
        (Example(subject, Unlabeled()),)
    )

    with pytest.raises(ValueError, match="row 1"):
        FeaturizedBatch(
            result.batch,
            result.hashed,
            result.supervision,
            ((1,),),
        )


def test_featurized_batch_refuses_a_swapped_anchor_row(
    hasher: ItemHasher,
    tensorize: CodeTensorizer,
    item_factory: Callable[..., Item],
) -> None:
    first = item_factory("first", "the")
    second = item_factory("second", "dog")
    result = UnlabeledFeaturizer(hasher, PAD, tensorize)(
        (Example(first, Unlabeled()), Example(second, Unlabeled()))
    )

    with pytest.raises(ValueError, match="first member row"):
        FeaturizedBatch(
            result.batch,
            result.hashed,
            result.supervision,
            ((1,), (0,)),
        )


@pytest.mark.parametrize(
    ("declaration", "accepted"),
    [
        ({"kind": "unlabeled"}, frozenset({Unlabeled})),
        ({"kind": "scalar"}, frozenset({Scalar})),
        ({"kind": "derivation"}, frozenset({GoldDerivation})),
        ({"kind": "span-constraint"}, frozenset({SpanConstraints})),
        ({"kind": "pair", "selection": "forced-choice"}, frozenset({ForcedChoice})),
        ({"kind": "pair", "selection": "relational"}, frozenset({Relational})),
        (
            {"kind": "pair", "selection": "relational-scalar"},
            frozenset({RelationalScalar}),
        ),
    ],
    ids=lambda value: value["kind"] if isinstance(value, dict) else "",
)
def test_every_installed_featurizer_declares_its_arms_without_being_built(
    declaration: dict[str, str], accepted: frozenset[SupervisionArm]
) -> None:
    spec = FeaturizerSpec.model_validate(declaration)
    plugin = featurizer_plugin(spec)

    assert isinstance(plugin, FeaturizerPlugin)
    assert plugin.accepted(spec) == accepted


def test_a_featurizer_entry_must_name_a_featurizer_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bare = ComponentPlugin(
        spec=UnlabeledFeaturizerSpec,
        build=lambda _settings, _resources: None,
    )
    monkeypatch.setattr(Registry, "get", lambda _registry, _tag: bare)

    with pytest.raises(RegistryError, match="not a featurizer plugin"):
        featurizer_plugin(UnlabeledFeaturizerSpec())

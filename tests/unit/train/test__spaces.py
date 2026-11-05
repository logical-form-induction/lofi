"""Assembly of the declared represented-space graph."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, cast

import pytest
import torch
from tests.conformance.component_plugin import plugin_context
from torch import nn

import lofi.nn.loss._readout as readout_module
from lofi._types import ModuleSharing, ReadoutId, SharedSubmodule
from lofi.judge import NoNormalForm, RuleSet
from lofi.nn._values import Encoded
from lofi.nn.codec import NeuralCodecView
from lofi.nn.loss._readout import TermsReadout
from lofi.nn.spacemap import TypeSpaceMap
from lofi.nn.spaces import ChartCombine, RelationKind
from lofi.registry import Registry
from lofi.spec.components.model import DataModelLanguageSpec, ModelSection
from lofi.spec.decoders.rnn import RnnTermDecoderSpec
from lofi.spec.encoders.rnn import TreeRnnEncoderSpec
from lofi.spec.readouts import ReadoutSource, TargetSource, TermsReadoutSpec
from lofi.spec.spaces import (
    CombinatorFamilySpec,
    Merge,
    SpaceMapForm,
    SpaceRelation,
    SpaceRelationSpec,
    SpaceSpec,
    TypeSpaceMapSpec,
)
from lofi.terms import (
    CompleteTree,
    Identity,
    Operator,
    OperatorIndex,
    Signature,
    TargetSet,
    TermCodec,
    TermStore,
)
from lofi.train._spaces import wire_spaces

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.judge import Rule
    from lofi.nn.build import BuildContext
    from lofi.nn.decoders import RecursiveTermDecoder
    from lofi.terms import Term


class CellCombination(nn.Module):
    """A chart-cell composition capability with the declared output width."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.projection = nn.Linear(2 * width, width)

    def compose(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Compose aligned daughter cells."""
        return self.projection(torch.cat((left, right), dim=-1))


class PublishingParser(nn.Module):
    """A parser test double publishing its real cell-combination module."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.combine = CellCombination(width)

    @property
    def shared(self) -> Mapping[str, SharedSubmodule[nn.Module]]:
        """Publish cell combination without parameter or state registration."""
        return MappingProxyType(
            {
                "combine": SharedSubmodule(
                    self.combine,
                    ModuleSharing(in_parameters=False, in_state_dict=False),
                )
            }
        )

    @property
    def span_width(self) -> int:
        """Return the width of a concatenated inside/outside representation."""
        return 2 * self.combine.projection.out_features


class _ComposeRule:
    """Minimal binary rule declaration for space-wiring tests."""

    name = "compose"
    arity = 2
    restriction = None

    def __init__(self, matching: Identity[Term]) -> None:
        self.matching = matching


def _context() -> BuildContext:
    """Return a context whose composing action names its installed rule."""
    matching = Identity()
    return plugin_context(
        rules=RuleSet(
            "space-wiring",
            (cast("Rule[Term]", _ComposeRule(matching)),),
            matching,
            NoNormalForm(),
        )
    )


def _model(
    *,
    merge: Merge = Merge.CHART_COMBINE,
    reads: ReadoutSource = ReadoutSource.INSIDE,
    targets: TargetSource | None = None,
) -> ModelSection:
    """Return one document-shaped declaration of a represented term space."""
    return ModelSection(
        language=DataModelLanguageSpec(),
        encoders={"encoding": TreeRnnEncoderSpec(width=4)},
        decoders={
            "identity": RnnTermDecoderSpec(width=4),
            "application": RnnTermDecoderSpec(width=4),
        },
        spaces={
            "terms": SpaceSpec(
                sort="T",
                encoder="encoding",
                identity="identity",
                combinators=CombinatorFamilySpec(
                    merge=merge,
                    decoders={"compose": "application"},
                ),
            )
        },
        readouts={
            "publishing": TermsReadoutSpec(space="terms", reads=reads, targets=targets)
        },
    )


def _relation_model(via: SpaceRelation) -> ModelSection:
    """Return two represented spaces under one declared relation."""
    target_width = 6 if via is SpaceRelation.MAPPED else 4

    return ModelSection(
        language=DataModelLanguageSpec(),
        encoders={
            "category-encoding": TreeRnnEncoderSpec(width=4),
            "type-encoding": TreeRnnEncoderSpec(width=target_width),
        },
        decoders={
            "category-identity": RnnTermDecoderSpec(width=4),
            "type-identity": RnnTermDecoderSpec(width=target_width),
        },
        spaces={
            "categories": SpaceSpec(
                sort="Cat",
                encoder="category-encoding",
                identity="category-identity",
                relation=SpaceRelationSpec(
                    space="types",
                    via=via,
                    mapping=(
                        TypeSpaceMapSpec(form=SpaceMapForm.LINEAR)
                        if via is SpaceRelation.MAPPED
                        else None
                    ),
                ),
            ),
            "types": SpaceSpec(
                sort="Ty",
                encoder="type-encoding",
                identity="type-identity",
            ),
        },
    )


def _relation_context() -> BuildContext:
    """Return a build context whose learner language has two represented sorts."""
    category = Operator("category", "Cat", family="primitive")
    semantic = Operator("type", "Ty", family="primitive")
    signature = Signature("related-spaces", ("Cat", "Ty"), (category, semantic))
    codec = TermCodec(
        OperatorIndex(signature),
        CompleteTree(signature, (category,), order="bf", depth=2),
    )

    return plugin_context(codec=NeuralCodecView(codec))


@pytest.fixture(autouse=True)
def current_readout_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Expose the current source plugin across the held editable-metadata skew."""
    original_get = Registry.get

    def get[MemberT](registry: Registry[MemberT], tag: str) -> MemberT:
        if registry.family == "readout" and tag == "terms":
            return cast("MemberT", readout_module.PLUGIN)

        return original_get(registry, tag)

    monkeypatch.setattr(Registry, "get", get)


def test_the_declared_space_graph_builds_named_components_and_chart_merge() -> None:
    parser = PublishingParser(4)
    graph = wire_spaces(
        _model(),
        _context(),
        parser=parser,
        pretrained_resources={},
        erasure=None,
    )

    space = graph.spaces["terms"]
    family = space.combinators

    assert set(graph.components) == {
        "encoding",
        "identity",
        "application",
        "publishing",
    }
    assert isinstance(graph.readouts["publishing"], TermsReadout)
    assert family is not None
    assert isinstance(family.merge, ChartCombine)
    assert family.decoders[family.composing[0]] is (graph.components["application"])


def test_a_span_readout_builds_its_decoder_at_the_derived_span_width() -> None:
    parser = PublishingParser(4)
    graph = wire_spaces(
        _model(reads=ReadoutSource.SPAN),
        _context(),
        parser=parser,
        pretrained_resources={},
        erasure=None,
    )
    decoder = cast("RecursiveTermDecoder", graph.components["identity"])

    assert graph.readouts["publishing"].reads == "span"
    assert decoder.input_width == parser.span_width


def test_each_readout_receives_only_its_fixed_and_injected_targets() -> None:
    context = _context()
    selected = TargetSet((context.categories[0],), Identity())
    run = context.with_targets(
        {"grammar-goal": selected, "another-run-target": selected}
    )
    graph = wire_spaces(
        _model(targets=TargetSource.GRAMMAR_GOAL),
        run,
        parser=PublishingParser(4),
        pretrained_resources={},
        role_targets={"publishing": {"pinning": selected}},
        erasure=None,
    )

    assert graph.readouts["publishing"].targets == (
        ReadoutId("grammar-goal"),
        ReadoutId("pinning"),
    )


def test_a_source_less_readout_receives_only_its_injected_targets() -> None:
    context = _context()
    selected = TargetSet((context.categories[0],), Identity())
    graph = wire_spaces(
        _model(),
        context.with_targets({"grammar-goal": selected}),
        parser=PublishingParser(4),
        pretrained_resources={},
        role_targets={"publishing": {"pinning": selected}},
        erasure=None,
    )

    assert graph.readouts["publishing"].targets == (ReadoutId("pinning"),)


def test_chart_merge_requires_the_declared_parser_publication() -> None:
    with pytest.raises(TypeError, match="publishes shared submodules"):
        wire_spaces(
            _model(),
            _context(),
            parser=object(),
            pretrained_resources={},
            erasure=None,
        )


def test_concatenation_needs_no_parser_publication() -> None:
    graph = wire_spaces(
        _model(merge=Merge.CONCATENATE),
        _context(),
        parser=None,
        pretrained_resources={},
        erasure=None,
    )

    assert graph.spaces["terms"].combinators is not None
    assert graph.components["encoding"] is graph.spaces["terms"].encoder


@pytest.mark.parametrize(
    ("declared", "kind"),
    [
        (SpaceRelation.ERASE_THEN_DECODE, RelationKind.ERASE_THEN_DECODE),
        (SpaceRelation.CONSISTENCY, RelationKind.CONSISTENCY),
        (SpaceRelation.MAPPED, RelationKind.MAPPED),
    ],
)
def test_each_declared_space_relation_builds_one_runtime_binding(
    declared: SpaceRelation,
    kind: RelationKind,
) -> None:
    context = _relation_context()
    store = TermStore(context.codec.codec.index.signature)
    graph = wire_spaces(
        _relation_model(declared),
        context,
        parser=None,
        pretrained_resources={},
        erasure=(
            None
            if declared is SpaceRelation.MAPPED
            else lambda _term: store.op(store.signature.by_name("type"))
        ),
    )
    relation = graph.relations["categories"]

    assert relation.kind is kind
    assert relation.source is graph.spaces["categories"]
    assert relation.target is graph.spaces["types"]

    if kind is RelationKind.MAPPED:
        assert graph.components["categories"] is relation
        assert isinstance(relation.mapping, TypeSpaceMap)
        projected = relation.project(Encoded(torch.ones(2, 4), ("term", "width")))
        assert tuple(projected.tensor.shape) == (2, 6)
    elif kind is RelationKind.CONSISTENCY:
        assert "categories" not in graph.components
        vectors = Encoded(torch.ones(2, 4), ("term", "width"))
        assert relation.project(vectors) is vectors
        assert (
            relation.erase(store.op(store.signature.by_name("category"))).op.name
            == "type"
        )
    else:
        assert "categories" not in graph.components
        with pytest.raises(ValueError, match="decoded terms"):
            relation.project(Encoded(torch.ones(2, 4), ("term", "width")))
        assert (
            relation.erase(store.op(store.signature.by_name("category"))).op.name
            == "type"
        )

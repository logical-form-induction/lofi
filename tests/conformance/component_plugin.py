"""Reusable laws for an installed component plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, NamedTuple, cast

import pytest
import torch
from torch import nn

from lofi._types import ComponentId
from lofi.judge import Action, ActionSet, NoNormalForm, RuleSet
from lofi.nn.build import BuildContext, build_component
from lofi.nn.codec import NeuralCodecView
from lofi.nn.policy import DevicePolicy
from lofi.nn.tokens.embedder import ReservedTokens, TokenVocabulary
from lofi.registry import plugin_for
from lofi.spec.language import LanguageCodec, SortBound
from lofi.terms import (
    CompleteTree,
    Identity,
    Operator,
    OperatorIndex,
    PreorderArityLayout,
    Signature,
    Term,
    TermCodec,
    TermStore,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from lofi.lf import SurfaceSyntax
    from lofi.nn.tokens.pretrained import PretrainedLoader
    from lofi.spec import ConfiguredSpec
    from lofi.spec.language import LanguageAssembly
    from lofi.terms import TargetSet

__all__ = ["ComponentPluginConformance", "ComponentPluginFixture", "plugin_context"]

_IDENTITY = Action("identity")


class _FixtureLanguage(NamedTuple):
    """A test learner language owning one codec collection."""

    codec: LanguageCodec
    store: TermStore
    syntax: SurfaceSyntax


def _fixture_language(codec: NeuralCodecView, /) -> _FixtureLanguage:
    """Return explicit learner-language resources for one fixture signature."""
    signature = codec.codec.index.signature
    term_codec = cast("TermCodec", codec.codec)
    layout = term_codec.layout

    if isinstance(layout, CompleteTree):
        bound = SortBound(
            "complete-tree",
            depth=layout.depth,
            nodes=None,
            order=layout.order,
            free_ranks=0,
        )
    elif isinstance(layout, PreorderArityLayout):
        bound = SortBound(
            "preorder-arity",
            depth=layout.capacity,
            nodes=layout.capacity,
            order="df",
            free_ranks=0,
        )
    else:
        msg = f"unsupported fixture term layout: {layout!r}"
        raise TypeError(msg)

    bounds = dict.fromkeys(signature.sorts, bound)

    return _FixtureLanguage(
        LanguageCodec(signature.name, signature, bounds),
        TermStore(signature),
        cast("SurfaceSyntax", object()),
    )


def _fixture_codec() -> NeuralCodecView:
    """Return a codec of one atom and one pairing constructor."""
    atom = Operator("atom", "T", family="primitive")
    pair = Operator("pair", "T", arg_sorts=("T", "T"), family="constructor")
    signature = Signature("plugin-context", ("T",), (atom, pair))

    return NeuralCodecView(
        TermCodec(
            OperatorIndex(signature),
            CompleteTree(signature, signature.operators, order="bf", depth=3),
        )
    )


def _fixture_atom(store: TermStore, /) -> Term:
    """Return the first closed nullary term the fixture signature admits."""
    signature = store.signature

    for operator in signature.operators:
        if operator.arity:
            continue

        if operator.payload_sort == "none":
            return store.op(operator)

        if operator.is_constant:
            vocabulary = signature.vocabulary(operator.sort)

            if vocabulary.names:
                return store.op(operator, payload=vocabulary.id(vocabulary.names[0]))

    msg = f"fixture signature {signature.name!r} declares no closed nullary term"
    raise ValueError(msg)


def plugin_context(
    *,
    codec: NeuralCodecView | None = None,
    language: LanguageAssembly | None = None,
    component: str = "component",
    producers: Mapping[str, object] | None = None,
    policy: DevicePolicy | None = None,
    vocabulary: TokenVocabulary | None = None,
    actions: ActionSet | None = None,
    rules: RuleSet[Term] | None = None,
    categories: tuple[Term, ...] | None = None,
    targets: Mapping[str, TargetSet[Term]] | None = None,
    loader: PretrainedLoader | None = None,
) -> BuildContext:
    """Return a build context over one codec, with test values for the rest.

    Parameters
    ----------
    codec : NeuralCodecView | None
        The codec view the component is built over, or None for a codec of one
        atom and one pairing constructor.
    language : LanguageAssembly | None
        The language assembly owning the codec's choice index, or None for an
        assembly synthesized from the codec.
    component : str
        The component handle.
    producers : Mapping[str, object] | None
        The wired producers, or None for none.
    policy : DevicePolicy | None
        The policy, or None for single precision on the CPU.
    vocabulary : TokenVocabulary | None
        The token vocabulary, or None for two words and an unknown token.
    actions : ActionSet | None
        The action inventory, or None for the identity and one composing action.
    rules : RuleSet[Term] | None
        The rule inventory, or None for an empty identity-matched inventory.
    categories : tuple[Term, ...] | None
        The chart categories, or None for the fixture signature's atom.
    targets : Mapping[str, TargetSet[Term]] | None
        The target sets, or None for none.
    loader : PretrainedLoader | None
        The pretrained backend, or None for a local-only Hugging Face loader.
    """
    held_codec = _fixture_codec() if codec is None else codec
    held_language = _fixture_language(held_codec) if language is None else language
    held_rules = (
        RuleSet(
            "plugin-context",
            (),
            Identity[Term](),
            NoNormalForm[Term](),
        )
        if rules is None
        else rules
    )
    held_categories = categories

    if held_categories is None:
        held_categories = (_fixture_atom(held_language.store),)

    return BuildContext(
        component=ComponentId(component),
        policy=DevicePolicy(torch.device("cpu"), torch.float32)
        if policy is None
        else policy,
        language=cast("LanguageAssembly", held_language),
        codec=held_codec,
        vocabulary=TokenVocabulary(
            ("dog", "cat"), reserved=ReservedTokens({"<unk>": 0}), unknown="<unk>"
        )
        if vocabulary is None
        else vocabulary,
        actions=ActionSet((_IDENTITY, Action("compose", "compose")), _IDENTITY)
        if actions is None
        else actions,
        rules=held_rules,
        categories=held_categories,
        producers={} if producers is None else producers,
        targets={} if targets is None else targets,
        loader=loader,
    )


class ComponentPluginFixture(NamedTuple):
    """One declaration, the context it is built in, and what the build shows.

    Attributes
    ----------
    declaration : ConfiguredSpec[object]
        A validated declaration of the variant under test.
    context : BuildContext
        The context the component is built in.
    expected_type : type
        The exact type of the component the declaration selects.
    observe_resources : Callable[[object], Mapping[str, bool]]
        Named evidence that the component read its settings and its context.
    """

    declaration: ConfiguredSpec[object]
    context: BuildContext
    expected_type: type
    observe_resources: Callable[[object], Mapping[str, bool]]


type _Snapshot = tuple[tuple[str, torch.Tensor], ...]


def _snapshot(tensors: Iterable[tuple[str, torch.Tensor]], /) -> _Snapshot:
    """Return detached copies of named tensors."""
    return tuple(
        (name, tensor.detach().clone())
        for name, tensor in tensors
        if isinstance(tensor, torch.Tensor)
    )


def _producer_state(producers: Mapping[str, object], /) -> dict[str, _Snapshot]:
    """Return the state of every module among the producers, by port."""
    return {
        port: _snapshot(producer.state_dict().items())
        for port, producer in producers.items()
        if isinstance(producer, nn.Module)
    }


def _defaults(function: object, /) -> tuple[object, ...]:
    """Return the positional and keyword defaults a function declares."""
    positional = getattr(function, "__defaults__", None) or ()
    keyword = getattr(function, "__kwdefaults__", None) or {}

    return (*positional, *keyword.values())


class ComponentPluginConformance:
    """Laws every installed component plugin obeys."""

    pytestmark: ClassVar[list[pytest.MarkDecorator]] = [pytest.mark.conformance]

    def test_the_installed_plugin_pairs_the_declarations_class(
        self, component_plugin_fixture: ComponentPluginFixture
    ) -> None:
        """Resolve the declaration's family and tag to a plugin for its class."""
        declaration = component_plugin_fixture.declaration

        assert plugin_for(declaration).spec is type(declaration)

    def test_the_component_reads_its_resources_without_changing_its_producers(
        self, component_plugin_fixture: ComponentPluginFixture
    ) -> None:
        """Build the component and observe its type, placement and resources."""
        fixture = component_plugin_fixture
        before = _producer_state(fixture.context.producers)
        built = build_component(fixture.declaration, fixture.context)
        after = _producer_state(fixture.context.producers)

        assert isinstance(built, fixture.expected_type)
        assert before.keys() == after.keys()

        for port, saved in before.items():
            for (name, current), (_, kept) in zip(after[port], saved, strict=True):
                assert torch.equal(current, kept), (port, name)

        if isinstance(built, nn.Module):
            policy = fixture.context.policy

            for tensor in (*built.parameters(), *built.buffers()):
                assert tensor.device == policy.device

                if tensor.is_floating_point():
                    assert tensor.dtype is policy.dtype

        observations = fixture.observe_resources(built)

        assert observations
        assert all(observations.values()), observations

    def test_neither_the_builder_nor_the_constructor_declares_a_default(
        self, component_plugin_fixture: ComponentPluginFixture
    ) -> None:
        """Keep every default on the declaration's fields."""
        fixture = component_plugin_fixture
        plugin = plugin_for(fixture.declaration)

        assert _defaults(plugin.build) == ()
        assert _defaults(fixture.expected_type.__init__) == ()

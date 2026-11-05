"""Assembly of named term encoders, decoders, spaces and readouts."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from lofi._frozen import FrozenValue, Unpicklable
from lofi._types import ComponentId
from lofi.nn.build import BuildContext, build_component
from lofi.nn.codec import NeuralCodecView
from lofi.nn.policy import place
from lofi.nn.ports import (
    FrontierDecoder,
    MarginalDecoder,
    TeacherForcedDecoder,
    TermEncoder,
)
from lofi.nn.sharing import ModuleSharingProducer
from lofi.nn.spacemap import MapForm, MultilayerOptions, TypeSpaceMap
from lofi.nn.spaces import (
    CellComposer,
    ChartCombine,
    CombinatorFamily,
    Concatenate,
    PairMerge,
    RelationKind,
    Space,
    SpaceRelationBinding,
    TermMorphism,
)
from lofi.spec.components.model import DecoderCapability
from lofi.spec.readouts import ReadoutSource
from lofi.spec.spaces import (
    Merge,
    SpaceMapForm,
    SpaceRelation,
    TypeSpaceMapSpec,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi._types import SubmoduleDeclaration
    from lofi.nn.loss._readout import TermsReadout
    from lofi.nn.tokens.pretrained import PretrainedLoader
    from lofi.spec import ConfiguredSpec
    from lofi.spec.components.model import ModelSection
    from lofi.spec.readouts import TermsReadoutSettings
    from lofi.terms import TargetSet, Term, TermCodec

__all__ = [
    "SpaceGraph",
    "assemble_space_graph",
    "build_space_encoders",
    "wire_spaces",
]


@runtime_checkable
class _WidthProducer(Protocol):
    """A built producer whose output width determines decoder input width."""

    @property
    def output_width(self) -> int:
        """Return the width of one produced vector."""


@runtime_checkable
class _SpanWidthProducer(Protocol):
    """A chart that publishes concatenated inside/outside span vectors."""

    @property
    def span_width(self) -> int:
        """Return the width of one published span representation."""


class _ReadoutVectors(FrozenValue):
    """The vector width selected by a chart-cell readout."""

    __slots__ = ("_output_width",)
    _noun = "readout vector producer"

    def __init__(self, output_width: int, /) -> None:
        object.__setattr__(self, "_output_width", output_width)

    @property
    def output_width(self) -> int:
        """Return the width handed to a readout's decoder."""
        return self._output_width


@runtime_checkable
class _CodecEncoder(TermEncoder, Protocol):
    """A term encoder bound to the codec of its represented sort."""

    @property
    def codec(self) -> NeuralCodecView:
        """Return the codec used to encode terms."""


class SpaceGraph(FrozenValue, Unpicklable):
    """Built term components and represented spaces of one model."""

    _noun = "space graph"
    _because = "holds built neural modules"
    __slots__ = ("_components", "_readouts", "_relations", "_spaces")

    def __init__(
        self,
        *,
        components: Mapping[str, object],
        spaces: Mapping[str, Space],
        relations: Mapping[str, SpaceRelationBinding],
        readouts: Mapping[str, TermsReadout],
    ) -> None:
        object.__setattr__(self, "_components", MappingProxyType(dict(components)))
        object.__setattr__(self, "_spaces", MappingProxyType(dict(spaces)))
        object.__setattr__(self, "_relations", MappingProxyType(dict(relations)))
        object.__setattr__(self, "_readouts", MappingProxyType(dict(readouts)))

    @property
    def components(self) -> Mapping[str, object]:
        """Return named encoders, decoders and readouts."""
        return self._components

    @property
    def spaces(self) -> Mapping[str, Space]:
        """Return represented spaces keyed by declared role."""
        return self._spaces

    @property
    def relations(self) -> Mapping[str, SpaceRelationBinding]:
        """Return cross-space relations keyed by their source-space role."""
        return self._relations

    @property
    def readouts(self) -> Mapping[str, TermsReadout]:
        """Return term readouts keyed by declared role."""
        return self._readouts


def _codec(run: BuildContext, sort: str, /) -> NeuralCodecView:
    """Return the learner-language codec view for one represented sort."""
    return NeuralCodecView(run.language.codec.at(sort))


def _resource(
    context: BuildContext,
    role: str,
    resources: Mapping[str, PretrainedLoader],
    /,
) -> BuildContext:
    """Return a component context with its declared pretrained resource."""
    loader = resources.get(role)

    return context if loader is None else context.with_loader(loader)


def _encoder_spaces(model: ModelSection) -> Mapping[str, tuple[str, ...]]:
    """Return the represented-space roles reading each encoder role."""
    grouped: dict[str, list[str]] = {}

    for space, declaration in model.spaces.items():
        grouped.setdefault(declaration.encoder, []).append(space)

    return MappingProxyType(
        {role: tuple(spaces) for role, spaces in sorted(grouped.items())}
    )


def build_space_encoders(
    model: ModelSection,
    run: BuildContext,
    resources: Mapping[str, PretrainedLoader],
    /,
) -> dict[str, _CodecEncoder]:
    """Build each named encoder against its represented sort codec."""
    unknown = sorted(set(resources) - set(model.encoders))

    if unknown:
        msg = f"pretrained resources name absent encoder roles {unknown}"
        raise ValueError(msg)

    encoders: dict[str, _CodecEncoder] = {}

    for role, space_roles in _encoder_spaces(model).items():
        sorts = {model.spaces[space].sort for space in space_roles}

        if len(sorts) != 1:
            msg = (
                f"encoder {role!r} is bound to multiple root sorts {sorted(sorts)}; "
                "a codec-bound encoder cannot realize that declared parameter share"
            )
            raise ValueError(msg)

        sort = next(iter(sorts))
        codec = _codec(run, sort)
        context = run.with_codec(codec).for_component(ComponentId(role), {})
        context = _resource(context, role, resources)
        declaration = cast("ConfiguredSpec[object]", model.encoders[role])
        built = build_component(declaration, context)

        if not isinstance(built, _CodecEncoder):
            msg = (
                f"encoder role {role!r} built {type(built).__name__}, not a "
                "codec-bound term encoder"
            )
            raise TypeError(msg)

        encoders[role] = built

    return encoders


def _chart_merge(
    declaration: SubmoduleDeclaration,
    parser: object,
    width: int,
    /,
) -> ChartCombine:
    """Resolve the chart's declared cell-combination publication."""
    if not isinstance(parser, ModuleSharingProducer):
        msg = "chart-combine requires a parser that publishes shared submodules"
        raise TypeError(msg)

    sharing = parser.shared.get("combine")

    if sharing is None or sharing.sharing != declaration.sharing:
        msg = "the parser's combine publication does not realize the space declaration"
        raise ValueError(msg)

    composer = sharing.module

    if not isinstance(composer, CellComposer):
        msg = "the parser's combine publication cannot compose paired cells"
        raise TypeError(msg)

    return ChartCombine(composer, width)


def _merge(
    kind: Merge,
    encoder: _CodecEncoder,
    parser: object | None,
    sharing: SubmoduleDeclaration | None,
    /,
) -> PairMerge:
    """Build the declared daughter merge from published capabilities."""
    if kind is Merge.CONCATENATE:
        return Concatenate(encoder.output_width)

    if parser is None or sharing is None:
        msg = "chart-combine requires a parser and its declared combine publication"
        raise ValueError(msg)

    return _chart_merge(sharing, parser, encoder.output_width)


def _decoder_uses(
    model: ModelSection,
    encoders: Mapping[str, _CodecEncoder],
    parser: object | None,
    /,
) -> tuple[dict[str, tuple[NeuralCodecView, _WidthProducer]], dict[str, PairMerge]]:
    """Resolve each decoder role's codec and vector producer."""
    uses: dict[str, list[tuple[NeuralCodecView, _WidthProducer]]] = {}
    merges: dict[str, PairMerge] = {}

    for space_role, space in model.spaces.items():
        encoder = encoders[space.encoder]
        codec = encoder.codec
        sources = {
            cast("ConfiguredSpec[TermsReadoutSettings]", declaration).settings().reads
            for declaration in model.readouts.values()
            if declaration.space == space_role
        }

        if len(sources) > 1:
            msg = (
                f"space {space_role!r} binds identity decoder {space.identity!r} "
                f"to incompatible chart representations "
                f"{sorted(source.value for source in sources)!r}"
            )
            raise ValueError(msg)

        source = next(iter(sources), ReadoutSource.INSIDE)

        if source is ReadoutSource.SPAN:
            if not isinstance(parser, _SpanWidthProducer):
                msg = (
                    f"space {space_role!r} reads the span representation, and "
                    "the declared parser publishes no span width"
                )
                raise ValueError(msg)

            identity_producer: _WidthProducer = _ReadoutVectors(parser.span_width)
        else:
            identity_producer = encoder

        uses.setdefault(space.identity, []).append((codec, identity_producer))
        combinators = space.combinators

        if combinators is None:
            continue

        if combinators.decoders:
            kind = combinators.merge

            if kind is None:
                msg = f"space {space_role!r} binds binary decoders without a merge"
                raise ValueError(msg)

            merge = _merge(
                kind,
                encoder,
                parser,
                combinators.chart_sharing(),
            )
            merges[space_role] = merge

            for role in combinators.decoders.values():
                uses.setdefault(role, []).append((codec, cast("_WidthProducer", merge)))

    resolved: dict[str, tuple[NeuralCodecView, _WidthProducer]] = {}

    for role, arrivals in uses.items():
        codecs = {id(codec) for codec, _ in arrivals}
        widths = {producer.output_width for _, producer in arrivals}

        if len(codecs) != 1 or len(widths) != 1:
            msg = (
                f"decoder {role!r} is bound across incompatible space codecs or "
                f"input widths {sorted(widths)}"
            )
            raise ValueError(msg)

        resolved[role] = arrivals[0]

    return resolved, merges


def _build_decoders(
    model: ModelSection,
    run: BuildContext,
    uses: Mapping[str, tuple[NeuralCodecView, _WidthProducer]],
    resources: Mapping[str, PretrainedLoader],
    /,
) -> dict[str, object]:
    """Build each named decoder once from its represented-space producer."""
    decoders: dict[str, object] = {}

    for role, (codec, producer) in uses.items():
        declaration = model.decoders[role]
        sharing = declaration.sharing().get("encoder")

        if sharing is not None:
            if not isinstance(producer, ModuleSharingProducer):
                msg = (
                    f"decoder {role!r} selects publication {sharing.name!r}, "
                    "but its space producer publishes no submodules"
                )
                raise TypeError(msg)

            published = producer.shared.get(sharing.name)

            if published is None or published.sharing != sharing.sharing:
                msg = (
                    f"decoder {role!r} selects publication {sharing.name!r} "
                    "with sharing the space producer does not publish"
                )
                raise ValueError(msg)

        context = run.with_codec(codec).for_component(
            ComponentId(role), {"encoder": producer}
        )
        context = _resource(context, role, resources)
        built = build_component(cast("ConfiguredSpec[object]", declaration), context)

        protocols = {
            DecoderCapability.FRONTIER: FrontierDecoder,
            DecoderCapability.TEACHER_FORCED: TeacherForcedDecoder,
            DecoderCapability.MARGINAL: MarginalDecoder,
        }
        capabilities = declaration.capabilities_for(cast("TermCodec", codec.codec))
        missing = tuple(
            capability.name.lower()
            for capability in capabilities
            if not isinstance(built, protocols[capability])
        )
        if missing:
            msg = (
                f"decoder role {role!r} built {type(built).__name__} without its "
                f"declared capabilities {missing!r}"
            )
            raise TypeError(msg)

        decoders[role] = built

    return decoders


def _family(
    model: ModelSection,
    space_role: str,
    encoder: _CodecEncoder,
    decoders: Mapping[str, object],
    merges: Mapping[str, PairMerge],
    run: BuildContext,
    /,
) -> CombinatorFamily | None:
    """Build one space's exact action-to-decoder family."""
    declaration = model.spaces[space_role].combinators

    if declaration is None:
        return None

    by_action = {action.name: action for action in run.actions.composing}
    resolved = {}

    for action_name, decoder_role in declaration.decoders.items():
        action = by_action.get(action_name)

        if action is None:
            known = ", ".join(by_action) or "no composing action"
            msg = f"space {space_role!r} names action {action_name!r}; known: {known}"
            raise ValueError(msg)

        resolved[action] = decoders[decoder_role]

    identity = decoders[model.spaces[space_role].identity]

    if not isinstance(identity, TeacherForcedDecoder):
        msg = f"space {space_role!r} identity cannot score unary conclusions"
        raise TypeError(msg)

    return CombinatorFamily(
        resolved,
        actions=run.actions,
        rules=run.rules,
        merge=merges.get(space_role),
        codec=cast("TermCodec", encoder.codec.codec),
        encoder=encoder,
        identity=identity,
        index=run.choices,
        root_sort=model.spaces[space_role].sort,
    )


def _space_map(
    source: Space,
    target: Space,
    declaration: TypeSpaceMapSpec,
    run: BuildContext,
    /,
) -> TypeSpaceMap:
    """Build and place one declared learned map between two spaces."""
    settings = declaration.settings()
    layers = (
        None
        if settings.form is SpaceMapForm.LINEAR
        else MultilayerOptions(
            hidden=cast("int", settings.hidden),
            depth=cast("int", settings.depth),
            dropout=cast("float", settings.dropout),
        )
    )
    built = TypeSpaceMap(
        source_width=source.encoder.output_width,
        target_width=target.encoder.output_width,
        form=MapForm(settings.form.value),
        layers=layers,
    )

    return place(built, run.policy)


def _relations(
    model: ModelSection,
    spaces: Mapping[str, Space],
    run: BuildContext,
    erasure: TermMorphism | None,
    /,
) -> dict[str, SpaceRelationBinding]:
    """Build each declared cross-space relation exactly once."""
    built: dict[str, SpaceRelationBinding] = {}

    for role, space in model.spaces.items():
        declaration = space.relation

        if declaration is None:
            continue

        source = spaces[role]
        target = spaces[declaration.space]
        mapping_declaration = declaration.mapping

        if declaration.via is SpaceRelation.MAPPED:
            if mapping_declaration is None:
                msg = f"mapped space relation {role!r} has no map declaration"
                raise ValueError(msg)

            mapping = _space_map(source, target, mapping_declaration, run)
        else:
            mapping = None
        built[role] = SpaceRelationBinding(
            source,
            target,
            kind=RelationKind(declaration.via.value),
            mapping=mapping,
            erasure=(None if declaration.via is SpaceRelation.MAPPED else erasure),
        )

    return built


def assemble_space_graph(
    model: ModelSection,
    run: BuildContext,
    encoders: Mapping[str, _CodecEncoder],
    /,
    *,
    parser: object | None,
    pretrained_resources: Mapping[str, PretrainedLoader],
    role_targets: Mapping[str, Mapping[str, TargetSet[Term]]] | None = None,
    erasure: TermMorphism | None,
) -> SpaceGraph:
    """Build decoders, spaces and readouts from their declared encoders."""
    if set(encoders) != set(model.encoders):
        missing = sorted(set(model.encoders) - set(encoders))
        extra = sorted(set(encoders) - set(model.encoders))
        msg = (
            "built space encoders differ from declarations; "
            f"missing={missing}, extra={extra}"
        )
        raise ValueError(msg)

    extra = sorted(set(pretrained_resources) - set(model.decoders))

    if extra:
        msg = f"pretrained resources name absent decoder roles {extra}"
        raise ValueError(msg)

    uses, merges = _decoder_uses(model, encoders, parser)
    decoders = _build_decoders(model, run, uses, pretrained_resources)
    spaces: dict[str, Space] = {}

    for role, declaration in model.spaces.items():
        encoder = encoders[declaration.encoder]
        spaces[role] = Space(
            declaration.sort,
            codec=encoder.codec,
            choices=run.choices,
            store=run.language.store,
            max_nodes=encoder.codec.capacity,
            encoder=encoder,
            identity=decoders[declaration.identity],
            combinators=_family(model, role, encoder, decoders, merges, run),
        )

    relations = _relations(model, spaces, run, erasure)

    targets = {} if role_targets is None else role_targets
    unknown_targets = sorted(set(targets) - set(model.readouts))

    if unknown_targets:
        msg = f"target sets name absent readout roles {unknown_targets}"
        raise ValueError(msg)

    readouts: dict[str, TermsReadout] = {}

    for role, declaration in model.readouts.items():
        configured = cast("ConfiguredSpec[TermsReadoutSettings]", declaration)
        settings = configured.settings()
        space_role = settings.space
        space = spaces[space_role]
        context = run.with_codec(space.codec).for_component(
            ComponentId(role), {"identity": space.identity, "space": space}
        )
        scoped_targets = dict(targets.get(role, {}))
        fixed = settings.targets

        if fixed is not None and str(fixed) in run.targets:
            source = str(fixed)
            declared = run.targets[source]
            injected = scoped_targets.get(source)

            if injected is not None and injected != declared:
                msg = f"readout {role!r} receives conflicting target source {source!r}"
                raise ValueError(msg)

            scoped_targets[source] = declared

        context = context.with_targets(scoped_targets)

        built = build_component(configured, context)
        readouts[role] = cast("TermsReadout", built)

    components: dict[str, object] = {
        **encoders,
        **decoders,
        **{
            role: relation
            for role, relation in relations.items()
            if relation.mapping is not None
        },
        **readouts,
    }

    return SpaceGraph(
        components=components,
        spaces=spaces,
        relations=relations,
        readouts=readouts,
    )


def wire_spaces(
    model: ModelSection,
    run: BuildContext,
    /,
    *,
    parser: object | None,
    pretrained_resources: Mapping[str, PretrainedLoader],
    role_targets: Mapping[str, Mapping[str, TargetSet[Term]]] | None = None,
    erasure: TermMorphism | None,
) -> SpaceGraph:
    """Build the complete represented-space graph in declaration order."""
    encoder_resources = {
        role: resource
        for role, resource in pretrained_resources.items()
        if role in model.encoders
    }
    decoder_resources = {
        role: resource
        for role, resource in pretrained_resources.items()
        if role in model.decoders
    }
    known = set(encoder_resources) | set(decoder_resources)
    extra = sorted(set(pretrained_resources) - known)

    if extra:
        msg = f"pretrained resources name term roles that load no weights {extra}"
        raise ValueError(msg)

    encoders = build_space_encoders(model, run, encoder_resources)

    return assemble_space_graph(
        model,
        run,
        encoders,
        parser=parser,
        pretrained_resources=decoder_resources,
        role_targets=role_targets,
        erasure=erasure,
    )

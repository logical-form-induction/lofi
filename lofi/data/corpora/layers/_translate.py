"""Translation from Layers records to data-layer values."""

from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING, Final, NamedTuple, override

from lairs.records import annotation as annotation_records
from lairs.records import defs as defs_records
from lairs.records import graph as graph_records

from lofi._types import ItemId
from lofi.data import (
    RESPONSE_MISSING,
    AnnotatorId,
    ByteSpan,
    Categorical,
    Dependency,
    Derivation,
    DerivationNode,
    GoldDependencies,
    GoldDerivation,
    GoldGraph,
    GraphEdge,
    GraphNode,
    LabelMembership,
    LicenseRef,
    Licensing,
    NodeId,
    Provenance,
    Relational,
    RelationalScalar,
    Reproducibility,
    Scalar,
    Scale,
    ScaleKind,
    SpanConstraints,
    Supervision,
    Token,
    TokenSpan,
    Unlabeled,
)
from lofi.lf import RELATION_LABELS

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from lairs.records import judgment as judgment_records
    from lairs.records import segmentation as segmentation_records

    from lofi.data import SupervisionArm

__all__ = [
    "EMITTED_ARMS",
    "GRAPH_LIFT",
    "LAYER_LIFTS",
    "RELATION_LIFT",
    "RELATION_NAMES",
    "RESPONSE_LIFTS",
    "GraphContext",
    "GraphLift",
    "LayerContext",
    "LayerLift",
    "LayersSkip",
    "NodeSource",
    "RecordSkippedError",
    "RelationLift",
    "RelationPair",
    "RelationReading",
    "ResponseContext",
    "ResponseLift",
    "TokenReading",
    "TreeLift",
    "annotator_of",
    "dependencies_of",
    "derivation_of",
    "features_of",
    "graph_of",
    "graph_start",
    "item_id_of",
    "lift_tree",
    "lower_tree",
    "pair_response_of",
    "provenance_of",
    "relation_layer_pair",
    "relation_pair_of",
    "relational_edge_of",
    "relational_layer_of",
    "response_of",
    "scale_of",
    "span_constraints_of",
    "supervision_of",
    "tokenization_of",
    "tokens_of",
]

_CONFIDENCE_CEILING: Final = 1000
"""The maximum confidence stored by Layers."""

_TREE_KIND: Final = "tree"
"""The layer kind for rooted analyses over token positions."""

_RELATION_KIND: Final = "relation"
"""The layer kind for labeled token relations."""

RELATION_NAMES: Final = frozenset(label.value for label in RELATION_LABELS)
"""The relation names shared by every pair encoding."""

_HYPOTHESIS_ROLE: Final = "hypothesis"
"""The argument role naming a relation pair's hypothesis."""

_DEPENDENCY_SUBKIND: Final = "dependency"
"""The relation subkind for dependency arcs."""

_TAG_KIND: Final = "token-tag"
"""The layer kind for token-label candidates."""

_WIRE_ROOT: Final = -1
"""The wire value for a dependency root."""

_ORDINAL_TASKS: Final = ("ordinal-scale", "binary")
"""The ordinal response instruments."""

_CONTINUOUS_TASK: Final = "continuous"
"""The continuous response instrument."""

_RATIO_TASK: Final = "magnitude"
"""The ratio-scale response instrument."""

_CATEGORICAL_TASK: Final = "categorical"
"""The categorical response instrument."""

_ENTAILMENT_RELATION: Final = "entails"
"""The relation whose probability a continuous inference judgment rates."""

_MODULUS_KEY: Final = "modulus"
"""The experiment feature for a ratio-scale modulus."""


def _exact_decimal(coefficient: int, exponent: int) -> Decimal:
    """Reconstruct one exact decimal without applying a numeric context."""
    return Decimal(f"{coefficient}e{exponent}")


class LayersSkip(StrEnum):
    """Why the reader set a Layers record aside, as the load report keys it."""

    #: A membership names an expression the corpus does not hold.
    UNHELD_EXPRESSION = "unheld-expression"

    #: A membership names no split.
    UNSPLIT = "unsplit"

    #: An annotation layer's nodes could not be read.
    UNREADABLE_LAYER = "unreadable-layer"

    #: A graph's nodes and edges could not be read.
    UNREADABLE_GRAPH = "unreadable-graph"

    #: A response names an experiment the corpus does not hold.
    UNHELD_EXPERIMENT = "unheld-experiment"

    #: An experiment's instrument puts its responses on no scale.
    UNSCALED = "unscaled"


class RecordSkippedError(Exception):
    """An error for an unreadable Layers record.

    Parameters
    ----------
    reason : str
        The load-report reason for skipping the record.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)

        #: Why the record was set aside, as the load report keys it.
        self.reason = reason


def features_of(features: defs_records.FeatureMap | None) -> Mapping[str, str]:
    """Return a Layers feature map as a plain mapping.

    Parameters
    ----------
    features : defs_records.FeatureMap | None
        The feature map, or None if the record has no features.

    Returns
    -------
    Mapping[str, str]
        The entries keyed by feature name. Later duplicate keys take precedence.
    """
    if features is None:
        return {}

    return {entry.key: entry.value for entry in features.entries}


def _node_features(features: defs_records.FeatureMap | None) -> Mapping[str, str]:
    """Return a tree node's feature map.

    Parameters
    ----------
    features : defs_records.FeatureMap | None
        The feature map, or None if the annotation has no features.

    Returns
    -------
    Mapping[str, str]
        The entries, keyed by their own keys.

    Raises
    ------
    RecordSkippedError
        If the map states one key twice.
    """
    read = features_of(features)

    if features is not None and len(read) != len(features.entries):
        reason = "a tree layer whose node states one feature key twice"

        raise RecordSkippedError(reason)

    return read


def _licensing_of(licensing: defs_records.Licensing | None) -> Licensing | None:
    """Return the terms a Layers licensing record states.

    Parameters
    ----------
    licensing : defs_records.Licensing | None
        The Layers licensing record, or None.

    Returns
    -------
    Licensing | None
        The terms, or None if the record states no SPDX expression.
    """
    if licensing is None or not licensing.expression:
        return None

    return Licensing(
        licensing.expression,
        tuple(
            LicenseRef(entry.spdx, entry.appliesTo)
            for entry in licensing.licenses
            if entry.spdx
        ),
    )


def _reproducibility_of(
    reproducibility: defs_records.ReproducibilityInfo | None,
) -> Reproducibility | None:
    """Return artifact reproducibility metadata.

    Parameters
    ----------
    reproducibility : defs_records.ReproducibilityInfo | None
        The Layers reproducibility record, or None.

    Returns
    -------
    Reproducibility | None
        The metadata, or None if the record has none.
    """
    if reproducibility is None:
        return None

    return Reproducibility(
        reproducibility.codeUri,
        reproducibility.commitHash,
        reproducibility.command,
        reproducibility.environment,
        reproducibility.randomSeed,
    )


def provenance_of(
    source_method: str | None,
    licensing: defs_records.Licensing | None,
    eprint_refs: tuple[str, ...] | None,
    reproducibility: defs_records.ReproducibilityInfo | None,
) -> Provenance | None:
    """Return annotation provenance.

    Parameters
    ----------
    source_method : str | None
        How the annotation was arrived at, in the corpus's own vocabulary.
    licensing : defs_records.Licensing | None
        The terms the annotation is released under.
    eprint_refs : tuple[str, ...] | None
        References to the publications describing the annotation's source.
    reproducibility : defs_records.ReproducibilityInfo | None
        How the annotation was computed, where it was computed.

    Returns
    -------
    Provenance | None
        The record when both a source method and SPDX licensing are present;
        otherwise, None.
    """
    if not source_method:
        return None

    terms = _licensing_of(licensing)

    if terms is None:
        return None

    return Provenance(
        source_method,
        terms,
        tuple(eprint_refs or ()),
        _reproducibility_of(reproducibility),
    )


def _read_tokenization(
    segmentations: Sequence[segmentation_records.Segmentation],
) -> segmentation_records.Tokenization | None:
    """Return the tokenization an item's tokens are read from.

    Parameters
    ----------
    segmentations : Sequence[segmentation_records.Segmentation]
        The segmentations targeting the expression, in pool order.

    Returns
    -------
    segmentation_records.Tokenization | None
        The tokenization, or None for an unsegmented expression.
    """
    for segmentation in segmentations:
        for tokenization in segmentation.tokenizations:
            return tokenization

    return None


def tokenization_of(
    segmentations: Sequence[segmentation_records.Segmentation],
) -> str | None:
    """Return the identifier of the tokenization an item's tokens come from.

    Parameters
    ----------
    segmentations : Sequence[segmentation_records.Segmentation]
        The segmentations targeting the expression, in pool order.

    Returns
    -------
    str | None
        The identifier, or None for an unsegmented expression.
    """
    read = _read_tokenization(segmentations)

    return None if read is None else read.uuid.value


def tokens_of(
    text: str | None,
    segmentations: Sequence[segmentation_records.Segmentation],
) -> tuple[Token, ...]:
    """Return the tokens of one expression, in position order.

    Parameters
    ----------
    text : str | None
        The expression's text, which the byte spans are offsets into. It takes
        no part in the tokens and is named only in the message.
    segmentations : Sequence[segmentation_records.Segmentation]
        The segmentations targeting the expression, in pool order.

    Returns
    -------
    tuple[Token, ...]
        The tokens, empty for an expression the corpus has not segmented.

    Raises
    ------
    RecordSkippedError
        If a tokenization's positions are not zero up to one less than its
        token count.
    """
    read = _read_tokenization(segmentations)

    if read is None:
        return ()

    ordered = sorted(read.tokens, key=lambda token: token.tokenIndex)

    for position, token in enumerate(ordered):
        if token.tokenIndex != position:
            reason = "a tokenization whose positions are not contiguous from zero"
            raise RecordSkippedError(reason)

    if text is None and any(token.textSpan is not None for token in ordered):
        reason = "a tokenization with byte spans into an expression holding no text"
        raise RecordSkippedError(reason)

    return tuple(
        Token(
            token.tokenIndex,
            token.text or "",
            None
            if token.textSpan is None
            else ByteSpan(token.textSpan.byteStart, token.textSpan.byteEnd),
        )
        for token in ordered
    )


def _share_of(confidence: int) -> float:
    """Return a wire confidence as a share of one.

    Parameters
    ----------
    confidence : int
        The number the record states.

    Returns
    -------
    float
        The share of one.

    Raises
    ------
    RecordSkippedError
        If the number is outside the range the wire form states.
    """
    if confidence < 0 or confidence > _CONFIDENCE_CEILING:
        reason = "an annotation whose confidence is outside the range the wire states"

        raise RecordSkippedError(reason)

    return confidence / _CONFIDENCE_CEILING


def _wire_share(share: float) -> int:
    """Return a share of one as the whole number the wire form states.

    Parameters
    ----------
    share : float
        The share of one.

    Returns
    -------
    int
        The number nearest the share times the ceiling, which is the number
        ``_share_of`` was given.
    """
    return round(share * _CONFIDENCE_CEILING)


class NodeSource:
    """The source of a lifted tree node.

    Parameters
    ----------
    uuid : str
        The identifier the annotation was recorded under.
    type_ref : str | None
        The ontology type the annotation named, absent where it named none.
    position : int | None
        The token position the annotation anchored, absent where it anchored
        none.
    tokenization : str | None
        The tokenization a token reference named, absent where the annotation
        gave a bare position or had no anchor.
    """

    __slots__ = ("_position", "_tokenization", "_type_ref", "_uuid")

    def __init__(
        self,
        uuid: str,
        type_ref: str | None = None,
        position: int | None = None,
        tokenization: str | None = None,
    ) -> None:
        self._uuid = uuid
        self._type_ref = type_ref
        self._position = position
        self._tokenization = tokenization

    @property
    def uuid(self) -> str:
        """Return the identifier the annotation was recorded under."""
        return self._uuid

    @property
    def type_ref(self) -> str | None:
        """Return the ontology type the annotation named, if it named one."""
        return self._type_ref

    @property
    def position(self) -> int | None:
        """Return the token position the annotation anchored, if it anchored one."""
        return self._position

    @property
    def tokenization(self) -> str | None:
        """Return the tokenization a token reference named, if one was given."""
        return self._tokenization

    def _identity(self) -> tuple[str, str | None, int | None, str | None]:
        """Return the four fields a source is equal on."""
        return (self._uuid, self._type_ref, self._position, self._tokenization)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value records the same origin."""
        if type(other) is not NodeSource:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the identifier, the type, the position and the id."""
        return hash((NodeSource, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this source."""
        return f"NodeSource{self._identity()!r}"


class TreeLift:
    """A lifted tree and its node sources.

    Parameters
    ----------
    derivation : Derivation[str]
        The analysis the layer records.
    sources : tuple[NodeSource, ...]
        Where each node came from, in node id order.
    """

    __slots__ = ("_derivation", "_sources")

    def __init__(
        self,
        derivation: Derivation[str],
        sources: tuple[NodeSource, ...],
    ) -> None:
        self._derivation = derivation
        self._sources = sources

    @property
    def derivation(self) -> Derivation[str]:
        """Return the analysis the layer records."""
        return self._derivation

    @property
    def sources(self) -> tuple[NodeSource, ...]:
        """Return where each node came from, in node id order."""
        return self._sources

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is the same analysis from the same layer."""
        if type(other) is not TreeLift:
            return NotImplemented

        return self._derivation == other._derivation and self._sources == other._sources

    @override
    def __hash__(self) -> int:
        """Return a hash over the analysis and the origins."""
        return hash((TreeLift, self._derivation, self._sources))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this lift."""
        return f"TreeLift({self._derivation!r}, {self._sources!r})"


def _source_of(annotation: annotation_records.Annotation) -> NodeSource:
    """Return the source named by an annotation anchor.

    Parameters
    ----------
    annotation : annotation_records.Annotation
        The annotation to place.

    Returns
    -------
    NodeSource
        Its identifier, the ontology type it named and where it attaches.

    Raises
    ------
    RecordSkippedError
        If the annotation sets two anchor variants at once, if the one it sets
        is not a reference to a token, or if it holds an anchor setting no
        variant at all.
    """
    anchor = annotation.anchor
    positions: list[tuple[int, str | None]] = []
    unread: tuple[str, ...] = ()

    if annotation.tokenIndex is not None:
        positions.append((annotation.tokenIndex, None))

    if anchor is not None:
        if anchor.tokenRef is not None:
            positions.append(
                (anchor.tokenRef.tokenIndex, anchor.tokenRef.tokenizationId.value)
            )

        unread = tuple(
            name
            for name, variant in (
                ("boundingBox", anchor.boundingBox),
                ("externalTarget", anchor.externalTarget),
                ("pageAnchor", anchor.pageAnchor),
                ("signalSpan", anchor.signalSpan),
                ("spatialRegion", anchor.spatialRegion),
                ("spatioTemporalAnchor", anchor.spatioTemporalAnchor),
                ("temporalSpan", anchor.temporalSpan),
                ("textSpan", anchor.textSpan),
                ("tokenRefSequence", anchor.tokenRefSequence),
            )
            if variant is not None
        )

        if anchor.tokenRef is None and not unread:
            reason = "an annotation whose anchor sets no variant"

            raise RecordSkippedError(reason)

    if len(positions) + len(unread) > 1:
        reason = "an annotation setting two anchor variants at once"

        raise RecordSkippedError(reason)

    if unread:
        reason = f"an annotation anchored by a {unread[0]}, which is no token"

        raise RecordSkippedError(reason)

    position, tokenization = positions[0] if positions else (None, None)

    return NodeSource(
        annotation.uuid.value, annotation.ontologyTypeRef, position, tokenization
    )


def _resolved(reference: str, labels: LabelMembership[str] | None) -> str:
    """Return the category an ontology type reference names.

    Parameters
    ----------
    reference : str
        The ontology type reference.
    labels : LabelMembership[str] | None
        The category labels the run admits.

    Returns
    -------
    str
        The declared operator's name.

    Raises
    ------
    RecordSkippedError
        If the reference names no type, if the run declares no label inventory,
        or if the inventory does not admit the type.
    """
    _, marked, fragment = reference.rpartition("#")
    name = fragment if marked else reference.rpartition("/")[2]

    if not name:
        reason = "an ontology type reference naming no type"

        raise RecordSkippedError(reason)

    if labels is None:
        reason = (
            "an ontology type reference under a run that declares no label inventory"
        )

        raise RecordSkippedError(reason)

    if not labels.contains(name):
        reason = "an ontology type the run's label inventory does not admit"

        raise RecordSkippedError(reason)

    return name


def _labeled(
    annotation: annotation_records.Annotation,
    labels: LabelMembership[str] | None,
) -> tuple[str, str | None]:
    """Return a node's category and the rule that licensed it.

    Parameters
    ----------
    annotation : annotation_records.Annotation
        The annotation to read.
    labels : LabelMembership[str] | None
        The inventory used to resolve ontology types, if available.

    Returns
    -------
    tuple[str, str | None]
        The category and optional rule name.

    Raises
    ------
    RecordSkippedError
        If no category is available or an ontology type cannot be resolved.
    """
    reference = annotation.ontologyTypeRef

    if reference is None:
        if annotation.label is None:
            reason = "a tree layer whose node has no label"

            raise RecordSkippedError(reason)

        return annotation.label, None

    return _resolved(reference, labels), annotation.label


def _refuse_parent_cycle(parents: Mapping[NodeId, NodeId | None]) -> None:
    """Raise if following parent references from any node returns to that node.

    Parameters
    ----------
    parents : Mapping[NodeId, NodeId | None]
        Each node's parent, by id, absent for a root.

    Raises
    ------
    RecordSkippedError
        If the parent references form a cycle.
    """
    settled: set[NodeId] = set()

    for start in parents:
        chain: list[NodeId] = []
        seen: set[NodeId] = set()
        walked: NodeId | None = start

        while walked is not None and walked not in settled:
            if walked in seen:
                reason = "a tree layer whose nodes form a cycle"

                raise RecordSkippedError(reason)

            seen.add(walked)
            chain.append(walked)
            walked = parents[walked]

        settled.update(chain)


def _post_order(
    root: NodeId,
    children: Mapping[NodeId, tuple[NodeId, ...]],
) -> list[NodeId]:
    """Return the nodes below a root, every child before its parent.

    Parameters
    ----------
    root : NodeId
        The node to walk from.
    children : Mapping[NodeId, tuple[NodeId, ...]]
        Each node's children, by id.

    Returns
    -------
    list[NodeId]
        The nodes reached, in post order.

    Raises
    ------
    RecordSkippedError
        If the child references form a cycle.
    """
    order: list[NodeId] = []
    on_path: set[NodeId] = set()
    stack: list[tuple[NodeId, bool]] = [(root, False)]

    while stack:
        node_id, walked = stack.pop()

        if walked:
            on_path.discard(node_id)
            order.append(node_id)

            continue

        if node_id in on_path:
            reason = "a tree layer whose nodes form a cycle"

            raise RecordSkippedError(reason)

        on_path.add(node_id)
        stack.append((node_id, True))
        stack.extend((child, False) for child in reversed(children[node_id]))

    return order


def _spanned(
    order: list[NodeId],
    children: Mapping[NodeId, tuple[NodeId, ...]],
    leaves: Mapping[NodeId, int],
) -> dict[NodeId, TokenSpan]:
    """Return the token positions each node covers.

    Parameters
    ----------
    order : list[NodeId]
        The nodes in post order, so every child is spanned before its parent.
    children : Mapping[NodeId, tuple[NodeId, ...]]
        Each node's children, by id.
    leaves : Mapping[NodeId, int]
        The token position each anchored node names, by id.

    Returns
    -------
    dict[NodeId, TokenSpan]
        Each node's extent, by id.

    Raises
    ------
    RecordSkippedError
        If a node with no children anchors no token position.
    """
    spans: dict[NodeId, TokenSpan] = {}

    for node_id in order:
        below = children[node_id]

        if below:
            covered = [spans[child] for child in below]
            spans[node_id] = TokenSpan(
                min(span.start for span in covered),
                max(span.end for span in covered),
            )
        else:
            position = leaves.get(node_id)

            if position is None:
                reason = "a tree layer whose leaf anchors no token"

                raise RecordSkippedError(reason)

            spans[node_id] = TokenSpan(position, position + 1)

    return spans


class _Skeleton(NamedTuple):
    """Decoded tree edges, anchors, and sources."""

    children: dict[NodeId, tuple[NodeId, ...]]
    parents: dict[NodeId, NodeId | None]
    leaves: dict[NodeId, int]
    sources: tuple[NodeSource, ...]
    roots: tuple[NodeId, ...]


def _identified(layer: annotation_records.AnnotationLayer) -> dict[str, NodeId]:
    """Return the node id each annotation identifier stands for.

    Parameters
    ----------
    layer : annotation_records.AnnotationLayer
        The layer to read.

    Returns
    -------
    dict[str, NodeId]
        The declaration position of each identifier.

    Raises
    ------
    RecordSkippedError
        If one identifier is recorded twice, since a reference to it would then
        name two nodes.
    """
    identifiers: dict[str, NodeId] = {}

    for position, annotation in enumerate(layer.annotations):
        if annotation.uuid.value in identifiers:
            reason = "a tree layer recording one identifier twice"

            raise RecordSkippedError(reason)

        identifiers[annotation.uuid.value] = position

    return identifiers


def _skeleton(
    layer: annotation_records.AnnotationLayer,
    identifiers: Mapping[str, NodeId],
) -> _Skeleton:
    """Return a layer's tree edges, anchors, and origins.

    Parameters
    ----------
    layer : annotation_records.AnnotationLayer
        The layer to read.
    identifiers : Mapping[str, NodeId]
        The node id each identifier stands for.

    Returns
    -------
    _Skeleton
        The children and parent of each node, the position each anchored node
        names, one source per node and the nodes naming no parent.

    Raises
    ------
    RecordSkippedError
        If a reference names a node the layer does not hold, if one node is
        named as a child twice, if a node names a parent other than the node
        naming it as a child, or if an annotation sets two anchor variants at
        once.
    """
    children: dict[NodeId, tuple[NodeId, ...]] = {}
    parents: dict[NodeId, NodeId | None] = {}
    leaves: dict[NodeId, int] = {}
    sources: list[NodeSource] = []
    roots: list[NodeId] = []
    claimed: dict[NodeId, NodeId] = {}

    for node_id, annotation in enumerate(layer.annotations):
        named = tuple(child.value for child in annotation.childIds or ())
        parent = None if annotation.parentId is None else annotation.parentId.value

        if any(name not in identifiers for name in named) or (
            parent is not None and parent not in identifiers
        ):
            reason = "a tree layer naming a node it does not hold"

            raise RecordSkippedError(reason)

        below = tuple(identifiers[name] for name in named)

        for child in below:
            if child in claimed:
                reason = "a tree layer naming one node as a child twice"

                raise RecordSkippedError(reason)

            claimed[child] = node_id

        children[node_id] = below
        parents[node_id] = None if parent is None else identifiers[parent]
        source = _source_of(annotation)
        sources.append(source)

        if source.position is not None:
            leaves[node_id] = source.position

        if parent is None:
            roots.append(node_id)

    for child, claimer in claimed.items():
        declared = parents[child]

        if declared is not None and declared != claimer:
            reason = "a tree layer whose parent and child references disagree"

            raise RecordSkippedError(reason)

    return _Skeleton(children, parents, leaves, tuple(sources), tuple(roots))


def lift_tree(
    layer: annotation_records.AnnotationLayer,
    labels: LabelMembership[str] | None = None,
) -> TreeLift:
    """Return a tree derivation and its node sources.

    Parameters
    ----------
    layer : annotation_records.AnnotationLayer
        Tree annotation layer.
    labels : LabelMembership[str] | None
        Inventory used to resolve ontology type references.

    Returns
    -------
    TreeLift
        Derivation and node sources in declaration order.

    Raises
    ------
    RecordSkippedError
        If identifiers, parent links, roots, anchors, reachability, features, or
        labels do not form a valid tree layer.
    ValueError
        If the translated nodes do not form a derivation.
    """
    identifiers = _identified(layer)
    skeleton = _skeleton(layer, identifiers)

    _refuse_parent_cycle(skeleton.parents)

    if len(skeleton.roots) != 1:
        reason = "a tree layer holding other than one root"

        raise RecordSkippedError(reason)

    root = skeleton.roots[0]
    order = _post_order(root, skeleton.children)
    spans = _spanned(order, skeleton.children, skeleton.leaves)

    if len(spans) < len(layer.annotations):
        reason = "a tree layer holding a node its root does not reach"

        raise RecordSkippedError(reason)

    nodes: dict[NodeId, DerivationNode[str]] = {}
    confidences: dict[NodeId, float] = {}

    for node_id, annotation in enumerate(layer.annotations):
        label, rule = _labeled(annotation, labels)
        nodes[node_id] = DerivationNode(
            spans[node_id],
            label,
            rule=rule,
            children=skeleton.children[node_id],
            features=_node_features(annotation.features),
        )

        if annotation.confidence is not None:
            confidences[node_id] = _share_of(annotation.confidence)

    return TreeLift(Derivation(root, nodes, confidences), skeleton.sources)


def lower_tree(lift: TreeLift) -> tuple[annotation_records.Annotation, ...]:
    """Return the annotations a lifted tree layer was read from.

    Parameters
    ----------
    lift : TreeLift
        The analysis and its sources.

    Returns
    -------
    tuple[annotation_records.Annotation, ...]
        The annotations, in node id order.
    """
    derivation = lift.derivation
    parents: dict[NodeId, NodeId] = {
        child: node_id
        for node_id, node in derivation.nodes.items()
        for child in node.children
    }
    written: list[annotation_records.Annotation] = []

    for node_id in sorted(derivation.nodes):
        node = derivation.node(node_id)
        source = lift.sources[node_id]
        parent = parents.get(node_id)
        confidence = derivation.confidence_of(node_id)
        entries = tuple(
            defs_records.Feature(key=key, value=value)
            for key, value in node.features.items()
        )
        reference = source.tokenization

        written.append(
            annotation_records.Annotation(
                uuid=defs_records.Uuid(value=source.uuid),
                label=node.label if source.type_ref is None else node.rule,
                ontologyTypeRef=source.type_ref,
                parentId=None
                if parent is None
                else defs_records.Uuid(value=lift.sources[parent].uuid),
                childIds=tuple(
                    defs_records.Uuid(value=lift.sources[child].uuid)
                    for child in node.children
                ),
                anchor=None
                if reference is None or source.position is None
                else defs_records.Anchor(
                    tokenRef=defs_records.TokenRef(
                        tokenIndex=source.position,
                        tokenizationId=defs_records.Uuid(value=reference),
                    )
                ),
                tokenIndex=None if reference is not None else source.position,
                features=None
                if not entries
                else defs_records.FeatureMap(entries=entries),
                confidence=None if confidence is None else _wire_share(confidence),
            )
        )

    return tuple(written)


def derivation_of(
    layer: annotation_records.AnnotationLayer,
    labels: LabelMembership[str] | None = None,
) -> GoldDerivation[str]:
    """Return the analysis arm a tree layer stands for.

    Parameters
    ----------
    layer : annotation_records.AnnotationLayer
        The layer, of the tree kind.
    labels : LabelMembership[str] | None
        The category labels the run admits.

    Returns
    -------
    GoldDerivation[str]
        The arm holding the layer's analysis.

    Raises
    ------
    RecordSkippedError
        If the layer is not a rooted analysis over token positions.
    """
    return GoldDerivation(lift_tree(layer, labels).derivation)


def dependencies_of(layer: annotation_records.AnnotationLayer) -> GoldDependencies:
    """Return the relation arm a dependency layer stands for.

    Parameters
    ----------
    layer : annotation_records.AnnotationLayer
        The layer, of the relation kind and the dependency subkind.

    Returns
    -------
    GoldDependencies
        The arm holding one relation per annotation, in the layer's order.

    Raises
    ------
    RecordSkippedError
        If an annotation names no head, no dependent or no label, or if it
        names a head position that is neither a position nor the root.
    """
    relations: list[Dependency] = []

    for annotation in layer.annotations:
        head = annotation.headIndex
        dependent = annotation.targetIndex

        if head is None or dependent is None or not annotation.label:
            reason = "a relation layer whose arc names no head, dependent or label"
            raise RecordSkippedError(reason)

        if head < _WIRE_ROOT:
            reason = "a relation layer whose arc governs from no position"
            raise RecordSkippedError(reason)

        relations.append(
            Dependency(
                dependent if head == _WIRE_ROOT else head,
                dependent,
                annotation.label,
            )
        )

    return GoldDependencies(tuple(relations))


class RelationPair(NamedTuple):
    """The premise and hypothesis of a directed relation."""

    premise: ItemId
    hypothesis: ItemId


class RelationReading(NamedTuple):
    """A directed pair with its relational supervision."""

    pair: RelationPair
    supervision: Relational


class RelationLift(NamedTuple):
    """The two decoders of relational supervision."""

    arm: SupervisionArm
    layer: Callable[
        [annotation_records.AnnotationLayer, annotation_records.Annotation],
        RelationReading | None,
    ]
    edge: Callable[[graph_records.GraphEdge], RelationReading]


def _metadata_annotator(
    metadata: defs_records.AnnotationMetadata | None,
) -> AnnotatorId | None:
    """Return the identifier of a Layers annotation's agent."""
    agent = None if metadata is None else metadata.agent

    return annotator_of(None if agent is None else agent.id)


def _relation_name(relation: str | None) -> str:
    """Return a relation from the canonical inference vocabulary."""
    if relation not in RELATION_NAMES:
        reason = "a relation outside the canonical inference vocabulary"

        raise RecordSkippedError(reason)

    return relation


def relation_pair_of(
    layer: annotation_records.AnnotationLayer,
    annotation: annotation_records.Annotation,
) -> RelationPair:
    """Return the directed expression pair one relation annotation names.

    Raises
    ------
    RecordSkippedError
        If the layer or annotation does not encode one hypothesis pair.
    """
    if layer.kind != _RELATION_KIND or layer.subkind is not None:
        reason = "an elicited relation naming no relation pair layer"

        raise RecordSkippedError(reason)

    arguments = tuple(annotation.arguments or ())

    if len(arguments) != 1 or arguments[0].role != _HYPOTHESIS_ROLE:
        reason = "a relation annotation without exactly one hypothesis argument"

        raise RecordSkippedError(reason)

    target = arguments[0].target

    if (
        target.recordRef is None
        or target.localId is not None
        or target.objectId is not None
        or target.knowledgeRef is not None
    ):
        reason = "a relation annotation whose hypothesis names no expression record"

        raise RecordSkippedError(reason)

    return RelationPair(ItemId(layer.expression), ItemId(target.recordRef))


def relation_layer_pair(
    layer: annotation_records.AnnotationLayer,
) -> RelationPair | None:
    """Return the one directed pair a relation layer's annotations name.

    Returns
    -------
    RelationPair | None
        The pair, or None for a layer that is not a relation-pair layer.

    Raises
    ------
    RecordSkippedError
        If a relation-pair layer's annotations name no pair or several.
    """
    if layer.kind != _RELATION_KIND or layer.subkind is not None:
        return None

    pairs = frozenset(
        relation_pair_of(layer, annotation) for annotation in layer.annotations
    )

    if len(pairs) != 1:
        reason = "a relation-pair layer whose annotations name no single pair"

        raise RecordSkippedError(reason)

    (pair,) = pairs

    return pair


def relational_layer_of(
    layer: annotation_records.AnnotationLayer,
    annotation: annotation_records.Annotation,
) -> RelationReading | None:
    """Return the annotated relation, or None for an elicitation carrier."""
    pair = relation_pair_of(layer, annotation)

    if annotation.label is None:
        return None

    return RelationReading(
        pair,
        Relational(
            pair.hypothesis,
            _relation_name(annotation.label),
            _metadata_annotator(layer.metadata),
        ),
    )


def relational_edge_of(edge: graph_records.GraphEdge) -> RelationReading:
    """Return the directed relation one expression-to-expression edge states.

    Raises
    ------
    RecordSkippedError
        If an endpoint is not an expression record or the relation is unsupported.
    """
    source = edge.source
    target = edge.target

    if (
        source.recordRef is None
        or target.recordRef is None
        or source.localId is not None
        or source.objectId is not None
        or source.knowledgeRef is not None
        or target.localId is not None
        or target.objectId is not None
        or target.knowledgeRef is not None
    ):
        reason = "a relation graph edge whose endpoint names no expression record"

        raise RecordSkippedError(reason)

    pair = RelationPair(ItemId(source.recordRef), ItemId(target.recordRef))
    supervision = Relational(
        pair.hypothesis,
        _relation_name(edge.edgeType),
        _metadata_annotator(edge.metadata),
    )

    return RelationReading(pair, supervision)


def pair_response_of(
    judgment: judgment_records.Judgment,
    experiment: judgment_records.ExperimentDef,
    pair: RelationPair,
    annotator: AnnotatorId | None,
) -> Supervision[str]:
    """Return the response an inference judgment assigns to one pair.

    Raises
    ------
    RecordSkippedError
        If the experiment or judgment does not encode a supported inference
        response.
    """
    if experiment.measureType != "inference":
        reason = "an elicited pair response whose experiment is not inference"

        raise RecordSkippedError(reason)

    if experiment.taskType == _CATEGORICAL_TASK:
        if judgment.scalarValue is not None or judgment.continuousValue is not None:
            reason = "a categorical pair response stored in a numeric arm"

            raise RecordSkippedError(reason)

        if judgment.categoricalValue is None:
            reason = "an elicited relation naming no relation"

            raise RecordSkippedError(reason)

        return Relational(
            pair.hypothesis,
            _relation_name(judgment.categoricalValue),
            annotator,
        )

    if experiment.taskType != _CONTINUOUS_TASK:
        reason = "an elicited pair response whose experiment has no supported task"

        raise RecordSkippedError(reason)

    return RelationalScalar(
        pair.hypothesis,
        _ENTAILMENT_RELATION,
        _continuous_arm(ResponseContext(judgment, experiment, annotator)),
    )


def span_constraints_of(
    layer: annotation_records.AnnotationLayer, token_count: int
) -> SpanConstraints[str]:
    """Return the per-position candidate arm a token-tag layer stands for.

    Parameters
    ----------
    layer : annotation_records.AnnotationLayer
        The layer, of the token-tag kind.
    token_count : int
        How many tokens the tagged item holds, which the beams run over.

    Returns
    -------
    SpanConstraints[str]
        Permitted labels in one width-one constraint per token position.

    Raises
    ------
    RecordSkippedError
        If an annotation sets two anchor variants at once, carries confidence,
        names no position or no label, if a position falls outside the item, or
        if some position admits no candidate.
    """
    permitted: list[set[str]] = [set() for _ in range(token_count)]

    for annotation in layer.annotations:
        if annotation.confidence is not None:
            reason = "a token-tag layer carrying confidence"
            raise RecordSkippedError(reason)

        position = _source_of(annotation).position

        if position is None or not annotation.label:
            reason = "a token-tag layer whose tag names no position or no label"
            raise RecordSkippedError(reason)

        if not 0 <= position < token_count:
            reason = "a token-tag layer tagging a position the item does not hold"
            raise RecordSkippedError(reason)

        permitted[position].add(annotation.label)

    if any(not labels for labels in permitted):
        reason = "a token-tag layer leaving a position untagged"

        raise RecordSkippedError(reason)

    return SpanConstraints.from_positions(
        tuple(frozenset(labels) for labels in permitted)
    )


def graph_start(edge: graph_records.GraphEdge, expression: str) -> str | None:
    """Return the object at an edge's other end where it names an expression.

    Parameters
    ----------
    edge : graph_records.GraphEdge
        The edge to read.
    expression : str
        The AT-URI of the expression the graph would be about.

    Returns
    -------
    str | None
        The record at the other end, or None if neither end names the expression.
    """
    source = edge.source.recordRef
    target = edge.target.recordRef

    if source == expression:
        return target

    if target == expression:
        return source

    return None


def _endpoints(edge: graph_records.GraphEdge) -> tuple[str, str] | None:
    """Return the two records an edge runs between, where it runs between two.

    Parameters
    ----------
    edge : graph_records.GraphEdge
        The edge to read.

    Returns
    -------
    tuple[str, str] | None
        The source and target, or None if either endpoint is not a record.
    """
    source = edge.source.recordRef
    target = edge.target.recordRef

    if source is None or target is None:
        return None

    return (source, target)


def _reached(
    start: str,
    nodes: Mapping[str, graph_records.GraphNode],
    edges: Mapping[str, graph_records.GraphEdge],
) -> frozenset[str]:
    """Return the graph nodes an edge's endpoint reaches, itself included.

    Parameters
    ----------
    start : str
        The node the attaching edge names.
    nodes : Mapping[str, graph_records.GraphNode]
        Every graph node the corpus holds, by AT-URI.
    edges : Mapping[str, graph_records.GraphEdge]
        Every graph edge the corpus holds, by AT-URI.

    Returns
    -------
    frozenset[str]
        The AT-URIs of the nodes reached.
    """
    found = {start}
    pending = [start]

    while pending:
        here = pending.pop()

        for edge in edges.values():
            ends = _endpoints(edge)

            if ends is None or here not in ends:
                continue

            for end in ends:
                if end in nodes and end not in found:
                    found.add(end)
                    pending.append(end)

    return frozenset(found)


class GraphContext(NamedTuple):
    """The records reachable from a graph attachment.

    Parameters
    ----------
    start : str
        The AT-URI of the graph node the attaching edge names.
    nodes : Mapping[str, graph_records.GraphNode]
        Every graph node the corpus holds, by AT-URI.
    edges : Mapping[str, graph_records.GraphEdge]
        Every graph edge the corpus holds, by AT-URI.
    """

    start: str
    nodes: Mapping[str, graph_records.GraphNode]
    edges: Mapping[str, graph_records.GraphEdge]


def graph_of(context: GraphContext) -> GoldGraph:
    """Return the meaning graph one attaching edge reaches.

    Parameters
    ----------
    context : GraphContext
        The attached node and the graph records the corpus holds.

    Returns
    -------
    GoldGraph
        The arm holding those nodes and edges.

    Raises
    ------
    RecordSkippedError
        If the attaching edge names no graph node the corpus holds, or if a
        node or an edge states no type.
    """
    if context.start not in context.nodes:
        reason = "a graph edge attaching an expression to no graph node"

        raise RecordSkippedError(reason)

    held = _reached(context.start, context.nodes, context.edges)
    nodes = tuple(
        GraphNode(
            uri,
            context.nodes[uri].nodeType,
            label=context.nodes[uri].label,
            properties=features_of(context.nodes[uri].properties),
        )
        for uri in sorted(held)
    )
    edges: list[GraphEdge] = []

    for uri in sorted(context.edges):
        edge = context.edges[uri]
        ends = _endpoints(edge)

        if ends is None or not held.issuperset(ends):
            continue

        edges.append(
            GraphEdge(
                ends[0],
                ends[1],
                edge.edgeType,
                label=edge.label,
                properties=features_of(edge.properties),
            )
        )

    return GoldGraph(nodes, tuple(edges))


class TokenReading(NamedTuple):
    """A tokenization and its token count.

    Parameters
    ----------
    identifier : str | None
        The tokenization's own identifier, absent for an item the corpus has
        not segmented.
    count : int
        How many tokens were read.
    """

    identifier: str | None
    count: int


class LayerContext(NamedTuple):
    """An annotation layer with its decoding context.

    Parameters
    ----------
    layer : annotation_records.AnnotationLayer
        The layer to read.
    tokens : TokenReading
        The tokenization the annotated item's tokens were read from.
    labels : LabelMembership[str] | None
        The category labels the run admits.
    """

    layer: annotation_records.AnnotationLayer
    tokens: TokenReading
    labels: LabelMembership[str] | None


class LayerLift(NamedTuple):
    """A decoder for one annotation layer kind.

    Parameters
    ----------
    kind : str
        The layer kind this lift reads.
    subkind : str | None
        The subkind it reads, or None if it reads every subkind.
    arm : SupervisionArm
        The arm of the supervision union it produces.
    read : Callable[[LayerContext], Supervision[str]]
        Callable that decodes one layer.
    """

    kind: str
    subkind: str | None
    arm: SupervisionArm
    read: Callable[[LayerContext], Supervision[str]]


def _tree_arm(context: LayerContext) -> Supervision[str]:
    """Return the analysis arm a tree layer stands for.

    Parameters
    ----------
    context : LayerContext
        The layer and what it is read against.

    Returns
    -------
    Supervision[str]
        The decoded derivation.
    """
    return derivation_of(context.layer, context.labels)


def _dependency_arm(context: LayerContext) -> Supervision[str]:
    """Return the relation arm a dependency layer stands for.

    Parameters
    ----------
    context : LayerContext
        The layer and what it is read against.

    Returns
    -------
    Supervision[str]
        The decoded dependencies.
    """
    return dependencies_of(context.layer)


def _tag_arm(context: LayerContext) -> Supervision[str]:
    """Return the per-position candidate arm a token-tag layer stands for.

    Parameters
    ----------
    context : LayerContext
        The layer and what it is read against.

    Returns
    -------
    Supervision[str]
        The decoded per-position candidates.
    """
    return span_constraints_of(context.layer, context.tokens.count)


LAYER_LIFTS: Final[tuple[LayerLift, ...]] = (
    LayerLift(_TREE_KIND, None, GoldDerivation, _tree_arm),
    LayerLift(_RELATION_KIND, _DEPENDENCY_SUBKIND, GoldDependencies, _dependency_arm),
    LayerLift(_TAG_KIND, None, SpanConstraints, _tag_arm),
)
"""The annotation layer decoders."""


class GraphLift(NamedTuple):
    """A graph-record decoder for one supervision arm.

    Parameters
    ----------
    arm : SupervisionArm
        The arm of the supervision union it produces.
    read : Callable[[GraphContext], GoldGraph]
        Callable that decodes one attaching edge.
    """

    arm: SupervisionArm
    read: Callable[[GraphContext], GoldGraph]


GRAPH_LIFT: Final = GraphLift(GoldGraph, graph_of)
"""The graph decoder."""


def _refuse_absent_tokenization(
    layer: annotation_records.AnnotationLayer,
    tokens: TokenReading,
) -> None:
    """Raise if a layer names a different tokenization.

    Parameters
    ----------
    layer : annotation_records.AnnotationLayer
        The layer to check.
    tokens : TokenReading
        The tokenization the annotated item's tokens were read from.

    Raises
    ------
    RecordSkippedError
        If the layer names a tokenization and it is not the one read.
    """
    named = layer.tokenizationId

    if named is not None and named.value != tokens.identifier:
        reason = "an annotation layer naming a tokenization the item does not hold"

        raise RecordSkippedError(reason)


def supervision_of(
    layer: annotation_records.AnnotationLayer,
    tokens: TokenReading,
    labels: LabelMembership[str] | None = None,
) -> Supervision[str]:
    """Return the supervision arm for an annotation layer.

    Parameters
    ----------
    layer : annotation_records.AnnotationLayer
        The layer to read.
    tokens : TokenReading
        The tokenization the annotated item's tokens were read from.
    labels : LabelMembership[str] | None
        The inventory against which a tree layer's ontology type references
        are resolved.

    Returns
    -------
    Supervision[str]
        The decoded supervision value.

    Raises
    ------
    RecordSkippedError
        If the layer names a tokenization the item does not hold, if no lift
        reads its kind, if none reads its subkind, or if it is not well formed
        for the lift that claims it.
    """
    _refuse_absent_tokenization(layer, tokens)

    for lift in LAYER_LIFTS:
        if lift.kind == layer.kind and lift.subkind in (None, layer.subkind):
            return lift.read(LayerContext(layer, tokens, labels))

    if any(lift.kind == layer.kind for lift in LAYER_LIFTS):
        reason = "an annotation layer of a subkind the reader does not read"

        raise RecordSkippedError(reason)

    reason = "an annotation layer of a kind the reader does not read"

    raise RecordSkippedError(reason)


def _modulus_of(experiment: judgment_records.ExperimentDef) -> float:
    """Return the reference value a magnitude experiment states.

    Parameters
    ----------
    experiment : judgment_records.ExperimentDef
        The experiment record.

    Returns
    -------
    float
        The modulus, read off the experiment's own features.

    Raises
    ------
    RecordSkippedError
        If the experiment states no modulus, or states one that is not a
        number.
    """
    stated = features_of(experiment.features).get(_MODULUS_KEY)

    if stated is None:
        reason = "a magnitude experiment stating no modulus"

        raise RecordSkippedError(reason)

    try:
        return float(stated)
    except ValueError as error:
        reason = "a magnitude experiment whose modulus is not a number"
        raise RecordSkippedError(reason) from error


def scale_of(experiment: judgment_records.ExperimentDef) -> Scale:
    """Return the scale an experiment collected its responses on.

    Parameters
    ----------
    experiment : judgment_records.ExperimentDef
        The experiment record.

    Returns
    -------
    Scale
        The range and the kind of distance used by its responses: ordinal for
        a rating or yes-or-no task, continuous for a bounded slider, and ratio
        for a magnitude task.

    Raises
    ------
    RecordSkippedError
        If the experiment declares an instrument whose responses are not on a
        scale, if it states no bounds where its instrument needs them, or if
        the bounds are not a scale.
    """
    task = experiment.taskType
    least = experiment.scaleMin

    if task in _ORDINAL_TASKS:
        if least is None or experiment.scaleMax is None:
            reason = "a rating experiment stating no bounds"

            raise RecordSkippedError(reason)

        kind: ScaleKind = "ordinal"

        return _built(kind, float(least), float(experiment.scaleMax), None)

    if task == _CONTINUOUS_TASK:
        if least is None or experiment.scaleMax is None:
            reason = "a continuous experiment stating no bounds"

            raise RecordSkippedError(reason)

        return _built("continuous", float(least), float(experiment.scaleMax), None)

    if task == _RATIO_TASK:
        if least is None:
            reason = "a magnitude experiment stating no least response"

            raise RecordSkippedError(reason)

        greatest = None if experiment.scaleMax is None else float(experiment.scaleMax)

        return _built("ratio", float(least), greatest, _modulus_of(experiment))

    raise RecordSkippedError(LayersSkip.UNSCALED)


def _built(
    kind: ScaleKind,
    least: float,
    greatest: float | None,
    modulus: float | None,
) -> Scale:
    """Return a scale, reporting a corpus that describes an impossible one.

    Parameters
    ----------
    kind : ScaleKind
        Whether the distances between two responses are interpretable.
    least : float
        The least response the task admitted.
    greatest : float | None
        The greatest response the task admitted, where it set one.
    modulus : float | None
        The value the reference item was given, on a ratio scale.

    Returns
    -------
    Scale
        The scale.

    Raises
    ------
    RecordSkippedError
        If the bounds and the kind do not describe a scale.
    """
    try:
        return Scale(kind, least, greatest, modulus=modulus)
    except ValueError as error:
        reason = "an experiment describing no scale"

        raise RecordSkippedError(reason) from error


class ResponseContext(NamedTuple):
    """A response with its decoding context.

    Parameters
    ----------
    judgment : judgment_records.Judgment
        The response.
    experiment : judgment_records.ExperimentDef
        The experiment it was collected under, which states the scale or the
        inventory it reads against.
    annotator : AnnotatorId | None
        Whoever responded, where the corpus records it.
    """

    judgment: judgment_records.Judgment
    experiment: judgment_records.ExperimentDef
    annotator: AnnotatorId | None


class ResponseLift(NamedTuple):
    """A decoder for one response-instrument family.

    Parameters
    ----------
    tasks : tuple[str, ...]
        The instruments this lift reads, as the experiment names them.
    arm : SupervisionArm
        The arm of the supervision union it produces.
    read : Callable[[ResponseContext], Supervision[str]]
        Callable that decodes one response.
    """

    tasks: tuple[str, ...]
    arm: SupervisionArm
    read: Callable[[ResponseContext], Supervision[str]]


def _rating_arm(context: ResponseContext) -> Scalar:
    """Return the rating arm one response on a scale stands for.

    Parameters
    ----------
    context : ResponseContext
        The response and what it is read against.

    Returns
    -------
    Scalar
        The decoded scalar response.

    Raises
    ------
    RecordSkippedError
        If the response holds no value, or if it is one the scale refuses.
    """
    if context.judgment.continuousValue is not None:
        reason = "an integer-scale response stored in the continuous arm"

        raise RecordSkippedError(reason)

    if context.judgment.categoricalValue is not None:
        reason = "an integer-scale response stored in the categorical arm"

        raise RecordSkippedError(reason)

    scale = scale_of(context.experiment)
    value = context.judgment.scalarValue

    if value is None:
        return Scalar(
            None,
            scale,
            context.annotator,
            features=frozenset({RESPONSE_MISSING}),
        )

    try:
        return Scalar(float(value), scale, context.annotator)
    except ValueError as error:
        reason = "a response the experiment's scale refuses"

        raise RecordSkippedError(reason) from error


def _continuous_arm(context: ResponseContext) -> Scalar:
    """Validate an exact-decimal wire value, then lower it to the continuous scale."""
    if context.judgment.scalarValue is not None:
        reason = "a continuous response stored in the integer scalar arm"

        raise RecordSkippedError(reason)

    if context.judgment.categoricalValue is not None:
        reason = "a continuous response stored in the categorical arm"

        raise RecordSkippedError(reason)

    scale = scale_of(context.experiment)
    exact = context.judgment.continuousValue

    if exact is None:
        return Scalar(
            None,
            scale,
            context.annotator,
            features=frozenset({RESPONSE_MISSING}),
        )

    value = _exact_decimal(exact.coefficient, exact.exponent)
    minimum = Decimal(str(scale.minimum))
    maximum = None if scale.maximum is None else Decimal(str(scale.maximum))

    if value < minimum or (maximum is not None and value > maximum):
        reason = "a response the experiment's scale refuses"

        raise RecordSkippedError(reason)

    try:
        return Scalar(float(value), scale, context.annotator)
    except ValueError as error:
        reason = "a response the experiment's scale refuses"

        raise RecordSkippedError(reason) from error


def _scalar_arm(context: ResponseContext) -> Scalar:
    """Return the scalar arm appropriate to the declared instrument."""
    if context.experiment.taskType == _CONTINUOUS_TASK:
        return _continuous_arm(context)

    return _rating_arm(context)


def _label_arm(context: ResponseContext) -> Supervision[str]:
    """Return the categorical arm for a response.

    Parameters
    ----------
    context : ResponseContext
        The response and what it is read against.

    Returns
    -------
    Supervision[str]
        The decoded label and its inventory.

    Raises
    ------
    RecordSkippedError
        If the response holds no label, if the experiment declares no
        inventory, or if the label is not in it.
    """
    if (
        context.judgment.scalarValue is not None
        or context.judgment.continuousValue is not None
    ):
        reason = "a categorical response stored in a numeric arm"

        raise RecordSkippedError(reason)

    chosen = context.judgment.categoricalValue
    inventory = frozenset(context.experiment.labels or ())

    if not inventory:
        reason = "a categorical experiment declaring no inventory"

        raise RecordSkippedError(reason)

    if chosen is None:
        return Categorical(
            None,
            inventory,
            context.annotator,
            features=frozenset({RESPONSE_MISSING}),
        )

    try:
        return Categorical(chosen, inventory, context.annotator)
    except ValueError as error:
        reason = "a response outside the inventory its experiment declares"
        raise RecordSkippedError(reason) from error


RESPONSE_LIFTS: Final[tuple[ResponseLift, ...]] = (
    ResponseLift((_CATEGORICAL_TASK,), Categorical, _label_arm),
    ResponseLift((*_ORDINAL_TASKS, _CONTINUOUS_TASK, _RATIO_TASK), Scalar, _scalar_arm),
)
"""The response-instrument decoders."""


RELATION_LIFT: Final = RelationLift(
    Relational,
    relational_layer_of,
    relational_edge_of,
)
"""The lift shared by each relation-pair representation."""


EMITTED_ARMS: Final[tuple[SupervisionArm, ...]] = (
    Unlabeled,
    *(lift.arm for lift in LAYER_LIFTS),
    *(lift.arm for lift in RESPONSE_LIFTS),
    GRAPH_LIFT.arm,
    RELATION_LIFT.arm,
    RelationalScalar,
)
"""The supervision arms emitted by a Layers load."""


def response_of(
    judgment: judgment_records.Judgment,
    experiment: judgment_records.ExperimentDef,
    annotator: AnnotatorId | None,
) -> Supervision[str]:
    """Return the arm one response to an experiment stands for.

    Parameters
    ----------
    judgment : judgment_records.Judgment
        The response.
    experiment : judgment_records.ExperimentDef
        The experiment it was collected under, which states the scale or the
        inventory it reads against.
    annotator : AnnotatorId | None
        Whoever responded, where the corpus records it.

    Returns
    -------
    Supervision[str]
        The decoded supervision value.

    Raises
    ------
    RecordSkippedError
        If no lift reads the experiment's instrument, if the response is
        missing, if it is one the scale does not admit, or if the experiment
        declares an inventory the label is not in.
    """
    for lift in RESPONSE_LIFTS:
        if experiment.taskType in lift.tasks:
            return lift.read(ResponseContext(judgment, experiment, annotator))

    raise RecordSkippedError(LayersSkip.UNSCALED)


def annotator_of(identifier: str | None) -> AnnotatorId | None:
    """Return the handle on whoever produced a response.

    Parameters
    ----------
    identifier : str | None
        The identifier the corpus records for the agent, or None.

    Returns
    -------
    AnnotatorId | None
        The handle, or None if the corpus names no agent.
    """
    return None if not identifier else AnnotatorId(identifier)


def item_id_of(uri: str, ordinal: int | None = None) -> ItemId:
    """Return the handle one datum of a Layers corpus is joined on.

    Parameters
    ----------
    uri : str
        The AT-URI of the record the datum comes from.
    ordinal : int | None
        The response's position within that record, where the record holds
        more than one datum.

    Returns
    -------
    ItemId
        The handle.
    """
    return ItemId(uri if ordinal is None else f"{uri}#{ordinal}")

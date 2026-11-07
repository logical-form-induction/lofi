"""Supervision types for corpus items."""

from typing import TYPE_CHECKING, Final, Literal, NewType, override

from lofi._checks import ensure_finite, ensure_name
from lofi._types import ItemId
from lofi.data._constraint import SpanConstraints

if TYPE_CHECKING:
    from collections.abc import Hashable

    from lofi.data._derivation import (
        Dependency,
        Derivation,
        GraphEdge,
        GraphNode,
    )
    from lofi.data._values import Item

__all__ = [
    "RESPONSE_MISSING",
    "SCALE_KINDS",
    "AnnotatorId",
    "Categorical",
    "Example",
    "ForcedChoice",
    "GoldDependencies",
    "GoldDerivation",
    "GoldGraph",
    "Relational",
    "RelationalScalar",
    "Scalar",
    "Scale",
    "ScaleKind",
    "SpanConstraints",
    "Supervision",
    "SupervisionArm",
    "Unlabeled",
    "Weighted",
]

_MINIMUM_LEVELS = 2
"""The minimum number of levels on an ordinal scale."""

_MINIMUM_ALTERNATIVES = 2
"""The minimum number of alternatives in a choice trial."""

RESPONSE_MISSING: Final = "response_missing"
"""The feature marking a missing response."""

AnnotatorId = NewType("AnnotatorId", str)
"""A handle for the annotator who produced an annotation."""

type ScaleKind = Literal["ordinal", "continuous", "ratio"]
"""The interpretation of distances between scale values."""

SCALE_KINDS: Final[tuple[ScaleKind, ...]] = ("ordinal", "continuous", "ratio")
"""Declared scale kinds in report order."""


def _ensure_levels(minimum: float, maximum: float) -> int:
    """Return the number of levels on an ordinal scale.

    Parameters
    ----------
    minimum : float
        Minimum response.
    maximum : float
        Maximum response.

    Returns
    -------
    int
        Inclusive number of integer levels.

    Raises
    ------
    ValueError
        If either bound is not a whole number, or if the two leave fewer than
        two levels.
    """
    for bound in (minimum, maximum):
        if not bound.is_integer():
            msg = (
                f"an ordinal scale is bounded at {bound}, which names no "
                "button of a rating task"
            )
            raise ValueError(msg)

    levels = int(maximum) - int(minimum) + 1

    if levels < _MINIMUM_LEVELS:
        msg = (
            f"an ordinal scale from {minimum} to {maximum} has {levels} level, "
            f"and an order needs {_MINIMUM_LEVELS}"
        )
        raise ValueError(msg)

    return levels


def _ensure_modulus(
    minimum: float,
    maximum: float | None,
    modulus: float | None,
) -> float:
    """Return a valid modulus for a ratio scale.

    Parameters
    ----------
    minimum : float
        Minimum response.
    maximum : float | None
        Maximum response, if bounded.
    modulus : float | None
        Reference value.

    Returns
    -------
    float
        The modulus.

    Raises
    ------
    ValueError
        If no modulus is stated, if it is not positive, or if it is a response
        the scale does not admit.
    """
    if modulus is None:
        msg = "a ratio scale states no modulus, so its responses read against nothing"
        raise ValueError(msg)

    if modulus <= 0.0:
        msg = f"a modulus is not positive: {modulus}"
        raise ValueError(msg)

    if modulus < minimum or (maximum is not None and modulus > maximum):
        msg = (
            f"a modulus of {modulus} is not a response on the scale that "
            f"states it, which runs from {minimum} to {maximum}"
        )
        raise ValueError(msg)

    return modulus


class Weighted[T: Hashable]:
    """A value paired with a score.

    Parameters
    ----------
    value : T
        Scored value.
    weight : float
        The score.

    Raises
    ------
    ValueError
        If the weight is not finite.
    """

    __slots__ = ("_value", "_weight")

    def __init__(self, value: T, weight: float) -> None:
        self._value = value
        self._weight = ensure_finite("a weight", weight)

    @property
    def value(self) -> T:
        """Return the thing scored."""
        return self._value

    @property
    def weight(self) -> float:
        """Return the score."""
        return self._weight

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value scores the same thing the same way."""
        if type(other) is not Weighted:
            return NotImplemented

        return self._value == other._value and self._weight == other._weight

    @override
    def __hash__(self) -> int:
        """Return a hash over the value and its weight."""
        return hash((Weighted, self._value, self._weight))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this wrapper."""
        return f"Weighted({self._value!r}, {self._weight!r})"


class Scale:
    """A rating scale.

    Parameters
    ----------
    kind : ScaleKind
        Measurement level: ordinal, interval, or ratio.
    minimum : float
        Minimum permitted response.
    maximum : float | None
        Maximum permitted response, or ``None`` for an unbounded ratio scale.
    modulus : float | None
        Reference value for a ratio scale.

    Raises
    ------
    ValueError
        If the kind, bounds, modulus, or level count is invalid.
    """

    __slots__ = ("_kind", "_levels", "_maximum", "_minimum", "_modulus")

    def __init__(
        self,
        kind: ScaleKind,
        minimum: float,
        maximum: float | None = None,
        *,
        modulus: float | None = None,
    ) -> None:
        if kind not in SCALE_KINDS:
            msg = f"a scale is of no declared kind: {kind!r}"
            raise ValueError(msg)

        least = ensure_finite("a scale's least response", minimum)
        greatest = (
            None
            if maximum is None
            else ensure_finite("a scale's greatest response", maximum)
        )

        if greatest is not None and greatest < least:
            msg = (
                f"a scale runs from {least} down to {greatest}, "
                "so it admits no response"
            )
            raise ValueError(msg)

        if modulus is not None:
            ensure_finite("a modulus", modulus)

        self._kind = kind
        self._minimum = least
        self._maximum = greatest
        self._levels: int | None = None
        self._modulus: float | None = None

        if kind == "ratio":
            self._settle_ratio(modulus)
        else:
            self._settle_bounded(modulus)

    def _settle_ratio(self, modulus: float | None) -> None:
        """Validate and store a ratio scale's modulus.

        Parameters
        ----------
        modulus : float | None
            Reference value.

        Raises
        ------
        ValueError
            If the least response is negative, or if the modulus is one the
            scale does not admit.
        """
        if self._minimum < 0.0:
            msg = (
                f"a ratio scale's least response is negative: {self._minimum}, "
                "and a negative response stands in no ratio to a modulus"
            )
            raise ValueError(msg)

        self._modulus = _ensure_modulus(self._minimum, self._maximum, modulus)

    def _settle_bounded(self, modulus: float | None) -> None:
        """Validate a bounded nonratio scale.

        Parameters
        ----------
        modulus : float | None
            Supplied modulus, which must be ``None``.

        Raises
        ------
        ValueError
            If the maximum is absent, a modulus is present, or ordinal bounds do
            not define integer levels.
        """
        if self._maximum is None:
            msg = f"a {self._kind} scale states no greatest response"
            raise ValueError(msg)

        if modulus is not None:
            msg = (
                f"a {self._kind} scale states a modulus of {modulus}, which "
                "only a ratio scale reads responses against"
            )
            raise ValueError(msg)

        if self._kind == "ordinal":
            self._levels = _ensure_levels(self._minimum, self._maximum)

    @property
    def kind(self) -> ScaleKind:
        """Return whether the scale's distances are interpretable."""
        return self._kind

    @property
    def minimum(self) -> float:
        """Return the least response the task admitted."""
        return self._minimum

    @property
    def maximum(self) -> float | None:
        """Return the greatest response the task admitted, where it set one."""
        return self._maximum

    @property
    def levels(self) -> int | None:
        """Return the ordinal level count, or None for another scale kind."""
        return self._levels

    @property
    def modulus(self) -> float | None:
        """Return the modulus for a ratio scale, or None for another scale kind."""
        return self._modulus

    @property
    def description(self) -> str:
        """Return the scale as a message or a report line names it."""
        if self._kind == "ordinal":
            return (
                f"an ordinal scale of {self._levels} levels from "
                f"{self._minimum} to {self._maximum}"
            )

        if self._kind == "ratio":
            return (
                f"a ratio scale above {self._minimum} with a modulus of {self._modulus}"
            )

        return f"a continuous scale from {self._minimum} to {self._maximum}"

    def covers(self, value: float) -> bool:
        """Return whether a response falls on the scale.

        Parameters
        ----------
        value : float
            Response value.

        Returns
        -------
        bool
            Whether the scale includes the value.
        """
        if value < self._minimum:
            return False

        if self._maximum is not None and value > self._maximum:
            return False

        if self._kind == "ordinal":
            return float(value).is_integer()

        if self._kind == "ratio":
            return value > 0.0

        return True

    def level_of(self, value: float) -> int:
        """Return a response's zero-based ordinal level.

        Parameters
        ----------
        value : float
            Response value.

        Returns
        -------
        int
            Zero-based level index.

        Raises
        ------
        ValueError
            If the scale has no levels, or if it does not admit the response.
        """
        if self._levels is None:
            msg = (
                f"a response of {value} has no level, because it was taken on "
                f"{self.description}"
            )
            raise ValueError(msg)

        if not self.covers(value):
            msg = f"a response of {value} is not a level of {self.description}"
            raise ValueError(msg)

        return int(value) - int(self._minimum)

    def _identity(self) -> tuple[ScaleKind, float, float | None, float | None]:
        """Return the four fields a scale is equal on."""
        return (self._kind, self._minimum, self._maximum, self._modulus)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is the same scale."""
        if type(other) is not Scale:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the kind, the bounds and the modulus."""
        return hash((Scale, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this scale."""
        return (
            f"Scale({self._kind!r}, {self._minimum!r}, {self._maximum!r}, "
            f"modulus={self._modulus!r})"
        )


class Unlabeled:
    """Absent supervision."""

    __slots__ = ()

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is also the unlabeled arm."""
        if type(other) is not Unlabeled:
            return NotImplemented

        return True

    @override
    def __hash__(self) -> int:
        """Return the shared hash for unlabeled arms."""
        return hash(Unlabeled)

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this arm."""
        return "Unlabeled()"


def _response_features(
    features: frozenset[str],
    *,
    missing: bool,
    subject: str,
) -> frozenset[str]:
    """Validate response features against response presence."""
    held = frozenset(ensure_name("a response feature", name) for name in features)
    marked = RESPONSE_MISSING in held

    if missing and not marked:
        msg = f"{subject} is null without the {RESPONSE_MISSING!r} feature"
        raise ValueError(msg)

    if not missing and marked:
        msg = f"{subject} has a value and the {RESPONSE_MISSING!r} feature"
        raise ValueError(msg)

    return held


class Scalar:
    """A scalar response.

    Parameters
    ----------
    value : float | None
        The response value, or ``None`` when missing.
    scale : Scale
        Response scale.
    annotator : AnnotatorId | None
        The recorded annotator identifier, if available.
    features : frozenset[str]
        Response features, including ``missing`` exactly when ``value`` is ``None``.

    Raises
    ------
    ValueError
        If the response is not finite, if it is one the scale does not admit,
        if the annotator or a feature is empty, or if the ``missing`` feature
        disagrees with the response value.
    """

    __slots__ = ("_annotator", "_features", "_scale", "_value")

    def __init__(
        self,
        value: float | None,
        scale: Scale,
        annotator: AnnotatorId | None = None,
        *,
        features: frozenset[str] = frozenset(),
    ) -> None:
        self._features = _response_features(
            features,
            missing=value is None,
            subject="a scalar response",
        )
        self._value = None if value is None else ensure_finite("a rating", value)
        self._scale = scale
        self._annotator = (
            None
            if annotator is None
            else AnnotatorId(ensure_name("an annotator id", annotator))
        )

        if self._value is not None and not scale.covers(self._value):
            msg = f"a rating of {self._value} is not a response on {scale.description}"
            raise ValueError(msg)

    @property
    def value(self) -> float | None:
        """Return the response value, if present."""
        return self._value

    @property
    def scale(self) -> Scale:
        """Return the range the response was collected on."""
        return self._scale

    @property
    def annotator(self) -> AnnotatorId | None:
        """Return the annotator identifier, if recorded."""
        return self._annotator

    @property
    def features(self) -> frozenset[str]:
        """Return response features."""
        return self._features

    @property
    def related_items(self) -> tuple[ItemId, ...]:
        """Return that an ordinary scalar response refers to no other item."""
        return ()

    def _identity(
        self,
    ) -> tuple[float | None, Scale, AnnotatorId | None, frozenset[str]]:
        """Return the fields a rating is equal on."""
        return (self._value, self._scale, self._annotator, self._features)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is the same rating by the same annotator."""
        if type(other) is not Scalar:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the response, the scale and the annotator."""
        return hash((Scalar, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this rating."""
        feature_arg = "" if not self._features else f", features={self._features!r}"

        return (
            f"Scalar({self._value!r}, {self._scale!r}, {self._annotator!r}"
            f"{feature_arg})"
        )


class Categorical:
    """A categorical response.

    Parameters
    ----------
    label : str | None
        The selected label, or ``None`` when missing.
    label_set : frozenset[str]
        Permitted labels.
    annotator : AnnotatorId | None
        The recorded annotator identifier, if available.
    features : frozenset[str]
        Response features, including ``missing`` exactly when ``label`` is ``None``.

    Raises
    ------
    ValueError
        If a label, annotator, or feature is empty; if the label set is empty;
        if the selected label is outside the set; or if the ``missing`` feature
        disagrees with the selected label.
    """

    __slots__ = ("_annotator", "_features", "_label", "_label_set")

    def __init__(
        self,
        label: str | None,
        label_set: frozenset[str],
        annotator: AnnotatorId | None = None,
        *,
        features: frozenset[str] = frozenset(),
    ) -> None:
        self._features = _response_features(
            features,
            missing=label is None,
            subject="a categorical response",
        )
        self._label = None if label is None else ensure_name("a label", label)

        if not label_set:
            msg = f"the label {label!r} was chosen from an empty inventory"
            raise ValueError(msg)

        for alternative in label_set:
            ensure_name("a label in an inventory", alternative)

        if self._label is not None and self._label not in label_set:
            msg = (
                f"the label {self._label!r} is not in the inventory it was chosen "
                f"from: {sorted(label_set)}"
            )
            raise ValueError(msg)

        self._label_set = label_set
        self._annotator = (
            None
            if annotator is None
            else AnnotatorId(ensure_name("an annotator id", annotator))
        )

    @property
    def label(self) -> str | None:
        """Return what was chosen."""
        return self._label

    @property
    def label_set(self) -> frozenset[str]:
        """Return the declared label set."""
        return self._label_set

    @property
    def annotator(self) -> AnnotatorId | None:
        """Return the annotator identifier, if recorded."""
        return self._annotator

    @property
    def features(self) -> frozenset[str]:
        """Return response features."""
        return self._features

    def _identity(
        self,
    ) -> tuple[str | None, frozenset[str], AnnotatorId | None, frozenset[str]]:
        """Return the fields a categorical response is equal on."""
        return (self._label, self._label_set, self._annotator, self._features)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is the same choice by the same annotator."""
        if type(other) is not Categorical:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the label, the inventory and the annotator."""
        return hash((Categorical, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this response."""
        feature_arg = "" if not self._features else f", features={self._features!r}"

        return (
            f"Categorical({self._label!r}, {self._label_set!r}, "
            f"{self._annotator!r}{feature_arg})"
        )


class ForcedChoice:
    """A forced-choice response.

    Parameters
    ----------
    alternatives : tuple[ItemId, ...]
        Presented item identifiers in display order.
    chosen : frozenset[ItemId]
        Selected item identifiers.
    annotator : AnnotatorId | None
        Annotator identifier, if recorded.

    Raises
    ------
    ValueError
        If identifiers are empty or duplicated, fewer than two alternatives
        exist, no item is selected, or a selected item is not an alternative.
    """

    __slots__ = ("_alternatives", "_annotator", "_chosen")

    def __init__(
        self,
        alternatives: tuple[ItemId, ...],
        chosen: frozenset[ItemId],
        annotator: AnnotatorId | None = None,
    ) -> None:
        if len(alternatives) < _MINIMUM_ALTERNATIVES:
            msg = (
                f"a choice was made between fewer than {_MINIMUM_ALTERNATIVES} "
                f"items: {list(alternatives)}"
            )
            raise ValueError(msg)

        for alternative in alternatives:
            ensure_name("an alternative's id", alternative)

        if len(set(alternatives)) != len(alternatives):
            msg = f"one item was shown twice in one trial: {list(alternatives)}"
            raise ValueError(msg)

        if not chosen:
            msg = f"nothing was picked from {list(alternatives)}"
            raise ValueError(msg)

        for pick in chosen:
            if pick not in alternatives:
                msg = (
                    f"the item {pick!r} was picked out of a trial that showed "
                    f"{list(alternatives)}"
                )
                raise ValueError(msg)

        self._alternatives = alternatives
        self._chosen = chosen
        self._annotator = (
            None
            if annotator is None
            else AnnotatorId(ensure_name("an annotator id", annotator))
        )

    @property
    def alternatives(self) -> tuple[ItemId, ...]:
        """Return the items shown together, in the order they were shown."""
        return self._alternatives

    @property
    def chosen(self) -> frozenset[ItemId]:
        """Return the items picked out of them."""
        return self._chosen

    @property
    def is_tie(self) -> bool:
        """Return whether more than one alternative was picked."""
        return len(self._chosen) > 1

    @property
    def annotator(self) -> AnnotatorId | None:
        """Return the annotator identifier, if recorded."""
        return self._annotator

    def _identity(
        self,
    ) -> tuple[tuple[ItemId, ...], frozenset[ItemId], AnnotatorId | None]:
        """Return the three fields a choice is equal on."""
        return (self._alternatives, self._chosen, self._annotator)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is the same pick from the same trial."""
        if type(other) is not ForcedChoice:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the alternatives, the pick and the annotator."""
        return hash((ForcedChoice, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this response."""
        return (
            f"ForcedChoice({self._alternatives!r}, {self._chosen!r}, "
            f"{self._annotator!r})"
        )


class GoldDerivation[L: Hashable]:
    """A gold derivation.

    Parameters
    ----------
    derivation : Derivation[L]
        Gold derivation.
    """

    __slots__ = ("_derivation",)

    def __init__(self, derivation: Derivation[L]) -> None:
        self._derivation = derivation

    @property
    def derivation(self) -> Derivation[L]:
        """Return the analysis the corpus asserts."""
        return self._derivation

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value asserts the same analysis."""
        if type(other) is not GoldDerivation:
            return NotImplemented

        return self._derivation == other._derivation

    @override
    def __hash__(self) -> int:
        """Return a hash over the analysis."""
        return hash((GoldDerivation, self._derivation))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this arm."""
        return f"GoldDerivation({self._derivation!r})"


class GoldDependencies:
    """Gold dependencies.

    Parameters
    ----------
    dependencies : tuple[Dependency, ...]
        Dependencies in source order.
    """

    __slots__ = ("_dependencies",)

    def __init__(self, dependencies: tuple[Dependency, ...]) -> None:
        self._dependencies = dependencies

    @property
    def dependencies(self) -> tuple[Dependency, ...]:
        """Return the relations the corpus asserts."""
        return self._dependencies

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value asserts the same relations in order."""
        if type(other) is not GoldDependencies:
            return NotImplemented

        return self._dependencies == other._dependencies

    @override
    def __hash__(self) -> int:
        """Return a hash over the relations."""
        return hash((GoldDependencies, self._dependencies))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this arm."""
        return f"GoldDependencies({self._dependencies!r})"


class GoldGraph:
    """A gold meaning graph.

    Parameters
    ----------
    nodes : tuple[GraphNode, ...]
        Graph nodes in source order.
    edges : tuple[GraphEdge, ...]
        Graph edges in source order.

    Raises
    ------
    ValueError
        If node identifiers repeat or an edge endpoint is absent.
    """

    __slots__ = ("_edges", "_nodes")

    def __init__(
        self,
        nodes: tuple[GraphNode, ...],
        edges: tuple[GraphEdge, ...] = (),
    ) -> None:
        identifiers = {node.identifier for node in nodes}

        if len(identifiers) != len(nodes):
            msg = "a meaning graph holds two nodes under one identifier"
            raise ValueError(msg)

        for edge in edges:
            for endpoint in (edge.source, edge.target):
                if endpoint not in identifiers:
                    msg = (
                        f"a meaning graph has an edge at {endpoint!r}, which "
                        "names no node it holds"
                    )
                    raise ValueError(msg)

        self._nodes = nodes
        self._edges = edges

    @property
    def nodes(self) -> tuple[GraphNode, ...]:
        """Return the graph's nodes, in the order the corpus recorded them."""
        return self._nodes

    @property
    def edges(self) -> tuple[GraphEdge, ...]:
        """Return the graph's edges, in the order the corpus recorded them."""
        return self._edges

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is the same graph, listed the same way."""
        if type(other) is not GoldGraph:
            return NotImplemented

        return self._nodes == other._nodes and self._edges == other._edges

    @override
    def __hash__(self) -> int:
        """Return a hash over the nodes and the edges."""
        return hash((GoldGraph, self._nodes, self._edges))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this graph."""
        return f"GoldGraph({self._nodes!r}, {self._edges!r})"


class Relational:
    """A relation between items.

    Parameters
    ----------
    other : ItemId
        Related item identifier.
    relation : str
        Relation label.
    annotator : AnnotatorId | None
        Annotator identifier, if recorded.

    Raises
    ------
    ValueError
        If the other item's id, the relation or the annotator is empty.
    """

    __slots__ = ("_annotator", "_other", "_relation")

    def __init__(
        self,
        other: ItemId,
        relation: str,
        annotator: AnnotatorId | None = None,
    ) -> None:
        self._other = ItemId(ensure_name("a related item's id", other))
        self._relation = ensure_name("a relation", relation)
        self._annotator = (
            None
            if annotator is None
            else AnnotatorId(ensure_name("an annotator id", annotator))
        )

    @property
    def other(self) -> ItemId:
        """Return the item this one is related to."""
        return self._other

    @property
    def relation(self) -> str:
        """Return the relation's name, in the corpus's own inventory."""
        return self._relation

    @property
    def annotator(self) -> AnnotatorId | None:
        """Return the annotator identifier, if recorded."""
        return self._annotator

    def _identity(self) -> tuple[ItemId, str, AnnotatorId | None]:
        """Return the three fields a relational response is equal on."""
        return (self._other, self._relation, self._annotator)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value asserts the same relation to the same item."""
        if type(other) is not Relational:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the other item, the relation and the annotator."""
        return hash((Relational, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this response."""
        return f"Relational({self._other!r}, {self._relation!r}, {self._annotator!r})"


class RelationalScalar:
    """A scored relation between items.

    Parameters
    ----------
    other : ItemId
        Related item identifier.
    relation : str
        Relation label.
    rating : Scalar
        Scalar relation rating.

    Raises
    ------
    ValueError
        If the other item's id or the relation is empty.
    """

    __slots__ = ("_other", "_rating", "_relation")

    def __init__(self, other: ItemId, relation: str, rating: Scalar) -> None:
        self._other = ItemId(ensure_name("a related item's id", other))
        self._relation = ensure_name("a relation", relation)
        self._rating = rating

    @property
    def other(self) -> ItemId:
        """Return the item the relation runs to."""
        return self._other

    @property
    def relation(self) -> str:
        """Return the relation's name, in the corpus's own inventory."""
        return self._relation

    @property
    def value(self) -> float | None:
        """Return how strongly the relation holds, if recorded."""
        return self._rating.value

    @property
    def scale(self) -> Scale:
        """Return the scale on which the relation was rated."""
        return self._rating.scale

    @property
    def related_items(self) -> tuple[ItemId, ...]:
        """Return the related item in semantic pair order."""
        return (self._other,)

    @property
    def annotator(self) -> AnnotatorId | None:
        """Return the annotator identifier, if recorded."""
        return self._rating.annotator

    def _identity(self) -> tuple[ItemId, str, Scalar]:
        """Return the three fields a graded relation is equal on."""
        return (self._other, self._relation, self._rating)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value rates the same relation the same way."""
        if type(other) is not RelationalScalar:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the other item, the relation and the rating."""
        return hash((RelationalScalar, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this response."""
        return (
            f"RelationalScalar({self._other!r}, {self._relation!r}, {self._rating!r})"
        )


type Supervision[L: Hashable] = (
    Unlabeled
    | Scalar
    | Categorical
    | ForcedChoice
    | SpanConstraints[L]
    | GoldDerivation[L]
    | GoldDependencies
    | GoldGraph
    | Relational
    | RelationalScalar
)
"""The supported supervision union."""


type SupervisionArm = type[Supervision[Hashable]]
"""A supervision arm identified by its type."""


class Example[S]:
    """An item paired with its supervision.

    Parameters
    ----------
    item : Item
        Corpus item.
    supervision : S
        Associated supervision.
    """

    __slots__ = ("_item", "_supervision")

    def __init__(self, item: Item, supervision: S) -> None:
        self._item = item
        self._supervision = supervision

    @property
    def item(self) -> Item:
        """Return the datum."""
        return self._item

    @property
    def supervision(self) -> S:
        """Return what the corpus says about the datum."""
        return self._supervision

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value pairs the same datum with the same claim."""
        if type(other) is not Example:
            return NotImplemented

        return self._item == other._item and self._supervision == other._supervision

    @override
    def __hash__(self) -> int:
        """Return a hash over the datum and the claim.

        Raises
        ------
        TypeError
            If the supervision value is not hashable.
        """
        return hash((Example, self._item, self._supervision))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this example."""
        return f"Example({self._item!r}, {self._supervision!r})"

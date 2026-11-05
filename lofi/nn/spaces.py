"""Runtime bindings for sort-indexed neural spaces."""

from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, cast, override, runtime_checkable

import torch
from torch import nn

from lofi.nn._values import Encoded, LogDistribution

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from torch import Tensor

    from lofi.judge import Action, ActionSet, Rule, RuleSet
    from lofi.nn.codec import NeuralCodecView
    from lofi.nn.ports import ForwardOptions, TermEncoder
    from lofi.terms import ChoiceIndex, Term, TermCodec, TermStore

from lofi.nn.ports import (
    ForwardMode,
    MarginalDecoder,
    TeacherForcedDecoder,
    ensure_mode,
)
from lofi.nn.raising import UnaryReencoding, reencode_unary_domain
from lofi.nn.strategies import exact_matches, score

__all__ = [
    "CellComposer",
    "ChartCombine",
    "CombinatorFamily",
    "Concatenate",
    "PairMerge",
    "RelationKind",
    "Space",
    "SpaceRelationBinding",
    "TermMorphism",
]


@runtime_checkable
class CellComposer(Protocol):
    """Compose paired chart-cell tensors in the chart's represented space."""

    def compose(self, left: Tensor, right: Tensor) -> Tensor:
        """Return one parent representation per paired child cell."""


def _paired(left: Encoded, right: Encoded, /) -> None:
    """Validate that two encoded values address the same pairs."""
    if left.axes != right.axes or left.tensor.shape != right.tensor.shape:
        msg = "paired child cells have different axes or shapes"
        raise ValueError(msg)

    if (left.mask is None) is not (right.mask is None):
        msg = "paired child cells have different masks"
        raise ValueError(msg)

    if (
        left.mask is not None
        and right.mask is not None
        and not bool(torch.equal(left.mask, right.mask))
    ):
        msg = "paired child cells have different masks"
        raise ValueError(msg)


@runtime_checkable
class PairMerge(Protocol):
    """Merge paired daughter vectors before a term decoder reads them."""

    @property
    def output_width(self) -> int:
        """Return the width of one merged pair."""

    def merge(self, left: Encoded, right: Encoded, /) -> Encoded:
        """Return one merged vector per paired daughter row."""


class RelationKind(StrEnum):
    """Names the runtime relation between category and semantic spaces."""

    ERASE_THEN_DECODE = "erase-then-decode"
    CONSISTENCY = "consistency"
    MAPPED = "mapped"


@runtime_checkable
class _SpaceMap(Protocol):
    """Map source-space vectors into a target represented space."""

    def apply_to(self, vectors: Encoded, /) -> Encoded:
        """Return vectors in the target space."""


@runtime_checkable
class _CombinatorDecoder(TeacherForcedDecoder, MarginalDecoder, Protocol):
    """Provide both training rows and marginal chart composition."""


@runtime_checkable
class TermMorphism(Protocol):
    """Map a source-space term into a target term language."""

    def __call__(self, term: Term, /) -> Term:
        """Return the target image of one source term."""


class SpaceRelationBinding(nn.Module):
    """Bind a represented category space to its semantic target space."""

    def __init__(
        self,
        source: Space,
        target: Space,
        /,
        *,
        kind: RelationKind,
        mapping: _SpaceMap | None,
        erasure: TermMorphism | None,
    ) -> None:
        super().__init__()

        if kind is RelationKind.MAPPED:
            if mapping is None:
                msg = "a mapped space relation requires its learned map"
                raise ValueError(msg)
            if erasure is not None:
                msg = "a mapped space relation cannot hold a term erasure"
                raise ValueError(msg)
        else:
            if mapping is not None:
                msg = f"a {kind.value} space relation cannot hold a learned map"
                raise ValueError(msg)
            if erasure is None:
                msg = f"a {kind.value} space relation requires its term erasure"
                raise ValueError(msg)

        self._source = source
        self._target = target
        self._kind = kind
        self._erasure = erasure
        self.mapping = mapping

    @property
    def source(self) -> Space:
        """Return the represented source space."""
        return self._source

    @property
    def target(self) -> Space:
        """Return the represented semantic target space."""
        return self._target

    @property
    def kind(self) -> RelationKind:
        """Return the declared relation kind."""
        return self._kind

    def project(self, vectors: Encoded, /) -> Encoded:
        """Return source vectors in the target embedding space.

        Consistency compares the two spaces directly and therefore requires equal
        widths. A mapped relation applies its learned map. Erase-then-decode is a
        relation between decoded terms, so it has no vector projection.
        """
        if self._kind is RelationKind.ERASE_THEN_DECODE:
            msg = "erase-then-decode projects decoded terms, not space vectors"
            raise ValueError(msg)

        if self._kind is RelationKind.CONSISTENCY:
            target_width = self._target.encoder.output_width

            if vectors.tensor.shape[-1] != target_width:
                msg = (
                    "a consistency relation compares equal-width spaces, not "
                    f"{vectors.tensor.shape[-1]} and {target_width}"
                )
                raise ValueError(msg)

            return vectors

        mapping = self.mapping

        if mapping is None:
            msg = "a mapped relation holds no space-map capability"
            raise TypeError(msg)

        return mapping.apply_to(vectors)

    def erase(self, term: Term, /) -> Term:
        """Return one source term in the target language.

        Mapped relations operate on vectors and are therefore refused here.
        """
        erasure = self._erasure

        if erasure is None:
            msg = "a mapped space relation projects vectors rather than terms"
            raise ValueError(msg)

        return erasure(term)


class Concatenate:
    """Merge paired daughters by concatenating their feature vectors."""

    __slots__ = ("_input_width",)

    def __init__(self, input_width: int) -> None:
        if input_width <= 0:
            msg = f"a merge input has positive width, not {input_width!r}"
            raise ValueError(msg)

        self._input_width = input_width

    @property
    def output_width(self) -> int:
        """Return twice the daughter width."""
        return 2 * self._input_width

    def merge(self, left: Encoded, right: Encoded, /) -> Encoded:
        """Return paired daughter features concatenated innermost."""
        _paired(left, right)

        if left.tensor.shape[-1] != self._input_width:
            msg = (
                f"the merge reads width {self._input_width}, not "
                f"{left.tensor.shape[-1]}"
            )
            raise ValueError(msg)

        return Encoded(
            torch.cat((left.tensor, right.tensor), dim=-1),
            left.axes,
            left.mask,
        )


class ChartCombine:
    """Merge paired daughters through the chart's published composition."""

    __slots__ = ("_composer", "_width")

    def __init__(self, composer: CellComposer, width: int) -> None:
        if width <= 0:
            msg = f"a chart merge has positive width, not {width!r}"
            raise ValueError(msg)

        self._composer = composer
        self._width = width

    @property
    def output_width(self) -> int:
        """Return the chart-cell width."""
        return self._width

    def merge(self, left: Encoded, right: Encoded, /) -> Encoded:
        """Return the chart's composition of paired daughter vectors."""
        _paired(left, right)

        if left.tensor.shape[-1] != self._width:
            msg = (
                f"the chart merge reads width {self._width}, not "
                f"{left.tensor.shape[-1]}"
            )
            raise ValueError(msg)

        merged = self._composer.compose(left.tensor, right.tensor)

        if merged.shape != left.tensor.shape:
            msg = (
                f"the chart merge returned shape {tuple(merged.shape)}, not "
                f"{tuple(left.tensor.shape)}"
            )
            raise ValueError(msg)

        return Encoded(merged, left.axes, left.mask)


def _checked_instances(
    premises: Sequence[Sequence[Term]], conclusions: Sequence[Term], arity: int, /
) -> tuple[tuple[Term, ...], ...]:
    """Return aligned premise tuples at one declared arity."""
    held = tuple(tuple(row) for row in premises)

    if len(held) != len(conclusions):
        msg = (
            f"a scorer received {len(held)} premise tuples and "
            f"{len(conclusions)} conclusions"
        )
        raise ValueError(msg)

    for row, inputs in enumerate(held):
        if len(inputs) != arity:
            msg = f"premise row {row} has arity {len(inputs)}, not {arity}"
            raise ValueError(msg)

    return held


class _MergedCombinatorScorer:
    """Score parent terms after the family's declared daughter merge."""

    __slots__ = ("_codec", "_decoder", "_encoder", "_index", "_merge")

    def __init__(
        self,
        decoder: TeacherForcedDecoder,
        merge: PairMerge,
        index: ChoiceIndex,
        codec: TermCodec,
        encoder: TermEncoder,
    ) -> None:
        self._decoder = decoder
        self._merge = merge
        self._index = index
        self._codec = codec
        self._encoder = encoder

    @property
    def arity(self) -> int:
        """Return binary arity."""
        return 2

    @property
    def slots(self) -> int:
        """Return the codec's conclusion capacity."""
        return self._codec.capacity

    @property
    def modes(self) -> frozenset[ForwardMode]:
        """Return the supervision schedule modes this scorer accepts."""
        return frozenset({ForwardMode.OBSERVE, ForwardMode.SAMPLE})

    def score(
        self,
        premises: Sequence[Sequence[Term]],
        /,
        *,
        conclusions: Sequence[Term],
        options: ForwardOptions,
    ) -> LogDistribution:
        """Merge paired daughter vectors and score their supplied parents."""
        ensure_mode(options, self.modes)
        rows = _checked_instances(premises, conclusions, self.arity)
        left = self._encoder.encode(
            self._codec.encode_batch(tuple(inputs[0] for inputs in rows))
        )
        right = self._encoder.encode(
            self._codec.encode_batch(tuple(inputs[1] for inputs in rows))
        )
        return score(
            self._decoder,
            self._merge.merge(left, right),
            self._codec.encode_batch(conclusions),
            index=self._index,
        )

    def exact_matches(
        self,
        premises: Sequence[Sequence[Term]],
        /,
        *,
        conclusions: Sequence[Term],
        options: ForwardOptions,
    ) -> Tensor:
        """Return exact teacher-forced conclusion matches."""
        ensure_mode(options, self.modes)
        rows = _checked_instances(premises, conclusions, self.arity)
        left = self._encoder.encode(
            self._codec.encode_batch(tuple(inputs[0] for inputs in rows))
        )
        right = self._encoder.encode(
            self._codec.encode_batch(tuple(inputs[1] for inputs in rows))
        )
        return exact_matches(
            self._decoder,
            self._merge.merge(left, right),
            self._codec.encode_batch(conclusions),
            index=self._index,
        )


class _UnaryCombinatorScorer:
    """Score a unary rule by re-encoding its symbolic conclusion."""

    __slots__ = ("_codec", "_decoder", "_encoder", "_index", "_rule")

    def __init__(
        self,
        rule: Rule[Term],
        decoder: TeacherForcedDecoder,
        index: ChoiceIndex,
        codec: TermCodec,
        encoder: TermEncoder,
    ) -> None:
        self._rule = rule
        self._decoder = decoder
        self._index = index
        self._codec = codec
        self._encoder = encoder

    @property
    def arity(self) -> int:
        """Return unary arity."""
        return 1

    @property
    def slots(self) -> int:
        """Return the codec's conclusion capacity."""
        return self._codec.capacity

    @property
    def modes(self) -> frozenset[ForwardMode]:
        """Return the supervision schedule modes this scorer accepts."""
        return frozenset({ForwardMode.OBSERVE, ForwardMode.SAMPLE})

    def score(
        self,
        premises: Sequence[Sequence[Term]],
        /,
        *,
        conclusions: Sequence[Term],
        options: ForwardOptions,
    ) -> LogDistribution:
        """Encode licensed unary conclusions and score the supplied conclusions."""
        ensure_mode(options, self.modes)
        rows = _checked_instances(premises, conclusions, self.arity)
        self._ensure_licensed(rows, conclusions)
        encoded = self._encoder.encode(self._codec.encode_batch(conclusions))
        return score(
            self._decoder,
            encoded,
            self._codec.encode_batch(conclusions),
            index=self._index,
        )

    def exact_matches(
        self,
        premises: Sequence[Sequence[Term]],
        /,
        *,
        conclusions: Sequence[Term],
        options: ForwardOptions,
    ) -> Tensor:
        """Return exact teacher-forced unary-conclusion matches."""
        ensure_mode(options, self.modes)
        rows = _checked_instances(premises, conclusions, self.arity)
        self._ensure_licensed(rows, conclusions)
        encoded = self._encoder.encode(self._codec.encode_batch(conclusions))
        return exact_matches(
            self._decoder,
            encoded,
            self._codec.encode_batch(conclusions),
            index=self._index,
        )

    def _ensure_licensed(
        self,
        rows: tuple[Sequence[Term], ...],
        conclusions: Sequence[Term],
        /,
    ) -> None:
        """Require each supplied conclusion to be licensed by its unary premise."""
        for row, (inputs, conclusion) in enumerate(zip(rows, conclusions, strict=True)):
            if conclusion not in self._rule.apply((inputs[0],)):
                msg = (
                    f"{self._rule.name!r} does not derive supplied conclusion row "
                    f"{row} from its unary premise"
                )
                raise ValueError(msg)


class CombinatorFamily:
    """Score each composing action at its rule's declared arity."""

    __slots__ = (
        "_actions",
        "_codec",
        "_composing",
        "_decoders",
        "_encoder",
        "_merge",
        "_root_sort",
        "_scorers",
        "_unary",
    )

    def __init__(
        self,
        decoders: Mapping[Action, object],
        /,
        *,
        actions: ActionSet,
        rules: RuleSet[Term],
        merge: PairMerge | None,
        codec: TermCodec,
        encoder: TermEncoder,
        identity: TeacherForcedDecoder,
        index: ChoiceIndex,
        root_sort: str,
    ) -> None:
        composing = actions.composing
        held = frozenset(decoders)
        by_name = {rule.name: rule for rule in rules}
        by_action: dict[Action, Rule[Term]] = {}

        for action in composing:
            rule = None if action.rule is None else by_name.get(action.rule)

            if rule is None:
                msg = f"composing action {action.name!r} names no installed rule"
                raise ValueError(msg)

            if rule.arity not in (1, 2):
                msg = (
                    f"composing action {action.name!r} applies arity {rule.arity}; "
                    "a combinator family supports unary and binary rules"
                )
                raise ValueError(msg)

            by_action[action] = rule

        binary = frozenset(
            action for action, rule in by_action.items() if rule.arity == 2
        )
        missing = binary - held
        extra = held - binary

        if missing:
            names = ", ".join(sorted(action.name for action in missing))
            msg = f"no decoder composes under binary actions {names}"
            raise ValueError(msg)

        if extra:
            names = ", ".join(sorted(action.name for action in extra))
            msg = f"a decoder composes under {names}, which are not binary actions"
            raise ValueError(msg)

        incapable = tuple(
            action.name
            for action, decoder in decoders.items()
            if not isinstance(decoder, _CombinatorDecoder)
        )
        if incapable:
            names = ", ".join(incapable)
            msg = (
                f"combinator decoders for {names} must provide teacher-forced and "
                "marginal rows"
            )
            raise TypeError(msg)

        held_decoders = {
            action: decoder
            for action, decoder in decoders.items()
            if isinstance(decoder, _CombinatorDecoder)
        }
        self._decoders: Mapping[Action, _CombinatorDecoder] = MappingProxyType(
            held_decoders
        )
        self._actions = actions
        self._composing = composing
        self._codec = codec
        self._encoder = encoder
        if binary and merge is None:
            msg = "binary combinator actions require one daughter merge"
            raise ValueError(msg)
        if not binary and merge is not None:
            msg = "a unary-only combinator family declares no daughter merge"
            raise ValueError(msg)
        self._merge = merge
        self._root_sort = root_sort
        self._unary = MappingProxyType(
            {action: rule for action, rule in by_action.items() if rule.arity == 1}
        )
        self._scorers = MappingProxyType(
            {
                action: (
                    _UnaryCombinatorScorer(rule, identity, index, codec, encoder)
                    if rule.arity == 1
                    else _MergedCombinatorScorer(
                        self._decoders[action],
                        cast("PairMerge", merge),
                        index,
                        codec,
                        encoder,
                    )
                )
                for action, rule in by_action.items()
            }
        )

    @property
    def decoders(self) -> Mapping[Action, _CombinatorDecoder]:
        """Return the raw term decoder bound to each composing action."""
        return self._decoders

    @property
    def scorers(
        self,
    ) -> Mapping[Action, _MergedCombinatorScorer | _UnaryCombinatorScorer]:
        """Return conclusion scorers bound to each composing action."""
        return self._scorers

    @property
    def merge(self) -> PairMerge | None:
        """Return how binary daughter vectors become a decoder input, if needed."""
        return self._merge

    @property
    def actions(self) -> ActionSet:
        """Return the complete action set the family covers."""
        return self._actions

    @property
    def composing(self) -> tuple[Action, ...]:
        """Return the composing actions in controller-column order."""
        return self._composing

    @property
    def identity(self) -> Action:
        """Return the action that leaves a represented cell unchanged."""
        return self._actions.identity

    @property
    def unary(self) -> tuple[Action, ...]:
        """Return unary composing actions in controller-column order."""
        return tuple(action for action in self._composing if action in self._unary)

    @property
    def reencodable_unary(self) -> tuple[Action, ...]:
        """Return unary actions with complete forward enumeration."""
        return tuple(
            action
            for action in self.unary
            if "forward" in self._unary[action].directions
            and self._unary[action].forward_is_complete
        )

    def reencode(self, action: Action, terms: Sequence[Term], /) -> UnaryReencoding:
        """Encode one unary action's conclusions over its source domain."""
        try:
            rule = self._unary[action]
        except KeyError:
            known = ", ".join(held.name for held in self._unary) or "no unary action"
            msg = f"no unary rule reencodes under {action.name!r}; known: {known}"
            raise KeyError(msg) from None

        return reencode_unary_domain(
            rule,
            terms,
            codec=self._codec,
            encoder=self._encoder,
        )

    def compose(
        self,
        action: Action,
        left: Encoded,
        /,
        *,
        right: Encoded,
    ) -> LogDistribution:
        """Merge paired daughters and marginalize their parent under one action."""
        try:
            decoder = self._decoders[action]

        except KeyError:
            known = ", ".join(held.name for held in self._decoders)
            msg = f"no binary decoder composes under {action.name!r}; known: {known}"
            raise KeyError(msg) from None

        merge = self._merge

        if merge is None:
            msg = "a binary decoder has no daughter merge"
            raise RuntimeError(msg)

        return decoder.marginal_rows(
            merge.merge(left, right), root_sort=self._root_sort
        )

    @override
    def __repr__(self) -> str:
        """Return action names, decoder types and the declared merge."""
        named = ", ".join(
            f"{action.name}: {type(decoder).__name__}"
            for action, decoder in self._decoders.items()
        )

        return f"CombinatorFamily({{{named}}}, merge={type(self._merge).__name__})"


class Space:
    """Bind one represented sort to its codec, encoder and decoders."""

    __slots__ = (
        "_choices",
        "_codec",
        "_combinators",
        "_encoder",
        "_identity",
        "_max_nodes",
        "_sort",
        "_store",
    )

    def __init__(
        self,
        sort: str,
        /,
        *,
        codec: NeuralCodecView,
        choices: ChoiceIndex,
        store: TermStore,
        max_nodes: int,
        encoder: TermEncoder,
        identity: object,
        combinators: CombinatorFamily | None,
    ) -> None:
        if not sort:
            msg = "a represented space has a sort"
            raise ValueError(msg)

        self._sort = sort
        self._codec = codec
        self._choices = choices
        self._store = store
        self._max_nodes = max_nodes
        self._encoder = encoder
        self._identity = identity
        self._combinators = combinators

    @property
    def sort(self) -> str:
        """Return the represented root sort."""
        return self._sort

    @property
    def codec(self) -> NeuralCodecView:
        """Return the exact codec view used by every component in this space."""
        return self._codec

    @property
    def choices(self) -> ChoiceIndex:
        """Return the learner language's payload-aware choice index."""
        return self._choices

    @property
    def store(self) -> TermStore:
        """Return the learner language's one interned term store."""
        return self._store

    @property
    def max_nodes(self) -> int:
        """Return the declared node bound for this represented sort."""
        return self._max_nodes

    @property
    def encoder(self) -> TermEncoder:
        """Return the encoder representing terms of this space."""
        return self._encoder

    @property
    def identity(self) -> object:
        """Return the decoder reading a span as one term of this space."""
        return self._identity

    @property
    def combinators(self) -> CombinatorFamily | None:
        """Return the action decoders, or None when this space does not compose."""
        return self._combinators

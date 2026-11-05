"""Shared featurizer values and batching operations."""

from enum import Enum
from typing import TYPE_CHECKING, Final, cast, override

from lofi.data._batch import Alignment, Batch, Mask
from lofi.data._constraint import SpanConstraints
from lofi.data._join import ItemMismatchError, JoinedExample
from lofi.data._supervision import (
    Categorical,
    Example,
    ForcedChoice,
    GoldDependencies,
    GoldDerivation,
    GoldGraph,
    Relational,
    RelationalScalar,
    Scalar,
    Unlabeled,
)
from lofi.registry import ComponentPlugin, RegistryError, plugin_for

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from lofi._types import ItemId
    from lofi.data._hashed import HashedItem, ItemHasher
    from lofi.data._ports import CodeTensorizer, FeaturizerResources
    from lofi.data._supervision import SupervisionArm
    from lofi.data._values import Item
    from lofi.registry import Builder
    from lofi.spec import ConfiguredSpec, FeaturizerSpec

__all__ = [
    "DeferredSupervisionError",
    "FeaturizedBatch",
    "FeaturizerPlugin",
    "MixedSupervisionError",
    "SupervisionDisposition",
    "disposition_for",
    "featurizer_plugin",
    "require_featurizable",
]


class SupervisionDisposition(Enum):
    """How the data layer prepares a supervision arm."""

    BATCH = "batch"
    HASHED_TARGET = "hashed-target"
    PREJOIN = "prejoin"
    EXPAND = "expand"
    DEFERRED = "deferred"


_HASHED: Final = frozenset({Scalar, GoldDerivation})
_PAIRED: Final = frozenset({Relational, RelationalScalar, ForcedChoice})
_DEFERRED: Final = frozenset({Categorical, GoldGraph, GoldDependencies})


def disposition_for(arm: SupervisionArm, /) -> SupervisionDisposition:
    """Return the data-layer disposition of a supervision arm.

    Parameters
    ----------
    arm : SupervisionArm
        Supervision variant.

    Returns
    -------
    SupervisionDisposition
        The arm's preparation path.

    Raises
    ------
    ValueError
        If the arm is not part of the supervision union.
    """
    if arm is Unlabeled:
        return SupervisionDisposition.BATCH
    if arm in _HASHED:
        return SupervisionDisposition.HASHED_TARGET
    if arm in _PAIRED:
        return SupervisionDisposition.PREJOIN
    if arm is SpanConstraints:
        return SupervisionDisposition.EXPAND
    if arm in _DEFERRED:
        return SupervisionDisposition.DEFERRED

    msg = f"{arm.__name__} is not an arm of the supervision union"
    raise ValueError(msg)


class DeferredSupervisionError(ValueError):
    """An error for an arm whose target representation is not declared.

    Parameters
    ----------
    arm : SupervisionArm
        Deferred supervision arm.
    """

    def __init__(self, arm: SupervisionArm) -> None:
        super().__init__(
            f"{arm.__name__} has no declared training target representation"
        )
        self.arm = arm


def require_featurizable(arm: SupervisionArm, /) -> SupervisionDisposition:
    """Return an arm's disposition if its representation is declared.

    Parameters
    ----------
    arm : SupervisionArm
        Supervision variant.

    Returns
    -------
    SupervisionDisposition
        A nondeferred disposition.

    Raises
    ------
    DeferredSupervisionError
        If no downstream target representation is declared.
    ValueError
        If the arm is not part of the supervision union.
    """
    disposition = disposition_for(arm)
    if disposition is SupervisionDisposition.DEFERRED:
        raise DeferredSupervisionError(arm)
    return disposition


class FeaturizerPlugin[V, M](ComponentPlugin[V, "FeaturizerResources", M]):
    """Pairs a featurizer declaration with its builder and the arms it accepts.

    Parameters
    ----------
    spec : type[ConfiguredSpec[V]]
        The declaration class.
    build : Builder[V, FeaturizerResources, M]
        The builder taking the declaration's settings and a dataset's resources.
    accepts : Callable[[V], frozenset[SupervisionArm]]
        The supervision arms a featurizer with the given settings accepts,
        decided from the settings alone.
    """

    __slots__ = ("_accepts",)

    _noun = "featurizer plugin"

    def __init__(
        self,
        *,
        spec: type[ConfiguredSpec[V]],
        build: Builder[V, FeaturizerResources, M],
        accepts: Callable[[V], frozenset[SupervisionArm]],
    ) -> None:
        super().__init__(spec=spec, build=build)
        object.__setattr__(self, "_accepts", accepts)

    def accepted(self, declaration: FeaturizerSpec, /) -> frozenset[SupervisionArm]:
        """Return the supervision arms the declared featurizer accepts.

        Parameters
        ----------
        declaration : FeaturizerSpec
            A validated declaration of this plugin's variant.

        Returns
        -------
        frozenset[SupervisionArm]
            The arms its examples may carry, without building the featurizer.
        """
        return self._accepts(cast("ConfiguredSpec[V]", declaration).settings())

    @override
    def __repr__(self) -> str:
        """Return the declaration class the plugin builds from."""
        return f"FeaturizerPlugin(spec={self.spec.__name__})"


def featurizer_plugin(
    declaration: FeaturizerSpec, /
) -> FeaturizerPlugin[object, object]:
    """Return the featurizer plugin installed for one declaration.

    Parameters
    ----------
    declaration : FeaturizerSpec
        A validated featurizer declaration.

    Returns
    -------
    FeaturizerPlugin[object, object]
        The plugin the featurizer family lists under the declaration's tag.

    Raises
    ------
    RegistryError
        If the entry names something other than a featurizer plugin for the
        declaration's variant.
    """
    configured = cast("ConfiguredSpec[object]", declaration)
    plugin = plugin_for(configured)

    if not isinstance(plugin, FeaturizerPlugin):
        msg = (
            f"the featurizer entry {configured.kind!r} names {plugin!r}, "
            "not a featurizer plugin"
        )
        raise RegistryError(msg, family="featurizer")

    return cast("FeaturizerPlugin[object, object]", plugin)


class MixedSupervisionError(ValueError):
    """An error for a batch containing an unselected supervision arm.

    Parameters
    ----------
    position : int
        Position of the mismatched example.
    selected : SupervisionArm
        Arm selected by the featurizer.
    observed : SupervisionArm
        Arm carried by the example.
    """

    def __init__(
        self,
        position: int,
        selected: SupervisionArm,
        observed: SupervisionArm,
    ) -> None:
        super().__init__(
            f"example {position} carries {observed.__name__}, but the "
            f"featurizer selects {selected.__name__}"
        )
        self.position = position
        self.selected = selected
        self.observed = observed


class FeaturizedBatch[S]:
    """Padded lexical inputs and supervision for a minibatch.

    Parameters
    ----------
    batch : Batch
        Padded lexical input rows.
    hashed : tuple[HashedItem, ...]
        Hashed anchor items in example order.
    supervision : tuple[S, ...]
        Supervision values in example order.
    member_rows : tuple[tuple[int, ...], ...]
        Batch row positions used by each example, in semantic order.

    Raises
    ------
    ValueError
        If example counts differ, a row is invalid, or a first member row does
        not hold its example's hashed anchor.
    """

    __slots__ = ("_batch", "_hashed", "_member_rows", "_supervision")

    def __init__(
        self,
        batch: Batch,
        hashed: tuple[HashedItem, ...],
        supervision: tuple[S, ...],
        member_rows: tuple[tuple[int, ...], ...],
    ) -> None:
        count = len(supervision)
        if len(hashed) != count or len(member_rows) != count:
            msg = (
                f"a featurized batch holds {count} supervision values, "
                f"{len(hashed)} hashed anchors, and {len(member_rows)} row groups"
            )
            raise ValueError(msg)

        for example, rows in enumerate(member_rows):
            if not rows:
                msg = f"example {example} refers to no batch row"
                raise ValueError(msg)

            for row in rows:
                if not 0 <= row < len(batch):
                    msg = (
                        f"example {example} refers to row {row} of a batch "
                        f"holding {len(batch)} rows"
                    )
                    raise ValueError(msg)

            anchor = batch.item_ids[rows[0]]
            if anchor != hashed[example].item_id:
                msg = (
                    f"example {example} hashes anchor {hashed[example].item_id!r}, "
                    f"but its first member row holds {anchor!r}"
                )
                raise ValueError(msg)

        self._batch = batch
        self._hashed = tuple(hashed)
        self._supervision = tuple(supervision)
        self._member_rows = tuple(tuple(rows) for rows in member_rows)

    @property
    def batch(self) -> Batch:
        """Return the padded lexical inputs."""
        return self._batch

    @property
    def hashed(self) -> tuple[HashedItem, ...]:
        """Return hashed anchor items in example order."""
        return self._hashed

    @property
    def supervision(self) -> tuple[S, ...]:
        """Return supervision values in example order."""
        return self._supervision

    @property
    def member_rows(self) -> tuple[tuple[int, ...], ...]:
        """Return the batch rows used by each example."""
        return self._member_rows

    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self._supervision)


def _padding_index(hasher: ItemHasher, padding: str) -> int:
    """Resolve a declared reserved padding symbol."""
    if not hasher.lexicon.is_reserved(padding):
        msg = f"the padding symbol {padding!r} is not reserved by {hasher.lexicon.name}"
        raise ValueError(msg)
    return hasher.lexicon.index(padding)


def _selected[S](examples: Sequence[Example[S]], selection: SupervisionArm) -> None:
    """Refuse examples outside the selected supervision arm."""
    for position, example in enumerate(examples):
        observed = type(example.supervision)
        if observed is not selection:
            raise MixedSupervisionError(
                position, selection, cast("SupervisionArm", observed)
            )


def _unique_items[S](
    joined: Sequence[JoinedExample[S]],
) -> tuple[tuple[Item, ...], dict[ItemId, int]]:
    """Collect joined items once in first-use order."""
    items: list[Item] = []
    rows: dict[ItemId, int] = {}
    for joined_example in joined:
        for item in joined_example.items:
            existing = rows.get(item.item_id)
            if existing is not None:
                if items[existing] != item:
                    raise ItemMismatchError(item.item_id)
                continue
            rows[item.item_id] = len(items)
            items.append(item)
    return tuple(items), rows


def _inputs(
    items: Sequence[Item],
    hasher: ItemHasher,
    padding: str,
    tensorize: CodeTensorizer,
) -> Batch:
    """Encode and pad lexical input rows."""
    hashed = tuple(hasher(Example(item, Unlabeled())) for item in items)
    lengths = tuple(len(item.symbols) for item in hashed)
    vectors, present = tensorize(
        tuple(item.symbols for item in hashed),
        padding=_padding_index(hasher, padding),
    )
    return Batch(
        vectors,
        lengths,
        mask=Mask(present),
        alignments=tuple(
            Alignment(tuple((token, token + 1) for token in range(length)))
            for length in lengths
        ),
        item_ids=tuple(item.item_id for item in items),
    )


def _featurize[S](
    examples: Sequence[Example[S]],
    joined: Sequence[JoinedExample[S]],
    hashed: tuple[HashedItem, ...],
    hasher: ItemHasher,
    padding: str,
    tensorize: CodeTensorizer,
) -> FeaturizedBatch[S]:
    """Build one minibatch from selected, joined examples."""
    items, rows = _unique_items(joined)
    return FeaturizedBatch(
        _inputs(items, hasher, padding, tensorize),
        hashed,
        tuple(example.supervision for example in examples),
        tuple(
            tuple(rows[item.item_id] for item in joined_example.items)
            for joined_example in joined
        ),
    )


def _unary_join[S](
    examples: Sequence[Example[S]],
) -> tuple[JoinedExample[S], ...]:
    """Join unary examples to their anchor items."""
    return tuple(JoinedExample(example, (example.item,)) for example in examples)

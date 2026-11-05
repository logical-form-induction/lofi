"""Featurization for span constraints."""

from typing import TYPE_CHECKING, Final

from lofi.data._constraint import SpanConstraints
from lofi.data._supervision import Example
from lofi.data.featurizers._core import (
    FeaturizedBatch,
    FeaturizerPlugin,
    _featurize,
    _padding_index,
    _selected,
    _unary_join,
)
from lofi.spec.featurizers.span_constraint import (
    SpanConstraintFeaturizerSettings,
    SpanConstraintFeaturizerSpec,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.data._hashed import ItemHasher
    from lofi.data._ports import CodeTensorizer, FeaturizerResources
    from lofi.data._supervision import SupervisionArm

__all__ = ["SpanConstraintFeaturizer"]


class SpanConstraintFeaturizer:
    """A featurizer that expands span constraints.

    Parameters
    ----------
    hasher : ItemHasher
        Encoder for item symbols.
    padding : str
        Reserved lexical padding symbol.
    tensorize : CodeTensorizer
        Backend for padded integer rows.

    Raises
    ------
    ValueError
        If the padding symbol is not reserved.
    """

    __slots__ = ("_hasher", "_padding", "_tensorize")
    accepts: Final[frozenset[SupervisionArm]] = frozenset({SpanConstraints})
    selection: Final[SupervisionArm] = SpanConstraints

    def __init__(
        self,
        hasher: ItemHasher,
        padding: str,
        tensorize: CodeTensorizer,
    ) -> None:
        _padding_index(hasher, padding)
        self._hasher = hasher
        self._padding = padding
        self._tensorize = tensorize

    def __call__(
        self, examples: Sequence[Example[SpanConstraints[str]]], /
    ) -> FeaturizedBatch[SpanConstraints[str]]:
        """Featurize and expand a span-constraint minibatch.

        Parameters
        ----------
        examples : Sequence[Example[SpanConstraints[str]]]
            Examples in minibatch order.

        Returns
        -------
        FeaturizedBatch[SpanConstraints[str]]
            Padded lexical inputs and occurrence-expanded constraints.

        Raises
        ------
        MixedSupervisionError
            If an example carries another supervision arm.
        ItemMismatchError
            If one identifier has conflicting item content.
        """
        _selected(examples, self.selection)
        expanded = tuple(
            Example(
                example.item,
                SpanConstraints(
                    example.supervision.instances(example.item.tokens),
                    len(example.item.tokens),
                ),
            )
            for example in examples
        )
        hashed = tuple(self._hasher(example) for example in expanded)
        return _featurize(
            expanded,
            _unary_join(expanded),
            hashed,
            self._hasher,
            self._padding,
            self._tensorize,
        )


def _build(
    settings: SpanConstraintFeaturizerSettings, resources: FeaturizerResources, /
) -> SpanConstraintFeaturizer:
    """Build a span-constraint featurizer over a dataset's resources.

    Parameters
    ----------
    settings : SpanConstraintFeaturizerSettings
        The reserved padding symbol.
    resources : FeaturizerResources
        The dataset's item encoder, tensor backend and item lookup.

    Raises
    ------
    ValueError
        If the padding symbol is not reserved.
    """
    return SpanConstraintFeaturizer(
        resources.hasher,
        settings.padding,
        resources.tensorizer,
    )


def _accepts(
    settings: SpanConstraintFeaturizerSettings, /
) -> frozenset[SupervisionArm]:
    """Return the one arm a span-constraint featurizer reads.

    No setting changes it.
    """
    del settings
    return SpanConstraintFeaturizer.accepts


PLUGIN: Final = FeaturizerPlugin(
    spec=SpanConstraintFeaturizerSpec, build=_build, accepts=_accepts
)
"""The span-constraint featurizer's plugin."""

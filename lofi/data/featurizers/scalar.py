"""Featurization for scalar supervision."""

from typing import TYPE_CHECKING, Final

from lofi.data._supervision import Scalar
from lofi.data.featurizers._core import (
    FeaturizedBatch,
    FeaturizerPlugin,
    _featurize,
    _padding_index,
    _selected,
    _unary_join,
)
from lofi.spec.featurizers.scalar import ScalarFeaturizerSettings, ScalarFeaturizerSpec

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.data._hashed import ItemHasher
    from lofi.data._ports import CodeTensorizer, FeaturizerResources
    from lofi.data._supervision import Example, SupervisionArm

__all__ = ["ScalarFeaturizer"]


class ScalarFeaturizer:
    """A featurizer for scalar-response examples.

    Parameters
    ----------
    hasher : ItemHasher
        Encoder for item symbols and responses.
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
    accepts: Final[frozenset[SupervisionArm]] = frozenset({Scalar})
    selection: Final[SupervisionArm] = Scalar

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
        self, examples: Sequence[Example[Scalar]], /
    ) -> FeaturizedBatch[Scalar]:
        """Featurize a scalar-response minibatch.

        Parameters
        ----------
        examples : Sequence[Example[Scalar]]
            Examples in minibatch order.

        Returns
        -------
        FeaturizedBatch[Scalar]
            Padded lexical inputs and scalar responses.

        Raises
        ------
        MixedSupervisionError
            If an example carries another supervision arm.
        ItemMismatchError
            If one identifier has conflicting item content.
        """
        _selected(examples, self.selection)
        hashed = tuple(self._hasher(example) for example in examples)
        return _featurize(
            examples,
            _unary_join(examples),
            hashed,
            self._hasher,
            self._padding,
            self._tensorize,
        )


def _build(
    settings: ScalarFeaturizerSettings, resources: FeaturizerResources, /
) -> ScalarFeaturizer:
    """Build a scalar featurizer over a dataset's resources.

    Parameters
    ----------
    settings : ScalarFeaturizerSettings
        The reserved padding symbol.
    resources : FeaturizerResources
        The dataset's item encoder, tensor backend and item lookup.

    Raises
    ------
    ValueError
        If the padding symbol is not reserved.
    """
    return ScalarFeaturizer(
        resources.hasher,
        settings.padding,
        resources.tensorizer,
    )


def _accepts(settings: ScalarFeaturizerSettings, /) -> frozenset[SupervisionArm]:
    """Return the one arm a scalar featurizer reads, which no setting changes."""
    del settings
    return ScalarFeaturizer.accepts


PLUGIN: Final = FeaturizerPlugin(
    spec=ScalarFeaturizerSpec, build=_build, accepts=_accepts
)
"""The scalar featurizer's plugin."""

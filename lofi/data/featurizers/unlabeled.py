"""Featurization for unlabeled items."""

from typing import TYPE_CHECKING, Final

from lofi.data._supervision import Unlabeled
from lofi.data.featurizers._core import (
    FeaturizedBatch,
    FeaturizerPlugin,
    _featurize,
    _padding_index,
    _selected,
    _unary_join,
)
from lofi.spec.featurizers.unlabeled import (
    UnlabeledFeaturizerSettings,
    UnlabeledFeaturizerSpec,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.data._hashed import ItemHasher
    from lofi.data._ports import CodeTensorizer, FeaturizerResources
    from lofi.data._supervision import Example, SupervisionArm

__all__ = ["UnlabeledFeaturizer"]


class UnlabeledFeaturizer:
    """A featurizer for lexical inputs without training targets.

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
    accepts: Final[frozenset[SupervisionArm]] = frozenset({Unlabeled})
    selection: Final[SupervisionArm] = Unlabeled

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
        self, examples: Sequence[Example[Unlabeled]], /
    ) -> FeaturizedBatch[Unlabeled]:
        """Featurize an unlabeled minibatch.

        Parameters
        ----------
        examples : Sequence[Example[Unlabeled]]
            Examples in minibatch order.

        Returns
        -------
        FeaturizedBatch[Unlabeled]
            Padded lexical inputs without training targets.

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
    settings: UnlabeledFeaturizerSettings, resources: FeaturizerResources, /
) -> UnlabeledFeaturizer:
    """Build an unlabeled featurizer over a dataset's resources.

    Parameters
    ----------
    settings : UnlabeledFeaturizerSettings
        The reserved padding symbol.
    resources : FeaturizerResources
        The dataset's item encoder, tensor backend and item lookup.

    Raises
    ------
    ValueError
        If the padding symbol is not reserved.
    """
    return UnlabeledFeaturizer(
        resources.hasher,
        settings.padding,
        resources.tensorizer,
    )


def _accepts(settings: UnlabeledFeaturizerSettings, /) -> frozenset[SupervisionArm]:
    """Return the one arm an unlabeled featurizer reads, which no setting changes."""
    del settings
    return UnlabeledFeaturizer.accepts


PLUGIN: Final = FeaturizerPlugin(
    spec=UnlabeledFeaturizerSpec, build=_build, accepts=_accepts
)
"""The unlabeled featurizer's plugin."""

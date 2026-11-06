"""Featurization for gold derivations."""

from typing import TYPE_CHECKING, Final

from lofi.data._supervision import GoldDerivation
from lofi.data.featurizers._core import (
    FeaturizedBatch,
    FeaturizerPlugin,
    _featurize,
    _padding_index,
    _selected,
    _unary_join,
)
from lofi.spec.featurizers.derivation import (
    DerivationFeaturizerSettings,
    DerivationFeaturizerSpec,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.data._hashed import ItemHasher
    from lofi.data._ports import CodeTensorizer, FeaturizerResources
    from lofi.data._supervision import Example, SupervisionArm

__all__ = ["DerivationFeaturizer"]


class DerivationFeaturizer:
    """A featurizer for derivations with encoded lexical categories.

    Parameters
    ----------
    hasher : ItemHasher
        Encoder for item symbols and categories.
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
    accepts: Final[frozenset[SupervisionArm]] = frozenset({GoldDerivation})
    selection: Final[SupervisionArm] = GoldDerivation

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
        self, examples: Sequence[Example[GoldDerivation[str]]], /
    ) -> FeaturizedBatch[GoldDerivation[str]]:
        """Featurize a derivation minibatch.

        Parameters
        ----------
        examples : Sequence[Example[GoldDerivation[str]]]
            Examples in minibatch order.

        Returns
        -------
        FeaturizedBatch[GoldDerivation[str]]
            Padded lexical inputs and encoded derivations.

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
    settings: DerivationFeaturizerSettings, resources: FeaturizerResources, /
) -> DerivationFeaturizer:
    """Build a derivation featurizer over a dataset's resources.

    Parameters
    ----------
    settings : DerivationFeaturizerSettings
        The reserved padding symbol.
    resources : FeaturizerResources
        The dataset's item encoder, tensor backend and item lookup.

    Raises
    ------
    ValueError
        If the padding symbol is not reserved.
    """
    return DerivationFeaturizer(
        resources.hasher,
        settings.padding,
        resources.tensorizer,
    )


def _accepts(settings: DerivationFeaturizerSettings, /) -> frozenset[SupervisionArm]:
    """Return the one arm a derivation featurizer reads, which no setting changes."""
    del settings
    return DerivationFeaturizer.accepts


PLUGIN: Final = FeaturizerPlugin(
    spec=DerivationFeaturizerSpec, build=_build, accepts=_accepts
)
"""The derivation featurizer's plugin."""

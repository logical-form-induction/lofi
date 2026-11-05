"""Featurization for pair-valued supervision."""

from typing import TYPE_CHECKING, Final

from lofi._frozen import FrozenValue
from lofi.data._join import ItemLookup, prejoin
from lofi.data._supervision import (
    Example,
    ForcedChoice,
    Relational,
    RelationalScalar,
    Unlabeled,
)
from lofi.data.featurizers._core import (
    _PAIRED,
    FeaturizedBatch,
    FeaturizerPlugin,
    _featurize,
    _padding_index,
    _selected,
)
from lofi.registry import Registry, RegistryError
from lofi.spec.featurizers.pair import PairFeaturizerSettings, PairFeaturizerSpec

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.data._hashed import ItemHasher
    from lofi.data._ports import CodeTensorizer, FeaturizerResources
    from lofi.data._supervision import Supervision, SupervisionArm

__all__ = ["PairFeaturizer", "PairSelectionPlugin", "pair_selection"]


class PairSelectionPlugin(FrozenValue):
    """Select one pair-valued supervision arm."""

    __slots__ = ("_arm",)
    _noun = "pair-selection plugin"

    def __init__(self, arm: SupervisionArm, /) -> None:
        object.__setattr__(self, "_arm", arm)

    @property
    def arm(self) -> SupervisionArm:
        """Return the supervision arm this selection reads."""
        return self._arm


def pair_selection(name: str, /) -> SupervisionArm:
    """Return the pair-valued supervision arm installed under ``name``."""
    member = Registry[object]("pair_selection").get(name)

    if not isinstance(member, PairSelectionPlugin):
        msg = (
            f"the pair_selection entry {name!r} names {member!r}, "
            "not a pair-selection plugin"
        )
        raise RegistryError(msg, family="pair_selection")

    return member.arm


FORCED_CHOICE: Final = PairSelectionPlugin(ForcedChoice)
RELATIONAL: Final = PairSelectionPlugin(Relational)
RELATIONAL_SCALAR: Final = PairSelectionPlugin(RelationalScalar)


class PairFeaturizer:
    """A prejoining featurizer for pair-valued supervision.

    Parameters
    ----------
    hasher : ItemHasher
        Encoder for item symbols.
    padding : str
        Reserved lexical padding symbol.
    tensorize : CodeTensorizer
        Backend for padded integer rows.
    lookup : ItemLookup
        Items available for referenced identifiers.
    selection : SupervisionArm
        Pair-valued supervision arm to read.

    Raises
    ------
    ValueError
        If the selection is not pair-valued or padding is not reserved.
    """

    __slots__ = ("_hasher", "_lookup", "_padding", "_selection", "_tensorize")
    accepts: Final[frozenset[SupervisionArm]] = _PAIRED

    def __init__(
        self,
        hasher: ItemHasher,
        padding: str,
        tensorize: CodeTensorizer,
        lookup: ItemLookup,
        selection: SupervisionArm,
    ) -> None:
        if selection not in self.accepts:
            names = ", ".join(sorted(arm.__name__ for arm in self.accepts))
            msg = f"a pair featurizer selects {selection.__name__}, not {names}"
            raise ValueError(msg)

        _padding_index(hasher, padding)
        self._hasher = hasher
        self._padding = padding
        self._lookup = lookup
        self._selection = selection
        self._tensorize = tensorize

    @property
    def selection(self) -> SupervisionArm:
        """Return the selected pair-valued supervision arm."""
        return self._selection

    def __call__(
        self, examples: Sequence[Example[Supervision[str]]], /
    ) -> FeaturizedBatch[Supervision[str]]:
        """Prejoin and featurize a pair minibatch.

        Parameters
        ----------
        examples : Sequence[Example[Supervision[str]]]
            Examples in minibatch order.

        Returns
        -------
        FeaturizedBatch[Supervision[str]]
            Padded lexical inputs with member-row groups for each pair.

        Raises
        ------
        MixedSupervisionError
            If an example carries another supervision arm.
        MissingItemError
            If a referenced item is unavailable.
        ItemMismatchError
            If one identifier has conflicting item content.
        PairAnchorError
            If a forced choice is not anchored on its first alternative.
        """
        _selected(examples, self.selection)
        joined = prejoin(examples, self._lookup)
        hashed = tuple(
            self._hasher(Example(example.item, Unlabeled())) for example in examples
        )
        return _featurize(
            examples,
            joined,
            hashed,
            self._hasher,
            self._padding,
            self._tensorize,
        )


def _build(
    settings: PairFeaturizerSettings, resources: FeaturizerResources, /
) -> PairFeaturizer:
    """Build a pair featurizer over a dataset's resources.

    Parameters
    ----------
    settings : PairFeaturizerSettings
        The reserved padding symbol and the pair-valued arm to read.
    resources : FeaturizerResources
        The dataset's item encoder, tensor backend and item lookup.

    Raises
    ------
    ValueError
        If the padding symbol is not reserved.
    """
    return PairFeaturizer(
        resources.hasher,
        settings.padding,
        resources.tensorizer,
        resources.lookup,
        pair_selection(settings.selection),
    )


def _accepts(settings: PairFeaturizerSettings, /) -> frozenset[SupervisionArm]:
    """Return the one pair-valued arm the declared selection reads.

    Raises
    ------
    RegistryError
        If no pair-selection plugin is installed under the selection's name.
    """
    return frozenset({pair_selection(settings.selection)})


PLUGIN: Final = FeaturizerPlugin(
    spec=PairFeaturizerSpec, build=_build, accepts=_accepts
)
"""The pair featurizer's plugin."""

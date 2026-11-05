"""Holographic encoding of linearized terms by binding addresses to constructors."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple, final

import torch
from torch import nn

from lofi._types import ModuleSharing, SharedSubmodule
from lofi.nn._values import Encoded
from lofi.nn.policy import DevicePolicy, place
from lofi.nn.sites import (
    Access,
    ComponentKind,
    DTypeClass,
    Role,
    Site,
    SiteMap,
    SiteSpec,
)
from lofi.nn.vsa import AtomTable, bind, bundle_scale
from lofi.registry import ComponentPlugin
from lofi.spec.encoders.hrr import HrrTermEncoderSettings, HrrTermEncoderSpec
from lofi.terms import Encoding

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor

    from lofi.nn.build import BuildContext
    from lofi.nn.codec import NeuralCodecView, PlacedEncoding
    from lofi.nn.sites import ComponentId
    from lofi.terms import BatchEncoding

__all__ = [
    "HolographicCodebook",
    "HolographicEncoderSettings",
    "HrrTermEncoder",
]

_TERM_AXES: Final[tuple[str, ...]] = ("term", "width")
"""The axes of a term encoding."""

_NODE_AXES: Final[tuple[str, ...]] = ("term", "slot", "width")
"""The axes of a node encoding."""


class HolographicEncoderSettings(NamedTuple):
    """Holographic term-encoder dimensions and atom initialization.

    Attributes
    ----------
    width : int
        The output width, which is the dimension of every atom.
    magnitude : float
        Where each coordinate of an atom starts, before its noise.
    floor : float
        The smallest magnitude any coordinate of an atom is read at.
    """

    width: int
    magnitude: float
    floor: float


def _encoder_sites(component: ComponentId) -> SiteMap:
    """Return the term and node sites published by an encoder.

    Parameters
    ----------
    component : ComponentId
        The component identifier assigned by the run.
    """
    return SiteMap(
        component,
        ComponentKind.TERM_ENCODER,
        {
            Site(component, Role.TERM_ENCODING): SiteSpec(
                _TERM_AXES,
                DTypeClass.REAL,
                differentiable=True,
                access=Access.OBSERVE,
            ),
            Site(component, Role.NODE_ENCODING): SiteSpec(
                _NODE_AXES,
                DTypeClass.REAL,
                differentiable=True,
                access=Access.OBSERVE,
            ),
        },
    )


@final
class HolographicCodebook(nn.Module):
    """The atoms a holographic encoder binds: one code per symbol, one role per slot.

    Matching encoders and decoders share these symbol codes and slot roles for binding
    and unbinding. The codebook contains no other encoder state.

    Parameters
    ----------
    symbols : int
        How many constructor codes to hold, one per codec symbol.
    capacity : int
        How many address roles to hold, one per codec slot.
    width : int
        How many coordinates each atom has.
    magnitude : float
        Where each coordinate of an atom starts, before its noise.
    floor : float
        The smallest magnitude any coordinate of an atom is read at.

    Raises
    ------
    TypeError
        If a count or the width is not an integer, or the magnitude or the floor is
        not a number.
    ValueError
        If a count or the width is not positive, if the magnitude or the floor is not
        finite, or if the floor is not above zero and at most the magnitude.
    """

    def __init__(
        self,
        *,
        symbols: int,
        capacity: int,
        width: int,
        magnitude: float,
        floor: float,
    ) -> None:
        super().__init__()
        self._codes = AtomTable(
            count=symbols, dimension=width, magnitude=magnitude, floor=floor
        )
        self._roles = AtomTable(
            count=capacity, dimension=width, magnitude=magnitude, floor=floor
        )

    @property
    def codes(self) -> AtomTable:
        """Return the constructor codes, one atom per codec symbol."""
        return self._codes

    @property
    def roles(self) -> AtomTable:
        """Return the address roles, one atom per codec slot."""
        return self._roles

    @property
    def width(self) -> int:
        """Return how many coordinates each atom has."""
        return self._codes.dimension


def _child_counts(placed: PlacedEncoding) -> tuple[Tensor, Tensor]:
    """Return where each slot's parent is and how many children every slot has.

    Parameters
    ----------
    placed : PlacedEncoding
        The parent and occupancy channels of the batch being encoded.

    Returns
    -------
    tuple[Tensor, Tensor]
        Over the term and slot axes: the parent slot to add each slot's encoding into,
        at zero where there is none, and how many occupied children each slot has.
    """
    parents = placed.parents
    is_child = placed.mask & parents.ne(Encoding.NO_PARENT)
    targets = parents.clamp_min(0)
    counts = torch.zeros_like(parents).scatter_add(
        1, targets, is_child.to(parents.dtype)
    )

    return targets, counts


@final
class HrrTermEncoder(nn.Module):
    """A holographic encoder binding a term codec's addresses to its constructors.

    Each occupied slot binds its address role to its constructor code. Child encodings
    are bundled with an arity normalization before being added to the parent. The
    shared codebook supplies both roles and codes to matching decoders.

    Parameters
    ----------
    codec : NeuralCodecView
        The grammar's tensor view of its term codec.
    component : ComponentId
        The component identifier assigned by the run.
    settings : HolographicEncoderSettings
        The atom dimension and initialization.

    Raises
    ------
    TypeError
        If the width is not an integer, or the magnitude or the floor is not a number.
    ValueError
        If the width is not positive, if the magnitude or the floor is not finite, or
        if the floor is not above zero and at most the magnitude.
    """

    def __init__(
        self,
        *,
        codec: NeuralCodecView,
        component: ComponentId,
        settings: HolographicEncoderSettings,
    ) -> None:
        super().__init__()
        width = settings.width
        self._codebook = HolographicCodebook(
            symbols=codec.n_symbols,
            capacity=codec.capacity,
            width=width,
            magnitude=settings.magnitude,
            floor=settings.floor,
        )
        self._codec = codec
        self._sites = _encoder_sites(component)
        self._width = width

    @property
    def output_width(self) -> int:
        """Return the size of the feature axis, which is the atoms' dimension."""
        return self._width

    @property
    def codec(self) -> NeuralCodecView:
        """Return the codec view supplied at construction."""
        return self._codec

    @property
    def codebook(self) -> HolographicCodebook:
        """Return the codes and roles this encoder binds."""
        return self._codebook

    @property
    def shared(self) -> Mapping[str, SharedSubmodule[nn.Module]]:
        """Return the submodules another component may hold, keyed by name.

        Returns
        -------
        Mapping[str, SharedSubmodule[nn.Module]]
            The codebook under ``"codebook"``, trained and saved with its holder.
        """
        return MappingProxyType(
            {
                "codebook": SharedSubmodule(
                    self._codebook,
                    ModuleSharing(in_parameters=True, in_state_dict=True),
                )
            }
        )

    @staticmethod
    def _policy(atoms: Tensor) -> DevicePolicy:
        """Return placement derived from the constructor codes.

        Parameters
        ----------
        atoms : Tensor
            The constructor codes, over the symbol and width axes.
        """
        return DevicePolicy(atoms.device, atoms.dtype)

    def _nodes(self, codes: Tensor, placed: PlacedEncoding) -> Tensor:
        """Return the encoding at every slot of a batch.

        Parameters
        ----------
        codes : Tensor
            The constructor code at each slot, over the term, slot and width axes.
        placed : PlacedEncoding
            The parent, depth and occupancy channels of the batch.

        Returns
        -------
        Tensor
            The encoding at every slot, over the term, slot and width axes, at zero
            wherever no node occupies the slot.
        """
        occupied = placed.mask
        roles = self._codebook.roles.atoms
        nodes = bind(roles, codes).masked_fill(~occupied.unsqueeze(-1), 0.0)

        if not bool(occupied.any()):
            return nodes

        targets, counts = _child_counts(placed)
        scales = nodes.new_tensor(
            [1.0, *(bundle_scale(rank) for rank in range(1, int(counts.max()) + 1))]
        )
        divisors = scales[counts].gather(1, targets).unsqueeze(-1)
        depths = placed.depths
        index = targets.unsqueeze(-1).expand_as(nodes)

        for depth in range(int(depths[occupied].max()), 0, -1):
            at = depths.eq(depth).unsqueeze(-1)
            contribution = (nodes / divisors).masked_fill(~at, 0.0)
            nodes = nodes.scatter_add(1, index, contribution)

        return nodes

    def _hard_nodes(self, terms: BatchEncoding) -> tuple[Tensor, PlacedEncoding]:
        """Return the encodings of a batch of terms and their placed channels.

        Parameters
        ----------
        terms : BatchEncoding
            The linearized terms to encode.
        """
        atoms = self._codebook.codes.atoms
        placed = self._codec.place_batch(terms, policy=self._policy(atoms))
        codes = self._codec.symbol_rows(placed, atoms)

        return self._nodes(codes.tensor, placed), placed

    @staticmethod
    def _roots(
        nodes: Tensor, placed: PlacedEncoding, structure: BatchEncoding
    ) -> Encoded:
        """Return the encoding at each term's root.

        Parameters
        ----------
        nodes : Tensor
            The encoding at every slot, over the term, slot and width axes.
        placed : PlacedEncoding
            The parent and occupancy channels the encodings were computed from.
        structure : BatchEncoding
            The source rows, read to decide whether a term mask is needed.

        Returns
        -------
        Encoded
            One vector per term, over the term and width axes, masked only where a row
            holds no term.
        """
        roots = placed.mask & placed.parents.eq(Encoding.NO_PARENT)
        selected = nodes.masked_fill(~roots.unsqueeze(-1), 0.0).sum(dim=1)
        mask = (
            None
            if all(row.length > 0 for row in structure.rows)
            else placed.mask.any(dim=1)
        )

        return Encoded(selected, _TERM_AXES, mask)

    def encode(self, terms: BatchEncoding, /) -> Encoded:
        """Return one vector per term.

        Parameters
        ----------
        terms : BatchEncoding
            The linearized terms to encode.

        Returns
        -------
        Encoded
            The encoding at each root, over the term and width axes, masked only where
            a row holds no term.
        """
        nodes, placed = self._hard_nodes(terms)

        return self._roots(nodes, placed, terms)

    def encode_nodes(self, terms: BatchEncoding, /) -> Encoded:
        """Return one vector per occupied slot of each term.

        Parameters
        ----------
        terms : BatchEncoding
            The linearized terms to encode.

        Returns
        -------
        Encoded
            The encoding at each slot, over the term, slot and width axes, masked by
            occupancy and at zero where no node occupies the slot.
        """
        nodes, placed = self._hard_nodes(terms)

        return Encoded(nodes, _NODE_AXES, placed.mask)

    def site_map(self) -> SiteMap:
        """Return the published term and node encoding sites."""
        return self._sites


def _build(
    settings: HrrTermEncoderSettings, context: BuildContext, /
) -> HrrTermEncoder:
    """Build and place a holographic term encoder over the run's codec.

    Parameters
    ----------
    settings : HrrTermEncoderSettings
        The atom dimension and the atoms' initialization.
    context : BuildContext
        The codec, the component handle and the device policy.
    """
    return place(
        HrrTermEncoder(
            codec=context.codec,
            component=context.component,
            settings=HolographicEncoderSettings(
                width=settings.width,
                magnitude=settings.magnitude,
                floor=settings.floor,
            ),
        ),
        context.policy,
    )


PLUGIN: Final = ComponentPlugin(spec=HrrTermEncoderSpec, build=_build)
"""The holographic term encoder's plugin."""

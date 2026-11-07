"""Device-tensor views of integer term encodings."""

from typing import TYPE_CHECKING, Final, NamedTuple

import torch

from lofi._checks import CHECKS, ensure_non_negative
from lofi.nn._values import Encoded
from lofi.nn.codec._ports import CodecMetadata, TermCoding
from lofi.nn.codec._relations import DeclaredStructure
from lofi.nn.codec._values import PlacedEncoding
from lofi.nn.ports import StructuralRelation
from lofi.terms import BatchEncoding, Encoding

if TYPE_CHECKING:
    from torch import Tensor

    from lofi.nn._values import LogDistribution
    from lofi.nn.policy import DevicePolicy
    from lofi.nn.ports import StructuralDeclaration

__all__ = ["NeuralCodecView"]

#: The rank of an embedding table: one row for each symbol, one column for each
#: feature of that symbol.
_TABLE_RANK: Final[int] = 2

#: The name of the axis an embedding table's columns run along.
_WIDTH_AXIS: Final[str] = "width"


class _FixedLayout(NamedTuple):
    """The tree positions the slots of a path-addressed layout denote.

    Attributes
    ----------
    parents : tuple[int, ...]
        The slot each slot descends from, with the root sentinel at the root.
    depths : tuple[int, ...]
        The number of edges between each slot and the root.
    children : tuple[tuple[int, ...], ...]
        One row for each slot, holding that slot's children in slot order.
    """

    parents: tuple[int, ...]
    depths: tuple[int, ...]
    children: tuple[tuple[int, ...], ...]


def _fixed_layout(codec: CodecMetadata) -> _FixedLayout | None:
    """Return the tree positions of the slots, or None for term-specific structure.

    Parameters
    ----------
    codec : CodecMetadata
        The codec whose layout is being read.

    Returns
    -------
    _FixedLayout | None
        The parent, the depth and the children of every slot, or None for a layout
        that has no fixed structure.

    Raises
    ------
    ValueError
        If the skeleton width, root count, parent order, or child depths are invalid.
    """
    skeleton = codec.skeleton

    if skeleton is None:
        return None

    if skeleton.width != codec.capacity:
        msg = (
            f"the skeleton is {skeleton.width} slots wide beside a capacity of "
            f"{codec.capacity}"
        )
        raise ValueError(msg)

    children: list[list[int]] = [[] for _ in range(skeleton.width)]
    roots = 0

    for slot, parent in enumerate(skeleton.parents):
        if parent == Encoding.NO_PARENT:
            roots += 1
            _require_depth("the root", skeleton.depths[slot], 0)
            continue

        if not 0 <= parent < slot:
            msg = f"slot {slot} has parent {parent}, which does not come before it"
            raise ValueError(msg)

        _require_depth(
            f"slot {slot}", skeleton.depths[slot], skeleton.depths[parent] + 1
        )
        children[parent].append(slot)

    if roots != 1:
        msg = f"the skeleton has {roots} slots without a parent, rather than one"
        raise ValueError(msg)

    return _FixedLayout(
        skeleton.parents, skeleton.depths, tuple(tuple(row) for row in children)
    )


def _require_depth(what: str, depth: int, expected: int) -> None:
    """Raise unless a slot has the depth implied by its parent.

    Parameters
    ----------
    what : str
        The slot being checked, for the message.
    depth : int
        The depth the skeleton records there.
    expected : int
        The depth its parent implies.

    Raises
    ------
    ValueError
        If the two disagree.
    """
    if depth != expected:
        msg = f"{what} is at depth {depth} where its parent puts it at {expected}"
        raise ValueError(msg)


def _checked_legality(codec: CodecMetadata) -> tuple[tuple[bool, ...], ...]:
    """Return the per-slot legality table.

    Parameters
    ----------
    codec : CodecMetadata
        The codec whose table is being read.

    Returns
    -------
    tuple[tuple[bool, ...], ...]
        One row for each slot, one flag for each symbol.

    Raises
    ------
    ValueError
        If the table does not cover every slot, if a row does not cover every symbol, or
        if some slot allows no symbol.
    """
    table = codec.legal_symbols

    if len(table) != codec.capacity:
        msg = (
            f"the legality table covers {len(table)} slots beside a capacity of "
            f"{codec.capacity}"
        )
        raise ValueError(msg)

    for slot, row in enumerate(table):
        if len(row) != codec.n_symbols:
            msg = (
                f"slot {slot} of the legality table covers {len(row)} symbols "
                f"beside an alphabet of {codec.n_symbols}"
            )
            raise ValueError(msg)

        if not any(row):
            msg = f"no symbol may occupy slot {slot}, so no term fills it"
            raise ValueError(msg)

    return tuple(tuple(row) for row in table)


def _checked_arities(codec: CodecMetadata) -> tuple[int, ...]:
    """Return the number of arguments each symbol takes.

    Parameters
    ----------
    codec : CodecMetadata
        The codec whose operator index is being read.

    Returns
    -------
    tuple[int, ...]
        One entry for each symbol, zero at padding.

    Raises
    ------
    ValueError
        If the index assigns another number of codes than the alphabet holds.
    """
    index = codec.index

    if index.width != codec.n_symbols:
        msg = (
            f"the operator index assigns {index.width} codes beside an alphabet "
            f"of {codec.n_symbols}"
        )
        raise ValueError(msg)

    return tuple(
        0 if index.is_pad(code) else index.arity(code) for code in range(index.width)
    )


def _check_beneath(
    codec: CodecMetadata,
    legality: tuple[tuple[bool, ...], ...],
    arities: tuple[int, ...],
    fixed: _FixedLayout,
) -> None:
    """Raise unless every slot admits a symbol beneath each symbol its parent admits.

    A symbol at a slot fills as many child slots as it takes arguments, from the
    leftmost. A child slot it fills is occupied by a symbol other than pad, and a
    child slot it does not fill holds pad, so the legality table admits the one or
    the other there.

    Parameters
    ----------
    codec : CodecMetadata
        The codec whose operator index assigns the pad code.
    legality : tuple[tuple[bool, ...], ...]
        One row for each slot, one flag for each symbol.
    arities : tuple[int, ...]
        The number of arguments each symbol takes, zero at padding.
    fixed : _FixedLayout
        The tree positions the slots denote.

    Raises
    ------
    ValueError
        If a slot admits pad alone where a symbol its parent admits fills it, or
        refuses pad where a symbol its parent admits leaves it empty.
    """
    pad = codec.index.PAD

    for parent, children in enumerate(fixed.children):
        admitted = [symbol for symbol, legal in enumerate(legality[parent]) if legal]

        for symbol in admitted:
            for side, child in enumerate(children):
                filled = side < arities[symbol]
                fits = any(
                    legal and (code != pad) == filled
                    for code, legal in enumerate(legality[child])
                )

                if fits:
                    continue

                if filled:
                    msg = (
                        f"slot {child} admits pad alone, and symbol {symbol} at slot "
                        f"{parent} fills it as argument {side}"
                    )
                    raise ValueError(msg)

                msg = (
                    f"slot {child} refuses pad, and symbol {symbol} at slot {parent} "
                    f"takes {arities[symbol]} arguments, which leaves it empty"
                )
                raise ValueError(msg)


def _codes(
    rows: tuple[tuple[int, ...], ...], width: int, device: torch.device
) -> Tensor:
    """Return an integer channel of a batch as a placed tensor.

    Parameters
    ----------
    rows : tuple[tuple[int, ...], ...]
        One tuple of codes for each term, all of the same width.
    width : int
        How many slots each term occupies, which a batch of no terms has no row to
        state.
    device : torch.device
        Where to put the result.

    Returns
    -------
    Tensor
        The codes, of rank two over the term and the slot, in ``int64``.
    """
    built = torch.tensor(rows, dtype=torch.int64, device=device)

    return built.reshape(len(rows), width)


def _flags(
    rows: tuple[tuple[bool, ...], ...], width: int, device: torch.device
) -> Tensor:
    """Return a boolean channel of a batch as a placed tensor.

    Parameters
    ----------
    rows : tuple[tuple[bool, ...], ...]
        One tuple of flags for each term, all of the same width.
    width : int
        How many slots each term occupies.
    device : torch.device
        Where to put the result.

    Returns
    -------
    Tensor
        The flags, of rank two over the term and the slot, in boolean dtype.
    """
    built = torch.tensor(rows, dtype=torch.bool, device=device)

    return built.reshape(len(rows), width)


class NeuralCodecView[C: CodecMetadata = TermCoding]:
    """Validated tensor placement for term encodings.

    Parameters
    ----------
    codec : C
        The finite alphabet and layout metadata to place in tensor space.

    Raises
    ------
    ValueError
        If capacity, legality-table coverage, the operator index's width, or the
        slot skeleton is invalid, or if a slot of a path-addressed layout admits no
        symbol beneath some symbol its parent admits.
    """

    __slots__ = ("_arities", "_codec", "_fixed", "_legality")

    def __init__(self, codec: C) -> None:
        if codec.capacity < 1:
            msg = f"a codec holding no slot encodes no term: capacity {codec.capacity}"
            raise ValueError(msg)

        self._codec = codec
        self._legality = _checked_legality(codec)
        self._arities = _checked_arities(codec)
        self._fixed = _fixed_layout(codec)

        if self._fixed is not None:
            _check_beneath(codec, self._legality, self._arities, self._fixed)

    @property
    def codec(self) -> C:
        """Return the codec this view was built from, by identity."""
        return self._codec

    @property
    def n_symbols(self) -> int:
        """Return the alphabet size, padding included."""
        return self._codec.n_symbols

    @property
    def capacity(self) -> int:
        """Return the number of slots every encoding of this codec occupies."""
        return self._codec.capacity

    @property
    def arities(self) -> tuple[int, ...]:
        """Return the number of arguments each symbol takes, zero at padding."""
        return self._arities

    @property
    def path_addressed(self) -> bool:
        """Return whether each slot denotes a fixed tree path."""
        return self._fixed is not None

    @property
    def served(self) -> frozenset[StructuralRelation]:
        """Return the structural relations this layout fixes for every term.

        Returns
        -------
        frozenset[StructuralRelation]
            Every relation when each slot denotes a fixed tree path, and the position
            of a slot alone when which slot holds a child depends on the term.
        """
        if self._fixed is None:
            return frozenset({StructuralRelation.POSITION})

        return frozenset(StructuralRelation)

    def serve(self, declaration: StructuralDeclaration, /) -> DeclaredStructure[C]:
        """Return the relations a component declared, read from this layout.

        Parameters
        ----------
        declaration : StructuralDeclaration
            The relations the component reads, each with how it uses it.

        Returns
        -------
        DeclaredStructure[C]
            One accessor per relation, each refusing a relation the declaration
            does not name.

        Raises
        ------
        TypeError
            If a key of the declaration is not a relation or a value is not a
            consumption.
        ValueError
            If the declaration names a relation this layout does not serve.
        """
        return DeclaredStructure(self, declaration)

    @property
    def branching(self) -> int:
        """Return the largest number of children a slot of this layout has.

        Raises
        ------
        ValueError
            If the layout is not path addressed, where the number of children a slot has
            depends on the term written into it.
        """
        return max(len(row) for row in self._require_fixed().children)

    def _require_fixed(self) -> _FixedLayout:
        """Return the tree positions the slots denote.

        Raises
        ------
        ValueError
            If the layout is not path addressed.
        """
        if self._fixed is None:
            msg = (
                "this layout is not path addressed, so which slot holds a "
                "child depends on the term rather than on the layout"
            )
            raise ValueError(msg)

        return self._fixed

    def child_index(self, parent: int, side: int) -> int:
        """Return the slot holding one child of a slot.

        Parameters
        ----------
        parent : int
            The slot whose child is wanted.
        side : int
            Which child, counting from the leftmost.

        Returns
        -------
        int
            The slot the child occupies.

        Raises
        ------
        TypeError
            If either argument is not an integer.
        ValueError
            If either argument is negative, or if the layout is not path addressed.
        IndexError
            If the layout has no such slot, or that slot has no such child.
        """
        children = self._require_fixed().children
        ensure_non_negative("parent", parent)
        ensure_non_negative("side", side)

        if parent >= len(children):
            msg = (
                f"this layout holds {len(children)} slots, so slot {parent} is not one"
            )
            raise IndexError(msg)

        row = children[parent]

        if side >= len(row):
            msg = f"slot {parent} has {len(row)} children, so child {side} is not one"
            raise IndexError(msg)

        return row[side]

    def legality(self, *, policy: DevicePolicy) -> Tensor:
        """Return which symbols may occupy which slot, as a placed mask.

        Parameters
        ----------
        policy : DevicePolicy
            Where this run's tensors live.

        Returns
        -------
        Tensor
            True where a symbol may occupy a slot, of rank two over the slot and the
            symbol, in boolean dtype.
        """
        return _flags(self._legality, self.n_symbols, policy.device)

    def place_batch(
        self, batch: BatchEncoding, *, policy: DevicePolicy
    ) -> PlacedEncoding:
        """Return a batch of linearized terms as placed channels.

        Parameters
        ----------
        batch : BatchEncoding
            The linearized terms, all of the codec's width.
        policy : DevicePolicy
            Where this run's tensors live.

        Returns
        -------
        PlacedEncoding
            The channels, of rank two over the term and the slot.

        Raises
        ------
        ValueError
            If a row is not as wide as the codec's capacity, or if some rows have a
            payload channel and others do not.
        """
        rows = batch.rows
        width = self.capacity

        for index, row in enumerate(rows):
            if row.width != width:
                msg = (
                    f"term {index} is {row.width} slots wide beside a capacity "
                    f"of {width}"
                )
                raise ValueError(msg)

        payloads = _payload_rows(rows)

        if CHECKS:
            self._check_codes(rows)

        return PlacedEncoding(
            _codes(batch.symbols, width, policy.device),
            _codes(batch.parents, width, policy.device),
            _codes(batch.depths, width, policy.device),
            _flags(batch.mask, width, policy.device),
            None if payloads is None else _codes(payloads, width, policy.device),
        )

    def _check_codes(self, rows: tuple[Encoding, ...]) -> None:
        """Raise unless every code of every row is one this codec could produce.

        Parameters
        ----------
        rows : tuple[Encoding, ...]
            The linearized terms, already known to be of the codec's width.

        Raises
        ------
        ValueError
            If a symbol falls outside the alphabet or falls at a slot the codec forbids,
            if a parent names no slot, or if a depth is negative.
        """
        for index, row in enumerate(rows):
            for slot, symbol in enumerate(row.symbols):
                if not 0 <= symbol < self.n_symbols:
                    msg = (
                        f"term {index} slot {slot} holds symbol {symbol} beside "
                        f"an alphabet of {self.n_symbols}"
                    )
                    raise ValueError(msg)

                if not self._legality[slot][symbol]:
                    msg = (
                        f"term {index} slot {slot} holds symbol {symbol}, which "
                        f"this codec does not admit there"
                    )
                    raise ValueError(msg)

            for slot, parent in enumerate(row.parents):
                if parent != Encoding.NO_PARENT and not 0 <= parent < row.width:
                    msg = (
                        f"term {index} slot {slot} has parent {parent}, which "
                        f"names no slot"
                    )
                    raise ValueError(msg)

            for slot, depth in enumerate(row.depths):
                ensure_non_negative(f"the depth of term {index} slot {slot}", depth)

    def assemble_batch(self, symbols: Tensor, /) -> BatchEncoding:
        """Return the terms a symbol assignment denotes, as linearized encodings.

        A term fills the root and, beneath each filled slot, one child slot per
        argument of the symbol held there. Every slot the walk from the root
        reaches is occupied, and the assignment holds pad at every other slot.

        Parameters
        ----------
        symbols : Tensor
            The symbol at each slot of each term, of rank two over the term and the
            slot, in ``int64``.

        Returns
        -------
        BatchEncoding
            One encoding per term, its parents and depths those of the layout, its
            occupancy the slots the walk reaches, and no payload channel.

        Raises
        ------
        TypeError
            If the symbols are not ``int64``.
        ValueError
            If the layout is not path addressed, if the symbols are not one per slot
            of the layout, or if an assignment denotes no term: a symbol outside the
            alphabet, pad at a slot the walk reaches, or a symbol other than pad at a
            slot it does not.
        """
        fixed = self._require_fixed()

        if symbols.dtype is not torch.int64:
            msg = f"symbols hold int64 codes, not {symbols.dtype}"
            raise TypeError(msg)

        if symbols.ndim != 2 or symbols.shape[1] != self.capacity:
            msg = (
                f"an assignment holds one symbol per slot of each term, over "
                f"(term, {self.capacity}), not {tuple(symbols.shape)}"
            )
            raise ValueError(msg)

        rows: list[Encoding] = []

        for index, codes in enumerate(symbols.tolist()):
            occupied = self._occupancy(index, codes, fixed)
            rows.append(Encoding(tuple(codes), fixed.parents, fixed.depths, occupied))

        return BatchEncoding(tuple(rows))

    def _occupancy(
        self, index: int, codes: list[int], fixed: _FixedLayout
    ) -> tuple[bool, ...]:
        """Return the slots the walk from the root reaches under one assignment.

        Parameters
        ----------
        index : int
            Which term of the batch the assignment is, for the message.
        codes : list[int]
            The symbol at each slot.
        fixed : _FixedLayout
            The tree positions the slots denote.

        Returns
        -------
        tuple[bool, ...]
            True at the root and at every slot that is an argument position of a
            reached slot's symbol.

        Raises
        ------
        ValueError
            If a symbol falls outside the alphabet, if pad sits at a reached slot, or
            if a symbol other than pad sits at a slot the walk does not reach.
        """
        pad = self._codec.index.PAD
        occupied = [False] * self.capacity
        occupied[0] = True

        for slot, symbol in enumerate(codes):
            if not 0 <= symbol < self.n_symbols:
                msg = (
                    f"term {index} slot {slot} holds symbol {symbol} beside "
                    f"an alphabet of {self.n_symbols}"
                )
                raise ValueError(msg)

            if occupied[slot] and symbol == pad:
                msg = f"term {index} slot {slot} holds pad, and a node fills it"
                raise ValueError(msg)

            if not occupied[slot] and symbol != pad:
                msg = (
                    f"term {index} slot {slot} holds symbol {symbol}, and no node "
                    f"fills it"
                )
                raise ValueError(msg)

            for side, child in enumerate(fixed.children[slot]):
                occupied[child] = occupied[slot] and side < self._arities[symbol]

        return tuple(occupied)

    def symbol_rows(self, placed: PlacedEncoding, table: Tensor) -> Encoded:
        """Return the table row each slot's symbol selects.

        Parameters
        ----------
        placed : PlacedEncoding
            The channels to read symbols from.
        table : Tensor
            The embedding table, of rank two over the symbol and the width.

        Returns
        -------
        Encoded
            The rows, of rank three over the term, the slot and the width, in the
            table's dtype, masked by the batch's occupancy.

        Raises
        ------
        ValueError
            If the table is not of rank two or does not hold one row for each symbol.
        """
        _check_table(table, self.n_symbols)

        return Encoded(table[placed.symbols], (*placed.axes, _WIDTH_AXIS), placed.mask)

    def soft_symbols(self, distribution: LogDistribution, table: Tensor) -> Encoded:
        """Return the table rows a distribution over symbols mixes.

        Parameters
        ----------
        distribution : LogDistribution
            The log space distribution over symbols, normalized over its innermost axis.
        table : Tensor
            The embedding table, of rank two over the symbol and the width.

        Returns
        -------
        Encoded
            The mixed rows, of the distribution's rank with its symbol axis replaced by
            the table's width, in the product's dtype, under the distribution's mask.

        Raises
        ------
        ValueError
            If symbol normalization is absent or uses the wrong axis, the table does
            not match the alphabet, or the mask covers the symbol axis.
        """
        _check_table(table, self.n_symbols)
        over = distribution.normalized_over

        if over is None:
            msg = (
                "a distribution over symbols sums to one over the symbol axis; "
                "this one names no axis its mass sums over"
            )
            raise ValueError(msg)

        axis = distribution.axis(over)
        mask = distribution.mask

        if axis != distribution.tensor.ndim - 1:
            msg = f"the symbol axis {over!r} is at {axis} rather than innermost"
            raise ValueError(msg)

        if distribution.size(over) != self.n_symbols:
            msg = (
                f"the symbol axis {over!r} holds {distribution.size(over)} "
                f"symbols beside an alphabet of {self.n_symbols}"
            )
            raise ValueError(msg)

        if mask is not None and mask.ndim > axis:
            msg = (
                f"the mask covers the symbol axis {over!r}, so it does not "
                f"cover the mixed rows"
            )
            raise ValueError(msg)

        axes = (*distribution.axes[:axis], _WIDTH_AXIS)

        return Encoded(distribution.tensor.exp() @ table, axes, mask)


def _check_table(table: Tensor, n_symbols: int) -> None:
    """Raise unless an embedding table holds one row for each symbol.

    Parameters
    ----------
    table : Tensor
        The table to check.
    n_symbols : int
        The alphabet size the rows are indexed by.

    Raises
    ------
    ValueError
        If the table is not of rank two, or has another number of rows than there are
        symbols.
    """
    if table.ndim != _TABLE_RANK:
        msg = (
            f"an embedding table has a symbol axis and a width axis, so its "
            f"rank is {_TABLE_RANK}, not {table.ndim}"
        )
        raise ValueError(msg)

    if table.shape[0] != n_symbols:
        msg = (
            f"an embedding table holds one row for each of {n_symbols} "
            f"symbols, not {table.shape[0]}"
        )
        raise ValueError(msg)


def _payload_rows(rows: tuple[Encoding, ...]) -> tuple[tuple[int, ...], ...] | None:
    """Return each row's payload channel, or None if no row has one.

    Parameters
    ----------
    rows : tuple[Encoding, ...]
        The linearized terms of one batch.

    Returns
    -------
    tuple[tuple[int, ...], ...] | None
        One payload channel for each row, or None where no row has one.

    Raises
    ------
    ValueError
        If some rows have a payload channel and others do not.
    """
    channels = tuple(row.payloads for row in rows if row.payloads is not None)

    if not channels:
        return None

    if len(channels) != len(rows):
        msg = (
            f"{len(channels)} of {len(rows)} terms have a payload channel, so "
            f"the batch has one for some terms and none for others"
        )
        raise ValueError(msg)

    return channels

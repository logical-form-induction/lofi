"""Integer sequence encodings for terms."""

from functools import cache
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, Never, final, override

from lofi._checks import ensure_non_negative
from lofi._frozen import FrozenValue
from lofi._native import (
    EncodingOverflowError,
    EncodingShapeError,
    UnknownOperatorError,
)
from lofi.terms._ports import BatchEncoding, Encoding
from lofi.terms._values import Signature

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from lofi._native import Term, TermStore
    from lofi.terms._frontier import HoleContext
    from lofi.terms._ports import Interner, Layout, OpName, Path, Sort
    from lofi.terms._values import Operator

__all__ = [
    "Choice",
    "ChoiceIndex",
    "CompleteTree",
    "OperatorIndex",
    "PreorderArityLayout",
    "TermCodec",
    "UnreachedArityDiagnostic",
]

_MIN_BRANCHING: Final = 2


class UnreachedArityDiagnostic(NamedTuple):
    """A declared operator with an unreachable arity."""

    operator: OpName
    arity: int

    @property
    def message(self) -> str:
        """Describe the unreachable declaration."""
        return f"operator {self.operator!r} declares unreached arity {self.arity}"


class Choice(NamedTuple):
    """One operator and its optional non-term payload."""

    operator: Operator
    payload: int | None


def _choice_width(
    signature: Signature,
    op: Operator,
    depths: Mapping[Sort, int],
    free_ranks: Mapping[Sort, int],
) -> int:
    """Return how many choices one operator contributes."""
    match op.payload_sort:
        case "none":
            return 0 if op.is_hole else 1
        case "constant":
            return len(signature.vocabulary(op.sort))
        case "index":
            return depths[op.sort]
        case "free":
            return free_ranks.get(op.sort, 0)


@final
class ChoiceIndex(FrozenValue):
    """The dense payload-aware choice alphabet of one language.

    Parameters
    ----------
    signature : Signature
        The language signature whose operators and vocabularies are indexed.
    depth_bounds : Mapping[Sort, int]
        The declared maximum depth of each sort. A bound-variable operator has
        one choice per index below its sort's bound.
    free_ranks : Mapping[Sort, int] | None
        The declared number of first-occurrence free-variable ranks of each
        ranked sort. Omitted sorts admit no free-variable choice.
    """

    PAD: Final = Signature.PAD
    """The choice reserved for padding."""

    _noun = "choice index"

    __slots__ = (
        "_choices",
        "_codes",
        "_depth_bounds",
        "_families",
        "_free_ranks",
        "_legal",
        "_signature",
    )

    def __init__(
        self,
        signature: Signature,
        *,
        depth_bounds: Mapping[Sort, int],
        free_ranks: Mapping[Sort, int] | None = None,
    ) -> None:
        declared = frozenset(signature.sorts)
        received = frozenset(depth_bounds)

        if received != declared:
            missing = tuple(sort for sort in signature.sorts if sort not in received)
            extra = tuple(sorted(received - declared))
            msg = (
                "choice depth bounds do not match the signature's sorts: "
                f"missing={missing!r}, extra={extra!r}"
            )
            raise ValueError(msg)

        depths = {
            sort: ensure_non_negative(f"depth_bounds[{sort!r}]", depth_bounds[sort])
            for sort in signature.sorts
        }
        ranked_source = {} if free_ranks is None else dict(free_ranks)
        extra_ranked = tuple(sorted(set(ranked_source) - declared))

        if extra_ranked:
            msg = f"free-variable ranks name undeclared sorts: {extra_ranked!r}"
            raise ValueError(msg)

        ranked = {
            sort: ensure_non_negative(
                f"free_ranks[{sort!r}]", ranked_source.get(sort, 0)
            )
            for sort in signature.sorts
        }
        choices = [
            Choice(op, payload)
            for op in signature.operators
            for payload in (
                (None,)
                if not op.has_payload and not op.is_hole
                else range(_choice_width(signature, op, depths, ranked))
            )
        ]

        codes = {choice: code for code, choice in enumerate(choices, start=1)}
        grouped: dict[tuple[Sort, str], list[int]] = {}

        for code, choice in enumerate(choices, start=1):
            key = (choice.operator.sort, choice.operator.family)
            grouped.setdefault(key, []).append(code)

        families = {
            key: range(codes[0], codes[-1] + 1) for key, codes in grouped.items()
        }
        legal = {
            sort: (False, *(choice.operator.sort == sort for choice in choices))
            for sort in signature.sorts
        }
        object.__setattr__(self, "_signature", signature)
        object.__setattr__(self, "_depth_bounds", MappingProxyType(depths))
        object.__setattr__(self, "_free_ranks", MappingProxyType(ranked))
        object.__setattr__(self, "_choices", tuple(choices))
        object.__setattr__(self, "_codes", MappingProxyType(codes))
        object.__setattr__(self, "_families", MappingProxyType(families))
        object.__setattr__(self, "_legal", MappingProxyType(legal))

    @property
    def signature(self) -> Signature:
        """Return the signature whose choices are indexed."""
        return self._signature

    @property
    def width(self) -> int:
        """Return the number of choices, including padding."""
        return len(self._choices) + 1

    @property
    def depth_bounds(self) -> Mapping[Sort, int]:
        """Return the declared depth bound of each sort."""
        return self._depth_bounds

    @property
    def free_ranks(self) -> Mapping[Sort, int]:
        """Return the declared free-variable rank count of each sort."""
        return self._free_ranks

    @property
    def legal_choices(self) -> Mapping[Sort, tuple[bool, ...]]:
        """Return the static legal-choice mask of each sort."""
        return self._legal

    def legal(self, context: HoleContext, /) -> tuple[bool, ...]:
        """Return the choices dynamically legal at one frontier hole."""
        try:
            static = self._legal[context.sort]
        except KeyError:
            msg = f"a hole names undeclared sort {context.sort!r}"
            raise ValueError(msg) from None

        held = list(static)

        for code, admitted in enumerate(static):
            if not admitted:
                continue

            choice = self.choice(code)
            op = choice.operator
            payload = choice.payload

            if op.arity > context.remaining:
                held[code] = False
            elif op.is_bound_variable:
                held[code] = payload is not None and payload < context.in_scope.get(
                    op.sort, 0
                )
            elif op.is_free_variable:
                held[code] = payload is not None and payload <= context.free_used.get(
                    op.sort, 0
                )

        return tuple(held)

    @property
    def family_slices(self) -> Mapping[tuple[Sort, str], range]:
        """Return each sort-family's contiguous choice range."""
        return self._families

    @property
    def digest(self) -> str:
        """Return the underlying signature digest."""
        return self._signature.digest

    def index(self, op: Operator | OpName, payload: int | None = None, /) -> int:
        """Return the code of one declared operator-payload choice."""
        declared = self._signature.operator(self._signature.code(op))
        held = self._codes.get(Choice(declared, payload))

        if held is None:
            msg = (
                f"operator {declared.name!r} has no choice with payload "
                f"{payload!r} in signature {self._signature.name!r}"
            )
            raise ValueError(msg)

        return held

    def choice(self, code: int, /) -> Choice:
        """Return the operator-payload choice at a positive code."""
        if isinstance(code, bool) or not isinstance(code, int):
            msg = f"a choice code is an integer, not {code!r}"
            raise TypeError(msg)

        if not self.PAD < code < self.width:
            msg = f"choice code {code!r} lies outside [1, {self.width})"
            raise ValueError(msg)

        return self._choices[code - 1]

    def instantiate(
        self,
        store: TermStore,
        code: int,
        /,
        *arguments: Term,
    ) -> Term:
        """Instantiate one ranked choice in a caller-owned term store."""
        choice = self.choice(code)
        declared = store.signature.by_name(choice.operator.name)
        payload = choice.payload

        if declared.is_free_variable:
            if payload is None:
                msg = f"free-variable choice {code!r} has no rank"
                raise ValueError(msg)

            if arguments:
                msg = f"free-variable choice {code!r} has term arguments"
                raise ValueError(msg)

            return store.fvar(declared.sort, f"$decoded{payload}")

        if declared.is_bound_variable:
            if payload is None:
                msg = f"bound-variable choice {code!r} has no index"
                raise ValueError(msg)

            if arguments:
                msg = f"bound-variable choice {code!r} has term arguments"
                raise ValueError(msg)

            return store.bvar(declared.sort, payload)

        return store.op(declared, *arguments, payload=payload)

    def operator(self, code: int, /) -> Operator:
        """Return the operator selected by one positive choice code."""
        return self.choice(code).operator

    def payload(self, code: int, /) -> int | None:
        """Return the payload selected by one positive choice code."""
        return self.choice(code).payload

    def is_pad(self, code: int, /) -> bool:
        """Return whether a code is padding."""
        return code == self.PAD

    def arity(self, code: int, /) -> int:
        """Return the selected operator's arity."""
        return self.operator(code).arity

    def codes(self, encoding: Encoding, /) -> tuple[int, ...]:
        """Return the payload-aware choice code at every encoded slot."""
        held: list[int] = []

        for slot, symbol in enumerate(encoding.symbols):
            if symbol == self.PAD:
                held.append(self.PAD)
                continue

            op = self._signature.operator(symbol)
            payload = (
                None
                if op.payload_sort == "none"
                else None
                if encoding.payloads is None
                else encoding.payloads[slot]
            )
            held.append(self.index(op, payload))

        return tuple(held)


def _declared_reachable(
    signature: Signature, reachable: Iterable[Operator], /
) -> tuple[Operator, ...]:
    """Resolve reached operators against their signature.

    Parameters
    ----------
    signature : Signature
        The declaring signature.
    reachable : Iterable[Operator]
        Operators reached by the compiled grammar.

    Raises
    ------
    UnknownOperatorError
        If an operator is undeclared or conflicts with its declaration.
    """
    held: list[Operator] = []

    for op in reachable:
        declared = signature.operator(signature.code(op))

        if declared is not op:
            raise UnknownOperatorError(operator=op.name, signature=signature.name)

        held.append(op)

    return tuple(held)


def _family_ranges(signature: Signature) -> Mapping[tuple[Sort, str], range]:
    """Return the code range of each family within each sort.

    Parameters
    ----------
    signature : Signature
        The indexed signature, whose codes order sort-major and then by
        family, so each range is one contiguous block.
    """
    held: dict[tuple[Sort, str], list[int]] = {}

    for op in signature.operators:
        held.setdefault((op.sort, op.family), []).append(signature.code(op))

    return MappingProxyType(
        {key: range(codes[0], codes[-1] + 1) for key, codes in held.items()}
    )


class OperatorIndex:
    """An operator-to-integer code table.

    Parameters
    ----------
    signature : Signature
        The indexed signature.
    """

    PAD: Final = Signature.PAD
    """The code reserved for padding."""

    __slots__ = ("_families", "_signature")

    def __init__(self, signature: Signature) -> None:
        self._signature = signature
        self._families = _family_ranges(signature)

    @property
    def signature(self) -> Signature:
        """Return the signature whose operators this table codes."""
        return self._signature

    @property
    def width(self) -> int:
        """Return the code count, including padding."""
        return len(self._signature.operators) + 1

    @property
    def family_slices(self) -> Mapping[tuple[Sort, str], range]:
        """Return the code range of each family within each sort, read-only."""
        return self._families

    @property
    def digest(self) -> str:
        """Return the signature declaration digest as lowercase hexadecimal."""
        return self._signature.digest

    def index(self, op: Operator | OpName, /) -> int:
        """Return an operator's positive code.

        Parameters
        ----------
        op : Operator | OpName
            The operator or its name.

        Raises
        ------
        UnknownOperatorError
            If the operator is undeclared or conflicts with the declaration.
        """
        return self._signature.code(op)

    def operator(self, i: int, /) -> Operator:
        """Return the operator mapped to a code.

        Parameters
        ----------
        i : int
            The operator code.

        Raises
        ------
        UnknownOperatorError
            If the code is padding or outside the alphabet.
        """
        return self._signature.operator(i)

    def is_pad(self, i: int, /) -> bool:
        """Return whether a code is reserved for padding.

        Parameters
        ----------
        i : int
            The code to test.
        """
        return i == OperatorIndex.PAD

    def is_nullary(self, i: int, /) -> bool:
        """Return whether a code maps to a nullary operator.

        Parameters
        ----------
        i : int
            The code to test.
        """
        return (
            OperatorIndex.PAD < i < self.width
            and self._signature.operator(i).arity == 0
        )

    def arity(self, i: int, /) -> int:
        """Return the arity of the operator mapped to a code.

        Parameters
        ----------
        i : int
            The operator code.

        Raises
        ------
        UnknownOperatorError
            If the code is padding or outside the alphabet.
        """
        return self._signature.operator(i).arity

    @override
    def __repr__(self) -> str:
        """Return a rendering naming the signature and the alphabet's width."""
        return f"OperatorIndex({self._signature.name!r}, {self.width} codes)"


@final
class TermCodec:
    """A symbol index paired with one term layout.

    Parameters
    ----------
    index : OperatorIndex
        The operator index used for symbol codes.
    layout : Layout[Term, Interner, OperatorIndex]
        The term layout used for integer channels.
    """

    __slots__ = ("_index", "_layout", "_legal_symbols", "_skeleton")

    def __init__(
        self,
        index: OperatorIndex,
        layout: Layout[Term, Interner, OperatorIndex],
    ) -> None:
        self._index = index
        self._layout = layout
        self._skeleton = layout.skeleton
        self._legal_symbols = layout.static_symbols(index)

    @property
    def index(self) -> OperatorIndex:
        """Return the codec's operator index by identity."""
        return self._index

    @property
    def layout(self) -> Layout[Term, Interner, OperatorIndex]:
        """Return the codec's layout by identity."""
        return self._layout

    @property
    def n_symbols(self) -> int:
        """Return the alphabet size, including padding."""
        return self._index.width

    @property
    def capacity(self) -> int:
        """Return the fixed number of slots per encoding."""
        return self._layout.capacity

    @property
    def skeleton(self) -> Encoding | None:
        """Return fixed slot structure, or None for term-dependent structure."""
        return self._skeleton

    @property
    def legal_symbols(self) -> tuple[tuple[bool, ...], ...]:
        """Return the cached static symbol mask for each slot."""
        return self._legal_symbols

    @staticmethod
    def tree_mask(parents: Sequence[int], /) -> tuple[tuple[float, ...], ...]:
        """Return an immutable ancestor-path mask for a parent array.

        Each row is ``0.0`` at its slot and ancestors and negative infinity elsewhere.

        Parameters
        ----------
        parents : Sequence[int]
            Parent slots, with each chain ending at ``Encoding.NO_PARENT``.
        """
        width = len(parents)
        refused = float("-inf")
        rows: list[tuple[float, ...]] = []

        for slot, parent in enumerate(parents):
            row = [refused] * width
            row[slot] = 0.0
            at = parent

            while at != Encoding.NO_PARENT:
                row[at] = 0.0
                at = parents[at]

            rows.append(tuple(row))

        return tuple(rows)

    def encode(self, term: Term, /) -> Encoding:
        """Encode a term as integer channels.

        Parameters
        ----------
        term : Term
            The term to encode.

        Raises
        ------
        EncodingOverflowError
            If the term exceeds the layout capacity.
        UnknownOperatorError
            If a term operator is not declared by the index.
        """
        return self._layout.encode(term, self._index)

    def encode_batch(self, terms: Sequence[Term], /) -> BatchEncoding:
        """Encode equal-width terms as one batch.

        Parameters
        ----------
        terms : Sequence[Term]
            The terms to encode in batch order.

        Raises
        ------
        EncodingOverflowError
            If any term exceeds the layout capacity.
        UnknownOperatorError
            If any term operator is not declared by the index.
        """
        return BatchEncoding(tuple(self.encode(term) for term in terms))

    def _path_slot_maps(
        self,
        term: Term,
        /,
    ) -> tuple[Mapping[Path, int], Mapping[int, Path]]:
        """Return the two directions of one encoded term's node addressing."""
        encoding = self.encode(term)
        children: dict[int, list[int]] = {}
        roots: list[int] = []

        for slot, occupied in enumerate(encoding.occupied):
            if not occupied:
                continue

            parent = encoding.parents[slot]

            if parent == Encoding.NO_PARENT:
                roots.append(slot)
            else:
                children.setdefault(parent, []).append(slot)

        if len(roots) != 1:
            msg = (
                f"{self._layout.name} encoded {term!r} with {len(roots)} occupied roots"
            )
            raise ValueError(msg)

        path_slots: dict[Path, int] = {}
        pending: list[tuple[Term, Path, int]] = [(term, (), roots[0])]

        while pending:
            node, path, slot = pending.pop()
            path_slots[path] = slot
            child_slots = children.get(slot, ())

            if len(child_slots) != len(node.args):
                msg = (
                    f"{self._layout.name} encoded the node at {path!r} with "
                    f"{len(child_slots)} children, not {len(node.args)}"
                )
                raise ValueError(msg)

            pending.extend(
                (child, (*path, position), child_slot)
                for position, (child, child_slot) in reversed(
                    tuple(enumerate(zip(node.args, child_slots, strict=True)))
                )
            )

        slot_paths = {slot: path for path, slot in path_slots.items()}

        return MappingProxyType(path_slots), MappingProxyType(slot_paths)

    def path_to_slot(self, term: Term, /) -> Mapping[Path, int]:
        """Map each node path of a term to its encoded slot.

        Parameters
        ----------
        term : Term
            The term whose layout-specific slots are addressed.
        """
        return self._path_slot_maps(term)[0]

    def slot_to_path(self, term: Term, /) -> Mapping[int, Path]:
        """Map each occupied encoded slot of a term to its node path.

        Parameters
        ----------
        term : Term
            The term whose layout-specific slots are addressed.
        """
        return self._path_slot_maps(term)[1]

    def decode(self, encoding: Encoding, store: Interner, /) -> Term:
        """Decode integer channels through a caller-supplied store.

        Parameters
        ----------
        encoding : Encoding
            The integer channels to decode.
        store : Interner
            The store that interns the result.

        Raises
        ------
        EncodingShapeError
            If the encoding is incomplete or contains trailing symbols.
        UnknownOperatorError
            If a symbol code is not in the alphabet.
        """
        return self._layout.decode(encoding, store, self._index)


def _preorder(t: Term) -> list[tuple[Term, int, int]]:
    """Return term nodes in preorder with parent slots and depths.

    Parameters
    ----------
    t : Term
        The term to traverse.
    """
    walk: list[tuple[Term, int, int]] = []
    pending: list[tuple[Term, int, int]] = [(t, Encoding.NO_PARENT, 0)]

    while pending:
        node, parent, depth = pending.pop()
        slot = len(walk)
        walk.append((node, parent, depth))
        pending.extend((arg, slot, depth + 1) for arg in reversed(node.args))

    return walk


def _payload_at(enc: Encoding, slot: int, op: Operator) -> int | None:
    """Return a slot payload when its operator accepts one.

    Parameters
    ----------
    enc : Encoding
        The encoding being read.
    slot : int
        The slot index.
    op : Operator
        The operator stored at the slot.
    """
    if enc.payloads is None or op.payload_sort == "none":
        return None

    return enc.payloads[slot]


def _depth_first(heaps: tuple[int, ...], /) -> tuple[int, ...]:
    """Keep complete-heap positions in the preorder they were visited in."""
    return heaps


def _breadth_first(heaps: tuple[int, ...], /) -> tuple[int, ...]:
    """Order complete-heap positions level by level, left to right."""
    return tuple(sorted(heaps))


_SLOT_ORDERS: Final[
    Mapping[Literal["df", "bf"], Callable[[tuple[int, ...]], tuple[int, ...]]]
] = MappingProxyType({"df": _depth_first, "bf": _breadth_first})
"""The arrangement of preorder heap positions each declared slot order selects."""


@cache
def heap_order(
    branching: int, depth: int, order: Literal["df", "bf"], /
) -> tuple[int, ...]:
    """Return complete-heap positions in the requested slot order.

    Parameters
    ----------
    branching : int
        The number of child positions under each node.
    depth : int
        The number of complete-tree levels.
    order : Literal["df", "bf"]
        The requested depth-first or breadth-first order.
    """
    capacity = (branching**depth - 1) // (branching - 1)
    depth_first: list[int] = []
    pending = [0]

    while pending:
        heap = pending.pop()

        if heap >= capacity:
            continue

        depth_first.append(heap)
        pending.extend(branching * heap + child for child in range(branching, 0, -1))

    return _SLOT_ORDERS[order](tuple(depth_first))


@final
class CompleteTree:
    """A padded complete-tree term layout.

    Parameters
    ----------
    signature : Signature
        The declared term signature.
    reachable : Iterable[Operator]
        Operators reached by the compiled grammar.
    order : Literal["df", "bf"]
        The depth-first or breadth-first slot order.
    depth : int
        The number of complete-tree levels.

    Raises
    ------
    TypeError
        If ``depth`` is not an integer.
    ValueError
        If ``order`` is unsupported or ``depth`` is not positive.
    UnknownOperatorError
        If a reachable operator is undeclared or conflicts with its declaration.
    """

    __slots__ = (
        "_branching",
        "_capacity",
        "_depth",
        "_depths",
        "_diagnostics",
        "_heap",
        "_order",
        "_parents",
        "_positions",
        "_skeleton",
    )

    def __init__(
        self,
        signature: Signature,
        reachable: Iterable[Operator],
        *,
        order: Literal["df", "bf"] = "df",
        depth: int,
    ) -> None:
        if order not in _SLOT_ORDERS:
            msg = f"order must be 'df' or 'bf', not {order!r}"
            raise ValueError(msg)

        checked_depth = ensure_non_negative("depth", depth)

        if checked_depth < 1:
            msg = f"depth must be positive, not {checked_depth}"
            raise ValueError(msg)

        reached = _declared_reachable(signature, reachable)
        reached_arities = frozenset(op.arity for op in reached)
        checked_branching = max(
            _MIN_BRANCHING, max((op.arity for op in reached), default=0)
        )
        diagnostics = tuple(
            UnreachedArityDiagnostic(op.name, op.arity)
            for op in signature.operators
            if op.arity not in reached_arities
        )

        capacity = (checked_branching**checked_depth - 1) // (checked_branching - 1)
        heap = heap_order(checked_branching, checked_depth, order)
        positions = [0] * capacity

        for slot, at in enumerate(heap):
            positions[at] = slot

        parents: list[int] = []
        depths: list[int] = []

        for at in heap:
            if at == 0:
                parents.append(Encoding.NO_PARENT)
                depths.append(0)
                continue

            parent = positions[(at - 1) // checked_branching]
            parents.append(parent)
            depths.append(depths[parent] + 1)

        object.__setattr__(self, "_order", order)
        object.__setattr__(self, "_branching", checked_branching)
        object.__setattr__(self, "_depth", checked_depth)
        object.__setattr__(self, "_diagnostics", diagnostics)
        object.__setattr__(self, "_capacity", capacity)
        object.__setattr__(self, "_heap", heap)
        object.__setattr__(self, "_positions", tuple(positions))
        object.__setattr__(self, "_parents", tuple(parents))
        object.__setattr__(self, "_depths", tuple(depths))
        object.__setattr__(
            self,
            "_skeleton",
            Encoding(
                (OperatorIndex.PAD,) * capacity,
                tuple(parents),
                tuple(depths),
                (False,) * capacity,
            ),
        )

    @override
    def __setattr__[ValueT](self, name: str, value: ValueT) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "a complete-tree layout is immutable"

        raise AttributeError(msg)

    @property
    def name(self) -> str:
        """Return the layout name."""
        return "complete-tree"

    @property
    def order(self) -> Literal["df", "bf"]:
        """Return the slot order."""
        return self._order

    @property
    def branching(self) -> int:
        """Return the number of child positions under each node."""
        return self._branching

    @property
    def depth(self) -> int:
        """Return the number of complete-tree levels."""
        return self._depth

    @property
    def diagnostics(self) -> tuple[UnreachedArityDiagnostic, ...]:
        """Return declarations whose arity is absent from the reachable set."""
        return self._diagnostics

    @property
    def path_addressed(self) -> bool:
        """Return whether each slot denotes one fixed tree path."""
        return True

    @property
    def capacity(self) -> int:
        """Return the number of slots in the complete tree."""
        return self._capacity

    @property
    def skeleton(self) -> Encoding:
        """Return the fixed complete-tree slot structure."""
        return self._skeleton

    def static_symbols(self, index: OperatorIndex, /) -> tuple[tuple[bool, ...], ...]:
        """Return the symbols that may occupy each fixed slot.

        Parameters
        ----------
        index : OperatorIndex
            The operator index addressed by each row.
        """
        arities = tuple(op.arity for op in index.signature.operators)
        rows: list[tuple[bool, ...]] = []

        for at in self._heap:
            room = self._capacity - self._branching * at
            rows.append(
                (
                    at != 0,
                    *(
                        arity == 0 or (arity <= self._branching and arity < room)
                        for arity in arities
                    ),
                )
            )

        return tuple(rows)

    def encode(self, t: Term, index: OperatorIndex, /) -> Encoding:
        """Return the term in fixed complete-tree slots.

        Parameters
        ----------
        t : Term
            The term to encode.
        index : OperatorIndex
            The operator index used for symbol codes.

        Raises
        ------
        EncodingOverflowError
            If the term exceeds the layout capacity or branching factor.
        UnknownOperatorError
            If a term operator is not declared by the index.
        """
        symbols = [OperatorIndex.PAD] * self._capacity
        occupied = [False] * self._capacity
        payloads = [0] * self._capacity if index.signature.has_payloads else None
        pending = [(t, 0)]

        while pending:
            node, at = pending.pop()

            if at >= self._capacity or node.op.arity > self._branching:
                raise EncodingOverflowError(
                    layout=self.name, capacity=self._capacity, size=t.size
                )

            slot = self._positions[at]
            symbols[slot] = index.index(node.op)
            occupied[slot] = True

            if payloads is not None and node.payload is not None:
                payloads[slot] = node.payload

            pending.extend(
                (arg, self._branching * at + child)
                for child, arg in reversed(tuple(enumerate(node.args, start=1)))
            )

        return Encoding(
            tuple(symbols),
            self._parents,
            self._depths,
            tuple(occupied),
            None if payloads is None else tuple(payloads),
        )

    def decode(self, enc: Encoding, store: Interner, index: OperatorIndex, /) -> Term:
        """Return the term held in fixed complete-tree slots.

        Parameters
        ----------
        enc : Encoding
            The encoding to decode.
        store : Interner
            The store that interns the result.
        index : OperatorIndex
            The operator index used for symbol codes.

        Raises
        ------
        EncodingShapeError
            If the encoding is incomplete or contains trailing symbols.
        UnknownOperatorError
            If a symbol code is not in the alphabet.
        """
        visited: set[int] = set()

        def build(at: int) -> Term:
            if at >= self._capacity:
                raise EncodingShapeError(
                    layout=self.name, reason="exhausted", slot=self._capacity
                )

            slot = self._positions[at]

            if slot >= enc.width or index.is_pad(enc.symbols[slot]):
                raise EncodingShapeError(
                    layout=self.name, reason="exhausted", slot=slot
                )

            visited.add(slot)
            op = index.operator(enc.symbols[slot])

            if op.arity > self._branching:
                raise EncodingShapeError(
                    layout=self.name, reason="exhausted", slot=slot
                )

            args = tuple(
                build(self._branching * at + child) for child in range(1, op.arity + 1)
            )

            return store.op(op, *args, payload=_payload_at(enc, slot, op))

        term = build(0)

        for slot, code in enumerate(enc.symbols):
            if slot in visited or index.is_pad(code):
                continue

            raise EncodingShapeError(layout=self.name, reason="trailing", slot=slot)

        return term

    def legal_next(
        self, prefix: Sequence[int], index: OperatorIndex, /
    ) -> tuple[bool, ...]:
        """Return the symbols allowed in the next fixed tree slot.

        Parameters
        ----------
        prefix : Sequence[int]
            The operator codes emitted so far.
        index : OperatorIndex
            The operator index addressed by the result.
        """
        allowed = [False] * index.width
        slot = len(prefix)

        if slot >= self._capacity:
            return tuple(allowed)

        at = self._heap[slot]
        expected = at == 0

        if at:
            parent_at = (at - 1) // self._branching
            parent_slot = self._positions[parent_at]
            parent_code = prefix[parent_slot]
            child = (at - 1) % self._branching
            expected = not index.is_pad(parent_code) and child < index.arity(
                parent_code
            )

        if not expected:
            allowed[OperatorIndex.PAD] = True
            return tuple(allowed)

        for op in index.signature.operators:
            final_child = self._branching * at + op.arity

            if op.arity == 0 or (
                op.arity <= self._branching and final_child < self._capacity
            ):
                allowed[index.index(op)] = True

        return tuple(allowed)

    @override
    def __repr__(self) -> str:
        """Return the constructor call for this layout."""
        return (
            f"CompleteTree(order={self._order!r}, branching={self._branching}, "
            f"depth={self._depth})"
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another layout has the same parameters."""
        if type(other) is not CompleteTree:
            return NotImplemented

        return (
            self._order,
            self._branching,
            self._depth,
            self._diagnostics,
        ) == (
            other._order,
            other._branching,
            other._depth,
            other._diagnostics,
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the layout parameters."""
        return hash(
            (
                CompleteTree,
                self._order,
                self._branching,
                self._depth,
                self._diagnostics,
            )
        )


def _owed(prefix: Sequence[int], index: OperatorIndex) -> int:
    """Return the number of subterms still required by a prefix.

    Parameters
    ----------
    prefix : Sequence[int]
        The emitted operator codes.
    index : OperatorIndex
        The operator index used to read arities.
    """
    owed = 1

    for code in prefix:
        if owed == 0 or index.is_pad(code):
            return 0

        owed += index.arity(code) - 1

    return owed


class PreorderArityLayout:
    """A preorder term layout with explicit arities.

    Parameters
    ----------
    max_nodes : int
        The fixed encoding capacity.

    Raises
    ------
    TypeError
        If ``max_nodes`` is not an integer.
    ValueError
        If ``max_nodes`` is negative.
    """

    __slots__ = ("_capacity",)

    def __init__(self, max_nodes: int) -> None:
        self._capacity = ensure_non_negative("max_nodes", max_nodes)

    @property
    def name(self) -> str:
        """Return the layout's name, as it appears in a grammar fingerprint."""
        return "preorder-arity"

    @property
    def path_addressed(self) -> bool:
        """Return whether each slot has a term-independent tree path."""
        return False

    @property
    def capacity(self) -> int:
        """Return the fixed slot count."""
        return self._capacity

    @property
    def skeleton(self) -> None:
        """Return no fixed structure for term-dependent preorder slots."""
        return None

    def static_symbols(self, index: OperatorIndex, /) -> tuple[tuple[bool, ...], ...]:
        """Return the symbols that may occupy each preorder slot.

        Parameters
        ----------
        index : OperatorIndex
            The operator index addressed by each row.
        """
        arities = tuple(op.arity for op in index.signature.operators)
        rows: list[tuple[bool, ...]] = []

        for slot in range(self._capacity):
            room = self._capacity - slot - 1
            rows.append((slot != 0, *(arity <= room for arity in arities)))

        return tuple(rows)

    def encode(self, t: Term, index: OperatorIndex, /) -> Encoding:
        """Return a term as padded preorder channels.

        Parameters
        ----------
        t : Term
            The term to encode.
        index : OperatorIndex
            The operator index used for symbol codes.

        Raises
        ------
        EncodingOverflowError
            If the term exceeds the layout capacity.
        UnknownOperatorError
            If a term operator is not declared by the index.
        """
        if t.size > self._capacity:
            raise EncodingOverflowError(
                layout=self.name, capacity=self._capacity, size=t.size
            )

        symbols = [OperatorIndex.PAD] * self._capacity
        parents = [Encoding.NO_PARENT] * self._capacity
        depths = [0] * self._capacity
        occupied = [False] * self._capacity
        payloads = [0] * self._capacity if index.signature.has_payloads else None

        for slot, (node, parent, depth) in enumerate(_preorder(t)):
            symbols[slot] = index.index(node.op)
            parents[slot] = parent
            depths[slot] = depth
            occupied[slot] = True

            if payloads is not None and node.payload is not None:
                payloads[slot] = node.payload

        return Encoding(
            tuple(symbols),
            tuple(parents),
            tuple(depths),
            tuple(occupied),
            None if payloads is None else tuple(payloads),
        )

    def decode(self, enc: Encoding, store: Interner, index: OperatorIndex, /) -> Term:
        """Return the interned term represented by a preorder encoding.

        Parameters
        ----------
        enc : Encoding
            The encoding to decode.
        store : Interner
            The store that interns the result.
        index : OperatorIndex
            The operator index used for symbol codes.

        Raises
        ------
        EncodingShapeError
            If the sequence is incomplete or contains trailing symbols.
        UnknownOperatorError
            If a symbol code is not in the alphabet.
        """
        frames: list[tuple[Operator, int | None, list[Term]]] = []
        done: Term | None = None

        for slot, code in enumerate(enc.symbols):
            if done is not None:
                if index.is_pad(code):
                    continue

                raise EncodingShapeError(layout=self.name, reason="trailing", slot=slot)

            if index.is_pad(code):
                raise EncodingShapeError(
                    layout=self.name, reason="exhausted", slot=slot
                )

            op = index.operator(code)
            payload = _payload_at(enc, slot, op)

            if op.arity:
                frames.append((op, payload, []))

                continue

            built: Term | None = store.op(op, payload=payload)

            while built is not None and frames:
                head, load, taken = frames[-1]
                taken.append(built)
                built = None

                if len(taken) == head.arity:
                    frames.pop()
                    built = store.op(head, *taken, payload=load)

            done = built

        if done is None:
            raise EncodingShapeError(
                layout=self.name, reason="exhausted", slot=enc.width
            )

        return done

    def legal_next(
        self, prefix: Sequence[int], index: OperatorIndex, /
    ) -> tuple[bool, ...]:
        """Return one admissibility flag per symbol after a partial encoding.

        Parameters
        ----------
        prefix : Sequence[int]
            The operator codes emitted so far.
        index : OperatorIndex
            The operator index addressed by the mask.
        """
        allowed = [False] * index.width
        slot = len(prefix)

        if slot >= self._capacity:
            return tuple(allowed)

        owed = _owed(prefix, index)

        if owed == 0:
            allowed[OperatorIndex.PAD] = True

            return tuple(allowed)

        room = self._capacity - slot - 1

        for op in index.signature.operators:
            if owed - 1 + op.arity <= room:
                allowed[index.index(op)] = True

        return tuple(allowed)

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this layout."""
        return f"PreorderArityLayout({self._capacity})"

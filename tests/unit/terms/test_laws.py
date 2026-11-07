from hashlib import sha256
from itertools import count
from types import FunctionType
from typing import TYPE_CHECKING, Final, NamedTuple, override

import pytest
from hypothesis import strategies as st

from lofi._checks import LawViolationError
from lofi.terms import (
    EncodingOverflowError,
    ForeignTermError,
    NotLocallyClosedError,
    laws,
)
from lofi.terms._ports import Encoding
from lofi.terms.laws import (
    LAYOUT_LAWS,
    MATCHING_LAWS,
    PRESENTATION_LAWS,
    SIGNATURE_INVARIANTS,
    TERM_LAWS,
    AlphabetFixture,
    BindingFixture,
    LayoutFixture,
    MatchingFixture,
    PresentationFixture,
    TermFixture,
    a_foreign_argument_is_refused,
    alpha_variants_are_one_node,
    check_layout_laws,
    check_matching_laws,
    check_presentation_laws,
    check_signature_invariants,
    check_term_laws,
    closing_undoes_opening,
    codes_are_a_bijection,
    codes_are_determined_by_the_declaration,
    codes_group_by_sort_then_family,
    derived_fields_agree_with_their_definitions,
    encoding_refuses_to_truncate,
    equivalence_coarsens_identity,
    equivalence_is_an_equivalence,
    equivalence_survives_construction,
    erasure_is_a_congruence,
    erasure_is_idempotent,
    hashing_is_stable,
    key_is_a_section,
    key_is_idempotent,
    legal_next_admits_the_next_symbol,
    mask_is_complete,
    mask_is_sound,
    nothing_refused_is_reachable,
    one_component_by_identity,
    padding_belongs_to_no_operator,
    path_addressing_is_honest,
    round_trip_is_identity,
    structure_is_identity,
    structure_is_store_independent,
    substitution_cancels_abstraction,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

_EXAMPLES: Final = 50
_CAPACITY: Final = 7
_PAD: Final = 0
_ERASURES: Final = {"Sdcl": "S"}
_ARITIES: Final = {"/": 2, "NP": 0, "S": 0, "Sdcl": 0}
_BINDS: Final = {"lam": 1}
_PARENTS: Final = tuple(
    Encoding.NO_PARENT if slot == 0 else (slot - 1) // 2 for slot in range(_CAPACITY)
)
_DEPTHS: Final = tuple((slot + 1).bit_length() - 1 for slot in range(_CAPACITY))


class Derived(NamedTuple):
    """Recomputed term summaries."""

    size: int
    maxidx: int
    free: int


class Node:
    """A test operator, payload, and arguments."""

    __slots__ = (
        "args",
        "core",
        "free",
        "generation",
        "maxidx",
        "op",
        "payload",
        "size",
    )

    def __init__(
        self,
        op: str,
        args: tuple[Node, ...],
        payload: int | None,
        generation: int,
        derived: Derived,
    ) -> None:
        """Hold an operator with its arguments, and the summaries of them."""
        self.op = op
        self.args = args
        self.payload = payload
        self.generation = generation
        self.size = derived.size
        self.maxidx = derived.maxidx
        self.free = derived.free
        self.core: Node = self

    @property
    def is_closed(self) -> bool:
        """Return whether the term leaves no bound index open."""
        return self.maxidx == 0

    @override
    def __repr__(self) -> str:
        """Render an atom as its name and an application in parentheses."""
        if self.payload is not None:
            return f"{self.op}{self.payload}"

        if not self.args:
            return self.op

        return f"({self.op} {' '.join(repr(arg) for arg in self.args)})"


class Store:
    """A hash-consing term store.

    Parameters
    ----------
    generation
        The store generation.
    parent
        The optional parent store.
    """

    def __init__(self, generation: int = 0, parent: Store | None = None) -> None:
        """Initialize empty interning tables."""
        self._nodes: dict[tuple[str, int | None, tuple[Node, ...]], Node] = {}
        self._generation = generation
        self._ancestry = frozenset({generation}) | (
            parent.ancestry if parent is not None else frozenset()
        )

    @property
    def ancestry(self) -> frozenset[int]:
        """Return the generations this store may build with."""
        return self._ancestry

    def op(self, op: str, *args: Node, payload: int | None = None) -> Node:
        """Return the node for an operator applied to arguments.

        Parameters
        ----------
        op
            The operator name.
        args
            The term arguments.
        payload
            The optional payload.

        Raises
        ------
        ForeignTermError
            If an argument comes from an unrelated store.
        """
        self.guard(op, args)
        key = (op, payload, args)
        node = self._nodes.get(key)

        if node is None:
            node = self.intern(op, args, payload)
            self._nodes[key] = node

        return node

    def guard(self, op: str, args: tuple[Node, ...]) -> None:
        """Reject arguments from unrelated stores.

        Parameters
        ----------
        op
            The operator name.
        args
            The proposed arguments.

        Raises
        ------
        ForeignTermError
            If an argument comes from an unrelated store.
        """
        for position, arg in enumerate(args):
            if arg.generation in self._ancestry:
                continue

            raise ForeignTermError(
                operator=op,
                position=position,
                expected=self._generation,
                received=arg.generation,
            )

    def intern(self, op: str, args: tuple[Node, ...], payload: int | None) -> Node:
        """Return a node with its summaries and erased twin.

        Parameters
        ----------
        op
            The operator name.
        args
            The term arguments.
        payload
            The optional payload.
        """
        node = Node(op, args, payload, self._generation, self.derive(op, args, payload))
        erased = _ERASURES.get(op, op)

        if erased != op or any(arg.core is not arg for arg in args):
            node.core = self.op(erased, *(arg.core for arg in args), payload=payload)

        return node

    def derive(self, op: str, args: tuple[Node, ...], payload: int | None) -> Derived:
        """Return a node's derived summaries.

        Parameters
        ----------
        op
            The operator name.
        args
            The term arguments.
        payload
            The optional payload.
        """
        index = 0 if payload is None else payload
        deepest = max((arg.maxidx for arg in args), default=0)

        return Derived(
            size=1 + sum(arg.size for arg in args),
            maxidx=index + 1 if op == "bvar" else max(0, deepest - _BINDS.get(op, 0)),
            free=1 << index if op == "fvar" else _union(args),
        )


def _union(args: tuple[Node, ...]) -> int:
    """Return the union of argument free-variable masks.

    Parameters
    ----------
    args
        The term arguments.
    """
    mask = 0

    for arg in args:
        mask |= arg.free

    return mask


class Codes:
    """An operator code table with zero reserved for padding.

    Parameters
    ----------
    arities
        Operator arities keyed by name.
    """

    first = 1
    """The code assigned to the lowest operator."""

    def __init__(self, arities: Mapping[str, int]) -> None:
        """Assign each operator a code, in the order this table orders them."""
        self._arities = dict(arities)
        ordered = self.order(self._arities)
        self._codes = {op: code for code, op in enumerate(ordered, start=self.first)}
        self._operators = {code: op for op, code in self._codes.items()}

    def order(self, arities: Mapping[str, int]) -> tuple[str, ...]:
        """Return operators in code order.

        Parameters
        ----------
        arities
            Operator arities keyed by name.
        """
        return tuple(sorted(arities, key=lambda op: (arities[op], op)))

    @property
    def width(self) -> int:
        """Return the number of codes, padding included."""
        return len(self._codes) + 1

    @property
    def operators(self) -> tuple[str, ...]:
        """Return the operator names, in code order."""
        return tuple(self._codes)

    def code(self, op: str) -> int:
        """Return an operator's code.

        Parameters
        ----------
        op
            The operator name.
        """
        return self._codes[op]

    def operator(self, code: int) -> str:
        """Return the operator mapped to a code.

        Parameters
        ----------
        code
            The operator code.
        """
        return self._operators[code]

    def arity(self, op: str) -> int:
        """Return an operator's arity.

        Parameters
        ----------
        op
            The operator name.
        """
        return self._arities[op]


class Declared(Codes):
    """A code table in declaration order."""

    @override
    def order(self, arities: Mapping[str, int]) -> tuple[str, ...]:
        """Return the operators in declaration order."""
        return tuple(arities)


class ByName(Codes):
    """A code table in name order."""

    @override
    def order(self, arities: Mapping[str, int]) -> tuple[str, ...]:
        """Return the operators in name order."""
        return tuple(sorted(arities))


class PadTaken(Codes):
    """A code table that assigns padding to an operator."""

    first = 0


class Gapped(Codes):
    """A code table with an unused code above padding."""

    first = 2


def digest(table: Codes) -> str:
    """Return a code table's declaration digest.

    Parameters
    ----------
    table
        The operator code table.
    """
    rows = "".join(
        f"{table.code(op)}:{op}:{table.arity(op)};" for op in sorted(table.operators)
    )

    return sha256(rows.encode()).hexdigest()


def size(term: Node) -> int:
    """Return a term's recursive node count.

    Parameters
    ----------
    term
        The term to count.
    """
    return 1 + sum(size(arg) for arg in term.args)


def fits_a_tree(term: Node) -> bool:
    """Return whether a term fits the complete-tree layout.

    Parameters
    ----------
    term
        The term to test.
    """
    pending = [(term, 0)]

    while pending:
        node, slot = pending.pop()

        if slot >= _CAPACITY:
            return False

        pending.extend((arg, 2 * slot + 1 + at) for at, arg in enumerate(node.args))

    return True


def fits_a_walk(term: Node) -> bool:
    """Return whether a term fits the preorder layout.

    Parameters
    ----------
    term
        The term to test.
    """
    return size(term) <= _CAPACITY


class Interning:
    """Identity matching for terms."""

    @property
    def name(self) -> str:
        """Name the relation."""
        return "identity"

    @property
    def has_key(self) -> bool:
        """Report that a class has a representative."""
        return True

    def key(self, t: Node, /) -> Node:
        """Return the term as its own representative."""
        return t

    def equivalent(self, a: Node, b: Node, /) -> bool:
        """Compare two terms by identity."""
        return a is b


class FeatureErasure:
    """Feature-erasure matching for categories.

    Parameters
    ----------
    store
        The representative interning store.
    """

    def __init__(self, store: Store) -> None:
        """Bind the relation to the store it interns representatives in."""
        self._store = store

    @property
    def name(self) -> str:
        """Name the relation."""
        return "feature-erasure"

    @property
    def has_key(self) -> bool:
        """Report that a class has a representative."""
        return True

    def _erase(self, term: Node) -> Node:
        """Return the term with every feature removed.

        Parameters
        ----------
        term
            The term to erase.
        """
        args = tuple(self._erase(arg) for arg in term.args)

        return self._store.op(_ERASURES.get(term.op, term.op), *args)

    def key(self, t: Node, /) -> Node:
        """Return the featureless term standing for the class."""
        return self._erase(t)

    def equivalent(self, a: Node, b: Node, /) -> bool:
        """Compare two terms once their features are gone."""
        return self._erase(a) is self._erase(b)


class SameSize:
    """Node-count matching without representatives."""

    @property
    def name(self) -> str:
        """Name the relation."""
        return "same-size"

    @property
    def has_key(self) -> bool:
        """Report that a class has no representative."""
        return False

    def key(self, t: Node, /) -> Node:
        """Raise because no representative exists.

        Parameters
        ----------
        t
            The unused term.

        Raises
        ------
        NotImplementedError
            Always raised because this relation has no representatives.
        """
        del t

        message = "a relation on node counts alone has no representatives"

        raise NotImplementedError(message)

    def equivalent(self, a: Node, b: Node, /) -> bool:
        """Compare two terms by node count."""
        return size(a) == size(b)


class OneDirection(Interning):
    """Asymmetric feature-erasure matching."""

    @override
    def equivalent(self, a: Node, b: Node, /) -> bool:
        """Compare by identity, and relate a featured atom to a bare one."""
        return a is b or (a.op == "Sdcl" and b.op == "S")


class KeyMissesTheClass(FeatureErasure):
    """Feature erasure with distinct keys for equivalent categories."""

    @override
    def key(self, t: Node, /) -> Node:
        """Return the term itself, feature and all."""
        return t


class RootErasure:
    """Outermost-only feature erasure.

    Parameters
    ----------
    store
        The representative interning store.
    """

    def __init__(self, store: Store) -> None:
        """Bind the relation to the store it interns representatives in."""
        self._store = store

    @property
    def name(self) -> str:
        """Name the relation."""
        return "root-erasure"

    @property
    def has_key(self) -> bool:
        """Report that a class has a representative."""
        return True

    def key(self, t: Node, /) -> Node:
        """Return the term with its outermost operator's feature dropped."""
        return self._store.op(_ERASURES.get(t.op, t.op), *t.args)

    def equivalent(self, a: Node, b: Node, /) -> bool:
        """Compare the representatives of two terms."""
        return self.key(a) is self.key(b)


class KeyLeavesTheClass(Interning):
    """Identity matching with an invalid representative.

    Parameters
    ----------
    store
        The store providing the fixed representative.
    """

    def __init__(self, store: Store) -> None:
        """Hold the term every class is said to be represented by."""
        self._canonical = store.op("NP")

    @override
    def key(self, t: Node, /) -> Node:
        """Return the fixed representative."""
        del t

        return self._canonical


class KeyGrowsTheTerm(Interning):
    """Identity matching whose representative grows.

    Parameters
    ----------
    store
        The store used to wrap representatives.
    """

    def __init__(self, store: Store) -> None:
        """Hold the store and the argument the wrapping is built with."""
        self._store = store
        self._filler = store.op("NP")

    @override
    def key(self, t: Node, /) -> Node:
        """Return the term under one more operator."""
        return self._store.op("/", t, self._filler)


class CompleteTree:
    """A path-addressed complete-binary-tree layout."""

    @property
    def name(self) -> str:
        """Name the layout."""
        return "complete-tree"

    @property
    def path_addressed(self) -> bool:
        """Return true for every slot."""
        return True

    @property
    def capacity(self) -> int:
        """Return the number of slots every encoding occupies."""
        return _CAPACITY

    @property
    def skeleton(self) -> Encoding:
        """Return the fixed binary-tree slot structure."""
        return Encoding(
            (_PAD,) * _CAPACITY,
            _PARENTS,
            _DEPTHS,
            (False,) * _CAPACITY,
        )

    def static_symbols(self, index: Codes, /) -> tuple[tuple[bool, ...], ...]:
        """Return a permissive static symbol mask."""
        return ((True,) * index.width,) * _CAPACITY

    def symbol(self, node: Node, index: Codes) -> int:
        """Return the code for a node's operator.

        Parameters
        ----------
        node
            The node to encode.
        index
            The operator code table.
        """
        return index.code(node.op)

    def encode(self, t: Node, index: Codes, /) -> Encoding:
        """Return a term in path-addressed slots.

        Parameters
        ----------
        t
            The term to encode.
        index
            The operator code table.

        Raises
        ------
        EncodingOverflowError
            If the term exceeds the fixed capacity.
        """
        symbols = [_PAD] * _CAPACITY
        occupied = [False] * _CAPACITY
        pending = [(t, 0)]

        while pending:
            node, slot = pending.pop()

            if slot >= _CAPACITY:
                raise EncodingOverflowError(
                    layout=self.name, capacity=_CAPACITY, size=size(t)
                )

            symbols[slot] = self.symbol(node, index)
            occupied[slot] = True
            pending.extend((arg, 2 * slot + 1 + at) for at, arg in enumerate(node.args))

        return Encoding(tuple(symbols), _PARENTS, _DEPTHS, tuple(occupied))

    def decode(self, enc: Encoding, store: Store, index: Codes, /) -> Node:
        """Return the term stored in path-addressed slots.

        Parameters
        ----------
        enc
            The encoding to decode.
        store
            The destination store.
        index
            The operator code table.
        """

        def build(slot: int) -> Node:
            op = index.operator(enc.symbols[slot])
            args = (build(2 * slot + 1 + at) for at in range(index.arity(op)))

            return store.op(op, *args)

        return build(0)

    def legal_next(self, prefix: Sequence[int], index: Codes, /) -> tuple[bool, ...]:
        """Return symbols allowed after a partial encoding.

        Parameters
        ----------
        prefix
            The emitted symbol codes.
        index
            The operator code table.
        """
        allowed = [False] * index.width
        slot = len(prefix)

        if slot >= _CAPACITY:
            return tuple(allowed)

        if not _is_expected(prefix, slot, index):
            allowed[_PAD] = True

            return tuple(allowed)

        for op in index.operators:
            arity = index.arity(op)

            if not arity or 2 * slot + arity < _CAPACITY:
                allowed[index.code(op)] = True

        return tuple(allowed)


def _is_expected(prefix: Sequence[int], slot: int, index: Codes) -> bool:
    """Return whether a tree slot should contain a node.

    Parameters
    ----------
    prefix
        The emitted symbol codes.
    slot
        The slot to test.
    index
        The operator code table.
    """
    if not slot:
        return True

    parent = (slot - 1) // 2
    code = prefix[parent]

    if code == _PAD:
        return False

    return slot - (2 * parent + 1) < index.arity(index.operator(code))


def _walk(term: Node) -> list[tuple[Node, int, int]]:
    """Return preorder nodes with parent slots and depths.

    Parameters
    ----------
    term
        The term to traverse.
    """
    walk: list[tuple[Node, int, int]] = []

    def visit(node: Node, parent: int, depth: int) -> None:
        slot = len(walk)
        walk.append((node, parent, depth))

        for arg in node.args:
            visit(arg, slot, depth + 1)

    visit(term, Encoding.NO_PARENT, 0)

    return walk


class Preorder:
    """A preorder term layout."""

    @property
    def name(self) -> str:
        """Name the layout."""
        return "preorder"

    @property
    def path_addressed(self) -> bool:
        """Report that a slot denotes no path of its own."""
        return False

    @property
    def capacity(self) -> int:
        """Return the number of slots every encoding occupies."""
        return _CAPACITY

    @property
    def skeleton(self) -> None:
        """Return no fixed structure for preorder slots."""
        return None

    def static_symbols(self, index: Codes, /) -> tuple[tuple[bool, ...], ...]:
        """Return a permissive static symbol mask."""
        return ((True,) * index.width,) * _CAPACITY

    def encode(self, t: Node, index: Codes, /) -> Encoding:
        """Return a term encoded in preorder.

        Parameters
        ----------
        t
            The term to encode.
        index
            The operator code table.

        Raises
        ------
        EncodingOverflowError
            If the term exceeds the fixed capacity.
        """
        walk = _walk(t)

        if len(walk) > _CAPACITY:
            raise EncodingOverflowError(
                layout=self.name, capacity=_CAPACITY, size=len(walk)
            )

        symbols = [_PAD] * _CAPACITY
        parents = [Encoding.NO_PARENT] * _CAPACITY
        depths = [0] * _CAPACITY
        occupied = [False] * _CAPACITY

        for slot, (node, parent, depth) in enumerate(walk):
            symbols[slot] = index.code(node.op)
            parents[slot] = parent
            depths[slot] = depth
            occupied[slot] = True

        return Encoding(tuple(symbols), tuple(parents), tuple(depths), tuple(occupied))

    def decode(self, enc: Encoding, store: Store, index: Codes, /) -> Node:
        """Return the term stored in preorder.

        Parameters
        ----------
        enc
            The encoding to decode.
        store
            The destination store.
        index
            The operator code table.
        """
        codes = iter(enc.symbols)

        def build() -> Node:
            op = index.operator(next(codes))
            args = (build() for _ in range(index.arity(op)))

            return store.op(op, *args)

        return build()

    def legal_next(self, prefix: Sequence[int], index: Codes, /) -> tuple[bool, ...]:
        """Return symbols allowed after a partial encoding.

        Parameters
        ----------
        prefix
            The emitted symbol codes.
        index
            The operator code table.
        """
        allowed = [False] * index.width
        slot = len(prefix)

        if slot >= _CAPACITY:
            return tuple(allowed)

        owed = _owed(prefix, index)

        if not owed:
            allowed[_PAD] = True

            return tuple(allowed)

        for op in index.operators:
            if owed - 1 + index.arity(op) <= _CAPACITY - slot - 1:
                allowed[index.code(op)] = True

        return tuple(allowed)


def _owed(prefix: Sequence[int], index: Codes) -> int:
    """Return the subterms still required by a prefix.

    Parameters
    ----------
    prefix
        The emitted symbol codes.
    index
        The operator code table.
    """
    owed = 1

    for code in prefix:
        if code == _PAD:
            return 0

        owed += index.arity(index.operator(code)) - 1

    return owed


class Featureless(CompleteTree):
    """A layout that drops operator features."""

    @override
    def symbol(self, node: Node, index: Codes) -> int:
        """Return the code of the operator with its feature dropped."""
        return index.code(_ERASURES.get(node.op, node.op))


class Truncating(CompleteTree):
    """A layout that truncates nodes beyond capacity."""

    @override
    def encode(self, t: Node, index: Codes, /) -> Encoding:
        """Return the term written into the slots it has, and no complaint."""
        symbols = [_PAD] * _CAPACITY
        occupied = [False] * _CAPACITY
        pending = [(t, 0)]

        while pending:
            node, slot = pending.pop()

            if slot >= _CAPACITY:
                continue

            symbols[slot] = self.symbol(node, index)
            occupied[slot] = True
            pending.extend((arg, 2 * slot + 1 + at) for at, arg in enumerate(node.args))

        return Encoding(tuple(symbols), _PARENTS, _DEPTHS, tuple(occupied))


class PhantomNode(CompleteTree):
    """A layout with an unconditionally occupied final slot."""

    @override
    def encode(self, t: Node, index: Codes, /) -> Encoding:
        """Return the encoding with one more slot marked than was written."""
        written = super().encode(t, index)
        occupied = (*written.occupied[:-1], True)

        return Encoding(written.symbols, written.parents, written.depths, occupied)


class LosesALeaf(CompleteTree):
    """A layout that omits the deepest occupied node."""

    @override
    def encode(self, t: Node, index: Codes, /) -> Encoding:
        """Return the encoding with its last occupied slot unmarked."""
        written = super().encode(t, index)

        if written.length < 2:
            return written

        last = max(slot for slot in range(written.width) if written.occupied[slot])
        occupied = tuple(
            held and slot != last for slot, held in enumerate(written.occupied)
        )

        return Encoding(written.symbols, written.parents, written.depths, occupied)


class Addressed(Preorder):
    """A preorder layout claiming path-addressed slots."""

    @property
    @override
    def path_addressed(self) -> bool:
        """Return true for every slot."""
        return True


class Deaf(CompleteTree):
    """A layout that rejects every next symbol."""

    @override
    def legal_next(self, prefix: Sequence[int], index: Codes, /) -> tuple[bool, ...]:
        """Reject every symbol."""
        del prefix

        return (False,) * index.width


class Shape(NamedTuple):
    """A layout capacity and order.

    Parameters
    ----------
    capacity
        The layout capacity.
    order
        The slot order.
    """

    capacity: int
    order: str


class Sized:
    """An object with a stored declaration.

    Parameters
    ----------
    shape
        The stored layout declaration.
    """

    def __init__(self, shape: Shape) -> None:
        """Keep the declaration this object was built from."""
        self._shape = shape

    def to_spec(self) -> Shape:
        """Return the declaration this object was built from."""
        return self._shape


class Restless:
    """An object whose declaration changes after each lookup."""

    def __init__(self) -> None:
        """Initialize the call count."""
        self._asked = 0

    def to_spec(self) -> Shape:
        """Return a declaration naming the number of questions so far."""
        self._asked += 1

        return Shape(self._asked, "complete-tree")


def rebuild(spec: Shape) -> Sized:
    """Return an object rebuilt from a shape.

    Parameters
    ----------
    spec
        The shape declaration.
    """
    return Sized(spec)


def rebuild_forgetting_the_order(spec: Shape) -> Sized:
    """Return a rebuild that discards the declared order.

    Parameters
    ----------
    spec
        The shape declaration.
    """
    return Sized(Shape(spec.capacity, "preorder"))


_STORE: Final = Store()
_CODES: Final = Codes(_ARITIES)
_ATOMS: Final = st.sampled_from(("NP", "S", "Sdcl")).map(_STORE.op)


def _slash(left: Node, right: Node) -> Node:
    """Return a slash node over two terms.

    Parameters
    ----------
    left
        The left term.
    right
        The right term.
    """
    return _STORE.op("/", left, right)


def _context(other: Node) -> Callable[[Node], Node]:
    """Return a context that places a term left of a slash.

    Parameters
    ----------
    other
        The fixed right argument.
    """

    def under(term: Node) -> Node:
        return _slash(term, other)

    return under


_PAIRS: Final = st.builds(_slash, _ATOMS, _ATOMS)
_BUSHES: Final = st.builds(_slash, _PAIRS, _PAIRS)
_SPINES: Final = st.builds(_slash, _BUSHES, _ATOMS)
_CATS: Final = st.one_of(_ATOMS, _PAIRS)
_TERMS: Final = st.one_of(_ATOMS, _PAIRS, _BUSHES, _SPINES)
_CONTEXTS: Final = _ATOMS.map(_context)

# restrict draws to a pair that exposes asymmetric matching.
_REFINED: Final = st.sampled_from(("Sdcl", "S")).map(_STORE.op)

_ATOM_NAMES: Final = 4
_FIRST_FRESH: Final = 100
_OTHER: Final = Store(generation=1)
_FRESH_ATOMS: Final = count(_FIRST_FRESH)


class Loose(Store):
    """A store that accepts arguments from another store."""

    @override
    def guard(self, op: str, args: tuple[Node, ...]) -> None:
        """Accept every argument."""
        del op, args


class Fresh(Store):
    """A store that allocates a new node for every build."""

    @override
    def op(self, op: str, *args: Node, payload: int | None = None) -> Node:
        """Return a newly allocated node."""
        self.guard(op, args)

        return self.intern(op, args, payload)


class Shallow(Store):
    """A store that records node depth as node count."""

    @override
    def derive(self, op: str, args: tuple[Node, ...], payload: int | None) -> Derived:
        """Return the summaries with a depth standing in for the node count."""
        derived = super().derive(op, args, payload)

        return derived._replace(size=1 + max((arg.size for arg in args), default=0))


class Drifting(Node):
    """A node whose hash changes after each lookup."""

    __slots__ = ("_asked",)

    def __init__(
        self,
        op: str,
        args: tuple[Node, ...],
        payload: int | None,
        generation: int,
        derived: Derived,
    ) -> None:
        """Initialize the call count."""
        super().__init__(op, args, payload, generation, derived)
        self._asked = 0

    @override
    def __hash__(self) -> int:
        """Return one more than the number last returned."""
        self._asked += 1

        return self._asked


class Counting(Store):
    """A store that interns nodes with unstable hashes."""

    @override
    def intern(self, op: str, args: tuple[Node, ...], payload: int | None) -> Node:
        """Return a node that recomputes its hash each time it is asked."""
        return Drifting(op, args, payload, 0, self.derive(op, args, payload))


_LOOSE: Final = Loose(generation=2)
_FRESH: Final = Fresh()
_SHALLOW: Final = Shallow()
_COUNTING: Final = Counting()


def _fvar(atom: int) -> Node:
    """Return a free-variable node.

    Parameters
    ----------
    atom
        The atom index.
    """
    return _STORE.op("fvar", payload=atom)


def _bvar(index: int) -> Node:
    """Return a bound-variable node.

    Parameters
    ----------
    index
        The bound index.
    """
    return _STORE.op("bvar", payload=index)


def _app(function: Node, argument: Node) -> Node:
    """Return an application node.

    Parameters
    ----------
    function
        The function term.
    argument
        The argument term.
    """
    return _STORE.op("app", function, argument)


def _fresh() -> int:
    return next(_FRESH_ATOMS)


def _rename_with(term: Node, renaming: Mapping[int, int]) -> Node:
    """Return a term with free atoms renamed.

    Parameters
    ----------
    term
        The term to rename.
    renaming
        Replacement atoms keyed by original atom.
    """
    if term.op == "fvar" and term.payload is not None:
        return _fvar(renaming.get(term.payload, term.payload))

    if not term.args:
        return term

    renamed = (_rename_with(arg, renaming) for arg in term.args)

    return _STORE.op(term.op, *renamed, payload=term.payload)


def _rename(term: Node, atoms: tuple[int, ...], fresh: tuple[int, ...]) -> Node:
    """Return a term with one atom tuple renamed to another.

    Parameters
    ----------
    term
        The term to rename.
    atoms
        The original atoms.
    fresh
        Their replacements.
    """
    return _rename_with(term, dict(zip(atoms, fresh, strict=True)))


def _rename_nothing(term: Node, atoms: tuple[int, ...], fresh: tuple[int, ...]) -> Node:
    """Return a term without applying the requested renaming.

    Parameters
    ----------
    term
        The term to return.
    atoms
        The ignored original atoms.
    fresh
        The ignored replacements.
    """
    del atoms, fresh

    return term


def _abstract_at(term: Node, atom: int, depth: int) -> Node:
    """Return a term with an atom abstracted at a depth.

    Parameters
    ----------
    term
        The term to transform.
    atom
        The atom to abstract.
    depth
        The current binder depth.
    """
    if term.op == "fvar":
        return _bvar(depth) if term.payload == atom else term

    if not term.args:
        return term

    under = depth + _BINDS.get(term.op, 0)
    abstracted = (_abstract_at(arg, atom, under) for arg in term.args)

    return _STORE.op(term.op, *abstracted, payload=term.payload)


def _abstract(term: Node, atom: int) -> Node:
    """Return a term with an atom abstracted.

    Parameters
    ----------
    term
        The term to transform.
    atom
        The atom to abstract.
    """
    return _abstract_at(term, atom, 0)


def _substitute_at(term: Node, value: Node, depth: int) -> Node:
    """Return a term with an index substituted at a depth.

    Parameters
    ----------
    term
        The term to transform.
    value
        The replacement term.
    depth
        The current binder depth.
    """
    if term.op == "bvar":
        return value if term.payload == depth else term

    if not term.args:
        return term

    under = depth + _BINDS.get(term.op, 0)
    substituted = (_substitute_at(arg, value, under) for arg in term.args)

    return _STORE.op(term.op, *substituted, payload=term.payload)


def _instantiate(term: Node, value: Node) -> Node:
    """Return a term instantiated with a locally closed value.

    Parameters
    ----------
    term
        The abstraction body.
    value
        The replacement term.

    Raises
    ------
    NotLocallyClosedError
        If the replacement leaves a bound index open.
    """
    if not value.is_closed:
        raise NotLocallyClosedError(max_index=value.maxidx, depth=0)

    return _substitute_at(term, value, 0)


def _instantiate_unguarded(term: Node, value: Node) -> Node:
    """Return a term instantiated without a closure check.

    Parameters
    ----------
    term
        The abstraction body.
    value
        The replacement term.
    """
    return _substitute_at(term, value, 0)


def _open_binder(term: Node) -> tuple[Node, tuple[int, ...]]:
    """Return a binder body opened with a fresh atom.

    Parameters
    ----------
    term
        The binder term.
    """
    atom = _fresh()

    return _substitute_at(term.args[0], _fvar(atom), 0), (atom,)


def _close_binder(body: Node, atoms: tuple[int, ...]) -> Node:
    """Return a body closed over its opening atoms.

    Parameters
    ----------
    body
        The opened body.
    atoms
        The opening atoms.
    """
    return _STORE.op("lam", _abstract(body, atoms[0]))


def _close_naively(body: Node, atoms: tuple[int, ...]) -> Node:
    """Return a body wrapped without abstracting its atoms.

    Parameters
    ----------
    body
        The opened body.
    atoms
        The ignored opening atoms.
    """
    del atoms

    return _STORE.op("lam", body)


def _binder_over(body: Node) -> Node:
    """Return a binder over a body.

    Parameters
    ----------
    body
        The binder body.
    """
    return _close_binder(body, (0,))


def _grow(inner: st.SearchStrategy[Node]) -> st.SearchStrategy[Node]:
    """Return recursive term growth strategies.

    Parameters
    ----------
    inner
        The recursive inner-term strategy.
    """
    return st.one_of(
        st.builds(_app, inner, inner),
        st.builds(_slash, inner, inner),
        st.builds(_binder_over, inner),
    )


_VARIABLES: Final = st.integers(0, _ATOM_NAMES - 1).map(_fvar)
_INDICES: Final = st.integers(0, 2).map(_bvar)
_CLOSED: Final = st.recursive(st.one_of(_ATOMS, _VARIABLES), _grow, max_leaves=4)
_OPEN: Final = st.one_of(_INDICES, st.builds(_app, _INDICES, _ATOMS))
_ALGEBRA: Final = st.one_of(_CLOSED, _OPEN)
_BINDERS: Final = _CLOSED.map(_binder_over)


def _root(term: Node) -> str:
    """Return a term's root operator.

    Parameters
    ----------
    term
        The term to inspect.
    """
    return term.op


def _arguments(term: Node) -> tuple[Node, ...]:
    """Return a term's arguments.

    Parameters
    ----------
    term
        The term to inspect.
    """
    return term.args


def _rebuild(term: Node, args: tuple[Node, ...]) -> Node:
    """Return a term rebuilt with supplied arguments.

    Parameters
    ----------
    term
        The source term.
    args
        The replacement arguments.
    """
    return _STORE.op(term.op, *args, payload=term.payload)


def _rebuild_afresh(term: Node, args: tuple[Node, ...]) -> Node:
    """Return a freshly allocated rebuild of a term.

    Parameters
    ----------
    term
        The source term.
    args
        The replacement arguments.
    """
    return _FRESH.op(term.op, *args, payload=term.payload)


def _core(term: Node) -> Node:
    """Return a term's erased core.

    Parameters
    ----------
    term
        The term to inspect.
    """
    return term.core


def _erase_root(term: Node) -> Node:
    """Return a term with only its root feature erased.

    Parameters
    ----------
    term
        The term to erase.
    """
    erased = _ERASURES.get(term.op, term.op)

    return _STORE.op(erased, *term.args, payload=term.payload)


def _erase_by_payload(term: Node) -> Node:
    """Return a payload-dependent erasure.

    Parameters
    ----------
    term
        The term to erase.
    """
    if term.op == "fvar" and term.payload is not None and not term.payload % 2:
        return _bvar(term.payload)

    return term


def _wrap(term: Node) -> Node:
    """Return a term wrapped in a slash context.

    Parameters
    ----------
    term
        The term to wrap.
    """
    return _slash(term, _STORE.op("NP"))


def _node_size(term: Node) -> int:
    """Return a term's cached node count.

    Parameters
    ----------
    term
        The term to inspect.
    """
    return term.size


def _open_indices(term: Node) -> dict[str, int]:
    """Return a term's cached open-index count, as the one sort's entry.

    Parameters
    ----------
    term
        The term to inspect.
    """
    return {"Cat": term.maxidx} if term.maxidx else {}


def _free(term: Node) -> int:
    """Return a term's cached free-variable mask.

    Parameters
    ----------
    term
        The term to inspect.
    """
    return term.free


def _shift(term: Node, position: int) -> dict[str, int]:
    """Return how many variables a term root binds over each of its arguments.

    Parameters
    ----------
    term
        The term to inspect.
    position
        The argument position, over which the root binds as over any other.
    """
    del position
    count = _BINDS.get(term.op, 0)

    return {"Cat": count} if count else {}


def _is_closed(term: Node) -> bool:
    """Return whether a term is locally closed.

    Parameters
    ----------
    term
        The term to inspect.
    """
    return term.is_closed


def _adopt() -> Node:
    return _OTHER.op("/", _STORE.op("NP"), _OTHER.op("NP"))


def _adopt_loosely() -> Node:
    return _LOOSE.op("/", _STORE.op("NP"), _LOOSE.op("NP"))


_BINDING: Final = BindingFixture(
    binders=_BINDERS,
    open_binder=_open_binder,
    close_binder=_close_binder,
    rename=_rename,
    fresh=_fresh,
    variable=_fvar,
    abstract=_abstract,
    instantiate=_instantiate,
    open_terms=_OPEN,
)


def term_fixture(
    terms: st.SearchStrategy[Node] = _ALGEBRA,
) -> TermFixture[Node, str, int]:
    """Return a term-law fixture.

    Parameters
    ----------
    terms
        The term strategy.
    """
    return TermFixture(
        terms=terms,
        root=_root,
        arguments=_arguments,
        rebuild=_rebuild,
        core=_core,
        size=_node_size,
        open_indices=_open_indices,
        free=_free,
        shift=_shift,
        is_closed=_is_closed,
        binding=_BINDING,
        adopt=_adopt,
        max_examples=_EXAMPLES,
    )


_REVERSED: Final = dict(reversed(list(_ARITIES.items())))


def alphabet_fixture(table: Codes, rebuilt: Codes) -> AlphabetFixture[Codes, str]:
    """Return an alphabet-law fixture.

    Parameters
    ----------
    table
        The code table under test.
    rebuilt
        The independently rebuilt table.
    """
    return AlphabetFixture(
        table=table,
        rebuild=lambda: rebuilt,
        operators=lambda held: held.operators,
        code=lambda held, op: held.code(op),
        operator=lambda held, code: held.operator(code),
        arity=lambda held, op: held.arity(op),
        sort=lambda _held, _op: "Cat",
        family=lambda _held, _op: "all",
        digest=digest,
        pad=_PAD,
    )


class Codec(NamedTuple):
    """A shared graph component.

    Attributes
    ----------
    order
        The codec slot order.
    """

    order: str


_CODEC: Final = Codec("complete-tree")


class Transient:
    """A boundary object that must not survive into a built graph."""


class Loaded(Sized):
    """A sized object holding further parts under its declaration.

    Parameters
    ----------
    shape
        The stored layout declaration.
    parts
        The parts the built graph holds.
    """

    def __init__(self, shape: Shape, parts: tuple[Codec | Transient, ...]) -> None:
        """Keep the declaration and the parts."""
        super().__init__(shape)
        self._parts = {"held": [parts]}


def matching_fixture(
    matching: Interning | FeatureErasure | SameSize | RootErasure,
    terms: st.SearchStrategy[Node] = _CATS,
) -> MatchingFixture[Node]:
    """Return a matching-law fixture.

    Parameters
    ----------
    matching
        The matching relation.
    terms
        The term strategy.
    """
    return MatchingFixture(matching, terms, _CONTEXTS, _EXAMPLES)


def tree_fixture(
    layout: CompleteTree, terms: st.SearchStrategy[Node]
) -> LayoutFixture[Node, Store, Codes]:
    """Return a complete-tree layout fixture.

    Parameters
    ----------
    layout
        The layout under test.
    terms
        The term strategy.
    """
    return LayoutFixture(layout, terms, _STORE, _CODES, size, fits_a_tree, _EXAMPLES)


def walk_fixture(
    layout: Preorder, terms: st.SearchStrategy[Node]
) -> LayoutFixture[Node, Store, Codes]:
    """Return a preorder layout fixture.

    Parameters
    ----------
    layout
        The layout under test.
    terms
        The term strategy.
    """
    return LayoutFixture(layout, terms, _STORE, _CODES, size, fits_a_walk, _EXAMPLES)


def presentation_fixture(
    subject: Sized | Restless,
    rebuilder: Callable[[Shape], Sized],
    components: tuple[Codec | Shape, ...] = (),
    refused: tuple[type, ...] = (),
) -> PresentationFixture[Shape, Codec | Shape]:
    """Return a presentation-law fixture.

    Parameters
    ----------
    subject
        The reifiable object under test.
    rebuilder
        The specification rebuild function.
    components
        References expected to share identity.
    refused
        Classes the built graph must not reach.
    """
    return PresentationFixture(subject, rebuilder, components, refused)


@pytest.mark.parametrize("law", MATCHING_LAWS, ids=lambda law: law.__name__)
def test_each_matching_law_collects_and_runs_on_its_own(
    law: Callable[[MatchingFixture[Node]], None],
) -> None:
    law(matching_fixture(Interning()))


def test_identity_holds_to_every_matching_law() -> None:
    check_matching_laws(matching_fixture(Interning()))


def test_a_coarser_congruence_holds_to_every_matching_law() -> None:
    check_matching_laws(matching_fixture(FeatureErasure(_STORE)))


def test_a_relation_with_no_representative_holds_to_every_matching_law() -> None:
    check_matching_laws(matching_fixture(SameSize()))


def test_the_matching_entry_point_reports_the_law_that_failed() -> None:
    with pytest.raises(LawViolationError, match="equivalence"):
        check_matching_laws(matching_fixture(OneDirection(), _REFINED))


def test_equivalence_catches_a_relation_that_holds_one_way() -> None:
    with pytest.raises(LawViolationError, match="one direction"):
        equivalence_is_an_equivalence(matching_fixture(OneDirection(), _REFINED))


def test_the_section_catches_a_key_that_splits_one_class() -> None:
    with pytest.raises(LawViolationError, match="section"):
        key_is_a_section(matching_fixture(KeyMissesTheClass(_STORE)))


def test_the_congruence_catches_a_relation_construction_undoes() -> None:
    with pytest.raises(LawViolationError, match="congruence"):
        equivalence_survives_construction(
            matching_fixture(RootErasure(_STORE), st.just(_STORE.op("Sdcl")))
        )


def test_coarsening_catches_a_representative_outside_the_class() -> None:
    with pytest.raises(LawViolationError, match="coarsening"):
        equivalence_coarsens_identity(matching_fixture(KeyLeavesTheClass(_STORE)))


def test_idempotence_catches_a_key_that_keeps_building() -> None:
    with pytest.raises(LawViolationError, match="idempotence"):
        key_is_idempotent(matching_fixture(KeyGrowsTheTerm(_STORE)))


def test_the_laws_over_representatives_leave_a_relation_without_one_alone() -> None:
    fixture = matching_fixture(SameSize())

    key_is_a_section(fixture)
    key_is_idempotent(fixture)


@pytest.mark.parametrize("law", LAYOUT_LAWS, ids=lambda law: law.__name__)
def test_each_layout_law_collects_and_runs_on_its_own(
    law: Callable[[LayoutFixture[Node, Store, Codes]], None],
) -> None:
    law(tree_fixture(CompleteTree(), _TERMS))


def test_a_complete_tree_holds_to_every_layout_law() -> None:
    check_layout_laws(tree_fixture(CompleteTree(), _TERMS))


def test_a_preorder_walk_holds_to_every_layout_law() -> None:
    check_layout_laws(walk_fixture(Preorder(), _TERMS))


def test_the_layout_entry_point_reports_the_law_that_failed() -> None:
    with pytest.raises(LawViolationError, match="round trip"):
        check_layout_laws(tree_fixture(Featureless(), st.just(_STORE.op("Sdcl"))))


def test_the_round_trip_catches_a_layout_that_loses_a_feature() -> None:
    with pytest.raises(LawViolationError, match="round trip"):
        round_trip_is_identity(tree_fixture(Featureless(), st.just(_STORE.op("Sdcl"))))


def test_the_round_trip_refuses_to_pass_on_terms_it_could_not_encode() -> None:
    with pytest.raises(LawViolationError, match="never exercised"):
        round_trip_is_identity(tree_fixture(CompleteTree(), _SPINES))


def test_no_truncation_catches_a_layout_that_drops_what_does_not_fit() -> None:
    with pytest.raises(LawViolationError, match="without complaint"):
        encoding_refuses_to_truncate(tree_fixture(Truncating(), _SPINES))


def test_no_truncation_refuses_to_pass_where_nothing_was_too_big() -> None:
    with pytest.raises(LawViolationError, match="never exercised"):
        encoding_refuses_to_truncate(tree_fixture(CompleteTree(), _ATOMS))


def test_mask_soundness_catches_a_node_marked_over_padding() -> None:
    with pytest.raises(LawViolationError, match="holds no node"):
        mask_is_sound(tree_fixture(PhantomNode(), _ATOMS))


def test_mask_completeness_catches_a_node_left_out_of_the_mask() -> None:
    with pytest.raises(LawViolationError, match="occupied slots"):
        mask_is_complete(tree_fixture(LosesALeaf(), _PAIRS))


def test_path_addressing_catches_a_walk_claiming_to_address_paths() -> None:
    leaf = _STORE.op("NP")
    pair = _slash(leaf, leaf)
    left_branching = _slash(pair, leaf)

    with pytest.raises(LawViolationError, match="path addressing"):
        path_addressing_is_honest(
            walk_fixture(Addressed(), st.sampled_from((pair, left_branching)))
        )


def test_path_addressing_leaves_a_layout_claiming_no_addresses_alone() -> None:
    path_addressing_is_honest(walk_fixture(Preorder(), _TERMS))


def test_legal_continuations_catch_a_layout_refusing_its_own_symbols() -> None:
    with pytest.raises(LawViolationError, match="legal continuation"):
        legal_next_admits_the_next_symbol(tree_fixture(Deaf(), _ATOMS))


def built_graph_fixture(
    components: tuple[Codec | Shape, ...] = (_CODEC, _CODEC, _CODEC),
    parts: tuple[Codec | Transient, ...] = (_CODEC,),
) -> PresentationFixture[Shape, Codec | Shape]:
    """Return a fixture for a built object graph refusing transients.

    Parameters
    ----------
    components
        References expected to share identity.
    parts
        The parts the built graph holds at depth.
    """
    return presentation_fixture(
        Loaded(Shape(_CAPACITY, "complete-tree"), parts),
        rebuild,
        components,
        (Transient,),
    )


@pytest.mark.parametrize("law", PRESENTATION_LAWS, ids=lambda law: law.__name__)
def test_each_presentation_law_collects_and_runs_on_its_own(
    law: Callable[[PresentationFixture[Shape, Codec | Shape]], None],
) -> None:
    law(built_graph_fixture())


def test_an_object_stating_its_declaration_holds_to_every_presentation_law() -> None:
    check_presentation_laws(built_graph_fixture())


def test_one_component_catches_a_graph_holding_two_equal_codecs() -> None:
    twin = Codec(_CODEC.order)

    assert twin == _CODEC

    with pytest.raises(LawViolationError, match="two objects"):
        one_component_by_identity(built_graph_fixture(components=(_CODEC, twin)))


def test_one_component_leaves_a_fixture_naming_none_alone() -> None:
    one_component_by_identity(built_graph_fixture(components=()))


def test_reachability_catches_a_transient_hidden_deep_in_the_built_graph() -> None:
    with pytest.raises(LawViolationError) as caught:
        nothing_refused_is_reachable(built_graph_fixture(parts=(_CODEC, Transient())))

    message = str(caught.value)

    assert message.startswith("reachability:")
    assert "Transient at ._parts['held'][0][1]" in message
    assert "refused there" in message


def test_reachability_leaves_a_graph_of_native_values_alone() -> None:
    nothing_refused_is_reachable(built_graph_fixture(parts=(_CODEC, _CODEC)))
    nothing_refused_is_reachable(
        presentation_fixture(Sized(Shape(_CAPACITY, "complete-tree")), rebuild)
    )


def test_the_round_trip_catches_a_declaration_the_rebuild_drops() -> None:
    fixture = presentation_fixture(
        Sized(Shape(_CAPACITY, "complete-tree")), rebuild_forgetting_the_order
    )

    with pytest.raises(LawViolationError, match="its rebuild"):
        check_presentation_laws(fixture)


def test_the_round_trip_catches_a_declaration_that_changes_between_calls() -> None:
    with pytest.raises(LawViolationError, match="and then"):
        check_presentation_laws(presentation_fixture(Restless(), rebuild))


@pytest.mark.parametrize("law", SIGNATURE_INVARIANTS, ids=lambda law: law.__name__)
def test_each_signature_invariant_collects_and_runs_on_its_own(
    law: Callable[[AlphabetFixture[Codes, str]], None],
) -> None:
    law(alphabet_fixture(Codes(_ARITIES), Codes(_REVERSED)))


def test_a_sorted_table_holds_to_every_signature_invariant() -> None:
    check_signature_invariants(alphabet_fixture(Codes(_ARITIES), Codes(_REVERSED)))


def test_the_signature_entry_point_reports_the_invariant_that_failed() -> None:
    with pytest.raises(LawViolationError, match="bijection"):
        check_signature_invariants(
            alphabet_fixture(Gapped(_ARITIES), Gapped(_REVERSED))
        )


def test_padding_catches_a_table_handing_an_operator_the_padding_code() -> None:
    with pytest.raises(LawViolationError, match="padding code"):
        padding_belongs_to_no_operator(
            alphabet_fixture(PadTaken(_ARITIES), PadTaken(_REVERSED))
        )


def test_the_bijection_catches_a_table_leaving_a_code_unheld() -> None:
    with pytest.raises(LawViolationError, match="bijection"):
        codes_are_a_bijection(alphabet_fixture(Gapped(_ARITIES), Gapped(_REVERSED)))


def test_segregation_catches_a_table_coding_by_name() -> None:
    with pytest.raises(LawViolationError, match="after arity"):
        codes_group_by_sort_then_family(
            alphabet_fixture(ByName(_ARITIES), ByName(_REVERSED))
        )


@pytest.mark.parametrize(
    ("kind", "match"),
    [("sort", "returns to the sort"), ("family", "returns to the family")],
)
def test_segregation_catches_a_sort_or_family_split_in_two(
    kind: str, match: str
) -> None:
    table = Codes(_ARITIES)
    ordered = sorted(table.operators, key=table.code)
    split = {op: ("a" if at % 2 else "b") for at, op in enumerate(ordered)}
    fixture = alphabet_fixture(table, Codes(_REVERSED))
    lookup = {
        "sort": fixture._replace(sort=lambda _held, op: split[op]),
        "family": fixture._replace(family=lambda _held, op: split[op]),
    }

    with pytest.raises(LawViolationError, match=match):
        codes_group_by_sort_then_family(lookup[kind])


def test_determinism_catches_codes_that_follow_the_declaration_order() -> None:
    with pytest.raises(LawViolationError, match="determinism"):
        codes_are_determined_by_the_declaration(
            alphabet_fixture(Declared(_ARITIES), Declared(_REVERSED))
        )


@pytest.mark.parametrize("law", TERM_LAWS, ids=lambda law: law.__name__)
def test_each_term_law_collects_and_runs_on_its_own(
    law: Callable[[TermFixture[Node, str, int]], None],
) -> None:
    law(term_fixture())


def test_the_interned_algebra_holds_to_every_term_law() -> None:
    check_term_laws(term_fixture())


def test_the_term_entry_point_reports_the_law_that_failed() -> None:
    with pytest.raises(LawViolationError, match="foreign argument"):
        check_term_laws(term_fixture()._replace(adopt=_adopt_loosely))


def test_the_foreign_guard_catches_a_store_that_asks_nothing() -> None:
    with pytest.raises(LawViolationError, match="outside the store's ancestry"):
        a_foreign_argument_is_refused(term_fixture()._replace(adopt=_adopt_loosely))


def test_the_foreign_guard_leaves_an_algebra_with_one_store_alone() -> None:
    a_foreign_argument_is_refused(term_fixture()._replace(adopt=None))


def test_interning_catches_a_store_that_hands_out_a_new_node() -> None:
    with pytest.raises(LawViolationError, match="interning"):
        structure_is_identity(term_fixture()._replace(rebuild=_rebuild_afresh))


def test_hashing_catches_a_node_that_recomputes_its_hash() -> None:
    counted = st.just(_COUNTING.op("/", _COUNTING.op("NP"), _COUNTING.op("S")))

    with pytest.raises(LawViolationError, match="hashes to"):
        hashing_is_stable(term_fixture(counted))


def test_erasure_idempotence_catches_an_erasure_that_keeps_building() -> None:
    with pytest.raises(LawViolationError, match="erasure idempotence"):
        erasure_is_idempotent(term_fixture()._replace(core=_wrap))


def test_the_congruence_catches_an_erasure_that_stops_at_the_root() -> None:
    featured = st.just(_slash(_STORE.op("Sdcl"), _STORE.op("NP")))
    fixture = term_fixture(featured)._replace(core=_erase_root)

    with pytest.raises(LawViolationError, match="erasure congruence"):
        erasure_is_a_congruence(fixture)


def test_the_congruence_checks_a_unary_argument_that_needs_erasure() -> None:
    featured = st.just(_STORE.op("lam", _STORE.op("Sdcl")))
    fixture = term_fixture(featured)._replace(core=_erase_root)

    with pytest.raises(LawViolationError, match="erasure congruence"):
        erasure_is_a_congruence(fixture)


def test_the_derived_fields_catch_a_depth_recorded_as_a_node_count() -> None:
    shallow = st.just(_SHALLOW.op("/", _SHALLOW.op("NP"), _SHALLOW.op("S")))

    with pytest.raises(LawViolationError, match="nodes, and covers"):
        derived_fields_agree_with_their_definitions(term_fixture(shallow))


def test_alpha_catches_a_renaming_that_leaves_the_atom_where_it_was() -> None:
    binding = _BINDING._replace(rename=_rename_nothing)

    with pytest.raises(LawViolationError, match="alpha"):
        alpha_variants_are_one_node(term_fixture()._replace(binding=binding))


def test_opening_and_closing_catch_a_binder_that_abstracts_nothing() -> None:
    binder = _binder_over(_fvar(0))
    binding = _BINDING._replace(
        binders=st.just(binder),
        close_binder=_close_naively,
    )

    with pytest.raises(LawViolationError, match="open and close"):
        closing_undoes_opening(term_fixture()._replace(binding=binding))


def test_substitution_catches_a_kernel_with_no_closedness_guard() -> None:
    binding = _BINDING._replace(instantiate=_instantiate_unguarded)

    with pytest.raises(LawViolationError, match="indices open"):
        substitution_cancels_abstraction(term_fixture()._replace(binding=binding))


def test_substitution_refuses_to_pass_where_every_term_was_closed() -> None:
    binding = _BINDING._replace(open_terms=_CLOSED)
    fixture = term_fixture()._replace(binding=binding)

    with pytest.raises(LawViolationError, match="never exercised"):
        substitution_cancels_abstraction(fixture)


def test_substitution_refuses_to_pass_where_every_term_was_open() -> None:
    with pytest.raises(LawViolationError, match="never exercised"):
        substitution_cancels_abstraction(term_fixture(_OPEN))


def test_the_binding_laws_leave_an_algebra_that_binds_nothing_alone() -> None:
    fixture = term_fixture()._replace(binding=None)

    alpha_variants_are_one_node(fixture)
    closing_undoes_opening(fixture)
    substitution_cancels_abstraction(fixture)


def test_the_congruence_catches_an_erasure_that_reads_the_payload() -> None:
    named = st.sampled_from((0, 1)).map(_fvar)
    fixture = term_fixture(named)._replace(core=_erase_by_payload)

    with pytest.raises(LawViolationError, match="erase to the root operators"):
        erasure_is_a_congruence(fixture)


def _exported(*, entry_points: bool) -> list[str]:
    return sorted(
        name
        for name in laws.__all__
        if isinstance(getattr(laws, name), FunctionType)
        and name.startswith("check_") is entry_points
    )


def test_every_exported_law_is_named_by_exactly_one_family() -> None:
    families = (
        SIGNATURE_INVARIANTS,
        TERM_LAWS,
        MATCHING_LAWS,
        LAYOUT_LAWS,
        PRESENTATION_LAWS,
    )
    collected = [law for family in families for law in family]

    assert sorted(law.__name__ for law in collected) == _exported(entry_points=False)
    assert len(set(collected)) == len(collected)


def test_the_exported_entry_points_are_the_five_the_families_are_run_by() -> None:
    entry_points = (
        check_signature_invariants,
        check_term_laws,
        check_matching_laws,
        check_layout_laws,
        check_presentation_laws,
    )

    assert sorted(entry.__name__ for entry in entry_points) == _exported(
        entry_points=True
    )


def test_store_independence_holds_of_a_reconstruction_that_keeps_structure() -> None:
    structure_is_store_independent(
        term_fixture()._replace(transplant=lambda term: _rebuild(term, term.args))
    )


def test_store_independence_catches_a_reconstruction_that_changes_a_payload() -> None:
    def shifted(term: Node) -> Node:
        return _STORE.op("bvar", payload=7) if term.op == "bvar" else term

    terms = st.just(_STORE.op("bvar", payload=0))

    with pytest.raises(LawViolationError, match="store independence"):
        structure_is_store_independent(term_fixture(terms)._replace(transplant=shifted))


def test_store_independence_is_silent_without_a_reconstruction() -> None:
    structure_is_store_independent(term_fixture()._replace(transplant=None))

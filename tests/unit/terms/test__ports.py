from typing import TYPE_CHECKING, Self

import pytest
from tests.support.term_reference import categories, formulas

from lofi.terms import (
    CategoryResolution,
    OperatorIndex,
    PreorderArityLayout,
    Term,
    TermStore,
)
from lofi.terms._ports import (
    BatchEncoding,
    Calculus,
    Encoding,
    Functors,
    Layout,
    Matching,
    Path,
    Reifiable,
    SortTag,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.terms import Operator, Signature
    from lofi.terms._ports import Interner


# a gapped occupancy mask for a complete binary-tree layout.
GAPPED = Encoding(
    symbols=(5, 2, 0, 0, 0, 0, 0),
    parents=(Encoding.NO_PARENT, 0, Encoding.NO_PARENT, 1, 1, -1, -1),
    depths=(0, 1, 0, 0, 0, 0, 0),
    occupied=(True, True, False, False, False, False, False),
)


class FirstLetter:
    """A first-letter string congruence."""

    def __init__(self, words: Sequence[str]) -> None:
        self._rep = {word[0]: word for word in reversed(words)}

    @property
    def name(self) -> str:
        """Return the relation's name."""
        return "first-letter"

    @property
    def has_key(self) -> bool:
        """Return whether the relation has representatives."""
        return True

    def key(self, t: str, /) -> str:
        """Return the representative word for a word's class."""
        return self._rep[t[0]]

    def equivalent(self, a: str, b: str, /) -> bool:
        """Return whether two words start alike."""
        return a[0] == b[0]


class HalfAMatching:
    """An incomplete matching implementation."""

    @property
    def name(self) -> str:
        """Return the relation's name."""
        return "half"

    def key(self, t: str, /) -> str:
        """Return the word unchanged."""
        return t


class FixedCategoryResolution:
    """Resolve one category spelling to one store-owned term."""

    def __init__(self, source: str, category: Term) -> None:
        self._source = source
        self._category = category

    def __call__(self, source: str, /) -> Term:
        """Return the category held under its declared spelling."""
        if source != self._source:
            raise ValueError(source)

        return self._category


class BinaryFunctors:
    """A complete binary-functor implementation."""

    @property
    def slashes(self) -> frozenset[str]:
        """Return the two functor names."""
        return frozenset({"fwd", "bwd"})

    @property
    def forward_slash(self) -> str:
        """Return the forward functor name."""
        return "fwd"

    @property
    def backward_slash(self) -> str:
        """Return the backward functor name."""
        return "bwd"

    @property
    def result_path(self) -> Path:
        """Return the result path."""
        return (0,)

    @property
    def argument_path(self) -> Path:
        """Return the argument path."""
        return (1,)

    @property
    def category_sort(self) -> str:
        """Return the sort the functor constructors build."""
        return "Cat"

    def is_functor(self, term: str, /) -> bool:
        """Return whether a string names a compound."""
        return "(" in term

    def slash(self, term: str, /) -> str:
        """Return a compound's operator."""
        return term.partition("(")[0]

    def make(self, slash: str, result: str, argument: str, /) -> str:
        """Return a compound string."""
        return f"{slash}({result},{argument})"

    def opposite(self, slash: str, /) -> str:
        """Return the other functor name."""
        return "bwd" if slash == "fwd" else "fwd"


class HalfFunctors:
    """A functor implementation without `make`."""

    slashes = frozenset({"fwd", "bwd"})
    forward_slash = "fwd"
    backward_slash = "bwd"
    result_path = (0,)
    argument_path = (1,)

    def is_functor(self, term: str, /) -> bool:
        """Return whether a string names a compound."""
        return "(" in term

    def slash(self, term: str, /) -> str:
        """Return a compound's operator."""
        return term.partition("(")[0]

    def opposite(self, slash: str, /) -> str:
        """Return the other functor name."""
        return "bwd" if slash == "fwd" else "fwd"


class Reversal:
    """A reversed-character string layout."""

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity

    @property
    def name(self) -> str:
        """Return the layout's name."""
        return "reversal"

    @property
    def path_addressed(self) -> bool:
        """Return whether a slot denotes a fixed path."""
        return False

    @property
    def capacity(self) -> int:
        """Return the number of slots an encoding occupies."""
        return self._capacity

    @property
    def skeleton(self) -> None:
        """Return no fixed structure for this layout."""
        return None

    def static_symbols(self, _index: None, /) -> tuple[tuple[bool, ...], ...]:
        """Return the symbol mask for each slot."""
        return ((True,),) * self._capacity

    def encode(self, t: str, _index: None, /) -> Encoding:
        """Return the characters of a word, in reverse, one to a slot."""
        codes = tuple(ord(c) for c in reversed(t))
        pad = (0,) * (self._capacity - len(codes))

        return Encoding(
            symbols=codes + pad,
            parents=(Encoding.NO_PARENT,) * self._capacity,
            depths=(0,) * self._capacity,
            occupied=(True,) * len(codes) + (False,) * len(pad),
        )

    def decode(self, enc: Encoding, _store: None, _index: None, /) -> str:
        """Return the word an encoding holds."""
        held = (c for c, live in zip(enc.symbols, enc.occupied, strict=True) if live)

        return "".join(chr(c) for c in held)[::-1]

    def legal_next(self, prefix: Sequence[int], _index: None, /) -> tuple[bool, ...]:
        """Return whether another character fits."""
        return (len(prefix) < self._capacity,)


class Sized:
    """An object with a stored declaration."""

    def __init__(self, spec: tuple[str, int]) -> None:
        self._spec = spec

    @classmethod
    def build(cls, spec: tuple[str, int]) -> Self:
        """Return the object a declaration describes."""
        return cls(spec)

    def to_spec(self) -> tuple[str, int]:
        """Return the declaration this object was built from."""
        return self._spec


def test_encoding_reports_width_and_occupancy_separately() -> None:
    assert GAPPED.width == 7
    assert GAPPED.length == 2


def test_encoding_walks_ancestors_up_to_the_root() -> None:
    assert GAPPED.ancestors(1) == (0,)
    assert GAPPED.ancestors(3) == (1, 0)
    assert GAPPED.ancestors(0) == ()


def test_encoding_is_equal_by_channel_and_frozen() -> None:
    twin = Encoding(
        symbols=GAPPED.symbols,
        parents=GAPPED.parents,
        depths=GAPPED.depths,
        occupied=GAPPED.occupied,
    )

    assert twin == GAPPED
    assert hash(twin) == hash(GAPPED)
    assert {GAPPED, twin} == {GAPPED}

    for field, value in (("symbols", ()), ("depths", ()), ("occupied", ())):
        with pytest.raises(AttributeError):
            setattr(GAPPED, field, value)


def test_encoding_with_a_payload_differs_from_one_without() -> None:
    paid = Encoding(
        symbols=GAPPED.symbols,
        parents=GAPPED.parents,
        depths=GAPPED.depths,
        occupied=GAPPED.occupied,
        payloads=(0,) * GAPPED.width,
    )

    assert paid != GAPPED
    assert paid.payloads is not None
    assert GAPPED.payloads is None


def test_batch_stacks_rows_and_keeps_them() -> None:
    small = Encoding(
        symbols=(9, 0),
        parents=(Encoding.NO_PARENT, Encoding.NO_PARENT),
        depths=(0, 0),
        occupied=(True, False),
    )
    wide = Encoding(
        symbols=(9, 4),
        parents=(Encoding.NO_PARENT, 0),
        depths=(0, 1),
        occupied=(True, True),
    )
    batch = BatchEncoding((small, wide))

    assert batch.rows == (small, wide)
    assert batch.width == 2
    assert batch.symbols == ((9, 0), (9, 4))
    assert batch.parents == ((-1, -1), (-1, 0))
    assert batch.depths == ((0, 0), (0, 1))
    assert batch.mask == ((True, False), (True, True))
    assert batch.lengths == (1, 2)
    assert batch.payloads is None


def test_batch_carries_the_payload_channel_when_the_rows_do() -> None:
    row = Encoding(
        symbols=(9,),
        parents=(Encoding.NO_PARENT,),
        depths=(0,),
        occupied=(True,),
        payloads=(7,),
    )

    assert BatchEncoding((row, row)).payloads == ((7,), (7,))


def test_empty_batch_has_no_width_and_no_rows() -> None:
    empty = BatchEncoding(())

    assert empty.width == 0
    assert empty.symbols == ()
    assert empty.lengths == ()
    assert empty.payloads is None


def test_calculus_names_two_operators_and_is_hashable() -> None:
    lifted = Calculus("lam", "eval")

    assert lifted.lam == "lam"
    assert lifted.app == "eval"
    assert lifted == Calculus("lam", "eval")
    assert lifted != Calculus("lam", "apply")
    assert len({lifted, Calculus("lam", "eval")}) == 1


def test_a_sort_tag_types_a_term_without_runtime_state() -> None:
    class Category(SortTag):
        """A static tag for categories."""

    store = TermStore(categories())
    view = store.view("Cat", Category)
    term = view.op("np")

    assert store.__class__ is TermStore
    assert term.__class__ is Term
    assert term is store.op(store.signature.by_name("np"))
    assert not hasattr(store, "__dict__")
    assert not hasattr(view, "__dict__")
    assert not hasattr(term, "__dict__")
    assert not hasattr(view, "__orig_class__")
    assert not hasattr(term, "__orig_class__")


def test_a_matching_instance_satisfies_the_laws_it_is_declared_by() -> None:
    words = ("apple", "anvil", "beet")
    relation = FirstLetter(words)

    assert isinstance(relation, Matching)
    assert relation.name == "first-letter"
    assert relation.has_key

    for a in words:
        assert relation.equivalent(a, a)
        assert relation.key(relation.key(a)) is relation.key(a)

        for b in words:
            assert relation.equivalent(a, b) == (relation.key(a) is relation.key(b))
            assert relation.equivalent(a, b) == relation.equivalent(b, a)


def test_an_incomplete_matching_is_not_one() -> None:
    assert not isinstance(HalfAMatching(), Matching)


def test_category_resolution_is_the_one_callable_term_boundary() -> None:
    store = TermStore(categories())
    category = store.op(store.signature.by_name("np"))
    resolution: CategoryResolution = FixedCategoryResolution("NP", category)

    assert isinstance(resolution, CategoryResolution)
    assert resolution("NP") is category
    assert not isinstance(object(), CategoryResolution)


def test_a_complete_functor_view_satisfies_the_runtime_protocol() -> None:
    functors = BinaryFunctors()

    assert isinstance(functors, Functors)
    assert functors.make("fwd", "s", "np") == "fwd(s,np)"
    assert functors.opposite(functors.forward_slash) == functors.backward_slash


def test_an_incomplete_functor_view_is_not_one() -> None:
    assert not isinstance(HalfFunctors(), Functors)


def test_a_layout_instance_round_trips_and_masks() -> None:
    layout = Reversal(6)

    assert isinstance(layout, Layout)
    assert layout.name == "reversal"
    assert not layout.path_addressed
    assert layout.capacity == 6

    encoded = layout.encode("term", None)

    assert encoded.width == 6
    assert encoded.length == 4
    assert layout.decode(encoded, None, None) == "term"
    assert layout.legal_next((1, 2), None) == (True,)
    assert layout.legal_next((1,) * 6, None) == (False,)


def test_a_reifiable_returns_the_declaration_it_was_built_from() -> None:
    spec = ("wide", 3)
    built = Sized.build(spec)

    assert isinstance(built, Reifiable)
    assert built.to_spec() == spec
    assert not isinstance(Reversal(2), Reifiable)


class Interning:
    """An implementation of only the interner protocol."""

    def __init__(self, signature: Signature) -> None:
        self._store = TermStore(signature)

    def op(
        self,
        op: Operator,
        /,
        *args: Term,
        payload: int | None = None,
    ) -> Term:
        """Return the node for an operator, payload, and arguments.

        Parameters
        ----------
        op
            The root operator.
        args
            The term arguments.
        payload
            The optional non-term payload.
        """
        return self._store.op(op, *args, payload=payload)


def test_the_one_call_the_protocol_declares_is_all_a_decode_makes() -> None:
    signature = categories()
    index = OperatorIndex(signature)
    layout = PreorderArityLayout(3)
    store: Interner = Interning(signature)
    written = (index.index("fwd"), index.index("s"), index.index("np"))
    encoding = Encoding(
        written, (Encoding.NO_PARENT, 0, 0), (0, 1, 1), (True, True, True)
    )
    built = layout.decode(encoding, store, index)

    assert built.op is signature.by_name("fwd")
    assert layout.decode(encoding, store, index) is built
    assert layout.encode(built, index).symbols == written


def test_the_payload_a_decode_passes_reaches_the_store_through_that_call() -> None:
    signature = formulas()
    index = OperatorIndex(signature)
    layout = PreorderArityLayout(2)
    store: Interner = Interning(signature)
    written = (index.index("lam"), index.index("#bvar[Lf]"))
    encoding = Encoding(
        written, (Encoding.NO_PARENT, 0), (0, 1), (True, True), payloads=(0, 2)
    )
    built = layout.decode(encoding, store, index)

    assert built.args[0] is store.op(signature.bound("Lf"), payload=2)
    assert built is store.op(signature.by_name("lam"), built.args[0])

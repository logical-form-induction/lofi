from typing import Final

import pytest

from lofi._types import ItemId
from lofi.data import FeaturizerResources, HashedResources, ItemLookup
from lofi.data._constraint import SpanConstraints
from lofi.data._derivation import Derivation, DerivationNode
from lofi.data._hashed import (
    Alphabet,
    HashedItem,
    ItemHasher,
    UnreadableArmError,
)
from lofi.data._inventory import Inventory, UnknownSymbolError
from lofi.data._supervision import (
    Example,
    GoldDependencies,
    GoldDerivation,
    Scalar,
    Scale,
    Supervision,
    Unlabeled,
)
from lofi.data._values import Item, Token, TokenSpan
from lofi.nn._tensorize import TorchCodeTensorizer

PROBE: Final = "probe"

RESERVED: Final = ("<unknown>",)

SURFACES: Final = ("the", "child", "sleeps")

CATEGORIES: Final = ("S", "NP", "N", "NP/N", "S\\NP")

ORDINAL: Final = Scale("ordinal", 1.0, 7.0)

SENTENCE: Final = tuple(Token(at, word) for at, word in enumerate(SURFACES))


def lexicon() -> Inventory:
    return Inventory("surfaces", SURFACES, RESERVED)


def categories() -> Inventory:
    return Inventory("categories", CATEGORIES)


def hasher(unknown: str | None = None) -> ItemHasher:
    return ItemHasher(lexicon(), categories(), unknown)


def item(
    handle: str = "s/1",
    tokens: tuple[Token, ...] = SENTENCE,
    text: str | None = "the child sleeps",
) -> Item:
    return Item(ItemId(handle), tokens, text=text)


def example(
    supervision: Supervision[str],
    handle: str = "s/1",
    tokens: tuple[Token, ...] = SENTENCE,
    text: str | None = "the child sleeps",
) -> Example[Supervision[str]]:
    return Example(item(handle, tokens, text), supervision)


def analysis() -> Derivation[str]:
    nodes = {
        0: DerivationNode(TokenSpan(0, 3), "S", children=(1, 4)),
        1: DerivationNode(TokenSpan(0, 2), "NP", children=(2, 3)),
        2: DerivationNode(TokenSpan(0, 1), "NP/N"),
        3: DerivationNode(TokenSpan(1, 2), "N"),
        4: DerivationNode(TokenSpan(2, 3), "S\\NP"),
    }

    return Derivation(0, nodes)


def positions(*candidates: tuple[str, ...]) -> SpanConstraints[str]:
    return SpanConstraints.from_positions(
        tuple(frozenset(position) for position in candidates)
    )


def test_the_alphabet_is_the_size_of_the_two_inventories() -> None:
    assert hasher().alphabet == Alphabet(len(SURFACES) + 1, len(CATEGORIES))


def test_the_adapter_publishes_the_two_inventories_it_codes_against() -> None:
    adapter = hasher()

    assert adapter.lexicon == lexicon()
    assert adapter.categories == categories()


def test_a_surface_form_hashes_to_the_code_the_inventory_declares() -> None:
    hashed = hasher()(example(Unlabeled()))
    declared = lexicon()

    assert hashed.symbols == tuple(declared.index(word) for word in SURFACES)
    assert hashed.item_id == "s/1"


def test_an_unlabeled_example_hashes_to_no_analysis_and_no_response() -> None:
    hashed = hasher()(example(Unlabeled()))

    assert hashed.analyses == ()
    assert hashed.response is None
    assert not hashed.is_ambiguous


def test_a_rating_travels_with_the_scale_it_was_collected_on() -> None:
    response = Scalar(6.0, ORDINAL)
    hashed = hasher()(example(response))

    assert hashed.response == response
    assert hashed.response is not None
    assert hashed.response.scale is ORDINAL
    assert hashed.analyses == ()


def test_a_gold_analysis_hashes_to_one_category_per_position() -> None:
    hashed = hasher()(example(GoldDerivation(analysis())))
    declared = categories()

    assert hashed.analyses == (
        frozenset({declared.index("NP/N")}),
        frozenset({declared.index("N")}),
        frozenset({declared.index("S\\NP")}),
    )
    assert not hashed.is_ambiguous


def test_every_candidate_a_corpus_offered_survives_hashing() -> None:
    hashed = hasher()(example(positions(("NP/N", "N"), ("N",), ("S\\NP", "S"))))
    declared = categories()

    assert hashed.analyses[0] == frozenset(
        {declared.index("NP/N"), declared.index("N")}
    )
    assert hashed.analyses[2] == frozenset(
        {declared.index("S\\NP"), declared.index("S")}
    )
    assert hashed.is_ambiguous


def test_the_order_candidates_arrive_in_does_not_change_the_hashed_form() -> None:
    one = hasher()(example(positions(("NP/N", "N"), ("N",), ("S\\NP",))))
    other = hasher()(example(positions(("N", "NP/N"), ("N",), ("S\\NP",))))

    assert one == other


def test_an_analysis_over_the_wrong_number_of_positions_is_refused() -> None:
    with pytest.raises(ValueError, match="positions"):
        hasher()(example(positions(("N",), ("N",))))


def test_an_analysis_leaving_a_position_uncovered_is_refused() -> None:
    short = Derivation(0, {0: DerivationNode(TokenSpan(0, 1), "NP/N")})

    with pytest.raises(ValueError, match="uncovered"):
        hasher()(example(GoldDerivation(short)))


def test_a_surface_the_inventory_does_not_declare_is_refused() -> None:
    stranger = (Token(0, "kangaroo"),)

    with pytest.raises(UnknownSymbolError, match="kangaroo"):
        hasher()(example(Unlabeled(), tokens=stranger, text="kangaroo"))


def test_a_run_may_declare_a_reserved_stand_in_for_a_surface_it_does_not_hold() -> None:
    stranger = (Token(0, "kangaroo"),)
    hashed = hasher("<unknown>")(example(Unlabeled(), tokens=stranger, text="kangaroo"))

    assert hashed.symbols == (lexicon().index("<unknown>"),)


def test_an_ordinary_symbol_cannot_stand_in_for_an_undeclared_surface() -> None:
    with pytest.raises(ValueError, match="ordinary symbol"):
        hasher("the")


def test_a_stand_in_the_inventory_does_not_declare_is_refused() -> None:
    with pytest.raises(UnknownSymbolError, match="kangaroo"):
        hasher("kangaroo")


def test_a_category_the_inventory_does_not_declare_is_refused() -> None:
    with pytest.raises(UnknownSymbolError, match="PP"):
        hasher()(example(positions(("PP",), ("N",), ("S\\NP",))))


def test_an_arm_the_adapter_does_not_read_is_refused_by_name() -> None:
    with pytest.raises(UnreadableArmError) as raised:
        hasher()(example(GoldDependencies(())))

    assert raised.value.arm == "GoldDependencies"
    assert "SpanConstraints" in raised.value.accepted


def test_nonpositional_constraints_have_no_dense_token_analyses() -> None:
    hashed = hasher()(example(SpanConstraints((), len(SURFACES))))

    assert hashed.analyses == ()


def test_the_arms_the_adapter_reads_are_the_ones_it_has_a_dense_form_for() -> None:
    assert {arm.__name__ for arm in hasher().accepts} == {
        "Unlabeled",
        "Scalar",
        "SpanConstraints",
        "GoldDerivation",
    }


def test_an_item_with_no_text_hashes_like_any_other() -> None:
    hashed = hasher()(example(Unlabeled(), text=None))

    assert hashed.symbols == tuple(lexicon().index(word) for word in SURFACES)


def test_a_hashed_item_refuses_a_surface_code_its_alphabet_cannot_reach() -> None:
    alphabet = hasher().alphabet

    with pytest.raises(ValueError, match="could have minted"):
        HashedItem(ItemId("s/1"), alphabet, (alphabet.lexical,))

    with pytest.raises(ValueError, match="could have minted"):
        HashedItem(ItemId("s/1"), alphabet, (-1,))


def test_a_hashed_item_refuses_a_category_code_its_alphabet_cannot_reach() -> None:
    alphabet = hasher().alphabet

    with pytest.raises(ValueError, match="could have minted"):
        HashedItem(ItemId("s/1"), alphabet, (0,), (frozenset({alphabet.categorial}),))


def test_a_hashed_item_refuses_analyses_over_a_different_number_of_positions() -> None:
    alphabet = hasher().alphabet

    with pytest.raises(ValueError, match="analyses over"):
        HashedItem(ItemId("s/1"), alphabet, (0, 1), (frozenset({0}),))


def test_a_hashed_item_refuses_a_position_admitting_no_category() -> None:
    alphabet = hasher().alphabet

    with pytest.raises(ValueError, match="admits no category"):
        HashedItem(ItemId("s/1"), alphabet, (0,), (frozenset(),))


def test_a_hashed_item_with_no_id_is_refused() -> None:
    with pytest.raises(ValueError, match="has no id"):
        HashedItem(ItemId(""), hasher().alphabet, ())


def test_two_hashed_items_over_one_example_are_one_value() -> None:
    adapter = hasher()
    one = adapter(example(Scalar(6.0, ORDINAL)))
    other = adapter(example(Scalar(6.0, ORDINAL)))

    assert one == other
    assert hash(one) == hash(other)
    assert one != adapter(example(Scalar(3.0, ORDINAL)))
    assert one != adapter(example(Unlabeled()))


def test_a_hashed_item_is_not_equal_to_something_else() -> None:
    assert hasher()(example(Unlabeled())) != (0, 1, 2)


def test_the_representation_names_the_id_and_the_codes() -> None:
    rendered = repr(hasher()(example(Unlabeled())))

    assert rendered.startswith("HashedItem('s/1', Alphabet(")
    assert "(1, 2, 3)" in rendered


def test_nothing_can_be_attached_to_a_hashed_item_after_it_is_built() -> None:
    hashed = hasher()(example(Unlabeled()))

    with pytest.raises(AttributeError):
        setattr(hashed, PROBE, 0)


def test_featurizer_resources_hold_what_they_were_given() -> None:
    hasher = ItemHasher(Inventory("lexicon", ("known",)), Inventory("categories", ()))
    tensorizer = TorchCodeTensorizer()
    lookup = ItemLookup(())
    resources = HashedResources(hasher, tensorizer, lookup)

    assert resources.hasher is hasher
    assert resources.tensorizer is tensorizer
    assert resources.lookup is lookup
    assert isinstance(resources, FeaturizerResources)

    with pytest.raises(AttributeError, match="immutable"):
        resources.__setattr__("_hasher", hasher)

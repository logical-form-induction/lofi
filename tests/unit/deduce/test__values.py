from math import inf, isclose, log

import pytest

from lofi.deduce._values import Derivation, Edge, ItemKey, PrunedMass

LEFT = ItemKey((0, 1), "NP/N", "lex")
RIGHT = ItemKey((1, 2), "N", "lex")


def axiom(reading: str) -> Edge[str, float]:
    return Edge("lex", (), reading, -0.5)


def combine(children: tuple[ItemKey[str], ...], reading: str) -> Edge[str, float]:
    return Edge("fa", children, reading, -0.25, split=children[0].span[1])


def test_a_key_is_a_span_a_reading_and_a_tag() -> None:
    key = ItemKey((0, 3), "S", "right")

    assert key.span == (0, 3)
    assert key.reading == "S"
    assert key.nf_tag == "right"


def test_a_key_takes_the_empty_tag_where_no_filter_runs() -> None:
    assert ItemKey((0, 1), "N").nf_tag == ""


def test_keys_agreeing_on_all_three_fields_are_one_item() -> None:
    one = ItemKey((0, 2), "S/NP", "left")
    other = ItemKey((0, 2), "S/NP", "left")

    assert one == other
    assert hash(one) == hash(other)
    assert len({one, other}) == 1


@pytest.mark.parametrize(
    "other",
    [
        ItemKey((0, 3), "S/NP", "left"),
        ItemKey((0, 2), "S", "left"),
        ItemKey((0, 2), "S/NP", "right"),
    ],
)
def test_a_key_differing_in_any_field_is_another_item(other: ItemKey[str]) -> None:
    assert ItemKey((0, 2), "S/NP", "left") != other


def test_a_key_is_not_equal_to_a_value_of_another_kind() -> None:
    assert ItemKey((0, 1), "N") != ((0, 1), "N", "")


def test_a_key_takes_on_nothing_after_it_is_built() -> None:
    key = ItemKey((0, 1), "N")

    for field in ("span", "reading", "nf_tag", "score"):
        with pytest.raises(AttributeError):
            setattr(key, field, None)

    assert not hasattr(key, "__dict__")


def test_a_key_renders_its_three_fields() -> None:
    assert repr(ItemKey((0, 2), "S", "left")) == (
        "ItemKey(span=(0, 2), reading='S', nf_tag='left')"
    )


def test_an_edge_names_the_step_that_drew_it() -> None:
    edge = combine((LEFT, RIGHT), "NP")

    assert edge.rule_name == "fa"
    assert edge.children == (LEFT, RIGHT)
    assert edge.output == "NP"
    assert edge.weight == -0.25
    assert edge.split == 1
    assert edge.order == 0
    assert edge.arity == 2


def test_an_axiom_edge_consumes_nothing() -> None:
    edge = axiom("N")

    assert edge.children == ()
    assert edge.arity == 0
    assert edge.split is None


def test_edges_agreeing_on_every_field_are_one_edge() -> None:
    one = combine((LEFT, RIGHT), "NP")
    other = combine((LEFT, RIGHT), "NP")

    assert one == other
    assert hash(one) == hash(other)


@pytest.mark.parametrize(
    "other",
    [
        Edge("ba", (LEFT, RIGHT), "NP", -0.25, split=1),
        Edge("fa", (RIGHT, LEFT), "NP", -0.25, split=1),
        Edge("fa", (LEFT, RIGHT), "S", -0.25, split=1),
        Edge("fa", (LEFT, RIGHT), "NP", -0.25, split=None),
        Edge("fa", (LEFT, RIGHT), "NP", -0.25, split=1, order=1),
    ],
)
def test_an_edge_differing_in_any_field_but_the_weight_is_another_edge(
    other: Edge[str, float],
) -> None:
    assert combine((LEFT, RIGHT), "NP") != other


def test_two_weightings_of_one_step_are_different_edges_that_hash_alike() -> None:
    cheap = Edge("fa", (LEFT, RIGHT), "NP", -0.25, split=1)
    dear = Edge("fa", (LEFT, RIGHT), "NP", -8.0, split=1)

    assert cheap != dear
    assert hash(cheap) == hash(dear)


def test_an_edge_hashes_whatever_the_algebra_weighs_it_with() -> None:
    # edge hashes exclude semiring weights, which may be unhashable.
    edge = Edge("fa", (LEFT, RIGHT), "NP", [-0.25], split=1)

    with pytest.raises(TypeError):
        hash(edge.weight)

    assert isinstance(hash(edge), int)


def test_an_edge_is_not_equal_to_a_value_of_another_kind() -> None:
    assert combine((LEFT, RIGHT), "NP") != ("fa", (LEFT, RIGHT), "NP", -0.25, 1, 0)


def test_an_edge_takes_on_nothing_after_it_is_built() -> None:
    edge = combine((LEFT, RIGHT), "NP")

    for field in ("rule_name", "children", "weight", "split", "order", "inside"):
        with pytest.raises(AttributeError):
            setattr(edge, field, None)

    assert not hasattr(edge, "__dict__")


def test_an_edge_renders_its_six_fields() -> None:
    rendered = repr(Edge("fa", (), "NP", -0.25))

    assert "rule_name='fa'" in rendered
    assert "output='NP'" in rendered
    assert "weight=-0.25" in rendered
    assert "split=None" in rendered
    assert "order=0" in rendered


def spine(depth: int) -> Derivation[str, float]:
    """Return a derivation whose left spine is one step deep for each level.

    Parameters
    ----------
    depth
        How many binary steps to stack.

    Returns
    -------
    Derivation
        The derivation, concluding a span of ``depth`` tokens.
    """
    edge = axiom("N")
    tree = Derivation(ItemKey((0, 1), "N", "lex"), edge)

    for end in range(2, depth + 1):
        leaf_key = ItemKey((end - 1, end), "N", "lex")
        leaf = Derivation(leaf_key, axiom("N"))
        children = (tree.key, leaf_key)
        step = combine(children, "N")
        tree = Derivation(ItemKey((0, end), "N", "step"), step, (tree, leaf))

    return tree


def depth_of(tree: Derivation[str, float]) -> int:
    if not tree.children:
        return 1

    return 1 + max(depth_of(child) for child in tree.children)


def test_a_derivation_is_an_item_a_step_and_its_sub_derivations() -> None:
    tree = spine(2)

    assert tree.key == ItemKey((0, 2), "N", "step")
    assert tree.edge.rule_name == "fa"
    assert len(tree.children) == 2
    assert tree.children[0].key == ItemKey((0, 1), "N", "lex")


def test_an_axiom_derivation_has_no_sub_derivations() -> None:
    assert spine(1).children == ()


def test_two_derivations_of_one_item_stay_two_objects() -> None:
    one = spine(3)
    other = spine(3)

    assert one.key == other.key
    assert one != other
    assert len({one, other}) == 2


def test_a_derivation_takes_on_nothing_after_it_is_built() -> None:
    tree = spine(2)

    for field in ("key", "edge", "children", "inside"):
        with pytest.raises(AttributeError):
            setattr(tree, field, None)

    assert not hasattr(tree, "__dict__")


def test_a_derivation_renders_its_item_and_its_rule() -> None:
    rendered = repr(spine(2))

    assert "edge='fa'" in rendered
    assert "children=2" in rendered


def test_an_edge_reaches_its_premises_and_nothing_under_them() -> None:
    deep = spine(21)
    fresh = (ItemKey((0, 20), "N", "step"), ItemKey((20, 21), "N", "lex"))
    rebuilt = Edge("fa", fresh, "N", -0.25, split=20)

    assert depth_of(deep) == 21
    assert deep.edge.children is not fresh
    assert deep.edge == rebuilt
    assert hash(deep.edge) == hash(rebuilt)


def test_a_key_names_the_exact_conclusion() -> None:
    assert ItemKey((0, 2), "S[dcl]") != ItemKey((0, 2), "S[b]")
    assert ItemKey((0, 2), "S[dcl]").reading == "S[dcl]"
    assert ItemKey((0, 2), "S[dcl]") == ItemKey((0, 2), "S[dcl]")
    assert ItemKey((0, 2), "S[dcl]") != ItemKey((0, 3), "S[dcl]")


def test_a_key_carries_the_normal_form_tag_it_was_given() -> None:
    assert ItemKey((0, 2), "S[dcl]", "right").nf_tag == "right"
    assert ItemKey((0, 2), "S[dcl]") != ItemKey((0, 2), "S[dcl]", "left")


def test_pruned_mass_reads_the_probability_and_share_a_beam_removed() -> None:
    mass = PrunedMass(log(0.8), log(0.6))

    assert isclose(mass.lost, 0.2)
    assert isclose(mass.lost_fraction, 0.25)
    assert PrunedMass(-inf, -inf).lost == 0.0
    assert PrunedMass(-inf, -inf).lost_fraction == 0.0

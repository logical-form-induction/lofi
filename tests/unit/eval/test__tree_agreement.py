from __future__ import annotations

from copy import copy, deepcopy
from pickle import dumps, loads
from typing import ClassVar, Final, cast

import pytest

import lofi.eval._tree_agreement as tree_agreement_module
from lofi.eval import Alignment, Contingency
from lofi.eval._tree_agreement import (
    TypeTreeAgreement,
    normalized_tree_distance,
    type_tree_agreement,
)
from lofi.terms import (
    Operator,
    Signature,
    Term,
    TermStore,
    TreeEditDistance,
    Vocabulary,
    class_sensitive_relabel,
)


def _signature(name: str, primitives: tuple[str, ...]) -> Signature:
    """Build one type signature with shared constructor declarations."""
    return Signature(
        name,
        ("Ty",),
        (
            *(Operator(primitive, "Ty", family=primitive) for primitive in primitives),
            Operator(
                "pair",
                "Ty",
                arg_sorts=("Ty", "Ty"),
                family="constructor",
            ),
            Operator(
                "within",
                "Ty",
                arg_sorts=("Ty", "Ty"),
                family="constructor",
            ),
            Operator(
                "left",
                "Ty",
                arg_sorts=("Ty", "Ty"),
                family="left-constructor",
            ),
            Operator(
                "right",
                "Ty",
                arg_sorts=("Ty", "Ty"),
                family="right-constructor",
            ),
        ),
    )


PREDICTED_SIGNATURE: Final = _signature("predicted", ("u0", "u1", "u2"))
GOLD_SIGNATURE: Final = _signature("gold", ("g0", "g1", "g2"))
PREDICTED_STORE: Final = TermStore(PREDICTED_SIGNATURE)
GOLD_STORE: Final = TermStore(GOLD_SIGNATURE)
U0: Final = PREDICTED_STORE.op(PREDICTED_SIGNATURE.by_name("u0"))
U1: Final = PREDICTED_STORE.op(PREDICTED_SIGNATURE.by_name("u1"))
U2: Final = PREDICTED_STORE.op(PREDICTED_SIGNATURE.by_name("u2"))
G0: Final = GOLD_STORE.op(GOLD_SIGNATURE.by_name("g0"))
G1: Final = GOLD_STORE.op(GOLD_SIGNATURE.by_name("g1"))
G2: Final = GOLD_STORE.op(GOLD_SIGNATURE.by_name("g2"))


def _alignment(
    mapping: dict[str, str] | None = None,
    *,
    gold_labels: tuple[str, ...] = ("g0", "g1", "g2"),
) -> Alignment:
    """Build a frozen primitive alignment over complete inventories."""
    counts = {
        (predicted, gold): 0 for predicted in ("u0", "u1", "u2") for gold in gold_labels
    }

    return Alignment(
        {"u0": "g0", "u1": "g1", "u2": "g2"} if mapping is None else mapping,
        Contingency(counts, split="dev"),
    )


def _node(store: TermStore, name: str, left: Term, right: Term) -> Term:
    """Build one binary fixture node."""
    return store.op(store.signature.by_name(name), left, right)


def test_aligned_identity_has_zero_distance_and_exact_shape_agreement() -> None:
    predicted = _node(PREDICTED_STORE, "pair", U0, U1)
    gold = _node(GOLD_STORE, "pair", G0, G1)
    predicted_nodes = PREDICTED_STORE.local_nodes
    gold_nodes = GOLD_STORE.local_nodes
    predicted_pair = PREDICTED_SIGNATURE.by_name("pair")
    gold_pair = GOLD_SIGNATURE.by_name("pair")

    report = type_tree_agreement(
        {"word": predicted},
        {"word": gold},
        (PREDICTED_STORE, GOLD_STORE),
        alignment=_alignment(),
    )

    assert report.vocabulary_size == 1
    assert report.mean_normalized_distance == 0.0
    assert report.exact_match_rate == 1.0
    assert report.arity_agreement == 1.0
    assert report.depth_difference_histogram == ((0, 1),)
    assert report.depth_spearman is None
    assert PREDICTED_STORE.local_nodes == predicted_nodes
    assert GOLD_STORE.local_nodes == gold_nodes
    assert predicted_pair == gold_pair
    assert predicted_pair is not gold_pair


def test_constants_are_compared_by_name_across_vocabularies() -> None:
    predicted_signature = Signature(
        "predicted-symbol",
        ("Ty",),
        (),
        vocabularies=(Vocabulary("Ty", ("distractor", "shared")),),
    )
    gold_signature = Signature(
        "gold-symbol",
        ("Ty",),
        (),
        vocabularies=(Vocabulary("Ty", ("shared",)),),
    )
    predicted_store = TermStore(predicted_signature)
    gold_store = TermStore(gold_signature)
    predicted = predicted_store.const("Ty", "shared")
    gold = gold_store.const("Ty", "shared")
    alignment = Alignment(
        {"#const[Ty]": "#const[Ty]"},
        Contingency({("#const[Ty]", "#const[Ty]"): 1}, split="dev"),
    )

    report = type_tree_agreement(
        {"word": predicted},
        {"word": gold},
        (predicted_store, gold_store),
        alignment=alignment,
    )

    assert predicted.payload != gold.payload
    assert report.mean_normalized_distance == 0.0
    assert report.exact_match_rate == 1.0


def test_primitive_alignment_preserves_declared_features() -> None:
    predicted_signature = Signature(
        "predicted-featured",
        ("Ty",),
        (Operator("u", "Ty"),),
        features={"u": ("marked",)},
    )
    gold_signature = Signature(
        "gold-featured",
        ("Ty",),
        (Operator("g", "Ty"),),
        features={"g": ("marked",)},
    )
    predicted_store = TermStore(predicted_signature)
    gold_store = TermStore(gold_signature)
    predicted = predicted_store.op(predicted_signature.featured("u", "marked"))
    marked_gold = gold_store.op(gold_signature.featured("g", "marked"))
    bare_gold = gold_store.op(gold_signature.by_name("g"))
    alignment = Alignment(
        {"u": "g"},
        Contingency({("u", "g"): 1}, split="dev"),
    )

    matched = type_tree_agreement(
        {"word": predicted},
        {"word": marked_gold},
        (predicted_store, gold_store),
        alignment=alignment,
    )
    unmarked = type_tree_agreement(
        {"word": predicted},
        {"word": bare_gold},
        (predicted_store, gold_store),
        alignment=alignment,
    )

    assert matched.mean_normalized_distance == 0.0
    assert matched.exact_match_rate == 1.0
    assert unmarked.mean_normalized_distance == 0.5
    assert unmarked.exact_match_rate == 0.0


def test_trees_sharing_no_aligned_node_have_normalized_distance_one() -> None:
    predicted = _node(PREDICTED_STORE, "left", U0, U0)
    gold = _node(GOLD_STORE, "right", G1, G1)

    report = type_tree_agreement(
        {"word": predicted},
        {"word": gold},
        (PREDICTED_STORE, GOLD_STORE),
        alignment=_alignment(),
    )

    assert report.mean_normalized_distance == 1.0
    assert report.exact_match_rate == 0.0
    assert report.arity_agreement == 1.0


def test_class_sensitive_cost_is_cheaper_within_an_operator_family() -> None:
    pair = GOLD_SIGNATURE.by_name("pair")
    within = GOLD_SIGNATURE.by_name("within")
    across = GOLD_SIGNATURE.by_name("left")

    assert class_sensitive_relabel((pair, None), (within, None)) == 1
    assert class_sensitive_relabel((pair, None), (across, None)) == 2


def test_report_aggregates_depth_differences_tied_ranks_and_root_arities() -> None:
    predicted_one = _node(PREDICTED_STORE, "pair", U0, U1)
    predicted_two = _node(PREDICTED_STORE, "pair", predicted_one, U2)
    gold_one = _node(GOLD_STORE, "pair", G0, G1)
    gold_other_one = _node(GOLD_STORE, "pair", G1, G2)
    gold_two = _node(GOLD_STORE, "pair", gold_one, G2)
    predicted = {
        "a": U0,
        "b": predicted_one,
        "c": predicted_two,
        "d": _node(PREDICTED_STORE, "pair", predicted_one, U0),
    }
    gold = {
        "d": gold_two,
        "c": gold_other_one,
        "b": gold_one,
        "a": G0,
    }

    report = type_tree_agreement(
        predicted,
        gold,
        (PREDICTED_STORE, GOLD_STORE),
        alignment=_alignment(),
    )

    assert report.vocabulary_size == 4
    assert report.arity_agreement == 1.0
    assert report.depth_difference_histogram == ((0, 3), (1, 1))
    assert report.depth_spearman == pytest.approx(5.0 / 6.0)


def test_root_arity_is_compared_separately_from_tree_distance() -> None:
    predicted = _node(PREDICTED_STORE, "pair", U0, U1)
    report = type_tree_agreement(
        {"word": predicted},
        {"word": G0},
        (PREDICTED_STORE, GOLD_STORE),
        alignment=_alignment(),
    )

    assert report.arity_agreement == 0.0
    assert report.depth_difference_histogram == ((1, 1),)


def test_depth_spearman_is_none_for_empty_short_or_constant_rank_inputs() -> None:
    empty = type_tree_agreement(
        {},
        {},
        (PREDICTED_STORE, GOLD_STORE),
        alignment=_alignment(),
    )
    constant = type_tree_agreement(
        {"a": U0, "b": U1},
        {"a": G0, "b": _node(GOLD_STORE, "pair", G0, G1)},
        (PREDICTED_STORE, GOLD_STORE),
        alignment=_alignment(),
    )

    assert empty.vocabulary_size == 0
    assert empty.mean_normalized_distance is None
    assert empty.exact_match_rate is None
    assert empty.arity_agreement is None
    assert empty.depth_difference_histogram == ()
    assert empty.depth_spearman is None
    assert constant.depth_spearman is None


def test_depth_spearman_preserves_rank_direction() -> None:
    predicted_one = _node(PREDICTED_STORE, "pair", U0, U1)
    predicted_two = _node(PREDICTED_STORE, "pair", predicted_one, U2)
    gold_one = _node(GOLD_STORE, "pair", G0, G1)
    gold_two = _node(GOLD_STORE, "pair", gold_one, G2)
    report = type_tree_agreement(
        {"a": U0, "b": predicted_one, "c": predicted_two},
        {"a": gold_two, "b": gold_one, "c": G0},
        (PREDICTED_STORE, GOLD_STORE),
        alignment=_alignment(),
    )

    assert report.depth_spearman == -1.0


def test_normalized_distance_divides_the_kernel_by_the_paired_node_count() -> None:
    distance: TreeEditDistance = TreeEditDistance()
    one = _node(GOLD_STORE, "pair", G0, G1)
    two = _node(GOLD_STORE, "pair", one, G2)

    assert normalized_tree_distance(distance, one, one) == 0.0
    assert normalized_tree_distance(distance, G0, G1) == 1.0
    assert normalized_tree_distance(distance, one, two) == distance(one, two) / (
        one.size + two.size
    )
    assert 0.0 < normalized_tree_distance(distance, one, two) < 1.0


class DistanceSpy:
    """A tree-distance spy for delegation checks."""

    calls: ClassVar[list[tuple[int, int]]] = []

    def __call__(self, left: Term, right: Term, /) -> int:
        """Record one delegated edit calculation."""
        self.calls.append((left.size, right.size))

        return 0


def test_every_edit_calculation_delegates_to_the_term_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DistanceSpy.calls = []
    monkeypatch.setattr(tree_agreement_module, "TreeEditDistance", DistanceSpy)

    report = type_tree_agreement(
        {"a": U0, "b": U1},
        {"a": G0, "b": G1},
        (PREDICTED_STORE, GOLD_STORE),
        alignment=_alignment(),
    )

    assert DistanceSpy.calls == [(1, 1), (1, 1)]
    assert report.mean_normalized_distance == 0.0


class RefusesScoring:
    """A kernel constructor that fails if validation reaches scoring."""

    def __init__(self) -> None:
        msg = "validation reached tree scoring"
        raise AssertionError(msg)


def test_mismatched_vocabulary_keys_are_rejected_before_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tree_agreement_module, "TreeEditDistance", RefusesScoring)

    with pytest.raises(ValueError, match="mismatched keys"):
        type_tree_agreement(
            {"predicted-only": U0},
            {"gold-only": G0},
            (PREDICTED_STORE, GOLD_STORE),
            alignment=_alignment(),
        )


@pytest.mark.parametrize(
    ("alignment", "message"),
    [
        (_alignment({}), "has no image"),
        (
            _alignment({"u0": "pair"}, gold_labels=("pair",)),
            "has arity 2, not zero",
        ),
        (
            _alignment({"u0": "missing"}, gold_labels=("missing",)),
            "absent from the gold signature",
        ),
    ],
)
def test_invalid_primitive_images_are_rejected_before_scoring(
    monkeypatch: pytest.MonkeyPatch,
    alignment: Alignment,
    message: str,
) -> None:
    monkeypatch.setattr(tree_agreement_module, "TreeEditDistance", RefusesScoring)

    with pytest.raises(ValueError, match=message):
        type_tree_agreement(
            {"word": U0},
            {"word": G0},
            (PREDICTED_STORE, GOLD_STORE),
            alignment=alignment,
        )


def test_incompatible_constructor_declarations_are_rejected_before_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed_signature = Signature(
        "changed",
        ("Ty",),
        (
            Operator("u0", "Ty", family="u0"),
            Operator(
                "pair",
                "Ty",
                arg_sorts=("Ty", "Ty"),
                family="changed-constructor",
            ),
        ),
    )
    changed_store = TermStore(changed_signature)
    atom = changed_store.op(changed_signature.by_name("u0"))
    predicted = _node(changed_store, "pair", atom, atom)
    monkeypatch.setattr(tree_agreement_module, "TreeEditDistance", RefusesScoring)

    with pytest.raises(ValueError, match="constructor 'pair' is incompatible"):
        type_tree_agreement(
            {"word": predicted},
            {"word": _node(GOLD_STORE, "pair", G0, G0)},
            (changed_store, GOLD_STORE),
            alignment=_alignment({"u0": "g0"}),
        )


def test_incompatible_primitive_result_sort_is_rejected_before_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predicted_signature = Signature(
        "predicted-sort",
        ("Ty", "Index"),
        (Operator("u", "Ty"),),
    )
    gold_signature = Signature(
        "gold-sort",
        ("Ty", "Index"),
        (Operator("g", "Index"),),
    )
    predicted_store = TermStore(predicted_signature)
    gold_store = TermStore(gold_signature)
    predicted = predicted_store.op(predicted_signature.by_name("u"))
    gold = gold_store.op(gold_signature.by_name("g"))
    alignment = Alignment(
        {"u": "g"},
        Contingency({("u", "g"): 1}, split="dev"),
    )
    monkeypatch.setattr(tree_agreement_module, "TreeEditDistance", RefusesScoring)

    with pytest.raises(ValueError, match="result sort 'Index', not 'Ty'"):
        type_tree_agreement(
            {"word": predicted},
            {"word": gold},
            (predicted_store, gold_store),
            alignment=alignment,
        )


def test_missing_aligned_feature_is_rejected_before_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predicted_signature = Signature(
        "predicted-featured",
        ("Ty",),
        (Operator("u", "Ty"),),
        features={"u": ("marked",)},
    )
    gold_signature = Signature(
        "gold-unfeatured",
        ("Ty",),
        (Operator("g", "Ty"),),
    )
    predicted_store = TermStore(predicted_signature)
    gold_store = TermStore(gold_signature)
    predicted = predicted_store.op(predicted_signature.featured("u", "marked"))
    gold = gold_store.op(gold_signature.by_name("g"))
    alignment = Alignment(
        {"u": "g"},
        Contingency({("u", "g"): 1}, split="dev"),
    )
    monkeypatch.setattr(tree_agreement_module, "TreeEditDistance", RefusesScoring)

    with pytest.raises(ValueError, match="has no feature 'marked'"):
        type_tree_agreement(
            {"word": predicted},
            {"word": gold},
            (predicted_store, gold_store),
            alignment=alignment,
        )


def test_foreign_terms_are_rejected_before_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    foreign_store = TermStore(PREDICTED_SIGNATURE)
    foreign = foreign_store.op(PREDICTED_SIGNATURE.by_name("u0"))
    monkeypatch.setattr(tree_agreement_module, "TreeEditDistance", RefusesScoring)

    with pytest.raises(ValueError, match="outside its supplied store"):
        type_tree_agreement(
            {"word": foreign},
            {"word": G0},
            (PREDICTED_STORE, GOLD_STORE),
            alignment=_alignment(),
        )


def test_alignment_must_be_the_frozen_public_value() -> None:
    with pytest.raises(TypeError, match="not frozen Alignment"):
        type_tree_agreement(
            {"word": U0},
            {"word": G0},
            (PREDICTED_STORE, GOLD_STORE),
            alignment=cast("Alignment", 1),
        )


def test_report_values_are_frozen_hashable_copyable_and_pickleable() -> None:
    report = type_tree_agreement(
        {"word": U0},
        {"word": G0},
        (PREDICTED_STORE, GOLD_STORE),
        alignment=_alignment(),
    )

    assert copy(report) == report
    assert copy(report) is not report
    assert deepcopy(report) == report
    assert loads(dumps(report)) == report
    assert hash(loads(dumps(report))) == hash(report)
    assert "__dict__" not in dir(report)

    with pytest.raises(AttributeError, match="immutable"):
        report.__setattr__("_vocabulary_size", 2)

    with pytest.raises(AttributeError, match="immutable"):
        report.__delattr__("_vocabulary_size")


@pytest.mark.parametrize(
    "report",
    [
        (0, 0.0, None, None, (), None),
        (1, None, 1.0, 1.0, ((0, 1),), None),
        (1, 0.0, 1.0, 1.0, ((0, 2),), None),
        (1, 0.0, 1.0, 1.0, ((0, 0), (1, 1)), None),
        (1, 0.0, 1.0, 1.0, ((1, 1), (0, 1)), None),
        (1, 0.0, 1.0, 1.0, ((0, 1),), 2.0),
    ],
)
def test_report_constructor_rejects_inconsistent_values(
    report: tuple[
        int,
        float | None,
        float | None,
        float | None,
        tuple[tuple[int, int], ...],
        float | None,
    ],
) -> None:
    with pytest.raises(
        ValueError,
        match=r"tree-agreement|histogram|correlation|missing rate",
    ):
        TypeTreeAgreement(report)

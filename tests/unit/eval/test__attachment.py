import json
from copy import copy, deepcopy
from pathlib import Path
from pickle import dumps, loads
from typing import TYPE_CHECKING, Final, cast

import pytest
from hypothesis import strategies as st

from lofi._types import ItemId
from lofi.data import (
    AttachmentTree,
    Dependency,
    Derivation,
    DerivationNode,
    Token,
    TokenSpan,
    attachment_tree,
)
from lofi.data.deps import (
    BiskHockenmaierDependencyProjector,
    HeadProjection,
    InducedDependencyState,
    project_bisk_hockenmaier_dependencies,
)
from lofi.eval._attachment import (
    ATTACHMENT_DIRECTIONS,
    WSJ_LENGTH_BANDS,
    AttachmentScorer,
    AttachmentScoreRow,
    LengthBand,
)
from lofi.eval._ports import Metric
from lofi.eval._values import ConventionVector, MetricResult, Resampling
from lofi.eval.laws import MetricFixture, check_metric_laws
from lofi.spec import HeadSelection

if TYPE_CHECKING:
    from lofi._types import JsonValue
    from lofi.data.deps import CoindexedPath
    from lofi.eval._attachment import AttachmentDirection
    from lofi.eval._values import Aggregation, CoveragePenalty, PunctuationPolicy

REFERENCE_POINTS_PATH: Final = (
    Path(__file__).resolve().parents[3]
    / "lofi"
    / "configs"
    / "eval"
    / "bisk_hockenmaier_2012.json"
)


# ---------------------------------------------------------------------------
# the seven-token fixture: a modifier on each conjunct, a coordination, and a
# sentence-final modifier, projected through the induced-grammar convention
# ---------------------------------------------------------------------------

TOKENS = ("big", "cats", "and", "small", "dogs", "sleep", "quietly")
PREDICTED_HEADS = (2, 6, 2, 5, 3, 0, 6)
PREDICTED_ARCS = frozenset({(2, 1), (5, 4), (2, 3), (3, 5), (6, 2), (6, 7), (0, 6)})
GOLD_HEADS = (2, 6, 5, 5, 2, 0, 6)


class _Algebra:
    """Structural operations over the fixture's string categories."""

    def __init__(self) -> None:
        self._functors = {
            "N/N": ("N", "N", True),
            "S\\N": ("S", "N", False),
            "S\\S": ("S", "S", True),
        }

    def is_functor(self, category: str, /) -> bool:
        return category in self._functors

    def result(self, category: str, /) -> str:
        return self._functors[category][0]

    def argument(self, category: str, /) -> str:
        return self._functors[category][1]

    def equivalent(self, left: str, right: str, /) -> bool:
        return left == right

    def shares_head(self, category: str, /) -> bool:
        return self._functors[category][2]

    def coindices(self, category: str, /) -> tuple[CoindexedPath, ...]:
        del category
        return ()


def _node(
    start: int,
    end: int,
    label: str,
    rule: str | None = None,
    children: tuple[int, ...] = (),
) -> DerivationNode[str]:
    return DerivationNode(TokenSpan(start, end), label, rule=rule, children=children)


def _seven_token_projection() -> HeadProjection[
    Dependency, InducedDependencyState[str]
]:
    nodes = {
        0: _node(0, 1, "N/N"),
        1: _node(1, 2, "N"),
        2: _node(0, 2, "N", "forward-modifier", (0, 1)),
        3: _node(2, 3, "conj"),
        4: _node(3, 4, "N/N"),
        5: _node(4, 5, "N"),
        6: _node(3, 5, "N", "forward-modifier", (4, 5)),
        7: _node(2, 5, "N[conj]", "coordination-mark", (3, 6)),
        8: _node(0, 5, "N", "coordination-merge", (2, 7)),
        9: _node(5, 6, "S\\N"),
        10: _node(0, 6, "S", "backward-application", (8, 9)),
        11: _node(6, 7, "S\\S"),
        12: _node(0, 7, "S", "backward-modifier", (10, 11)),
    }
    selections = {
        "forward-modifier": HeadSelection("functor", 0),
        "coordination-mark": HeadSelection("non-absorbing", 1),
        "coordination-merge": HeadSelection("left-conjunct", 0),
        "backward-application": HeadSelection("functor", 1),
        "backward-modifier": HeadSelection("functor", 1),
    }
    projector = BiskHockenmaierDependencyProjector(
        _Algebra(), frozenset({"coordination-mark"})
    )

    return project_bisk_hockenmaier_dependencies(
        Derivation(12, nodes),
        tuple(Token(index, form) for index, form in enumerate(TOKENS)),
        selections,
        projector,
    )


def _tokens() -> tuple[Token, ...]:
    return tuple(Token(index, form) for index, form in enumerate(TOKENS))


def _conventions(
    *,
    punctuation: PunctuationPolicy = "keep",
    minimum_length: int = 0,
    aggregation: Aggregation = "corpus",
    coverage: CoveragePenalty = "ignore",
) -> ConventionVector:
    return ConventionVector(
        punctuation=punctuation,
        unary_chains="keep",
        trivial_spans="keep",
        minimum_length=minimum_length,
        aggregation=aggregation,
        coverage=coverage,
        self_arcs="forbid",
        gold_derivation="canonical",
    )


def _row(
    scorer: AttachmentScorer, band: str, direction: AttachmentDirection
) -> AttachmentScoreRow:
    return next(
        row
        for row in scorer.rows()
        if row.band.name == band and row.direction == direction
    )


# ---------------------------------------------------------------------------
# the projection
# ---------------------------------------------------------------------------


def test_the_published_reference_points_name_their_source_and_verification() -> None:
    fixture: JsonValue = json.loads(REFERENCE_POINTS_PATH.read_text(encoding="utf-8"))

    assert isinstance(fixture, dict)
    assert fixture["metric"] == "unlabeled directed attachment"
    assert fixture["training_band"] == "wsj10"
    source = fixture["source"]
    points = fixture["points"]
    assert isinstance(source, dict)
    assert source["venue"] == "AAAI 2012"
    assert source["figure"] == "Figure 4(a)"
    assert isinstance(source["url"], str)
    assert source["url"].endswith("/AAAI2012/Induction.pdf")
    assert isinstance(points, list)
    assert [point["metric"] for point in points if isinstance(point, dict)] == [
        "dependencies-length-10",
        "dependencies-length-20",
        "dependencies-all-lengths",
    ]
    assert [point["score_percent"] for point in points if isinstance(point, dict)] == [
        71.5,
        60.3,
        53.3,
    ]


def test_the_seven_token_fixture_projects_to_the_hand_checked_arcs() -> None:
    projection = _seven_token_projection()
    arcs = frozenset(
        (dependency.head, dependency.dependent)
        for dependency in projection.dependencies
    )

    assert arcs == PREDICTED_ARCS
    assert (0, 6) in arcs
    assert len(projection.dependencies) == len(TOKENS)


def test_the_projection_reads_as_a_tree_with_one_head_per_token() -> None:
    tree = attachment_tree(_seven_token_projection(), _tokens())

    assert tree.forms == TOKENS
    assert tree.heads == PREDICTED_HEADS
    assert tree.arcs() == PREDICTED_ARCS
    assert tree.length == 7
    assert tree.punctuation == frozenset()


def test_modifiers_depend_on_their_heads_in_the_projection() -> None:
    tree = attachment_tree(_seven_token_projection(), _tokens())

    assert tree.head_of(1) == 2
    assert tree.head_of(4) == 5
    assert tree.head_of(7) == 6


def test_coordination_chains_conjunct_to_marker_to_conjunct() -> None:
    tree = attachment_tree(_seven_token_projection(), _tokens())

    assert tree.head_of(3) == 2
    assert tree.head_of(5) == 3
    assert tree.head_of(2) == 6
    assert tree.head_of(6) == 0


def test_the_adapter_marks_punctuation_and_rejects_bad_inputs() -> None:
    projection = _seven_token_projection()

    assert attachment_tree(projection, _tokens(), punctuation=(7,)).punctuation == {7}

    with pytest.raises(TypeError, match="not one"):
        attachment_tree(cast("HeadProjection", projection.dependencies), _tokens())

    with pytest.raises(ValueError, match="has index"):
        attachment_tree(projection, tuple(reversed(_tokens())))

    with pytest.raises(ValueError, match="have no head"):
        attachment_tree(projection, (*_tokens(), Token(7, "extra")))

    with pytest.raises(ValueError, match="outside 6 tokens"):
        attachment_tree(projection, _tokens()[:6])

    with pytest.raises(TypeError, match="projected dependency is not one"):
        attachment_tree(
            HeadProjection(projection.heads, cast("tuple[Dependency, ...]", ((0, 6),))),
            _tokens(),
        )


def test_the_adapter_rejects_a_token_with_two_heads() -> None:
    projection = _seven_token_projection()
    doubled = HeadProjection(
        projection.heads,
        (*projection.dependencies, projection.dependencies[0]),
        projection.state,
    )

    with pytest.raises(ValueError, match="more than one head"):
        attachment_tree(doubled, _tokens())


# ---------------------------------------------------------------------------
# the tree value
# ---------------------------------------------------------------------------


def test_a_tree_validates_heads_root_and_reachability() -> None:
    with pytest.raises(ValueError, match="are given"):
        AttachmentTree(("a", "b"), (0,))

    with pytest.raises(ValueError, match="heads itself"):
        AttachmentTree(("a", "b"), (0, 2))

    with pytest.raises(ValueError, match="outside 2 tokens"):
        AttachmentTree(("a", "b"), (0, 3))

    with pytest.raises(ValueError, match="0 tokens attach to the root"):
        AttachmentTree(("a", "b"), (2, 1))

    with pytest.raises(ValueError, match="3 tokens attach to the root"):
        AttachmentTree(("a", "b", "c"), (0, 0, 0))

    with pytest.raises(ValueError, match="2 tokens attach to the root"):
        AttachmentTree(("a", "b", "c"), (0, 1, 0))

    with pytest.raises(ValueError, match="does not reach"):
        AttachmentTree(("a", "b", "c"), (0, 3, 2))

    with pytest.raises(ValueError, match="punctuation position 3"):
        AttachmentTree(("a", "b"), (0, 1), punctuation=(3,))

    with pytest.raises(ValueError, match="punctuation position 0"):
        AttachmentTree(("a", "b"), (0, 1), punctuation=(0,))

    with pytest.raises(TypeError, match="not an integer"):
        AttachmentTree(("a",), cast("tuple[int, ...]", (0.0,)))


def test_an_empty_tree_is_allowed_and_head_of_checks_its_range() -> None:
    empty = AttachmentTree((), ())
    tree = AttachmentTree(("a", "b"), (0, 1), punctuation=(2,))

    assert empty.length == 0
    assert empty.arcs() == frozenset()
    assert tree.head_of(2) == 1

    with pytest.raises(ValueError, match="outside"):
        tree.head_of(0)

    with pytest.raises(ValueError, match="outside"):
        empty.head_of(1)


def test_a_tree_is_a_frozen_value() -> None:
    tree = AttachmentTree(("a", "b"), (0, 1), punctuation=(2,))
    same = AttachmentTree(("a", "b"), (0, 1), punctuation=[2])

    assert tree == same
    assert hash(tree) == hash(same)
    assert tree != AttachmentTree(("a", "b"), (0, 1))
    assert tree.__eq__(object()) is NotImplemented
    assert loads(dumps(tree)) == tree
    assert copy(tree) is tree
    assert deepcopy(tree) is tree
    assert repr(tree) == "AttachmentTree(('a', 'b'), (0, 1), [2])"

    field = "_heads"

    with pytest.raises(AttributeError, match="immutable"):
        setattr(tree, field, (0, 0))

    with pytest.raises(AttributeError, match="immutable"):
        delattr(tree, field)


# ---------------------------------------------------------------------------
# the hand count
# ---------------------------------------------------------------------------


def test_directed_matches_the_hand_count_and_undirected_is_at_least_as_high() -> None:
    scorer = AttachmentScorer(_conventions())
    scorer.update(
        attachment_tree(_seven_token_projection(), _tokens()),
        AttachmentTree(TOKENS, GOLD_HEADS),
        key=ItemId("wsj:1"),
    )
    directed = _row(scorer, "wsj-infinity", "directed").result
    undirected = _row(scorer, "wsj-infinity", "undirected").result

    assert directed.value == pytest.approx(5 / 7)
    assert directed.components["matched"] == 5.0
    assert directed.components["tokens"] == 7.0
    assert undirected.value == pytest.approx(6 / 7)
    assert undirected.components["matched"] == 6.0
    assert directed.value is not None
    assert undirected.value is not None
    assert undirected.value >= directed.value
    assert directed.n == 1
    assert directed.resampling == Resampling("sentence")


def test_a_reversed_arc_counts_undirected_only_and_a_root_arc_has_no_reverse() -> None:
    scorer = AttachmentScorer(_conventions())
    scorer.update(
        AttachmentTree(("a", "b"), (0, 1)),
        AttachmentTree(("a", "b"), (2, 0)),
        key=ItemId("i"),
    )

    assert _row(scorer, "wsj10", "directed").result.value == 0.0
    assert _row(scorer, "wsj10", "undirected").result.value == 0.5


def test_a_perfect_prediction_scores_one_in_both_directions() -> None:
    scorer = AttachmentScorer(_conventions())
    tree = AttachmentTree(TOKENS, GOLD_HEADS)
    scorer.update(tree, tree, key=ItemId("i"))

    assert all(row.result.value == 1.0 for row in scorer.rows())


# ---------------------------------------------------------------------------
# bands
# ---------------------------------------------------------------------------


def test_rows_cross_bands_and_directions_once_in_report_order() -> None:
    scorer = AttachmentScorer(_conventions())
    rows = scorer.rows()

    assert scorer.name == "attachment"
    assert scorer.order_invariant is True
    assert scorer.bands == WSJ_LENGTH_BANDS
    assert tuple(row.name for row in rows) == tuple(
        f"attachment:{band.name}:{direction}"
        for band in WSJ_LENGTH_BANDS
        for direction in ATTACHMENT_DIRECTIONS
    )
    assert all(row.result.value is None and row.result.n == 0 for row in rows)
    assert all(row.conventions == _conventions() for row in rows)


def test_a_band_admits_sentences_up_to_its_limit() -> None:
    scorer = AttachmentScorer(_conventions(aggregation="sentence"))
    short = AttachmentTree(("a", "b"), (0, 1))
    long_forms = tuple(f"w{index}" for index in range(12))
    long_gold = AttachmentTree(long_forms, (0, *range(1, 12)))
    long_wrong = AttachmentTree(long_forms, (0, *(1,) * 11))
    scorer.update(short, short, key=ItemId("short"))
    scorer.update(long_wrong, long_gold, key=ItemId("long"))

    assert _row(scorer, "wsj10", "directed").result.value == 1.0
    assert _row(scorer, "wsj10", "directed").result.n == 1
    assert _row(scorer, "wsj20", "directed").result.n == 2
    assert _row(scorer, "wsj20", "directed").result.value == pytest.approx(
        (1.0 + 2 / 12) / 2
    )
    assert _row(scorer, "wsj-infinity", "directed").result.n == 2


def test_a_band_boundary_is_inclusive() -> None:
    assert LengthBand("ten", 10).admits(10)
    assert not LengthBand("ten", 10).admits(11)
    assert LengthBand("all", None).admits(10**6)


def test_stripping_punctuation_shortens_the_counted_length_and_the_score() -> None:
    forms = ("a", "b", ".")
    gold = AttachmentTree(forms, (0, 1, 1), punctuation=(3,))
    prediction = AttachmentTree(forms, (0, 1, 2))
    kept = AttachmentScorer(_conventions(), (LengthBand("two", 2),))
    stripped = AttachmentScorer(
        _conventions(punctuation="strip"), (LengthBand("two", 2),)
    )
    kept.update(prediction, gold, key=ItemId("i"))
    stripped.update(prediction, gold, key=ItemId("i"))

    assert _row(kept, "two", "directed").result.n == 0
    assert _row(stripped, "two", "directed").result.value == 1.0
    assert _row(stripped, "two", "directed").result.components["tokens"] == 2.0


def test_the_minimum_length_excludes_shorter_sentences_and_admits_equal_ones() -> None:
    scorer = AttachmentScorer(_conventions(minimum_length=3))
    short = AttachmentTree(("a", "b"), (0, 1))
    exact = AttachmentTree(("a", "b", "c"), (0, 1, 1))
    scorer.update(short, short, key=ItemId("short"))
    scorer.update(exact, exact, key=ItemId("exact"))

    assert all(row.result.n == 1 for row in scorer.rows())
    assert all(row.result.components["tokens"] == 3.0 for row in scorer.rows())


def test_a_sentence_of_only_punctuation_is_not_scored_under_strip() -> None:
    scorer = AttachmentScorer(_conventions(punctuation="strip"))
    dot = AttachmentTree((".",), (0,), punctuation=(1,))
    scorer.update(dot, dot, key=ItemId("i"))

    assert all(row.result.n == 0 for row in scorer.rows())


# ---------------------------------------------------------------------------
# aggregation and coverage
# ---------------------------------------------------------------------------


def test_corpus_aggregation_weights_tokens_and_sentence_aggregation_sentences() -> None:
    corpus = AttachmentScorer(_conventions(aggregation="corpus"))
    sentence = AttachmentScorer(_conventions(aggregation="sentence"))
    one = AttachmentTree(("a",), (0,))
    three_gold = AttachmentTree(("a", "b", "c"), (0, 1, 2))
    three_wrong = AttachmentTree(("a", "b", "c"), (0, 1, 1))

    for scorer in (corpus, sentence):
        scorer.update(one, one, key=ItemId("one"))
        scorer.update(three_wrong, three_gold, key=ItemId("three"))

    assert _row(corpus, "wsj10", "directed").result.value == pytest.approx(3 / 4)
    assert _row(sentence, "wsj10", "directed").result.value == pytest.approx(
        (1.0 + 2 / 3) / 2
    )


def test_a_failed_parse_is_ignored_or_scored_at_zero_by_the_coverage_policy() -> None:
    ignore = AttachmentScorer(_conventions(coverage="ignore"))
    zero = AttachmentScorer(_conventions(coverage="zero"))
    gold = AttachmentTree(("a", "b"), (0, 1))

    for scorer in (ignore, zero):
        scorer.update(gold, gold, key=ItemId("parsed"))
        scorer.update(None, gold, key=ItemId("failed"))

    ignored = _row(ignore, "wsj10", "directed").result
    zeroed = _row(zero, "wsj10", "directed").result

    assert ignored.value == 1.0
    assert ignored.n == 1
    assert ignored.components["parsed"] == 1.0
    assert zeroed.value == 0.5
    assert zeroed.n == 2
    assert zeroed.components["parsed"] == 1.0
    assert zeroed.components["tokens"] == 4.0


def test_updates_are_idempotent_by_key_and_order_invariant() -> None:
    forward = AttachmentScorer(_conventions())
    backward = AttachmentScorer(_conventions())
    first = (AttachmentTree(("a", "b"), (0, 1)), AttachmentTree(("a", "b"), (2, 0)))
    second = (AttachmentTree(("c",), (0,)), AttachmentTree(("c",), (0,)))
    forward.update(*first, key=ItemId("1"))
    forward.update(*second, key=ItemId("2"))
    forward.update(*first, key=ItemId("1"))
    backward.update(*second, key=ItemId("2"))
    backward.update(*first, key=ItemId("1"))

    assert forward.rows() == backward.rows()
    assert _row(forward, "wsj10", "directed").result.n == 2

    with pytest.raises(ValueError, match="already scored"):
        forward.update(*second, key=ItemId("1"))


def test_reset_forgets_every_sentence() -> None:
    scorer = AttachmentScorer(_conventions())
    tree = AttachmentTree(("a",), (0,))
    scorer.update(tree, tree, key=ItemId("1"))
    scorer.reset()

    assert all(row.result.n == 0 for row in scorer.rows())


def test_update_rejects_wrong_types_and_mismatched_forms() -> None:
    scorer = AttachmentScorer(_conventions())
    gold = AttachmentTree(("a", "b"), (0, 1))

    with pytest.raises(TypeError, match="gold attachment tree"):
        scorer.update(gold, cast("AttachmentTree", gold.heads), key=ItemId("1"))

    with pytest.raises(TypeError, match="predicted attachment tree"):
        scorer.update(cast("AttachmentTree", gold.heads), gold, key=ItemId("1"))

    with pytest.raises(ValueError, match="scored against"):
        scorer.update(AttachmentTree(("a", "c"), (0, 1)), gold, key=ItemId("1"))

    with pytest.raises(ValueError, match="empty"):
        scorer.update(gold, gold, key=ItemId(""))


def test_the_scorer_validates_its_conventions_and_bands() -> None:
    with pytest.raises(TypeError, match="not a vector"):
        AttachmentScorer(cast("ConventionVector", "corpus"))

    with pytest.raises(TypeError, match="band 0 is not one"):
        AttachmentScorer(_conventions(), cast("tuple[LengthBand, ...]", ("wsj10",)))

    with pytest.raises(ValueError, match="at least one"):
        AttachmentScorer(_conventions(), ())

    with pytest.raises(ValueError, match="share a name"):
        AttachmentScorer(_conventions(), (LengthBand("ten", 10), LengthBand("ten", 20)))

    with pytest.raises(ValueError, match="negative"):
        AttachmentScorer(_conventions(), seed=-1)

    assert AttachmentScorer(_conventions(), seed=3).rows()[0].result.resampling == (
        Resampling("sentence", 3)
    )
    assert AttachmentScorer(_conventions()).conventions == _conventions()


# ---------------------------------------------------------------------------
# the metric protocol
# ---------------------------------------------------------------------------

type _Example = tuple[str, AttachmentTree | None, AttachmentTree]

_POOL: tuple[_Example, ...] = (
    (
        "s-a",
        AttachmentTree(TOKENS, PREDICTED_HEADS),
        AttachmentTree(TOKENS, GOLD_HEADS),
    ),
    ("s-b", AttachmentTree(("a", "b"), (0, 1)), AttachmentTree(("a", "b"), (2, 0))),
    ("s-c", AttachmentTree(("c",), (0,)), AttachmentTree(("c",), (0,))),
    ("s-d", None, AttachmentTree(("d", "e", "."), (0, 1, 1), punctuation=(3,))),
    (
        "s-e",
        AttachmentTree(("d", "e", "."), (0, 1, 2), punctuation=(3,)),
        AttachmentTree(("d", "e", "."), (0, 1, 1), punctuation=(3,)),
    ),
)


def _build() -> AttachmentScorer:
    return AttachmentScorer(
        _conventions(punctuation="strip", minimum_length=2, aggregation="sentence")
    )


def test_the_scorer_holds_to_every_metric_law() -> None:
    check_metric_laws(MetricFixture(_build, st.sampled_from(_POOL), max_examples=25))


def test_the_scorer_is_usable_through_the_protocol_a_report_holds_it_by() -> None:
    held: Metric[AttachmentTree | None, AttachmentTree] = AttachmentScorer(
        _conventions()
    )
    held.update(*_POOL[0][1:], key=ItemId("s-a"))

    assert isinstance(held, Metric)
    assert held.compute().value == pytest.approx(5 / 7)
    assert held.conventions == _conventions()


def test_the_aggregate_is_directed_accuracy_at_any_length() -> None:
    scorer = AttachmentScorer(_conventions(), (LengthBand("one", 1),))
    long_forms = tuple(f"w{index}" for index in range(12))
    long_gold = AttachmentTree(long_forms, (0, *range(1, 12)))
    long_wrong = AttachmentTree(long_forms, (0, *(1,) * 11))
    scorer.update(long_wrong, long_gold, key=ItemId("long"))
    scorer.update(*_POOL[1][1:], key=ItemId("short"))

    assert all(row.result.n == 0 for row in scorer.rows())
    assert scorer.compute() == MetricResult(
        2 / 14,
        2,
        Resampling("sentence"),
        components={"matched": 2.0, "tokens": 14.0, "parsed": 2.0},
    )


def test_each_sentence_decomposes_to_its_own_directed_accuracy() -> None:
    scorer = _build()

    for key, prediction, gold in _POOL:
        scorer.update(prediction, gold, key=ItemId(key))

    decomposed = scorer.per_item()

    assert set(decomposed) == {ItemId(key) for key, _, _ in _POOL}
    assert decomposed[ItemId("s-a")].value == pytest.approx(5 / 7)
    assert decomposed[ItemId("s-a")].n == 1
    assert decomposed[ItemId("s-b")].value == 0.0
    assert decomposed[ItemId("s-c")] == MetricResult(None, 0, Resampling("sentence"))
    assert decomposed[ItemId("s-d")] == MetricResult(None, 0, Resampling("sentence"))
    assert decomposed[ItemId("s-e")].value == 1.0
    assert decomposed[ItemId("s-e")].components["tokens"] == 2.0

    zeroed = AttachmentScorer(_conventions(coverage="zero"))
    zeroed.update(*_POOL[3][1:], key=ItemId("s-d"))

    assert zeroed.per_item()[ItemId("s-d")].value == 0.0
    assert zeroed.per_item()[ItemId("s-d")].n == 1
    assert zeroed.compute().value == 0.0

    with pytest.raises(TypeError, match="does not support item assignment"):
        cast("dict[ItemId, MetricResult]", zeroed.per_item())[ItemId("x")] = (
            MetricResult(None, 0, Resampling("sentence"))
        )


# ---------------------------------------------------------------------------
# rows and bands as values
# ---------------------------------------------------------------------------


def test_a_row_serializes_its_band_conventions_and_components() -> None:
    scorer = AttachmentScorer(_conventions(aggregation="sentence"))
    tree = AttachmentTree(("a", "b"), (0, 1))
    scorer.update(tree, tree, key=ItemId("1"))
    row = _row(scorer, "wsj20", "undirected")

    assert row.to_spec() == {
        "name": "attachment:wsj20:undirected",
        "band": "wsj20",
        "length_limit": 20,
        "direction": "undirected",
        "value": 1.0,
        "sentences": 1,
        "aggregation": "sentence",
        "coverage": "ignore",
        "punctuation": "keep",
        "minimum_length": 0,
        "matched": 2.0,
        "tokens": 2.0,
        "parsed": 1.0,
    }
    assert row == AttachmentScoreRow(
        LengthBand("wsj20", 20), "undirected", row.conventions, row.result
    )
    assert hash(row) == hash(
        AttachmentScoreRow(
            LengthBand("wsj20", 20), "undirected", row.conventions, row.result
        )
    )
    assert row != _row(scorer, "wsj20", "directed")
    assert row.__eq__(object()) is NotImplemented
    assert repr(row).startswith("AttachmentScoreRow(LengthBand('wsj20', 20)")


def test_a_row_validates_its_fields() -> None:
    result = MetricResult(None, 0, Resampling("sentence"))

    band = LengthBand("b", 1)

    with pytest.raises(TypeError, match="band is not one"):
        AttachmentScoreRow(
            cast("LengthBand", "wsj10"), "directed", _conventions(), result
        )

    with pytest.raises(ValueError, match="unknown attachment direction"):
        AttachmentScoreRow(
            band, cast("AttachmentDirection", "labeled"), _conventions(), result
        )

    with pytest.raises(TypeError, match="not a vector"):
        AttachmentScoreRow(band, "directed", cast("ConventionVector", "keep"), result)

    with pytest.raises(TypeError, match="result is not one"):
        AttachmentScoreRow(band, "directed", _conventions(), cast("MetricResult", 1.0))


def test_a_band_is_a_frozen_value() -> None:
    band = LengthBand("wsj10", 10)

    assert band == LengthBand("wsj10", 10)
    assert band != LengthBand("wsj10", 11)
    assert band.__eq__(object()) is NotImplemented
    assert hash(band) == hash(LengthBand("wsj10", 10))
    assert loads(dumps(band)) == band
    assert copy(band) is band
    assert repr(band) == "LengthBand('wsj10', 10)"
    assert WSJ_LENGTH_BANDS[2].limit is None

    with pytest.raises(ValueError, match="empty"):
        LengthBand("", 10)

    with pytest.raises(ValueError, match="negative"):
        LengthBand("neg", -1)

    field = "_limit"

    with pytest.raises(AttributeError, match="immutable"):
        setattr(band, field, 3)

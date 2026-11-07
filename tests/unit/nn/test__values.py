import math
from typing import TYPE_CHECKING, cast

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from lofi.data import Alignment
from lofi.judge import Action, ActionSet
from lofi.nn import _values
from lofi.nn._values import (
    ChartActionScores,
    ChartReadouts,
    DerivationSetScore,
    Encoded,
    LogDistribution,
    NormalFormDerivationScores,
    TokenEncoding,
    TruncationReport,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

TOKEN_AXES = ("item", "row", "width")
CHART_AXES = ("item", "span", "split")
ACTION_AXES = (*CHART_AXES, "action")

IDENTITY = Action("identity")
APPLY = Action("apply", "application")
ACTIONS = ActionSet((IDENTITY, APPLY), IDENTITY)


def test_derivation_scores_are_frozen_validated_values() -> None:
    constrained = DerivationSetScore("eisner", -2.0, -1.0)
    unrestricted = DerivationSetScore("none", -2.0, 0.0)
    pair = NormalFormDerivationScores(constrained, unrestricted)

    assert pair.constrained is constrained
    assert pair.unconstrained is unrestricted
    assert pair == NormalFormDerivationScores(constrained, unrestricted)
    assert hash(pair) == hash(NormalFormDerivationScores(constrained, unrestricted))
    assert repr(pair).startswith("NormalFormDerivationScores(")

    with pytest.raises(AttributeError, match="immutable"):
        pair.__setattr__("constrained", unrestricted)


def test_derivation_scores_refuse_invalid_names_and_numbers() -> None:
    with pytest.raises(ValueError, match="normal form"):
        DerivationSetScore("", 0.0, 0.0)

    with pytest.raises(ValueError, match="finite"):
        DerivationSetScore("eisner", math.inf, 0.0)


def row_mask(lengths: tuple[int, ...], rows: int) -> Tensor:
    counts = torch.tensor(lengths).unsqueeze(1)

    return torch.arange(rows).unsqueeze(0) < counts


def token_encoding(
    lengths: tuple[int, ...], rows: int, width: int = 2
) -> tuple[Encoded, Tensor]:
    mask = row_mask(lengths, rows)
    vectors = torch.arange(len(lengths) * rows * width, dtype=torch.float32)

    return Encoded(vectors.reshape(len(lengths), rows, width), TOKEN_AXES, mask), mask


def split_posterior(items: int, spans: int, splits: int, live: int) -> LogDistribution:
    mask = torch.zeros(items, spans, splits, dtype=torch.bool)
    mask[:, :, :live] = True
    scores = torch.randn(items, spans, splits).masked_fill(~mask, -math.inf)

    return LogDistribution(
        scores.log_softmax(dim=2), CHART_AXES, mask, normalized_over="split"
    )


def chart_action_scores(
    actions: ActionSet = ACTIONS,
) -> tuple[ChartActionScores, Tensor]:
    mask = torch.tensor([[[False, False], [True, False], [True, True]]])
    source = -torch.arange(
        1,
        1 + mask.numel() * len(actions.actions),
        dtype=torch.float64,
    ).reshape(*mask.shape, len(actions.actions))
    source.requires_grad_()
    tensor = source.masked_fill(~mask.unsqueeze(-1), -math.inf)
    scores = LogDistribution(tensor, ACTION_AXES, mask)

    return ChartActionScores(actions, scores), source


def test_chart_action_scores_pair_every_column_with_its_exact_action() -> None:
    actions, source = chart_action_scores()

    assert actions.action_set is ACTIONS
    assert actions.scores.axes == ACTION_AXES
    assert actions.scores.mask is not None
    assert tuple(actions.by_action) == ACTIONS.actions
    assert all(
        held is expected
        for held, expected in zip(actions.by_action, ACTIONS.actions, strict=True)
    )

    for column, action in enumerate(ACTIONS.actions):
        projected = actions.by_action[action]

        assert projected.axes == CHART_AXES
        assert projected.mask is actions.scores.mask
        assert torch.equal(projected.tensor, actions.scores.tensor[..., column])

    assert actions.identity is actions.by_action[IDENTITY]

    actions.by_action[APPLY].tensor[actions.scores.mask].sum().backward()
    expected = torch.zeros_like(source)
    expected[..., ACTIONS.index(APPLY.name)][actions.scores.mask] = 1.0

    assert source.grad is not None
    assert torch.equal(source.grad, expected)


def test_chart_action_scores_are_frozen() -> None:
    actions, _ = chart_action_scores()

    with pytest.raises(AttributeError, match="immutable"):
        actions.__setattr__("action_set", ACTIONS)

    with pytest.raises(AttributeError, match="immutable"):
        actions.__delattr__("scores")

    with pytest.raises(TypeError):
        cast("dict[Action, LogDistribution]", actions.by_action)[IDENTITY] = (
            actions.identity
        )


def test_chart_action_scores_require_exact_axes_mask_and_inventory_extent() -> None:
    mask = torch.ones(1, 2, 3, dtype=torch.bool)

    with pytest.raises(ValueError, match=r"item.*span.*split.*action"):
        ChartActionScores(
            ACTIONS,
            LogDistribution(
                torch.full((1, 2, 3, 2), -1.0), (*CHART_AXES, "rule"), mask
            ),
        )

    with pytest.raises(ValueError, match="valid-split mask"):
        ChartActionScores(
            ACTIONS,
            LogDistribution(torch.full((1, 2, 3, 2), -1.0), ACTION_AXES),
        )

    with pytest.raises(ValueError, match="valid-split mask"):
        ChartActionScores(
            ACTIONS,
            LogDistribution(
                torch.full((1, 2, 3, 2), -1.0),
                ACTION_AXES,
                torch.ones(1, 2, dtype=torch.bool),
            ),
        )

    normalized = torch.full((1, 2, 3, 2), -math.log(2.0))

    with pytest.raises(ValueError, match="unnormalized parser scores"):
        ChartActionScores(
            ACTIONS,
            LogDistribution(
                normalized,
                ACTION_AXES,
                mask,
                normalized_over="action",
            ),
        )

    with pytest.raises(ValueError, match=r"extent 3.*holds 2 actions"):
        ChartActionScores(
            ACTIONS,
            LogDistribution(torch.full((1, 2, 3, 3), -1.0), ACTION_AXES, mask),
        )


def test_permuting_the_inventory_and_columns_preserves_action_association() -> None:
    original, _ = chart_action_scores()
    permuted_set = ActionSet((APPLY, IDENTITY), IDENTITY)
    permutation = [ACTIONS.index(action.name) for action in permuted_set.actions]
    permuted = ChartActionScores(
        permuted_set,
        LogDistribution(
            original.scores.tensor[..., permutation],
            ACTION_AXES,
            original.scores.mask,
        ),
    )

    for action in ACTIONS.actions:
        assert torch.equal(
            original.by_action[action].tensor,
            permuted.by_action[action].tensor,
        )


def test_an_encoding_reads_its_axes_by_name() -> None:
    encoded = Encoded(torch.zeros(2, 3, 4), TOKEN_AXES)

    assert encoded.axes == TOKEN_AXES
    assert encoded.axis("row") == 1
    assert encoded.size("width") == 4
    assert encoded.mask is None
    assert encoded.mask_axes == ()


def test_an_encoding_refuses_an_axis_it_does_not_name() -> None:
    encoded = Encoded(torch.zeros(2, 3), ("item", "row"))

    with pytest.raises(ValueError, match="no axis named 'width'"):
        encoded.axis("width")

    with pytest.raises(ValueError, match="no axis named 'width'"):
        encoded.size("width")


def test_an_encoding_names_every_axis_exactly_once() -> None:
    with pytest.raises(ValueError, match="needs 3 axis names"):
        Encoded(torch.zeros(2, 3, 4), ("item", "row"))

    with pytest.raises(ValueError, match="axis name is empty"):
        Encoded(torch.zeros(2, 3), ("item", ""))

    with pytest.raises(ValueError, match="used twice"):
        Encoded(torch.zeros(2, 3), ("item", "item"))


def test_a_mask_covers_the_outermost_axes() -> None:
    mask = torch.ones(2, 3, dtype=torch.bool)
    encoded = Encoded(torch.zeros(2, 3, 4), TOKEN_AXES, mask)

    assert encoded.mask is mask
    assert encoded.mask_axes == ("item", "row")


def test_a_mask_that_does_not_cover_the_outermost_axes_is_refused() -> None:
    with pytest.raises(TypeError, match="holds booleans"):
        Encoded(torch.zeros(2, 3), ("item", "row"), torch.ones(2, 3))

    with pytest.raises(ValueError, match="rank is between 1 and 2"):
        Encoded(
            torch.zeros(2, 3), ("item", "row"), torch.ones(2, 3, 4, dtype=torch.bool)
        )

    with pytest.raises(ValueError, match="does not cover the outermost"):
        Encoded(torch.zeros(2, 3), ("item", "row"), torch.ones(3, dtype=torch.bool))


def test_an_encoding_cannot_be_rebound_or_emptied() -> None:
    encoded = Encoded(torch.zeros(2, 3), ("item", "row"))

    for field in ("tensor", "axes", "mask"):
        with pytest.raises(AttributeError):
            setattr(encoded, field, None)

        with pytest.raises(AttributeError):
            delattr(encoded, field)

    assert not hasattr(encoded, "__dict__")
    assert encoded.tensor.shape == (2, 3)


def test_a_log_space_value_is_floating_point() -> None:
    with pytest.raises(TypeError, match="floating point"):
        LogDistribution(torch.zeros(2, 3, dtype=torch.int64), ("item", "span"))


def test_a_normalized_axis_is_one_of_the_declared_axes() -> None:
    with pytest.raises(ValueError, match="no axis named 'k'"):
        LogDistribution(torch.zeros(2, 3), ("item", "span"), normalized_over="k")


def test_an_unnormalized_log_score_carries_no_normalized_axis() -> None:
    inside = LogDistribution(torch.full((2, 3), -1.5), ("item", "span"))

    assert inside.normalized_over is None
    assert inside.size("span") == 3


def test_a_split_posterior_normalizes_over_its_split_axis() -> None:
    alpha = split_posterior(items=2, spans=5, splits=4, live=3)

    assert alpha.normalized_over == "split"
    assert alpha.mask is not None
    assert alpha.mask_axes == CHART_AXES

    live = alpha.tensor[:, :, :3].exp().sum(dim=2)

    assert torch.allclose(live, torch.ones_like(live), atol=1e-5)


def test_the_debug_checks_refuse_a_zero_in_a_padded_log_position(
    reload_with_checks: Callable[..., None],
) -> None:
    reload_with_checks(_values, on=True)
    mask = torch.tensor([[True, False]])

    with pytest.raises(ValueError, match="reads as probability one"):
        _values.LogDistribution(torch.zeros(1, 2), ("item", "span"), mask)

    padded = _values.LogDistribution(
        torch.tensor([[-0.5, -math.inf]]), ("item", "span"), mask
    )

    assert bool(padded.tensor[0, 1].isneginf())


def test_the_debug_checks_refuse_mass_that_does_not_sum_to_one(
    reload_with_checks: Callable[..., None],
) -> None:
    reload_with_checks(_values, on=True)
    mask = torch.ones(1, 2, dtype=torch.bool)

    with pytest.raises(ValueError, match="does not sum to one"):
        _values.LogDistribution(
            torch.full((1, 2), -1.5), ("item", "split"), mask, normalized_over="split"
        )


@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_the_debug_checks_report_mass_that_is_not_a_number_as_numerical(
    reload_with_checks: Callable[..., None], value: float
) -> None:
    reload_with_checks(_values, on=True)
    mask = torch.ones(1, 2, dtype=torch.bool)

    with pytest.raises(_values.NumericalFailureError, match="'split'") as raised:
        _values.LogDistribution(
            torch.tensor([[value, -1.0]]),
            ("item", "split"),
            mask,
            normalized_over="split",
        )

    assert raised.value.quantity == "the log mass over the axis named 'split'"


def test_the_debug_checks_refuse_a_row_with_no_mass_as_unnormalized(
    reload_with_checks: Callable[..., None],
) -> None:
    reload_with_checks(_values, on=True)

    with pytest.raises(ValueError, match="does not sum to one"):
        _values.LogDistribution(
            torch.full((1, 2), -math.inf),
            ("item", "split"),
            torch.ones(1, 2, dtype=torch.bool),
            normalized_over="split",
        )


def test_the_debug_checks_exclude_masked_values_from_normalized_mass(
    reload_with_checks: Callable[..., None],
) -> None:
    reload_with_checks(_values, on=True)
    mask = torch.tensor([[True, True, False]])

    normalized = _values.LogDistribution(
        torch.tensor([[-math.log(2.0), -math.log(2.0), math.nan]]),
        ("item", "split"),
        mask,
        normalized_over="split",
    )

    assert bool(normalized.tensor[0, 2].isnan())


def test_the_debug_checks_leave_a_wholly_padded_row_alone(
    reload_with_checks: Callable[..., None],
) -> None:
    reload_with_checks(_values, on=True)
    mask = torch.tensor([[True, True], [False, False]])
    scores = torch.tensor([[-0.5, -0.5], [-math.inf, -math.inf]])

    normalized = _values.LogDistribution(
        scores.log_softmax(dim=1).masked_fill(~mask, -math.inf),
        ("item", "split"),
        mask,
        normalized_over="split",
    )

    assert bool(normalized.tensor[1].isneginf().all())


def test_without_the_debug_checks_the_log_values_are_taken_unread(
    reload_with_checks: Callable[..., None],
) -> None:
    reload_with_checks(_values, on=False)

    unexamined = _values.LogDistribution(
        torch.zeros(1, 2), ("item", "split"), torch.tensor([[True, False]])
    )

    assert unexamined.tensor[0, 1].item() == 0.0


def test_an_alignment_says_which_rows_encode_each_word() -> None:
    alignment = Alignment(((0, 2), (2, 3), None, (3, 6)))

    assert alignment.word_count == 4
    assert alignment.rows_for(1) == (2, 3)
    assert alignment.rows_for(2) is None
    assert alignment.dropped == (2,)
    assert not alignment.is_total


def test_an_alignment_over_every_word_is_total() -> None:
    alignment = Alignment(((0, 1), (1, 2)))

    assert alignment.is_total
    assert alignment.dropped == ()


def test_an_alignment_lays_its_rows_out_in_word_order() -> None:
    with pytest.raises(ValueError, match="before the rows of the tokens before"):
        Alignment(((2, 4), (1, 2)))

    with pytest.raises(ValueError, match="aligned to no row"):
        Alignment(((0, 1), (1, 1)))


def test_a_negative_word_position_is_refused() -> None:
    with pytest.raises(IndexError, match="none at position -1"):
        Alignment(((0, 1),)).rows_for(-1)


def test_alignments_over_the_same_words_are_one_alignment() -> None:
    alignment = Alignment(((0, 2), None))

    assert alignment == Alignment(((0, 2), None))
    assert hash(alignment) == hash(Alignment(((0, 2), None)))
    assert alignment != Alignment(((0, 2), (2, 3)))
    assert alignment != ((0, 2), None)


def test_a_token_encoding_reads_its_lengths_off_its_mask() -> None:
    vectors, mask = token_encoding((3, 1), rows=3)
    encoding = TokenEncoding(vectors, (Alignment(((0, 3),)), Alignment(((0, 1),))))

    assert encoding.vectors is vectors
    assert encoding.lengths.tolist() == [3, 1]
    assert encoding.lengths.tolist() == mask.sum(dim=1).tolist()
    assert encoding.truncated == ((), ())


def test_a_token_encoding_reports_the_words_no_row_encodes() -> None:
    vectors, _ = token_encoding((2, 2), rows=2)
    encoding = TokenEncoding(
        vectors,
        (Alignment(((0, 1), (1, 2), None)), Alignment(((0, 2),))),
        words=(("the", "quick", "fox"), ("brown",)),
    )

    assert encoding.truncated == ((2,), ())
    assert encoding.alignments[0].dropped == (2,)
    assert encoding.truncation == (
        TruncationReport((2,), ("fox",)),
        TruncationReport((), ()),
    )


def test_a_truncation_report_names_each_dropped_word_at_its_position() -> None:
    report = TruncationReport((3, 4), ("fox", "jumps"))

    assert report.positions == (3, 4)
    assert report.words == ("fox", "jumps")
    assert len(report) == 2
    assert report
    assert not TruncationReport((), ())
    assert len(TruncationReport((), ())) == 0


def test_a_truncation_report_pairs_every_position_with_one_word() -> None:
    with pytest.raises(ValueError, match="one word per position"):
        TruncationReport((3, 4), ("fox",))

    with pytest.raises(ValueError, match="increase"):
        TruncationReport((4, 3), ("fox", "jumps"))

    with pytest.raises(ValueError, match="increase"):
        TruncationReport((2, 2), ("fox", "fox"))

    with pytest.raises(ValueError, match="increase"):
        TruncationReport((-1,), ("fox",))

    with pytest.raises(ValueError, match="contiguous"):
        TruncationReport((2, 4), ("fox", "jumps"))


def test_reports_of_the_same_words_at_the_same_positions_are_one_report() -> None:
    report = TruncationReport((1,), ("fox",))

    assert report == TruncationReport((1,), ("fox",))
    assert hash(report) == hash(TruncationReport((1,), ("fox",)))
    assert report != TruncationReport((2,), ("fox",))
    assert report != TruncationReport((1,), ("dog",))
    assert report != ((1,), ("fox",))
    assert repr(report) == "TruncationReport((1,), ('fox',))"


def test_a_token_encoding_names_the_dropped_words_when_it_records_its_words() -> None:
    vectors, _ = token_encoding((2, 2), rows=2)
    encoding = TokenEncoding(
        vectors,
        (Alignment(((0, 1), (1, 2), None)), Alignment(((0, 2),))),
        words=(["the", "quick", "fox"], ("brown",)),
    )

    assert encoding.words == (("the", "quick", "fox"), ("brown",))
    assert encoding.truncated == ((2,), ())
    assert encoding.truncation == (
        TruncationReport((2,), ("fox",)),
        TruncationReport((), ()),
    )


def test_an_incomplete_token_encoding_requires_its_original_words() -> None:
    vectors, _ = token_encoding((2, 2), rows=2)

    with pytest.raises(ValueError, match="must record its original words"):
        TokenEncoding(
            vectors,
            (Alignment(((0, 1), (1, 2), None)), Alignment(((0, 2),))),
        )


def test_an_incomplete_token_encoding_omits_one_contiguous_suffix() -> None:
    vectors, _ = token_encoding((2, 2), rows=2)

    with pytest.raises(ValueError, match="omit only one contiguous suffix"):
        TokenEncoding(
            vectors,
            (Alignment(((0, 1), None, (1, 2))), Alignment(((0, 2),))),
            words=(("the", "quick", "fox"), ("brown",)),
        )


def test_a_complete_token_encoding_need_not_retain_its_original_words() -> None:
    vectors, _ = token_encoding((2, 2), rows=2)
    encoding = TokenEncoding(
        vectors,
        (Alignment(((0, 1), (1, 2))), Alignment(((0, 2),))),
    )

    assert encoding.words is None
    assert encoding.truncation is None
    assert encoding.truncated == ((), ())


def test_a_token_encodings_words_number_its_alignments_entries() -> None:
    vectors, _ = token_encoding((2, 2), rows=2)
    alignments = (Alignment(((0, 1), (1, 2))), Alignment(((0, 2),)))

    with pytest.raises(ValueError, match="holds 3 words and its alignment holds 2"):
        TokenEncoding(vectors, alignments, words=(("a", "b", "c"), ("d",)))

    with pytest.raises(ValueError, match="words of each of its 2 items"):
        TokenEncoding(vectors, alignments, words=(("a", "b"),))


def test_a_token_encoding_declares_the_three_axes_it_is_over() -> None:
    mask = torch.ones(2, 3, dtype=torch.bool)
    wrong = Encoded(torch.zeros(2, 3, 4), ("item", "piece", "width"), mask)

    with pytest.raises(ValueError, match="over the axes"):
        TokenEncoding(wrong, (Alignment(((0, 3),)), Alignment(((0, 3),))))


def test_a_token_encoding_requires_a_mask_over_its_items_and_rows() -> None:
    unmasked = Encoded(torch.zeros(1, 3, 4), TOKEN_AXES)

    with pytest.raises(ValueError, match="requires a mask"):
        TokenEncoding(unmasked, (Alignment(((0, 3),)),))

    deep = Encoded(
        torch.zeros(1, 3, 4), TOKEN_AXES, torch.ones(1, 3, 4, dtype=torch.bool)
    )

    with pytest.raises(ValueError, match="requires a mask"):
        TokenEncoding(deep, (Alignment(((0, 3),)),))


def test_a_token_encoding_carries_one_alignment_per_item() -> None:
    vectors, _ = token_encoding((2, 2), rows=2)

    with pytest.raises(ValueError, match="one alignment per item"):
        TokenEncoding(vectors, (Alignment(((0, 2),)),))


def test_a_token_encoding_refuses_an_alignment_beyond_its_row_axis() -> None:
    vectors, _ = token_encoding((2,), rows=2)

    with pytest.raises(ValueError, match="beyond the row extent 2"):
        TokenEncoding(vectors, (Alignment(((0, 3),)),))


def test_a_token_encoding_refuses_an_alignment_to_a_masked_row() -> None:
    vectors, _ = token_encoding((1,), rows=2)

    with pytest.raises(ValueError, match="includes a masked row"):
        TokenEncoding(vectors, (Alignment(((0, 2),)),))


def test_readouts_hand_back_what_the_chart_published() -> None:
    inside = LogDistribution(torch.full((2, 5), -1.0), ("item", "span"))
    alpha = split_posterior(items=2, spans=5, splits=4, live=2)
    mu = Encoded(torch.full((2, 5), 0.25), ("item", "span"))
    cells = Encoded(torch.zeros(2, 5, 3), ("item", "span", "width"))
    outside_cells = Encoded(torch.ones(2, 5, 3), ("item", "span", "width"))
    spans = Encoded(torch.ones(2, 5, 6), ("item", "span", "span_width"))
    entropy = Encoded(torch.ones(2), ("item",))
    mbr = Encoded(torch.zeros(2, 5), ("item", "span"))

    readouts = ChartReadouts(
        inside=inside,
        alpha=alpha,
        mu=mu,
        cells=cells,
        outside_cells=outside_cells,
        spans=spans,
        entropy=entropy,
        mbr=mbr,
    )

    assert readouts.inside is inside
    assert readouts.alpha is alpha
    assert readouts.mu is mu
    assert readouts.outside is None
    assert readouts.log_z is None
    assert readouts.actions is None
    assert readouts.identity is None
    assert readouts.cells is cells
    assert readouts.outside_cells is outside_cells
    assert readouts.spans is spans
    assert readouts.entropy is entropy
    assert readouts.mbr is mbr


def test_readouts_that_disagree_about_a_shared_axis_are_refused() -> None:
    inside = LogDistribution(torch.full((2, 5), -1.0), ("item", "span"))
    outside = LogDistribution(torch.full((3, 5), -1.0), ("item", "span"))

    with pytest.raises(ValueError, match="disagree about the axis named 'item'"):
        ChartReadouts(inside=inside, outside=outside)


def test_a_chart_may_publish_nothing() -> None:
    empty = ChartReadouts()

    assert empty.inside is None
    assert "ChartReadouts" in repr(empty)


@settings(max_examples=50)
@given(
    shape=st.lists(st.integers(min_value=1, max_value=4), min_size=1, max_size=4),
    depth=st.integers(min_value=1, max_value=4),
)
def test_an_encodings_mask_and_axes_agree_in_rank(shape: list[int], depth: int) -> None:
    covered = min(depth, len(shape))
    axes = tuple(f"axis{position}" for position in range(len(shape)))
    mask = torch.ones(tuple(shape[:covered]), dtype=torch.bool)
    encoded = Encoded(torch.zeros(tuple(shape)), axes, mask)

    assert len(encoded.axes) == encoded.tensor.ndim
    assert encoded.mask is not None
    assert encoded.mask_axes == axes[:covered]
    assert [encoded.size(axis) for axis in axes] == shape


@settings(max_examples=50)
@given(
    spans=st.integers(min_value=1, max_value=5),
    splits=st.integers(min_value=1, max_value=5),
    live=st.integers(min_value=1, max_value=5),
)
def test_a_split_posteriors_mask_and_axes_agree_in_rank(
    spans: int, splits: int, live: int
) -> None:
    alpha = split_posterior(items=2, spans=spans, splits=splits, live=min(live, splits))

    assert len(alpha.axes) == alpha.tensor.ndim == 3
    assert alpha.mask is not None
    assert alpha.mask_axes == alpha.axes
    assert alpha.size("split") == splits
    assert not bool(alpha.tensor.eq(0.0).logical_and(~alpha.mask).any())


@settings(max_examples=25)
@given(
    lengths=st.lists(st.integers(min_value=1, max_value=6), min_size=1, max_size=4),
    slack=st.integers(min_value=0, max_value=3),
)
def test_a_token_encodings_lengths_and_alignments_follow_its_mask(
    lengths: list[int], slack: int
) -> None:
    rows = max(lengths) + slack
    vectors, mask = token_encoding(tuple(lengths), rows)
    alignments = tuple(
        Alignment(tuple((row, row + 1) for row in range(length))) for length in lengths
    )
    encoding = TokenEncoding(vectors, alignments)

    assert len(encoding.vectors.axes) == encoding.vectors.tensor.ndim
    assert encoding.vectors.mask_axes == TOKEN_AXES[:2]
    assert encoding.lengths.tolist() == lengths
    assert encoding.lengths.tolist() == mask.sum(dim=1).tolist()
    assert encoding.truncated == ((),) * len(lengths)


@settings(max_examples=25)
@given(
    items=st.integers(min_value=1, max_value=3),
    spans=st.integers(min_value=1, max_value=5),
    splits=st.integers(min_value=1, max_value=4),
)
def test_the_published_readouts_agree_on_every_axis_they_share(
    items: int, spans: int, splits: int
) -> None:
    readouts = ChartReadouts(
        inside=LogDistribution(torch.full((items, spans), -1.0), ("item", "span")),
        alpha=split_posterior(items, spans, splits, live=splits),
        mu=Encoded(torch.full((items, spans), 0.5), ("item", "span")),
        log_z=LogDistribution(torch.full((items,), -2.0), ("item",)),
    )
    published = (readouts.inside, readouts.alpha, readouts.mu, readouts.log_z)

    for readout in published:
        assert readout is not None
        assert len(readout.axes) == readout.tensor.ndim
        assert readout.mask_axes == readout.axes[: len(readout.mask_axes)]
        assert readout.size("item") == items

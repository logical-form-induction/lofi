from typing import Final

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from lofi.eval import (
    AccuracyCurve,
    BeamNormalization,
    ScoredBeam,
    SweepPoint,
    accuracy_at_beam_size,
    beam_size_sweep,
    depth_truncated_accuracy,
    mean_reciprocal_rank,
    typeless_accuracy,
)
from lofi.terms import (
    DeclaredConstructors,
    NeighborhoodError,
    Operator,
    Signature,
    Term,
    TermStore,
)

SIGNATURE: Final = Signature(
    "beamed-categories",
    ("Cat",),
    (
        Operator("n", "Cat"),
        Operator("np", "Cat"),
        Operator("s", "Cat"),
        Operator("fwd", "Cat", arg_sorts=("Cat", "Cat"), family="slash"),
        Operator("bwd", "Cat", arg_sorts=("Cat", "Cat"), family="slash"),
        Operator("wrap", "Cat", arg_sorts=("Cat",), family="wrapper"),
    ),
)
STORE: Final = TermStore(SIGNATURE)
OTHER: Final = TermStore(SIGNATURE)
N: Final = STORE.op(SIGNATURE.by_name("n"))
NP: Final = STORE.op(SIGNATURE.by_name("np"))
S: Final = STORE.op(SIGNATURE.by_name("s"))
NORMALIZATIONS: Final = tuple(BeamNormalization)


VIEW: Final = DeclaredConstructors(
    STORE,
    {"fwd": ("result", "argument"), "bwd": ("result", "argument")},
)
INTRANSITIVE: Final = VIEW.make("bwd", (S, NP))
TRANSITIVE: Final = VIEW.make("fwd", (INTRANSITIVE, NP))
DITRANSITIVE: Final = VIEW.make("fwd", (TRANSITIVE, NP))
MODIFIER: Final = VIEW.make("fwd", (N, N))
BUSHY: Final = VIEW.make("fwd", (INTRANSITIVE, INTRANSITIVE))
LOPSIDED: Final = VIEW.make("fwd", (TRANSITIVE, INTRANSITIVE))
WRAPPED: Final = STORE.op(SIGNATURE.by_name("wrap"), NP)

#: A beam whose first entry differs under every normalization.
CONTESTED: Final = ((NP, -2.0), (TRANSITIVE, -3.0), (BUSHY, -3.5), (LOPSIDED, -4.4))
TOP: Final = {
    BeamNormalization.NONE: NP,
    BeamNormalization.DEPTH: TRANSITIVE,
    BeamNormalization.NODES: LOPSIDED,
    BeamNormalization.TERMINALS: BUSHY,
}


def scored(*terms: Term) -> ScoredBeam:
    """Return a beam over the terms in the order given, at one log weight."""
    return tuple((term, 0.0) for term in terms)


def primitives() -> st.SearchStrategy[Term]:
    return st.sampled_from((N, NP, S))


def categories(depth: int = 4) -> st.SearchStrategy[Term]:
    """Return a strategy over functor terms of at most the given depth."""
    if depth <= 1:
        return primitives()

    return st.one_of(
        primitives(),
        st.builds(
            lambda constructor, result, argument: VIEW.make(
                constructor, (result, argument)
            ),
            st.sampled_from(("fwd", "bwd")),
            categories(depth - 1),
            categories(depth - 1),
        ),
    )


def weights() -> st.SearchStrategy[float]:
    return st.floats(min_value=-40.0, max_value=0.0)


def beams(max_size: int = 4) -> st.SearchStrategy[ScoredBeam]:
    return st.lists(st.tuples(categories(), weights()), max_size=max_size)


def items(
    max_size: int = 6, *, min_size: int = 0, beam_size: int = 4
) -> st.SearchStrategy[list[tuple[ScoredBeam, Term]]]:
    return st.lists(
        st.tuples(beams(beam_size), categories()), min_size=min_size, max_size=max_size
    )


@pytest.mark.parametrize("normalization", NORMALIZATIONS)
def test_each_normalization_ranks_a_different_entry_first(
    normalization: BeamNormalization,
) -> None:
    ranked = normalization.ranked(CONTESTED)

    assert ranked[0] is TOP[normalization]
    assert set(ranked) == {term for term, _ in CONTESTED}


def test_the_divisors_are_one_depth_nodes_and_terminals() -> None:
    assert [n.divisor(NP) for n in NORMALIZATIONS] == [1, 1, 1, 1]
    assert [n.divisor(TRANSITIVE) for n in NORMALIZATIONS] == [1, 3, 5, 3]
    assert [n.divisor(BUSHY) for n in NORMALIZATIONS] == [1, 3, 7, 4]
    assert [n.divisor(LOPSIDED) for n in NORMALIZATIONS] == [1, 4, 9, 5]


def test_the_four_normalizations_rank_the_contested_beam_four_ways() -> None:
    # log weights: NP -2.0, TRANSITIVE -3.0, BUSHY -3.5, LOPSIDED -4.4
    # depth divides them to -2.0, -1.0, -3.5 / 3, -1.1
    # nodes divides them to -2.0, -0.6, -0.5, -4.4 / 9
    # terminals divides them to -2.0, -1.0, -0.875, -0.88
    rankings = {n: n.ranked(CONTESTED) for n in NORMALIZATIONS}

    assert rankings == {
        BeamNormalization.NONE: (NP, TRANSITIVE, BUSHY, LOPSIDED),
        BeamNormalization.DEPTH: (TRANSITIVE, LOPSIDED, BUSHY, NP),
        BeamNormalization.NODES: (LOPSIDED, BUSHY, TRANSITIVE, NP),
        BeamNormalization.TERMINALS: (BUSHY, LOPSIDED, TRANSITIVE, NP),
    }
    assert len(set(rankings.values())) == len(NORMALIZATIONS)


@pytest.mark.parametrize("normalization", NORMALIZATIONS)
def test_entries_at_one_normalized_weight_keep_their_order(
    normalization: BeamNormalization,
) -> None:
    beam = scored(TRANSITIVE, NP, BUSHY, MODIFIER)

    assert normalization.ranked(beam) == (TRANSITIVE, NP, BUSHY, MODIFIER)


def test_ranking_under_none_follows_the_raw_log_weight() -> None:
    beam = ((NP, -3.0), (TRANSITIVE, -1.0), (BUSHY, -2.0))

    assert BeamNormalization.NONE.ranked(beam) == (TRANSITIVE, BUSHY, NP)


@pytest.mark.parametrize("weight", [float("inf"), float("-inf"), float("nan")])
def test_a_log_weight_that_is_not_finite_is_refused(weight: float) -> None:
    with pytest.raises(ValueError, match="not finite"):
        accuracy_at_beam_size(
            (((NP, weight),),),
            (NP,),
            normalization=BeamNormalization.NONE,
            max_width=1,
        )


@pytest.mark.parametrize("normalization", NORMALIZATIONS)
def test_accuracy_at_beam_size_steps_up_where_the_ranking_first_admits_gold(
    normalization: BeamNormalization,
) -> None:
    ranked = normalization.ranked(CONTESTED)
    gold = TOP[normalization]

    at_top = accuracy_at_beam_size(
        (CONTESTED,), (gold,), normalization=normalization, max_width=4
    )
    at_last = accuracy_at_beam_size(
        (CONTESTED,), (ranked[3],), normalization=normalization, max_width=4
    )

    assert at_top.hits == (1, 1, 1, 1)
    assert at_last.hits == (0, 0, 0, 1)


def test_accuracy_at_beam_size_steps_up_exactly_where_gold_first_appears() -> None:
    beam = scored(INTRANSITIVE, MODIFIER, TRANSITIVE, NP, TRANSITIVE)

    curve = accuracy_at_beam_size(
        (beam,), (TRANSITIVE,), normalization=BeamNormalization.NONE, max_width=5
    )

    assert curve.items == 1
    assert curve.hits == (0, 0, 1, 1, 1)
    assert curve.rates == (0.0, 0.0, 1.0, 1.0, 1.0)
    assert curve.at(2) == 0.0
    assert curve.at(3) == 1.0


def test_accuracy_at_beam_size_averages_over_items() -> None:
    beams = (
        scored(TRANSITIVE, NP),
        scored(NP, TRANSITIVE),
        scored(INTRANSITIVE, MODIFIER),
        (),
    )
    gold = (TRANSITIVE, TRANSITIVE, TRANSITIVE, TRANSITIVE)

    curve = accuracy_at_beam_size(
        beams, gold, normalization=BeamNormalization.NONE, max_width=3
    )

    assert curve.items == 4
    assert curve.hits == (1, 2, 2)
    assert curve.rates == (0.25, 0.5, 0.5)


def test_accuracy_at_beam_size_ignores_entries_beyond_the_width() -> None:
    beam = scored(NP, N, S, TRANSITIVE)

    curve = accuracy_at_beam_size(
        (beam,), (TRANSITIVE,), normalization=BeamNormalization.NONE, max_width=3
    )

    assert curve.hits == (0, 0, 0)


def test_types_are_compared_by_structure_rather_than_by_store() -> None:
    foreign = OTHER.op(
        SIGNATURE.by_name("fwd"),
        OTHER.op(
            SIGNATURE.by_name("bwd"),
            OTHER.op(SIGNATURE.by_name("s")),
            OTHER.op(SIGNATURE.by_name("np")),
        ),
        OTHER.op(SIGNATURE.by_name("np")),
    )

    assert foreign is not TRANSITIVE
    assert accuracy_at_beam_size(
        (scored(foreign),),
        (TRANSITIVE,),
        normalization=BeamNormalization.NONE,
        max_width=1,
    ).hits == (1,)


@settings(max_examples=60, deadline=None)
@given(items(), st.sampled_from(NORMALIZATIONS), st.integers(1, 5))
def test_accuracy_at_beam_size_is_non_decreasing_in_the_width(
    scored_items: list[tuple[ScoredBeam, Term]],
    normalization: BeamNormalization,
    max_width: int,
) -> None:
    beams = tuple(beam for beam, _ in scored_items)
    gold = tuple(target for _, target in scored_items)

    curve = accuracy_at_beam_size(
        beams, gold, normalization=normalization, max_width=max_width
    )

    assert len(curve.hits) == max_width
    assert curve.items == len(scored_items)
    assert all(
        earlier <= later
        for earlier, later in zip(curve.hits, curve.hits[1:], strict=False)
    )
    assert curve.hits[-1] == sum(
        any(
            candidate.key() == target.key()
            for candidate in normalization.ranked(beam)[:max_width]
        )
        for beam, target in scored_items
    )


@pytest.mark.parametrize("normalization", NORMALIZATIONS)
def test_depth_truncated_accuracy_scores_the_entry_the_normalization_ranks_first(
    normalization: BeamNormalization,
) -> None:
    curve = depth_truncated_accuracy(
        (CONTESTED,),
        (TOP[normalization],),
        N,
        VIEW,
        normalization=normalization,
        max_depth=4,
    )

    assert curve.hits == (1, 1, 1, 1)


def test_depth_truncated_accuracy_reads_how_deep_the_best_entry_is_right() -> None:
    wrong_argument = VIEW.make("fwd", (INTRANSITIVE, N))
    wrong_root = VIEW.make("bwd", (INTRANSITIVE, NP))
    wrong_inside = VIEW.make("fwd", (VIEW.make("bwd", (S, N)), NP))
    beams = (
        scored(TRANSITIVE, NP),
        scored(wrong_argument, TRANSITIVE),
        scored(wrong_root),
        scored(wrong_inside),
        (),
    )
    gold = (TRANSITIVE,) * 5

    curve = depth_truncated_accuracy(
        beams, gold, N, VIEW, normalization=BeamNormalization.NONE, max_depth=4
    )

    assert curve.items == 5
    assert curve.hits == (4, 2, 1, 1)
    assert curve.rates == (0.8, 0.4, 0.2, 0.2)


def test_depth_one_keeps_a_primitive_and_collapses_every_functor() -> None:
    curve = depth_truncated_accuracy(
        (
            scored(NP),
            scored(S),
            scored(TRANSITIVE),
            scored(MODIFIER),
            scored(INTRANSITIVE),
        ),
        (NP, NP, NP, TRANSITIVE, TRANSITIVE),
        N,
        VIEW,
        normalization=BeamNormalization.NONE,
        max_depth=2,
    )

    assert curve.hits == (3, 1)


def test_a_beam_scored_at_depth_beyond_both_types_stays_exact() -> None:
    curve = depth_truncated_accuracy(
        (scored(DITRANSITIVE), scored(TRANSITIVE)),
        (DITRANSITIVE, DITRANSITIVE),
        N,
        VIEW,
        normalization=BeamNormalization.NONE,
        max_depth=6,
    )

    assert curve.hits == (2, 2, 1, 1, 1, 1)


@settings(max_examples=60, deadline=None)
@given(items(beam_size=3), st.sampled_from(NORMALIZATIONS), st.integers(1, 5))
def test_depth_truncated_accuracy_is_non_increasing_in_the_depth(
    scored_items: list[tuple[ScoredBeam, Term]],
    normalization: BeamNormalization,
    max_depth: int,
) -> None:
    beams = tuple(beam for beam, _ in scored_items)
    gold = tuple(target for _, target in scored_items)

    curve = depth_truncated_accuracy(
        beams, gold, N, VIEW, normalization=normalization, max_depth=max_depth
    )

    assert len(curve.hits) == max_depth
    assert curve.items == len(scored_items)
    assert all(
        earlier >= later
        for earlier, later in zip(curve.hits, curve.hits[1:], strict=False)
    )
    assert curve.hits[-1] >= sum(
        bool(beam) and normalization.ranked(beam)[0].key() == target.key()
        for beam, target in scored_items
    )


def test_depth_truncated_accuracy_at_full_depth_is_exact_accuracy_at_width_one() -> (
    None
):
    beams = (
        scored(TRANSITIVE, NP),
        scored(INTRANSITIVE, TRANSITIVE),
        scored(DITRANSITIVE),
        (),
    )
    gold = (TRANSITIVE, TRANSITIVE, DITRANSITIVE, NP)

    by_depth = depth_truncated_accuracy(
        beams, gold, N, VIEW, normalization=BeamNormalization.NONE, max_depth=5
    )
    by_width = accuracy_at_beam_size(
        beams, gold, normalization=BeamNormalization.NONE, max_width=1
    )

    assert by_depth.hits[-1] == by_width.hits[0] == 2


def test_typeless_accuracy_counts_a_right_shape_under_wrong_labels() -> None:
    relabeled = VIEW.make("fwd", (VIEW.make("bwd", (N, S)), N))
    reshaped = VIEW.make("fwd", (NP, INTRANSITIVE))
    reslashed = VIEW.make("bwd", (INTRANSITIVE, NP))
    beams = (
        scored(relabeled, TRANSITIVE),
        scored(reshaped, relabeled),
        scored(reslashed),
        scored(TRANSITIVE),
        (),
    )
    gold = (TRANSITIVE,) * 5

    shape = typeless_accuracy(
        beams, gold, N, VIEW, normalization=BeamNormalization.NONE, max_width=2
    )
    exact = accuracy_at_beam_size(
        beams, gold, normalization=BeamNormalization.NONE, max_width=2
    )

    assert shape.items == 5
    assert shape.hits == (2, 3)
    assert exact.hits == (1, 2)


@pytest.mark.parametrize("normalization", NORMALIZATIONS)
def test_typeless_accuracy_scores_the_entry_the_normalization_ranks_first(
    normalization: BeamNormalization,
) -> None:
    curve = typeless_accuracy(
        (CONTESTED,),
        (TOP[normalization],),
        N,
        VIEW,
        normalization=normalization,
        max_width=1,
    )

    assert curve.hits == (1,)


def test_typeless_accuracy_takes_its_representative_from_the_caller() -> None:
    beams = (scored(MODIFIER),)

    under_n = typeless_accuracy(
        beams,
        (VIEW.make("fwd", (S, S)),),
        N,
        VIEW,
        normalization=BeamNormalization.NONE,
        max_width=1,
    )
    under_np = typeless_accuracy(
        beams,
        (VIEW.make("fwd", (S, S)),),
        NP,
        VIEW,
        normalization=BeamNormalization.NONE,
        max_width=1,
    )

    assert under_n.hits == under_np.hits == (1,)


@settings(max_examples=20, deadline=None)
@given(items(60, min_size=50), st.sampled_from(NORMALIZATIONS), st.integers(1, 4))
def test_typeless_accuracy_dominates_exact_accuracy_at_every_width(
    scored_items: list[tuple[ScoredBeam, Term]],
    normalization: BeamNormalization,
    max_width: int,
) -> None:
    beams = tuple(beam for beam, _ in scored_items)
    gold = tuple(target for _, target in scored_items)

    shape = typeless_accuracy(
        beams, gold, N, VIEW, normalization=normalization, max_width=max_width
    )
    exact = accuracy_at_beam_size(
        beams, gold, normalization=normalization, max_width=max_width
    )

    assert shape.items == exact.items == len(scored_items)
    assert len(shape.hits) == max_width
    assert all(
        by_shape >= by_label
        for by_shape, by_label in zip(shape.hits, exact.hits, strict=True)
    )


def test_mean_reciprocal_rank_is_exact_with_gold_at_rank_three() -> None:
    beam = scored(INTRANSITIVE, MODIFIER, TRANSITIVE, NP)

    assert (
        mean_reciprocal_rank(
            (beam,), (TRANSITIVE,), normalization=BeamNormalization.NONE
        )
        == 1 / 3
    )


def test_mean_reciprocal_rank_averages_over_items_with_absent_gold_at_zero() -> None:
    beams = (
        scored(TRANSITIVE, NP),
        scored(NP, N, TRANSITIVE),
        scored(NP, N),
        (),
    )
    gold = (TRANSITIVE,) * 4

    assert mean_reciprocal_rank(
        beams, gold, normalization=BeamNormalization.NONE
    ) == pytest.approx((1 + 1 / 3) / 4)


@pytest.mark.parametrize("normalization", NORMALIZATIONS)
def test_mean_reciprocal_rank_follows_the_normalization(
    normalization: BeamNormalization,
) -> None:
    ranked = normalization.ranked(CONTESTED)

    assert mean_reciprocal_rank(
        (CONTESTED,), (ranked[0],), normalization=normalization
    ) == pytest.approx(1.0)
    assert mean_reciprocal_rank(
        (CONTESTED,), (ranked[2],), normalization=normalization
    ) == pytest.approx(1 / 3)


@settings(max_examples=60, deadline=None)
@given(items(min_size=1), st.sampled_from(NORMALIZATIONS))
def test_mean_reciprocal_rank_is_the_mean_of_one_over_the_gold_rank(
    scored_items: list[tuple[ScoredBeam, Term]],
    normalization: BeamNormalization,
) -> None:
    beams = tuple(beam for beam, _ in scored_items)
    gold = tuple(target for _, target in scored_items)
    expected = 0.0

    for beam, target in scored_items:
        ranked = normalization.ranked(beam)
        ranks = [
            rank
            for rank, candidate in enumerate(ranked, start=1)
            if candidate.key() == target.key()
        ]
        expected += 1 / ranks[0] if ranks else 0.0

    assert mean_reciprocal_rank(
        beams, gold, normalization=normalization
    ) == pytest.approx(expected / len(scored_items))


def test_the_sweep_re_decodes_at_each_width_rather_than_taking_prefixes() -> None:
    decoded: dict[int, tuple[ScoredBeam, ...]] = {
        1: (scored(NP), scored(TRANSITIVE)),
        2: (scored(NP, TRANSITIVE), scored(TRANSITIVE, NP)),
        3: (scored(N, NP, S), scored(TRANSITIVE, NP, N)),
    }
    asked: list[int] = []

    def decode(width: int) -> tuple[ScoredBeam, ...]:
        asked.append(width)

        return decoded[width]

    points = beam_size_sweep(decode, (TRANSITIVE, TRANSITIVE), widths=(1, 2, 3))

    assert asked == [1, 2, 3]
    assert points == (SweepPoint(1, 1, 2), SweepPoint(2, 2, 2), SweepPoint(3, 1, 2))
    assert [point.rate for point in points] == [0.5, 1.0, 0.5]


def test_a_sweep_over_no_widths_decodes_nothing() -> None:
    def decode(width: int) -> tuple[ScoredBeam, ...]:
        msg = f"decoded at {width}"
        raise AssertionError(msg)

    assert beam_size_sweep(decode, (TRANSITIVE,), widths=()) == ()


def test_a_sweep_over_no_items_has_no_rate() -> None:
    def decode(width: int) -> tuple[ScoredBeam, ...]:
        del width

        return ()

    points = beam_size_sweep(decode, (), widths=(2,))

    assert points == (SweepPoint(2, 0, 0),)
    assert points[0].rate is None


@pytest.mark.parametrize("widths", [(0,), (2, 2), (3, 1), (1, -1)])
def test_sweep_widths_ascend_from_one(widths: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="ascend from one"):
        beam_size_sweep(lambda width: (scored(NP)[:width],), (NP,), widths=widths)


def test_a_sweep_refuses_a_beam_wider_than_its_width() -> None:
    def decode(width: int) -> tuple[ScoredBeam, ...]:
        del width

        return (scored(NP, N),)

    with pytest.raises(ValueError, match="width 1 holds 2"):
        beam_size_sweep(decode, (NP,), widths=(1,))


def test_a_sweep_refuses_a_decode_without_one_beam_per_gold_type() -> None:
    with pytest.raises(ValueError, match="1 beams and 2 gold"):
        beam_size_sweep(lambda width: (scored(NP),)[:width], (NP, S), widths=(1,))


def test_truncation_rebuilds_a_foreign_constructor_in_the_declared_store() -> None:
    foreign = OTHER.op(
        SIGNATURE.by_name("fwd"),
        OTHER.op(SIGNATURE.by_name("s")),
        OTHER.op(SIGNATURE.by_name("np")),
    )

    curve = depth_truncated_accuracy(
        (scored(foreign),),
        (VIEW.make("fwd", (S, NP)),),
        N,
        VIEW,
        normalization=BeamNormalization.NONE,
        max_depth=3,
    )

    assert curve.hits == (1, 1, 1)


def test_typeless_projection_rebuilds_a_foreign_shape_in_the_constructor_store() -> (
    None
):
    foreign = OTHER.op(
        SIGNATURE.by_name("fwd"),
        OTHER.op(SIGNATURE.by_name("s")),
        OTHER.op(SIGNATURE.by_name("np")),
    )

    curve = typeless_accuracy(
        (scored(foreign),),
        (MODIFIER,),
        N,
        VIEW,
        normalization=BeamNormalization.NONE,
        max_width=1,
    )

    assert curve.hits == (1,)


def test_arrow_product_beams_use_the_declared_constructor_view() -> None:
    signature = Signature(
        "arrow-product-beams",
        ("Cat",),
        (
            Operator("n", "Cat"),
            Operator("np", "Cat"),
            Operator("arrow", "Cat", arg_sorts=("Cat", "Cat")),
            Operator("product", "Cat", arg_sorts=("Cat", "Cat")),
        ),
    )
    store = TermStore(signature)
    view = DeclaredConstructors(
        store,
        {
            "arrow": ("result", "argument"),
            "product": ("left-factor", "right-factor"),
        },
    )
    n = store.op(signature.by_name("n"))
    np = store.op(signature.by_name("np"))
    predicted = view.make("product", (view.make("arrow", (n, n)), n))
    gold = view.make("product", (view.make("arrow", (np, n)), n))

    shaped = typeless_accuracy(
        (scored(predicted),),
        (gold,),
        n,
        view,
        normalization=BeamNormalization.NONE,
        max_width=1,
    )
    truncated = depth_truncated_accuracy(
        (scored(predicted),),
        (gold,),
        n,
        view,
        normalization=BeamNormalization.NONE,
        max_depth=3,
    )

    assert shaped.hits == (1,)
    assert truncated.hits == (1, 1, 0)


def test_no_items_give_no_rates() -> None:
    by_width = accuracy_at_beam_size(
        (), (), normalization=BeamNormalization.NONE, max_width=3
    )
    by_depth = depth_truncated_accuracy(
        (), (), N, VIEW, normalization=BeamNormalization.NONE, max_depth=2
    )
    by_shape = typeless_accuracy(
        (), (), N, VIEW, normalization=BeamNormalization.NONE, max_width=2
    )

    assert by_width.items == 0
    assert by_width.hits == (0, 0, 0)
    assert by_width.rates == ()
    assert by_width.at(3) is None
    assert by_depth.rates == ()
    assert by_shape.rates == ()
    assert mean_reciprocal_rank((), (), normalization=BeamNormalization.NONE) is None


def test_one_beam_per_gold_type_is_required() -> None:
    with pytest.raises(ValueError, match="2 beams and 1 gold"):
        accuracy_at_beam_size(
            (scored(NP), scored(S)),
            (NP,),
            normalization=BeamNormalization.NONE,
            max_width=1,
        )

    with pytest.raises(ValueError, match="1 beams and 2 gold"):
        depth_truncated_accuracy(
            (scored(NP),),
            (NP, S),
            N,
            VIEW,
            normalization=BeamNormalization.NONE,
            max_depth=1,
        )

    with pytest.raises(ValueError, match="1 beams and 2 gold"):
        typeless_accuracy(
            (scored(NP),),
            (NP, S),
            N,
            VIEW,
            normalization=BeamNormalization.NONE,
            max_width=1,
        )

    with pytest.raises(ValueError, match="2 beams and 1 gold"):
        mean_reciprocal_rank(
            (scored(NP), scored(S)), (NP,), normalization=BeamNormalization.NONE
        )


@pytest.mark.parametrize("bound", [0, -2])
def test_a_width_or_depth_below_one_is_refused(bound: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        accuracy_at_beam_size(
            (scored(NP),), (NP,), normalization=BeamNormalization.NONE, max_width=bound
        )

    with pytest.raises(ValueError, match="positive"):
        depth_truncated_accuracy(
            (scored(NP),),
            (NP,),
            N,
            VIEW,
            normalization=BeamNormalization.NONE,
            max_depth=bound,
        )

    with pytest.raises(ValueError, match="positive"):
        typeless_accuracy(
            (scored(NP),),
            (NP,),
            N,
            VIEW,
            normalization=BeamNormalization.NONE,
            max_width=bound,
        )


def test_a_type_outside_the_declared_constructor_algebra_is_refused() -> None:
    with pytest.raises(NeighborhoodError):
        depth_truncated_accuracy(
            (scored(WRAPPED),),
            (NP,),
            N,
            VIEW,
            normalization=BeamNormalization.NONE,
            max_depth=2,
        )

    with pytest.raises(NeighborhoodError):
        depth_truncated_accuracy(
            (scored(NP),),
            (NP,),
            TRANSITIVE,
            VIEW,
            normalization=BeamNormalization.NONE,
            max_depth=2,
        )

    with pytest.raises(NeighborhoodError):
        typeless_accuracy(
            (scored(WRAPPED),),
            (NP,),
            N,
            VIEW,
            normalization=BeamNormalization.NONE,
            max_width=1,
        )

    with pytest.raises(NeighborhoodError):
        typeless_accuracy(
            (scored(NP),),
            (NP,),
            TRANSITIVE,
            VIEW,
            normalization=BeamNormalization.NONE,
            max_width=1,
        )


def test_a_curve_is_a_frozen_value_equal_on_its_hits_and_items() -> None:
    curve = AccuracyCurve([1, 2], 3)

    assert curve == AccuracyCurve((1, 2), 3)
    assert curve != AccuracyCurve((1, 2), 4)
    assert hash(curve) == hash(AccuracyCurve((1, 2), 3))
    assert repr(curve) == "AccuracyCurve([1, 2], 3)"
    assert curve.rates == (1 / 3, 2 / 3)

    with pytest.raises(AttributeError, match="immutable"):
        curve.__setattr__("_items", 5)

    with pytest.raises(AttributeError, match="immutable"):
        curve.__delattr__("_items")


def test_a_curve_rejects_more_hits_than_items_and_levels_it_lacks() -> None:
    with pytest.raises(ValueError, match="level 2 counts 4 hits among 3"):
        AccuracyCurve([1, 4], 3)

    with pytest.raises(ValueError, match="negative"):
        AccuracyCurve([1], -1)

    with pytest.raises(IndexError, match="levels 1 to 2"):
        AccuracyCurve([1, 2], 3).at(3)

    with pytest.raises(IndexError, match="levels 1 to 2"):
        AccuracyCurve([1, 2], 3).at(0)

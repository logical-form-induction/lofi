import pytest
import torch

from lofi.deduce import KBest, SpanIndex
from lofi.deduce import LogInside as SymbolicLogInside
from lofi.nn.parsers._decode import k_best, mbr_trees
from lofi.nn.parsers._inside import InsideKernel
from lofi.nn.parsers._marginals import log_constituent_marginals
from lofi.nn.parsers._outside import OutsideKernel
from lofi.nn.parsers._reduction import LogSumExpReduction, MaximumReduction

NEG_INF = float("-inf")

#: Absolute tolerance for log weights.
TOLERANCE = 1e-9

#: Number of derivations used for k-best comparisons.
KEPT = 3

#: Seed for the deterministic MBR comparison.
DIVERGENT_SEED = 1

#: Input length for the deterministic MBR comparison.
DIVERGENT_LENGTH = 5


def scored(
    index: SpanIndex, seed: int, batch: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    leaves = torch.randn(batch, index.n, generator=generator, dtype=torch.float64)
    splits = torch.randn(
        batch,
        index.num_spans,
        index.max_splits,
        generator=generator,
        dtype=torch.float64,
    )

    return (leaves, splits)


def charts(
    leaves: torch.Tensor,
    splits: torch.Tensor,
    index: SpanIndex,
    lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    inside = InsideKernel(LogSumExpReduction()).inside(leaves, splits, index)
    outside = OutsideKernel(LogSumExpReduction()).outside(
        inside, splits, index, lengths=lengths
    )

    return (inside, outside)


def derivations(
    index: SpanIndex, leaves: torch.Tensor, splits: torch.Tensor, i: int, j: int
) -> list[tuple[float, frozenset[tuple[int, int]]]]:
    if j - i == 1:
        return [(leaves[i].item(), frozenset())]

    found = []

    for k in index.splits(i, j):
        edge = splits[index.span_id(i, j), k - i - 1].item()
        found += [
            (left + right + edge, frozenset({(i, j)}) | held | rest)
            for left, held in derivations(index, leaves, splits, i, k)
            for right, rest in derivations(index, leaves, splits, k, j)
        ]

    return found


def ranked(
    index: SpanIndex, leaves: torch.Tensor, splits: torch.Tensor
) -> list[tuple[float, frozenset[tuple[int, int]]]]:
    return sorted(
        derivations(index, leaves, splits, 0, index.n),
        key=lambda tree: -tree[0],
    )


def symbolic_k_best(
    index: SpanIndex, leaves: torch.Tensor, splits: torch.Tensor, k: int
) -> tuple[float, ...]:
    reduction = KBest(k)
    chart: dict[tuple[int, int], tuple[float, ...]] = {}

    for i, j in index.spans_by_length():
        if j - i == 1:
            chart[(i, j)] = (leaves[i].item(),)
            continue

        total = reduction.zero

        for cut in index.splits(i, j):
            edge = (splits[index.span_id(i, j), cut - i - 1].item(),)
            total = reduction.plus(
                total,
                reduction.times(
                    edge, reduction.times(chart[(i, cut)], chart[(cut, j)])
                ),
            )

        chart[(i, j)] = total

    return chart[(0, index.n)]


def constituent_gain(
    index: SpanIndex, marginals: torch.Tensor, held: frozenset[tuple[int, int]]
) -> float:
    return sum(marginals[index.span_id(i, j)].item() for i, j in held)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_the_best_few_trees_are_the_best_few_found_by_enumeration(n: int) -> None:
    index = SpanIndex(n)
    leaves, splits = scored(index, seed=n)
    reference = ranked(index, leaves[0], splits[0])[:KEPT]

    found = k_best(leaves, splits, index, k=KEPT)

    assert len(found[0]) == len(reference)

    for tree, (score, held) in zip(found[0], reference, strict=True):
        assert tree.constituents == tuple(
            sorted(held, key=lambda cell: index.span_id(cell[0], cell[1]))
        )
        assert tree.total == pytest.approx(score, abs=TOLERANCE)


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_the_best_few_weights_are_the_symbolic_k_best_reduction(n: int) -> None:
    index = SpanIndex(n)
    leaves, splits = scored(index, seed=n + 1)

    found = k_best(leaves, splits, index, k=KEPT)

    assert [tree.total for tree in found[0]] == pytest.approx(
        list(symbolic_k_best(index, leaves[0], splits[0], KEPT)), abs=TOLERANCE
    )


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_the_heaviest_tree_weighs_what_viterbi_says_it_does(n: int) -> None:
    index = SpanIndex(n)
    leaves, splits = scored(index, seed=n + 2)

    found = k_best(leaves, splits, index, k=KEPT)
    best = InsideKernel(MaximumReduction()).log_partition(leaves, splits, index)

    assert found[0][0].total == pytest.approx(best.item(), abs=TOLERANCE)


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_the_trees_come_back_heaviest_first_and_all_different(n: int) -> None:
    index = SpanIndex(n)
    leaves, splits = scored(index, seed=n + 3)

    found = k_best(leaves, splits, index, k=KEPT)
    totals = [tree.total for tree in found[0]]

    assert totals == sorted(totals, reverse=True)
    assert len(set(found[0])) == len(found[0])


def test_an_item_with_fewer_trees_than_asked_for_yields_what_it_has() -> None:
    index = SpanIndex(2)
    leaves, splits = scored(index, seed=5)

    found = k_best(leaves, splits, index, k=KEPT)

    assert len(found[0]) == 1
    assert found[0][0].constituents == ((0, 2),)


def test_a_one_token_item_yields_one_tree_holding_no_constituent() -> None:
    index = SpanIndex(1)
    leaves, splits = scored(index, seed=7)

    found = k_best(leaves, splits, index, k=KEPT)

    assert len(found[0]) == 1
    assert found[0][0].constituents == ()
    assert found[0][0].total == pytest.approx(leaves[0, 0].item())


def test_an_item_nothing_derives_yields_no_tree() -> None:
    index = SpanIndex(4)
    leaves = torch.zeros(1, index.n, dtype=torch.float64)
    splits = torch.full(
        (1, index.num_spans, index.max_splits), NEG_INF, dtype=torch.float64
    )
    inside, outside = charts(leaves, splits, index)

    assert k_best(leaves, splits, index, k=KEPT) == ((),)
    assert mbr_trees(inside, outside, splits, index) == (None,)


def test_a_ragged_batch_decodes_each_item_over_its_own_tokens() -> None:
    index = SpanIndex(5)
    leaves, splits = scored(index, seed=11, batch=3)
    lengths = torch.tensor([5, 3, 1])

    found = k_best(leaves, splits, index, k=1, lengths=lengths)

    for item, length in enumerate(lengths.tolist()):
        tree = found[item][0]
        reference = max(
            derivations(index, leaves[item], splits[item], 0, length),
            key=lambda candidate: candidate[0],
        )

        assert len(tree.constituents) == length - 1
        assert set(tree.constituents) == set(reference[1])
        assert tree.total == pytest.approx(reference[0], abs=TOLERANCE)


def test_the_decode_refuses_a_width_of_less_than_one() -> None:
    index = SpanIndex(3)
    leaves, splits = scored(index, seed=13)

    with pytest.raises(ValueError, match="at least one derivation"):
        k_best(leaves, splits, index, k=0)


def test_the_decode_refuses_scores_it_cannot_read() -> None:
    index = SpanIndex(3)
    leaves, splits = scored(index, seed=17)

    with pytest.raises(TypeError, match="leaf scores are log weights"):
        k_best(leaves.long(), splits, index, k=1)

    with pytest.raises(ValueError, match="split scores"):
        k_best(leaves, splits[:, :, :-1], index, k=1)


def test_the_decode_carries_no_gradient_out_of_the_chart() -> None:
    index = SpanIndex(4)
    leaves, splits = scored(index, seed=19)
    splits.requires_grad_()

    found = k_best(leaves, splits, index, k=1)

    assert isinstance(found[0][0].total, float)
    assert splits.grad is None


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_the_mbr_tree_holds_the_most_constituent_mass_of_any_tree(n: int) -> None:
    index = SpanIndex(n)
    leaves, splits = scored(index, seed=n + 23)
    inside, outside = charts(leaves, splits, index)
    marginals = log_constituent_marginals(inside, outside, index).exp()[0]

    tree = mbr_trees(inside, outside, splits, index)[0]

    assert tree is not None
    assert tree.total == pytest.approx(
        max(
            constituent_gain(index, marginals, held)
            for _, held in derivations(index, leaves[0], splits[0], 0, n)
        ),
        abs=TOLERANCE,
    )
    assert tree.total == pytest.approx(
        constituent_gain(index, marginals, frozenset(tree.constituents)), abs=TOLERANCE
    )


def test_the_mbr_tree_is_not_always_the_heaviest_tree() -> None:
    index = SpanIndex(DIVERGENT_LENGTH)
    leaves, splits = scored(index, seed=DIVERGENT_SEED)
    inside, outside = charts(leaves, splits, index)

    tree = mbr_trees(inside, outside, splits, index)[0]
    heaviest = k_best(leaves, splits, index, k=1)[0][0]

    assert tree is not None
    assert tree != heaviest


def test_a_chart_certain_of_one_tree_decodes_that_tree_either_way() -> None:
    index = SpanIndex(5)
    leaves, splits = scored(index, seed=29)
    # fixed split points admit exactly one derivation
    splits = splits.masked_fill(
        torch.tensor(index.split_mask).unsqueeze(0), NEG_INF
    ).index_put(
        (torch.tensor([0]), torch.arange(index.num_spans), torch.zeros(1).long()),
        torch.zeros(1, dtype=torch.float64),
    )
    inside, outside = charts(leaves, splits, index)

    tree = mbr_trees(inside, outside, splits, index)[0]
    heaviest = k_best(leaves, splits, index, k=1)[0][0]

    assert tree == heaviest
    assert tree is not None
    assert tree.total == pytest.approx(index.n - 1, abs=TOLERANCE)


def test_the_mbr_tree_uses_no_decomposition_the_chart_gives_no_mass() -> None:
    index = SpanIndex(4)
    leaves = torch.zeros(1, index.n, dtype=torch.float64)
    splits = torch.zeros(1, index.num_spans, index.max_splits, dtype=torch.float64)
    # the goal split makes every crossing span unreachable
    splits[0, index.span_id(0, 4), 1:] = NEG_INF
    inside, outside = charts(leaves, splits, index)

    tree = mbr_trees(inside, outside, splits, index)[0]

    assert tree is not None
    assert (0, 4) in tree.constituents
    assert not {(0, 2), (0, 3)} & set(tree.constituents)


def test_the_symbolic_and_tensor_partitions_agree_on_the_fixture() -> None:
    index = SpanIndex(5)
    leaves, splits = scored(index, seed=31)
    reduction = SymbolicLogInside()
    total = reduction.zero

    for score, _ in derivations(index, leaves[0], splits[0], 0, index.n):
        total = reduction.plus(total, score)

    assert InsideKernel(LogSumExpReduction()).log_partition(
        leaves, splits, index
    ).item() == pytest.approx(total, abs=TOLERANCE)


def test_a_ragged_batch_brackets_each_item_over_its_own_tokens() -> None:
    index = SpanIndex(5)
    leaves, splits = scored(index, seed=37, batch=3)
    lengths = torch.tensor([5, 3, 1])
    inside, outside = charts(leaves, splits, index, lengths)
    marginals = log_constituent_marginals(inside, outside, index, lengths=lengths).exp()

    found = mbr_trees(inside, outside, splits, index, lengths=lengths)

    for item, length in enumerate(lengths.tolist()):
        tree = found[item]

        assert tree is not None
        assert len(tree.constituents) == length - 1
        assert all(j <= length for _, j in tree.constituents)
        assert tree.total == pytest.approx(
            max(
                constituent_gain(index, marginals[item], held)
                for _, held in derivations(index, leaves[item], splits[item], 0, length)
            ),
            abs=TOLERANCE,
        )

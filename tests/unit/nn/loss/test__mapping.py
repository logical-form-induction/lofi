import itertools
import math
from typing import TYPE_CHECKING, Final, cast

import pytest
import torch

from lofi.nn._values import LogDistribution
from lofi.nn.loss._divergence import (
    KL,
    CrossEntropy,
    FenchelYoung,
)
from lofi.nn.loss._errors import SparseDivergenceError
from lofi.nn.loss._kernels import (
    cross_entropy,
    fenchel_young,
    kullback_leibler,
)
from lofi.nn.loss._mapping import (
    SOFTMAX,
    Entmax,
    _solved,
    ensure_finite_pairing,
)

if TYPE_CHECKING:
    from torch import Tensor

SYMBOL_AXES: Final[tuple[str, ...]] = ("node", "symbol")
SCORES: Final = torch.tensor(
    [[0.3, 1.1, 1.7, -2.0], [0.0, 0.2, -0.1, 0.4]], dtype=torch.float64
)


def distribution(log_probabilities: Tensor) -> LogDistribution:
    return LogDistribution(
        log_probabilities, SYMBOL_AXES, normalized_over=SYMBOL_AXES[-1]
    )


def test_alpha_below_one_leaves_the_family_the_mapping_is_defined_over() -> None:
    with pytest.raises(ValueError, match="from one upward"):
        Entmax(0.5, bisections=50)


def test_a_non_numeric_alpha_is_refused_by_type() -> None:
    with pytest.raises(TypeError, match="alpha is not a number"):
        Entmax(cast("float", "1.5"), bisections=50)


def test_an_infinite_alpha_is_refused() -> None:
    with pytest.raises(ValueError, match="alpha"):
        Entmax(float("inf"), bisections=50)


def test_the_dense_mapping_is_the_family_at_alpha_one() -> None:
    assert Entmax(1.0, bisections=0) == SOFTMAX
    assert hash(SOFTMAX) == hash(Entmax(1.0, bisections=0))
    assert SOFTMAX.alpha == 1.0
    assert not SOFTMAX.sparse
    assert repr(SOFTMAX) == "Entmax(1.0, bisections=0)"


def test_only_alpha_above_one_makes_the_mapping_sparse() -> None:
    assert not Entmax(1.0, bisections=50).sparse
    assert Entmax(1.0001, bisections=50).sparse
    assert Entmax(2.0, bisections=50).sparse


def test_two_mappings_at_different_alphas_are_distinct() -> None:
    assert Entmax(1.5, bisections=50) != Entmax(2.0, bisections=50)
    assert Entmax(1.5, bisections=50) != 1.5


def test_the_dense_mapping_is_the_log_softmax_of_the_scores() -> None:
    assert torch.allclose(SOFTMAX.project(SCORES), SCORES.log_softmax(dim=-1))


def test_the_dense_mapping_leaves_no_symbol_at_exactly_zero_mass() -> None:
    assert bool((SOFTMAX.project(SCORES).exp() > 0.0).all())


def test_the_sparse_mapping_leaves_symbols_at_exactly_zero_mass() -> None:
    mapped = Entmax(1.5, bisections=50).project(SCORES)

    assert bool(mapped[0].isneginf().any())
    assert float(mapped[0].exp()[3]) == 0.0


def test_the_sparse_mapping_still_normalizes_what_it_keeps() -> None:
    for alpha in (1.5, 2.0, 3.0):
        mapped = Entmax(alpha, bisections=50).project(SCORES).exp()

        assert torch.allclose(mapped.sum(dim=-1), torch.ones(2, dtype=torch.float64))


def test_the_sparse_mapping_agrees_with_the_closed_form_at_alpha_two() -> None:
    row = torch.tensor([[0.0, 0.2, -0.1, 0.4]], dtype=torch.float64)
    expected = row - (row.sum() - 1.0) / 4.0

    assert torch.allclose(Entmax(2.0, bisections=50).project(row).exp(), expected)


def test_the_mapping_is_invariant_to_a_shift_shared_by_a_row() -> None:
    for mapping in (SOFTMAX, Entmax(1.5, bisections=50), Entmax(2.0, bisections=50)):
        shifted = mapping.project(SCORES + 7.0)

        assert torch.allclose(shifted, mapping.project(SCORES))


def test_a_readout_that_already_normalizes_publishes_the_same_distribution() -> None:
    for mapping in (SOFTMAX, Entmax(1.5, bisections=50), Entmax(2.0, bisections=50)):
        through = mapping.project(SOFTMAX.project(SCORES))

        assert torch.allclose(through, mapping.project(SCORES))


def test_a_symbol_the_decoder_rules_out_keeps_no_mass() -> None:
    scores = torch.tensor([[0.3, 1.1, -torch.inf, 1.0]], dtype=torch.float64)

    for mapping in (SOFTMAX, Entmax(1.5, bisections=50), Entmax(2.0, bisections=50)):
        mapped = mapping.project(scores)

        assert float(mapped[0, 2]) == -torch.inf
        assert torch.isclose(mapped.exp().sum(), torch.ones((), dtype=torch.float64))


def test_a_row_with_no_available_symbol_has_no_distribution_to_read() -> None:
    scores = torch.full((1, 4), -torch.inf, dtype=torch.float64)

    for mapping in (SOFTMAX, Entmax(1.5, bisections=50), Entmax(2.0, bisections=50)):
        assert bool(mapping.project(scores).isneginf().all())


@pytest.mark.parametrize("diverged", [float("nan"), float("inf")])
def test_a_diverged_score_reads_as_a_numerical_failure_not_an_absent_row(
    diverged: float,
) -> None:
    scores = torch.full((1, 3), diverged, dtype=torch.float64)

    for mapping in (SOFTMAX, Entmax(1.5, bisections=50), Entmax(2.0, bisections=50)):
        assert bool(mapping.project(scores).isnan().any())


def test_the_threshold_search_normalizes_what_it_returns() -> None:
    weights = _solved(SCORES, 1.5, 50)

    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, dtype=torch.float64))
    assert bool((weights >= 0.0).all())


def test_the_mapping_carries_the_gradient_of_the_mapping_it_computes() -> None:
    scores = SCORES.clone().requires_grad_()

    for alpha in (1.5, 2.0, 3.0):
        mapping = Entmax(alpha, bisections=50)

        assert torch.autograd.gradcheck(
            lambda given, mapping=mapping: mapping.project(given).exp(),
            (scores,),
            eps=1e-6,
            atol=1e-6,
        )


def test_a_symbol_off_the_support_takes_no_gradient() -> None:
    scores = SCORES.clone().requires_grad_()
    mapped = Entmax(1.5, bisections=50).project(scores).exp()
    mapped.sum().backward()

    assert scores.grad is not None
    assert float(scores.grad[0, 3]) == 0.0


def test_the_dense_mapping_admits_every_divergence() -> None:
    assert ensure_finite_pairing(SOFTMAX, KL) == KL
    assert ensure_finite_pairing(SOFTMAX, CrossEntropy()) == CrossEntropy()
    assert ensure_finite_pairing(SOFTMAX, FenchelYoung(1.5)) == FenchelYoung(1.5)


def test_a_sparse_mapping_refuses_the_divergences_it_makes_infinite() -> None:
    for divergence in (KL, CrossEntropy()):
        with pytest.raises(SparseDivergenceError) as raised:
            ensure_finite_pairing(Entmax(1.5, bisections=50), divergence)

        message = str(raised.value)

        assert "Entmax(1.5, bisections=50)" in message
        assert repr(divergence) in message
        assert "FenchelYoung(1.5)" in message


def test_a_sparse_mapping_admits_the_loss_built_on_its_own_entropy() -> None:
    admitted = ensure_finite_pairing(Entmax(1.5, bisections=50), FenchelYoung(1.5))

    assert admitted == FenchelYoung(1.5)


def test_the_fenchel_young_loss_at_alpha_one_computes_the_kl_divergence() -> None:
    reference = distribution(SOFTMAX.project(SCORES))
    shifted = distribution(
        SOFTMAX.project(SCORES + torch.tensor([[0.0, 0.0, 0.0, 3.0]]))
    )
    sparse = distribution(Entmax(1.5, bisections=50).project(SCORES))

    assert float(fenchel_young(reference, shifted, alpha=1.0)) == pytest.approx(
        float(kullback_leibler(reference, shifted)), abs=1e-12
    )
    assert float(fenchel_young(reference, sparse, alpha=1.0)) == torch.inf


def test_a_sparse_mapping_refuses_the_fenchel_young_loss_that_is_kl() -> None:
    with pytest.raises(SparseDivergenceError) as raised:
        ensure_finite_pairing(Entmax(1.5, bisections=50), FenchelYoung(1.0))

    message = str(raised.value)

    assert "Entmax(1.5, bisections=50)" in message
    assert "FenchelYoung(1.0)" in message
    assert "FenchelYoung(1.5)" in message


def test_the_dense_mapping_admits_the_fenchel_young_loss_that_is_kl() -> None:
    assert ensure_finite_pairing(SOFTMAX, FenchelYoung(1.0)) == FenchelYoung(1.0)
    assert ensure_finite_pairing(SOFTMAX, KL) == KL


def test_the_divergences_a_sparse_mapping_refuses_are_genuinely_infinite() -> None:
    reference = distribution(SOFTMAX.project(SCORES))
    model = distribution(Entmax(1.5, bisections=50).project(SCORES))

    assert float(kullback_leibler(reference, model)) == torch.inf
    assert float(cross_entropy(reference, model)) == torch.inf


def test_the_fenchel_young_loss_is_finite_on_the_inputs_that_make_kl_infinite() -> None:
    reference = distribution(SOFTMAX.project(SCORES))
    model = distribution(Entmax(1.5, bisections=50).project(SCORES))
    measured = float(fenchel_young(reference, model, alpha=1.5))

    assert math.isfinite(measured)
    assert measured > 0.0


def test_the_fenchel_young_loss_is_zero_exactly_where_the_distributions_match() -> None:
    mapping = Entmax(1.5, bisections=50)
    matched = distribution(mapping.project(SCORES))
    apart = distribution(mapping.project(SCORES + torch.tensor([[0.0, 0.0, 0.0, 3.0]])))

    assert float(fenchel_young(matched, matched, alpha=1.5)) == pytest.approx(
        0.0, abs=1e-12
    )
    assert float(fenchel_young(matched, apart, alpha=1.5)) > 0.0


def test_the_mass_outside_a_sparse_support_is_exactly_zero() -> None:
    sparse = Entmax(1.5, bisections=50).project(SCORES).exp()
    dense = SOFTMAX.project(SCORES).exp()
    support = sparse > 0.0

    assert not bool(support.all())
    assert float(sparse[~support].sum()) == 0.0
    assert float(sparse[support].sum()) == pytest.approx(2.0, abs=1e-12)
    assert float(dense[~support].sum()) > 0.0
    assert float(dense[support].sum()) < 2.0


def test_the_fenchel_young_loss_separates_every_pair_that_differs() -> None:
    mapping = Entmax(1.5, bisections=50)
    generator = torch.Generator().manual_seed(11)
    scores = torch.randn(16, 2, 6, generator=generator, dtype=torch.float64)

    for left in scores:
        for right in scores:
            first = distribution(mapping.project(left))
            second = distribution(mapping.project(right))
            measured = float(fenchel_young(first, second, alpha=1.5))

            if torch.equal(first.tensor, second.tensor):
                assert measured == pytest.approx(0.0, abs=1e-12)

            else:
                assert measured > 1e-9


def test_the_types_a_sparse_readout_gives_mass_to_are_a_finite_exact_set() -> None:
    generator = torch.Generator().manual_seed(5)
    scores = torch.randn(3, 5, generator=generator, dtype=torch.float64) * 2.0
    sparse = Entmax(2.0, bisections=50).project(scores).exp()
    dense = SOFTMAX.project(scores).exp()
    supports = [row.nonzero().flatten().tolist() for row in sparse]
    enumerated = list(itertools.product(*supports))

    assert len(enumerated) < 5**3
    assert sum(
        math.prod(float(sparse[slot, symbol]) for slot, symbol in enumerate(kind))
        for kind in enumerated
    ) == pytest.approx(1.0, abs=1e-12)
    assert (
        sum(
            math.prod(float(dense[slot, symbol]) for slot, symbol in enumerate(kind))
            for kind in enumerated
        )
        < 1.0
    )


def test_the_sparse_mapping_normalizes_tightly_at_single_precision() -> None:
    generator = torch.Generator().manual_seed(7)
    wide = (torch.randn(512, 64, generator=generator, dtype=torch.float64) * 3.0).to(
        torch.float32
    )

    for alpha in (1.5, 2.0, 3.0, 5.0):
        mass = Entmax(alpha, bisections=50).project(wide).exp().sum(dim=-1)

        assert float((mass - 1.0).abs().max()) < 1e-6


def test_the_threshold_search_runs_the_declared_number_of_bisections() -> None:
    coarse = Entmax(1.5, bisections=2).project(SCORES).exp()
    fine = Entmax(1.5, bisections=50).project(SCORES).exp()

    assert Entmax(1.5, bisections=2) != Entmax(1.5, bisections=50)
    assert not torch.allclose(coarse, fine)
    assert torch.allclose(fine.sum(dim=-1), torch.ones(2, dtype=torch.float64))


def test_a_sparse_mapping_needs_a_positive_bisection_count() -> None:
    with pytest.raises(ValueError, match="bisection"):
        Entmax(1.5, bisections=0)

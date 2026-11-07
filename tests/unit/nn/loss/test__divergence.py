from typing import cast

import pytest

from lofi.nn.loss._divergence import (
    KL,
    CrossEntropy,
    FenchelYoung,
    KullbackLeibler,
    resolved,
)


def test_the_default_divergence_is_forward_kullback_leibler() -> None:
    assert KullbackLeibler() == KL
    assert isinstance(KL, KullbackLeibler)


def test_each_divergence_is_equal_to_itself_and_to_nothing_else() -> None:
    assert KullbackLeibler() == KullbackLeibler()
    assert CrossEntropy() == CrossEntropy()
    assert KullbackLeibler() != CrossEntropy()
    assert CrossEntropy() != KullbackLeibler()


def test_a_divergence_hashes_by_what_it_measures() -> None:
    chosen = {KullbackLeibler(), KullbackLeibler(), CrossEntropy()}

    assert len(chosen) == 2


def test_the_fenchel_young_loss_carries_the_entropy_it_is_built_on() -> None:
    loss = FenchelYoung(2.0)

    assert loss.alpha == 2.0
    assert loss == FenchelYoung(2.0)
    assert loss != FenchelYoung(1.5)
    assert len({FenchelYoung(2.0), FenchelYoung(2.0)}) == 1


def test_the_fenchel_young_loss_has_a_default_between_the_two_named_ones() -> None:
    assert 1.0 < FenchelYoung().alpha < 2.0


@pytest.mark.parametrize("alpha", [0.5, 0.0, -1.0])
def test_an_alpha_below_one_leaves_the_family_and_is_refused(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha indexes the entropy family"):
        FenchelYoung(alpha)


def test_an_alpha_that_is_not_a_finite_number_is_refused() -> None:
    with pytest.raises(ValueError, match="not finite"):
        FenchelYoung(float("inf"))

    with pytest.raises(TypeError, match="not a number"):
        FenchelYoung(cast("float", "2"))


def test_a_divergence_compares_unequal_to_a_value_of_another_kind() -> None:
    assert KullbackLeibler() != "kl"
    assert CrossEntropy() != 1
    assert FenchelYoung(1.5) != 1.5


def test_a_divergence_reads_back_as_the_call_that_rebuilds_it() -> None:
    assert repr(KullbackLeibler()) == "KullbackLeibler()"
    assert repr(CrossEntropy()) == "CrossEntropy()"
    assert repr(FenchelYoung(2.0)) == "FenchelYoung(2.0)"


def test_the_fenchel_young_loss_on_the_shannon_entropy_resolves_to_kl() -> None:
    assert resolved(FenchelYoung(1.0)) == KL
    assert isinstance(resolved(FenchelYoung(1.0)), KullbackLeibler)


@pytest.mark.parametrize(
    "divergence",
    [KullbackLeibler(), CrossEntropy(), FenchelYoung(1.5), FenchelYoung(2.0)],
)
def test_a_divergence_that_is_not_kl_by_another_name_resolves_to_itself(
    divergence: KullbackLeibler | CrossEntropy | FenchelYoung,
) -> None:
    assert resolved(divergence) == divergence

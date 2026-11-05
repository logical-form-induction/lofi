"""Tests for the bilinear controller declaration."""

from lofi.spec.controllers import (
    BilinearControllerSpec,
    EntmaxControllerNormalizerSpec,
)


def test_a_bilinear_controller_lowers_its_common_choices() -> None:
    declared = BilinearControllerSpec(
        input="span",
        features=("left", "product"),
        normalizer=EntmaxControllerNormalizerSpec(alpha=1.4),
    )

    assert declared.settings() == (
        "span",
        ("left", "product"),
        (1.4, 50),
        "none",
    )

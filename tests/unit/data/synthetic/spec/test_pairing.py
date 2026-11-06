"""Tests for declared premise--hypothesis pairing strategies."""

from lofi.data.synthetic._pairing import IndependentPairing, OverlapMatchedPairing
from lofi.data.synthetic.spec.pairing import (
    IndependentPairingSpec,
    OverlapMatchedPairingSpec,
)


def test_pairing_declarations_build_their_registered_strategy() -> None:
    assert isinstance(IndependentPairingSpec().build(), IndependentPairing)
    assert isinstance(OverlapMatchedPairingSpec().build(), OverlapMatchedPairing)

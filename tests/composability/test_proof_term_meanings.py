"""Meaning induction remains an independently swappable model axis."""

from typing import cast

from tests.acceptance.driver import composed

from lofi.run import plan_training
from lofi.spec.readouts import (
    DecodedMeaningSpec,
    ProofTermMeaningSpec,
    TermsReadoutSpec,
)


def test_decoded_and_proof_term_meanings_compose_with_the_same_training_point() -> None:
    decoded = composed(
        "proof-term-meanings.yaml",
        'model.readouts.forms.meanings={"kind":"decoded"}',
    )
    proof_term = composed("proof-term-meanings.yaml")
    decoded_readout = cast("TermsReadoutSpec", decoded.model.readouts["forms"])
    proof_term_readout = cast("TermsReadoutSpec", proof_term.model.readouts["forms"])

    assert isinstance(decoded_readout.meanings, DecodedMeaningSpec)
    assert isinstance(proof_term_readout.meanings, ProofTermMeaningSpec)
    assert decoded.model.parser == proof_term.model.parser
    assert decoded.loss == proof_term.loss
    assert decoded.trainer == proof_term.trainer
    assert (
        plan_training(decoded).sampler_streams
        == plan_training(proof_term).sampler_streams
    )

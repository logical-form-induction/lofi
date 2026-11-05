import didactic.api as dx
import pytest

from lofi.spec.decoding import ExhaustiveDecodingSpec
from lofi.spec.groundings import ExpectedEmbeddingSpec
from lofi.spec.readouts import ReadoutSource, TermsReadoutSpec


def test_expected_form_grounding_names_its_readout_and_likelihood_signal() -> None:
    declared = ExpectedEmbeddingSpec(
        readout="forms",
        vectors="expected-form",
        expectation="enumerate",
        hidden=8,
    )

    assert declared.form_readouts() == ("forms", "forms")
    assert declared.supported_signals() == frozenset({"likelihood"})
    declared.validate_readouts(
        {
            "forms": TermsReadoutSpec(
                space="semantics",
                reads=ReadoutSource.INSIDE,
                decoding=ExhaustiveDecodingSpec(residual_mass=0.1, cap=8),
            )
        }
    )


def test_expected_form_grounding_refuses_a_readout_without_complete_terms() -> None:
    declared = ExpectedEmbeddingSpec(
        readout="forms",
        vectors="expected-form",
        expectation="enumerate",
        hidden=8,
    )

    with pytest.raises(ValueError, match="no complete terms"):
        declared.validate_readouts(
            {
                "forms": TermsReadoutSpec(
                    space="semantics",
                    reads=ReadoutSource.INSIDE,
                )
            }
        )


def test_soft_expected_form_grounding_requires_marginal_rows() -> None:
    declared = ExpectedEmbeddingSpec(
        readout="forms",
        vectors="expected-form",
        expectation="soft",
        hidden=8,
    )

    declared.validate_readouts(
        {
            "forms": TermsReadoutSpec(
                space="semantics",
                reads=ReadoutSource.INSIDE,
            )
        }
    )

    with pytest.raises(ValueError, match="requires marginal rows"):
        declared.validate_readouts(
            {
                "forms": TermsReadoutSpec(
                    space="semantics",
                    reads=ReadoutSource.INSIDE,
                    decoding=ExhaustiveDecodingSpec(residual_mass=0.1, cap=8),
                )
            }
        )


def test_sentence_grounding_refuses_a_soft_term_expectation() -> None:
    with pytest.raises(dx.ValidationError, match="sentence-vector grounding"):
        ExpectedEmbeddingSpec(
            readout="forms",
            vectors="sentence",
            expectation="soft",
            hidden=8,
        )


def test_sentence_grounding_reads_no_form_readout() -> None:
    declared = ExpectedEmbeddingSpec(
        readout="forms",
        vectors="sentence",
        expectation="enumerate",
        hidden=8,
    )

    assert declared.form_readouts() is None
    declared.validate_readouts({})

"""Tests for the logical form of an item and the entailment certificate."""

from math import inf, nan
from typing import TYPE_CHECKING, Final, cast

import pytest
from tests.verdicts import proved, refuted

from lofi._types import ItemId
from lofi.data.synthetic import (
    ENTAILMENT_LABELS,
    EntailmentCertificate,
    GoldForm,
)
from lofi.lf import Abstain

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi.data.synthetic import EntailmentLabel
    from lofi.lf import Verdict

PROOF: Final = proved("no countermodel among 4 assignments to 2 atoms")
COUNTERMODEL: Final = refuted("rain=true, wet=false")
EXHAUSTION: Final = Abstain("the wall clock of 1.0 seconds ran out after 3 assignments")
DIGEST: Final = "d" * 64


def _certificate(
    verdict: Verdict = PROOF,
    *,
    prover: str = "truth-table",
    profile_digest: str = DIGEST,
    timeout: float = 1.5,
) -> EntailmentCertificate:
    return EntailmentCertificate(
        verdict, prover=prover, profile_digest=profile_digest, timeout=timeout
    )


def test_the_labels_are_listed_in_report_order() -> None:
    assert ENTAILMENT_LABELS == ("entailment", "non-entailment", "undecided")


@pytest.mark.parametrize(
    ("verdict", "label"),
    [
        (PROOF, "entailment"),
        (COUNTERMODEL, "non-entailment"),
        (EXHAUSTION, "undecided"),
    ],
)
def test_a_certificate_is_labeled_by_the_kind_of_its_verdict(
    verdict: Verdict, label: EntailmentLabel
) -> None:
    certificate = _certificate(verdict)

    assert certificate.label == label


def test_a_gold_form_names_its_item_and_its_text() -> None:
    form = GoldForm(ItemId("x:1"), "sleep(john)")

    assert form.item_id == "x:1"
    assert form.text == "sleep(john)"
    assert form == GoldForm(ItemId("x:1"), "sleep(john)")
    assert hash(form) == hash(GoldForm(ItemId("x:1"), "sleep(john)"))
    assert form != GoldForm(ItemId("x:2"), "sleep(john)")
    assert form != GoldForm(ItemId("x:1"), "run(john)")
    assert form.__eq__("sleep(john)") is NotImplemented
    assert repr(form) == "GoldForm('x:1', 'sleep(john)')"


def test_a_gold_form_refuses_an_empty_text() -> None:
    with pytest.raises(ValueError, match="logical form of 'x:1'"):
        GoldForm(ItemId("x:1"), "")


def test_a_certificate_holds_the_verdict_and_the_proof_system() -> None:
    certificate = _certificate(COUNTERMODEL)

    assert certificate.verdict is COUNTERMODEL
    assert certificate.prover == "truth-table"
    assert certificate.profile_digest == DIGEST
    assert certificate.timeout == 1.5
    assert certificate == _certificate(COUNTERMODEL)
    assert hash(certificate) == hash(_certificate(COUNTERMODEL))
    assert certificate != _certificate(PROOF)
    assert certificate != _certificate(refuted("rain=false, wet=false"))
    assert certificate != _certificate(COUNTERMODEL, prover="z3")
    assert certificate != _certificate(COUNTERMODEL, profile_digest="e" * 64)
    assert certificate != _certificate(COUNTERMODEL, timeout=2.0)
    assert certificate.__eq__(COUNTERMODEL) is NotImplemented
    assert repr(certificate) == (
        f"EntailmentCertificate({COUNTERMODEL!r}, prover='truth-table', "
        f"profile_digest={DIGEST!r}, timeout=1.5)"
    )


@pytest.mark.parametrize(
    ("build", "error", "fault"),
    [
        (lambda: _certificate(prover=""), ValueError, "prover"),
        (lambda: _certificate(profile_digest=""), ValueError, "profile digest"),
        (lambda: _certificate(timeout=0.0), ValueError, "positive number of seconds"),
        (lambda: _certificate(timeout=-1.0), ValueError, "positive number of seconds"),
        (lambda: _certificate(timeout=inf), ValueError, "timeout"),
        (lambda: _certificate(timeout=nan), ValueError, "timeout"),
        (lambda: _certificate(timeout=cast("float", "1")), TypeError, "timeout"),
    ],
)
def test_a_certificate_refuses_an_empty_prover_or_digest_and_a_bad_clock(
    build: Callable[[], EntailmentCertificate], error: type[Exception], fault: str
) -> None:
    with pytest.raises(error, match=fault):
        build()

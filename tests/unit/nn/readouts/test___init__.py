from lofi.nn import readouts
from lofi.nn.readouts import ProofTermMeaning, ProofTermPushforward


def test_the_package_publishes_its_proof_term_capabilities() -> None:
    assert readouts.__all__ == ["ProofTermMeaning", "ProofTermPushforward"]
    assert readouts.ProofTermMeaning is ProofTermMeaning
    assert readouts.ProofTermPushforward is ProofTermPushforward

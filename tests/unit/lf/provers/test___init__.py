from lofi.lf import provers
from lofi.lf.provers import SmtOracle, TptpOracle, tptp_problem


def test_optional_prover_package_exports_both_oracles() -> None:
    assert provers.SmtOracle is SmtOracle
    assert provers.TptpOracle is TptpOracle
    assert provers.tptp_problem is tptp_problem

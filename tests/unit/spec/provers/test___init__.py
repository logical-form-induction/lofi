from lofi.spec import provers
from lofi.spec.provers import SmtProverSpec, TptpProverSpec


def test_prover_specs_are_exported() -> None:
    assert provers.SmtProverSpec is SmtProverSpec
    assert provers.TptpProverSpec is TptpProverSpec

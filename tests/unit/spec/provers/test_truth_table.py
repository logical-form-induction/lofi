import didactic.api as dx
import pytest

from lofi.spec.provers import TruthTableProverSpec


def test_truth_table_spec_owns_its_atom_cap() -> None:
    declared = TruthTableProverSpec(atoms=7)

    assert declared.family() == "prover"
    settings = declared.settings()
    assert settings.atoms == 7
    assert settings.service.timeout == 30.0

    with pytest.raises(dx.ValidationError, match="atom cap"):
        TruthTableProverSpec(atoms=0)

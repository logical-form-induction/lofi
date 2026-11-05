import didactic.api as dx
import pytest

from lofi.spec import Dependency
from lofi.spec.components.loss import LossSpec
from lofi.spec.losses.form_supervision import FormSupervisionLossSpec


def test_the_form_supervision_variant_declares_its_term_readout() -> None:
    declared = LossSpec.model_validate(
        {"kind": "form-supervision", "readout": "logical-forms"}
    )

    assert isinstance(declared, FormSupervisionLossSpec)
    assert declared == FormSupervisionLossSpec(readout="logical-forms")
    assert declared.dependencies() == (
        Dependency("logical-forms", "form-supervision-readout"),
    )
    assert declared.settings().readout == "logical-forms"


def test_a_form_supervision_readout_is_named() -> None:
    with pytest.raises(dx.ValidationError, match="readout"):
        FormSupervisionLossSpec(readout="")

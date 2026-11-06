import didactic.api as dx
import pytest

from lofi.spec import Dependency
from lofi.spec.components.loss import LossSpec
from lofi.spec.losses.form_typing import FormTypingLossSpec


def test_the_form_typing_variant_declares_its_term_readout() -> None:
    declared = LossSpec.model_validate(
        {"kind": "form-typing", "readout": "logical-forms"}
    )

    assert isinstance(declared, FormTypingLossSpec)
    assert declared == FormTypingLossSpec(readout="logical-forms")
    assert declared.dependencies() == (
        Dependency("logical-forms", "form-typing-readout"),
    )
    assert declared.settings().readout == "logical-forms"


@pytest.mark.parametrize("readout", [""])
def test_a_form_typing_readout_is_named(readout: str) -> None:
    with pytest.raises(dx.ValidationError, match="readout"):
        FormTypingLossSpec(readout=readout)


def test_no_placeholder_probability_or_regime_field_is_accepted() -> None:
    for field in ("probability", "regime", "negative_weight"):
        with pytest.raises(dx.ValidationError, match=field):
            LossSpec.model_validate(
                {"kind": "form-typing", "readout": "forms", field: 1.0}
            )

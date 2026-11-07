import didactic.api as dx
import pytest

from lofi.spec import FeaturizerSpec
from lofi.spec.featurizers.derivation import DerivationFeaturizerSpec


def test_derivation_declaration_names_its_padding() -> None:
    spec = FeaturizerSpec.model_validate({"kind": "derivation", "padding": "[PAD]"})

    assert isinstance(spec, DerivationFeaturizerSpec)
    assert spec.padding == "[PAD]"


def test_derivation_declaration_requires_a_padding_symbol() -> None:
    with pytest.raises(dx.ValidationError, match="is empty"):
        DerivationFeaturizerSpec.model_validate({"padding": ""})

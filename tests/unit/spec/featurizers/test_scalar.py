import didactic.api as dx
import pytest

from lofi.spec import FeaturizerSpec
from lofi.spec.featurizers.scalar import ScalarFeaturizerSpec


def test_scalar_declaration_names_its_padding() -> None:
    spec = FeaturizerSpec.model_validate({"kind": "scalar", "padding": "[PAD]"})

    assert isinstance(spec, ScalarFeaturizerSpec)
    assert spec.padding == "[PAD]"


def test_scalar_declaration_requires_a_padding_symbol() -> None:
    with pytest.raises(dx.ValidationError, match="is empty"):
        ScalarFeaturizerSpec.model_validate({"padding": ""})

import didactic.api as dx
import pytest

from lofi.spec import FeaturizerSpec
from lofi.spec.featurizers.unlabeled import UnlabeledFeaturizerSpec


def test_unlabeled_declaration_names_its_padding() -> None:
    spec = FeaturizerSpec.model_validate({"kind": "unlabeled", "padding": "[PAD]"})

    assert isinstance(spec, UnlabeledFeaturizerSpec)
    assert spec.padding == "[PAD]"


def test_unlabeled_declaration_requires_a_padding_symbol() -> None:
    with pytest.raises(dx.ValidationError, match="is empty"):
        UnlabeledFeaturizerSpec.model_validate({"padding": ""})

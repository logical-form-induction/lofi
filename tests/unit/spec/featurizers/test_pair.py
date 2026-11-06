import didactic.api as dx
import pytest

from lofi.spec import FeaturizerSpec
from lofi.spec.featurizers.pair import PairFeaturizerSpec


def test_pair_declaration_names_its_padding_and_arm() -> None:
    spec = FeaturizerSpec.model_validate(
        {
            "kind": "pair",
            "padding": "[PAD]",
            "selection": "relational-scalar",
        }
    )

    assert isinstance(spec, PairFeaturizerSpec)
    assert spec.padding == "[PAD]"
    assert spec.selection == "relational-scalar"


def test_pair_declaration_requires_a_padding_symbol() -> None:
    with pytest.raises(dx.ValidationError, match="is empty"):
        PairFeaturizerSpec.model_validate({"padding": ""})


def test_pair_selection_names_are_open_to_installed_plugins() -> None:
    assert PairFeaturizerSpec.model_validate({"selection": "scalar"}).selection == (
        "scalar"
    )

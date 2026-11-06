import didactic.api as dx
import pytest

from lofi.spec import FeaturizerSpec
from lofi.spec.featurizers.span_constraint import SpanConstraintFeaturizerSpec


def test_span_constraint_declaration_names_its_padding() -> None:
    spec = FeaturizerSpec.model_validate(
        {"kind": "span-constraint", "padding": "[PAD]"}
    )

    assert isinstance(spec, SpanConstraintFeaturizerSpec)
    assert spec.padding == "[PAD]"


def test_span_constraint_declaration_requires_a_padding_symbol() -> None:
    with pytest.raises(dx.ValidationError, match="is empty"):
        SpanConstraintFeaturizerSpec.model_validate({"padding": ""})

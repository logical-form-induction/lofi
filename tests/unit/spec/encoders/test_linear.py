from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.spec.components.model import TermEncoderSpec
from lofi.spec.encoders.linear import LinearRnnEncoderSettings, LinearRnnEncoderSpec

if TYPE_CHECKING:
    from lofi._types import JsonValue


def test_the_linear_recurrence_is_one_registered_encoder_variant() -> None:
    declared = TermEncoderSpec.model_validate(
        {
            "kind": "linear-rnn",
            "width": 12,
            "cell": "lstm",
            "layers": 3,
            "direction": "bidirectional",
            "merge": "concatenation",
            "variables": "level",
        }
    )

    assert type(declared) is LinearRnnEncoderSpec
    assert declared.settings() == LinearRnnEncoderSettings(
        12, "lstm", 3, "bidirectional", "level", "concatenation"
    )
    assert declared.publications() == frozenset({"choice-embedding"})


def test_the_cell_is_any_installed_recurrent_cell_name() -> None:
    declared = LinearRnnEncoderSpec(width=8, cell="test-package-cell")

    assert declared.settings().cell == "test-package-cell"


def test_the_linear_encoder_defaults_are_a_forward_index_recurrence() -> None:
    declared = LinearRnnEncoderSpec(width=8)

    assert declared.settings() == LinearRnnEncoderSettings(
        8, "gru", 1, "forward", "index", None
    )


@pytest.mark.parametrize(
    ("changed", "detail"),
    [
        ({"width": 0}, "state of no coordinates"),
        ({"layers": 0}, "recurrence has a layer"),
        ({"cell": ""}, "recurrent cell"),
        ({"direction": "bidirectional"}, "merge"),
        ({"merge": "mean"}, "merge"),
    ],
)
def test_invalid_linear_encoder_dimensions_are_refused(
    changed: dict[str, JsonValue], detail: str
) -> None:
    with pytest.raises(dx.ValidationError, match=detail):
        LinearRnnEncoderSpec.model_validate({"width": 8, **changed})

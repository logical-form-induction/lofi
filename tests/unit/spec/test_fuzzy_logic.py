import didactic.api as dx
import pytest

from lofi.spec.fuzzy_logic import FuzzyLogicSettings, FuzzyLogicSpec


def test_a_fuzzy_logic_declaration_resolves_immutable_interior_settings() -> None:
    settings = FuzzyLogicSpec(
        tnorm="lukasiewicz",
        implication="residuum",
        quantifier="extreme",
        equality="margin",
        margin=0.2,
    ).settings()

    assert settings == FuzzyLogicSettings(
        "lukasiewicz",
        "residuum",
        "extreme",
        "margin",
        0.2,
    )


@pytest.mark.parametrize("margin", [-0.1, float("nan"), float("inf")])
def test_a_fuzzy_logic_declaration_refuses_an_invalid_margin(margin: float) -> None:
    with pytest.raises(dx.ValidationError):
        FuzzyLogicSpec(
            tnorm="product",
            implication="residuum",
            quantifier="extreme",
            equality="crisp",
            margin=margin,
        )


def test_a_fuzzy_logic_declaration_refuses_a_t_norm_no_plugin_installs() -> None:
    with pytest.raises(dx.ValidationError, match="installed t-norms are"):
        FuzzyLogicSpec(
            tnorm="hamacher",
            implication="residuum",
            quantifier="extreme",
            equality="crisp",
            margin=0.0,
        )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("implication", "installed implications are"),
        ("quantifier", "installed quantifiers are"),
        ("equality", "installed equalitys are"),
    ],
)
def test_a_fuzzy_logic_declaration_refuses_an_operation_no_plugin_installs(
    field: str, message: str
) -> None:
    declared = {
        "tnorm": "product",
        "implication": "residuum",
        "quantifier": "extreme",
        "equality": "crisp",
        "margin": 0.0,
    }
    declared[field] = "uninstalled"

    with pytest.raises(dx.ValidationError, match=message):
        FuzzyLogicSpec.model_validate(declared)

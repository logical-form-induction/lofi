import didactic.api as dx
import pytest

from lofi.spec.groundings._policies import (
    BestCompositionSpec,
    CompositionRuleSpec,
    MarginalCompositionSpec,
    TemperedCompositionSpec,
)


def test_each_composition_rule_is_reached_by_its_tag_and_no_other() -> None:
    marginal = CompositionRuleSpec.model_validate({"rule": "marginal"})
    best = CompositionRuleSpec.model_validate({"rule": "best"})
    tempered = CompositionRuleSpec.model_validate(
        {"rule": "tempered", "temperature": 0.5}
    )

    assert marginal == MarginalCompositionSpec()
    assert best == BestCompositionSpec()
    assert tempered == TemperedCompositionSpec(temperature=0.5)
    assert isinstance(marginal, MarginalCompositionSpec)
    assert isinstance(best, BestCompositionSpec)
    assert isinstance(tempered, TemperedCompositionSpec)
    assert len({type(marginal), type(best), type(tempered)}) == 3
    assert marginal.settings() == ("marginal", None)
    assert best.settings() == ("best", None)
    assert tempered.settings() == ("tempered", 0.5)


def test_an_unknown_rule_is_refused_with_the_rule_named() -> None:
    with pytest.raises(dx.ValidationError, match="'annealed'"):
        CompositionRuleSpec.model_validate({"rule": "annealed"})


def test_a_tempered_rule_needs_a_positive_finite_temperature() -> None:
    with pytest.raises(dx.ValidationError, match=r"positive, not 0\.0"):
        TemperedCompositionSpec(temperature=0.0)

    with pytest.raises(dx.ValidationError, match=r"positive, not -1\.0"):
        TemperedCompositionSpec(temperature=-1.0)

    with pytest.raises(dx.ValidationError, match="is not finite: nan"):
        TemperedCompositionSpec(temperature=float("nan"))

    with pytest.raises(dx.ValidationError, match="required"):
        CompositionRuleSpec.model_validate({"rule": "tempered"})


def test_a_rule_takes_no_setting_it_does_not_declare() -> None:
    with pytest.raises(dx.ValidationError):
        CompositionRuleSpec.model_validate({"rule": "marginal", "temperature": 1.0})


@pytest.mark.parametrize(
    "document",
    [
        '{"rule": "marginal"}',
        '{"rule": "best"}',
        '{"rule": "tempered", "temperature": 2.5}',
    ],
)
def test_each_rule_round_trips_through_its_json_form_byte_for_byte(
    document: str,
) -> None:
    rule = CompositionRuleSpec.model_validate_json(document)

    assert rule.model_dump_json() == document
    assert CompositionRuleSpec.model_validate_json(rule.model_dump_json()) == rule

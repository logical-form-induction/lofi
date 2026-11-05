import didactic.api as dx
import pytest

from lofi.spec.provers.smt import SmtProverSpec


def test_smt_spec_resolves_every_worker_setting() -> None:
    declared = SmtProverSpec(engine="cvc5", executable="/solver")

    assert declared.family() == "prover"
    settings = declared.settings()
    assert settings.engine == "cvc5"
    assert settings.executable == "/solver"
    assert settings.logic == "ALL"
    assert settings.options == {}
    assert settings.arguments == ()
    assert SmtProverSpec().settings().engine == "z3"


def test_smt_logic_options_and_common_service_settings_are_declared() -> None:
    declared = SmtProverSpec(
        engine="third-party",
        logic="QF_UF",
        options={"produce-models": "true"},
        arguments=("--incremental",),
        certificates=("model",),
        polymorphism="erase",
    )
    settings = declared.settings()

    assert settings.engine == "third-party"
    assert settings.logic == "QF_UF"
    assert settings.options == {"produce-models": "true"}
    assert settings.arguments == ("--incremental",)
    assert settings.service.certificates == {"model"}
    assert settings.service.polymorphism == "erase"


def test_a_prover_declaration_does_not_accept_an_engine_capability_override() -> None:
    assert "serves" not in SmtProverSpec.__field_specs__


def test_an_smt_prefix_defaults_to_the_engine_and_may_be_replaced() -> None:
    assert SmtProverSpec().settings().prefix is None
    assert SmtProverSpec(prefix=("-in",)).settings().prefix == ("-in",)


def test_an_smt_prefix_refuses_an_empty_argument() -> None:
    with pytest.raises(dx.ValidationError):
        SmtProverSpec(prefix=("",))


def test_an_smt_terminate_grace_period_is_declared() -> None:
    assert SmtProverSpec().settings().terminate_grace == 0.2
    assert SmtProverSpec(terminate_grace=1.5).settings().terminate_grace == 1.5


@pytest.mark.parametrize("grace", [0.0, -1.0, float("inf")])
def test_an_smt_terminate_grace_period_is_finite_and_positive(grace: float) -> None:
    with pytest.raises(dx.ValidationError):
        SmtProverSpec(terminate_grace=grace)

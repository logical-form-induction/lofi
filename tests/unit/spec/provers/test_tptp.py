import didactic.api as dx
import pytest

from lofi.spec.provers.tptp import TptpProverSpec


def test_tptp_spec_resolves_every_process_setting() -> None:
    declared = TptpProverSpec(
        engine="vampire",
        executable="/prover",
        dialect="tff",
        timeout=2.0,
        workers=3,
        batch_size=16,
    )
    assert declared.timeout == 2.0
    assert declared.workers == 3
    assert declared.batch_size == 16

    assert declared.family() == "prover"
    settings = declared.settings()
    assert settings.engine == "vampire"
    assert settings.executable == "/prover"
    assert settings.dialect == "tff"
    assert settings.options == ()
    assert settings.service.timeout == 2.0
    assert settings.service.workers == 3
    assert settings.service.batch_size == 16
    assert TptpProverSpec().settings().dialect == "fof"


def test_tptp_engine_and_process_options_are_open_declarations() -> None:
    declared = TptpProverSpec(
        engine="third-party",
        dialect="thf",
        options=("--strategy=language",),
        memory_mb=2048,
        cpu_seconds=17,
        seed=11,
        polymorphism="monomorphize",
        certificates=("proof",),
    )
    settings = declared.settings()

    assert settings.engine == "third-party"
    assert settings.options == ("--strategy=language",)
    assert settings.service.memory_mb == 2048
    assert settings.service.cpu_seconds == 17
    assert settings.service.seed == 11
    assert settings.service.polymorphism == "monomorphize"
    assert settings.service.certificates == {"proof"}
    assert "serves" not in TptpProverSpec.__field_specs__


def test_dialect_compatibility_belongs_to_the_registered_engine() -> None:
    assert TptpProverSpec(engine="vampire", dialect="thf").settings().dialect == "thf"
    assert TptpProverSpec(engine="leo3", dialect="thf").settings().engine == "leo3"


@pytest.mark.parametrize(
    ("field", "value"),
    [("timeout", 0.0), ("timeout", float("inf")), ("workers", 0), ("batch_size", 0)],
)
def test_a_prover_refuses_an_invalid_service_setting(field: str, value: float) -> None:
    with pytest.raises(dx.ValidationError):
        TptpProverSpec.model_validate({field: value})


def test_a_tptp_prefix_defaults_to_the_engine_and_may_be_replaced() -> None:
    assert TptpProverSpec().settings().prefix is None
    assert TptpProverSpec(prefix=()).settings().prefix == ()
    assert TptpProverSpec(prefix=("--mode", "portfolio")).settings().prefix == (
        "--mode",
        "portfolio",
    )


def test_a_tptp_prefix_refuses_an_empty_argument() -> None:
    with pytest.raises(dx.ValidationError):
        TptpProverSpec(prefix=("",))


def test_a_tptp_problem_file_name_is_declared() -> None:
    assert TptpProverSpec().settings().problem_file == "problem.p"
    assert TptpProverSpec(problem_file="query.p").settings().problem_file == "query.p"


@pytest.mark.parametrize("name", ["", "nested/problem.p", "..", "."])
def test_a_tptp_problem_file_is_one_plain_file_name(name: str) -> None:
    with pytest.raises(dx.ValidationError):
        TptpProverSpec(problem_file=name)

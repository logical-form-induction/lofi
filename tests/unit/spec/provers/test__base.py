import didactic.api as dx
import pytest

from lofi.spec.provers import TruthTableProverSpec


def test_common_prover_resources_lower_to_one_immutable_service_setting() -> None:
    declared = TruthTableProverSpec(
        timeout=2.5,
        workers=3,
        batch_size=8,
        cache_size=21,
        memory_mb=512,
        cpu_seconds=7,
        seed=11,
        polymorphism="monomorphize",
        certificates=("proof", "model"),
    )
    settings = declared.settings().service

    assert settings.timeout == 2.5
    assert settings.workers == 3
    assert settings.batch_size == 8
    assert settings.cache_size == 21
    assert settings.memory_mb == 512
    assert settings.cpu_seconds == 7
    assert settings.seed == 11
    assert settings.polymorphism == "monomorphize"
    assert settings.certificates == {"proof", "model"}


@pytest.mark.parametrize("field", ["cache_size", "memory_mb", "cpu_seconds"])
def test_common_prover_resources_refuse_nonpositive_limits(field: str) -> None:
    with pytest.raises(dx.ValidationError):
        TruthTableProverSpec.model_validate({field: 0})


def test_common_prover_resources_refuse_duplicate_certificates() -> None:
    with pytest.raises(dx.ValidationError, match="certificate twice"):
        TruthTableProverSpec(certificates=("proof", "proof"))


def test_a_prover_seed_is_nonnegative() -> None:
    with pytest.raises(dx.ValidationError, match="seed"):
        TruthTableProverSpec(seed=-1)

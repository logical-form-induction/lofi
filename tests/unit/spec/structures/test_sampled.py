import didactic.api as dx
import pytest

from lofi.lf.models import SampledSettings
from lofi.spec.structures import SampledStructureSourceSpec


def declaration() -> SampledStructureSourceSpec:
    return SampledStructureSourceSpec(
        structures=12,
        carriers={"entity": 3},
        density=0.4,
        full_function_space_cap=64,
        henkin_pool=8,
        seed=17,
        resample="per-step",
        atomic_sampling="uniform",
        denotation_streams="independent",
        function_sampling="table",
        truth_sampling="bernoulli",
    )


def test_a_sampled_source_declaration_discards_its_boundary_model() -> None:
    resolved = declaration().settings()

    assert isinstance(resolved, SampledSettings)
    assert resolved.structures == 12
    assert resolved.carriers == {"entity": 3}
    assert resolved.resample == "per-step"
    assert resolved.atomic_sampling == "uniform"
    assert resolved.denotation_streams == "independent"
    assert resolved.function_sampling == "table"
    assert resolved.truth_sampling == "bernoulli"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("structures", 0),
        ("carriers", {}),
        ("density", 1.1),
        ("full_function_space_cap", 0),
        ("henkin_pool", 0),
        ("seed", -1),
    ],
)
def test_a_sampled_source_refuses_an_invalid_boundary_value(
    field: str, value: float | dict[str, int]
) -> None:
    with pytest.raises(dx.ValidationError):
        declaration().with_(**{field: value})

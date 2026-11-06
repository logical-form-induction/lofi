import didactic.api as dx
import pytest

from lofi.spec._ports import Dependency
from lofi.spec.evaluators.grammaticality import GrammaticalityEvaluatorSpec

CONVENTIONS = {
    "punctuation": "keep",
    "unary_chains": "keep",
    "trivial_spans": "keep",
    "minimum_length": 0,
    "aggregation": "sentence",
    "coverage": "zero",
    "self_arcs": "allow",
    "gold_derivation": "canonical",
}


def test_grammaticality_declares_its_readout_and_full_battery() -> None:
    spec = GrammaticalityEvaluatorSpec.model_validate(
        {
            "kind": "grammaticality",
            "interval": {"replicates": 20, "level": 0.95},
            "readout": "root-mass",
            "output": "sentences",
            "flip_rate": 0.1,
            "bins": 5,
            "conventions": CONVENTIONS,
        }
    )

    assert spec.dependencies() == (Dependency("root-mass", "span-readout"),)
    assert spec.settings().metrics == spec.metrics()
    assert len(spec.metrics()) == 9


@pytest.mark.parametrize("flip_rate", [-0.1, 0.5])
def test_grammaticality_refuses_uninformative_flip_rates(flip_rate: float) -> None:
    with pytest.raises(dx.ValidationError, match="flip"):
        GrammaticalityEvaluatorSpec.model_validate(
            {
                "kind": "grammaticality",
                "interval": {"replicates": 20, "level": 0.95},
                "readout": "root-mass",
                "output": "sentences",
                "flip_rate": flip_rate,
                "bins": 5,
                "conventions": CONVENTIONS,
            }
        )

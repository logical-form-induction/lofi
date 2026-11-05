from pathlib import Path

import didactic.api as dx
import pytest

from lofi.spec import (
    DatasetSpec,
    RunContext,
    SpecLoader,
    register_resolvers,
    run_context,
)
from lofi.spec._interpolation import lower_package_resource
from lofi.spec.datasets.expert import ExpertRatingsDatasetSpec


def test_expert_ratings_are_a_registered_dataset_declaration() -> None:
    declared = DatasetSpec.model_validate(
        {"kind": "expert-ratings", "source": "ratings.json"}
    )

    assert type(declared) is ExpertRatingsDatasetSpec
    assert declared.source == "ratings.json"
    assert DatasetSpec.model_validate_json(declared.model_dump_json()) == declared


def test_an_empty_expert_ratings_source_is_refused_at_its_location() -> None:
    with pytest.raises(dx.ValidationError) as raised:
        DatasetSpec.model_validate({"kind": "expert-ratings", "source": ""})

    assert raised.value.entries[0].loc == ("source",)


def test_the_packaged_ratings_option_composes_beneath_the_data_root(
    tmp_path: Path,
) -> None:
    register_resolvers(replace=True)
    path = Path(
        lower_package_resource(
            "lofi.resource:evaluators/expert/ratings/expert-ratings.yaml"
        )
    )

    with run_context(RunContext(tmp_path, 31)):
        declared = (
            SpecLoader().compose_traced(path, schema=ExpertRatingsDatasetSpec).spec
        )

    assert declared.source == str(tmp_path / "expert-ratings.json")

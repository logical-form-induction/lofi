import json
from typing import TYPE_CHECKING

import pytest

from lofi.data.datasets import artifact_dataset_plugin, dataset_plugin
from lofi.data.expert import ExpertRating, ExpertRatings
from lofi.registry import RegistryError
from lofi.spec.datasets.expert import ExpertRatingsDatasetSpec

if TYPE_CHECKING:
    from pathlib import Path


def _document() -> dict[str, object]:
    return {
        "kind": "expert-ratings",
        "package_digest": "a" * 64,
        "ratings": [
            {
                "category_id": "I-A",
                "annotator": "rater-a",
                "rating": "correct",
                "errors": [],
            }
        ],
    }


def test_the_artifact_dataset_plugin_loads_expert_ratings_once(
    tmp_path: Path,
) -> None:
    source = tmp_path / "ratings.json"
    source.write_text(json.dumps(_document()), encoding="utf-8", newline="\n")
    declaration = ExpertRatingsDatasetSpec(source=str(source))
    loaded = artifact_dataset_plugin(declaration).load(declaration)

    assert isinstance(loaded, ExpertRatings)
    assert loaded.package_digest == "a" * 64
    assert loaded.ratings == (ExpertRating("I-A", "rater-a", "correct", ()),)


def test_the_training_dataset_resolver_refuses_an_artifact_plugin(
    tmp_path: Path,
) -> None:
    source = tmp_path / "ratings.json"
    source.write_text(json.dumps(_document()), encoding="utf-8", newline="\n")
    ratings = ExpertRatingsDatasetSpec(source=str(source))

    with pytest.raises(RegistryError, match="not name a training dataset plugin"):
        dataset_plugin(ratings)

import json
from typing import TYPE_CHECKING

import pytest
from tests.unit.run.conftest import a_point
from tests.unit.run.test__experiment import an_experiment_document, write_experiment
from tests.unit.spec.acquisition.test_active import (
    a_document as an_active_learning_document,
)

from lofi.run import (
    ExperimentSeed,
    ExperimentStore,
    ManifestStatus,
    encode_seed_derivation,
    read_experiment,
)
from lofi.run._active_learning_configuration import ActiveLearningConfiguration
from lofi.spec.acquisition import decode_active_learning

if TYPE_CHECKING:
    from pathlib import Path

    from lofi._types import JsonValue


def _active_manifest(**changed: JsonValue) -> bytes:
    document: dict[str, JsonValue] = {
        "document": json.loads(an_active_learning_document()),
        "status": "completed",
        "rounds": 2,
        "awaiting": None,
        "stop_reason": "max-rounds",
        "initial_judgments": 24,
        "initial_skipped": 0,
        "source_digests": {},
        "seed": 3,
        "seed_derivation": json.loads(
            encode_seed_derivation(
                ExperimentSeed(7).point("0123456789abcdef", 3)
            ).model_dump_json()
        ),
        "package_versions": {},
        "git_revision": "0" * 40,
    }
    document.update(changed)
    return json.dumps(document).encode()


def test_an_active_learning_configuration_reports_its_rounds_under_one_role(
    tmp_path: Path,
) -> None:
    configuration = ActiveLearningConfiguration(
        "active-learning", tmp_path / "active.json"
    )

    assert configuration.kind == "active-learning"
    assert configuration.manifest_name == "active-learning.json"
    assert not configuration.enumerates
    assert configuration.evaluators((tmp_path,), (), 0) == {
        "active-learning": frozenset(
            {"agreement", "rank-correlation", "batch-variance", "stratum-share"}
        )
    }
    assert configuration.roles({}) == frozenset({"active-learning"})


def test_an_active_learning_manifest_records_a_completed_or_awaiting_point(
    tmp_path: Path,
) -> None:
    configuration = ActiveLearningConfiguration(
        "active-learning", tmp_path / "active.json"
    )

    completed = configuration.manifest_point(_active_manifest())
    awaiting = configuration.manifest_point(
        _active_manifest(status="awaiting-responses", awaiting=3, stop_reason=None)
    )

    assert completed.status is ManifestStatus.COMPLETED
    assert awaiting.status is ManifestStatus.KILLED
    assert awaiting.awaiting_responses
    assert not completed.awaiting_responses
    assert completed.seed.point_id == "0123456789abcdef"
    assert decode_active_learning(
        json.dumps(completed.config.document).encode()
    ) == decode_active_learning(an_active_learning_document())

    with pytest.raises(ValueError, match="no seed derivation"):
        configuration.manifest_point(_active_manifest(seed_derivation=None))


def test_an_active_learning_run_executes_in_process_untimed_and_unfrozen(
    tmp_path: Path,
) -> None:
    configuration = ActiveLearningConfiguration(
        "active-learning", tmp_path / "active.json"
    )
    store = ExperimentStore(
        tmp_path, read_experiment(write_experiment(tmp_path, an_experiment_document()))
    )
    rows: tuple[dict[str, JsonValue], ...] = ({"metric": "agreement", "value": 0.5},)

    assert configuration.in_process
    assert not configuration.timed
    assert configuration.identities((a_point(1),)) == {}
    assert configuration.result_rows(rows, (a_point(1),), store) == rows

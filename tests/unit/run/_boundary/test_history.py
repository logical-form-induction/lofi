import json
import threading
from hashlib import sha256
from typing import TYPE_CHECKING, Final, cast

import didactic.api as dx
import panproto
import pytest
from tests.unit.run._boundary.conftest import a_run_spec

from lofi.run import CommittedRun, TamperedRecordError
from lofi.run._boundary import history as history_module
from lofi.run._boundary.history import (
    CommittedLanguageRecord,
    CommittedRunIso,
    CommittedRunRecord,
    RunHistory,
    UncommittedRunError,
    _exclusive,
)
from lofi.spec import (
    ArchivedLanguage,
    BoundaryMismatchError,
    FingerprintIso,
    IndexedLanguage,
    RunFingerprint,
    fingerprint_run,
)

if TYPE_CHECKING:
    from pathlib import Path

    from lofi._types import JsonValue


HISTORY_ROUTE: Final = dx.LoweringRoute("decl-1", "native-1")


def history_language(specification: bytes = b"theory one") -> ArchivedLanguage:
    return ArchivedLanguage(
        b'{"name":"toy"}',
        IndexedLanguage(
            "decl-1->native-1",
            specification,
            sha256(specification).hexdigest(),
            HISTORY_ROUTE,
        ),
    )


def committed_run(
    point: str = "0123456789abcdef",
    *,
    configuration: bytes | None = None,
    specification: bytes = b"theory one",
) -> CommittedRun:
    language = history_language(specification)
    fingerprint = fingerprint_run({"grammar": language.theory})
    resolved = (
        json.dumps(json.loads(a_run_spec().model_dump_json()), indent=2) + "\n"
    ).encode()

    return CommittedRun(
        point,
        configuration=resolved if configuration is None else configuration,
        fingerprint=FingerprintIso().forward(fingerprint),
        languages={"grammar": language},
    )


def test_a_committed_run_round_trips_through_its_record() -> None:
    run = committed_run()
    record = CommittedRunIso().forward(run)

    assert record.point_id == run.point_id
    assert record.configuration == run.configuration.hex()
    assert record.configuration_identifier == sha256(run.configuration).hexdigest()
    assert record.fingerprint == run.fingerprint.hex()
    assert record.fingerprint_identifier == sha256(run.fingerprint).hexdigest()
    assert record.languages == {
        "grammar": CommittedLanguageRecord(
            declaration=b'{"name":"toy"}'.hex(),
            version="decl-1->native-1",
            specification=b"theory one".hex(),
            digest=sha256(b"theory one").hexdigest(),
            source_version="decl-1",
            target_version="native-1",
        )
    }
    assert CommittedRunIso().backward(record) == run
    assert (
        CommittedRunIso().backward(
            CommittedRunRecord.model_validate_json(record.model_dump_json())
        )
        == run
    )


def test_a_record_whose_identifier_misses_its_bytes_is_tampered() -> None:
    record = CommittedRunIso().forward(committed_run())
    forged = record.model_dump()
    forged["configuration"] = b'{\n  "trainer": {\n    "epochs": 9\n  }\n}\n'.hex()

    with pytest.raises(TamperedRecordError) as refused:
        CommittedRunIso().backward(CommittedRunRecord.model_validate(forged))

    assert refused.value.subject == "configuration"
    assert refused.value.recorded == record.configuration_identifier


def test_a_record_whose_fingerprint_identifier_misses_is_tampered() -> None:
    forged = CommittedRunIso().forward(committed_run()).model_dump()
    forged["fingerprint_identifier"] = "0" * 64

    with pytest.raises(TamperedRecordError) as refused:
        CommittedRunIso().backward(CommittedRunRecord.model_validate(forged))

    assert refused.value.subject == "fingerprint"


def test_a_record_whose_theory_digest_misses_its_specification_is_refused() -> None:
    forged = json.loads(CommittedRunIso().forward(committed_run()).model_dump_json())
    language = cast("dict[str, JsonValue]", forged["languages"])
    grammar = cast("dict[str, JsonValue]", language["grammar"])
    grammar["specification"] = b"theory two".hex()

    with pytest.raises(ValueError, match="digest"):
        CommittedRunIso().backward(CommittedRunRecord.model_validate(forged))


@pytest.mark.parametrize("spelled", ["7B7D", "7b7", "zz", "7b 7d"])
def test_a_record_spelled_other_than_lowercase_hex_is_refused(spelled: str) -> None:
    forged = CommittedRunIso().forward(committed_run()).model_dump()
    forged["configuration"] = spelled

    with pytest.raises(ValueError, match="hex"):
        CommittedRunIso().backward(CommittedRunRecord.model_validate(forged))


def test_a_fresh_history_holds_no_runs(tmp_path: Path) -> None:
    history = RunHistory(tmp_path / "runs")

    assert history.root == tmp_path / "runs"
    assert history.committed() == {}
    assert repr(history) == f"RunHistory({tmp_path / 'runs'!r})"

    with pytest.raises(UncommittedRunError) as refused:
        history.replay("0123456789abcdef")

    assert refused.value.point_id == "0123456789abcdef"


def test_a_committed_run_replays_byte_for_byte_from_its_tag(tmp_path: Path) -> None:
    history = RunHistory(tmp_path)
    run = committed_run()
    commit = history.commit(run)
    replayed = RunHistory(tmp_path).replay(run.point_id)

    assert history.committed() == {run.point_id: commit}
    assert replayed == run
    assert replayed.configuration == run.configuration
    assert replayed.fingerprint == run.fingerprint
    assert replayed.languages["grammar"].theory.specification == b"theory one"
    assert replayed.languages["grammar"].declaration == b'{"name":"toy"}'


def test_replay_decodes_records_once_at_the_full_tagged_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history = RunHistory(tmp_path)
    run = committed_run()
    commit = history.commit(run)
    refs: list[str] = []
    decoded_data_at = dx.Repository.decoded_data_at

    def spied(repository: dx.Repository, ref: str) -> list[dx.DecodedCommittedDataset]:
        refs.append(ref)
        return decoded_data_at(repository, ref)

    monkeypatch.setattr(dx.Repository, "decoded_data_at", spied)

    assert history.replay(run.point_id) == run
    assert refs == [commit]


def test_each_run_binds_configuration_and_metadata_to_their_persisted_schemas(
    tmp_path: Path,
) -> None:
    history = RunHistory(tmp_path)
    first = committed_run()
    second = committed_run("fedcba9876543210", specification=b"theory two")
    first_commit = history.commit(first)
    second_commit = history.commit(second)
    repository = dx.Repository.open(tmp_path)
    first_data = {dataset.key: dataset for dataset in repository.data_at(first_commit)}
    second_data = {
        dataset.key: dataset for dataset in repository.data_at(second_commit)
    }

    assert repository.resolve_ref(first.point_id) == first_commit
    assert repository.resolve_ref(second.point_id) == second_commit
    assert set(first_data) == {first.point_id, f"{first.point_id}:configuration"}
    assert set(second_data) == {second.point_id, f"{second.point_id}:configuration"}
    assert (
        first_data[f"{first.point_id}:configuration"].schema_id
        == second_data[f"{second.point_id}:configuration"].schema_id
    )
    assert (
        first_data[first.point_id].schema_id == second_data[second.point_id].schema_id
    )
    assert (
        first_data[first.point_id].schema_id
        != first_data[f"{first.point_id}:configuration"].schema_id
    )


def test_a_configuration_missing_a_current_required_field_is_not_committed(
    tmp_path: Path,
) -> None:
    run = committed_run()
    missing = json.loads(run.configuration)
    del missing["optimizer"]
    malformed = CommittedRun(
        run.point_id,
        configuration=(json.dumps(missing, indent=2) + "\n").encode(),
        fingerprint=run.fingerprint,
        languages=run.languages,
    )
    history = RunHistory(tmp_path)

    with pytest.raises(dx.ValidationError):
        history.commit(malformed)

    assert history.committed() == {}
    assert dx.Repository.open(tmp_path).head() is None


def test_a_run_from_another_boundary_is_refused_before_any_commit(
    tmp_path: Path,
) -> None:
    run = committed_run()
    other = RunFingerprint(
        {
            **run.boundary.schemas,
            "lofi.spec.components.run.RunSpec": "didactic://different-run-spec",
        },
        run.boundary.languages,
        run.boundary.packages,
    )
    refused = CommittedRun(
        run.point_id,
        configuration=run.configuration,
        fingerprint=FingerprintIso().forward(other),
        languages=run.languages,
    )
    history = RunHistory(tmp_path)

    with pytest.raises(BoundaryMismatchError) as mismatch:
        history.commit(refused)

    assert mismatch.value.members == ("schemas.lofi.spec.components.run.RunSpec",)
    assert history.committed() == {}
    assert dx.Repository.open(tmp_path).head() is None


def test_the_first_commit_stages_the_committed_run_record_schema(
    tmp_path: Path,
) -> None:
    history = RunHistory(tmp_path)
    history.commit(committed_run())
    history.commit(committed_run("fedcba9876543210"))
    rows = dx.Repository.open(tmp_path).log()

    assert [row["message"] for row in rows] == ["fedcba9876543210", "0123456789abcdef"]
    assert {row["protocol"] for row in rows} == {"CommittedRunRecord"}


def test_nominal_record_identity_survives_a_field_rename_in_blame(
    tmp_path: Path,
) -> None:
    original = history_module.CommittedRunRecord

    class CommittedRunRecord(dx.Model, extra="forbid"):
        run_id: str = dx.field(nominal=True)
        configuration: str
        configuration_identifier: str = dx.field(nominal=True)
        fingerprint: str
        fingerprint_identifier: str = dx.field(nominal=True)
        languages: dict[str, CommittedLanguageRecord]

    repository = dx.Repository.init(tmp_path)
    repository.add(original)
    introduced = repository.commit("introduce record", author="test")
    repository.add(CommittedRunRecord)
    renamed = repository.commit("rename point", author="test")

    assert (
        repository.blame_field(renamed, CommittedRunRecord, "run_id").commit_id
        == introduced
    )


def test_the_nominal_theory_version_survives_a_field_rename_in_blame(
    tmp_path: Path,
) -> None:
    original = history_module.CommittedLanguageRecord

    class CommittedLanguageRecord(dx.Model, extra="forbid"):
        declaration: str
        theory_version: str = dx.field(nominal=True)
        specification: str
        digest: str = dx.field(nominal=True)
        source_version: str = dx.field(nominal=True)
        target_version: str = dx.field(nominal=True)

    repository = dx.Repository.init(tmp_path)
    repository.add(original)
    introduced = repository.commit("introduce language", author="test")
    repository.add(CommittedLanguageRecord)
    renamed = repository.commit("rename theory version", author="test")

    assert (
        repository.blame_field(
            renamed,
            CommittedLanguageRecord,
            "theory_version",
        ).commit_id
        == introduced
    )


def test_each_point_replays_its_own_run_after_later_commits(tmp_path: Path) -> None:
    history = RunHistory(tmp_path)
    first = committed_run()
    second = committed_run("fedcba9876543210", specification=b"theory two")
    first_commit = history.commit(first)
    second_commit = history.commit(second)

    assert first_commit != second_commit
    assert history.committed() == {
        first.point_id: first_commit,
        second.point_id: second_commit,
    }
    assert history.replay(first.point_id) == first
    assert history.replay(second.point_id) == second


def test_committing_the_same_run_twice_adds_no_commit(tmp_path: Path) -> None:
    history = RunHistory(tmp_path)
    run = committed_run()
    commit = history.commit(run)

    assert history.commit(run) == commit
    assert len(dx.Repository.open(tmp_path).log()) == 1


def test_a_point_committed_with_another_record_is_refused(tmp_path: Path) -> None:
    history = RunHistory(tmp_path)
    original = committed_run()
    changed = json.loads(original.configuration)
    changed["device_policy"]["deterministic"] = False
    configuration = (json.dumps(changed, indent=2) + "\n").encode()
    history.commit(original)

    with pytest.raises(ValueError, match="different record"):
        history.commit(committed_run(configuration=configuration))

    assert history.replay("0123456789abcdef") == committed_run()
    assert len(dx.Repository.open(tmp_path).log()) == 1


def test_a_tag_on_a_commit_holding_no_dataset_for_its_point_is_uncommitted(
    tmp_path: Path,
) -> None:
    history = RunHistory(tmp_path)
    commit = history.commit(committed_run())
    dx.Repository.open(tmp_path).create_tag("ghost", commit)

    with pytest.raises(UncommittedRunError):
        history.replay("ghost")


def test_a_dataset_keyed_by_one_point_holding_another_is_refused(
    tmp_path: Path,
) -> None:
    history = RunHistory(tmp_path)
    history.commit(committed_run())
    repository = dx.Repository.open(tmp_path)
    staged = tmp_path / "forged.json"
    staged.write_text(
        CommittedRunIso().forward(committed_run()).model_dump_json(),
        encoding="utf-8",
        newline="\n",
    )
    repository.add_data(staged, key="fedcba9876543210")
    repository.create_tag(
        "fedcba9876543210", repository.commit("forged", author="test")
    )

    with pytest.raises(UncommittedRunError):
        history.replay("fedcba9876543210")


def test_a_stored_object_altered_on_disk_fails_replay(tmp_path: Path) -> None:
    history = RunHistory(tmp_path)
    run = committed_run()
    history.commit(run)
    marker = run.configuration.hex()[:8].encode()
    stored = next(
        path
        for directory in (tmp_path / ".panproto" / "objects").iterdir()
        for path in directory.iterdir()
        if path.is_file() and marker in path.read_bytes()
    )
    altered = bytearray(stored.read_bytes())
    at = altered.index(marker)
    altered[at] = ord("8") if altered[at] != ord("8") else ord("9")
    stored.write_bytes(bytes(altered))

    with pytest.raises(panproto.VcsError, match="corrupted"):
        history.replay(run.point_id)


def test_a_point_committed_with_another_declaration_is_refused(
    tmp_path: Path,
) -> None:
    history = RunHistory(tmp_path)
    run = committed_run()
    history.commit(run)
    redeclared = CommittedRun(
        run.point_id,
        configuration=run.configuration,
        fingerprint=run.fingerprint,
        languages={
            "grammar": ArchivedLanguage(b'{"name":"other"}', history_language().theory)
        },
    )

    with pytest.raises(ValueError, match="different record"):
        history.commit(redeclared)

    assert history.replay(run.point_id) == run


def test_a_point_whose_dataset_holds_two_records_is_uncommitted(
    tmp_path: Path,
) -> None:
    history = RunHistory(tmp_path)
    history.commit(committed_run())
    run = committed_run("fedcba9876543210")
    record = json.loads(CommittedRunIso().forward(run).model_dump_json())
    repository = dx.Repository.open(tmp_path)
    staged = tmp_path / "twice.json"
    staged.write_text(json.dumps([record, record]), encoding="utf-8", newline="\n")
    repository.add_data(staged, key=run.point_id)
    repository.create_tag(run.point_id, repository.commit("twice", author="test"))

    with pytest.raises(UncommittedRunError):
        history.replay(run.point_id)


def test_a_point_keyed_by_two_datasets_is_uncommitted(tmp_path: Path) -> None:
    history = RunHistory(tmp_path)
    history.commit(committed_run())
    run = committed_run("fedcba9876543210")
    repository = dx.Repository.open(tmp_path)
    staged = tmp_path / "once.json"
    staged.write_text(
        CommittedRunIso().forward(run).model_dump_json(), encoding="utf-8", newline="\n"
    )
    repository.add_data(staged, key=run.point_id)
    repository.add_data(staged, key=run.point_id)
    repository.create_tag(run.point_id, repository.commit("twice", author="test"))

    with pytest.raises(UncommittedRunError):
        history.replay(run.point_id)


def test_a_commit_waits_for_any_holder_of_the_history_lock(tmp_path: Path) -> None:
    history = RunHistory(tmp_path)
    run = committed_run()
    outcome: list[str] = []
    committer = threading.Thread(target=lambda: outcome.append(history.commit(run)))

    with _exclusive(tmp_path):
        committer.start()
        committer.join(timeout=1.0)

        assert committer.is_alive()
        assert history.committed() == {}

    committer.join(timeout=30.0)

    assert history.committed() == {run.point_id: outcome[0]}


def test_opening_a_history_waits_for_any_holder_of_its_lock(tmp_path: Path) -> None:
    opened: list[RunHistory] = []
    opener = threading.Thread(target=lambda: opened.append(RunHistory(tmp_path)))

    with _exclusive(tmp_path):
        opener.start()
        opener.join(timeout=1.0)

        assert opener.is_alive()
        assert not (tmp_path / ".panproto").exists()

    opener.join(timeout=30.0)

    assert opened[0].committed() == {}
    assert (tmp_path / ".panproto").is_dir()

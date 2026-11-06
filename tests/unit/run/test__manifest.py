import json
import platform
from hashlib import sha256
from importlib.metadata import version
from typing import TYPE_CHECKING, Final, Literal, cast

import didactic.api as dx
import pytest
import torch

from lofi.data import BatchShape
from lofi.run import (
    GRID_COST_MODEL,
    ExperimentManifest,
    ExperimentSeed,
    GridCell,
    GridTiming,
    HardwareDescriptor,
    ManifestStatus,
    close_manifest,
    open_manifest,
)
from lofi.run._boundary.manifest import (
    FailureKind,
    PointBinding,
    PointFailure,
    RecordedRows,
    encode_seed_derivation,
)
from lofi.run._manifest import (
    FAILURE_NAME,
    INTERPRETER,
    MANIFEST_NAME,
    POINT_NAME,
    PRICE_NAME,
    RECORDED_DISTRIBUTIONS,
    REPORTED_NAME,
    RESOLVED_NAME,
    SCHEMA_FINGERPRINT_NAME,
    SOURCES_NAME,
    RunRecord,
    StoredRecord,
    accelerator_runtime,
    installed_packages,
    read_binding,
    read_failure,
    read_manifest,
    read_price,
    read_record,
    read_rows,
    remove_failure,
    write_binding,
    write_failure,
    write_manifest,
    write_price,
    write_record,
    write_rows,
)
from lofi.run._values import (
    ExperimentPrice,
    MemoryBudget,
    StackPrice,
    StackTiming,
)
from lofi.spec import (
    ArchivedLanguage,
    ConfigSource,
    DatasetSpec,
    FeaturizerSpec,
    IndexedLanguage,
    LeafOrigin,
    LossSpec,
    OptimizerSpec,
    ParserSpec,
    Provenance,
    RunFingerprint,
    RunSpec,
    TrainerSpec,
)
from lofi.training_plugins import TrainingLength, TrainingUnit

if TYPE_CHECKING:
    from pathlib import Path

    from lofi._types import JsonValue


class RecordParser(ParserSpec):
    kind: Literal["record-unit-parser"] = "record-unit-parser"


class RecordLoss(LossSpec):
    kind: Literal["record-unit-loss"] = "record-unit-loss"


class RecordDataset(DatasetSpec):
    kind: Literal["record-unit-dataset"] = "record-unit-dataset"


class RecordFeaturizer(FeaturizerSpec):
    kind: Literal["record-unit-featurizer"] = "record-unit-featurizer"


class RecordTrainer(TrainerSpec):
    kind: Literal["record-unit-trainer"] = "record-unit-trainer"
    epochs: int = 1


class RecordOptimizer(OptimizerSpec):
    kind: Literal["record-unit-optimizer"] = "record-unit-optimizer"


PACKAGES: Final = {
    "didactic": "0.17.1",
    "panproto": "0.74.3",
    "quivers": "0.17.0",
    "torch": "2.14.0",
    "lairs": "1.0",
    "numpy": "1.0",
    "python": "1.0",
    "transformers": "1.0",
}

SEED: Final = ExperimentSeed(7).point("0123456789abcdef", 0)

SEED_POLICY: Final = SEED.policy(accelerator=None, samplers=())

GRAMMAR_SPEC: Final = (
    b'{"route":{"source":"decl-1","target":"native-1"},"theory":"S NP"}'
)


def a_document(epochs: int = 1) -> dict[str, JsonValue]:
    return {
        "model": {
            "language": {"from": "data"},
            "parser": {"kind": "record-unit-parser"},
        },
        "loss": {"terms": [{"objective": {"kind": "record-unit-loss"}}]},
        "data": {
            "datasets": {
                "train": {
                    "dataset": {"kind": "record-unit-dataset"},
                    "featurizer": {"kind": "record-unit-featurizer"},
                }
            }
        },
        "trainer": {"kind": "record-unit-trainer", "epochs": epochs},
        "parse": {"retain": "forest", "reachability": "all"},
        "release": {"checkpoint": "resume-latest"},
        "optimizer": {"kind": "record-unit-optimizer"},
        "device_policy": {
            "device": "cpu",
            "compute_dtype": "float32",
            "deterministic": True,
            "matmul_precision": "highest",
            "activation_memory_bytes": 8_000_000_000,
        },
    }


def a_manifest(
    epochs: int = 1, *, packages: dict[str, str] | None = None
) -> ExperimentManifest:
    return open_manifest(
        run_spec=RunSpec.model_validate(a_document(epochs)),
        component_fingerprints={
            role: f"fingerprint:{role}"
            for role in RunSpec.model_validate(a_document(epochs)).component_roles()
        },
        grammar_fingerprint="grammar:ccg",
        signature_fingerprint="signature:ccg",
        corpus_revision="ccgbank:LDC2005T13:auto-v1",
        seed_policy=SEED_POLICY,
        seed_derivation=SEED,
        package_versions=PACKAGES if packages is None else packages,
        git_revision="0123456789abcdef",
        accelerator="cpu",
    )


def a_provenance() -> Provenance:
    return Provenance(
        origins={
            "trainer.epochs": LeafOrigin(source=ConfigSource.FILE, layer="run.json"),
            "trainer.kind": LeafOrigin(source=ConfigSource.GROUP, layer="base.json"),
            "device_policy.device": LeafOrigin(
                source=ConfigSource.OVERRIDE, layer="device_policy.device=cpu"
            ),
        }
    )


def a_fingerprint(spec: bytes = GRAMMAR_SPEC) -> RunFingerprint:
    return RunFingerprint(
        {"run": "urn:manifest-unit:run-spec"},
        {
            "grammar": IndexedLanguage(
                "decl-1->native-1",
                spec,
                sha256(spec).hexdigest(),
                dx.LoweringRoute("decl-1", "native-1"),
            )
        },
        PACKAGES,
    )


DECLARATIONS: Final = {"grammar": b'{"name":"manifest-grammar"}'}


def a_record(declarations: dict[str, bytes] = DECLARATIONS) -> RunRecord:
    return RunRecord(
        provenance=a_provenance(),
        fingerprint=a_fingerprint(),
        declarations=declarations,
    )


def test_installed_packages_reports_the_installed_versions() -> None:
    packages = installed_packages()

    assert tuple(packages) == (INTERPRETER, *RECORDED_DISTRIBUTIONS)
    assert {"lairs", "numpy", "transformers"} <= set(RECORDED_DISTRIBUTIONS)
    assert packages == {
        INTERPRETER: platform.python_version(),
        **{name: version(name) for name in RECORDED_DISTRIBUTIONS if name != "torch"},
        "torch": str(torch.__version__),
    }


def test_installed_packages_records_the_framework_build_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch, "__version__", f"{version('torch')}+cu130")

    assert installed_packages()["torch"] == f"{version('torch')}+cu130"


def test_the_accelerator_runtime_reports_the_framework_build() -> None:
    reported = accelerator_runtime()
    family, _, held = reported.partition(" ")

    if torch.version.cuda is not None:
        assert (family, held) == ("cuda", torch.version.cuda)

    elif torch.backends.mps.is_built():
        assert (family, held) == ("mps", platform.mac_ver()[0])

    else:
        assert reported == "cpu"


def test_run_records_compare_by_every_field() -> None:
    assert a_record() == a_record()
    assert hash(a_record()) == hash(a_record())
    assert a_record() != RunRecord(
        provenance=a_provenance(),
        fingerprint=a_fingerprint(GRAMMAR_SPEC + b" "),
        declarations=DECLARATIONS,
    )
    assert a_record() != a_record(declarations={"grammar": b'{"name":"other"}'})
    assert hash(a_record()) != hash(
        a_record(declarations={"grammar": b'{"name":"other"}'})
    )
    assert a_record().__eq__(object()) is NotImplemented
    assert repr(a_record()).endswith(f"declarations={DECLARATIONS!r})")


def test_a_run_record_pairs_each_declaration_with_its_fingerprinted_theory() -> None:
    record = a_record()

    assert dict(record.languages) == {
        "grammar": ArchivedLanguage(
            b'{"name":"manifest-grammar"}', a_fingerprint().languages["grammar"]
        )
    }


def test_a_run_record_pairs_each_of_several_declarations_with_its_own_theory() -> None:
    grammar = a_fingerprint().languages["grammar"]
    lexicon = a_fingerprint(b"lexicon spec").languages["grammar"]
    record = RunRecord(
        provenance=a_provenance(),
        fingerprint=RunFingerprint(
            {"run": "urn:manifest-unit:run-spec"},
            {"lexicon": lexicon, "grammar": grammar},
            PACKAGES,
        ),
        declarations={"lexicon": b'{"name":"lex"}', "grammar": b'{"name":"gram"}'},
    )

    assert list(record.languages) == ["grammar", "lexicon"]
    assert dict(record.languages) == {
        "grammar": ArchivedLanguage(b'{"name":"gram"}', grammar),
        "lexicon": ArchivedLanguage(b'{"name":"lex"}', lexicon),
    }


@pytest.mark.parametrize(
    "declarations",
    [{}, {"grammar": b"{}", "lexicon": b"{}"}, {"lexicon": b"{}"}],
    ids=["none", "extra", "renamed"],
)
def test_a_run_record_refuses_declarations_its_fingerprint_does_not_index(
    declarations: dict[str, bytes],
) -> None:
    with pytest.raises(ValueError, match="declares languages"):
        a_record(declarations=declarations)


def test_a_run_record_refuses_a_declaration_that_is_not_bytes() -> None:
    with pytest.raises(TypeError, match="not bytes"):
        a_record(declarations={"grammar": cast("bytes", "text")})


def test_writing_a_record_leaves_the_three_files_beside_the_manifest(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "runs" / "point"
    manifest = a_manifest()

    write_record(directory, manifest, a_record())
    write_manifest(directory, manifest)

    assert sorted(path.name for path in directory.iterdir()) == sorted(
        (
            MANIFEST_NAME,
            RESOLVED_NAME,
            SCHEMA_FINGERPRINT_NAME,
            SOURCES_NAME,
        )
    )
    assert read_manifest(directory) == manifest
    assert manifest.accelerator == "cpu"
    assert read_record(directory) == StoredRecord(
        manifest.run_spec, a_provenance(), a_fingerprint()
    )


def test_the_written_files_are_indented_json_a_reader_can_open(tmp_path: Path) -> None:
    manifest = a_manifest(epochs=3)

    write_record(tmp_path, manifest, a_record())

    resolved = json.loads((tmp_path / RESOLVED_NAME).read_text(encoding="utf-8"))
    sources = json.loads((tmp_path / SOURCES_NAME).read_text(encoding="utf-8"))
    fingerprint = json.loads(
        (tmp_path / SCHEMA_FINGERPRINT_NAME).read_text(encoding="utf-8")
    )

    assert resolved["trainer"]["kind"] == "record-unit-trainer"
    assert resolved["trainer"]["epochs"] == 3
    assert sources["origins"]["trainer.epochs"]["layer"] == "run.json"
    assert fingerprint["digest"] == a_fingerprint().digest
    assert (tmp_path / RESOLVED_NAME).read_text(encoding="utf-8").count("\n") > 1


def test_the_resolved_configuration_is_the_manifest_run_specification(
    tmp_path: Path,
) -> None:
    write_record(tmp_path, a_manifest(epochs=2), a_record())

    assert read_record(tmp_path).resolved == RunSpec.model_validate(a_document(2))
    assert read_record(tmp_path).resolved != RunSpec.model_validate(a_document(1))


def test_a_manifest_missing_a_recorded_distribution_writes_nothing(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError, match="didactic, lairs, numpy, panproto, python, quivers"
    ):
        write_record(tmp_path, a_manifest(packages={"torch": "2.14.0"}), a_record())

    assert list(tmp_path.iterdir()) == []


def test_rewriting_the_manifest_replaces_it_and_leaves_no_staged_file(
    tmp_path: Path,
) -> None:
    manifest = a_manifest()

    write_manifest(tmp_path, manifest)
    write_manifest(tmp_path, close_manifest(manifest, ManifestStatus.KILLED))

    assert [path.name for path in tmp_path.iterdir()] == [MANIFEST_NAME]
    stored = read_manifest(tmp_path)
    assert stored is not None
    assert stored.status is ManifestStatus.KILLED


def test_an_empty_directory_holds_no_manifest(tmp_path: Path) -> None:
    assert read_manifest(tmp_path) is None
    assert read_manifest(tmp_path / "absent") is None


def test_reading_an_incomplete_record_names_the_missing_file(tmp_path: Path) -> None:
    write_record(tmp_path, a_manifest(), a_record())
    (tmp_path / SOURCES_NAME).unlink()

    with pytest.raises(FileNotFoundError, match=SOURCES_NAME):
        read_record(tmp_path)


def test_a_tampered_fingerprint_is_refused_on_read(tmp_path: Path) -> None:
    write_record(tmp_path, a_manifest(), a_record())
    path = tmp_path / SCHEMA_FINGERPRINT_NAME
    record = json.loads(path.read_text(encoding="utf-8"))
    record["digest"] = "0" * 64
    path.write_text(json.dumps(record), encoding="utf-8", newline="\n")

    with pytest.raises(ValueError, match="does not match"):
        read_record(tmp_path)


def an_experiment_price(seconds: float = 4.0) -> ExperimentPrice:
    stacks = [
        StackPrice(
            stack=f"stack-{epochs}",
            points=[f"point-{epochs}-{seed}" for seed in range(3)],
            length=TrainingLength(epochs, TrainingUnit.EPOCHS),
            timing=StackTiming(
                steps_per_unit=5,
                positions=[0, 1],
                seconds=[seconds * epochs, seconds],
                peak_allocated_bytes=1_024 * epochs,
                shape=BatchShape(12.0, 8.0),
            ),
            budget=MemoryBudget(100_000),
        )
        for epochs in (1, 2)
    ]

    return ExperimentPrice(
        experiment="headline",
        freeze_id="freeze/1",
        sweep_id="0123456789abcdef",
        device="cuda:0",
        hardware=HardwareDescriptor(
            device_name="NVIDIA A40",
            driver=None,
            cuda="12.8",
            torch="2.8.0",
            host_cpu="x86_64",
        ),
        warmup=1,
        repeats=2,
        resamples=100,
        bootstrap_seed=0,
        stacks=stacks,
        grid=None,
    )


def test_a_written_price_is_read_back_as_the_price_it_was_written_from(
    tmp_path: Path,
) -> None:
    price = an_experiment_price()
    write_price(tmp_path, price)
    stored = read_price(tmp_path)

    assert (tmp_path / PRICE_NAME).is_file()
    assert stored == price


def test_the_hours_are_recomputed_from_the_raw_timings_the_record_holds(
    tmp_path: Path,
) -> None:
    price = an_experiment_price()
    write_price(tmp_path, price)
    written = json.loads((tmp_path / PRICE_NAME).read_text(encoding="utf-8"))
    stored = read_price(tmp_path)

    assert stored is not None
    assert stored.p50_hours == price.p50_hours
    assert stored.p90_hours == price.p90_hours
    assert [stack["seconds"] for stack in written["stacks"]] == [
        list(stack.seconds) for stack in price.stacks
    ]
    assert "p50_hours" not in written
    assert "p90_hours" not in written


def test_a_price_written_twice_holds_the_later_price(tmp_path: Path) -> None:
    write_price(tmp_path, an_experiment_price())
    write_price(tmp_path, an_experiment_price(8.0))
    stored = read_price(tmp_path)

    assert stored == an_experiment_price(8.0)


def test_a_directory_holding_no_price_reads_as_none(tmp_path: Path) -> None:
    assert read_price(tmp_path) is None
    assert read_price(tmp_path / "absent") is None


def test_a_price_is_written_into_a_directory_that_does_not_exist_yet(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "sweeps" / "0123456789abcdef"
    write_price(directory, an_experiment_price())

    assert read_price(directory) == an_experiment_price()


def test_a_price_record_missing_a_member_is_refused_on_read(tmp_path: Path) -> None:
    write_price(tmp_path, an_experiment_price())
    path = tmp_path / PRICE_NAME
    record = json.loads(path.read_text(encoding="utf-8"))
    del record["device"]
    path.write_text(json.dumps(record), encoding="utf-8", newline="\n")

    with pytest.raises(ValueError, match="device"):
        read_price(tmp_path)


def test_a_price_record_naming_another_cost_model_than_its_grid_is_refused(
    tmp_path: Path,
) -> None:
    write_price(tmp_path, an_experiment_price())
    path = tmp_path / PRICE_NAME
    record = json.loads(path.read_text(encoding="utf-8"))
    record["cost_model"] = GRID_COST_MODEL
    path.write_text(json.dumps(record), encoding="utf-8", newline="\n")

    with pytest.raises(ValueError, match="cost model"):
        read_price(tmp_path)


def test_a_price_record_holding_a_value_its_price_refuses_is_refused(
    tmp_path: Path,
) -> None:
    write_price(tmp_path, an_experiment_price())
    path = tmp_path / PRICE_NAME
    record = json.loads(path.read_text(encoding="utf-8"))
    record["stacks"][0]["budget"]["max_bytes"] = 0
    path.write_text(json.dumps(record), encoding="utf-8", newline="\n")

    with pytest.raises(ValueError, match="room for a byte"):
        read_price(tmp_path)


def test_a_price_round_trips_with_its_grid_cells(tmp_path: Path) -> None:
    ungridded = an_experiment_price()
    cells = [
        GridCell(length, batch, (0.1 * length, 0.2 * batch))
        for length in (10, 20)
        for batch in (8, 16)
    ]
    price = ExperimentPrice(
        experiment=ungridded.experiment,
        freeze_id=ungridded.freeze_id,
        sweep_id=ungridded.sweep_id,
        device=ungridded.device,
        hardware=ungridded.hardware,
        warmup=ungridded.warmup,
        repeats=ungridded.repeats,
        resamples=ungridded.resamples,
        bootstrap_seed=ungridded.bootstrap_seed,
        stacks=ungridded.stacks,
        grid=GridTiming(cells),
    )
    write_price(tmp_path, price)
    stored = read_price(tmp_path)

    assert stored == price
    assert stored is not None
    assert stored.cost_model == GRID_COST_MODEL
    assert stored.p90_hours == price.p90_hours


DERIVATION: Final = ExperimentSeed(7).point("p", 0)


def test_a_point_records_its_rows_with_their_freeze_and_reads_them_back(
    tmp_path: Path,
) -> None:
    rows: tuple[dict[str, JsonValue], ...] = (
        {"metric": "f1", "value": 0.5, "residual_chart-cap": None},
        {"metric": "coverage", "value": None, "n": 0},
    )
    recorded = RecordedRows("freeze/1", DERIVATION, rows)

    assert read_rows(tmp_path) is None

    write_rows(tmp_path / "point", recorded)
    read = read_rows(tmp_path / "point")

    assert read is not None
    assert (read.freeze_id, read.seed_derivation, read.rows) == (
        "freeze/1",
        DERIVATION,
        rows,
    )


def test_rows_produced_outside_an_experiment_record_no_freeze(tmp_path: Path) -> None:
    write_rows(tmp_path, RecordedRows(None, DERIVATION, ()))
    read = read_rows(tmp_path)

    assert read is not None
    assert (read.freeze_id, read.rows) == (None, ())


@pytest.mark.parametrize(
    "content",
    [
        "[]",
        '{"freeze_id": "freeze/1", "seed_derivation": {}, "rows": [1]}',
        '{"freeze_id": 1, "seed_derivation": {}, "rows": []}',
        '{"freeze_id": "freeze/1", "rows": []}',
    ],
)
def test_malformed_rows_are_refused(tmp_path: Path, content: str) -> None:
    (tmp_path / REPORTED_NAME).write_text(content, encoding="utf-8", newline="\n")

    with pytest.raises(ValueError, match="malformed"):
        read_rows(tmp_path)


def test_a_binding_is_recorded_and_read_back(tmp_path: Path) -> None:
    binding = PointBinding("freeze/1", DERIVATION)

    assert read_binding(tmp_path) is None

    write_binding(tmp_path / "point", binding)
    read = read_binding(tmp_path / "point")

    assert read is not None
    assert (read.freeze_id, read.seed_derivation) == ("freeze/1", DERIVATION)


def test_a_malformed_binding_is_refused(tmp_path: Path) -> None:
    (tmp_path / POINT_NAME).write_text(
        '{"freeze_id": "freeze/1"}', "utf-8", newline="\n"
    )

    with pytest.raises(ValueError, match="malformed"):
        read_binding(tmp_path)


@pytest.mark.parametrize("kind", list(FailureKind))
def test_a_failure_is_recorded_with_its_seed_and_read_back(
    tmp_path: Path, kind: FailureKind
) -> None:
    failure = PointFailure(kind, "the loss is non-finite", 3, DERIVATION)

    assert read_failure(tmp_path) is None

    write_failure(tmp_path / "point", failure)
    read = read_failure(tmp_path / "point")

    assert read is not None
    assert (read.kind, read.diagnostic, read.attempts) == (
        kind,
        "the loss is non-finite",
        3,
    )
    assert read.seed_derivation == DERIVATION

    remove_failure(tmp_path / "point")
    remove_failure(tmp_path / "point")

    assert read_failure(tmp_path / "point") is None


@pytest.mark.parametrize(
    "record",
    [
        {"status": "numerical-failure", "diagnostic": "nan", "seed_derivation": {}},
        {
            "status": "numerical-failure",
            "diagnostic": "nan",
            "attempts": 0,
            "seed_derivation": {},
        },
        {"status": "numerical-failure", "diagnostic": "nan", "attempts": 1},
        [],
    ],
)
def test_a_malformed_failure_record_is_refused(
    tmp_path: Path, record: JsonValue
) -> None:
    (tmp_path / FAILURE_NAME).write_text(
        json.dumps(record), encoding="utf-8", newline="\n"
    )

    with pytest.raises(ValueError, match="malformed"):
        read_failure(tmp_path)


def test_a_failure_of_an_unknown_kind_is_refused(tmp_path: Path) -> None:
    (tmp_path / FAILURE_NAME).write_text(
        json.dumps(
            {
                "status": "vanished",
                "diagnostic": "",
                "attempts": 1,
                "seed_derivation": encode_seed_derivation(DERIVATION).model_dump(),
            }
        ),
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ValueError, match="vanished"):
        read_failure(tmp_path)


def test_no_field_of_the_manifest_is_written_twice_in_the_run_directory(
    tmp_path: Path,
) -> None:
    manifest = a_manifest()
    write_manifest(tmp_path, manifest)
    write_record(tmp_path, manifest, a_record())
    recorded = json.loads((tmp_path / MANIFEST_NAME).read_text(encoding="utf-8"))
    others = {
        name: json.loads((tmp_path / name).read_text(encoding="utf-8"))
        for name in (SOURCES_NAME, SCHEMA_FINGERPRINT_NAME)
    }

    for name, fields in others.items():
        assert set(fields).isdisjoint(set(recorded) - {"run_spec"}), name


def test_a_malformed_manifest_is_refused_as_the_decoder_refuses_it(
    tmp_path: Path,
) -> None:
    (tmp_path / MANIFEST_NAME).write_text(
        '{"status": "running"}', encoding="utf-8", newline="\n"
    )

    with pytest.raises(ValueError, match="experiment manifest"):
        read_manifest(tmp_path)

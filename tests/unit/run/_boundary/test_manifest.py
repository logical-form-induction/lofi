import json
from typing import TYPE_CHECKING, cast

import didactic.api as dx
import pytest
import torch
from tests.conformance.experiment_manifest import (
    EXPERIMENT_MANIFEST_LAWS,
    ManifestFixture,
)
from tests.unit.run._boundary.conftest import ROLES, a_run_spec
from torch import nn

from lofi.eval import ConventionVector, ReportIdentity, ReportRow
from lofi.judgment import (
    Measurement,
    posterior_variance_row,
    predictive_density_row,
    pseudo_r_squared,
)
from lofi.run import (
    ExperimentSeed,
    SeedPolicy,
    decode_seed_derivation,
    encode_seed_derivation,
)
from lofi.run._boundary.manifest import (
    BindingIso,
    ExperimentManifest,
    FailureIso,
    FailureKind,
    ManifestIso,
    ManifestStatus,
    MeasuredRole,
    PointBinding,
    PointFailure,
    PointManifest,
    RecordedRows,
    RoleParameterRecord,
    RowsIso,
    close_manifest,
    decode_file,
    decode_manifest,
    indented_json,
    judgment_report_row,
    measure_point,
    open_manifest,
    read_file,
)
from lofi.run._boundary.sweep import SweepPoint
from lofi.spec import ComputeDtype, DevicePolicySpec, MatmulPrecision
from lofi.spec.sinks.wandb import WandbSinkSpec

if TYPE_CHECKING:
    from pathlib import Path

    from lofi._types import JsonValue


class _ReportConfig:
    def model_dump(self) -> dict[str, JsonValue]:
        return {"run": "judgment-report"}


class _ReportGrammar:
    fingerprint = "grammar:judgment-report"


def _report_identity() -> ReportIdentity:
    return ReportIdentity(
        _ReportConfig(),
        _ReportGrammar(),
        ConventionVector(
            punctuation="strip",
            unary_chains="collapse",
            trivial_spans="drop",
            minimum_length=0,
            aggregation="sentence",
            coverage="zero",
            self_arcs="forbid",
            gold_derivation="canonical",
        ),
        freeze_id="freeze/8",
    )


def _lower_measurement(metric: str, measurement: Measurement) -> ReportRow:
    return judgment_report_row(
        measurement,
        _report_identity(),
        suite="judgment-models",
        task="held-out-fit",
        dataset="mega-veridicality",
        split="test",
        metric=metric,
        seed=19,
        normal_form=None,
        specification_hash="specification-a1",
        fold=1,
        rung="verb-by-frame",
        held_out_convention="marginal",
    )


def test_real_judgment_measurements_lower_without_revalidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    density = predictive_density_row(
        "verb-by-frame", torch.log(torch.tensor((0.5, 0.25)))
    ).predictive_density
    variance = posterior_variance_row(
        "verb-by-frame", "participant", torch.tensor((0.1, 0.2, 0.3))
    ).posterior_variance
    pseudo = pseudo_r_squared(
        "verb-by-frame",
        Measurement(value=-2.0, n=2, unit="response"),
        reference="intercept-only",
        references={"intercept-only": Measurement(value=-4.0, n=2, unit="response")},
    ).pseudo_r_squared

    def refuse_revalidation(field: str, value: float) -> float:
        del field, value
        message = "validated measurements must not be revalidated"
        raise AssertionError(message)

    monkeypatch.setattr(
        "lofi.eval._values.ensure_finite",
        refuse_revalidation,
    )
    monkeypatch.setattr("lofi.stats._values.ensure_finite", refuse_revalidation)
    rows = (
        _lower_measurement("predictive-density", density),
        _lower_measurement("posterior-variance", variance),
        _lower_measurement("pseudo-r-squared", pseudo),
    )

    assert [row.value for row in rows] == [
        density.value,
        variance.value,
        pseudo.value,
    ]
    assert [row.resampling_unit for row in rows] == [
        density.unit,
        variance.unit,
        pseudo.unit,
    ]
    assert [row.metric for row in rows] == [
        "predictive-density",
        "posterior-variance",
        "pseudo-r-squared",
    ]
    assert all(row.fold == 1 for row in rows)


def a_device_policy() -> DevicePolicySpec:
    return DevicePolicySpec(
        device="cpu",
        compute_dtype=ComputeDtype.FLOAT32,
        deterministic=True,
        matmul_precision=MatmulPrecision.HIGHEST,
        activation_memory_bytes=8_000_000_000,
    )


SEED = ExperimentSeed(7).point("0123456789abcdef", 0)


def a_manifest() -> ExperimentManifest:
    return open_manifest(
        run_spec=a_run_spec(),
        component_fingerprints={
            role: f"fingerprint:{role}" for role in a_run_spec().component_roles()
        },
        grammar_fingerprint="grammar:ccg",
        signature_fingerprint="signature:ccg",
        corpus_revision="ccgbank:LDC2005T13:auto-v1",
        seed_policy=SeedPolicy(SEED.root, loaders=("training",)),
        seed_derivation=SEED,
        package_versions={
            "didactic": "0.0.2",
            "panproto": "0.1.0",
            "quivers": "0.1.0",
            "torch": "2.8.0",
        },
        git_revision="0123456789abcdef",
        accelerator="cpu",
    )


@pytest.mark.parametrize(
    ("policy", "match"),
    [
        ({"loaders": ["training", "evaluation"]}, "canonical order"),
        ({"derivation": "blake2b-32"}, "derivation"),
        ({"version": 1}, "version"),
        ({"root": -1}, "negative"),
    ],
)
def test_a_seed_policy_record_naming_no_policy_is_refused(
    policy: dict[str, JsonValue], match: str
) -> None:
    record = cast("dict[str, JsonValue]", json.loads(a_manifest().model_dump_json()))
    seed_policy = record["seed_policy"]
    assert isinstance(seed_policy, dict)

    with pytest.raises(dx.ValidationError, match=match):
        ExperimentManifest.model_validate(
            {**record, "seed_policy": {**seed_policy, **policy}}
        )


def test_a_manifest_records_the_full_run_provenance() -> None:
    manifest = a_manifest()

    assert manifest.run_spec == a_run_spec()
    assert frozenset(manifest.component_fingerprints) == ROLES
    assert manifest.grammar_fingerprint == "grammar:ccg"
    assert manifest.corpus_revision == "ccgbank:LDC2005T13:auto-v1"
    assert manifest.seed_policy.policy() == SeedPolicy(SEED.root, loaders=("training",))
    assert decode_seed_derivation(manifest.seed_derivation) == SEED
    assert manifest.seed_derivation.experiment_root == 7
    assert manifest.device_policy == a_device_policy()
    assert manifest.package_versions["torch"] == "2.8.0"
    assert manifest.git_revision == "0123456789abcdef"


def test_a_manifest_serializes_one_device_policy_source() -> None:
    manifest = a_manifest()
    document = json.loads(manifest.model_dump_json())

    assert "device_policy" not in document
    assert document["run_spec"]["device_policy"] == a_device_policy().model_dump()

    with pytest.raises(dx.ValidationError, match="device_policy"):
        ExperimentManifest.model_validate(
            document | {"device_policy": a_device_policy().model_dump()}
        )


def test_the_manifest_status_vocabulary_distinguishes_each_lifecycle_state() -> None:
    assert tuple(ManifestStatus) == (
        ManifestStatus.RUNNING,
        ManifestStatus.COMPLETED,
        ManifestStatus.KILLED,
    )


def test_a_manifest_opens_running_and_closes_completed() -> None:
    opened = a_manifest()
    closed = close_manifest(opened)

    assert opened.status is ManifestStatus.RUNNING
    assert closed.status is ManifestStatus.COMPLETED
    assert opened is not closed


def test_a_killed_manifest_cannot_be_mistaken_for_a_completed_one() -> None:
    opened = a_manifest()
    killed = close_manifest(opened, ManifestStatus.KILLED)
    completed = close_manifest(opened)

    assert killed.status is ManifestStatus.KILLED
    assert killed != completed


def test_closing_returns_success_instead_of_raising_it() -> None:
    result = close_manifest(a_manifest())

    assert isinstance(result, ExperimentManifest)
    assert result.status is ManifestStatus.COMPLETED


def test_a_closed_manifest_cannot_keep_the_running_status() -> None:
    with pytest.raises(ValueError, match="cannot remain running"):
        close_manifest(a_manifest(), ManifestStatus.RUNNING)


def test_component_fingerprints_cover_every_role_in_the_run_spec() -> None:
    manifest = a_manifest()
    document = json.loads(manifest.model_dump_json())

    with pytest.raises(
        dx.ValidationError, match=r"missing roles: output\.sinks\.metrics"
    ):
        ExperimentManifest.model_validate(
            document
            | {
                "component_fingerprints": {
                    role: fingerprint
                    for role, fingerprint in manifest.component_fingerprints.items()
                    if role != "output.sinks.metrics"
                }
            }
        )

    with pytest.raises(dx.ValidationError, match="undeclared roles: stray-component"):
        ExperimentManifest.model_validate(
            document
            | {
                "component_fingerprints": manifest.component_fingerprints
                | {"stray-component": "fingerprint:stray"}
            }
        )


def test_manifest_mappings_are_stored_in_name_order() -> None:
    manifest = a_manifest().with_(
        package_versions={"torch": "2.8.0", "didactic": "0.0.2"}
    )

    assert tuple(manifest.component_fingerprints) == tuple(sorted(ROLES))
    assert tuple(manifest.package_versions) == ("didactic", "torch")


def test_manifest_identity_fields_and_mapping_values_must_be_nonempty() -> None:
    with pytest.raises(dx.ValidationError):
        a_manifest().with_(git_revision="")

    with pytest.raises(dx.ValidationError):
        a_manifest().with_(package_versions={"torch": ""})

    document = json.loads(a_manifest().model_dump_json())
    document["package_versions"] = {}

    with pytest.raises(dx.ValidationError, match="cannot be empty"):
        ExperimentManifest.model_validate(document)

    document = json.loads(a_manifest().model_dump_json())
    document["seed_policy"] = {}

    with pytest.raises(dx.ValidationError, match=r"seed_policy.*missing"):
        ExperimentManifest.model_validate(document)


def test_a_manifest_refuses_a_seed_its_derivation_does_not_give() -> None:
    document = json.loads(a_manifest().model_dump_json())
    document["seed_policy"] = dict(SeedPolicy(7, loaders=("training",)).record)

    with pytest.raises(dx.ValidationError, match="seed_derivation derives"):
        ExperimentManifest.model_validate(document)

    document = json.loads(a_manifest().model_dump_json())
    document["seed_derivation"] = dict(
        encode_seed_derivation(
            ExperimentSeed(8).point(SEED.point_id, SEED.seed)
        ).model_dump()
    )

    with pytest.raises(dx.ValidationError, match="seed_derivation derives"):
        ExperimentManifest.model_validate(document)

    document = json.loads(a_manifest().model_dump_json())
    del document["seed_derivation"]

    with pytest.raises(dx.ValidationError, match="seed_derivation"):
        ExperimentManifest.model_validate(document)


def test_the_manifest_iso_round_trips_json_bytes() -> None:
    codec = ManifestIso()
    manifest = a_manifest()
    encoded = codec.forward(manifest)

    assert encoded.startswith(b'{"run_spec":')
    assert codec.backward(encoded) == manifest


def a_manifest_with_sink_entity(entity: str | None) -> ExperimentManifest:
    manifest = a_manifest()
    sink = WandbSinkSpec(project="grammar", entity=entity)
    run_spec = manifest.run_spec.with_(sinks={**manifest.run_spec.sinks, "wandb": sink})

    return manifest.with_(
        run_spec=run_spec,
        component_fingerprints={
            **manifest.component_fingerprints,
            "output.sinks.wandb": "fingerprint:wandb",
        },
    )


def test_the_manifest_iso_keeps_a_present_null_distinct_from_an_absent_field() -> None:
    codec = ManifestIso()
    spelled = a_manifest_with_sink_entity("null")
    absent = a_manifest_with_sink_entity(None)

    assert codec.forward(spelled) != codec.forward(absent)
    assert (
        codec.backward(codec.forward(spelled)).run_spec.sinks["wandb"]
        == spelled.run_spec.sinks["wandb"]
    )
    assert (
        codec.backward(codec.forward(absent)).run_spec.sinks["wandb"]
        == absent.run_spec.sinks["wandb"]
    )
    assert codec.backward(codec.forward(spelled)) != codec.backward(
        codec.forward(absent)
    )


def test_the_model_refuses_an_unknown_field() -> None:
    with pytest.raises(dx.ValidationError, match="finished"):
        ExperimentManifest.model_validate(
            a_manifest().model_dump() | {"finished": True}
        )


def test_the_working_path_satisfies_every_manifest_law() -> None:
    fixture = ManifestFixture(a_manifest, close_manifest)

    for law in EXPERIMENT_MANIFEST_LAWS:
        law(fixture)


def a_measured_document(
    encoder: dict[str, JsonValue] | None, /
) -> dict[str, JsonValue]:
    model: dict[str, JsonValue] = {
        "language": {"from": "data"},
        "parser": {"kind": "manifest-unit-parser"},
    }

    if encoder is not None:
        model["encoders"] = {"terms": encoder}

    return {
        "model": model,
        "loss": {"terms": [{"objective": {"kind": "manifest-unit-loss"}}]},
        "data": {
            "datasets": {
                "train": {
                    "dataset": {"kind": "manifest-unit-dataset"},
                    "featurizer": {"kind": "manifest-unit-featurizer"},
                }
            }
        },
        "trainer": {"kind": "manifest-unit-trainer"},
        "optimizer": {"kind": "manifest-unit-optimizer"},
        "device_policy": {
            "device": "cpu",
            "compute_dtype": "float32",
            "deterministic": True,
            "matmul_precision": "highest",
            "activation_memory_bytes": 8_000_000_000,
        },
    }


def test_a_point_manifest_measures_each_role_and_the_whole_model() -> None:
    point = SweepPoint(
        a_measured_document({"kind": "measured-unit-encoder", "width": 64}),
        overrides=("model.encoders.terms.width=64",),
        seed=3,
    )
    encoder = nn.Linear(3, 4)
    parser = nn.Linear(2, 2).requires_grad_(requires_grad=False)
    manifest = measure_point(
        point,
        {
            "model.encoders.terms": MeasuredRole(
                "measured-unit-encoder", tuple(encoder.parameters())
            ),
            "model.parser": MeasuredRole(
                "manifest-unit-parser", tuple(parser.parameters())
            ),
        },
        corpus_fingerprint="corpus:a",
    )

    assert manifest.point_id == point.point_id
    assert manifest.seed == 3
    assert manifest.corpus_fingerprint == "corpus:a"
    assert list(manifest.roles) == ["model.encoders.terms", "model.parser"]
    assert manifest.roles["model.encoders.terms"] == RoleParameterRecord(
        kind="measured-unit-encoder", parameters=16, trainable=16
    )
    assert manifest.roles["model.parser"] == RoleParameterRecord(
        kind="manifest-unit-parser", parameters=6, trainable=0
    )
    assert (manifest.parameter_count, manifest.trainable_count) == (22, 16)
    assert PointManifest.model_validate_json(manifest.model_dump_json()) == manifest


def test_a_parameter_two_roles_share_counts_once_in_the_whole_model() -> None:
    shared = nn.Linear(2, 2)
    manifest = measure_point(
        SweepPoint(a_measured_document(None)),
        {
            "first": MeasuredRole("unit", tuple(shared.parameters())),
            "second": MeasuredRole("unit", tuple(shared.parameters())),
        },
        corpus_fingerprint="corpus:a",
    )

    assert manifest.roles["first"].parameters == 6
    assert manifest.roles["second"].parameters == 6
    assert manifest.parameter_count == 6


def test_a_point_manifest_counts_a_model_of_no_parameters_as_none() -> None:
    point = SweepPoint(a_measured_document(None))
    manifest = measure_point(point, {}, corpus_fingerprint="corpus:a")

    assert manifest.roles == {}
    assert (manifest.parameter_count, manifest.trainable_count) == (0, 0)
    assert manifest.seed == 0


def test_a_point_manifest_refuses_an_empty_identity_or_an_impossible_size() -> None:
    role: dict[str, JsonValue] = {"kind": "unit", "parameters": 4, "trainable": 4}
    fields: dict[str, JsonValue] = {
        "point_id": "0123456789abcdef",
        "seed": 0,
        "corpus_fingerprint": "corpus:a",
        "roles": {"parser": role},
        "parameter_count": 4,
        "trainable_count": 4,
    }

    assert PointManifest.model_validate(fields).parameter_count == 4

    for field, value in (
        ("point_id", ""),
        ("corpus_fingerprint", ""),
        ("seed", -1),
        ("parameter_count", -1),
        ("trainable_count", -1),
        ("trainable_count", 5),
    ):
        with pytest.raises(dx.ValidationError):
            PointManifest.model_validate(fields | {field: value})

    for field, value in (
        ("kind", ""),
        ("parameters", -1),
        ("trainable", -1),
        ("trainable", 5),
    ):
        with pytest.raises(dx.ValidationError):
            RoleParameterRecord.model_validate(role | {field: value})


def test_two_roles_of_one_kind_configured_apart_are_recorded_apart() -> None:
    manifest = a_manifest()
    run_spec = manifest.run_spec.with_(
        sinks={
            "first": WandbSinkSpec(project="grammar"),
            "second": WandbSinkSpec(project="lexicon"),
        }
    )
    fingerprints = {
        role: fingerprint
        for role, fingerprint in manifest.component_fingerprints.items()
        if not role.startswith("output.sinks.")
    }
    recorded = manifest.with_(
        run_spec=run_spec,
        component_fingerprints={
            **fingerprints,
            "output.sinks.first": "fingerprint:first",
            "output.sinks.second": "fingerprint:second",
        },
    )

    assert (
        recorded.component_fingerprints["output.sinks.first"]
        != (recorded.component_fingerprints["output.sinks.second"])
    )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("derivation", "sha256-v0", "derivation"),
        ("version", 2, "version"),
        ("root", -1, "root"),
        ("samplers", ["fit", "fit"], "sampler streams"),
    ],
)
def test_a_seed_policy_record_refuses_what_encodes_no_policy(
    field: str, value: JsonValue, match: str
) -> None:
    document = json.loads(a_manifest().model_dump_json())
    document["seed_policy"][field] = value

    with pytest.raises(dx.ValidationError, match=match):
        ExperimentManifest.model_validate(document)


def test_the_seed_records_return_the_policy_and_derivation_they_encode() -> None:
    manifest = a_manifest()

    assert manifest.seed_policy.policy() == SeedPolicy(SEED.root, loaders=("training",))
    assert decode_seed_derivation(manifest.seed_derivation) == SEED
    assert json.loads(manifest.model_dump_json())["seed_derivation"] == dict(
        encode_seed_derivation(SEED).model_dump()
    )


POINT_SEED = ExperimentSeed(7).point("p", 0)


def test_each_point_record_round_trips_through_its_codec() -> None:
    binding = PointBinding("freeze/1", POINT_SEED)
    rows = RecordedRows(None, POINT_SEED, ({"metric": "f1", "value": None},))
    failure = PointFailure(FailureKind.NUMERICAL, "nan", 2, POINT_SEED)

    assert BindingIso().backward(BindingIso().forward(binding)) == binding
    assert RowsIso().backward(RowsIso().forward(rows)) == rows
    assert FailureIso().backward(FailureIso().forward(failure)) == failure
    assert json.loads(FailureIso().forward(failure))["status"] == "numerical-failure"


def test_recorded_rows_read_back_each_json_array_as_a_list() -> None:
    rows = RecordedRows(None, POINT_SEED, ({"distribution": [0.25, 0.75]},))

    read = RowsIso().backward(RowsIso().forward(rows))

    assert read.rows == ({"distribution": [0.25, 0.75]},)
    assert isinstance(read.rows[0]["distribution"], list)


def test_rows_holding_a_number_json_cannot_hold_are_refused() -> None:
    rows = RecordedRows(None, POINT_SEED, ({"value": float("nan")},))

    with pytest.raises(ValueError, match="JSON compliant"):
        RowsIso().forward(rows)


def test_a_record_is_written_as_indented_json_ending_in_a_newline() -> None:
    written = indented_json(a_manifest())

    assert written.endswith(b"}\n")
    assert written.count(b"\n") > 1
    assert json.loads(written) == json.loads(a_manifest().model_dump_json())


def test_a_file_that_does_not_validate_is_refused_as_a_value_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "point.json"

    assert read_file(path, BindingIso()) is None

    path.write_text('{"freeze_id": "freeze/1"}', encoding="utf-8", newline="\n")

    with pytest.raises(ValueError, match=r"point\.json is malformed"):
        decode_file(path, BindingIso())

    with pytest.raises(FileNotFoundError):
        decode_file(tmp_path / "absent.json", BindingIso())


def test_the_manifest_decoder_refuses_bytes_that_do_not_validate() -> None:
    with pytest.raises(ValueError, match="experiment manifest"):
        decode_manifest(b"{}")

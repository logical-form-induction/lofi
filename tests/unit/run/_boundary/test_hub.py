import json
from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.run._boundary.hub import (
    HubBundleManifest,
    HubBundleManifestIso,
    HubComponentRecord,
    HubFileRecord,
    HubLicenseRecord,
    HubReleaseDeclaration,
    HubReleaseRecord,
    HubTensorRecord,
)
from lofi.run._boundary.manifest import encode_seed_derivation
from lofi.run._seed import ExperimentSeed

if TYPE_CHECKING:
    from lofi._types import JsonValue


def _hub_manifest() -> HubBundleManifest:
    """Return one complete model-only bundle boundary value."""
    digest = "a" * 64

    return HubBundleManifest(
        format="lofi-hub-model",
        seed_derivation=encode_seed_derivation(
            ExperimentSeed(7).point("hub-bundle", 0)
        ),
        release=HubReleaseRecord(
            model_name="parser",
            datasets=("synthetic",),
            languages=("en",),
            grammar_identity=digest,
            codec_identity=digest,
            action_set_identity=digest,
            corpus_revision="corpus-1",
            source_method="generated",
            license_expression="CC-BY-4.0",
            licenses=(HubLicenseRecord(identifier="CC-BY-4.0", applies_to=None),),
            eprint_refs=(),
            reproducibility=None,
            pretrained_roles=(),
        ),
        configuration=HubFileRecord(path="config.json", digest=digest),
        card=HubFileRecord(path="README.md", digest=digest),
        components={
            "parser": HubComponentRecord(
                tag="parser",
                spec_fingerprint=digest,
                state=HubFileRecord(path="states/000000.pt", digest=digest),
                tensors={
                    "weight": HubTensorRecord(
                        kind="tensor", shape=(2, 3), dtype="float32"
                    )
                },
            )
        },
        resources={},
    )


def test_the_hub_manifest_codec_preserves_the_closed_model_only_record() -> None:
    manifest = _hub_manifest()
    codec = HubBundleManifestIso()

    assert codec.backward(codec.forward(manifest)) == manifest


@pytest.mark.parametrize("path", ["/absolute", "../escape", "states/../escape"])
def test_a_hub_file_refuses_a_noncanonical_bundle_path(path: str) -> None:
    with pytest.raises(dx.ValidationError, match="canonical relative path"):
        HubFileRecord(path=path, digest="a" * 64)


def test_a_hub_manifest_refuses_resource_roles_absent_from_the_release() -> None:
    manifest = json.loads(HubBundleManifestIso().forward(_hub_manifest()))
    manifest["resources"] = {"parser": {}}

    with pytest.raises(dx.ValidationError, match="pretrained roles"):
        HubBundleManifest.model_validate(manifest)


def test_a_hub_manifest_refuses_release_metadata_for_another_point_seed() -> None:
    manifest = json.loads(HubBundleManifestIso().forward(_hub_manifest()))
    release = manifest["release"]
    derivation = manifest["seed_derivation"]
    release["reproducibility"] = {
        "code_uri": None,
        "commit_hash": None,
        "command": None,
        "environment": None,
        "random_seed": derivation["root"] + 1,
    }

    with pytest.raises(dx.ValidationError, match="point derivation records"):
        HubBundleManifest.model_validate(manifest)


def test_a_hub_manifest_refuses_an_old_or_extended_shape() -> None:
    manifest = json.loads(HubBundleManifestIso().forward(_hub_manifest()))
    manifest["optimizer"] = {}

    with pytest.raises(dx.ValidationError, match="unknown field 'optimizer'"):
        HubBundleManifest.model_validate(manifest)


def test_a_hub_manifest_refuses_a_shape_without_the_current_format() -> None:
    manifest = json.loads(HubBundleManifestIso().forward(_hub_manifest()))
    del manifest["format"]

    with pytest.raises(
        dx.ValidationError, match="required field 'format' not supplied"
    ):
        HubBundleManifest.model_validate(manifest)


DECLARED: dict[str, JsonValue] = {
    "model_name": "parser",
    "datasets": ["synthetic"],
    "languages": ["en"],
    "source_method": "generated",
    "license_expression": "CC-BY-4.0",
    "licenses": [{"identifier": "CC-BY-4.0", "applies_to": None}],
    "code_uri": "https://example.org/lofi",
    "command": "lofi run",
    "environment": "uv",
}


def test_a_release_declaration_names_metadata_and_no_identity() -> None:
    declared = HubReleaseDeclaration.model_validate(DECLARED)

    assert declared.datasets == ("synthetic",)
    assert declared.licenses == (
        HubLicenseRecord(identifier="CC-BY-4.0", applies_to=None),
    )
    assert declared.eprint_refs == ()
    assert declared.pretrained_resources == {}

    with pytest.raises(dx.ValidationError):
        HubReleaseDeclaration.model_validate(DECLARED | {"grammar_identity": "a" * 64})


@pytest.mark.parametrize(
    "changed",
    [
        {"model_name": ""},
        {"command": ""},
        {"datasets": [""]},
        {"pretrained_resources": {"token_encoder": ""}},
        {"pretrained_resources": {"": "/assets"}},
    ],
)
def test_a_release_declaration_with_blank_metadata_is_refused(
    changed: dict[str, JsonValue],
) -> None:
    with pytest.raises((dx.ValidationError, ValueError)):
        HubReleaseDeclaration.model_validate(DECLARED | changed)


@pytest.mark.parametrize("identity", ["", "A" * 64, "a" * 63, "g" * 64])
@pytest.mark.parametrize(
    "field", ["grammar_identity", "codec_identity", "action_set_identity"]
)
def test_a_release_record_refuses_an_identity_that_is_not_lowercase_sha256(
    identity: str, field: str
) -> None:
    record = json.loads(HubBundleManifestIso().forward(_hub_manifest()))["release"]
    record[field] = identity

    with pytest.raises(dx.ValidationError, match=field):
        HubReleaseRecord.model_validate(record)


def _declaration() -> dict[str, JsonValue]:
    return {
        "model_name": "parser",
        "datasets": ["synthetic"],
        "languages": ["en"],
        "source_method": "generated",
        "license_expression": "CC-BY-4.0",
        "licenses": [{"identifier": "CC-BY-4.0", "applies_to": None}],
        "code_uri": "https://example.org/lofi",
        "command": "lofi run",
        "environment": "cpu",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_name", ""),
        ("code_uri", ""),
        ("datasets", [""]),
        ("languages", [""]),
    ],
)
def test_a_malformed_release_declaration_is_refused_at_its_boundary_naming_the_field(
    field: str, value: JsonValue
) -> None:
    declared = _declaration()
    declared[field] = value

    with pytest.raises(dx.ValidationError) as refused:
        HubReleaseDeclaration.model_validate(declared)

    assert field in str(refused.value)


def test_a_well_formed_release_declaration_is_accepted() -> None:
    assert HubReleaseDeclaration.model_validate(_declaration()).model_name == "parser"


@pytest.mark.parametrize("field", ["datasets", "languages", "eprint_refs"])
def test_a_release_record_with_a_blank_list_entry_is_refused_naming_the_field(
    field: str,
) -> None:
    record = json.loads(HubBundleManifestIso().forward(_hub_manifest()))["release"]
    record[field] = ["named", ""]

    with pytest.raises(dx.ValidationError) as refused:
        HubReleaseRecord.model_validate(record)

    assert field in str(refused.value)


@pytest.mark.parametrize(
    "roles", [["token_encoder", "parser"], ["parser", "parser"], [""]]
)
def test_a_release_record_refuses_pretrained_roles_out_of_order_repeated_or_blank(
    roles: list[JsonValue],
) -> None:
    record = json.loads(HubBundleManifestIso().forward(_hub_manifest()))["release"]
    record["pretrained_roles"] = roles

    with pytest.raises(dx.ValidationError, match="pretrained_roles"):
        HubReleaseRecord.model_validate(record)

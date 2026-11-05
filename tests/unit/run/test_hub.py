from __future__ import annotations

import json
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
from hashlib import file_digest
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple, Never, cast

import pytest
import torch
from httpx import Request, Response
from huggingface_hub import ModelCard
from huggingface_hub.errors import RepositoryNotFoundError
from tests.unit.run.conftest import EXPERIMENT_SEED, a_point, record_manifest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast
from transformers.models.bert import BertConfig

from lofi._artifacts import PinnedCheckpoint
from lofi.data import LicenseRef, Licensing, Provenance, Reproducibility
from lofi.data.synthetic.training import SyntheticTrainingBatch
from lofi.nn.tokens.pretrained import LocalPretrainedLoader
from lofi.run import (
    MANIFEST_NAME,
    ExperimentSeed,
    ManifestStatus,
    RandomStreams,
    RunStore,
    close_manifest,
    open_manifest,
    plan_training,
    read_manifest,
    resolve_device_policy,
    write_manifest,
)
from lofi.run import hub as hub_module
from lofi.run import main as cli_main
from lofi.run._entrypoint import main
from lofi.run._ports import ReleaseAuthorization
from lofi.run.hub import (
    ModelRelease,
    export_hub_model,
    export_trained_point,
    load_hub_bundle,
    restore_checkpoint,
)
from lofi.spec import (
    PACKAGED_CONFIG_ROOT,
    ConfigError,
    ReleaseCheckpoint,
    ReleaseSpec,
    RunSpec,
    RunSpecCodec,
    config_search_path,
)
from lofi.train import (
    ChartForward,
    Checkpoint,
    CheckpointId,
    CheckpointManifest,
    ComponentRecord,
    DirectoryStore,
    ParserFittingPlan,
    ParserFittingRecord,
    PrecisionRecord,
    RngRecord,
    StateReference,
    prepare_training,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from lofi._types import JsonValue
    from lofi.run import ExperimentManifest, PointSeed, TrainingBlueprint
    from lofi.train import PreparedTraining, StateTree

CONFIG = Path(__file__).resolve().parents[3] / "lofi/configs/run/synthetic-parser.yaml"
PACKAGED_CODEC = RunSpecCodec(search_path=(CONFIG.parent, CONFIG.parent.parent))
LOCAL = PinnedCheckpoint("lofi/tiny-local", "0123456789abcdef" * 2 + "01234567")
EPRINT = "at://did:plc:lofi/pub.layers.eprint.eprint/1"
PAD: Final = "[PAD]"
UNKNOWN: Final = "[UNK]"
CLASSIFIER: Final = "[CLS]"
SEPARATOR: Final = "[SEP]"
MASK: Final = "[MASK]"
BUNDLE_DERIVATION = EXPERIMENT_SEED.point("hub-bundle", 0)


class _NetworkUsedError(Exception):
    """Raised if offline bundle work opens a socket."""


def _refusing[**P, R](_replaced: Callable[P, R]) -> Callable[P, R]:
    def refuse(*_args: P.args, **_kwargs: P.kwargs) -> R:
        raise _NetworkUsedError

    return refuse


def _prepared() -> tuple[TrainingBlueprint, PreparedTraining]:
    """Return one seeded configured parser and its checked blueprint."""
    run_spec = PACKAGED_CODEC(CONFIG)
    blueprint = plan_training(run_spec)
    streams = RandomStreams(
        BUNDLE_DERIVATION.policy(accelerator=None, samplers=blueprint.sampler_streams)
    )
    streams.apply()

    return (
        blueprint,
        prepare_training(blueprint, streams, resolve_device_policy(run_spec)),
    )


def _release(
    blueprint: TrainingBlueprint,
    /,
    *,
    seed_derivation: PointSeed = BUNDLE_DERIVATION,
    grammar_identity: str | None = None,
    codec_identity: str | None = None,
    action_set_identity: str | None = None,
) -> ModelRelease:
    """Return complete model-card metadata for one blueprint."""
    return ModelRelease(
        "lofi-synthetic-parser",
        ("lofi/synthetic-parser",),
        ("en",),
        Provenance(
            "generated",
            Licensing(
                "CC-BY-4.0",
                (LicenseRef("CC-BY-4.0", "generated-corpus"),),
            ),
            (EPRINT,),
            Reproducibility(
                "https://github.com/logical-form-induction/lofi",
                "0123456789abcdef",
                "lofi run",
                "uv",
                seed_derivation.root,
            ),
        ),
        grammar_identity=(
            blueprint.grammar_fingerprint
            if grammar_identity is None
            else grammar_identity
        ),
        codec_identity=(
            blueprint.codec_fingerprint if codec_identity is None else codec_identity
        ),
        action_set_identity=(
            blueprint.action_set_fingerprint
            if action_set_identity is None
            else action_set_identity
        ),
        corpus_revision=blueprint.corpus_revision,
        pretrained_resources={},
    )


def _states_equal(
    expected: Mapping[str, Mapping[str, StateTree]],
    found: Mapping[str, Mapping[str, StateTree]],
    /,
) -> bool:
    """Return whether two complete model states are bit-identical."""
    if tuple(expected) != tuple(found):
        return False

    for role in expected:
        if tuple(expected[role]) != tuple(found[role]):
            return False

        for key in expected[role]:
            left = expected[role][key]
            right = found[role][key]

            if isinstance(left, torch.Tensor):
                if not isinstance(right, torch.Tensor) or not torch.equal(left, right):
                    return False
            elif left != right:
                return False

    return True


def _manifest(bundle: Path, /) -> dict[str, JsonValue]:
    """Return a mutable manifest record for corruption tests."""
    return cast(
        "dict[str, JsonValue]",
        json.loads((bundle / "manifest.json").read_text(encoding="utf-8")),
    )


def _write_manifest(bundle: Path, record: Mapping[str, JsonValue], /) -> None:
    """Write one deliberately modified manifest record."""
    (bundle / "manifest.json").write_text(
        json.dumps(record, separators=(",", ":")), encoding="utf-8", newline="\n"
    )


def _export(tmp_path: Path) -> tuple[Path, TrainingBlueprint, PreparedTraining]:
    """Export one changed tiny configured parser."""
    blueprint, prepared = _prepared()

    with torch.no_grad():
        next(iter(prepared.model.parameters)).add_(0.125)

    bundle = tmp_path / "bundle"
    export_hub_model(
        bundle, _release(blueprint), blueprint, prepared, BUNDLE_DERIVATION
    )

    return bundle, blueprint, prepared


def _construction_assets(directory: Path, /) -> None:
    """Write a tiny local BERT configuration and fast tokenizer, but no weights."""
    vocabulary = {
        token: position
        for position, token in enumerate(
            (
                PAD,
                UNKNOWN,
                CLASSIFIER,
                SEPARATOR,
                MASK,
                *(f"word-{position}" for position in range(8)),
            )
        )
    }
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token=UNKNOWN))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token=PAD,
        unk_token=UNKNOWN,
        cls_token=CLASSIFIER,
        sep_token=SEPARATOR,
        mask_token=MASK,
    ).save_pretrained(directory)
    BertConfig(
        hidden_size=4,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=8,
        vocab_size=len(vocabulary),
        max_position_embeddings=16,
        type_vocab_size=2,
        pad_token_id=vocabulary[PAD],
    ).save_pretrained(directory)
    (directory / "pytorch_model.bin").write_bytes(b"must not be copied")


def _pretrained(
    tmp_path: Path,
) -> tuple[TrainingBlueprint, PreparedTraining, Path]:
    """Return a parser constructed from local-only tokenizer and config assets."""
    document = tmp_path / "pretrained.yaml"
    document.write_text(
        "defaults:\n  - run/synthetic-parser\n  - model.token_encoder: pretrained\n",
        encoding="utf-8",
        newline="\n",
    )
    run_spec = RunSpecCodec(search_path=config_search_path(tmp_path))(
        document,
        overrides=(
            "model.token_encoder.checkpoint.repository=lofi/tiny-local",
            f"model.token_encoder.checkpoint.revision={LOCAL.revision}",
            f"artifacts.root={tmp_path / 'artifacts'}",
            "artifacts.offline=true",
            "model.token_encoder.window.rows=16",
        ),
    )
    blueprint = plan_training(run_spec)
    streams = RandomStreams(
        BUNDLE_DERIVATION.policy(accelerator=None, samplers=blueprint.sampler_streams)
    )
    streams.apply()
    assets = tmp_path / "construction-assets"
    assets.mkdir()
    _construction_assets(assets)
    prepared = prepare_training(
        blueprint,
        streams,
        resolve_device_policy(run_spec),
        pretrained_resources={"token_encoder": LocalPretrainedLoader(assets, LOCAL)},
    )

    return blueprint, prepared, assets


def test_export_writes_a_parsed_card_with_complete_release_metadata(
    tmp_path: Path,
) -> None:
    bundle, blueprint, _ = _export(tmp_path)
    fields = ModelCard.load(bundle / "README.md").data.to_dict()

    assert fields["model_name"] == "lofi-synthetic-parser"
    assert fields["datasets"] == ["lofi/synthetic-parser"]
    assert fields["language"] == ["en"]
    assert fields["license"] == "CC-BY-4.0"
    assert fields["source_method"] == "generated"
    assert fields["eprint_refs"] == [EPRINT]
    assert fields["grammar_identity"] == blueprint.grammar_fingerprint
    assert fields["codec_identity"] == blueprint.codec_fingerprint
    assert fields["action_set_identity"] == blueprint.action_set_fingerprint
    assert fields["corpus_revision"] == blueprint.corpus_revision
    assert fields["component_licenses"] == [
        {"identifier": "CC-BY-4.0", "applies_to": "generated-corpus"}
    ]
    assert fields["reproducibility"] == {
        "code_uri": "https://github.com/logical-form-induction/lofi",
        "commit_hash": "0123456789abcdef",
        "command": "lofi run",
        "environment": "uv",
        "random_seed": BUNDLE_DERIVATION.root,
    }


def test_export_uses_a_truthful_model_only_manifest(tmp_path: Path) -> None:
    bundle, _blueprint, prepared = _export(tmp_path)
    record = _manifest(bundle)

    assert set(record) == {
        "format",
        "seed_derivation",
        "release",
        "configuration",
        "card",
        "components",
        "resources",
    }
    assert set(cast("dict[str, JsonValue]", record["components"])) == set(
        prepared.model.modules
    )
    assert all(
        forbidden not in record
        for forbidden in ("optimizer", "scheduler", "rng", "progress", "step")
    )


def test_export_and_load_restore_every_exact_role_bitwise(tmp_path: Path) -> None:
    bundle, blueprint, prepared = _export(tmp_path)
    loaded = load_hub_bundle(bundle)

    assert _states_equal(prepared.model.state(), loaded.prepared.model.state())
    assert loaded.release == _release(blueprint)
    assert loaded.release.provenance == _release(blueprint).provenance


def test_a_bundle_restores_a_grammar_drawn_from_a_derived_data_seed(
    tmp_path: Path,
) -> None:
    document = json.loads(PACKAGED_CODEC(CONFIG).model_dump_json())
    data = cast("dict[str, JsonValue]", document["data"])
    datasets = cast("dict[str, JsonValue]", data["datasets"])
    train = cast("dict[str, JsonValue]", datasets["train"])
    dataset = cast("dict[str, JsonValue]", train["dataset"])
    dataset["seed"] = "derived"
    run_spec = RunSpec.model_validate(document)
    derivation = EXPERIMENT_SEED.point("derived-hub-bundle", 3)
    blueprint = plan_training(run_spec, point=derivation)
    streams = RandomStreams(
        derivation.policy(accelerator=None, samplers=blueprint.sampler_streams)
    )
    streams.apply()
    prepared = prepare_training(
        blueprint, streams, resolve_device_policy(blueprint.run_spec)
    )
    release = _release(blueprint, seed_derivation=derivation)
    bundle = tmp_path / "derived-bundle"
    export_hub_model(bundle, release, blueprint, prepared, derivation)

    loaded = load_hub_bundle(bundle)

    assert loaded.release == release
    assert _states_equal(prepared.model.state(), loaded.prepared.model.state())


def test_same_class_roles_restore_their_distinct_weights_without_swapping(
    tmp_path: Path,
) -> None:
    document = json.loads(PACKAGED_CODEC(CONFIG).model_dump_json())
    model = cast("dict[str, JsonValue]", document["model"])
    decoders = cast("dict[str, JsonValue]", model["decoders"])
    decoders["secondary"] = dict(cast("dict[str, JsonValue]", decoders["application"]))
    spaces = cast("dict[str, JsonValue]", model["spaces"])
    categories = cast("dict[str, JsonValue]", spaces["categories"])
    combinators = cast("dict[str, JsonValue]", categories["combinators"])
    roles = cast("dict[str, JsonValue]", combinators["decoders"])
    roles["application:\\"] = "secondary"
    run_spec = RunSpec.model_validate(document)
    blueprint = plan_training(run_spec)
    streams = RandomStreams(
        BUNDLE_DERIVATION.policy(accelerator=None, samplers=blueprint.sampler_streams)
    )
    streams.apply()
    prepared = prepare_training(
        blueprint, streams, resolve_device_policy(blueprint.run_spec)
    )
    primary = next(prepared.model.modules["application"].parameters())
    secondary = next(prepared.model.modules["secondary"].parameters())

    with torch.no_grad():
        primary.fill_(0.125)
        secondary.fill_(0.875)

    bundle = tmp_path / "bundle"
    export_hub_model(
        bundle, _release(blueprint), blueprint, prepared, BUNDLE_DERIVATION
    )
    loaded = load_hub_bundle(bundle)
    loaded_primary = next(loaded.prepared.model.modules["application"].parameters())
    loaded_secondary = next(loaded.prepared.model.modules["secondary"].parameters())

    assert not torch.equal(primary, secondary)
    assert torch.equal(loaded_primary, primary)
    assert torch.equal(loaded_secondary, secondary)


def test_a_span_decoder_role_is_fingerprinted_and_restored_as_a_span_decoder(
    tmp_path: Path,
) -> None:
    document = json.loads(PACKAGED_CODEC(CONFIG).model_dump_json())
    model = cast("dict[str, JsonValue]", document["model"])
    model["span_decoders"] = {"reconstruction": {"kind": "rnn-span-decoder"}}
    run_spec = RunSpec.model_validate(document)
    blueprint = plan_training(run_spec)
    streams = RandomStreams(
        BUNDLE_DERIVATION.policy(accelerator=None, samplers=blueprint.sampler_streams)
    )
    streams.apply()
    prepared = prepare_training(
        blueprint, streams, resolve_device_policy(blueprint.run_spec)
    )
    weight = next(prepared.model.modules["reconstruction"].parameters())

    with torch.no_grad():
        weight.fill_(0.375)

    bundle = tmp_path / "bundle"
    export_hub_model(
        bundle, _release(blueprint), blueprint, prepared, BUNDLE_DERIVATION
    )
    components = cast(
        "dict[str, dict[str, JsonValue]]", _manifest(bundle)["components"]
    )
    fingerprints = blueprint.component_fingerprints
    loaded = load_hub_bundle(bundle)

    assert components["reconstruction"]["tag"] == "rnn-span-decoder"
    assert (
        components["reconstruction"]["spec_fingerprint"]
        == (fingerprints["model.span_decoders.reconstruction"])
    )
    assert (
        components["application"]["spec_fingerprint"]
        == (fingerprints["model.decoders.application"])
    )
    assert torch.equal(
        next(loaded.prepared.model.modules["reconstruction"].parameters()), weight
    )
    assert _states_equal(prepared.model.state(), loaded.prepared.model.state())


def test_pretrained_bundle_carries_construction_assets_without_a_weight_copy(
    tmp_path: Path,
) -> None:
    blueprint, prepared, assets = _pretrained(tmp_path)
    ordinary = _release(blueprint)
    release = ModelRelease(
        ordinary.model_name,
        ordinary.datasets,
        ordinary.languages,
        ordinary.provenance,
        grammar_identity=ordinary.grammar_identity,
        codec_identity=ordinary.codec_identity,
        action_set_identity=ordinary.action_set_identity,
        corpus_revision=ordinary.corpus_revision,
        pretrained_resources={"token_encoder": assets},
    )
    bundle = tmp_path / "bundle"
    export_hub_model(bundle, release, blueprint, prepared, BUNDLE_DERIVATION)
    resources = bundle / "resources/000000"
    copied = {
        path.relative_to(resources).as_posix()
        for path in resources.rglob("*")
        if path.is_file()
    }

    assert {"config.json", "tokenizer.json"} <= copied
    assert not any(
        Path(path).suffix
        in {".bin", ".h5", ".msgpack", ".onnx", ".pt", ".pth", ".safetensors"}
        for path in copied
    )

    loaded = load_hub_bundle(bundle)

    assert _states_equal(prepared.model.state(), loaded.prepared.model.state())
    assert loaded.release.pretrained_resources == {"token_encoder": resources}


@pytest.mark.parametrize("existing", ["file", "directory"])
def test_an_existing_destination_is_refused_without_mutation(
    tmp_path: Path, existing: str
) -> None:
    blueprint, prepared = _prepared()
    destination = tmp_path / "bundle"

    if existing == "file":
        destination.write_text("sentinel", encoding="utf-8", newline="\n")
    else:
        destination.mkdir()
        (destination / "sentinel").write_text("kept", encoding="utf-8", newline="\n")

    before = tuple(
        (path.relative_to(tmp_path), path.read_bytes())
        for path in tmp_path.rglob("*")
        if path.is_file()
    )

    with pytest.raises(FileExistsError, match="already exists"):
        export_hub_model(
            destination,
            _release(blueprint),
            blueprint,
            prepared,
            BUNDLE_DERIVATION,
        )

    after = tuple(
        (path.relative_to(tmp_path), path.read_bytes())
        for path in tmp_path.rglob("*")
        if path.is_file()
    )
    assert after == before


def test_release_identity_mismatch_precedes_destination_creation(
    tmp_path: Path,
) -> None:
    blueprint, prepared = _prepared()
    destination = tmp_path / "bundle"
    wrong = "0" * 64 if blueprint.grammar_fingerprint != "0" * 64 else "1" * 64

    with pytest.raises(ValueError, match="grammar identity"):
        export_hub_model(
            destination,
            _release(blueprint, grammar_identity=wrong),
            blueprint,
            prepared,
            BUNDLE_DERIVATION,
        )

    assert not destination.exists()


def test_a_resource_for_a_nonpretrained_role_is_refused_before_destination(
    tmp_path: Path,
) -> None:
    blueprint, prepared = _prepared()
    resource = tmp_path / "resource"
    resource.mkdir()
    (resource / "config.json").write_text("{}", encoding="utf-8", newline="\n")
    ordinary = _release(blueprint)
    release = ModelRelease(
        ordinary.model_name,
        ordinary.datasets,
        ordinary.languages,
        ordinary.provenance,
        grammar_identity=ordinary.grammar_identity,
        codec_identity=ordinary.codec_identity,
        action_set_identity=ordinary.action_set_identity,
        corpus_revision=ordinary.corpus_revision,
        pretrained_resources={"token_encoder": resource},
    )
    destination = tmp_path / "bundle"

    with pytest.raises(ValueError, match="built-in pretrained construction settings"):
        export_hub_model(destination, release, blueprint, prepared, BUNDLE_DERIVATION)

    assert not destination.exists()


@pytest.mark.parametrize(
    ("field", "identity"),
    [
        ("grammar_identity", "grammar"),
        ("codec_identity", "codec"),
        ("action_set_identity", "action set"),
        ("corpus_revision", "corpus"),
    ],
)
def test_a_changed_release_identity_is_named_before_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    identity: str,
) -> None:
    bundle, _, _ = _export(tmp_path)
    record = _manifest(bundle)
    release = cast("dict[str, JsonValue]", record["release"])
    release[field] = "0" * 64 if field != "corpus_revision" else "another-corpus"
    _write_manifest(bundle, record)

    def refuse(*_args: Never, **_kwargs: Never) -> None:
        pytest.fail("release identity mismatch reached model allocation")

    monkeypatch.setattr(torch.nn.Module, "__init__", refuse)

    with pytest.raises(ValueError, match=rf"stored {identity} identity"):
        load_hub_bundle(bundle)


def test_a_file_digest_mismatch_is_named_before_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, _, _ = _export(tmp_path)
    card = bundle / "README.md"
    card.write_text(
        card.read_text(encoding="utf-8") + "\nchanged\n", encoding="utf-8", newline="\n"
    )

    def refuse(*_args: Never, **_kwargs: Never) -> None:
        pytest.fail("file digest mismatch reached model allocation")

    monkeypatch.setattr(torch.nn.Module, "__init__", refuse)

    with pytest.raises(ValueError, match=r"README\.md.*digests to .*not stored"):
        load_hub_bundle(bundle)


def test_a_component_fingerprint_mismatch_is_named_before_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, _, _ = _export(tmp_path)
    record = _manifest(bundle)
    components = cast("dict[str, JsonValue]", record["components"])
    role = next(iter(components))
    component = cast("dict[str, JsonValue]", components[role])
    component["spec_fingerprint"] = "0" * 64
    _write_manifest(bundle, record)

    def refuse(*_args: Never, **_kwargs: Never) -> None:
        pytest.fail("component fingerprint mismatch reached model allocation")

    monkeypatch.setattr(torch.nn.Module, "__init__", refuse)

    with pytest.raises(ValueError, match=rf"component role {role!r} stores"):
        load_hub_bundle(bundle)


def test_an_extra_component_role_is_named_before_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, _, _ = _export(tmp_path)
    record = _manifest(bundle)
    components = cast("dict[str, JsonValue]", record["components"])
    components["extra-role"] = next(iter(components.values()))
    _write_manifest(bundle, record)

    def refuse(*_args: Never, **_kwargs: Never) -> None:
        pytest.fail("role mismatch reached model allocation")

    monkeypatch.setattr(torch.nn.Module, "__init__", refuse)

    with pytest.raises(ValueError, match=r"roles lack \[\] and add \['extra-role'\]"):
        load_hub_bundle(bundle)


def test_a_missing_component_role_is_named_before_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, _, _ = _export(tmp_path)
    record = _manifest(bundle)
    components = cast("dict[str, JsonValue]", record["components"])
    role = next(iter(components))
    component = cast("dict[str, JsonValue]", components.pop(role))
    state = cast("dict[str, JsonValue]", component["state"])
    (bundle / cast("str", state["path"])).unlink()
    _write_manifest(bundle, record)

    def refuse(*_args: Never, **_kwargs: Never) -> None:
        pytest.fail("missing role reached model allocation")

    monkeypatch.setattr(torch.nn.Module, "__init__", refuse)

    with pytest.raises(ValueError, match=rf"model component roles lack \['{role}'\]"):
        load_hub_bundle(bundle)


def test_an_extra_state_key_is_named_before_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, _, _ = _export(tmp_path)
    record = _manifest(bundle)
    components = cast("dict[str, JsonValue]", record["components"])
    role = next(iter(components))
    component = cast("dict[str, JsonValue]", components[role])
    state_record = cast("dict[str, JsonValue]", component["state"])
    state_path = bundle / cast("str", state_record["path"])
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    assert isinstance(state, dict)
    state["extra-key"] = torch.zeros(1)
    torch.save(state, state_path)

    with state_path.open("rb") as handle:
        state_record["digest"] = file_digest(handle, "sha256").hexdigest()

    _write_manifest(bundle, record)

    def refuse(*_args: Never, **_kwargs: Never) -> None:
        pytest.fail("state-key mismatch reached model allocation")

    monkeypatch.setattr(torch.nn.Module, "__init__", refuse)

    with pytest.raises(
        ValueError, match=r"state keys lack \[\] and add \['extra-key'\]"
    ):
        load_hub_bundle(bundle)


def test_a_missing_state_key_is_named_before_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, _, _ = _export(tmp_path)
    record = _manifest(bundle)
    components = cast("dict[str, JsonValue]", record["components"])
    role = next(iter(components))
    component = cast("dict[str, JsonValue]", components[role])
    state_record = cast("dict[str, JsonValue]", component["state"])
    state_path = bundle / cast("str", state_record["path"])
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    assert isinstance(state, dict)
    key = next(iter(state))
    del state[key]
    torch.save(state, state_path)

    with state_path.open("rb") as handle:
        state_record["digest"] = file_digest(handle, "sha256").hexdigest()

    _write_manifest(bundle, record)

    def refuse(*_args: Never, **_kwargs: Never) -> None:
        pytest.fail("missing state key reached model allocation")

    monkeypatch.setattr(torch.nn.Module, "__init__", refuse)

    with pytest.raises(ValueError, match=rf"state keys lack \['{re.escape(key)}'\]"):
        load_hub_bundle(bundle)


@pytest.mark.parametrize(("field", "value"), [("shape", [999]), ("dtype", "float64")])
def test_state_shape_and_dtype_mismatches_are_named_before_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: JsonValue,
) -> None:
    bundle, _, _ = _export(tmp_path)
    record = _manifest(bundle)
    components = cast("dict[str, JsonValue]", record["components"])
    role = next(iter(components))
    component = cast("dict[str, JsonValue]", components[role])
    tensors = cast("dict[str, JsonValue]", component["tensors"])
    key = next(
        key
        for key, raw in tensors.items()
        if cast("dict[str, JsonValue]", raw)["kind"] == "tensor"
    )
    tensor = cast("dict[str, JsonValue]", tensors[key])
    tensor[field] = value
    _write_manifest(bundle, record)

    def refuse(*_args: Never, **_kwargs: Never) -> None:
        pytest.fail("state description mismatch reached model allocation")

    monkeypatch.setattr(torch.nn.Module, "__init__", refuse)

    with pytest.raises(ValueError, match=rf"state key {key!r} has {field}"):
        load_hub_bundle(bundle)


def test_export_and_load_open_no_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blueprint, prepared = _prepared()
    monkeypatch.setattr(socket, "socket", _refusing(socket.socket))
    monkeypatch.setattr(
        socket, "create_connection", _refusing(socket.create_connection)
    )
    bundle = tmp_path / "bundle"
    export_hub_model(
        bundle, _release(blueprint), blueprint, prepared, BUNDLE_DERIVATION
    )
    assert load_hub_bundle(bundle).release.model_name == "lofi-synthetic-parser"


RELEASE: Final[dict[str, JsonValue]] = {
    "model_name": "lofi-synthetic-parser",
    "datasets": ["lofi/synthetic-parser"],
    "languages": ["en"],
    "source_method": "generated",
    "license_expression": "CC-BY-4.0",
    "licenses": [{"identifier": "CC-BY-4.0", "applies_to": "generated-corpus"}],
    "eprint_refs": [EPRINT],
    "code_uri": "https://github.com/logical-form-induction/lofi",
    "command": "lofi run --config synthetic-parser.yaml --experiment-seed 7",
    "environment": "uv",
}


def _trained_point(root: Path, /) -> Path:
    """Train the packaged synthetic parser through the command line."""
    status = main(
        [
            "run",
            "--config",
            str(CONFIG),
            "--seed",
            "3",
            "--experiment-seed",
            "7",
            "--root",
            str(root),
        ]
    )
    (run,) = tuple((root / "runs").glob("[0-9a-f]" * 16))

    assert status == 0

    return run


def _release_document(root: Path, /, **changed: JsonValue) -> Path:
    path = root / "release.json"
    path.write_text(json.dumps(RELEASE | changed), encoding="utf-8", newline="\n")

    return path


def _held_out_scores(prepared: PreparedTraining, /) -> tuple[torch.Tensor, ...]:
    """Return the item scores the forward pass publishes over the held-out split."""
    plan = prepared.plan.parser_plan()
    forward = plan.forward
    scores: list[torch.Tensor] = []

    if not isinstance(forward, ChartForward):
        msg = "the parser release fixture has no chart forward pass"
        raise TypeError(msg)

    for module in prepared.model.modules.values():
        module.eval()

    with torch.no_grad():
        for batch in plan.fit_batches:
            if not isinstance(batch, SyntheticTrainingBatch):
                msg = "the parser release fixture has no sentence batch"
                raise TypeError(msg)

            published = forward.publish(batch).scores
            assert published is not None
            scores.append(published.tensor.detach().cpu())

    return tuple(scores)


def _trained_model(run: Path, /) -> tuple[TrainingBlueprint, PreparedTraining]:
    """Rebuild a trained point from the checkpoint its run releases."""
    manifest = read_manifest(run)
    assert manifest is not None
    blueprint = plan_training(manifest.run_spec)
    streams = RandomStreams(manifest.seed_policy.policy())
    streams.apply()
    prepared = prepare_training(
        blueprint, streams, resolve_device_policy(manifest.run_spec)
    )
    restore_checkpoint(
        run / "checkpoints",
        blueprint,
        prepared,
        manifest.run_spec.release.checkpoint,
    )

    return blueprint, prepared


def test_checkpoint_restoration_selects_the_newest_checkpoint_of_each_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _trained_point(tmp_path / "store")
    manifest = read_manifest(run)
    assert manifest is not None
    blueprint = plan_training(manifest.run_spec)
    streams = RandomStreams(manifest.seed_policy.policy())
    streams.apply()
    prepared = prepare_training(
        blueprint, streams, resolve_device_policy(manifest.run_spec)
    )
    store = DirectoryStore(run / "checkpoints")
    expected = {
        kind: tuple(
            handle
            for handle in store.list()
            if (fitting := store.manifest(handle).fitting) is not None
            and fitting.kind == kind
        )[-1]
        for kind in ("validation-best", "resume-latest")
    }
    loaded: list[tuple[str, str]] = []
    original = DirectoryStore.load

    def record_load(
        current: DirectoryStore, checkpoint: CheckpointId, template: CheckpointManifest
    ) -> Checkpoint[StateTree]:
        fitting = template.fitting
        assert fitting is not None
        loaded.append((str(checkpoint), cast("str", fitting.kind)))
        return original(current, checkpoint, template)

    monkeypatch.setattr(DirectoryStore, "load", record_load)

    restore_checkpoint(run / "checkpoints", blueprint, prepared, "validation-best")
    restore_checkpoint(run / "checkpoints", blueprint, prepared, "resume-latest")

    assert loaded == [
        (str(expected["validation-best"]), "validation-best"),
        (str(expected["resume-latest"]), "resume-latest"),
    ]


def test_the_export_command_bundles_the_packaged_trained_point_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _trained_point(tmp_path / "store")
    trained = read_manifest(run)
    bundle = tmp_path / "bundle"
    monkeypatch.setattr(socket, "socket", _refusing(socket.socket))
    monkeypatch.setattr(
        socket, "create_connection", _refusing(socket.create_connection)
    )

    status = main(
        [
            "export",
            "--run",
            str(run),
            "--release",
            str(_release_document(tmp_path)),
            "--out",
            str(bundle),
        ]
    )
    reloaded = plan_training(RunSpecCodec()(bundle / "config.json"))
    loaded = load_hub_bundle(bundle)
    source_blueprint, source = _trained_model(run)
    reproducibility = loaded.release.provenance.reproducibility

    assert status == 0
    assert trained is not None
    assert trained.status is ManifestStatus.COMPLETED
    assert reloaded.grammar_fingerprint == source_blueprint.grammar_fingerprint
    assert loaded.release.grammar_identity == source_blueprint.grammar_fingerprint
    assert reloaded.declaration_fingerprint == source_blueprint.declaration_fingerprint
    assert reproducibility is not None
    assert reproducibility.commit_hash == trained.git_revision
    assert reproducibility.random_seed == trained.seed_policy.policy().root
    assert reproducibility.command == RELEASE["command"]
    assert loaded.release.model_name == "lofi-synthetic-parser"

    source_scores = _held_out_scores(source)
    bundled_scores = _held_out_scores(loaded.prepared)

    assert len(source_scores) == len(bundled_scores) > 0
    assert all(
        torch.equal(left, right)
        for left, right in zip(source_scores, bundled_scores, strict=True)
    )
    assert _states_equal(source.model.state(), loaded.prepared.model.state())


def test_a_fresh_process_exports_a_completed_point(tmp_path: Path) -> None:
    run = _trained_point(tmp_path / "store")
    bundle = tmp_path / "bundle"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lofi.run",
            "export",
            "--run",
            str(run),
            "--release",
            str(_release_document(tmp_path)),
            "--out",
            str(bundle),
        ],
        capture_output=True,
        encoding="utf-8",
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert load_hub_bundle(bundle).release.model_name == "lofi-synthetic-parser"


def test_a_fresh_process_restores_a_completed_point(tmp_path: Path) -> None:
    run = _trained_point(tmp_path / "store")
    program = """
import sys
from pathlib import Path

from lofi.run.hub import restore_trained_point

prepared = restore_trained_point(Path(sys.argv[1]))
print(sorted(prepared.model.modules))
"""

    result = subprocess.run(
        [sys.executable, "-c", program, str(run)],
        capture_output=True,
        encoding="utf-8",
        check=False,
    )
    _, prepared = _trained_model(run)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(sorted(prepared.model.modules))


def test_the_export_command_refuses_a_run_that_did_not_complete(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    unfinished = tmp_path / "runs" / "0123456789abcdef"
    unfinished.mkdir(parents=True)

    status = main(
        [
            "export",
            "--run",
            str(unfinished),
            "--release",
            str(_release_document(tmp_path)),
            "--out",
            str(tmp_path / "bundle"),
        ]
    )

    assert status == 2
    assert "records no completed run" in capsys.readouterr().err
    assert not (tmp_path / "bundle").exists()


def test_a_release_declaring_an_unknown_field_is_refused_before_any_bundle(
    tmp_path: Path,
) -> None:
    run = _trained_point(tmp_path / "store")

    with pytest.raises(ConfigError):
        export_trained_point(
            run,
            _release_document(tmp_path, revision="main"),
            tmp_path / "bundle",
        )

    assert not (tmp_path / "bundle").exists()


def test_a_release_naming_a_pretrained_role_the_point_lacks_is_refused(
    tmp_path: Path,
) -> None:
    run = _trained_point(tmp_path / "store")
    assets = tmp_path / "assets"
    assets.mkdir()

    with pytest.raises(ValueError, match="absent component roles"):
        export_trained_point(
            run,
            _release_document(
                tmp_path, pretrained_resources={"not-a-role": str(assets)}
            ),
            tmp_path / "bundle",
        )

    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize("status", [ManifestStatus.RUNNING, ManifestStatus.KILLED])
def test_a_run_whose_manifest_is_not_completed_is_not_exported(
    tmp_path: Path, status: ManifestStatus
) -> None:
    store = RunStore(tmp_path / "store", freeze_id=None, seed=EXPERIMENT_SEED)
    point = a_point()
    record_manifest(store, point, status)

    with pytest.raises(ValueError, match="records no completed run"):
        export_trained_point(
            store.directory(point), _release_document(tmp_path), tmp_path / "bundle"
        )

    assert not (tmp_path / "bundle").exists()


def test_a_checkpoint_written_for_another_declaration_is_not_exported(
    tmp_path: Path,
) -> None:
    run = _trained_point(tmp_path / "store")
    recorded = run / "manifest.json"
    document = json.loads(recorded.read_text(encoding="utf-8"))
    document["run_spec"]["optimizer"]["learning_rate"] = 0.5
    recorded.write_text(json.dumps(document), encoding="utf-8", newline="\n")

    with pytest.raises(ValueError, match="belongs to the run"):
        export_trained_point(run, _release_document(tmp_path), tmp_path / "bundle")

    assert not (tmp_path / "bundle").exists()


RELEASE_CONFIG = (
    Path(__file__).resolve().parents[3] / "lofi/configs/run/synthetic-parser.yaml"
)
CODEC: Final = RunSpecCodec(search_path=(PACKAGED_CONFIG_ROOT,))
DERIVATION = ExperimentSeed(317).point("hub-release", 0)
SEED = DERIVATION.root
LICENSE = "CC-BY-4.0"


class ReleaseWorkspace(NamedTuple):
    root: Path
    bundle: Path
    validation: Path
    resume: Path


class FakeHub:
    def __init__(
        self,
        root: Path,
        *,
        evidence: Path | None = None,
        exists: bool = False,
        corrupt_authenticated: bool = False,
        stale_authenticated: bool = False,
        corrupt_anonymous: bool = False,
        mutate_staging: bool = False,
        publicize_then_raise: bool = False,
        privatize_failure: bool = False,
        revision: str = "a" * 40,
    ) -> None:
        self.root = root
        self.evidence = evidence
        self.exists = exists
        self.corrupt_authenticated = corrupt_authenticated
        self.stale_authenticated = stale_authenticated
        self.corrupt_anonymous = corrupt_anonymous
        self.mutate_staging = mutate_staging
        self.publicize_then_raise = publicize_then_raise
        self.privatize_failure = privatize_failure
        self.revision = revision
        self.private = False
        self.events: list[tuple[object, ...]] = []
        self.uploads: list[tuple[Path, str, str]] = []
        self.authenticated_downloads: list[tuple[str, str, str, Path]] = []
        self.anonymous_downloads: list[tuple[str, str, Path]] = []
        self.uploaded_paths: tuple[str, ...] = ()

    def require_absent(self, repo_id: str, token: str, /) -> None:
        reservation = None

        if self.evidence is not None and self.evidence.is_file():
            reservation = json.loads(self.evidence.read_text(encoding="utf-8"))[
                "format"
            ]

        self.events.append(("require-absent", repo_id, token, reservation))

        if self.exists:
            msg = f"repository already exists: {repo_id}"
            raise FileExistsError(msg)

    def create_private(self, repo_id: str, token: str, /) -> None:
        self.events.append(("create-private", repo_id, token))

        if self.exists:
            msg = f"repository already exists: {repo_id}"
            raise FileExistsError(msg)

        self.root.mkdir()
        self.exists = True
        self.private = True

    def upload_private(self, bundle: Path, repo_id: str, token: str, /) -> str:
        self.events.append(("upload-private", bundle, repo_id, token))
        self.uploads.append((bundle, repo_id, token))
        self.uploaded_paths = tuple(
            path.relative_to(bundle).as_posix()
            for path in sorted(bundle.rglob("*"))
            if path.is_file()
        )
        shutil.copytree(bundle, self.root / self.revision)

        if self.mutate_staging:
            readme = bundle / "README.md"
            readme.chmod(0o644)
            readme.write_text("changed\n", encoding="utf-8", newline="\n")

        return self.revision

    def download_authenticated(
        self,
        repo_id: str,
        revision: str,
        token: str,
        destination: Path,
        /,
    ) -> Path:
        self.events.append(
            ("download-authenticated", repo_id, revision, token, destination)
        )
        self.authenticated_downloads.append((repo_id, revision, token, destination))
        shutil.copytree(self.root / revision, destination)

        if self.corrupt_authenticated:
            readme = destination / "README.md"
            readme.chmod(0o644)
            readme.write_text("changed\n", encoding="utf-8", newline="\n")

        if self.stale_authenticated:
            (destination / "stale.bin").write_bytes(b"stale")

        return destination

    def publicize(self, repo_id: str, token: str, /) -> None:
        self.events.append(("publicize", repo_id, token))
        self.private = False

        if self.publicize_then_raise:
            msg = "public visibility changed before the API failed"
            raise RuntimeError(msg)

    def download_anonymous(
        self,
        repo_id: str,
        revision: str,
        destination: Path,
        /,
    ) -> Path:
        self.events.append(("download-anonymous", repo_id, revision, destination))
        self.anonymous_downloads.append((repo_id, revision, destination))
        shutil.copytree(self.root / revision, destination)

        if self.corrupt_anonymous:
            readme = destination / "README.md"
            readme.chmod(0o644)
            readme.write_text("changed\n", encoding="utf-8", newline="\n")

        return destination

    def privatize(self, repo_id: str, token: str, /) -> None:
        self.events.append(("privatize", repo_id, token))

        if self.privatize_failure:
            msg = "private visibility could not be restored"
            raise OSError(msg)

        self.private = True


def _components(
    blueprint: TrainingBlueprint,
    prepared: PreparedTraining,
    /,
) -> dict[str, ComponentRecord]:
    return {
        role: ComponentRecord(
            tag=role,
            spec_fingerprint=blueprint.component_fingerprints[
                blueprint.run_spec.model.checkpointed_paths()[role]
            ],
            signature_fingerprint=blueprint.codec_fingerprint,
            codec_identity=blueprint.grammar_fingerprint,
            state=StateReference(key=role, digest="0" * 64),
            labels=(
                None
                if role not in prepared.model.checkpoint_labels
                else dict(prepared.model.checkpoint_labels[role])
            ),
        )
        for role in prepared.model.modules
    }


def _release_checkpoint_manifest(
    blueprint: TrainingBlueprint,
    prepared: PreparedTraining,
    streams: RandomStreams,
    fitting: ParserFittingRecord,
    /,
) -> CheckpointManifest:
    plan = cast("ParserFittingPlan", prepared.plan)

    return CheckpointManifest(
        run_id=blueprint.declaration_fingerprint,
        step=fitting.completed_epochs * plan.steps_per_epoch,
        code_version="test",
        action_set="actions-a1",
        components=_components(blueprint, prepared),
        rng=RngRecord(streams=dict(streams.snapshot())),
        precision=PrecisionRecord(
            compute_dtype="float32",
            autocast_device_type="cpu",
            gradient_scaling=False,
        ),
        data_position=fitting.completed_epochs * plan.items_per_epoch,
        fitting=fitting,
        optimizer=StateReference(key="optimizer", digest="0" * 64),
    )


def a_run_manifest(
    derivation: PointSeed, run_spec: RunSpec | None = None
) -> ExperimentManifest:
    if run_spec is None:
        run_spec = CODEC(RELEASE_CONFIG)

    blueprint = plan_training(run_spec)

    return close_manifest(
        open_manifest(
            run_spec=run_spec,
            component_fingerprints=blueprint.component_fingerprints,
            grammar_fingerprint=blueprint.grammar_fingerprint,
            signature_fingerprint=blueprint.codec_fingerprint,
            corpus_revision=blueprint.corpus_revision,
            seed_policy=derivation.policy(
                accelerator=None, samplers=blueprint.sampler_streams
            ),
            seed_derivation=derivation,
            package_versions={"torch": "test"},
            git_revision="0123456789abcdef",
            accelerator="cpu",
        ),
        ManifestStatus.COMPLETED,
    )


def _states(prepared: PreparedTraining, /) -> Mapping[str, StateTree]:
    states = cast("dict[str, StateTree]", dict(prepared.model.state()))
    states["optimizer"] = {}

    return states


def _workspace(root: Path, checkpoint: ReleaseCheckpoint, /) -> ReleaseWorkspace:
    run_spec = CODEC(RELEASE_CONFIG).with_(release=ReleaseSpec(checkpoint=checkpoint))
    blueprint = plan_training(run_spec)
    streams = RandomStreams(
        DERIVATION.policy(accelerator=None, samplers=blueprint.sampler_streams)
    )
    streams.apply()
    write_manifest(root, a_run_manifest(DERIVATION, run_spec))
    prepared = prepare_training(
        blueprint,
        streams,
        resolve_device_policy(run_spec),
    )
    store = DirectoryStore(root / "checkpoints")
    validation_fitting = ParserFittingRecord(
        kind="validation-best",
        completed_epochs=1,
        metric="loss/grammaticality",
        best=1.0,
        failures=0,
        best_checkpoint=None,
    )
    validation_id = store.save(
        _release_checkpoint_manifest(blueprint, prepared, streams, validation_fitting),
        _states(prepared),
    )
    resume_fitting = ParserFittingRecord(
        kind="resume-latest",
        completed_epochs=2,
        metric="loss/grammaticality",
        best=1.0,
        failures=1,
        best_checkpoint=str(validation_id),
    )
    resume_id = store.save(
        _release_checkpoint_manifest(blueprint, prepared, streams, resume_fitting),
        _states(prepared),
    )
    release = ModelRelease(
        "lofi-release-test",
        ("lofi/synthetic",),
        ("en",),
        Provenance(
            "generated",
            Licensing(LICENSE, (LicenseRef(LICENSE, "synthetic"),)),
            (),
            Reproducibility(
                "https://github.com/logical-form-induction/lofi",
                "0123456789abcdef",
                "lofi run",
                "test",
                SEED,
            ),
        ),
        grammar_identity=blueprint.grammar_fingerprint,
        codec_identity=blueprint.codec_fingerprint,
        action_set_identity=blueprint.action_set_fingerprint,
        corpus_revision=blueprint.corpus_revision,
        pretrained_resources={},
    )
    bundle = root / "bundle"
    export_hub_model(bundle, release, blueprint, prepared, DERIVATION)

    return ReleaseWorkspace(
        root,
        bundle,
        store.root / validation_id,
        store.root / resume_id,
    )


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> ReleaseWorkspace:
    return _workspace(tmp_path_factory.mktemp("hub-release"), "validation-best")


@pytest.fixture(scope="module")
def resume_workspace(tmp_path_factory: pytest.TempPathFactory) -> ReleaseWorkspace:
    return _workspace(tmp_path_factory.mktemp("hub-release-resume"), "resume-latest")


def _prepare(
    workspace: ReleaseWorkspace, *, bundle: Path | None = None
) -> hub_module.PreparedRelease:
    return hub_module.prepare_release(
        workspace.bundle if bundle is None else bundle,
        "factslab/lofi-release-test",
        workspace.root,
    )


def _authorization(**changes: object) -> ReleaseAuthorization:
    values: dict[str, object] = {
        "reviewer": "Authorized Reviewer",
        "license_identifier": LICENSE,
        "corpus_redistribution_confirmed": True,
        "artifact_redistribution_confirmed": True,
    }
    values.update(changes)

    return ReleaseAuthorization(
        cast("str", values["reviewer"]),
        cast("str", values["license_identifier"]),
        cast("bool", values["corpus_redistribution_confirmed"]),
        cast("bool", values["artifact_redistribution_confirmed"]),
    )


def _copied_checkpoints(
    workspace: ReleaseWorkspace,
    destination: Path,
    /,
) -> tuple[Path, Path]:
    copied = destination / "checkpoints"
    shutil.copytree(workspace.validation.parent, copied)
    shutil.copy(workspace.root / MANIFEST_NAME, destination / MANIFEST_NAME)

    return copied / workspace.validation.name, copied / workspace.resume.name


def _rewrite_manifest(
    checkpoint: Path,
    /,
    *,
    fields: Mapping[str, object] | None = None,
    fitting_fields: Mapping[str, object] | None = None,
) -> None:
    path = checkpoint / "manifest.json"
    record = cast(
        "dict[str, object]",
        json.loads(path.read_text(encoding="utf-8")),
    )

    if fields is not None:
        record.update(fields)

    if fitting_fields is not None:
        fitting = cast("dict[str, object]", record["fitting"])
        fitting.update(fitting_fields)

    path.write_text(
        f"{json.dumps(record, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
        newline="\n",
    )


def test_preparation_proves_the_bundle_matches_validation_best_bitwise(
    workspace: ReleaseWorkspace,
) -> None:
    prepared = _prepare(workspace)

    assert prepared.repo_id == "factslab/lofi-release-test"
    assert prepared.run == workspace.root
    assert prepared.checkpoint == workspace.validation
    assert prepared.checkpoint_id == workspace.validation.name
    assert prepared.model_name == "lofi-release-test"
    assert prepared.license_identifier == LICENSE
    assert prepared.seed == SEED
    assert prepared.held_out.items == 1
    assert len(prepared.held_out.sha256) == 64
    bundle_manifest = cast(
        "dict[str, object]",
        json.loads((prepared.bundle / "manifest.json").read_text(encoding="utf-8")),
    )
    components = cast("dict[str, dict[str, object]]", bundle_manifest["components"])
    state_paths = {
        cast("str", cast("dict[str, object]", component["state"])["path"])
        for component in components.values()
    }
    paths = {record.path for record in prepared.files}

    assert paths == {"README.md", "config.json", "manifest.json", *state_paths}


def test_preparation_uses_the_run_declared_resume_checkpoint(
    resume_workspace: ReleaseWorkspace,
) -> None:
    prepared = _prepare(resume_workspace)

    assert prepared.run == resume_workspace.root
    assert prepared.checkpoint == resume_workspace.resume
    assert prepared.checkpoint_id == resume_workspace.resume.name


def test_preparation_refuses_a_checkpoint_metric_other_than_the_run_spec(
    workspace: ReleaseWorkspace,
    tmp_path: Path,
) -> None:
    _, resume = _copied_checkpoints(workspace, tmp_path)
    _rewrite_manifest(resume, fitting_fields={"metric": "loss/other"})

    with pytest.raises(ValueError, match="not configured metric"):
        hub_module.prepare_release(
            workspace.bundle,
            "factslab/lofi-release-test",
            tmp_path,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("step", 1), ("data_position", 3)],
)
def test_preparation_refuses_a_checkpoint_position_inconsistent_with_its_epochs(
    workspace: ReleaseWorkspace,
    tmp_path: Path,
    field: str,
    value: int,
) -> None:
    _, resume = _copied_checkpoints(workspace, tmp_path)
    _rewrite_manifest(resume, fields={field: value})

    with pytest.raises(ValueError, match="implies step/data position"):
        hub_module.prepare_release(
            workspace.bundle,
            "factslab/lofi-release-test",
            tmp_path,
        )


def test_preparation_refuses_a_latest_checkpoint_from_a_nonterminal_run(
    workspace: ReleaseWorkspace,
    tmp_path: Path,
) -> None:
    _, resume = _copied_checkpoints(workspace, tmp_path)
    _rewrite_manifest(
        resume,
        fields={"step": 1, "data_position": 3},
        fitting_fields={"completed_epochs": 1, "failures": 0},
    )

    with pytest.raises(ValueError, match="resume-latest is not terminal"):
        hub_module.prepare_release(
            workspace.bundle,
            "factslab/lofi-release-test",
            tmp_path,
        )


def test_preparation_refuses_a_directory_without_a_completed_run(
    workspace: ReleaseWorkspace,
    tmp_path: Path,
) -> None:
    _copied_checkpoints(workspace, tmp_path)
    (tmp_path / MANIFEST_NAME).unlink()

    with pytest.raises(ValueError, match="records no completed run"):
        hub_module.prepare_release(
            workspace.bundle,
            "factslab/lofi-release-test",
            tmp_path,
        )


def test_preparation_refuses_a_bundle_released_under_another_seed(
    workspace: ReleaseWorkspace,
    tmp_path: Path,
) -> None:
    _copied_checkpoints(workspace, tmp_path)
    other = ExperimentSeed(318).point("hub-release", 0)
    write_manifest(tmp_path, a_run_manifest(other))

    with pytest.raises(ValueError, match="not the run seed"):
        hub_module.prepare_release(
            workspace.bundle,
            "factslab/lofi-release-test",
            tmp_path,
        )


def test_read_only_command_prepares_without_calling_the_hub(
    workspace: ReleaseWorkspace,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = FakeHub(tmp_path / "remote")

    status = cli_main(
        [
            "release",
            "--bundle",
            str(workspace.bundle),
            "--repo-id",
            "factslab/lofi-release-test",
            "--run",
            str(workspace.root),
        ],
        release=hub_module.HubRelease(client),
        environ={},
    )

    output = json.loads(capsys.readouterr().out)
    assert status == 0
    assert output["repository"] == "factslab/lofi-release-test"
    assert output["held_out_items"] == 1
    assert client.events == []


def test_the_concrete_download_explicitly_disables_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cached = tmp_path / "cached"
    cached.mkdir()
    (cached / "manifest.json").write_text("{}\n", encoding="utf-8", newline="\n")
    calls: list[tuple[str, str, str, Path, bool, bool]] = []

    def snapshot(
        repo_id: str,
        *,
        repo_type: str,
        revision: str,
        cache_dir: Path,
        token: bool,
        force_download: bool,
    ) -> str:
        calls.append((repo_id, repo_type, revision, cache_dir, token, force_download))

        return str(cached)

    monkeypatch.setattr(hub_module, "snapshot_download", snapshot)
    destination = tmp_path / "clean"

    downloaded = hub_module.HuggingFaceHub().download_anonymous(
        "factslab/model",
        "b" * 40,
        destination,
    )

    assert calls == [
        (
            "factslab/model",
            "model",
            "b" * 40,
            tmp_path / ".clean.cache",
            False,
            True,
        )
    ]
    assert downloaded == destination
    assert (downloaded / "manifest.json").read_text(encoding="utf-8") == "{}\n"


def test_the_concrete_client_has_an_explicit_private_release_lifecycle(
    workspace: ReleaseWorkspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[object, ...]] = []

    class Commit(NamedTuple):
        oid: str

    class Api:
        def repo_info(
            self,
            repo_id: str,
            *,
            repo_type: str,
            token: str,
        ) -> None:
            events.append(("inspect", repo_id, repo_type, token))
            msg = "missing"
            raise RepositoryNotFoundError(
                msg,
                response=Response(
                    404,
                    request=Request("GET", "https://huggingface.co/api/models/x"),
                ),
            )

        def create_repo(
            self,
            repo_id: str,
            *,
            token: str,
            private: bool,
            repo_type: str,
            exist_ok: bool,
        ) -> None:
            events.append(("create", repo_id, token, private, repo_type, exist_ok))

        def upload_folder(
            self,
            *,
            repo_id: str,
            folder_path: Path,
            token: str,
            repo_type: str,
            commit_message: str,
        ) -> Commit:
            events.append(
                (
                    "upload",
                    repo_id,
                    folder_path,
                    token,
                    repo_type,
                    commit_message,
                )
            )

            return Commit("c" * 40)

        def update_repo_settings(
            self,
            repo_id: str,
            *,
            private: bool,
            token: str,
            repo_type: str,
        ) -> None:
            events.append(("visibility", repo_id, private, token, repo_type))

    monkeypatch.setattr(hub_module, "HfApi", Api)

    client = hub_module.HuggingFaceHub()
    client.require_absent("factslab/model", "secret")
    client.create_private("factslab/model", "secret")
    revision = client.upload_private(
        workspace.bundle,
        "factslab/model",
        "secret",
    )
    client.publicize("factslab/model", "secret")
    client.privatize("factslab/model", "secret")

    assert revision == "c" * 40
    assert events == [
        ("inspect", "factslab/model", "model", "secret"),
        ("create", "factslab/model", "secret", True, "model", False),
        (
            "upload",
            "factslab/model",
            workspace.bundle,
            "secret",
            "model",
            "Publish verified LoFI model bundle",
        ),
        ("visibility", "factslab/model", False, "secret", "model"),
        ("visibility", "factslab/model", True, "secret", "model"),
    ]


@pytest.mark.parametrize(
    ("authorization", "token", "message"),
    [
        (_authorization(reviewer=""), "secret", "reviewer"),
        (
            _authorization(license_identifier="MIT"),
            "secret",
            "does not match bundle license",
        ),
        (
            _authorization(corpus_redistribution_confirmed=False),
            "secret",
            "corpus redistribution",
        ),
        (
            _authorization(artifact_redistribution_confirmed=False),
            "secret",
            "artifact redistribution",
        ),
        (_authorization(), "", "credentials"),
    ],
)
def test_every_authorization_gate_precedes_the_first_remote_write(
    workspace: ReleaseWorkspace,
    tmp_path: Path,
    authorization: ReleaseAuthorization,
    token: str,
    message: str,
) -> None:
    client = FakeHub(tmp_path / "remote")

    with pytest.raises(ValueError, match=message):
        hub_module.publish_release(
            _prepare(workspace),
            authorization,
            client,
            token,
            tmp_path / "evidence.json",
        )

    assert client.events == []


@pytest.mark.parametrize("stale", ["evidence", "finalizing", "recovering"])
def test_stale_evidence_files_prevent_the_first_network_call(
    workspace: ReleaseWorkspace,
    tmp_path: Path,
    stale: str,
) -> None:
    evidence = tmp_path / "evidence.json"
    occupied = {
        "evidence": evidence,
        "finalizing": evidence.with_name(f".{evidence.name}.finalizing"),
        "recovering": evidence.with_name(f".{evidence.name}.recovering"),
    }[stale]
    occupied.write_text("stale\n", encoding="utf-8", newline="\n")
    client = FakeHub(tmp_path / "remote", evidence=evidence)

    with pytest.raises(FileExistsError, match="release evidence"):
        hub_module.publish_release(
            _prepare(workspace),
            _authorization(),
            client,
            "secret",
            evidence,
        )

    assert client.events == []
    assert occupied.read_text(encoding="utf-8") == "stale\n"


def test_an_existing_repository_is_refused_after_durable_evidence_reservation(
    workspace: ReleaseWorkspace,
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence.json"
    client = FakeHub(
        tmp_path / "remote",
        evidence=evidence,
        exists=True,
    )

    with pytest.raises(FileExistsError, match="already exists"):
        hub_module.publish_release(
            _prepare(workspace),
            _authorization(),
            client,
            "secret",
            evidence,
        )

    assert client.events == [
        (
            "require-absent",
            "factslab/lofi-release-test",
            "secret",
            "lofi-hub-release-reservation",
        )
    ]
    assert json.loads(evidence.read_text(encoding="utf-8"))["format"] == (
        "lofi-hub-release-reservation"
    )


def test_a_noncanonical_upload_revision_never_becomes_public(
    workspace: ReleaseWorkspace,
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence.json"
    client = FakeHub(
        tmp_path / "remote",
        evidence=evidence,
        revision="main",
    )

    with pytest.raises(ValueError, match="noncanonical immutable revision"):
        hub_module.publish_release(
            _prepare(workspace),
            _authorization(),
            client,
            "secret",
            evidence,
        )

    assert [event[0] for event in client.events] == [
        "require-absent",
        "create-private",
        "upload-private",
    ]
    assert client.private


def test_a_staged_byte_mutation_during_upload_never_becomes_public(
    workspace: ReleaseWorkspace,
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence.json"
    client = FakeHub(
        tmp_path / "remote",
        evidence=evidence,
        mutate_staging=True,
    )

    with pytest.raises(ValueError, match="staged bundle bytes changed"):
        hub_module.publish_release(
            _prepare(workspace),
            _authorization(),
            client,
            "secret",
            evidence,
        )

    assert [event[0] for event in client.events] == [
        "require-absent",
        "create-private",
        "upload-private",
    ]
    assert client.private


def test_stale_authenticated_remote_files_prevent_public_visibility(
    workspace: ReleaseWorkspace,
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence.json"
    client = FakeHub(
        tmp_path / "remote",
        evidence=evidence,
        stale_authenticated=True,
    )

    with pytest.raises(ValueError, match=r"add.*stale\.bin"):
        hub_module.publish_release(
            _prepare(workspace),
            _authorization(),
            client,
            "secret",
            evidence,
        )

    assert [event[0] for event in client.events] == [
        "require-absent",
        "create-private",
        "upload-private",
        "download-authenticated",
    ]
    assert client.private


def test_a_publicize_call_that_changes_visibility_then_raises_is_rolled_back(
    workspace: ReleaseWorkspace,
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence.json"
    client = FakeHub(
        tmp_path / "remote",
        evidence=evidence,
        publicize_then_raise=True,
    )

    with pytest.raises(RuntimeError, match="visibility changed"):
        hub_module.publish_release(
            _prepare(workspace),
            _authorization(),
            client,
            "secret",
            evidence,
        )

    assert [event[0] for event in client.events][-2:] == ["publicize", "privatize"]
    assert client.private
    assert json.loads(evidence.read_text(encoding="utf-8"))["format"] == (
        "lofi-hub-release-reservation"
    )


def test_a_rollback_failure_preserves_both_errors_and_recovery_evidence(
    workspace: ReleaseWorkspace,
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence.json"
    client = FakeHub(
        tmp_path / "remote",
        evidence=evidence,
        corrupt_anonymous=True,
        privatize_failure=True,
    )

    with pytest.raises(BaseExceptionGroup) as caught:
        hub_module.publish_release(
            _prepare(workspace),
            _authorization(),
            client,
            "secret",
            evidence,
        )

    failures = caught.value.exceptions
    assert len(failures) == 2
    assert isinstance(failures[0], ValueError)
    assert "digests to" in str(failures[0])
    assert isinstance(failures[1], OSError)
    assert "could not be restored" in str(failures[1])
    assert not client.private
    recovery = json.loads(evidence.read_text(encoding="utf-8"))
    assert recovery["format"] == "lofi-hub-release-recovery"
    assert recovery["visibility"] == "unknown-requires-private-verification"
    assert recovery["primary_failure"]["type"] == "ValueError"
    assert recovery["rollback_failure"]["type"] == "OSError"
    assert recovery["revision"] == "a" * 40


def test_a_post_replace_directory_sync_failure_restores_intent_before_rollback(
    workspace: ReleaseWorkspace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = tmp_path / "evidence.json"
    client = FakeHub(tmp_path / "remote", evidence=evidence)
    real_fsync = os.fsync
    directory_syncs = 0

    def injected_fsync(descriptor: int) -> None:
        nonlocal directory_syncs

        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            directory_syncs += 1

            if directory_syncs == 3:
                msg = "injected post-replace directory sync failure"
                raise OSError(msg)

        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", injected_fsync)

    with pytest.raises(OSError, match="post-replace directory sync failure"):
        hub_module.publish_release(
            _prepare(workspace),
            _authorization(),
            client,
            "secret",
            evidence,
        )

    assert directory_syncs >= 5
    assert client.events[-1][0] == "privatize"
    assert client.private
    restored = json.loads(evidence.read_text(encoding="utf-8"))
    assert restored["format"] == "lofi-hub-release-reservation"
    assert "revision" not in restored


def test_publication_uploads_then_round_trips_without_download_credentials(
    workspace: ReleaseWorkspace,
    tmp_path: Path,
) -> None:
    evidence_path = tmp_path / "evidence.json"
    client = FakeHub(tmp_path / "remote", evidence=evidence_path)
    prepared = _prepare(workspace)

    evidence = hub_module.publish_release(
        prepared,
        _authorization(),
        client,
        "secret",
        evidence_path,
    )

    assert len(client.uploads) == 1
    uploaded, uploaded_repo, uploaded_token = client.uploads[0]
    assert uploaded != workspace.bundle.resolve()
    assert uploaded_repo == "factslab/lofi-release-test"
    assert client.uploaded_paths == tuple(record.path for record in prepared.files)
    assert len(client.authenticated_downloads) == 1
    assert uploaded_token == client.authenticated_downloads[0][2]
    assert client.authenticated_downloads[0][:3] == (
        "factslab/lofi-release-test",
        "a" * 40,
        "secret",
    )
    assert len(client.anonymous_downloads) == 1
    assert client.anonymous_downloads[0][:2] == (
        "factslab/lofi-release-test",
        "a" * 40,
    )
    assert client.events[0] == (
        "require-absent",
        "factslab/lofi-release-test",
        "secret",
        "lofi-hub-release-reservation",
    )
    assert [event[0] for event in client.events] == [
        "require-absent",
        "create-private",
        "upload-private",
        "download-authenticated",
        "publicize",
        "download-anonymous",
    ]
    assert not client.private
    assert evidence.held_out == prepared.held_out
    assert evidence.files == prepared.files
    assert json.loads(evidence_path.read_text(encoding="utf-8")) == {
        "format": "lofi-hub-release",
        "repository": "factslab/lofi-release-test",
        "revision": "a" * 40,
        "checkpoint": workspace.validation.name,
        "corpus_revision": prepared.corpus_revision,
        "license": LICENSE,
        "reviewer": "Authorized Reviewer",
        "bundle_files": [
            {"path": record.path, "sha256": record.sha256} for record in prepared.files
        ],
        "held_out": {
            "items": 1,
            "sha256": prepared.held_out.sha256,
        },
    }


def test_a_remote_byte_change_prevents_release_evidence(
    workspace: ReleaseWorkspace,
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence.json"
    client = FakeHub(
        tmp_path / "remote",
        evidence=evidence,
        corrupt_anonymous=True,
    )

    with pytest.raises(ValueError, match="digests to"):
        hub_module.publish_release(
            _prepare(workspace),
            _authorization(),
            client,
            "secret",
            evidence,
        )

    assert len(client.uploads) == 1
    assert len(client.authenticated_downloads) == 1
    assert len(client.anonymous_downloads) == 1
    assert client.private
    assert client.events[-1][0] == "privatize"
    assert json.loads(evidence.read_text(encoding="utf-8"))["format"] == (
        "lofi-hub-release-reservation"
    )


def test_a_local_change_after_preparation_prevents_upload(
    workspace: ReleaseWorkspace,
    tmp_path: Path,
) -> None:
    copied = tmp_path / "bundle"
    shutil.copytree(workspace.bundle, copied)
    prepared = _prepare(workspace, bundle=copied)
    (copied / "unrecorded.txt").write_text("changed\n", encoding="utf-8", newline="\n")
    client = FakeHub(tmp_path / "remote")

    with pytest.raises(ValueError, match="bundle files"):
        hub_module.publish_release(
            prepared,
            _authorization(),
            client,
            "secret",
            tmp_path / "evidence.json",
        )

    assert client.uploads == []

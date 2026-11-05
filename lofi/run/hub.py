"""Hugging Face model bundles and their authorized release.

A bundle holds one trained model's weights, the resolved run document that
reproduces its grammar, and a model card. ``export_trained_point`` writes the
bundle of a trained point from its run directory and a release declaration.
``load_hub_bundle`` reconstructs a run from those persisted contents. Neither
opens a network connection.

``prepare_release`` verifies, without a network, that a bundle reproduces the
release checkpoint its run declares bitwise on the held-out items, and
``publish_release`` uploads an authorized release privately, downloads it back
with and without credentials, verifies both copies and only then records its
evidence, returning the repository to private visibility if anything fails
once it is public. Every route selects a run's checkpoint with
``newest_checkpoint`` and prepares the recorded point the same way.
"""

import json
import os
import re
import shutil
import sys
from collections.abc import Mapping
from hashlib import file_digest, sha256
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple, Never, Protocol, cast, final, override

import torch
from httpx import HTTPError
from huggingface_hub import HfApi, ModelCard, ModelCardData, snapshot_download
from huggingface_hub.errors import RepositoryNotFoundError
from torch import Tensor

from lofi._checks import refuse_reassignment
from lofi._frozen import FrozenValue
from lofi.data import LicenseRef, Licensing, Provenance, Reproducibility
from lofi.data.synthetic.training import SyntheticTrainingBatch
from lofi.nn.tokens.pretrained import LocalPretrainedLoader
from lofi.run._boundary.hub import (
    HubBundleManifest,
    HubBundleManifestIso,
    HubComponentRecord,
    HubFileRecord,
    HubLicenseRecord,
    HubReleaseDeclaration,
    HubReleaseRecord,
    HubReproducibilityRecord,
    HubTensorRecord,
)
from lofi.run._boundary.manifest import (
    ManifestStatus,
    decode_seed_derivation,
    encode_seed_derivation,
)
from lofi.run._device import accelerator_of, resolve_device_policy
from lofi.run._executor import CHECKPOINTS_NAME
from lofi.run._manifest import read_manifest
from lofi.run._seed import RandomStreams
from lofi.run._training import (
    TrainingBlueprint,
    plan_training,
)
from lofi.spec import ReleaseCheckpoint, RunSpecCodec, SpecLoader
from lofi.spec.artifacts import PretrainedSourceSpec, source_artifact
from lofi.spec.tokens.pretrained import PretrainedTokenEncoderSettings
from lofi.train import (
    ChartForward,
    CheckpointId,
    CheckpointManifest,
    DirectoryStore,
    ParserFittingPlan,
    ParserFittingRecord,
    PreparedTraining,
    prepare_training,
)

if TYPE_CHECKING:
    from lofi._artifacts import PinnedCheckpoint
    from lofi._types import FrozenJsonValue, JsonValue
    from lofi.nn.policy import DevicePolicy
    from lofi.nn.tokens.pretrained import PretrainedLoader
    from lofi.run._boundary.manifest import ExperimentManifest
    from lofi.run._ports import ReleaseAuthorization, ReleasePublication
    from lofi.run._seed import PointSeed
    from lofi.spec import ComponentSpec
    from lofi.train import (
        Checkpoint,
        ComponentRecord,
        RestorableRandomStreams,
        StateTree,
    )

__all__ = [
    "BundleFile",
    "HubClient",
    "HubRelease",
    "HuggingFaceHub",
    "LoadedHubModel",
    "ModelRelease",
    "PredictionDigest",
    "PreparedRelease",
    "ReleaseEvidence",
    "export_hub_model",
    "export_trained_point",
    "load_hub_bundle",
    "newest_checkpoint",
    "prepare_release",
    "publish_release",
    "restore_checkpoint",
    "restore_trained_point",
]

_MANIFEST = "manifest.json"
_CONFIGURATION = "config.json"
_CARD = "README.md"
_STATE_DIRECTORY = "states"
_RESOURCE_DIRECTORY = "resources"
_WEIGHT_SUFFIXES = frozenset(
    {".bin", ".h5", ".msgpack", ".onnx", ".pt", ".pth", ".safetensors"}
)


@final
class ModelRelease(FrozenValue):
    """Model-card metadata, exact identities, and export-only local resources.

    The card's text is the text a release declaration or a bundle's release
    record validated, and the identities are the trained point's own, so the
    value holds what it is given.
    """

    _noun = "model release"

    __slots__ = (
        "_action_set_identity",
        "_codec_identity",
        "_corpus_revision",
        "_datasets",
        "_grammar_identity",
        "_languages",
        "_model_name",
        "_pretrained_resources",
        "_provenance",
    )

    def __init__(
        self,
        model_name: str,
        datasets: tuple[str, ...],
        languages: tuple[str, ...],
        provenance: Provenance,
        *,
        grammar_identity: str,
        codec_identity: str,
        action_set_identity: str,
        corpus_revision: str,
        pretrained_resources: Mapping[str, Path],
    ) -> None:
        object.__setattr__(self, "_model_name", model_name)
        object.__setattr__(self, "_datasets", datasets)
        object.__setattr__(self, "_languages", languages)
        object.__setattr__(self, "_provenance", provenance)
        object.__setattr__(self, "_grammar_identity", grammar_identity)
        object.__setattr__(self, "_codec_identity", codec_identity)
        object.__setattr__(self, "_action_set_identity", action_set_identity)
        object.__setattr__(self, "_corpus_revision", corpus_revision)
        object.__setattr__(
            self,
            "_pretrained_resources",
            MappingProxyType(dict(sorted(pretrained_resources.items()))),
        )

    @property
    def model_name(self) -> str:
        """Return the name displayed by the model card."""
        return self._model_name

    @property
    def datasets(self) -> tuple[str, ...]:
        """Return corpus identifiers displayed by the model card."""
        return self._datasets

    @property
    def languages(self) -> tuple[str, ...]:
        """Return language tags displayed by the model card."""
        return self._languages

    @property
    def provenance(self) -> Provenance:
        """Return the corpus provenance carried into the release."""
        return self._provenance

    @property
    def grammar_identity(self) -> str:
        """Return the exact grammar identity."""
        return self._grammar_identity

    @property
    def codec_identity(self) -> str:
        """Return the exact codec identity."""
        return self._codec_identity

    @property
    def action_set_identity(self) -> str:
        """Return the exact action-set identity."""
        return self._action_set_identity

    @property
    def corpus_revision(self) -> str:
        """Return the exact corpus revision."""
        return self._corpus_revision

    @property
    def pretrained_resources(self) -> Mapping[str, Path]:
        """Return export-only local resources keyed by exact component role."""
        return self._pretrained_resources

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another release carries the same persisted metadata."""
        if type(other) is not ModelRelease:
            return NotImplemented

        return self._persistent_fields() == other._persistent_fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over persisted metadata and resource role names."""
        return hash((ModelRelease, self._persistent_fields()))

    def _persistent_fields(
        self,
    ) -> tuple[
        str,
        tuple[str, ...],
        tuple[str, ...],
        Provenance,
        str,
        str,
        str,
        str,
        tuple[str, ...],
    ]:
        """Return metadata persisted by a bundle."""
        return (
            self._model_name,
            self._datasets,
            self._languages,
            self._provenance,
            self._grammar_identity,
            self._codec_identity,
            self._action_set_identity,
            self._corpus_revision,
            tuple(self._pretrained_resources),
        )


@final
class LoadedHubModel(FrozenValue):
    """A loaded release and its configured model graph."""

    _noun = "loaded Hub model"
    __slots__ = ("_prepared", "_release")

    def __init__(self, release: ModelRelease, prepared: PreparedTraining) -> None:
        object.__setattr__(self, "_release", release)
        object.__setattr__(self, "_prepared", prepared)

    @property
    def release(self) -> ModelRelease:
        """Return the stored release metadata."""
        return self._release

    @property
    def prepared(self) -> PreparedTraining:
        """Return the configured graph with released weights restored."""
        return self._prepared

    @override
    def __setattr__[ValueT](self, name: str, value: ValueT) -> Never:
        del name, value
        refuse_reassignment("loaded Hub model")


def _digest(path: Path, /) -> str:
    """Return the SHA-256 digest of one file's exact bytes."""
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def _file(root: Path, path: Path, /) -> HubFileRecord:
    """Return one manifest file record relative to a bundle root."""
    return HubFileRecord(path=path.relative_to(root).as_posix(), digest=_digest(path))


def _write_bytes(path: Path, content: bytes, /) -> None:
    """Write exact bytes to one new file."""
    path.write_bytes(content)


def _release_record(release: ModelRelease, /) -> HubReleaseRecord:
    """Return the typed persisted form of release metadata."""
    provenance = release.provenance
    reproducibility = provenance.reproducibility

    return HubReleaseRecord(
        model_name=release.model_name,
        datasets=release.datasets,
        languages=release.languages,
        grammar_identity=release.grammar_identity,
        codec_identity=release.codec_identity,
        action_set_identity=release.action_set_identity,
        corpus_revision=release.corpus_revision,
        source_method=provenance.source_method,
        license_expression=provenance.licensing.expression,
        licenses=tuple(
            HubLicenseRecord(
                identifier=reference.identifier,
                applies_to=reference.applies_to,
            )
            for reference in provenance.licensing.licenses
        ),
        eprint_refs=provenance.eprint_refs,
        reproducibility=(
            None
            if reproducibility is None
            else HubReproducibilityRecord(
                code_uri=reproducibility.code_uri,
                commit_hash=reproducibility.commit_hash,
                command=reproducibility.command,
                environment=reproducibility.environment,
                random_seed=reproducibility.random_seed,
            )
        ),
        pretrained_roles=tuple(release.pretrained_resources),
    )


def _release_from_record(
    record: HubReleaseRecord,
    resources: Mapping[str, Path],
    /,
) -> ModelRelease:
    """Return public release metadata reconstructed from a bundle."""
    reproducibility = record.reproducibility

    return ModelRelease(
        record.model_name,
        record.datasets,
        record.languages,
        Provenance(
            record.source_method,
            Licensing(
                record.license_expression,
                tuple(
                    LicenseRef(reference.identifier, reference.applies_to)
                    for reference in record.licenses
                ),
            ),
            record.eprint_refs,
            (
                None
                if reproducibility is None
                else Reproducibility(
                    reproducibility.code_uri,
                    reproducibility.commit_hash,
                    reproducibility.command,
                    reproducibility.environment,
                    reproducibility.random_seed,
                )
            ),
        ),
        grammar_identity=record.grammar_identity,
        codec_identity=record.codec_identity,
        action_set_identity=record.action_set_identity,
        corpus_revision=record.corpus_revision,
        pretrained_resources=resources,
    )


def _card(release: ModelRelease, /) -> str:
    """Return a parsed Hugging Face model card with complete release metadata."""
    provenance = release.provenance
    reproducibility = provenance.reproducibility
    data = ModelCardData(
        datasets=list(release.datasets),
        language=list(release.languages),
        license=provenance.licensing.expression,
        model_name=release.model_name,
        source_method=provenance.source_method,
        eprint_refs=list(provenance.eprint_refs),
        grammar_identity=release.grammar_identity,
        codec_identity=release.codec_identity,
        action_set_identity=release.action_set_identity,
        corpus_revision=release.corpus_revision,
        component_licenses=[
            {
                "identifier": reference.identifier,
                "applies_to": reference.applies_to,
            }
            for reference in provenance.licensing.licenses
        ],
        reproducibility=(
            None
            if reproducibility is None
            else {
                "code_uri": reproducibility.code_uri,
                "commit_hash": reproducibility.commit_hash,
                "command": reproducibility.command,
                "environment": reproducibility.environment,
                "random_seed": reproducibility.random_seed,
            }
        ),
        tags=["lofi"],
    )
    body = (
        f"# {release.model_name}\n\n"
        "This bundle contains one LoFI model and the local resources needed "
        "to reconstruct it.\n"
    )

    return str(ModelCard(f"---\n{data.to_yaml()}\n---\n\n{body}"))


def _pretrained_checkpoint(
    role: str, declaration: ComponentSpec, /
) -> PinnedCheckpoint:
    """Return the pinned checkpoint after structural pretrained-resource preflight."""
    values = cast("dict[str, FrozenJsonValue]", declaration.model_dump())
    values.pop("kind")
    values.pop("declared_output", None)
    expected = frozenset(PretrainedTokenEncoderSettings._fields)

    if frozenset(values) != expected:
        msg = (
            f"construction resource role {role!r} does not implement the built-in "
            f"pretrained construction settings {sorted(expected)}"
        )
        raise ValueError(msg)

    checkpoint = values["checkpoint"]

    if not isinstance(checkpoint, Mapping):
        msg = f"construction resource role {role!r} has no pinned checkpoint"
        raise TypeError(msg)

    return source_artifact(
        PretrainedSourceSpec.model_validate(dict(checkpoint))
    ).hub_checkpoint(f"construction resource role {role!r}")


def _check_release(release: ModelRelease, blueprint: TrainingBlueprint, /) -> None:
    """Refuse release identities that differ from the checked blueprint."""
    compared = (
        ("grammar", release.grammar_identity, blueprint.grammar_fingerprint),
        ("codec", release.codec_identity, blueprint.codec_fingerprint),
        ("action set", release.action_set_identity, blueprint.action_set_fingerprint),
        ("corpus", release.corpus_revision, blueprint.corpus_revision),
    )

    for field, stored, expected in compared:
        if stored != expected:
            msg = f"{field} identity is {stored!r}, expected {expected!r}"
            raise ValueError(msg)

    roles = blueprint.run_spec.model.checkpointed_components()
    unknown = sorted(set(release.pretrained_resources) - set(roles))

    if unknown:
        msg = f"pretrained resources name absent component roles {unknown}"
        raise ValueError(msg)

    for role in release.pretrained_resources:
        _pretrained_checkpoint(role, roles[role])


def _state_kind(value: StateTree, /) -> str:
    """Return the exact top-level kind of one non-tensor state value."""
    if value is None:
        return "none"

    return type(value).__name__


def _tensor_state(
    state: Mapping[str, StateTree], /
) -> tuple[dict[str, StateTree], dict[str, HubTensorRecord]]:
    """Return one exact role block and its top-level structural record."""
    values: dict[str, StateTree] = {}
    records: dict[str, HubTensorRecord] = {}

    for key, value in state.items():
        if isinstance(value, Tensor):
            values[key] = value.detach().cpu()
            records[key] = HubTensorRecord(
                kind="tensor",
                shape=tuple(value.shape),
                dtype=str(value.dtype).removeprefix("torch."),
            )
        else:
            values[key] = value
            records[key] = HubTensorRecord(
                kind="value",
                shape=(),
                dtype=_state_kind(value),
            )

    return values, records


def _write_resources(
    staging: Path,
    release: ModelRelease,
    /,
) -> dict[str, dict[str, HubFileRecord]]:
    """Write exact-role construction assets and return their file records."""
    resources: dict[str, dict[str, HubFileRecord]] = {}
    root = staging / _RESOURCE_DIRECTORY
    root.mkdir()

    for position, role in enumerate(release.pretrained_resources):
        declared_source = release.pretrained_resources[role]

        if not declared_source.is_dir():
            msg = f"pretrained resource source for role {role!r} is not a directory"
            raise ValueError(msg)

        destination = root / f"{position:06d}"
        destination.mkdir()
        files: dict[str, HubFileRecord] = {}

        for source in sorted(declared_source.rglob("*")):
            if source.is_symlink():
                msg = f"construction resource for role {role!r} is not a regular file"
                raise ValueError(msg)

            if not source.is_file():
                continue

            if source.suffix in _WEIGHT_SUFFIXES:
                continue

            relative_path = source.relative_to(declared_source)
            copied = destination / relative_path
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, copied)
            relative = relative_path.as_posix()
            files[relative] = _file(staging, copied)

        if not files:
            msg = f"pretrained resource role {role!r} wrote no construction assets"
            raise ValueError(msg)

        resources[role] = files

    return resources


def export_hub_model(
    destination: Path,
    release: ModelRelease,
    blueprint: TrainingBlueprint,
    prepared: PreparedTraining,
    seed_derivation: PointSeed,
    /,
) -> None:
    """Write one complete model-only bundle without opening a network connection."""
    if destination.exists():
        msg = f"Hub bundle destination already exists: {destination}"
        raise FileExistsError(msg)

    if not destination.parent.is_dir():
        msg = f"Hub bundle parent is not a directory: {destination.parent}"
        raise FileNotFoundError(msg)

    _check_release(release, blueprint)
    declarations = blueprint.run_spec.model.checkpointed_components()
    paths = blueprint.run_spec.model.checkpointed_paths()
    state = prepared.model.state()

    if set(state) != set(declarations):
        msg = (
            f"prepared model roles are {sorted(state)}, but the blueprint declares "
            f"{sorted(declarations)}"
        )
        raise ValueError(msg)

    staging = Path(mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))

    try:
        configuration = staging / _CONFIGURATION
        card = staging / _CARD
        state_root = staging / _STATE_DIRECTORY
        state_root.mkdir()
        _write_bytes(configuration, blueprint.configuration)
        _write_bytes(card, _card(release).encode())
        components: dict[str, HubComponentRecord] = {}

        for position, role in enumerate(sorted(state)):
            tensors, tensor_records = _tensor_state(state[role])
            block = state_root / f"{position:06d}.pt"
            torch.save(tensors, block)
            declaration = declarations[role]
            components[role] = HubComponentRecord(
                tag=cast("str", declaration.model_dump()["kind"]),
                spec_fingerprint=blueprint.component_fingerprints[paths[role]],
                state=_file(staging, block),
                tensors=tensor_records,
            )

        resources = _write_resources(staging, release)
        manifest = HubBundleManifest(
            format="lofi-hub-model",
            seed_derivation=encode_seed_derivation(seed_derivation),
            release=_release_record(release),
            configuration=_file(staging, configuration),
            card=_file(staging, card),
            components=components,
            resources=resources,
        )
        _write_bytes(staging / _MANIFEST, HubBundleManifestIso().forward(manifest))
        staging.rename(destination)

    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _read_file(source: Path, record: HubFileRecord, /) -> Path:
    """Return one recorded file after verifying its exact bytes."""
    path = source / record.path

    if not path.is_file():
        msg = f"bundle file {record.path!r} is missing"
        raise ValueError(msg)

    found = _digest(path)

    if found != record.digest:
        msg = (
            f"bundle file {record.path!r} digests to {found}, not stored "
            f"{record.digest}"
        )
        raise ValueError(msg)

    return path


def _expected_files(manifest: HubBundleManifest, /) -> frozenset[str]:
    """Return every file path the manifest claims, including itself."""
    files = {_MANIFEST, manifest.configuration.path, manifest.card.path}
    files.update(component.state.path for component in manifest.components.values())
    files.update(
        record.path
        for resources in manifest.resources.values()
        for record in resources.values()
    )

    return frozenset(files)


def _validate_manifest(
    source: Path,
    manifest: HubBundleManifest,
    expected: TrainingBlueprint,
    /,
) -> dict[str, dict[str, StateTree]]:
    """Validate all persisted identities and state before construction."""
    found_files = frozenset(
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_file()
    )
    expected_files = _expected_files(manifest)

    if found_files != expected_files:
        missing = sorted(expected_files - found_files)
        extra = sorted(found_files - expected_files)
        msg = f"bundle files lack {missing} and add {extra}"
        raise ValueError(msg)

    configuration = _read_file(source, manifest.configuration).read_bytes()
    _read_file(source, manifest.card)

    if configuration != expected.configuration:
        stored = sha256(configuration).hexdigest()
        wanted = sha256(expected.configuration).hexdigest()
        msg = f"configuration digest is {stored}, expected {wanted}"
        raise ValueError(msg)

    release = manifest.release
    compared = (
        ("grammar", release.grammar_identity, expected.grammar_fingerprint),
        ("codec", release.codec_identity, expected.codec_fingerprint),
        ("action set", release.action_set_identity, expected.action_set_fingerprint),
        ("corpus", release.corpus_revision, expected.corpus_revision),
    )

    for field, stored, wanted in compared:
        if stored != wanted:
            msg = f"stored {field} identity is {stored!r}, expected {wanted!r}"
            raise ValueError(msg)

    declarations = expected.run_spec.model.checkpointed_components()
    paths = expected.run_spec.model.checkpointed_paths()

    if set(manifest.components) != set(declarations):
        missing = sorted(set(declarations) - set(manifest.components))
        extra = sorted(set(manifest.components) - set(declarations))
        msg = f"model component roles lack {missing} and add {extra}"
        raise ValueError(msg)

    states: dict[str, dict[str, StateTree]] = {}

    for role, component in manifest.components.items():
        declaration = declarations[role]
        wanted_tag = cast("str", declaration.model_dump()["kind"])
        wanted_fingerprint = expected.component_fingerprints[paths[role]]

        if component.tag != wanted_tag:
            msg = (
                f"component role {role!r} stores tag {component.tag!r}, "
                f"expected {wanted_tag!r}"
            )
            raise ValueError(msg)

        if component.spec_fingerprint != wanted_fingerprint:
            msg = (
                f"component role {role!r} stores specification fingerprint "
                f"{component.spec_fingerprint}, expected {wanted_fingerprint}"
            )
            raise ValueError(msg)

        block = _read_file(source, component.state)
        loaded = torch.load(block, map_location="cpu", weights_only=True)

        if not isinstance(loaded, dict) or any(
            not isinstance(key, str) for key in loaded
        ):
            msg = f"component role {role!r} does not hold a named state mapping"
            raise ValueError(msg)

        tensors = cast("dict[str, StateTree]", loaded)

        if set(tensors) != set(component.tensors):
            missing = sorted(set(component.tensors) - set(tensors))
            extra = sorted(set(tensors) - set(component.tensors))
            msg = f"component role {role!r} state keys lack {missing} and add {extra}"
            raise ValueError(msg)

        for key, tensor in tensors.items():
            record = component.tensors[key]
            shape = tuple(tensor.shape) if isinstance(tensor, Tensor) else ()
            dtype = (
                str(tensor.dtype).removeprefix("torch.")
                if isinstance(tensor, Tensor)
                else _state_kind(tensor)
            )
            kind = "tensor" if isinstance(tensor, Tensor) else "value"

            if kind != record.kind:
                msg = (
                    f"component role {role!r} state key {key!r} has kind "
                    f"{kind!r}, not stored {record.kind!r}"
                )
                raise ValueError(msg)

            if shape != record.shape:
                msg = (
                    f"component role {role!r} state key {key!r} has shape "
                    f"{shape}, not stored {record.shape}"
                )
                raise ValueError(msg)

            if dtype != record.dtype:
                msg = (
                    f"component role {role!r} state key {key!r} has dtype "
                    f"{dtype!r}, not stored {record.dtype!r}"
                )
                raise ValueError(msg)

        states[role] = tensors

    for position, role in enumerate(manifest.release.pretrained_roles):
        resources = manifest.resources[role]

        if not resources:
            msg = f"pretrained resource role {role!r} has no construction assets"
            raise ValueError(msg)

        for relative, record in resources.items():
            expected_path = f"{_RESOURCE_DIRECTORY}/{position:06d}/{relative}"

            if record.path != expected_path:
                msg = (
                    f"construction resource role {role!r} records {record.path!r}, "
                    f"expected {expected_path!r}"
                )
                raise ValueError(msg)

            if Path(record.path).suffix in _WEIGHT_SUFFIXES:
                msg = f"construction resource {record.path!r} duplicates model weights"
                raise ValueError(msg)

            _read_file(source, record)

    return states


def _resource_loaders(
    source: Path,
    manifest: HubBundleManifest,
    expected: TrainingBlueprint,
    /,
) -> tuple[dict[str, PretrainedLoader], dict[str, Path]]:
    """Return exact-role local-only loaders and their verified directories."""
    declarations = expected.run_spec.model.checkpointed_components()
    loaders: dict[str, PretrainedLoader] = {}
    paths: dict[str, Path] = {}

    for position, role in enumerate(manifest.release.pretrained_roles):
        declaration = declarations[role]
        checkpoint = _pretrained_checkpoint(role, declaration)

        directory = source / _RESOURCE_DIRECTORY / f"{position:06d}"
        loaders[role] = LocalPretrainedLoader(directory, checkpoint)
        paths[role] = directory

    return loaders, paths


def _read_hub_manifest(source: Path, /) -> HubBundleManifest:
    """Validate and return the boundary manifest of one local bundle."""
    if not source.is_dir():
        msg = f"Hub bundle source is not a local directory: {source}"
        raise FileNotFoundError(msg)

    try:
        manifest_bytes = (source / _MANIFEST).read_bytes()

    except OSError as error:
        msg = f"Hub bundle manifest cannot be read: {error}"
        raise ValueError(msg) from error

    return HubBundleManifestIso().backward(manifest_bytes)


def _restore_bundle(
    source: Path,
    manifest: HubBundleManifest,
    expected: TrainingBlueprint,
    streams: RestorableRandomStreams,
    policy: DevicePolicy,
    /,
) -> LoadedHubModel:
    """Load a bundle whose boundary manifest was already validated."""
    states = _validate_manifest(source, manifest, expected)
    loaders, resource_paths = _resource_loaders(source, manifest, expected)
    prepared = prepare_training(
        expected,
        streams,
        policy,
        pretrained_resources=loaders,
    )
    restore = prepared.model.prepare_restore(
        cast("Mapping[str, Mapping[str, StateTree]]", states)
    )
    restore()

    return LoadedHubModel(
        _release_from_record(manifest.release, resource_paths),
        prepared,
    )


def load_hub_bundle(source: Path, /) -> LoadedHubModel:
    """Reconstruct a model from one bundle's configuration and point seed."""
    manifest = _read_hub_manifest(source)

    if manifest.configuration.path != _CONFIGURATION:
        msg = (
            f"Hub bundle configuration is {manifest.configuration.path!r}, "
            f"not {_CONFIGURATION!r}"
        )
        raise ValueError(msg)

    derivation = decode_seed_derivation(manifest.seed_derivation)
    run_spec = RunSpecCodec()(source / _CONFIGURATION)
    blueprint = plan_training(run_spec, point=derivation)
    streams = RandomStreams(
        derivation.policy(
            accelerator=accelerator_of(run_spec.device_policy.device),
            samplers=blueprint.sampler_streams,
        )
    )
    streams.apply()

    return _restore_bundle(
        source,
        manifest,
        blueprint,
        streams,
        resolve_device_policy(run_spec),
    )


def newest_checkpoint(
    store: DirectoryStore, kind: ReleaseCheckpoint, /
) -> tuple[CheckpointId, CheckpointManifest] | None:
    """Return the newest checkpoint of one fitting class a run holds, or None.

    Only each checkpoint's manifest is read, so no state is loaded to select
    one.

    Parameters
    ----------
    store : DirectoryStore
        The run's checkpoint store.
    kind : ReleaseCheckpoint
        The fitting class, ``validation-best`` or ``resume-latest``.
    """
    candidates = tuple(
        (handle, manifest)
        for handle in store.list()
        if (manifest := store.manifest(handle)).fitting is not None
        and manifest.fitting.kind == kind
    )

    return candidates[-1] if candidates else None


def _restore_states(
    prepared: PreparedTraining, checkpoint: Checkpoint[StateTree], /
) -> None:
    """Restore a loaded checkpoint's model states into a prepared run."""
    prepared.model.prepare_restore(
        cast("Mapping[str, Mapping[str, StateTree]]", checkpoint.states)
    )()


def restore_checkpoint(
    checkpoints: Path,
    blueprint: TrainingBlueprint,
    prepared: PreparedTraining,
    kind: ReleaseCheckpoint,
    /,
) -> None:
    """Restore the newest checkpoint of the declared class into a run's model.

    Raises
    ------
    ValueError
        If the run holds no checkpoint of the declared class, or that checkpoint
        was written for another run declaration.
    """
    store = DirectoryStore(checkpoints)
    newest = newest_checkpoint(store, kind)

    if newest is None:
        msg = f"the run holds no {kind!r} checkpoint under {checkpoints}"
        raise ValueError(msg)

    selected, manifest = newest

    if manifest.run_id != blueprint.declaration_fingerprint:
        msg = (
            f"checkpoint {selected} belongs to the run {manifest.run_id!r}, not "
            f"the run declared as {blueprint.declaration_fingerprint!r}"
        )
        raise ValueError(msg)

    _restore_states(prepared, store.load(selected, manifest))


def _prepared_point(
    manifest: ExperimentManifest, /
) -> tuple[TrainingBlueprint, PreparedTraining]:
    """Plan and prepare one recorded run under its declaration and point seed."""
    blueprint = plan_training(
        manifest.run_spec,
        point=decode_seed_derivation(manifest.seed_derivation),
    )
    streams = RandomStreams(manifest.seed_policy.policy())
    streams.apply()

    return blueprint, prepare_training(
        blueprint,
        streams,
        resolve_device_policy(manifest.run_spec),
    )


def _restore_trained_point(
    run: Path,
    manifest: ExperimentManifest,
    /,
) -> tuple[TrainingBlueprint, PreparedTraining]:
    """Restore one completed run through its recorded declaration and point seed."""
    blueprint, prepared = _prepared_point(manifest)
    restore_checkpoint(
        run / CHECKPOINTS_NAME,
        blueprint,
        prepared,
        manifest.run_spec.release.checkpoint,
    )
    return blueprint, prepared


def restore_trained_point(run: Path, /) -> PreparedTraining:
    """Restore the released prediction point recorded by one completed run."""
    manifest = read_manifest(run)

    if manifest is None or manifest.status is not ManifestStatus.COMPLETED:
        msg = f"the run directory {run} records no completed run"
        raise ValueError(msg)

    _, prepared = _restore_trained_point(run, manifest)
    return prepared


def export_trained_point(run: Path, release: Path, destination: Path, /) -> None:
    """Write the bundle of the trained point a run directory holds, offline.

    The point's recorded run document is planned again and its model is
    restored from the checkpoint class the run declares for release. The
    release declaration supplies the model
    card's metadata; the model identities and corpus revision come from the
    planned run, and the source revision and random seed from the point's
    manifest.

    Parameters
    ----------
    run : Path
        The run directory of a completed point.
    release : Path
        The release declaration document.
    destination : Path
        The bundle directory to create, which must not exist.

    Raises
    ------
    ValueError
        If the directory records no completed run, holds no checkpoint for its
        run declaration, or the release names an absent pretrained role.
    ConfigError
        If the release declaration is refused.
    FileExistsError
        If the destination exists.
    """
    manifest = read_manifest(run)

    if manifest is None or manifest.status is not ManifestStatus.COMPLETED:
        msg = f"the run directory {run} records no completed run"
        raise ValueError(msg)

    declared = SpecLoader().compose_traced(release, schema=HubReleaseDeclaration).spec
    seed_policy = manifest.seed_policy.policy()
    blueprint, prepared = _restore_trained_point(run, manifest)
    model_release = ModelRelease(
        declared.model_name,
        declared.datasets,
        declared.languages,
        Provenance(
            declared.source_method,
            Licensing(
                declared.license_expression,
                tuple(
                    LicenseRef(record.identifier, record.applies_to)
                    for record in declared.licenses
                ),
            ),
            declared.eprint_refs,
            Reproducibility(
                declared.code_uri,
                manifest.git_revision,
                declared.command,
                declared.environment,
                seed_policy.root,
            ),
        ),
        grammar_identity=blueprint.grammar_fingerprint,
        codec_identity=blueprint.codec_fingerprint,
        action_set_identity=blueprint.action_set_fingerprint,
        corpus_revision=blueprint.corpus_revision,
        pretrained_resources={
            role: Path(directory)
            for role, directory in declared.pretrained_resources.items()
        },
    )
    export_hub_model(
        destination,
        model_release,
        blueprint,
        prepared,
        decode_seed_derivation(manifest.seed_derivation),
    )


class BundleFile(NamedTuple):
    """One bundle path and its exact SHA-256 digest."""

    path: str
    sha256: str


class PredictionDigest(NamedTuple):
    """The exact bytes of all held-out item scores, represented by a digest."""

    items: int
    sha256: str


class PreparedRelease(NamedTuple):
    """A locally verified bundle and the declared checkpoint it reproduces."""

    bundle: Path
    repo_id: str
    run: Path
    checkpoint: Path
    checkpoint_id: str
    model_name: str
    corpus_revision: str
    license_identifier: str
    seed: int
    files: tuple[BundleFile, ...]
    held_out: PredictionDigest


class ReleaseEvidence(NamedTuple):
    """The immutable identity and round-trip result of one public release."""

    repo_id: str
    revision: str
    checkpoint_id: str
    corpus_revision: str
    license_identifier: str
    reviewer: str
    files: tuple[BundleFile, ...]
    held_out: PredictionDigest


class HubClient(Protocol):
    """Carry one model repository through the explicit release lifecycle."""

    def require_absent(self, repo_id: str, token: str, /) -> None:
        """Refuse every collision with an existing destination repository."""
        ...

    def create_private(self, repo_id: str, token: str, /) -> None:
        """Create one new private model repository."""
        ...

    def upload_private(self, bundle: Path, repo_id: str, token: str, /) -> str:
        """Upload staged bytes and return their immutable remote revision."""
        ...

    def download_authenticated(
        self,
        repo_id: str,
        revision: str,
        token: str,
        destination: Path,
        /,
    ) -> Path:
        """Download one exact private revision into a fresh destination."""
        ...

    def publicize(self, repo_id: str, token: str, /) -> None:
        """Make the privately verified repository public."""
        ...

    def download_anonymous(
        self,
        repo_id: str,
        revision: str,
        destination: Path,
        /,
    ) -> Path:
        """Download one exact public revision without credentials."""
        ...

    def privatize(self, repo_id: str, token: str, /) -> None:
        """Return a public repository to private visibility."""
        ...


class HuggingFaceHub:
    """Perform the release lifecycle through ``huggingface_hub``."""

    __slots__ = ("_api",)

    def __init__(self) -> None:
        self._api = HfApi()

    def require_absent(self, repo_id: str, token: str, /) -> None:
        """Refuse an existing model repository, including an empty private one."""
        try:
            self._api.repo_info(repo_id, repo_type="model", token=token)
        except RepositoryNotFoundError:
            return

        msg = f"Hugging Face repository already exists: {repo_id}"
        raise FileExistsError(msg)

    def create_private(self, repo_id: str, token: str, /) -> None:
        """Create one new private model repository without collision tolerance."""
        self._api.create_repo(
            repo_id,
            token=token,
            private=True,
            repo_type="model",
            exist_ok=False,
        )

    def upload_private(self, bundle: Path, repo_id: str, token: str, /) -> str:
        """Upload one complete staged bundle to a private repository."""
        commit = self._api.upload_folder(
            repo_id=repo_id,
            folder_path=bundle,
            token=token,
            repo_type="model",
            commit_message="Publish verified LoFI model bundle",
        )

        if not commit.oid:
            msg = "Hugging Face returned no immutable revision for the upload"
            raise RuntimeError(msg)

        return commit.oid

    def _download(
        self,
        repo_id: str,
        revision: str,
        destination: Path,
        /,
        *,
        token: str | bool,
    ) -> Path:
        """Download an exact revision into a destination that does not exist."""
        cache = destination.with_name(f".{destination.name}.cache")

        if destination.exists():
            msg = f"Hub download destination already exists: {destination}"
            raise FileExistsError(msg)

        if cache.exists():
            msg = f"Hub download cache already exists: {cache}"
            raise FileExistsError(msg)

        if not destination.parent.is_dir():
            msg = f"Hub download parent is not a directory: {destination.parent}"
            raise FileNotFoundError(msg)

        cached = snapshot_download(
            repo_id,
            repo_type="model",
            revision=revision,
            cache_dir=cache,
            token=token,
            force_download=True,
        )
        downloaded = destination
        shutil.copytree(Path(cached), downloaded)

        return downloaded

    def download_authenticated(
        self,
        repo_id: str,
        revision: str,
        token: str,
        destination: Path,
        /,
    ) -> Path:
        """Download an exact private revision using release credentials."""
        return self._download(repo_id, revision, destination, token=token)

    def publicize(self, repo_id: str, token: str, /) -> None:
        """Make the verified private repository public."""
        self._api.update_repo_settings(
            repo_id,
            private=False,
            token=token,
            repo_type="model",
        )

    def download_anonymous(
        self,
        repo_id: str,
        revision: str,
        destination: Path,
        /,
    ) -> Path:
        """Download one public revision with authentication explicitly disabled."""
        return self._download(repo_id, revision, destination, token=False)

    def privatize(self, repo_id: str, token: str, /) -> None:
        """Return a repository to private visibility after a public failure."""
        self._api.update_repo_settings(
            repo_id,
            private=True,
            token=token,
            repo_type="model",
        )


class _BundleInspection(NamedTuple):
    """The release facts reproduced from one local bundle."""

    model_name: str
    corpus_revision: str
    license_identifier: str
    seed: int
    files: tuple[BundleFile, ...]
    held_out: PredictionDigest


def _held_out_digest(plan: ParserFittingPlan, /) -> PredictionDigest:
    """Return an exact digest over validation-item score tensors."""
    if not isinstance(plan.forward, ChartForward):
        msg = "a parser model release needs a chart forward pass"
        raise TypeError(msg)

    digest = sha256()
    items = 0

    with torch.no_grad():
        for batch in plan.fit_batches:
            if not isinstance(batch, SyntheticTrainingBatch):
                msg = "a parser model release needs synthetic parser batches"
                raise TypeError(msg)

            published = plan.forward.publish(batch)
            scores = published.scores

            if scores is None:
                msg = "the validation forward pass publishes no item scores"
                raise ValueError(msg)

            tensor = scores.tensor.detach().cpu().contiguous()

            if tensor.numel() != len(batch):
                msg = (
                    f"a validation batch of {len(batch)} items published "
                    f"{tensor.numel()} item scores"
                )
                raise ValueError(msg)

            metadata = json.dumps(
                {
                    "dtype": str(tensor.dtype),
                    "shape": list(tensor.shape),
                },
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
            payload = tensor.view(torch.uint8).numpy().tobytes()
            digest.update(len(metadata).to_bytes(8, "big"))
            digest.update(metadata)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
            items += len(batch)

    if items == 0:
        msg = "the selected parser checkpoint has no held-out items"
        raise ValueError(msg)

    return PredictionDigest(items, digest.hexdigest())


def _prediction_digest(prepared: PreparedTraining, /) -> PredictionDigest:
    """Return exact held-out predictions from one prepared parser."""
    for module in prepared.model.modules.values():
        module.eval()

    return _held_out_digest(prepared.plan.parser_plan())


def _bundle_files(bundle: Path, /) -> tuple[BundleFile, ...]:
    """Return every regular bundle file in canonical path order."""
    records: list[BundleFile] = []

    for path in sorted(bundle.rglob("*")):
        relative = path.relative_to(bundle).as_posix()

        if path.is_symlink():
            msg = f"bundle path {relative!r} is a symbolic link"
            raise ValueError(msg)

        if not path.is_file():
            continue

        with path.open("rb") as handle:
            digest = file_digest(handle, "sha256").hexdigest()

        records.append(BundleFile(relative, digest))

    if not records:
        msg = f"Hub bundle contains no files: {bundle}"
        raise ValueError(msg)

    return tuple(records)


def _inspect_bundle(bundle: Path, seed: int, /) -> _BundleInspection:
    """Load and evaluate one local bundle through the public Hub API."""
    loaded = load_hub_bundle(bundle)
    release = loaded.release
    reproducibility = release.provenance.reproducibility

    if not release.datasets:
        msg = "the release names no corpus whose redistribution terms can be reviewed"
        raise ValueError(msg)

    if reproducibility is None or reproducibility.random_seed != seed:
        found = None if reproducibility is None else reproducibility.random_seed
        msg = f"bundle release seed is {found!r}, not the run seed {seed}"
        raise ValueError(msg)

    return _BundleInspection(
        release.model_name,
        release.corpus_revision,
        release.provenance.licensing.expression,
        seed,
        _bundle_files(bundle),
        _prediction_digest(loaded.prepared),
    )


def _component_identity(record: ComponentRecord, /) -> tuple[object, ...]:
    """Return checkpoint component fields that do not change with its weights."""
    return (
        record.tag,
        record.spec_fingerprint,
        record.signature_fingerprint,
        record.codec_identity,
        record.state.key,
        None if record.labels is None else tuple(sorted(record.labels.items())),
    )


def _validate_checkpoint_pair(
    validation_directory: Path,
    resume_directory: Path,
    validation: CheckpointManifest,
    resume: CheckpointManifest,
    plan: ParserFittingPlan,
    /,
) -> ParserFittingRecord:
    """Validate one terminal parser run's distinct best and latest states."""
    validation_id = validation_directory.name
    resume_id = resume_directory.name

    if validation_id == resume_id:
        msg = "validation-best and resume-latest checkpoints are the same"
        raise ValueError(msg)

    validation_fitting = validation.fitting
    resume_fitting = resume.fitting

    if (
        not isinstance(validation_fitting, ParserFittingRecord)
        or validation_fitting.kind != "validation-best"
    ):
        msg = f"checkpoint {validation_id!r} is not parser validation-best state"
        raise ValueError(msg)

    if (
        not isinstance(resume_fitting, ParserFittingRecord)
        or resume_fitting.kind != "resume-latest"
    ):
        msg = f"checkpoint {resume_id!r} is not parser resume-latest state"
        raise ValueError(msg)

    best = validation_fitting
    latest = resume_fitting

    if latest.best_checkpoint != validation_id:
        msg = (
            f"resume-latest names validation checkpoint "
            f"{latest.best_checkpoint!r}, not {validation_id!r}"
        )
        raise ValueError(msg)

    if best.metric != plan.early_stopping.metric:
        msg = (
            f"validation-best monitored {best.metric!r}, not configured metric "
            f"{plan.early_stopping.metric!r}"
        )
        raise ValueError(msg)

    if latest.metric != plan.early_stopping.metric:
        msg = (
            f"resume-latest monitored {latest.metric!r}, not configured metric "
            f"{plan.early_stopping.metric!r}"
        )
        raise ValueError(msg)

    for checkpoint_id, manifest, fitting in (
        (validation_id, validation, best),
        (resume_id, resume, latest),
    ):
        expected_step = fitting.completed_epochs * plan.steps_per_epoch
        expected_position = fitting.completed_epochs * plan.items_per_epoch

        if (manifest.step, manifest.data_position) != (
            expected_step,
            expected_position,
        ):
            msg = (
                f"checkpoint {checkpoint_id!r} completed "
                f"{fitting.completed_epochs} epochs, which implies step/data "
                f"position {(expected_step, expected_position)!r}, not "
                f"{(manifest.step, manifest.data_position)!r}"
            )
            raise ValueError(msg)

        if fitting.completed_epochs > plan.epochs:
            msg = (
                f"checkpoint {checkpoint_id!r} completed "
                f"{fitting.completed_epochs} epochs beyond configured horizon "
                f"{plan.epochs}"
            )
            raise ValueError(msg)

    if latest.completed_epochs < best.completed_epochs:
        msg = "resume-latest predates the selected validation-best checkpoint"
        raise ValueError(msg)

    if latest.best != best.best:
        msg = "resume-latest does not preserve the selected validation-best metric"
        raise ValueError(msg)

    if best.failures != 0:
        msg = "a validation-best checkpoint records non-improving epochs"
        raise ValueError(msg)

    expected_failures = latest.completed_epochs - best.completed_epochs

    if latest.failures != expected_failures:
        msg = (
            f"resume-latest records {latest.failures} non-improving epochs, not "
            f"{expected_failures} since validation-best"
        )
        raise ValueError(msg)

    at_horizon = latest.completed_epochs == plan.epochs
    stopped_early = (
        latest.completed_epochs < plan.epochs
        and latest.failures == plan.early_stopping.patience
    )

    if not at_horizon and not stopped_early:
        msg = (
            f"resume-latest is not terminal: completed "
            f"{latest.completed_epochs}/{plan.epochs} epochs with "
            f"{latest.failures}/{plan.early_stopping.patience} stopping failures"
        )
        raise ValueError(msg)

    if resume.optimizer is None:
        msg = "resume-latest checkpoint has no optimizer state"
        raise ValueError(msg)

    same_run = (
        validation.run_id == resume.run_id
        and validation.code_version == resume.code_version
        and validation.precision == resume.precision
        and validation.grounding == resume.grounding
        and validation.stage == resume.stage
        and set(validation.components) == set(resume.components)
        and all(
            _component_identity(validation.components[role])
            == _component_identity(resume.components[role])
            for role in validation.components
        )
    )

    if not same_run:
        msg = "validation-best and resume-latest do not belong to one run"
        raise ValueError(msg)

    return best


def _source_predictions(
    bundle: Path,
    run: Path,
    /,
) -> tuple[_BundleInspection, PredictionDigest, Path]:
    """Reconstruct the declared release point and the bundle after preflight."""
    run_manifest = read_manifest(run)

    if run_manifest is None or run_manifest.status is not ManifestStatus.COMPLETED:
        msg = f"the run directory {run} records no completed run"
        raise ValueError(msg)

    run_spec = RunSpecCodec()(bundle / _CONFIGURATION)

    if run_manifest.run_spec != run_spec:
        msg = "the bundle declaration differs from the run declaration"
        raise ValueError(msg)

    blueprint, source = _prepared_point(run_manifest)
    policy = run_manifest.seed_policy.policy()
    root_seed = policy.root
    store = DirectoryStore(run / CHECKPOINTS_NAME)
    newest = newest_checkpoint(store, "resume-latest")

    if newest is None:
        msg = f"the run holds no 'resume-latest' checkpoint under {store.root}"
        raise ValueError(msg)

    resume_id, resume = newest
    resume_fitting = resume.fitting

    if (
        not isinstance(resume_fitting, ParserFittingRecord)
        or resume_fitting.best_checkpoint is None
    ):
        msg = f"checkpoint {resume_id!s} names no parser validation-best checkpoint"
        raise ValueError(msg)

    validation_id = CheckpointId(resume_fitting.best_checkpoint)
    validation = store.manifest(validation_id)
    validation_directory = store.root / validation_id
    resume_directory = store.root / resume_id

    if validation.run_id != blueprint.declaration_fingerprint:
        msg = (
            f"validation checkpoint run identity is {validation.run_id!r}, "
            f"not bundle declaration {blueprint.declaration_fingerprint!r}"
        )
        raise ValueError(msg)

    declared_samplers = policy.samplers
    expected_samplers = tuple(sorted(blueprint.sampler_streams))

    if declared_samplers != expected_samplers:
        msg = (
            f"checkpoint sampler streams are {declared_samplers!r}, "
            f"not bundle streams {expected_samplers!r}"
        )
        raise ValueError(msg)

    _validate_checkpoint_pair(
        validation_directory,
        resume_directory,
        validation,
        resume,
        source.plan.parser_plan(),
    )
    expected_roles = frozenset(source.model.modules)

    for manifest_name, candidate in (
        ("validation-best", validation),
        ("resume-latest", resume),
    ):
        if candidate.roles != expected_roles:
            msg = (
                f"{manifest_name} component roles are {sorted(candidate.roles)}, "
                f"not bundle roles {sorted(expected_roles)}"
            )
            raise ValueError(msg)

        for role, record in candidate.components.items():
            path = run_spec.model.checkpointed_paths()[role]
            expected_fingerprint = blueprint.component_fingerprints.get(path)

            if record.spec_fingerprint != expected_fingerprint:
                msg = (
                    f"{manifest_name} role {role!r} has specification "
                    f"{record.spec_fingerprint!r}, not {expected_fingerprint!r}"
                )
                raise ValueError(msg)

            if record.signature_fingerprint != blueprint.codec_fingerprint:
                msg = (
                    f"{manifest_name} role {role!r} has signature "
                    f"{record.signature_fingerprint!r}, not "
                    f"{blueprint.codec_fingerprint!r}"
                )
                raise ValueError(msg)

            if record.codec_identity != blueprint.grammar_fingerprint:
                msg = (
                    f"{manifest_name} role {role!r} has codec identity "
                    f"{record.codec_identity!r}, not "
                    f"{blueprint.grammar_fingerprint!r}"
                )
                raise ValueError(msg)

    released_id, released = (
        (validation_id, validation)
        if run_spec.release.checkpoint == "validation-best"
        else (resume_id, resume)
    )
    checkpoint = store.load(released_id, released)
    counterpart_id, counterpart = (
        (resume_id, resume)
        if released_id == validation_id
        else (validation_id, validation)
    )
    store.load(counterpart_id, counterpart)
    _restore_states(source, checkpoint)
    source_held_out = _prediction_digest(source)
    inspection = _inspect_bundle(bundle, root_seed)

    return inspection, source_held_out, store.root / released_id


def _repo_id(value: str, /) -> str:
    """Return one explicit Hugging Face owner/repository identity."""
    owner, separator, name = value.partition("/")

    if not separator or not owner or not name or "/" in name:
        msg = f"Hub repository identity is not owner/name: {value!r}"
        raise ValueError(msg)

    return value


def prepare_release(
    bundle: Path,
    repo_id: str,
    run: Path,
    /,
) -> PreparedRelease:
    """Verify a run's declared release checkpoint and bundle without a network."""
    held_bundle = bundle.resolve()
    held_run = run.resolve()

    if not held_bundle.is_dir():
        msg = f"Hub bundle is not a directory: {held_bundle}"
        raise FileNotFoundError(msg)

    inspection, source_held_out, checkpoint = _source_predictions(held_bundle, held_run)

    if inspection.held_out != source_held_out:
        msg = (
            "offline bundle held-out predictions do not match the run's declared "
            "release checkpoint bitwise"
        )
        raise ValueError(msg)

    return PreparedRelease(
        held_bundle,
        _repo_id(repo_id),
        held_run,
        checkpoint,
        checkpoint.name,
        inspection.model_name,
        inspection.corpus_revision,
        inspection.license_identifier,
        inspection.seed,
        inspection.files,
        inspection.held_out,
    )


def _authorization(
    prepared: PreparedRelease,
    authorization: ReleaseAuthorization,
    token: str,
    /,
) -> None:
    """Validate every irreversible-publication gate before a remote call."""
    if not authorization.reviewer.strip():
        msg = "a PI or authorized legal reviewer must be named"
        raise ValueError(msg)

    if authorization.license_identifier != prepared.license_identifier:
        msg = (
            f"reviewed license {authorization.license_identifier!r} does not match "
            f"bundle license {prepared.license_identifier!r}"
        )
        raise ValueError(msg)

    if not authorization.corpus_redistribution_confirmed:
        msg = "corpus redistribution terms have not been confirmed"
        raise ValueError(msg)

    if not authorization.artifact_redistribution_confirmed:
        msg = "artifact redistribution terms have not been confirmed"
        raise ValueError(msg)

    if not token:
        msg = "release credentials are missing"
        raise ValueError(msg)


def _staging_path(evidence: Path, /) -> Path:
    """Return the deterministic atomic-finalization path for evidence."""
    return evidence.with_name(f".{evidence.name}.finalizing")


def _recovery_path(evidence: Path, /) -> Path:
    """Return the deterministic atomic-recovery path for evidence."""
    return evidence.with_name(f".{evidence.name}.recovering")


def _sync_directory(directory: Path, /) -> None:
    """Durably record a directory-entry change.

    Windows cannot open a directory to flush it, so there an entry is as
    durable as the file system records it.
    """
    if sys.platform == "win32":
        return

    descriptor = os.open(directory, os.O_RDONLY)

    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_durable_exclusive(path: Path, payload: str, /) -> None:
    """Create, flush and sync one text file without replacing anything."""
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{payload}\n")
        handle.flush()
        os.fsync(handle.fileno())

    _sync_directory(path.parent)


def _reservation_record(
    prepared: PreparedRelease,
    authorization: ReleaseAuthorization,
    /,
) -> dict[str, JsonValue]:
    """Return the durable intent recorded before the first network call."""
    return {
        "format": "lofi-hub-release-reservation",
        "repository": prepared.repo_id,
        "checkpoint": prepared.checkpoint_id,
        "corpus_revision": prepared.corpus_revision,
        "license": prepared.license_identifier,
        "reviewer": authorization.reviewer,
        "bundle_files": [
            {"path": record.path, "sha256": record.sha256} for record in prepared.files
        ],
        "held_out": {
            "items": prepared.held_out.items,
            "sha256": prepared.held_out.sha256,
        },
    }


def _reserve_evidence(
    prepared: PreparedRelease,
    authorization: ReleaseAuthorization,
    evidence: Path,
    /,
) -> None:
    """Durably reserve a clean evidence destination before any network call."""
    if evidence.exists():
        msg = f"release evidence already exists: {evidence}"
        raise FileExistsError(msg)

    if not evidence.parent.is_dir():
        msg = f"release evidence parent is not a directory: {evidence.parent}"
        raise FileNotFoundError(msg)

    for staging in (_staging_path(evidence), _recovery_path(evidence)):
        if staging.exists():
            msg = f"stale release evidence staging file exists: {staging}"
            raise FileExistsError(msg)

    payload = json.dumps(
        _reservation_record(prepared, authorization),
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )
    _write_durable_exclusive(evidence, payload)


def _finalize_evidence(evidence: Path, release: ReleaseEvidence, /) -> None:
    """Atomically replace a durable reservation with durable final evidence."""
    staging = _staging_path(evidence)
    reservation = evidence.read_text(encoding="utf-8").removesuffix("\n")
    payload = json.dumps(
        _evidence_record(release),
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )
    _write_durable_exclusive(staging, payload)
    staging.replace(evidence)

    try:
        _sync_directory(evidence.parent)
    except BaseException as failure:
        try:
            _write_durable_exclusive(staging, reservation)
            staging.replace(evidence)
            _sync_directory(evidence.parent)
        except OSError as restoration:
            msg = "release evidence finalization and reservation restoration failed"
            raise BaseExceptionGroup(msg, [failure, restoration]) from failure

        raise


def _error_record(error: BaseException, /) -> dict[str, JsonValue]:
    """Return a stable error summary without traceback-local state."""
    return {"type": type(error).__name__, "message": str(error)}


def _recovery_record(
    prepared: PreparedRelease,
    authorization: ReleaseAuthorization,
    revision: str,
    primary: BaseException,
    rollback: BaseException,
    /,
) -> dict[str, JsonValue]:
    """Return durable evidence that public visibility may require intervention."""
    return {
        "format": "lofi-hub-release-recovery",
        "repository": prepared.repo_id,
        "revision": revision,
        "visibility": "unknown-requires-private-verification",
        "checkpoint": prepared.checkpoint_id,
        "corpus_revision": prepared.corpus_revision,
        "license": prepared.license_identifier,
        "reviewer": authorization.reviewer,
        "primary_failure": _error_record(primary),
        "rollback_failure": _error_record(rollback),
        "bundle_files": [
            {"path": record.path, "sha256": record.sha256} for record in prepared.files
        ],
        "held_out": {
            "items": prepared.held_out.items,
            "sha256": prepared.held_out.sha256,
        },
    }


def _record_recovery_required(
    prepared: PreparedRelease,
    authorization: ReleaseAuthorization,
    revision: str,
    primary: BaseException,
    rollback: BaseException,
    evidence: Path,
    /,
) -> None:
    """Durably replace release intent with a recovery-required record."""
    staging = _recovery_path(evidence)
    payload = json.dumps(
        _recovery_record(
            prepared,
            authorization,
            revision,
            primary,
            rollback,
        ),
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )
    _write_durable_exclusive(staging, payload)
    staging.replace(evidence)
    _sync_directory(evidence.parent)


def _rollback_publication(
    prepared: PreparedRelease,
    authorization: ReleaseAuthorization,
    client: HubClient,
    token: str,
    revision: str,
    evidence: Path,
    primary: BaseException,
    /,
) -> None:
    """Restore private visibility or preserve every failure for recovery."""
    try:
        client.privatize(prepared.repo_id, token)
    except (HTTPError, OSError) as rollback:
        failures: list[BaseException] = [primary, rollback]

        try:
            _record_recovery_required(
                prepared,
                authorization,
                revision,
                primary,
                rollback,
                evidence,
            )
        except OSError as recovery:
            failures.append(recovery)

        msg = "public release failed and private-visibility rollback failed"
        raise BaseExceptionGroup(msg, failures) from primary


def _stage_bundle(
    prepared: PreparedRelease,
    destination: Path,
    /,
) -> Path:
    """Snapshot the exact prepared bundle bytes into a fresh local directory."""
    if destination.exists():
        msg = f"release staging destination already exists: {destination}"
        raise FileExistsError(msg)

    shutil.copytree(prepared.bundle, destination)
    found = _bundle_files(destination)

    if found != prepared.files:
        msg = "bundle files changed while staging immutable release bytes"
        raise ValueError(msg)

    for path in destination.rglob("*"):
        if path.is_file():
            path.chmod(0o444)

    return destination


def _verify_staged_bundle(prepared: PreparedRelease, staged: Path, /) -> None:
    """Refuse any staged-byte mutation during the upload call."""
    if _bundle_files(staged) != prepared.files:
        msg = "immutable staged bundle bytes changed during upload"
        raise ValueError(msg)


def _immutable_revision(revision: str, /) -> str:
    """Return one canonical full Git object identity from the Hub."""
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        msg = f"Hub upload returned a noncanonical immutable revision: {revision!r}"
        raise ValueError(msg)

    return revision


def _exact_download_path(found: Path, expected: Path, kind: str, /) -> Path:
    """Return one client download only when it occupies the requested path."""
    resolved = found.resolve()

    if resolved != expected.resolve():
        msg = f"Hub client returned an unexpected {kind} download path"
        raise ValueError(msg)

    return resolved


def _same_preparation(expected: PreparedRelease, found: PreparedRelease, /) -> None:
    """Refuse any local input change after the operator's preflight."""
    if found != expected:
        msg = "release inputs changed after local preparation"
        raise ValueError(msg)


def _same_round_trip(
    prepared: PreparedRelease,
    downloaded: _BundleInspection,
    /,
) -> None:
    """Refuse any remote identity or held-out prediction disagreement."""
    expected = _BundleInspection(
        prepared.model_name,
        prepared.corpus_revision,
        prepared.license_identifier,
        prepared.seed,
        prepared.files,
        prepared.held_out,
    )

    if downloaded != expected:
        msg = "downloaded Hub revision does not match the prepared release exactly"
        raise ValueError(msg)


def _evidence_record(evidence: ReleaseEvidence, /) -> dict[str, JsonValue]:
    """Return the canonical JSON record for one verified public revision."""
    return {
        "format": "lofi-hub-release",
        "repository": evidence.repo_id,
        "revision": evidence.revision,
        "checkpoint": evidence.checkpoint_id,
        "corpus_revision": evidence.corpus_revision,
        "license": evidence.license_identifier,
        "reviewer": evidence.reviewer,
        "bundle_files": [
            {"path": record.path, "sha256": record.sha256} for record in evidence.files
        ],
        "held_out": {
            "items": evidence.held_out.items,
            "sha256": evidence.held_out.sha256,
        },
    }


def publish_release(
    prepared: PreparedRelease,
    authorization: ReleaseAuthorization,
    client: HubClient,
    token: str,
    evidence_path: Path,
    /,
) -> ReleaseEvidence:
    """Upload, download anonymously, verify, and record one public revision."""
    evidence_destination = evidence_path.resolve()
    _authorization(
        prepared,
        authorization,
        token,
    )
    current = prepare_release(
        prepared.bundle,
        prepared.repo_id,
        prepared.run,
    )
    _same_preparation(prepared, current)

    with TemporaryDirectory(prefix="lofi-hub-release-") as directory:
        clean_root = Path(directory)
        staged_bundle = _stage_bundle(prepared, clean_root / "staged-bundle")
        _reserve_evidence(prepared, authorization, evidence_destination)
        client.require_absent(prepared.repo_id, token)
        client.create_private(prepared.repo_id, token)
        uploaded_revision = client.upload_private(
            staged_bundle,
            prepared.repo_id,
            token,
        )
        _verify_staged_bundle(prepared, staged_bundle)
        revision = _immutable_revision(uploaded_revision)
        authenticated_destination = clean_root / "authenticated-bundle"
        authenticated_path = _exact_download_path(
            client.download_authenticated(
                prepared.repo_id,
                revision,
                token,
                authenticated_destination,
            ),
            authenticated_destination,
            "authenticated",
        )

        authenticated = _inspect_bundle(authenticated_path, prepared.seed)
        _same_round_trip(prepared, authenticated)

        try:
            client.publicize(prepared.repo_id, token)
            anonymous_destination = clean_root / "anonymous-bundle"
            anonymous_path = _exact_download_path(
                client.download_anonymous(
                    prepared.repo_id,
                    revision,
                    anonymous_destination,
                ),
                anonymous_destination,
                "anonymous",
            )

            anonymous = _inspect_bundle(anonymous_path, prepared.seed)
            _same_round_trip(prepared, anonymous)
            evidence = ReleaseEvidence(
                prepared.repo_id,
                revision,
                prepared.checkpoint_id,
                prepared.corpus_revision,
                prepared.license_identifier,
                authorization.reviewer,
                prepared.files,
                prepared.held_out,
            )
            _finalize_evidence(evidence_destination, evidence)
        except BaseException as primary:
            _rollback_publication(
                prepared,
                authorization,
                client,
                token,
                revision,
                evidence_destination,
                primary,
            )
            raise

    return evidence


def _prepared_record(prepared: PreparedRelease, /) -> dict[str, JsonValue]:
    """Return the read-only preflight summary printed by the command."""
    return {
        "repository": prepared.repo_id,
        "model": prepared.model_name,
        "checkpoint": prepared.checkpoint_id,
        "corpus_revision": prepared.corpus_revision,
        "license": prepared.license_identifier,
        "bundle_files": len(prepared.files),
        "held_out_items": prepared.held_out.items,
        "held_out_sha256": prepared.held_out.sha256,
    }


@final
class HubRelease:
    """Prepares, and when authorized publishes, one model release.

    Parameters
    ----------
    client : HubClient | None
        The Hub the release is published to; the Hugging Face Hub when None.
    """

    __slots__ = ("_client",)

    def __init__(self, client: HubClient | None = None) -> None:
        self._client = client

    def __call__(
        self,
        bundle: Path,
        repo_id: str,
        run: Path,
        /,
        *,
        publication: ReleasePublication | None,
        environ: Mapping[str, str],
    ) -> dict[str, JsonValue]:
        """Verify a release locally and publish it when a publication is given.

        Parameters
        ----------
        bundle : Path
            The offline bundle ``lofi export`` wrote.
        repo_id : str
            The ``owner/name`` repository the release is published to.
        run : Path
            The run directory whose declared release checkpoint the bundle
            reproduces.
        publication : ReleasePublication | None
            The authorization of a public release, or None for the read-only
            preflight.
        environ : Mapping[str, str]
            The environment the Hub token is read from.

        Returns
        -------
        dict[str, JsonValue]
            The preflight summary, or the evidence of the published release.
        """
        prepared = prepare_release(bundle, repo_id, run)

        if publication is None:
            return _prepared_record(prepared)

        released = publish_release(
            prepared,
            publication.authorization,
            HuggingFaceHub() if self._client is None else self._client,
            environ.get(publication.token_environment, ""),
            publication.evidence,
        )

        return _evidence_record(released)

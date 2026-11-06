"""The records of a model-only Hugging Face bundle manifest, and its codec."""

from pathlib import PurePosixPath
from typing import Literal, override

import didactic.api as dx

from lofi._checks import ensure_name, ensure_non_negative
from lofi.spec import SeedDerivationRecord

__all__ = [
    "HubBundleManifest",
    "HubBundleManifestIso",
    "HubComponentRecord",
    "HubFileRecord",
    "HubLicenseRecord",
    "HubReleaseDeclaration",
    "HubReleaseRecord",
    "HubReproducibilityRecord",
    "HubTensorRecord",
]


def _sha256(field: str, value: str, /) -> str:
    """Return one lowercase SHA-256 identity."""
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        msg = f"{field} is not a lowercase SHA-256 digest: {value!r}"
        raise ValueError(msg)

    return value


class HubFileRecord(dx.Model, extra="forbid"):
    """A bundle-relative file and the digest of its exact bytes."""

    path: str
    digest: str

    @dx.validates("path")
    @classmethod
    def _is_a_relative_file(cls, value: str) -> str:
        """Validate a canonical bundle-relative file name."""
        named = ensure_name("a bundle file", value)
        path = PurePosixPath(named)

        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            msg = f"a bundle file is a canonical relative path, not {value!r}"
            raise ValueError(msg)

        return named

    @dx.validates("digest")
    @classmethod
    def _has_a_digest(cls, value: str) -> str:
        """Validate the exact file digest."""
        return _sha256("a bundle file digest", value)


class HubTensorRecord(dx.Model, extra="forbid"):
    """The kind, shape and dtype of one top-level role-state value."""

    kind: Literal["tensor", "value"]
    shape: tuple[int, ...]
    dtype: str

    @dx.model_validator()
    def _is_a_tensor_description(self) -> HubTensorRecord:
        """Validate a named dtype and nonnegative dimensions."""
        ensure_name("a tensor dtype", self.dtype)

        for dimension in self.shape:
            ensure_non_negative("a tensor dimension", dimension)

        if self.kind == "value" and self.shape:
            msg = "a non-tensor state value cannot declare tensor dimensions"
            raise ValueError(msg)

        return self


class HubComponentRecord(dx.Model, extra="forbid"):
    """One exact component role and its sole model-state block."""

    tag: str
    spec_fingerprint: str
    state: HubFileRecord
    tensors: dict[str, HubTensorRecord]

    @dx.validates("tag")
    @classmethod
    def _has_a_tag(cls, value: str) -> str:
        """Validate the registered component tag."""
        return ensure_name("a Hub component tag", value)

    @dx.validates("spec_fingerprint")
    @classmethod
    def _has_a_specification_digest(cls, value: str) -> str:
        """Validate the component declaration digest."""
        return _sha256("a Hub component specification fingerprint", value)

    @dx.validates("tensors")
    @classmethod
    def _has_named_state(
        cls, value: dict[str, HubTensorRecord]
    ) -> dict[str, HubTensorRecord]:
        """Validate every state key."""
        for key in value:
            ensure_name("a component state key", key)

        return value


class HubLicenseRecord(dx.Model, extra="forbid"):
    """One provenance-sourced license and its optional scope."""

    identifier: str
    applies_to: str | None

    @dx.validates("identifier")
    @classmethod
    def _has_an_identifier(cls, value: str) -> str:
        """Validate the license identifier."""
        return ensure_name("a license identifier", value)


class HubReproducibilityRecord(dx.Model, extra="forbid"):
    """Recorded production metadata carried into a model release."""

    code_uri: str | None
    commit_hash: str | None
    command: str | None
    environment: str | None
    random_seed: int | None


class HubReleaseRecord(dx.Model, extra="forbid"):
    """Hub-card metadata and exact model identities."""

    model_name: str
    datasets: tuple[str, ...]
    languages: tuple[str, ...]
    grammar_identity: str
    codec_identity: str
    action_set_identity: str
    corpus_revision: str
    source_method: str
    license_expression: str
    licenses: tuple[HubLicenseRecord, ...]
    eprint_refs: tuple[str, ...]
    reproducibility: HubReproducibilityRecord | None
    pretrained_roles: tuple[str, ...]

    @dx.validates(
        "model_name",
        "corpus_revision",
        "source_method",
        "license_expression",
    )
    @classmethod
    def _has_required_text(cls, value: str) -> str:
        """Validate required release text."""
        return ensure_name("Hub release metadata", value)

    @dx.validates("grammar_identity", "codec_identity", "action_set_identity")
    @classmethod
    def _has_a_model_identity(cls, value: str) -> str:
        """Validate a lowercase SHA-256 model identity."""
        return _sha256("a Hub model identity", value)

    @dx.validates("datasets", "languages", "eprint_refs")
    @classmethod
    def _names_each_entry(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Validate a release list whose every entry is named."""
        return tuple(ensure_name("a release list entry", value) for value in values)

    @dx.validates("pretrained_roles")
    @classmethod
    def _roles_are_ordered(cls, roles: tuple[str, ...]) -> tuple[str, ...]:
        """Validate distinct, named pretrained roles in lexical order."""
        named = tuple(ensure_name("a release pretrained role", role) for role in roles)

        if named != tuple(sorted(set(named))):
            msg = "pretrained roles must be distinct and in lexical order"
            raise ValueError(msg)

        return named


class HubReleaseDeclaration(dx.Model, extra="forbid"):
    """Declares the model-card metadata a trained point is released under.

    The model identities, the corpus revision, the source revision and the
    random seed are read from the trained point itself, so a declaration
    names none of them.

    Parameters
    ----------
    model_name : str
        The name the model card displays.
    datasets : tuple[str, ...]
        The corpora the model was trained on, as the card lists them.
    languages : tuple[str, ...]
        The language tags the card lists.
    source_method : str
        How the training annotation was arrived at.
    license_expression : str
        The license the release is distributed under.
    licenses : tuple[HubLicenseRecord, ...]
        Each license the training data is released under, with its scope.
    eprint_refs : tuple[str, ...]
        The publications describing the training data.
    code_uri : str
        Where the code that trained the point is published.
    command : str
        The command line that trained the point.
    environment : str
        The environment the point was trained in.
    pretrained_resources : dict[str, str]
        The local directory holding each pretrained component's construction
        assets, by exact component role.
    """

    model_name: str
    datasets: tuple[str, ...]
    languages: tuple[str, ...]
    source_method: str
    license_expression: str
    licenses: tuple[HubLicenseRecord, ...]
    eprint_refs: tuple[str, ...] = ()
    code_uri: str
    command: str
    environment: str
    pretrained_resources: dict[str, str] = dx.field(default_factory=dict)

    @dx.validates(
        "model_name",
        "source_method",
        "license_expression",
        "code_uri",
        "command",
        "environment",
    )
    @classmethod
    def _has_required_text(cls, value: str) -> str:
        """Validate required release text."""
        return ensure_name("Hub release metadata", value)

    @dx.validates("datasets", "languages", "eprint_refs")
    @classmethod
    def _names_each_entry(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Validate a release list whose every entry is named."""
        return tuple(ensure_name("a release list entry", value) for value in values)

    @dx.validates("pretrained_resources")
    @classmethod
    def _resources_are_named(cls, resources: dict[str, str]) -> dict[str, str]:
        """Validate named pretrained roles and resource directories."""
        for role, directory in resources.items():
            ensure_name("a release pretrained role", role)
            ensure_name(f"the resource directory of {role!r}", directory)

        return resources


class HubBundleManifest(dx.Model, extra="forbid"):
    """A truthful model-only offline bundle manifest."""

    format: Literal["lofi-hub-model"]
    seed_derivation: SeedDerivationRecord
    release: HubReleaseRecord
    configuration: HubFileRecord
    card: HubFileRecord
    components: dict[str, HubComponentRecord]
    resources: dict[str, dict[str, HubFileRecord]]

    @dx.model_validator()
    def _seed_and_resource_roles_are_exact(self) -> HubBundleManifest:
        """Validate the seed and construction-resource role names."""
        reproducibility = self.release.reproducibility

        if (
            reproducibility is not None
            and reproducibility.random_seed is not None
            and reproducibility.random_seed != self.seed_derivation.root
        ):
            msg = (
                f"release metadata records the random seed "
                f"{reproducibility.random_seed}, but the point derivation records "
                f"{self.seed_derivation.root}"
            )
            raise ValueError(msg)

        for role in self.components:
            ensure_name("a Hub component role", role)

        if tuple(sorted(self.resources)) != self.release.pretrained_roles:
            msg = (
                f"release metadata names pretrained roles "
                f"{list(self.release.pretrained_roles)}, but resources name "
                f"{sorted(self.resources)}"
            )
            raise ValueError(msg)

        if not set(self.resources) <= set(self.components):
            extra = sorted(set(self.resources) - set(self.components))
            msg = f"construction resources name absent component roles {extra}"
            raise ValueError(msg)

        return self


class HubBundleManifestIso(dx.Iso[HubBundleManifest, bytes]):
    """Canonical JSON codec for the sole current Hub-bundle manifest."""

    @override
    def forward(self, source: HubBundleManifest, /) -> bytes:
        """Encode a model-only bundle manifest."""
        return source.model_dump_json().encode()

    @override
    def backward(self, target: bytes, /) -> HubBundleManifest:
        """Decode and validate a model-only bundle manifest."""
        return HubBundleManifest.model_validate_json(target)

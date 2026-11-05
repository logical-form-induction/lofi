"""Boundary models for independently fitted baseline systems."""

import json
from typing import TYPE_CHECKING, Final, Literal, Protocol, cast

import didactic.api as dx

from lofi.run._boundary.manifest import SeedDerivationRecord
from lofi.spec import Composition, SpecLoader, load_plugins
from lofi.spec.components.baseline import BaselineSystemSpec

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from lofi._types import JsonValue

__all__ = [
    "BASELINE_FIT_MANIFEST_NAME",
    "BASELINE_ROLE",
    "BaselineFitCodec",
    "BaselineFitManifest",
    "BaselineFitSpec",
    "decode_baseline_fit",
    "decode_baseline_manifest",
    "encode_baseline_manifest",
]

BASELINE_ROLE: Final = "baseline"
"""The evaluator role under which a baseline fit reports its rows."""

BASELINE_FIT_MANIFEST_NAME: Final = "baseline-fit.json"
"""The file a completed baseline-fit point writes its manifest to."""


class _ConfiguredBaseline(Protocol):
    """A baseline declaration exposing the rows it promises."""

    def metrics(self) -> tuple[str, ...]:
        """Return the promised metric names in row order."""
        ...


class BaselineFitSpec(dx.Model, extra="forbid"):
    """Declare one external baseline fit."""

    system: BaselineSystemSpec

    @dx.model_validator()
    def _system_reports_metrics(self) -> BaselineFitSpec:
        """Return a fit whose system promises distinct named metrics."""
        metrics = cast("_ConfiguredBaseline", self.system).metrics()
        if not metrics or any(not metric.strip() for metric in metrics):
            msg = "a baseline system must declare at least one named metric"
            raise ValueError(msg)
        if len(metrics) != len(set(metrics)):
            msg = f"a baseline system repeats a metric: {metrics!r}"
            raise ValueError(msg)
        return self


class BaselineFitManifest(dx.Model, extra="forbid"):
    """Record one completed baseline fit and its execution identity."""

    kind: Literal["baseline-fit"] = "baseline-fit"
    fit: BaselineFitSpec
    source_digests: dict[str, str]
    seed_derivation: SeedDerivationRecord
    package_versions: dict[str, str]
    git_revision: str

    @dx.validates("source_digests")
    @classmethod
    def _sources_have_content_identities(
        cls, source_digests: dict[str, str]
    ) -> dict[str, str]:
        """Return a nonempty mapping of named SHA-256 content identities."""
        if not source_digests:
            msg = "a baseline-fit manifest has no source content identities"
            raise ValueError(msg)
        for source, digest in source_digests.items():
            if not source.strip():
                msg = "a baseline-fit manifest has an unnamed source"
                raise ValueError(msg)
            if (
                not digest.startswith("sha256:")
                or len(digest) != 71
                or any(character not in "0123456789abcdef" for character in digest[7:])
            ):
                msg = f"baseline source {source!r} has no SHA-256 identity"
                raise ValueError(msg)
        return dict(sorted(source_digests.items()))


def decode_baseline_fit(document: Mapping[str, JsonValue], /) -> BaselineFitSpec:
    """Validate a resolved baseline-fit declaration at its run boundary."""
    load_plugins()
    return BaselineFitSpec.model_validate_json(json.dumps(dict(document)))


def decode_baseline_manifest(data: bytes, /) -> BaselineFitManifest:
    """Validate persisted baseline-fit manifest bytes."""
    load_plugins()
    try:
        return BaselineFitManifest.model_validate_json(data)
    except dx.ValidationError as refused:
        msg = f"the bytes do not validate as a baseline-fit manifest: {refused}"
        raise ValueError(msg) from refused


def encode_baseline_manifest(manifest: BaselineFitManifest, /) -> bytes:
    """Encode a validated baseline-fit manifest deterministically."""
    document = json.loads(manifest.model_dump_json())
    return (json.dumps(document, sort_keys=True, indent=2) + "\n").encode()


class BaselineFitCodec:
    """Compose baseline-fit documents through their declared boundary."""

    __slots__ = ("_loader", "_plugins_loaded")

    def __init__(self, *, search_path: tuple[Path, ...] = ()) -> None:
        self._loader = SpecLoader(search_path=search_path)
        self._plugins_loaded = False

    @property
    def search_path(self) -> tuple[Path, ...]:
        """Return the roots searched after the document's directory."""
        return self._loader.search_path

    def __call__(
        self,
        path: Path,
        /,
        *,
        overrides: tuple[str, ...] = (),
    ) -> BaselineFitSpec:
        """Compose and validate one baseline-fit document."""
        return self.compose_traced(path, overrides=overrides).spec

    def compose_traced(
        self,
        path: Path,
        /,
        *,
        overrides: tuple[str, ...] = (),
    ) -> Composition[BaselineFitSpec]:
        """Compose a baseline fit and retain the source of each leaf."""
        if not self._plugins_loaded:
            load_plugins()
            self._plugins_loaded = True
        return self._loader.compose_traced(
            path, schema=BaselineFitSpec, overrides=overrides
        )

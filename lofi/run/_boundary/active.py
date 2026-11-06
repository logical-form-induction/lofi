"""Boundary records of an active learning run.

A run records its manifest beside its round rows. The manifest holds the
resolved active learning document, whether the run completed or stopped with a
batch awaiting its responses, why a completed run stopped, how many rounds it
recorded, the judgments it started from, and the content identity of every
document it read. A run that
an experiment executed also records the seed derivation of its point.
"""

import json
from typing import TYPE_CHECKING, Final, Literal

import didactic.api as dx

from lofi.run._boundary.manifest import SeedDerivationRecord
from lofi.spec import SpecLoader, load_plugins
from lofi.spec.acquisition import ActiveLearningSpec, ActiveStopReason

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from lofi._types import JsonValue

__all__ = [
    "ACTIVE_LEARNING_MANIFEST_NAME",
    "ACTIVE_LEARNING_METRICS",
    "ACTIVE_LEARNING_ROLE",
    "BATCH_NAME",
    "RESPONSES_NAME",
    "ROUNDS_NAME",
    "ActiveLearningManifest",
    "ActiveLearningStatus",
    "awaited_batch",
    "compose_active_learning",
    "decode_active_learning_document",
    "decode_active_learning_manifest",
    "encode_active_learning_manifest",
    "round_directory",
]

ACTIVE_LEARNING_ROLE: Final = "active-learning"
"""The role an active learning run's round rows are reported under."""

ACTIVE_LEARNING_METRICS: Final = (
    "agreement",
    "rank-correlation",
    "batch-variance",
    "stratum-share",
)
"""The metrics each recorded round reports, one row each, a share per stratum."""

ACTIVE_LEARNING_MANIFEST_NAME: Final = "active-learning.json"
"""The file an active learning run records its manifest in."""

ROUNDS_NAME: Final = "rounds"
"""The directory, under a run's, holding one directory per round."""

BATCH_NAME: Final = "batch.jsonl"
"""The bead items of a round's batch, one per line."""

RESPONSES_NAME: Final = "responses.jsonl"
"""The bead annotation records answering a round's batch, one per line."""

type ActiveLearningStatus = Literal["completed", "awaiting-responses"]
"""Whether a run completed or stopped with a batch awaiting its responses."""


class ActiveLearningManifest(dx.Model, extra="forbid"):
    """Record one active learning run and what it read.

    Attributes
    ----------
    kind : Literal["active-learning"]
        The manifest kind.
    document : ActiveLearningSpec
        The resolved active learning document.
    status : ActiveLearningStatus
        Whether the run completed or awaits the responses to a batch.
    rounds : int
        Rounds whose responses were collected.
    awaiting : int | None
        The round whose batch awaits responses, 0 for an initial draw, or None
        for a completed run.
    stop_reason : ActiveStopReason | None
        Why a completed run stopped: the stopping rule's kind, an exhausted
        pool, or the round bound; None while a batch is awaited.
    initial_judgments : int
        Judgments read before the first round.
    initial_skipped : int
        Initial records set aside because they answer another question.
    source_digests : dict[str, str]
        SHA-256 content identity of every document read, by declared path.
    seed : int
        The run seed every stream of the run derives from.
    seed_derivation : SeedDerivationRecord | None
        How an experiment derived the run seed, or None for a run outside one.
    package_versions : dict[str, str]
        Installed package versions.
    git_revision : str
        Source revision of the executing checkout.
    """

    kind: Literal["active-learning"] = "active-learning"
    document: ActiveLearningSpec
    status: ActiveLearningStatus
    rounds: int
    awaiting: int | None
    stop_reason: ActiveStopReason | None
    initial_judgments: int
    initial_skipped: int
    source_digests: dict[str, str]
    seed: int
    seed_derivation: SeedDerivationRecord | None
    package_versions: dict[str, str]
    git_revision: str

    @dx.model_validator()
    def _status_names_its_round(self) -> ActiveLearningManifest:
        """Validate that exactly an awaiting run names the round it awaits."""
        if (self.status == "awaiting-responses") != (self.awaiting is not None):
            msg = (
                f"an active-learning manifest {self.status!r} names awaited "
                f"round {self.awaiting!r}"
            )
            raise ValueError(msg)
        if (self.status == "completed") != (self.stop_reason is not None):
            msg = (
                f"an active-learning manifest {self.status!r} records stop "
                f"reason {self.stop_reason!r}"
            )
            raise ValueError(msg)
        return self


def round_directory(directory: Path, round_index: int, /) -> Path:
    """Return the directory a run keeps one round's batch and responses in."""
    return directory / ROUNDS_NAME / str(round_index)


def awaited_batch(directory: Path, manifest: ActiveLearningManifest, /) -> Path | None:
    """Return the batch file a manifest's run awaits responses to, or None."""
    if manifest.awaiting is None:
        return None
    return round_directory(directory, manifest.awaiting) / BATCH_NAME


def decode_active_learning_document(
    document: Mapping[str, JsonValue], /
) -> ActiveLearningSpec:
    """Validate a resolved active learning document at its run boundary."""
    load_plugins()
    return ActiveLearningSpec.model_validate_json(json.dumps(dict(document)))


def compose_active_learning(
    path: Path, /, *, search_path: tuple[Path, ...] = ()
) -> ActiveLearningSpec:
    """Compose and validate one active learning document from its file."""
    load_plugins()
    return (
        SpecLoader(search_path=search_path)
        .compose_traced(path, schema=ActiveLearningSpec)
        .spec
    )


def decode_active_learning_manifest(data: bytes, /) -> ActiveLearningManifest:
    """Validate persisted active learning manifest bytes.

    Raises
    ------
    ValueError
        If the bytes are not an active learning manifest.
    """
    load_plugins()
    try:
        return ActiveLearningManifest.model_validate_json(data)
    except dx.ValidationError as refused:
        msg = f"the bytes do not validate as an active-learning manifest: {refused}"
        raise ValueError(msg) from refused


def encode_active_learning_manifest(manifest: ActiveLearningManifest, /) -> bytes:
    """Encode a validated active learning manifest deterministically."""
    document = json.loads(manifest.model_dump_json())
    return (json.dumps(document, sort_keys=True, indent=2) + "\n").encode()

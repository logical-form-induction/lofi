"""Checkpoint cadence and retention declaration."""

import didactic.api as dx

from lofi._checks import ensure_positive

__all__ = ["CheckpointPolicySpec"]


class CheckpointPolicySpec(dx.Model, extra="forbid"):
    """Declares when a trainer writes resumable checkpoints and how many it keeps.

    Attributes
    ----------
    interval : int
        Positive cadence in the trainer's own unit of progress: completed epochs
        for a trainer that fits in epochs, completed optimizer steps for one
        that fits in steps. The final position is always written.
    keep : int
        Positive number of the newest resumable checkpoints the run keeps. An
        older one is removed once a newer one is written.
    """

    interval: int
    keep: int

    @dx.validates("interval", "keep")
    @classmethod
    def _counts_are_positive(cls, value: int) -> int:
        """Validate a positive cadence or retention count."""
        return ensure_positive("a checkpoint policy count", value)

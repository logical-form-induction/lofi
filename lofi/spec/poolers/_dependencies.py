"""What every subword pooler takes from the graph."""

from typing import Final

from lofi.spec._ports import Dependency

__all__ = ["ENCODER_DEPENDENCY"]

ENCODER_DEPENDENCY: Final[tuple[Dependency, ...]] = (
    Dependency("token_encoder", "encoder"),
)
"""The rows a pooler gathers, whose width it runs at, come from the token encoder."""

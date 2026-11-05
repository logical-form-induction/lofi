"""Names shared by declared decoder forward modes."""

from typing import Literal

__all__ = ["ForwardModeName"]


type ForwardModeName = Literal["observe", "sample", "marginalize"]
"""The name of a decoder forward-pass mode."""

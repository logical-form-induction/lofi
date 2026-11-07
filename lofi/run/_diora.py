"""Construct the released DIORA backend from a fetched artifact."""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING, override
from zipfile import ZipFile

import torch

from lofi.nn.parsers import ReleasedDioraMLP

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor
    from torch.nn import Module


__all__ = ["ReleasedDioraCheckpointFactory"]


def _member(artifact: Path, member: str, /) -> bytes:
    """Return one stored file of a fetched snapshot directory or zip archive."""
    if artifact.is_dir():
        return (artifact / member).read_bytes()

    with ZipFile(artifact) as bundle:
        return bundle.read(member)


def _released_state(artifact: Path, member: str, /) -> dict[str, Tensor]:
    """Return the tensor state stored at one member of the released artifact."""
    serialized = _member(artifact, member)

    loaded: object = torch.load(
        BytesIO(serialized), map_location="cpu", weights_only=True
    )

    if not isinstance(loaded, dict):
        msg = "the released DIORA checkpoint is not a named checkpoint record"
        raise TypeError(msg)

    state: object = loaded.get("state_dict")

    if not isinstance(state, dict):
        msg = "the released DIORA checkpoint has no tensor state dictionary"
        raise TypeError(msg)

    tensors: dict[str, Tensor] = {}

    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            msg = "the released DIORA state dictionary is not string-to-tensor"
            raise TypeError(msg)

        tensors[name] = value

    return tensors


class ReleasedDioraCheckpointFactory:
    """Build the released MLP DIORA backend from one stored state file.

    Parameters
    ----------
    member : str
        The relative path of the state file inside the fetched artifact: a
        member of a zip archive, or a file of a snapshot directory.
    """

    __slots__ = ("_member",)

    def __init__(self, member: str) -> None:
        self._member = member

    def build(self, checkpoint: Path, /) -> Module:
        """Return the MLP DIORA stored in the verified artifact."""
        return ReleasedDioraMLP(_released_state(checkpoint, self._member))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this factory."""
        return f"ReleasedDioraCheckpointFactory({self._member!r})"

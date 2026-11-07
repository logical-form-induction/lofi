from __future__ import annotations

from collections import OrderedDict
from hashlib import sha256
from io import BytesIO
from typing import TYPE_CHECKING
from zipfile import ZIP_STORED, ZipFile

import pytest
import torch

from lofi.deduce import SpanIndex
from lofi.nn.sites import Role
from lofi.probe import ExternalModel, diora_model_spec
from lofi.run._diora import ReleasedDioraCheckpointFactory
from lofi.spec.artifacts import UrlSourceSpec

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


MEMBER = "diora-checkpoints/mlp-softmax/model.pt"


def _serialized_state() -> bytes:
    width = 2
    input_width = 3
    state: OrderedDict[str, torch.Tensor] = OrderedDict(
        {
            "module.embed.mat": torch.eye(width, input_width),
            "module.diora.root_vector_out_h": torch.ones(width),
            "module.diora.inside_score_func.mat": torch.eye(width),
            "module.diora.outside_score_func.mat": torch.eye(width),
            "module.diora.inside_compose_func.V": torch.eye(width),
            "module.diora.inside_compose_func.W_0": torch.ones(width * 2, width),
            "module.diora.inside_compose_func.W_1": torch.eye(width),
            "module.diora.inside_compose_func.B": torch.zeros(width),
            "module.diora.inside_compose_func.B_1": torch.zeros(width),
            "module.diora.outside_compose_func.W_0": torch.ones(width * 2, width),
            "module.diora.outside_compose_func.W_1": torch.eye(width),
            "module.diora.outside_compose_func.B": torch.zeros(width),
            "module.diora.outside_compose_func.B_1": torch.zeros(width),
            "module.reconstruct_softmax_loss.mat": torch.ones(width, input_width),
        }
    )
    serialized = BytesIO()
    torch.save({"state_dict": state, "word2idx": {}}, serialized)

    return serialized.getvalue()


def _archive(root: Path) -> Path:
    archive = root / "diora-checkpoints.zip"

    with ZipFile(archive, "w", compression=ZIP_STORED) as bundle:
        bundle.writestr(MEMBER, _serialized_state())

    return archive


def _snapshot(root: Path) -> Path:
    directory = root / "snapshot"
    stored = directory / MEMBER
    stored.parent.mkdir(parents=True)
    stored.write_bytes(_serialized_state())

    return directory


@pytest.mark.parametrize("artifact", [_archive, _snapshot])
def test_the_released_factory_builds_the_published_chart_from_its_member(
    tmp_path: Path, artifact: Callable[[Path], Path]
) -> None:
    width = 2
    input_width = 3
    fetched = artifact(tmp_path)
    release = UrlSourceSpec(
        url="https://example.test/diora-checkpoints.zip",
        sha256=sha256(b"identity only").hexdigest(),
    )
    backend = ReleasedDioraCheckpointFactory(MEMBER).build(fetched)
    external = ExternalModel(backend, diora_model_spec("diora", release).site_map())
    sites = frozenset(external.site_map())
    index = SpanIndex(4)

    with external.instrument(sites, patches={}) as scope:
        output = external.run({"embeddings": torch.ones(2, index.n, input_width)})

    assert isinstance(output, torch.Tensor)
    assert output.shape == (2, width)
    readings = {site.role: scope.readings[site] for site in sites}
    assert readings[Role.SPLIT_DISTRIBUTION].tensor.shape == (
        2,
        index.num_spans,
        index.max_splits,
    )
    assert readings[Role.INSIDE_SCORE].tensor.shape == (2, index.num_spans)
    assert readings[Role.OUTSIDE_SCORE].tensor.shape == (2, index.num_spans)
    assert torch.isfinite(readings[Role.INSIDE_SCORE].tensor).all()
    assert torch.isfinite(readings[Role.OUTSIDE_SCORE].tensor).all()


def test_a_missing_member_is_refused(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match=r"absent\.pt"):
        ReleasedDioraCheckpointFactory("absent.pt").build(_archive(tmp_path))

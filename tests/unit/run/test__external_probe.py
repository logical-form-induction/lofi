from __future__ import annotations

from collections import OrderedDict
from hashlib import sha256
from io import BytesIO
from types import MappingProxyType
from typing import TYPE_CHECKING, cast
from zipfile import ZIP_STORED, ZipFile

import pytest
import torch

from lofi._artifacts import PinnedArchive
from lofi.probe import ProbeBuildContext, TrainedPoint
from lofi.run._boundary.external_probe import ExternalProbeSpec
from lofi.run._external_probe import PLUGIN
from lofi.spec import RunSpecCodec, config_search_path, load_plugins

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from lofi.nn.ports import Reading
    from lofi.probe import HeldOutPass, Probed


class _HostPoint:
    @property
    def components(self) -> Mapping[str, Probed[Reading]]:
        return MappingProxyType({})

    @property
    def passes(self) -> tuple[HeldOutPass, ...]:
        return ()


def _released_archive(root: Path) -> tuple[Path, str]:
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
        }
    )
    serialized = BytesIO()
    torch.save({"state_dict": state}, serialized)
    source = root / "release.zip"

    with ZipFile(source, "w", compression=ZIP_STORED) as bundle:
        bundle.writestr("diora-checkpoints/mlp-softmax/model.pt", serialized.getvalue())

    return source, sha256(source.read_bytes()).hexdigest()


def _document(cache: Path, digest: str) -> ExternalProbeSpec:
    return ExternalProbeSpec.model_validate(
        {
            "kind": "external",
            "artifacts": {"root": str(cache), "offline": True},
            "stimulus_lengths": [3, 4],
            "backend": {
                "kind": "released-mlp",
                "member": "diora-checkpoints/mlp-softmax/model.pt",
            },
            "model": {
                "model": "diora-test",
                "checkpoint": {
                    "kind": "url",
                    "url": "https://example.test/release.zip",
                    "sha256": digest,
                },
                "sites": [
                    {
                        "path": "alpha",
                        "role": "SPLIT_DISTRIBUTION",
                        "layer": 0,
                        "axes": ["item", "span", "split"],
                        "dtype_class": "log-probability",
                        "differentiable": True,
                        "access": ["OBSERVE"],
                    },
                    {
                        "path": "inside",
                        "role": "INSIDE_SCORE",
                        "layer": 0,
                        "axes": ["item", "span"],
                        "dtype_class": "real",
                        "differentiable": True,
                        "access": ["OBSERVE"],
                    },
                    {
                        "path": "outside",
                        "role": "OUTSIDE_SCORE",
                        "layer": 0,
                        "axes": ["item", "span"],
                        "dtype_class": "real",
                        "differentiable": True,
                        "access": ["OBSERVE"],
                    },
                ],
            },
        }
    )


def test_an_external_probe_document_loads_offline_and_writes_rows_by_length(
    tmp_path: Path,
) -> None:
    source, digest = _released_archive(tmp_path)
    cache = tmp_path / "cache"
    checkpoint = cache / "archives" / digest
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(source.read_bytes())
    host = _HostPoint()
    probe = PLUGIN.construct(_document(cache, digest), ProbeBuildContext("diora", host))
    probe.admit({})

    values = probe.run(cast("TrainedPoint", host))

    assert {value.measure for value in values} == {
        "split-posterior-kl/kl:length:3",
        "split-posterior-kl/kl:length:4",
    }
    assert all(value.count == 1 for value in values)


def test_the_packaged_diora_release_is_selected_from_a_document_and_fetched_offline(
    tmp_path: Path,
) -> None:
    document = tmp_path / "run.yaml"
    document.write_text(
        "defaults:\n"
        "  - run/synthetic-parser\n"
        "  - probes.diora: naacl-2019\n"
        "probes:\n"
        "  diora:\n"
        "    artifacts:\n"
        f"      root: {tmp_path / 'cache'}\n"
        "      offline: true\n"
        "    stimulus_lengths: [3]\n",
        encoding="utf-8",
    )
    load_plugins()
    run = RunSpecCodec(search_path=config_search_path(tmp_path))(document)
    declared = cast("ExternalProbeSpec", run.probes["diora"])

    assert declared.settings().model.checkpoint == PinnedArchive(
        "https://diora-naacl-2019.s3.amazonaws.com/diora-checkpoints.zip",
        "6794928e34afa4a3731e0312efdfbbd9e41bedf9cc27dea30c3d21f9c85dcde8",
    )

    with pytest.raises(FileNotFoundError, match="offline"):
        PLUGIN.construct(declared, ProbeBuildContext("diora", _HostPoint()))

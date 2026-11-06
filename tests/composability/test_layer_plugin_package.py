"""An installed package can extend neural-layer vocabularies in a run."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import torch
from tests.unit.nn.encoders import test_rnn as rnn_tests

from lofi.nn._layers import feedforward, recurrence
from lofi.nn.encoders import TreeRnnEncoder
from lofi.nn.sites import Role
from lofi.spec import PACKAGED_CONFIG_ROOT, RunSpec, RunSpecCodec
from lofi.spec.encoders.rnn import TreeRnnEncoderSpec

if TYPE_CHECKING:
    import pytest

    from lofi._types import JsonValue
    from lofi.spec.controllers import MlpControllerSpec
    from lofi.spec.poolers.attention import AttentionSubwordPoolerSpec


CONFIG: Final = (
    Path(__file__).resolve().parents[2] / "lofi/configs/run/synthetic-parser.yaml"
)
CODEC: Final = RunSpecCodec(search_path=(PACKAGED_CONFIG_ROOT,))


def _install_layer_plugins(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[str, str]:
    """Install one activation and recurrent cell through distribution metadata."""
    module = f"lofi_layer_plugin_{root.name.replace('-', '_')}"
    (root / f"{module}.py").write_text(
        """from typing import override
import torch
from torch import Tensor, nn
from lofi.nn._layers import ActivationPlugin, CellStep, Gate, RecurrentCellPlugin
from lofi.nn.sites import DTypeClass

class Square(nn.Module):
    @override
    def forward(self, values: Tensor) -> Tensor:
        return values.square()

def square() -> nn.Module:
    return Square()

SQUARE = ActivationPlugin(square)
def leaky(
    inputs: tuple[Tensor, ...],
    hidden: tuple[Tensor, ...],
    previous: Tensor,
    memory: Tensor,
    /,
) -> CellStep:
    del memory
    keep = torch.sigmoid(inputs[0] + hidden[0])
    candidate = torch.tanh(inputs[1] + hidden[1])
    reached = keep * previous + (1.0 - keep) * candidate
    return CellStep(reached, reached, (keep, candidate))

CUSTOM_GRU = RecurrentCellPlugin(
    nn.GRU,
    leaky,
    gates=(Gate("keep", DTypeClass.PROBABILITY), Gate("candidate", DTypeClass.REAL)),
    memory=False,
)
""",
        encoding="utf-8",
        newline="\n",
    )
    metadata = root / "lofi_layer_fixture-1.0.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: lofi-layer-fixture\nVersion: 1.0.0\n",
        encoding="utf-8",
        newline="\n",
    )
    (metadata / "entry_points.txt").write_text(
        f"[lofi.activation]\ntest-square = {module}:SQUARE\n"
        f"[lofi.recurrent_cell]\ntest-gru = {module}:CUSTOM_GRU\n",
        encoding="utf-8",
        newline="\n",
    )
    monkeypatch.syspath_prepend(str(root))

    return "test-square", "test-gru"


def _declared_run(activation: str, cell: str) -> RunSpec:
    """Return a run selecting the two external layer plugins."""
    base = CODEC(CONFIG)
    document = cast("dict[str, JsonValue]", json.loads(base.model_dump_json()))
    model = cast("dict[str, JsonValue]", document["model"])
    controller = cast("dict[str, JsonValue]", model["controller"])
    controller["hidden"] = [
        {"width": 64, "activation": activation, "dropout": 0.1},
        {"width": 32, "activation": activation, "dropout": 0.1},
    ]
    model["subword_pooler"] = {
        "kind": "attention",
        "context": {"cell": cell, "layers": 1, "bidirectional": False},
    }
    return RunSpec.model_validate(document)


def test_external_layer_plugins_selected_from_a_run_train_a_tiny_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    activation, cell = _install_layer_plugins(tmp_path, monkeypatch)
    run = _declared_run(activation, cell)
    controller = cast("MlpControllerSpec", run.model.controller)
    pooler = cast("AttentionSubwordPoolerSpec", run.model.subword_pooler)
    context = pooler.context

    assert context is not None
    assert [layer.width for layer in controller.hidden] == [64, 32]
    assert [layer.dropout for layer in controller.hidden] == [0.1, 0.1]
    assert [layer.activation for layer in controller.hidden] == [activation, activation]
    assert context.cell == cell

    head = feedforward(
        tuple(layer.settings() for layer in controller.hidden),
        inputs=4,
        output=2,
        bias=True,
    )
    encoder = recurrence(
        context.cell,
        inputs=4,
        width=3,
        layers=context.layers,
        bidirectional=context.bidirectional,
    )
    parameters = (*head.parameters(), *encoder.parameters())
    before = tuple(parameter.detach().clone() for parameter in parameters)
    optimizer = torch.optim.SGD(parameters, lr=0.01)
    encoded, _ = encoder(torch.randn(2, 3, 4))
    loss = head(torch.randn(2, 4)).square().mean() + encoded.square().mean()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert any(
        not torch.equal(initial, trained)
        for initial, trained in zip(before, parameters, strict=True)
    )


def test_an_external_cell_steps_a_tree_encoder_and_publishes_its_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, cell = _install_layer_plugins(tmp_path, monkeypatch)
    declared = TreeRnnEncoderSpec(width=rnn_tests.WIDTH, cell=cell, layers=1)
    settings = declared.settings()
    encoder = TreeRnnEncoder(
        codec=rnn_tests.VIEW,
        choices=rnn_tests.CHOICES,
        component=rnn_tests.COMPONENT,
        scheme=rnn_tests.scheme(width=settings.width),
        cell=settings.cell,
        layers=settings.layers,
        norm_cap=settings.norm_cap,
        inventory=settings.inventory,
    )
    before = tuple(parameter.detach().clone() for parameter in encoder.parameters())
    optimizer = torch.optim.SGD(encoder.parameters(), lr=0.1)

    with encoder.instrument(frozenset(encoder.site_map()), patches={}) as scope:
        loss = encoder.encode(rnn_tests.TERMS).tensor.square().mean()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    assert encoder.cell == cell
    assert {site.key for site in scope.readings if site.role is Role.GATE} == {
        (0, "keep"),
        (0, "candidate"),
    }
    assert any(
        not torch.equal(initial, trained)
        for initial, trained in zip(before, encoder.parameters(), strict=True)
    )

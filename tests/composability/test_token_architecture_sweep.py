"""A sweep over the span decoder's and attention pooler's architectures trains.

The run document is the packaged parser run with an attention pooler, a span
decoder and a reconstruction term; the sweep document varies every architecture
field of the two, and the command line trains each point it expands to.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import torch

from lofi.run import (
    CHECKPOINTS_NAME,
    ExperimentSeed,
    GridSweepExpander,
    PointExecutor,
    RunStore,
    Selection,
    SerialLauncher,
    Sweep,
    SweepPoint,
    composer,
)
from lofi.run._boundary.experiment import FailurePolicySpec
from lofi.run.operations import TRAINING_OPERATIONS
from lofi.spec import (
    PACKAGED_CONFIG_ROOT,
    RunSpec,
    RunSpecCodec,
    SpecLoader,
    config_search_path,
)
from lofi.spec.layers import BUILTIN_ACTIVATIONS, BUILTIN_RECURRENT_CELLS

if TYPE_CHECKING:
    from torch import Tensor

    from lofi._types import JsonValue
    from lofi.spec.poolers.attention import AttentionSubwordPoolerSpec
    from lofi.spec.span_decoders.rnn import RnnSpanDecoderSpec

CONFIG: Final = (
    Path(__file__).resolve().parents[2] / "lofi/configs/run/synthetic-parser.yaml"
)
PACKAGED_CODEC: Final = RunSpecCodec(search_path=(PACKAGED_CONFIG_ROOT,))
ROLE: Final = "reconstruction"
POOLER: Final = "model.subword_pooler"
DECODER: Final = f"model.span_decoders.{ROLE}"

SWEEP: Final = {
    "grid": [{"path": f"{DECODER}.layers", "values": [1, 2]}],
    "yoked": [
        [
            {
                "path": f"{POOLER}.context.cell",
                "values": ["gru", "lstm", "elman-tanh", "elman-relu"],
            },
            {"path": f"{POOLER}.context.layers", "values": [1, 2, 1, 2]},
            {"path": f"{POOLER}.context.width", "values": [None, 4, 6, 3]},
            {
                "path": f"{POOLER}.context.bidirectional",
                "values": [True, False, False, True],
            },
            {
                "path": f"{POOLER}.scorer",
                "values": [
                    [],
                    [{"width": 5, "activation": "tanh", "dropout": 0.1}],
                    [
                        {"width": 6, "activation": "relu"},
                        {"width": 3, "activation": "gelu"},
                    ],
                    [{"width": 2, "activation": "silu"}],
                ],
            },
            {
                "path": f"{DECODER}.cell",
                "values": ["lstm", "elman-tanh", "elman-relu", "gru"],
            },
            {"path": f"{DECODER}.state_width", "values": [5, None, 3, 6]},
            {
                "path": f"{DECODER}.head",
                "values": [
                    [{"width": 4, "activation": "silu"}],
                    [],
                    [{"width": 3, "activation": "tanh", "dropout": 0.1}],
                    [
                        {"width": 4, "activation": "relu"},
                        {"width": 2, "activation": "gelu"},
                    ],
                ],
            },
        ]
    ],
    "seeds": [0],
    "expected_points": 8,
}


def run_document() -> dict[str, JsonValue]:
    """Return the packaged parser run with an attention pooler and a span decoder."""
    declared = PACKAGED_CODEC(CONFIG)
    document = cast("dict[str, JsonValue]", json.loads(declared.model_dump_json()))
    model = cast("dict[str, JsonValue]", document["model"])
    model["subword_pooler"] = {"kind": "attention", "context": {}}
    model["span_decoders"] = {ROLE: {"kind": "rnn-span-decoder"}}
    trainer = cast("dict[str, JsonValue]", document["trainer"])
    trainer["epochs"] = 1
    terms = cast(
        "list[JsonValue]", cast("dict[str, JsonValue]", document["loss"])["terms"]
    )
    terms.append(
        {"objective": {"kind": "span-reconstruction", "decoder": ROLE}, "weight": 1.0}
    )

    return document


def expanded(config: Path, sweep: Path, root: Path) -> tuple[SweepPoint, ...]:
    """Return the points the sweep document expands to over the run document."""
    selection = Selection(
        config=config,
        profile=None,
        overrides=(),
        overlays=(),
        search_path=config_search_path(root),
    )
    declared = (
        SpecLoader(search_path=selection.search_path)
        .compose_traced(sweep, schema=Sweep)
        .spec
    )

    compose_point = composer(selection)

    return GridSweepExpander(
        compose_point,
        provenance=compose_point.provenance,
        sources=compose_point.sources,
    ).expand(declared)


GATES: Final = {
    "gru": 3,
    "lstm": 4,
    "elman-tanh": 1,
    "elman-relu": 1,
}


def trained(store: RunStore, point: SweepPoint, role: str) -> dict[str, Tensor]:
    """Return the state one component's last checkpoint of a point saved."""
    checkpoints = store.directory(point) / CHECKPOINTS_NAME
    last = max(entry.name for entry in checkpoints.iterdir() if entry.is_dir())

    return torch.load(checkpoints / last / "roles" / f"{role}.pt", weights_only=True)


def weight_shapes(state: dict[str, Tensor], prefix: str) -> list[tuple[int, ...]]:
    """Return the shapes of the weight matrices under one submodule, in order."""
    return [
        tuple(value.shape)
        for name, value in state.items()
        if name.startswith(prefix) and name.endswith(".weight")
    ]


def test_every_point_of_the_architecture_sweep_trains(tmp_path: Path) -> None:
    config = tmp_path / "run.json"
    config.write_text(json.dumps(run_document()), encoding="utf-8", newline="\n")
    sweep = tmp_path / "sweep.json"
    sweep.write_text(json.dumps(SWEEP), encoding="utf-8", newline="\n")
    root = tmp_path / "runs"
    points = expanded(config, sweep, tmp_path)
    store = RunStore(root, freeze_id=None, seed=ExperimentSeed(7))
    executor = PointExecutor(
        store,
        codec=RunSpecCodec(),
        operations=TRAINING_OPERATIONS,
        infrastructure_attempts=1,
        failure_policy=FailurePolicySpec(numerical="final", out_of_memory="retry"),
    )
    handles = SerialLauncher(store, execute=executor).submit(points)
    declared = [RunSpec.model_validate(point.config).model for point in points]
    poolers = [
        cast("AttentionSubwordPoolerSpec", model.subword_pooler) for model in declared
    ]
    decoders = [
        cast("RnnSpanDecoderSpec", model.span_decoders[ROLE]) for model in declared
    ]
    contexts = [pooler.context for pooler in poolers if pooler.context is not None]

    assert len(points) == len(contexts) == 8
    assert len(handles) == 8
    assert all(store.is_completed(point) for point in points)
    assert {context.cell for context in contexts} == set(BUILTIN_RECURRENT_CELLS)
    assert {decoder.cell for decoder in decoders} == set(BUILTIN_RECURRENT_CELLS)
    assert {decoder.layers for decoder in decoders} == {1, 2}
    assert {layer.activation for pooler in poolers for layer in pooler.scorer} == set(
        BUILTIN_ACTIVATIONS
    )
    assert {layer.activation for decoder in decoders for layer in decoder.head} == set(
        BUILTIN_ACTIVATIONS
    )
    assert {context.width for context in contexts} == {None, 3, 4, 6}
    assert {layer.width for pooler in poolers for layer in pooler.scorer} == {
        2,
        3,
        5,
        6,
    }
    assert {decoder.state_width for decoder in decoders} == {None, 3, 5, 6}
    assert {layer.width for decoder in decoders for layer in decoder.head} == {
        2,
        3,
        4,
    }

    for point, pooler, context, decoder in zip(
        points, poolers, contexts, decoders, strict=True
    ):
        pooled = trained(store, point, "subword_pooler")
        decoded = trained(store, point, ROLE)
        rows = pooled["_context.weight_ih_l0"].shape[1]
        state = rows if context.width is None else context.width
        read = state * (2 if context.bidirectional else 1)
        scored = [read, *(layer.width for layer in pooler.scorer)]
        cells = decoded["_recur.weight_ih_l0"].shape[1]
        steps = decoder.state_width or cells
        heads = [steps, *(layer.width for layer in decoder.head)]

        assert pooled["_context.weight_ih_l0"].shape[0] == GATES[context.cell] * state
        assert f"_context.weight_ih_l{context.layers - 1}" in pooled
        assert f"_context.weight_ih_l{context.layers}" not in pooled
        assert ("_context.weight_ih_l0_reverse" in pooled) is context.bidirectional
        assert weight_shapes(pooled, "_query.") == [
            *zip(scored[1:], scored[:-1], strict=True),
            (1, scored[-1]),
        ]
        assert decoded["_recur.weight_ih_l0"].shape[0] == GATES[decoder.cell] * steps
        assert f"_recur.weight_ih_l{decoder.layers - 1}" in decoded
        assert f"_recur.weight_ih_l{decoder.layers}" not in decoded
        assert weight_shapes(decoded, "_score.")[:-1] == list(
            zip(heads[1:], heads[:-1], strict=True)
        )
        assert weight_shapes(decoded, "_score.")[-1][1] == heads[-1]

import re
from typing import TYPE_CHECKING, Final, override

import pytest
import torch
from torch import nn

from lofi.train._boundary import (
    CheckpointManifest,
    ComponentRecord,
    PrecisionRecord,
    RngRecord,
    StateReference,
)
from lofi.train._checkpoint import CheckpointId
from lofi.train._grouping import declare
from lofi.train._groups import ActiveGroups, GroupError, ParameterGroup, ParameterGroups
from lofi.train._staging import Stage, StageCompletion, StagedProtocol
from lofi.train._steps import TrainResult
from lofi.train._store import DirectoryStore

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from lofi.train._grouping import GroupResolution
    from lofi.train._store import StateTree

WIDTH: Final = 3

LAYERS: Final = 2

IDS: Final = torch.tensor([[0, 1, 2]])

CHART: Final = "chart"

TYPES: Final = "types"


class Chart(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(4, WIDTH)
        self.layers = nn.ModuleList(nn.Linear(WIDTH, WIDTH) for _ in range(LAYERS))

    @property
    def parameter_groups(self) -> ParameterGroups:
        return ParameterGroups(
            [
                ParameterGroup("embeddings", "embeddings"),
                ParameterGroup("layers", "layers"),
            ]
        )

    @override
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        vectors = self.embeddings(ids)

        for layer in self.layers:
            vectors = torch.tanh(layer(vectors))

        return vectors


class TypeMap(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.score = nn.Linear(WIDTH, 2)

    @property
    def parameter_groups(self) -> ParameterGroups:
        return ParameterGroups([ParameterGroup("weights", "")])

    @override
    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        return self.score(vectors)


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.chart = Chart()
        self.types = TypeMap()

    @override
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.types(self.chart(ids))


def wired(seed: int = 0) -> Model:
    torch.manual_seed(seed)

    return Model()


def declared(model: Model) -> ParameterGroups:
    return declare({CHART: model.chart, TYPES: model.types})


def component(role: str) -> ComponentRecord:
    return ComponentRecord(
        tag=f"{role}-tag",
        spec_fingerprint=f"spec-{role}",
        signature_fingerprint="sig-1",
        codec_identity="codec-1",
        state=StateReference(key=role, digest=f"digest-{role}"),
    )


def plain(step: int) -> CheckpointManifest:
    return CheckpointManifest(
        run_id="run-0001",
        step=step,
        data_position=0,
        code_version="0.1.0",
        action_set="actions-a1",
        components={CHART: component(CHART), TYPES: component(TYPES)},
        rng=RngRecord(streams={"torch": "1c20"}),
        precision=PrecisionRecord(
            compute_dtype="float32",
            autocast_device_type="cpu",
            gradient_scaling=False,
        ),
    )


class Fitting:
    """Fits whatever a stage left active, and records what moved while it did."""

    def __init__(self, model: Model) -> None:
        self.model = model
        self.active: list[frozenset[str]] = []
        self.moved: list[frozenset[str]] = []
        self.gradients: list[frozenset[str]] = []
        self.stepping: list[tuple[int, ...]] = []
        self.entered: list[int] = []

    def fit(self, resolution: GroupResolution, step: int, /) -> TrainResult:
        before = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
        }
        optimizer = torch.optim.SGD(resolution.parameters, lr=0.5)
        self.model.zero_grad(set_to_none=True)
        self.model(IDS).sum().backward()
        optimizer.step()

        self.entered.append(step)
        self.active.append(resolution.trainable)
        self.stepping.append(tuple(id(held) for held in resolution.parameters))
        self.gradients.append(
            frozenset(
                name
                for name, parameter in self.model.named_parameters()
                if parameter.grad is not None
            )
        )
        self.moved.append(
            frozenset(
                name
                for name, parameter in self.model.named_parameters()
                if not torch.equal(before[name], parameter.detach())
            )
        )

        return TrainResult(step + 1, WIDTH)

    def checkpoint(
        self, step: int, /
    ) -> tuple[CheckpointManifest, Mapping[str, StateTree]]:
        return plain(step), {
            CHART: {
                name: parameter.detach().clone()
                for name, parameter in self.model.chart.named_parameters()
            },
            TYPES: {
                name: parameter.detach().clone()
                for name, parameter in self.model.types.named_parameters()
            },
        }


def staged() -> tuple[Stage, ...]:
    return (
        Stage("chart", ActiveGroups(["chart.embeddings", "chart.layers"])),
        Stage("type-map", ActiveGroups(["types.weights"])),
        Stage(
            "joint",
            ActiveGroups(["chart.embeddings", "chart.layers", "types.weights"]),
        ),
    )


def protocol(
    stages: Sequence[Stage], model: Model, root: Path
) -> tuple[StagedProtocol[StateTree], Fitting, DirectoryStore]:
    fitting = Fitting(model)
    store = DirectoryStore(root / "checkpoints")

    return (
        StagedProtocol(stages, declared(model), model, fitting, store),
        fitting,
        store,
    )


def written(
    store: DirectoryStore, completions: Sequence[StageCompletion]
) -> list[CheckpointManifest]:
    return [
        store.load(completion.checkpoint, plain(completion.step)).manifest
        for completion in completions
    ]


def test_a_frozen_stage_leaves_every_parameter_it_froze_where_it_was(
    tmp_path: Path,
) -> None:
    model = wired()
    held = frozenset(name for name, _ in model.named_parameters())
    staging, fitting, _ = protocol(staged(), model, tmp_path)

    completions = staging.run()

    assert fitting.moved == fitting.active
    assert fitting.gradients == fitting.active
    assert fitting.stepping == [
        tuple(
            id(parameter)
            for name, parameter in model.named_parameters()
            if name in active
        )
        for active in fitting.active
    ]
    assert [completion.trained for completion in completions] == fitting.active
    assert [completion.frozen for completion in completions] == [
        held - active for active in fitting.active
    ]


def test_each_stage_trains_the_groups_it_names_and_freezes_the_others(
    tmp_path: Path,
) -> None:
    model = wired()
    held = frozenset(name for name, _ in model.named_parameters())
    staging, _, _ = protocol(staged(), model, tmp_path)

    completions = staging.run()

    assert [completion.stage for completion in completions] == [
        "chart",
        "type-map",
        "joint",
    ]
    assert completions[0].trained == frozenset(
        name for name in held if name.startswith("chart.")
    )
    assert completions[1].trained == frozenset(
        name for name in held if name.startswith("types.")
    )
    assert completions[2].trained == held
    assert completions[2].frozen == frozenset()

    for completion in completions:
        assert completion.trained | completion.frozen == held
        assert not completion.trained & completion.frozen


def test_the_protocol_writes_one_manifest_entry_naming_each_stage(
    tmp_path: Path,
) -> None:
    staging, _, store = protocol(staged(), wired(), tmp_path)

    completions = staging.run()
    manifests = written(store, completions)

    assert len(store.list()) == len(staged())
    assert [held.stage.name for held in manifests if held.stage] == [
        "chart",
        "type-map",
        "joint",
    ]
    assert [held.stage.position for held in manifests if held.stage] == [0, 1, 2]
    assert [held.step for held in manifests] == [1, 2, 3]
    assert [completion.checkpoint for completion in completions] == list(store.list())


def test_a_manifest_entry_names_the_groups_its_stage_trained(tmp_path: Path) -> None:
    staging, _, store = protocol(staged(), wired(), tmp_path)

    manifests = written(store, staging.run())
    first = manifests[0].stage
    second = manifests[1].stage

    assert first is not None
    assert second is not None
    assert first.active == ("chart.embeddings", "chart.layers")
    assert second.active == ("types.weights",)


def test_the_stage_count_is_configuration_and_not_a_module_per_stage(
    tmp_path: Path,
) -> None:
    model = wired()
    two = (staged()[0], staged()[2])
    four = (*staged(), Stage("second-joint", staged()[2].active))

    short, _, _ = protocol(two, model, tmp_path / "short")
    long, _, _ = protocol(four, wired(1), tmp_path / "long")

    assert [completion.stage for completion in short.run()] == ["chart", "joint"]
    assert [completion.stage for completion in long.run()] == [
        "chart",
        "type-map",
        "joint",
        "second-joint",
    ]
    assert len(short) == 2
    assert len(long) == 4
    assert [stage.name for stage in short] == ["chart", "joint"]


def test_a_stage_fits_for_as_many_epochs_as_it_declares(tmp_path: Path) -> None:
    model = wired()
    stages = (
        Stage("chart", ActiveGroups(["chart.embeddings", "chart.layers"]), epochs=3),
        Stage("type-map", ActiveGroups(["types.weights"]), epochs=2),
    )
    staging, fitting, _ = protocol(stages, model, tmp_path)

    completions = staging.run(4)

    assert fitting.entered == [4, 5, 6, 7, 8]
    assert [len(completion.results) for completion in completions] == [3, 2]
    assert [completion.step for completion in completions] == [7, 9]


def test_a_protocol_resumes_from_the_step_count_it_is_given(tmp_path: Path) -> None:
    staging, fitting, _ = protocol(staged(), wired(), tmp_path)

    staging.run(10)

    assert fitting.entered == [10, 11, 12]


def test_the_weights_a_stage_wrote_are_the_weights_it_ended_with(
    tmp_path: Path,
) -> None:
    model = wired()
    staging, _, store = protocol(staged(), model, tmp_path)

    completions = staging.run()
    restored = store.load(completions[-1].checkpoint, plain(3)).states

    for role, block in restored.items():
        held = dict(getattr(model, role).named_parameters())

        assert isinstance(block, dict)

        for name, saved in block.items():
            assert isinstance(saved, torch.Tensor)
            assert torch.equal(saved, held[str(name)].detach())


def test_a_protocol_fits_at_least_one_stage(tmp_path: Path) -> None:
    model = wired()
    store = DirectoryStore(tmp_path / "checkpoints")

    with pytest.raises(ValueError, match="at least one stage"):
        StagedProtocol((), declared(model), model, Fitting(model), store)


def test_two_stages_may_not_share_a_name(tmp_path: Path) -> None:
    model = wired()
    store = DirectoryStore(tmp_path / "checkpoints")
    twice = (staged()[0], Stage("chart", ActiveGroups(["types.weights"])))

    with pytest.raises(ValueError, match=re.escape("two stages are called ['chart']")):
        StagedProtocol(twice, declared(model), model, Fitting(model), store)


def test_a_stage_naming_an_undeclared_group_is_refused_when_it_is_reached(
    tmp_path: Path,
) -> None:
    model = wired()
    stages = (Stage("chart", ActiveGroups(["chart.attention"])),)
    staging, _, _ = protocol(stages, model, tmp_path)

    with pytest.raises(GroupError, match=re.escape("chart.attention")):
        staging.run()


def test_a_stage_fits_for_at_least_one_epoch() -> None:
    with pytest.raises(ValueError, match="epoch count is not positive"):
        Stage("chart", ActiveGroups(["types.weights"]), epochs=0)


def test_a_stage_is_named() -> None:
    with pytest.raises(ValueError, match="a stage name is empty"):
        Stage("", ActiveGroups(["types.weights"]))


def test_a_protocol_starts_at_a_nonnegative_step_count(tmp_path: Path) -> None:
    staging, _, _ = protocol(staged(), wired(), tmp_path)

    with pytest.raises(ValueError, match="step is negative"):
        staging.run(-1)


def test_a_stage_is_a_frozen_value() -> None:
    stage = Stage("chart", ActiveGroups(["types.weights"]), epochs=2)

    assert stage == Stage("chart", ActiveGroups(["types.weights"]), 2)
    assert stage != Stage("chart", ActiveGroups(["types.weights"]))
    assert stage != Stage("types", ActiveGroups(["types.weights"]), 2)
    assert len({stage, Stage("chart", ActiveGroups(["types.weights"]), 2)}) == 1
    assert stage.epochs == 2
    assert repr(stage) == ("Stage('chart', ActiveGroups(['types.weights']), 2)")

    with pytest.raises(AttributeError, match="immutable"):
        stage.__setattr__("_name", "other")


def test_a_completion_records_one_stage_and_refuses_an_impossible_one() -> None:
    completion = StageCompletion(
        "chart",
        CheckpointId("checkpoint-1"),
        frozenset({"a"}),
        frozenset({"b"}),
        [TrainResult(1, 2)],
    )

    assert completion == StageCompletion(
        "chart",
        CheckpointId("checkpoint-1"),
        frozenset({"a"}),
        frozenset({"b"}),
        (TrainResult(1, 2),),
    )
    assert completion != StageCompletion(
        "other",
        CheckpointId("checkpoint-1"),
        frozenset({"a"}),
        frozenset({"b"}),
        [TrainResult(1, 2)],
    )
    assert len({completion, completion}) == 1
    assert completion.step == 1
    assert repr(completion) == (
        "StageCompletion(stage='chart', checkpoint='checkpoint-1', trained=1, "
        "frozen=1, epochs=1)"
    )

    with pytest.raises(AttributeError, match="immutable"):
        completion.__setattr__("_stage", "other")


def test_a_completion_reports_at_least_one_epoch() -> None:
    with pytest.raises(ValueError, match="reports no epoch"):
        StageCompletion(
            "chart", CheckpointId("checkpoint-1"), frozenset(), frozenset(), []
        )


def test_a_parameter_is_never_both_trained_and_frozen() -> None:
    with pytest.raises(ValueError, match=re.escape("both trained and froze ['a']")):
        StageCompletion(
            "chart",
            CheckpointId("checkpoint-1"),
            frozenset({"a"}),
            frozenset({"a"}),
            [TrainResult(1, 2)],
        )

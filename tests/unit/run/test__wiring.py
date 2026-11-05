import subprocess
import sys
from importlib import import_module
from typing import TYPE_CHECKING, Final, Literal, cast

import pytest

from lofi.data import ClassRole, QuantilePopulation
from lofi.data._selection import (
    CategoryClass,
    CategoryRole,
    FrequencyQuantile,
    MajorityRule,
    SelectionPolicy,
)
from lofi.registry import Registry
from lofi.run import NullSink, SinkContext
from lofi.run._wiring import resolve_constraint_policy, resolve_metric_sinks
from lofi.spec import (
    AmbiguityMode,
    ConstraintPolicySpec,
    SinkSpec,
)
from lofi.spec.constraint_policy import (
    CategoryClassSelectorSpec,
    CategoryRoleSelectorSpec,
    QuantileSelectorSpec,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_a_constraint_policy_is_lowered_at_the_run_boundary() -> None:
    declaration = ConstraintPolicySpec(
        ambiguity=AmbiguityMode.MAJORITY,
        selector=QuantileSelectorSpec(share=0.6, over=QuantilePopulation.TOKENS),
        coverage=0.75,
        seed=13,
    )

    assert resolve_constraint_policy(declaration) == SelectionPolicy(
        MajorityRule(),
        selector=FrequencyQuantile(0.6, over=QuantilePopulation.TOKENS),
        coverage=0.75,
        seed=13,
    )


def test_a_category_class_declaration_lowers_to_its_runtime_class() -> None:
    declaration = ConstraintPolicySpec(
        selector=CategoryClassSelectorSpec(categories=frozenset({"N", "NP"}))
    )
    selected = resolve_constraint_policy(declaration).selector

    assert selected == CategoryClass({"N", "NP"})


def test_a_category_role_declaration_lowers_to_its_runtime_role() -> None:
    declaration = ConstraintPolicySpec(
        selector=CategoryRoleSelectorSpec(role=ClassRole.OPEN, share=0.3)
    )
    selected = resolve_constraint_policy(declaration).selector

    assert selected == CategoryRole(ClassRole.OPEN, 0.3)


def _install_sink_plugin(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[str, str]:
    suffix = root.name.replace("-", "_")
    module = f"lofi_sink_plugin_{suffix}"
    distribution = f"lofi-sink-plugin-{suffix}"
    source = """from typing import Literal, NamedTuple

from lofi.registry import ComponentPlugin
from lofi.spec import SinkSpec

EVENTS = []


class FixtureSinkSettings(NamedTuple):
    label: str


class FixtureSinkSpec(SinkSpec):
    kind: Literal["fixture-sink"] = "fixture-sink"
    label: str

    def settings(self):
        return FixtureSinkSettings(label=self.label)


class FixtureSink:
    def __init__(self, label, point_id):
        self.label = label
        self.point_id = point_id

    def log_scalars(self, mapping, step):
        EVENTS.append((self.label, self.point_id, dict(mapping), step))

    def log_table(self, name, rows):
        pass

    def close(self):
        pass


def build(settings, context):
    return FixtureSink(settings.label, context.point_id)


PLUGIN = ComponentPlugin(spec=FixtureSinkSpec, build=build)
"""
    (root / f"{module}.py").write_text(source, encoding="utf-8", newline="\n")
    metadata = root / f"{distribution.replace('-', '_')}-1.0.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {distribution}\nVersion: 1.0.0\n",
        encoding="utf-8",
        newline="\n",
    )
    (metadata / "entry_points.txt").write_text(
        f"[lofi.sink]\nfixture-sink = {module}:PLUGIN\n",
        encoding="utf-8",
        newline="\n",
    )
    monkeypatch.syspath_prepend(str(root))

    return module, distribution


def test_a_configuration_builds_each_declared_sink_through_its_plugin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, distribution = _install_sink_plugin(tmp_path, monkeypatch)
    (entry,) = (
        point
        for point in Registry[object]("sink").entries()
        if point.name == "fixture-sink"
    )
    entry.load()
    specs = {
        role: SinkSpec.model_validate({"kind": "fixture-sink", "label": role})
        for role in ("second", "first")
    }
    sinks = resolve_metric_sinks(specs, SinkContext("0123abcd"))

    for sink in sinks:
        sink.log_scalars({"loss": 0.75}, 3)

    plugin = import_module(module)
    events = cast("list[tuple[str, str, dict[str, float], int]]", plugin.EVENTS)
    assert entry.dist is not None
    assert entry.dist.name == distribution
    assert events == [
        ("first", "0123abcd", {"loss": 0.75}, 3),
        ("second", "0123abcd", {"loss": 0.75}, 3),
    ]


def test_an_empty_sink_configuration_resolves_to_one_null_sink() -> None:
    sinks = resolve_metric_sinks({}, SinkContext("0123abcd"))

    assert len(sinks) == 1
    assert isinstance(sinks[0], NullSink)


_NESTED_RUNTIME_PROGRAM: Final = """
import sys
from typing import Literal, cast

import didactic.api as dx
import torch
from torch import nn

from lofi._types import JsonValue
from lofi.spec import (
    DatasetSpec,
    FeaturizerSpec,
    LossSpec,
    OptimizerSpec,
    ParserSpec,
    RunSpec,
    SinkSpec,
    TrainerSpec,
)


class NestedRuntimeLeaf(dx.Model, extra="forbid"):
    value: JsonValue


class NestedRuntimeSettings(dx.Model, extra="forbid"):
    nested: NestedRuntimeLeaf


class NestedRuntimeParser(ParserSpec):
    kind: Literal["nested-runtime-parser"] = "nested-runtime-parser"
    settings: NestedRuntimeSettings


class NestedRuntimeLoss(LossSpec):
    kind: Literal["nested-runtime-loss"] = "nested-runtime-loss"


class NestedRuntimeDataset(DatasetSpec):
    kind: Literal["nested-runtime-dataset"] = "nested-runtime-dataset"


class NestedRuntimeFeaturizer(FeaturizerSpec):
    kind: Literal["nested-runtime-featurizer"] = "nested-runtime-featurizer"


class NestedRuntimeTrainer(TrainerSpec):
    kind: Literal["nested-runtime-trainer"] = "nested-runtime-trainer"


class NestedRuntimeOptimizer(OptimizerSpec):
    kind: Literal["nested-runtime-optimizer"] = "nested-runtime-optimizer"


class NestedRuntimeSink(SinkSpec):
    kind: Literal["nested-runtime-sink"] = "nested-runtime-sink"


case = sys.argv[1]

if case == "valid":
    value = {"values": [1, "ready", True, None], "options": {"depth": 2}}
elif case == "module":
    value = nn.Linear(2, 1)
elif case == "tensor":
    value = torch.ones(2)
elif case == "class":
    value = nn.Linear
else:
    raise AssertionError(f"unknown nested-runtime case: {case!r}")

document = {
    "model": {
        "language": {"from": "data"},
        "parser": {
            "kind": "nested-runtime-parser",
            "settings": {"nested": {"value": value}},
        }
    },
    "loss": {"terms": [{"objective": {"kind": "nested-runtime-loss"}}]},
    "data": {
        "datasets": {
            "train": {
                "dataset": {"kind": "nested-runtime-dataset"},
                "featurizer": {"kind": "nested-runtime-featurizer"},
            }
        }
    },
    "trainer": {"kind": "nested-runtime-trainer"},
    "release": {"checkpoint": "resume-latest"},
    "parse": {"retain": "forest", "reachability": "all"},
    "optimizer": {"kind": "nested-runtime-optimizer"},
    "device_policy": {
        "device": "cpu",
        "compute_dtype": "float32",
        "deterministic": True,
        "matmul_precision": "highest",
        "activation_memory_bytes": 8_000_000_000,
    },
    "sinks": {"metrics": {"kind": "nested-runtime-sink"}},
}

if case == "valid":
    declared = RunSpec.model_validate(document)
    parser = cast("NestedRuntimeParser", declared.model.parser)
    assert parser.settings.nested.value == {
        "values": (1, "ready", True, None),
        "options": {"depth": 2},
    }
else:
    runtime_type = {"module": "Linear", "tensor": "Tensor", "class": "type"}[case]

    try:
        RunSpec.model_validate(document)
    except dx.ValidationError as error:
        message = str(error)
        assert "model" in message
        assert f"Object of type {runtime_type} is not JSON serializable" in message
    else:
        raise AssertionError(f"RunSpec accepted a nested {runtime_type}")
"""


def _run_nested_runtime_case(
    case: Literal["valid", "module", "tensor", "class"], /
) -> None:
    """Construct and validate one nested-settings run in a fresh interpreter.

    Parameters
    ----------
    case : Literal["valid", "module", "tensor", "class"]
        Nested leaf to construct and validate.
    """
    subprocess.run(
        [sys.executable, "-c", _NESTED_RUNTIME_PROGRAM, case],
        check=True,
        encoding="utf-8",
        capture_output=True,
    )


def test_a_run_spec_accepts_a_nested_json_settings_leaf() -> None:
    _run_nested_runtime_case("valid")


@pytest.mark.parametrize(
    "case",
    [
        pytest.param("module", id="module"),
        pytest.param("tensor", id="tensor"),
        pytest.param("class", id="class"),
    ],
)
def test_a_run_spec_refuses_a_nested_non_json_settings_leaf_before_wiring(
    case: Literal["module", "tensor", "class"],
) -> None:
    _run_nested_runtime_case(case)

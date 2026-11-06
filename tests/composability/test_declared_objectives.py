"""Run documents declare every packaged objective, and each trains one step."""

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import pytest
import torch

from lofi.data.synthetic.training import SyntheticTrainingBatch
from lofi.registry import Registry
from lofi.run import (
    RandomStreams,
    SeedPolicy,
    packaged_options,
    plan_training,
    resolve_device_policy,
)
from lofi.spec import ConfigDocumentError, ConfigError, RunSpecCodec, config_search_path
from lofi.train import Featurized, ParserFittingPlan, ParserStep, prepare_training

if TYPE_CHECKING:
    from lofi._types import JsonValue
    from lofi.nn.ports import SubwordPooler, TermEncoder
    from lofi.train import PreparedTraining

ROOT: Final = Path(__file__).resolve().parents[2]
CONFIG: Final = ROOT / "lofi/configs/run/synthetic-parser.yaml"

DECLARED: Final[dict[str, dict[str, JsonValue]]] = {
    "vector-anchor": {"kind": "vector-anchor", "readout": "types"},
    "simplicity": {"kind": "simplicity", "readout": "types", "scope": "lexicon"},
    "span-reconstruction": {
        "kind": "span-reconstruction",
        "decoder": "reconstruction",
        "weighting": "marginal",
    },
    "outside-reconstruction": {
        "kind": "outside-reconstruction",
        "decoder": "reconstruction",
    },
    "chart-scale": {"kind": "chart-scale", "target": 1.0},
}
"""One declaration of each newly declarable objective."""


TRAINED: Final[dict[str, str]] = {
    "vector-anchor": "parser",
    "simplicity": "identity",
    "span-reconstruction": "reconstruction",
    "outside-reconstruction": "reconstruction",
    "chart-scale": "parser",
}
"""A model role each objective's gradient reaches."""


def _document(directory: Path, terms: list[dict[str, JsonValue]], /) -> Path:
    """Write the packaged parser run with its loss section replaced.

    A run anchoring vectors selects every gold lexical constraint, and a run
    reconstructing spans declares and trains the span decoder it names.

    Parameters
    ----------
    directory : Path
        Where the document is written.
    terms : list[dict[str, JsonValue]]
        The weighted loss terms the document declares.
    """
    resolved = RunSpecCodec(search_path=config_search_path(ROOT))(CONFIG)
    document = cast("dict[str, JsonValue]", json.loads(resolved.model_dump_json()))
    model = cast("dict[str, JsonValue]", document["model"])
    data = cast("dict[str, JsonValue]", document["data"])
    datasets = cast("dict[str, JsonValue]", data["datasets"])
    train = cast("dict[str, JsonValue]", datasets["train"])
    trainer = cast("dict[str, JsonValue]", document["trainer"])
    stopping = cast("dict[str, JsonValue]", trainer["early_stopping"])
    groups = cast("list[JsonValue]", trainer["active_groups"])
    kinds = [cast("dict[str, JsonValue]", term["objective"])["kind"] for term in terms]
    document["loss"] = {"terms": cast("JsonValue", terms)}
    stopping["metric"] = f"loss/{kinds[0]}"

    if "vector-anchor" in kinds:
        train["constraint_source"] = {"kind": "gold-lexicon", "policy": "syntactic"}
        train["constraint_policy"] = {
            "syntactic": {"ambiguity": "union", "coverage": 1.0}
        }

    if {"span-reconstruction", "outside-reconstruction"} & set(kinds):
        model["span_decoders"] = {"reconstruction": {"kind": "rnn-span-decoder"}}
        groups.append("reconstruction")

    if "outside-reconstruction" in kinds:
        parser = cast("dict[str, JsonValue]", model["parser"])
        parser["outside_weights"] = "contextual"

    path = directory / "run.json"
    path.write_text(json.dumps(document), encoding="utf-8", newline="\n")

    return path


def _prepared(path: Path, /) -> PreparedTraining:
    """Compose, plan and prepare one run document under a fixed seed.

    Parameters
    ----------
    path : Path
        The run document.
    """
    run_spec = RunSpecCodec(search_path=config_search_path(ROOT))(path)
    blueprint = plan_training(run_spec)
    streams = RandomStreams(SeedPolicy(23, samplers=blueprint.sampler_streams))
    streams.apply()

    return prepare_training(blueprint, streams, resolve_device_policy(run_spec))


def _plan(prepared: PreparedTraining, /) -> ParserFittingPlan:
    """Return the parser fitting plan a prepared run holds.

    Parameters
    ----------
    prepared : PreparedTraining
        The prepared run.
    """
    plan = prepared.plan
    assert isinstance(plan, ParserFittingPlan)

    return plan


def test_each_objective_is_installed_as_a_packaged_loss() -> None:
    assert set(DECLARED) <= set(Registry[object]("loss").available())
    assert set(DECLARED) <= set(packaged_options("loss"))


@pytest.mark.parametrize("tag", sorted(DECLARED))
def test_a_declared_objective_builds_and_trains_one_synthetic_step(
    tag: str, tmp_path: Path
) -> None:
    prepared = _prepared(
        _document(tmp_path, [{"objective": DECLARED[tag], "weight": 0.5}])
    )
    plan = _plan(prepared)

    assert plan.step.assembly.names == (tag,)
    assert plan.step.assembly.weights == (0.5,)

    batch = plan.batches[0]

    assert isinstance(plan.step, ParserStep)
    assert isinstance(batch, SyntheticTrainingBatch)

    result = plan.step.assembly.evaluate(plan.step.forward.publish(batch), step=0)
    (record,) = result.terms

    assert record.term.count > 0
    assert math.isfinite(result.total.item())

    result.total.backward()
    gradients = tuple(
        parameter.grad
        for parameter in prepared.model.modules[TRAINED[tag]].parameters()
        if parameter.grad is not None
    )

    assert any(bool(torch.any(gradient != 0.0)) for gradient in gradients)

    outcome = plan.step.run(Featurized(batch, 0))

    assert math.isfinite(outcome.result.metrics[f"loss/{tag}"])


@pytest.mark.parametrize("tag", sorted(DECLARED))
def test_an_unknown_setting_is_refused_when_the_document_is_read(
    tag: str, tmp_path: Path
) -> None:
    declared = {**DECLARED[tag], "unknown_setting": 0.1}
    path = _document(tmp_path, [{"objective": declared, "weight": 1.0}])

    with pytest.raises(ConfigError, match="unknown_setting"):
        RunSpecCodec(search_path=config_search_path(ROOT))(path)


@pytest.mark.parametrize(
    ("tag", "first", "second"),
    [
        (
            "vector-anchor",
            {"kind": "vector-anchor", "readout": "types"},
            {
                "kind": "vector-anchor",
                "readout": "types",
                "distance": "squared-euclidean",
                "aggregate": "mean",
                "margin": 2.0,
            },
        ),
        (
            "chart-scale",
            {"kind": "chart-scale", "target": 1.0},
            {
                "kind": "chart-scale",
                "target": 1.0,
                "penalty": "absolute-distance",
            },
        ),
        (
            "span-reconstruction",
            {
                "kind": "span-reconstruction",
                "decoder": "reconstruction",
                "weighting": "marginal",
            },
            {
                "kind": "span-reconstruction",
                "decoder": "reconstruction",
                "weighting": "marginal",
                "max_span_length": 2,
                "normalization": "token",
            },
        ),
    ],
)
def test_auxiliary_objective_shapes_train_and_change_the_tiny_point(
    tag: str,
    first: dict[str, JsonValue],
    second: dict[str, JsonValue],
    tmp_path: Path,
) -> None:
    """Check that declared objective shapes reach a training step and its loss."""
    losses: list[float] = []

    for index, declared in enumerate((first, second)):
        directory = tmp_path / str(index)
        directory.mkdir()
        prepared = _prepared(
            _document(directory, [{"objective": declared, "weight": 1.0}])
        )
        plan = _plan(prepared)
        batch = plan.batches[0]

        assert isinstance(plan.step, ParserStep)
        assert isinstance(batch, SyntheticTrainingBatch)

        result = plan.step.assembly.evaluate(plan.step.forward.publish(batch), step=0)
        losses.append(result.total.item())
        outcome = plan.step.run(Featurized(batch, 0))

        assert math.isfinite(outcome.result.metrics[f"loss/{tag}"])

    assert losses[0] != pytest.approx(losses[1])


def test_span_reconstruction_trains_when_the_chart_is_wider_than_the_pooler(
    tmp_path: Path,
) -> None:
    path = _document(
        tmp_path, [{"objective": DECLARED["span-reconstruction"], "weight": 1.0}]
    )
    document = cast("dict[str, JsonValue]", json.loads(path.read_text("utf-8")))
    model = cast("dict[str, dict[str, JsonValue]]", document["model"])
    encoders = cast("dict[str, dict[str, JsonValue]]", model["encoders"])
    encoders["terms"]["width"] = 6
    model["parser"]["width"] = 6
    cast("dict[str, JsonValue]", document["trainer"])["accumulation_steps"] = 1
    path.write_text(json.dumps(document), encoding="utf-8", newline="\n")
    prepared = _prepared(path)
    plan = _plan(prepared)
    pooler = cast("SubwordPooler", prepared.model.modules["subword_pooler"])
    encoder = cast("TermEncoder", prepared.model.modules["terms"])
    decoder = prepared.model.modules["reconstruction"]
    before = tuple(parameter.detach().clone() for parameter in decoder.parameters())
    batch = plan.batches[0]

    assert pooler.output_width != encoder.output_width
    assert isinstance(plan.step, ParserStep)
    assert isinstance(batch, SyntheticTrainingBatch)

    outcome = plan.step.run(Featurized(batch, 0))

    assert math.isfinite(outcome.result.metrics["loss/span-reconstruction"])
    assert any(
        not torch.equal(initial, trained)
        for initial, trained in zip(before, decoder.parameters(), strict=True)
    )


def test_outside_reconstruction_refuses_reused_outside_weights(tmp_path: Path) -> None:
    path = _document(
        tmp_path,
        [{"objective": DECLARED["outside-reconstruction"], "weight": 1.0}],
    )
    document = cast("dict[str, JsonValue]", json.loads(path.read_text("utf-8")))
    model = cast("dict[str, JsonValue]", document["model"])
    parser = cast("dict[str, JsonValue]", model["parser"])
    parser["outside_weights"] = "reuse"
    path.write_text(json.dumps(document), encoding="utf-8", newline="\n")

    with pytest.raises(ConfigDocumentError, match="outside_cells"):
        _prepared(path)

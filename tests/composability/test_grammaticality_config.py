import json
import math
from typing import Final, NamedTuple

import pytest
import torch
from didactic.settings import load_document

from lofi._types import ReadoutId
from lofi.deduce import SpanIndex
from lofi.nn._values import LogDistribution
from lofi.nn.loss._grammaticality import GrammaticalityObjective
from lofi.nn.loss._parser import SPAN_AXES
from lofi.nn.loss._readout import MASS_AXES
from lofi.nn.loss._values import CellChart, ObjectiveOutputs, SpanReadout
from lofi.registry import Registry
from lofi.run import packaged_options, section_root
from lofi.spec import LossSection
from lofi.spec._plugins import load_plugins
from lofi.train import LossPlugin, resolve_loss_section


class LabeledBatch(NamedTuple):
    """A single sentence with its grammaticality label."""

    ids: tuple[str, ...]
    lengths: tuple[int, ...]
    labels: tuple[bool | None, ...]


SPANS: Final = SpanIndex(1)
READOUT: Final = ReadoutId("sentences")


def outputs() -> ObjectiveOutputs[LabeledBatch]:
    """Return one grammatical item with a published root-mass pair."""
    computed = torch.ones((1, SPANS.num_spans), dtype=torch.bool)
    pair = torch.tensor(
        [[[math.log(0.8), math.log(0.2)]]],
        dtype=torch.float64,
        requires_grad=True,
    )

    return ObjectiveOutputs(
        spans=SPANS,
        chart=CellChart(
            LogDistribution(
                torch.zeros((1, SPANS.num_spans), dtype=torch.float64),
                SPAN_AXES,
                computed,
            )
        ),
        batch=LabeledBatch(("one",), (1,), (True,)),
        readouts={
            READOUT: SpanReadout(
                SPANS,
                LogDistribution(
                    pair,
                    MASS_AXES,
                    computed,
                    normalized_over="outcome",
                ),
            )
        },
    )


def test_the_packaged_loss_resolves_through_its_installed_factory() -> None:
    loaded = load_plugins()
    document = load_document(section_root("loss") / "grammaticality.yaml")
    section = LossSection.model_validate_json(json.dumps(document))
    registry: Registry[LossPlugin[LabeledBatch]] = Registry("loss")

    assembly = resolve_loss_section(
        section,
        registry,
        published=frozenset({"batch", "readouts"}),
        readouts={},
        spaces={},
        readout_spaces={},
        relations={},
        span_decoders={},
        proof_reward=None,
    )
    entry = next(iter(assembly))
    built = entry.objective

    assert "grammaticality" in loaded
    assert "grammaticality" in packaged_options("loss")
    assert registry.available() == (
        "chart-scale",
        "coherence",
        "derivation-entropy",
        "form-supervision",
        "form-typing",
        "grammaticality",
        "inference",
        "outside-reconstruction",
        "pinning",
        "shaped-proof",
        "simplicity",
        "span-reconstruction",
        "vector-anchor",
    )
    assert assembly.names == ("grammaticality",)
    assert assembly.weights == (1.0,)
    assert isinstance(built, GrammaticalityObjective)
    assert built.name == "grammaticality"
    assert built.readout == READOUT
    assert built.noise == 0.0
    assert built.requires == frozenset({"batch", "readouts"})

    published = outputs()
    actual = assembly.evaluate(published, step=0)
    expected = GrammaticalityObjective(
        readout=READOUT, noise=0.0, name="grammaticality"
    ).evaluate(published)

    assert actual.terms[0].term.count == expected.count == 1
    assert actual.total.item() == pytest.approx(expected.value.item(), rel=1e-12)
    assert actual.metrics["loss/grammaticality"] == pytest.approx(
        expected.value.item(), rel=1e-12
    )
    assert actual.metrics["loss/grammaticality/range"] == pytest.approx(0.0)

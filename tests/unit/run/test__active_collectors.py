"""Tests for the installed collectors of an active-learning run."""

import json
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest
from tests.unit.run.test__active_inputs import (
    encoding_document,
    fixture_generator,
    item_document,
)

from lofi.judgment import AcquisitionBatch, CollectorContext
from lofi.judgment.synthetic_designs import (
    KNOWN_RANDOM_EFFECTS,
    KNOWN_RANDOM_EFFECTS_TAG,
)
from lofi.run._active_collection import BeadCollector, SimulatedCollector
from lofi.run._active_collectors import (
    BEAD_COLLECTOR_PLUGIN,
    DRAWN_EFFECTS_COLLECTOR_PLUGIN,
    PREDICTOR_TRUTH_COLLECTOR_PLUGIN,
    REGISTERED_DESIGN_COLLECTOR_PLUGIN,
)
from lofi.run._active_inputs import read_encoding, read_pool
from lofi.spec import load_plugins
from lofi.spec.acquisition import (
    BeadCollectorSettings,
    DrawnEffectsCollectorSettings,
    LatentRangeSettings,
    PredictorTruthCollectorSettings,
    RegisteredDesignCollectorSettings,
)

if TYPE_CHECKING:
    from pathlib import Path

    from bead.items.item import Item

PROBE = """\
import json
import sys

import lofi.run._active_collectors
from lofi.spec import load_plugins

load_plugins()
print(json.dumps(sorted(
    name
    for name in ("lofi.judgment", "bead", "quivers", "lairs", "panproto")
    if name in sys.modules
)))
"""
"""Report which optional capability packages discovering the plugins loads."""

RANGE = LatentRangeSettings(-3.0, 3.0)
"""The latent range every drawn design of these tests divides."""


def _items(tmp_path: Path, /) -> tuple[Item, ...]:
    document = tmp_path / "items.json"
    document.write_text(json.dumps(item_document()), encoding="utf-8", newline="\n")
    return read_pool(document, generator_for=fixture_generator).items


def _joined(items: tuple[Item, ...], /) -> tuple[Item, ...]:
    return tuple(
        item.with_(item_metadata={**item.item_metadata, "design_item": f"item-{index}"})
        for index, item in enumerate(items)
    )


def _context(
    tmp_path: Path, items: tuple[Item, ...], /, *, levels: int = 3
) -> CollectorContext:
    path = tmp_path / "encoding.json"
    path.write_text(json.dumps(encoding_document()), encoding="utf-8", newline="\n")
    return CollectorContext(
        tmp_path / "run",
        tmp_path,
        items,
        {
            str(item.id): {"p_gram": float(position)}
            for position, item in enumerate(items)
        },
        levels=levels,
        seed=0,
        encoding=read_encoding(path),
    )


def test_discovering_the_plugins_loads_no_judgment_capability() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", PROBE],
        check=True,
        capture_output=True,
        encoding="utf-8",
    )

    assert json.loads(completed.stdout) == []


def test_the_bead_plugin_builds_a_bead_collector(tmp_path: Path) -> None:
    load_plugins()
    items = _items(tmp_path)
    collector = BEAD_COLLECTOR_PLUGIN.build(
        BeadCollectorSettings("bead"), _context(tmp_path, items)
    )

    assert isinstance(collector, BeadCollector)
    assert (
        collector.collect(
            1, AcquisitionBatch(items[:1], {}, None, performance={}, floored=())
        )
        is None
    )


@pytest.mark.parametrize("participants", [1, 2])
def test_the_simulated_plugins_build_simulated_collectors(
    tmp_path: Path, participants: int
) -> None:
    load_plugins()
    items = _items(tmp_path)
    batch = AcquisitionBatch(items[:1], {}, None, performance={}, floored=())
    context = _context(tmp_path, items)
    collectors = (
        PREDICTOR_TRUTH_COLLECTOR_PLUGIN.build(
            PredictorTruthCollectorSettings(
                "predictor-truth", "p_gram", participants, "raw", RANGE
            ),
            context,
        ),
        DRAWN_EFFECTS_COLLECTOR_PLUGIN.build(
            DrawnEffectsCollectorSettings("drawn-effects", participants, RANGE),
            context,
        ),
    )

    for collector in collectors:
        assert isinstance(collector, SimulatedCollector)
        assert len(collector.collect(1, batch)) == participants


def test_the_registered_design_plugin_builds_a_simulated_collector(
    tmp_path: Path,
) -> None:
    load_plugins()
    items = _items(tmp_path)
    collector = REGISTERED_DESIGN_COLLECTOR_PLUGIN.build(
        RegisteredDesignCollectorSettings(
            "registered-design", KNOWN_RANDOM_EFFECTS_TAG, "item['design_item']"
        ),
        _context(tmp_path, _joined(items), levels=KNOWN_RANDOM_EFFECTS.levels),
    )

    assert isinstance(collector, SimulatedCollector)
    assert (
        len(
            collector.collect(
                1,
                AcquisitionBatch(
                    _joined(items)[:1], {}, None, performance={}, floored=()
                ),
            )
        )
        == KNOWN_RANDOM_EFFECTS.participants
    )

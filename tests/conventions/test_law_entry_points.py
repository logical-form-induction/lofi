"""Installed discovery of the documented conformance-law modules."""

from importlib.metadata import entry_points
from types import ModuleType

EXPECTED = {
    "data": "lofi.data.laws",
    "deduction": "lofi.deduce.laws",
    "evaluation": "lofi.eval.laws",
    "interpretation": "lofi.interpret.laws",
    "judgment": "lofi.judge.laws",
    "lexicon": "lofi.lexicon.laws",
    "neural-loss": "lofi.nn.loss.laws",
    "probe": "lofi.probe.laws",
    "run": "lofi.run.laws",
    "statistics": "lofi.stats.laws",
    "terms": "lofi.terms.laws",
    "training": "lofi.train.laws",
    "typing": "lofi.typecheck.laws",
}


def test_every_documented_law_module_is_discoverable_and_loadable() -> None:
    points = {point.name: point for point in entry_points(group="lofi.laws")}

    assert {name: point.value for name, point in points.items()} == EXPECTED
    assert all(isinstance(point.load(), ModuleType) for point in points.values())

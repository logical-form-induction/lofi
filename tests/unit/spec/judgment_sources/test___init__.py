"""Tests for the judgment-source declaration package surface."""

from lofi.spec import judgment_sources
from lofi.spec.judgment_sources import predicted


def test_the_package_publishes_the_current_predicted_source_declarations() -> None:
    assert judgment_sources.__all__ == predicted.__all__

    for name in predicted.__all__:
        assert vars(judgment_sources)[name] is vars(predicted)[name]


def test_the_public_decoder_returns_immutable_source_settings() -> None:
    declaration = judgment_sources.decode_predicted_judgment_source(
        b'{"kind":"predicted","producer_run":"run-1",'
        b'"producer_point":"point-1","item_set":"items.json",'
        b'"inputs":{"premise":"a","hypothesis":"b"},'
        b'"factors":{"kind":"item[\'kind\']"},'
        b'"question":"entailment","labels":["yes","no"]}'
    )

    assert declaration.settings() == judgment_sources.PredictedJudgmentSourceSettings(
        "run-1",
        "point-1",
        "items.json",
        {"premise": "a", "hypothesis": "b"},
        {"kind": "item['kind']"},
        "entailment",
        ("yes", "no"),
        "lofi-prediction",
    )

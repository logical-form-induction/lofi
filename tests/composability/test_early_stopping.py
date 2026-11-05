"""Composition tests for early-stopping declarations and runtime rules."""

from lofi.spec.trainers.early_stopping import EarlyStoppingSpec
from lofi.train import EarlyStopping, EarlyStoppingState


def test_a_closed_declaration_lowers_to_the_same_runtime_rule() -> None:
    declared = EarlyStoppingSpec(
        metric="loss/parser",
        mode="max",
        patience=2,
        minimum_delta=0.1,
        keep_best=1,
    )
    rule = EarlyStopping(
        declared.metric,
        declared.mode,
        declared.patience,
        declared.minimum_delta,
    )
    first = rule.observe(EarlyStoppingState(), 0.5)
    equal_boundary = rule.observe(first.state, 0.6)
    strict = rule.observe(equal_boundary.state, 0.61)

    assert rule.metric == declared.metric
    assert rule.mode == declared.mode
    assert rule.patience == declared.patience
    assert rule.minimum_delta == declared.minimum_delta
    assert first.improved
    assert not equal_boundary.improved
    assert strict.improved

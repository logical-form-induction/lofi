from itertools import pairwise
from typing import TYPE_CHECKING, Final

import pytest

from lofi.spec import load_plugins
from lofi.spec.components.data import DatasetSpec, FeaturizerSpec
from lofi.spec.components.loss import LossSpec
from lofi.spec.components.model import (
    ControllerSpec,
    ParserSpec,
    TermDecoderSpec,
    TermEncoderSpec,
)
from lofi.spec.components.run import EvaluatorSpec, ProbeSpec, SinkSpec
from lofi.spec.components.trainer import OptimizerSpec, SchedulerSpec, TrainerSpec
from lofi.train._optimization import (
    Optimization,
    OptimizationResult,
    OptInSet,
    Rewrite,
)

if TYPE_CHECKING:
    import didactic.api as dx

_MEMO: Final = Optimization("memo", Rewrite.VALUE)
_POOL: Final = Optimization("buffer-pool", Rewrite.VALUE)
_BATCHING: Final = Optimization("cell-batching", Rewrite.SCHEDULE)

_COMPONENT_FAMILIES: Final = (
    TermEncoderSpec,
    TermDecoderSpec,
    ControllerSpec,
    ParserSpec,
    LossSpec,
    DatasetSpec,
    FeaturizerSpec,
    TrainerSpec,
    OptimizerSpec,
    SchedulerSpec,
    SinkSpec,
    ProbeSpec,
    EvaluatorSpec,
)

_OPTIMIZATION_FIELD_WORDS: Final = frozenset(
    {
        "cache",
        "cached",
        "caching",
        "fastpath",
        "memo",
        "memoization",
        "memoize",
        "memoized",
        "pool",
        "pooled",
        "vectorization",
        "vectorize",
        "vectorized",
    }
)


def _optimization_fields(fields: tuple[str, ...]) -> frozenset[str]:
    found: set[str] = set()

    for field in fields:
        words = field.lower().replace("-", "_").split("_")
        adjacent = set(pairwise(words))

        if _OPTIMIZATION_FIELD_WORDS.intersection(words):
            found.add(field)

        if ("fast", "path") in adjacent:
            found.add(field)

        if adjacent.intersection(
            {
                ("buffer", "reuse"),
                ("buffers", "reuse"),
                ("reuse", "buffer"),
                ("reuse", "buffers"),
            }
        ):
            found.add(field)

    return frozenset(found)


def _registered_specifications() -> tuple[type[dx.Model], ...]:
    load_plugins()

    registered: list[type[dx.Model]] = []

    for family in _COMPONENT_FAMILIES:
        registered.append(family)
        registered.extend(family.__variants__.values())

    return tuple(registered)


def test_only_a_value_rewrite_has_to_be_named() -> None:
    assert Rewrite.VALUE.requires_opt_in
    assert not Rewrite.SCHEDULE.requires_opt_in


def test_an_optimization_carries_its_name_and_what_it_rewrites() -> None:
    assert _MEMO.name == "memo"
    assert _MEMO.rewrite is Rewrite.VALUE
    assert repr(_MEMO) == "Optimization('memo', Rewrite.VALUE)"


@pytest.mark.parametrize("name", ["", " ", "memo ", " memo"])
def test_an_optimization_refuses_a_name_a_reader_cannot_act_on(name: str) -> None:
    with pytest.raises(ValueError, match="non-empty and stripped"):
        Optimization(name, Rewrite.VALUE)


def test_two_optimizations_agree_when_the_name_and_the_rewrite_agree() -> None:
    same = Optimization("memo", Rewrite.VALUE)
    reclassified = Optimization("memo", Rewrite.SCHEDULE)

    assert same == _MEMO
    assert hash(same) == hash(_MEMO)
    assert reclassified != _MEMO
    assert _MEMO != "memo"


def test_an_optimization_cannot_be_rewritten_or_extended() -> None:
    with pytest.raises(AttributeError, match="immutable"):
        _MEMO.__setattr__("_name", "renamed")

    with pytest.raises(AttributeError, match="immutable"):
        _MEMO.__setattr__("extra", "anything")


def test_a_schedule_rewrite_cannot_be_switched() -> None:
    with pytest.raises(ValueError, match="cell-batching") as caught:
        OptInSet([_BATCHING])

    assert "cannot be opted into" in str(caught.value)


def test_registered_component_specs_expose_no_optimization_flags() -> None:
    specifications = _registered_specifications()
    forbidden = {
        specification.__name__: sorted(
            _optimization_fields(tuple(specification.__field_specs__))
        )
        for specification in specifications
        if _optimization_fields(tuple(specification.__field_specs__))
    }

    assert specifications
    assert set(_COMPONENT_FAMILIES).issubset(specifications)
    assert forbidden == {}


def test_the_specification_field_check_rejects_each_optimization_flag_form() -> None:
    fields = (
        "cache",
        "use_memoization",
        "buffer_pool",
        "reuse_buffer",
        "enable_fast_path",
        "vectorized",
    )

    assert _optimization_fields(fields) == frozenset(fields)


def test_the_specification_field_check_keeps_component_choices() -> None:
    fields = ("pooler", "vector_width", "path")

    assert _optimization_fields(fields) == frozenset()


def test_the_empty_regime_asks_for_nothing() -> None:
    reference = OptInSet()

    assert len(reference) == 0
    assert list(reference) == []
    assert _MEMO not in reference
    assert repr(reference) == "OptInSet([])"


def test_a_regime_holds_the_value_rewrites_it_was_given() -> None:
    regime = OptInSet([_POOL, _MEMO, _MEMO])

    assert len(regime) == 2
    assert _MEMO in regime
    assert _POOL in regime
    assert Optimization("hoisting", Rewrite.VALUE) not in regime


def test_a_regime_iterates_in_name_order() -> None:
    regime = OptInSet([_MEMO, _POOL])

    assert [optimization.name for optimization in regime] == ["buffer-pool", "memo"]
    assert repr(regime) == (
        "OptInSet([Optimization('buffer-pool', Rewrite.VALUE), "
        "Optimization('memo', Rewrite.VALUE)])"
    )


def test_two_regimes_agree_when_they_ask_for_the_same_rewrites() -> None:
    one_order = OptInSet([_MEMO, _POOL])
    other_order = OptInSet([_POOL, _MEMO])

    assert one_order == other_order
    assert hash(one_order) == hash(other_order)
    assert one_order != OptInSet([_MEMO])
    assert one_order != frozenset({_MEMO, _POOL})


def test_a_regime_cannot_be_extended_after_it_is_built() -> None:
    regime = OptInSet([_MEMO])

    with pytest.raises(AttributeError, match="immutable"):
        regime.__setattr__("_members", frozenset({_POOL}))

    with pytest.raises(AttributeError, match="immutable"):
        regime.__setattr__("extra", _POOL)


def test_an_optimization_result_reports_the_loss_step_and_scale() -> None:
    result = OptimizationResult(loss_finite=True, stepped=True, scale=16.0)

    assert result.loss_finite
    assert result.stepped
    assert result.scale == 16.0
    assert repr(result) == (
        "OptimizationResult(loss_finite=True, stepped=True, scale=16.0)"
    )


def test_optimization_results_have_value_semantics() -> None:
    result = OptimizationResult(loss_finite=True, stepped=False, scale=None)
    same = OptimizationResult(loss_finite=True, stepped=False, scale=None)
    stepped = OptimizationResult(loss_finite=True, stepped=True, scale=None)

    assert result == same
    assert hash(result) == hash(same)
    assert result != stepped
    assert result != (True, False, None)


def test_an_optimization_result_is_frozen() -> None:
    result = OptimizationResult(loss_finite=False, stepped=False, scale=None)

    with pytest.raises(AttributeError, match="immutable"):
        result.__setattr__("scale", 2.0)

    with pytest.raises(AttributeError, match="immutable"):
        result.__setattr__("_scale", 2.0)

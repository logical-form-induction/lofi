import ast
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

import pytest

from lofi._types import ItemId
from lofi.eval import _baselines
from lofi.eval._baselines import (
    BaselineFamily,
    BaselineHarness,
    BaselineItem,
    BaselineRow,
    BaselineSystem,
    BaselineTable,
    FamilyMismatchError,
    ReleasedFile,
    RerunConfiguration,
)
from lofi.eval._churn import ARMS, ArmMixingError
from lofi.eval._ports import Metric
from lofi.eval._values import ConventionVector, MetricResult, Resampling

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi.eval._churn import Arm

_AGGREGATING = frozenset(
    {"sum", "fsum", "fmean", "mean", "median", "stdev", "variance", "total"}
)


def _conventions(*, minimum_length: int = 0) -> ConventionVector:
    return ConventionVector(
        punctuation="keep",
        unary_chains="keep",
        trivial_spans="keep",
        minimum_length=minimum_length,
        aggregation="corpus",
        coverage="zero",
        self_arcs="forbid",
        gold_derivation="canonical",
    )


class _Counted:
    """A metric that reports the fraction of items whose prediction is the gold."""

    def __init__(self, conventions: ConventionVector | None) -> None:
        self._conventions = conventions
        self._items: dict[ItemId, tuple[str, str]] = {}

    @property
    def name(self) -> str:
        return "counted"

    @property
    def conventions(self) -> ConventionVector | None:
        return self._conventions

    @property
    def order_invariant(self) -> bool:
        return True

    def reset(self) -> None:
        self._items = {}

    def update(self, pred: str, gold: str, *, key: ItemId) -> None:
        self._items[key] = (pred, gold)

    def compute(self) -> MetricResult:
        if not self._items:
            return MetricResult(None, 0, Resampling("item"))

        matched = [1 for pred, gold in self._items.values() if pred == gold]

        return MetricResult(
            len(matched) / len(self._items), len(self._items), Resampling("item")
        )

    def per_item(self) -> Mapping[ItemId, MetricResult]:
        return MappingProxyType(
            {
                key: MetricResult(float(pred == gold), 1, Resampling("item"))
                for key, (pred, gold) in self._items.items()
            }
        )


def _family(name: str = "counted-family") -> BaselineFamily[str, str]:
    return BaselineFamily(name, _Counted(_conventions()))


def _items(*pairs: tuple[str, str, str]) -> tuple[BaselineItem[str, str], ...]:
    return tuple(BaselineItem(ItemId(key), pred, gold) for key, pred, gold in pairs)


def _released(name: str = "released-system") -> BaselineSystem[str, str]:
    return BaselineSystem(
        name,
        family="counted-family",
        reason="the authors published their system output",
        items=_items(("i1", "a", "a"), ("i2", "b", "c")),
        released=ReleasedFile("predictions/released.auto", digest="sha256:released"),
    )


def _rerun(name: str = "rerun-system") -> BaselineSystem[str, str]:
    return BaselineSystem(
        name,
        family="counted-family",
        reason="no output was published, so the system was re-run here",
        items=_items(("i3", "a", "a"), ("i4", "c", "c")),
        rerun=RerunConfiguration("rerun-wsj10", config_hash="sha256:configuration"),
    )


def test_a_released_file_puts_its_system_in_the_rescoring_arm_and_cites_the_file() -> (
    None
):
    evidence = ReleasedFile("predictions/released.auto", digest="sha256:released")

    assert evidence.arm == "re-scoring"
    assert evidence.cites == "predictions/released.auto"
    assert evidence.to_spec() == {
        "arm": "re-scoring",
        "cites": "predictions/released.auto",
        "digest": "sha256:released",
    }
    assert _released().arm == "re-scoring"
    assert _released().cites == "predictions/released.auto"


def test_a_rerun_puts_its_system_in_the_rerunning_arm_and_cites_its_config_hash() -> (
    None
):
    evidence = RerunConfiguration("rerun-wsj10", config_hash="sha256:configuration")

    assert evidence.arm == "re-running"
    assert evidence.cites == "sha256:configuration"
    assert evidence.to_spec() == {
        "arm": "re-running",
        "cites": "sha256:configuration",
        "digest": "rerun-wsj10",
    }
    assert _rerun().arm == "re-running"
    assert _rerun().cites == "sha256:configuration"


def test_the_harness_emits_one_row_per_family_system_and_arm() -> None:
    harness = BaselineHarness()
    rows = harness.run(_family(), (_released(), _rerun()))
    table = harness.table()

    assert len(rows) == 2
    assert [row.system for row in rows] == ["released-system", "rerun-system"]
    assert [row.arm for row in rows] == ["re-scoring", "re-running"]
    assert [row.family for row in rows] == ["counted-family", "counted-family"]
    assert table.rescoring == (rows[0],)
    assert table.rerunning == (rows[1],)
    assert table.families() == ("counted-family",)
    assert table.systems("re-scoring") == ("released-system",)
    assert table.systems("re-running") == ("rerun-system",)


def test_a_row_cites_the_prediction_file_or_the_configuration_hash() -> None:
    harness = BaselineHarness()
    rescored, rerun = harness.run(_family(), (_released(), _rerun()))

    assert rescored.evidence == ReleasedFile(
        "predictions/released.auto", digest="sha256:released"
    )
    assert rerun.evidence == RerunConfiguration(
        "rerun-wsj10", config_hash="sha256:configuration"
    )
    assert rescored.cites == "predictions/released.auto"
    assert rescored.reason == "the authors published their system output"
    assert rerun.cites == "sha256:configuration"
    assert rerun.reason == "no output was published, so the system was re-run here"
    assert rerun.to_spec()["digest"] == "rerun-wsj10"


def test_a_row_carries_the_conventions_its_family_reports_under() -> None:
    family = _family()
    harness = BaselineHarness()
    row = harness.run(family, (_released(),))[0]

    assert row.conventions == family.conventions
    assert row.conventions == family.metric.conventions
    assert row.metric == "counted"


def _in(family: str) -> BaselineSystem[str, str]:
    return BaselineSystem(
        "released-system",
        family=family,
        reason="the authors published their system output",
        items=_items(("i1", "a", "a")),
        released=ReleasedFile("predictions/o.auto", digest="sha256:o"),
    )


def test_each_family_is_scored_under_its_own_convention_vector() -> None:
    harness = BaselineHarness()
    wide = BaselineFamily("wide", _Counted(_conventions()))
    narrow = BaselineFamily("narrow", _Counted(_conventions(minimum_length=3)))
    first = harness.run(wide, (_in("wide"),))[0]
    second = harness.run(narrow, (_in("narrow"),))[0]

    assert first.conventions != second.conventions
    assert first.conventions.minimum_length == 0
    assert second.conventions.minimum_length == 3
    assert harness.ran == ("wide", "narrow")
    assert harness.table().families() == ("wide", "narrow")


def test_the_metric_is_reset_between_systems_so_one_row_holds_one_system() -> None:
    harness = BaselineHarness()
    rescored, rerun = harness.run(_family(), (_released(), _rerun()))

    assert rescored.result.n == 2
    assert rescored.result.value == 0.5
    assert rerun.result.n == 2
    assert rerun.result.value == 1.0


def test_the_table_keeps_the_two_arms_apart() -> None:
    harness = BaselineHarness()
    harness.run(_family(), (_released(), _rerun()))
    table = harness.table()
    spec = table.to_spec()

    assert set(spec) == set(ARMS)
    assert [
        entry["arm"] for entry in cast("list[dict[str, str]]", spec["re-scoring"])
    ] == ["re-scoring"]
    assert [
        entry["arm"] for entry in cast("list[dict[str, str]]", spec["re-running"])
    ] == ["re-running"]
    assert set(table.rescoring).isdisjoint(table.rerunning)
    assert table.arm("re-scoring") == table.rescoring
    assert table.arm("re-running") == table.rerunning


def test_a_table_refuses_a_row_filed_under_the_arm_it_does_not_record() -> None:
    harness = BaselineHarness()
    rescored, rerun = harness.run(_family(), (_released(), _rerun()))

    with pytest.raises(ArmMixingError, match="re-running"):
        BaselineTable([rescored, rerun], [])

    with pytest.raises(ArmMixingError, match="re-scoring"):
        BaselineTable([], [rescored])


def test_a_table_refuses_an_undeclared_arm_and_a_row_that_is_not_one() -> None:
    table = BaselineTable([], [])

    with pytest.raises(ValueError, match="undeclared arm"):
        table.arm(cast("Arm", "pooled"))

    with pytest.raises(ValueError, match="undeclared arm"):
        table.systems(cast("Arm", "pooled"))

    with pytest.raises(TypeError, match="baseline row is not one"):
        BaselineTable([cast("BaselineRow", "row")], [])


def test_the_harness_source_makes_no_aggregating_call() -> None:
    tree = ast.parse(Path(_baselines.__file__).read_text(encoding="utf-8"))
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    called.update(
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    )

    assert _AGGREGATING.isdisjoint(called)
    assert not any(
        isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add)
        for node in ast.walk(tree)
    )
    assert not any(
        isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Add)
        for node in ast.walk(tree)
    )


def test_a_system_offered_to_another_family_is_refused() -> None:
    harness = BaselineHarness()

    with pytest.raises(FamilyMismatchError, match="counted-family"):
        harness.run(BaselineFamily("other", _Counted(_conventions())), (_released(),))

    assert harness.ran == ()
    assert harness.table() == BaselineTable([], [])


def test_a_family_runs_once() -> None:
    harness = BaselineHarness()
    harness.run(_family(), (_released(),))

    with pytest.raises(ValueError, match="already run"):
        harness.run(_family(), (_rerun(),))

    assert harness.ran == ("counted-family",)
    assert len(harness.table().rescoring) == 1


def test_a_family_run_over_no_system_or_a_repeated_one_is_refused() -> None:
    harness = BaselineHarness()

    with pytest.raises(ValueError, match="run over no system"):
        harness.run(_family(), ())

    with pytest.raises(ValueError, match="twice"):
        harness.run(_family(), (_released(), _released()))

    with pytest.raises(TypeError, match="baseline system is not one"):
        harness.run(
            _family(), cast("Sequence[BaselineSystem[str, str]]", ("released",))
        )

    with pytest.raises(TypeError, match="baseline family is not one"):
        BaselineHarness().run(
            cast("BaselineFamily[str, str]", "family"), (_released(),)
        )

    assert harness.ran == ()


def test_a_family_whose_metric_declares_no_conventions_is_refused() -> None:
    with pytest.raises(TypeError, match="no convention vector"):
        BaselineFamily("unconventional", _Counted(None))


def test_a_system_names_exactly_one_kind_of_evidence() -> None:
    with pytest.raises(ValueError, match="arm is undecided"):
        BaselineSystem(
            "both",
            family="counted-family",
            reason="two kinds of evidence",
            items=_items(("i1", "a", "a")),
            released=ReleasedFile("p.auto", digest="sha256:p"),
            rerun=RerunConfiguration("c", config_hash="sha256:c"),
        )

    with pytest.raises(ValueError, match="no arm holds it"):
        BaselineSystem(
            "neither",
            family="counted-family",
            reason="no evidence at all",
            items=_items(("i1", "a", "a")),
        )

    with pytest.raises(TypeError, match="released predictions are not a file"):
        BaselineSystem(
            "wrong",
            family="counted-family",
            reason="a path is not a file record",
            items=_items(("i1", "a", "a")),
            released=cast("ReleasedFile", "p.auto"),
        )

    with pytest.raises(TypeError, match="re-run configuration is not one"):
        BaselineSystem(
            "wrong",
            family="counted-family",
            reason="a hash is not a configuration record",
            items=_items(("i1", "a", "a")),
            rerun=cast("RerunConfiguration", "sha256:c"),
        )


def test_a_system_predicts_each_item_once_and_predicts_at_least_one() -> None:
    with pytest.raises(ValueError, match="twice"):
        BaselineSystem(
            "repeated",
            family="counted-family",
            reason="the released file lists one item twice",
            items=_items(("i1", "a", "a"), ("i1", "b", "b")),
            released=ReleasedFile("p.auto", digest="sha256:p"),
        )

    with pytest.raises(ValueError, match="predicts no item"):
        BaselineSystem(
            "empty",
            family="counted-family",
            reason="the released file is empty",
            items=(),
            released=ReleasedFile("p.auto", digest="sha256:p"),
        )

    with pytest.raises(TypeError, match="baseline item is not one"):
        BaselineSystem(
            "wrong",
            family="counted-family",
            reason="a tuple is not an item",
            items=cast("Sequence[BaselineItem[str, str]]", ("i1",)),
            released=ReleasedFile("p.auto", digest="sha256:p"),
        )


def test_an_empty_name_is_refused_everywhere_a_row_reads_one() -> None:
    with pytest.raises(ValueError, match="released prediction file"):
        ReleasedFile("", digest="sha256:p")

    with pytest.raises(ValueError, match="digest"):
        ReleasedFile("p.auto", digest="")

    with pytest.raises(ValueError, match="re-run configuration"):
        RerunConfiguration("", config_hash="sha256:c")

    with pytest.raises(ValueError, match="hash"):
        RerunConfiguration("c", config_hash="")

    with pytest.raises(ValueError, match="baseline family"):
        BaselineFamily("", _Counted(_conventions()))


def test_a_row_serializes_its_arm_citation_conventions_and_value() -> None:
    harness = BaselineHarness()
    row = harness.run(_family(), (_released(),))[0]
    spec = row.to_spec()

    assert spec["family"] == "counted-family"
    assert spec["system"] == "released-system"
    assert spec["metric"] == "counted"
    assert spec["arm"] == "re-scoring"
    assert spec["cites"] == "predictions/released.auto"
    assert spec["digest"] == "sha256:released"
    assert spec["reason"] == "the authors published their system output"
    assert spec["value"] == 0.5
    assert spec["n"] == 2
    assert spec["unit"] == "item"
    assert spec["seed"] is None
    assert spec["punctuation"] == "keep"
    assert spec["minimum_length"] == 0


def test_evidence_and_tables_compare_by_value() -> None:
    harness = BaselineHarness()
    harness.run(_family(), (_released(), _rerun()))
    table = harness.table()
    rebuilt = BaselineHarness()
    rebuilt.run(_family(), (_released(), _rerun()))

    assert table == rebuilt.table()
    assert hash(table) == hash(rebuilt.table())
    assert table != BaselineTable([], [])
    assert table.__eq__("table") is NotImplemented
    assert ReleasedFile("p.auto", digest="d") == ReleasedFile("p.auto", digest="d")
    assert ReleasedFile("p.auto", digest="d") != ReleasedFile("q.auto", digest="d")
    assert ReleasedFile("p.auto", digest="d").__eq__("p.auto") is NotImplemented
    assert hash(RerunConfiguration("c", config_hash="h")) == hash(
        RerunConfiguration("c", config_hash="h")
    )
    assert RerunConfiguration("c", config_hash="h") != RerunConfiguration(
        "c", config_hash="g"
    )
    assert RerunConfiguration("c", config_hash="h").__eq__("c") is NotImplemented


def test_every_value_reprs_as_the_call_that_rebuilds_it() -> None:
    released = ReleasedFile("p.auto", digest="sha256:p")
    rerun = RerunConfiguration("c", config_hash="sha256:c")

    assert repr(released) == "ReleasedFile('p.auto', digest='sha256:p')"
    assert repr(rerun) == "RerunConfiguration('c', config_hash='sha256:c')"
    assert "BaselineSystem(" in repr(_released())
    assert "released=ReleasedFile(" in repr(_released())
    assert "rerun=RerunConfiguration(" in repr(_rerun())
    assert repr(_family()).startswith("BaselineFamily('counted-family'")
    assert repr(BaselineTable([], [])) == "BaselineTable([], [])"


def test_the_stand_in_metric_is_one() -> None:
    assert isinstance(_Counted(_conventions()), Metric)

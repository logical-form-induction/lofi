from typing import TYPE_CHECKING, cast

import pytest
import torch

import lofi.nn.loss._suite as loss_suite
from lofi.deduce import SpanIndex
from lofi.nn._values import Encoded, LogDistribution
from lofi.nn.loss._errors import MissingOutputError, UndeclaredOutputError
from lofi.nn.loss._ports import LossTerm
from lofi.nn.loss._suite import ObjectiveMeans, ObjectiveSuite, SuiteResult
from lofi.nn.loss._values import OUTSIDE_CHART, CellChart, ObjectiveOutputs

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from torch import Tensor

    from lofi.nn.loss._ports import Objective, PublishedOutputs

type Outputs = PublishedOutputs[tuple[str, ...]]


def bundle(*, cells: Encoded | None = None) -> ObjectiveOutputs[tuple[str, ...]]:
    inside = LogDistribution(torch.full((2, 3), -1.0), ("item", "span"))

    return ObjectiveOutputs(
        spans=SpanIndex(3),
        chart=CellChart(inside),
        batch=("first", "second"),
        cells=cells,
    )


class Fixed:
    def __init__(
        self,
        name: str,
        value: float,
        count: int = 1,
        diagnostics: Mapping[str, Tensor] | None = None,
    ) -> None:
        self.name = name
        self.requires = frozenset({"batch"})
        self._value = value
        self._count = count
        self._diagnostics = dict(diagnostics or {})

    def evaluate(self, outputs: Outputs, /) -> LossTerm:
        items = len(outputs.batch)

        return LossTerm(
            torch.tensor(self._value * items / 2),
            self._count,
            diagnostics=self._diagnostics,
        )


class Truncating:
    def __init__(self, name: str, residual: float) -> None:
        self.name = name
        self.requires = frozenset({"batch"})
        self._residual = residual

    def evaluate(self, outputs: Outputs, /) -> LossTerm:
        return LossTerm(
            torch.tensor(float(len(outputs.batch))),
            1,
            residual=torch.tensor(self._residual),
        )


class Inapplicable:
    def __init__(self, name: str, like: Tensor) -> None:
        self.name = name
        self.requires = frozenset({"batch"})
        self._like = like

    def evaluate(self, outputs: Outputs, /) -> LossTerm:
        _ = outputs.batch

        return LossTerm.inapplicable(self._like)


class Reaching:
    name = "reaching"
    requires = frozenset({"batch"})

    def evaluate(self, outputs: Outputs, /) -> LossTerm:
        cells = outputs.cells

        if cells is None:
            return LossTerm(torch.tensor(0.0), 0)

        return LossTerm(cells.tensor.sum(), 1)


class Wanting:
    def __init__(self, name: str, requires: frozenset[str]) -> None:
        self.name = name
        self.requires = requires

    def evaluate(self, outputs: Outputs, /) -> LossTerm:
        return LossTerm(outputs.chart.inside.tensor.sum(), 1)


class Unbound:
    def __init__(self, name: str) -> None:
        self.name = name
        self.requires = frozenset({"batch"})

    def bind(self, context: float, /) -> Fixed:
        return Fixed(self.name, context)


class Misbinding:
    name = "misbinding"
    requires = frozenset({"batch"})

    def bind(self, context: float, /) -> Objective[Outputs]:
        return cast("Objective[Outputs]", context)


def suite(
    *entries: tuple[float, Objective[Outputs]],
    published: frozenset[str] | None = None,
) -> ObjectiveSuite[tuple[str, ...]]:
    return ObjectiveSuite(list(entries), published=published)


def test_a_suite_keeps_its_objectives_in_the_order_it_was_given() -> None:
    composed = suite((0.5, Fixed("first", 2.0)), (2.0, Fixed("second", 3.0)))

    assert composed.names == ("first", "second")
    assert composed.weights == (0.5, 2.0)
    assert len(composed) == 2
    assert [objective.name for _, objective in composed] == ["first", "second"]


def test_a_suite_reports_every_name_its_objectives_read() -> None:
    composed = suite(
        (1.0, Fixed("first", 2.0)),
        (1.0, Wanting("second", frozenset({"chart"}))),
    )

    assert composed.requires == frozenset({"batch", "chart"})


def test_the_total_is_the_weighted_sum_of_the_terms() -> None:
    composed = suite((0.5, Fixed("first", 2.0)), (2.0, Fixed("second", 3.0)))

    result = composed.evaluate(bundle())

    assert result.total.item() == pytest.approx(0.5 * 2.0 + 2.0 * 3.0)


def test_an_objective_never_sees_the_weight_it_is_scaled_by() -> None:
    composed = suite((7.0, Fixed("first", 2.0)))

    result = composed.evaluate(bundle())

    assert result.terms["first"].value.item() == pytest.approx(2.0)


def test_doubling_one_weight_doubles_that_term_alone() -> None:
    entries = ((1.0, Fixed("first", 2.0)), (1.0, Fixed("second", 3.0)))
    once = ObjectiveSuite(list(entries)).evaluate(bundle())
    twice = ObjectiveSuite(
        [(2.0, entries[0][1]), entries[1]],
    ).evaluate(bundle())

    assert twice.total.item() - once.total.item() == pytest.approx(2.0)
    assert twice.terms["second"].value.item() == once.terms["second"].value.item()


def test_a_term_that_does_not_apply_leaves_the_total_alone() -> None:
    like = torch.ones(2, 2)
    composed = suite((1.0, Fixed("first", 2.0)), (100.0, Inapplicable("second", like)))

    result = composed.evaluate(bundle())

    assert result.total.item() == pytest.approx(2.0)
    assert result.applied == frozenset({"first"})
    assert result.terms["second"].count == 0


def test_a_step_nothing_applied_to_still_reaches_the_graph() -> None:
    like = torch.ones(2, 2, requires_grad=True)
    composed = suite((1.0, Inapplicable("only", like)))

    result = composed.evaluate(bundle())
    result.total.backward()

    assert result.total.item() == 0.0
    assert like.grad is not None
    assert bool((like.grad == 0.0).all())


def test_the_diagnostics_of_every_objective_land_in_one_namespaced_mapping() -> None:
    composed = suite(
        (1.0, Fixed("first", 2.0, diagnostics={"gap": torch.tensor(0.25)})),
        (1.0, Truncating("second", 0.125)),
    )

    result = composed.evaluate(bundle())

    assert set(result.diagnostics) == {
        "first/loss",
        "first/gap",
        "second/loss",
        "second/residual",
    }
    assert result.diagnostics["first/gap"].item() == 0.25
    assert result.diagnostics["second/residual"].item() == 0.125


def test_a_result_holds_snapshots_rather_than_the_callers_mappings() -> None:
    terms = {"first": LossTerm(torch.tensor(1.0), 1)}
    diagnostics = {"first/loss": torch.tensor(1.0)}

    result = SuiteResult(torch.tensor(1.0), terms, diagnostics)
    terms["second"] = LossTerm(torch.tensor(2.0), 1)
    diagnostics["second/loss"] = torch.tensor(2.0)

    assert set(result.terms) == {"first"}
    assert set(result.diagnostics) == {"first/loss"}


def test_a_suite_refuses_two_objectives_of_the_same_name() -> None:
    with pytest.raises(ValueError, match="two objectives are named"):
        suite((1.0, Fixed("same", 1.0)), (1.0, Fixed("same", 2.0)))


def test_a_suite_with_no_objectives_is_refused() -> None:
    with pytest.raises(ValueError, match="trains nothing"):
        suite()


@pytest.mark.parametrize("weight", [-1.0, float("inf"), float("nan")])
def test_a_weight_that_cannot_scale_a_term_is_refused(weight: float) -> None:
    with pytest.raises(ValueError, match="weight"):
        suite((weight, Fixed("first", 1.0)))


def test_an_objective_declaring_a_name_no_step_publishes_is_refused() -> None:
    with pytest.raises(ValueError, match="alpha"):
        suite((1.0, Wanting("first", frozenset({"alpha"}))))


def test_an_objective_wanting_what_the_graph_omits_is_refused_at_build() -> None:
    published = bundle().published

    with pytest.raises(MissingOutputError) as raised:
        suite(
            (1.0, Wanting("reconstruction", frozenset({OUTSIDE_CHART}))),
            published=published,
        )

    assert raised.value.objective == "reconstruction"
    assert raised.value.missing == frozenset({OUTSIDE_CHART})
    assert OUTSIDE_CHART in str(raised.value)


def test_an_objective_the_graph_satisfies_builds() -> None:
    composed = suite(
        (1.0, Wanting("coherence", frozenset({"chart"}))),
        published=bundle().published,
    )

    assert composed.names == ("coherence",)


def test_an_unbound_objective_cannot_reach_a_suite() -> None:
    entries = [(1.0, cast("Objective[Outputs]", Unbound("anchor")))]

    with pytest.raises(TypeError, match="not an objective"):
        ObjectiveSuite(entries)


def test_binding_returns_a_new_objective_and_leaves_the_receiver_alone() -> None:
    unbound = Unbound("anchor")

    composed = ObjectiveSuite.bound([(1.0, unbound)], 4.0)
    result = composed.evaluate(bundle())

    assert composed.names == ("anchor",)
    assert result.terms["anchor"].value.item() == pytest.approx(4.0)
    assert unbound.name == "anchor"
    assert not hasattr(unbound, "evaluate")


def test_binding_passes_an_objective_through_untouched() -> None:
    already = Fixed("first", 2.0)

    composed = ObjectiveSuite.bound([(1.0, already)], 4.0)

    assert [objective for _, objective in composed] == [already]


def test_binding_that_hands_back_something_else_is_refused() -> None:
    with pytest.raises(TypeError, match="not an objective"):
        ObjectiveSuite.bound([(1.0, Misbinding())], 4.0)


def test_without_the_debug_checks_an_objective_reads_the_bundle_itself(
    reload_with_checks: Callable[..., None],
) -> None:
    reload_with_checks(loss_suite, on=False)
    cells = Encoded(torch.ones(2, 3, 4), ("item", "span", "width"))
    composed = loss_suite.ObjectiveSuite([(1.0, Reaching())])

    result = composed.evaluate(bundle(cells=cells))

    assert result.terms["reaching"].value.item() == pytest.approx(24.0)


def test_the_debug_checks_refuse_a_read_of_an_undeclared_field(
    reload_with_checks: Callable[..., None],
) -> None:
    reload_with_checks(loss_suite, on=True)
    cells = Encoded(torch.ones(2, 3, 4), ("item", "span", "width"))
    composed = loss_suite.ObjectiveSuite([(1.0, Reaching())])

    with pytest.raises(UndeclaredOutputError, match="cells"):
        composed.evaluate(bundle(cells=cells))


def test_the_debug_checks_leave_a_declared_read_alone(
    reload_with_checks: Callable[..., None],
) -> None:
    reload_with_checks(loss_suite, on=True)
    composed = loss_suite.ObjectiveSuite([(1.0, Fixed("first", 2.0))])

    result = composed.evaluate(bundle())

    assert result.total.item() == pytest.approx(2.0)


def test_a_running_mean_is_over_the_items_each_objective_scored() -> None:
    means = ObjectiveMeans()
    composed = suite((1.0, Fixed("first", 2.0, count=3)))

    means.update(composed.evaluate(bundle()))
    means.update(
        ObjectiveSuite([(1.0, Fixed("first", 6.0, count=1))]).evaluate(bundle())
    )

    assert means.means()["first"].item() == pytest.approx((2.0 * 3 + 6.0) / 4)
    assert means.counts()["first"] == 4
    assert means.steps == 2


def test_a_term_that_did_not_apply_enters_no_mean() -> None:
    means = ObjectiveMeans()
    like = torch.ones(2)
    composed = suite(
        (1.0, Fixed("first", 2.0)),
        (1.0, Inapplicable("second", like)),
    )

    means.update(composed.evaluate(bundle()))
    means.update(composed.evaluate(bundle()))

    assert means.means()["first"].item() == pytest.approx(2.0)
    assert "second" not in means.means()
    assert "second" not in means.counts()


def test_the_mean_total_is_over_the_steps_that_ran() -> None:
    means = ObjectiveMeans()

    assert means.total is None

    means.update(suite((1.0, Fixed("first", 2.0))).evaluate(bundle()))
    means.update(suite((1.0, Fixed("first", 4.0))).evaluate(bundle()))

    assert means.total is not None
    assert means.total.item() == pytest.approx(3.0)


def test_resetting_forgets_every_step_folded_in() -> None:
    means = ObjectiveMeans()
    means.update(suite((1.0, Fixed("first", 2.0))).evaluate(bundle()))

    means.reset()

    assert means.total is None
    assert means.steps == 0
    assert dict(means.means()) == {}


def test_the_debug_checks_refuse_a_diagnostic_that_lands_on_the_term(
    reload_with_checks: Callable[..., None],
) -> None:
    reload_with_checks(loss_suite, on=True)
    shadowing = Fixed("first", 2.0, diagnostics={"loss": torch.tensor(0.0)})
    composed = loss_suite.ObjectiveSuite([(1.0, shadowing)])

    with pytest.raises(ValueError, match="lands on the term"):
        composed.evaluate(bundle())

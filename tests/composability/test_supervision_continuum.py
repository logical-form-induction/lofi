"""The packaged supervision continuum composes through one parser-training route."""

from pathlib import Path

import pytest

from lofi.data import ConstraintScope
from lofi.data.synthetic.training import SyntheticTrainingBatch
from lofi.nn.loss._constraint import BatchLocalConstraint
from lofi.run import RandomStreams, SeedPolicy, plan_training, resolve_device_policy
from lofi.spec import RankSelectorSpec, RunSpec, RunSpecCodec, config_search_path
from lofi.terms import Term
from lofi.train import ParserFittingPlan, PreparedTraining, prepare_training

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "lofi/configs/run"
ARMS = ("none", "partial", "lexical", "constituent")
type LocalConstraint = BatchLocalConstraint[Term, ConstraintScope]


def _prepared(arm: str, /) -> tuple[RunSpec, PreparedTraining]:
    run_spec = RunSpecCodec(search_path=config_search_path(ROOT))(
        CONFIGS / f"synthetic-pinning-{arm}.yaml"
    )
    blueprint = plan_training(run_spec)
    streams = RandomStreams(SeedPolicy(17, samplers=blueprint.sampler_streams))
    streams.apply()

    return run_spec, prepare_training(
        blueprint, streams, resolve_device_policy(run_spec)
    )


def _plan(prepared: PreparedTraining, /) -> ParserFittingPlan:
    plan = prepared.plan
    assert isinstance(plan, ParserFittingPlan)

    return plan


def _constraints(plan: ParserFittingPlan, /) -> tuple[LocalConstraint, ...]:
    held: list[LocalConstraint] = []

    for batch in (*plan.batches, *plan.fit_batches):
        if not isinstance(batch, SyntheticTrainingBatch):
            pytest.fail("a parser fitting plan contains synthetic parser batches")

        for constraint in batch.constraints:
            if not isinstance(constraint, BatchLocalConstraint):
                pytest.fail("a target-route source localizes target constraints")

            held.append(constraint)

    return tuple(held)


def _constraint_shape(
    constraint: LocalConstraint, /
) -> tuple[
    str, tuple[object, ...], tuple[tuple[int, int], ...], tuple[int, ...], object
]:
    """Return one store-independent localized-constraint shape."""
    return (
        str(constraint.readout),
        tuple(category.key() for category in constraint.targets.categories),
        constraint.spans,
        constraint.item_positions,
        constraint.scope,
    )


def test_four_packaged_arms_resolve_to_one_weighted_pinning_parser_step() -> None:
    prepared = {arm: _prepared(arm) for arm in ARMS}

    for run_spec, training in prepared.values():
        plan = _plan(training)
        terms = tuple(plan.step.assembly)

        assert tuple(term.name for term in terms) == (
            "grammaticality",
            "coherence",
            "pinning",
        )
        assert tuple(term.weight for term in terms) == (1.0, 0.1, 1.0)
        assert type(plan.step) is type(_plan(prepared["none"][1]).step)
        assert run_spec.loss is not None
        assert run_spec.loss.terms[-1].weight == 1.0


def test_each_arm_reports_and_realizes_its_exact_declared_budget() -> None:
    prepared = {arm: _prepared(arm) for arm in ARMS}
    plans = {arm: _plan(training) for arm, (_, training) in prepared.items()}
    reports = {arm: plan.constraint_selection for arm, plan in plans.items()}

    assert all(report is not None for report in reports.values())
    none = reports["none"]
    partial = reports["partial"]
    lexical = reports["lexical"]
    constituent = reports["constituent"]
    assert none is not None
    assert partial is not None
    assert lexical is not None
    assert constituent is not None

    assert (none.types_pinned, none.tokens_covered, none.mass_share) == (0, 0, 0.0)
    assert _constraints(plans["none"]) == ()
    assert (partial.types_pinned, partial.tokens_covered, partial.mass_share) == (
        4,
        6,
        0.5,
    )
    assert len(_constraints(plans["partial"])) == partial.tokens_covered
    assert len(_constraints(plans["lexical"])) == lexical.tokens_covered == 12
    assert lexical.mass_share == 1.0
    assert len(_constraints(plans["constituent"])) == constituent.tokens_covered == 20
    assert constituent.mass_share == 1.0

    partial_spec = (
        prepared["partial"][0].data.datasets["train"].constraint_policy.syntactic
    )
    assert partial_spec.selector == RankSelectorSpec(first=0.0, last=1.0)
    assert partial_spec.coverage == 0.5
    assert partial_spec.seed == 20261009


def test_partial_selection_and_target_catalog_are_stable_and_share_equal_sets() -> None:
    _, first = _prepared("partial")
    _, repeated = _prepared("partial")
    first_plan = _plan(first)
    repeated_plan = _plan(repeated)
    first_constraints = _constraints(first_plan)
    repeated_constraints = _constraints(repeated_plan)

    assert first_plan.constraint_selection == repeated_plan.constraint_selection
    assert tuple(map(_constraint_shape, first_constraints)) == tuple(
        map(_constraint_shape, repeated_constraints)
    )

    by_targets: dict[object, set[str]] = {}

    for constraint in _constraints(_plan(_prepared("constituent")[1])):
        by_targets.setdefault(constraint.targets, set()).add(str(constraint.readout))

    assert all(len(readouts) == 1 for readouts in by_targets.values())
    assert len({next(iter(readouts)) for readouts in by_targets.values()}) == len(
        by_targets
    )

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from gc import get_referents
from pathlib import Path
from types import FunctionType, ModuleType
from typing import TYPE_CHECKING, Protocol, cast

import didactic.api as dx
import pytest
import torch
from tests.pretrained_stand_ins import StandInLoader

from lofi.data.synthetic.spec.corpus import (
    CheckedSyntheticCorpus,
    SyntheticCorpusLowerer,
)
from lofi.data.synthetic.training import SyntheticTrainingBatch
from lofi.run import (
    MeasuredRole,
    RandomStreams,
    SeedPolicy,
    SweepPoint,
    measure_point,
    plan_training,
    resolve_device_policy,
)
from lofi.spec import (
    ComponentSpec,
    ParameterCapacitySpec,
    RunSpec,
    RunSpecCodec,
    config_search_path,
)
from lofi.train import (
    ChartForward,
    ConfiguredModel,
    Featurized,
    OptimizationState,
    ParserFittingPlan,
    ParserStep,
    StepFittingPlan,
    prepare_training,
)
from lofi.train._capacity import CAPACITY_ROLE
from lofi.train._configured import _ConfiguredTopology
from lofi.train._pretraining import distance_batch_seed

if TYPE_CHECKING:
    from lofi._types import JsonValue
    from lofi.data.synthetic import SyntheticDataset
    from lofi.nn.loss._coherence import CoherenceObjective
    from lofi.nn.parsers import DeepInsideOutside
    from lofi.nn.policy import DevicePolicy
    from lofi.nn.ports import Controller
    from lofi.spec.trainers import DistanceTrainerSpec, ParserTrainerSpec
    from lofi.train import PreparedTraining
    from lofi.train._store import StateTree

CONFIG = Path(__file__).resolve().parents[3] / "lofi/configs/run/synthetic-parser.yaml"
PACKAGED_CODEC = RunSpecCodec(search_path=(CONFIG.parent, CONFIG.parent.parent))


class _Default:
    __slots__ = ()


class _Inspectable(Protocol):
    pass


_DEFAULT = _Default()


def _prepared(
    active: tuple[str, ...] | _Default | None = _DEFAULT,
) -> tuple[PreparedTraining, DevicePolicy]:
    run_spec = PACKAGED_CODEC(CONFIG)

    if active is not _DEFAULT:
        run_spec = run_spec.with_(
            trainer=run_spec.trainer.with_(
                active_groups=cast("tuple[str, ...] | None", active)
            )
        )

    blueprint = plan_training(run_spec)
    streams = RandomStreams(SeedPolicy(7, samplers=blueprint.sampler_streams))
    streams.apply()

    policy = resolve_device_policy(run_spec)

    return (
        prepare_training(blueprint, streams, policy),
        policy,
    )


def _train(prepared: PreparedTraining) -> None:
    """Run the configured parser step across its declared epoch horizon."""
    plan = prepared.plan
    assert isinstance(plan, ParserFittingPlan)
    step = plan.step
    assert isinstance(step, ParserStep)
    completed = 0

    for _ in range(plan.epochs):
        for batch in plan.batches:
            assert isinstance(batch, SyntheticTrainingBatch)
            completed = step.run(Featurized(batch, completed)).result.step


def assert_state_equal(
    expected: StateTree,
    found: StateTree,
    /,
) -> None:
    """Assert recursively over checkpoint state including tensors."""
    if isinstance(expected, torch.Tensor):
        assert isinstance(found, torch.Tensor)
        assert torch.equal(expected, found)

        return

    if isinstance(expected, Mapping):
        assert isinstance(found, Mapping)
        assert tuple(expected) == tuple(found)

        for expected_value, found_value in zip(
            expected.values(), found.values(), strict=True
        ):
            assert_state_equal(expected_value, found_value)

        return

    if isinstance(expected, (list, tuple)):
        assert isinstance(found, (list, tuple))
        assert type(found) is type(expected)
        assert len(found) == len(expected)

        for left, right in zip(expected, found, strict=True):
            assert_state_equal(left, right)

        return

    assert type(expected) is type(found)
    assert expected == found


def assert_component_state_equal(
    expected: Mapping[str, dict[str, StateTree]],
    found: Mapping[str, dict[str, StateTree]],
    /,
) -> None:
    """Assert exact role and state-key topology before comparing values."""
    assert tuple(expected) == tuple(found)

    for role in expected:
        expected_role = expected[role]
        found_role = found[role]
        assert tuple(expected_role) == tuple(found_role)

        for expected_value, found_value in zip(
            expected_role.values(), found_role.values(), strict=True
        ):
            assert_state_equal(expected_value, found_value)


def _reachable(root: _Inspectable, /) -> tuple[_Inspectable, ...]:
    """Return reachable runtime values without entering executable/module graphs."""
    pending = [root]
    seen: set[int] = set()
    found: list[_Inspectable] = []

    while pending:
        value = pending.pop()

        if id(value) in seen:
            continue

        seen.add(id(value))
        found.append(value)

        if not isinstance(
            value,
            (type, FunctionType, ModuleType, torch.Tensor, torch.nn.Module),
        ):
            pending.extend(cast("list[_Inspectable]", get_referents(value)))

    return tuple(found)


def test_preparation_has_one_policy_inventory_and_parameter_ownership() -> None:
    prepared, policy = _prepared()
    plan = prepared.plan
    assert isinstance(plan, ParserFittingPlan)
    controller = cast("Controller", prepared.model.modules["controller"])
    actions = controller.action_set
    forward = plan.forward
    batch = plan.batches[0]
    assert isinstance(forward, ChartForward)
    assert isinstance(batch, SyntheticTrainingBatch)
    published = forward.publish(batch)

    assert prepared.model.policy is policy
    assert prepared.prediction is None
    assert tuple(published.actions) == actions.actions
    assert all(
        found is declared
        for found, declared in zip(published.actions, actions.actions, strict=True)
    )
    assert actions is controller.action_set
    parser = cast("DeepInsideOutside", prepared.model.modules["parser"])
    assert actions is parser.rules.action_set
    coherence = cast(
        "CoherenceObjective[SyntheticTrainingBatch[object]]",
        next(
            entry.objective for entry in plan.step.assembly if entry.name == "coherence"
        ),
    )
    family = coherence.space.combinators
    assert family is not None
    assert family.actions is actions
    assert len({id(parameter) for parameter in prepared.model.parameters}) == len(
        prepared.model.parameters
    )
    assert all(parameter.requires_grad for parameter in prepared.model.parameters)
    assert all(
        not parameter.requires_grad
        for parameter in prepared.model.modules["terms"].parameters()
    )
    assert plan.steps_per_epoch == 1


def test_the_parser_plan_holds_the_corpus_its_epochs_are_drawn_from() -> None:
    prepared, _ = _prepared()
    plan = prepared.plan
    assert isinstance(plan, ParserFittingPlan)
    source = plan.corpus
    assert source is not None
    corpus = {example.item.item_id for example in source.observed()}
    drawn = {
        item_id
        for batch in (*plan.batches, *plan.fit_batches)
        for item_id in batch.item_ids
    }

    language = plan.grammar.language
    assert language.codec.at(language.category_sort) is prepared.model.codec.codec
    assert drawn == corpus


def test_a_per_epoch_sampler_redraws_order_but_keeps_the_split_fixed() -> None:
    run_spec = PACKAGED_CODEC(CONFIG)
    trainer = cast("ParserTrainerSpec", run_spec.trainer)
    declared = run_spec.with_(
        trainer=trainer.with_(
            epochs=4,
            batch_sampler=trainer.batch_sampler.with_(reshuffle_each_epoch=True),
        )
    )
    blueprint = plan_training(declared)
    streams = RandomStreams(SeedPolicy(7, samplers=blueprint.sampler_streams))
    streams.apply()
    prepared = prepare_training(blueprint, streams, resolve_device_policy(declared))
    plan = prepared.plan
    assert isinstance(plan, ParserFittingPlan)
    source = plan.corpus
    assert source is not None
    corpus = {example.item.item_id for example in source.observed()}
    epochs = tuple(plan.epoch(position) for position in range(plan.epochs))
    training_orders = tuple(
        tuple(item_id for batch in epoch.training for item_id in batch.item_ids)
        for epoch in epochs
    )
    validation_memberships = tuple(
        frozenset(
            item_id
            for batch in epoch.retained[plan.fit_split]
            for item_id in batch.item_ids
        )
        for epoch in epochs
    )

    assert len(set(training_orders)) > 1
    assert len(set(validation_memberships)) == 1
    assert all(
        set(training) | set(validation) == corpus
        for training, validation in zip(
            training_orders, validation_memberships, strict=True
        )
    )


@pytest.mark.parametrize("active", [("parser",), ("controller",)])
def test_nested_parser_and_controller_groups_have_most_specific_ownership(
    active: tuple[str, ...],
) -> None:
    prepared, _ = _prepared(active)
    named = prepared.model.named_parameters
    free = prepared.model.groups.trainable
    frozen = prepared.model.groups.frozen
    before = {name: parameter.detach().clone() for name, parameter in named.items()}

    assert prepared.model.groups.members("parser").isdisjoint(
        prepared.model.groups.members("controller")
    )
    assert free == prepared.model.groups.members(active[0])
    assert len(prepared.model.parameters) == len(free)
    assert {id(parameter) for parameter in prepared.model.parameters} == {
        id(named[name]) for name in free
    }

    _train(prepared)

    engaged = frozenset(name for name in free if named[name].grad is not None)
    idle = free - engaged

    assert engaged
    assert all(not torch.equal(before[name], named[name]) for name in engaged)
    assert all(torch.equal(before[name], named[name]) for name in idle)
    assert all(named[name].grad is None for name in frozen)
    assert all(torch.equal(before[name], named[name]) for name in frozen)


@pytest.mark.parametrize(
    "active",
    [
        None,
        (
            "application",
            "controller",
            "identity",
            "parser",
            "terms",
            "token_encoder",
        ),
    ],
)
def test_all_active_groups_cover_every_parameter_exactly_once(
    active: tuple[str, ...] | None,
) -> None:
    prepared, _ = _prepared(active)
    named = prepared.model.named_parameters

    assert prepared.model.groups.trainable == frozenset(named)
    assert not prepared.model.groups.frozen
    assert len(prepared.model.parameters) == len(named)
    assert len({id(parameter) for parameter in prepared.model.parameters}) == len(named)
    assert {id(parameter) for parameter in prepared.model.parameters} == {
        id(parameter) for parameter in named.values()
    }


def test_prepared_training_retains_no_declaration_or_corpus_lowerer() -> None:
    prepared, _ = _prepared()
    retained = _reachable(prepared)

    assert not any(
        isinstance(value, (ComponentSpec, RunSpec, CheckedSyntheticCorpus))
        for value in retained
    )
    assert not any(isinstance(value, SyntheticCorpusLowerer) for value in retained)


def test_complete_state_round_trips_into_a_fresh_seeded_assembly() -> None:
    trained, _ = _prepared()
    _train(trained)
    state = trained.state()
    fresh, _ = _prepared()

    fresh.restore(*state)

    restored = fresh.state()
    assert set(restored[0]) == set(trained.model.modules)
    assert not any(
        name.startswith("_rules._controller.") for name in state[0]["parser"]
    )
    assert state[0]["controller"]
    assert_component_state_equal(state[0], restored[0])
    assert_state_equal(state[1].optimizer, restored[1].optimizer)
    assert_state_equal(state[1].scaler, restored[1].scaler)
    assert_state_equal(state[1].scheduler, restored[1].scheduler)


def test_restore_refuses_aliased_or_optimization_mismatch_before_mutation() -> None:
    trained, _ = _prepared()
    _train(trained)
    components, optimization = trained.state()
    fresh, _ = _prepared()
    untouched = fresh.state()
    aliased = {role: dict(state) for role, state in components.items()}
    controller_tensor = next(
        value
        for value in aliased["controller"].values()
        if isinstance(value, torch.Tensor)
    )
    aliased["parser"]["_rules._controller.planted"] = controller_tensor.clone()

    with pytest.raises(ValueError, match=r"extra keys"):
        fresh.restore(aliased, optimization)

    after_alias = fresh.state()
    assert_component_state_equal(untouched[0], after_alias[0])
    assert_state_equal(untouched[1].optimizer, after_alias[1].optimizer)
    assert_state_equal(untouched[1].scheduler, after_alias[1].scheduler)

    bad_optimizer = deepcopy(optimization.optimizer)
    bad_optimizer["unexpected"] = 1
    incompatible = OptimizationState(
        bad_optimizer,
        scaler=deepcopy(optimization.scaler),
        scheduler=deepcopy(optimization.scheduler),
    )

    with pytest.raises(ValueError, match=r"optimizer state has missing keys"):
        fresh.restore(components, incompatible)

    after_optimizer = fresh.state()
    assert_component_state_equal(untouched[0], after_optimizer[0])
    assert_state_equal(untouched[1].optimizer, after_optimizer[1].optimizer)
    assert_state_equal(untouched[1].scheduler, after_optimizer[1].scheduler)


def test_an_incomplete_accumulation_window_is_refused_before_preparation() -> None:
    run_spec = PACKAGED_CODEC(CONFIG)

    with pytest.raises(dx.ValidationError, match=r"2 microbatches"):
        run_spec.with_(trainer=run_spec.trainer.with_(accumulation_steps=3))


@pytest.mark.parametrize("active", [None, ("frozen", "tuned")])
def test_a_component_built_frozen_stays_frozen_whatever_the_active_groups(
    active: tuple[str, ...] | None,
) -> None:
    frozen = torch.nn.Linear(2, 2).requires_grad_(requires_grad=False)
    tuned = torch.nn.Linear(2, 2)
    topology = _ConfiguredTopology({"frozen": frozen, "tuned": tuned})

    groups = topology.resolve(active)

    assert {id(parameter) for parameter in groups.parameters} == {
        id(parameter) for parameter in tuned.parameters()
    }
    assert len(groups.frozen) == len(list(frozen.parameters()))
    assert not any(parameter.requires_grad for parameter in frozen.parameters())
    assert all(parameter.requires_grad for parameter in tuned.parameters())


def test_a_run_naming_only_frozen_components_trains_nothing_and_is_refused() -> None:
    frozen = torch.nn.Linear(2, 2).requires_grad_(requires_grad=False)
    topology = _ConfiguredTopology({"frozen": frozen})

    with pytest.raises(ValueError, match="no free parameter"):
        topology.resolve(("frozen",))


WORDS = tuple(f"word-{index}" for index in range(8))


def _pretrained_prepared(
    directory: Path, *overrides: str, trainable: bool
) -> tuple[PreparedTraining, RunSpec]:
    """Prepare the packaged run over a stand-in pretrained token encoder."""
    document = directory / "pretrained.yaml"
    document.write_text(
        "defaults:\n  - synthetic-parser\n  - model.token_encoder: pretrained\n"
        f"artifacts:\n  root: {directory / 'artifacts'}\n",
        encoding="utf-8",
        newline="\n",
    )
    run_spec = RunSpecCodec(
        search_path=(CONFIG.parent, *config_search_path(directory))
    )(
        document,
        overrides=(
            f"model.token_encoder.trainable={str(trainable).lower()}",
            "model.parser.width=6",
            "model/encoders/terms=rnn",
            "model.encoders.terms.width=6",
            *overrides,
        ),
    )
    blueprint = plan_training(run_spec)
    streams = RandomStreams(SeedPolicy(7, samplers=blueprint.sampler_streams))
    streams.apply()
    prepared = prepare_training(
        blueprint,
        streams,
        resolve_device_policy(run_spec),
        pretrained_resources={"token_encoder": StandInLoader(WORDS, width=6, seed=3)},
    )

    return prepared, run_spec


def test_a_frozen_pretrained_role_is_pinned_and_left_out_of_checkpoint_state(
    tmp_path: Path,
) -> None:
    prepared, run_spec = _pretrained_prepared(tmp_path, trainable=False)
    encoder = prepared.model.modules["token_encoder"]
    names = {name for name, _ in encoder.named_parameters()}
    loaded = {
        name: value.detach().clone() for name, value in encoder.state_dict().items()
    }
    declared = cast("ComponentSpec", run_spec.model.token_encoder)

    trainer = cast("ParserTrainerSpec", run_spec.trainer)

    assert trainer.active_groups is not None
    assert "token_encoder" in trainer.active_groups
    assert names
    assert prepared.model.pinned == {"token_encoder": declared.pretrained_weights()}
    assert not any(parameter.requires_grad for parameter in encoder.parameters())
    assert names.isdisjoint(prepared.model.groups.trainable)
    assert names <= set(prepared.model.state()["token_encoder"])
    assert names.isdisjoint(prepared.model.checkpoint_state()["token_encoder"])
    assert names.isdisjoint(prepared.state()[0]["token_encoder"])
    assert set(prepared.model.checkpoint_state()["application"]) == set(
        prepared.model.state()["application"]
    )

    _train(prepared)

    assert_state_equal(
        cast("StateTree", loaded), cast("StateTree", dict(encoder.state_dict()))
    )


def test_a_pinned_role_restores_from_complete_or_checkpoint_state(
    tmp_path: Path,
) -> None:
    trained, _ = _pretrained_prepared(tmp_path, trainable=False)
    _train(trained)
    complete = trained.model.state()
    checkpointed = trained.model.checkpoint_state()

    for states in (complete, checkpointed):
        fresh, _ = _pretrained_prepared(tmp_path, trainable=False)
        fresh.model.restore(states)

        assert_component_state_equal(complete, fresh.model.state())

    partial = {role: dict(state) for role, state in checkpointed.items()}
    partial["token_encoder"] = dict(complete["token_encoder"])
    dropped = next(
        name for name, _ in trained.model.modules["token_encoder"].named_parameters()
    )
    del partial["token_encoder"][dropped]
    fresh, _ = _pretrained_prepared(tmp_path, trainable=False)

    with pytest.raises(ValueError, match=r"missing keys"):
        fresh.model.restore(partial)


def test_a_trainable_pretrained_role_is_not_pinned_and_is_checkpointed_whole(
    tmp_path: Path,
) -> None:
    prepared, _ = _pretrained_prepared(tmp_path, trainable=True)
    encoder = prepared.model.modules["token_encoder"]

    assert prepared.model.pinned == {}
    assert all(parameter.requires_grad for parameter in encoder.parameters())
    assert set(prepared.model.checkpoint_state()["token_encoder"]) == set(
        prepared.model.state()["token_encoder"]
    )


def test_a_trainable_pretrained_role_left_out_of_the_active_groups_is_pinned(
    tmp_path: Path,
) -> None:
    prepared, run_spec = _pretrained_prepared(
        tmp_path,
        'trainer.active_groups=["application","controller","parser"]',
        trainable=True,
    )
    encoder = prepared.model.modules["token_encoder"]
    names = {name for name, _ in encoder.named_parameters()}
    loaded = {
        name: value.detach().clone() for name, value in encoder.state_dict().items()
    }
    declared = cast("ComponentSpec", run_spec.model.token_encoder)

    assert names
    assert prepared.model.pinned == {"token_encoder": declared.pretrained_weights()}
    assert not any(parameter.requires_grad for parameter in encoder.parameters())
    assert names.isdisjoint(prepared.model.checkpoint_state()["token_encoder"])

    _train(prepared)

    assert_state_equal(
        cast("StateTree", loaded), cast("StateTree", dict(encoder.state_dict()))
    )


def test_a_module_built_frozen_stays_frozen_whatever_groups_are_active() -> None:
    frozen = torch.nn.Linear(2, 2)
    frozen.requires_grad_(requires_grad=False)
    trained = torch.nn.Linear(2, 2)
    topology = _ConfiguredTopology({"frozen": frozen, "trained": trained})

    for active in (None, ("frozen", "trained")):
        resolution = topology.resolve(active)

        assert topology.frozen == frozenset({"frozen"})
        assert resolution.members("frozen") <= resolution.frozen
        assert not any(parameter.requires_grad for parameter in frozen.parameters())
        assert all(parameter.requires_grad for parameter in trained.parameters())

    with pytest.raises(ValueError, match="no free parameter"):
        topology.resolve(("frozen",))


def test_a_frozen_nested_role_leaves_the_role_holding_it_trainable() -> None:
    nested = torch.nn.Linear(2, 2)
    nested.requires_grad_(requires_grad=False)
    outer = torch.nn.Sequential(nested, torch.nn.Linear(2, 2))
    topology = _ConfiguredTopology({"nested": nested, "outer": outer})
    resolution = topology.resolve(None)

    assert topology.frozen == frozenset({"nested"})
    assert resolution.members("nested") <= resolution.frozen
    assert resolution.members("outer") <= resolution.trainable
    assert resolution.members("outer")


def test_each_role_owns_exactly_its_most_specific_parameters() -> None:
    prepared, _ = _prepared(("parser",))
    owned = prepared.model.role_parameters()
    named = prepared.model.named_parameters
    groups = prepared.model.groups

    assert set(owned) == set(prepared.model.modules)

    for role, parameters in owned.items():
        if not any(True for _ in prepared.model.modules[role].parameters()):
            assert parameters == ()
            continue

        assert {id(parameter) for parameter in parameters} == {
            id(named[name]) for name in groups.members(role)
        }

    assert sum(len(parameters) for parameters in owned.values()) == len(named)
    assert owned["parser"]
    assert owned["controller"]
    assert all(parameter.requires_grad for parameter in owned["parser"])
    assert not any(parameter.requires_grad for parameter in owned["controller"])


def test_a_role_whose_module_holds_no_parameter_owns_none() -> None:
    prepared, policy = _prepared()
    trained = torch.nn.Linear(2, 2)
    topology = _ConfiguredTopology({"empty": torch.nn.Identity(), "trained": trained})
    model = ConfiguredModel(
        topology,
        codec=prepared.model.codec,
        groups=topology.resolve(None),
        policy=policy,
        checkpoint_labels={},
        pretrained={},
    )
    owned = model.role_parameters()

    assert owned["empty"] == ()
    assert {id(parameter) for parameter in owned["trained"]} == {
        id(parameter) for parameter in trained.parameters()
    }


def test_the_configured_topology_owns_and_restores_frozen_capacity_padding() -> None:
    prepared, policy = _prepared()
    trained = torch.nn.Linear(2, 3)
    topology = _ConfiguredTopology({"trained": trained}, target_parameter_count=14)
    model = ConfiguredModel(
        topology,
        codec=prepared.model.codec,
        groups=topology.resolve(None),
        policy=policy,
        checkpoint_labels={},
        pretrained={},
    )
    owned = model.role_parameters()

    assert set(model.modules) == {"trained", CAPACITY_ROLE}
    assert sum(parameter.numel() for parameter in model.named_parameters.values()) == 14
    assert (
        sum(
            parameter.numel()
            for parameter in model.named_parameters.values()
            if parameter.requires_grad
        )
        == 9
    )
    assert len(owned[CAPACITY_ROLE]) == 1
    assert not owned[CAPACITY_ROLE][0].requires_grad
    assert model.groups.members(CAPACITY_ROLE) <= model.groups.frozen
    assert model.groups.members(CAPACITY_ROLE).isdisjoint(model.groups.trainable)
    assert (
        model.activate(("trained", CAPACITY_ROLE)).trainable == model.groups.trainable
    )

    checkpoint = model.checkpoint_state()

    with torch.no_grad():
        owned[CAPACITY_ROLE][0].fill_(7.0)

    model.restore(checkpoint)
    assert torch.equal(
        owned[CAPACITY_ROLE][0], torch.zeros_like(owned[CAPACITY_ROLE][0])
    )


def test_a_production_model_reaches_and_reports_its_declared_parameter_target() -> None:
    natural, _ = _prepared()
    natural_total = sum(
        parameter.numel() for parameter in natural.model.named_parameters.values()
    )
    natural_trainable = sum(
        parameter.numel()
        for parameter in natural.model.named_parameters.values()
        if parameter.requires_grad
    )
    target = natural_total + 7
    run_spec = PACKAGED_CODEC(CONFIG)
    run_spec = run_spec.with_(
        model=run_spec.model.with_(capacity=ParameterCapacitySpec(target=target))
    )
    blueprint = plan_training(run_spec)
    assert run_spec.component_roles()["model.capacity"] == run_spec.model.capacity
    assert "model.capacity" in blueprint.component_fingerprints
    streams = RandomStreams(SeedPolicy(7, samplers=blueprint.sampler_streams))
    streams.apply()
    prepared = prepare_training(blueprint, streams, resolve_device_policy(run_spec))
    declarations = run_spec.model.checkpointed_components()
    owned = prepared.model.role_parameters()
    point = SweepPoint(
        cast("dict[str, JsonValue]", json.loads(run_spec.model_dump_json())),
        seed=7,
    )
    manifest = measure_point(
        point,
        {
            role: MeasuredRole(cast("str", declarations[role].kind), owned[role])
            for role in sorted(owned)
        },
        corpus_fingerprint=blueprint.corpus_revision,
    )

    assert manifest.parameter_count == target
    assert manifest.trainable_count == natural_trainable
    assert manifest.roles[CAPACITY_ROLE].parameters == 7
    assert manifest.roles[CAPACITY_ROLE].trainable == 0


def test_the_distance_plan_holds_the_corpus_it_was_prepared_from() -> None:
    run_spec = RunSpecCodec()(CONFIG.parent / "synthetic-distance.yaml")
    blueprint = plan_training(run_spec)
    streams = RandomStreams(SeedPolicy(7, samplers=blueprint.sampler_streams))
    streams.apply()
    prepared = prepare_training(blueprint, streams, resolve_device_policy(run_spec))
    plan = cast("StepFittingPlan", prepared.plan)
    lowered = cast("SyntheticDataset", blueprint.lower_dataset("train"))

    assert plan.dataset == lowered
    language = blueprint.learner.grammar.language
    trainer = cast("DistanceTrainerSpec", run_spec.trainer)
    assert language.codec.at(trainer.sort) is prepared.model.codec.codec


def test_a_step_plan_rotates_its_steps_and_seeds_each_position() -> None:
    run_spec = RunSpecCodec()(CONFIG.parent / "synthetic-distance.yaml")
    blueprint = plan_training(run_spec)
    streams = RandomStreams(SeedPolicy(7, samplers=blueprint.sampler_streams))
    streams.apply()
    prepared = prepare_training(blueprint, streams, resolve_device_policy(run_spec))
    plan = cast("StepFittingPlan", prepared.plan)
    (step,) = plan.trainers

    assert plan.trainer_at(0) is plan.trainer_at(3) is step
    assert plan.batch_seed(plan.stream_seed, 2) == distance_batch_seed(
        plan.stream_seed, 2
    )
    assert plan.stop is None
    assert plan.validation_seed is None

    with pytest.raises(ValueError, match="at least one step"):
        StepFittingPlan(
            (),
            distance_batch_seed,
            plan.stream_seed,
            validation_seed=None,
            stop=None,
            steps=plan.steps,
            batch_size=plan.batch_size,
            checkpoints=plan.checkpoints,
            dataset=plan.dataset,
        )

    with pytest.raises(ValueError, match="held-out stream"):
        StepFittingPlan(
            plan.trainers,
            distance_batch_seed,
            plan.stream_seed,
            validation_seed=3,
            stop=None,
            steps=plan.steps,
            batch_size=plan.batch_size,
            checkpoints=plan.checkpoints,
            dataset=plan.dataset,
        )


def test_a_role_built_frozen_stays_frozen_around_a_trainable_nested_role() -> None:
    nested = torch.nn.Linear(2, 2)
    own = torch.nn.Linear(2, 2)
    own.requires_grad_(requires_grad=False)
    outer = torch.nn.Sequential(nested, own)
    topology = _ConfiguredTopology({"nested": nested, "outer": outer})
    resolution = topology.resolve(None)

    assert topology.frozen == frozenset({"outer"})
    assert resolution.members("outer") <= resolution.frozen
    assert resolution.members("nested") <= resolution.trainable


def test_a_role_holding_only_a_frozen_nested_role_is_not_itself_frozen() -> None:
    nested = torch.nn.Linear(2, 2)
    nested.requires_grad_(requires_grad=False)
    outer = torch.nn.Sequential(nested)
    trainable = torch.nn.Linear(2, 2)
    topology = _ConfiguredTopology(
        {"nested": nested, "outer": outer, "trainable": trainable}
    )
    resolution = topology.resolve(("trainable",))

    assert topology.frozen == frozenset({"nested"})
    with pytest.raises(ValueError, match="no group named 'outer'"):
        resolution.members("outer")


def test_a_step_plan_fits_no_parser_and_names_its_residuals_and_stack() -> None:
    run_spec = RunSpecCodec()(CONFIG.parent / "synthetic-distance.yaml")
    blueprint = plan_training(run_spec)
    streams = RandomStreams(SeedPolicy(7, samplers=blueprint.sampler_streams))
    streams.apply()
    prepared = prepare_training(blueprint, streams, resolve_device_policy(run_spec))
    plan = cast("StepFittingPlan", prepared.plan)
    (step,) = plan.trainers

    assert plan.held_out_parser() is None
    assert dict(plan.residual_keys) == dict(step.assembly.residual_keys)
    assert plan.timed_stack().positions == plan.steps
    assert plan.timed_stack().steps_per_unit == 1

    with pytest.raises(ValueError, match="fits no parser"):
        plan.parser_plan()


def test_a_parser_plan_is_its_own_parser_and_times_its_epoch() -> None:
    run_spec = PACKAGED_CODEC(CONFIG)
    blueprint = plan_training(run_spec)
    streams = RandomStreams(SeedPolicy(7, samplers=blueprint.sampler_streams))
    streams.apply()
    prepared = prepare_training(blueprint, streams, resolve_device_policy(run_spec))
    plan = cast("ParserFittingPlan", prepared.plan)

    assert plan.parser_plan() is plan
    assert plan.held_out_parser() is plan
    assert dict(plan.residual_keys) == dict(plan.step.assembly.residual_keys)
    assert plan.timed_stack().positions == len(plan.batches)

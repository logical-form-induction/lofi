from __future__ import annotations

import random
from copy import deepcopy
from json import loads
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Never, cast, override

import didactic.api as dx
import pytest
import torch
import yaml

from lofi.data.synthetic.spec.corpus import (
    CheckedSyntheticCorpus,
    SyntheticCorpusLowerer,
)
from lofi.formalism import (
    BoundedConsumer,
    ProfileBound,
)
from lofi.judge import ActionSet
from lofi.registry import Registry
from lofi.run import (
    ExperimentSeed,
    Selection,
    compose,
    plan_training,
    training_length,
)
from lofi.run._entrypoint import main
from lofi.run._training import (
    _learner_signature_inventories,
    _shared_source,
)
from lofi.spec import (
    ActionsSpec,
    BootstrapIntervalSpec,
    ConfigDocumentError,
    EvaluatorSpec,
    ProbeSpec,
    RunSpec,
    RunSpecCodec,
    SignatureAssignmentsSpec,
    SinkSpec,
    UnknownSettingError,
    config_search_path,
    schema_uris,
)
from lofi.spec.components.data import TrainingBatchKind
from lofi.spec.components.loss import LossSection, WeightedLossSpec
from lofi.spec.components.trainer import TrainerSpec
from lofi.spec.constraint_sources import GoldLexiconConstraintSourceSpec
from lofi.spec.grammar import declare_grammar
from lofi.spec.losses.chart_scale import ChartScaleLossSpec
from lofi.spec.losses.pinning import PinningLossSpec
from lofi.spec.parsers.flat import FlatParserSpec
from lofi.spec.span_decoders.rnn import RnnSpanDecoderSpec
from lofi.training_plugins import (
    FeaturizerMismatchError,
    TrainerBuildContext,
    TrainerPlugin,
    TrainingLength,
    TrainingUnit,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi._types import JsonValue
    from lofi.data import SupervisionArm
    from lofi.data.datasets import CheckedDataset
    from lofi.data.synthetic import SyntheticDataset
    from lofi.data.synthetic._interpretation import InterpretationFunction
    from lofi.data.synthetic.spec.corpus import SyntheticCorpusDecl
    from lofi.spec.grammar import GrammarAssembly
    from lofi.spec.trainers import ParserTrainerSpec

CONFIG = Path(__file__).resolve().parents[3] / "lofi/configs/run/synthetic-parser.yaml"
DISTANCE_CONFIG = (
    Path(__file__).resolve().parents[3] / "lofi/configs/run/synthetic-distance.yaml"
)
MAPPED_CONFIG = (
    Path(__file__).resolve().parents[3]
    / "lofi/configs/run/synthetic-parser-mapped.yaml"
)
INFERENCE_DATASET = (
    Path(__file__).resolve().parents[3]
    / "lofi/configs/data/datasets/train/dataset/synthetic-inference.yaml"
)
CATEGORY_EVALUATION = (
    Path(__file__).resolve().parents[2]
    / "acceptance/documents/lf-category-metrics.yaml"
)
PACKAGED_CODEC = RunSpecCodec(search_path=(CONFIG.parent, CONFIG.parent.parent))


class TrainingRoleSink(SinkSpec):
    kind: Literal["training-role-sink"] = "training-role-sink"


class TrainingRoleProbe(ProbeSpec):
    kind: Literal["training-role-probe"] = "training-role-probe"


class TrainingRoleEvaluator(EvaluatorSpec):
    kind: Literal["training-role-evaluator"] = "training-role-evaluator"


class ExternalPlanningTrainer(TrainerSpec):
    kind: Literal["external-planning"] = "external-planning"

    @override
    def dataset_requirements(
        self,
    ) -> MappingProxyType[str, frozenset[TrainingBatchKind]]:
        """Require sentence batches from the declared training dataset."""
        return MappingProxyType({"train": frozenset({TrainingBatchKind.SENTENCES})})


def _plan_external(
    declaration: ExternalPlanningTrainer, run_spec: RunSpec, /
) -> tuple[str, ...]:
    del declaration, run_spec

    return ("external-sampler",)


def _do_not_build_external(
    declaration: ExternalPlanningTrainer, context: TrainerBuildContext, /
) -> Never:
    del declaration, context
    msg = "planning must not build the external trainer"
    raise AssertionError(msg)


def _price_external(declaration: ExternalPlanningTrainer, /) -> None:
    del declaration


def _present_external(
    declaration: ExternalPlanningTrainer, run_spec: RunSpec, /
) -> Mapping[str, frozenset[SupervisionArm]]:
    del declaration, run_spec

    return {}


EXTERNAL_PLANNING = TrainerPlugin(
    spec=ExternalPlanningTrainer,
    plan=_plan_external,
    present=_present_external,
    build=_do_not_build_external,
    price=_price_external,
)


def test_an_installed_trainer_plans_without_a_central_identity_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_get = Registry.get

    def installed[MemberT](registry: Registry[MemberT], tag: str) -> MemberT:
        if registry.family == "trainer" and tag == "external-planning":
            return cast("MemberT", EXTERNAL_PLANNING)

        return original_get(registry, tag)

    monkeypatch.setattr(Registry, "get", installed)
    run_spec = PACKAGED_CODEC(CONFIG).with_(trainer=ExternalPlanningTrainer())

    assert plan_training(run_spec).sampler_streams == ("external-sampler",)


def test_planning_is_pure_and_covers_every_structural_component_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_spec = PACKAGED_CODEC(CONFIG)
    python = random.getstate()
    torch_state = torch.random.get_rng_state().clone()
    original_get = Registry.get

    def refuse_module_allocation(_module: torch.nn.Module) -> None:
        msg = "planning allocated a neural module"
        raise AssertionError(msg)

    def refuse_resolution[FactoryT](registry: Registry[FactoryT], tag: str) -> FactoryT:
        if registry.family in {"dataset", "featurizer", "trainer"}:
            return original_get(registry, tag)

        msg = "planning resolved a component factory"
        raise AssertionError(msg)

    def refuse_lowering(
        _lowerer: SyntheticCorpusLowerer, _checked: CheckedSyntheticCorpus
    ) -> SyntheticDataset:
        msg = "planning lowered a checked corpus"
        raise AssertionError(msg)

    monkeypatch.setattr(torch.nn.Module, "__init__", refuse_module_allocation)
    monkeypatch.setattr(Registry, "get", refuse_resolution)
    monkeypatch.setattr(SyntheticCorpusLowerer, "lower", refuse_lowering)
    blueprint = plan_training(run_spec)

    assert random.getstate() == python
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    assert set(blueprint.component_fingerprints) == set(run_spec.component_roles())
    assert {
        "model.token_encoder",
        "model.subword_pooler",
        "model.encoders.terms",
        "model.decoders.identity",
        "model.decoders.application",
        "model.readouts.types",
        "loss.terms.0",
        "loss.terms.1",
        "data.datasets.train.dataset",
        "data.datasets.train.featurizer",
        "training.trainer",
        "training.optimizer",
        "training.scheduler",
    } <= set(run_spec.component_roles())
    assert blueprint.run_fingerprint.languages["grammar"].digest
    assert dict(blueprint.run_fingerprint.schemas) == schema_uris(RunSpec)
    assert len(blueprint.codec_fingerprint) == 64
    assert len(blueprint.action_set_fingerprint) == 64
    assert blueprint.configuration == run_spec.model_dump_json().encode()
    assert set(run_spec.model.checkpointed_components()) == {
        "token_encoder",
        "subword_pooler",
        "terms",
        "controller",
        "parser",
        "identity",
        "application",
    }
    assert blueprint.sampler_streams == (
        "training:train:validation-split",
        "training:train:training-batches",
        "training:train:validation-batches",
    )


def test_component_roles_are_the_exact_complete_structural_map() -> None:
    run_spec = PACKAGED_CODEC(CONFIG)
    declared = run_spec.with_(
        sinks={"metrics": TrainingRoleSink()},
        probes={"chart": TrainingRoleProbe()},
        evaluators={
            "heldout": TrainingRoleEvaluator(
                interval=BootstrapIntervalSpec(replicates=50, level=0.95)
            )
        },
    )
    loss = declared.loss

    assert loss is not None

    assert dict(declared.component_roles()) == {
        "data.datasets.train.dataset": declared.data.datasets["train"].dataset,
        "data.datasets.train.featurizer": declared.data.datasets["train"].featurizer,
        "loss.terms.0": loss.terms[0].objective,
        "loss.terms.1": loss.terms[1].objective,
        "model.decoders.application": declared.model.decoders["application"],
        "model.decoders.identity": declared.model.decoders["identity"],
        "model.encoders.terms": declared.model.encoders["terms"],
        "model.controller": declared.model.controller,
        "model.parser": declared.model.parser,
        "model.readouts.category-terms": declared.model.readouts["category-terms"],
        "model.readouts.types": declared.model.readouts["types"],
        "model.subword_pooler": declared.model.subword_pooler,
        "model.token_encoder": declared.model.token_encoder,
        "output.evaluators.heldout": declared.evaluators["heldout"],
        "output.probes.chart": declared.probes["chart"],
        "output.sinks.metrics": declared.sinks["metrics"],
        "training.optimizer": declared.optimizer,
        "training.scheduler": declared.scheduler,
        "training.trainer": declared.trainer,
    }


def test_a_mapped_space_relation_has_one_checkpoint_component_declaration() -> None:
    run_spec = PACKAGED_CODEC(MAPPED_CONFIG)
    relation = run_spec.model.spaces["categories"].relation

    assert relation is not None
    assert relation.mapping is not None
    assert (
        run_spec.component_roles()["model.spaces.categories.relation.mapping"]
        == relation.mapping
    )
    assert run_spec.model.checkpointed_components()["categories"] == relation.mapping


def test_the_packaged_vector_resource_keeps_its_stable_identity() -> None:
    run_spec = PACKAGED_CODEC(CONFIG)
    token_encoder = run_spec.model.token_encoder

    assert token_encoder is not None
    assert token_encoder.model_dump()["vectors"] == (
        "lofi.resource:run/synthetic-parser.vec"
    )


def test_distance_planning_uses_only_its_declared_graph_and_sampler_stream() -> None:
    run_spec = RunSpecCodec()(DISTANCE_CONFIG)
    blueprint = plan_training(run_spec)

    assert blueprint.sampler_streams == ("training:train:distance-batches",)
    assert set(run_spec.component_roles()) == {
        "data.datasets.train.dataset",
        "data.type_sampler",
        "model.encoders.terms",
        "training.optimizer",
        "training.scheduler",
        "training.trainer",
    }
    assert run_spec.model.parser is None
    assert run_spec.loss is None
    assert run_spec.data.datasets["train"].featurizer is None


def test_a_declared_span_decoder_is_a_structural_and_a_stateful_model_role() -> None:
    run_spec = PACKAGED_CODEC(CONFIG)
    declared = run_spec.with_(
        model=run_spec.model.with_(
            span_decoders={"reconstruction": RnnSpanDecoderSpec()}
        )
    )

    assert (
        declared.component_roles()["model.span_decoders.reconstruction"]
        == RnnSpanDecoderSpec()
    )
    assert (
        declared.model.checkpointed_components()["reconstruction"]
        == RnnSpanDecoderSpec()
    )


def test_distance_composition_refuses_a_declared_span_decoder() -> None:
    run_spec = RunSpecCodec()(DISTANCE_CONFIG)

    with pytest.raises(dx.ValidationError, match=r"model\.span_decoders"):
        run_spec.with_(
            model=run_spec.model.with_(
                span_decoders={"reconstruction": RnnSpanDecoderSpec()}
            )
        )


def _pinning_document() -> dict[str, JsonValue]:
    """Return the parser declaration with one current pinning path."""
    document = cast(
        "dict[str, JsonValue]",
        deepcopy(loads(PACKAGED_CODEC(CONFIG).model_dump_json())),
    )
    data = document["data"]
    loss = document["loss"]

    assert isinstance(data, dict)
    assert isinstance(loss, dict)
    datasets = data["datasets"]
    assert isinstance(datasets, dict)
    train = datasets["train"]
    assert isinstance(train, dict)
    terms = loss["terms"]
    assert isinstance(terms, list)
    train["constraint_source"] = {"kind": "gold-lexicon", "policy": "syntactic"}
    terms.append(
        {
            "objective": {"kind": "pinning", "readout": "types"},
            "weight": 1.0,
        }
    )

    return document


def test_pinning_planning_names_the_source_and_keeps_the_parser_step_streams() -> None:
    document = _pinning_document()
    run_spec = RunSpec.model_validate(document)
    blueprint = plan_training(run_spec)

    assert isinstance(
        run_spec.data.datasets["train"].constraint_source,
        GoldLexiconConstraintSourceSpec,
    )
    loss = run_spec.loss
    assert loss is not None
    assert isinstance(loss.terms[-1].objective, PinningLossSpec)
    assert "data.datasets.train.constraint_source" in run_spec.component_roles()
    assert blueprint.sampler_streams == (
        "training:train:validation-split",
        "training:train:training-batches",
        "training:train:validation-batches",
    )


@pytest.mark.parametrize(
    ("remove", "pattern"),
    [
        (
            "source",
            (
                "requires both one source and a parser constraint objective "
                "or composed token-warm-up step"
            ),
        ),
        (
            "pinning",
            (
                "constraint source 'gold-lexicon' requires at least one 'target' "
                "constraint consumer"
            ),
        ),
    ],
)
def test_parser_planning_refuses_inert_or_unresolved_pinning_wiring(
    remove: str, pattern: str
) -> None:
    document = _pinning_document()
    data = document["data"]
    loss = document["loss"]
    assert isinstance(data, dict)
    assert isinstance(loss, dict)
    datasets = data["datasets"]
    assert isinstance(datasets, dict)
    train = datasets["train"]
    assert isinstance(train, dict)
    terms = loss["terms"]
    assert isinstance(terms, list)

    if remove == "source":
        train["constraint_source"] = None
    else:
        terms.pop()

    run_spec = RunSpec.model_validate(document)

    with pytest.raises(ValueError, match=pattern):
        plan_training(run_spec)


def test_a_pinning_objective_on_an_undeclared_readout_is_refused_at_composition() -> (
    None
):
    document = _pinning_document()
    loss = document["loss"]
    assert isinstance(loss, dict)
    terms = loss["terms"]
    assert isinstance(terms, list)
    pinning = terms[-1]
    assert isinstance(pinning, dict)
    objective = pinning["objective"]
    assert isinstance(objective, dict)
    objective["readout"] = "missing"

    with pytest.raises(
        dx.ValidationError,
        match=r"loss\.terms\.pinning: the objective names readout role 'missing'",
    ):
        RunSpec.model_validate(document)


def _anchored_document(readout: str, /) -> dict[str, JsonValue]:
    document = _pinning_document()
    loss = document["loss"]
    assert isinstance(loss, dict)
    terms = loss["terms"]
    assert isinstance(terms, list)
    terms.append(
        {
            "objective": {"kind": "vector-anchor", "readout": readout},
            "weight": 1.0,
        }
    )

    return document


def test_constraint_objectives_on_one_readout_share_one_selection() -> None:
    run_spec = RunSpec.model_validate(_anchored_document("types"))

    assert plan_training(run_spec).sampler_streams == (
        "training:train:validation-split",
        "training:train:training-batches",
        "training:train:validation-batches",
    )


def test_category_side_planning_requires_its_retained_derivation_splits() -> None:
    declared = RunSpecCodec(search_path=config_search_path(CATEGORY_EVALUATION.parent))(
        CATEGORY_EVALUATION
    )
    document = loads(declared.model_dump_json())
    data = document["data"]
    assert isinstance(data, dict)
    datasets = data["datasets"]
    assert isinstance(datasets, dict)
    train = datasets["train"]
    assert isinstance(train, dict)
    dataset = train["dataset"]
    assert isinstance(dataset, dict)
    dataset["retained"] = {"validation": 0.25}
    run_spec = RunSpec.model_validate(document)

    with pytest.raises(ValueError, match="retained pair-side derivation splits"):
        plan_training(run_spec)


def test_constraint_objectives_on_two_readouts_are_refused() -> None:
    run_spec = RunSpec.model_validate(_anchored_document("category-terms"))

    with pytest.raises(ValueError, match=r"more than one term-readout role"):
        plan_training(run_spec)


@pytest.mark.parametrize(
    ("document", "featurizer", "presented"),
    [
        ("docs/examples/synthetic-parser.yaml", "pair", "Unlabeled"),
        ("experiments/l4/l4-run.yaml", "derivation", "Relational"),
    ],
)
def test_planning_refuses_a_featurizer_that_does_not_accept_the_trainer_arm(
    document: str, featurizer: str, presented: str
) -> None:
    root = Path(__file__).resolve().parents[3]
    run_spec = RunSpecCodec(search_path=config_search_path(root))(
        root / document,
        overrides=(f"data/datasets/train/featurizer={featurizer}",),
    )

    assert run_spec.data.datasets["train"].featurizer is not None

    with pytest.raises(
        FeaturizerMismatchError,
        match=f"presents {presented} examples, but the {featurizer!r} featurizer",
    ):
        plan_training(run_spec, point=ExperimentSeed(0).point("featurizer", 0))


def test_parser_planning_refuses_weighted_syntactic_pinning() -> None:
    document = _pinning_document()
    data = document["data"]
    assert isinstance(data, dict)
    datasets = data["datasets"]
    assert isinstance(datasets, dict)
    train = datasets["train"]
    assert isinstance(train, dict)
    policies = train["constraint_policy"]
    assert isinstance(policies, dict)
    syntactic = policies["syntactic"]
    assert isinstance(syntactic, dict)
    syntactic["ambiguity"] = "weighted"
    run_spec = RunSpec.model_validate(document)

    with pytest.raises(ValueError, match="does not admit weighted labels"):
        plan_training(run_spec)


_CONVENTIONS: dict[str, JsonValue] = {
    "punctuation": "keep",
    "unary_chains": "keep",
    "trivial_spans": "keep",
    "minimum_length": 0,
    "aggregation": "sentence",
    "coverage": "zero",
    "self_arcs": "allow",
    "gold_derivation": "canonical",
}


def _evaluated_document(readout: str) -> dict[str, JsonValue]:
    """Return the parser declaration with one lexical-type evaluator."""
    document = cast(
        "dict[str, JsonValue]",
        deepcopy(loads(PACKAGED_CODEC(CONFIG).model_dump_json())),
    )
    document["evaluators"] = {
        "lexicon": {
            "kind": "lexical-types",
            "interval": {"replicates": 50, "level": 0.95},
            "readout": readout,
            "alignment": {
                "kind": "identity",
                "fit_on": "dev",
                "score_on": "test",
                "random_floor": {"replicates": 50, "level": 0.95},
                "stability": {"replicates": 50, "level": 0.95},
            },
            "conventions": dict(_CONVENTIONS),
        }
    }

    return document


def test_parser_planning_admits_an_evaluator_over_a_declared_readout() -> None:
    run_spec = RunSpec.model_validate(_evaluated_document("types"))

    assert "output.evaluators.lexicon" in plan_training(run_spec).component_fingerprints


def test_parser_planning_refuses_an_evaluator_over_an_undeclared_readout() -> None:
    run_spec = RunSpec.model_validate(_evaluated_document("missing"))

    with pytest.raises(
        ValueError, match="evaluator 'lexicon' names readout role 'missing'"
    ):
        plan_training(run_spec)


def test_distance_composition_refuses_a_held_out_evaluator() -> None:
    run_spec = RunSpecCodec()(DISTANCE_CONFIG)

    with pytest.raises(dx.ValidationError, match=r"output\.evaluators"):
        run_spec.with_(
            evaluators={
                "heldout": TrainingRoleEvaluator(
                    interval=BootstrapIntervalSpec(replicates=50, level=0.95)
                )
            }
        )


def test_parser_composition_refuses_a_run_missing_a_required_role() -> None:
    run_spec = PACKAGED_CODEC(CONFIG)

    with pytest.raises(
        dx.ValidationError,
        match=r"'parser' trainer requires component roles \['loss'\]",
    ):
        run_spec.with_(loss=None)


def test_parser_composition_refuses_a_type_sampler() -> None:
    run_spec = PACKAGED_CODEC(CONFIG)
    sampler = RunSpecCodec()(DISTANCE_CONFIG).type_sampler

    with pytest.raises(dx.ValidationError, match=r"\['data\.type_sampler'\]"):
        run_spec.with_(type_sampler=sampler)


def _inference_parser_document(root: Path, /) -> Path:
    """Write a document composing the parser run over the inference dataset."""
    document = root / "inference-parser.yaml"
    document.write_text(
        "defaults:\n"
        "  - run/synthetic-parser\n"
        "  - data.datasets.train.dataset: synthetic-inference\n",
        encoding="utf-8",
        newline="\n",
    )

    return document


def test_inference_pairs_without_a_grounding_are_refused_at_composition(
    tmp_path: Path,
) -> None:
    selection = Selection(
        config=_inference_parser_document(tmp_path),
        profile=None,
        overrides=(),
        overlays=(),
        search_path=config_search_path(tmp_path),
    )

    with pytest.raises(ConfigDocumentError) as refused:
        compose(selection)

    message = str(refused.value)
    assert "trainer does not consume them" in message
    assert "model.grounding is absent" in message


def test_lofi_run_refuses_an_inference_document_under_the_parser_before_any_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    document = _inference_parser_document(tmp_path)

    status = main(
        [
            "run",
            "--config",
            str(document),
            "--experiment-seed",
            "5",
            "--root",
            str(tmp_path / "runs"),
        ]
    )

    assert status != 0
    assert "model.grounding is absent" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


def test_planning_names_the_sentence_lowering_a_dataset_does_not_provide(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_get = Registry.get

    def installed[MemberT](registry: Registry[MemberT], tag: str) -> MemberT:
        if registry.family == "trainer" and tag == "external-planning":
            return cast("MemberT", EXTERNAL_PLANNING)

        return original_get(registry, tag)

    monkeypatch.setattr(Registry, "get", installed)
    document = loads(PACKAGED_CODEC(CONFIG).model_dump_json())
    document["trainer"] = {"kind": "external-planning"}
    document["data"]["datasets"]["train"]["dataset"] = yaml.safe_load(
        INFERENCE_DATASET.read_text(encoding="utf-8")
    )
    with pytest.raises(
        dx.ValidationError,
        match=r"trainer does not consume them and model\.grounding is absent",
    ):
        RunSpec.model_validate(document)


IDENTITY_RULE = "identity-combinator"


def test_the_identity_rule_document_composes_into_a_run_and_draws_data(
    tmp_path: Path,
) -> None:
    document = tmp_path / "identity.yaml"
    document.write_text(
        "defaults:\n"
        "  - run/synthetic-parser\n"
        "  - data.datasets.train.dataset.grammar.rules: application-identity\n"
        "data:\n"
        "  datasets:\n"
        "    train:\n"
        "      dataset:\n"
        "        grammar:\n"
        "          budget: {unary_rounds: 1}\n"
        "model:\n"
        "  parser: {kind: tagged, width: 4, unary_depth: 1}\n",
        encoding="utf-8",
        newline="\n",
    )
    run_spec = compose(
        Selection(
            config=document,
            profile=None,
            overrides=(),
            overlays=(),
            search_path=config_search_path(tmp_path),
        )
    )
    blueprint = plan_training(run_spec)
    dataset = cast("SyntheticDataset", blueprint.lower_dataset("train"))
    rules = tuple(rule.name for rule in blueprint.learner.grammar.grammar.rules)
    declared = cast("SyntheticCorpusDecl", run_spec.data.datasets["train"].dataset)
    grammar = declare_grammar(declared.grammar)
    lexicon = blueprint.learner.grammar.lexicon
    categories = {
        entry.category for form in lexicon.forms for entry in lexicon.entries(form)
    }
    identity = next(
        rule
        for rule in blueprint.learner.grammar.grammar.rules
        if rule.name == IDENTITY_RULE
    )

    assert tuple(rule.rule for rule in grammar.rules.rules) == (
        "application",
        "application",
        "pattern",
    )
    assert categories
    assert all(
        list(identity.apply((category,))) == [category] for category in categories
    )
    assert len(dataset.corpus.examples) == declared.size
    assert blueprint.action_set_fingerprint == (
        ActionSet.derive(rules, identity="identity", null=None).digest
    )


def test_the_blueprint_derives_the_declared_actions_into_its_action_set() -> None:
    plain = plan_training(PACKAGED_CODEC(CONFIG))
    run_spec = PACKAGED_CODEC(CONFIG)
    declared = run_spec.with_(
        model=run_spec.model.with_(
            actions=ActionsSpec(identity="keep", null="none-applies")
        )
    )
    blueprint = plan_training(declared)
    rules = tuple(rule.name for rule in blueprint.learner.grammar.grammar.rules)

    assert blueprint.actions == (
        ActionSet.derive(rules, identity="keep", null="none-applies")
    )
    assert blueprint.action_set_fingerprint == blueprint.actions.digest
    assert blueprint.action_set_fingerprint != plain.action_set_fingerprint


def test_soft_raising_without_a_raising_rule_is_refused_before_a_run_directory() -> (
    None
):
    run_spec = PACKAGED_CODEC(CONFIG)
    controller = run_spec.model.controller

    assert controller is not None

    declared = run_spec.with_(
        model=run_spec.model.with_(controller=controller.with_(**{"raise": "soft"}))
    )

    with pytest.raises(ValueError, match="normal-form role 'raising'"):
        plan_training(declared)


def test_signature_assignments_reorder_learner_owned_action_columns() -> None:
    run_spec = PACKAGED_CODEC(CONFIG)
    learner = plan_training(run_spec).learner.grammar
    primitives, constructors, combinators = _learner_signature_inventories(learner)
    assigned = run_spec.with_(
        signature_assignment=SignatureAssignmentsSpec(
            primitives=tuple(reversed(primitives)),
            constructors=tuple(reversed(constructors)),
            combinators=tuple(reversed(combinators)),
        )
    )

    blueprint = plan_training(assigned)

    assert tuple(
        action.rule for action in blueprint.actions.actions if action.rule is not None
    ) == tuple(reversed(combinators))


@pytest.mark.parametrize("field", ["primitives", "constructors", "combinators"])
def test_signature_assignments_must_permute_each_learner_inventory(field: str) -> None:
    run_spec = PACKAGED_CODEC(CONFIG)
    learner = plan_training(run_spec).learner.grammar
    primitives, constructors, combinators = _learner_signature_inventories(learner)
    inventories = {
        "primitives": primitives,
        "constructors": constructors,
        "combinators": combinators,
    }
    inventories[field] = (*inventories[field][1:], "not-a-learner-name")
    assigned = run_spec.with_(
        signature_assignment=SignatureAssignmentsSpec(**inventories)
    )

    with pytest.raises(ValueError, match=rf"signature_assignment\.{field}"):
        plan_training(assigned)


class _InterpretationVariant:
    """A checked source with one deliberately different interpretation."""

    def __init__(self, source: CheckedDataset, /) -> None:
        self._source = source

    @property
    def declaration(self) -> bytes:
        return self._source.declaration

    @property
    def source_grammar(self) -> GrammarAssembly:
        return self._source.source_grammar

    @property
    def source_interpretation(self) -> InterpretationFunction | None:
        return self._source.source_interpretation

    @property
    def source_interpretation_declaration(self) -> dict[str, JsonValue] | None:
        return {"entity": "different"}


def test_data_owned_learning_requires_one_shared_source_interpretation() -> None:
    checked = plan_training(PACKAGED_CODEC(CONFIG)).checked_dataset("train")
    sources = cast(
        "Mapping[str, CheckedDataset]",
        {"train": checked, "validation": _InterpretationVariant(checked)},
    )

    with pytest.raises(ValueError, match="grammar, language, and interpretation"):
        _shared_source(sources)


COORDINATION = """\
defaults:
  - run/synthetic-parser
data:
  datasets:
    train:
      dataset:
        grammar:
          rules:
            name: application-coordination
            rules:
              - {rule: application, view: /}
              - {rule: application, view: "\\\\"}
              - rule: pattern
                name: coordinate
                premises:
                  - &np {pattern: constant, name: NP}
                  - {pattern: constant, name: conj}
                  - *np
                conclusion: *np
"""


def _composed(document: Path, search: Path, *overrides: str) -> RunSpec:
    return compose(
        Selection(
            config=document,
            profile=None,
            overrides=overrides,
            overlays=(),
            search_path=config_search_path(search),
        )
    )


def _coordinating(tmp_path: Path, *overrides: str) -> RunSpec:
    document = tmp_path / "coordination.yaml"
    document.write_text(COORDINATION, encoding="utf-8", newline="\n")

    return _composed(document, tmp_path, *overrides)


def test_a_ternary_rule_plans_under_the_span_chart(
    tmp_path: Path,
) -> None:
    blueprint = plan_training(_coordinating(tmp_path))
    assert blueprint.learner.grammar.grammar.profile.ranks == frozenset({2, 3})


def test_a_ternary_rule_plans_under_a_parser_that_combines_no_cells(
    tmp_path: Path,
) -> None:
    run_spec = _coordinating(tmp_path)
    trainer = cast("ParserTrainerSpec", run_spec.trainer)
    flat = run_spec.with_(
        trainer=trainer.with_(
            early_stopping=trainer.early_stopping.with_(metric="loss/chart-scale")
        ),
        model=run_spec.model.with_(
            parser=FlatParserSpec(
                width=4,
                features=("left", "right", "bilinear"),
                projection_initialization="identity",
            ),
            readouts={},
        ),
        loss=LossSection(terms=(WeightedLossSpec(objective=ChartScaleLossSpec()),)),
    )
    blueprint = plan_training(flat)
    dataset = cast("SyntheticDataset", blueprint.lower_dataset("train"))

    assert blueprint.learner.grammar.grammar.profile.ranks == frozenset({2, 3})
    checked = cast("CheckedSyntheticCorpus", blueprint.checked_dataset("train"))
    assert len(dataset.corpus.examples) == checked.source.size


@pytest.mark.parametrize("setting", ["fanout=2", "lexicalized=false"])
def test_a_grammar_cannot_declare_its_own_profile(tmp_path: Path, setting: str) -> None:
    field, _, value = setting.partition("=")
    document = tmp_path / "invalid-profile.yaml"
    document.write_text(
        COORDINATION.replace(
            "        grammar:\n",
            f"        grammar:\n          {field}: {value}\n",
            1,
        ),
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(UnknownSettingError, match=rf"grammar\.{field}"):
        _composed(document, tmp_path)


@pytest.mark.parametrize(
    "document",
    sorted(CONFIG.parent.glob("*.yaml")),
    ids=lambda path: path.stem,
)
def test_every_packaged_run_plans_with_a_profile_its_parser_serves(
    tmp_path: Path, document: Path
) -> None:
    run_spec = _composed(document, tmp_path)
    grammar = plan_training(run_spec).learner.grammar.grammar
    parser = run_spec.model.parser
    bound = (
        parser.profile_bound()
        if isinstance(parser, BoundedConsumer)
        else ProfileBound("lofi.deduce.parse", 1, frozenset({1, 2, 3}))
    )

    assert grammar.profile.fanout == 1
    assert grammar.profile.lexicalized is True
    assert grammar.profile.ranks <= bound.ranks


def test_planning_draws_a_derived_corpus_under_the_points_seed() -> None:
    run_spec = PACKAGED_CODEC(
        CONFIG, overrides=("data.datasets.train.dataset.seed=derived",)
    )
    first = ExperimentSeed(3).point("0123456789abcdef", 0)
    second = ExperimentSeed(3).point("fedcba9876543210", 1)
    planned = plan_training(run_spec, point=first)
    checked = cast("CheckedSyntheticCorpus", planned.checked_dataset("train")).source

    assert planned.run_spec == run_spec
    assert checked.seed == first.data_seed("data.datasets.train.dataset.seed")
    assert planned.corpus_revision != (
        plan_training(run_spec, point=second).corpus_revision
    )


def test_planning_a_fixed_corpus_ignores_the_point() -> None:
    run_spec = PACKAGED_CODEC(CONFIG)
    point = ExperimentSeed(3).point("0123456789abcdef", 0)

    assert (
        plan_training(run_spec, point=point).corpus_revision
        == plan_training(run_spec).corpus_revision
    )


PLUGIN_TRAINER = """from types import MappingProxyType
from typing import Literal, override

from lofi.spec import RunRole, RunSpec, TrainingBatchKind
from lofi.spec.components.trainer import TrainerSpec
from lofi.spec.trainers import CheckpointPolicySpec, DistanceTrainerSpec
from lofi.training_plugins import (
    TrainerBuildContext,
    TrainerPlugin,
    TrainingLength,
    TrainingUnit,
    trainer_plugin,
)


class FixedHorizonTrainer(TrainerSpec):
    kind: Literal["{tag}"] = "{tag}"
    dataset: str
    horizon: int

    @override
    def required_roles(self) -> frozenset[RunRole]:
        return frozenset({{RunRole.TERM_ENCODERS, RunRole.TYPE_SAMPLER}})

    @override
    def forbidden_roles(self) -> frozenset[RunRole]:
        return frozenset({{RunRole.PARSER, RunRole.LOSS, RunRole.EVALUATORS}})

    @override
    def dataset_requirements(
        self,
    ) -> MappingProxyType[str, frozenset[TrainingBatchKind]]:
        return MappingProxyType(
            {{self.dataset: frozenset({{TrainingBatchKind.SENTENCES}})}}
        )


def _distance(declaration: FixedHorizonTrainer) -> DistanceTrainerSpec:
    return DistanceTrainerSpec(
        encoder="terms",
        dataset=declaration.dataset,
        sort="Ty",
        steps=declaration.horizon,
        batch_size=2,
        checkpoints=CheckpointPolicySpec(interval=declaration.horizon, keep=1),
        inverse_distance_power=0.0,
        embedding_penalty=0.0,
    )


def _plan(declaration: FixedHorizonTrainer, run_spec: RunSpec) -> tuple[str, ...]:
    distance = _distance(declaration)
    return trainer_plugin(distance).plan(distance, run_spec)


def _build(declaration: FixedHorizonTrainer, context: TrainerBuildContext) -> object:
    distance = _distance(declaration)
    return trainer_plugin(distance).prepare(distance, context)


def _price(declaration: FixedHorizonTrainer) -> TrainingLength:
    return TrainingLength(declaration.horizon, TrainingUnit.STEPS)


def _present(
    declaration: FixedHorizonTrainer, run_spec: RunSpec, /
) -> dict[str, frozenset[type]]:
    del declaration, run_spec
    return {{}}


PLUGIN = TrainerPlugin(
    spec=FixedHorizonTrainer,
    plan=_plan,
    present=_present,
    build=_build,
    price=_price,
)
"""


def _install_trainer_package(root: Path, monkeypatch: pytest.MonkeyPatch, /) -> str:
    """Install a distribution whose one trainer entry point names its plugin."""
    suffix = root.name.replace("-", "_").lower()
    module = f"lofi_trainer_plugin_{suffix}"
    tag = f"fixed-horizon-{suffix.replace('_', '-')}"
    package = root / "site"
    package.mkdir()
    (package / f"{module}.py").write_text(
        PLUGIN_TRAINER.format(tag=tag), encoding="utf-8", newline="\n"
    )
    metadata = package / f"{module}-0.1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {module}\nVersion: 0.1.0\n",
        encoding="utf-8",
        newline="\n",
    )
    (metadata / "entry_points.txt").write_text(
        f"[lofi.trainer]\n{tag} = {module}:PLUGIN\n", encoding="utf-8", newline="\n"
    )
    monkeypatch.syspath_prepend(str(package))

    return tag


def test_a_trainer_from_an_installed_package_plans_prices_and_trains_from_a_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tag = _install_trainer_package(tmp_path, monkeypatch)
    document = yaml.safe_load(DISTANCE_CONFIG.read_text(encoding="utf-8"))
    document["trainer"] = {"kind": tag, "dataset": "train", "horizon": 3}
    config = tmp_path / "plugin-trainer.yaml"
    config.write_text(yaml.safe_dump(document), encoding="utf-8", newline="\n")
    root = tmp_path / "runs"

    run_spec = RunSpecCodec()(config)

    assert type(run_spec.trainer).__module__.startswith("lofi_trainer_plugin_")
    assert plan_training(run_spec).sampler_streams == (
        "training:train:distance-batches",
    )
    assert training_length(run_spec) == TrainingLength(3, TrainingUnit.STEPS)

    status = main(
        [
            "run",
            "--config",
            str(config),
            "--seed",
            "3",
            "--experiment-seed",
            "5",
            "--root",
            str(root),
        ]
    )

    assert status == 0
    (directory,) = (root / "runs").glob("[0-9a-f]*")
    manifest = loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"

    parser_run = PACKAGED_CODEC(CONFIG)
    with pytest.raises(dx.ValidationError, match=r"\['loss'\]"):
        run_spec.with_(
            approximation=parser_run.approximation,
            loss=parser_run.loss,
        )

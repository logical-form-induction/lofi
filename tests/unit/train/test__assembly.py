"""Assembly of declared training components, losses, and fitting plans."""

from json import loads
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, Never, cast
from unittest.mock import Mock

import pytest
import torch
from tests.acceptance.driver import composed
from tests.conformance.component_plugin import plugin_context
from tests.support.installed import installed

from lofi import train as lofi_train
from lofi._types import ItemId
from lofi.data import (
    ConstraintSelection,
    Example,
    Item,
    SpanConstraint,
    SpanConstraints,
    Token,
    TokenSpan,
)
from lofi.data.synthetic.spec.corpus import SyntheticCorpusDecl, SyntheticCorpusLowerer
from lofi.data.synthetic.training import InferenceTrainingBatch
from lofi.deduce import SpanIndex
from lofi.judgment import InferenceText
from lofi.lexicon import ClampPolicy, ClampScope, UnsatisfiableClampError
from lofi.lf import InferenceLabel
from lofi.nn.codec import NeuralCodecView
from lofi.nn.loss._constraint import BatchLocalNegativeBracket
from lofi.nn.loss._errors import MissingOutputError
from lofi.nn.loss._ports import (
    LossComponentBindings,
    LossTerm,
)
from lofi.nn.parsers import ScoreChannels, TaggedInsideKernel
from lofi.registry import ComponentPlugin, Registry, UnknownTagError
from lofi.run import RandomStreams, SeedPolicy, plan_training, resolve_device_policy
from lofi.spec import (
    PACKAGED_CONFIG_ROOT,
    ComputeDtype,
    ConstantPatternSpec,
    GradScalerSpec,
    InlineModelLanguageSpec,
    LearnerInterpretationSpec,
    LossSection,
    LossSpec,
    ModelRole,
    ModelSection,
    PatternRuleSpec,
    RuleSetSpec,
    RunSpecCodec,
    WeightedLossSpec,
    config_search_path,
)
from lofi.spec._ports import Dependency
from lofi.spec.components.trainer import TrainerSpec
from lofi.spec.constraint_policy import PolicyDomain
from lofi.spec.constraint_sources import ClampSpec
from lofi.spec.grammar import (
    BudgetSpec,
    GrammarDecl,
    GrammarIso,
    InlineLexiconSpec,
    LexicalEntrySpec,
    RootGoalSpec,
    SignatureSpec,
)
from lofi.spec.groundings import PairwiseGroundingSpec
from lofi.spec.language import registered_language
from lofi.spec.losses.pinning import PinningLossSpec
from lofi.spec.pair_samplers.rule import RulePairSamplerSpec
from lofi.spec.parsers.tagged import TaggedParserSpec
from lofi.spec.provers import ProofBudgetSpec
from lofi.spec.scorers import ProverScorerSpec
from lofi.spec.spaces import SpaceSpec
from lofi.spec.trainers import (
    CheckpointPolicySpec,
    TermGenerationSpec,
    TermsTrainerSpec,
)
from lofi.spec.type_samplers.recursive import RecursiveTypeSamplerSpec
from lofi.terms import EncodingOverflowError, Normalization, Term, TermCodec
from lofi.train import (
    Draw,
    GradScaling,
    ParserFittingPlan,
    StepFittingPlan,
)
from lofi.train import _assembly as assembly_module
from lofi.train._assembly import (
    _build_grounding,
    _constraint_catalog,
    _constraint_readout_role,
    _learner_category_inventory,
    _learner_category_support,
    _learner_reachability_chart,
    _LearnerCategorySupport,
    _retained_fit_split,
    prepare_training,
    resolve_loss_section,
    wire_components,
)
from lofi.train._losses import LossAssembly, LossPlugin, MisnamedObjectiveError
from lofi.training_plugins import TrainerBuildContext, TrainerPlugin

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi.data import SupervisionArm
    from lofi.data.datasets import PairTraining
    from lofi.data.synthetic import InterpretationFunction, SyntheticDataset
    from lofi.data.synthetic.spec.inference import (
        SyntheticInferenceDataset,
        SyntheticInferenceDecl,
    )
    from lofi.data.synthetic.training import SyntheticTrainingBatch
    from lofi.nn.build import BuildContext
    from lofi.nn.loss._ports import PublishedOutputs
    from lofi.nn.loss._values import ObjectiveOutputs
    from lofi.nn.ports import SubwordPooler
    from lofi.spec import ComponentSpec, RunSpec
    from lofi.spec.grammar import GrammarAssembly
    from lofi.terms import Matching
    from lofi.train import PreparedTraining
    from lofi.train._pretraining import TermsStep


CONFIG = Path(__file__).resolve().parents[3] / "lofi/configs/run/synthetic-parser.yaml"
DISTANCE = CONFIG.with_name("synthetic-distance.yaml")
PACKAGED_CODEC = RunSpecCodec(search_path=(CONFIG.parent, CONFIG.parent.parent))
INFERENCE_PARTITIONS = (
    'data.datasets.train.dataset.retained={"dev":0.16666666666666666,'
    '"test":0.16666666666666666}'
)


def _prepared(run_spec: RunSpec, /) -> PreparedTraining:
    """Prepare one run under its declared deterministic device policy."""
    blueprint = plan_training(run_spec)
    streams = RandomStreams(SeedPolicy(23, samplers=blueprint.sampler_streams))
    streams.apply()

    return prepare_training(blueprint, streams, resolve_device_policy(run_spec))


class _DatasetWithHeldOutDefault:
    """Dataset declaration with multiple retained partitions."""

    def retained_splits(self) -> frozenset[str]:
        """Return the declared retained partitions."""
        return frozenset({"development", "test"})

    def held_out_split(self) -> str:
        """Return the partition selected by the dataset."""
        return "test"


def test_a_held_out_dataset_selects_the_fit_partition_without_an_override() -> None:
    run_spec = cast(
        "RunSpec",
        SimpleNamespace(
            data=SimpleNamespace(
                datasets={
                    "train": SimpleNamespace(dataset=_DatasetWithHeldOutDefault())
                }
            ),
            evaluators={},
        ),
    )

    assert _retained_fit_split(run_spec, "train") == "test"


def test_the_parser_plan_builds_the_declared_current_component_roles() -> None:
    run_spec = PACKAGED_CODEC(CONFIG)
    prepared = _prepared(run_spec)

    assert isinstance(prepared.plan, ParserFittingPlan)
    assert set(prepared.model.modules) == {
        "application",
        "controller",
        "identity",
        "parser",
        "subword_pooler",
        "terms",
        "token_encoder",
    }
    assert set(prepared.model.role_parameters()) == set(prepared.model.modules)


def test_grounded_preparation_publishes_prediction_for_new_inference_text() -> None:
    prepared = _prepared(
        composed(
            "inference-point.yaml",
            INFERENCE_PARTITIONS,
            "/evaluators.inference=inference",
        )
    )
    prediction = prepared.prediction
    plan = prepared.plan

    assert prediction is not None
    assert isinstance(plan, ParserFittingPlan)
    assert prediction.relation_labels == (
        InferenceLabel.ENTAILS,
        InferenceLabel.CONTRADICTS,
        InferenceLabel.NEUTRAL,
    )

    retained = {
        (
            " ".join(batch.words[premise]),
            " ".join(batch.words[hypothesis]),
        )
        for batches in (plan.batches, *plan.retained_batches.values())
        for batch in batches
        if isinstance(batch, InferenceTrainingBatch)
        for premise, hypothesis in zip(
            batch.premise_rows, batch.hypothesis_rows, strict=True
        )
    }
    unseen = ("john sees mary", "mary sleeps")
    assert unseen not in retained

    pairs = (InferenceText("bead-item-new", *unseen),)
    relations = prediction.predict_relations(pairs)
    responses = prediction.predict_responses(pairs)

    assert len(relations) == len(responses) == 1
    assert responses[0] == pytest.approx(relations[0])
    assert len(relations[0]) == len(prediction.relation_labels)
    assert len(responses[0]) == prediction.response_levels
    assert sum(relations[0]) == pytest.approx(1.0)
    assert sum(responses[0]) == pytest.approx(1.0)
    assert prediction.predict_relations(()) == ()
    assert prediction.predict_responses(()) == ()


class _LearnerBuildContextObservedError(Exception):
    """Stop after grounded preparation constructs its run-wide context."""


class _GoalRefusingCodec:
    """Delegate encoding except for one required goal class."""

    __slots__ = ("_codec", "_goal", "_matching")

    def __init__(
        self,
        codec: TermCodec,
        goal: Term,
        matching: Matching[Term],
        /,
    ) -> None:
        self._codec = codec
        self._goal = goal
        self._matching = matching

    def encode(self, category: Term, /) -> object:
        """Encode a non-goal category and refuse every goal representative."""
        if self._matching.equivalent(category, self._goal):
            raise EncodingOverflowError(layout="test-goal", capacity=1, size=2)

        return self._codec.encode(category)


def test_a_codec_overflowing_the_grammar_goal_fails_at_the_inventory_boundary() -> None:
    run_spec = PACKAGED_CODEC(CONFIG)
    grammar = plan_training(run_spec).learner.grammar
    language = grammar.language
    codec = language.codec.at(language.category_sort)
    goal = grammar.grammar.targets.categories[0]
    refusing = cast(
        "NeuralCodecView[TermCodec]",
        SimpleNamespace(
            codec=_GoalRefusingCodec(codec, goal, grammar.grammar.matching)
        ),
    )

    with pytest.raises(EncodingOverflowError):
        _learner_category_inventory(grammar, refusing)


def test_parser_context_uses_the_learner_closure_without_source_or_gold_leakage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_spec = PACKAGED_CODEC(CONFIG)
    declaration = cast("SyntheticCorpusDecl", run_spec.data.datasets["train"].dataset)
    learner = InlineModelLanguageSpec(
        language=declaration.language.with_(name="assembly-test-learner"),
        grammar=cast("GrammarDecl", declaration.grammar),
    )
    run_spec = run_spec.with_(model=run_spec.model.with_(language=learner))
    blueprint = plan_training(run_spec)
    dataset = cast("SyntheticDataset", blueprint.lower_dataset("train"))
    grammar = blueprint.learner.grammar
    language = grammar.language
    expected = _learner_category_inventory(
        grammar,
        NeuralCodecView(language.codec.at(language.category_sort)),
    )
    observed: list[BuildContext] = []

    def observe(
        model: object,
        run: BuildContext,
        pretrained_resources: object,
        /,
    ) -> Never:
        del model, pretrained_resources
        observed.append(run)
        raise _LearnerBuildContextObservedError

    monkeypatch.setattr(assembly_module, "build_space_encoders", observe)

    streams = RandomStreams(SeedPolicy(23, samplers=blueprint.sampler_streams))
    streams.apply()
    with pytest.raises(_LearnerBuildContextObservedError):
        prepare_training(blueprint, streams, resolve_device_policy(run_spec))

    assert len(observed) == 1
    assert observed[0].categories == expected
    assert observed[0].language is blueprint.learner.grammar.language
    assert observed[0].language is not dataset.source_grammar.language


def test_grounded_preparation_builds_only_in_the_declared_learner_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_spec = composed(
        "inference-point.yaml",
        INFERENCE_PARTITIONS,
        "/evaluators.inference=inference",
    )
    data = run_spec.data
    section = data.datasets["train"]
    declaration = cast("SyntheticInferenceDecl", section.dataset)
    learner = InlineModelLanguageSpec(
        language=declaration.source.language.with_(name="assembly-test-learner"),
        grammar=declaration.grammar,
        interpretation=LearnerInterpretationSpec.model_validate(
            loads(declaration.source.interpretation.model_dump_json())
        ),
    )
    run_spec = run_spec.with_(model=run_spec.model.with_(language=learner))
    blueprint = plan_training(run_spec)
    dataset = cast("SyntheticInferenceDataset", blueprint.lower_dataset("train"))
    grammar = blueprint.learner.grammar
    language = grammar.language
    categories = _learner_category_inventory(
        grammar,
        NeuralCodecView(language.codec.at(language.category_sort)),
    )
    observed: list[BuildContext] = []

    def observe(
        model: object,
        run: BuildContext,
        pretrained_resources: object,
        /,
    ) -> Never:
        del model, pretrained_resources
        observed.append(run)
        raise _LearnerBuildContextObservedError

    monkeypatch.setattr(assembly_module, "build_space_encoders", observe)

    streams = RandomStreams(SeedPolicy(23, samplers=blueprint.sampler_streams))
    streams.apply()
    with pytest.raises(_LearnerBuildContextObservedError):
        prepare_training(blueprint, streams, resolve_device_policy(run_spec))

    assert len(observed) == 1
    context = observed[0]
    assert context.language is blueprint.learner.grammar.language
    assert context.meaning is blueprint.learner.interpretation
    assert context.language is not dataset.source_grammar.language
    assert context.categories == categories
    assert tuple(
        blueprint.learner.grammar.syntax.render(
            blueprint.learner.grammar.store, category
        )
        for category in context.categories
    )
    assert not hasattr(dataset, "grammar")
    assert not hasattr(dataset, "interpretation")


class _GroundingConstructionObservedError(Exception):
    """Stop after the nested scorer receives its assembly context."""


def test_grounding_assembly_wires_the_interpretation_normalization_by_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    normalization = Normalization(
        fuel=13,
        order="weak-head",
        eta="contract",
        associative_commutative=(),
    )
    service = object()
    observed: list[tuple[bool, bool]] = []

    def construct(component: object, context: BuildContext, /) -> object:
        if isinstance(component, ProverScorerSpec):
            observed.append(
                (
                    context.producer("normalization") is normalization,
                    context.producer("service") is service,
                )
            )
            return object()

        raise _GroundingConstructionObservedError

    monkeypatch.setattr(assembly_module, "_construct", construct)
    declaration = PairwiseGroundingSpec(
        premise="premise-forms",
        hypothesis="hypothesis-forms",
        scorer=ProverScorerSpec(budget=ProofBudgetSpec(calls_per_step=4)),
    )
    pairs = cast("PairTraining", SimpleNamespace(labeling=object()))
    interpretation = SimpleNamespace(
        normalization=normalization,
        service=service,
    )

    with pytest.raises(_GroundingConstructionObservedError):
        _build_grounding(
            declaration,
            plugin_context(),
            pairs,
            cast("InterpretationFunction", interpretation),
            {},
            cast("SubwordPooler", object()),
        )

    assert observed == [(True, True)]


def test_the_declared_schedules_span_each_fitting_horizon() -> None:
    parser = _prepared(PACKAGED_CODEC(CONFIG))
    distance = _prepared(RunSpecCodec()(DISTANCE))
    parser_schedule = parser.optimization.state().scheduler
    distance_schedule = distance.optimization.state().scheduler

    assert isinstance(parser.plan, ParserFittingPlan)
    assert isinstance(distance.plan, StepFittingPlan)
    assert parser_schedule is not None
    assert distance_schedule is not None
    assert parser_schedule["T_max"] == parser.plan.epochs * parser.plan.steps_per_epoch
    assert distance_schedule["T_max"] == distance.plan.steps


def _float16(run_spec: RunSpec, scaler: GradScalerSpec | None, /) -> RunSpec:
    """Return the run computing in float16 under the given gradient scaler."""
    return run_spec.with_(
        device_policy=run_spec.device_policy.with_(
            compute_dtype=ComputeDtype.FLOAT16, scaler=scaler
        )
    )


@pytest.mark.parametrize("trainer", ["parser", "distance", "terms"])
def test_every_trainer_steps_under_the_declared_gradient_scaler(trainer: str) -> None:
    runs = {
        "parser": lambda: PACKAGED_CODEC(CONFIG),
        "distance": lambda: RunSpecCodec()(DISTANCE),
        "terms": _terms_run,
    }
    declared = GradScalerSpec(
        init_scale=1024.0, growth_factor=3.0, backoff_factor=0.25, growth_interval=7
    )
    scaled = _prepared(_float16(runs[trainer](), declared)).optimization
    unscaled = _prepared(_float16(runs[trainer](), None)).optimization

    assert scaled.options.scaling == GradScaling(
        init_scale=1024.0, growth_factor=3.0, backoff_factor=0.25, growth_interval=7
    )
    assert scaled.scales_gradients
    assert unscaled.options.scaling is None
    assert not unscaled.scales_gradients


def _terms_run() -> RunSpec:
    """Return the packaged learner language under generated-term pretraining."""
    parser = PACKAGED_CODEC(CONFIG)
    categories = parser.model.spaces["categories"]
    model = ModelSection(
        language=parser.model.language,
        encoders={"terms": parser.model.encoders["terms"]},
        decoders={
            "identity": parser.model.decoders["identity"],
            "application": parser.model.decoders["application"],
        },
        spaces={
            "categories": SpaceSpec(
                sort=categories.sort,
                encoder=categories.encoder,
                identity=categories.identity,
                combinators=categories.combinators,
            )
        },
    )

    return parser.with_(
        model=model,
        loss=None,
        data=parser.data.with_(
            datasets={"train": parser.data.datasets["train"].with_(featurizer=None)}
        ),
        trainer=TermsTrainerSpec(
            dataset="train",
            spaces=("categories",),
            steps=2,
            batch_size=1,
            generation=TermGenerationSpec(max_depth=3),
            checkpoints=CheckpointPolicySpec(interval=1, keep=1),
        ),
        type_sampler=RecursiveTypeSamplerSpec(max_size=5),
        pair_sampler=RulePairSamplerSpec(),
        probes={},
        evaluators={},
    )


def test_terms_training_builds_one_joint_step_for_each_declared_space() -> None:
    run_spec = _terms_run()
    blueprint = plan_training(run_spec)
    streams = RandomStreams(SeedPolicy(23, samplers=blueprint.sampler_streams))
    streams.apply()
    prepared = prepare_training(blueprint, streams, resolve_device_policy(run_spec))
    plan = cast("StepFittingPlan", prepared.plan)

    assert isinstance(plan, StepFittingPlan)
    assert len(plan.trainers) == 1
    assert cast("TermsStep", plan.trainers[0]).roles == (
        "terms",
        "identity",
        "application",
    )
    assert plan.trainer_at(0) is plan.trainer_at(1)
    assert (
        blueprint.learner.grammar.language.codec.at("Cat") is prepared.model.codec.codec
    )


def test_a_declared_terms_run_executes_its_generated_joint_objectives() -> None:
    prepared = _prepared(_terms_run())
    plan = cast("StepFittingPlan", prepared.plan)
    outcome = plan.trainer_at(0).run(Draw(29, 1, step=0, horizon=plan.steps))

    assert outcome.outputs == (
        lofi_train.Trained("terms"),
        lofi_train.Trained("identity"),
        lofi_train.Trained("application"),
    )
    assert outcome.result.items == 1
    assert outcome.result.step == 1
    assert outcome.result.metrics["optimizer/loss_finite"] == 1.0
    assert outcome.result.metrics["loss/categories/autoencoding"] > 0.0
    assert outcome.result.metrics["loss/categories/combinators"] > 0.0


class ExternalTrainer(TrainerSpec):
    """Trainer declaration whose installed builder is observed directly."""

    kind: Literal["external-assembly"] = "external-assembly"


class ExternalBuilderReachedError(Exception):
    """Signal that trainer preparation dispatched through the installed plugin."""


def _plan_external(
    declaration: ExternalTrainer, run_spec: RunSpec, /
) -> tuple[str, ...]:
    del declaration, run_spec
    return ()


def _present_external(
    declaration: ExternalTrainer, run_spec: RunSpec, /
) -> Mapping[str, frozenset[SupervisionArm]]:
    del declaration, run_spec
    return {}


def _build_external(
    declaration: ExternalTrainer, context: TrainerBuildContext, /
) -> Never:
    del declaration, context
    raise ExternalBuilderReachedError


def _price_external(declaration: ExternalTrainer, /) -> None:
    del declaration


EXTERNAL = TrainerPlugin(
    spec=ExternalTrainer,
    plan=_plan_external,
    present=_present_external,
    build=_build_external,
    price=_price_external,
)


def test_an_installed_trainer_builds_without_a_central_identity_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = Registry.get

    def installed[MemberT](registry: Registry[MemberT], tag: str) -> MemberT:
        if registry.family == "trainer" and tag == "external-assembly":
            return cast("MemberT", EXTERNAL)

        return original(registry, tag)

    monkeypatch.setattr(Registry, "get", installed)
    run_spec = PACKAGED_CODEC(CONFIG).with_(trainer=ExternalTrainer())

    with pytest.raises(ExternalBuilderReachedError):
        _prepared(run_spec)


class ScaleSettings(NamedTuple):
    """Resolved scalar held by one fixture loss."""

    scale: float


class ScaledLossSpec(LossSpec):
    """Fixture declaration exercising the general loss plugin boundary."""

    kind: Literal["scaled"] = "scaled"
    scale: float = 1.0

    def settings(self) -> ScaleSettings:
        """Return the declared scalar."""
        return ScaleSettings(self.scale)


class Batch(NamedTuple):
    """Minimal batch carried by the fixture outputs."""

    size: int


class Outputs:
    """Minimal published-output bundle read only through its batch."""

    def __init__(self, size: int) -> None:
        self.batch = Batch(size)


class Scaled:
    """Objective returning its declared scale once per fixture item."""

    def __init__(self, name: str, scale: float) -> None:
        self._name = name
        self._scale = scale

    @property
    def name(self) -> str:
        """Return the declaration-selected term name."""
        return self._name

    @property
    def requires(self) -> frozenset[str]:
        """Return the one published field this objective reads."""
        return frozenset({"batch"})

    def evaluate(self, outputs: PublishedOutputs[Batch], /) -> LossTerm:
        """Return a differentiable scaled item count."""
        count = outputs.batch.size
        return LossTerm(torch.tensor(self._scale * count), count)


def _scaled_plugin(*, honor_name: bool = True) -> LossPlugin[Batch]:
    """Return the fixture plugin, optionally violating declared naming."""

    def build(settings: ScaleSettings, context: LossComponentBindings, /) -> Scaled:
        name = context.name if honor_name else "scaled"
        return Scaled(name, settings.scale)

    return cast(
        "LossPlugin[Batch]",
        ComponentPlugin(spec=ScaledLossSpec, build=build),
    )


def _resolve(
    section: LossSection,
    registry: Registry[LossPlugin[Batch]],
    /,
    *,
    published: frozenset[str] | None = None,
    unavailable_outputs: Mapping[str, str] | None = None,
) -> LossAssembly[Batch]:
    """Resolve fixture losses through the complete current binding surface."""
    return resolve_loss_section(
        section,
        registry,
        published=published,
        unavailable_outputs=unavailable_outputs,
        readouts={},
        spaces={},
        readout_spaces={},
        relations={},
        span_decoders={},
        proof_reward=None,
    )


def test_loss_terms_receive_fresh_contexts_and_report_declared_names() -> None:
    section = LossSection(
        terms=(
            WeightedLossSpec(objective=ScaledLossSpec(scale=2.0), weight=0.5),
            WeightedLossSpec(
                objective=ScaledLossSpec(scale=3.0),
                weight=0.25,
                name="scaled-wide",
            ),
        )
    )
    registry = installed("loss-assembly-fixture", {"scaled": _scaled_plugin()})
    assembly = _resolve(section, registry)
    result = assembly.evaluate(cast("ObjectiveOutputs[Batch]", Outputs(2)), step=0)

    assert assembly.names == ("scaled", "scaled-wide")
    assert assembly.weights == (0.5, 0.25)
    assert result.total.item() == pytest.approx(3.5)
    assert result.metrics["loss/scaled"] == 4.0
    assert result.metrics["loss/scaled-wide"] == 6.0


def test_loss_resolution_refuses_unknown_misnamed_and_unpublished_terms() -> None:
    section = LossSection(
        terms=(WeightedLossSpec(objective=ScaledLossSpec(), weight=1.0),)
    )
    empty = Registry[LossPlugin[Batch]]("loss-empty-fixture")

    with pytest.raises(UnknownTagError):
        _resolve(section, empty)

    named = LossSection(
        terms=(
            WeightedLossSpec(
                objective=ScaledLossSpec(), weight=1.0, name="declared-name"
            ),
        )
    )

    misnamed = installed(
        "loss-misnamed-fixture", {"scaled": _scaled_plugin(honor_name=False)}
    )

    with pytest.raises(MisnamedObjectiveError):
        _resolve(named, misnamed)

    current = installed("loss-published-fixture", {"scaled": _scaled_plugin()})

    with pytest.raises(MissingOutputError):
        _resolve(section, current, published=frozenset({"chart"}))


def test_loss_resolution_reports_a_declared_configuration_refusal() -> None:
    section = LossSection(
        terms=(WeightedLossSpec(objective=ScaledLossSpec(), weight=1.0),)
    )

    registry = installed("loss-refusal-fixture", {"scaled": _scaled_plugin()})

    with pytest.raises(ValueError, match="outside_weights"):
        _resolve(
            section,
            registry,
            published=frozenset({"chart"}),
            unavailable_outputs={"batch": "model.parser.outside_weights is 'reuse'"},
        )


def _drawn(run_spec: RunSpec) -> SyntheticDataset:
    """Lower the synthetic dataset declared by one run."""
    lowerer = SyntheticCorpusLowerer()
    return lowerer.lower(
        lowerer.check(
            cast("SyntheticCorpusDecl", run_spec.data.datasets["train"].dataset)
        ),
        run_spec.parse.build().policy,
    )


def _category_support(
    dataset: SyntheticDataset,
    learner: GrammarAssembly,
    run_spec: RunSpec,
    /,
) -> _LearnerCategorySupport:
    """Return finite learner support for one test run's unlabeled items."""
    language = learner.language
    codec = NeuralCodecView(language.codec.at(language.category_sort))

    return _learner_category_support(
        tuple(example.item for example in dataset.corpus.observed()),
        learner,
        _learner_category_inventory(learner, codec),
        run_spec.model.parser,
    )


def test_an_untagged_parser_offers_the_full_inventory_at_every_span() -> None:
    run = PACKAGED_CODEC(CONFIG)
    blueprint = plan_training(run)
    dataset = cast("SyntheticDataset", blueprint.lower_dataset("train"))
    grammar = blueprint.learner.grammar
    support = _category_support(dataset, grammar, run)

    for length, chart in support.charts.items():
        for width in range(1, length + 1):
            for start in range(length - width + 1):
                offered = tuple(
                    category
                    for category, _ in chart.categories_at((start, start + width))
                )
                assert offered == support.categories


def _overflows(codec: NeuralCodecView[TermCodec], category: Term, /) -> bool:
    """Return whether one category lies outside a finite codec."""
    try:
        codec.codec.encode(category)
    except EncodingOverflowError:
        return True

    return False


def test_constraint_selection_is_present_exactly_when_declared() -> None:
    plain = PACKAGED_CODEC(CONFIG)
    constrained = RunSpecCodec(search_path=config_search_path(CONFIG.parents[2]))(
        CONFIG.with_name("synthetic-pinning-lexical.yaml")
    )

    assert lofi_train.constraint_source(plain, "train", _drawn(plain)) is None
    assert lofi_train.select_constraints(plain, "train", _drawn(plain)) is None
    assert (
        lofi_train.constraint_source(constrained, "train", _drawn(constrained))
        is not None
    )
    assert (
        lofi_train.select_constraints(constrained, "train", _drawn(constrained))
        is not None
    )


def _selection_with_one_of_five_labels_flipped() -> tuple[
    ConstraintSelection[str, str], SyntheticDataset, RunSpec
]:
    """Return five selected items with one token label made unsatisfiable."""
    loaded = RunSpecCodec(search_path=config_search_path(CONFIG.parents[2]))(
        CONFIG.with_name("synthetic-pinning-lexical.yaml")
    )
    dataset_decl = cast(
        "SyntheticCorpusDecl", loaded.data.datasets["train"].dataset
    ).with_(size=5)
    run = loaded.with_(
        data=loaded.data.with_(
            datasets={
                "train": loaded.data.datasets["train"].with_(dataset=dataset_decl)
            }
        )
    )
    dataset = _drawn(run)
    selection = lofi_train.select_constraints(run, "train", dataset)

    assert selection is not None
    assert len(selection.examples) == 5

    example = selection.examples[0]
    original = example.supervision.constraints[0]
    grammar = dataset.source_grammar
    language = grammar.language
    matching = grammar.grammar.matching
    inventory = _learner_category_inventory(
        grammar,
        NeuralCodecView(language.codec.at(language.category_sort)),
    )
    impossible = next(
        label
        for label in ("S/S", "S\\S", "NP\\NP")
        if not any(
            matching.equivalent(grammar.syntax.parse(grammar.store, label), held)
            for held in inventory
        )
    )
    flipped = Example(
        example.item,
        SpanConstraints(
            (
                SpanConstraint(
                    original.span,
                    frozenset({impossible}),
                    original.scope,
                ),
            ),
            len(example.item.tokens),
        ),
    )

    return (
        ConstraintSelection(
            (flipped, *selection.examples[1:]),
            selection.report,
            selection.budget,
        ),
        dataset,
        run,
    )


def _constituent_run(*overrides: str) -> RunSpec:
    """Return the constituent-pinning run over five deterministic items."""
    loaded = RunSpecCodec(search_path=config_search_path(CONFIG.parents[2]))(
        CONFIG.with_name("synthetic-pinning-constituent.yaml"), overrides=overrides
    )
    declaration = cast(
        "SyntheticCorpusDecl", loaded.data.datasets["train"].dataset
    ).with_(size=5)

    return loaded.with_(
        model=loaded.model.with_(parser=TaggedParserSpec(width=4, unary_depth=1)),
        data=loaded.data.with_(
            datasets={"train": loaded.data.datasets["train"].with_(dataset=declaration)}
        ),
    )


def _constituent_selection() -> tuple[
    ConstraintSelection[str, str], SyntheticDataset, RunSpec
]:
    """Return gold-derivation constraints over five deterministic items."""
    run = _constituent_run()
    dataset = _drawn(run)
    selection = lofi_train.select_constraints(run, "train", dataset)

    assert selection is not None

    return selection, dataset, run


NEGATIVE_BRACKETS = (
    "data.datasets.train.constraint_source.clamp.require_constituent=false",
    "data.datasets.train.constraint_source.clamp.on_unsatisfiable.kind=drop_clamp",
)


def _negative_bracket_run(*overrides: str) -> RunSpec:
    """Return configured hard negative-bracket supervision without a readout."""
    run = _constituent_run(*overrides, *NEGATIVE_BRACKETS)
    loss = cast("LossSection", run.loss)

    return run.with_(
        loss=loss.with_(
            terms=tuple(
                term.with_(objective=PinningLossSpec())
                if isinstance(term.objective, PinningLossSpec)
                else term
                for term in loss.terms
            )
        ),
    )


def _one_span_selection(
    selection: ConstraintSelection[str, str], span: tuple[int, int], /
) -> ConstraintSelection[str, str]:
    """Return the first item with only its constraint at one exact span."""
    example = selection.examples[0]
    selected = next(
        constraint
        for constraint in example.supervision.constraints
        if (constraint.span.start, constraint.span.end) == span
    )

    return ConstraintSelection(
        (
            Example(
                example.item,
                SpanConstraints((selected,), len(example.item.tokens)),
            ),
        ),
        selection.report,
        selection.budget,
    )


def test_negative_brackets_have_no_target_or_readout() -> None:
    selection, dataset, run = _constituent_selection()
    selected = _one_span_selection(selection, (0, 2))
    targets, batcher, report = _constraint_catalog(
        selected,
        dataset.source_grammar,
        _category_support(dataset, dataset.source_grammar, run),
        domain=PolicyDomain.SYNTACTIC,
        policy=ClampSpec(require_constituent=False).settings(),
    )
    localized = batcher.for_items((selected.examples[0].item.item_id,))[0]

    assert not targets
    assert localized
    assert all(isinstance(held, BatchLocalNegativeBracket) for held in localized)
    assert all(not hasattr(held, "targets") for held in localized)
    assert all(not hasattr(held, "readout") for held in localized)
    assert report.dropped_clamps == 0
    assert report.skipped_items == 0


@pytest.mark.parametrize("span", [(0, 1), (0, 3)])
def test_a_guaranteed_negative_bracket_is_rejected_before_training(
    span: tuple[int, int],
) -> None:
    selection, dataset, run = _constituent_selection()
    selected = _one_span_selection(selection, span)

    with pytest.raises(UnsatisfiableClampError) as raised:
        _constraint_catalog(
            selected,
            dataset.source_grammar,
            _category_support(dataset, dataset.source_grammar, run),
            domain=PolicyDomain.SYNTACTIC,
            policy=ClampSpec(require_constituent=False).settings(),
        )

    assert raised.value.clamp.span == span
    assert not raised.value.clamp.require_constituent


@pytest.mark.parametrize(
    ("action", "dropped", "skipped"),
    [("drop_clamp", 1, 0), ("skip_item", 0, 1)],
)
def test_a_guaranteed_negative_bracket_uses_the_declared_refusal_policy(
    action: str, dropped: int, skipped: int
) -> None:
    selection, dataset, run = _constituent_selection()
    selected = _one_span_selection(selection, (0, 3))
    targets, batcher, report = _constraint_catalog(
        selected,
        dataset.source_grammar,
        _category_support(dataset, dataset.source_grammar, run),
        domain=PolicyDomain.SYNTACTIC,
        policy=ClampSpec.model_validate(
            {
                "require_constituent": False,
                "on_unsatisfiable": {"kind": action},
            }
        ).settings(),
    )

    assert not targets
    assert not batcher.for_items((selected.examples[0].item.item_id,))[0]
    assert report.dropped_clamps == dropped
    assert report.skipped_items == skipped


def test_an_occurrence_scoped_negative_bracket_expands_to_guaranteed_leaves() -> None:
    selection, dataset, run = _constituent_selection()
    selected = _one_span_selection(selection, (0, 2))
    base = ClampSpec(require_constituent=False).settings()
    occurrence = ClampPolicy(
        mode=base.mode,
        penalty=base.penalty,
        scope=ClampScope.OCCURRENCE,
        matching=base.matching,
        meaning_matching=base.meaning_matching,
        require_constituent=False,
        negate=base.negate,
        on_unsatisfiable=base.on_unsatisfiable,
    )

    with pytest.raises(UnsatisfiableClampError) as raised:
        _constraint_catalog(
            selected,
            dataset.source_grammar,
            _category_support(dataset, dataset.source_grammar, run),
            domain=PolicyDomain.SYNTACTIC,
            policy=occurrence,
        )

    assert raised.value.clamp.span == (0, 1)
    assert raised.value.position == 0


def test_a_gold_derivation_source_is_refused_on_the_negative_bracket_route() -> None:
    run = _negative_bracket_run()

    with pytest.raises(ValueError, match="cannot serve the 'negative-bracket' route"):
        _prepared(run)


def test_a_configured_negative_bracket_run_uses_no_constraint_readout() -> None:
    run = _negative_bracket_run("/data.datasets.train.constraint_source=declared-pins")

    assert _constraint_readout_role(run) is None

    prepared = _prepared(run)
    plan = cast("ParserFittingPlan", prepared.plan)
    batches = cast("tuple[SyntheticTrainingBatch[object], ...]", plan.batches)
    constraints = tuple(
        constraint for batch in batches for constraint in batch.constraints
    )

    assert constraints
    assert all(isinstance(held, BatchLocalNegativeBracket) for held in constraints)


def _one_item_selection(
    selection: ConstraintSelection[str, str], /
) -> ConstraintSelection[str, str]:
    """Return the first selected item under its existing selection metadata."""
    return ConstraintSelection(
        (selection.examples[0],), selection.report, selection.budget
    )


def _span_unreachable_selection(
    selection: ConstraintSelection[str, str],
    dataset: SyntheticDataset,
    run: RunSpec,
    /,
) -> ConstraintSelection[str, str]:
    """Relabel one derived span with a neural-inventory category absent there."""
    example = selection.examples[0]
    constraints = list(example.supervision.constraints)
    victim = next(
        constraint
        for constraint in constraints
        if constraint.span.start == 0 and constraint.span.end == 2
    )
    grammar = dataset.source_grammar
    support = _category_support(dataset, grammar, run)
    offered = tuple(
        category
        for category, _ in support.charts[len(example.item.tokens)].categories_at(
            (victim.span.start, victim.span.end)
        )
    )
    replacement = next(
        category
        for category in support.categories
        if not any(
            grammar.grammar.matching.equivalent(category, candidate)
            for candidate in offered
        )
    )
    position = constraints.index(victim)
    constraints[position] = SpanConstraint(
        victim.span,
        frozenset({grammar.syntax.render(grammar.store, replacement)}),
        victim.scope,
    )
    flipped = Example(
        example.item,
        SpanConstraints(tuple(constraints), len(example.item.tokens)),
    )

    return ConstraintSelection((flipped,), selection.report, selection.budget)


def test_reachable_multitoken_constituent_is_retained() -> None:
    selection, dataset, run = _constituent_selection()
    selected = _one_item_selection(selection)
    support = _category_support(dataset, dataset.source_grammar, run)
    grammar = dataset.source_grammar
    internal = next(
        constraint
        for constraint in selected.examples[0].supervision.constraints
        if constraint.span.start == 0 and constraint.span.end == 2
    )
    category = grammar.syntax.parse(grammar.store, next(iter(internal.permitted)))
    matching = grammar.grammar.matching

    targets, batcher, report = _constraint_catalog(
        selected,
        grammar,
        support,
        domain=PolicyDomain.SYNTACTIC,
        policy=ClampSpec().settings(),
    )
    localized = batcher.for_items((selected.examples[0].item.item_id,))[0]

    assert targets
    assert not any(
        matching.equivalent(category, value)
        for value in grammar.grammar.targets.categories
    )
    assert any(matching.equivalent(category, value) for value in support.categories)
    assert report.dropped_clamps == 0
    assert report.skipped_items == 0
    assert any((0, 2) in constraint.spans for constraint in localized)


def test_nl_diamond_uses_finite_learner_support_and_refuses_codec_overflow() -> None:
    run = composed("nl-diamond-pack.yaml")
    blueprint = plan_training(run)
    dataset = cast("SyntheticDataset", blueprint.lower_dataset("train"))
    grammar = blueprint.learner.grammar
    support = _category_support(dataset, grammar, run)
    codec = NeuralCodecView(grammar.language.codec.at(grammar.language.category_sort))
    categories = _learner_category_inventory(grammar, codec)
    overflow = next(
        category
        for category in grammar.grammar.closure.reached
        if _overflows(codec, category)
    )
    selection, _, _ = _constituent_selection()
    reference = selection.examples[0].supervision.constraints[0]
    item = dataset.corpus.observed()[0].item
    overflow_label = grammar.syntax.render(grammar.store, overflow)
    local, _ = support.charts[len(item.tokens)].categories_at(
        (reference.span.start, reference.span.end)
    )[0]
    local_label = grammar.syntax.render(grammar.store, local)

    def selected(permitted: frozenset[str]) -> ConstraintSelection[str, str]:
        return ConstraintSelection(
            (
                Example(
                    item,
                    SpanConstraints(
                        (
                            SpanConstraint(
                                reference.span,
                                permitted,
                                reference.scope,
                            ),
                        ),
                        len(item.tokens),
                    ),
                ),
            ),
            selection.report,
            selection.budget,
        )

    refused = selected(frozenset({overflow_label}))

    assert support.categories == categories
    assert len(grammar.grammar.closure.reached) == 256
    assert len(support.categories) < len(grammar.grammar.closure.reached)
    assert any(category.op.name == "dot_linear" for category in support.categories)
    assert all(not _overflows(codec, category) for category in support.categories)
    assert all(
        any(
            grammar.grammar.matching.equivalent(category, admitted)
            for admitted in support.categories
        )
        and not grammar.grammar.matching.equivalent(category, overflow)
        for length, chart in support.charts.items()
        for width in range(1, length + 1)
        for start in range(length - width + 1)
        for category, _ in chart.categories_at((start, start + width))
    )
    assert any(
        len(chart.categories_at((start, start + width))) < len(categories)
        for length, chart in support.charts.items()
        for width in range(2, length + 1)
        for start in range(length - width + 1)
    )
    with pytest.raises(UnsatisfiableClampError):
        _constraint_catalog(
            refused,
            grammar,
            support,
            domain=PolicyDomain.SYNTACTIC,
            policy=ClampSpec().settings(),
        )

    targets, _, report = _constraint_catalog(
        selected(frozenset({local_label, overflow_label})),
        grammar,
        support,
        domain=PolicyDomain.SYNTACTIC,
        policy=ClampSpec().settings(),
    )

    assert targets
    assert report.dropped_clamps == 0
    assert report.skipped_items == 0
    assert all(
        not grammar.grammar.matching.equivalent(overflow, admitted)
        for target in targets.values()
        for admitted in target.categories
    )

    excluded, _, negated_report = _constraint_catalog(
        refused,
        grammar,
        support,
        domain=PolicyDomain.SYNTACTIC,
        policy=ClampSpec(negate=True).settings(),
    )

    assert not excluded
    assert negated_report.dropped_clamps == 0
    assert negated_report.skipped_items == 0


def test_globally_available_but_span_unreachable_category_errors() -> None:
    selection, dataset, run = _constituent_selection()
    selected = _span_unreachable_selection(selection, dataset, run)
    grammar = dataset.source_grammar
    label = next(
        iter(
            next(
                constraint
                for constraint in selected.examples[0].supervision.constraints
                if constraint.span.start == 0 and constraint.span.end == 2
            ).permitted
        )
    )
    category = grammar.syntax.parse(grammar.store, label)

    assert any(
        grammar.grammar.matching.equivalent(category, reached)
        for reached in grammar.grammar.closure.reached
    )
    with pytest.raises(UnsatisfiableClampError):
        _constraint_catalog(
            selected,
            grammar,
            _category_support(dataset, grammar, run),
            domain=PolicyDomain.SYNTACTIC,
            policy=ClampSpec().settings(),
        )


class _UniversalMatching:
    """Treat every test category as one class."""

    __slots__ = ()

    @property
    def name(self) -> str:
        """Return the test congruence name."""
        return "universal-test"

    @property
    def has_key(self) -> bool:
        """Return that the category itself can stand for its class."""
        return True

    def key(self, category: Term, /) -> Term:
        """Return the supplied category."""
        return category

    def equivalent(self, left: Term, right: Term, /) -> bool:
        """Return that every pair belongs to the same class."""
        del left, right

        return True


def test_constraint_support_intersection_uses_the_policy_matching_override() -> None:
    selection, dataset, run = _constituent_selection()
    selected = _span_unreachable_selection(selection, dataset, run)
    support = _category_support(dataset, dataset.source_grammar, run)
    base = ClampSpec().settings()
    override = _UniversalMatching()
    policy = ClampPolicy(
        mode=base.mode,
        penalty=base.penalty,
        scope=base.scope,
        matching=override,
        meaning_matching=base.meaning_matching,
        require_constituent=base.require_constituent,
        negate=base.negate,
        on_unsatisfiable=base.on_unsatisfiable,
    )
    targets, _, report = _constraint_catalog(
        selected,
        dataset.source_grammar,
        support,
        domain=PolicyDomain.SYNTACTIC,
        policy=policy,
    )

    assert targets
    assert all(target.matching is override for target in targets.values())
    assert report.dropped_clamps == 0
    assert report.skipped_items == 0


@pytest.mark.parametrize(
    ("action", "dropped", "skipped"),
    [("drop_clamp", 1, 0), ("skip_item", 0, 1)],
)
def test_span_unreachable_category_uses_the_declared_refusal_policy(
    action: str, dropped: int, skipped: int
) -> None:
    selection, dataset, run = _constituent_selection()
    targets, _, report = _constraint_catalog(
        _span_unreachable_selection(selection, dataset, run),
        dataset.source_grammar,
        _category_support(dataset, dataset.source_grammar, run),
        domain=PolicyDomain.SYNTACTIC,
        policy=ClampSpec.model_validate(
            {"on_unsatisfiable": {"kind": action}}
        ).settings(),
    )

    assert bool(targets) is (action == "drop_clamp")
    assert report.dropped_clamps == dropped
    assert report.skipped_items == skipped


def test_constraint_reachability_compiles_each_selected_length_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, dataset, run = _constituent_selection()
    selected = _one_item_selection(selection)
    chart_spy = Mock(wraps=_learner_reachability_chart)
    monkeypatch.setattr(assembly_module, "_learner_reachability_chart", chart_spy)
    language = dataset.source_grammar.language
    codec = NeuralCodecView(language.codec.at(language.category_sort))
    categories = _learner_category_inventory(dataset.source_grammar, codec)
    item = selected.examples[0].item
    support = _learner_category_support(
        (item, item),
        dataset.source_grammar,
        categories,
        run.model.parser,
    )
    _constraint_catalog(
        selected,
        dataset.source_grammar,
        support,
        domain=PolicyDomain.SYNTACTIC,
        policy=ClampSpec().settings(),
    )

    assert tuple(support.charts) == (len(item.tokens),)
    assert chart_spy.call_count == 1


def test_learner_inventory_is_independent_of_served_sentence_lengths() -> None:
    run = PACKAGED_CODEC(CONFIG)
    blueprint = plan_training(run)
    dataset = cast("SyntheticDataset", blueprint.lower_dataset("train"))
    grammar = blueprint.learner.grammar
    language = grammar.language
    categories = _learner_category_inventory(
        grammar,
        NeuralCodecView(language.codec.at(language.category_sort)),
    )
    observed = tuple(example.item for example in dataset.corpus.observed())
    one = _learner_category_support(
        observed[:1],
        grammar,
        categories,
        run.model.parser,
    )
    all_items = _learner_category_support(
        observed,
        grammar,
        categories,
        run.model.parser,
    )

    assert one.categories == all_items.categories == categories


def test_drop_clamp_completes_and_reports_one_drop_under_twenty_percent_flips() -> None:
    selection, dataset, run = _selection_with_one_of_five_labels_flipped()
    targets, _, report = _constraint_catalog(
        selection,
        dataset.source_grammar,
        _category_support(dataset, dataset.source_grammar, run),
        domain=PolicyDomain.SYNTACTIC,
        policy=ClampSpec.model_validate(
            {"on_unsatisfiable": {"kind": "drop_clamp"}}
        ).settings(),
    )

    assert targets
    assert report.dropped_clamps == 1
    assert report.skipped_items == 0


def test_error_policy_aborts_under_the_same_twenty_percent_flips() -> None:
    selection, dataset, run = _selection_with_one_of_five_labels_flipped()

    with pytest.raises(UnsatisfiableClampError):
        _constraint_catalog(
            selection,
            dataset.source_grammar,
            _category_support(dataset, dataset.source_grammar, run),
            domain=PolicyDomain.SYNTACTIC,
            policy=ClampSpec().settings(),
        )


def test_skip_item_reports_the_item_removed_by_an_unsatisfiable_clamp() -> None:
    selection, dataset, run = _selection_with_one_of_five_labels_flipped()
    targets, _, report = _constraint_catalog(
        selection,
        dataset.source_grammar,
        _category_support(dataset, dataset.source_grammar, run),
        domain=PolicyDomain.SYNTACTIC,
        policy=ClampSpec.model_validate(
            {"on_unsatisfiable": {"kind": "skip_item"}}
        ).settings(),
    )

    assert targets
    assert report.dropped_clamps == 0
    assert report.skipped_items == 1


def _primitive(name: str, /) -> ConstantPatternSpec:
    return ConstantPatternSpec(name=name)


def _pairing(name: str, left: str, right: str, output: str, /) -> PatternRuleSpec:
    return PatternRuleSpec(
        name=name,
        premises=(_primitive(left), _primitive(right)),
        conclusion=_primitive(output),
        head_policy="pass-through",
    )


def _paired_grammar() -> GrammarAssembly:
    """Return a grammar whose only goal derivation of three tokens is (A A) A."""
    return GrammarIso().forward(
        GrammarDecl(
            name="paired",
            signature=SignatureSpec(
                name="paired", primitives=("S", "A", "B"), syntax="s-expression"
            ),
            rules=RuleSetSpec(
                name="pairing",
                rules=(
                    _pairing("pair", "A", "A", "B"),
                    _pairing("close", "B", "A", "S"),
                ),
            ),
            lexicon=InlineLexiconSpec(
                entries=(LexicalEntrySpec(form="a", category='("#const[Cat]" "A")'),)
            ),
            goal=RootGoalSpec(categories=('("#const[Cat]" "S")',)),
            budget=BudgetSpec(unary_rounds=0),
        ),
        registered_language("propositional"),
    )


def _paired_item(name: str, length: int, /) -> Item:
    return Item(
        ItemId(name),
        tuple(Token(position, "a") for position in range(length)),
    )


def _paired_support(
    grammar: GrammarAssembly, items: tuple[Item, ...], /
) -> _LearnerCategorySupport:
    language = grammar.language

    return _learner_category_support(
        items,
        grammar,
        _learner_category_inventory(
            grammar, NeuralCodecView(language.codec.at(language.category_sort))
        ),
        TaggedParserSpec(width=4, unary_depth=0),
    )


def _bracket_selection(
    brackets: tuple[tuple[Item, tuple[int, int]], ...], /
) -> ConstraintSelection[str, str]:
    """Return one negative bracket per item over its declared span."""
    selection, _, _ = _constituent_selection()

    return ConstraintSelection(
        tuple(
            Example(
                item,
                SpanConstraints(
                    (SpanConstraint(TokenSpan(*span), frozenset({"S"})),),
                    len(item.tokens),
                ),
            )
            for item, span in brackets
        ),
        selection.report,
        selection.budget,
    )


def test_tagged_support_holds_only_categories_on_some_goal_derivation() -> None:
    grammar = _paired_grammar()
    support = _paired_support(grammar, (_paired_item("three", 3),))
    chart = support.charts[3]

    def rendered(span: tuple[int, int], /) -> set[str]:
        return {
            grammar.syntax.render(grammar.store, category)
            for category, _ in chart.categories_at(span)
        }

    assert {span: rendered(span) for span in SpanIndex(3).spans_by_length()} == {
        (0, 1): {'("#const[Cat]" "A")'},
        (1, 2): {'("#const[Cat]" "A")'},
        (2, 3): {'("#const[Cat]" "A")'},
        (0, 2): {'("#const[Cat]" "B")'},
        (1, 3): set(),
        (0, 3): {'("#const[Cat]" "S")'},
    }


def test_a_bracket_every_goal_derivation_builds_is_refused_before_training() -> None:
    grammar = _paired_grammar()
    item = _paired_item("three", 3)

    with pytest.raises(UnsatisfiableClampError) as raised:
        _constraint_catalog(
            _bracket_selection(((item, (0, 2)),)),
            grammar,
            _paired_support(grammar, (item,)),
            domain=PolicyDomain.SYNTACTIC,
            policy=ClampSpec(require_constituent=False).settings(),
        )

    assert raised.value.clamp.span == (0, 2)
    assert not raised.value.clamp.require_constituent


@pytest.mark.parametrize(
    ("action", "dropped", "skipped"),
    [("drop_clamp", 1, 0), ("skip_item", 0, 1)],
)
def test_a_bracket_every_goal_derivation_builds_uses_the_declared_policy(
    action: str, dropped: int, skipped: int
) -> None:
    grammar = _paired_grammar()
    mandatory = _paired_item("mandatory", 3)
    optional = _paired_item("optional", 3)
    targets, batcher, report = _constraint_catalog(
        _bracket_selection(((mandatory, (0, 2)), (optional, (1, 3)))),
        grammar,
        _paired_support(grammar, (mandatory, optional)),
        domain=PolicyDomain.SYNTACTIC,
        policy=ClampSpec.model_validate(
            {
                "require_constituent": False,
                "on_unsatisfiable": {"kind": action},
            }
        ).settings(),
    )
    localized = batcher.for_items((mandatory.item_id, optional.item_id))

    assert not targets
    assert not localized[0]
    assert len(localized[1]) == 1
    assert report.dropped_clamps == dropped
    assert report.skipped_items == skipped


def test_an_item_with_no_goal_derivation_uses_the_declared_policy() -> None:
    grammar = _paired_grammar()
    item = _paired_item("four", 4)

    targets, batcher, report = _constraint_catalog(
        _bracket_selection(((item, (1, 3)),)),
        grammar,
        _paired_support(grammar, (item,)),
        domain=PolicyDomain.SYNTACTIC,
        policy=ClampSpec.model_validate(
            {
                "require_constituent": False,
                "on_unsatisfiable": {"kind": "skip_item"},
            }
        ).settings(),
    )

    assert not targets
    assert not batcher.for_items((item.item_id,))[0]
    assert report.skipped_items == 1


def test_bracket_requirements_are_decided_once_per_length_and_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grammar = _paired_grammar()
    items = tuple(_paired_item(f"item-{position}", 3) for position in range(3))
    decided: list[tuple[int, int]] = []
    requires_span = TaggedInsideKernel.requires_span

    def counted(
        kernel: TaggedInsideKernel[Term],
        channels: ScoreChannels,
        index: SpanIndex,
        span: tuple[int, int],
        /,
        *,
        goals: Sequence[Term] | None = None,
    ) -> torch.Tensor:
        decided.append(span)
        return requires_span(kernel, channels, index, span, goals=goals)

    monkeypatch.setattr(TaggedInsideKernel, "requires_span", counted)

    _constraint_catalog(
        _bracket_selection(tuple((item, (1, 3)) for item in items)),
        grammar,
        _paired_support(grammar, items),
        domain=PolicyDomain.SYNTACTIC,
        policy=ClampSpec(require_constituent=False).settings(),
    )

    assert decided == [(1, 3)]


SYNTHETIC_PARSER: Final = (
    Path(__file__).resolve().parents[3] / "lofi/configs/run/synthetic-parser.yaml"
)


def _front_declarations() -> dict[str, ComponentSpec]:
    """Return the token encoder and subword pooler of the packaged parser run."""
    declared = RunSpecCodec(search_path=(PACKAGED_CONFIG_ROOT,))(
        SYNTHETIC_PARSER
    ).model.base_components()

    return {
        role: declared[role]
        for role in (ModelRole.TOKENS_ENCODER, ModelRole.SUBWORD_POOLER)
    }


def test_a_built_producer_enters_the_graph_without_being_rebuilt() -> None:
    declarations = _front_declarations()
    context = plugin_context(component="model")
    tokens = wire_components(
        {ModelRole.TOKENS_ENCODER: declarations[ModelRole.TOKENS_ENCODER]},
        context,
        pretrained_resources={},
        built={},
        derived={},
    ).components[ModelRole.TOKENS_ENCODER]

    graph = wire_components(
        declarations,
        context,
        pretrained_resources={},
        built={ModelRole.TOKENS_ENCODER: tokens},
        derived={},
    )

    assert graph.components[ModelRole.TOKENS_ENCODER] is tokens
    assert graph.dependency(ModelRole.SUBWORD_POOLER, "encoder") is tokens


def test_a_derived_edge_wires_a_producer_the_declaration_does_not_name() -> None:
    declarations = _front_declarations()
    pooler = declarations[ModelRole.SUBWORD_POOLER]

    graph = wire_components(
        declarations,
        plugin_context(component="model"),
        pretrained_resources={},
        built={},
        derived={
            ModelRole.SUBWORD_POOLER: (Dependency(ModelRole.TOKENS_ENCODER, "context"),)
        },
    )

    assert pooler.dependencies()
    assert (
        graph.dependency(ModelRole.SUBWORD_POOLER, "context")
        is (graph.components[ModelRole.TOKENS_ENCODER])
    )


def test_a_built_producer_cannot_read_a_producer() -> None:
    declarations = _front_declarations()

    with pytest.raises(ValueError, match="built before the graph cannot read"):
        wire_components(
            declarations,
            plugin_context(component="model"),
            pretrained_resources={},
            built={ModelRole.SUBWORD_POOLER: object()},
            derived={},
        )


def test_wiring_refuses_a_built_component_the_graph_does_not_hold() -> None:
    with pytest.raises(ValueError, match="built components name absent"):
        wire_components(
            _front_declarations(),
            plugin_context(component="model"),
            pretrained_resources={},
            built={"absent": object()},
            derived={},
        )


def test_wiring_refuses_a_derived_edge_into_a_role_the_graph_does_not_hold() -> None:
    with pytest.raises(ValueError, match="derived edges name absent"):
        wire_components(
            _front_declarations(),
            plugin_context(component="model"),
            pretrained_resources={},
            built={},
            derived={"absent": ()},
        )

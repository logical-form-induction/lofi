"""Pure planning for configured training."""

from __future__ import annotations

import json
from hashlib import sha256
from types import MappingProxyType
from typing import TYPE_CHECKING, Never, final, override

from lofi._frozen import FrozenValue
from lofi.data.datasets import (
    CheckedDataset,
    DatasetPlugin,
    DatasetVocabulary,
    LoweredDataset,
    PairTraining,
    ReservedForms,
    SentenceTraining,
    dataset_plugin,
)
from lofi.data.synthetic.spec.interpretation import (
    InterpretationSpec,
    lower_interpretation,
)
from lofi.formalism import BoundedConsumer
from lofi.judge import ActionSet
from lofi.judge._raising import soft_action_set
from lofi.spec import (
    DataModelLanguageSpec,
    DeclarationError,
    EvaluationData,
    IndexedLanguage,
    InlineModelLanguageSpec,
    LearnerInterpretationSpec,
    RetainedPairSideDerivationsSpec,
    RunSpec,
    fingerprint_run,
)
from lofi.spec.grammar import GrammarAssembly, GrammarDecl, GrammarIso
from lofi.spec.language import LanguageDecl, LanguageIso
from lofi.training_plugins import trainer_plugin

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.data._ports import CodeTensorizer, FeaturizerResources
    from lofi.data.synthetic import InterpretationFunction
    from lofi.deduce import ParsePolicy, Retain
    from lofi.run._seed import PointSeed
    from lofi.spec import ApproximationPlan, ComponentSpec, DatasetSpec, RunFingerprint
    from lofi.spec.components.model import ModelLanguageSpec
    from lofi.spec.grammar import CompileReport
    from lofi.train import ParserFittingPlan, StepFittingPlan

    type FittingPlan = ParserFittingPlan | StepFittingPlan

__all__ = [
    "LearnerAssembly",
    "TrainingBlueprint",
    "materialize_learner_language",
    "plan_training",
    "resolve_learner",
]


@final
class LearnerAssembly(FrozenValue):
    """The one grammar and optional semantics owned by a model."""

    __slots__ = ("_grammar", "_interpretation")
    _noun = "learner assembly"

    def __init__(
        self,
        grammar: GrammarAssembly,
        interpretation: InterpretationFunction | None,
        /,
    ) -> None:
        object.__setattr__(self, "_grammar", grammar)
        object.__setattr__(self, "_interpretation", interpretation)

    @property
    def grammar(self) -> GrammarAssembly:
        """Return the learner grammar in its one language store."""
        return self._grammar

    @property
    def interpretation(self) -> InterpretationFunction | None:
        """Return the learner semantics, when one is declared."""
        return self._interpretation


def _shared_source(checked_datasets: Mapping[str, CheckedDataset], /) -> CheckedDataset:
    """Return the first checked source after validating exact source identity."""
    if not checked_datasets:
        msg = "model.language.from 'data' requires at least one selected dataset"
        raise ValueError(msg)

    first_role = next(iter(checked_datasets))
    first = checked_datasets[first_role]
    shared = first.source_grammar
    shared_interpretation = first.source_interpretation_declaration

    for role, checked in checked_datasets.items():
        grammar = checked.source_grammar
        if (
            grammar.declaration_digest != shared.declaration_digest
            or grammar.language.declaration_digest != shared.language.declaration_digest
            or checked.source_interpretation_declaration != shared_interpretation
        ):
            msg = (
                "model.language.from 'data' requires all selected datasets to "
                "share one source grammar, language, and interpretation; "
                f"data.datasets.{role}.dataset does not match "
                f"data.datasets.{first_role}.dataset"
            )
            raise ValueError(msg)
    return first


def materialize_learner_language(
    declaration: ModelLanguageSpec,
    checked_datasets: Mapping[str, CheckedDataset],
    /,
) -> InlineModelLanguageSpec:
    """Return an explicit inline declaration for a model language selection."""
    if isinstance(declaration, InlineModelLanguageSpec):
        return declaration
    if not isinstance(declaration, DataModelLanguageSpec):
        msg = f"unknown model language declaration {type(declaration).__name__}"
        raise TypeError(msg)

    checked = _shared_source(checked_datasets)
    source = checked.source_grammar
    document = checked.source_interpretation_declaration
    interpretation = (
        None if document is None else LearnerInterpretationSpec.model_validate(document)
    )
    return InlineModelLanguageSpec(
        language=LanguageDecl.model_validate_json(source.language.declaration),
        grammar=GrammarDecl.model_validate_json(source.declaration),
        interpretation=interpretation,
    )


def resolve_learner(
    declaration: ModelLanguageSpec,
    checked_datasets: Mapping[str, CheckedDataset],
    /,
) -> LearnerAssembly:
    """Lower one model-owned learner language exactly once."""
    if isinstance(declaration, DataModelLanguageSpec):
        checked = _shared_source(checked_datasets)
        return LearnerAssembly(
            checked.source_grammar,
            checked.source_interpretation,
        )

    explicit = materialize_learner_language(declaration, checked_datasets)
    languages = LanguageIso()
    grammars = GrammarIso(languages)
    if explicit.interpretation is None:
        grammar = grammars.forward(explicit.grammar, explicit.language)
        interpretation = None
    else:
        semantic = InterpretationSpec.model_validate_json(
            explicit.interpretation.model_dump_json()
        )
        grammar, interpretation = lower_interpretation(
            explicit.grammar,
            explicit.language,
            semantic,
            grammars,
            languages,
        )
    return LearnerAssembly(grammar, interpretation)


def _fingerprint(declaration: ComponentSpec, /) -> str:
    canonical = json.dumps(
        json.loads(declaration.model_dump_json()),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()

    return sha256(canonical).hexdigest()


def _learner_signature_inventories(
    grammar: GrammarAssembly, /
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Return learner primitive, constructor and rule names in owner order."""
    signature = GrammarDecl.model_validate_json(grammar.declaration).signature
    primitives = signature.primitives
    constructors = tuple(constructor.name for constructor in signature.constructors)
    combinators = tuple(rule.name for rule in grammar.grammar.rules)
    return primitives, constructors, combinators


def _validate_assignment_inventory(
    field: str, assigned: tuple[str, ...], owned: tuple[str, ...], /
) -> None:
    """Require one signature assignment to permute its learner-owned inventory."""
    missing = tuple(name for name in owned if name not in assigned)
    extra = tuple(name for name in assigned if name not in owned)
    if missing or extra:
        msg = (
            f"signature_assignment.{field} must be an exact permutation of the "
            f"learner {field}; missing {missing!r}, extra {extra!r}"
        )
        raise ValueError(msg)


def _validate_category_side_data(run_spec: RunSpec, /) -> None:
    """Refuse category side scoring without retained pair-side derivations."""
    evaluators = tuple(
        (role, evaluator)
        for role, evaluator in run_spec.evaluators.items()
        if evaluator.evaluation_data() is EvaluationData.PAIR_SIDE_DERIVATIONS
    )
    if not evaluators:
        return
    dataset_role = run_spec.trainer.evaluation_dataset()
    if dataset_role is None:
        msg = "category-side evaluation requires an evaluation dataset"
        raise TypeError(msg)
    dataset = run_spec.data.datasets[dataset_role].dataset
    if not isinstance(dataset, RetainedPairSideDerivationsSpec):
        msg = (
            f"category-side evaluation requires retained pair-side derivations, "
            f"which dataset {dataset_role!r} does not declare"
        )
        raise TypeError(msg)
    retained = dataset.retained_pair_side_derivation_splits()
    for role, evaluator in evaluators:
        required = tuple(
            split
            for split in (evaluator.fit_split(), evaluator.score_split())
            if split is not None
        )
        missing = tuple(split for split in required if split not in retained)
        if missing:
            msg = (
                f"category-side evaluator {role!r} requires retained pair-side "
                f"derivation splits {missing!r}, but dataset {dataset_role!r} "
                f"provides {tuple(retained)!r}"
            )
            raise ValueError(msg)


@final
class TrainingBlueprint(FrozenValue):
    """Checked, allocation-free inputs to seeded training preparation."""

    _noun = "training blueprint"

    __slots__ = (
        "_actions",
        "_approximation",
        "_checked_datasets",
        "_codec_fingerprint",
        "_compile_report",
        "_component_fingerprints",
        "_configuration",
        "_corpus_revision",
        "_dataset_plugins",
        "_declaration_fingerprint",
        "_grammar_fingerprint",
        "_learner",
        "_parse_policy",
        "_run_fingerprint",
        "_run_spec",
        "_sampler_streams",
    )

    def __init__(
        self,
        run_spec: RunSpec,
        checked_datasets: Mapping[str, CheckedDataset],
        dataset_plugins: Mapping[
            str, DatasetPlugin[DatasetSpec, CheckedDataset, LoweredDataset]
        ],
        learner: LearnerAssembly,
        *,
        component_fingerprints: Mapping[str, str],
        run_fingerprint: RunFingerprint,
        sampler_streams: tuple[str, ...],
    ) -> None:
        declaration = run_spec.model_dump_json().encode()
        grammar = learner.grammar
        declared = run_spec.model.actions
        primitive_order, constructor_order, rule_order = _learner_signature_inventories(
            grammar
        )
        assignment = run_spec.signature_assignment

        if assignment is not None:
            _validate_assignment_inventory(
                "primitives", assignment.primitives, primitive_order
            )
            _validate_assignment_inventory(
                "constructors", assignment.constructors, constructor_order
            )
            _validate_assignment_inventory(
                "combinators", assignment.combinators, rule_order
            )
            rule_order = assignment.combinators

        actions = ActionSet.derive(
            rule_order,
            identity=declared.identity,
            null=declared.null,
        )
        controller = run_spec.model.controller

        if controller is not None and controller.raising() == "soft":
            raising_rules = tuple(
                rule
                for rule in grammar.grammar.rules
                if rule.normal_form_role.raising is not None
            )

            if not raising_rules:
                msg = (
                    "model.controller.raise 'soft' requires a grammar rule with "
                    "normal-form role 'raising'"
                )
                raise ValueError(msg)

            actions = soft_action_set(actions, grammar.grammar.rules)
        object.__setattr__(self, "_run_spec", run_spec)
        object.__setattr__(self, "_learner", learner)
        object.__setattr__(self, "_approximation", run_spec.approximation_plan())
        object.__setattr__(self, "_parse_policy", run_spec.parse.policy())
        object.__setattr__(
            self, "_checked_datasets", MappingProxyType(dict(checked_datasets))
        )
        object.__setattr__(
            self, "_dataset_plugins", MappingProxyType(dict(dataset_plugins))
        )
        object.__setattr__(self, "_configuration", declaration)
        object.__setattr__(
            self,
            "_component_fingerprints",
            MappingProxyType(dict(component_fingerprints)),
        )
        object.__setattr__(self, "_grammar_fingerprint", grammar.grammar.fingerprint)
        object.__setattr__(self, "_compile_report", grammar.compile_report)
        object.__setattr__(
            self, "_codec_fingerprint", grammar.language.codec.index.digest
        )
        object.__setattr__(self, "_actions", actions)
        object.__setattr__(
            self,
            "_corpus_revision",
            sha256(
                json.dumps(
                    {
                        role: sha256(checked.declaration).hexdigest()
                        for role, checked in checked_datasets.items()
                    },
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode()
            ).hexdigest(),
        )
        object.__setattr__(
            self, "_declaration_fingerprint", sha256(declaration).hexdigest()
        )
        object.__setattr__(self, "_run_fingerprint", run_fingerprint)
        object.__setattr__(self, "_sampler_streams", sampler_streams)

    @override
    def __setattr__[ValueT](self, name: str, value: ValueT) -> Never:
        del name, value
        msg = "a training blueprint is immutable"
        raise AttributeError(msg)

    @property
    def run_spec(self) -> RunSpec:
        """Return the validated run declaration."""
        return self._run_spec

    @property
    def learner(self) -> LearnerAssembly:
        """Return the model-owned learner assembly resolved at planning."""
        return self._learner

    @property
    def checked_datasets(self) -> Mapping[str, CheckedDataset]:
        """Return checked dataset inputs keyed by their declared role."""
        return self._checked_datasets

    def checked_dataset(self, role: str, /) -> CheckedDataset:
        """Return the checked inputs for one declared dataset role."""
        return self._checked_datasets[role]

    def lower_dataset(self, role: str, /) -> LoweredDataset:
        """Lower and validate one checked dataset through its installed plugin."""
        plugin = self._dataset_plugins[role]
        dataset = plugin.lower(self._checked_datasets[role], self._parse_policy)
        plugin.validate(dataset)

        return dataset

    @property
    def parse_policy(self) -> ParsePolicy[Retain]:
        """Return the run's once-lowered symbolic parse policy."""
        return self._parse_policy

    def dataset_vocabulary(
        self, role: str, dataset: LoweredDataset, /
    ) -> DatasetVocabulary:
        """Return the vocabulary declared by one dataset's installed plugin."""
        return self._dataset_plugins[role].vocabulary(dataset, self._reserved(role))

    def featurizer_resources(
        self, role: str, dataset: LoweredDataset, tensorizer: CodeTensorizer, /
    ) -> FeaturizerResources:
        """Return the featurizer resources one dataset's installed plugin builds."""
        return self._dataset_plugins[role].resources(
            dataset, self._reserved(role), tensorizer
        )

    def sentence_training(
        self, role: str, dataset: LoweredDataset, /
    ) -> SentenceTraining:
        """Return the sentence training one dataset's installed plugin supplies."""
        return self._dataset_plugins[role].sentences(dataset)

    def pair_training(self, role: str, dataset: LoweredDataset, /) -> PairTraining:
        """Return the pair training one dataset's installed plugin supplies."""
        return self._dataset_plugins[role].pairs(dataset)

    def _reserved(self, role: str, /) -> ReservedForms:
        """Return the reserved forms one dataset section declares."""
        section = self._run_spec.data.datasets[role]

        return ReservedForms(section.reserved_forms, unknown=section.unknown_form)

    @property
    def component_fingerprints(self) -> Mapping[str, str]:
        """Return declaration fingerprints keyed by structural component role."""
        return self._component_fingerprints

    @property
    def configuration(self) -> bytes:
        """Return the validated run declaration's exact JSON bytes."""
        return self._configuration

    @property
    def grammar_fingerprint(self) -> str:
        """Return the compiled grammar identity."""
        return self._grammar_fingerprint

    @property
    def compile_report(self) -> CompileReport:
        """Return the compiled grammar diagnostics persisted with this point."""
        return self._compile_report

    @property
    def approximation(self) -> ApproximationPlan:
        """Return the resolved residual budgets recorded with this point."""
        return self._approximation

    @property
    def codec_fingerprint(self) -> str:
        """Return the allocation-free term-codec identity."""
        return self._codec_fingerprint

    @property
    def actions(self) -> ActionSet:
        """Return the controller actions the grammar and the declaration define."""
        return self._actions

    @property
    def action_set_fingerprint(self) -> str:
        """Return the allocation-free controller action set identity."""
        return self._actions.digest

    @property
    def corpus_revision(self) -> str:
        """Return the canonical corpus-declaration identity."""
        return self._corpus_revision

    @property
    def declaration_fingerprint(self) -> str:
        """Return the validated run-declaration identity."""
        return self._declaration_fingerprint

    @property
    def run_fingerprint(self) -> RunFingerprint:
        """Return the schema, theory and substrate fingerprint."""
        return self._run_fingerprint

    @property
    def sampler_streams(self) -> tuple[str, ...]:
        """Return the named sampler streams preparation consumes."""
        return self._sampler_streams


def plan_training(
    run_spec: RunSpec, /, *, point: PointSeed | None = None
) -> TrainingBlueprint:
    """Check a composed run without resolving or allocating factories.

    The run's trainer and dataset were checked against each other when the
    run was composed. Planning asks the trainer's installed plugin for its
    sampler streams and checks the sentence corpus the dataset lowers to. The
    blueprint keeps the run declaration as written, so a data seed declared
    as derived stays derived in it, while the checked corpus is drawn under
    the seeds the point derives.

    Parameters
    ----------
    run_spec : RunSpec
        The composed run.
    point : PointSeed | None
        The point the run executes for, which derives every data seed the run
        declares as derived, or None for a run whose data seeds are all fixed.

    Raises
    ------
    ValueError
        If the trainer's plugin refuses the run's wiring.
    ProfileUnsupportedError
        If the declared parser cannot serve the grammar's fan-out or rule arities.
    UnresolvedSeedError
        If a data seed is derived and no point is given.
    """
    _validate_category_side_data(run_spec)
    trainer = run_spec.trainer
    checked_datasets: dict[str, CheckedDataset] = {}
    dataset_plugins: dict[
        str, DatasetPlugin[DatasetSpec, CheckedDataset, LoweredDataset]
    ] = {}
    dataset_streams: list[str] = []

    for role, section in run_spec.data.datasets.items():
        dataset = section.dataset

        if point is not None:
            dataset = dataset.with_derived_seeds(
                point.data_seed, f"data.datasets.{role}.dataset"
            )

        plugin = dataset_plugin(dataset)
        try:
            checked_datasets[role] = plugin.check(dataset)
        except DeclarationError as error:
            path = f"data.datasets.{role}.dataset"
            raise error.prefixed(path) from error
        dataset_plugins[role] = plugin
        dataset_streams.extend(
            f"dataset:{role}:{stream}" for stream in plugin.sampler_streams(dataset)
        )

    sampler_streams = (
        *dataset_streams,
        *trainer_plugin(trainer).plan(trainer, run_spec),
    )

    if len(set(sampler_streams)) != len(sampler_streams):
        msg = (
            f"dataset and trainer declare repeated sampler streams: {sampler_streams!r}"
        )
        raise ValueError(msg)

    learner = resolve_learner(run_spec.model.language, checked_datasets)
    learner_grammar = learner.grammar
    grammar = learner_grammar.grammar
    parser = run_spec.model.parser

    if isinstance(parser, BoundedConsumer):
        parser.profile_bound().check(grammar.profile, grammar.name)

    language = IndexedLanguage.from_versions(
        grammar.theory_version,
        grammar.theory_specification,
        grammar.theory_digest,
        source_version=learner_grammar.source_version,
        target_version=learner_grammar.target_version,
    )
    roles = run_spec.component_roles()

    return TrainingBlueprint(
        run_spec,
        checked_datasets,
        dataset_plugins,
        learner,
        component_fingerprints={
            role: _fingerprint(spec) for role, spec in roles.items()
        },
        run_fingerprint=fingerprint_run({"grammar": language}),
        sampler_streams=sampler_streams,
    )

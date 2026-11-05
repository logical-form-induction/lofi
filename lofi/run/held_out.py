"""Held-out evaluation of a trained parser through its declared evaluators.

A run document declares its held-out evaluators under ``evaluators``, each
keyed by the role its rows are reported under. After training, every declared
evaluator is built by the plugin its kind names in the ``lofi.evaluator``
entry-point group, from its declaration's settings and the trained grammar, the
trained forward pass is published once over the held-out batches, and each
evaluator's metric rows are emitted through the result sink with full identity,
each value beside the bootstrap interval its declaration asks for, drawn over
the units its metric counts under the run seed.
An evaluator that publishes artifacts emits them through that same sink under
its declared role after its report has completed.
Every row is scored under the approximation budget the grammar declares, and
no residual of that budget is measured while scoring, so each report reaches
the sink with that budget and an unmeasured residual.

The freeze a row is reported against is not part of the run document. A run
document is the configuration an experiment freezes and sweeps, and the experiment
declares the freeze it runs against, so the caller passes the freeze of the
experiment that dispatched the run, or ``UNFROZEN`` for a run scored outside
any experiment, whose rows no freeze ingests.
"""

from collections import Counter
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple, cast, final

import torch

from lofi._types import ItemId
from lofi.eval import (
    Approximation,
    ArtifactEvaluator,
    DevelopmentAlignedEvaluator,
    HeldOutEvaluationContext,
    InferenceEvaluationContext,
    ReportIdentity,
    ReportRow,
    report,
)
from lofi.eval._held_out import LexicalTrainingEvidence
from lofi.nn.loss._ports import (
    CoherenceEvaluationObjective,
    TargetSetSource,
)
from lofi.registry import plugin_for
from lofi.run._evaluator import Evaluator
from lofi.spec import EvaluationData
from lofi.spec.components.data import (
    HeldOutDatasetSpec,
    InferenceLabeledDatasetSpec,
    RetainedDatasetSpec,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from lofi.data import EvalDataset, Example, GoldDerivation, Item
    from lofi.eval import (
        AlignmentScore,
        HeldOutEvaluator,
        HeldOutItems,
        InferenceItems,
        Report,
    )
    from lofi.eval._ports import (
        GrammaticalityItems,
    )
    from lofi.formalism import Grammar
    from lofi.nn.loss._values import ObjectiveOutputs
    from lofi.registry import ComponentPlugin
    from lofi.run._ports import ResultSink
    from lofi.spec import ConfiguredSpec, EvaluatorSpec, RunSpec
    from lofi.terms import TermKey
    from lofi.train import (
        ParserFittingPlan,
        PreparedTraining,
    )

__all__ = [
    "HELD_OUT_SUITE",
    "HeldOutPredictions",
    "evaluate_held_out",
    "held_out_predictions",
]

HELD_OUT_SUITE: Final = "held-out"
"""The suite every held-out row is reported under."""

type _EvaluatorPlugin = ComponentPlugin[
    object,
    HeldOutEvaluationContext | InferenceEvaluationContext,
    HeldOutEvaluator[object, object],
]
"""The plugin an evaluator kind lists, built over held-out dataset resources."""


def _evaluator(
    declaration: EvaluatorSpec,
    context: HeldOutEvaluationContext | InferenceEvaluationContext,
    /,
) -> HeldOutEvaluator[object, object]:
    """Return the evaluator one declaration describes in its held-out context.

    Raises
    ------
    ValueError
        If the built evaluator's metric is not one its declaration reports, so
        an experiment's rows would name a metric no row carries.
    """
    configured = cast("ConfiguredSpec[object]", declaration)
    plugin = cast("_EvaluatorPlugin", plugin_for(configured))
    evaluator = plugin.construct(configured, context)
    if isinstance(evaluator, DevelopmentAlignedEvaluator):
        return evaluator
    reported = tuple(metric.name for metric in evaluator.metrics)
    declared = _declared_metrics(declaration, context)
    if reported != declared:
        msg = (
            f"a {declaration.kind!r} evaluator reports metrics {reported}, "
            f"not its declaration's {declared}"
        )
        raise ValueError(msg)

    return evaluator


def _declared_metrics(
    declaration: EvaluatorSpec,
    context: HeldOutEvaluationContext | InferenceEvaluationContext,
    /,
) -> tuple[str, ...]:
    """Return report rows after resolving any data-owned response scale."""
    if isinstance(context, InferenceEvaluationContext):
        return declaration.metrics(context.labels)

    return declaration.metrics()


def _retained_derivations(
    plan: ParserFittingPlan,
    /,
) -> Mapping[str, tuple[Example[GoldDerivation[str]], ...]]:
    """Return exact retained derivation partitions by declared split name.

    Corpus examples and both sides of inference pairs retain their gold
    derivations; a retained item outside them, such as a corruption, has no
    gold analysis.
    The dataset's ordinary held-out partition contains every retained gold item.
    Named item splits add exact subpartitions and may not disagree with that
    partition when they use the same name.
    """
    if plan.corpus is not None:
        derivations = plan.corpus.examples
    elif plan.pairs is not None:
        derivations = plan.pairs.derivations
    else:
        msg = "the parser plan holds no gold derivation corpus"
        raise ValueError(msg)

    ordered = _unique_derivations(derivations)
    examples = {example.item.item_id: example for example in ordered}
    exact: dict[str, tuple[Example[GoldDerivation[str]], ...]] = {}
    for split, batches in plan.retained_batches.items():
        selected = tuple(
            examples[item_id]
            for batch in batches
            for item_id in batch.item_ids
            if item_id in examples
        )
        misplaced = tuple(
            example.item.item_id
            for example in selected
            if example.item.split is not None and example.item.split != split
        )
        if misplaced:
            msg = (
                f"retained partition {split!r} contains items assigned elsewhere: "
                f"{misplaced!r}"
            )
            raise ValueError(msg)
        exact[split] = _unique_derivations(selected)
    return MappingProxyType(exact)


def _unique_derivations(
    examples: Sequence[Example[GoldDerivation[str]]], /
) -> tuple[Example[GoldDerivation[str]], ...]:
    """Return the first occurrence of each consistently annotated item.

    Inference generation may expose the same sentence in several premise or
    hypothesis pairs. Held-out derivation metrics score that sentence once. An
    identifier that names different item content or a different gold derivation
    is not a repeated exposure and is refused.
    """
    unique: dict[ItemId, Example[GoldDerivation[str]]] = {}
    for example in examples:
        item_id = example.item.item_id
        prior = unique.get(item_id)
        if prior is not None and prior != example:
            msg = f"item {item_id!r} has conflicting held-out gold derivations"
            raise ValueError(msg)
        unique.setdefault(item_id, example)
    return tuple(unique.values())


def _partition(
    retained: Mapping[str, tuple[Example[GoldDerivation[str]], ...]],
    split: str,
    /,
) -> tuple[Example[GoldDerivation[str]], ...]:
    """Return one nonempty exact retained partition or refuse its absence."""
    selected = retained.get(split)
    if selected is None:
        msg = (
            f"held-out evaluation declares split {split!r}, but retained item "
            f"metadata provides {sorted(retained)!r}"
        )
        raise ValueError(msg)
    if not selected:
        msg = f"held-out evaluation split {split!r} contains no gold item"
        raise ValueError(msg)
    return selected


def _batches(
    plan: ParserFittingPlan,
    split: str,
    /,
) -> tuple[object, ...]:
    """Return one nonempty exact retained batch partition."""
    selected = plan.retained_batches.get(split)
    if selected is None:
        msg = (
            f"held-out evaluation declares split {split!r}, but retained batches "
            f"provide {sorted(plan.retained_batches)!r}"
        )
        raise ValueError(msg)
    if not selected:
        msg = f"held-out evaluation split {split!r} contains no batch"
        raise ValueError(msg)
    return cast("tuple[object, ...]", selected)


def _published(
    prepared: PreparedTraining,
    plan: ParserFittingPlan,
    partitions: Mapping[str, Sequence[object]],
    /,
) -> Mapping[str, tuple[ObjectiveOutputs[HeldOutItems], ...]]:
    """Publish the trained forward pass over the batches of some partitions."""
    modules = prepared.model.modules.values()
    training = tuple(module.training for module in modules)

    try:
        for module in modules:
            module.eval()

        with torch.no_grad():
            publish = cast(
                "Callable[[HeldOutItems], ObjectiveOutputs[HeldOutItems]]",
                plan.forward.publish,
            )
            return MappingProxyType(
                {
                    split: tuple(
                        publish(cast("HeldOutItems", batch)) for batch in batches
                    )
                    for split, batches in partitions.items()
                }
            )

    finally:
        for module, mode in zip(modules, training, strict=True):
            module.train(mode)


def _ordinary_split(run_spec: RunSpec, dataset: RetainedDatasetSpec, /) -> str:
    """Return the run's exact declared scoring partition."""
    retained = dataset.retained_splits()
    requested = frozenset(
        split
        for evaluator in run_spec.evaluators.values()
        if (split := evaluator.score_split()) is not None
    )
    if len(requested) > 1:
        msg = (
            f"evaluators score incompatible retained partitions: {sorted(requested)!r}"
        )
        raise ValueError(msg)
    if requested:
        selected = next(iter(requested))
        if selected not in retained:
            msg = (
                f"evaluators score {selected!r}, but the dataset retains "
                f"{sorted(retained)!r}"
            )
            raise ValueError(msg)
        return selected
    if isinstance(dataset, HeldOutDatasetSpec):
        selected = dataset.held_out_split()
        if selected not in retained:
            msg = (
                f"the dataset selects {selected!r} for held-out scoring, but "
                f"retains {sorted(retained)!r}"
            )
            raise ValueError(msg)
        return selected
    if len(retained) != 1:
        msg = (
            "an evaluator without fitted partitions requires exactly one retained "
            f"split, not {sorted(retained)!r}"
        )
        raise ValueError(msg)
    return next(iter(retained))


def _gold[Gold](dataset: EvalDataset[Gold], /) -> Mapping[ItemId, Gold]:
    """Return one immutable annotation value per evaluation item."""
    return MappingProxyType(
        {annotation.item_id: annotation.value for annotation in dataset.annotations}
    )


def _pair_id(position: int, /) -> ItemId:
    """Return the stable identifier shared by retained pair resources and batches."""
    return ItemId(f"inference-pair/{position}")


def _predictions[Pred, Gold](
    evaluator: HeldOutEvaluator[Pred, Gold],
    published: Sequence[ObjectiveOutputs[HeldOutItems]],
    scored: frozenset[ItemId],
    /,
) -> Mapping[ItemId, Pred]:
    """Return one evaluator's predictions for an exact retained partition."""
    predictions: dict[ItemId, Pred] = {}

    for outputs in published:
        predicted = evaluator.predict(outputs)
        predictions.update(
            (item_id, prediction)
            for item_id, prediction in predicted.items()
            if item_id in scored
        )

    return MappingProxyType(predictions)


def _report[Pred, Gold](
    evaluator: HeldOutEvaluator[Pred, Gold],
    scorer: Evaluator[Gold, Pred],
    published: Sequence[ObjectiveOutputs[HeldOutItems]],
    scored: frozenset[ItemId],
    /,
) -> Report:
    """Return one evaluator's report over an exact retained partition."""
    return scorer.evaluate(_predictions(evaluator, published, scored))


def _aligned_report(
    headline: Report,
    score: AlignmentScore,
    *,
    evaluator: HeldOutEvaluator[object, object],
    run_spec: RunSpec,
    grammar: Grammar,
    freeze_id: str,
    role: str,
    dataset_name: str,
    split: str,
    seed: int,
    approximation: Approximation,
    normal_form: str,
) -> Report:
    """Append alignment accuracy and floor rows to a headline test report."""
    identity = ReportIdentity(
        run_spec,
        grammar,
        evaluator.conventions,
        freeze_id=freeze_id,
    )
    rows = tuple(
        ReportRow(
            identity=identity,
            method=role,
            metric=name,
            result=result,
            approximation=approximation,
            suite=HELD_OUT_SUITE,
            task=role,
            dataset=dataset_name,
            split=split,
            seed=seed,
            normal_form=normal_form,
        )
        for name, result in score.rows.items()
    )
    return report(
        (*headline.rows, *rows),
        unpredicted=headline.unpredicted,
        unmatched=headline.unmatched,
    )


def _lexical_training_evidence(
    plan: ParserFittingPlan, /
) -> Mapping[str, LexicalTrainingEvidence]:
    """Measure training frequency and source-gold category ambiguity by form."""
    corpus = plan.corpus
    if corpus is None:
        return MappingProxyType({})

    source = plan.dataset.source_grammar
    examples = {example.item.item_id: example for example in corpus.examples}
    frequencies: Counter[str] = Counter()
    categories: dict[str, set[TermKey]] = {}
    batches = cast("tuple[HeldOutItems, ...]", plan.batches)
    for batch in batches:
        for item_id, words in zip(batch.item_ids, batch.words, strict=True):
            frequencies.update(words)
            example = examples.get(item_id)
            if example is None:
                continue
            leaves = sorted(
                example.supervision.derivation.leaves(),
                key=lambda leaf: leaf.span.start,
            )
            spans = tuple((leaf.span.start, leaf.span.end) for leaf in leaves)
            expected = tuple((position, position + 1) for position in range(len(words)))
            if spans != expected:
                msg = (
                    f"training derivation {item_id!r} does not tile its "
                    f"{len(words)} tokens"
                )
                raise ValueError(msg)
            for word, leaf in zip(words, leaves, strict=True):
                category = source.syntax.parse(source.store, leaf.label).key()
                categories.setdefault(word, set()).add(category)

    return MappingProxyType(
        {
            form: LexicalTrainingEvidence(count, len(categories.get(form, ())))
            for form, count in sorted(frequencies.items())
        }
    )


class _Contexts(NamedTuple):
    """The retained dataset a run scores and the contexts its evaluators read."""

    dataset: RetainedDatasetSpec
    derivation: HeldOutEvaluationContext | None
    inference: InferenceEvaluationContext | None


def _contexts(plan: ParserFittingPlan, run_spec: RunSpec, /) -> _Contexts:
    """Return the evaluation dataset and the held-out contexts of a fitted plan.

    Raises
    ------
    TypeError
        If the trainer declares no evaluation dataset, the dataset declares no
        named retained partitions, or an inference dataset declares no label
        inventory.
    ValueError
        If an item has conflicting held-out surface forms.
    """
    dataset_role = run_spec.trainer.evaluation_dataset()

    if dataset_role is None:
        msg = f"the {run_spec.trainer.kind!r} trainer declares no evaluation dataset"
        raise TypeError(msg)

    dataset = run_spec.data.datasets[dataset_role].dataset
    if not isinstance(dataset, RetainedDatasetSpec):
        msg = (
            f"the {type(dataset).__name__} dataset does not declare named retained "
            "partitions"
        )
        raise TypeError(msg)
    corpus = plan.corpus
    pairs = plan.pairs
    coherence = {
        entry.name: entry.objective
        for entry in plan.step.assembly
        if isinstance(entry.objective, CoherenceEvaluationObjective)
    }
    target_sets = {}
    if corpus is not None:
        if not isinstance(plan.forward, TargetSetSource):
            msg = "a derivation forward pass does not publish target-set ownership"
            raise TypeError(msg)
        target_sets = plan.forward.target_sets
    lexical_training = _lexical_training_evidence(plan)
    training_items = len(
        {
            item_id
            for batch in cast("tuple[HeldOutItems, ...]", plan.batches)
            for item_id in batch.item_ids
        }
    )
    parse = run_spec.parse.build()
    if corpus is not None:
        derivation_context = HeldOutEvaluationContext(
            plan.grammar,
            interpretation=plan.interpretation,
            forms={item_id: form.text for item_id, form in corpus.forms.items()},
            coherence=coherence,
            target_sets=target_sets,
            lexical_training=lexical_training,
            parse=parse,
            training_items=training_items,
        )
    elif pairs is not None:
        forms: dict[ItemId, str] = {}
        for item_id, form in pairs.gold_forms:
            prior = forms.get(item_id)
            if prior is not None and prior != form:
                msg = f"item {item_id!r} has conflicting held-out surface forms"
                raise ValueError(msg)
            forms.setdefault(item_id, form)
        derivation_context = HeldOutEvaluationContext(
            plan.grammar,
            interpretation=plan.interpretation,
            forms=forms,
            coherence=coherence,
            target_sets={},
            lexical_training=lexical_training,
            parse=parse,
            training_items=training_items,
        )
    else:
        derivation_context = None
    interpretation = plan.interpretation
    gold_pairs = (
        ()
        if pairs is None
        else tuple(
            (position, pair, pair.gold)
            for position, pair in enumerate(pairs.corpus_pairs)
            if pair.relation is not None and pair.gold is not None
        )
    )
    if interpretation is None or not gold_pairs:
        inference_context = None
    else:
        if not isinstance(dataset, InferenceLabeledDatasetSpec):
            msg = "an inference evaluation dataset declares its label inventory"
            raise TypeError(msg)
        inference_context = InferenceEvaluationContext(
            plan.grammar,
            dataset.inference_labels(),
            interpretation=interpretation,
            forms={
                _pair_id(position): (
                    interpretation.read_text(premise),
                    interpretation.read_text(hypothesis),
                )
                for position, _, (premise, hypothesis) in gold_pairs
            },
            tokens={
                _pair_id(position): (
                    tuple(token.surface for token in pair.premise.tokens),
                    tuple(token.surface for token in pair.hypothesis.tokens),
                )
                for position, pair, _ in gold_pairs
            },
            parse=parse,
        )

    return _Contexts(dataset, derivation_context, inference_context)


class _Scoring(NamedTuple):
    """What every row one declared evaluator reports is identified by."""

    run_spec: RunSpec
    grammar: Grammar
    freeze_id: str
    role: str
    seed: int
    approximation: Approximation
    declaration: EvaluatorSpec

    def scorer(
        self,
        evaluator: HeldOutEvaluator[object, object],
        data: EvalDataset[object],
        split: str,
        /,
    ) -> Evaluator[object, object]:
        """Return the scorer of one evaluator's metrics over one partition."""
        return Evaluator(
            data,
            evaluator.metrics,
            run_spec=self.run_spec,
            grammar=self.grammar,
            freeze_id=self.freeze_id,
            fallback_conventions=evaluator.conventions,
            suite=HELD_OUT_SUITE,
            task=self.role,
            dataset_name=self.grammar.name,
            split=split,
            seed=self.seed,
            approximation=self.approximation,
            normal_form=self.grammar.normal_form.name,
            interval=self.declaration.interval.settings(),
        )


@final
class _HeldOut:
    """The retained partitions one declared evaluator scores, by data shape."""

    __slots__ = ("_derivation", "_inference", "_plan", "_retained", "_role")

    def __init__(
        self,
        plan: ParserFittingPlan,
        retained: Mapping[str, tuple[Example[GoldDerivation[str]], ...]],
        role: str,
        /,
        *,
        derivation: bool,
        inference: bool,
    ) -> None:
        self._plan = plan
        self._retained = retained
        self._role = role
        self._derivation = derivation
        self._inference = inference

    def derivations(self, split: str, /) -> tuple[Example[GoldDerivation[str]], ...]:
        """Return the retained corpus derivations of a partition.

        Raises
        ------
        ValueError
            If the run holds out no corpus derivations.
        """
        if not self._derivation:
            msg = (
                f"the evaluator {self._role!r} reads derivation items, which this "
                "run holds out none of"
            )
            raise ValueError(msg)

        return _partition(self._retained, split)

    def side_derivations(
        self, split: str, /
    ) -> tuple[Example[GoldDerivation[str]], ...]:
        """Return the retained pair-side derivations of a partition.

        Raises
        ------
        ValueError
            If the run holds out no inference pairs.
        """
        if not self._inference:
            msg = (
                f"the evaluator {self._role!r} reads pair-side derivations, which "
                "this run holds out none of"
            )
            raise ValueError(msg)

        return _partition(self._retained, split)

    def pair_batches(self, split: str, /) -> Sequence[InferenceItems]:
        """Return the retained inference-pair batches of a partition.

        Raises
        ------
        ValueError
            If the run holds out no inference pairs.
        """
        if not self._inference:
            msg = (
                f"the evaluator {self._role!r} scores inference pairs, which this "
                "run holds out none of"
            )
            raise ValueError(msg)

        return cast("Sequence[InferenceItems]", _batches(self._plan, split))

    def judgment_batches(self, split: str, /) -> Sequence[GrammaticalityItems]:
        """Return the retained grammaticality-judgment batches of a partition.

        Raises
        ------
        ValueError
            If the run holds out inference pairs rather than judged sentences.
        """
        if self._plan.corpus is None:
            msg = (
                f"the evaluator {self._role!r} scores grammaticality, but this run "
                "holds out inference pairs"
            )
            raise ValueError(msg)

        return cast("Sequence[GrammaticalityItems]", _batches(self._plan, split))

    def data(
        self, evaluator: HeldOutEvaluator[object, object], split: str, /
    ) -> tuple[EvalDataset[object], Mapping[ItemId, object]]:
        """Return one partition as the evaluator reads it, with its gold by item."""
        dataset = evaluator.held_out(self, split)

        return dataset, _gold(dataset)


def evaluate_held_out(
    prepared: PreparedTraining,
    run_spec: RunSpec,
    *,
    freeze_id: str,
    seed: int,
    sink: ResultSink,
) -> tuple[Report, ...]:
    """Score a trained parser's held-out split with every declared evaluator.

    Parameters
    ----------
    prepared : PreparedTraining
        The trained model and the plan it was fitted under.
    run_spec : RunSpec
        The validated run document the model was trained from.
    freeze_id : str
        The freeze of the experiment that dispatched the run, ``freeze/<n>``,
        or ``UNFROZEN`` for a run scored outside any experiment.
    seed : int
        The run seed every row records.
    sink : ResultSink
        Where each evaluator's report is emitted, with the grammar's budget
        and an unmeasured residual.

    Returns
    -------
    tuple[Report, ...]
        One report per declared evaluator, in role order, each already emitted.

    Raises
    ------
    TypeError
        If the dataset does not declare the name of its held-out split.
    ValueError
        If the freeze identifier is malformed, the plan holds out no parse, or an
        evaluator cannot read what the trained pass published.
    UnknownTagError
        If no installed evaluator is registered under a declared kind.
    """
    plan = prepared.plan.parser_plan()
    grammar = plan.grammar
    dataset, derivation_context, inference_context = _contexts(plan, run_spec)
    if not run_spec.evaluators:
        return ()
    retained = _retained_derivations(plan)
    published = _published(prepared, plan, plan.retained_batches)
    reports: list[Report] = []

    for role, declaration in run_spec.evaluators.items():
        context = (
            inference_context
            if declaration.evaluation_data() is EvaluationData.INFERENCE_PAIRS
            else derivation_context
        )

        if context is None:
            msg = "the parser plan holds neither derivations nor inference pairs"
            raise ValueError(msg)

        evaluator = _evaluator(declaration, context)
        held = _HeldOut(
            plan,
            retained,
            role,
            derivation=derivation_context is not None,
            inference=inference_context is not None,
        )
        scoring = _Scoring(
            run_spec,
            grammar.grammar,
            freeze_id,
            role,
            seed,
            Approximation(grammar.grammar.budget, residual=None),
            declaration,
        )

        if isinstance(evaluator, DevelopmentAlignedEvaluator):
            aligned = cast("DevelopmentAlignedEvaluator[object, object]", evaluator)
            fit_split = aligned.alignment.fit_on
            score_split = aligned.alignment.score_on
            fit_data, fit_gold = held.data(evaluator, fit_split)
            score_data, score_gold = held.data(evaluator, score_split)
            fit_ids = frozenset(fit_data.ids)
            score_ids = frozenset(score_data.ids)
            overlap = sorted(fit_ids & score_ids)
            if overlap:
                msg = (
                    f"alignment fitting split {fit_split!r} and scoring split "
                    f"{score_split!r} overlap on {overlap!r}"
                )
                raise ValueError(msg)
            fit_predictions = _predictions(evaluator, published[fit_split], fit_ids)
            aligned.fit_alignment(fit_predictions, fit_gold)
            reported = tuple(metric.name for metric in evaluator.metrics)
            declared = _declared_metrics(declaration, context)
            if reported != declared[: len(reported)]:
                msg = (
                    f"a {declaration.kind!r} evaluator reports metrics {reported}, "
                    f"not the leading metrics of its declaration's {declared}"
                )
                raise ValueError(msg)
            score_predictions = _predictions(
                evaluator, published[score_split], score_ids
            )
            headline = scoring.scorer(evaluator, score_data, score_split).evaluate(
                score_predictions
            )
            scored_alignment = aligned.score_alignment(
                score_predictions, score_gold, seed=seed
            )
            completed = _aligned_report(
                headline,
                scored_alignment,
                evaluator=evaluator,
                run_spec=run_spec,
                grammar=grammar.grammar,
                freeze_id=freeze_id,
                role=role,
                dataset_name=grammar.grammar.name,
                split=score_split,
                seed=seed,
                approximation=scoring.approximation,
                normal_form=grammar.grammar.normal_form.name,
            )
            names = tuple(row.metric for row in completed.rows)
            if names != declared:
                msg = (
                    f"a {declaration.kind!r} evaluator reports metrics {names}, "
                    f"not its declaration's {declared}"
                )
                raise ValueError(msg)
        else:
            score_split = _ordinary_split(run_spec, dataset)
            evaluation_data, _ = held.data(evaluator, score_split)
            completed = _report(
                evaluator,
                scoring.scorer(evaluator, evaluation_data, score_split),
                published[score_split],
                frozenset(evaluation_data.ids),
            )

        reports.append(completed)
        sink.emit(completed)
        if isinstance(evaluator, ArtifactEvaluator):
            for artifact in evaluator.artifacts():
                sink.emit_artifact(role, artifact)

    return tuple(reports)


class HeldOutPredictions(NamedTuple):
    """One declared evaluator's predictions over one retained partition.

    Attributes
    ----------
    evaluator : HeldOutEvaluator[object, object]
        The evaluator the role declares, built over the trained grammar.
    items : tuple[Item, ...]
        The partition's gold-analyzed items, in retained order.
    predictions : Mapping[ItemId, object]
        The evaluator's prediction for each of those items it predicted.
    """

    evaluator: HeldOutEvaluator[object, object]
    items: tuple[Item, ...]
    predictions: Mapping[ItemId, object]


def held_out_predictions(
    prepared: PreparedTraining, run_spec: RunSpec, role: str, split: str, /
) -> HeldOutPredictions:
    """Rerun one declared evaluator's predictions over one retained partition.

    The evaluator is built as held-out scoring builds it, and the trained
    forward pass is published over the partition's batches alone.

    Parameters
    ----------
    prepared : PreparedTraining
        The trained model and the plan it was fitted under.
    run_spec : RunSpec
        The validated run document the model was trained from.
    role : str
        The role the evaluator is declared under.
    split : str
        The retained partition whose items are predicted.

    Raises
    ------
    ValueError
        If the run declares no evaluator under the role, holds out nothing the
        evaluator reads, or retains no gold item under the split.
    TypeError
        If the dataset does not declare named retained partitions.
    """
    plan = prepared.plan.parser_plan()
    declaration = run_spec.evaluators.get(role)

    if declaration is None:
        msg = (
            f"the run declares no evaluator {role!r}; it declares "
            f"{sorted(run_spec.evaluators)!r}"
        )
        raise ValueError(msg)

    _, derivation_context, inference_context = _contexts(plan, run_spec)
    context = (
        inference_context
        if declaration.evaluation_data() is EvaluationData.INFERENCE_PAIRS
        else derivation_context
    )

    if context is None:
        msg = (
            f"the run holds out nothing the {declaration.kind!r} evaluator "
            f"{role!r} reads"
        )
        raise ValueError(msg)

    evaluator = _evaluator(declaration, context)
    examples = _partition(_retained_derivations(plan), split)
    published = _published(prepared, plan, {split: _batches(plan, split)})
    predictions = _predictions(
        evaluator,
        published[split],
        frozenset(example.item.item_id for example in examples),
    )

    return HeldOutPredictions(
        evaluator, tuple(example.item for example in examples), predictions
    )

"""Registered LM formalization pipelines evaluated through one proof boundary."""

from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple, cast, final

from lofi._types import ItemId
from lofi.eval._baselines import (
    BaselineFamily,
    BaselineHarness,
    BaselineItem,
    BaselineSystem,
    RerunConfiguration,
)
from lofi.eval._entailment_family import EntailmentPair, entailment_family
from lofi.eval._inference_proof import open_inference_proof_step, prove_inference_pairs
from lofi.eval._logical_form import LogicalFormMetric
from lofi.eval._logical_form_evaluator import (
    AlignedLogicalFormMetric,
    fit_logical_form_alignment,
    score_logical_form_alignment,
)
from lofi.eval._ports import BaselineBuildContext, BaselineMeasurement, BaselineResult
from lofi.eval._values import ConventionVector, MetricResult
from lofi.lf import RELATION_LABELS, InferenceCell, InferenceLabel
from lofi.lf.provers import ProofService
from lofi.registry import ComponentPlugin, plugin_for
from lofi.spec._interpolation import lower_package_resource
from lofi.spec.baselines.lm_prover import (
    FormalizationPipeline,
    LmProverBaselineSettings,
    LmProverBaselineSpec,
)
from lofi.spec.baselines.lm_prover_trace import (
    GeneratedFormalization,
    LmProverData,
    LmProverGenerationItem,
    read_lm_prover_data,
    read_lm_prover_generations,
)
from lofi.spec.language import LanguageError, LanguageIso
from lofi.terms import Nominal, Term, to_nominal

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi.eval._baselines import BaselineRow
    from lofi.lf import EntailmentOracle
    from lofi.spec.baselines.lm_prover_trace import LmProverGenerations
    from lofi.spec.components.base import ConfiguredSpec
    from lofi.spec.evaluators.conventions import ConventionSettings
    from lofi.spec.language import LanguageAssembly

__all__ = ["PLUGIN"]

_REASON: Final = (
    "recorded raw model generations are replayed through the declared "
    "formalization-selection and proof configuration"
)
_LF_FAMILY: Final = "logical-form-inference"


class _ParsedAttempt(NamedTuple):
    """One generated premise and hypothesis that both parse and type-check."""

    premise: Term
    hypothesis: Term


class _Prediction(NamedTuple):
    """One selected formalization and its strict three-way proof label."""

    premise: Term | None
    hypothesis: Term | None
    label: InferenceLabel | None


def _resolved(root: Path, declared: str, /) -> Path:
    """Resolve one declared source against the fit document directory."""
    lowered = Path(lower_package_resource(declared))
    return lowered if lowered.is_absolute() else root / lowered


def _digest(content: bytes, /) -> str:
    """Return the content identity reported beside a baseline result."""
    return f"sha256:{sha256(content).hexdigest()}"


def _read_sources(
    root: Path, settings: LmProverBaselineSettings, /
) -> tuple[LmProverData, LmProverGenerations, Mapping[str, str]]:
    """Read and validate the dataset, raw generations, and prompt identity."""
    paths = {
        "data": _resolved(root, settings.data_source),
        "generations": _resolved(root, settings.generations_source),
        "prompt": _resolved(root, settings.prompt_source),
    }
    contents = {name: path.read_bytes() for name, path in paths.items()}
    digests = {name: _digest(content) for name, content in contents.items()}
    data = read_lm_prover_data(contents["data"].decode("utf-8"))
    generations = read_lm_prover_generations(contents["generations"].decode("utf-8"))
    if generations.pipeline != settings.pipeline.name:
        msg = (
            f"generation trace declares {generations.pipeline!r}, not the "
            f"configured {settings.pipeline.name!r} pipeline"
        )
        raise ValueError(msg)
    if generations.model != settings.model:
        msg = (
            f"generation trace declares model {generations.model!r}, not "
            f"{settings.model!r}"
        )
        raise ValueError(msg)
    if generations.prompt_digest != digests["prompt"]:
        msg = (
            f"generation trace declares prompt {generations.prompt_digest!r}, not "
            f"the prompt content identity {digests['prompt']!r}"
        )
        raise ValueError(msg)
    expected = {pair.item_id for pair in data.pairs}
    observed = {item.item_id for item in generations.items}
    if expected != observed:
        msg = (
            "generation trace item identifiers do not exactly match the data; "
            f"missing={sorted(expected - observed)!r}, "
            f"extra={sorted(observed - expected)!r}"
        )
        raise ValueError(msg)
    return data, generations, digests


def _parse_attempt(
    language: LanguageAssembly, generated: GeneratedFormalization, /
) -> _ParsedAttempt | None:
    """Return a typed generated pair, or absence for an invalid formalization."""
    try:
        return _ParsedAttempt(
            language.read(generated.premise),
            language.read(generated.hypothesis),
        )
    except LanguageError, ValueError:
        return None


def _validated_attempts(
    language: LanguageAssembly,
    item: LmProverGenerationItem,
    settings: LmProverBaselineSettings,
    /,
) -> tuple[_ParsedAttempt | None, ...]:
    """Parse every raw attempt and refuse a trace the pipeline cannot produce."""
    candidates = item.candidates
    parsed = tuple(_parse_attempt(language, candidate) for candidate in candidates)
    settings.pipeline.check_trace(
        item.item_id,
        tuple(candidate.feedback is not None for candidate in candidates),
        tuple(attempt is not None for attempt in parsed),
    )
    return parsed


def _strict_label(cell: InferenceCell | None, /) -> InferenceLabel | None:
    """Return a label only for one of the three nondegenerate decided cells."""
    if cell in (
        InferenceCell.ENTAILS,
        InferenceCell.CONTRADICTS,
        InferenceCell.NEUTRAL,
    ):
        return InferenceLabel(cell.value)
    return None


def _select(
    attempts: Sequence[_ParsedAttempt | None],
    cells: Sequence[InferenceCell | None],
    *,
    pipeline: FormalizationPipeline,
) -> _Prediction:
    """Select one proof-labeled formalization under the declared pipeline."""
    parsed = tuple(attempt for attempt in attempts if attempt is not None)
    labels = tuple(_strict_label(cell) for cell in cells)
    chosen = pipeline.select(labels) if parsed else None
    if chosen is None:
        return _Prediction(None, None, None)
    selected = parsed[chosen]
    return _Prediction(selected.premise, selected.hypothesis, labels[chosen])


def _predictions(
    data: LmProverData,
    generations: LmProverGenerations,
    language: LanguageAssembly,
    service: ProofService,
    settings: LmProverBaselineSettings,
    /,
) -> Mapping[str, _Prediction]:
    """Run every generated candidate through the shared inference proof cell."""
    by_id = {item.item_id: item for item in generations.items}
    produced: dict[str, _Prediction] = {}
    normalization = settings.normalization.build(fuel=settings.normalization_fuel)
    budget = settings.budget.build()
    for pair in data.pairs:
        attempts = _validated_attempts(language, by_id[pair.item_id], settings)
        parsed = tuple(attempt for attempt in attempts if attempt is not None)
        if not parsed:
            produced[pair.item_id] = _Prediction(None, None, None)
            continue
        open_inference_proof_step(service)
        proof = prove_inference_pairs(
            language,
            service,
            budget,
            tuple((attempt.premise, attempt.hypothesis) for attempt in parsed),
            normalization=normalization,
        )
        produced[pair.item_id] = _select(
            attempts,
            proof.cells,
            pipeline=settings.pipeline,
        )
    return produced


def _conventions(settings: ConventionSettings, /) -> ConventionVector:
    """Build the metric convention vector from the validated declaration."""
    return ConventionVector(
        punctuation=settings.punctuation,
        unary_chains=settings.unary_chains,
        trivial_spans=settings.trivial_spans,
        minimum_length=settings.minimum_length,
        aggregation=settings.aggregation,
        coverage=settings.coverage,
        self_arcs=settings.self_arcs,
        gold_derivation=settings.gold_derivation,
    )


def _nominal(
    language: LanguageAssembly, term: Term | None, names: Sequence[str], /
) -> Nominal | None:
    """Present one selected core form under the language-owned name capacity."""
    return None if term is None else to_nominal(language.store, term, names)


def _row_measurement(row: BaselineRow, *, metric: str) -> BaselineMeasurement:
    """Lower one harness row into the baseline-fit reporting boundary."""
    result = row.result
    return BaselineMeasurement(
        row.family,
        row.system,
        metric,
        row.arm,
        row.cites,
        row.reason,
        result.value,
        result.n,
        result.resampling.unit,
    )


def _direct_measurement(
    *,
    system: str,
    metric: str,
    evidence: RerunConfiguration,
    result: MetricResult,
    family: str = _LF_FAMILY,
) -> BaselineMeasurement:
    """Lower a deterministic metric result not owned by a harness family."""
    return BaselineMeasurement(
        family,
        system,
        metric,
        evidence.arm,
        evidence.cites,
        _REASON,
        result.value,
        result.n,
        result.resampling.unit,
    )


@final
class _Operation:
    """Replay raw generations, prove them, and score held-out LF predictions."""

    __slots__ = ("_context", "_settings")

    def __init__(
        self, settings: LmProverBaselineSettings, context: BaselineBuildContext
    ) -> None:
        self._settings = settings
        self._context = context

    def __call__(self, /) -> BaselineResult:
        """Return LF, alignment, inference, coverage, and prover measurements."""
        settings = self._settings
        data, generations, source_digests = _read_sources(self._context.root, settings)
        language = LanguageIso().forward(settings.language)
        declared_prover = settings.language.logic.prover
        configured = cast("ConfiguredSpec[object]", declared_prover)

        def oracle_factory() -> EntailmentOracle:
            return cast(
                "EntailmentOracle",
                plugin_for(configured).construct(configured, language),
            )

        service = ProofService(
            oracle_factory,
            profile=language.profile,
            workers=declared_prover.workers,
            batch_size=declared_prover.batch_size,
            timeout=declared_prover.timeout,
            cache_size=declared_prover.cache_size,
        )
        try:
            predicted = _predictions(data, generations, language, service, settings)
            report = service.report
        finally:
            service.close()

        names = tuple(
            f"x{index}"
            for index in range(language.codec.at(language.term_sort).capacity)
        )
        gold_terms = {
            pair.item_id: _ParsedAttempt(
                language.read(pair.gold_premise),
                language.read(pair.gold_hypothesis),
            )
            for pair in data.pairs
        }
        dev_predictions: dict[ItemId, Nominal | None] = {}
        dev_gold: dict[ItemId, Nominal] = {}
        test_predictions: dict[ItemId, Nominal | None] = {}
        test_gold: dict[ItemId, Nominal] = {}
        for pair in data.pairs:
            prediction = predicted[pair.item_id]
            gold = gold_terms[pair.item_id]
            prediction_terms = (prediction.premise, prediction.hypothesis)
            gold_pair = (gold.premise, gold.hypothesis)
            target_predictions = (
                dev_predictions if pair.split == "dev" else test_predictions
            )
            target_gold = dev_gold if pair.split == "dev" else test_gold
            for role, pred_term, gold_term in zip(
                ("premise", "hypothesis"),
                prediction_terms,
                gold_pair,
                strict=True,
            ):
                key = ItemId(f"{pair.item_id}:{role}")
                target_predictions[key] = _nominal(language, pred_term, names)
                held_gold = _nominal(language, gold_term, names)
                if held_gold is None:
                    msg = f"gold logical form {key!r} is absent"
                    raise ValueError(msg)
                target_gold[key] = held_gold

        fitted = fit_logical_form_alignment(
            dev_predictions,
            dev_gold,
            language.calculus,
            settings.alignment,
        )
        alignment = score_logical_form_alignment(
            fitted,
            test_predictions,
            test_gold,
            language.calculus,
            seed=self._context.seed,
        )
        conventions = _conventions(settings.scoring).extended(
            {
                "normalization_order": settings.normalization.order,
                "normalization_eta": settings.normalization.eta,
                "normalization_fuel": settings.normalization_fuel,
                "normalization_policy": settings.normalization.policy,
                "normalization_associative_commutative": ",".join(
                    settings.normalization.associative_commutative
                ),
            }
        )
        metric = AlignedLogicalFormMetric(
            LogicalFormMetric(
                language.store.signature,
                language.calculus,
                settings.measure,
                normalization=settings.normalization.build(
                    fuel=settings.normalization_fuel
                ),
                store_nodes=language.store.max_nodes,
                conventions=conventions,
                seed=self._context.seed,
            ),
            fitted,
            language.store.signature,
            language.calculus,
        )
        system_name = settings.pipeline.name
        evidence = RerunConfiguration(
            settings.generations_source,
            config_hash=source_digests["generations"],
        )
        harness = BaselineHarness()
        lf_items = tuple(
            BaselineItem(key, test_predictions[key], test_gold[key])
            for key in sorted(test_gold)
        )
        lf_row = harness.run(
            BaselineFamily(_LF_FAMILY, metric),
            (
                BaselineSystem(
                    system_name,
                    family=_LF_FAMILY,
                    reason=_REASON,
                    items=lf_items,
                    rerun=evidence,
                ),
            ),
        )[0]
        test_pairs = tuple(pair for pair in data.pairs if pair.split == "test")
        inference_family = entailment_family(
            labels=frozenset(label.value for label in RELATION_LABELS),
            conventions=conventions,
            seed=self._context.seed,
        )
        inference_row = harness.run(
            inference_family,
            (
                BaselineSystem(
                    system_name,
                    family=inference_family.name,
                    reason=_REASON,
                    items=tuple(
                        BaselineItem(
                            ItemId(pair.item_id),
                            (
                                None
                                if (label := predicted[pair.item_id].label) is None
                                else label.value
                            ),
                            EntailmentPair(
                                pair.label.value,
                                len(pair.premise_tokens),
                                len(pair.hypothesis_tokens),
                            ),
                        )
                        for pair in test_pairs
                    ),
                    rerun=evidence,
                ),
            ),
        )[0]
        covered = sum(predicted[pair.item_id].label is not None for pair in test_pairs)
        count = len(test_pairs)
        alignment_rows = tuple(alignment.rows.values())
        abstentions = sum(report.abstentions.values())
        measurements = (
            _row_measurement(lf_row, metric=f"logical-form-{settings.measure}"),
            _direct_measurement(
                system=system_name,
                metric=f"{settings.alignment.kind}-accuracy",
                evidence=evidence,
                result=alignment_rows[0],
            ),
            _direct_measurement(
                system=system_name,
                metric="random-relabeling-floor",
                evidence=evidence,
                result=alignment_rows[1],
            ),
            _row_measurement(inference_row, metric="inference-accuracy"),
            BaselineMeasurement(
                inference_row.family,
                system_name,
                "inference-coverage",
                evidence.arm,
                evidence.cites,
                _REASON,
                covered / count,
                count,
                "pair",
            ),
            BaselineMeasurement(
                "proof-service",
                system_name,
                "prover-calls",
                evidence.arm,
                evidence.cites,
                _REASON,
                report.calls,
                count,
                "pair",
            ),
            BaselineMeasurement(
                "proof-service",
                system_name,
                "prover-memo-hit-rate",
                evidence.arm,
                evidence.cites,
                _REASON,
                report.memo_hit_rate,
                count,
                "pair",
            ),
            BaselineMeasurement(
                "proof-service",
                system_name,
                "prover-cpu-seconds",
                evidence.arm,
                evidence.cites,
                _REASON,
                report.cpu_seconds,
                count,
                "pair",
            ),
            BaselineMeasurement(
                "proof-service",
                system_name,
                "prover-abstention-rate",
                evidence.arm,
                evidence.cites,
                _REASON,
                0.0 if report.questions == 0 else abstentions / report.questions,
                count,
                "pair",
            ),
        )
        return BaselineResult(measurements, source_digests)


def _build(
    settings: LmProverBaselineSettings,
    context: BaselineBuildContext,
    /,
) -> _Operation:
    """Bind one validated LM-plus-prover declaration to run resources."""
    return _Operation(settings, context)


PLUGIN: Final[
    ComponentPlugin[LmProverBaselineSettings, BaselineBuildContext, _Operation]
] = ComponentPlugin(spec=LmProverBaselineSpec, build=_build)

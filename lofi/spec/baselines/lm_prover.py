"""Declaration of language-model-plus-prover baselines."""

from collections import Counter
from hashlib import sha256
from typing import (
    TYPE_CHECKING,
    Literal,
    NamedTuple,
    Protocol,
    Self,
    cast,
    final,
    override,
)

import didactic.api as dx

from lofi._checks import ensure_name, ensure_non_negative, ensure_positive
from lofi.spec.components.baseline import BaselineSystemSpec
from lofi.spec.evaluators.alignment import AlignmentSpec
from lofi.spec.evaluators.conventions import ConventionSpec
from lofi.spec.evaluators.logical_form import LogicalFormMeasure
from lofi.spec.language import LANGUAGE_FIELD, LanguageDecl, canonical_bytes
from lofi.spec.normalization import NormalizationSpec
from lofi.spec.provers import ProofBudgetSpec

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence

    from lofi._types import JsonValue
    from lofi.spec.evaluators.alignment import (
        AlignmentDeclaration,
        AlignmentSettings,
    )
    from lofi.spec.evaluators.conventions import ConventionSettings


class FormalizationPipeline(Protocol):
    """How one LM-plus-prover pipeline records and selects formalizations."""

    @property
    def name(self) -> str:
        """Return the pipeline's declared name."""
        ...

    @property
    def candidates(self) -> int:
        """Return how many independent formalizations the pipeline samples."""
        ...

    @property
    def refinements(self) -> int:
        """Return how many solver-feedback refinements the pipeline allows."""
        ...

    def check_trace(
        self, item: str, feedback: Sequence[bool], valid: Sequence[bool], /
    ) -> None:
        """Refuse a recorded attempt sequence this pipeline cannot produce.

        Parameters
        ----------
        item : str
            The item the attempts were generated for.
        feedback : Sequence[bool]
            Whether each attempt was given solver feedback.
        valid : Sequence[bool]
            Whether each attempt parses and type-checks.

        Raises
        ------
        ValueError
            If the attempts break the pipeline's declared shape.
        """
        ...

    def select[L: Hashable](self, labels: Sequence[L | None], /) -> int | None:
        """Return the position of the selected valid attempt, or None.

        Parameters
        ----------
        labels : Sequence[L | None]
            The proof label of each valid attempt, in attempt order.

        Returns
        -------
        int | None
            The selected attempt's position among the valid attempts, or None
            when the pipeline selects nothing.
        """
        ...


class _PipelineDeclaration(Protocol):
    """A validated pipeline declaration exposing its runtime pipeline."""

    def settings(self) -> FormalizationPipeline:
        """Return the runtime pipeline."""
        ...


__all__ = [
    "FormalizationPipeline",
    "LincPipeline",
    "LincPipelineSpec",
    "LmProverBaselineSettings",
    "LmProverBaselineSpec",
    "LmProverPipelineSpec",
    "LogicLmPipeline",
    "LogicLmPipelineSpec",
]


@final
class LincPipeline:
    """Vote over independent formalization samples by their proof labels.

    Parameters
    ----------
    samples : int
        How many independent formalizations every item records.
    """

    __slots__ = ("_samples",)

    def __init__(self, samples: int) -> None:
        self._samples = ensure_positive("a LINC formalization-sample count", samples)

    @property
    def name(self) -> str:
        """Return the pipeline's declared name."""
        return "linc"

    @property
    def candidates(self) -> int:
        """Return the declared sample count."""
        return self._samples

    @property
    def refinements(self) -> int:
        """Return that LINC refines nothing."""
        return 0

    def check_trace(
        self, item: str, feedback: Sequence[bool], valid: Sequence[bool], /
    ) -> None:
        """Require exactly the declared samples and no solver feedback."""
        del valid
        if len(feedback) != self._samples:
            msg = (
                f"LINC item {item!r} records {len(feedback)} candidates, "
                f"not the declared {self._samples}"
            )
            raise ValueError(msg)
        if any(feedback):
            msg = f"LINC item {item!r} records solver feedback"
            raise ValueError(msg)

    def select[L: Hashable](self, labels: Sequence[L | None], /) -> int | None:
        """Return the first attempt carrying the unique majority label."""
        counts = Counter(label for label in labels if label is not None)
        if not counts:
            return None
        largest = max(counts.values())
        winners = tuple(label for label, count in counts.items() if count == largest)
        if len(winners) != 1:
            return None
        return labels.index(winners[0])

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another pipeline votes over as many samples."""
        return type(other) is LincPipeline and self._samples == other._samples

    @override
    def __hash__(self) -> int:
        """Return a hash over the sample count."""
        return hash((LincPipeline, self._samples))

    @override
    def __repr__(self) -> str:
        """Return the constructor call."""
        return f"LincPipeline({self._samples!r})"


@final
class LogicLmPipeline:
    """Refine one formalization under solver feedback until it is valid.

    Parameters
    ----------
    refinements : int
        The most solver-feedback refinements an item may record.
    """

    __slots__ = ("_refinements",)

    def __init__(self, refinements: int) -> None:
        self._refinements = ensure_non_negative(
            "a Logic-LM refinement limit", refinements
        )

    @property
    def name(self) -> str:
        """Return the pipeline's declared name."""
        return "logic-lm"

    @property
    def candidates(self) -> int:
        """Return the single initial formalization."""
        return 1

    @property
    def refinements(self) -> int:
        """Return the declared refinement limit."""
        return self._refinements

    def check_trace(
        self, item: str, feedback: Sequence[bool], valid: Sequence[bool], /
    ) -> None:
        """Require feedback on every refinement and a stop at the first valid one."""
        if len(feedback) > 1 + self._refinements:
            msg = f"Logic-LM item {item!r} exceeds its declared refinement limit"
            raise ValueError(msg)
        if feedback[0]:
            msg = f"Logic-LM item {item!r} gives feedback before its first attempt"
            raise ValueError(msg)
        for position, (given, held) in enumerate(zip(feedback, valid, strict=True)):
            if position and not given:
                msg = (
                    f"Logic-LM item {item!r} attempt {position} has no solver feedback"
                )
                raise ValueError(msg)
            if held and position + 1 != len(feedback):
                msg = (
                    f"Logic-LM item {item!r} continues after a valid "
                    f"formalization at attempt {position}"
                )
                raise ValueError(msg)

    def select[L: Hashable](self, labels: Sequence[L | None], /) -> int | None:
        """Return the one valid attempt, whatever its proof label."""
        return 0 if labels else None

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another pipeline allows as many refinements."""
        return type(other) is LogicLmPipeline and (
            self._refinements == other._refinements
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the refinement limit."""
        return hash((LogicLmPipeline, self._refinements))

    @override
    def __repr__(self) -> str:
        """Return the constructor call."""
        return f"LogicLmPipeline({self._refinements!r})"


class LmProverPipelineSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declare how generated formalizations are selected."""


class LincPipelineSpec(LmProverPipelineSpec):
    """Declare independent formalization samples followed by prover voting."""

    kind: Literal["linc"] = "linc"
    samples: int

    @dx.validates("samples")
    @classmethod
    def _samples_are_positive(cls, value: int) -> int:
        """Return a positive number of independent formalization samples."""
        return ensure_positive("a LINC formalization-sample count", value)

    def settings(self) -> FormalizationPipeline:
        """Return the LINC voting policy."""
        return LincPipeline(self.samples)


class LogicLmPipelineSpec(LmProverPipelineSpec):
    """Declare one formalization followed by solver-feedback refinements."""

    kind: Literal["logic-lm"] = "logic-lm"
    refinements: int

    @dx.validates("refinements")
    @classmethod
    def _refinements_are_nonnegative(cls, value: int) -> int:
        """Return a nonnegative solver-feedback refinement limit."""
        return ensure_non_negative("a Logic-LM refinement limit", value)

    def settings(self) -> FormalizationPipeline:
        """Return the Logic-LM refinement policy."""
        return LogicLmPipeline(self.refinements)


class LmProverBaselineSettings(NamedTuple):
    """Resolved controls for one LM-plus-prover baseline run."""

    data_source: str
    generations_source: str
    prompt_source: str
    model: str
    pipeline: FormalizationPipeline
    language: LanguageDecl
    budget: ProofBudgetSpec
    normalization: NormalizationSpec
    normalization_fuel: int
    measure: LogicalFormMeasure
    alignment: AlignmentSettings
    scoring: ConventionSettings


class LmProverBaselineSpec(BaselineSystemSpec):
    """Declare replayable LM formalizations scored by LoFI's prover and LF metrics.

    The source records model generations rather than scores. LoFI parses and
    type-checks those generations, runs its declared prover, selects candidates
    according to the named pipeline, and computes every reported measurement.
    """

    kind: Literal["lm-prover"] = "lm-prover"
    data_source: str
    generations_source: str
    prompt_source: str
    model: str
    pipeline: LmProverPipelineSpec
    language: LanguageDecl = LANGUAGE_FIELD
    budget: ProofBudgetSpec
    normalization_fuel: int
    normalization: NormalizationSpec = dx.field(default_factory=NormalizationSpec)
    measure: LogicalFormMeasure
    alignment: AlignmentSpec
    scoring: ConventionSpec

    @dx.validates("data_source", "generations_source", "prompt_source", "model")
    @classmethod
    def _sources_and_model_are_named(cls, value: str) -> str:
        """Return a nonempty source path or model identity."""
        return ensure_name("an LM-plus-prover source or model identity", value)

    @dx.model_validator()
    def _alignment_uses_the_declared_partitions(self) -> Self:
        """Require development-fitted alignment frozen for test scoring."""
        alignment = cast("AlignmentDeclaration", self.alignment).settings()
        if alignment.fit_on != "dev" or alignment.score_on != "test":
            msg = "LM-plus-prover alignment must fit on dev and score on test"
            raise ValueError(msg)
        return self

    @dx.validates("normalization_fuel")
    @classmethod
    def _normalization_fuel_is_positive(cls, value: int) -> int:
        """Return a positive boundary-owned reduction budget."""
        return ensure_positive("an LM-plus-prover normalization fuel", value)

    @override
    def metrics(self) -> tuple[str, ...]:
        """Return LF, inference, coverage, and prover-effort measurements."""
        alignment = cast("AlignmentDeclaration", self.alignment).settings()
        return (
            f"logical-form-{self.measure}",
            f"{alignment.kind}-accuracy",
            "random-relabeling-floor",
            "inference-accuracy",
            "inference-coverage",
            "prover-calls",
            "prover-memo-hit-rate",
            "prover-cpu-seconds",
            "prover-abstention-rate",
        )

    @override
    def conventions(self) -> Mapping[str, JsonValue]:
        """Return the full baseline selection and scoring convention vector."""
        pipeline = cast("_PipelineDeclaration", self.pipeline).settings()
        declared_prover = self.language.logic.prover
        prover: dict[str, JsonValue] = {
            "kind": cast("str", declared_prover.kind),
            "timeout": declared_prover.timeout,
            "workers": declared_prover.workers,
            "batch_size": declared_prover.batch_size,
            "cache_size": declared_prover.cache_size,
            "memory_mb": declared_prover.memory_mb,
            "cpu_seconds": declared_prover.cpu_seconds,
            "seed": declared_prover.seed,
            "polymorphism": declared_prover.polymorphism,
            "certificates": list(declared_prover.certificates),
        }
        budget: dict[str, JsonValue] = {
            "calls_per_step": self.budget.calls_per_step,
            "wall_per_step": self.budget.wall_per_step,
            "cpu_seconds": self.budget.cpu_seconds,
        }
        alignment = cast("AlignmentDeclaration", self.alignment).settings()
        scoring = self.scoring.settings()
        return {
            "pipeline": pipeline.name,
            "formalization_candidates": pipeline.candidates,
            "solver_feedback_refinements": pipeline.refinements,
            "model": self.model,
            "language_digest": (
                "sha256:" + sha256(canonical_bytes(self.language)).hexdigest()
            ),
            "prover": prover,
            "proof_budget": budget,
            "normalization": dict(self.normalization.conventions()),
            "normalization_fuel": self.normalization_fuel,
            "logical_form_measure": self.measure,
            "alignment": alignment.kind,
            "alignment_tie_break": alignment.tie_break,
            "alignment_fit_on": alignment.fit_on,
            "alignment_score_on": alignment.score_on,
            "punctuation": scoring.punctuation,
            "unary_chains": scoring.unary_chains,
            "trivial_spans": scoring.trivial_spans,
            "minimum_length": scoring.minimum_length,
            "aggregation": scoring.aggregation,
            "coverage": scoring.coverage,
            "self_arcs": scoring.self_arcs,
            "gold_derivation": scoring.gold_derivation,
        }

    def settings(self) -> LmProverBaselineSettings:
        """Return validated immutable settings for baseline execution."""
        return LmProverBaselineSettings(
            self.data_source,
            self.generations_source,
            self.prompt_source,
            self.model,
            cast("_PipelineDeclaration", self.pipeline).settings(),
            self.language,
            self.budget,
            self.normalization,
            self.normalization_fuel,
            self.measure,
            cast("AlignmentDeclaration", self.alignment).settings(),
            self.scoring.settings(),
        )

"""Conditional-satisfaction scoring over declared finite structures."""

from typing import TYPE_CHECKING, Final, Literal, Protocol, final, runtime_checkable

import torch
from torch import nn

from lofi.lf.models import (
    EvaluationError,
    Evaluator,
    StructureSource,
    StructureSourceFactory,
    TruthAlgebra,
    fuzzy_algebra,
)
from lofi.nn.grounding._scorer import Satisfaction, Scores
from lofi.nn.grounding._values import EvidenceLedger
from lofi.registry import ComponentPlugin, Registry
from lofi.spec.language import LanguageAssembly
from lofi.spec.scorers.model_checking import (
    ModelCheckingScorerSettings,
    ModelCheckingScorerSpec,
)

if TYPE_CHECKING:
    from torch import Tensor

    from lofi.lf import LogicFragment
    from lofi.lf.models import Structure
    from lofi.lf.provers import Questions
    from lofi.nn.build import BuildContext
    from lofi.nn.grounding._values import ProofMetadata
    from lofi.terms import Term

__all__ = ["PLUGIN", "ModelCheckingScorer"]


@runtime_checkable
class _AuxiliarySource(Protocol):
    """Expose an optional source-owned satisfiability term."""

    def auxiliary(self) -> Tensor | None: ...


@final
class ModelCheckingScorer(nn.Module):
    """Estimate conditional satisfaction without claiming a proof verdict."""

    def __init__(
        self,
        language: LanguageAssembly,
        source: StructureSource,
        algebra: TruthAlgebra,
        /,
        *,
        truth_threshold: float,
        abstain_band: float,
        premise_floor: float,
        unsatisfied_premise: Literal["undecided", "zero"],
    ) -> None:
        super().__init__()
        self._language = language
        self._source = source
        self._algebra = algebra
        self._fragment = language.profile.fragment
        self._truth_threshold = truth_threshold
        self._abstain_band = abstain_band
        self._premise_floor = premise_floor
        self._unsatisfied_premise = unsatisfied_premise

    @property
    def fragment(self) -> LogicFragment:
        return self._fragment

    @property
    def scorer_kind(self) -> str:
        """Return the registered evidence-scoring treatment."""
        return "model-checking"

    @property
    def source_identity(self) -> str:
        """Return the declared structure source supplying the evidence."""
        return self._source.name

    @property
    def differentiable(self) -> bool:
        return False

    @property
    def certified(self) -> bool:
        return False

    @property
    def proof_metadata(self) -> ProofMetadata | None:
        return None

    @property
    def source(self) -> StructureSource:
        """Return the declared source whose structures this scorer reads."""
        return self._source

    def auxiliary(self) -> Tensor | None:
        """Return the source's independent satisfiability term, when present."""
        if isinstance(self._source, _AuxiliarySource):
            return self._source.auxiliary()

        return None

    def score(self, questions: Questions, /) -> Scores[Satisfaction]:
        """Score each question by hypothesis truth among its premise models."""
        structures = self._source.structures()
        values: list[float] = []
        decisions: list[bool] = []
        evidence: list[Satisfaction] = []

        for premise, goal in zip(questions.premises, questions.goals, strict=True):
            score, decided, premise_count, both_count = self._score_pair(
                premise,
                goal,
                questions.theory,
                structures,
            )
            values.append(score)
            decisions.append(decided)
            evidence.append(
                Satisfaction(
                    self._source.name,
                    structures=len(structures),
                    premise_holding=premise_count,
                    both_holding=both_count,
                )
            )

        held_evidence = tuple(evidence)
        return Scores(
            torch.tensor(values, dtype=torch.float64),
            torch.tensor(decisions, dtype=torch.bool),
            held_evidence,
            EvidenceLedger(
                {self._source.evidence_label: len(held_evidence)},
                {},
                calls=len(held_evidence),
                memo_hits=0,
                cpu_seconds=0.0,
                questions=len(held_evidence),
                wall_seconds=0.0,
                process_starts=None,
                verified_share=None,
            ),
        )

    def relaxation(
        self,
        premise: Term,
        hypothesis: Term,
        /,
        *,
        theory: tuple[Term, ...] = (),
    ) -> float:
        """Return conditional satisfaction for an evaluation pair.

        The value is defined independently of the scorer's abstention decision.
        This lets an evaluation metric compare the relaxation with a symbolic
        oracle on exactly the pairs that oracle decided.
        """
        structures = self._source.structures()
        score, _, _, _ = self._score_pair(
            premise,
            hypothesis,
            theory,
            structures,
        )
        return score

    def open_step(self) -> None:
        """Advance the source's declared resampling schedule."""
        self._source.open_step()

    def _score_pair(
        self,
        premise: Term,
        goal: Term,
        theory: tuple[Term, ...],
        structures: tuple[Structure, ...],
    ) -> tuple[float, bool, int, int]:
        premise_mass = 0.0
        joint_mass = 0.0
        premise_count = 0
        both_count = 0

        for candidate in structures:
            try:
                evaluator = Evaluator(self._language, candidate, self._algebra)
                premise_value = self._premise_value(evaluator, premise, theory)
                goal_value = evaluator.truth(goal)
            except EvaluationError:
                return 0.0, False, premise_count, both_count

            if premise_value < self._premise_floor:
                continue

            premise_count += 1
            premise_mass += premise_value
            joint_mass += self._algebra.conjunction(premise_value, goal_value)

            if goal_value >= self._truth_threshold:
                both_count += 1

        if premise_count == 0 or premise_mass == 0.0:
            return (
                0.0,
                self._unsatisfied_premise == "zero",
                premise_count,
                both_count,
            )

        score = joint_mass / premise_mass
        distance = abs(score - self._truth_threshold)
        decided = self._abstain_band == 0.0 or distance > self._abstain_band
        return score, decided, premise_count, both_count

    def _premise_value(
        self,
        evaluator: Evaluator,
        premise: Term,
        theory: tuple[Term, ...],
    ) -> float:
        value = evaluator.truth(premise)

        for sentence in theory:
            value = self._algebra.conjunction(value, evaluator.truth(sentence))

        return value


def _build(
    settings: ModelCheckingScorerSettings,
    context: BuildContext,
    /,
) -> ModelCheckingScorer:
    """Build scored finite-model checking from one declared source."""
    language = context.language

    if not isinstance(language, LanguageAssembly):
        msg = f"model checking reads a LanguageAssembly, not {language!r}"
        raise TypeError(msg)

    algebra = fuzzy_algebra(*settings.logic)
    factory = Registry[object]("structure_source").get(settings.source_kind)

    if not isinstance(factory, StructureSourceFactory):
        msg = f"structure-source entry {settings.source_kind!r} names no factory"
        raise TypeError(msg)

    source = factory.build(
        settings.source,
        language,
        algebra,
        theory_floor=settings.truth_threshold,
    )
    return ModelCheckingScorer(
        language,
        source,
        algebra,
        truth_threshold=settings.truth_threshold,
        abstain_band=settings.abstain_band,
        premise_floor=settings.premise_floor,
        unsatisfied_premise=settings.unsatisfied_premise,
    )


PLUGIN: Final = ComponentPlugin(spec=ModelCheckingScorerSpec, build=_build)

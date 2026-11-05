"""One pairwise grounding over enumerated, truncated, or sampled forms."""

from itertools import product
from math import log
from typing import TYPE_CHECKING, Final, cast, final, override

import torch
from torch import nn

from lofi.control_variate import (
    ControlVariate,
    ControlVariateDeclaration,
    ControlVariatePlugin,
)
from lofi.lf import InferenceCell, InferenceLabel, InferenceLabeling, LabelMap
from lofi.lf.provers import Questions
from lofi.nn._values import LogDistribution
from lofi.nn.forms import Estimate
from lofi.nn.grounding._compose import PairTable, compose
from lofi.nn.grounding._ports import GroundingQuery
from lofi.nn.grounding._scorer import (
    AuxiliaryScorer,
    PairScorer,
    PositiveQuestionScorer,
    Scores,
)
from lofi.nn.grounding._values import (
    CellMass,
    Composition,
    EvidenceLedger,
    GradientPath,
    GroundedBatch,
    GroundingMetadata,
    Phase,
    checked_composition,
    composition_from_settings,
)
from lofi.nn.sites import ComponentId, SiteMap
from lofi.registry import ComponentPlugin, Registry, RegistryError
from lofi.reward import CONTRADICTION_DEFINITIONS
from lofi.spec.groundings import PairwiseGroundingSettings, PairwiseGroundingSpec

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

    from lofi.nn.build import BuildContext
    from lofi.nn.forms import WeightedTerms
    from lofi.nn.grounding._ports import GroundingInput
    from lofi.reward import InferenceReward
    from lofi.terms import Term, TermStore

__all__ = ["PLUGIN", "PairwiseGrounding"]

type _Candidate = tuple[
    int,
    int,
    Tensor,
    int | None,
    tuple[int, ...],
    InferenceLabel | None,
]

_CELL_COLUMN = {cell: column for column, cell in enumerate(InferenceCell)}


def _counts(batch: GroundingInput, /) -> int:
    pairs = batch.pairs
    counts = (
        len(pairs.premises),
        len(pairs.hypotheses),
        len(pairs.alignments),
    )

    if batch.query is GroundingQuery.OBSERVED_LABEL:
        counts = (*counts, len(pairs.labels))

    if len(set(counts)) != 1:
        msg = (
            "a grounding input has one premise, hypothesis, alignment set, and "
            f"label per pair, not counts {counts}"
        )
        raise ValueError(msg)

    if any(not alignments for alignments in pairs.alignments):
        msg = "each grounded pair declares at least one alignment"
        raise ValueError(msg)

    return counts[0]


def _forms(batch: GroundingInput, name: str, /) -> WeightedTerms:
    forms = batch.forms.get(name)

    if forms is None:
        offered = ", ".join(sorted(batch.forms)) or "nothing"
        msg = (
            f"pairwise grounding reads forms {name!r}, and the step published {offered}"
        )
        raise ValueError(msg)

    return forms


def _path(estimates: Sequence[Estimate], *, differentiable: bool) -> GradientPath:
    sampled = tuple(estimate is Estimate.SAMPLED for estimate in estimates)

    if sampled and any(sampled) and not all(sampled):
        msg = "one grounding batch cannot mix sampled and enumerated form estimates"
        raise ValueError(msg)

    if sampled and all(sampled):
        return GradientPath.MIXED if differentiable else GradientPath.SCORE_FUNCTION

    return GradientPath.PATHWISE


@final
class PairwiseGrounding(nn.Module):
    """Compose evidence over every retained form pair and alignment."""

    def __init__(
        self,
        store: TermStore,
        scorer: PairScorer[object],
        /,
        *,
        negation: Term,
        application: str,
        labels: tuple[InferenceLabel, ...],
        label_map: LabelMap,
        contradiction: str,
        composition: Composition,
        baseline: ControlVariate | None,
        reward: InferenceReward | None,
    ) -> None:
        super().__init__()

        if tuple(label_map.labels) != labels:
            msg = f"pairwise labels {labels!r} disagree with map {label_map.labels!r}"
            raise ValueError(msg)

        if contradiction not in CONTRADICTION_DEFINITIONS:
            msg = f"unknown contradiction definition {contradiction!r}"
            raise ValueError(msg)

        positive_only = (
            isinstance(scorer, PositiveQuestionScorer)
            and scorer.positive_questions_only
        )

        if positive_only:
            scorer.validate_labels(labels)

        self._store = store
        self._scorer = scorer
        self._negation = negation
        self._application = application
        self._labels = labels
        self._label_map = label_map
        self._contradiction = contradiction
        self._composition = checked_composition(composition)
        self._baseline = baseline
        self._reward = reward
        self._positive_questions_only = positive_only

    def _probability_baseline(self, query: GroundingQuery, /) -> float:
        """Return the declared baseline on the unit label-probability scale."""
        baseline = self._baseline

        if baseline is None:
            return 0.0

        reward = self._reward
        if query is GroundingQuery.OBSERVED_LABEL and reward is not None:
            return reward.probability_baseline(baseline.value)

        return baseline.value

    @property
    def labels(self) -> tuple[InferenceLabel, ...]:
        return self._labels

    @property
    def requires(self) -> frozenset[str]:
        return frozenset({"forms"})

    @property
    def scorer(self) -> PairScorer[object]:
        return self._scorer

    @property
    def composition(self) -> Composition:
        return self._composition

    @property
    def grounding_metadata(self) -> GroundingMetadata:
        """Return the scorer's stable proof identity and budget."""
        proof = self._scorer.proof_metadata
        return GroundingMetadata(
            "pairwise",
            scorer=self._scorer.scorer_kind,
            source=self._scorer.source_identity,
            prover=None if proof is None else proof.prover,
            prover_version=None if proof is None else proof.version,
            budget=None if proof is None else proof.budget,
        )

    def _negated(self, goal: Term, /) -> Term:
        operator = self._store.signature.by_name(self._application)
        return self._store.op(operator, self._negation, goal)

    def ground(self, batch: GroundingInput, /) -> GroundedBatch:
        """Score every distinct question once and compose per data pair."""
        count = _counts(batch)
        estimates: list[Estimate] = []
        candidates: list[_Candidate] = []
        questions: dict[tuple[Term, Term], int] = {}
        premises: list[Term] = []
        goals: list[Term] = []

        def question(premise: Term, goal: Term) -> int:
            key = (premise, goal)
            found = questions.get(key)

            if found is not None:
                return found

            position = len(questions)
            questions[key] = position
            premises.append(premise)
            goals.append(goal)
            return position

        for pair in range(count):
            left = _forms(batch, batch.pairs.premises[pair])
            right = _forms(batch, batch.pairs.hypotheses[pair])
            estimates.extend((left.estimate, right.estimate))
            target = (
                batch.pairs.labels[pair]
                if batch.query is GroundingQuery.OBSERVED_LABEL
                else None
            )

            for alignment_position, alignment in enumerate(
                batch.pairs.alignments[pair]
            ):
                renamed = tuple(
                    alignment.apply(self._store, term) for term in right.terms
                )

                for (left_position, premise), (right_position, goal) in product(
                    enumerate(left.terms), enumerate(renamed)
                ):
                    positive = None
                    negatives: tuple[int, ...] = ()

                    if (
                        batch.phase is Phase.EVALUATE
                        or batch.query is GroundingQuery.FULL
                        or target is not InferenceLabel.CONTRADICTS
                    ):
                        positive = question(premise, goal)

                    if not self._positive_questions_only and (
                        batch.phase is Phase.EVALUATE
                        or batch.query is GroundingQuery.FULL
                        or target is not InferenceLabel.ENTAILS
                    ):
                        negatives = (question(premise, self._negated(goal)),)

                        if self._contradiction == "joint-inconsistency":
                            negatives = (
                                *negatives,
                                question(goal, self._negated(premise)),
                            )

                    candidates.append(
                        (
                            pair,
                            alignment_position,
                            left.log_weights[left_position]
                            + right.log_weights[right_position],
                            positive,
                            negatives,
                            target,
                        )
                    )

        path = _path(estimates, differentiable=self._scorer.differentiable)

        if path is not GradientPath.PATHWISE and self._baseline is None:
            msg = "sampled pairwise grounding requires a control variate"
            raise ValueError(msg)

        scores = self._score(premises, goals)
        by_pair = self._compose(
            batch,
            candidates,
            scores,
            count=count,
            path=path,
        )
        probabilities, residual, decided, cells, variance = by_pair
        ledger = scores.ledger

        return GroundedBatch(
            LogDistribution(
                probabilities,
                ("pair", "label"),
                decided,
                normalized_over="label",
            ),
            labels=self._labels,
            complete=batch.phase is Phase.EVALUATE,
            residual=residual,
            decided=decided,
            cells=CellMass(cells),
            gradient_path=path,
            baseline=self._baseline,
            auxiliary=(
                self._scorer.auxiliary()
                if isinstance(self._scorer, AuxiliaryScorer)
                else None
            ),
            ledger=ledger,
            variance=variance,
            reward=self._reward,
        )

    def _score(self, premises: list[Term], goals: list[Term]) -> Scores[object]:
        if premises:
            return self._scorer.score(Questions(tuple(premises), tuple(goals)))

        return Scores(
            torch.empty(0),
            torch.empty(0, dtype=torch.bool),
            (),
            EvidenceLedger.empty(),
        )

    def _compose(
        self,
        batch: GroundingInput,
        candidates: list[_Candidate],
        scores: Scores[object],
        /,
        *,
        count: int,
        path: GradientPath,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None]:
        like = next(iter(batch.forms.values())).log_weights
        rows: list[Tensor] = []
        residuals: list[Tensor] = []
        decisions: list[bool] = []
        cell_rows: list[Tensor] = []
        variances: list[Tensor] = []

        for pair in range(count):
            pair_candidates = [
                candidate for candidate in candidates if candidate[0] == pair
            ]
            alignments = len(batch.pairs.alignments[pair])
            label_tables: list[list[PairTable]] = [[] for _ in self._labels]
            cell_tables: list[list[PairTable]] = [[] for _ in InferenceCell]
            observed: list[Tensor] = []

            for alignment in range(alignments):
                selected = [
                    candidate
                    for candidate in pair_candidates
                    if candidate[1] == alignment
                ]
                joint = (
                    torch.stack([candidate[2] for candidate in selected])
                    if selected
                    else like.new_empty((0,))
                )
                label_values: list[list[Tensor | None]] = [[] for _ in self._labels]
                cell_values: list[list[Tensor | None]] = [[] for _ in InferenceCell]

                for candidate in selected:
                    distribution = self._distribution(
                        scores,
                        positive=candidate[3],
                        negatives=candidate[4],
                        target=candidate[5],
                    )

                    if distribution is None:
                        for values in (*label_values, *cell_values):
                            values.append(None)
                        continue

                    labels, cells = distribution
                    observed.append(labels)

                    for values, value in zip(label_values, labels, strict=True):
                        values.append(value)

                    for values, value in zip(cell_values, cells, strict=True):
                        values.append(value)

                for tables, values in zip(label_tables, label_values, strict=True):
                    tables.append(self._table(joint, values))

                for tables, values in zip(cell_tables, cell_values, strict=True):
                    tables.append(self._table(joint, values))

            composed_labels = [
                compose(
                    tables,
                    composition=self._composition,
                    gradient_path=path,
                    baseline=self._probability_baseline(batch.query),
                )
                for tables in label_tables
            ]
            held_labels = [one.log_probability for one in composed_labels]
            live = all(value is not None for value in held_labels)
            decisions.append(live)

            if live:
                stacked = torch.stack(
                    [value for value in held_labels if value is not None]
                )
                rows.append(stacked - stacked.logsumexp(0))
                residuals.append(
                    torch.stack([one.residual for one in composed_labels]).max()
                )
                composed_cells = [
                    compose(
                        tables,
                        composition=self._composition,
                        gradient_path=path,
                        baseline=self._probability_baseline(batch.query),
                    ).log_probability
                    for tables in cell_tables
                ]
                cell_rows.append(
                    torch.stack(
                        [
                            like.new_zeros(())
                            if value is None or bool(value == -torch.inf)
                            else value.exp()
                            for value in composed_cells
                        ]
                    )
                )
            else:
                rows.append(
                    like.new_full((len(self._labels),), -log(len(self._labels)))
                )
                residuals.append(like.new_ones(()))
                cell_rows.append(like.new_zeros((len(InferenceCell),)))

            if observed:
                values = torch.stack(observed).detach()
                variances.append(values.var(dim=0, correction=0).mean())
            else:
                variances.append(like.new_zeros(()))

        variance = torch.stack(variances) if path is not GradientPath.PATHWISE else None
        return (
            torch.stack(rows),
            torch.stack(residuals),
            torch.tensor(decisions, dtype=torch.bool, device=like.device),
            torch.stack(cell_rows),
            variance,
        )

    def _distribution(
        self,
        scores: Scores[object],
        /,
        *,
        positive: int | None,
        negatives: tuple[int, ...],
        target: InferenceLabel | None,
    ) -> tuple[Tensor, Tensor] | None:
        like = scores.values

        if positive is not None and not bool(scores.decided[positive]):
            return None

        if any(not bool(scores.decided[negative]) for negative in negatives):
            return None

        half = like.new_tensor(0.5)
        entailed = half if positive is None else scores.values[positive]
        contradicted = (
            half
            if not negatives
            else torch.stack([scores.values[negative] for negative in negatives]).amin()
        )
        cells = torch.stack(
            (
                entailed * (1.0 - contradicted),
                (1.0 - entailed) * contradicted,
                (1.0 - entailed) * (1.0 - contradicted),
                entailed * contradicted,
            )
        )
        labels = torch.stack(
            [
                sum(
                    (
                        cells[_CELL_COLUMN[cell]]
                        for cell in self._label_map.cells[label]
                    ),
                    start=cells.new_zeros(()),
                )
                for label in self._labels
            ]
        )
        total = labels.sum()

        if bool(total.detach() <= 0.0):
            return None

        normalized = labels / total

        if target is InferenceLabel.NEUTRAL and (positive is None or not negatives):
            msg = "neutral training requires both entailment questions"
            raise RuntimeError(msg)

        return normalized, cells

    @staticmethod
    def _table(joint: Tensor, values: list[Tensor | None], /) -> PairTable:
        decided = torch.tensor(
            [value is not None for value in values],
            dtype=torch.bool,
            device=joint.device,
        )
        truth = (
            torch.stack(
                [
                    joint.new_zeros(()) if value is None else value.to(joint)
                    for value in values
                ]
            )
            if values
            else joint.new_empty((0,))
        )
        return PairTable(joint, truth, decided)

    def site_map(self) -> SiteMap:
        return SiteMap.custom(ComponentId("pairwise-grounding"), {})

    def open_step(self) -> None:
        """Open the scorer's next telemetry window without clearing its memo."""
        self._scorer.open_step()

    @override
    def __repr__(self) -> str:
        return (
            f"PairwiseGrounding(scorer={self._scorer!r}, "
            f"composition={self._composition!r})"
        )


def _negation(context: BuildContext, /) -> Term:
    """Return the negation designated by the language's logic declaration."""
    name = context.language.profile.negation

    if name is None:
        msg = "a pairwise grounding requires a designated negation"
        raise ValueError(msg)

    return context.language.constant(name)


def _control_variate(declaration: ControlVariateDeclaration, /) -> ControlVariate:
    """Return the control variate built by its declared plugin."""
    member = Registry[object]("control_variate").get(declaration.kind)

    if not isinstance(member, ControlVariatePlugin):
        msg = (
            f"the control_variate entry {declaration.kind!r} names {member!r}, "
            "not a control-variate plugin"
        )
        raise RegistryError(msg, family="control_variate")

    return member.construct(declaration.settings)


def _build(
    settings: PairwiseGroundingSettings, context: BuildContext, /
) -> PairwiseGrounding:
    """Build pairwise evidence composition over the declared form readouts."""
    scorer = context.producer("scorer")
    labeling = context.producer("labeling")

    if not isinstance(scorer, PairScorer):
        msg = f"a pairwise grounding reads a PairScorer, not {scorer!r}"
        raise TypeError(msg)

    if not isinstance(labeling, InferenceLabeling):
        msg = f"a pairwise grounding reads InferenceLabeling, not {labeling!r}"
        raise TypeError(msg)

    return PairwiseGrounding(
        context.language.store,
        cast("PairScorer[object]", scorer),
        negation=_negation(context),
        application=context.language.profile.application,
        labels=labeling.labels,
        label_map=labeling.label_map(context.language.profile.label_map),
        contradiction=(
            "negated-hypothesis"
            if settings.reward is None
            else settings.reward.contradiction
        ),
        composition=composition_from_settings(*settings.composition),
        baseline=None
        if settings.control_variate is None
        else _control_variate(settings.control_variate),
        reward=settings.reward,
    )


PLUGIN: Final = ComponentPlugin(spec=PairwiseGroundingSpec, build=_build)
"""The single pairwise inference-grounding plugin."""

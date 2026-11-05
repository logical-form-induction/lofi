"""Budgeted three-way reproduction of retained inference forms."""

from heapq import nsmallest
from itertools import product
from typing import TYPE_CHECKING, NamedTuple

from lofi.lf import Decided, InferenceCell
from lofi.lf.provers import ProofService, ProofSite, Questions
from lofi.terms import normalize

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.data.synthetic import InterpretationFunction
    from lofi.lf.provers import ProofBudget
    from lofi.nn.forms import WeightedTerms
    from lofi.spec.evaluators.inference import InferenceProofSettings
    from lofi.spec.language import LanguageAssembly
    from lofi.terms import Normalization, Term

__all__ = [
    "InferenceProof",
    "InferenceProofBatch",
    "open_inference_proof_step",
    "prove_inference_beam",
    "prove_inference_pairs",
]

_SITE = ProofSite("evaluation.inference")


def open_inference_proof_step(service: ProofService, /) -> None:
    """Open the inference evaluator's next proof-service budget window."""
    service.open_step(_SITE)


class InferenceProof(NamedTuple):
    """Reference and ranked candidate cells beside proof-service telemetry."""

    reference: InferenceCell | None
    candidates: tuple[InferenceCell | None, ...]
    calls: int
    memo_hits: int
    cpu_seconds: float
    questions: int
    wall_seconds: float
    abstentions: tuple[tuple[str, int], ...]


class InferenceProofBatch(NamedTuple):
    """Ordered four-cell decisions beside proof-service telemetry."""

    cells: tuple[InferenceCell | None, ...]
    calls: int
    memo_hits: int
    cpu_seconds: float
    questions: int
    wall_seconds: float
    abstentions: tuple[tuple[str, int], ...]


def prove_inference_beam(
    interpretation: InterpretationFunction,
    settings: InferenceProofSettings,
    reference: tuple[Term, Term],
    premise: WeightedTerms,
    hypothesis: WeightedTerms,
    /,
) -> InferenceProof:
    """Prove both polarities of the reference and ranked joint candidates.

    Joint candidates are ordered by the sum of their published log weights,
    with source beam order breaking ties. A partial proof remains undecided;
    proving both polarities retains the degenerate cell explicitly.
    """
    left = premise.log_weights.detach().tolist()
    right = hypothesis.log_weights.detach().tolist()
    ranked = nsmallest(
        settings.depth,
        product(range(len(premise)), range(len(hypothesis))),
        key=lambda pair: -(left[pair[0]] + right[pair[1]]),
    )
    pairs = (
        reference,
        *((premise.terms[p], hypothesis.terms[h]) for p, h in ranked),
    )
    proved = prove_inference_pairs(
        interpretation.language,
        interpretation.service,
        settings.budget,
        pairs,
        normalization=interpretation.normalization,
    )
    return InferenceProof(proved.cells[0], proved.cells[1:], *proved[1:])


def prove_inference_pairs(
    language: LanguageAssembly,
    service: ProofService,
    budget: ProofBudget,
    pairs: Sequence[tuple[Term, Term]],
    /,
    *,
    normalization: Normalization,
) -> InferenceProofBatch:
    """Decide both polarities for ordered pairs through one budgeted service.

    Deduplicate normalized questions before dispatch, then restore their original
    pair order. Preserve abstentions and both-proved cells without coercion.
    The caller opens the evaluation site's step boundary once per evaluation step.
    """
    negation_name = language.profile.negation
    if negation_name is None:
        msg = "inference proof evaluation needs a designated negation"
        raise ValueError(msg)
    negation = language.constant(negation_name)
    application = language.store.signature.by_name(language.calculus.app)
    premises: list[Term] = []
    goals: list[Term] = []
    for source, target in pairs:
        held_source = normalize(
            language.store,
            source,
            language.calculus,
            normalization=normalization,
        )
        held_target = normalize(
            language.store,
            target,
            language.calculus,
            normalization=normalization,
        )
        negative = language.store.op(application, negation, held_target)
        premises.extend((held_source, held_source))
        goals.extend((held_target, negative))
    before = service.step_report(_SITE)
    questions = Questions(tuple(premises), tuple(goals))
    verdicts = service.decide(questions, budget=budget, site=_SITE)
    settled = dict(
        zip(
            zip(questions.premises, questions.goals, strict=True), verdicts, strict=True
        )
    )
    expanded = tuple(settled[pair] for pair in zip(premises, goals, strict=True))
    after = service.step_report(_SITE)
    cells: list[InferenceCell | None] = []
    for index in range(0, len(expanded), 2):
        decided = Decided.from_verdicts(expanded[index], expanded[index + 1])
        cells.append(next(iter(decided.cells)) if decided.complete else None)
    return InferenceProofBatch(
        tuple(cells),
        after.calls - before.calls,
        after.memo_hits - before.memo_hits,
        after.cpu_seconds - before.cpu_seconds,
        after.questions - before.questions,
        after.wall_seconds - before.wall_seconds,
        tuple(
            (cause, count - before.abstentions[cause])
            for cause, count in after.abstentions.items()
        ),
    )

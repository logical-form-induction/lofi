"""Certified pair scores dispatched through the shared proof service."""

from typing import TYPE_CHECKING, Final, final

import torch

from lofi._frozen import FrozenValue, Unpicklable
from lofi.lf import (
    Abstain,
    Graded,
    LogicFragment,
    Proved,
    Refuted,
    Verdict,
)
from lofi.lf.models import CountermodelFilter
from lofi.lf.provers import ProofService, ProofSite, Questions
from lofi.nn.grounding._scorer import Scores
from lofi.nn.grounding._values import EvidenceLedger, ProofMetadata
from lofi.registry import ComponentPlugin
from lofi.spec.language import LanguageAssembly
from lofi.spec.scorers import ProverScorerSettings, ProverScorerSpec
from lofi.terms import Normalization, normalize

if TYPE_CHECKING:
    from lofi.lf.provers import ProofBudget
    from lofi.nn.build import BuildContext
    from lofi.terms import Term

__all__ = ["PLUGIN", "ProverScorer"]


@final
class ProverScorer(FrozenValue, Unpicklable):
    """Turn proof service verdicts into certified zero-one pair scores."""

    _noun = "prover scorer"
    _because = "holds a live proof service"
    __slots__ = (
        "_budget",
        "_countermodels",
        "_language",
        "_normalization",
        "_service",
        "_site",
    )

    def __init__(
        self,
        service: ProofService,
        budget: ProofBudget,
        language: LanguageAssembly,
        normalization: Normalization,
        site: ProofSite,
        /,
        *,
        countermodels: CountermodelFilter | None,
    ) -> None:
        object.__setattr__(self, "_service", service)
        object.__setattr__(self, "_budget", budget)
        object.__setattr__(self, "_language", language)
        object.__setattr__(self, "_normalization", normalization)
        object.__setattr__(self, "_site", site)
        object.__setattr__(self, "_countermodels", countermodels)

    @property
    def fragment(self) -> LogicFragment:
        return self._language.profile.fragment

    @property
    def scorer_kind(self) -> str:
        """Return the registered evidence-scoring treatment."""
        return "prover"

    @property
    def source_identity(self) -> str:
        """Return the declared prover supplying the evidence."""
        return self._service.prover

    @property
    def differentiable(self) -> bool:
        return False

    @property
    def certified(self) -> bool:
        return True

    @property
    def proof_metadata(self) -> ProofMetadata:
        """Return the proof identity and use-site budget this scorer owns."""
        return ProofMetadata(
            self._service.prover,
            self._budget,
            version=None,
        )

    @property
    def service(self) -> ProofService:
        return self._service

    @property
    def budget(self) -> ProofBudget:
        return self._budget

    @property
    def normalization(self) -> Normalization:
        """Return the interpretation's exact normalization treatment."""
        return self._normalization

    def _normalize(self, term: Term) -> Term:
        """Normalize one logical form within the declared scorer budget."""
        return normalize(
            self._language.store,
            term,
            self._language.calculus,
            normalization=self._normalization,
        )

    def score(self, questions: Questions, /) -> Scores[Verdict]:
        """Dispatch a proof step and score only proved and refuted questions."""
        theory = tuple(self._normalize(term) for term in questions.theory)
        pairs = tuple(
            (self._normalize(premise), self._normalize(goal))
            for premise, goal in zip(
                questions.premises,
                questions.goals,
                strict=True,
            )
        )
        unique_pairs = tuple(dict.fromkeys(pairs))
        filtered: dict[tuple[Term, Term], Verdict | None] = {}
        pending_premises = []
        pending_goals = []

        for premise, goal in unique_pairs:
            countermodel = (
                None
                if self._countermodels is None
                else self._countermodels.refute(
                    premise,
                    goal,
                    theory=theory,
                )
            )

            if countermodel is None:
                filtered[(premise, goal)] = None
                pending_premises.append(premise)
                pending_goals.append(goal)
            else:
                filtered[(premise, goal)] = Refuted(countermodel)

        pending = Questions(
            tuple(pending_premises),
            tuple(pending_goals),
            theory=theory,
        )
        proved = (
            ()
            if not pending.premises
            else self._service.decide(pending, budget=self._budget, site=self._site)
        )
        proof_at = iter(proved)
        settled = {
            pair: next(proof_at) if item is None else item
            for pair, item in filtered.items()
        }
        evidence = tuple(settled[pair] for pair in pairs)
        values = torch.tensor(
            [
                item.degree
                if isinstance(item, Graded)
                else 1.0
                if isinstance(item, Proved)
                else 0.0
                for item in evidence
            ],
            dtype=torch.float64,
        )
        decided = torch.tensor(
            [isinstance(item, Proved | Refuted | Graded) for item in evidence],
            dtype=torch.bool,
        )

        if any(
            not isinstance(item, Proved | Refuted | Graded | Abstain)
            for item in evidence
        ):
            msg = "a proof service returned something other than a Verdict"
            raise TypeError(msg)

        return Scores(
            values,
            decided,
            evidence,
            EvidenceLedger.from_service(
                evidence,
                self._service.step_report(self._site),
            ),
        )

    def open_step(self) -> None:
        """Reset only the proof service's step telemetry."""
        self._service.open_step(self._site)


def _build(settings: ProverScorerSettings, context: BuildContext, /) -> ProverScorer:
    """Build certified scoring from the run's shared proof service."""
    service = context.producer("service")
    normalization = context.producer("normalization")
    language = context.language

    if not isinstance(service, ProofService):
        msg = f"a prover scorer reads a ProofService, not {service!r}"
        raise TypeError(msg)

    if not isinstance(language, LanguageAssembly):
        msg = f"a prover scorer reads a LanguageAssembly, not {language!r}"
        raise TypeError(msg)

    if not isinstance(normalization, Normalization):
        msg = f"a prover scorer reads a Normalization, not {normalization!r}"
        raise TypeError(msg)

    countermodels = (
        None
        if settings.countermodels is None
        else CountermodelFilter(language, settings.countermodels)
    )
    return ProverScorer(
        service,
        settings.budget,
        language,
        normalization,
        ProofSite(str(context.component)),
        countermodels=countermodels,
    )


PLUGIN: Final = ComponentPlugin(spec=ProverScorerSpec, build=_build)
"""The certified proof service pair scorer plugin."""

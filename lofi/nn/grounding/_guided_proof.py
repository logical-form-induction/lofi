"""Encoder-guided signed-tableau proof scoring with exact verification."""

from collections import Counter
from math import sqrt
from typing import TYPE_CHECKING, Final, cast, final, override

import torch
from torch import nn

from lofi.lf import (
    RTE,
    Abstain,
    InferenceLabel,
    LogicFragment,
    Proved,
    SearchBound,
    SequentCalculus,
    SequentLimits,
)
from lofi.lf.provers import ProofService, ProofSite, Questions
from lofi.nn.grounding._scorer import ProofSearch, Scores
from lofi.nn.grounding._values import EvidenceLedger, ProofMetadata
from lofi.nn.policy import place
from lofi.nn.ports import TermEncoder
from lofi.nn.sites import ComponentId, SiteMap
from lofi.nn.spaces import Space
from lofi.registry import ComponentPlugin
from lofi.spec.groundings.guided_proof import (
    GuidedProofScorerSettings,
    GuidedProofScorerSpec,
)
from lofi.spec.language import LanguageAssembly
from lofi.terms import Normalization, normalize

if TYPE_CHECKING:
    from torch import Tensor

    from lofi.lf import SequentProof, Verdict
    from lofi.lf.provers import ProofBudget
    from lofi.nn.build import BuildContext
    from lofi.nn.codec import TermCoding
    from lofi.terms import Term

__all__ = ["PLUGIN", "GuidedProofScorer"]


@final
class GuidedProofScorer(nn.Module):
    """Learn a ranking over declared proof steps and verify found proofs."""

    _codec: TermCoding
    _encoder: TermEncoder

    def __init__(
        self,
        service: ProofService,
        budget: ProofBudget,
        language: LanguageAssembly,
        normalization: Normalization,
        encoder: TermEncoder,
        codec: TermCoding,
        component: ComponentId,
        /,
        *,
        order: str,
        limits: SequentLimits,
        hidden: int,
    ) -> None:
        super().__init__()
        calculus = SequentCalculus(language, normalization, order=order)
        object.__setattr__(self, "_encoder", encoder)
        self._service = service
        self._budget = budget
        self._site = ProofSite(str(component))
        self._language = language
        self._normalization = normalization
        self._codec = codec
        self._limits = limits
        self._calculus = calculus
        self._rule_names = calculus.rules
        self._rule_index = {name: index for index, name in enumerate(self._rule_names)}
        self._focus = nn.Linear(encoder.output_width, hidden)
        self._rules = nn.Embedding(len(self._rule_names), hidden)
        self._sites = SiteMap.custom(component, {})
        fragment = language.profile.fragment
        self._fragment = LogicFragment(
            order=order,
            quantifiers=fragment.quantifiers,
            connectives=fragment.connectives,
            functions=fragment.functions,
            equality=fragment.equality,
        )

    @property
    def scorer_kind(self) -> str:
        return "guided-proof"

    @property
    def source_identity(self) -> str:
        return self._service.prover

    @property
    def fragment(self) -> LogicFragment:
        return self._fragment

    @property
    def differentiable(self) -> bool:
        return True

    @property
    def certified(self) -> bool:
        return False

    @property
    def proof_metadata(self) -> ProofMetadata:
        return ProofMetadata(self._service.prover, self._budget, version=None)

    @property
    def positive_questions_only(self) -> bool:
        return True

    def validate_labels(self, labels: tuple[InferenceLabel, ...], /) -> None:
        """Require the exact binary inventory parameterized by one question."""
        if labels != RTE.labels:
            names = tuple(label.value for label in labels)
            expected = tuple(label.value for label in RTE.labels)
            msg = (
                f"guided-proof supports only binary labels {expected!r}, not {names!r}"
            )
            raise ValueError(msg)

    def site_map(self) -> SiteMap:
        """Return this scorer's component-scoped instrumentation surface."""
        return self._sites

    def open_step(self) -> None:
        """Open the proof service's next telemetry window."""
        self._service.open_step(self._site)

    def _normalize(self, term: Term, /) -> Term:
        return normalize(
            self._language.store,
            term,
            self._language.calculus,
            normalization=self._normalization,
        )

    def _rule_code(self, name: str, /) -> int:
        if name.startswith("instantiate:"):
            name = ":".join(name.split(":", maxsplit=2)[:2])
        return self._rule_index[name]

    def _proof_score(self, proof: SequentProof, /) -> Tensor:
        foci = tuple(step.focus for step in proof.steps)
        encoded = self._encoder.encode(self._codec.encode_batch(foci)).tensor
        rule_codes = torch.tensor(
            [self._rule_code(step.rule) for step in proof.steps],
            dtype=torch.int64,
            device=encoded.device,
        )
        focus = torch.tanh(self._focus(encoded))
        rules = self._rules(rule_codes)
        steps = torch.sigmoid((focus * rules).sum(dim=-1) / sqrt(rules.shape[-1]))
        return steps.prod()

    def _search_score(self, proofs: tuple[SequentProof, ...], /) -> Tensor:
        if not proofs:
            return (self._focus.weight.sum() + self._rules.weight.sum()) * 0.0
        values = torch.stack([self._proof_score(proof) for proof in proofs])
        return 1.0 - (1.0 - values).prod()

    def score(self, questions: Questions, /) -> Scores[ProofSearch]:
        """Rank bounded proofs and recheck each found question exactly."""
        theory = tuple(self._normalize(term) for term in questions.theory)
        pairs = tuple(
            (self._normalize(premise), self._normalize(goal))
            for premise, goal in zip(
                questions.premises,
                questions.goals,
                strict=True,
            )
        )
        searches = tuple(
            self._calculus.search(
                premise,
                goal,
                theory=theory,
                limits=self._limits,
            )
            for premise, goal in pairs
        )
        values = tuple(self._search_score(search.proofs) for search in searches)
        found = tuple(bool(search.proofs) for search in searches)
        pending = tuple(position for position, held in enumerate(found) if held)
        pending_pairs = tuple(dict.fromkeys(pairs[position] for position in pending))
        verification = (
            ()
            if not pending_pairs
            else self._service.decide(
                Questions(
                    tuple(pair[0] for pair in pending_pairs),
                    tuple(pair[1] for pair in pending_pairs),
                    theory=theory,
                ),
                budget=self._budget,
                site=self._site,
            )
        )
        by_pair = dict(zip(pending_pairs, verification, strict=True))
        checked: dict[int, Verdict] = {
            position: by_pair[pairs[position]] for position in pending
        }
        evidence: list[ProofSearch] = []

        for position, (search, value) in enumerate(zip(searches, values, strict=True)):
            if not search.supported:
                verdict: Verdict = Abstain(
                    cast("str", search.reason),
                    cause="out-of-fragment",
                )
            elif not search.proofs:
                resource = "proof-steps" if search.exhausted else "proof-depth"
                bound = self._limits.steps if search.exhausted else self._limits.depth
                verdict = Abstain(
                    "the declared guided proof search found no closed tableau",
                    cause="search-bound",
                    bound=SearchBound(resource, bound),
                )
            else:
                verdict = checked[position]
            evidence.append(
                ProofSearch(
                    verdict,
                    found=found[position],
                    verified=isinstance(verdict, Proved),
                    score=float(value.detach()),
                )
            )

        report = self._service.step_report(self._site)
        abstentions = Counter(
            item.verdict.cause for item in evidence if isinstance(item.verdict, Abstain)
        )
        discovered = sum(item.found for item in evidence)
        verified = sum(item.verified for item in evidence)
        tensor = torch.stack(values) if values else self._focus.weight.new_empty((0,))
        supported = torch.tensor(
            [search.supported for search in searches],
            dtype=torch.bool,
            device=tensor.device,
        )
        return Scores(
            tensor,
            supported,
            tuple(evidence),
            EvidenceLedger(
                {"proofsearch": len(evidence)},
                abstentions,
                calls=report.calls,
                memo_hits=report.memo_hits,
                cpu_seconds=report.cpu_seconds,
                questions=len(evidence),
                wall_seconds=report.wall_seconds,
                process_starts=report.process_starts,
                verified_share=(0.0 if discovered == 0 else verified / discovered),
            ),
        )

    @override
    def __repr__(self) -> str:
        return (
            f"GuidedProofScorer(order={self._fragment.order!r}, "
            f"rules={len(self._rule_names)!r})"
        )


def _build(
    settings: GuidedProofScorerSettings,
    context: BuildContext,
    /,
) -> GuidedProofScorer:
    """Build guided proof search from the declared LF readout's public space."""
    service = context.producer("service")
    normalization = context.producer("normalization")
    produced = context.producer(settings.readout)
    language = context.language

    if not isinstance(service, ProofService):
        msg = f"a guided proof scorer reads ProofService, not {service!r}"
        raise TypeError(msg)
    if not isinstance(language, LanguageAssembly):
        msg = f"a guided proof scorer reads LanguageAssembly, not {language!r}"
        raise TypeError(msg)
    if not isinstance(normalization, Normalization):
        msg = f"a guided proof scorer reads Normalization, not {normalization!r}"
        raise TypeError(msg)
    if not isinstance(produced, Space):
        msg = f"a guided proof scorer reads a represented Space, not {produced!r}"
        raise TypeError(msg)
    if produced.sort != language.term_sort:
        msg = (
            f"guided proof reads logical-form sort {language.term_sort!r}, "
            f"not represented sort {produced.sort!r}"
        )
        raise ValueError(msg)
    if not isinstance(produced.encoder, TermEncoder):
        msg = f"a guided proof scorer reads a TermEncoder, not {produced.encoder!r}"
        raise TypeError(msg)

    return place(
        GuidedProofScorer(
            service,
            settings.verification,
            language,
            normalization,
            produced.encoder,
            produced.codec.codec,
            context.component,
            order=settings.order,
            limits=settings.limits,
            hidden=settings.hidden,
        ),
        context.policy,
    )


PLUGIN: Final = ComponentPlugin(spec=GuidedProofScorerSpec, build=_build)
"""The single guided symbolic pair-scorer plugin."""

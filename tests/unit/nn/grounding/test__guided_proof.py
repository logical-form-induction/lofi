from typing import TYPE_CHECKING, cast

import torch
from tests.verdicts import proved
from torch import nn

from lofi.lf import (
    RTE,
    Abstain,
    Proved,
    SequentLimits,
)
from lofi.lf.provers import (
    ProofBudget,
    ProofService,
    ProofSite,
    Questions,
    ServiceReport,
)
from lofi.nn._values import Encoded
from lofi.nn.grounding import GuidedProofScorer, ProofSearch
from lofi.nn.sites import ComponentId, SiteMap
from lofi.spec.language import ConstantSpec, LanguageIso, LogicSpec, registered_language
from lofi.terms import BatchEncoding, Normalization

if TYPE_CHECKING:
    from lofi.lf import Verdict
    from lofi.spec.language import LanguageAssembly


NORMALIZATION = Normalization(
    fuel=32,
    order="leftmost-outermost",
    eta="contract",
    associative_commutative=(),
)
LIMITS = SequentLimits(steps=64, depth=8, top_k=4, instantiations=8)


def _language() -> LanguageAssembly:
    source = registered_language("neo-davidsonian")
    return LanguageIso().forward(
        source.with_(
            name="guided-second-order",
            constants=(
                *source.constants,
                ConstantSpec(name="dog", type="e -> t"),
                ConstantSpec(name="mammal", type="e -> t"),
                ConstantSpec(name="rex", type="e"),
                ConstantSpec(name="fido", type="e"),
            ),
            logic=LogicSpec(order="second"),
        )
    )


DECLARED = _language()


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(4))
        self.calls: list[BatchEncoding] = []

    @property
    def output_width(self) -> int:
        return 4

    def encode(self, terms: BatchEncoding, /) -> Encoded:
        self.calls.append(terms)
        rows = len(terms.rows)
        return Encoded(self.scale.unsqueeze(0).expand(rows, -1), ("term", "width"))

    def encode_nodes(self, terms: BatchEncoding, /) -> Encoded:
        rows = len(terms.rows)
        width = terms.width
        return Encoded(
            self.scale.reshape(1, 1, -1).expand(rows, width, -1),
            ("term", "slot", "width"),
        )

    def site_map(self) -> SiteMap:
        return SiteMap.custom(ComponentId("forms-encoder"), {})


class Service:
    def __init__(self, *, proves: bool = True) -> None:
        self.received: Questions | None = None
        self.proves = proves

    @property
    def prover(self) -> str:
        return "thf-test"

    def step_report(self, site: ProofSite, /) -> ServiceReport:
        del site
        questions = 0 if self.received is None else len(self.received.premises)
        return ServiceReport(questions, 0, {}, 0.25, questions, 0.5, None)

    def decide(
        self,
        questions: Questions,
        /,
        *,
        budget: ProofBudget,
        site: ProofSite,
    ) -> tuple[Verdict, ...]:
        del budget, site
        self.received = questions
        return tuple(
            proved("thf-checked")
            if self.proves
            else Abstain("THF rejected the candidate", cause="gave-up")
            for _ in questions.premises
        )

    def open_step(self, site: ProofSite, /) -> None:
        del site
        self.received = None


def scorer(
    *, order: str = "first", proves: bool = True
) -> tuple[GuidedProofScorer, Encoder, Service]:
    encoder = Encoder()
    service = Service(proves=proves)
    checked = GuidedProofScorer(
        cast("ProofService", service),
        ProofBudget(calls_per_step=8),
        DECLARED,
        NORMALIZATION,
        encoder,
        DECLARED.codec.at(DECLARED.term_sort),
        ComponentId("guided-proof-test"),
        order=order,
        limits=LIMITS,
        hidden=6,
    )
    return checked, encoder, service


def test_guided_steps_read_the_published_lf_encoder_and_are_pathwise() -> None:
    checked, encoder, service = scorer()
    premise = DECLARED.read("(forall x. dog(x) => mammal(x)) & dog(rex)")
    goal = DECLARED.read("mammal(rex)")

    scores = checked.score(Questions((premise,), (goal,)))
    scores.values.sum().backward()

    evidence = scores.evidence[0]
    assert isinstance(evidence, ProofSearch)
    assert evidence.found
    assert evidence.verified
    assert isinstance(evidence.verdict, Proved)
    assert encoder.calls
    assert service.received == Questions((premise,), (goal,))
    assert scores.values.requires_grad
    assert any(parameter.grad is not None for parameter in checked.parameters())


def test_guided_second_order_instantiates_predicates_and_rechecks_with_thf() -> None:
    checked, _, service = scorer(order="second")
    premise = DECLARED.read("(forall P:(e -> t). P(rex) => P(fido)) & dog(rex)")
    goal = DECLARED.read("dog(fido)")

    scores = checked.score(Questions((premise,), (goal,)))

    assert scores.evidence[0].found
    assert scores.evidence[0].verified
    assert service.received == Questions((premise,), (goal,))
    assert scores.ledger.verified_share == 1.0


def test_guided_first_order_refuses_the_same_predicate_binder() -> None:
    checked, encoder, service = scorer(order="first")
    premise = DECLARED.read("forall P:(e -> t). P(rex) => P(fido)")

    scores = checked.score(Questions((premise,), (DECLARED.read("dog(fido)"),)))

    evidence = scores.evidence[0]
    assert isinstance(evidence.verdict, Abstain)
    assert evidence.verdict.cause == "out-of-fragment"
    assert "first" in str(evidence.verdict)
    assert "second" in str(evidence.verdict)
    assert not encoder.calls
    assert service.received is None


def test_guided_second_order_refuses_a_nonpattern_application() -> None:
    checked, encoder, service = scorer(order="second")
    premise = DECLARED.read("forall F:(t -> t). F(F(dog(rex)))")

    scores = checked.score(Questions((premise,), (DECLARED.read("dog(rex)"),)))

    evidence = scores.evidence[0]
    assert isinstance(evidence.verdict, Abstain)
    assert evidence.verdict.cause == "out-of-fragment"
    assert not evidence.found
    assert not encoder.calls
    assert service.received is None


def test_guided_proof_refuses_a_quantifier_without_an_abstraction() -> None:
    checked, encoder, service = scorer(order="second")
    quantified_constant = DECLARED.apply(
        DECLARED.constant("forall"), DECLARED.constant("dog")
    )

    scores = checked.score(Questions((quantified_constant,), (quantified_constant,)))

    evidence = scores.evidence[0]
    assert isinstance(evidence.verdict, Abstain)
    assert evidence.verdict.cause == "out-of-fragment"
    assert not evidence.found
    assert not encoder.calls
    assert service.received is None


def test_unverified_guided_findings_lower_the_verified_share() -> None:
    checked, _, service = scorer(proves=False)
    premise = DECLARED.read("dog(rex)")
    goal = DECLARED.read("dog(rex)")

    scores = checked.score(Questions((premise,), (goal,)))

    assert scores.evidence[0].found
    assert not scores.evidence[0].verified
    assert service.received == Questions((premise,), (goal,))
    assert scores.ledger.verified_share == 0.0


def test_guided_proof_accepts_only_binary_labels() -> None:
    checked, _, _ = scorer()

    checked.validate_labels(RTE.labels)

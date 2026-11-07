"""Differentiable type-directed proof search with discrete verification."""

from collections import Counter
from typing import TYPE_CHECKING, Final, Literal, final, override

import torch
from torch import nn

from lofi.lf import (
    RTE,
    Abstain,
    InferenceLabel,
    LogicFragment,
    Proved,
    SearchBound,
)
from lofi.lf.provers import ProofService, ProofSite, Questions
from lofi.nn.embeddings import ChoiceEmbedding
from lofi.nn.grounding._scorer import ProofSearch, Scores
from lofi.nn.grounding._similarity import similarity_kernel
from lofi.nn.grounding._values import EvidenceLedger, ProofMetadata
from lofi.nn.policy import place
from lofi.registry import ComponentPlugin
from lofi.spec.groundings.neural_proof import (
    NeuralProofScorerSettings,
    NeuralProofScorerSpec,
    RuleTemplate,
    SimilarityKernelSettings,
)
from lofi.spec.language import LanguageAssembly
from lofi.terms import ChoiceIndex, Normalization, instantiate, normalize

if TYPE_CHECKING:
    from torch import Tensor

    from lofi.lf import Verdict
    from lofi.lf.provers import ProofBudget
    from lofi.nn.build import BuildContext
    from lofi.nn.sites import ComponentId, SiteMap
    from lofi.terms import Term

__all__ = ["PLUGIN", "NeuralProofScorer"]

type _Search = tuple[Tensor, bool]
type _Position = Literal["goal", "premise", "term"]


@final
class NeuralProofScorer(nn.Module):
    """Score bounded backward chains and verify their discrete proofs."""

    def __init__(
        self,
        service: ProofService,
        budget: ProofBudget,
        language: LanguageAssembly,
        normalization: Normalization,
        choices: ChoiceIndex,
        component: ComponentId,
        /,
        *,
        order: str,
        instantiations: int,
        templates: tuple[RuleTemplate, ...],
        depth: int,
        width: int,
        similarity: SimilarityKernelSettings,
    ) -> None:
        super().__init__()
        self._service = service
        self._budget = budget
        self._site = ProofSite(str(component))
        self._language = language
        self._normalization = normalization
        self._order = order
        self._instantiations = instantiations
        self._templates = frozenset(templates)
        self._depth = depth
        self._kernel = similarity_kernel(similarity.kind)
        self._temperature = similarity.temperature
        self._embedding = ChoiceEmbedding(
            index=choices,
            width=width,
            component=component,
        )
        allowed_roles = self._allowed_roles()
        connectives = tuple(
            name
            for name, role in language.profile.logical_roles.items()
            if role.name in allowed_roles
        )
        self._fragment = LogicFragment(
            order=order,
            quantifiers=(
                tuple(language.profile.fragment.quantifiers)
                if order == "second"
                else ()
            ),
            connectives=connectives,
            functions=True,
            equality=False,
        )

    def _allowed_roles(self) -> frozenset[str]:
        roles: set[str] = set()

        if {
            "conjunction-elimination",
            "conjunction-introduction",
        } & self._templates:
            roles.add("conjunction")

        if "modus-ponens" in self._templates:
            roles.add("conditional")

        return frozenset(roles)

    @property
    def scorer_kind(self) -> str:
        return "neural-proof"

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
        """Return true because RTE nonentailment is the positive complement."""
        return True

    def validate_labels(self, labels: tuple[InferenceLabel, ...], /) -> None:
        """Require the exact binary RTE inventory this score parameterizes."""
        if labels != RTE.labels:
            names = tuple(label.value for label in labels)
            expected = tuple(label.value for label in RTE.labels)
            msg = (
                f"neural-proof supports only binary labels {expected!r}, not {names!r}"
            )
            raise ValueError(msg)

    def site_map(self) -> SiteMap:
        """Publish only the shared choice-embedding parameter site."""
        return self._embedding.site_map()

    def open_step(self) -> None:
        """Open the verifier's next telemetry window."""
        self._service.open_step(self._site)

    def _normalize(self, term: Term, /) -> Term:
        return normalize(
            self._language.store,
            term,
            self._language.calculus,
            normalization=self._normalization,
        )

    def _logical(
        self,
        term: Term,
        /,
    ) -> tuple[str, tuple[Term, ...]] | None:
        head, arguments = self._language.spine(term)

        if not head.op.is_constant:
            return None

        name = self._language.store.payload_name(head)
        role = self._language.profile.logical_roles.get(name)

        if role is None or len(arguments) != role.arity:
            return None

        return role.name, arguments

    def _is_formula(self, term: Term, /) -> bool:
        inferred = self._language.type_of(term)
        return (
            inferred.op.is_constant
            and self._language.store.payload_name(inferred)
            in self._language.profile.proposition_types
        )

    def _abstraction(self, term: Term, /) -> tuple[Term, Term] | None:
        logical = self._logical(term)

        if logical is None:
            return None

        role, arguments = logical

        if role != "universal" or len(arguments) != 1:
            return None

        abstraction = arguments[0]

        if not abstraction.op.is_binder or len(abstraction.op.binding) != 1:
            return None

        site = abstraction.op.binding[0]

        if len(site.arguments) != 1:
            return None

        abstraction_type = self._language.type_of(abstraction)

        if (
            abstraction_type.op.name != self._language.function_type
            or len(abstraction_type.args) != 2
        ):
            return None

        return abstraction.args[site.arguments[0]], abstraction_type.args[0]

    def _type_order(self, term: Term, /) -> int:
        if term.op.name != self._language.function_type or len(term.args) != 2:
            return 0

        return max(
            self._type_order(term.args[0]) + 1,
            self._type_order(term.args[1]),
        )

    def _contains_bound(self, term: Term, index: int, /) -> bool:
        return (term.op.is_bound_variable and term.payload == index) or any(
            self._contains_bound(argument, index) for argument in term.args
        )

    def _pattern_body(self, body: Term, /) -> bool:
        head, arguments = self._language.spine(body)

        if head.op.is_bound_variable and head.payload == 0:
            return bool(arguments) and all(
                not self._contains_bound(argument, 0) for argument in arguments
            )

        if body.op.is_bound_variable and body.payload == 0:
            return False

        return all(self._pattern_body(argument) for argument in body.args)

    def _supported(self, term: Term, /, *, goal: bool) -> bool:
        if not self._is_formula(term):
            return False

        def visit(node: Term, /, *, position: _Position) -> bool:
            abstraction = self._abstraction(node)

            if abstraction is not None:
                body, quantified_type = abstraction
                return (
                    self._order == "second"
                    and position == "premise"
                    and self._type_order(quantified_type) == 1
                    and self._pattern_body(body)
                )

            if (
                node.op.is_binder
                or node.op.is_variable
                or node.op.is_hole
                or not node.is_closed
                or (position == "term" and self._is_formula(node))
                or (
                    node.op.name == self._language.calculus.app
                    and len(node.args) == 2
                    and self._language.type_of(node.args[1]).op.name
                    == self._language.function_type
                )
            ):
                return False

            logical = self._logical(node)

            if logical is not None:
                role, arguments = logical

                if role == "conjunction":
                    template = (
                        "conjunction-introduction"
                        if position == "goal"
                        else "conjunction-elimination"
                    )
                    return template in self._templates and all(
                        visit(argument, position=position) for argument in arguments
                    )

                if role == "conditional":
                    return (
                        position == "premise"
                        and "modus-ponens" in self._templates
                        and all(
                            visit(argument, position="goal") for argument in arguments
                        )
                    )

                return False

            return all(visit(argument, position="term") for argument in node.args)

        return visit(term, position="goal" if goal else "premise")

    def _unsupported_reason(self, term: Term, /, *, goal: bool) -> str | None:
        if self._supported(term, goal=goal):
            return None

        def quantified(node: Term) -> bool:
            abstraction = self._abstraction(node)
            if abstraction is not None and self._type_order(abstraction[1]) > 0:
                return True
            return any(quantified(argument) for argument in node.args)

        if quantified(term) and self._order == "first":
            return "first-order neural proof cannot serve a second-order binder"

        return "the neural proof templates do not serve this logical form"

    def _subterms(self, term: Term, /) -> tuple[Term, ...]:
        return (
            term,
            *(nested for argument in term.args for nested in self._subterms(argument)),
        )

    def _instantiation_candidates(
        self,
        quantified_type: Term,
        sentences: tuple[Term, ...],
        /,
    ) -> tuple[Term, ...]:
        candidates: list[Term] = []

        def offer(term: Term) -> None:
            if (
                term.is_closed
                and term not in candidates
                and self._language.type_of(term) == quantified_type
            ):
                candidates.append(term)

        for sentence in sentences:
            for term in self._subterms(sentence):
                offer(term)

        vocabulary = self._language.store.signature.vocabulary(self._language.term_sort)
        for name in vocabulary.names:
            offer(self._language.constant(name))

        return tuple(candidates[: self._instantiations])

    def _conjuncts(self, term: Term, /) -> tuple[Term, ...]:
        logical = self._logical(term)

        if (
            "conjunction-elimination" in self._templates
            and logical is not None
            and logical[0] == "conjunction"
        ):
            return tuple(
                nested
                for argument in logical[1]
                for nested in self._conjuncts(argument)
            )

        return (term,)

    def _knowledge(
        self,
        premise: Term,
        theory: tuple[Term, ...],
        goal: Term,
        /,
    ) -> tuple[tuple[Term, ...], tuple[tuple[Term, Term], ...]]:
        facts: list[Term] = []
        rules: list[tuple[Term, Term]] = []

        sentences = (*theory, premise, goal)

        for sentence in (*theory, premise):
            for item in self._conjuncts(sentence):
                abstraction = self._abstraction(item)

                if abstraction is not None:
                    body, quantified_type = abstraction
                    for candidate in self._instantiation_candidates(
                        quantified_type,
                        sentences,
                    ):
                        instance = instantiate(
                            self._language.store,
                            body,
                            candidate,
                        )
                        logical = self._logical(instance)
                        if (
                            "modus-ponens" in self._templates
                            and logical is not None
                            and logical[0] == "conditional"
                        ):
                            rules.append((logical[1][0], logical[1][1]))
                        else:
                            facts.append(instance)
                    continue

                logical = self._logical(item)

                if (
                    "modus-ponens" in self._templates
                    and logical is not None
                    and logical[0] == "conditional"
                ):
                    rules.append((logical[1][0], logical[1][1]))
                else:
                    facts.append(item)

        return tuple(facts), tuple(rules)

    def _choice_row(self, term: Term, /) -> Tensor:
        code = self._embedding.index.index(term.op, term.payload)
        return self._embedding.weight[code]

    def _unify(self, left: Term, right: Term, /) -> Tensor:
        zero = self._embedding.weight.sum() * 0.0

        if self._language.type_of(left) != self._language.type_of(right):
            return zero

        if left.op.is_constant and right.op.is_constant:
            return self._kernel.similarity(
                self._choice_row(left),
                self._choice_row(right),
                temperature=self._temperature,
            )

        if (
            left.op != right.op
            or left.payload != right.payload
            or len(left.args) != len(right.args)
        ):
            return zero

        if not left.args:
            return zero + 1.0

        return torch.stack(
            [
                self._unify(one, other)
                for one, other in zip(left.args, right.args, strict=True)
            ]
        ).prod()

    def _compatible(self, left: Term, right: Term, /) -> bool:
        """Return whether two terms admit a type-directed soft unifier."""
        if self._language.type_of(left) != self._language.type_of(right):
            return False

        if left.op.is_constant and right.op.is_constant:
            return True

        return (
            left.op == right.op
            and left.payload == right.payload
            and len(left.args) == len(right.args)
            and all(
                self._compatible(one, other)
                for one, other in zip(left.args, right.args, strict=True)
            )
        )

    def _prove(
        self,
        goal: Term,
        facts: tuple[Term, ...],
        rules: tuple[tuple[Term, Term], ...],
        depth: int,
        /,
    ) -> _Search:
        candidates: list[Tensor] = []
        found = False

        if "identity" in self._templates:
            compatible = tuple(fact for fact in facts if self._compatible(fact, goal))
            candidates.extend(self._unify(fact, goal) for fact in compatible)
            found = bool(compatible)

        logical = self._logical(goal)

        if (
            depth > 0
            and "conjunction-introduction" in self._templates
            and logical is not None
            and logical[0] == "conjunction"
        ):
            left = self._prove(logical[1][0], facts, rules, depth - 1)
            right = self._prove(logical[1][1], facts, rules, depth - 1)
            candidates.append(left[0] * right[0])
            found = found or (left[1] and right[1])

        if depth > 0 and "modus-ponens" in self._templates:
            for antecedent, consequence in rules:
                if not self._compatible(consequence, goal):
                    continue

                premise = self._prove(antecedent, facts, rules, depth - 1)
                candidates.append(self._unify(consequence, goal) * premise[0])
                found = found or premise[1]

        if not candidates:
            return self._embedding.weight.sum() * 0.0, found

        return torch.stack(candidates).max(), found

    def score(self, questions: Questions, /) -> Scores[ProofSearch]:
        """Run soft search, then recheck every discovered discrete derivation."""
        theory = tuple(self._normalize(term) for term in questions.theory)
        pairs = tuple(
            (self._normalize(premise), self._normalize(goal))
            for premise, goal in zip(
                questions.premises,
                questions.goals,
                strict=True,
            )
        )
        values: list[Tensor] = []
        supported: list[bool] = []
        reasons: list[str | None] = []
        found: list[bool] = []

        theory_supported = all(self._supported(term, goal=False) for term in theory)

        for premise, goal in pairs:
            inside = (
                theory_supported
                and self._supported(premise, goal=False)
                and self._supported(goal, goal=True)
            )
            supported.append(inside)
            reasons.append(
                next(
                    (
                        reason
                        for term, is_goal in (
                            *((term, False) for term in theory),
                            (premise, False),
                            (goal, True),
                        )
                        if (reason := self._unsupported_reason(term, goal=is_goal))
                        is not None
                    ),
                    None,
                )
            )

            if not inside:
                values.append(self._embedding.weight.sum() * 0.0)
                found.append(False)
                continue

            facts, rules = self._knowledge(premise, theory, goal)
            result = self._prove(goal, facts, rules, self._depth)
            values.append(result[0])
            found.append(result[1])

        pending = tuple(position for position, item in enumerate(found) if item)
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
        verdicts: dict[int, Verdict] = {
            position: by_pair[pairs[position]] for position in pending
        }
        evidence: list[ProofSearch] = []

        for position, value in enumerate(values):
            if not supported[position]:
                verdict: Verdict = Abstain(
                    reasons[position]
                    or "the neural proof templates do not serve this logical form",
                    cause="out-of-fragment",
                )
            elif not found[position]:
                verdict = Abstain(
                    "the declared neural proof depth was exhausted",
                    cause="search-bound",
                    bound=SearchBound("proof-depth", self._depth),
                )
            else:
                verdict = verdicts[position]

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
        verified = sum(item.verified for item in evidence)
        discovered = sum(item.found for item in evidence)
        return Scores(
            torch.stack(values) if values else self._embedding.weight.new_empty((0,)),
            torch.tensor(
                supported,
                dtype=torch.bool,
                device=self._embedding.weight.device,
            ),
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
        templates = tuple(sorted(self._templates))
        return f"NeuralProofScorer(templates={templates!r}, depth={self._depth!r})"


def _build(
    settings: NeuralProofScorerSettings,
    context: BuildContext,
    /,
) -> NeuralProofScorer:
    """Build one neural proof scorer from shared language and proof services."""
    service = context.producer("service")
    normalization = context.producer("normalization")
    language = context.language

    if not isinstance(service, ProofService):
        msg = f"a neural proof scorer reads ProofService, not {service!r}"
        raise TypeError(msg)

    if not isinstance(language, LanguageAssembly):
        msg = f"a neural proof scorer reads LanguageAssembly, not {language!r}"
        raise TypeError(msg)

    if not isinstance(normalization, Normalization):
        msg = f"a neural proof scorer reads Normalization, not {normalization!r}"
        raise TypeError(msg)

    return place(
        NeuralProofScorer(
            service,
            settings.verification,
            language,
            normalization,
            context.choices,
            context.component,
            order=settings.order,
            instantiations=settings.instantiations,
            templates=settings.templates,
            depth=settings.depth,
            width=settings.width,
            similarity=settings.similarity,
        ),
        context.policy,
    )


PLUGIN: Final = ComponentPlugin(spec=NeuralProofScorerSpec, build=_build)
"""The differentiable proof search pair-scorer plugin."""

"""Rule, proof-search, and normal-form law checks."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Final, NamedTuple, cast

from hypothesis import given, seed, settings
from hypothesis import strategies as st

from lofi._checks import (
    LawViolationError,
    ensure_name,
    ensure_non_negative,
    fail_law,
)
from lofi.judge._ports import HOLE
from lofi.registry import Registry
from lofi.terms import MVar, Node, PVar, Spine

if TYPE_CHECKING:
    from collections.abc import Mapping

    from hypothesis.strategies import SearchStrategy

    from lofi._types import JsonValue
    from lofi.judge._ports import (
        NormalForm,
        ProofCalculus,
        ProofSchema,
        Rule,
        SchematicSequent,
        Slot,
        StructuralErasure,
    )
    from lofi.terms import Pattern, Term

__all__ = [
    "NORMAL_FORM_LAWS",
    "PROOF_CALCULUS_LAWS",
    "PROOF_SCHEMA_LAWS",
    "RULE_LAWS",
    "RULE_LAW_FAMILY",
    "STRUCTURAL_ERASURE_LAWS",
    "NormalFormFixture",
    "ProofCalculusFixture",
    "ProofSchemaFixture",
    "RuleFixture",
    "RuleLawCase",
    "RuleLawCases",
    "RuleLawProvider",
    "RuleLawReport",
    "RuleTotalityCase",
    "StructuralErasureFixture",
    "adjointness",
    "admits_is_total",
    "application_and_abstraction_channels_are_exact",
    "check_normal_form_laws",
    "check_proof_calculus_laws",
    "check_proof_schema_laws",
    "check_registered_rule_laws",
    "check_rule_law_cases",
    "check_rule_laws",
    "check_structural_erasure_laws",
    "conditional_completeness",
    "discover_rule_law_cases",
    "erasure_is_core_stable",
    "erasure_matches_structural_cases",
    "erasure_preserves_declared_lineages",
    "filtering_preserves_goal_terms",
    "hypotheses_are_stable_and_injective",
    "non_mutation",
    "proof_constructors_return_owned_terms",
    "schemata_are_stable",
    "schemata_are_structurally_deterministic",
    "schemata_match_declared_orientations",
    "soundness",
    "spec_round_trip",
    "stability",
    "tagging_is_pure",
    "tags_are_declared",
    "variable_channels_are_canonical",
]

RULE_LAW_FAMILY: Final = "rule_laws"
"""Registry family whose ``lofi.rule_laws`` entry points provide rule laws."""

_CANDIDATES: Final = 16
"""Conditional-completeness candidate count."""


class ProofSchemaFixture(NamedTuple):
    """Inputs for proof-schema law checks."""

    schema: ProofSchema
    expected: tuple[SchematicSequent, ...]


def schemata_match_declared_orientations(fixture: ProofSchemaFixture) -> None:
    """Check the declared physical orientations and order."""
    received = fixture.schema.proof_schemata

    if not isinstance(received, tuple) or received != fixture.expected:
        fail_law(
            f"received {received!r}, expected {fixture.expected!r}",
            law="schemata match declared orientations",
        )


def schemata_are_stable(fixture: ProofSchemaFixture) -> None:
    """Check that repeated reads preserve the schema value."""
    first = fixture.schema.proof_schemata
    second = fixture.schema.proof_schemata

    if first != second:
        fail_law(
            f"successive reads returned {first!r} and {second!r}",
            law="schemata are stable",
        )


def _pattern_is_deterministic(pattern: Pattern) -> bool:
    """Return whether a pattern has no operator, payload, or spine variables."""
    if isinstance(pattern, MVar):
        return True

    if isinstance(pattern, Spine):
        return False

    return (
        isinstance(pattern, Node)
        and not isinstance(pattern.op, PVar)
        and not isinstance(pattern.payload, PVar)
        and all(_pattern_is_deterministic(child) for child in pattern.args)
    )


def schemata_are_structurally_deterministic(fixture: ProofSchemaFixture) -> None:
    """Check that every schema uses deterministic term patterns."""
    for at, (premises, conclusion) in enumerate(fixture.schema.proof_schemata):
        patterns = (*premises, conclusion)

        if not all(_pattern_is_deterministic(pattern) for pattern in patterns):
            fail_law(
                f"schema {at} contains a nondeterministic pattern",
                law="schemata are structurally deterministic",
            )


PROOF_SCHEMA_LAWS: Final = (
    schemata_match_declared_orientations,
    schemata_are_stable,
    schemata_are_structurally_deterministic,
)
"""Proof-schema laws in conformance order."""


def check_proof_schema_laws(fixture: ProofSchemaFixture) -> None:
    """Run every proof-schema law."""
    for law in PROOF_SCHEMA_LAWS:
        law(fixture)


class StructuralErasureFixture(NamedTuple):
    """Inputs for structural-erasure law checks."""

    erasure: StructuralErasure
    cases: tuple[tuple[Term, Term], ...]


def erasure_preserves_declared_lineages(
    fixture: StructuralErasureFixture,
) -> None:
    """Check store ownership for declared cases and atom images.

    Raises
    ------
    LawViolationError
        If cases are absent, lineage ownership fails, or erasure raises.
    """
    erasure = fixture.erasure

    if not fixture.cases:
        fail_law("the fixture declares no cases", law="erasure preserves lineages")

    for name, image in erasure.atoms.items():
        if not erasure.store.contains(image):
            fail_law(
                f"atom image {name!r} is outside the language-store lineage",
                law="erasure preserves lineages",
            )

    for source, expected in fixture.cases:
        if not erasure.store.contains(source):
            fail_law(
                f"source case {source!r} is outside the language-store lineage",
                law="erasure preserves lineages",
            )

        if not erasure.store.contains(expected):
            fail_law(
                f"expected result {expected!r} is outside the language-store lineage",
                law="erasure preserves lineages",
            )

        try:
            received = erasure(source)
        except Exception as error:
            message = f"erasure preserves lineages: erasing {source!r} raised"
            raise LawViolationError(message) from error

        if not erasure.store.contains(received):
            fail_law(
                f"erasing {source!r} returned foreign term {received!r}",
                law="erasure preserves lineages",
            )


def erasure_matches_structural_cases(
    fixture: StructuralErasureFixture,
) -> None:
    """Check each declared structural input and output pair.

    Raises
    ------
    LawViolationError
        If erasure raises or a declared pair is unstable or incorrect.
    """
    for source, expected in fixture.cases:
        try:
            first = fixture.erasure(source)
            second = fixture.erasure(source)
        except Exception as error:
            message = f"erasure matches structural cases: erasing {source!r} raised"
            raise LawViolationError(message) from error

        if first is not expected or second is not first:
            fail_law(
                f"erasing {source!r} twice returned {first!r} and {second!r}, "
                f"expected interned {expected!r}",
                law="erasure matches structural cases",
            )


def erasure_is_core_stable(
    fixture: StructuralErasureFixture,
) -> None:
    """Check that feature erasure does not alter the semantic image.

    Raises
    ------
    LawViolationError
        If erasure raises or a term and its core have different images.
    """
    for source, _ in fixture.cases:
        try:
            received = fixture.erasure(source)
            core = fixture.erasure(source.core)
        except Exception as error:
            message = f"erasure is core stable: erasing {source!r} raised"
            raise LawViolationError(message) from error

        if received is not core:
            fail_law(
                f"{source!r} and its core map to {received!r} and {core!r}",
                law="erasure is core stable",
            )


STRUCTURAL_ERASURE_LAWS: Final = (
    erasure_preserves_declared_lineages,
    erasure_matches_structural_cases,
    erasure_is_core_stable,
)
"""Structural-erasure laws in conformance order."""


def check_structural_erasure_laws(
    fixture: StructuralErasureFixture,
) -> None:
    """Run every structural-erasure law."""
    for law in STRUCTURAL_ERASURE_LAWS:
        law(fixture)


class ProofCalculusFixture(NamedTuple):
    """Inputs for proof-calculus law checks."""

    calculus: ProofCalculus
    indices: tuple[int, ...]


def hypotheses_are_stable_and_injective(
    fixture: ProofCalculusFixture,
) -> None:
    """Check stable, injective, canonical hypotheses.

    Raises
    ------
    LawViolationError
        If the indices are invalid or hypotheses violate the checked properties.
    """
    if not fixture.indices or any(index < 0 for index in fixture.indices):
        fail_law(
            f"the fixture indices are invalid: {fixture.indices!r}",
            law="hypotheses are stable and injective",
        )

    calculus = fixture.calculus

    try:
        first = tuple(calculus.hypothesis(index) for index in fixture.indices)
        second = tuple(calculus.hypothesis(index) for index in fixture.indices)
    except Exception as error:
        message = "hypotheses are stable and injective: construction raised"
        raise LawViolationError(message) from error

    if any(left is not right for left, right in zip(first, second, strict=True)):
        fail_law(
            f"successive reads returned {first!r} and {second!r}",
            law="hypotheses are stable and injective",
        )

    if len(set(first)) != len(first):
        fail_law(
            f"indices {fixture.indices!r} returned repeated hypotheses {first!r}",
            law="hypotheses are stable and injective",
        )

    for term in first:
        if not calculus.store.contains(term) or term.payload is None:
            fail_law(
                f"hypothesis {term!r} is not a store-owned free variable",
                law="hypotheses are stable and injective",
            )

        try:
            name = calculus.store.atom_name(term.payload)
        except ValueError as error:
            message = "hypotheses are stable and injective: invalid atom payload"
            raise LawViolationError(message) from error

        if calculus.store.fvar(term.op.sort, name) is not term:
            fail_law(
                f"hypothesis {term!r} is not a canonical free variable",
                law="hypotheses are stable and injective",
            )


def variable_channels_are_canonical(
    fixture: ProofCalculusFixture,
) -> None:
    """Check the named free-variable and indexed bound-variable channels.

    Raises
    ------
    LawViolationError
        If construction raises or either channel returns a noncanonical term.
    """
    calculus = fixture.calculus
    name = "__proof_conformance_variable"
    sort = calculus.store.signature.by_name(calculus.operators.lam).binding[0].sort

    try:
        free = calculus.free_variable(name)
        bound = calculus.bound_variable(2)
        expected_free = calculus.store.fvar(sort, name)
        expected_bound = calculus.store.bvar(sort, 2)
    except Exception as error:
        message = "variable channels are canonical: construction raised"
        raise LawViolationError(message) from error

    if free is not expected_free or bound is not expected_bound:
        fail_law(
            f"received free {free!r} and bound {bound!r}",
            law="variable channels are canonical",
        )


def application_and_abstraction_channels_are_exact(
    fixture: ProofCalculusFixture,
) -> None:
    """Check application and abstraction against direct store construction.

    Raises
    ------
    LawViolationError
        If construction raises or either channel differs from the store.
    """
    calculus = fixture.calculus
    variable = "__proof_conformance_binder"

    try:
        function = calculus.free_variable("__proof_conformance_function")
        argument = calculus.free_variable("__proof_conformance_argument")
        application = calculus.application(function, argument)
        expected_application = calculus.store.op(
            calculus.store.signature.by_name(calculus.operators.app),
            function,
            argument,
        )
        abstraction = calculus.abstraction(variable, argument)
        expected_abstraction = calculus.store.bind(
            calculus.operators.lam,
            variable,
            argument,
        )
    except Exception as error:
        message = "application and abstraction channels are exact: construction raised"
        raise LawViolationError(message) from error

    if (
        application is not expected_application
        or abstraction is not expected_abstraction
    ):
        fail_law(
            f"received application {application!r} and abstraction {abstraction!r}",
            law="application and abstraction channels are exact",
        )


def proof_constructors_return_owned_terms(
    fixture: ProofCalculusFixture,
) -> None:
    """Check that every proof-construction channel returns an owned term.

    Raises
    ------
    LawViolationError
        If construction raises or a channel returns a foreign term.
    """
    calculus = fixture.calculus

    try:
        hypothesis = calculus.hypothesis(fixture.indices[0])
        free = calculus.free_variable("__proof_conformance_owned")
        terms = (
            hypothesis,
            free,
            calculus.bound_variable(0),
            calculus.application(hypothesis, free),
            calculus.abstraction("__proof_conformance_owned", free),
        )
    except Exception as error:
        message = "proof constructors return owned terms: construction raised"
        raise LawViolationError(message) from error

    foreign = tuple(term for term in terms if not calculus.store.contains(term))

    if foreign:
        fail_law(
            f"constructor channels returned foreign terms {foreign!r}",
            law="proof constructors return owned terms",
        )


PROOF_CALCULUS_LAWS: Final = (
    hypotheses_are_stable_and_injective,
    variable_channels_are_canonical,
    application_and_abstraction_channels_are_exact,
    proof_constructors_return_owned_terms,
)
"""Proof-calculus laws in conformance order."""


def check_proof_calculus_laws(
    fixture: ProofCalculusFixture,
) -> None:
    """Run every proof-calculus law."""
    for law in PROOF_CALCULUS_LAWS:
        law(fixture)


type _Step[T] = tuple[Rule[T], tuple[str, ...], tuple[T, ...]]
"""Rule, child tags, and premises."""


class RuleFixture[T](NamedTuple):
    """Inputs for rule-law checks.

    Attributes
    ----------
    rule : Rule[T]
        Rule under test.
    terms : SearchStrategy[T]
        Term strategy.
    rebuild : Callable[[Mapping[str, JsonValue]], Rule[T]]
        Factory that rebuilds the rule from its declaration.
    max_examples : int
        Maximum Hypothesis examples per law.
    seed : int
        Hypothesis seed applied to each law.
    """

    rule: Rule[T]
    terms: SearchStrategy[T]
    rebuild: Callable[[Mapping[str, JsonValue]], Rule[T]]
    max_examples: int = 300
    seed: int = 0


class RuleLawCase(NamedTuple):
    """A named zero-argument rule-law check.

    Attributes
    ----------
    name : str
        Globally unique case name.
    rule_name : str
        Runtime rule name.
    check : Callable[[], None]
        Check that runs every ordinary rule law.
    """

    name: str
    rule_name: str
    check: Callable[[], None]


class RuleTotalityCase(NamedTuple):
    """A named zero-argument totality check.

    Attributes
    ----------
    name : str
        Globally unique case name.
    rule_name : str
        Runtime rule name.
    check : Callable[[], int]
        Check that returns its admission-call count.
    """

    name: str
    rule_name: str
    check: Callable[[], int]


class RuleLawCases(NamedTuple):
    """The cases contributed by one provider.

    Attributes
    ----------
    provider : str
        Entry-point name.
    ordinary : tuple[RuleLawCase, ...]
        Cases that run all rule laws.
    totality : tuple[RuleTotalityCase, ...]
        Standalone totality cases.
    """

    provider: str
    ordinary: tuple[RuleLawCase, ...]
    totality: tuple[RuleTotalityCase, ...] = ()


class RuleLawReport(NamedTuple):
    """Coverage report for one rule-law run.

    Attributes
    ----------
    providers : tuple[str, ...]
        Providers in discovery order.
    ordinary_cases : tuple[str, ...]
        Ordinary case names in provider order.
    totality_cases : tuple[str, ...]
        Totality case names in provider order.
    runtime_rules : tuple[str, ...]
        Case-qualified runtime rule names.
    totality_calls : int
        Number of admission calls made by totality cases.
    """

    providers: tuple[str, ...]
    ordinary_cases: tuple[str, ...]
    totality_cases: tuple[str, ...]
    runtime_rules: tuple[str, ...]
    totality_calls: int

    @property
    def covered(self) -> tuple[str, ...]:
        """Return every covered provider, case, and runtime rule."""
        cases = (*self.ordinary_cases, *self.totality_cases)

        return (
            *(f"provider:{provider}" for provider in self.providers),
            *(f"case:{case}" for case in cases),
            *(f"rule:{rule}" for rule in self.runtime_rules),
        )


def _premises[T](fixture: RuleFixture[T]) -> SearchStrategy[tuple[T, ...]]:
    """Return a strategy over premise tuples of the rule's arity.

    Parameters
    ----------
    fixture : RuleFixture[T]
        Rule and term strategy.

    Returns
    -------
    SearchStrategy[tuple[T, ...]]
        Premise-tuple strategy.
    """
    return st.tuples(*[fixture.terms] * fixture.rule.arity)


def _candidates[T](fixture: RuleFixture[T]) -> SearchStrategy[list[T]]:
    """Return a strategy over candidate-conclusion batches.

    Parameters
    ----------
    fixture : RuleFixture[T]
        Rule and term strategy.

    Returns
    -------
    SearchStrategy[list[T]]
        Fixed-size candidate-batch strategy.
    """
    return st.lists(fixture.terms, min_size=_CANDIDATES, max_size=_CANDIDATES)


def _budget(max_examples: int) -> settings:
    """Return Hypothesis settings for a law.

    Parameters
    ----------
    max_examples : int
        Maximum generated examples.

    Returns
    -------
    settings
        Settings with no per-example deadline.
    """
    # Disable per-example deadlines because runtime varies by machine.
    return settings(max_examples=max_examples, deadline=None)


def soundness[T](fixture: RuleFixture[T]) -> None:
    """Check that ``admits`` accepts every enumerated conclusion.

    Parameters
    ----------
    fixture : RuleFixture[T]
        Rule and generated terms.

    Raises
    ------
    LawViolationError
        If an enumerated conclusion is rejected or no case exercises the law.
    """
    rule = fixture.rule

    if not {"forward", "check"} <= rule.directions:
        return

    fired: list[tuple[tuple[T, ...], T]] = []

    @seed(fixture.seed)
    @given(_premises(fixture))
    @_budget(fixture.max_examples)
    def law(inputs: tuple[T, ...]) -> None:
        for output in rule.apply(inputs):
            fired.append((inputs, output))

            if not rule.admits(inputs, output):
                detail = (
                    f"{rule.name} enumerated {output!r} from {inputs!r} "
                    f"and then rejected it"
                )

                fail_law(detail, law="soundness")

    law()

    if not fired:
        detail = (
            f"{rule.name} fired on no generated premise tuple, so soundness "
            f"was never exercised"
        )

        fail_law(detail, law="soundness")


def conditional_completeness[T](fixture: RuleFixture[T]) -> None:
    """Check complete forward enumeration against admitted outputs.

    Parameters
    ----------
    fixture : RuleFixture[T]
        Rule and generated terms.

    Raises
    ------
    LawViolationError
        If an admitted output has no equivalent enumerated output.
    """
    rule = fixture.rule

    if not rule.forward_is_complete or not {"forward", "check"} <= rule.directions:
        return

    matching = rule.matching
    fired: list[T] = []

    @seed(fixture.seed)
    @given(_premises(fixture))
    @_budget(fixture.max_examples)
    def witness(inputs: tuple[T, ...]) -> None:
        fired.extend(rule.apply(inputs))

    witness()

    if not fired:
        detail = (
            f"{rule.name} fired on no generated premise tuple, so completeness "
            f"was never exercised"
        )

        fail_law(detail, law="conditional completeness")

    @seed(fixture.seed)
    @given(_premises(fixture), _candidates(fixture))
    @_budget(fixture.max_examples)
    def law(inputs: tuple[T, ...], outputs: list[T]) -> None:
        admitted_here = [o for o in outputs if rule.admits(inputs, o)]

        if not admitted_here:
            return

        enumerated = tuple(rule.apply(inputs))

        for output in admitted_here:
            if not any(matching.equivalent(output, o) for o in enumerated):
                detail = (
                    f"{rule.name} admits {output!r} from {inputs!r} but "
                    f"enumerates nothing equivalent to it"
                )

                fail_law(detail, law="conditional completeness")

    law()


def _verdict[T](rule: Rule[T], inputs: tuple[T, ...], output: T) -> bool:
    """Return a verdict or report a totality failure.

    Parameters
    ----------
    rule : Rule[T]
        Rule to query.
    inputs : tuple[T, ...]
        Premises in rule order.
    output : T
        Candidate conclusion.

    Returns
    -------
    bool
        Rule verdict.

    Raises
    ------
    LawViolationError
        If the rule raises.
    """
    try:
        return rule.admits(inputs, output)

    except Exception as exc:
        message = (
            f"totality: {rule.name}.admits raised {exc!r} on {inputs!r} and {output!r}"
        )

        raise LawViolationError(message) from exc


def admits_is_total[T](fixture: RuleFixture[T]) -> None:
    """Check that ``admits`` terminates and never raises.

    Parameters
    ----------
    fixture : RuleFixture[T]
        Rule and generated terms.

    Raises
    ------
    LawViolationError
        If ``admits`` raises or accepts a malformed premise tuple.
    """
    rule = fixture.rule

    @seed(fixture.seed)
    @given(_premises(fixture), fixture.terms)
    @_budget(fixture.max_examples)
    def law(inputs: tuple[T, ...], output: T) -> None:
        _verdict(rule, inputs, output)
        mis_arity: list[tuple[T, ...]] = [(*inputs, output)]

        if rule.arity:
            mis_arity.append(inputs[:-1])

        for probe in mis_arity:
            if _verdict(rule, probe, output):
                detail = (
                    f"{rule.name} of arity {rule.arity} admitted {len(probe)} premises"
                )

                fail_law(detail, law="totality")

    law()


def _check_residual_declaration[T](rule: Rule[T]) -> None:
    """Check residual-slot range and direction agreement."""
    valid_slots = frozenset(range(rule.arity))
    invalid_slots = rule.residual_slots - valid_slots

    if invalid_slots:
        detail = f"{rule.name} declares out-of-range residual slots {invalid_slots!r}"

        fail_law(detail, law="adjointness")

    has_residual_direction = "residual" in rule.directions

    if has_residual_direction != bool(rule.residual_slots):
        detail = (
            f"{rule.name} declares residual direction {has_residual_direction!r} "
            f"and residual slots {rule.residual_slots!r}"
        )

        fail_law(detail, law="adjointness")


def _known_with_hole[T](inputs: tuple[T, ...], slot: int) -> tuple[Slot[T], ...]:
    """Replace one premise with the shared hole."""
    return tuple(HOLE if at == slot else term for at, term in enumerate(inputs))


def _check_undeclared_residuals[T](
    rule: Rule[T], inputs: tuple[T, ...], output: T
) -> None:
    """Check that undeclared premise slots yield no residuals."""
    undeclared = frozenset(range(rule.arity)) - rule.residual_slots

    for slot in undeclared:
        if tuple(rule.residuate(output, _known_with_hole(inputs, slot))):
            detail = f"{rule.name} residuated undeclared premise slot {slot}"

            fail_law(detail, law="adjointness")


def _check_declared_residuals[T](
    rule: Rule[T], inputs: tuple[T, ...], output: T
) -> None:
    """Check that every declared premise slot recovers its input."""
    for slot in rule.residual_slots:
        recovered = rule.residuate(output, _known_with_hole(inputs, slot))

        if not any(
            rule.matching.equivalent(inputs[slot], candidate) for candidate in recovered
        ):
            detail = (
                f"{rule.name} draws {output!r} from {inputs!r}, but "
                f"residuating slot {slot} misses {inputs[slot]!r}"
            )

            fail_law(detail, law="adjointness")


def adjointness[T](fixture: RuleFixture[T]) -> None:
    """Check premise-granular inverse images of application.

    Parameters
    ----------
    fixture : RuleFixture[T]
        Rule and generated terms.

    Raises
    ------
    LawViolationError
        If residual declarations or computed inverse images are inconsistent.
    """
    rule = fixture.rule
    _check_residual_declaration(rule)
    fired: list[T] = []

    @seed(fixture.seed)
    @given(_premises(fixture), fixture.terms)
    @_budget(fixture.max_examples)
    def law(inputs: tuple[T, ...], probe: T) -> None:
        _check_undeclared_residuals(rule, inputs, probe)

        if "forward" not in rule.directions:
            return

        for output in rule.apply(inputs):
            fired.append(output)
            _check_undeclared_residuals(rule, inputs, output)
            _check_declared_residuals(rule, inputs, output)

    law()

    if "forward" in rule.directions and rule.residual_slots and not fired:
        detail = (
            f"{rule.name} fired on no generated premise pair, so adjointness "
            f"was never exercised"
        )

        fail_law(detail, law="adjointness")


def non_mutation[T](fixture: RuleFixture[T]) -> None:
    """Check that no computation direction mutates its inputs.

    Parameters
    ----------
    fixture : RuleFixture[T]
        Rule and generated terms.

    Raises
    ------
    LawViolationError
        If any rule computation mutates a premise or conclusion.
    """
    rule = fixture.rule

    @seed(fixture.seed)
    @given(_premises(fixture), fixture.terms)
    @_budget(fixture.max_examples)
    def law(inputs: tuple[T, ...], output: T) -> None:
        before = tuple((term, hash(term)) for term in (*inputs, output))

        if "forward" in rule.directions:
            tuple(rule.apply(inputs))

        if "check" in rule.directions:
            rule.admits(inputs, output)

        for slot in rule.residual_slots:
            known = tuple(
                HOLE if at == slot else term for at, term in enumerate(inputs)
            )
            tuple(rule.residuate(output, known))

        if tuple((term, hash(term)) for term in (*inputs, output)) != before:
            detail = f"{rule.name} changed an input among {inputs!r} and {output!r}"

            fail_law(detail, law="non-mutation")

    law()


def stability[T](fixture: RuleFixture[T]) -> None:
    """Check premise handling under the rule's congruence.

    Parameters
    ----------
    fixture : RuleFixture[T]
        Rule and generated terms.

    Raises
    ------
    LawViolationError
        If equivalent premises produce inequivalent conclusions.
    """
    rule = fixture.rule
    matching = rule.matching

    if "forward" not in rule.directions or not matching.has_key:
        return

    @seed(fixture.seed)
    @given(_premises(fixture))
    @_budget(fixture.max_examples)
    def law(inputs: tuple[T, ...]) -> None:
        keyed = tuple(matching.key(term) for term in inputs)
        got = {matching.key(o) for o in rule.apply(keyed)}
        want = {matching.key(o) for o in rule.apply(inputs)}

        if got != want:
            detail = (
                f"{rule.name} on {inputs!r} draws {want!r}, but on their "
                f"representatives {keyed!r} draws {got!r}"
            )

            fail_law(detail, law="stability")

    law()


def spec_round_trip[T](fixture: RuleFixture[T]) -> None:
    """Check that a rule's declaration rebuilds it.

    Parameters
    ----------
    fixture : RuleFixture[T]
        Rule, generated terms, and rebuild function.

    Raises
    ------
    LawViolationError
        If declarations or behavior differ after rebuilding.
    """
    rule = fixture.rule
    spec = rule.to_spec()
    rebuilt = fixture.rebuild(spec)

    if rebuilt.to_spec() != spec:
        detail = f"rebuilding {rule.name} from {spec!r} declares {rebuilt.to_spec()!r}"

        fail_law(detail, law="specification round trip")

    declared = (
        rule.name,
        rule.arity,
        rule.directions,
        rule.residual_slots,
        rule.forward_is_complete,
        rule.probe,
    )
    rebuilt_declared = (
        rebuilt.name,
        rebuilt.arity,
        rebuilt.directions,
        rebuilt.residual_slots,
        rebuilt.forward_is_complete,
        rebuilt.probe,
    )

    if declared != rebuilt_declared:
        detail = f"{rule.name} declares {declared!r}, its rebuild {rebuilt_declared!r}"

        fail_law(detail, law="specification round trip")

    matching = rule.matching

    @seed(fixture.seed)
    @given(_premises(fixture), fixture.terms)
    @_budget(fixture.max_examples)
    def law(inputs: tuple[T, ...], output: T) -> None:
        if "check" in rule.directions and rebuilt.admits(inputs, output) != rule.admits(
            inputs, output
        ):
            detail = (
                f"{rule.name} and its rebuild disagree on whether {inputs!r} "
                f"admits {output!r}"
            )

            fail_law(detail, law="specification round trip")

        if "forward" in rule.directions:
            got = {matching.key(o) for o in rebuilt.apply(inputs)}
            want = {matching.key(o) for o in rule.apply(inputs)}

            if got != want:
                detail = (
                    f"{rule.name} draws {want!r} from {inputs!r}, its rebuild {got!r}"
                )

                fail_law(detail, law="specification round trip")

    law()


RULE_LAWS: Final = (
    soundness,
    conditional_completeness,
    admits_is_total,
    adjointness,
    non_mutation,
    stability,
    spec_round_trip,
)
"""Rule laws in conformance order."""


def check_rule_laws[T](fixture: RuleFixture[T]) -> None:
    """Run every rule law against one rule.

    Parameters
    ----------
    fixture : RuleFixture[T]
        Rule-law fixture.

    Raises
    ------
    LawViolationError
        If any rule law fails.
    """
    for law in RULE_LAWS:
        law(fixture)


type RuleLawProvider = Callable[[], RuleLawCases]
"""Zero-argument rule-law provider."""


def _validate_rule_law_cases(
    providers: tuple[RuleLawCases, ...],
) -> tuple[RuleLawCases, ...]:
    """Validate discovered provider and case names.

    Parameters
    ----------
    providers : tuple[RuleLawCases, ...]
        Provider contributions in discovery order.

    Raises
    ------
    ValueError
        If a provider name repeats, a provider has no ordinary cases, or a case
        name repeats.
    """
    provider_names: set[str] = set()
    case_names: set[str] = set()

    for contribution in providers:
        provider = ensure_name("rule-law provider", contribution.provider)

        if provider in provider_names:
            msg = f"duplicate rule-law provider name: {provider!r}"
            raise ValueError(msg)

        if not contribution.ordinary:
            msg = f"rule-law provider {provider!r} contributes no ordinary cases"
            raise ValueError(msg)

        provider_names.add(provider)

        for case in (*contribution.ordinary, *contribution.totality):
            name = ensure_name("rule-law case", case.name)
            ensure_name("runtime rule", case.rule_name)

            if name in case_names:
                msg = f"duplicate rule-law case name: {name!r}"
                raise ValueError(msg)

            case_names.add(name)

    return providers


def discover_rule_law_cases(
    registry: Registry[RuleLawProvider] | None = None,
) -> tuple[RuleLawCases, ...]:
    """Load rule-law providers in stable tag order.

    Parameters
    ----------
    registry : Registry[RuleLawProvider] | None
        The rule-law family to discover, or None for the installed one.

    Returns
    -------
    tuple[RuleLawCases, ...]
        Validated provider contributions.

    Raises
    ------
    ValueError
        If a provider name disagrees with its entry point or validation fails.
    """
    contributions: list[RuleLawCases] = []

    family = (
        Registry[RuleLawProvider](RULE_LAW_FAMILY) if registry is None else registry
    )

    for point in family.entries():
        provider = cast("RuleLawProvider", point.load())
        contribution = provider()

        if contribution.provider != point.name:
            msg = (
                f"rule-law entry point {point.name!r} returned provider "
                f"{contribution.provider!r}"
            )
            raise ValueError(msg)

        contributions.append(contribution)

    return _validate_rule_law_cases(tuple(contributions))


def check_rule_law_cases(
    providers: tuple[RuleLawCases, ...],
) -> RuleLawReport:
    """Run validated ordinary and totality rule-law cases.

    Parameters
    ----------
    providers : tuple[RuleLawCases, ...]
        Provider contributions to run.

    Returns
    -------
    RuleLawReport
        Coverage and totality call counts.

    Raises
    ------
    ValueError
        If provider validation fails or a totality check reports no calls.
    LawViolationError
        If an ordinary rule law fails.
    """
    contributions = _validate_rule_law_cases(providers)

    for contribution in contributions:
        for case in contribution.ordinary:
            case.check()

    totality_calls = 0

    for contribution in contributions:
        for case in contribution.totality:
            calls = ensure_non_negative("R3 admission-call count", case.check())

            if calls == 0:
                msg = f"R3 case {case.name!r} made no admission calls"
                raise ValueError(msg)

            totality_calls += calls

    ordinary = tuple(
        case for contribution in contributions for case in contribution.ordinary
    )
    totality = tuple(
        case for contribution in contributions for case in contribution.totality
    )
    runtime_rules = tuple(
        f"{case.name}:{case.rule_name}" for case in (*ordinary, *totality)
    )
    report = RuleLawReport(
        providers=tuple(contribution.provider for contribution in contributions),
        ordinary_cases=tuple(case.name for case in ordinary),
        totality_cases=tuple(case.name for case in totality),
        runtime_rules=runtime_rules,
        totality_calls=totality_calls,
    )

    if not report.covered:
        msg = "rule-law coverage is empty"
        raise ValueError(msg)

    return report


def check_registered_rule_laws(
    registry: Registry[RuleLawProvider] | None = None,
) -> RuleLawReport:
    """Discover and run every registered rule-law provider.

    Parameters
    ----------
    registry : Registry[RuleLawProvider] | None
        The rule-law family to discover, or None for the installed one.

    Returns
    -------
    RuleLawReport
        Coverage and totality call counts.

    Raises
    ------
    ValueError
        If provider validation fails or a totality check reports no calls.
    LawViolationError
        If an ordinary rule law fails.
    """
    return check_rule_law_cases(discover_rule_law_cases(registry))


class NormalFormFixture[T](NamedTuple):
    """Inputs for normal-form law checks.

    Attributes
    ----------
    normal_form : NormalForm[T]
        Filter under test.
    rules : tuple[Rule[T], ...]
        Rules used to generate steps.
    terms : SearchStrategy[T]
        Term strategy.
    tokens : SearchStrategy[tuple[str, ...]]
        Token-sequence strategy.
    reachable : Callable[[NormalForm[T] | None, tuple[str, ...]], frozenset[T]]
        Function that computes reachable goal terms.
    max_examples : int
        Maximum Hypothesis examples per law.
    """

    normal_form: NormalForm[T]
    rules: tuple[Rule[T], ...]
    terms: SearchStrategy[T]
    tokens: SearchStrategy[tuple[str, ...]]
    reachable: Callable[[NormalForm[T] | None, tuple[str, ...]], frozenset[T]]
    max_examples: int = 200


def _tags[T](fixture: NormalFormFixture[T], *, law: str) -> tuple[str, ...]:
    """Return the ordered nonempty tag alphabet.

    Parameters
    ----------
    fixture : NormalFormFixture[T]
        Normal-form fixture.
    law : str
        Current law name.

    Returns
    -------
    tuple[str, ...]
        Declared tags in ascending order.

    Raises
    ------
    LawViolationError
        If no rules or tags are declared.
    """
    normal_form = fixture.normal_form

    if not fixture.rules:
        fail_law(f"{normal_form.name} was given no rules to tag steps of", law=law)

    tags = tuple(sorted(normal_form.alphabet))

    if not tags:
        fail_law(f"{normal_form.name} declares an empty tag alphabet", law=law)

    return tags


def _steps[T](
    fixture: NormalFormFixture[T], tags: tuple[str, ...]
) -> SearchStrategy[_Step[T]]:
    """Return a strategy over derivation steps.

    Parameters
    ----------
    fixture : NormalFormFixture[T]
        Normal-form fixture.
    tags : tuple[str, ...]
        Available child tags.

    Returns
    -------
    SearchStrategy[_Step[T]]
        Generated rule, child-tag, and premise tuples.
    """

    def of(rule: Rule[T]) -> SearchStrategy[_Step[T]]:
        return st.tuples(
            st.just(rule),
            st.tuples(*[st.sampled_from(tags)] * rule.arity),
            st.tuples(*[fixture.terms] * rule.arity),
        )

    return st.sampled_from(fixture.rules).flatmap(of)


def tags_are_declared[T](fixture: NormalFormFixture[T]) -> None:
    """Check that every returned tag belongs to the declared alphabet.

    Parameters
    ----------
    fixture : NormalFormFixture[T]
        Filter and generated steps.

    Raises
    ------
    LawViolationError
        If a lexical or derived tag is undeclared.
    """
    normal_form = fixture.normal_form
    tags = _tags(fixture, law="declared tags")
    alphabet = normal_form.alphabet
    lexical = normal_form.lexical_tag()

    if lexical not in alphabet:
        detail = (
            f"{normal_form.name} tags a lexical item {lexical!r}, "
            f"which is not among {tags!r}"
        )

        fail_law(detail, law="declared tags")

    @given(_steps(fixture, tags))
    @_budget(fixture.max_examples)
    def law(step: _Step[T]) -> None:
        rule, children, _ = step
        tag = normal_form.tag(rule, children)

        if tag not in alphabet:
            detail = (
                f"{normal_form.name} tags a {rule.name} step over {children!r} "
                f"{tag!r}, which is not among {tags!r}"
            )

            fail_law(detail, law="declared tags")

    law()


def tagging_is_pure[T](fixture: NormalFormFixture[T]) -> None:
    """Check that tagging and filtering depend only on their arguments.

    Parameters
    ----------
    fixture : NormalFormFixture[T]
        Filter and generated steps.

    Raises
    ------
    LawViolationError
        If repeated calls return different tags or verdicts.
    """
    normal_form = fixture.normal_form
    tags = _tags(fixture, law="purity")
    steps = _steps(fixture, tags)

    @given(steps, steps, fixture.terms, fixture.terms)
    @_budget(fixture.max_examples)
    def law(step: _Step[T], other: _Step[T], output: T, other_output: T) -> None:
        rule, children, inputs = step
        other_rule, other_children, other_inputs = other

        normal_form.tag(other_rule, other_children)
        normal_form.admits(other_rule, other_children, other_inputs, other_output)

        lexical = normal_form.lexical_tag()
        answered = [normal_form.tag(rule, children) for _ in (0, 1)]
        verdicts = [normal_form.admits(rule, children, inputs, output) for _ in (0, 1)]

        normal_form.tag(other_rule, other_children)
        normal_form.admits(other_rule, other_children, other_inputs, other_output)

        answered.append(normal_form.tag(rule, children))
        verdicts.append(normal_form.admits(rule, children, inputs, output))

        if len(set(answered)) > 1:
            detail = (
                f"{normal_form.name} tags one {rule.name} step over "
                f"{children!r} {answered!r}"
            )

            fail_law(detail, law="purity")

        if len(set(verdicts)) > 1:
            detail = (
                f"{normal_form.name} answers {verdicts!r} to one question about "
                f"a {rule.name} step from {inputs!r} to {output!r}"
            )

            fail_law(detail, law="purity")

        if normal_form.lexical_tag() != lexical:
            detail = (
                f"{normal_form.name} tags a lexical item {lexical!r}, then "
                f"{normal_form.lexical_tag()!r}"
            )

            fail_law(detail, law="purity")

    law()


def filtering_preserves_goal_terms[T](fixture: NormalFormFixture[T]) -> None:
    """Check that filtering preserves reachable goal terms.

    Parameters
    ----------
    fixture : NormalFormFixture[T]
        Filter and reachable-term function.

    Raises
    ------
    LawViolationError
        If filtering changes goal terms or no input exercises the law.
    """
    normal_form = fixture.normal_form
    reached: list[tuple[str, ...]] = []

    @given(fixture.tokens)
    @_budget(fixture.max_examples)
    def law(tokens: tuple[str, ...]) -> None:
        want = fixture.reachable(None, tokens)
        got = fixture.reachable(normal_form, tokens)

        if want:
            reached.append(tokens)

        if got != want:
            detail = (
                f"{normal_form.name} on {tokens!r} drops "
                f"{sorted(map(repr, want - got))} from the goal terms and adds "
                f"{sorted(map(repr, got - want))}"
            )

            fail_law(detail, law="goal completeness")

    law()

    if not reached:
        detail = (
            f"no generated sentence reached a goal term at all, so goal "
            f"completeness was never exercised on {normal_form.name}"
        )

        fail_law(detail, law="goal completeness")


NORMAL_FORM_LAWS: Final = (
    tags_are_declared,
    tagging_is_pure,
    filtering_preserves_goal_terms,
)
"""Normal-form laws in conformance order."""


def check_normal_form_laws[T](fixture: NormalFormFixture[T]) -> None:
    """Run every normal-form law against one filter.

    Parameters
    ----------
    fixture : NormalFormFixture[T]
        Normal-form fixture.

    Raises
    ------
    LawViolationError
        If any normal-form law fails.
    """
    for law in NORMAL_FORM_LAWS:
        law(fixture)

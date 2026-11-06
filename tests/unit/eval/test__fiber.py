from itertools import permutations, product
from math import factorial
from random import Random
from typing import TYPE_CHECKING, Final, cast

import pytest
from tests.conformance._suite import Conformance, ConformanceSuite
from tests.language import language

from lofi.eval import (
    FIBER_MEASURES,
    LogicalVocabulary,
    OneToOne,
    Predicate,
    fiber_floor,
    fiber_match,
    fiber_recovery,
    fiber_size,
    permuted_body,
)
from lofi.lf import (
    ProverBound,
    TruthTableOracle,
)
from lofi.lf.provers import ProofBudget, ProofService
from lofi.terms import Normalization

if TYPE_CHECKING:
    from lofi.terms import Term


LANGUAGE: Final = language(
    (
        "F",
        "a0",
        "a1",
        "bark",
        "c",
        "chase",
        "d",
        "f",
        "g",
        "legal",
        "own",
        "p0",
        "p1",
        "p2",
        "pet",
        "q0",
        "q1",
        "r",
        "r0",
        "rain",
        "u",
        "v",
        "w",
        "wet",
        "wide",
    ),
)

STORE: Final = LANGUAGE.store
CALCULUS: Final = LANGUAGE.calculus
SLOTS: Final = ("a0", "a1")
POLICY: Final = OneToOne(tie_break="least-gold-label")
TIMEOUT: Final = 30.0
FUEL: Final = 1_000
NORMALIZATION: Final = Normalization(
    fuel=FUEL,
    order="leftmost-outermost",
    eta="contract",
    associative_commutative=(),
)
BUDGET: Final = ProofBudget()
_SERVICES: list[ProofService] = []


def const(name: str) -> Term:
    return LANGUAGE.constant(name)


def apply(head: Term, *arguments: Term) -> Term:
    return LANGUAGE.apply(head, *arguments)


def conj(left: Term, right: Term) -> Term:
    return LANGUAGE.apply(LANGUAGE.constant("and"), left, right)


def disj(left: Term, right: Term) -> Term:
    return LANGUAGE.apply(LANGUAGE.constant("or"), left, right)


A0: Final = const("a0")
A1: Final = const("a1")

CHASE: Final = apply(const("chase"), A0, A1)
CHASE_SWAPPED: Final = apply(const("chase"), A1, A0)
OWN: Final = conj(apply(const("own"), A0, A1), apply(const("legal"), A0))
OWN_COMMUTED: Final = conj(apply(const("legal"), A0), apply(const("own"), A0, A1))
RAIN: Final = const("rain")

GOLD: Final = LogicalVocabulary(
    (
        Predicate("chase", 2, CHASE),
        Predicate("own", 2, OWN),
        Predicate("rain", 0, RAIN),
    ),
    slots=SLOTS,
)

RELABELED: Final = LogicalVocabulary(
    (
        Predicate("p0", 2, CHASE_SWAPPED),
        Predicate("p1", 2, OWN),
        Predicate("p2", 0, RAIN),
    ),
    slots=SLOTS,
)

FRAGMENT: Final = LANGUAGE.profile.fragment
BOUND: Final = ProverBound("truth-table", FRAGMENT)


def proof_service() -> ProofService:
    """Return a proof service over exhaustive truth assignments."""
    service = ProofService(
        lambda: TruthTableOracle(STORE, LANGUAGE.profile, bound=BOUND, atoms=8),
        profile=LANGUAGE.profile,
        workers=1,
        batch_size=8,
        timeout=TIMEOUT,
    )
    _SERVICES.append(service)
    return service


def teardown_module() -> None:
    for service in _SERVICES:
        service.close()


def test_the_two_measures_are_the_recovery_a_reader_reads() -> None:
    """Check that both headline measures are reachable by name."""
    recovered = fiber_recovery(STORE, RELABELED, GOLD, policy=POLICY, split="dev")

    assert {measure: recovered.measure(measure) for measure in FIBER_MEASURES} == {
        "up-to-fiber": recovered.up_to_fiber,
        "fixed-alphabet": recovered.fixed_alphabet,
    }


def test_a_relabeled_permuted_vocabulary_is_exact_only_up_to_the_fiber() -> None:
    """Check that the relabeling is recovered while the gold alphabet is not."""
    recovered = fiber_recovery(STORE, RELABELED, GOLD, policy=POLICY, split="dev")

    assert recovered.up_to_fiber.value == 1.0
    assert recovered.fixed_alphabet.value == 0.0
    assert dict(recovered.alignment.mapping) == {
        "p0": "chase",
        "p1": "own",
        "p2": "rain",
    }
    assert recovered.permutation("p0") == (1, 0)
    assert recovered.permutation("p1") == (0, 1)


def test_keeping_the_gold_names_recovers_every_relation_the_order_left_alone() -> None:
    """Check that a permuted argument order costs the fixed-alphabet measure."""
    kept = LogicalVocabulary(
        (
            Predicate("chase", 2, CHASE_SWAPPED),
            Predicate("own", 2, OWN),
            Predicate("rain", 0, RAIN),
        ),
        slots=SLOTS,
    )
    recovered = fiber_recovery(STORE, kept, GOLD, policy=POLICY, split="dev")

    assert recovered.up_to_fiber.value == 1.0
    assert recovered.fixed_alphabet.value == pytest.approx(2 / 3)


def test_a_vocabulary_missing_a_meaning_recovers_only_what_it_holds() -> None:
    """Check that an unrecoverable relation is not counted."""
    partial = LogicalVocabulary(
        (Predicate("q0", 2, CHASE), Predicate("q1", 2, apply(const("bark"), A0, A1))),
        slots=SLOTS,
    )
    recovered = fiber_recovery(STORE, partial, GOLD, policy=POLICY, split="dev")

    assert recovered.up_to_fiber.value == pytest.approx(1 / 3)
    assert recovered.up_to_fiber.components["induced_labels"] == 2.0
    assert recovered.up_to_fiber.components["gold_labels"] == 3.0


def test_equivalent_vocabularies_score_alike_and_syntax_alone_does_not() -> None:
    """Check that proof, not term shape, decides an identification."""
    reference = LogicalVocabulary((Predicate("own", 2, OWN),), slots=SLOTS)
    same_shape = LogicalVocabulary((Predicate("r0", 2, OWN),), slots=SLOTS)
    other_shape = LogicalVocabulary((Predicate("r0", 2, OWN_COMMUTED),), slots=SLOTS)
    asked = proof_service()

    assert OWN is not OWN_COMMUTED
    assert (
        fiber_recovery(
            STORE,
            other_shape,
            reference,
            policy=POLICY,
            split="dev",
            service=asked,
            budget=BUDGET,
        ).up_to_fiber.value
        == fiber_recovery(
            STORE,
            same_shape,
            reference,
            policy=POLICY,
            split="dev",
            service=asked,
            budget=BUDGET,
        ).up_to_fiber.value
        == 1.0
    )
    assert (
        fiber_recovery(
            STORE, other_shape, reference, policy=POLICY, split="dev"
        ).up_to_fiber.value
        == 0.0
    )


def test_permuting_a_relation_twice_returns_the_body_it_started_from() -> None:
    """Check that the argument permutations act as a group on a body."""
    once = permuted_body(STORE, GOLD, "chase", (1, 0))
    twice = permuted_body(
        STORE,
        LogicalVocabulary((Predicate("chase", 2, once),), slots=SLOTS),
        "chase",
        (1, 0),
    )

    assert once is not CHASE
    assert twice is CHASE
    assert permuted_body(STORE, GOLD, "chase", (0, 1)) is CHASE


def test_free_variable_slots_are_permuted_without_becoming_symbols() -> None:
    """Check that canonical open slots carry order without alphabet names."""
    first = STORE.fvar(LANGUAGE.term_sort, "slot0")
    second = STORE.fvar(LANGUAGE.term_sort, "slot1")
    body = apply(const("chase"), first, second)
    vocabulary = LogicalVocabulary(
        (Predicate("chase", 2, body),), slots=("slot0", "slot1")
    )

    assert permuted_body(STORE, vocabulary, "chase", (1, 0)) is apply(
        const("chase"), second, first
    )


def test_permuting_a_relation_that_names_no_slot_leaves_it_alone() -> None:
    """Check that a slot-free body is fixed by every argument permutation."""
    blind = LogicalVocabulary((Predicate("r", 2, RAIN),), slots=SLOTS)

    assert permuted_body(STORE, blind, "r", (1, 0)) is RAIN


def test_a_permutation_over_the_wrong_positions_is_refused() -> None:
    """Check that a permutation must order the relation's own arguments."""
    with pytest.raises(ValueError, match="is not a permutation"):
        permuted_body(STORE, GOLD, "chase", (0, 0))

    with pytest.raises(ValueError, match="is not a permutation"):
        permuted_body(STORE, GOLD, "rain", (0,))


def _group(vocabulary: LogicalVocabulary) -> int:
    """Count the group elements that leave a vocabulary where it was.

    Parameters
    ----------
    vocabulary : LogicalVocabulary
        The vocabulary whose stabilizer to enumerate.

    Returns
    -------
    int
        How many relabelings with argument permutations fix it.
    """
    symbols = vocabulary.symbols
    orders = tuple(
        tuple(permutations(range(vocabulary.predicate(symbol).arity)))
        for symbol in symbols
    )
    fixed = 0

    for relabeling in permutations(symbols):
        for chosen in product(*orders):
            moved = {
                relabeling[at]: permuted_body(
                    STORE, vocabulary, symbols[at], chosen[at]
                )
                for at in range(len(symbols))
            }

            if all(
                moved[symbol] is vocabulary.predicate(symbol).body for symbol in symbols
            ):
                fixed += 1

    return fixed


SYMMETRIC: Final = LogicalVocabulary(
    (
        Predicate("r", 2, RAIN),
        Predicate("u", 2, apply(const("f"), A0, A1)),
        Predicate("v", 2, apply(const("f"), A0, A1)),
        Predicate("w", 2, apply(const("f"), A0, A1)),
    ),
    slots=SLOTS,
)


def test_the_computed_fiber_size_matches_the_enumerated_one() -> None:
    """Check the closed-form stabilizer against a brute-force enumeration."""
    enumerated = _group(SYMMETRIC)
    whole = factorial(len(SYMMETRIC.symbols)) * 2 ** len(SYMMETRIC.symbols)

    assert fiber_size(STORE, SYMMETRIC, limit=5_040) == enumerated
    assert 1 < enumerated < whole


def test_a_vocabulary_with_no_symmetry_has_a_fiber_of_one() -> None:
    """Check that an asymmetric, distinguishable vocabulary is rigid."""
    rigid = LogicalVocabulary(
        (
            Predicate("u", 2, apply(const("f"), A0, A1)),
            Predicate("w", 2, apply(const("g"), A0, A1)),
        ),
        slots=SLOTS,
    )

    assert fiber_size(STORE, rigid, limit=5_040) == _group(rigid) == 1


def test_the_oracle_widens_the_fiber_a_syntactic_reading_misses() -> None:
    """Check that a semantically symmetric relation counts as symmetric."""
    symmetric = LogicalVocabulary(
        (
            Predicate(
                "r",
                2,
                disj(apply(const("f"), A0, A1), apply(const("f"), A1, A0)),
            ),
        ),
        slots=SLOTS,
    )

    assert fiber_size(STORE, symmetric, limit=5_040) == 1
    assert (
        fiber_size(
            STORE,
            symmetric,
            service=proof_service(),
            budget=BUDGET,
            limit=5_040,
        )
        == 2
    )


def test_a_relation_wider_than_the_permutation_limit_is_refused() -> None:
    """Check that the enumeration budget is a refusal and not a silent cut."""
    with pytest.raises(ValueError, match="exceed the limit"):
        fiber_size(STORE, GOLD, limit=1)


def test_the_floor_is_computed_at_each_signature_size() -> None:
    """Check that the floor moves with the size and is not one number."""
    pool = GOLD.predicates
    narrow = fiber_floor(
        STORE,
        GOLD,
        pool,
        size=1,
        policy=POLICY,
        split="dev",
        rng=Random(11),
        draws=32,
        level=0.95,
    )
    wide = fiber_floor(
        STORE,
        GOLD,
        pool,
        size=6,
        policy=POLICY,
        split="dev",
        rng=Random(11),
        draws=32,
        level=0.95,
    )

    narrowest = narrow.value
    widest = wide.value

    assert narrowest == pytest.approx(1 / len(GOLD.symbols))
    assert narrow.components["induced_labels"] == 1.0
    assert wide.components["induced_labels"] == 6.0
    assert narrowest is not None
    assert widest is not None
    assert widest > narrowest


def test_a_recovery_is_flagged_against_the_floor_it_is_read_with() -> None:
    """Check that a recovery at or below its floor reports itself below."""
    floor = fiber_floor(
        STORE,
        GOLD,
        GOLD.predicates,
        size=1,
        policy=POLICY,
        split="dev",
        rng=Random(5),
        draws=16,
        level=0.95,
    )
    exact = fiber_recovery(
        STORE, RELABELED, GOLD, policy=POLICY, split="dev", floor=floor
    )
    thin = fiber_recovery(
        STORE,
        LogicalVocabulary((Predicate("q0", 2, CHASE),), slots=SLOTS),
        GOLD,
        policy=POLICY,
        split="dev",
        floor=floor,
    )

    assert exact.beats_floor is True
    assert thin.up_to_fiber.value == pytest.approx(floor.value)
    assert thin.beats_floor is False
    assert exact.floor is floor


def test_a_recovery_without_a_floor_reports_no_flag() -> None:
    """Check that the flag is withheld rather than guessed."""
    assert (
        fiber_recovery(STORE, RELABELED, GOLD, policy=POLICY, split="dev").beats_floor
        is None
    )


def test_a_floor_refuses_an_empty_pool_a_zero_size_and_a_zero_draw() -> None:
    """Check that a floor over nothing is an error and not a zero."""
    with pytest.raises(ValueError, match="pool, which is empty"):
        fiber_floor(
            STORE,
            GOLD,
            (),
            size=1,
            policy=POLICY,
            split="dev",
            rng=Random(1),
            draws=4,
            level=0.95,
        )

    with pytest.raises(ValueError, match="signature size of at least one"):
        fiber_floor(
            STORE,
            GOLD,
            GOLD.predicates,
            size=0,
            policy=POLICY,
            split="dev",
            rng=Random(1),
            draws=4,
            level=0.95,
        )

    with pytest.raises(ValueError, match="at least one draw"):
        fiber_floor(
            STORE,
            GOLD,
            GOLD.predicates,
            size=1,
            policy=POLICY,
            split="dev",
            rng=Random(1),
            draws=0,
            level=0.95,
        )


def test_a_declared_permutation_settles_a_form_whose_arguments_moved() -> None:
    """Check that the declared orders are searched and not assumed."""
    gold = apply(const("chase"), const("c"), const("d"))
    predicted = apply(const("chase"), const("d"), const("c"))

    assert fiber_match(
        STORE,
        gold,
        predicted,
        {"chase": ((0, 1), (1, 0))},
        calculus=CALCULUS,
        normalization=NORMALIZATION,
    )
    assert not fiber_match(
        STORE,
        gold,
        predicted,
        {"chase": ((0, 1),)},
        calculus=CALCULUS,
        normalization=NORMALIZATION,
    )
    assert not fiber_match(
        STORE,
        gold,
        predicted,
        {},
        calculus=CALCULUS,
        normalization=NORMALIZATION,
    )
    assert fiber_match(
        STORE,
        gold,
        gold,
        {},
        calculus=CALCULUS,
        normalization=NORMALIZATION,
    )


def test_a_declared_permutation_moves_only_the_symbol_it_was_declared_for() -> None:
    """Check that an unrelated symbol's arguments stay where they were."""
    gold = apply(const("chase"), apply(const("pet"), const("c"), const("d")))
    predicted = apply(const("chase"), apply(const("pet"), const("d"), const("c")))

    assert not fiber_match(
        STORE,
        gold,
        predicted,
        {"chase": ((0, 1), (1, 0))},
        calculus=CALCULUS,
        normalization=NORMALIZATION,
    )
    assert fiber_match(
        STORE,
        gold,
        predicted,
        {"pet": ((0, 1), (1, 0))},
        calculus=CALCULUS,
        normalization=NORMALIZATION,
    )


def test_a_match_the_oracle_settles_is_not_settled_by_interning() -> None:
    """Check that an equivalent form with another shape needs proof."""
    left = conj(const("rain"), const("wet"))
    right = conj(const("wet"), const("rain"))

    assert not fiber_match(
        STORE,
        left,
        right,
        {},
        calculus=CALCULUS,
        normalization=NORMALIZATION,
    )
    assert fiber_match(
        STORE,
        left,
        right,
        {},
        calculus=CALCULUS,
        normalization=NORMALIZATION,
        service=proof_service(),
        budget=BUDGET,
    )


def test_a_vocabulary_refuses_a_repeated_symbol() -> None:
    """Check that a vocabulary names each relation once."""
    with pytest.raises(ValueError, match="declares a symbol twice"):
        LogicalVocabulary(
            (Predicate("chase", 2, CHASE), Predicate("chase", 2, OWN)), slots=SLOTS
        )


def test_a_vocabulary_refuses_a_relation_wider_than_its_slots() -> None:
    """Check that a relation cannot take an argument the slots do not name."""
    with pytest.raises(ValueError, match="takes 3 arguments"):
        LogicalVocabulary((Predicate("wide", 3, CHASE),), slots=SLOTS)


def test_a_vocabulary_refuses_a_repeated_slot() -> None:
    """Check that two argument positions cannot share a name."""
    with pytest.raises(ValueError, match="slots repeat a name"):
        LogicalVocabulary((Predicate("chase", 2, CHASE),), slots=("a0", "a0"))


def test_a_predicate_refuses_a_body_that_is_not_a_term() -> None:
    """Check that a relation's meaning must be a formula."""
    with pytest.raises(TypeError, match="body is not a term"):
        Predicate("chase", 2, cast("Term", "chase(a0, a1)"))


def test_a_vocabulary_refuses_an_entry_that_is_not_a_predicate() -> None:
    """Check that a vocabulary holds relations and nothing else."""
    with pytest.raises(TypeError, match="not a predicate"):
        LogicalVocabulary((cast("Predicate", CHASE),), slots=SLOTS)


def test_recovery_refuses_vocabularies_over_different_slots() -> None:
    """Check that two vocabularies must name their argument positions alike."""
    other = LogicalVocabulary((Predicate("chase", 2, CHASE),), slots=("b0", "b1"))

    with pytest.raises(ValueError, match="cannot be compared"):
        fiber_recovery(STORE, other, GOLD, policy=POLICY, split="dev")


def test_recovery_requires_a_normalization_treatment_with_its_calculus() -> None:
    with pytest.raises(ValueError, match="supplied together"):
        fiber_recovery(
            STORE,
            RELABELED,
            GOLD,
            policy=POLICY,
            split="dev",
            calculus=CALCULUS,
        )

    with pytest.raises(ValueError, match="supplied together"):
        fiber_recovery(
            STORE,
            RELABELED,
            GOLD,
            policy=POLICY,
            split="dev",
            normalization=NORMALIZATION,
        )


def test_recovery_refuses_a_vocabulary_that_is_not_one() -> None:
    """Check that the metric names what it was handed."""
    with pytest.raises(TypeError, match="not one"):
        fiber_recovery(
            STORE,
            cast("LogicalVocabulary", CHASE),
            GOLD,
            policy=POLICY,
            split="dev",
        )


def test_recovery_accepts_any_named_evidence_split() -> None:
    """Check that the metric leaves split roles to its declared caller."""
    recovered = fiber_recovery(
        STORE, RELABELED, GOLD, policy=POLICY, split="development-fold-2"
    )

    assert recovered.alignment.split == "development-fold-2"


def test_an_unknown_relation_is_a_key_error() -> None:
    """Check that asking for a relation the vocabulary lacks is an error."""
    with pytest.raises(KeyError):
        GOLD.predicate("bark")


class TestAPredicate(ConformanceSuite):
    """Tests predicates against value laws."""

    @pytest.fixture
    def conformance(self) -> Conformance[Predicate]:
        """Return the conformance the laws run a predicate through."""
        return Conformance(
            build=lambda: Predicate("chase", 2, CHASE),
            unequal=Predicate("chase", 2, OWN),
            operations=(
                ("symbol", lambda predicate: predicate.symbol),
                ("arity", lambda predicate: predicate.arity),
            ),
        )


class TestAVocabulary(ConformanceSuite):
    """Tests logical vocabularies against value laws."""

    @pytest.fixture
    def conformance(self) -> Conformance[LogicalVocabulary]:
        """Return the conformance the laws run a vocabulary through."""
        return Conformance(
            build=lambda: LogicalVocabulary(
                (Predicate("chase", 2, CHASE), Predicate("rain", 0, RAIN)),
                slots=SLOTS,
            ),
            unequal=LogicalVocabulary(
                (Predicate("chase", 2, OWN), Predicate("rain", 0, RAIN)),
                slots=SLOTS,
            ),
            operations=(
                ("slots", lambda vocabulary: vocabulary.slots),
                ("symbols", lambda vocabulary: vocabulary.symbols),
                ("arity", lambda vocabulary: vocabulary.predicate("chase").arity),
            ),
        )

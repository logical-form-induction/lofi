"""Tests for the labeled premise-hypothesis pairs and the draw that produces them."""

from math import nextafter
from typing import TYPE_CHECKING, Final, cast

import pytest
from tests.unit.data.synthetic.conftest import (
    ABSTAINING_TPTP,
    APPLICATION,
    NEGATED_ENTRIES,
    NEGATED_INVENTORY,
    SCOPED_ENTRIES,
    SCOPED_INVENTORY,
    SCOPING,
    built,
    grammar,
    interpretation,
    language,
    negated_generator,
)

from lofi.data.synthetic import (
    ComplexityMeter,
    ComplexityTarget,
    ComplexityUnreachableError,
    ConfusionReport,
    InferenceNoise,
    InferenceReport,
    LabelConfusion,
    LabelProportion,
    NoisedLabel,
    PairComplexityTarget,
    PairRelations,
    PairRelationUnreachableError,
    ProportionUnreachableError,
    RelationReport,
    SyntheticGenerator,
    SyntheticInference,
    Target,
    UniformLength,
    relation_complexity,
    synthetic_inference_item,
)
from lofi.data.synthetic.spec.interpretation import lower_interpretation
from lofi.deduce import ParsePolicy, Reachability, Retain
from lofi.lf import RELATION_LABELS, InferenceLabel
from lofi.spec import DeclarationError
from lofi.spec.grammar import GrammarIso
from lofi.spec.language import LanguageIso, registered_language

if TYPE_CHECKING:
    from tests.unit.data.synthetic.conftest import Generator

    from lofi._types import ItemId
    from lofi.terms import Term

TWO: Final = UniformLength(2, 2)
NO_TRIVIAL: Final = PairRelations(trivial=0.0, entailment=0)
LABELS: Final = RELATION_LABELS
ENTAILS: Final = InferenceLabel.ENTAILS
CONTRADICTS: Final = InferenceLabel.CONTRADICTS
NEUTRAL: Final = InferenceLabel.NEUTRAL
BALANCED: Final = LabelProportion(
    {ENTAILS: 1 / 3, CONTRADICTS: 1 / 3, NEUTRAL: 1 / 3}, tolerance=0.0
)


def _uniform_noise(
    rate: float,
    confusion: LabelConfusion | None = None,
    *,
    tolerance: float,
) -> InferenceNoise:
    """Return the same flip rate for every decided relation."""
    return InferenceNoise(
        dict.fromkeys(LABELS, rate),
        confusion=LabelConfusion.uniform() if confusion is None else confusion,
        tolerance=tolerance,
    )


def _run(
    generator: Generator | None = None,
    *,
    count: int = 12,
    seed: int = 7,
    candidates: int = 12,
    attempts: int = 400,
    complexity: PairComplexityTarget | None = None,
    proportion: LabelProportion | None = None,
    noise: InferenceNoise | None = None,
    relations: PairRelations = NO_TRIVIAL,
) -> SyntheticInference:
    held = negated_generator() if generator is None else generator

    return held.infer(
        count,
        seed=seed,
        premise=TWO,
        hypothesis=TWO,
        candidates=candidates,
        attempts=attempts,
        complexity=complexity,
        proportion=proportion,
        noise=noise,
        relations=relations,
    )


def test_a_quantifier_free_fragment_yields_forms_with_no_quantifier() -> None:
    """Parsing every retained form back finds no binder that quantifies."""
    generator = negated_generator()
    interpretation = generator.interpretation
    assert interpretation is not None
    meter = ComplexityMeter(
        interpretation.store,
        interpretation.language.profile,
        abstraction=interpretation.calculus.operators.lam,
    )
    run = _run(generator)

    assert run.pairs

    for pair in run.pairs:
        for gold in (pair.premise, pair.hypothesis):
            measured = meter.measure(interpretation.read_text(gold.text))

            assert measured.quantifiers == frozenset()
            assert measured.value("quantifier_depth") == 0


def test_a_universal_language_cannot_write_an_existential_meaning() -> None:
    """A quantifier the language does not declare has no notation to read."""
    with pytest.raises(DeclarationError, match="a-book"):
        built(
            "forall",
            SCOPED_ENTRIES,
            SCOPED_INVENTORY,
            SCOPING,
            "forall",
            prover=ABSTAINING_TPTP,
        )


def test_a_generator_without_an_interpretation_is_refused() -> None:
    """Pairs are labeled under an interpretation, and one is required."""
    generator = SyntheticGenerator(
        GrammarIso().forward(
            grammar("bare", NEGATED_ENTRIES, APPLICATION),
            registered_language("propositional"),
        ),
        policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
    )

    with pytest.raises(ValueError, match="interpretation"):
        generator.infer(
            1,
            seed=1,
            premise=TWO,
            hypothesis=TWO,
            candidates=2,
            attempts=4,
            relations=NO_TRIVIAL,
        )


def test_every_pair_retains_both_forms() -> None:
    """A pair carries the logical form of each side, not only its identifier."""
    run = _run()

    for pair in run.pairs:
        assert pair.premise.text
        assert pair.hypothesis.text
        assert pair.label in LABELS


def test_every_label_agrees_with_the_oracle_rerun_on_the_retained_forms() -> None:
    """Relabeling a pair from the text it retains reproduces its certificates."""
    generator = negated_generator()
    run = _run(generator)

    assert run.pairs
    assert all(generator.verify_inference(pair) for pair in run.pairs)


def test_a_relabeled_pair_with_another_label_is_declined() -> None:
    """A pair whose label is changed no longer verifies against the oracle."""
    generator = negated_generator()
    run = _run(generator)
    pair = next(p for p in run.pairs if p.label is not NEUTRAL)
    swapped = type(pair)(
        pair.premise,
        pair.hypothesis,
        label=NEUTRAL if pair.label is not NEUTRAL else ENTAILS,
        entailment=pair.entailment,
        contradiction=pair.contradiction,
    )

    assert not generator.verify_inference(swapped)


def test_the_three_relations_are_all_reachable() -> None:
    """A grammar holding negation decides pairs each of the three ways."""
    run = _run(count=60, candidates=24, attempts=800)

    for label in LABELS:
        assert run.inference.labels[label] > 0


def test_a_designated_negation_under_another_name_labels_contradictions() -> None:
    declared = language()
    renamed = declared.with_(
        constants=tuple(
            constant.with_(name="deny") if constant.name == "not" else constant
            for constant in declared.constants
        ),
        logic=declared.logic.with_(negation="deny"),
    )
    languages = LanguageIso()
    assembly, interpreted = lower_interpretation(
        grammar("renamed-negation", NEGATED_ENTRIES, APPLICATION),
        renamed,
        interpretation(NEGATED_INVENTORY),
        GrammarIso(languages),
        languages,
    )
    generator = SyntheticGenerator(
        assembly,
        policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
        interpretation=interpreted,
    )
    run = _run(generator, count=30, candidates=24, attempts=800)

    assert run.inference.labels[CONTRADICTS] > 0
    assert all(generator.verify_inference(pair) for pair in run.pairs)


def test_a_declared_proportion_is_achieved_within_tolerance() -> None:
    """Each relation fills its quota and the report reads back satisfied."""
    run = _run(count=9, proportion=BALANCED)

    assert run.inference.satisfied()
    assert {label: run.inference.labels[label] for label in LABELS} == {
        ENTAILS: 3,
        CONTRADICTS: 3,
        NEUTRAL: 3,
    }


def test_an_unreachable_proportion_is_refused_with_the_achieved_one_named() -> None:
    """A quota no candidate fills raises, naming what the draw did reach."""
    generator = built(
        "decided-only",
        (
            ("john", "NP", "john"),
            ("sleeps", "S\\NP", "\\x. sleep(x)"),
            ("wakes", "S\\NP", "\\x. ~sleep(x)"),
        ),
        NEGATED_INVENTORY,
        APPLICATION,
    )

    with pytest.raises(ProportionUnreachableError) as caught:
        generator.infer(
            3,
            seed=1,
            premise=TWO,
            hypothesis=TWO,
            candidates=2,
            attempts=6,
            relations=NO_TRIVIAL,
            proportion=LabelProportion(
                {ENTAILS: 0.0, CONTRADICTS: 0.0, NEUTRAL: 1.0}, tolerance=0.0
            ),
        )

    assert caught.value.attempts == 6
    assert dict(caught.value.achieved) == {}
    assert dict(caught.value.reached) == {CONTRADICTS: 2}
    assert (
        "the draw achieved entails=0, contradicts=0, neutral=0, "
        "and the last quota's candidates reached entails=0, contradicts=2, "
        "neutral=0, with undecided=0"
    ) in str(caught.value)


def test_a_complexity_band_bounds_every_accepted_form() -> None:
    """Every achieved bucket lies in the band, and the report reads satisfied."""
    run = _run(
        count=6,
        complexity=PairComplexityTarget(
            premise=ComplexityTarget({"connective_count": Target(1, 0)}),
            overlap=Target(1, 0),
        ),
    )

    assert run.complexity.satisfied()
    assert set(run.complexity.premise["connective_count"].counts) == {1}
    assert set(run.complexity.overlap.counts) == {1}


def test_an_unreachable_complexity_band_is_refused() -> None:
    """A band no candidate falls in exhausts the budget and raises."""
    with pytest.raises(ComplexityUnreachableError):
        _run(
            count=1,
            attempts=5,
            complexity=PairComplexityTarget(
                premise=ComplexityTarget({"predicate_count": Target(99, 0)})
            ),
        )


def test_the_two_sides_are_drawn_from_their_own_distributions() -> None:
    """Each side's length report names the distribution that side was drawn from."""
    generator = negated_generator()
    run = generator.infer(
        4,
        seed=3,
        premise=UniformLength(2, 2),
        hypothesis=UniformLength(1, 2),
        candidates=6,
        attempts=200,
        relations=NO_TRIVIAL,
    )

    assert run.premises.length.distribution == UniformLength(2, 2)
    assert run.hypotheses.length.distribution == UniformLength(1, 2)
    assert set(run.premises.length.achieved.counts) == {2}


def test_equal_seeds_draw_equal_runs() -> None:
    """Two runs at one seed hold the same pairs in the same order."""
    assert _run(seed=5).pairs == _run(seed=5).pairs


def test_a_negative_count_is_refused() -> None:
    """A run cannot be asked for a negative number of pairs."""
    with pytest.raises(ValueError, match="pair count"):
        _run(count=-1)


def test_a_zero_pool_is_refused() -> None:
    """A run needs at least one string on each side to pair among."""
    with pytest.raises(ValueError, match="candidate count"):
        _run(candidates=0)


def test_a_zero_attempt_budget_is_refused() -> None:
    """A run needs at least one candidate per accepted pair."""
    with pytest.raises(ValueError, match="attempt budget"):
        _run(attempts=0)


def test_quotas_apportion_by_largest_remainder() -> None:
    """Quotas sum to the requested count however the shares round."""
    quotas = BALANCED.quotas(10)

    assert sum(quotas.values()) == 10
    assert set(quotas) == set(LABELS)


def test_shares_must_name_every_decided_label() -> None:
    """A proportion that leaves a relation unnamed is refused."""
    with pytest.raises(ValueError, match="does not name every decided label"):
        LabelProportion({ENTAILS: 1.0}, tolerance=0.0)


def test_undecided_pairs_are_outside_the_proportions() -> None:
    """The shares are taken over the decided pairs alone."""
    report = InferenceReport(
        {ENTAILS: 1, CONTRADICTS: 1, NEUTRAL: 2},
        undecided=4,
        requested=None,
        rejected=0,
    )

    assert report.total == 8
    assert report.decided == 4
    assert report.proportions()[NEUTRAL] == 0.5


def test_a_report_with_no_decided_pair_reports_zero_shares() -> None:
    """An all-undecided run reports a zero share for each relation."""
    report = InferenceReport({}, undecided=3, requested=None, rejected=0)

    assert report.decided == 0
    assert all(report.proportions()[label] == 0.0 for label in LABELS)


@pytest.mark.parametrize("label", LABELS)
def test_every_label_is_countable(label: InferenceLabel) -> None:
    """A report holds a bucket for every declared label, named or not."""
    report = InferenceReport({label: 2}, undecided=0, requested=None, rejected=0)

    assert report.labels[label] == 2
    assert report.total == 2


def test_the_relation_is_decided_in_the_declared_direction() -> None:
    """A conjunction entails its conjunct, and the premise is the side assumed."""
    run = _run(
        count=6,
        complexity=PairComplexityTarget(
            premise=ComplexityTarget(
                {
                    "connective_count": Target(1, 0),
                    "predicate_count": Target(2, 0),
                }
            ),
            hypothesis=ComplexityTarget(
                {
                    "connective_count": Target(0, 0),
                    "predicate_count": Target(1, 0),
                }
            ),
            overlap=Target(1, 0),
        ),
    )

    assert run.inference.labels[ENTAILS] > 0
    assert all(pair.premise.text != pair.hypothesis.text for pair in run.pairs)


def test_each_side_is_drawn_from_its_own_pool() -> None:
    """A length only one side draws at reaches forms the other side never holds."""
    generator = negated_generator()
    run = generator.infer(
        8,
        seed=4,
        premise=UniformLength(3, 3),
        hypothesis=UniformLength(2, 2),
        candidates=8,
        attempts=400,
        relations=NO_TRIVIAL,
    )

    assert run.pairs
    assert all("love" in pair.premise.text for pair in run.pairs)
    assert all("love" not in pair.hypothesis.text for pair in run.pairs)


ENTAILMENT_TO_NEUTRAL: Final = LabelConfusion({ENTAILS: {NEUTRAL: 1.0}})
UNIFORM: Final = LabelConfusion.uniform()


def test_a_uniform_confusion_splits_each_relations_flips_evenly() -> None:
    """Each relation sends half its flips to each of the two it is not."""
    for source in LABELS:
        row = UNIFORM.rows[source]

        assert row[source] == 0.0
        assert all(row[target] == 0.5 for target in LABELS if target != source)


def test_a_relation_with_no_row_is_never_confused() -> None:
    """A relation the structure leaves out sends no flip anywhere."""
    assert ENTAILMENT_TO_NEUTRAL.confuses(ENTAILS)
    assert not ENTAILMENT_TO_NEUTRAL.confuses(CONTRADICTS)

    with pytest.raises(ValueError, match="lands nowhere"):
        ENTAILMENT_TO_NEUTRAL.lands(CONTRADICTS, 0.0)


def test_a_coarsened_output_label_is_not_a_proof_relation() -> None:
    with pytest.raises(ValueError, match="not a decided label"):
        LabelConfusion({InferenceLabel.NON_ENTAILS: {ENTAILS: 1.0}})


def test_the_destinations_partition_the_unit_interval_in_report_order() -> None:
    """A flip lands on the destination whose stretch of the interval holds the draw."""
    assert UNIFORM.lands(NEUTRAL, 0.0) is ENTAILS
    assert UNIFORM.lands(NEUTRAL, 0.49) is ENTAILS
    assert UNIFORM.lands(NEUTRAL, 0.5) is CONTRADICTS
    assert UNIFORM.lands(NEUTRAL, 0.99) is CONTRADICTS


def test_a_draw_beyond_the_rounded_stretches_lands_on_the_last_destination() -> None:
    """Shares that sum just short of one leave a sliver owned by the last stretch."""
    confusion = LabelConfusion(
        {ENTAILS: {ENTAILS: 0.06, CONTRADICTS: 0.57, NEUTRAL: 0.37}}
    )

    assert confusion.lands(ENTAILS, nextafter(1.0, 0.0)) is NEUTRAL


def test_a_draw_outside_the_unit_interval_is_refused() -> None:
    """A flip is drawn at a point of the unit interval and nowhere else."""
    with pytest.raises(ValueError, match="unit interval"):
        UNIFORM.lands(ENTAILS, 1.0)


def test_the_declared_matrix_spreads_the_rate_and_keeps_the_rest() -> None:
    """A confused relation keeps one less the rate and spreads the rate by the row."""
    declared = _uniform_noise(
        0.4, confusion=ENTAILMENT_TO_NEUTRAL, tolerance=0.0
    ).matrix()

    assert declared[ENTAILS][ENTAILS] == pytest.approx(0.6)
    assert declared[ENTAILS][NEUTRAL] == pytest.approx(0.4)
    assert declared[ENTAILS][CONTRADICTS] == 0.0
    assert declared[CONTRADICTS][CONTRADICTS] == 1.0
    assert declared[NEUTRAL][NEUTRAL] == 1.0


def test_class_conditional_noise_changes_only_the_declared_relation() -> None:
    """A sparse rate table leaves every relation it omits untouched."""
    declared = InferenceNoise(
        {ENTAILS: 1.0}, confusion=ENTAILMENT_TO_NEUTRAL, tolerance=0.0
    )

    assert declared.rate(ENTAILS) == 1.0
    assert declared.rate(CONTRADICTS) == 0.0
    assert declared.matrix()[ENTAILS][NEUTRAL] == 1.0
    assert declared.matrix()[CONTRADICTS][CONTRADICTS] == 1.0


def test_a_zero_rate_declares_the_identity_matrix() -> None:
    """At no rate every relation keeps all of its mass whatever the structure."""
    declared = _uniform_noise(0.0, confusion=UNIFORM, tolerance=0.0).matrix()

    assert all(declared[source][source] == 1.0 for source in LABELS)


def test_a_record_never_flips_an_undecided_pair() -> None:
    """A pair bearing no relation has none to flip, in either direction."""
    with pytest.raises(ValueError, match="absence of a decision"):
        NoisedLabel(None, ENTAILS)

    with pytest.raises(ValueError, match="absence of a decision"):
        NoisedLabel(ENTAILS, None)


def test_a_record_reads_back_whether_it_was_flipped() -> None:
    """A record holds the oracle's relation beside the emitted one."""
    kept = NoisedLabel.of(ENTAILS)
    moved = NoisedLabel(ENTAILS, NEUTRAL)

    assert kept.oracle is ENTAILS
    assert kept.observed is ENTAILS
    assert not kept.flipped
    assert moved.oracle is ENTAILS
    assert moved.observed is NEUTRAL
    assert moved.flipped


def test_the_report_counts_the_oracle_against_the_observed_relation() -> None:
    """The matrix counts each pair in the row of the relation the oracle reached."""
    requested = _uniform_noise(1.0, confusion=ENTAILMENT_TO_NEUTRAL, tolerance=0.0)
    report = ConfusionReport.of(
        requested,
        (
            NoisedLabel(ENTAILS, NEUTRAL),
            NoisedLabel(ENTAILS, NEUTRAL),
            NoisedLabel.of(CONTRADICTS),
            NoisedLabel.of(None),
        ),
    )

    assert report.counts[ENTAILS][NEUTRAL] == 2
    assert report.counts[CONTRADICTS][CONTRADICTS] == 1
    assert report.undecided == 1
    assert report.decided == 3
    assert report.flip_count == 2
    assert report.achieved_rate == pytest.approx(2 / 3)
    assert report.satisfied()


def test_a_relation_no_pair_bears_is_left_out_of_the_achieved_matrix() -> None:
    """A row nothing measures is not reported and does not fail the tolerance."""
    requested = _uniform_noise(0.5, confusion=UNIFORM, tolerance=0.0)
    report = ConfusionReport.of(requested, (NoisedLabel(ENTAILS, NEUTRAL),))

    assert set(report.matrix()) == {"entails"}
    assert report.matrix()[ENTAILS][NEUTRAL] == 1.0


def test_an_achieved_row_outside_the_tolerance_is_not_satisfied() -> None:
    """A row half a point from the declared one fails a tolerance below that."""
    requested = _uniform_noise(0.5, confusion=UNIFORM, tolerance=0.4)
    report = ConfusionReport.of(requested, (NoisedLabel(ENTAILS, NEUTRAL),))

    assert not report.satisfied()


def test_a_zero_rate_reproduces_the_run_drawn_without_noise() -> None:
    """At no rate the pairs, the records and the matrix are the clean run's."""
    clean = _run()
    quiet = _run(noise=_uniform_noise(0.0, confusion=UNIFORM, tolerance=0.0))
    report = quiet.noise

    assert quiet.pairs == clean.pairs
    assert quiet.labels == clean.labels
    assert quiet.inference == clean.inference
    assert clean.noise is None
    assert report is not None
    assert report.flip_count == 0
    assert all(report.matrix()[source][source] == 1.0 for source in report.matrix())
    assert report.satisfied()


def test_every_noised_pair_retains_its_oracle_label() -> None:
    """However a relation is emitted, the record keeps what the oracle reached."""
    run = _run(
        count=24,
        attempts=800,
        noise=_uniform_noise(1.0, confusion=UNIFORM, tolerance=0.0),
    )

    assert len(run.labels) == len(run.pairs)
    assert all(
        record.oracle == pair.label
        for record, pair in zip(run.labels, run.pairs, strict=True)
    )


def test_a_full_rate_flips_every_decided_pair() -> None:
    """At the full rate no decided pair is emitted under its own relation."""
    run = _run(
        count=24,
        attempts=800,
        noise=_uniform_noise(1.0, confusion=UNIFORM, tolerance=0.0),
    )
    report = run.noise

    assert report is not None
    assert report.decided > 0
    assert report.flip_count == report.decided
    assert all(
        record.observed != record.oracle
        for record in run.labels
        if record.oracle is not None
    )


def test_a_relation_the_confusion_leaves_out_is_never_flipped() -> None:
    """Only the relations the structure routes are confused, whatever the rate."""
    run = _run(
        count=24,
        attempts=800,
        noise=_uniform_noise(1.0, confusion=ENTAILMENT_TO_NEUTRAL, tolerance=0.0),
    )
    report = run.noise

    assert report is not None
    assert report.counts[ENTAILS][NEUTRAL] == run.inference.labels[ENTAILS]
    assert report.counts[ENTAILS][ENTAILS] == 0
    assert all(
        record.observed == record.oracle
        for record in run.labels
        if record.oracle is not ENTAILS
    )
    assert report.satisfied()


@pytest.mark.parametrize("seed", range(5))
def test_the_achieved_matrix_lies_within_tolerance_of_the_declared_one(
    seed: int,
) -> None:
    """A balanced run at half the rate measures the declared matrix, from the report."""
    run = _run(
        count=150,
        seed=seed,
        candidates=16,
        attempts=2000,
        proportion=BALANCED,
        noise=_uniform_noise(0.5, confusion=UNIFORM, tolerance=0.2),
    )
    report = run.noise

    assert report is not None
    assert set(report.matrix()) == set(LABELS)
    assert report.satisfied()


def test_the_pairs_drawn_at_a_seed_are_the_same_at_every_rate() -> None:
    """The flips are drawn last, so no rate moves the pairs or their labels."""
    clean = _run(seed=3)
    noised = _run(seed=3, noise=_uniform_noise(1.0, confusion=UNIFORM, tolerance=0.0))

    assert noised.pairs == clean.pairs
    assert noised.inference.labels == clean.inference.labels


def _rebuilt(
    run: SyntheticInference,
    labels: tuple[NoisedLabel, ...],
    noise: ConfusionReport | None,
) -> SyntheticInference:
    return SyntheticInference(
        seed=run.seed,
        pairs=run.pairs,
        labels=labels,
        noise=noise,
        premises=run.premises,
        hypotheses=run.hypotheses,
        complexity=run.complexity,
        inference=run.inference,
        relations=run.relations,
        semantic=run.semantic,
    )


def test_a_record_disagreeing_with_its_pair_is_refused() -> None:
    """A record that renames the oracle's relation is not a record of that pair."""
    run = _run(count=4)
    swapped = (
        NoisedLabel.of(NEUTRAL if run.pairs[0].label is not NEUTRAL else ENTAILS),
        *run.labels[1:],
    )

    with pytest.raises(ValueError, match="another relation than the oracle"):
        _rebuilt(run, swapped, None)


def test_a_record_count_other_than_the_pair_count_is_refused() -> None:
    """The records cover the pairs one for one."""
    run = _run(count=4)

    with pytest.raises(ValueError, match="label records"):
        _rebuilt(run, run.labels[:-1], None)


def test_a_flipped_record_without_a_report_is_refused() -> None:
    """A run that flipped a relation says at what rate and to where."""
    run = _run(
        count=4,
        noise=_uniform_noise(1.0, confusion=UNIFORM, tolerance=0.0),
    )

    with pytest.raises(ValueError, match="without a confusion report"):
        _rebuilt(run, run.labels, None)


def test_a_report_that_does_not_measure_the_records_is_refused() -> None:
    """The report counts the records the run ships, not another run's."""
    requested = _uniform_noise(1.0, confusion=UNIFORM, tolerance=0.0)
    run = _run(count=4, noise=requested)
    unrelated = ConfusionReport.of(requested, (NoisedLabel(ENTAILS, NEUTRAL),))

    with pytest.raises(ValueError, match="the records measure"):
        _rebuilt(run, run.labels, unrelated)


def test_two_samples_drawn_from_different_grammars_are_refused() -> None:
    """Both sides of a run are drawn from one grammar."""
    run = _run(count=4)
    other = _run(
        built("renamed", NEGATED_ENTRIES, NEGATED_INVENTORY, APPLICATION), count=4
    )

    assert other.declaration != run.declaration

    with pytest.raises(ValueError, match="drawn from different grammars"):
        SyntheticInference(
            seed=run.seed,
            pairs=run.pairs,
            labels=run.labels,
            noise=None,
            premises=run.premises,
            hypotheses=other.hypotheses,
            complexity=run.complexity,
            inference=run.inference,
            relations=run.relations,
            semantic=run.semantic,
        )


def test_a_pair_naming_an_item_its_sample_does_not_hold_is_refused() -> None:
    """Each side of a pair is a form the sample that side was drawn as holds."""
    run = _run(count=4)

    with pytest.raises(ValueError, match="bears no such form in its sample"):
        SyntheticInference(
            seed=run.seed,
            pairs=run.pairs,
            labels=run.labels,
            noise=None,
            premises=run.hypotheses,
            hypotheses=run.premises,
            complexity=run.complexity,
            inference=run.inference,
            relations=run.relations,
            semantic=run.semantic,
        )


def _form(text: str) -> Term:
    interpretation = negated_generator().interpretation
    assert interpretation is not None

    return interpretation.read_text(text)


def test_alphabetic_variants_lie_zero_apart_and_other_forms_do_not() -> None:
    """Bound variables are nameless, so renaming one changes nothing."""
    assert relation_complexity(_form("\\x. sleep(x)"), _form("\\y. sleep(y)")) == 0
    assert relation_complexity(_form("sleep(john)"), _form("sleep(john)")) == 0
    assert relation_complexity(_form("sleep(john)"), _form("~sleep(john)")) == 2
    assert relation_complexity(
        _form("sleep(john) & snore(john)"), _form("sleep(john)")
    ) == relation_complexity(_form("sleep(john)"), _form("sleep(john) & snore(john)"))
    assert relation_complexity(_form("sleep(john)"), _form("sleep(mary)")) == 4


def test_a_relation_policy_sets_trivial_pairs_and_bounds_entailing_complexity() -> None:
    policy = PairRelations(trivial=0.25, entailment=3)

    assert policy.quota(10) == 2
    assert PairRelations(trivial=0.35, entailment=0).quota(10) == 4
    assert NO_TRIVIAL.quota(10) == 0
    assert policy.admits(ENTAILS, 3)
    assert not policy.admits(ENTAILS, 2)
    assert policy.admits(NEUTRAL, 0)
    assert policy.admits(None, 0)
    assert policy != PairRelations(trivial=0.25, entailment=0)
    assert hash(policy) == hash(PairRelations(trivial=0.25, entailment=3))
    assert repr(policy) == "PairRelations(trivial=0.25, entailment=3)"
    assert policy != "policy"


@pytest.mark.parametrize(
    ("trivial", "entailment", "error"),
    [(1.5, 0, ValueError), (-0.1, 0, ValueError), (0.0, -1, ValueError)],
)
def test_a_relation_policy_refuses_a_bad_bound(
    trivial: float, entailment: int, error: type[Exception]
) -> None:
    with pytest.raises(error):
        PairRelations(trivial=trivial, entailment=entailment)


def test_a_relation_report_is_satisfied_by_the_declared_share_alone() -> None:
    half = PairRelations(trivial=0.5, entailment=0)
    report = RelationReport(half, pairs=4, trivial=2, screened=3)

    assert report.satisfied()
    assert not RelationReport(NO_TRIVIAL, pairs=4, trivial=1, screened=0).satisfied()
    assert not RelationReport(half, pairs=4, trivial=1, screened=0).satisfied()
    assert report == RelationReport(half, pairs=4, trivial=2, screened=3)
    assert report != RelationReport(half, pairs=4, trivial=2, screened=4)
    assert "screened=3" in repr(report)

    with pytest.raises(ValueError, match="are trivial"):
        RelationReport(NO_TRIVIAL, pairs=1, trivial=2, screened=0)


def test_a_draw_holds_no_trivial_pair_by_default() -> None:
    """Identical pairs, which entail trivially, are drawn again."""
    run = _run(count=40, candidates=24, attempts=800)

    assert run.relations.requested == NO_TRIVIAL
    assert run.relations.trivial == 0
    assert run.relations.screened > 0
    assert all(pair.premise.text != pair.hypothesis.text for pair in run.pairs)


@pytest.mark.parametrize("share", [0.25, 0.5])
def test_a_declared_share_of_trivial_pairs_is_drawn_exactly(share: float) -> None:
    policy = PairRelations(trivial=share, entailment=0)
    run = _run(count=40, candidates=24, attempts=800, relations=policy)
    identical = sum(pair.premise.text == pair.hypothesis.text for pair in run.pairs)

    assert run.relations.trivial == identical == round(share * 40)
    assert run.relations.satisfied()


def test_a_least_entailing_complexity_bounds_every_entailing_pair() -> None:
    least = 3
    run = _run(
        count=9,
        candidates=24,
        attempts=2000,
        proportion=BALANCED,
        relations=PairRelations(trivial=0.0, entailment=least),
    )
    entailing = [pair for pair in run.pairs if pair.label is ENTAILS]

    assert len(entailing) == 3
    assert all(
        relation_complexity(_form(pair.premise.text), _form(pair.hypothesis.text))
        >= least
        for pair in entailing
    )


def test_an_entailment_quota_no_complex_enough_pair_fills_is_refused() -> None:
    with pytest.raises(ProportionUnreachableError, match="entails=0, contradicts"):
        _run(
            count=3,
            candidates=24,
            attempts=200,
            proportion=LabelProportion(
                {ENTAILS: 1.0, CONTRADICTS: 0.0, NEUTRAL: 0.0}, tolerance=0.0
            ),
            relations=PairRelations(trivial=0.0, entailment=6),
        )


def test_a_draw_whose_every_candidate_is_trivial_is_refused() -> None:
    generator = built(
        "one-string",
        (("john", "NP", "john"), ("sleeps", "S\\NP", "\\x. sleep(x)")),
        NEGATED_INVENTORY,
        APPLICATION,
    )

    with pytest.raises(PairRelationUnreachableError, match="refused by"):
        _run(generator, count=2, candidates=4, attempts=20)

    assert (
        _run(
            generator,
            count=2,
            candidates=4,
            attempts=20,
            relations=PairRelations(trivial=1.0, entailment=0),
        ).relations.trivial
        == 2
    )


def test_retained_pair_splits_draw_from_disjoint_candidate_pools() -> None:
    generator = built(
        "one-string",
        (("john", "NP", "john"), ("sleeps", "S\\NP", "\\x. sleep(x)")),
        NEGATED_INVENTORY,
        APPLICATION,
    )

    run = generator.infer(
        2,
        seed=1,
        splits=("dev", "test"),
        premise=TWO,
        hypothesis=TWO,
        candidates=1,
        attempts=20,
        relations=PairRelations(trivial=1.0, entailment=0),
    )

    dev, test = run.pairs
    assert dev.split == "dev"
    assert test.split == "test"
    assert {dev.premise.item_id, dev.hypothesis.item_id}.isdisjoint(
        {test.premise.item_id, test.hypothesis.item_id}
    )


def test_a_confusion_row_keeping_its_relation_lowers_its_flip_rate() -> None:
    confusion = LabelConfusion({NEUTRAL: {NEUTRAL: 0.75, ENTAILS: 0.25}})
    noise = _uniform_noise(1.0, confusion=confusion, tolerance=0.0)

    assert confusion.confuses(NEUTRAL)
    assert confusion.kept(NEUTRAL) == 0.75
    assert noise.matrix()[NEUTRAL][NEUTRAL] == pytest.approx(0.75)
    assert noise.matrix()[NEUTRAL][ENTAILS] == pytest.approx(0.25)
    assert not LabelConfusion({ENTAILS: {ENTAILS: 1.0}}).confuses(ENTAILS)
    assert confusion.lands(NEUTRAL, 0.0) is ENTAILS
    assert confusion.lands(NEUTRAL, 0.5) is NEUTRAL


def test_entailment_only_data_can_carry_a_share_of_mislabeled_non_entailments() -> None:
    """Every neutral is emitted as entails, so all pairs read as entailment."""
    run = _run(
        count=10,
        candidates=24,
        attempts=2000,
        proportion=LabelProportion(
            {ENTAILS: 0.8, CONTRADICTS: 0.0, NEUTRAL: 0.2}, tolerance=0.0
        ),
        noise=InferenceNoise(
            {NEUTRAL: 1.0},
            confusion=LabelConfusion({NEUTRAL: {ENTAILS: 1.0}}),
            tolerance=0.0,
        ),
    )
    report = run.noise

    assert {record.observed for record in run.labels} == {ENTAILS}
    assert sum(record.oracle is NEUTRAL for record in run.labels) == 2
    assert report is not None
    assert report.satisfied()


def test_synthetic_inference_text_uses_its_declared_surface_boundary() -> None:
    item = synthetic_inference_item(
        cast("ItemId", "new-pair:premise"),
        "josé sleeps",
    )

    assert item.text == "josé sleeps"
    assert tuple(token.surface for token in item.tokens) == ("josé", "sleeps")
    assert tuple(
        (token.span.start, token.span.end) if token.span is not None else None
        for token in item.tokens
    ) == ((0, 5), (6, 12))

    for refused in ("", " john", "john ", "john  sleeps", "john\tsleeps"):
        with pytest.raises(ValueError, match="one ASCII space"):
            synthetic_inference_item(cast("ItemId", "refused"), refused)

import io
import json
import re
from json import loads
from typing import TYPE_CHECKING, Final, Literal, cast, override

import didactic.api as dx
import pytest
from hypothesis import given, reject, settings
from tests.declarations import SMALL, DataKind, DataSection, data_sections
from tests.unit.data.synthetic.conftest import (
    APPLICATION,
    NEGATED_ENTRIES,
    NEGATED_INVENTORY,
    grammar,
    interpretation,
    language,
)

from lofi.data.datasets import ReservedForms
from lofi.data.synthetic import (
    ComplexityUnreachableError,
    CorruptionExhaustedError,
    LabelConfusion,
    PairRelationUnreachableError,
    ProportionUnreachableError,
    RelationReport,
    ScopeUnreachableError,
    SyntheticInference,
)
from lofi.data.synthetic.spec.corpus import (
    CorruptionProportionSpec,
    IllTypingOperatorsSpec,
    IllTypingSpec,
    ScopeSpec,
    TargetSpec,
    UniformLengthSpec,
)
from lofi.data.synthetic.spec.inference import (
    INFERENCE_SOURCE_VERSION,
    INFERENCE_TARGET_VERSION,
    PLUGIN,
    ComplexityControlSpec,
    ConfusionRowSpec,
    InferenceNoiseRateSpec,
    InferenceNoiseSpec,
    InferenceSemanticsSpec,
    LabelProportionSpec,
    PairComplexitySpec,
    PairRelationsSpec,
    SyntheticInferenceDataset,
    SyntheticInferenceDecl,
    SyntheticInferenceLowerer,
    TabularLabelConfusionSpec,
    UniformLabelConfusionSpec,
    inference_bytes,
    inference_fingerprint,
)
from lofi.data.synthetic.spec.operators import (
    ArgumentSwapSpec,
    OverApplicationSpec,
    SortMismatchSpec,
    WeightedIllTypingOperatorSpec,
)
from lofi.data.synthetic.spec.pairing import OverlapMatchedPairingSpec
from lofi.deduce import ParsePolicy, Reachability, Retain
from lofi.lf import RELATION_LABELS, InferenceLabel
from lofi.run import ExperimentError
from lofi.run import launching as launching_module
from lofi.run.launching import ExperimentLaunching
from lofi.run.operations import TRAINING_OPERATIONS
from lofi.spec import (
    DERIVED_SEED,
    PACKAGED_CONFIG_ROOT,
    ConfigValueError,
    DeclarationError,
    FeaturizerSpec,
    RunSpec,
    SpecLoader,
    TrainerSpec,
    TrainingBatchKind,
    UnresolvedSeedError,
)
from lofi.spec.grammar import (
    GeneratedLexiconSpec,
    ParametricAmbiguityLawSpec,
    UniformRankLawSpec,
)
from lofi.spec.language import InfixNotationSpec
from lofi.spec.optimizers.adamw import AdamWOptimizerSpec
from lofi.spec.policy import ComputeDtype, DevicePolicySpec, MatmulPrecision

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from lofi._types import JsonValue


TWO: Final = UniformLengthSpec(min_length=2, max_length=2)
ENTAILS: Final = InferenceLabel.ENTAILS


def _rates(rate: float) -> tuple[InferenceNoiseRateSpec, ...]:
    """Return one shared rate spelled for every decided relation."""
    return tuple(
        InferenceNoiseRateSpec(relation=label, rate=rate) for label in RELATION_LABELS
    )


CONTRADICTS: Final = InferenceLabel.CONTRADICTS
NEUTRAL: Final = InferenceLabel.NEUTRAL
FOREST_POLICY: Final = ParsePolicy(Retain.FOREST, Reachability.ALL)


def _ill_typing(*, attempts: int) -> IllTypingSpec:
    return IllTypingSpec(
        proportion=CorruptionProportionSpec(value=0.5),
        operators=IllTypingOperatorsSpec(
            entries=(
                WeightedIllTypingOperatorSpec(operator=ArgumentSwapSpec()),
                WeightedIllTypingOperatorSpec(operator=OverApplicationSpec()),
                WeightedIllTypingOperatorSpec(operator=SortMismatchSpec()),
            )
        ),
        attempts=attempts,
    )


def _semantics(name: str) -> InferenceSemanticsSpec:
    """Return one named treatment of the test grammar's meanings."""
    return InferenceSemanticsSpec(
        language=language().with_(name=name),
        interpretation=interpretation(NEGATED_INVENTORY),
    )


def _inference_declaration(
    *,
    size: int = 9,
    seed: int | str = 7,
    complexity: PairComplexitySpec | None = None,
    proportion: LabelProportionSpec | None = None,
    label_noise: InferenceNoiseSpec | None = None,
) -> SyntheticInferenceDecl:
    return SyntheticInferenceDecl(
        labeling="three-way",
        retained={"validation": 0.25},
        grammar=grammar("negated", NEGATED_ENTRIES, APPLICATION),
        source=_semantics("source"),
        size=size,
        seed=seed,
        candidates=12,
        attempts=400,
        premise=TWO,
        hypothesis=TWO,
        complexity=complexity,
        proportion=proportion,
        label_noise=label_noise,
    )


def _lowered(declaration: SyntheticInferenceDecl) -> SyntheticInference:
    lowerer = SyntheticInferenceLowerer()
    run = lowerer.lower(declaration, FOREST_POLICY)
    lowerer.validate(declaration, run)

    return run


def test_a_synthetic_inference_corpus_declares_exact_retained_allocations() -> None:
    declaration = _inference_declaration().with_(retained={"dev": 0.2, "test": 0.2})

    assert declaration.retained_splits() == frozenset(("dev", "test"))
    assert declaration.retained_pair_side_derivation_splits() == frozenset(
        ("dev", "test")
    )

    with pytest.raises(dx.ValidationError, match="retained declares"):
        _inference_declaration().with_(retained={})


def test_an_inference_corpus_declares_its_training_split() -> None:
    declaration = _inference_declaration(size=2).with_(training_split="fit")
    checked = PLUGIN.check(declaration)
    dataset = PLUGIN.lower(checked, ParsePolicy(Retain.FOREST, Reachability.ALL))

    assert "training_split='fit'" in repr(PLUGIN.pairs(dataset))
    assert _inference_declaration().training_split == "train"

    with pytest.raises(dx.ValidationError, match="is the training split"):
        _inference_declaration().with_(training_split="validation")


def test_the_lowerer_names_its_route() -> None:
    """The lowerer implements the one declared version pair."""
    route = SyntheticInferenceLowerer().route

    assert route.source_version == INFERENCE_SOURCE_VERSION
    assert route.target_version == INFERENCE_TARGET_VERSION


def test_the_installed_dataset_plugin_lowers_and_validates_pairs() -> None:
    declaration = _inference_declaration(size=2)
    checked = PLUGIN.check(declaration)
    dataset = cast(
        "SyntheticInferenceDataset",
        PLUGIN.lower(checked, ParsePolicy(Retain.FOREST, Reachability.ALL)),
    )

    PLUGIN.validate(dataset)
    vocabulary = PLUGIN.vocabulary(
        dataset, ReservedForms(("<pad>", "<unk>"), unknown="<unk>")
    )

    assert dataset.fingerprint == inference_fingerprint(declaration)
    assert dataset.labeling.name == "three-way"
    assert len(dataset.inference.pairs) == 2
    assert dataset.source_interpretation.store is dataset.source_grammar.language.store
    assert vocabulary.forms
    assert dict(vocabulary.reserved) == {"<pad>": 0, "<unk>": 1}
    assert PLUGIN.pairs(dataset).corpus is dataset
    assert dataset.drawn is dataset.inference
    assert len(dataset.corpus_pairs) == len(dataset.inference.pairs)
    assert tuple(pair.relation for pair in dataset.corpus_pairs) == tuple(
        label.observed for label in dataset.inference.labels
    )
    assert tuple(pair.gold for pair in dataset.corpus_pairs) == tuple(
        (pair.premise.text, pair.hypothesis.text) for pair in dataset.inference.pairs
    )

    with pytest.raises(ValueError, match="not sentence training"):
        PLUGIN.sentences(dataset)


def test_a_declaration_lowers_to_a_run_of_the_declared_size() -> None:
    """The run holds as many pairs as the declaration asks for, at its seed."""
    declaration = _inference_declaration()
    run = _lowered(declaration)

    assert len(run.pairs) == declaration.size
    assert run.seed == declaration.seed


def test_the_inference_declaration_owns_only_fields_that_pair_drawing_uses() -> None:
    """Source semantics, scope and ill-typing belong to the pair draw."""
    declaration = SyntheticInferenceDecl(
        labeling="three-way",
        retained={"validation": 0.25},
        grammar=grammar("negated", NEGATED_ENTRIES, APPLICATION),
        source=_semantics("source"),
        scope=ScopeSpec(derivations=4, attempts=8),
        ill_typing=_ill_typing(attempts=8),
        size=1,
        seed=1,
        candidates=4,
        attempts=8,
        premise=TWO,
        hypothesis=TWO,
    )

    assert declaration.grammar.name == "negated"
    assert declaration.source.language.logic.prover.timeout == 5.0
    assert not hasattr(declaration, "language")
    assert not hasattr(declaration, "interpretation")
    assert not hasattr(declaration, "corpus")
    assert not hasattr(declaration, "timeout")


def test_a_declaration_without_an_interpretation_is_refused() -> None:
    """Pairs are labeled by proof, so the declaration must name an interpretation."""
    document = loads(inference_bytes(_inference_declaration()))
    del document["source"]["interpretation"]

    with pytest.raises(dx.ValidationError, match="interpretation"):
        SyntheticInferenceDecl.model_validate(document)


def test_the_superseded_flat_semantic_fields_are_refused() -> None:
    document = loads(inference_bytes(_inference_declaration()))
    source = document.pop("source")
    document["language"] = source["language"]
    document["interpretation"] = source["interpretation"]

    with pytest.raises(dx.ValidationError, match=r"language|interpretation"):
        SyntheticInferenceDecl.model_validate(document)


def test_the_declared_proportion_is_reached_and_validated() -> None:
    """A balanced target fills each quota and survives validation."""
    run = _lowered(
        _inference_declaration(
            proportion=LabelProportionSpec(
                entails=1 / 3, contradicts=1 / 3, neutral=1 / 3, tolerance=0.0
            )
        )
    )

    assert run.inference.satisfied()
    assert run.inference.labels[ENTAILS] == 3
    assert run.inference.labels[CONTRADICTS] == 3
    assert run.inference.labels[NEUTRAL] == 3


def test_an_unreachable_proportion_refuses_the_declaration() -> None:
    """A share no candidates can fill raises rather than silently missing the target."""
    entries = (
        ("john", "NP", "john"),
        ("sleeps", "S\\NP", "\\x. sleep(x)"),
        ("wakes", "S\\NP", "\\x. ~sleep(x)"),
    )
    declaration = SyntheticInferenceDecl(
        labeling="three-way",
        retained={"validation": 0.25},
        grammar=grammar("decided", entries, APPLICATION),
        source=_semantics("source"),
        size=4,
        seed=1,
        candidates=2,
        attempts=8,
        premise=TWO,
        hypothesis=TWO,
        proportion=LabelProportionSpec(
            entails=0.0, contradicts=0.0, neutral=1.0, tolerance=0.0
        ),
    )

    with pytest.raises(ProportionUnreachableError, match="last quota's candidates"):
        SyntheticInferenceLowerer().lower(declaration, FOREST_POLICY)


def test_overlap_matched_pairing_reaches_an_entailment_quota_independent_misses() -> (
    None
):
    target = LabelProportionSpec(
        entails=1.0,
        contradicts=0.0,
        neutral=0.0,
        tolerance=0.0,
    )
    independent = _conjoined().with_(
        size=2,
        seed=4,
        candidates=8,
        attempts=1,
        proportion=target,
    )

    with pytest.raises(ProportionUnreachableError):
        _lowered(independent)

    matched = _lowered(independent.with_(pairing=OverlapMatchedPairingSpec()))

    assert matched.inference.labels == {
        ENTAILS: 2,
        CONTRADICTS: 0,
        NEUTRAL: 0,
    }


def test_shares_that_do_not_sum_to_one_are_refused_by_the_declaration() -> None:
    """A proportion whose shares miss one is refused when it is declared."""
    with pytest.raises(dx.ValidationError, match=r"sums to 1\.16"):
        LabelProportionSpec(entails=0.5, contradicts=0.33, neutral=0.33, tolerance=0.0)

    assert LabelProportionSpec(
        entails=0.1, contradicts=0.2, neutral=0.7, tolerance=0.0
    ).build().shares == {"entails": 0.1, "contradicts": 0.2, "neutral": 0.7}


def test_a_complexity_declaration_bounds_the_achieved_histograms() -> None:
    """Every achieved bucket lies in its declared band, read from the report."""
    run = _lowered(
        _inference_declaration(
            size=6,
            complexity=PairComplexitySpec(
                premise=ComplexityControlSpec(
                    controls={"connective_count": TargetSpec(value=1, tolerance=0)}
                ),
                overlap=TargetSpec(value=1, tolerance=0),
            ),
        )
    )

    assert run.complexity.satisfied()
    assert set(run.complexity.premise["connective_count"].counts) == {1}
    assert set(run.complexity.overlap.counts) == {1}


def test_the_length_reports_name_the_declared_distributions() -> None:
    """Each side's report carries the distribution that side was drawn from."""
    run = _lowered(_inference_declaration())

    assert run.premises.length.distribution == TWO.build()
    assert run.hypotheses.length.distribution == TWO.build()


def test_canonical_bytes_rebuild_an_equal_declaration() -> None:
    """Validating a declaration's bytes gives back the same declaration."""
    declaration = _inference_declaration()
    rebuilt = SyntheticInferenceDecl.model_validate_json(inference_bytes(declaration))

    assert inference_bytes(rebuilt) == inference_bytes(declaration)
    assert inference_fingerprint(rebuilt) == inference_fingerprint(declaration)


def test_two_declarations_differing_in_seed_fingerprint_apart() -> None:
    """A seed is part of the declaration, so it changes the fingerprint."""
    assert inference_fingerprint(
        _inference_declaration(seed=1)
    ) != inference_fingerprint(_inference_declaration(seed=2))


def test_the_dataset_exposes_only_its_source_semantics() -> None:
    declaration = _inference_declaration(size=2)
    checked = PLUGIN.check(declaration)
    dataset = cast("SyntheticInferenceDataset", PLUGIN.lower(checked, FOREST_POLICY))

    assert dataset.source_grammar.language.name == "source"
    assert dataset.source_interpretation.language.name == "source"
    assert not hasattr(dataset, "grammar")
    assert not hasattr(dataset, "interpretation")


def test_validation_refuses_a_run_of_another_size() -> None:
    """A run holding another number of pairs than declared is declined."""
    lowerer = SyntheticInferenceLowerer()
    declaration = _inference_declaration()
    run = lowerer.lower(declaration, FOREST_POLICY)

    with pytest.raises(DeclarationError, match="another number of pairs"):
        lowerer.validate(_inference_declaration(size=declaration.size + 1), run)


def test_validation_refuses_a_run_at_another_seed() -> None:
    """A run drawn at another seed than declared is declined."""
    lowerer = SyntheticInferenceLowerer()
    run = lowerer.lower(_inference_declaration(), FOREST_POLICY)

    with pytest.raises(DeclarationError, match="another seed"):
        lowerer.validate(_inference_declaration(seed=99), run)


def test_validation_refuses_a_run_drawn_under_another_source_grammar() -> None:
    declaration = _inference_declaration()
    other = declaration.with_(
        grammar=declaration.grammar.with_(name="other-source-grammar")
    )
    lowerer = SyntheticInferenceLowerer()
    run = lowerer.lower(other, FOREST_POLICY)

    with pytest.raises(DeclarationError, match="another source grammar"):
        lowerer.validate(declaration, run)


def test_a_nonpositive_pool_is_refused_by_the_declaration() -> None:
    """A candidates of no strings has nothing to pair among."""
    with pytest.raises(dx.ValidationError, match="candidates"):
        _inference_declaration().with_(candidates=0)


def test_the_uniform_confusion_spelling_spreads_the_flips_evenly() -> None:
    """The uniform spelling builds the structure that splits every relation's flips."""
    assert UniformLabelConfusionSpec().build() == LabelConfusion.uniform()


def test_the_tabular_confusion_spelling_routes_only_the_relations_it_names() -> None:
    """A relation with no row is never confused, whatever the rate."""
    built = TabularLabelConfusionSpec(
        rows=(ConfusionRowSpec(oracle=ENTAILS, neutral=1.0),)
    ).build()

    assert built == LabelConfusion({ENTAILS: {NEUTRAL: 1.0}})
    assert built.confuses(ENTAILS)
    assert not built.confuses(NEUTRAL)


def test_a_tabular_confusion_listing_a_relation_twice_is_refused() -> None:
    """One relation has one row, since two would leave its destinations undefined."""
    with pytest.raises(dx.ValidationError, match="twice"):
        TabularLabelConfusionSpec(
            rows=(
                ConfusionRowSpec(oracle=ENTAILS, neutral=1.0),
                ConfusionRowSpec(oracle=ENTAILS, contradicts=1.0),
            )
        )


def test_a_confusion_row_may_keep_a_share_on_its_own_relation() -> None:
    """A row keeping part of its flips lowers that relation's flip rate."""
    kept = ConfusionRowSpec(oracle=NEUTRAL, neutral=0.75, entails=0.25)

    assert kept.shares() == {"entails": 0.25, "contradicts": 0.0, "neutral": 0.75}

    with pytest.raises(dx.ValidationError, match="rather than one or nothing"):
        ConfusionRowSpec(oracle=ENTAILS, neutral=0.5, contradicts=0.2)

    assert ConfusionRowSpec(oracle=ENTAILS).shares() == {
        "entails": 0.0,
        "contradicts": 0.0,
        "neutral": 0.0,
    }


def test_a_declared_confusion_is_achieved_and_validated() -> None:
    """The run reports the achieved matrix and the lowerer accepts it."""
    run = _lowered(
        _inference_declaration(
            label_noise=InferenceNoiseSpec(
                rates=_rates(1.0),
                confusion=TabularLabelConfusionSpec(
                    rows=(ConfusionRowSpec(oracle=ENTAILS, neutral=1.0),)
                ),
                tolerance=0.0,
            )
        )
    )
    report = run.noise

    assert report is not None
    assert report.satisfied()
    assert report.counts[ENTAILS][NEUTRAL] == run.inference.labels[ENTAILS]
    assert all(
        record.oracle == pair.label
        for record, pair in zip(run.labels, run.pairs, strict=True)
    )


def test_a_zero_rate_declaration_flips_nothing_and_still_reports() -> None:
    """A rate of nothing leaves every relation as the oracle reached it."""
    run = _lowered(
        _inference_declaration(
            label_noise=InferenceNoiseSpec(
                rates=_rates(0.0),
                confusion=UniformLabelConfusionSpec(),
                tolerance=0.0,
            )
        )
    )
    report = run.noise

    assert report is not None
    assert report.flip_count == 0
    assert report.satisfied()
    assert not any(record.flipped for record in run.labels)


def test_noise_rates_are_declared_by_relation() -> None:
    """A sparse rate declaration changes only the relation it names."""
    built = InferenceNoiseSpec(
        rates=(InferenceNoiseRateSpec(relation=ENTAILS, rate=1.0),),
        confusion=TabularLabelConfusionSpec(
            rows=(ConfusionRowSpec(oracle=ENTAILS, neutral=1.0),)
        ),
        tolerance=0.0,
    ).build()

    assert built.rate(ENTAILS) == 1.0
    assert built.rate(CONTRADICTS) == 0.0


def test_a_declared_rate_and_tolerance_are_the_ones_the_run_is_held_to() -> None:
    """Half the pairs of each relation are flipped, within the declared tolerance."""
    run = _lowered(
        _inference_declaration(
            size=30,
            proportion=LabelProportionSpec(
                entails=1 / 3, contradicts=1 / 3, neutral=1 / 3, tolerance=0.0
            ),
            label_noise=InferenceNoiseSpec(
                rates=_rates(0.5),
                confusion=UniformLabelConfusionSpec(),
                tolerance=0.4,
            ),
        )
    )
    report = run.noise

    assert report is not None
    assert report.satisfied()
    assert 0 < report.flip_count < report.decided


def test_validation_refuses_a_run_that_missed_the_declared_matrix() -> None:
    """A tolerance a small run cannot meet is refused rather than reported met."""
    lowerer = SyntheticInferenceLowerer()
    declaration = _inference_declaration(
        label_noise=InferenceNoiseSpec(
            rates=_rates(0.5),
            confusion=UniformLabelConfusionSpec(),
            tolerance=0.0,
        )
    )
    run = lowerer.lower(declaration, FOREST_POLICY)

    with pytest.raises(DeclarationError, match="confusion matrix unmet"):
        lowerer.validate(declaration, run)


def test_a_declaration_without_noise_lowers_to_a_run_without_a_report() -> None:
    """A run that was asked to flip nothing reports no confusion."""
    run = _lowered(_inference_declaration())

    assert run.noise is None
    assert not any(record.flipped for record in run.labels)


def test_a_noised_declaration_rebuilds_from_its_canonical_bytes() -> None:
    """The confusion spelling dispatches back to the variant it was declared as."""
    declaration = _inference_declaration(
        label_noise=InferenceNoiseSpec(
            rates=_rates(0.25),
            confusion=UniformLabelConfusionSpec(),
            tolerance=0.1,
        )
    )
    rebuilt = SyntheticInferenceDecl.model_validate_json(inference_bytes(declaration))
    noise = rebuilt.label_noise

    assert noise is not None
    assert type(noise.confusion) is UniformLabelConfusionSpec
    assert inference_fingerprint(rebuilt) == inference_fingerprint(declaration)


def test_validation_refuses_a_run_drawn_under_another_noise_declaration() -> None:
    """A run that flipped nothing does not answer a declaration that asked for flips."""
    lowerer = SyntheticInferenceLowerer()
    run = lowerer.lower(_inference_declaration(), FOREST_POLICY)

    with pytest.raises(DeclarationError, match="another noise declaration"):
        lowerer.validate(
            _inference_declaration(
                label_noise=InferenceNoiseSpec(
                    rates=_rates(0.5),
                    confusion=UniformLabelConfusionSpec(),
                    tolerance=0.1,
                )
            ),
            run,
        )


def test_validation_refuses_a_run_drawn_under_another_relation_policy() -> None:
    lowerer = SyntheticInferenceLowerer()
    run = lowerer.lower(_inference_declaration(), FOREST_POLICY)

    with pytest.raises(DeclarationError, match="another relation policy"):
        lowerer.validate(
            _inference_declaration().with_(relations=PairRelationsSpec(entailment=1)),
            run,
        )


def test_validation_refuses_a_run_holding_another_share_of_trivial_pairs() -> None:
    declaration = _inference_declaration()
    lowerer = SyntheticInferenceLowerer()
    run = lowerer.lower(declaration, FOREST_POLICY)
    forged = SyntheticInference(
        seed=run.seed,
        pairs=run.pairs,
        labels=run.labels,
        noise=run.noise,
        premises=run.premises,
        hypotheses=run.hypotheses,
        complexity=run.complexity,
        inference=run.inference,
        relations=RelationReport(
            run.relations.requested,
            pairs=run.relations.pairs,
            trivial=1,
            screened=run.relations.screened,
        ),
        semantic=run.semantic,
    )

    with pytest.raises(DeclarationError, match="another share of trivial pairs"):
        lowerer.validate(declaration, forged)


def test_a_rate_outside_the_unit_interval_is_refused_by_the_declaration() -> None:
    """A flip happens with a probability, so the rate lies in the unit interval."""
    with pytest.raises(dx.ValidationError, match="at most one"):
        InferenceNoiseSpec(
            rates=(InferenceNoiseRateSpec(relation=ENTAILS, rate=1.5),),
            confusion=UniformLabelConfusionSpec(),
            tolerance=0.0,
        )


CONJOINED_ENTRIES: Final = (
    *NEGATED_ENTRIES,
    ("and", "(S\\S)/S", "\\q. \\p. (p & q)"),
)
UP_TO_EIGHT: Final = UniformLengthSpec(min_length=2, max_length=8)
BALANCED: Final = LabelProportionSpec(
    entails=1 / 3, contradicts=1 / 3, neutral=1 / 3, tolerance=0.0
)

type Change = Callable[[SyntheticInferenceDecl], SyntheticInferenceDecl]


def _right_associative_conjunction(
    held: SyntheticInferenceDecl,
) -> SyntheticInferenceDecl:
    """Return a declaration whose conjunction groups in the other direction."""
    constants = []

    for constant in held.source.language.constants:
        if constant.name != "and":
            constants.append(constant)
            continue

        notation = constant.notation

        if not isinstance(notation, InfixNotationSpec):
            msg = "the propositional conjunction needs infix notation"
            raise TypeError(msg)

        constants.append(constant.with_(notation=notation.with_(associativity="right")))

    return held.with_(
        source=held.source.with_(
            language=held.source.language.with_(constants=tuple(constants))
        )
    )


def _conjoined() -> SyntheticInferenceDecl:
    return SyntheticInferenceDecl(
        labeling="three-way",
        retained={"validation": 0.25},
        grammar=grammar("conjoined", CONJOINED_ENTRIES, APPLICATION),
        source=_semantics("source"),
        size=9,
        seed=7,
        candidates=16,
        attempts=400,
        premise=UP_TO_EIGHT,
        hypothesis=UP_TO_EIGHT,
        proportion=BALANCED,
    )


def _drawn(run: SyntheticInference) -> tuple[object, ...]:
    return (
        run.pairs,
        run.labels,
        run.premises.ill_typed,
        run.hypotheses.ill_typed,
    )


FIELD_CHANGES: Final[dict[str, Change]] = {
    "labeling": lambda held: held.with_(labeling="rte"),
    "retained": lambda held: held.with_(retained={"test": 0.2}),
    "training_split": lambda held: held.with_(training_split="fit"),
    "grammar": lambda held: held.with_(
        grammar=grammar("renamed", CONJOINED_ENTRIES, APPLICATION)
    ),
    "source": _right_associative_conjunction,
    "scope": lambda held: held.with_(scope=ScopeSpec(derivations=1, attempts=200)),
    "ill_typing": lambda held: held.with_(ill_typing=_ill_typing(attempts=60)),
    "size": lambda held: held.with_(size=12),
    "seed": lambda held: held.with_(seed=8),
    "candidates": lambda held: held.with_(candidates=15),
    "attempts": lambda held: held.with_(attempts=1),
    "premise": lambda held: held.with_(
        premise=UniformLengthSpec(min_length=2, max_length=3)
    ),
    "hypothesis": lambda held: held.with_(
        hypothesis=UniformLengthSpec(min_length=2, max_length=3)
    ),
    "complexity": lambda held: held.with_(
        complexity=PairComplexitySpec(overlap=TargetSpec(value=1, tolerance=0))
    ),
    "proportion": lambda held: held.with_(
        proportion=LabelProportionSpec(
            entails=2 / 3, contradicts=0.0, neutral=1 / 3, tolerance=0.0
        )
    ),
    "label_noise": lambda held: held.with_(
        label_noise=InferenceNoiseSpec(
            rates=_rates(1.0),
            confusion=UniformLabelConfusionSpec(),
            tolerance=1.0,
        )
    ),
    "judgments": lambda held: held.with_(judgments=(0,) * held.size),
    "relations": lambda held: held.with_(relations=PairRelationsSpec(entailment=64)),
    "pairing": lambda held: held.with_(pairing=OverlapMatchedPairingSpec()),
}
DATA_CHANGES: Final = FIELD_CHANGES


def test_the_field_changes_cover_every_declared_field() -> None:
    assert set(FIELD_CHANGES) == set(SyntheticInferenceDecl.__field_specs__) - {"kind"}


@pytest.mark.parametrize("change", DATA_CHANGES.values(), ids=DATA_CHANGES.keys())
def test_every_data_field_changes_the_drawn_pairs_or_is_refused(
    change: Change,
) -> None:
    base = _conjoined()
    changed = change(base)
    lowerer = SyntheticInferenceLowerer()
    drawn = (
        base.inference_labels(),
        base.retained_splits(),
        base.judgments,
        *_drawn(_lowered(base)),
    )

    assert changed != base

    try:
        run = lowerer.lower(changed, FOREST_POLICY)
    except (
        ComplexityUnreachableError,
        CorruptionExhaustedError,
        DeclarationError,
        PairRelationUnreachableError,
        ProportionUnreachableError,
        ScopeUnreachableError,
    ):
        return

    lowerer.validate(changed, run)

    assert (
        changed.inference_labels(),
        changed.retained_splits(),
        changed.judgments,
        *_drawn(run),
    ) != drawn


def test_a_readings_target_redraws_the_strings_each_side_pairs_among() -> None:
    base = _lowered(_conjoined())
    scoped = _lowered(_conjoined().with_(scope=ScopeSpec(derivations=1, attempts=200)))

    for sample in (scoped.premises, scoped.hypotheses):
        assert sample.scope is not None
        assert sample.scope.exceeded > 0
        assert set(sample.scope.derivations.counts) == {1}

    assert base.premises.scope is None
    assert scoped.pairs != base.pairs


def test_declared_refused_forms_reach_both_sides_without_moving_the_pairs() -> None:
    base = _lowered(_conjoined())
    injected = _lowered(_conjoined().with_(ill_typing=_ill_typing(attempts=60)))

    assert injected.pairs == base.pairs
    assert injected.labels == base.labels

    for sample in (injected.premises, injected.hypotheses):
        report = sample.ill_typing
        assert report is not None
        assert report.injection.sample_size == len(sample.examples)
        assert report.injection.corrupted == len(sample.ill_typed)
        assert report.injection.achieved_proportion == pytest.approx(0.5)


def test_an_unmeetable_scope_target_is_refused_naming_the_achieved_readings() -> None:
    declaration = _conjoined().with_(
        scope=ScopeSpec(
            readings=TargetSpec(value=7.0, tolerance=0.0), derivations=8, attempts=20
        )
    )

    with pytest.raises(
        ScopeUnreachableError,
        match=(
            r"20 draws all fell outside the readings target Target\(7\.0, 0\.0\), "
            r"which the draws reach at Histogram\(\{1: 20\}\)"
        ),
    ):
        SyntheticInferenceLowerer().lower(declaration, FOREST_POLICY)


def test_the_lowerer_refuses_a_run_drawn_under_another_scope_or_injection() -> None:
    lowerer = SyntheticInferenceLowerer()
    run = lowerer.lower(_conjoined(), FOREST_POLICY)

    with pytest.raises(DeclarationError, match="another readings target"):
        lowerer.validate(
            _conjoined().with_(scope=ScopeSpec(derivations=1, attempts=200)), run
        )

    with pytest.raises(DeclarationError, match="refused forms against"):
        lowerer.validate(
            _conjoined().with_(ill_typing=_ill_typing(attempts=60)),
            run,
        )


def test_a_band_no_candidate_meets_is_refused_as_a_complexity_failure() -> None:
    declaration = _conjoined().with_(
        attempts=30,
        complexity=PairComplexitySpec(
            premise=ComplexityControlSpec(
                controls={"connective_count": TargetSpec(value=40, tolerance=0)}
            )
        ),
    )

    with pytest.raises(
        ComplexityUnreachableError,
        match=r"the candidates measured premise connective_count Histogram\(\{\d+: ",
    ):
        SyntheticInferenceLowerer().lower(declaration, FOREST_POLICY)


class InferencePairsTrainer(TrainerSpec):
    kind: Literal["inference-pairs-unit-trainer"] = "inference-pairs-unit-trainer"

    @override
    def dataset_requirements(self) -> dict[str, frozenset[TrainingBatchKind]]:
        return {"train": frozenset({TrainingBatchKind.INFERENCE_PAIRS})}

    @override
    def featurized_datasets(self) -> frozenset[str]:
        return frozenset({"train"})


class InferencePairsFeaturizer(FeaturizerSpec):
    kind: Literal["inference-pairs-unit-featurizer"] = "inference-pairs-unit-featurizer"


def test_the_inference_declaration_yields_labeled_pairs_and_their_sentences() -> None:
    assert _conjoined().training_batches() == frozenset(
        {
            TrainingBatchKind.SENTENCES,
            TrainingBatchKind.INFERENCE_PAIRS,
            TrainingBatchKind.GOLD_FORMS,
        }
    )


def test_declared_judgments_add_their_training_batch_capability() -> None:
    declaration = _conjoined().with_(judgments=(0,) * _conjoined().size)

    assert TrainingBatchKind.JUDGMENTS in declaration.training_batches()


def test_declared_judgments_are_aligned_to_the_generated_pairs() -> None:
    with pytest.raises(dx.ValidationError, match="declares 1 judgment responses"):
        _conjoined().with_(judgments=(0,))

    with pytest.raises(dx.ValidationError, match="zero-based levels"):
        _conjoined().with_(judgments=(-1,) * _conjoined().size)


def _run_document(neutral: float) -> dict[str, JsonValue]:
    dataset = json.loads(inference_bytes(_conjoined()))
    dataset["proportion"] = {
        "entails": 0.5,
        "contradicts": 0.33,
        "neutral": neutral,
        "tolerance": 0.0,
    }
    trainer = InferencePairsTrainer()
    policy = DevicePolicySpec(
        device="cpu",
        compute_dtype=ComputeDtype.FLOAT32,
        deterministic=True,
        matmul_precision=MatmulPrecision.HIGHEST,
        activation_memory_bytes=1,
    )

    return {
        "model": {"language": {"from": "data"}},
        "loss": None,
        "data": {
            "datasets": {
                "train": {
                    "dataset": dataset,
                    "featurizer": {
                        "kind": "inference-pairs-unit-featurizer",
                    },
                }
            }
        },
        "trainer": json.loads(trainer.model_dump_json()),
        "type_sampler": None,
        "optimizer": json.loads(AdamWOptimizerSpec().model_dump_json()),
        "device_policy": json.loads(policy.model_dump_json()),
        "scheduler": None,
        "parse": {"retain": "forest", "reachability": "all"},
        "release": {"checkpoint": "validation-best"},
        "sinks": {},
        "probes": {},
        "evaluators": {},
    }


def _write(path: Path, document: JsonValue) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8", newline="\n")

    return path


def test_a_proportion_past_one_fails_at_composition_and_no_dry_run_lists_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summing = _write(tmp_path / "summing.json", _run_document(0.17))
    past = _write(tmp_path / "past.json", _run_document(0.33))
    loader = SpecLoader(search_path=(tmp_path,))

    composed = loader.compose_traced(summing, schema=RunSpec).spec

    assert isinstance(composed.data.datasets["train"].dataset, SyntheticInferenceDecl)

    with pytest.raises(
        ConfigValueError,
        match=r"data\.datasets\.train\.dataset\.proportion .*sums to 1\.16",
    ):
        loader.compose_traced(past, schema=RunSpec)

    experiment = tmp_path / "experiments/headline.yaml"
    experiment.parent.mkdir(parents=True)
    (experiment.parent / "run.json").write_bytes(past.read_bytes())
    _write(
        experiment.parent / "preregistration.json",
        {
            "grid": {"data.datasets.train.dataset.seed": [1, 2]},
            "hypotheses": [
                {
                    "name": "synthetic-inference",
                    "left": "one",
                    "right": "two",
                    "family": "primary",
                }
            ],
        },
    )
    row: dict[str, JsonValue] = {"evaluator": "relations", "metric": "accuracy"}
    _write(
        experiment,
        {
            "role": "preflight",
            "freeze_id": "freeze/1",
            "config": {"kind": "run", "path": "run.json"},
            "launch": {"kind": "serial"},
            "failure": {"numerical": "final", "out_of_memory": "retry"},
            "sweep": {
                "grid": [
                    {"path": "data.datasets.train.dataset.seed", "values": [1, 2]},
                ],
                "seeds": [0, 1, 2],
                "expected_points": 6,
            },
            "seed_policy": {"root": 5},
            "rows": [row],
            "primary": row,
            "preregistration": "preregistration.json",
            "compute": {"kind": "unpriced"},
            "contraction": {"rungs": [{"kind": "fallback"}], "factorial": "none"},
            "stop_rules": [],
            "decision_point": {
                "date": "2027-01-01",
                "signal": {
                    "kind": "metric",
                    "row": row,
                    "statistic": "mean",
                    "direction": "above",
                    "threshold": 0.5,
                },
                "fallback": "No relation-proportion claim is made.",
            },
            "evaluation": {
                "alpha": 0.05,
                "correction": {"kind": "benjamini-hochberg"},
                "dependence": "prds",
                "minimum_seeds": 3,
                "seed_budget": {
                    "metric": "recovery",
                    "target": "mean",
                    "precision": 0.05,
                    "level": 0.95,
                },
            },
        },
    )
    output = tmp_path / "launch.json"
    monkeypatch.setattr(
        launching_module,
        "require_frozen_revision",
        lambda _checkout, _freeze: "0" * 40,
    )
    monkeypatch.setattr("lofi.spec._loader.load_plugins", lambda: ())

    with pytest.raises(ExperimentError, match=r"proportion .*sums to 1\.16"):
        ExperimentLaunching(checkout=tmp_path).launch(
            experiment,
            freeze=tmp_path / "FREEZE.json",
            root=None,
            overrides=("launch.kind=array",),
            dry_run=True,
            companions=(),
            training=TRAINING_OPERATIONS,
            environ={},
            out=output,
            stdout=io.StringIO(),
        )

    assert not output.exists()


@settings(max_examples=2 * SMALL.examples, deadline=None)
@given(data_sections(SMALL, DataKind.INFERENCE))
def test_a_generated_interpreted_grammar_draws_its_declared_count_of_labeled_pairs(
    section: DataSection,
) -> None:
    declaration = SyntheticInferenceDecl.model_validate_json(
        json.dumps(section.dataset)
    )
    symbols = {
        constant.name for constant in declaration.source.interpretation.constants
    }

    try:
        drawn = SyntheticInferenceLowerer().lower(declaration, FOREST_POLICY)
    except PairRelationUnreachableError:
        # An infeasible relation policy lies outside this success property's domain.
        reject()

    assert len(drawn.pairs) == declaration.size
    assert {pair.label for pair in drawn.pairs} <= set(RELATION_LABELS)
    assert all(
        set(re.findall(r"[A-Za-z_]\w*", form.text)) <= symbols
        for pair in drawn.pairs
        for form in (pair.premise, pair.hypothesis)
    )


def test_an_inference_seed_is_fixed_or_derived() -> None:
    assert _inference_declaration().with_(seed=DERIVED_SEED).seed == DERIVED_SEED

    with pytest.raises(dx.ValidationError, match="nonnegative integer or 'derived'"):
        _inference_declaration().with_(seed="later")


def test_derived_inference_and_lexicon_seeds_use_their_own_paths() -> None:
    asked: list[str] = []

    def derive(path: str) -> int:
        asked.append(path)

        return 40 + len(asked)

    lexicon = GeneratedLexiconSpec(
        forms=6,
        categories=3,
        category_shares=UniformRankLawSpec(),
        coupling=0.0,
        ambiguity=ParametricAmbiguityLawSpec(
            mean=1.0, dispersion=0.0, maximum=1, placement=0.0
        ),
        rank_law=UniformRankLawSpec(),
        seed=DERIVED_SEED,
        max_depth=2,
    )
    declared = _inference_declaration(seed=DERIVED_SEED)
    declared = declared.with_(grammar=declared.grammar.with_(lexicon=lexicon))
    resolved = declared.with_derived_seeds(derive, "data.datasets.train.dataset")

    assert asked == [
        "data.datasets.train.dataset.seed",
        "data.datasets.train.dataset.grammar.lexicon.seed",
    ]
    assert resolved.seed == 41
    assert resolved.grammar.lexicon == lexicon.with_(seed=42)


def test_an_inference_run_under_an_unresolved_derived_seed_is_refused() -> None:
    with pytest.raises(UnresolvedSeedError, match="inference run's seed is derived"):
        SyntheticInferenceLowerer().lower(
            _inference_declaration().with_(seed=DERIVED_SEED), FOREST_POLICY
        )


def _document(name: str, **changes: JsonValue) -> SyntheticInferenceDecl:
    declared = (
        SpecLoader()
        .compose_traced(
            PACKAGED_CONFIG_ROOT / f"data/datasets/train/dataset/{name}.yaml",
            schema=SyntheticInferenceDecl,
        )
        .spec
    )

    return SyntheticInferenceDecl.model_validate(
        {**loads(declared.model_dump_json()), **changes}
    )


@pytest.mark.parametrize(
    "name",
    [
        "synthetic-inference",
        "synthetic-inference-balanced",
        "synthetic-inference-entailment",
        "synthetic-inference-overlap",
    ],
)
def test_the_packaged_documents_hold_no_identical_pair_and_meet_their_proportions(
    name: str,
) -> None:
    declaration = _document(name)
    run = _lowered(declaration)

    assert declaration.relations == PairRelationsSpec()
    assert run.relations.trivial == 0
    assert all(pair.premise.text != pair.hypothesis.text for pair in run.pairs)
    assert run.inference.satisfied()


@pytest.mark.parametrize(
    "name",
    [
        "synthetic-inference-balanced",
        "synthetic-inference-entailment",
    ],
)
def test_the_entailing_pairs_of_a_document_are_lexical_entailments(name: str) -> None:
    run = _lowered(_document(name))
    entailing = {
        (pair.premise.text, pair.hypothesis.text)
        for pair in run.pairs
        if pair.label is ENTAILS
    }

    assert entailing
    assert ("sleep(john) & snore(john)", "sleep(john)") in entailing or (
        "sleep(mary) & snore(mary)",
        "sleep(mary)",
    ) in entailing


def test_entailment_only_data_can_be_declared_with_mislabeled_non_entailments() -> None:
    declaration = _document(
        "synthetic-inference",
        size=10,
        candidates=24,
        attempts=2000,
        proportion={
            "entails": 0.8,
            "contradicts": 0.0,
            "neutral": 0.2,
            "tolerance": 0.0,
        },
        label_noise={
            "rates": [{"relation": "neutral", "rate": 1.0}],
            "confusion": {
                "kind": "tabular",
                "rows": [{"oracle": "neutral", "entails": 1.0}],
            },
            "tolerance": 0.0,
        },
    )
    run = _lowered(declaration)

    assert {record.observed for record in run.labels} == {ENTAILS}
    assert sum(record.oracle is NEUTRAL for record in run.labels) == 2
    assert sum(record.oracle is ENTAILS for record in run.labels) == 8


def test_a_document_declaring_a_trivial_share_draws_exactly_that_share() -> None:
    run = _lowered(_document("synthetic-inference", relations={"trivial": 1 / 3}))
    identical = sum(pair.premise.text == pair.hypothesis.text for pair in run.pairs)

    assert run.relations.trivial == identical == 3
    assert run.relations.satisfied()


def test_trivial_pairs_and_a_positive_entailing_complexity_are_refused() -> None:
    with pytest.raises(dx.ValidationError, match="which no trivial pair does"):
        PairRelationsSpec(trivial=0.25, entailment=1)

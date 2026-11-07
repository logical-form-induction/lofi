"""Tests for the synthetic-data package surface."""

import subprocess
import sys

import lofi.data.synthetic._entailment as entailment
import lofi.data.synthetic._gold_sources as gold_sources
import lofi.data.synthetic._illtyped as illtyped
import lofi.data.synthetic._inference as inference
import lofi.data.synthetic._interpretation as interpretation
import lofi.data.synthetic._lengths as lengths
import lofi.data.synthetic._pairs as pairs
import lofi.data.synthetic._ports as ports
import lofi.data.synthetic._resemblance as resemblance
import lofi.data.synthetic._scope as scope
import lofi.data.synthetic._semantics as semantics
import lofi.data.synthetic._types as types
import lofi.data.synthetic._values as values
from lofi.data.synthetic import (
    ENTAILMENT_LABELS,
    GRAMMATICALITY_LABELS,
    SOURCE_METHOD,
    AmbiguityReport,
    AmbiguityTarget,
    AmbiguityUnreachableError,
    ConstraintSourceContext,
    Corrupted,
    Corruption,
    CorruptionDraw,
    CorruptionExhaustedError,
    CorruptionReport,
    EntailmentCertificate,
    ExhaustiveTypeInventory,
    ExhaustiveTypeSampler,
    GoldForm,
    GrammarKnobs,
    Histogram,
    IllTyped,
    IllTypedFormError,
    IllTypingReport,
    InterpretationError,
    InterpretationFunction,
    LabelNoise,
    LambdaCalculus,
    LengthDistribution,
    LengthOutOfSupportError,
    LengthReport,
    LexiconEstimate,
    LexiconReport,
    LogicalClassContext,
    LogicalClassPlugin,
    Meaning,
    NegativeBinomialLength,
    NoiseReport,
    NonDerivability,
    ObservedLabel,
    OutOfSupportPolicy,
    ReadingCount,
    Readings,
    RecursiveTypeSampler,
    RulePairSampler,
    RulePairSamplerContext,
    ScopeReport,
    ScopeTarget,
    ScopeUnreachableError,
    SemanticLexicon,
    SemanticLexiconEstimate,
    SemanticReport,
    SemanticSource,
    ShapeMeasurement,
    ShapeReport,
    ShapeTarget,
    ShapeUnreachableError,
    SourceLexicon,
    SymbolInventory,
    SyntheticCorpus,
    SyntheticDataset,
    SyntheticGenerator,
    TabularLength,
    Target,
    TypingConflict,
    TypingFailure,
    UncertifiedDerivationError,
    UniformLength,
    UninterpretableDerivationError,
    UnproductiveGrammarError,
    WeightedOperator,
    build_arity_branching,
    build_child_depth,
    estimate,
    generator,
    interpretation_function,
    logical_classes,
    readings,
    semantic_estimate,
    synthetic_inference_item,
)


def test_the_constraint_source_context_is_published() -> None:
    assert ConstraintSourceContext is gold_sources.ConstraintSourceContext


def test_synthetic_sampler_names_are_published() -> None:
    assert ExhaustiveTypeInventory is types.ExhaustiveTypeInventory
    assert ExhaustiveTypeSampler is types.ExhaustiveTypeSampler
    assert RecursiveTypeSampler is types.RecursiveTypeSampler
    assert RulePairSampler is pairs.RulePairSampler
    assert RulePairSamplerContext is pairs.RulePairSamplerContext
    assert build_arity_branching is types.build_arity_branching
    assert build_child_depth is types.build_child_depth


def test_the_inference_tokenizer_is_published() -> None:
    assert synthetic_inference_item is inference.synthetic_inference_item


def test_importing_the_package_loads_no_neural_dependency() -> None:
    source = (
        "import sys\n"
        "import lofi.data.synthetic\n"
        "print(sorted({'numpy', 'torch'} & set(sys.modules)))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", source],
        check=True,
        capture_output=True,
        encoding="utf-8",
    )

    assert result.stdout.strip() == "[]"


def test_synthetic_generator_names_are_published() -> None:
    assert SyntheticGenerator is generator.SyntheticGenerator
    assert SOURCE_METHOD is generator.SOURCE_METHOD
    assert SyntheticCorpus is values.SyntheticCorpus
    assert SyntheticDataset is values.SyntheticDataset
    assert GrammarKnobs is values.GrammarKnobs
    assert UnproductiveGrammarError is values.UnproductiveGrammarError
    assert UncertifiedDerivationError is values.UncertifiedDerivationError


def test_ambiguity_and_corruption_names_are_published() -> None:
    assert Histogram is values.Histogram
    assert Target is values.Target
    assert AmbiguityTarget is values.AmbiguityTarget
    assert AmbiguityReport is values.AmbiguityReport
    assert LexiconReport is values.LexiconReport
    assert AmbiguityUnreachableError is values.AmbiguityUnreachableError
    assert CorruptionDraw is values.CorruptionDraw
    assert Corruption is values.Corruption
    assert WeightedOperator is values.WeightedOperator
    assert Corrupted is values.Corrupted
    assert CorruptionReport is values.CorruptionReport
    assert CorruptionExhaustedError is values.CorruptionExhaustedError
    assert NonDerivability is values.NonDerivability


def test_ill_typing_names_are_published() -> None:
    assert IllTyped is illtyped.IllTyped
    assert TypingConflict is illtyped.TypingConflict
    assert TypingFailure is illtyped.TypingFailure
    assert IllTypingReport is values.IllTypingReport


def test_scope_names_are_published() -> None:
    assert ScopeTarget is values.ScopeTarget
    assert ScopeReport is values.ScopeReport
    assert ReadingCount is values.ReadingCount
    assert ScopeUnreachableError is values.ScopeUnreachableError
    assert Readings is scope.Readings
    assert readings is scope.readings


def test_label_noise_names_are_published() -> None:
    assert GRAMMATICALITY_LABELS is values.GRAMMATICALITY_LABELS
    assert LabelNoise is values.LabelNoise
    assert ObservedLabel is values.ObservedLabel
    assert NoiseReport is values.NoiseReport


def test_length_names_are_published() -> None:
    assert LengthDistribution is lengths.LengthDistribution
    assert LengthOutOfSupportError is lengths.LengthOutOfSupportError
    assert OutOfSupportPolicy is lengths.OutOfSupportPolicy
    assert UniformLength is lengths.UniformLength
    assert NegativeBinomialLength is lengths.NegativeBinomialLength
    assert TabularLength is lengths.TabularLength
    assert LengthReport is values.LengthReport


def test_derivation_shape_names_are_published() -> None:
    assert ShapeMeasurement is values.ShapeMeasurement
    assert ShapeReport is values.ShapeReport
    assert ShapeTarget is values.ShapeTarget
    assert ShapeUnreachableError is values.ShapeUnreachableError


def test_lexicon_resemblance_names_are_published() -> None:
    assert SourceLexicon is ports.SourceLexicon
    assert LexiconEstimate is resemblance.LexiconEstimate
    assert estimate is resemblance.estimate


def test_semantic_resemblance_names_are_published() -> None:
    assert SemanticSource is ports.SemanticSource
    assert SemanticLexiconEstimate is resemblance.SemanticLexiconEstimate
    assert semantic_estimate is resemblance.semantic_estimate


def test_semantic_lexicon_names_are_published() -> None:
    assert Meaning is semantics.Meaning
    assert SemanticLexicon is semantics.SemanticLexicon
    assert SemanticReport is semantics.SemanticReport
    assert LogicalClassContext is semantics.LogicalClassContext
    assert LogicalClassPlugin is semantics.LogicalClassPlugin
    assert logical_classes is semantics.logical_classes


def test_interpretation_and_entailment_names_are_published() -> None:
    assert SymbolInventory is values.SymbolInventory
    assert UninterpretableDerivationError is values.UninterpretableDerivationError
    assert IllTypedFormError is values.IllTypedFormError
    assert InterpretationFunction is interpretation.InterpretationFunction
    assert InterpretationError is interpretation.InterpretationError
    assert LambdaCalculus is interpretation.LambdaCalculus
    assert interpretation_function is interpretation.interpretation_function
    assert ENTAILMENT_LABELS is entailment.ENTAILMENT_LABELS
    assert EntailmentCertificate is entailment.EntailmentCertificate
    assert GoldForm is entailment.GoldForm

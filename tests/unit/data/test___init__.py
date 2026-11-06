from pathlib import Path

import pytest
import torch

import lofi.data._constraint_sources as constraint_sources
import lofi.data._ports as ports
import lofi.data._reachability as reachability
import lofi.data._sampling as sampling
import lofi.data._selection as selection
from lofi.data import (
    RESPONSE_MISSING,
    Alignment,
    AmbiguityRule,
    AnalysisCounts,
    Annotation,
    AnnotatorId,
    ArityBranching,
    AttachmentTree,
    Batch,
    BatchPartition,
    BucketOccupancy,
    ByteSpan,
    Categorical,
    ChildDepthDistribution,
    ConstraintPattern,
    ConstraintSource,
    DataRoot,
    DeclaredSource,
    Dependency,
    DerivationOutcome,
    DerivationOutcomeKind,
    EvalDataset,
    Example,
    ExhaustiveWeighting,
    FeaturizerComponent,
    FeaturizerResources,
    GoldDerivationSource,
    GoldLexiconSource,
    InventoriedTypeSampler,
    Item,
    ItemId,
    Join,
    LabelBlocker,
    LabelMembership,
    Licensing,
    LoadReport,
    LogicalFormSource,
    Mask,
    PairDistribution,
    Pairing,
    PairSampler,
    PairSampleReport,
    PathDeclarationError,
    PinnedConstraintSource,
    PowerChildDepth,
    Provenance,
    ReachabilityReport,
    ReachableDerivation,
    RuleBlocker,
    Scalar,
    Scale,
    ScopeKind,
    ScoreQuantile,
    SelectionPolicy,
    SizeExhaustiveWeighting,
    StepRecovery,
    Supervision,
    SupervisionArm,
    Token,
    TokenSpan,
    TypeDistribution,
    TypeInventory,
    TypeSampler,
    TypeSampleReport,
    UniformChildDepth,
    UniformExhaustiveWeighting,
    Unlabeled,
    UnrecoverableStep,
    Weighted,
    attachment_tree,
    deps,
    reachability_report,
)
from lofi.data.deps import DependencyProjector, HeadProjection, project_dependencies

SENTENCE = (Token(0, "the"), Token(1, "child"), Token(2, "slept"))


def test_reachability_names_are_published_at_the_package_boundary() -> None:
    assert DerivationOutcome is reachability.DerivationOutcome
    assert DerivationOutcomeKind is reachability.DerivationOutcomeKind
    assert LabelBlocker is reachability.LabelBlocker
    assert ReachabilityReport is reachability.ReachabilityReport
    assert ReachableDerivation is reachability.ReachableDerivation
    assert RuleBlocker is reachability.RuleBlocker
    assert UnrecoverableStep is reachability.UnrecoverableStep
    assert reachability_report is reachability.reachability_report
    assert LabelMembership is ports.LabelMembership
    assert StepRecovery is ports.StepRecovery


def test_sampling_names_are_published_at_the_package_boundary() -> None:
    assert TypeSampler is ports.TypeSampler
    assert InventoriedTypeSampler is ports.InventoriedTypeSampler
    assert TypeInventory is ports.TypeInventory
    assert PairSampler is ports.PairSampler
    assert TypeDistribution is sampling.TypeDistribution
    assert PairDistribution is sampling.PairDistribution
    assert TypeSampleReport is sampling.TypeSampleReport
    assert PairSampleReport is sampling.PairSampleReport
    assert ArityBranching is sampling.ArityBranching
    assert ChildDepthDistribution is sampling.ChildDepthDistribution
    assert ExhaustiveWeighting is sampling.ExhaustiveWeighting
    assert PowerChildDepth is sampling.PowerChildDepth
    assert UniformChildDepth is sampling.UniformChildDepth
    assert SizeExhaustiveWeighting is sampling.SizeExhaustiveWeighting
    assert UniformExhaustiveWeighting is sampling.UniformExhaustiveWeighting


def test_featurizer_ports_are_published_at_the_package_boundary() -> None:
    assert FeaturizerComponent is ports.FeaturizerComponent
    assert FeaturizerResources is ports.FeaturizerResources


def test_constraint_source_names_are_published_at_the_package_boundary() -> None:
    assert ConstraintSource is ports.ConstraintSource
    assert AmbiguityRule is ports.AmbiguityRule
    assert ScopeKind is selection.ScopeKind
    assert SelectionPolicy is selection.SelectionPolicy
    assert GoldDerivationSource is constraint_sources.GoldDerivationSource
    assert GoldLexiconSource is constraint_sources.GoldLexiconSource
    assert DeclaredSource is constraint_sources.DeclaredSource
    assert LogicalFormSource is constraint_sources.LogicalFormSource
    assert ScoreQuantile is constraint_sources.ScoreQuantile
    assert ConstraintPattern is constraint_sources.ConstraintPattern
    assert PinnedConstraintSource is constraint_sources.PinnedConstraintSource


def test_dependency_projection_names_are_published_at_the_package_boundary() -> None:
    assert DependencyProjector is deps.DependencyProjector
    assert HeadProjection is deps.HeadProjection
    assert project_dependencies is deps.project_dependencies


def test_attachment_values_are_published_at_the_package_boundary() -> None:
    tree = AttachmentTree(("a",), (0,))
    projection = HeadProjection((0,), (Dependency(0, 1, "root"),))

    assert attachment_tree(projection, (Token(0, "a"),)) == tree


def dataset() -> EvalDataset[int]:
    return EvalDataset(
        [Item(ItemId("s/1"), SENTENCE), Item(ItemId("s/2"), SENTENCE)],
        [
            Annotation(ItemId("s/1"), 7, "annotator-a"),
            Annotation(ItemId("s/2"), 3, "annotator-b"),
        ],
    )


def build_item() -> Item:
    text = "el niño duerme"
    offsets = ((0, 2), (3, 8), (9, 15))
    tokens = tuple(
        Token(index, ByteSpan(start, end).substring(text), ByteSpan(start, end))
        for index, (start, end) in enumerate(offsets)
    )

    return Item(
        ItemId("at://example.invalid/expression/1"),
        tokens,
        text=text,
        languages=("es",),
        split="train",
        provenance=Provenance("manual-native", Licensing("CC-BY-4.0")),
    )


def test_the_published_surface_joins_gold_to_predictions_on_the_item_id() -> None:
    join = dataset().join({ItemId("s/1"): 6, ItemId("s/9"): 1})

    assert isinstance(join, Join)
    assert isinstance(join.pairs[0], Pairing)
    assert [pair.item_id for pair in join.pairs] == [ItemId("s/1")]
    assert join.pairs[0].gold.value == 7
    assert join.pairs[0].prediction == 6
    assert join.unpredicted == (ItemId("s/2"),)
    assert join.unmatched == (ItemId("s/9"),)


def test_the_published_surface_reads_one_items_gold_back_by_id() -> None:
    annotated = dataset()

    assert len(annotated) == 2
    assert annotated.ids == (ItemId("s/1"), ItemId("s/2"))
    assert annotated.item(ItemId("s/2")).tokens == SENTENCE
    assert annotated.annotation(ItemId("s/2")) == Annotation(
        ItemId("s/2"), 3, "annotator-b"
    )


def test_a_configured_corpus_path_resolves_through_the_published_root() -> None:
    root = DataRoot(Path("/srv/corpora").absolute())

    assert (
        root.resolve("ccgbank/wsj", key="dataset.path")
        == root.directory / "ccgbank/wsj"
    )

    with pytest.raises(PathDeclarationError):
        root.resolve("/etc/passwd", key="dataset.path")


def test_the_published_values_build_a_usable_item() -> None:
    item = build_item()

    assert [token.surface for token in item.tokens] == ["el", "niño", "duerme"]
    assert item.tokens[1].index == 1
    assert item.languages == ("es",)
    assert item.split == "train"


def test_a_span_over_the_published_tokens_names_the_words_it_covers() -> None:
    item = build_item()
    subject = TokenSpan(0, 2)
    covered = item.tokens[subject.start : subject.end]

    assert subject.length == len(covered)
    assert [token.surface for token in covered] == ["el", "niño"]
    assert subject.concat(TokenSpan(2, 3)).length == len(item.tokens)


def test_the_published_provenance_includes_its_license() -> None:
    provenance = build_item().provenance

    assert provenance is not None
    assert provenance.licensing.expression == "CC-BY-4.0"
    assert provenance.source_method == "manual-native"


def test_the_published_surface_pairs_an_item_with_what_is_said_about_it() -> None:
    item = build_item()
    rated: Example[Supervision[str]] = Example(
        item,
        Scalar(6.0, Scale("ordinal", 1.0, 7.0), AnnotatorId("annotator-1")),
    )
    unlabeled: Example[Supervision[str]] = Example(item, Unlabeled())

    assert rated.item == unlabeled.item
    assert rated != unlabeled
    assert len({rated, unlabeled}) == 2


def test_the_published_surface_keeps_a_score_beside_a_label() -> None:
    scored = frozenset({Weighted("NP", -0.2), Weighted("N", -1.6)})

    assert {candidate.value for candidate in scored} == {"NP", "N"}
    assert Weighted("NP", -0.2) != Weighted("NP", -1.9)


def test_the_published_surface_names_the_inventory_a_label_came_from() -> None:
    choice = Categorical("entailed", frozenset({"entailed", "contradicted"}))

    assert choice.label in choice.label_set
    assert choice.annotator is None


def test_the_published_surface_names_the_missing_response_feature() -> None:
    assert RESPONSE_MISSING == "response_missing"


def test_the_published_report_states_a_rate_as_a_quotient_of_counts() -> None:
    report = LoadReport(20, 18, {"invalid record": 2}, AnalysisCounts(10, 7, 6))

    assert report.records_seen == report.emitted + report.skipped
    assert report.reachability == 7 / 10
    assert report.normal_form_conformance == 6 / 10


def test_the_published_batch_includes_its_mask_and_alignment() -> None:
    rows = torch.zeros(1, 3, 2)
    batch = Batch(
        rows,
        (3,),
        mask=Mask(torch.ones(1, 3, dtype=torch.bool)),
        alignments=(Alignment(((0, 1), (1, 2), (2, 3))),),
        item_ids=(ItemId("s/1"),),
    )

    assert len(batch) == 1
    assert batch.mask is not None
    assert batch.mask.width == 3
    assert batch.alignments[0].is_total


def test_the_published_surface_includes_batch_partitions() -> None:
    partition = BatchPartition((3, 1, 2), 2, ((0, 2), (1,)))

    assert sorted(index for batch in partition for index in batch) == [0, 1, 2]
    assert isinstance(partition.occupancy, BucketOccupancy)


def test_the_published_surface_names_a_supervision_arm_by_its_type() -> None:
    arms: frozenset[SupervisionArm] = frozenset({Unlabeled, Scalar})

    assert Unlabeled in arms
    assert type(Scalar(6.0, Scale("ordinal", 1.0, 7.0))) in arms
    assert Categorical not in arms

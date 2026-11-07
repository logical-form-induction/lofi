from typing import TYPE_CHECKING, Final

import pytest
from tests.conformance.evaluator import (
    HeldOutEvaluatorSuite,
    HeldOutSubject,
    packaged_held_out_pass,
)

from lofi._types import ItemId, ReadoutId
from lofi.eval import ConventionVector
from lofi.eval._lexical_type_evaluator import (
    PLUGIN,
    LexicalAnalysis,
    LexicalTypeEvaluator,
    LexicalTypeF1,
    fit_lexical_alignment,
    lexical_relabeling_base,
    score_lexical_alignment,
    stable_lexical_alignments,
)
from lofi.nn.loss._values import ObjectiveOutputs
from lofi.registry import plugin_for
from lofi.spec.evaluators.alignment import AlignmentKind, AlignmentSettings
from lofi.spec.evaluators.lexical_types import LexicalTypeEvaluatorSpec

if TYPE_CHECKING:
    from lofi.spec.grammar import GrammarAssembly
    from lofi.terms import Term

CONVENTIONS: Final = ConventionVector(
    punctuation="keep",
    unary_chains="keep",
    trivial_spans="keep",
    minimum_length=0,
    aggregation="sentence",
    coverage="zero",
    self_arcs="allow",
    gold_derivation="canonical",
)
STRIPPED: Final = ConventionVector(
    punctuation="strip",
    unary_chains="keep",
    trivial_spans="keep",
    minimum_length=3,
    aggregation="sentence",
    coverage="zero",
    self_arcs="allow",
    gold_derivation="canonical",
)
ALIGNMENT: Final = AlignmentSettings(
    kind="identity",
    tie_break=None,
    fit_on="dev",
    score_on="test",
    floor_replicates=50,
    floor_level=0.95,
    stability_replicates=50,
    stability_level=0.95,
)


class TestTheLexicalTypeEvaluatorObeysTheLaws(HeldOutEvaluatorSuite):
    @pytest.fixture
    def held_out_subject(
        self,
    ) -> HeldOutSubject[tuple[Term, ...], LexicalAnalysis]:
        grammar = packaged_held_out_pass().grammar
        held_out = packaged_held_out_pass()
        built = LexicalTypeEvaluator(
            grammar,
            readout="category-terms",
            conventions=CONVENTIONS,
            alignment=ALIGNMENT,
        )
        predictions = {
            item_id: prediction
            for outputs in held_out.published
            for item_id, prediction in built.predict(outputs).items()
        }
        gold = {
            item.item_id: built.annotate(item, derivation)
            for item, derivation in held_out.gold
        }
        built.fit_alignment(predictions, gold)
        return HeldOutSubject(built, held_out)


def category(grammar: GrammarAssembly, text: str) -> Term:
    return grammar.syntax.parse(grammar.store, text)


def metric(grammar: GrammarAssembly, conventions: ConventionVector) -> LexicalTypeF1:
    categories = tuple(
        entry.category
        for form in grammar.lexicon.forms
        for entry in grammar.lexicon.entries(form)
    )
    words = tuple(f"w{position}" for position in range(len(categories)))
    item_id = ItemId("development")
    fitted = fit_lexical_alignment(
        {item_id: categories},
        {item_id: LexicalAnalysis(words, categories, frozenset())},
        grammar,
        conventions,
        ALIGNMENT,
    )
    assert fitted.constructors is not None
    return LexicalTypeF1(
        conventions,
        grammar.store,
        primitive_alignment=fitted.primitives.alignment,
        constructor_alignment=fitted.constructors.alignment,
    )


def test_each_token_takes_the_highest_weight_complete_decoded_type() -> None:
    held_out = packaged_held_out_pass()
    grammar = held_out.grammar
    evaluator = LexicalTypeEvaluator(
        grammar,
        readout="category-terms",
        conventions=CONVENTIONS,
        alignment=ALIGNMENT,
    )
    seen = 0

    for outputs in held_out.published:
        decoded = outputs.terms[ReadoutId("category-terms")]
        predicted = evaluator.predict(outputs)

        for row, (item_id, words) in enumerate(
            zip(outputs.batch.item_ids, outputs.batch.words, strict=True)
        ):
            for position, _form in enumerate(words):
                cell = outputs.spans.span_id(position, position + 1)
                frontier = decoded.at(row, cell)
                seen += 1

                assert (
                    predicted[item_id][position]
                    is frontier.terms[int(frontier.log_weights.argmax())]
                )

    assert seen


def test_decoded_types_are_scored_after_alignment() -> None:
    held_out = packaged_held_out_pass()
    evaluator = LexicalTypeEvaluator(
        held_out.grammar,
        readout="category-terms",
        conventions=CONVENTIONS,
        alignment=ALIGNMENT,
    )
    predictions: dict[ItemId, tuple[Term, ...]] = {}

    for outputs in held_out.published:
        predictions.update(evaluator.predict(outputs))

    analyses = {
        item.item_id: evaluator.annotate(item, derivation)
        for item, derivation in held_out.gold
    }
    evaluator.fit_alignment(predictions, analyses)
    for item, _derivation in held_out.gold:
        evaluator.metrics[0].update(
            predictions[item.item_id],
            analyses[item.item_id],
            key=item.item_id,
        )

    result = evaluator.metrics[0].compute()

    assert result.value is not None
    assert 0.0 <= result.value <= 1.0
    assert 0.0 <= result.components["token_accuracy"] <= 1.0
    assert result.n == len(held_out.gold)


def test_the_gold_analysis_reads_the_leaf_over_each_token() -> None:
    held_out = packaged_held_out_pass()
    grammar = held_out.grammar
    evaluator = LexicalTypeEvaluator(
        grammar,
        readout="category-terms",
        conventions=CONVENTIONS,
        alignment=ALIGNMENT,
    )

    for item, derivation in held_out.gold:
        analysis = evaluator.annotate(item, derivation)
        leaves = sorted(derivation.leaves(), key=lambda leaf: leaf.span.start)

        assert analysis.words == tuple(token.surface for token in item.tokens)
        assert analysis.categories == tuple(
            category(grammar, leaf.label) for leaf in leaves
        )
        assert analysis.punctuation == frozenset()


def test_word_level_multiset_f1_averages_over_word_types() -> None:
    grammar = packaged_held_out_pass().grammar
    noun = category(grammar, "NP")
    verb = category(grammar, "S\\NP")
    scored = metric(grammar, CONVENTIONS)
    gold = LexicalAnalysis(("a", "b", "a"), (noun, verb, noun), frozenset())

    scored.update((noun, verb, verb), gold, key=ItemId("item"))
    result = scored.compute()

    assert result.value == pytest.approx((0.5 + 1.0) / 2)
    assert result.components["token_accuracy"] == pytest.approx(2 / 3)
    assert result.components["words"] == 2.0
    assert scored.per_item()[ItemId("item")].value == pytest.approx(0.75)

    scored.reset()

    assert scored.compute().value is None
    assert scored.compute().n == 0


def test_declared_aggregation_and_coverage_change_applicable_scores() -> None:
    grammar = packaged_held_out_pass().grammar
    noun = category(grammar, "NP")
    verb = category(grammar, "S\\NP")
    corpus = ConventionVector(
        punctuation="keep",
        unary_chains="keep",
        trivial_spans="keep",
        minimum_length=0,
        aggregation="corpus",
        coverage="zero",
        self_arcs="allow",
        gold_derivation="canonical",
    )
    ignored = ConventionVector(
        punctuation="keep",
        unary_chains="keep",
        trivial_spans="keep",
        minimum_length=0,
        aggregation="sentence",
        coverage="ignore",
        self_arcs="allow",
        gold_derivation="canonical",
    )
    sentence_metric = metric(grammar, CONVENTIONS)
    corpus_metric = metric(grammar, corpus)

    for scored in (sentence_metric, corpus_metric):
        scored.update(
            (verb, verb),
            LexicalAnalysis(("a", "a"), (noun, noun), frozenset()),
            key=ItemId("first"),
        )
        scored.update(
            (noun, noun),
            LexicalAnalysis(("a", "b"), (noun, noun), frozenset()),
            key=ItemId("second"),
        )

    ignored_metric = metric(grammar, ignored)
    ignored_metric.update(
        (noun, verb),
        LexicalAnalysis(("a",), (noun,), frozenset()),
        key=ItemId("unequal"),
    )

    assert sentence_metric.compute().value == pytest.approx(0.5)
    assert corpus_metric.compute().value == pytest.approx(2 / 3)
    assert ignored_metric.compute().components["token_accuracy"] == 1.0
    assert ignored_metric.compute().components["tokens"] == 1.0


def test_zero_coverage_scores_a_constructor_absent_from_development_as_zero() -> None:
    grammar = packaged_held_out_pass().grammar
    noun = category(grammar, "NP")
    sentence = category(grammar, "S")
    backward = category(grammar, "S\\NP")
    forward = category(grammar, "S/NP")
    development = ItemId("development")
    settings = AlignmentSettings(
        kind="many-to-one",
        tie_break="least-gold-label",
        fit_on="dev",
        score_on="test",
        floor_replicates=50,
        floor_level=0.95,
        stability_replicates=50,
        stability_level=0.95,
    )
    fitted = fit_lexical_alignment(
        {development: (noun, sentence, backward)},
        {
            development: LexicalAnalysis(
                ("noun", "sentence", "verb"),
                (noun, sentence, backward),
                frozenset(),
            )
        },
        grammar,
        CONVENTIONS,
        settings,
    )
    constructors = fitted.constructors
    assert constructors is not None
    assert "/" not in constructors.alignment.mapping
    scored = LexicalTypeF1(
        CONVENTIONS,
        grammar.store,
        primitive_alignment=fitted.primitives.alignment,
        constructor_alignment=constructors.alignment,
    )
    scored.update(
        (forward,),
        LexicalAnalysis(("verb",), (backward,), frozenset()),
        key=ItemId("test"),
    )

    result = scored.compute()

    assert result.value == 0.0
    assert result.components["token_accuracy"] == 0.0


def test_zero_coverage_scores_a_primitive_absent_from_development_as_zero() -> None:
    grammar = packaged_held_out_pass().grammar
    noun = category(grammar, "NP")
    sentence = category(grammar, "S")
    development = ItemId("development")
    settings = AlignmentSettings(
        kind="many-to-one",
        tie_break="least-gold-label",
        fit_on="dev",
        score_on="test",
        floor_replicates=50,
        floor_level=0.95,
        stability_replicates=50,
        stability_level=0.95,
    )
    fitted = fit_lexical_alignment(
        {development: (noun,)},
        {
            development: LexicalAnalysis(
                ("noun",),
                (noun,),
                frozenset(),
            )
        },
        grammar,
        CONVENTIONS,
        settings,
    )
    assert "S" not in fitted.primitives.alignment.mapping
    scored = LexicalTypeF1(
        CONVENTIONS,
        grammar.store,
        primitive_alignment=fitted.primitives.alignment,
        constructor_alignment=None,
    )
    scored.update(
        (sentence,),
        LexicalAnalysis(("sentence",), (sentence,), frozenset()),
        key=ItemId("test"),
    )

    result = scored.compute()

    assert result.value == 0.0
    assert result.components["token_accuracy"] == 0.0


def test_stripped_punctuation_and_short_items_are_not_scored() -> None:
    grammar = packaged_held_out_pass().grammar
    noun = category(grammar, "NP")
    verb = category(grammar, "S\\NP")
    scored = metric(grammar, STRIPPED)

    scored.update(
        (noun, verb, verb, noun),
        LexicalAnalysis(("a", ",", "b", "a"), (noun, noun, verb, noun), frozenset({1})),
        key=ItemId("kept"),
    )
    scored.update(
        (verb, verb),
        LexicalAnalysis(("a", "b"), (noun, verb), frozenset()),
        key=ItemId("short"),
    )
    result = scored.compute()

    assert result.value == pytest.approx(1.0)
    assert result.n == 1
    assert result.components["tokens"] == 3.0
    assert result.components["dropped"] == 1.0
    assert set(scored.per_item()) == {ItemId("kept")}


def test_unequal_token_counts_are_scored_and_history_stays_consistent() -> None:
    grammar = packaged_held_out_pass().grammar
    noun = category(grammar, "NP")
    verb = category(grammar, "S\\NP")
    scored = metric(grammar, CONVENTIONS)
    gold = LexicalAnalysis(("a",), (noun,), frozenset())

    scored.update((noun, noun), gold, key=ItemId("item"))
    scored.update((noun, noun), gold, key=ItemId("item"))

    result = scored.compute()

    assert result.value == 1.0
    assert result.components["token_accuracy"] == 0.5
    assert result.components["tokens"] == 2.0

    with pytest.raises(ValueError, match="already scored"):
        scored.update((verb,), gold, key=ItemId("item"))


def test_a_readout_the_pass_did_not_publish_is_refused() -> None:
    held_out = packaged_held_out_pass()
    evaluator = LexicalTypeEvaluator(
        held_out.grammar,
        readout="absent",
        conventions=CONVENTIONS,
        alignment=ALIGNMENT,
    )
    published = held_out.published[0]
    bare = ObjectiveOutputs(
        spans=published.spans, chart=published.chart, batch=published.batch
    )

    with pytest.raises(ValueError, match="'absent'"):
        evaluator.predict(published)

    with pytest.raises(ValueError, match="decoded \\[\\]"):
        evaluator.predict(bare)


def test_the_plugin_reads_the_declared_readout_and_conventions() -> None:
    declared = ConventionVector(
        punctuation="strip",
        unary_chains="collapse",
        trivial_spans="drop",
        minimum_length=3,
        aggregation="corpus",
        coverage="ignore",
        self_arcs="forbid",
        gold_derivation="oracle",
    )
    declaration = LexicalTypeEvaluatorSpec.model_validate(
        {
            "kind": "lexical-types",
            "interval": {"replicates": 50, "level": 0.95},
            "readout": "lexicon-readout",
            "conventions": dict(declared.factors),
            "alignment": {
                "kind": "identity",
                "fit_on": "dev",
                "score_on": "test",
                "random_floor": {"replicates": 50, "level": 0.95},
                "stability": {"replicates": 50, "level": 0.95},
            },
        }
    )
    built = PLUGIN.construct(declaration, packaged_held_out_pass().context)
    grammar = packaged_held_out_pass().grammar
    categories = tuple(
        entry.category
        for form in grammar.lexicon.forms
        for entry in grammar.lexicon.entries(form)
    )
    words = tuple(f"w{position}" for position in range(len(categories)))
    item_id = ItemId("development")
    built.fit_alignment(
        {item_id: categories},
        {item_id: LexicalAnalysis(words, categories, frozenset())},
    )

    assert built.alignment == ALIGNMENT
    assert built.conventions.factors["alignment"] == "identity"
    assert built.metrics[0].conventions == built.conventions
    assert built.metrics[0].name == "lexical-type-f1"
    assert declaration.metrics()[:1] == tuple(metric.name for metric in built.metrics)
    assert "readout='lexicon-readout'" in repr(built)


@pytest.mark.parametrize("kind", ["many-to-one", "one-to-one"])
def test_real_development_predictions_fit_and_freeze_a_planted_permutation(
    kind: AlignmentKind,
) -> None:
    held_out = packaged_held_out_pass()
    grammar = held_out.grammar
    noun = category(grammar, "NP")
    sentence = category(grammar, "S")
    slash_noun = category(grammar, "S\\NP")
    slash_sentence = category(grammar, "NP\\S")
    settings = AlignmentSettings(
        kind=kind,
        tie_break="least-gold-label",
        fit_on="dev",
        score_on="test",
        floor_replicates=50,
        floor_level=0.95,
        stability_replicates=50,
        stability_level=0.95,
    )
    dev_id = ItemId("development")
    test_id = ItemId("test")
    development_predictions = {dev_id: (noun, sentence, slash_noun, slash_noun)}
    development_gold = {
        dev_id: LexicalAnalysis(
            ("a", "b", "c", "d"),
            (sentence, noun, slash_sentence, slash_sentence),
            frozenset(),
        )
    }
    test_predictions = {test_id: (noun, sentence, slash_noun, slash_noun)}
    test_gold = {
        test_id: LexicalAnalysis(
            ("a", "b", "c", "d"),
            (sentence, noun, slash_sentence, slash_sentence),
            frozenset(),
        )
    }

    fitted = fit_lexical_alignment(
        development_predictions,
        development_gold,
        grammar,
        CONVENTIONS,
        settings,
    )
    scored = score_lexical_alignment(
        fitted,
        test_predictions,
        test_gold,
        grammar,
        CONVENTIONS,
        seed=7,
    )

    assert dict(fitted.primitives.alignment.mapping) == {"NP": "S", "S": "NP"}
    assert fitted.constructors is not None
    assert dict(fitted.constructors.alignment.mapping) == {"\\": "\\"}
    assert scored.primitives.observed.value == 1.0
    primitive_value = scored.primitives.observed.value
    assert primitive_value is not None
    assert primitive_value > scored.primitives.floor.value
    assert "primitive-random-relabeling-floor" in scored.rows
    assert "constructor-random-relabeling-floor" in scored.rows
    prediction = lexical_relabeling_base(fitted, ambiguity_budget=1)
    assert prediction.group_size == 2
    assert prediction.base_size == 1

    stability = stable_lexical_alignments((fitted, fitted, fitted, fitted), seed=11)
    assert stability.primitives.agreement.value == 1.0
    assert stability.constructors is not None
    assert stability.constructors.agreement.value == 1.0


def test_the_evaluator_family_lists_this_plugin_under_its_kind() -> None:
    declaration = LexicalTypeEvaluatorSpec.model_validate(
        {
            "kind": "lexical-types",
            "interval": {"replicates": 50, "level": 0.95},
            "readout": "types",
            "conventions": dict(CONVENTIONS.factors),
            "alignment": {
                "kind": "identity",
                "fit_on": "dev",
                "score_on": "test",
                "random_floor": {"replicates": 50, "level": 0.95},
                "stability": {"replicates": 50, "level": 0.95},
            },
        }
    )

    assert plugin_for(declaration) is PLUGIN

"""Tests for the readers that lower collected releases."""

from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from lofi.data import GoldGraph, Relational, RelationalScalar
from lofi.data._collected_reading import (
    lower_chaosnli,
    lower_fracas,
    lower_mnli,
    lower_pmb_ccg,
    lower_pmb_drs,
    lower_unli,
)
from lofi.data.collected import (
    CHAOS_NLI_PLUGIN,
    FRACAS_PLUGIN,
    MNLI_PLUGIN,
    PMB_CCG_PLUGIN,
    PMB_DRS_PLUGIN,
    UNLI_PLUGIN,
)
from lofi.deduce import ParsePolicy, Reachability, Retain
from lofi.spec.datasets import (
    ChaosNliDatasetSpec,
    CorpusReleaseSpec,
    CorpusSourceSpec,
    FracasDatasetSpec,
    MnliDatasetSpec,
    PmbCcgDatasetSpec,
    PmbDrsDatasetSpec,
    UnliDatasetSpec,
    ccgbank_grammar,
    pmb_grammar,
)
from lofi.spec.language import LanguageDecl, registered_language
from lofi.spec.tokenizers import RegexTokenizerSpec

if TYPE_CHECKING:
    from lofi.data.datasets import CheckedDataset, DatasetPlugin, LoweredDataset
    from lofi.spec import DatasetSpec
    from lofi.spec.grammar import GrammarDecl

FIXTURES: Final = Path(__file__).parent / "corpora"
POLICY: Final = ParsePolicy(Retain.FOREST, Reachability.ALL)
EPRINT: Final = "at://did:plc:test/pub.layers.eprint.eprint/collected"


def collected_language() -> LanguageDecl:
    """Return a small language accepted by the CCGbank test grammar."""
    return LanguageDecl.model_validate(
        {
            "name": "collected-test",
            "atomic_types": [{"name": "e"}, {"name": "t"}],
            "constructors": [
                {
                    "name": "->",
                    "kind": ["Ty", "Ty", "Ty"],
                    "notation": {
                        "style": "infix",
                        "spelling": "->",
                        "precedence": 1,
                        "associativity": "right",
                    },
                }
            ],
            "constants": [],
            "binders": {"style": "curry"},
            "logic": {"equality": False},
        }
    )


def _checked[C: CheckedDataset](
    plugin: DatasetPlugin[DatasetSpec, C, LoweredDataset], declaration: DatasetSpec, /
) -> C:
    """Return the checked value the plugin's reader lowers."""
    return cast("C", plugin.check(declaration))


def nli_release() -> CorpusReleaseSpec:
    """Return metadata for the committed NLI fixtures."""
    return CorpusReleaseSpec(
        source_id="synthetic-nli",
        release_id="v1",
        split="test",
        language="en",
        source_method="manual",
        spdx="CC0-1.0",
        eprint_refs=(EPRINT,),
    )


def _pmb_release() -> CorpusReleaseSpec:
    """Return metadata for the committed PMB fixtures."""
    return CorpusReleaseSpec(
        source_id="synthetic-pmb",
        release_id="v1",
        split="test",
        language="en",
        source_method="manual",
        spdx="CC0-1.0",
        eprint_refs=(EPRINT,),
    )


def _fracas_release() -> CorpusReleaseSpec:
    """Return metadata for the committed FraCaS fixture."""
    return CorpusReleaseSpec(
        source_id="synthetic-fracas",
        release_id="v1",
        split="test",
        language="en",
        source_method="manual",
        spdx="CC0-1.0",
        eprint_refs=(EPRINT,),
    )


def _pmb_grammar() -> GrammarDecl:
    """Return the test grammar with PMB's lowercase category atoms declared."""
    grammar = ccgbank_grammar()
    signature = grammar.signature
    return grammar.with_(
        signature=signature.with_(
            primitives=(*signature.primitives, "np", "n"),
        )
    )


def test_pmb_ccg_keeps_source_categories_and_names_no_rule_for_a_foreign_step(
    tmp_path: Path,
) -> None:
    """Keep PMB categories and record no rule for a step no compiled rule licenses."""
    source = tmp_path / "sample.ccg"
    source.write_text(
        ":- op(601, xfx, (/)).\n"
        ":- op(601, xfx, (\\)).\n"
        "ccg(sample,rp(np,t(np,'café',[lemma:café,from:0,to:4,pos:'N',"
        "sem:'CON',wordnet:'cafe.n.01',verbnet:[]]))).\n",
        encoding="utf-8",
        newline="\n",
    )
    declaration = PmbCcgDatasetSpec(
        sources=(CorpusSourceSpec(source=str(source), release=_pmb_release()),),
        grammar=_pmb_grammar(),
        language=collected_language(),
    )
    dataset = lower_pmb_ccg(_checked(PMB_CCG_PLUGIN, declaration), POLICY)
    example = dataset.corpus.examples[0]
    derivation = example.supervision.derivation

    assert tuple(derivation.node(node).label for node in derivation.preorder) == (
        "np",
        "np",
    )
    assert derivation.node(derivation.root).rule is None
    assert example.item.languages == ("en",)
    assert example.item.split == "test"
    assert dataset.report.reachability_report is not None
    assert dataset.report.analyses is not None
    assert dataset.report.analyses.checked == 1


def test_packaged_pmb_grammar_reaches_the_collected_corpus() -> None:
    """Measure exact corpus reachability under the packaged English grammar."""
    declaration = PmbCcgDatasetSpec(
        sources=(
            CorpusSourceSpec(
                source=str(FIXTURES / "pmb" / "fixtures" / "pawnshop.tags"),
                release=_pmb_release(),
            ),
        ),
        grammar=pmb_grammar("en"),
        language=registered_language("propositional"),
    )
    dataset = lower_pmb_ccg(_checked(PMB_CCG_PLUGIN, declaration), POLICY)
    report = dataset.report.reachability_report

    assert report is not None
    assert report.checked_derivations == 1
    assert report.reachable_derivations == 1
    assert report.derivation_reachability == 1.0
    assert report.step_reachability == 1.0
    rules = {rule.name for rule in dataset.source_grammar.grammar.rules}
    steps = tuple(
        node
        for example in dataset.corpus.examples
        for node in example.supervision.derivation.nodes.values()
        if not node.is_lexical
    )
    assert steps
    assert all(node.rule in rules for node in steps)


def test_pmb_drs_lowers_to_the_gold_graph_carrier() -> None:
    """Expose a PMB clausal DRS through the one graph supervision arm."""
    declaration = PmbDrsDatasetSpec(
        sources=(
            CorpusSourceSpec(
                source=str(FIXTURES / "pmb" / "fixtures" / "basic.clf"),
                release=_pmb_release(),
            ),
        ),
        grammar=ccgbank_grammar(),
        language=collected_language(),
    )
    dataset = lower_pmb_drs(_checked(PMB_DRS_PLUGIN, declaration), POLICY)

    assert isinstance(dataset.corpus.observed("test")[0].supervision, GoldGraph)


def test_chaosnli_majority_reduces_one_hundred_votes_to_one_relation() -> None:
    """Aggregate repeated source responses at the declared dataset boundary."""
    declaration = ChaosNliDatasetSpec(
        sources=(
            CorpusSourceSpec(
                source=str(FIXTURES / "nli" / "fixtures" / "chaosnli.jsonl"),
                release=nli_release(),
            ),
        ),
        grammar=ccgbank_grammar(),
        language=collected_language(),
    )
    dataset = lower_chaosnli(_checked(CHAOS_NLI_PLUGIN, declaration), POLICY)
    observed = dataset.corpus.observed("test")

    assert len(observed) == 1
    assert isinstance(observed[0].supervision, Relational)
    assert observed[0].supervision.relation == "entails"


def test_mnli_and_unli_use_their_exact_supervision_arms() -> None:
    """Retain categorical and graded inference releases without coercion."""
    mnli = MnliDatasetSpec(
        sources=(
            CorpusSourceSpec(
                source=str(FIXTURES / "nli" / "fixtures" / "mnli.jsonl"),
                release=nli_release(),
            ),
        ),
        grammar=ccgbank_grammar(),
        language=collected_language(),
    )
    unli = UnliDatasetSpec(
        sources=(
            CorpusSourceSpec(
                source=str(FIXTURES / "nli" / "fixtures" / "unli.tsv"),
                release=nli_release(),
            ),
        ),
        grammar=ccgbank_grammar(),
        language=collected_language(),
    )

    mnli_data = lower_mnli(_checked(MNLI_PLUGIN, mnli), POLICY)
    unli_data = lower_unli(_checked(UNLI_PLUGIN, unli), POLICY)

    assert all(
        isinstance(example.supervision, Relational)
        for example in mnli_data.corpus.observed("test")
    )
    unli_responses = tuple(
        example.supervision for example in unli_data.corpus.observed("test")
    )
    assert sum(isinstance(response, Relational) for response in unli_responses) == 2
    assert (
        sum(isinstance(response, RelationalScalar) for response in unli_responses) == 2
    )


def test_fracas_joins_ordered_premises_into_binary_pairs() -> None:
    """Present every usable FraCaS problem to the pair featurizer."""
    declaration = FracasDatasetSpec(
        sources=(
            CorpusSourceSpec(
                source=str(FIXTURES / "fracas" / "fixtures" / "sample.xml"),
                release=_fracas_release(),
            ),
        ),
        grammar=ccgbank_grammar(),
        language=collected_language(),
        tokenizer=RegexTokenizerSpec(pattern=r"\w+|[^\w\s]"),
    )
    dataset = lower_fracas(_checked(FRACAS_PLUGIN, declaration), POLICY)
    observed = dataset.corpus.observed("test")

    assert len(observed) == 4
    assert dict(dataset.corpus.report.skips_by_reason) == {
        "a FraCaS problem without a pair": 1
    }
    assert any(
        example.item.text
        == "No delegate finished the survey on time. Smith is a delegate. "
        "Smith finished the survey."
        for example in observed
    )

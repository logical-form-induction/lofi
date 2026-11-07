"""Tests for collected-corpus dataset declarations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

import didactic.api as dx
import pytest

from lofi.spec import (
    DatasetSection,
    RunContext,
    SpecLoader,
    TrainingBatchKind,
    register_resolvers,
    run_context,
)
from lofi.spec._interpolation import lower_package_resource
from lofi.spec.datasets import (
    ChaosNliDatasetSpec,
    CorpusReleaseSpec,
    CorpusSourceSpec,
    FracasDatasetSpec,
    MnliDatasetSpec,
    PmbCcgDatasetSpec,
    PmbDrsDatasetSpec,
    SnliDatasetSpec,
    UnliDatasetSpec,
    ccgbank_grammar,
    pmb_grammar,
)
from lofi.spec.gold import MajorityGoldSpec
from lofi.spec.language import LanguageDecl
from lofi.spec.tokenizers import RegexTokenizerSpec

if TYPE_CHECKING:
    from lofi._types import JsonValue
    from lofi.spec.grammar import GrammarDecl

_EPRINT: Final = "at://did:plc:test/pub.layers.eprint.eprint/collected"


class _HeldOutDataset(Protocol):
    """Expose the held-out split shared by the collected variants."""

    def held_out_split(self) -> str:
        """Return the held-out split."""
        ...


def _language() -> LanguageDecl:
    """Return a small language accepted by the grammar boundary."""
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


def _nli() -> CorpusReleaseSpec:
    """Return release metadata shared by the NLI declaration variants."""
    return CorpusReleaseSpec(
        source_id="synthetic-nli",
        release_id="v1",
        split="test",
        language="en",
        source_method="manual",
        spdx="CC0-1.0",
        eprint_refs=(_EPRINT,),
    )


def _pmb() -> CorpusReleaseSpec:
    """Return release metadata shared by the PMB declaration variants."""
    return CorpusReleaseSpec(
        source_id="synthetic-pmb",
        release_id="v1",
        split="test",
        language="en",
        source_method="manual",
        spdx="CC0-1.0",
        eprint_refs=(_EPRINT,),
    )


def _fracas() -> CorpusReleaseSpec:
    """Return release metadata for a FraCaS declaration."""
    return CorpusReleaseSpec(
        source_id="synthetic-fracas",
        release_id="v1",
        split="test",
        language="en",
        source_method="manual",
        spdx="CC0-1.0",
        eprint_refs=(_EPRINT,),
    )


@pytest.mark.parametrize(
    "field",
    ["source_id", "release_id", "split", "language", "source_method", "spdx"],
)
def test_a_corpus_release_refuses_blank_required_metadata(field: str) -> None:
    values = _nli().model_dump()
    values[field] = " "

    with pytest.raises(dx.ValidationError, match="cannot be empty"):
        CorpusReleaseSpec.model_validate(values)


def test_a_corpus_release_refuses_blank_publication_references() -> None:
    values = _nli().model_dump()
    values["eprint_refs"] = ("",)

    with pytest.raises(dx.ValidationError, match="cannot be empty"):
        CorpusReleaseSpec.model_validate(values)


def test_every_collected_kind_is_reachable_through_the_dataset_union(
    tmp_path: Path,
) -> None:
    """Build every exact dataset arm through a run-document section."""
    common = {
        "grammar": json.loads(ccgbank_grammar().model_dump_json()),
        "language": json.loads(_language().model_dump_json()),
    }

    def sources(release: CorpusReleaseSpec) -> list[JsonValue]:
        return [
            {
                "source": str(tmp_path / "release"),
                "release": json.loads(release.model_dump_json()),
            }
        ]

    documents = (
        {**common, "kind": "pmb-ccg", "sources": sources(_pmb())},
        {**common, "kind": "pmb-drs", "sources": sources(_pmb())},
        {**common, "kind": "snli", "sources": sources(_nli())},
        {**common, "kind": "mnli", "sources": sources(_nli())},
        {**common, "kind": "chaosnli", "sources": sources(_nli())},
        {**common, "kind": "unli", "sources": sources(_nli())},
        {**common, "kind": "fracas", "sources": sources(_fracas())},
    )

    built = tuple(
        DatasetSection.model_validate({"dataset": document}).dataset
        for document in documents
    )

    assert tuple(dataset.kind for dataset in built) == (
        "pmb-ccg",
        "pmb-drs",
        "snli",
        "mnli",
        "chaosnli",
        "unli",
        "fracas",
    )
    assert all(
        cast("_HeldOutDataset", dataset).held_out_split() == "test" for dataset in built
    )


def test_pair_corpora_declare_tokenization_aggregation_and_pair_batches() -> None:
    """Keep the tokenizer and majority rule in the run declaration."""
    declaration = SnliDatasetSpec(
        sources=(CorpusSourceSpec(source="fixture.jsonl", release=_nli()),),
        grammar=ccgbank_grammar(),
        language=_language(),
        tokenizer=RegexTokenizerSpec(pattern=r"\w+|[^\w\s]"),
        aggregation=MajorityGoldSpec(),
    )

    assert declaration.tokenizer.pattern == r"\w+|[^\w\s]"
    assert declaration.aggregation.kind == "majority"
    assert declaration.training_batches() == frozenset(
        {TrainingBatchKind.INFERENCE_PAIRS}
    )


type _Collected = (
    PmbCcgDatasetSpec
    | PmbDrsDatasetSpec
    | SnliDatasetSpec
    | MnliDatasetSpec
    | ChaosNliDatasetSpec
    | UnliDatasetSpec
    | FracasDatasetSpec
)


@pytest.mark.parametrize(
    ("option", "schema", "grammar"),
    [
        ("pmb-ccg", PmbCcgDatasetSpec, pmb_grammar("en")),
        ("pmb-drs", PmbDrsDatasetSpec, pmb_grammar("en")),
        ("snli", SnliDatasetSpec, ccgbank_grammar()),
        ("mnli", MnliDatasetSpec, ccgbank_grammar()),
        ("chaosnli", ChaosNliDatasetSpec, ccgbank_grammar()),
        ("unli", UnliDatasetSpec, ccgbank_grammar()),
        ("fracas", FracasDatasetSpec, ccgbank_grammar()),
    ],
)
def test_each_packaged_collected_option_composes_beneath_the_data_root(
    option: str, schema: type[_Collected], grammar: GrammarDecl, tmp_path: Path
) -> None:
    register_resolvers(replace=True)
    path = Path(
        lower_package_resource(
            f"lofi.resource:data/datasets/train/dataset/{option}.yaml"
        )
    )
    extra: dict[str, JsonValue] = {
        "grammar": json.loads(grammar.model_dump_json()),
        "language": json.loads(_language().model_dump_json()),
    }

    with run_context(RunContext(tmp_path, 31)):
        declaration = SpecLoader().compose_traced(path, schema=schema, extra=extra).spec

    assert declaration.kind == option
    assert declaration.grammar == grammar
    assert declaration.sources
    assert all(
        Path(entry.source).is_relative_to(tmp_path) for entry in declaration.sources
    )


def _split_release(split: str, /) -> CorpusSourceSpec:
    """Return one NLI release file assigned to a split."""
    return CorpusSourceSpec(source=f"{split}.jsonl", release=_nli().with_(split=split))


def test_a_collected_declaration_retains_every_release_split_but_the_training_one() -> (
    None
):
    declaration = SnliDatasetSpec(
        sources=(_split_release("train"), _split_release("validation")),
        training_split="train",
        grammar=ccgbank_grammar(),
        language=_language(),
    )

    assert declaration.retained_splits() == frozenset({"validation"})
    assert declaration.held_out_split() == "validation"
    assert declaration.inference_labels() == ("entails", "contradicts", "neutral")


def test_a_collected_declaration_refuses_a_training_split_no_release_reads() -> None:
    with pytest.raises(dx.ValidationError, match="which no declared release reads"):
        SnliDatasetSpec(
            sources=(_split_release("validation"),),
            training_split="train",
            grammar=ccgbank_grammar(),
            language=_language(),
        )


def test_a_collected_declaration_reads_each_split_from_one_release() -> None:
    with pytest.raises(dx.ValidationError, match="each split from one release"):
        SnliDatasetSpec(
            sources=(_split_release("train"), _split_release("train")),
            grammar=ccgbank_grammar(),
            language=_language(),
        )


def test_a_collected_pair_declaration_refuses_an_uninstalled_labeling() -> None:
    with pytest.raises(dx.ValidationError, match="inference_labeling"):
        SnliDatasetSpec(
            sources=(_split_release("train"),),
            labeling="absent",
            grammar=ccgbank_grammar(),
            language=_language(),
        )

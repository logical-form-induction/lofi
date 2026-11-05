from __future__ import annotations

from typing import TYPE_CHECKING, Final

import didactic.api as dx
import pytest

from lofi.data.synthetic.spec.ccgbank import (
    CcgbankReskinDatasetSpec,
    FrequencyReskinSpec,
)
from lofi.spec import RunContext, SpecLoader, register_resolvers, run_context
from lofi.spec.components.data import RetainedDatasetSpec, TrainingBatchKind
from lofi.spec.datasets import ccgbank_grammar
from lofi.spec.language import registered_language

if TYPE_CHECKING:
    from pathlib import Path


GRAMMAR: Final = ccgbank_grammar()
LANGUAGE: Final = registered_language("propositional")


def _declaration(source: str = "ccgbank") -> CcgbankReskinDatasetSpec:
    return CcgbankReskinDatasetSpec(
        source=source,
        splits=("train",),
        grammar=GRAMMAR,
        language=LANGUAGE,
        mapping=FrequencyReskinSpec(prefix="type", minimum_frequency=2),
        seed=7,
    )


def test_the_reskin_declares_a_nondefault_sentence_dataset() -> None:
    declaration = _declaration()

    assert declaration.kind == "ccgbank-reskin"
    assert declaration.training_batches() == frozenset({TrainingBatchKind.SENTENCES})
    assert isinstance(declaration, RetainedDatasetSpec)
    assert declaration.retained_splits() == frozenset()
    assert declaration.mapping.settings() == ("type", 2)


def test_the_reskin_trains_on_its_declared_source_split() -> None:
    declaration = _declaration().with_(
        splits=("train", "validation"), training_split="validation"
    )

    assert declaration.retained_splits() == frozenset({"train"})

    with pytest.raises(dx.ValidationError, match="does not read from"):
        _declaration().with_(training_split="test")


def test_the_frequency_mapping_requires_a_named_prefix_and_positive_cutoff() -> None:
    with pytest.raises(dx.ValidationError, match="prefix is empty"):
        FrequencyReskinSpec(prefix="", minimum_frequency=1)

    with pytest.raises(dx.ValidationError, match="minimum_frequency must be positive"):
        FrequencyReskinSpec(prefix="type", minimum_frequency=0)


def test_a_derived_reskin_seed_is_resolved_at_its_dataset_path() -> None:
    declaration = _declaration().with_(seed="derived")
    reached: list[str] = []

    def derive(path: str) -> int:
        reached.append(path)
        return 19

    resolved = declaration.with_derived_seeds(derive, "data.datasets.train.dataset")

    assert resolved.seed == 19
    assert reached == ["data.datasets.train.dataset.seed"]


def test_a_reskin_document_resolves_its_source_beneath_the_data_root(
    tmp_path: Path,
) -> None:
    register_resolvers(replace=True)
    document = tmp_path / "reskin.yaml"
    document.write_text(
        "\n".join(
            (
                "kind: ccgbank-reskin",
                "source: ${lofi.path:ccgbank}",
                "splits: [train]",
                f"grammar: {GRAMMAR.model_dump_json()}",
                f"language: {LANGUAGE.model_dump_json()}",
                "mapping: {prefix: type, minimum_frequency: 2}",
                "seed: 7",
            )
        ),
        encoding="utf-8",
        newline="\n",
    )

    with run_context(RunContext(tmp_path, 31)):
        declaration = (
            SpecLoader().compose_traced(document, schema=CcgbankReskinDatasetSpec).spec
        )

    assert declaration.source == str(tmp_path / "ccgbank")

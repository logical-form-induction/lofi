from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from tests.language import apply, language

from lofi.lf.models import (
    CRISP,
    EXTREME,
    PRODUCT,
    RESIDUUM,
    WIKIDATA,
    Element,
    EvaluationError,
    Evaluator,
    TruthAlgebra,
    WikidataSettings,
)

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Literal

    from lofi.lf.models import Frame, ModelLanguage, ModelValue
    from lofi.spec.language import LanguageAssembly


def algebra() -> TruthAlgebra:
    return PRODUCT.build(
        implication=RESIDUUM,
        quantifier=EXTREME,
        equality=CRISP,
        margin=0.0,
    )


def files(
    root: Path, *, world: Literal["open", "closed"]
) -> tuple[WikidataSettings, LanguageAssembly]:
    declared = language(
        ("alice", "bob", "human"),
        types={"alice": "e", "bob": "e", "human": "e -> t"},
    )
    vocabulary = declared.store.signature.vocabulary(declared.term_sort)
    snapshot = root / "snapshot.jsonl"
    snapshot.write_text(
        json.dumps(
            {
                "subject": "Q1",
                "property": "P31",
                "object": "Q5",
                "value": True,
            }
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    mapping = root / "mapping.json"
    mapping.write_text(
        json.dumps(
            {
                "carriers": {"e": ["Q1", "Q2"], "v": ["E1"]},
                "constants": {
                    str(vocabulary.id("alice")): {"kind": "item", "item": "Q1"},
                    str(vocabulary.id("bob")): {"kind": "item", "item": "Q2"},
                    str(vocabulary.id("human")): {
                        "kind": "predicate",
                        "property": "P31",
                        "object": "Q5",
                    },
                },
            }
        ),
        encoding="utf-8",
        newline="\n",
    )
    return (
        WikidataSettings(
            str(snapshot),
            str(mapping),
            world,
            "undecided",
        ),
        declared,
    )


def test_an_open_world_snapshot_distinguishes_false_from_unknown(
    tmp_path: Path,
) -> None:
    settings, declared = files(tmp_path, world="open")
    structure = WIKIDATA.read(
        settings,
        declared,
        full_function_space_cap=16,
        henkin_pool=4,
    )
    evaluator = Evaluator(declared, structure, algebra())
    human = declared.constant("human")

    assert evaluator.truth(apply(declared, human, declared.constant("alice"))) == 1.0

    with pytest.raises(EvaluationError, match="unknown"):
        evaluator.truth(apply(declared, human, declared.constant("bob")))


def test_a_closed_world_snapshot_makes_an_absent_fact_false(tmp_path: Path) -> None:
    settings, declared = files(tmp_path, world="closed")
    structure = WIKIDATA.read(
        settings,
        declared,
        full_function_space_cap=16,
        henkin_pool=4,
    )
    evaluator = Evaluator(declared, structure, algebra())

    assert (
        evaluator.truth(
            apply(declared, declared.constant("human"), declared.constant("bob"))
        )
        == 0.0
    )


class SecondEntity:
    def resolve(
        self, identifier: int, language: ModelLanguage, frame: Frame, /
    ) -> ModelValue | None:
        del frame
        vocabulary = language.store.signature.vocabulary(language.term_sort)
        return Element("e", 1) if vocabulary.names[identifier] == "alice" else None


SECOND_ENTITY = SecondEntity()


def test_the_declared_unmapped_policy_supplies_missing_denotations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, declared = files(tmp_path, world="closed")
    vocabulary = declared.store.signature.vocabulary(declared.term_sort)
    mapping_path = tmp_path / "mapping.json"
    mapping = json.loads(mapping_path.read_text())
    del mapping["constants"][str(vocabulary.id("alice"))]
    mapping_path.write_text(json.dumps(mapping))
    distribution = tmp_path / "unmapped_policy-1.0.dist-info"
    distribution.mkdir()
    (distribution / "METADATA").write_text("Name: unmapped-policy\nVersion: 1.0\n")
    (distribution / "entry_points.txt").write_text(
        "[lofi.unmapped_constant]\n"
        "second-entity = tests.unit.lf.models.test__wikidata:SECOND_ENTITY\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    changed = WikidataSettings(
        settings.snapshot, settings.mapping, settings.world, "second-entity"
    )
    structure = WIKIDATA.read(
        changed, declared, full_function_space_cap=16, henkin_pool=4
    )

    assert structure.denotation(vocabulary.id("alice")) == Element("e", 1)

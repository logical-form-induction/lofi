import json
from typing import TYPE_CHECKING, cast
from uuid import UUID

import pytest

from lofi.data.corpora.entailment_graph import (
    EntailmentGraphGeneratorSpec,
    EntailmentGraphLexiconGenerator,
    read_entailment_graph,
)
from lofi.judgment import construct_items, decode_item_construction

if TYPE_CHECKING:
    from pathlib import Path

NAMESPACE = UUID("6697679f-2262-4436-b45b-558a39b7836e")
FILLED = {"element_type": "filled_template_ref"}


def _declared(tmp_path: Path, **changed: object) -> dict[str, object]:
    graph = tmp_path / "graph.tsv"
    graph.write_text("admire\tlike\nbuy\town\n", encoding="utf-8")
    return {
        "kind": "entailment-graph",
        "graph": {"kind": "edge-list", "source": str(graph), "language": "en"},
        "lexicon": "nodes",
        "feature": "paired",
        "linked": "plausible",
        "sampled": "presumed-non-inference",
        "negative_ratio": 1,
        "seed": 17,
        **changed,
    }


def _generator(tmp_path: Path, **changed: object) -> EntailmentGraphLexiconGenerator:
    declaration = EntailmentGraphGeneratorSpec.model_validate_json(
        json.dumps(_declared(tmp_path, **changed))
    )
    settings = declaration.settings()
    return EntailmentGraphLexiconGenerator(
        settings, read_entailment_graph(settings.graph)
    )


def test_each_node_records_its_labeled_pairs(tmp_path: Path) -> None:
    lexicon = _generator(tmp_path).generate(NAMESPACE)
    paired = {
        item.lemma: cast("dict[str, str]", item.features["paired"]) for item in lexicon
    }

    assert sorted(paired) == ["admire", "buy", "like", "own"]
    assert paired["admire"]["like"] == "plausible"
    assert paired["buy"]["own"] == "plausible"
    sampled = [
        (left, right)
        for left, pairs in paired.items()
        for right, label in pairs.items()
        if label == "presumed-non-inference"
    ]
    assert len(sampled) == 2
    assert all(left != right for left, right in sampled)
    assert lexicon.items[0].id == _generator(tmp_path).generate(NAMESPACE).items[0].id


def test_an_undeclared_seed_is_derived_from_the_namespace(tmp_path: Path) -> None:
    first = _generator(tmp_path, seed=None).generate(NAMESPACE)
    again = _generator(tmp_path, seed=None).generate(NAMESPACE)

    assert [item.features for item in first] == [item.features for item in again]


def test_more_drawn_pairs_than_the_graph_admits_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside pairs"):
        _generator(tmp_path, negative_ratio=9).generate(NAMESPACE)


def _template(identity: str, name: str) -> dict[str, object]:
    return {
        "kind": "inline",
        "template": {
            "id": identity,
            "name": name,
            "template_string": "Someone {node} someone.",
            "language_code": "en",
            "slots": {"node": {"name": "node"}},
        },
    }


def test_the_graph_seeds_pair_items_through_the_one_item_construction(
    tmp_path: Path,
) -> None:
    generator = _generator(tmp_path)
    first = "2b3c4d5e-6f70-4a81-9b2c-3d4e5f607181"
    second = "2b3c4d5e-6f70-4a81-9b2c-3d4e5f607182"
    paired = "templates.first.slots.node.features['paired']"
    other = "templates.second.slots.node.lemma"
    document = {
        "namespace": str(NAMESPACE),
        "templates": [_template(first, "first"), _template(second, "second")],
        "lexicons": [],
        "generators": [_declared(tmp_path)],
        "filling": {"strategy": "exhaustive"},
        "template_fillings": {},
        "item_template": {
            "kind": "inline",
            "item_template": {
                "id": "9a3e5c71-1d2b-4f6a-8c4e-7b0d2e1f3a01",
                "name": "pair",
                "judgment_type": "inference",
                "task_type": "binary",
                "task_spec": {"prompt": "Does the first entail the second?"},
                "presentation_spec": {},
                "elements": [
                    {**FILLED, "element_name": "left", "filled_template_ref_id": first},
                    {
                        **FILLED,
                        "element_name": "right",
                        "filled_template_ref_id": second,
                    },
                ],
            },
        },
        "constraints": [],
        "admit": f"{other} in {paired}",
        "attributes": {
            "left": "templates.first.slots.node.lemma",
            "right": other,
            "seed": f"{paired}[{other}]",
        },
    }
    items = construct_items(
        decode_item_construction(json.dumps(document)),
        tmp_path,
        generator_for=lambda _declared: generator,
    )

    labels = {
        (item.item_metadata["left"], item.item_metadata["right"]): item.item_metadata[
            "seed"
        ]
        for item in items
    }
    assert len(items) == 4
    assert labels[("admire", "like")] == "plausible"
    assert labels[("buy", "own")] == "plausible"
    assert list(labels.values()).count("presumed-non-inference") == 2
    assert {item.rendered_elements["left"] for item in items} <= {
        "Someone admire someone.",
        "Someone buy someone.",
        "Someone like someone.",
        "Someone own someone.",
    }

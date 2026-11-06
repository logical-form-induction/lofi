"""Tests for declared bead item construction."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast
from uuid import UUID

import didactic.api as dx
import pytest

from lofi._artifacts import PinnedCheckpoint
from lofi.judgment._items import (
    MaskedLMLexiconGenerator,
    construct_items,
    item_identity,
)
from lofi.judgment._spec import (
    ItemConstructionSpec,
    MaskedLMFilling,
    MaskedLMGeneratorSpec,
    MaskedLMSettings,
    decode_item_construction,
)
from lofi.spec.artifacts import ArtifactCacheSettings

if TYPE_CHECKING:
    from bead.items.item import Item
    from bead.templates.adapters.huggingface import HuggingFaceMLMAdapter

    from lofi.spec.lexicon_generators import LexiconGeneratorSpec

FIXTURE = Path(__file__).parents[2] / "data" / "judgment" / "items"
NAMESPACE = UUID("d1547144-8a00-4c23-b341-6d87f38b0f65")
REVISION = "0123456789abcdef0123456789abcdef01234567"


class _FillMask:
    """Return scored token records for each fixture prompt."""

    def __call__(self, prompt: str, /, *, top_k: int) -> tuple[dict[str, object], ...]:
        assert top_k == 2
        if prompt == "Someone <mask> someone.":
            return (
                {"token_str": " admired", "score": 0.8},
                {"token_str": "called", "score": 0.3},
            )
        assert prompt == "People <mask> people."
        return (
            {"token_str": "helped", "score": 0.9},
            {"token_str": "admired", "score": 0.7},
            {"token_str": "two words", "score": 0.95},
        )


def _generator(declaration: LexiconGeneratorSpec, /) -> MaskedLMLexiconGenerator:
    assert isinstance(declaration, MaskedLMGeneratorSpec)
    settings = declaration.settings()
    return MaskedLMLexiconGenerator(settings, _FillMask())


def _document(**changes: object) -> dict[str, object]:
    document = json.loads((FIXTURE / "items.json").read_text(encoding="utf-8"))
    document["generators"][0]["artifacts"]["root"] = str(
        Path("/var/cache/lofi/artifacts").absolute()
    )
    document.update(changes)
    return document


def _construct(tmp_path: Path, **changes: object) -> tuple[Item, ...]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    for source in FIXTURE.iterdir():
        (tmp_path / source.name).write_bytes(source.read_bytes())
    path = tmp_path / "items.json"
    path.write_text(json.dumps(_document(**changes)), encoding="utf-8")
    declaration = decode_item_construction(path.read_bytes())
    return construct_items(declaration, tmp_path, generator_for=_generator)


def test_a_declared_construction_fills_every_template_with_bead_lexicons(
    tmp_path: Path,
) -> None:
    items = _construct(tmp_path)

    assert tuple(item.rendered_elements for item in items) == (
        {"sentence": "Someone admired someone."},
        {"sentence": "Someone called someone."},
        {"sentence": "someone was admired by Someone."},
        {"sentence": "someone was called by Someone."},
    )
    assert tuple(item.item_metadata for item in items) == (
        {"structure": "transitive", "verb": "admired"},
        {"structure": "transitive", "verb": "called"},
        {"structure": "passive", "verb": "admired"},
        {"structure": "passive", "verb": "called"},
    )
    assert {item.item_template_id for item in items} == {
        UUID("9a3e5c71-1d2b-4f6a-8c4e-7b0d2e1f3a00")
    }


def test_item_identity_is_the_template_and_its_sorted_slot_fillers(
    tmp_path: Path,
) -> None:
    first = _construct(tmp_path / "first")
    again = _construct(tmp_path / "again")
    template = UUID("6f1c3a52-6c55-4a8e-9a0b-0d7c3e2f1a01")
    subject = UUID("0b7e2d10-2f4c-4c55-8f0e-5d1f8a3c2b01")
    obj = UUID("0b7e2d10-2f4c-4c55-8f0e-5d1f8a3c2b02")

    assert tuple(item.id for item in first) == tuple(item.id for item in again)
    assert len({item.id for item in first}) == len(first)
    assert item_identity(
        NAMESPACE, {template: {"subject": subject, "object": obj}}
    ) == item_identity(NAMESPACE, {template: {"object": obj, "subject": subject}})
    assert item_identity(NAMESPACE, {template: {"subject": subject}}) != (
        item_identity(NAMESPACE, {template: {"subject": obj}})
    )


def test_generated_lexical_items_record_the_pinned_model_as_their_source() -> None:
    generator = MaskedLMLexiconGenerator(
        MaskedLMSettings(
            "masked-lm",
            "verbs",
            "eng",
            {"pos": "VERB"},
            PinnedCheckpoint("fixture/masked-lm", REVISION),
            ArtifactCacheSettings(
                Path("/var/cache/lofi/artifacts").absolute(), offline=True
            ),
            "<mask>",
            2,
            ("Someone <mask> someone.", "People <mask> people."),
        ),
        _FillMask(),
    )

    lexicon = generator.generate(NAMESPACE)

    assert lexicon.name == "verbs"
    assert tuple(item.lemma for item in lexicon.items) == ("helped", "admired")
    assert {item.source for item in lexicon.items} == {
        f"masked-lm:fixture/masked-lm@{REVISION}"
    }
    assert all(item.features == {"pos": "VERB"} for item in lexicon.items)
    assert lexicon.items[0].id == generator.generate(NAMESPACE).items[0].id


def test_an_inline_resource_without_an_identity_is_refused(tmp_path: Path) -> None:
    template = json.loads(
        (FIXTURE / "templates.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    del template["id"]

    with pytest.raises(ValueError, match="declares no id"):
        _construct(tmp_path, templates=[{"kind": "inline", "template": template}])


def test_a_template_constraint_is_refused_by_an_enumerating_strategy(
    tmp_path: Path,
) -> None:
    lines = (FIXTURE / "templates.jsonl").read_text(encoding="utf-8").splitlines()
    template = json.loads(lines[0])
    template["constraints"] = [{"expression": "subject.lemma != object.lemma"}]
    inline = [{"kind": "inline", "template": template}]

    with pytest.raises(ValueError, match="template constraint"):
        _construct(tmp_path / "exhaustive", templates=inline)

    constrained = _construct(
        tmp_path / "csp",
        templates=inline,
        filling={"strategy": "csp", "count": 10, "max_attempts": 1000},
    )
    assert len(constrained) == 2


def test_an_attribute_naming_no_slot_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="attribute 'argument'"):
        _construct(tmp_path, attributes={"argument": "slots.theme.lemma"})


def test_an_item_template_without_elements_is_refused(tmp_path: Path) -> None:
    declared = json.loads((FIXTURE / "item-template.json").read_text())

    with pytest.raises(ValueError, match="element"):
        _construct(
            tmp_path,
            item_template={
                "kind": "inline",
                "item_template": {**declared, "elements": []},
            },
        )


def test_the_declaration_is_validated_at_its_boundary() -> None:
    declaration = decode_item_construction(json.dumps(_document()).encode())

    assert isinstance(declaration, ItemConstructionSpec)
    assert tuple(declaration.attributes) == ("structure", "verb")
    with pytest.raises(dx.ValidationError, match="attribute"):
        decode_item_construction(json.dumps(_document(attributes={})).encode("utf-8"))


def _templates() -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (FIXTURE / "templates.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]


def _item_template(**changes: object) -> dict[str, object]:
    declared = json.loads((FIXTURE / "item-template.json").read_text())
    declared.update(changes)
    return {"kind": "inline", "item_template": declared}


def test_no_slot_name_is_reserved_by_the_attribute_context(tmp_path: Path) -> None:
    line = (FIXTURE / "templates.jsonl").read_text(encoding="utf-8").splitlines()[0]
    template = json.loads(line.replace("predicate", "template"))

    items = _construct(
        tmp_path,
        templates=[{"kind": "inline", "template": template}],
        attributes={"name": "template.name", "verb": "slots.template.lemma"},
    )

    assert {item.item_metadata["verb"] for item in items} == {"admired", "called"}
    assert {item.item_metadata["name"] for item in items} == {"transitive"}


def test_template_and_element_metadata_are_readable_by_an_attribute(
    tmp_path: Path,
) -> None:
    declared = json.loads((FIXTURE / "item-template.json").read_text())
    element = {**declared["elements"][0], "element_metadata": {"shown": "first"}}

    items = _construct(
        tmp_path,
        item_template=_item_template(
            template_metadata={"study": "s1"}, elements=[element]
        ),
        attributes={
            "study": "item_template.template_metadata['study']",
            "shown": "elements.sentence.element_metadata['shown']",
        },
    )

    assert {item.item_metadata["study"] for item in items} == {"s1"}
    assert {item.item_metadata["shown"] for item in items} == {"first"}


def test_each_named_template_is_filled_by_its_own_strategy(tmp_path: Path) -> None:
    items = _construct(
        tmp_path,
        template_fillings={"passive": {"strategy": "random", "samples": 1, "seed": 3}},
    )

    structures = [item.item_metadata["structure"] for item in items]
    assert structures.count("transitive") == 2
    assert structures.count("passive") == 1
    with pytest.raises(ValueError, match="undeclared templates"):
        _construct(
            tmp_path / "unknown",
            template_fillings={"other": {"strategy": "exhaustive"}},
        )


def test_an_undeclared_random_seed_is_derived_from_the_namespace(
    tmp_path: Path,
) -> None:
    filling = {"strategy": "random", "samples": 1, "seed": None}
    first = _construct(tmp_path / "first", filling=filling)
    again = _construct(tmp_path / "again", filling=filling)
    moved = _construct(
        tmp_path / "moved",
        filling=filling,
        namespace="5b0e2f38-8d6f-4f42-9a51-3c7d1e0b9a10",
    )

    assert tuple(item.id for item in first) == tuple(item.id for item in again)
    assert len(first) == 2
    assert {item.id for item in moved}.isdisjoint({item.id for item in first})


def test_a_stratified_filling_balances_a_filler_property(tmp_path: Path) -> None:
    items = _construct(
        tmp_path,
        filling={
            "strategy": "stratified",
            "samples": 2,
            "grouping_property": "lemma",
            "seed": 5,
        },
    )

    assert len(items) == 4


def test_a_mixed_filling_fills_each_named_slot_by_its_strategy(
    tmp_path: Path,
) -> None:
    items = _construct(
        tmp_path,
        filling={
            "strategy": "mixed",
            "slots": {"predicate": {"strategy": "random", "samples": 1, "seed": 1}},
            "masked_lm_slots": [],
            "masked_lm": None,
            "default": {"strategy": "exhaustive"},
        },
    )

    assert len(items) == 2
    assert len({item.item_metadata["verb"] for item in items}) == 1
    with pytest.raises(dx.ValidationError, match="exactly for its slots"):
        decode_item_construction(
            json.dumps(
                _document(
                    filling={
                        "strategy": "mixed",
                        "slots": {},
                        "masked_lm_slots": ["predicate"],
                        "masked_lm": None,
                        "default": {"strategy": "exhaustive"},
                    }
                )
            ).encode()
        )


def test_a_streaming_filling_bounds_the_fillings_of_each_template(
    tmp_path: Path,
) -> None:
    items = _construct(
        tmp_path, filling={"strategy": "streaming", "max_combinations": 1}
    )

    assert len(items) == 2


class _Adapter:
    """Answer bead's masked-LM strategy with fixed completions."""

    model_name = "fixture/masked-lm"

    def is_loaded(self) -> bool:
        return True

    def get_mask_token(self) -> str:
        return "<mask>"

    def predict_masked_token(
        self, text: str, mask_position: int, top_k: int
    ) -> list[tuple[str, float]]:
        del text, mask_position, top_k
        return [("admired", -0.1), ("called", -2.0), ("Someone", -3.0)]

    def predict_masked_token_batch(
        self, texts: list[str], mask_position: int, top_k: int
    ) -> list[list[tuple[str, float]]]:
        return [self.predict_masked_token(text, mask_position, top_k) for text in texts]


def test_a_masked_lm_filling_ranks_lexicon_items_by_bead_beam_search(
    tmp_path: Path,
) -> None:
    template = {
        "id": "6f1c3a52-6c55-4a8e-9a0b-0d7c3e2f1a09",
        "name": "single",
        "template_string": "Someone {predicate} someone.",
        "language_code": "eng",
        "slots": {
            "predicate": {
                "name": "predicate",
                "constraints": [{"expression": "self.features.get('pos') == 'VERB'"}],
            }
        },
    }
    filling = {
        "strategy": "mlm",
        "checkpoint": {
            "kind": "hub",
            "repository": "fixture/masked-lm",
            "revision": REVISION,
        },
        "artifacts": {
            "root": str(Path("/var/cache/lofi/artifacts").absolute()),
            "offline": True,
        },
        "device": "cpu",
        "beam_size": 1,
        "fill_direction": "left_to_right",
        "custom_order": None,
        "top_k": 3,
        "budget": None,
        "per_slot_max_fills": {},
        "per_slot_enforce_unique": {},
    }
    loaded: list[str] = []

    def adapter_for(declared: MaskedLMFilling, /) -> HuggingFaceMLMAdapter:
        loaded.append(declared.pinned().repository)
        return cast("HuggingFaceMLMAdapter", _Adapter())

    for source in FIXTURE.iterdir():
        (tmp_path / source.name).write_bytes(source.read_bytes())
    document = _document(
        templates=[{"kind": "inline", "template": template}], filling=filling
    )
    items = construct_items(
        decode_item_construction(json.dumps(document).encode()),
        tmp_path,
        generator_for=_generator,
        adapter_for=adapter_for,
    )

    assert loaded == ["fixture/masked-lm"]
    assert [item.rendered_elements["sentence"] for item in items] == [
        "Someone admired someone."
    ]


def test_an_item_combines_the_templates_its_elements_name(tmp_path: Path) -> None:
    transitive, passive = _templates()
    declared = json.loads((FIXTURE / "item-template.json").read_text())
    elements = [
        {
            **declared["elements"][0],
            "element_name": "first",
            "filled_template_ref_id": transitive["id"],
        },
        {
            **declared["elements"][0],
            "element_name": "second",
            "filled_template_ref_id": passive["id"],
        },
    ]

    items = _construct(
        tmp_path,
        item_template=_item_template(elements=elements),
        attributes={
            "left": "templates.transitive.slots.predicate.lemma",
            "right": "templates.passive.slots.predicate.lemma",
        },
    )

    assert len(items) == 4
    pairs = {
        (item.item_metadata["left"], item.item_metadata["right"]) for item in items
    }
    assert pairs == {
        ("admired", "admired"),
        ("admired", "called"),
        ("called", "admired"),
        ("called", "called"),
    }
    assert all(
        item.rendered_elements["second"].startswith("someone was") for item in items
    )


def test_an_item_template_naming_some_templates_only_is_refused(
    tmp_path: Path,
) -> None:
    transitive, _ = _templates()
    declared = json.loads((FIXTURE / "item-template.json").read_text())
    element = {**declared["elements"][0], "filled_template_ref_id": transitive["id"]}

    with pytest.raises(ValueError, match="no element renders"):
        _construct(tmp_path, item_template=_item_template(elements=[element]))


def test_item_template_constraints_are_honored_through_bead(tmp_path: Path) -> None:
    constraint = {
        "id": "3c1f0e2a-9b7d-4e5f-8a6b-2d4c6e8f0a10",
        "expression": "'called' not in sentence",
    }

    items = _construct(
        tmp_path,
        item_template=_item_template(constraints=[constraint["id"]]),
        constraints=[{"kind": "inline", "constraint": constraint}],
    )

    assert {item.item_metadata["verb"] for item in items} == {"admired"}
    with pytest.raises(ValueError, match="does not declare"):
        _construct(
            tmp_path / "undeclared",
            item_template=_item_template(constraints=[constraint["id"]]),
        )
    with pytest.raises(ValueError, match="does not name"):
        _construct(
            tmp_path / "unused",
            constraints=[{"kind": "inline", "constraint": constraint}],
        )


def test_a_template_collection_file_is_a_template_source(tmp_path: Path) -> None:
    collection = {
        "id": "7a1b2c3d-4e5f-4a6b-8c7d-9e0f1a2b3c4d",
        "name": "fixture",
        "templates": _templates(),
    }
    (tmp_path / "collection.json").write_text(json.dumps(collection))

    items = _construct(
        tmp_path, templates=[{"kind": "collection", "path": "collection.json"}]
    )

    assert len(items) == 4


def test_a_table_lexicon_is_read_by_bead_with_derived_identities(
    tmp_path: Path,
) -> None:
    (tmp_path / "arguments.tsv").write_text(
        "word\trole\nSomeone\tsubject\nsomeone\tobject\n", encoding="utf-8"
    )
    table = {
        "kind": "table",
        "path": "arguments.tsv",
        "name": "arguments",
        "format": "tsv",
        "language_code": "eng",
        "column_mapping": {"word": "lemma"},
        "feature_columns": ["role"],
        "pos": None,
    }

    first = _construct(tmp_path, lexicons=[table])
    again = _construct(tmp_path, lexicons=[table])

    assert len(first) == 4
    assert tuple(item.id for item in first) == tuple(item.id for item in again)

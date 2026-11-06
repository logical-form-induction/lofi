from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.run._boundary.corpus_designs import (
    BeadDesignSpec,
    CausalLmReadoutsSpec,
    FragmentDesignSpec,
    LayersExportSpec,
    RelationResponsesSpec,
    RootMassSpec,
    declared_data_root,
    decode_design,
)
from lofi.spec.artifacts import HubSourceSpec

if TYPE_CHECKING:
    from lofi._types import JsonValue


def _fragment(**changes: JsonValue) -> dict[str, JsonValue]:
    document: dict[str, JsonValue] = {
        "source": "fragment",
        "fragment": {
            "kind": "codec",
            "codec": "megaacceptability",
            "declaration": "release/release.json",
            "responses": "release/responses.tsv",
        },
        "responses": {"kind": "ordinal"},
        "factors": {"structure": "item['structure']"},
        "readouts": None,
        "keep": None,
    }
    document.update(changes)
    return document


def _decode(document: dict[str, JsonValue]) -> object:
    return decode_design(json.dumps(document).encode())


def test_a_fragment_design_reads_factors_as_named_expressions() -> None:
    spec = _decode(_fragment(factors={"lemma": "item['lemma']", "kind": "item.k"}))

    assert isinstance(spec, FragmentDesignSpec)
    assert spec.factors == {"lemma": "item['lemma']", "kind": "item.k"}


def test_a_factor_names_a_nonblank_expression() -> None:
    with pytest.raises(dx.ValidationError, match="expression"):
        _decode(_fragment(factors={"lemma": "  "}))


def test_a_layers_export_is_a_fragment_source() -> None:
    spec = _decode(_fragment(fragment={"kind": "layers", "path": "export.jsonl"}))

    assert isinstance(spec, FragmentDesignSpec)
    assert spec.fragment == LayersExportSpec(path="export.jsonl")


def test_a_relation_reading_declares_its_inventory_order() -> None:
    spec = _decode(
        _fragment(responses={"kind": "relation", "inventory": ["no", "yes"]})
    )

    assert isinstance(spec, FragmentDesignSpec)
    assert spec.responses == RelationResponsesSpec(inventory=("no", "yes"))
    with pytest.raises(dx.ValidationError, match="at least two"):
        _decode(_fragment(responses={"kind": "relation", "inventory": ["no"]}))


def test_readouts_join_only_an_ordinal_design() -> None:
    readouts: dict[str, JsonValue] = {
        "kind": "root-mass",
        "path": "runs/root-mass.json",
        "mass": "score",
        "complement": "score_complement",
    }
    spec = _decode(_fragment(readouts=readouts))

    assert isinstance(spec, FragmentDesignSpec)
    assert isinstance(spec.readouts, RootMassSpec)
    with pytest.raises(dx.ValidationError, match="ordinal"):
        _decode(_fragment(readouts=readouts, responses={"kind": "bounded"}))
    with pytest.raises(dx.ValidationError, match="both fill"):
        RootMassSpec(path="runs/root-mass.json", mass="score", complement="score")


def test_a_causal_readout_names_the_element_it_scores() -> None:
    spec = _decode(
        _fragment(
            readouts={
                "kind": "causal-lm",
                "element": "stimulus",
                "unigrams": "morcela/unigrams.tsv",
                "artifacts": {
                    "root": str(Path("/var/cache/lofi/artifacts").absolute()),
                    "offline": True,
                },
                "checkpoint": {
                    "kind": "hub",
                    "repository": "org/model",
                    "revision": "a" * 40,
                },
                "columns": {
                    "log_probability": "log_p",
                    "unigram_log_probability": "unigram",
                    "length": "n",
                },
            }
        )
    )

    assert isinstance(spec, FragmentDesignSpec)
    assert isinstance(spec.readouts, CausalLmReadoutsSpec)
    assert spec.readouts.element == "stimulus"
    assert spec.readouts.columns.log_probability == "log_p"


def test_a_bead_design_names_its_construction_and_records() -> None:
    spec = _decode(
        {
            "source": "bead",
            "items": "items/items.json",
            "judgments": "items/judgments.jsonl",
            "encoding": "items/encoding.json",
            "factors": {"structure": "item['structure']"},
        }
    )

    assert isinstance(spec, BeadDesignSpec)
    assert spec.items == "items/items.json"


def test_each_document_refuses_unknown_fields() -> None:
    with pytest.raises(dx.ValidationError):
        _decode(_fragment(unexpected=True))


def test_the_environment_declares_the_data_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOFI_DATA_ROOT", "/srv/corpora")

    assert declared_data_root() == "/srv/corpora"


@pytest.mark.parametrize(
    ("value", "message"), [(None, "not supplied"), ("  ", "cannot be blank")]
)
def test_an_undeclared_data_root_is_refused(
    monkeypatch: pytest.MonkeyPatch, value: str | None, message: str
) -> None:
    if value is None:
        monkeypatch.delenv("LOFI_DATA_ROOT", raising=False)
    else:
        monkeypatch.setenv("LOFI_DATA_ROOT", value)

    with pytest.raises(ValueError, match=message):
        declared_data_root()


@pytest.mark.parametrize("revision", ["main", "A" * 40, "a" * 39])
def test_a_checkpoint_revision_is_a_full_commit(revision: str) -> None:
    with pytest.raises(dx.ValidationError, match="full lowercase"):
        HubSourceSpec(repository="org/model", revision=revision)

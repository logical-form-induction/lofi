from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final

from lairs.records import expression as expression_records
from lairs.records import ontology as ontology_records
from lairs.store.repository import Repository

from lofi.data.corpora.layers.spec import records_at

if TYPE_CHECKING:
    from pathlib import Path

CREATED: Final = datetime(2026, 1, 1, tzinfo=UTC)

EXPRESSION_URI: Final = "at://did:plc:test/pub.layers.expression.expression/valid"

INVALID_URI: Final = "at://did:plc:test/pub.layers.expression.expression/invalid"

UNKNOWN_URI: Final = "at://did:plc:test/pub.layers.ontology.typeDef/type"


def expression(identifier: str) -> expression_records.Expression:
    return expression_records.Expression(
        id=identifier,
        kind="sentence",
        createdAt=CREATED,
        text=identifier,
        languages=("en",),
    )


def test_repository_records_validate_known_collections_and_ignore_others(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "repository"
    repository = Repository.init(directory)
    repository.save(INVALID_URI, expression("invalid"))
    invalid_path = next((directory / "records").glob("*.json"))
    invalid_path.write_text('{"id": "invalid"}', encoding="utf-8", newline="\n")
    repository.save(EXPRESSION_URI, expression("valid"))
    repository.save(
        UNKNOWN_URI,
        ontology_records.TypeDef(
            createdAt=CREATED,
            name="type",
            typeKind="entity",
            ontologyRef="at://did:plc:test/pub.layers.ontology.ontology/main",
        ),
    )

    reading = records_at(repository)

    assert [uri for uri, _ in reading.records] == [EXPRESSION_URI]
    assert reading.corpus_uris == ()
    assert reading.skips == ("a Layers record that fails schema validation",)

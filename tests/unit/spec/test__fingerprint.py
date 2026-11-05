import copy
import json
import os
import re
import subprocess
import sys
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, cast

import didactic.api as dx
import pytest

from lofi.formalism import TheoryArtifact
from lofi.spec import (
    SUBSTRATE_DISTRIBUTIONS,
    BoundaryMismatchError,
    FingerprintIso,
    IndexedLanguage,
    RunFingerprint,
    RunSpec,
    decode_fingerprint,
    fingerprint_run,
    installed_versions,
    require_unchanged,
    schema_uris,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi._types import JsonValue

TREE: Final = Path(__file__).resolve().parents[3]
FIXTURE: Final = Path(__file__).with_name("fixtures") / "run_fingerprint.json"

GRAMMAR_SPEC: Final = (
    b'{"route":{"source":"decl-1","target":"native-1"},"theory":"S NP"}'
)
FORMULA_SPEC: Final = b'{"route":{"source":"lf-1","target":"lambda-1"},"theory":"e t"}'

GRAMMAR: Final = TheoryArtifact.from_specification("decl-1->native-1", GRAMMAR_SPEC)
GRAMMAR_ROUTE: Final = dx.LoweringRoute("decl-1", "native-1")
FORMULAS: Final = TheoryArtifact.from_specification("lf-1->lambda-1", FORMULA_SPEC)


def indexed(artifact: TheoryArtifact, route: dx.LoweringRoute) -> IndexedLanguage:
    return IndexedLanguage(
        artifact.version, artifact.specification, artifact.digest, route
    )


LANGUAGES: Final[dict[str, IndexedLanguage]] = {
    "grammar": indexed(GRAMMAR, GRAMMAR_ROUTE),
    "formulas": indexed(FORMULAS, dx.LoweringRoute("lf-1", "lambda-1")),
}

FRESH: Final = """
import json
import sys

from lofi.spec import FingerprintIso, decode_fingerprint, fingerprint_run

committed = decode_fingerprint(sys.stdin.buffer.read())
fresh = fingerprint_run(committed.languages)

sys.stdout.buffer.write(FingerprintIso().forward(fresh))
"""

TASK_IDENTIFIER: Final = re.compile(r"\b[A-Z]{2,6}-\d{3}\b")


def committed_record() -> dict[str, JsonValue]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def decoded(record: JsonValue) -> RunFingerprint:
    return decode_fingerprint(json.dumps(record).encode())


def record_of(fingerprint: RunFingerprint) -> dict[str, JsonValue]:
    return json.loads(FingerprintIso().forward(fingerprint))


def grammar_of(record: dict[str, JsonValue]) -> dict[str, JsonValue]:
    languages = cast("dict[str, dict[str, JsonValue]]", record["languages"])

    return languages["grammar"]


def with_spec(spec: bytes) -> IndexedLanguage:
    return indexed(
        TheoryArtifact.from_specification("decl-1->native-1", spec),
        dx.LoweringRoute("decl-1", "native-1"),
    )


def renamed(name: str) -> dict[str, IndexedLanguage]:
    return {
        name if key == "grammar" else key: value for key, value in LANGUAGES.items()
    }


def replaced(language: IndexedLanguage) -> dict[str, IndexedLanguage]:
    return {**LANGUAGES, "grammar": language}


def fresh_in_a_new_process(seed: str) -> RunFingerprint:
    finished = subprocess.run(
        [sys.executable, "-c", FRESH],
        input=FIXTURE.read_text(encoding="utf-8"),
        capture_output=True,
        encoding="utf-8",
        check=True,
        cwd=TREE,
        env={**os.environ, "PYTHONHASHSEED": seed},
    )

    return decode_fingerprint(finished.stdout.encode())


def test_the_committed_record_matches_a_fresh_fingerprint() -> None:
    committed = decoded(committed_record())
    fresh = fresh_in_a_new_process("0")

    assert committed == fresh
    assert committed.digest == fresh.digest
    assert committed.document == fresh.document
    require_unchanged(committed, fresh)


def test_the_committed_digest_is_the_sha256_of_the_committed_document() -> None:
    committed = decoded(committed_record())

    assert committed.digest == sha256(committed.document).hexdigest()
    assert committed_record()["digest"] == committed.digest


@pytest.mark.parametrize("seed", ["0", "4242"])
def test_a_fresh_process_under_another_hash_seed_reproduces_the_record(
    seed: str,
) -> None:
    fresh = fresh_in_a_new_process(seed)
    committed = decoded(committed_record())

    assert fresh.document == committed.document
    assert fresh.digest == committed.digest


def test_identical_inputs_produce_identical_bytes_and_digest() -> None:
    first = fingerprint_run(LANGUAGES)
    second = fingerprint_run(dict(LANGUAGES))

    assert first.document == second.document
    assert first.digest == second.digest
    assert first == second
    assert hash(first) == hash(second)


def test_the_run_specification_schema_uri_is_the_public_one() -> None:
    schemas = fingerprint_run(LANGUAGES).schemas

    assert schemas["lofi.spec.components.run.RunSpec"] == dx.schema_uri(RunSpec)
    assert dict(schemas) == schema_uris(RunSpec)


def test_language_order_does_not_enter_the_document() -> None:
    reversed_languages = dict(reversed(LANGUAGES.items()))

    assert (
        fingerprint_run(reversed_languages).document
        == fingerprint_run(LANGUAGES).document
    )


def test_the_document_orders_every_member_by_name() -> None:
    document = json.loads(fingerprint_run(LANGUAGES).document)

    assert list(document) == ["languages", "packages", "schemas"]
    assert list(document["languages"]) == sorted(LANGUAGES)
    assert list(document["schemas"]) == sorted(schema_uris(RunSpec))
    assert list(document["packages"]) == sorted(SUBSTRATE_DISTRIBUTIONS)


def test_the_document_holds_the_lowercase_hex_specification() -> None:
    document = json.loads(fingerprint_run(LANGUAGES).document)
    grammar = document["languages"]["grammar"]

    assert bytes.fromhex(grammar["specification"]) == GRAMMAR_SPEC
    assert grammar["specification"] == grammar["specification"].lower()
    assert grammar["digest"] == sha256(GRAMMAR_SPEC).hexdigest()
    assert grammar["route"] == {"source": "decl-1", "target": "native-1"}
    assert grammar["version"] == "decl-1->native-1"


CHANGES: Final[dict[str, Callable[[], RunFingerprint]]] = {
    "schema uri": lambda: RunFingerprint(
        {**schema_uris(RunSpec), "lofi.spec.components.run.RunSpec": "didactic://v1/0"},
        LANGUAGES,
        installed_versions(),
    ),
    "one specification byte": lambda: fingerprint_run(
        replaced(with_spec(GRAMMAR_SPEC[:-1] + b" "))
    ),
    "artifact version": lambda: fingerprint_run(
        replaced(
            indexed(
                TheoryArtifact("decl-2->native-1", GRAMMAR_SPEC, GRAMMAR.digest),
                dx.LoweringRoute("decl-1", "native-1"),
            )
        )
    ),
    "route source": lambda: fingerprint_run(
        replaced(indexed(GRAMMAR, dx.LoweringRoute("decl-2", "native-1")))
    ),
    "route target": lambda: fingerprint_run(
        replaced(indexed(GRAMMAR, dx.LoweringRoute("decl-1", "native-2")))
    ),
    "language name": lambda: fingerprint_run(renamed("syntax")),
    "package version": lambda: fingerprint_run(
        LANGUAGES, packages={**installed_versions(), "lairs": "0.0.0"}
    ),
}


@pytest.mark.parametrize("change", CHANGES, ids=CHANGES)
def test_a_change_to_any_member_changes_the_fingerprint(change: str) -> None:
    baseline = fingerprint_run(LANGUAGES)
    changed = CHANGES[change]()

    assert changed.document != baseline.document
    assert changed.digest != baseline.digest
    assert changed != baseline


@pytest.mark.parametrize("change", CHANGES, ids=CHANGES)
def test_a_changed_fingerprint_is_refused(change: str) -> None:
    committed = decoded(committed_record())

    with pytest.raises(BoundaryMismatchError) as failure:
        require_unchanged(committed, CHANGES[change]())

    assert failure.value.committed is committed
    assert failure.value.members
    assert committed.digest in str(failure.value)


def test_the_boundary_error_names_the_members_that_differ() -> None:
    committed = fingerprint_run(LANGUAGES)

    with pytest.raises(BoundaryMismatchError) as failure:
        require_unchanged(
            committed, fingerprint_run(replaced(with_spec(GRAMMAR_SPEC + b" ")))
        )

    assert failure.value.members == (
        "languages.grammar.digest",
        "languages.grammar.specification",
    )
    assert "languages.grammar.digest" in str(failure.value)


def test_a_renamed_language_is_reported_at_both_names() -> None:
    committed = fingerprint_run(LANGUAGES)

    with pytest.raises(BoundaryMismatchError) as failure:
        require_unchanged(committed, fingerprint_run(renamed("syntax")))

    assert failure.value.members == ("languages.grammar", "languages.syntax")


def test_a_tampered_artifact_digest_is_refused_at_rebuild() -> None:
    record = committed_record()
    grammar_of(record)["digest"] = sha256(GRAMMAR_SPEC + b" ").hexdigest()

    with pytest.raises(ValueError, match="does not match"):
        decoded(record)


def test_a_tampered_record_digest_is_refused_at_rebuild() -> None:
    record = committed_record()
    record["digest"] = sha256(b"").hexdigest()

    with pytest.raises(ValueError, match="does not match the rebuilt"):
        decoded(record)


def test_the_record_holds_exactly_the_substrate_distributions_as_installed() -> None:
    committed = decoded(committed_record())
    installed = {name: version(name) for name in SUBSTRATE_DISTRIBUTIONS}

    assert dict(committed.packages) == installed
    assert installed_versions() == installed
    assert set(installed) == {"didactic", "didactic-settings", "lairs", "panproto"}


def test_the_record_round_trips_through_its_json_form() -> None:
    fingerprint = fingerprint_run(LANGUAGES)
    record = record_of(fingerprint)
    rebuilt = decoded(record)

    assert rebuilt == fingerprint
    assert rebuilt.languages == fingerprint.languages
    assert rebuilt.packages == fingerprint.packages
    assert rebuilt.schemas == fingerprint.schemas


def test_the_record_handed_out_is_not_the_fingerprint_state() -> None:
    fingerprint = fingerprint_run(LANGUAGES)
    record = record_of(fingerprint)
    before = fingerprint.document
    record["schemas"] = {"run": "didactic://v1/0"}
    grammar_of(record)["version"] = "tampered"

    assert fingerprint.document == before
    assert record_of(fingerprint) != record


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (lambda record: record.pop("schemas"), "schemas"),
        (lambda record: record.pop("languages"), "languages"),
        (lambda record: record.pop("digest"), "digest"),
        (lambda record: grammar_of(record).pop("route"), "route"),
        (lambda record: grammar_of(record).pop("version"), "version"),
        (lambda record: grammar_of(record).update(previous="old"), "previous"),
        (lambda record: record.update(previous="old"), "previous"),
        (
            lambda record: grammar_of(record).update(
                specification=str(grammar_of(record)["specification"]).upper()
            ),
            "not lowercase hex",
        ),
        (
            lambda record: grammar_of(record).update(specification="zz"),
            "not lowercase hex",
        ),
    ],
)
def test_a_record_missing_or_corrupting_a_member_is_refused(
    edit: Callable[[dict[str, JsonValue]], JsonValue], match: str
) -> None:
    record = committed_record()
    edit(record)

    with pytest.raises(ValueError, match=match):
        decoded(record)


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (lambda record: record.update(schemas=7), "schemas"),
        (lambda record: record.update(schemas={"run": 7}), "schemas"),
        (lambda record: record.update(packages=["didactic"]), "packages"),
        (
            lambda record: cast("dict[str, JsonValue]", record["languages"]).update(
                grammar="x"
            ),
            "languages",
        ),
        (
            lambda record: cast("dict[str, JsonValue]", record["packages"]).update(
                lairs=8
            ),
            "packages",
        ),
    ],
)
def test_a_record_with_a_member_of_the_wrong_shape_is_refused(
    edit: Callable[[dict[str, JsonValue]], JsonValue], match: str
) -> None:
    record = committed_record()
    edit(record)

    with pytest.raises(ValueError, match=match):
        decoded(record)


def test_a_record_that_is_not_an_object_is_refused() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        decoded([])


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (
            lambda: RunFingerprint({"run": ""}, LANGUAGES, installed_versions()),
            "schema URI is empty",
        ),
        (
            lambda: RunFingerprint({"": "didactic://v1/0"}, LANGUAGES, {}),
            "model name is empty",
        ),
        (
            lambda: RunFingerprint({}, LANGUAGES, {}),
            "at least one model",
        ),
        (
            lambda: RunFingerprint(
                {"run": "didactic://v1/0"}, {"": LANGUAGES["grammar"]}, {}
            ),
            "language name is empty",
        ),
        (
            lambda: RunFingerprint({"run": "didactic://v1/0"}, {}, {"didactic": ""}),
            "package version is empty",
        ),
    ],
)
def test_an_empty_name_or_version_is_refused(
    build: Callable[[], RunFingerprint], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        build()


def test_a_language_that_is_not_an_indexed_language_is_refused() -> None:
    bare = cast("IndexedLanguage", GRAMMAR)

    with pytest.raises(TypeError, match="not an IndexedLanguage"):
        RunFingerprint({"run": "didactic://v1/0"}, {"grammar": bare}, {})


def test_an_indexed_language_holds_the_artifact_it_was_built_from() -> None:
    language = LANGUAGES["grammar"]

    assert (language.version, language.specification, language.digest) == (
        GRAMMAR.version,
        GRAMMAR.specification,
        GRAMMAR.digest,
    )
    assert language.route == dx.LoweringRoute("decl-1", "native-1")


@pytest.mark.parametrize(
    ("build", "failure", "match"),
    [
        (
            lambda: IndexedLanguage("", GRAMMAR_SPEC, GRAMMAR.digest, GRAMMAR_ROUTE),
            ValueError,
            "version is empty",
        ),
        (
            lambda: IndexedLanguage("v", b"", sha256(b"").hexdigest(), GRAMMAR_ROUTE),
            ValueError,
            "specification is empty",
        ),
        (
            lambda: IndexedLanguage(
                "v", cast("bytes", GRAMMAR_SPEC.decode()), GRAMMAR.digest, GRAMMAR_ROUTE
            ),
            TypeError,
            "not bytes",
        ),
        (
            lambda: IndexedLanguage("v", GRAMMAR_SPEC, FORMULAS.digest, GRAMMAR_ROUTE),
            ValueError,
            "does not match",
        ),
        (
            lambda: IndexedLanguage(
                "v", GRAMMAR_SPEC, GRAMMAR.digest, cast("dx.LoweringRoute", ("a", "b"))
            ),
            TypeError,
            "not a lowering route",
        ),
    ],
    ids=["empty version", "empty specification", "text", "wrong digest", "bare route"],
)
def test_an_indexed_language_refuses_a_malformed_field(
    build: Callable[[], IndexedLanguage], failure: type[Exception], match: str
) -> None:
    with pytest.raises(failure, match=match):
        build()


def test_an_indexed_language_compares_every_field_and_the_route() -> None:
    same = IndexedLanguage(
        "decl-1->native-1", GRAMMAR_SPEC, GRAMMAR.digest, GRAMMAR_ROUTE
    )
    other_route = indexed(GRAMMAR, dx.LoweringRoute("decl-1", "native-2"))
    other_version = indexed(
        TheoryArtifact("decl-2->native-1", GRAMMAR_SPEC, GRAMMAR.digest), GRAMMAR_ROUTE
    )

    assert same == LANGUAGES["grammar"]
    assert hash(same) == hash(LANGUAGES["grammar"])
    assert other_route != LANGUAGES["grammar"]
    assert other_version != LANGUAGES["grammar"]
    assert LANGUAGES["grammar"] != GRAMMAR


def test_fingerprint_values_are_immutable_and_copy_to_themselves() -> None:
    fingerprint = fingerprint_run(LANGUAGES)
    language = LANGUAGES["grammar"]

    with pytest.raises(AttributeError, match="immutable"):
        fingerprint.__setattr__("_digest", "0")

    with pytest.raises(AttributeError, match="immutable"):
        language.__delattr__("_route")

    assert copy.copy(fingerprint) is fingerprint
    assert copy.deepcopy(language) is language

    with pytest.raises(TypeError):
        cast("dict[str, IndexedLanguage]", fingerprint.languages)["grammar"] = language

    with pytest.raises(TypeError):
        cast("dict[str, str]", fingerprint.packages)["lairs"] = "0"


def test_reprs_name_the_constructor_and_every_field() -> None:
    fingerprint = fingerprint_run(LANGUAGES)
    language = LANGUAGES["grammar"]

    assert repr(fingerprint).startswith(
        f"RunFingerprint({dict(fingerprint.schemas)!r}, "
    )
    assert repr(language) in repr(fingerprint)
    assert repr(language) == (
        f"IndexedLanguage({GRAMMAR.version!r}, {GRAMMAR_SPEC!r}, "
        f"{GRAMMAR.digest!r}, {language.route!r})"
    )


def test_neither_the_module_nor_its_diagnostics_name_a_planning_task() -> None:
    source = (TREE / "lofi" / "spec" / "_fingerprint.py").read_text(encoding="utf-8")

    with pytest.raises(BoundaryMismatchError) as failure:
        require_unchanged(
            fingerprint_run(LANGUAGES), fingerprint_run(renamed("syntax"))
        )

    assert TASK_IDENTIFIER.search(source) is None
    assert TASK_IDENTIFIER.search(str(failure.value)) is None


PLANTED: Final = ("nested", "sequence", "mapping", "optional", "variant")
"""The positions a field is planted at in a model the root reaches."""


def a_hierarchy(planted: str | None) -> type[dx.Model]:
    """Return a root model reaching nested models, with a field planted at one."""

    class Nested(dx.Model, extra="forbid"):
        width: int

        if planted == "nested":
            depth: int = 0

    class Listed(dx.Model, extra="forbid"):
        width: int

        if planted == "sequence":
            depth: int = 0

    class Mapped(dx.Model, extra="forbid"):
        width: int

        if planted == "mapping":
            depth: int = 0

    class Maybe(dx.Model, extra="forbid"):
        width: int

        if planted == "optional":
            depth: int = 0

    class Shape(dx.TaggedUnion, discriminator="kind", extra="forbid"):
        """Declares a shape."""

    class Round(Shape):
        kind: Literal["round"] = "round"
        radius: float = 1.0

        if planted == "variant":
            corners: int = 0

    class Root(dx.Model, extra="forbid"):
        nested: Nested
        listed: tuple[Listed, ...]
        mapped: dict[str, Mapped]
        maybe: Maybe | None
        shape: Shape

    return Root


def test_a_nested_field_leaves_the_root_schema_uri_alone() -> None:
    assert dx.schema_uri(a_hierarchy("nested")) == dx.schema_uri(a_hierarchy(None))


@pytest.mark.parametrize("planted", PLANTED)
def test_a_field_planted_in_any_nested_model_changes_the_fingerprint(
    planted: str,
) -> None:
    before = schema_uris(a_hierarchy(None))
    after = schema_uris(a_hierarchy(planted))
    changed = sorted(name for name in before if before[name] != after[name])

    assert set(before) == set(after)
    assert len(changed) == 1
    assert RunFingerprint(before, LANGUAGES, {}).digest != (
        RunFingerprint(after, LANGUAGES, {}).digest
    )

    with pytest.raises(BoundaryMismatchError) as refused:
        require_unchanged(
            RunFingerprint(before, LANGUAGES, {}), RunFingerprint(after, LANGUAGES, {})
        )

    assert refused.value.members == (f"schemas.{changed[0]}",)


def test_the_run_fingerprint_reaches_models_nested_inside_union_variants() -> None:
    schemas = fingerprint_run(LANGUAGES).schemas

    assert "lofi.spec.trainers.parser.ParserTrainerSpec" in schemas
    assert "lofi.spec.trainers.early_stopping.EarlyStoppingSpec" in schemas
    assert "lofi.data.synthetic.spec.corpus.LengthSpec" in schemas

"""Contract tests for the didactic and panproto substrate.

The tests cover indexed terms and models, canonical encodings, staged lowering,
and persisted-repository integrity.
"""

from __future__ import annotations

import json
import shutil
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final, Literal

import didactic.api as dx
import panproto
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

TREE: Final = Path(__file__).resolve().parents[2]

#: The committed layer-sample repository.
SAMPLE_REPOSITORY: Final = TREE / "tests" / "data" / "layers_sample" / "repository"


# ---------------------------------------------------------------------------
# a dependent expression language
# ---------------------------------------------------------------------------


class ExpressionLanguage:
    """A closed expression family indexed by a type code, with a typed evaluator.

    ``Ty`` is the closed sort of codes, ``El(t)`` the open carrier at each code,
    and ``Expr(t)`` the closed family of expressions denoting an element of
    ``El(t)``. ``evaluate`` eliminates ``Expr(t)`` into ``El(t)``, with ``t`` an
    implicit input inferred from the expression.

    Parameters
    ----------
    name : str
        The theory name. Every sort and operation name is derived from it, so
        two languages can coexist in one process.
    bool_literal_body : Callable[[ExpressionLanguage], dx.Term] | None
        The body of the evaluator's ``BoolLit`` branch. The default returns the
        literal's own value, which inhabits ``El(bool_code())``. A body at the
        wrong index makes the theory ill-typed.
    """

    def __init__(
        self,
        name: str,
        *,
        bool_literal_body: Callable[[ExpressionLanguage], dx.Term] | None = None,
    ) -> None:
        self.theory = dx.GADT(name)
        self.ty = self.theory.sort(f"{name}Ty", closed=True)
        self.el = self.theory.family(f"{name}El", t=self.ty)
        self.expr = self.theory.family(f"{name}Expr", t=self.ty, closed=True)
        self.int_code = self.theory.constructor(f"{name}_int", returns=self.ty)
        self.bool_code = self.theory.constructor(f"{name}_bool", returns=self.ty)
        self.int_value = self.theory.operation(
            f"{name}_int_value", returns=self.el[self.int_code()]
        )
        self.bool_value = self.theory.operation(
            f"{name}_bool_value", returns=self.el[self.bool_code()]
        )
        self.int_literal = self.theory.constructor(
            f"{name}IntLit",
            value=self.el[self.int_code()],
            returns=self.expr[self.int_code()],
        )
        self.bool_literal = self.theory.constructor(
            f"{name}BoolLit",
            value=self.el[self.bool_code()],
            returns=self.expr[self.bool_code()],
        )
        self.evaluate = self.theory.eliminator(
            f"{name}_evaluate",
            t=dx.Implicit(self.ty),
            expression=lambda t: self.expr[t],
            returns=lambda t: self.el[t],
        )
        bool_body: dx.Body = (
            (lambda value: value)
            if bool_literal_body is None
            else (lambda value: bool_literal_body(self))
        )
        self.definition = self.evaluate.define(
            lambda t, expression: dx.match(
                expression,
                **{
                    self.int_literal.name: lambda value: value,
                    self.bool_literal.name: bool_body,
                },
            )
        )


def test_dependent_motive_family_compiles_and_evaluates_at_each_index() -> None:
    language = ExpressionLanguage("Motive")
    compiled = language.theory.compile()

    assert compiled.eq_count == 1
    assert language.evaluate.motive == dx.Motive(language.el(dx.Var("t")))
    assert language.evaluate.motive.at(t=language.int_code()) == language.el(
        language.int_code()
    )

    int_term = language.evaluate(language.int_literal(language.int_value()))
    bool_term = language.evaluate(language.bool_literal(language.bool_value()))
    assert language.theory.normalize(int_term) == language.int_value()
    assert language.theory.normalize(bool_term) == language.bool_value()


def test_a_branch_at_the_wrong_index_is_rejected_at_compile() -> None:
    language = ExpressionLanguage(
        "WrongBranch", bool_literal_body=lambda self: self.int_value()
    )

    with pytest.raises(panproto.GatError, match=language.bool_literal.name):
        language.theory.compile()


def test_a_constructor_argument_at_the_wrong_index_is_rejected() -> None:
    language = ExpressionLanguage("WrongArgument")

    with pytest.raises(dx.GADTDeclarationError, match="index mismatch"):
        language.theory.infer_sort(language.int_literal(language.bool_value()))


def test_an_implicit_index_is_inferred_in_an_application() -> None:
    language = ExpressionLanguage("ImplicitApplication")

    assert language.evaluate.explicit_inputs == (language.evaluate.inputs[1],)
    int_term = language.evaluate(language.int_literal(language.int_value()))
    bool_term = language.evaluate(language.bool_literal(language.bool_value()))
    assert language.theory.infer_sort(int_term) == language.el(language.int_code())
    assert language.theory.infer_sort(bool_term) == language.el(language.bool_code())


def test_an_implicit_index_is_inferred_in_a_defining_equation() -> None:
    language = ExpressionLanguage("ImplicitEquation")

    assert language.definition.lhs == language.evaluate(dx.Var("expression"))
    assert "t" not in language.definition.lhs.free_vars()
    panproto.typecheck_theory(language.theory.compile())
    assert language.theory.compile().eq_count == 1


# ---------------------------------------------------------------------------
# a model field selected by an index
# ---------------------------------------------------------------------------

PAYLOAD: Final = dx.Universe("SubstratePayload", count=int, label=str)


class Cell(dx.Model, extra="forbid"):
    """A payload whose Python type is selected by its kind."""

    kind: Literal["count", "label"]
    body: Annotated[int | str, PAYLOAD.at("kind")]


def index_mismatches(
    caught: dx.ValidationError,
) -> tuple[tuple[str | int, ...], ...]:
    """Return the locations of the index-mismatch entries in a validation error.

    Parameters
    ----------
    caught : dx.ValidationError
        The error a construction or update raised.

    Returns
    -------
    tuple[tuple[str | int, ...], ...]
        One location per entry whose type is an index mismatch, in order.
    """
    return tuple(
        entry.loc for entry in caught.entries if entry.type == "index_mismatch"
    )


def test_a_payload_is_validated_at_the_carrier_its_index_selects() -> None:
    assert Cell(kind="count", body=3).body == 3
    assert Cell(kind="label", body="three").body == "three"

    with pytest.raises(dx.ValidationError) as caught:
        Cell(kind="count", body="three")

    assert index_mismatches(caught.value) == (("body",),)


def test_the_indexed_relation_is_preserved_through_json_validation() -> None:
    cell = Cell(kind="label", body="three")

    assert Cell.model_validate_json(cell.model_dump_json()) == cell

    swapped = json.dumps({"kind": "count", "body": "three"})

    with pytest.raises(dx.ValidationError) as caught:
        Cell.model_validate_json(swapped)

    assert index_mismatches(caught.value) == (("body",),)


def test_with_revalidates_selector_and_payload_as_one_change() -> None:
    cell = Cell(kind="count", body=3)

    with pytest.raises(dx.ValidationError) as selector_alone:
        cell.with_(kind="label")

    with pytest.raises(dx.ValidationError) as payload_alone:
        cell.with_(body="three")

    assert index_mismatches(selector_alone.value) == (("body",),)
    assert index_mismatches(payload_alone.value) == (("body",),)
    assert cell.with_(kind="label", body="three") == Cell(kind="label", body="three")
    assert cell == Cell(kind="count", body=3)


# ---------------------------------------------------------------------------
# canonical theory bytes
# ---------------------------------------------------------------------------


def canonical_theory_bytes(theory: panproto.Theory) -> bytes:
    """Return the one byte encoding LoFI gives a theory.

    Parameters
    ----------
    theory : panproto.Theory
        A compiled theory.

    Returns
    -------
    bytes
        The theory's dictionary form as JSON with keys sorted, no whitespace,
        every non-ASCII character escaped, encoded as UTF-8. Two theories with
        the same declarations produce the same bytes.
    """
    document = json.dumps(
        theory.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return document.encode("utf-8")


def theory_digest(theory: panproto.Theory) -> str:
    """Return the SHA-256 of a theory's canonical bytes as lowercase hexadecimal.

    Parameters
    ----------
    theory : panproto.Theory
        A compiled theory.
    """
    return sha256(canonical_theory_bytes(theory)).hexdigest()


def reload_theory(encoded: bytes) -> panproto.Theory:
    """Return the theory a canonical byte string encodes.

    Parameters
    ----------
    encoded : bytes
        Bytes produced by :func:`canonical_theory_bytes`.
    """
    return panproto.create_theory(json.loads(encoded.decode("utf-8")))


class Plain(dx.Model, extra="forbid"):
    """A model with one scalar field and no index."""

    name: str


def test_canonical_theory_bytes_round_trip() -> None:
    theory = ExpressionLanguage("Canonical").theory.compile()
    encoded = canonical_theory_bytes(theory)

    reloaded = reload_theory(encoded)
    assert canonical_theory_bytes(reloaded) == encoded
    assert reloaded.to_dict() == theory.to_dict()
    assert canonical_theory_bytes(ExpressionLanguage("Canonical").theory.compile()) == (
        encoded
    )

    renamed = reload_theory(encoded.replace(b'"Canonical"', b'"Renamed"', 1))
    assert renamed.name == "Renamed"
    assert renamed.sort_count == theory.sort_count
    assert canonical_theory_bytes(renamed) != encoded


def test_the_digest_covers_the_canonical_bytes_and_nothing_else() -> None:
    theory = ExpressionLanguage("Digested").theory.compile()
    encoded = canonical_theory_bytes(theory)

    assert theory_digest(theory) == sha256(encoded).hexdigest()
    assert theory_digest(reload_theory(encoded)) == theory_digest(theory)

    renamed = ExpressionLanguage("Digestee").theory.compile()
    assert theory_digest(renamed) != theory_digest(theory)


def test_reconstruction_from_an_indexed_theory_alone_fails_closed() -> None:
    rebuilt = dx.model_from_theory(Plain.__theory__)
    assert rebuilt(name="x").name == "x"

    with pytest.raises(NotImplementedError, match="indexed"):
        dx.model_from_theory(Cell.__theory__)


# ---------------------------------------------------------------------------
# checked lowering
# ---------------------------------------------------------------------------


class RecordingLowerer:
    """A lowerer over strings that records the order its stages run in.

    ``check`` splits the source into words, ``lower`` joins them with hyphens,
    and ``validate`` rejects an empty target. Each stage appends its name to
    ``stages`` when it runs.

    Parameters
    ----------
    source_version : str
        The source version the lowerer accepts.
    target_version : str
        The target version the lowerer produces.
    """

    def __init__(self, source_version: str, target_version: str) -> None:
        self._route = dx.LoweringRoute(source_version, target_version)
        self.stages: list[str] = []

    @property
    def route(self) -> dx.LoweringRoute:
        """Return the one version pair this lowerer implements."""
        return self._route

    def check(self, source: str, /) -> tuple[str, ...]:
        """Split the source into words, refusing a blank source.

        Parameters
        ----------
        source : str
            The text to lower.

        Raises
        ------
        ValueError
            If the source has no words.
        """
        self.stages.append("check")
        words = tuple(source.split())

        if not words:
            msg = "a blank source has nothing to lower"
            raise ValueError(msg)

        return words

    def lower(self, checked: tuple[str, ...], /) -> str:
        """Join the checked words with hyphens.

        Parameters
        ----------
        checked : tuple[str, ...]
            The words ``check`` returned.
        """
        self.stages.append("lower")
        return "-".join(checked)

    def validate(self, target: str, /) -> None:
        """Reject an empty target.

        Parameters
        ----------
        target : str
            The text ``lower`` returned.

        Raises
        ------
        ValueError
            If the target is empty.
        """
        self.stages.append("validate")

        if not target:
            msg = "an empty target is malformed"
            raise ValueError(msg)


def test_lower_checked_runs_check_lower_and_validate_in_order() -> None:
    lowerer = RecordingLowerer("src-1", "core-2")

    assert dx.supports_lowering(
        lowerer, source_version="src-1", target_version="core-2"
    )
    lowered = dx.lower_checked(
        "two words", lowerer, source_version="src-1", target_version="core-2"
    )

    assert lowered == "two-words"
    assert lowerer.stages == ["check", "lower", "validate"]


def test_lower_checked_runs_no_stage_for_a_wrong_route() -> None:
    lowerer = RecordingLowerer("src-1", "core-2")

    assert not dx.supports_lowering(
        lowerer, source_version="src-1", target_version="core-3"
    )

    with pytest.raises(dx.UnsupportedLoweringRouteError, match="core-3"):
        dx.lower_checked(
            "two words", lowerer, source_version="src-1", target_version="core-3"
        )

    with pytest.raises(dx.UnsupportedLoweringRouteError, match="src-0"):
        dx.lower_checked(
            "two words", lowerer, source_version="src-0", target_version="core-2"
        )

    assert lowerer.stages == []


def test_a_failing_check_runs_no_later_stage() -> None:
    lowerer = RecordingLowerer("src-1", "core-2")

    with pytest.raises(ValueError, match="blank"):
        dx.lower_checked(
            "   ", lowerer, source_version="src-1", target_version="core-2"
        )

    assert lowerer.stages == ["check"]


# ---------------------------------------------------------------------------
# persisted repository objects
# ---------------------------------------------------------------------------


def object_path(store: Path, identifier: str) -> Path:
    """Return the file a content-addressed object is stored in.

    Parameters
    ----------
    store : Path
        The ``.panproto`` directory.
    identifier : str
        The object's identifier as lowercase hexadecimal.
    """
    return store / "objects" / identifier[:2] / identifier[2:]


def commit_field_count(encoded: bytes) -> int:
    """Return how many fields a stored commit object has.

    Parameters
    ----------
    encoded : bytes
        The bytes of a commit object file: a one-entry MessagePack map from the
        fixed string ``Commit`` to an array of the commit's fields.

    Raises
    ------
    ValueError
        If the bytes do not begin with that envelope followed by an array
        header.
    """
    tag = b"Commit"
    envelope = bytes((0x81, 0xA0 | len(tag))) + tag
    msg = "the object is not a commit in the one-entry map envelope"

    if not encoded.startswith(envelope) or len(encoded) == len(envelope):
        raise ValueError(msg)

    header = encoded[len(envelope)]
    length = encoded[len(envelope) + 1 :]

    if header & 0xF0 == 0x90:
        return header & 0x0F

    width = {0xDC: 2, 0xDD: 4}.get(header)

    if width is None or len(length) < width:
        raise ValueError(msg)

    return int.from_bytes(length[:width], "big")


def test_commit_field_count_reads_each_array_header() -> None:
    envelope = b"\x81\xa6Commit"

    assert commit_field_count(envelope + b"\x90") == 0
    assert commit_field_count(envelope + b"\x9f") == 15
    assert commit_field_count(envelope + b"\xdc\x00\x10") == 16
    assert commit_field_count(envelope + b"\xdd\x00\x01\x00\x00") == 65536

    for malformed in (
        b"\x81\xa6Commi",
        envelope,
        envelope + b"\x81",
        envelope + b"\xdc\x00",
        b"\x82\xa6Commit\x9f",
    ):
        with pytest.raises(ValueError, match="envelope"):
            commit_field_count(malformed)


@pytest.fixture
def sample_repository(tmp_path: Path) -> Path:
    """Copy the sample repository so a test can alter it.

    Parameters
    ----------
    tmp_path : Path
    """
    copied = tmp_path / "sample"
    shutil.copytree(SAMPLE_REPOSITORY, copied)
    return copied


def test_sample_objects_keep_their_stored_identifiers(sample_repository: Path) -> None:
    store = sample_repository / ".panproto"
    recorded = (store / "refs" / "heads" / "main").read_text(encoding="utf-8").strip()
    repository = dx.Repository.open(sample_repository)

    assert repository.head() == recorded
    assert repository.resolve_ref("main") == recorded
    assert repository.resolve_ref("HEAD") == recorded

    (row,) = repository.log()
    schema_id = row["schema_id"]
    data_ids = row["data_ids"]
    assert isinstance(schema_id, str)
    assert isinstance(data_ids, list)
    assert object_path(store, schema_id).is_file()
    assert all(
        isinstance(data_id, str) and object_path(store, data_id).is_file()
        for data_id in data_ids
    )

    datasets = repository.data_at("HEAD")
    assert len(datasets) == len(data_ids) > 0
    assert all(object_path(store, dataset.schema_id).is_file() for dataset in datasets)
    assert all(dataset.record_count > 0 for dataset in datasets)


def test_a_substituted_sample_object_is_refused(sample_repository: Path) -> None:
    store = sample_repository / ".panproto"
    recorded = (store / "refs" / "heads" / "main").read_text(encoding="utf-8").strip()
    commit = object_path(store, recorded)
    (row,) = dx.Repository.open(sample_repository).log()
    message = row["message"]
    assert isinstance(message, str)

    stored = bytearray(commit.read_bytes())
    at = stored.find(message.encode("utf-8"))
    assert at >= 0
    stored[at] ^= 0x01
    commit.write_bytes(bytes(stored))

    repository = dx.Repository.open(sample_repository)

    with pytest.raises(panproto.VcsError, match=recorded):
        repository.log()

    with pytest.raises(panproto.VcsError, match=recorded):
        repository.data_at("HEAD")

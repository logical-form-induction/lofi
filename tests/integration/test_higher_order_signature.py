"""Declared typed constants determine logical-form syntax and typing."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import pytest
import yaml

from lofi.data.synthetic.spec.interpretation import (
    InterpretationSpec,
    lower_interpretation,
)
from lofi.lf import LogicalFormParseError
from lofi.run import MANIFEST_NAME, ManifestStatus
from lofi.run._entrypoint import main
from lofi.spec import (
    ConstructorSpec,
    InfixConstructorNotationSpec,
    RunSpecCodec,
    config_search_path,
)
from lofi.spec.grammar import GrammarDecl, GrammarIso
from lofi.spec.language import (
    AtomicTypeSpec,
    BinderNotationSpec,
    BinderSpec,
    ConstantSpec,
    InfixNotationSpec,
    LanguageDecl,
    LanguageIso,
    LogicSpec,
    PrefixNotationSpec,
)

if TYPE_CHECKING:
    from lofi._types import JsonValue
    from lofi.data.synthetic import InterpretationFunction


def _language() -> LanguageDecl:
    return LanguageDecl(
        name="higher-order",
        atomic_types=(
            AtomicTypeSpec(name="e"),
            AtomicTypeSpec(name="v", kind="event"),
            AtomicTypeSpec(name="s", kind="world"),
            AtomicTypeSpec(name="t", kind="proposition"),
        ),
        constructors=(
            ConstructorSpec(
                name="->",
                kind=("Ty", "Ty", "Ty"),
                notation=InfixConstructorNotationSpec(
                    spelling="->", precedence=1, associativity="right"
                ),
            ),
        ),
        constants=(
            ConstantSpec(name="john", type="e"),
            ConstantSpec(name="sleep", type="e -> t"),
            ConstantSpec(name="agent", type="v -> e -> t"),
            ConstantSpec(
                name="and",
                type="a -> a -> a",
                quantifies=("a",),
                classes=("sentential", "conjunction"),
                notation=InfixNotationSpec(
                    ascii="&", unicode="∧", precedence=4, associativity="left"
                ),
            ),
            ConstantSpec(
                name="most",
                type="(e -> t) -> (e -> t) -> t",
                classes=("quantifier",),
                notation=BinderNotationSpec(ascii="most", unicode="most", bodies=2),
            ),
            ConstantSpec(
                name="iota",
                type="(e -> t) -> e",
                classes=("description",),
                notation=BinderNotationSpec(
                    ascii="iota", unicode="\N{GREEK SMALL LETTER IOTA}", bodies=1
                ),
            ),
            ConstantSpec(
                name="box",
                type="(s -> t) -> (s -> t)",
                classes=("modal",),
                notation=PrefixNotationSpec(ascii="box", unicode="□", precedence=6),
            ),
            ConstantSpec(name="sleep_at", type="s -> e -> t"),
        ),
        binders=BinderSpec(style="church"),
        logic=LogicSpec(order="higher"),
    )


def test_declared_notation_lowers_to_constants_applied_to_abstractions() -> None:
    language = LanguageIso().forward(_language())

    for source in (
        "iota x. sleep(x)",
        "box \\w:s. sleep_at(w)(john)",
        "most x.(sleep(x), sleep(x))",
        "sleep(john) & sleep(john)",
    ):
        term = language.read(source)
        written = language.print(term)
        rebuilt = language.read(written)

        assert rebuilt.key() == term.key()

    most = language.read("most x.(sleep(x), sleep(x))")
    head, arguments = language.spine(most)
    assert language.store.payload_name(head) == "most"
    assert len(arguments) == 2
    assert all(argument.op.name == "lam" for argument in arguments)


def test_open_terms_round_trip_and_ty2_meanings_type() -> None:
    language = LanguageIso().forward(_language())
    open_term = language.read("sleep(x)")
    intensional = language.read(r"\w:s. sleep_at(w)(john)")

    assert language.read(language.print(open_term)).key() == open_term.key()
    assert language.type_of(intensional).key() == language.read_type("s -> t").key()
    assert language.profile.world_types == {"s"}
    assert language.profile.fragment.connectives == {"and"}


def test_declared_higher_order_schemes_are_reachable() -> None:
    language = LanguageIso().forward(_language())

    assert (
        language.scheme("agent").body.key() == language.read_type("v -> e -> t").key()
    )
    assert (
        language.scheme("most").body.key()
        == language.read_type("(e -> t) -> (e -> t) -> t").key()
    )
    assert language.scheme("and").quantified


ROOT: Final = Path(__file__).resolve().parents[2]
CONFIGS: Final = ROOT / "lofi/configs"
PARSER: Final = CONFIGS / "run/synthetic-parser.yaml"
INTERPRETATION: Final = (
    Path(__file__).resolve().parent / "fixtures/higher-order-interpretation.yaml"
)

type Document = dict[str, JsonValue]


def _interpretation() -> Document:
    return cast("Document", yaml.safe_load(INTERPRETATION.read_text(encoding="utf-8")))


def _language_of(document: Document) -> Document:
    return cast("Document", document["language"])


def _interpretation_of(document: Document) -> Document:
    return cast("Document", document["interpretation"])


def _lexicon_of(document: Document) -> Document:
    return cast("Document", document["lexicon"])


def _constants_of(document: Document) -> list[Document]:
    return cast("list[Document]", document["constants"])


def _parser_document() -> Document:
    """Return the composed packaged parser run as mutable JSON data."""
    run = RunSpecCodec(search_path=config_search_path(ROOT))(PARSER)

    return cast("Document", json.loads(run.model_dump_json()))


def _dataset(document: Document) -> Document:
    data = cast("Document", document["data"])
    datasets = cast("Document", data["datasets"])
    train = cast("Document", datasets["train"])

    return cast("Document", train["dataset"])


def _run_document(path: Path, declaration: Document) -> Path:
    document: Document = {
        "defaults": ["run/synthetic-parser"],
        "data": {
            "datasets": {
                "train": {
                    "dataset": {
                        "language": _language_of(declaration),
                        "interpretation": _interpretation_of(declaration),
                        "grammar": {"lexicon": _lexicon_of(declaration)},
                    }
                }
            }
        },
    }
    path.write_text(
        yaml.safe_dump(document, allow_unicode=True), encoding="utf-8", newline="\n"
    )
    return path


def _built(declaration: Document) -> InterpretationFunction:
    document = _parser_document()
    grammar_document = cast("Document", _dataset(document)["grammar"])
    grammar_document["lexicon"] = _lexicon_of(declaration)
    grammar = GrammarDecl.model_validate_json(json.dumps(grammar_document))
    language = LanguageDecl.model_validate_json(json.dumps(_language_of(declaration)))
    declared = InterpretationSpec.model_validate_json(
        json.dumps(_interpretation_of(declaration))
    )
    _, built = lower_interpretation(
        grammar, language, declared, GrammarIso(), LanguageIso()
    )

    return built


def test_a_document_writes_iota_box_and_most_and_trains_a_tiny_point(
    tmp_path: Path,
) -> None:
    declaration = _interpretation()
    built = _built(declaration)
    language = built.language

    for text in ("iota x:e. dog(x)", "most z.(dog(z), sleep(z))"):
        term = language.read(text)
        head, arguments = language.spine(term)

        assert head.op.payload_sort == "constant"
        assert arguments
        assert all(argument.op.name == language.calculus.lam for argument in arguments)
        assert language.read(language.print(term)).key() == term.key()

    boxed = language.read(r"box \w:s. sleep_at(w)(y)")
    head, arguments = language.spine(boxed)
    assert language.store.payload_name(head) == "box"
    assert arguments[0].op.name == language.calculus.lam
    assert language.read(language.print(boxed)).key() == boxed.key()
    assert (
        language.read("dog(john) && sleep(john)").key()
        == language.read("dog(john) ∧ sleep(john)").key()
    )

    for undeclared in (
        "dog(john) & sleep(john)",
        "dog(john) | sleep(john)",
        "dog(john) -> sleep(john)",
        "~dog(john)",
        "forall x. dog(x)",
        "every x. dog(x)",
        "exists x. dog(x)",
        "the x. dog(x)",
    ):
        with pytest.raises(LogicalFormParseError):
            language.read(undeclared)

    config = _run_document(tmp_path / "higher-order.yaml", declaration)
    root = tmp_path / "root"
    argv = ["run", "--config", str(config), "--experiment-seed", "5"]

    assert main([*argv, "--root", str(root)]) == 0

    (directory,) = (root / "runs").glob("[0-9a-f]*")
    manifest = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))

    assert manifest["status"] == ManifestStatus.COMPLETED.value


def test_a_document_infers_left_constants_and_types_declared_ones() -> None:
    declaration = _interpretation()
    language = _built(declaration).language

    assert language.print_type(language.scheme("snore").body) == "e -> t"
    assert (
        language.print_type(language.scheme("most").body) == "(e -> t) -> (e -> t) -> t"
    )
    assert language.print_type(language.scheme("agent").body) == "v -> e -> t"
    assert language.print_type(language.scheme("and").body) == "a -> a -> a"
    assert language.scheme("and").quantified

    for constant in _constants_of(_language_of(declaration)):
        if constant["name"] == "and":
            constant["type"] = "infer"
            constant["quantifies"] = []

    inferred = _built(declaration).language

    assert inferred.print_type(inferred.scheme("and").body) == "t -> t -> t"
    assert inferred.print_type(inferred.scheme("snore").body) == "e -> t"

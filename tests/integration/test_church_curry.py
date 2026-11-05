"""Church and Curry languages lower through the same one-store boundary."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, cast

import pytest
import yaml

from lofi.run._entrypoint import main
from lofi.spec import (
    ConstructorSpec,
    DeclarationError,
    InfixConstructorNotationSpec,
)
from lofi.spec.language import (
    AnnotationError,
    AtomicTypeSpec,
    BinderSpec,
    ConstantSpec,
    LanguageDecl,
    LanguageIso,
)

if TYPE_CHECKING:
    from lofi._types import JsonValue

type Document = dict[str, JsonValue]

INTERPRETATION: Final = (
    Path(__file__).resolve().parent / "fixtures/higher-order-interpretation.yaml"
)


def _language(style: Literal["church", "curry"]) -> LanguageDecl:
    return LanguageDecl(
        name=f"lambda-{style}",
        atomic_types=(
            AtomicTypeSpec(name="e"),
            AtomicTypeSpec(name="v"),
            AtomicTypeSpec(name="t"),
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
        constants=(ConstantSpec(name="sleep", type="e -> t"),),
        binders=BinderSpec(style=style),
    )


@pytest.mark.parametrize("style", ["church", "curry"])
def test_binder_style_lowers_to_one_store(style: Literal["church", "curry"]) -> None:
    language = LanguageIso().forward(_language(style))
    term = language.read(r"\x:e. sleep(x)")

    assert language.checker.store is language.store
    assert language.store.contains(term)
    assert term.op.name == "lam"

    if style == "church":
        assert len(term.args) == 2
        assert language.store.payload_name(term.args[0]) == "e"
    else:
        assert len(term.args) == 1


@pytest.mark.parametrize("style", ["church", "curry"])
def test_disagreeing_surface_annotation_is_located(
    style: Literal["church", "curry"],
) -> None:
    language = LanguageIso().forward(_language(style))

    with pytest.raises(AnnotationError) as refused:
        language.read(r"\x:v. sleep(x)")

    assert refused.value.residual
    assert {constraint.kind for constraint in refused.value.residual} == {"annotation"}
    assert refused.value.residual[0].path == ()


def _interpretation(style: Literal["church", "curry"]) -> Document:
    interpretation = cast(
        "Document", yaml.safe_load(INTERPRETATION.read_text(encoding="utf-8"))
    )
    language = cast("Document", interpretation["language"])
    language["binders"] = {"style": style}
    return interpretation


def _document_language(style: Literal["church", "curry"]) -> LanguageDecl:
    language = cast("Document", _interpretation(style)["language"])
    constants = cast("list[JsonValue]", language["constants"])
    language["constants"] = [
        constant
        for constant in constants
        if cast("Document", constant)["type"] != "infer"
    ] + [{"name": "sleep", "type": "e -> t"}]
    return LanguageDecl.model_validate_json(json.dumps(language))


@pytest.mark.parametrize("style", ["church", "curry"])
def test_documents_differing_only_in_binder_style_lower_to_one_store(
    style: Literal["church", "curry"],
) -> None:
    language = LanguageIso().forward(_document_language(style))
    term = language.read(r"\x:e. sleep(x)")

    assert language.store.contains(term)
    assert len(term.args) == (2 if style == "church" else 1)

    if style == "church":
        assert term.args[0] is language.read_type("e")

    with pytest.raises(AnnotationError) as refused:
        language.read(r"\x:v. sleep(x)")

    assert {item.kind for item in refused.value.residual} == {"annotation"}


def test_a_disagreeing_annotation_in_a_run_document_refuses_before_a_run(
    tmp_path: Path,
) -> None:
    interpretation = _interpretation("curry")
    source = cast("Document", interpretation["lexicon"])
    meanings = cast("list[Document]", source["entries"])
    meanings[1]["meaning"] = r"\x:v. sleep(x)"
    meanings[5]["meaning"] = r"\x. snore(x)"
    document: Document = {
        "defaults": ["run/synthetic-parser"],
        "data": {
            "datasets": {
                "train": {
                    "dataset": {
                        "language": cast("Document", interpretation["language"]),
                        "interpretation": cast(
                            "Document", interpretation["interpretation"]
                        ),
                        "grammar": {"lexicon": source},
                    }
                }
            }
        },
    }
    config = tmp_path / "refused.yaml"
    config.write_text(
        yaml.safe_dump(document, allow_unicode=True), encoding="utf-8", newline="\n"
    )
    root = tmp_path / "root"

    with pytest.raises(DeclarationError, match="annotation disagrees"):
        main(
            [
                "run",
                "--config",
                str(config),
                "--experiment-seed",
                "5",
                "--root",
                str(root),
            ]
        )

    assert not root.exists()

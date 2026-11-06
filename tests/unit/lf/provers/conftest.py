from importlib.metadata import EntryPoint
from itertools import count
from typing import TYPE_CHECKING

import pytest
from tests.stand_ins import stand_in
from tests.support.installed import install

from lofi.spec import ConstructorSpec, InfixConstructorNotationSpec
from lofi.spec.language import (
    AtomicTypeSpec,
    BinderNotationSpec,
    BinderSpec,
    ConstantSpec,
    InfixNotationSpec,
    LanguageAssembly,
    LanguageDecl,
    LanguageIso,
    LogicSpec,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@pytest.fixture(autouse=True)
def _logical_constant_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Expose the packaged semantics without relying on editable metadata."""
    install(
        monkeypatch,
        "logical_constant",
        (
            EntryPoint(
                name="most",
                value="lofi.lf._semantics:MOST",
                group="lofi.logical_constant",
            ),
        ),
    )


def _declaration() -> LanguageDecl:
    return LanguageDecl(
        name="first-order-entailment",
        atomic_types=(
            AtomicTypeSpec(name="e"),
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
            ConstantSpec(name="dog", type="e -> t"),
            ConstantSpec(name="mammal", type="e -> t"),
            ConstantSpec(name="rex", type="e"),
            ConstantSpec(
                name="both",
                type="t -> t -> t",
                classes=("sentential", "conjunction"),
                notation=InfixNotationSpec(
                    ascii="&", unicode="∧", precedence=4, associativity="left"
                ),
            ),
            ConstantSpec(
                name="implies",
                type="t -> t -> t",
                classes=("sentential", "conditional"),
                notation=InfixNotationSpec(
                    ascii="=>", unicode="→", precedence=2, associativity="right"
                ),
            ),
            ConstantSpec(
                name="all",
                type="(e -> t) -> t",
                classes=("quantifier", "universal"),
                notation=BinderNotationSpec(ascii="all", unicode="∀", bodies=1),
            ),
        ),
        binders=BinderSpec(style="curry"),
    )


@pytest.fixture
def first_order_language() -> LanguageAssembly:
    return LanguageIso().forward(_declaration())


@pytest.fixture
def intensional_language() -> LanguageAssembly:
    declaration = _declaration()
    return LanguageIso().forward(
        declaration.with_(
            name="intensional-entailment",
            atomic_types=(
                *declaration.atomic_types,
                AtomicTypeSpec(name="s", kind="world"),
            ),
            logic=LogicSpec(),
        )
    )


@pytest.fixture
def script(tmp_path: Path) -> Callable[[str], str]:
    """Return a factory writing Python programs that stand in for provers."""
    written = count()

    def write(program: str) -> str:
        return str(stand_in(tmp_path / f"prover-{next(written)}", program))

    return write


@pytest.fixture
def python_program(tmp_path: Path) -> Callable[[str], str]:
    """Write prover fixture programs for direct execution by the interpreter."""
    written = count()

    def write(program: str) -> str:
        source = tmp_path / f"prover-{next(written)}.py"
        source.write_text(program, encoding="utf-8", newline="\n")
        return str(source)

    return write

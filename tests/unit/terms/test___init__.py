import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Final

from lofi import _native, terms
from lofi._native import _compile_grammar, _CompiledGrammar, _CompiledRule
from lofi.spec.datasets import ccgbank_grammar
from lofi.spec.grammar import GrammarIso
from lofi.spec.language import registered_language
from lofi.terms import (
    Calculus,
    Encoding,
    FeatureErasure,
    Identity,
    Nominal,
    Operator,
    OperatorIndex,
    Product,
    RegionStateError,
    SExpSyntax,
    Signature,
    SignatureStableMatching,
    SortError,
    TargetSet,
    TermCodec,
    TextDepthError,
    TextParseError,
    TreeEditDistance,
    _codec,
    _distance,
    _frontier,
    _matching,
    _neighborhood,
    _normalize,
    _path,
    _pattern,
    _ports,
    _present,
    _reach,
    _rules,
    _targets,
    _text,
    _values,
)

TREE: Final = Path(__file__).resolve().parents[3]

CLOSURE: Final = """
import json
import sys

before = set(sys.modules)

import lofi.terms

print(json.dumps(sorted(set(sys.modules) - before)))
"""


def test_the_package_re_exports_its_modules_and_nothing_else() -> None:
    source = (
        dict.fromkeys(_ports.__all__, _ports)
        | dict.fromkeys(_frontier.__all__, _frontier)
        | dict.fromkeys(_values.__all__, _values)
        | dict.fromkeys(_targets.__all__, _targets)
        | dict.fromkeys(_text.__all__, _text)
        | dict.fromkeys(_codec.__all__, _codec)
        | dict.fromkeys(_distance.__all__, _distance)
        | dict.fromkeys(_matching.__all__, _matching)
        | dict.fromkeys(_neighborhood.__all__, _neighborhood)
        | dict.fromkeys(_normalize.__all__, _normalize)
        | dict.fromkeys(_path.__all__, _path)
        | dict.fromkeys(_pattern.__all__, _pattern)
        | dict.fromkeys(_present.__all__, _present)
        | dict.fromkeys(_reach.__all__, _reach)
        | dict.fromkeys(_rules.__all__, _rules)
    )

    native = {
        "ArityError",
        "BindingAtomCountError",
        "BindingShapeError",
        "DuplicateNameError",
        "EncodingOverflowError",
        "EncodingShapeError",
        "ErasureError",
        "ForeignTermError",
        "NonTerminatingError",
        "NotLocallyClosedError",
        "PayloadError",
        "RegionStateError",
        "RepresentativeKernel",
        "ReservedDeclarationError",
        "SortError",
        "SortView",
        "StoreFullError",
        "Term",
        "TermError",
        "TermStore",
        "UndeclaredSortError",
        "UnknownConstantError",
        "UnknownOperatorError",
        "VacuousBindingError",
        "abstract",
        "close_binder",
        "instantiate",
        "open_binder",
        "substitute",
    }

    assert sorted(terms.__all__) == sorted((*source, *native))

    for name, module in source.items():
        assert getattr(terms, name) is getattr(module, name)

    for name in native:
        assert getattr(terms, name) is getattr(_native, name)


def test_the_re_exported_names_are_usable_through_the_package() -> None:
    root = Encoding(
        symbols=(4,),
        parents=(Encoding.NO_PARENT,),
        depths=(0,),
        occupied=(True,),
    )

    assert root.length == 1
    assert Calculus("lam", "eval").app == "eval"
    assert (
        SortError(operator="fwd", position=0, expected="Ty", received="Lf").received
        == "Lf"
    )
    assert Identity[int]().key(root.width) == 1
    product = Product(Identity[int](), Identity[int]())
    assert product.equivalent(1, 1)
    assert not product.equivalent(1, 2)
    assert "FeatureErasure" in terms.__all__
    assert terms.FeatureErasure is FeatureErasure
    assert "SignatureStableMatching" in terms.__all__
    assert terms.SignatureStableMatching is SignatureStableMatching

    signature = Signature("categories", ("Cat",), (Operator("np", "Cat"),))
    nominal = Nominal(signature.operator(1))
    store = terms.TermStore(signature)
    sexp = SExpSyntax(signature)

    assert nominal.op is signature.operator(1)
    assert sexp.parse(
        store, sexp.render(store, store.op(signature.operator(1)))
    ) is store.op(signature.operator(1))
    assert TextDepthError is terms.TextDepthError
    assert TextParseError is terms.TextParseError
    assert signature.code(signature.operator(1)) == 1
    assert OperatorIndex(signature).is_nullary(1)
    assert (
        TermCodec(OperatorIndex(signature), _codec.PreorderArityLayout(1)).n_symbols
        == 2
    )
    assert TreeEditDistance()(None, None) == 0
    assert TargetSet(("np",), Identity()).categories == ("np",)
    assert (
        RegionStateError(region="sentence", operation="release", state="active").state
        == "active"
    )


def test_a_native_plan_adopts_the_matching_public_signature() -> None:
    language = registered_language("propositional")
    grammar = ccgbank_grammar()
    plan = _compile_grammar(
        language.compile_envelope.canonical_declaration,
        grammar.compile_envelope.canonical_declaration,
    )
    assembly = GrammarIso().forward(grammar, language)
    rules = tuple(plan.rules)
    store = plan.term_store(assembly.grammar.signature)

    assert type(plan) is _CompiledGrammar
    assert rules
    assert all(type(rule) is _CompiledRule for rule in rules)
    assert tuple(rule.identifier for rule in rules) == tuple(range(len(rules)))
    assert plan.signature_fingerprint == assembly.grammar.signature.digest
    assert store.signature is assembly.grammar.signature
    assert "_compile_grammar" not in terms.__all__


def test_importing_the_package_reaches_nothing_but_the_standard_library() -> None:
    finished = subprocess.run(
        [sys.executable, "-"],
        input=CLOSURE,
        capture_output=True,
        encoding="utf-8",
        check=True,
        cwd=TREE,
    )
    reached = json.loads(finished.stdout)
    outside = [
        name
        for name in reached
        if not name.startswith("lofi")
        and name.partition(".")[0] not in sys.stdlib_module_names
    ]

    assert reached
    assert outside == []
    assert "lofi.terms.laws" not in reached


def enumerations_named_by(module: Path) -> list[str]:
    tree = ast.parse(module.read_text(encoding="utf-8"))
    named: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "enum":
            named.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            named.extend(
                alias.name for alias in node.names if alias.name.split(".")[0] == "enum"
            )
        elif isinstance(node, ast.ClassDef):
            named.extend(
                ast.unparse(base)
                for base in node.bases
                if ast.unparse(base).rpartition(".")[2] in {"Enum", "Flag", "IntEnum"}
            )

    return named


def test_no_module_of_the_package_declares_or_imports_an_enumeration() -> None:
    package = Path(terms.__file__).resolve().parent
    modules = sorted(package.glob("*.py"))
    offending = {
        module.name: found
        for module in modules
        if (found := enumerations_named_by(module))
    }

    assert [module.name for module in modules]
    assert offending == {}


def test_the_enumeration_check_reads_a_module_that_declares_one() -> None:
    source = Path(sys.modules["enum"].__file__ or "")

    assert "Enum" in enumerations_named_by(source)

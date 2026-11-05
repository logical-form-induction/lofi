from lofi.lf import Proved, ProverBound, Refuted, TruthTableOracle
from lofi.spec.language import (
    ConstantSpec,
    InfixNotationSpec,
    LanguageAssembly,
    LanguageIso,
    registered_language,
)


def _language() -> LanguageAssembly:
    base = registered_language("propositional")
    constants = tuple(
        constant for constant in base.constants if "conjunction" not in constant.classes
    )
    return LanguageIso().forward(
        base.with_(
            constants=(
                *constants,
                ConstantSpec(
                    name="conjunct",
                    type="t -> t -> t",
                    classes=("sentential", "conjunction"),
                    notation=InfixNotationSpec(
                        ascii="&&",
                        unicode="⋏",
                        precedence=4,
                        associativity="left",
                    ),
                ),
                ConstantSpec(
                    name="xor",
                    type="t -> t -> t",
                    classes=("exclusive-or",),
                    notation=InfixNotationSpec(
                        ascii="xor",
                        unicode="xor",
                        precedence=3,
                        associativity="left",
                    ),
                ),
                ConstantSpec(name="rain", type="t", classes=("predicate",)),
                ConstantSpec(name="wet", type="t", classes=("predicate",)),
            )
        )
    )


def test_oracle_reads_truth_functions_from_declared_constant_classes() -> None:
    language = _language()
    bound = ProverBound(
        "truth-table",
        language.profile.fragment,
    )
    oracle = TruthTableOracle(language.store, language.profile, bound=bound, atoms=12)

    assert isinstance(
        oracle.entails(
            language.read("rain && wet"), language.read("rain"), timeout=1.0
        ),
        Proved,
    )
    assert isinstance(
        oracle.entails(language.read("rain"), language.read("wet"), timeout=1.0),
        Refuted,
    )
    assert isinstance(
        oracle.entails(
            language.read("rain xor wet"),
            language.read("rain | wet"),
            timeout=1.0,
        ),
        Proved,
    )

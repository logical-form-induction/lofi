"""Concrete conformance cases for proof-search ports."""

from typing import TYPE_CHECKING, Final, Protocol, cast, override

import pytest
from tests.conformance.proof import (
    PROOF_CALCULUS_CONFORMANCE_LAWS,
    PROOF_SCHEMA_CONFORMANCE_LAWS,
    STRUCTURAL_ERASURE_CONFORMANCE_LAWS,
    ProofCalculusConformance,
    ProofSchemaConformance,
    StructuralErasureConformance,
)

from lofi._checks import LawViolationError
from lofi.judge.laws import (
    PROOF_CALCULUS_LAWS,
    PROOF_SCHEMA_LAWS,
    STRUCTURAL_ERASURE_LAWS,
    ProofCalculusFixture,
    ProofSchemaFixture,
    StructuralErasureFixture,
    application_and_abstraction_channels_are_exact,
    erasure_matches_structural_cases,
    schemata_are_stable,
)
from lofi.spec import (
    FixedOperatorSpec,
    NodePatternSpec,
    PatternRuleSpec,
    TermPatternSpec,
)
from lofi.spec._rule_laws import LawGrammar
from lofi.terms import (
    BindingSite,
    Calculus,
    MVar,
    Node,
    Operator,
    Signature,
    Term,
    TermStore,
    Vocabulary,
)
from lofi.typecheck import Erasure

if TYPE_CHECKING:
    from lofi.judge import ProofSchema


class NamedLaw(Protocol):
    """Callable with a stable test name."""

    @property
    def __name__(self) -> str:
        """Return the callable name."""


LANGUAGE_SIGNATURE: Final = Signature(
    "proof-port-language",
    ("C", "T"),
    (
        Operator("fwd", "C", arg_sorts=("C", "C")),
        Operator("tx", "T"),
        Operator("ty", "T"),
        Operator("->", "T", arg_sorts=("T", "T")),
    ),
    vocabularies=(Vocabulary("C", ("x", "x[marked]", "y")),),
    constant_cores={"C": {"x[marked]": "x"}},
    open_sorts=("T",),
)
PROOF_SIGNATURE: Final = Signature(
    "proof-port-terms",
    ("P",),
    (
        Operator("app", "P", arg_sorts=("P", "P")),
        Operator("lam", "P", arg_sorts=("P",), binding=(BindingSite((0,), "P"),)),
    ),
)
LANGUAGE_STORE: Final = TermStore(LANGUAGE_SIGNATURE)
PROOF_STORE: Final = TermStore(PROOF_SIGNATURE)
X: Final = LANGUAGE_STORE.const("C", "x")
MARKED_X: Final = LANGUAGE_STORE.const("C", "x[marked]")
Y: Final = LANGUAGE_STORE.const("C", "y")
FWD_X_Y: Final = LANGUAGE_STORE.op(LANGUAGE_SIGNATURE.by_name("fwd"), X, Y)
TYPE_X: Final = LANGUAGE_STORE.op(LANGUAGE_SIGNATURE.by_name("tx"))
TYPE_Y: Final = LANGUAGE_STORE.op(LANGUAGE_SIGNATURE.by_name("ty"))
TYPE_FWD_X_Y: Final = LANGUAGE_STORE.op(
    LANGUAGE_SIGNATURE.by_name("->"), TYPE_Y, TYPE_X
)
TYPE_FWD_IMAGE: Final = LANGUAGE_STORE.op(
    LANGUAGE_SIGNATURE.by_name("->"),
    LANGUAGE_STORE.fvar("T", "a1"),
    LANGUAGE_STORE.fvar("T", "a0"),
)
ERASURE: Final = Erasure(
    LANGUAGE_STORE,
    {"x": TYPE_X, "y": TYPE_Y},
    {"fwd": TYPE_FWD_IMAGE},
    category_sort="C",
    function_type="->",
)
SCHEMATA: Final = (
    (
        (Node("fwd", (MVar("X", "C"), MVar("Y", "C"))), MVar("Y", "C")),
        MVar("X", "C"),
    ),
)
NATIVE_SCHEMATA: Final = (
    (
        (Node("fwd", (MVar("X", "Cat"), MVar("Y", "Cat"))), MVar("Y", "Cat")),
        MVar("X", "Cat"),
    ),
)


class FixedSchema:
    """External fixed proof schema."""

    @property
    def proof_schemata(self) -> tuple[tuple[tuple[Node | MVar, ...], Node | MVar], ...]:
        """Return the declared schema."""
        return SCHEMATA


class LambdaCalculus:
    """External proof calculus backed by one store."""

    @property
    def store(self) -> TermStore:
        """Return the proof store."""
        return PROOF_STORE

    @property
    def operators(self) -> Calculus:
        """Return the binder and application names."""
        return Calculus("lam", "app")

    def hypothesis(self, index: int, /) -> Term:
        """Return a canonical indexed hypothesis."""
        return self.free_variable(f"p{index}")

    def free_variable(self, name: str, /) -> Term:
        """Return a named free variable."""
        return self.store.fvar("P", name)

    def bound_variable(self, index: int, /) -> Term:
        """Return an indexed bound variable."""
        return self.store.bvar("P", index)

    def application(
        self,
        function: Term,
        argument: Term,
        /,
    ) -> Term:
        """Return an application."""
        return self.store.op(PROOF_SIGNATURE.by_name("app"), function, argument)

    def abstraction(self, variable: str, body: Term, /) -> Term:
        """Return an abstraction."""
        return self.store.bind("lam", variable, body)


class TestExternalProofSchema(ProofSchemaConformance):
    """Proof-schema conformance for an external structural implementation."""

    @pytest.fixture
    def proof_schema_fixture(self) -> ProofSchemaFixture:
        """Return the external schema fixture."""
        return ProofSchemaFixture(FixedSchema(), SCHEMATA)


class TestNativeProofSchema(ProofSchemaConformance):
    """Proof-schema conformance for a native pattern rule."""

    @pytest.fixture
    def proof_schema_fixture(self) -> ProofSchemaFixture:
        """Return the native schema fixture."""
        rule = cast(
            "ProofSchema",
            LawGrammar("direct").rule(
                PatternRuleSpec(
                    name="forward-application",
                    premises=(
                        NodePatternSpec(
                            operator=FixedOperatorSpec(name="fwd"),
                            arguments=(
                                TermPatternSpec(name="X", sort="Cat"),
                                TermPatternSpec(name="Y", sort="Cat"),
                            ),
                        ),
                        TermPatternSpec(name="Y", sort="Cat"),
                    ),
                    conclusion=TermPatternSpec(name="X", sort="Cat"),
                )
            ),
        )
        return ProofSchemaFixture(rule, NATIVE_SCHEMATA)


class TestNativeStructuralErasure(StructuralErasureConformance):
    """Structural-erasure conformance for the native erasure."""

    @pytest.fixture
    def structural_erasure_fixture(
        self,
    ) -> StructuralErasureFixture:
        """Return atom, feature, and constructor cases."""
        return StructuralErasureFixture(
            ERASURE,
            ((X, TYPE_X), (MARKED_X, TYPE_X), (FWD_X_Y, TYPE_FWD_X_Y)),
        )


class TestExternalProofCalculus(ProofCalculusConformance):
    """Proof-calculus conformance for an external implementation."""

    @pytest.fixture
    def proof_calculus_fixture(self) -> ProofCalculusFixture:
        """Return a three-hypothesis calculus fixture."""
        return ProofCalculusFixture(LambdaCalculus(), (0, 1, 2))


class ChangingSchema(FixedSchema):
    """Schema whose order changes between reads."""

    def __init__(self) -> None:
        self._calls = 0

    @property
    @override
    def proof_schemata(self) -> tuple[tuple[tuple[Node | MVar, ...], Node | MVar], ...]:
        """Return alternating schema order."""
        self._calls += 1
        extra = (((MVar("X", "C"),), MVar("X", "C")),)

        return (*SCHEMATA, *extra) if self._calls % 2 else (*extra, *SCHEMATA)


class WrongErasure:
    """Erasure that maps one constructor to the wrong type."""

    @property
    def store(self) -> TermStore:
        """Return the shared language store."""
        return ERASURE.store

    @property
    def category_sort(self) -> str:
        """Return the category sort."""
        return ERASURE.category_sort

    @property
    def atoms(self) -> dict[str, Term]:
        """Return the atom declarations."""
        return dict(ERASURE.atoms)

    @property
    def constructors(self) -> dict[str, Term]:
        """Return the constructor declarations."""
        return dict(ERASURE.constructors)

    @property
    def function_type(self) -> str:
        """Return the function-type name."""
        return ERASURE.function_type

    @property
    def target_sort(self) -> str:
        """Return the target type sort."""
        return ERASURE.target_sort

    def image(self, constructor: str, arguments: tuple[Term, ...], /) -> Term:
        """Instantiate one constructor image."""
        return ERASURE.image(constructor, arguments)

    def __call__(self, category: Term, /) -> Term:
        """Return the wrong result for the constructor case."""
        return TYPE_Y if category is FWD_X_Y else ERASURE(category)


class WrongApplicationCalculus(LambdaCalculus):
    """Calculus whose application channel returns its argument."""

    @override
    def application(
        self,
        function: Term,
        argument: Term,
        /,
    ) -> Term:
        """Return the argument instead of an application."""
        del function

        return argument


def test_the_schema_laws_reject_unstable_order() -> None:
    with pytest.raises(LawViolationError, match="schemata are stable"):
        schemata_are_stable(ProofSchemaFixture(ChangingSchema(), SCHEMATA))


def test_the_erasure_laws_reject_a_wrong_structural_image() -> None:
    fixture = StructuralErasureFixture(
        WrongErasure(),
        ((X, TYPE_X), (FWD_X_Y, TYPE_FWD_X_Y)),
    )

    with pytest.raises(LawViolationError, match="structural cases"):
        erasure_matches_structural_cases(fixture)


def test_the_calculus_laws_reject_a_wrong_application() -> None:
    fixture = ProofCalculusFixture(WrongApplicationCalculus(), (0, 1, 2))

    with pytest.raises(LawViolationError, match="channels are exact"):
        application_and_abstraction_channels_are_exact(fixture)


@pytest.mark.parametrize(
    ("laws", "conformance_laws"),
    [
        (PROOF_SCHEMA_LAWS, PROOF_SCHEMA_CONFORMANCE_LAWS),
        (STRUCTURAL_ERASURE_LAWS, STRUCTURAL_ERASURE_CONFORMANCE_LAWS),
        (PROOF_CALCULUS_LAWS, PROOF_CALCULUS_CONFORMANCE_LAWS),
    ],
)
def test_each_conformance_method_binds_one_named_law(
    laws: tuple[NamedLaw, ...],
    conformance_laws: tuple[NamedLaw, ...],
) -> None:
    bound = {method.__name__.removeprefix("test_") for method in conformance_laws}

    assert bound == {law.__name__ for law in laws}

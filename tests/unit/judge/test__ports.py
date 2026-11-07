from typing import TYPE_CHECKING, Final

import pytest

from lofi.judge import TypeRaising
from lofi.judge._ports import (
    HOLE,
    CompositionRole,
    Direction,
    Hole,
    NormalFormRole,
    ProofCalculus,
    ProofSchema,
    Rule,
    SchematicSequent,
    Slot,
    StructuralErasure,
)
from lofi.spec import TypeRaisingSpec
from lofi.spec._rule_laws import LawGrammar
from lofi.terms import (
    BindingSite,
    Calculus,
    Operator,
    OpName,
    Signature,
    Term,
    TermRuleKernel,
    TermStore,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from lofi._types import JsonValue
    from lofi.judge._index import RuleProbe
    from lofi.judge._values import RuleRestriction

_DOUBLING: Final[frozenset[Direction]] = frozenset({"forward", "check", "residual"})


def _enumerates_nothing() -> TermRuleKernel:
    """Return a compiled kernel with no forward clauses."""
    rule = LawGrammar("direct").rule(TypeRaisingSpec(view="fwd", arguments=None))

    assert isinstance(rule, TypeRaising)

    return rule.lowering


_ENUMERATES_NOTHING: Final = _enumerates_nothing()
"""A compiled kernel with no forward clauses, for structural fixtures."""


LANGUAGE_SIGNATURE: Final = Signature(
    "port-language",
    ("C", "T"),
    (
        Operator("category-x", "C"),
        Operator("x", "T"),
        Operator("->", "T", arg_sorts=("T", "T")),
    ),
)
PROOF_SIGNATURE: Final = Signature(
    "port-proofs",
    ("P",),
    (
        Operator("app", "P", arg_sorts=("P", "P")),
        Operator("lam", "P", arg_sorts=("P",), binding=(BindingSite((0,), "P"),)),
    ),
)
LANGUAGE_STORE: Final = TermStore(LANGUAGE_SIGNATURE)
PROOF_STORE: Final = TermStore(PROOF_SIGNATURE)


class SchemaShaped:
    """Value with the full proof-schema shape."""

    @property
    def proof_schemata(self) -> tuple[SchematicSequent, ...]:
        """Return no oriented schemata."""
        return ()


class MissingSchemata:
    """Value without proof schemata."""


class ErasureShaped:
    """Value with the full structural-erasure shape."""

    @property
    def store(self) -> TermStore:
        """Return the unified language store."""
        return LANGUAGE_STORE

    @property
    def category_sort(self) -> str:
        """Return the category sort."""
        return "C"

    @property
    def atoms(self) -> Mapping[OpName, Term]:
        """Return the atom map."""
        return {"category-x": LANGUAGE_STORE.op(LANGUAGE_SIGNATURE.by_name("x"))}

    @property
    def constructors(self) -> Mapping[OpName, Term]:
        """Return no constructor declarations."""
        return {}

    @property
    def function_type(self) -> OpName:
        """Return the function-type constructor."""
        return "->"

    @property
    def target_sort(self) -> str:
        """Return the type sort."""
        return "T"

    def image(self, constructor: OpName, arguments: tuple[Term, ...], /) -> Term:
        """Instantiate a constructor image."""
        del constructor, arguments
        return self.atoms["category-x"]

    def __call__(self, category: Term, /) -> Term:
        """Erase the sole category atom."""
        del category

        return self.atoms["category-x"]


class MissingFunctionType:
    """Structural erasure without its function-type constructor."""

    store: Final = LANGUAGE_STORE
    category_sort: Final = "C"
    atoms: Final = {"category-x": LANGUAGE_STORE.op(LANGUAGE_SIGNATURE.by_name("x"))}
    constructors: Final[dict[OpName, Term]] = {}
    target_sort: Final = "T"

    def image(self, constructor: OpName, arguments: tuple[Term, ...], /) -> Term:
        """Instantiate a constructor image."""
        del constructor, arguments
        return self.atoms["category-x"]

    def __call__(self, category: Term, /) -> Term:
        """Erase the sole category atom."""
        del category

        return self.atoms["category-x"]


class CalculusShaped:
    """Value with the full proof-calculus shape."""

    @property
    def store(self) -> TermStore:
        """Return the proof store."""
        return PROOF_STORE

    @property
    def operators(self) -> Calculus:
        """Return the calculus declaration."""
        return Calculus("lam", "app")

    def hypothesis(self, index: int, /) -> Term:
        """Return an indexed hypothesis."""
        return self.free_variable(f"p{index}")

    def free_variable(self, name: str, /) -> Term:
        """Return a free variable."""
        return self.store.fvar("P", name)

    def bound_variable(self, index: int, /) -> Term:
        """Return a bound variable."""
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


class MissingAbstraction:
    """Value with only part of the proof-calculus shape."""

    @property
    def store(self) -> TermStore:
        """Return the proof store."""
        return PROOF_STORE

    @property
    def operators(self) -> Calculus:
        """Return the calculus declaration."""
        return Calculus("lam", "app")

    def hypothesis(self, index: int, /) -> Term:
        """Return an indexed hypothesis."""
        return PROOF_STORE.fvar("P", f"p{index}")

    def free_variable(self, name: str, /) -> Term:
        """Return a free variable."""
        return PROOF_STORE.fvar("P", name)

    def bound_variable(self, index: int, /) -> Term:
        """Return a bound variable."""
        return PROOF_STORE.bvar("P", index)

    def application(
        self,
        function: Term,
        argument: Term,
        /,
    ) -> Term:
        """Return an application."""
        return PROOF_STORE.op(PROOF_SIGNATURE.by_name("app"), function, argument)


class Exact:
    """Integer-identity congruence."""

    @property
    def name(self) -> str:
        """Name the congruence."""
        return "exact"

    @property
    def has_key(self) -> bool:
        """Report that classes have representatives."""
        return True

    def key(self, term: int, /) -> int:
        """Return the term, which is its own representative."""
        return term

    def equivalent(self, left: int, right: int, /) -> bool:
        """Compare integers."""
        return left == right


class Doubling:
    """Integer-doubling rule."""

    @property
    def name(self) -> str:
        """Name the rule."""
        return "doubling"

    @property
    def arity(self) -> int:
        """Take one premise."""
        return 1

    @property
    def directions(self) -> frozenset[Direction]:
        """Compute in all three directions."""
        return _DOUBLING

    @property
    def normal_form_role(self) -> NormalFormRole:
        return NormalFormRole()

    @property
    def spine_order(self) -> int:
        return 0

    @property
    def residual_slots(self) -> frozenset[int]:
        """Solve the sole premise."""
        return frozenset({0})

    @property
    def forward_is_complete(self) -> bool:
        """Enumerate every conclusion the rule admits."""
        return True

    @property
    def matching(self) -> Exact:
        """Read premises under equality."""
        return Exact()

    @property
    def restriction(self) -> RuleRestriction | None:
        """Declare no rule-instance restriction."""
        return None

    @property
    def probe(self) -> RuleProbe | None:
        """Decline premise-head indexing."""
        return None

    @property
    def lowering(self) -> TermRuleKernel:
        """Return an empty native kernel for this structural fixture."""
        return _ENUMERATES_NOTHING

    @property
    def reachability_schemata(self) -> tuple[SchematicSequent, ...]:
        """Declare unrestricted root reachability."""
        return ()

    def apply(self, inputs: tuple[int, ...], /) -> Iterator[int]:
        """Double the premise."""
        if len(inputs) != 1:
            return

        yield inputs[0] * 2

    def admits(self, inputs: tuple[int, ...], output: int, /) -> bool:
        """Decide whether the conclusion is twice the premise."""
        return len(inputs) == 1 and output == inputs[0] * 2

    def residuate(self, output: int, known: tuple[Slot[int], ...], /) -> Iterator[int]:
        """Halve the conclusion when the single premise is unknown."""
        if len(known) != 1 or known[0] is not HOLE or output % 2:
            return

        yield output // 2

    def to_spec(self) -> Mapping[str, JsonValue]:
        """Return the declaration this rule was built from."""
        return {"rule": self.name}


class MissingResidualSlots:
    """Rule-shaped value without residual-slot metadata."""

    @property
    def name(self) -> str:
        """Name the rule."""
        return "uncheckable"

    @property
    def arity(self) -> int:
        """Take one premise."""
        return 1

    @property
    def directions(self) -> frozenset[Direction]:
        """Compute forward only."""
        return frozenset({"forward"})

    @property
    def forward_is_complete(self) -> bool:
        """Enumerate every conclusion the rule admits."""
        return True

    @property
    def matching(self) -> Exact:
        """Read premises under equality."""
        return Exact()

    @property
    def restriction(self) -> RuleRestriction | None:
        """Declare no rule-instance restriction."""
        return None

    @property
    def probe(self) -> RuleProbe | None:
        """Decline premise-head indexing."""
        return None

    def apply(self, inputs: tuple[int, ...], /) -> Iterator[int]:
        """Double the premise."""
        yield inputs[0] * 2

    def admits(self, inputs: tuple[int, ...], output: int, /) -> bool:
        """Decide whether the conclusion is twice the premise."""
        return len(inputs) == 1 and output == inputs[0] * 2

    def residuate(self, output: int, known: tuple[Slot[int], ...], /) -> Iterator[int]:
        """Yield no residuals."""
        del output, known
        return iter(())

    def to_spec(self) -> Mapping[str, JsonValue]:
        """Return the declaration this rule was built from."""
        return {"rule": self.name}


class Successor:
    """Forward-only integer-successor rule."""

    @property
    def name(self) -> str:
        """Name the rule."""
        return "successor"

    @property
    def arity(self) -> int:
        """Take one premise."""
        return 1

    @property
    def directions(self) -> frozenset[Direction]:
        """Declare forward and checking support."""
        return frozenset({"forward", "check"})

    @property
    def normal_form_role(self) -> NormalFormRole:
        return NormalFormRole()

    @property
    def spine_order(self) -> int:
        return 0

    @property
    def residual_slots(self) -> frozenset[int]:
        """Declare no computable residual."""
        return frozenset()

    @property
    def forward_is_complete(self) -> bool:
        """Enumerate every conclusion the rule admits."""
        return True

    @property
    def matching(self) -> Exact:
        """Read premises under equality."""
        return Exact()

    @property
    def restriction(self) -> RuleRestriction | None:
        """Declare no rule-instance restriction."""
        return None

    @property
    def probe(self) -> RuleProbe | None:
        """Decline premise-head indexing."""
        return None

    @property
    def lowering(self) -> TermRuleKernel:
        """Return an empty native kernel for this structural fixture."""
        return _ENUMERATES_NOTHING

    @property
    def reachability_schemata(self) -> tuple[SchematicSequent, ...]:
        """Declare unrestricted root reachability."""
        return ()

    def apply(self, inputs: tuple[int, ...], /) -> Iterator[int]:
        """Add one to the premise."""
        if len(inputs) != 1:
            return

        yield inputs[0] + 1

    def admits(self, inputs: tuple[int, ...], output: int, /) -> bool:
        """Decide whether the conclusion is the premise's successor."""
        return len(inputs) == 1 and output == inputs[0] + 1

    def residuate(self, output: int, known: tuple[Slot[int], ...], /) -> Iterator[int]:
        """Yield no residuals."""
        del output, known
        return iter(())

    def to_spec(self) -> Mapping[str, JsonValue]:
        """Return the declaration this rule was built from."""
        return {"rule": self.name}


def test_the_hole_is_the_only_one() -> None:
    assert isinstance(HOLE, Hole)
    assert repr(HOLE) == "HOLE"


def test_conformance_is_structural() -> None:
    assert isinstance(Doubling(), Rule)
    assert not isinstance(MissingResidualSlots(), Rule)


def test_a_rule_declares_its_normal_form_role() -> None:
    roleless = type(
        "Roleless",
        (),
        {
            name: value
            for name, value in vars(Doubling).items()
            if name != "normal_form_role"
        },
    )

    assert Doubling().normal_form_role == NormalFormRole()
    assert isinstance(Doubling(), Rule)
    assert not isinstance(roleless(), Rule)


def test_a_normal_form_role_declares_its_facets() -> None:
    composition = CompositionRole("forward", principal_slot=0, order=1, crossed=True)
    role = NormalFormRole(composition=composition)

    assert role.composition == composition
    assert role.raising is None
    assert not role.coordination
    assert NormalFormRole(raising="undirected").raising == "undirected"
    assert NormalFormRole(coordination=True).coordination
    assert NormalFormRole() == NormalFormRole()
    assert NormalFormRole() != role
    assert repr(role) == (
        "NormalFormRole(composition=CompositionRole('forward', principal_slot=0, "
        "order=1, crossed=True), raising=None, coordination=False)"
    )


def test_a_normal_form_role_refuses_two_facets() -> None:
    with pytest.raises(ValueError, match="at most one facet"):
        NormalFormRole(raising="forward", coordination=True)


@pytest.mark.parametrize(
    ("slot", "order", "error"),
    [(-1, 0, ValueError), (0, -1, ValueError), (0, True, TypeError)],
)
def test_a_composition_role_refuses_a_malformed_slot_or_order(
    slot: int, order: int, error: type[Exception]
) -> None:
    with pytest.raises(error, match="composition normal-form"):
        CompositionRole("forward", principal_slot=slot, order=order)


def test_proof_schema_conformance_is_structural() -> None:
    assert isinstance(SchemaShaped(), ProofSchema)
    assert not isinstance(MissingSchemata(), ProofSchema)


def test_structural_erasure_conformance_is_structural() -> None:
    assert isinstance(ErasureShaped(), StructuralErasure)
    assert not isinstance(MissingFunctionType(), StructuralErasure)


def test_proof_calculus_conformance_is_structural() -> None:
    assert isinstance(CalculusShaped(), ProofCalculus)
    assert not isinstance(MissingAbstraction(), ProofCalculus)


def test_the_three_directions_agree_on_one_rule() -> None:
    rule: Rule[int] = Doubling()
    premises = (7,)
    outputs = list(rule.apply(premises))

    assert outputs == [14]
    assert rule.residual_slots == frozenset({0})
    assert rule.admits(premises, 14)
    assert not rule.admits(premises, 15)
    assert list(rule.residuate(14, (HOLE,))) == [7]
    assert list(rule.residuate(15, (HOLE,))) == []


def test_a_missing_direction_is_read_off_the_instance() -> None:
    residuating: Rule[int] = Doubling()
    forward_only: Rule[int] = Successor()

    assert "residual" in residuating.directions
    assert "residual" not in forward_only.directions
    assert residuating.residual_slots == frozenset({0})
    assert forward_only.residual_slots == frozenset()
    assert list(forward_only.residuate(8, (HOLE,))) == []
    assert forward_only.to_spec() == {"rule": "successor"}


def test_a_malformed_query_answers_rather_than_raising() -> None:
    rule: Rule[int] = Doubling()

    assert rule.admits((), 14) is False
    assert rule.admits((7, 7), 14) is False
    assert list(rule.residuate(14, (HOLE, HOLE))) == []
    assert list(rule.residuate(14, (7,))) == []

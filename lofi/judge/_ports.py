"""Rule and normal-form protocols."""

from typing import TYPE_CHECKING, Final, Literal, Protocol, override, runtime_checkable

from lofi._frozen import FrozenValue

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterator, Mapping, Sequence

    from lofi._types import JsonValue
    from lofi.judge._index import RuleProbe
    from lofi.judge._values import RuleRestriction
    from lofi.terms import (
        Calculus,
        Matching,
        OpName,
        Pattern,
        Term,
        TermStore,
    )

__all__ = [
    "HOLE",
    "BinaryJoin",
    "CompositionRole",
    "DeclaredRelation",
    "Direction",
    "ElaboratedSchema",
    "GroundPremises",
    "HeadIndex",
    "Hole",
    "NormalForm",
    "NormalFormDecision",
    "NormalFormDecisionTable",
    "NormalFormRole",
    "ProofCalculus",
    "ProofSchema",
    "ProofWitness",
    "Rule",
    "SchematicSequent",
    "Slot",
    "StructuralErasure",
]


type Direction = Literal["forward", "check", "residual"]
"""Rule-computation direction."""

type NormalFormDecision = Literal["admit", "refuse", "premises"]
"""Whether a finite-state normal form decides a step without its terms."""

type NormalFormDecisionTable = tuple[Mapping[tuple[str, ...], NormalFormDecision], ...]
"""One child-tag decision row per rule, in rule declaration order."""

type NormalFormDirection = Literal["forward", "backward", "undirected"]
"""Direction of a composition or raising role."""

_DIRECTIONS: Final = frozenset({"forward", "backward", "undirected"})
"""Declared normal-form directions."""


def _direction(direction: NormalFormDirection, /) -> NormalFormDirection:
    """Validate one normal-form direction."""
    if direction not in _DIRECTIONS:
        msg = f"unknown normal-form direction: {direction!r}"
        raise ValueError(msg)
    return direction


class CompositionRole(FrozenValue):
    """The composition facet of a rule's normal-form role.

    Parameters
    ----------
    direction : NormalFormDirection
        Direction the principal functor combines in.
    principal_slot : int
        The principal premise slot.
    order : int | None
        The fixed composition order, or None when each step reads its own.
    crossed : bool
        Whether the deepest constructor is crossed.

    Raises
    ------
    TypeError
        If the slot or order has the wrong type.
    ValueError
        If the direction is unknown or the slot or order is negative.
    """

    __slots__ = ("_crossed", "_direction", "_order", "_principal_slot")

    _noun = "composition role"

    def __init__(
        self,
        direction: NormalFormDirection,
        *,
        principal_slot: int,
        order: int | None,
        crossed: bool = False,
    ) -> None:
        if (
            isinstance(principal_slot, bool)
            or not isinstance(principal_slot, int)
            or principal_slot < 0
        ):
            msg = "a composition normal-form role needs a nonnegative principal slot"
            raise ValueError(msg)
        if isinstance(order, bool) or not isinstance(order, int | None):
            msg = "a composition normal-form order must be an integer or None"
            raise TypeError(msg)
        if order is not None and order < 0:
            msg = "a composition normal-form order must be nonnegative"
            raise ValueError(msg)

        object.__setattr__(self, "_direction", _direction(direction))
        object.__setattr__(self, "_principal_slot", principal_slot)
        object.__setattr__(self, "_order", order)
        object.__setattr__(self, "_crossed", crossed)

    @property
    def direction(self) -> NormalFormDirection:
        """Return the direction the principal functor combines in."""
        return self._direction

    @property
    def principal_slot(self) -> int:
        """Return the principal premise slot."""
        return self._principal_slot

    @property
    def order(self) -> int | None:
        """Return the fixed composition order, or None when step-dependent."""
        return self._order

    @property
    def crossed(self) -> bool:
        """Report whether the composition crosses its deepest constructor."""
        return self._crossed

    def _fields(self) -> tuple[NormalFormDirection, int, int | None, bool]:
        """Return the fields in constructor order."""
        return (self._direction, self._principal_slot, self._order, self._crossed)

    @override
    def __eq__(self, other: object) -> bool:
        """Compare composition facets by their declarations."""
        if type(other) is not CompositionRole:
            return NotImplemented
        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Hash the immutable declaration."""
        return hash((CompositionRole, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return a rebuilding representation."""
        return (
            f"CompositionRole({self._direction!r}, "
            f"principal_slot={self._principal_slot!r}, order={self._order!r}, "
            f"crossed={self._crossed!r})"
        )


class NormalFormRole(FrozenValue):
    """Declared structural role of one rule in a derivation normal form.

    The role is independent of the rule's implementation, serialized kind and
    constructor spelling.  A normal form can therefore constrain a third-party
    or pattern rule exactly as it constrains a built-in rule. A role declares
    at most one facet; a role with none is one no normal form constrains.

    Parameters
    ----------
    composition : CompositionRole | None
        The rule's composition facet, if it composes functors.
    raising : NormalFormDirection | None
        The direction the rule raises a category in, if it type-raises.
    coordination : bool
        Whether the rule takes a coordination step.

    Raises
    ------
    ValueError
        If more than one facet is declared or the raising direction is unknown.
    """

    __slots__ = ("_composition", "_coordination", "_raising")

    _noun = "normal-form role"

    def __init__(
        self,
        *,
        composition: CompositionRole | None = None,
        raising: NormalFormDirection | None = None,
        coordination: bool = False,
    ) -> None:
        declared = (composition is not None) + (raising is not None) + coordination
        if declared > 1:
            msg = "a normal-form role declares at most one facet"
            raise ValueError(msg)

        object.__setattr__(self, "_composition", composition)
        object.__setattr__(
            self, "_raising", None if raising is None else _direction(raising)
        )
        object.__setattr__(self, "_coordination", coordination)

    @property
    def composition(self) -> CompositionRole | None:
        """Return the composition facet, if declared."""
        return self._composition

    @property
    def raising(self) -> NormalFormDirection | None:
        """Return the raising direction, if declared."""
        return self._raising

    @property
    def coordination(self) -> bool:
        """Report whether the rule takes a coordination step."""
        return self._coordination

    def _fields(
        self,
    ) -> tuple[CompositionRole | None, NormalFormDirection | None, bool]:
        """Return the fields in constructor order."""
        return (self._composition, self._raising, self._coordination)

    @override
    def __eq__(self, other: object) -> bool:
        """Compare roles by their declarations."""
        if type(other) is not NormalFormRole:
            return NotImplemented
        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Hash the immutable declaration."""
        return hash((NormalFormRole, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return a rebuilding representation."""
        return (
            f"NormalFormRole(composition={self._composition!r}, "
            f"raising={self._raising!r}, coordination={self._coordination!r})"
        )


class Hole:
    """Missing premise in a residuation query."""

    __slots__ = ()

    @override
    def __repr__(self) -> str:
        """Return the missing-premise spelling."""
        return "HOLE"


HOLE: Final = Hole()
"""Unknown-premise marker for residuation."""


type Slot[T] = T | Hole
"""Known or missing premise."""

type SchematicSequent = tuple[tuple[Pattern, ...], Pattern]
"""Premise and conclusion patterns for one physical rule orientation."""


@runtime_checkable
class ProofWitness(Protocol):
    """One derivation step's own elaboration into a proof term."""

    def elaborate(self, calculus: ProofCalculus, /) -> Term:
        """Return the proof term the step denotes.

        Parameters
        ----------
        calculus : ProofCalculus
            The proof-term construction channels.
        """


@runtime_checkable
class ElaboratedSchema[T](Protocol):
    """A rule that elaborates proof terms for each step it licenses.

    A rule standing for a family of proof schemata has no finite deterministic
    presentation, so proof search over its erased sequents cannot reach it. It
    matches a step instead and returns the witnesses of what that step did.
    """

    @property
    def witness_is_unique(self) -> bool:
        """Report whether a licensed step has exactly one witness."""

    def witnesses(
        self, inputs: tuple[T, ...], output: T, /
    ) -> tuple[ProofWitness, ...]:
        """Return the witnesses of one step.

        Parameters
        ----------
        inputs : tuple[T, ...]
            Premises in rule order.
        output : T
            The conclusion drawn from them.

        Returns
        -------
        tuple[ProofWitness, ...]
            The witnesses, or an empty tuple when the rule licenses no such
            step.
        """


@runtime_checkable
class ProofSchema(Protocol):
    """Oriented schemata available for proof search."""

    @property
    def proof_schemata(self) -> tuple[SchematicSequent, ...]:
        """Return deterministic oriented schemata."""


@runtime_checkable
class StructuralErasure(Protocol):
    """Structural category-to-type erasure used by proof search."""

    @property
    def store(self) -> TermStore:
        """Return the one language store containing categories and target terms."""

    @property
    def category_sort(self) -> str:
        """Return the source category sort."""

    @property
    def atoms(self) -> Mapping[OpName, Term]:
        """Return core-atom images."""

    @property
    def constructors(self) -> Mapping[OpName, Term]:
        """Return constructor type-term images."""

    @property
    def function_type(self) -> OpName:
        """Return the target function-type constructor."""

    @property
    def target_sort(self) -> str:
        """Return the target sort of every image."""

    def image(self, constructor: OpName, arguments: tuple[Term, ...], /) -> Term:
        """Instantiate one constructor's image over erased arguments."""

    def __call__(self, category: Term, /) -> Term:
        """Erase a source category."""


@runtime_checkable
class ProofCalculus(Protocol):
    """Caller-owned proof-term construction."""

    @property
    def store(self) -> TermStore:
        """Return the proof-term store."""

    @property
    def operators(self) -> Calculus:
        """Return the lambda and application operators."""

    def hypothesis(self, index: int, /) -> Term:
        """Construct a stable store-owned free variable for this index."""

    def free_variable(self, name: str, /) -> Term:
        """Construct a named free variable."""

    def bound_variable(self, index: int, /) -> Term:
        """Construct a bound variable."""

    def application(
        self,
        function: Term,
        argument: Term,
        /,
    ) -> Term:
        """Construct an application."""

    def abstraction(self, variable: str, body: Term, /) -> Term:
        """Construct an abstraction."""


@runtime_checkable
class BinaryJoin[T](Protocol):
    """A binary rule with hashable keys its two premises must share.

    Every instance the rule licenses has premises sharing at least one key
    across the two slots, so pairs sharing none need not be tried. Extra keys
    only cost trials.
    """

    def join_keys(self, term: T, slot: int, /) -> tuple[Hashable, ...]:
        """Return the keys one premise offers in one slot.

        Parameters
        ----------
        term : T
            The premise.
        slot : int
            The premise slot it would fill.

        Returns
        -------
        tuple[Hashable, ...]
            The keys, none where the term can fill no instance in that slot.
        """


@runtime_checkable
class GroundPremises[T](Protocol):
    """A rule whose premise slots may each admit exactly one representative."""

    @property
    def ground_premises(self) -> tuple[T | None, ...]:
        """Return, per premise slot, the one representative the slot admits.

        A premise fills a slot holding a term only when its representative under
        the rule's matching, or the premise itself under a matching with no
        representatives, is that term. A slot holding None constrains nothing.
        """


@runtime_checkable
class HeadIndex[T](Protocol):
    """Signature-scoped head codes for rule indexing."""

    @property
    def codes(self) -> frozenset[int]:
        """Return all valid head codes."""

    def term(self, term: T, /) -> int:
        """Map a term to its head code."""

    def named(self, name: str, /) -> int | None:
        """Map a declared operator name to its head code, if present."""


@runtime_checkable
class DeclaredRelation(Protocol):
    """A rule that declares which relation over categories it denotes."""

    @property
    def relation(self) -> Mapping[str, JsonValue]:
        """Return the parameters that fix the relation the rule denotes.

        Two rules with equal relations draw the same conclusions from every
        premise tuple. Parameters that change only the rule's presentation,
        such as its name, or that the other parameters make inert, are left
        out.
        """


@runtime_checkable
class Rule[T](Protocol):
    """Relation from premises to a conclusion."""

    @property
    def name(self) -> str:
        """Return the rule name."""

    @property
    def arity(self) -> int:
        """Return the premise count."""

    @property
    def directions(self) -> frozenset[Direction]:
        """Return the supported computation directions."""

    @property
    def residual_slots(self) -> frozenset[int]:
        """Return the premise slots with computable residuals."""

    @property
    def normal_form_role(self) -> NormalFormRole:
        """Return the rule's declared structural role in a normal form."""

    @property
    def spine_order(self) -> int:
        """Return the spine order every instance of the rule carries.

        A rule with no spine, or one whose instances carry different orders,
        returns zero.
        """

    @property
    def forward_is_complete(self) -> bool:
        """Report whether forward enumeration is complete."""

    @property
    def matching(self) -> Matching[T]:
        """Return the rule's term congruence."""

    @property
    def restriction(self) -> RuleRestriction | None:
        """Return the rule-instance restriction, if declared."""

    @property
    def probe(self) -> RuleProbe | None:
        """Return conservative head constraints, if available."""

    def apply(self, inputs: tuple[T, ...], /) -> Iterator[T]:
        """Enumerate conclusions licensed by the premises.

        Parameters
        ----------
        inputs : tuple[T, ...]
            Premises in rule order.

        Returns
        -------
        Iterator[T]
            Licensed conclusions.
        """

    def admits(self, inputs: tuple[T, ...], output: T, /) -> bool:
        """Decide whether the supplied rule instance holds.

        Parameters
        ----------
        inputs : tuple[T, ...]
            Premises in rule order.
        output : T
            Candidate conclusion.

        Returns
        -------
        bool
            ``False`` for malformed or unsupported instances.
        """

    def residuate(self, output: T, known: tuple[Slot[T], ...], /) -> Iterator[T]:
        """Enumerate fillers for the single hole in ``known``.

        Parameters
        ----------
        output : T
            Rule conclusion.
        known : tuple[Slot[T], ...]
            Premises with exactly one ``HOLE`` marker.

        Returns
        -------
        Iterator[T]
            Terms that complete a valid rule instance.
        """

    def to_spec(self) -> Mapping[str, JsonValue]:
        """Return the rule declaration."""


@runtime_checkable
class NormalForm[T](Protocol):
    """Derivation-step filter."""

    @property
    def name(self) -> str:
        """Return the filter name."""

    @property
    def alphabet(self) -> frozenset[str]:
        """Return the finite tag alphabet."""

    def lexical_tag(self) -> str:
        """Return the lexical-item tag."""

    def decisions(self, rules: Sequence[Rule[T]], /) -> NormalFormDecisionTable:
        """Compile finite child-tag decisions for rules.

        A ``"premises"`` entry means :meth:`admits` must inspect the exact
        premise or conclusion terms. The other entries are final.
        """

    def tag(self, rule: Rule[T], children: tuple[str, ...], /) -> str:
        """Return the derived-item tag.

        Parameters
        ----------
        rule : Rule[T]
            Applied rule.
        children : tuple[str, ...]
            Child tags in premise order.

        Returns
        -------
        str
            Tag for the derived item.
        """

    def admits(
        self,
        rule: Rule[T],
        children: tuple[str, ...],
        inputs: tuple[T, ...],
        output: T,
        /,
    ) -> bool:
        """Decide whether the filter keeps a step.

        Parameters
        ----------
        rule : Rule[T]
            Applied rule.
        children : tuple[str, ...]
            Child tags in premise order.
        inputs : tuple[T, ...]
            Premise terms in rule order.
        output : T
            Rule conclusion.

        Returns
        -------
        bool
            Whether the step is admitted; unreadable steps are admitted.
        """

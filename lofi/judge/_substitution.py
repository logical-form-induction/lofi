"""Generalized substitution over term functors."""

from typing import TYPE_CHECKING, Final, Never, final, override

from lofi.judge._index import RuleProbe
from lofi.judge._ports import HOLE, NormalFormRole
from lofi.judge._proof import _functor_pattern, _spine_patterns
from lofi.terms import MVar, TermRuleKernel

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterator, Mapping

    from lofi._types import JsonValue
    from lofi.judge._ports import Direction, SchematicSequent, Slot
    from lofi.judge._values import RuleRestriction
    from lofi.terms import Functors, OpName, Term, TermMatching, _CompiledRule

__all__ = ["GeneralizedSubstitution"]

_ARITY: Final = 2
"""Binary-rule arity."""

type _Orientation = tuple[int, int]
"""Functor and container premise slots."""


@final
class GeneralizedSubstitution:
    """Generalized-substitution rule.

    The rule is adopted from one compiled substitution recipe, which fixes its
    order and crossing; every direction of the relation runs the recipe's
    native kernel.

    Parameters
    ----------
    compiled : _CompiledRule
        The compiled substitution recipe.
    functors : Functors[Term]
        The functor view the recipe names.
    slash : OpName
        The principal constructor of that view.
    matching : TermMatching
        The premise congruence.
    restriction : RuleRestriction | None
        The rule instance restriction, if declared.
    """

    __slots__ = (
        "_crossed",
        "_functors",
        "_lowering",
        "_matching",
        "_name",
        "_order",
        "_orientations",
        "_probe",
        "_residual_slots",
        "_restriction",
        "_slash",
    )

    def __init__(
        self,
        compiled: _CompiledRule,
        functors: Functors[Term],
        slash: OpName,
        matching: TermMatching,
        /,
        *,
        restriction: RuleRestriction | None = None,
    ) -> None:
        order = compiled.order
        crossed = compiled.crossed
        orientations: tuple[_Orientation, ...] = tuple(
            orientation
            for enabled, orientation in (
                (slash == functors.forward_slash, (0, 1)),
                (slash == functors.backward_slash, (1, 0)),
            )
            if enabled
        )
        residual_slots = frozenset(
            slot
            for slot in range(_ARITY)
            if all(container == slot for _, container in orientations)
        )
        constraint = functors.slashes
        variant = "crossed" if crossed else "harmonic"
        declared = "any" if order is None else order
        name = f"substitution:{slash}:{declared}:{variant}"
        object.__setattr__(self, "_functors", functors)
        object.__setattr__(self, "_matching", matching)
        object.__setattr__(self, "_order", order)
        object.__setattr__(self, "_slash", slash)
        object.__setattr__(self, "_crossed", crossed)
        object.__setattr__(self, "_orientations", orientations)
        object.__setattr__(
            self, "_probe", RuleProbe(_ARITY, ((constraint, constraint),))
        )
        object.__setattr__(self, "_restriction", restriction)
        object.__setattr__(self, "_residual_slots", residual_slots)
        object.__setattr__(
            self,
            "_lowering",
            TermRuleKernel(compiled.substitution_kernel(matching, functors)),
        )
        object.__setattr__(
            self,
            "_name",
            name if restriction is None else restriction.qualify(name),
        )

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "a generalized-substitution rule is immutable"

        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "a generalized-substitution rule is immutable"

        raise AttributeError(msg)

    @property
    def name(self) -> str:
        """Return the rule name."""
        return self._name

    @property
    def arity(self) -> int:
        """Return the premise count."""
        return _ARITY

    @property
    def directions(self) -> frozenset[Direction]:
        """Return the computable directions."""
        directions: set[Direction] = {"forward", "check"}

        if self._residual_slots:
            directions.add("residual")

        return frozenset(directions)

    @property
    def normal_form_role(self) -> NormalFormRole:
        """Return the rule's role, which no normal form constrains."""
        return NormalFormRole()

    @property
    def spine_order(self) -> int:
        """Return the declared order, or zero when every order is admitted."""
        return 0 if self._order is None else self._order

    @property
    def residual_slots(self) -> frozenset[int]:
        """Return the physical container slot when it is unambiguous."""
        return self._residual_slots

    @property
    def forward_is_complete(self) -> bool:
        """Report whether forward enumeration is complete."""
        return True

    @property
    def matching(self) -> TermMatching:
        """Return the term congruence."""
        return self._matching

    @property
    def restriction(self) -> RuleRestriction | None:
        """Return the rule instance restriction, if declared."""
        return self._restriction

    @property
    def probe(self) -> RuleProbe:
        """Return conservative premise-head constraints."""
        return self._probe

    @property
    def lowering(self) -> TermRuleKernel:
        """Return the native implementation of this relation."""
        return self._lowering

    @property
    def functors(self) -> Functors[Term]:
        """Return the functor view."""
        return self._functors

    @property
    def order(self) -> int | None:
        """Return the substitution order, or no finite upper bound."""
        return self._order

    @property
    def slash(self) -> OpName:
        """Return the principal slash."""
        return self._slash

    @property
    def crossed(self) -> bool:
        """Report whether the deepest shared slash is crossed."""
        return self._crossed

    @property
    def proof_schemata(self) -> tuple[SchematicSequent, ...]:
        """Return every deterministic physical orientation."""
        if self._order is None:
            return ()

        sort = self._functors.category_sort
        result = MVar("X", sort)
        argument = MVar("Y", sort)
        foot = _functor_pattern(self._functors, self._slash, result, argument)

        if foot is None:
            return ()

        arguments = tuple(MVar(f"Z{at}", sort) for at in range(self._order))
        deepest = self._functors.opposite(self._slash) if self._crossed else self._slash
        functors = tuple(
            _spine_patterns(
                self._functors,
                self._order,
                deepest,
                foot,
                arguments,
            )
        )
        containers = tuple(
            _spine_patterns(
                self._functors,
                self._order,
                deepest,
                argument,
                arguments,
            )
        )
        conclusions = tuple(
            _spine_patterns(
                self._functors,
                self._order,
                deepest,
                result,
                arguments,
            )
        )
        schemata: list[SchematicSequent] = []

        for functor, container, conclusion in zip(
            functors,
            containers,
            conclusions,
            strict=True,
        ):
            for functor_slot, _ in self._orientations:
                premises = (
                    (functor, container) if functor_slot == 0 else (container, functor)
                )
                schemata.append((premises, conclusion))

        return tuple(schemata)

    @property
    def reachability_schemata(self) -> tuple[SchematicSequent, ...]:
        """Return the consumed premises and introduced conclusions."""
        return self.proof_schemata

    def join_keys(self, term: Term, slot: int, /) -> tuple[Hashable, ...]:
        """Return the keys a premise offers in one slot.

        A functor offers, at each admitted order, the key of the argument its
        foot takes, and a container the key of its foot at that order, both
        tagged with the orientation, the order and the parallel frames, so a
        licensed pair always shares a key.

        Parameters
        ----------
        term : Term
            The premise.
        slot : int
            The premise slot it would fill.
        """
        return self._lowering.join_keys(term, slot)

    def apply(self, inputs: tuple[Term, ...], /) -> Iterator[Term]:
        """Enumerate conclusions for a premise pair."""
        yield from self._lowering.apply(inputs)

    def admits(
        self,
        inputs: tuple[Term, ...],
        output: Term,
        /,
    ) -> bool:
        """Return whether the rule relates the premises and conclusion."""
        return self._lowering.admits(inputs, output)

    def residuate(
        self,
        output: Term,
        known: tuple[Slot[Term], ...],
        /,
    ) -> Iterator[Term]:
        """Enumerate a filler for the supported container hole."""
        yield from self._lowering.residuate(
            output, tuple(None if value is HOLE else value for value in known)
        )

    @property
    def relation(self) -> Mapping[str, JsonValue]:
        """Return the parameters that fix the relation the rule denotes."""
        return self.to_spec()

    def to_spec(self) -> Mapping[str, JsonValue]:
        """Return the rule declaration."""
        return {
            "rule": "substitution",
            "matching": self._matching.name,
            "order": self._order,
            "slash": self._slash,
            "crossed": self._crossed,
            "restriction": (
                None if self._restriction is None else self._restriction.to_spec()
            ),
        }

"""Generalized composition over term functors.

A composition rule relates a principal functor to a container whose result
spine holds the arguments the composition carries through. The rule either
fixes one order or admits every finite order above zero; in the second case the
order an application uses is the one the container in front of it supplies, so
the rule walks a finite category rather than an enumeration of the naturals.
Order zero is application, which has no spine, and is declared on its own.

The rule's direction restriction fixes which slashes that spine may use. Under
``"steedman"`` every spine slash is the one the crossing declaration names.
Under ``"kuhlmann"`` only the deepest slash, the one adjacent to the shared
category, is fixed and the shallower ones are free. Under ``"unrestricted"`` no
spine slash is constrained. The three coincide at order zero, where the
container has no spine.
"""

from typing import TYPE_CHECKING, Final, Never, cast, final, override

from lofi.judge._index import RuleProbe
from lofi.judge._ports import HOLE, CompositionRole, NormalFormRole
from lofi.judge._proof import CompositionWitness, _functor_pattern, _spine_patterns
from lofi.terms import MVar, TermRuleKernel

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterator, Mapping

    from lofi._types import JsonValue
    from lofi.judge._ports import Direction, SchematicSequent, Slot
    from lofi.judge._values import CompositionRestriction, RuleRestriction
    from lofi.terms import Functors, OpName, Pattern, Term, TermMatching, _CompiledRule

__all__ = ["GeneralizedComposition"]

_ARITY: Final = 2
"""Binary-rule arity."""

_RESIDUAL_SLOTS: Final = frozenset(range(_ARITY))
"""Premise slots whose residual is finite."""


type _Orientation = tuple[int, int]
"""Principal and container premise slots."""


@final
class GeneralizedComposition:
    """Generalized composition rule.

    The rule is adopted from one compiled application or composition recipe,
    which fixes its order, direction restriction and crossing; every
    direction of the relation runs the recipe's native kernel.

    Parameters
    ----------
    compiled : _CompiledRule
        The compiled application or composition recipe.
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
        "_direction",
        "_functors",
        "_lowering",
        "_matching",
        "_name",
        "_order",
        "_orientations",
        "_probe",
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
        application = compiled.kind == "application"
        order = 0 if application else compiled.order
        direction = cast(
            "CompositionRestriction",
            "unrestricted" if application else compiled.direction,
        )
        crossed = compiled.crossed
        orientations: tuple[_Orientation, ...] = tuple(
            orientation
            for enabled, orientation in (
                (slash == functors.forward_slash, (0, 1)),
                (slash == functors.backward_slash, (1, 0)),
            )
            if enabled
        )
        alternatives = []
        for principal, container in orientations:
            constraints: list[frozenset[OpName] | None] = [None, None]
            constraints[principal] = frozenset({slash})
            if order != 0:
                constraints[container] = functors.slashes
            alternatives.append(tuple(constraints))

        variant = "crossed" if crossed else "harmonic"
        declared = "any" if order is None else order
        name = (
            f"application:{slash}"
            if order == 0
            else f"composition:{slash}:{declared}:{variant}:{direction}"
        )
        object.__setattr__(self, "_functors", functors)
        object.__setattr__(self, "_matching", matching)
        object.__setattr__(self, "_order", order)
        object.__setattr__(self, "_slash", slash)
        object.__setattr__(self, "_crossed", crossed)
        object.__setattr__(self, "_direction", direction)
        object.__setattr__(self, "_orientations", orientations)
        object.__setattr__(self, "_probe", RuleProbe(_ARITY, alternatives))
        object.__setattr__(self, "_restriction", restriction)
        object.__setattr__(
            self,
            "_lowering",
            TermRuleKernel(compiled.composition_kernel(matching, functors)),
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
        msg = "a generalized-composition rule is immutable"

        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "a generalized-composition rule is immutable"

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
        return frozenset({"forward", "check", "residual"})

    @property
    def spine_order(self) -> int:
        """Return the declared order, or zero when every order is admitted."""
        return 0 if self._order is None else self._order

    @property
    def residual_slots(self) -> frozenset[int]:
        """Return the premise slots with finite residuals."""
        return _RESIDUAL_SLOTS

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
    def normal_form_role(self) -> NormalFormRole:
        """Return the composition metadata used by normal form filters."""
        if len(self._orientations) == 1:
            principal, _ = self._orientations[0]
            direction = "forward" if principal == 0 else "backward"
        else:
            principal = 0
            direction = "undirected"

        return NormalFormRole(
            composition=CompositionRole(
                direction,
                principal_slot=principal,
                order=self._order,
                crossed=self._crossed,
            )
        )

    @property
    def functors(self) -> Functors[Term]:
        """Return the functor view."""
        return self._functors

    @property
    def order(self) -> int | None:
        """Return the declared order, or ``None`` for every finite order."""
        return self._order

    @property
    def slash(self) -> OpName:
        """Return the principal slash."""
        return self._slash

    @property
    def crossed(self) -> bool:
        """Report whether the deepest slash is crossed."""
        return self._crossed

    @property
    def direction(self) -> CompositionRestriction:
        """Return the direction restriction on the container spine."""
        return self._direction

    @property
    def proof_schemata(self) -> tuple[SchematicSequent, ...]:
        """Return every deterministic physical orientation.

        A rule admitting every finite order stands for one schema per order and
        so has no finite deterministic presentation; it returns no schema.
        """
        if self._order is None:
            return ()

        sort = self._functors.category_sort
        result = MVar("X", sort)
        argument = MVar("Y", sort)
        principal = _functor_pattern(
            self._functors,
            self._slash,
            result,
            argument,
        )

        if principal is None:
            return ()

        pairs: tuple[tuple[Pattern, Pattern], ...]

        if self._order == 0:
            pairs = ((argument, result),)
        else:
            arguments = tuple(MVar(f"Z{at}", sort) for at in range(self._order))
            collected: list[tuple[Pattern, Pattern]] = []

            for deepest in self._deepest_slashes():
                prefixes = self._prefix_slashes(deepest)
                containers = tuple(
                    _spine_patterns(
                        self._functors,
                        self._order,
                        deepest,
                        argument,
                        arguments,
                        prefixes,
                    )
                )
                conclusions = tuple(
                    _spine_patterns(
                        self._functors,
                        self._order,
                        deepest,
                        result,
                        arguments,
                        prefixes,
                    )
                )
                collected.extend(zip(containers, conclusions, strict=True))

            pairs = tuple(collected)

        schemata: list[SchematicSequent] = []

        for container, conclusion in pairs:
            for principal_slot, _ in self._orientations:
                premises = (
                    (principal, container)
                    if principal_slot == 0
                    else (container, principal)
                )
                schemata.append((premises, conclusion))

        return tuple(schemata)

    @property
    def witness_is_unique(self) -> bool:
        """Report whether every licensed step has one physical orientation."""
        return len(self._orientations) == 1

    @property
    def reachability_schemata(self) -> tuple[SchematicSequent, ...]:
        """Return the consumed premises and introduced conclusions."""
        return self.proof_schemata

    def _required_slash(self) -> OpName:
        """Return the slash the crossing declaration names."""
        return self._functors.opposite(self._slash) if self._crossed else self._slash

    def _deepest_slashes(self) -> tuple[OpName, ...]:
        """Return the slashes the deepest spine frame may take."""
        if self._direction == "unrestricted":
            return tuple(sorted(self._functors.slashes))

        return (self._required_slash(),)

    def _prefix_slashes(self, deepest: OpName) -> tuple[OpName, ...]:
        """Return the slashes the shallower spine frames may take."""
        if self._direction == "steedman":
            return (deepest,)

        return tuple(sorted(self._functors.slashes))

    def _spine_license(self) -> list[JsonValue] | None:
        """Return the slashes each spine frame may take, deepest frame first.

        Application has no spine. Otherwise each slash the deepest frame may
        take is listed with the slashes the shallower frames may then take,
        which a rule of order one never has. The direction restriction and the
        crossing declaration affect conclusions only through these sets.
        """
        if self._order == 0:
            return None

        return [
            [deepest, [] if self._order == 1 else list(self._prefix_slashes(deepest))]
            for deepest in self._deepest_slashes()
        ]

    def join_keys(self, term: Term, slot: int, /) -> tuple[Hashable, ...]:
        """Return the matching keys a premise offers in one slot.

        A principal offers its argument's key and a container the key of each
        foot an admitted order peels to, both tagged with the orientation, so a
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

    def witnesses(
        self, inputs: tuple[Term, ...], output: Term, /
    ) -> tuple[CompositionWitness, ...]:
        """Return the witnesses of the step the premises and conclusion make.

        Each witness names the premise slots one physical orientation used,
        the principal operator, and the container spine the conclusion
        carries, from which the order and the crossing of the step follow.

        Parameters
        ----------
        inputs : tuple[Term, ...]
            Premise categories in rule order.
        output : Term
            The conclusion whose witness is wanted.

        Returns
        -------
        tuple[CompositionWitness, ...]
            The witnesses, or an empty tuple when the rule does not relate
            the premises to the conclusion.
        """
        return tuple(
            CompositionWitness(principal, container, slash, spine)
            for principal, container, slash, spine in self._lowering.witnesses(
                inputs, output
            )
        )

    def applied_order(self, inputs: tuple[Term, ...], output: Term, /) -> int | None:
        """Return the order at which the rule draws a conclusion.

        Parameters
        ----------
        inputs : tuple[Term, ...]
            Premise categories in rule order.
        output : Term
            The conclusion whose order is wanted.

        Returns
        -------
        int | None
            The composition order the derivation used, or ``None`` when the
            rule does not relate the premises to the conclusion.

        Raises
        ------
        ValueError
            If witnesses of the same step report inconsistent orders.
        """
        return self._lowering.applied_order(inputs, output)

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
        """Enumerate fillers for one supported premise hole."""
        yield from self._lowering.residuate(
            output, tuple(None if value is HOLE else value for value in known)
        )

    @property
    def relation(self) -> Mapping[str, JsonValue]:
        """Return the parameters that fix the relation the rule denotes.

        The direction restriction and the crossing enter only through the
        slashes they leave each spine frame, so declarations that leave every
        frame the same slashes denote the same relation.
        """
        return {
            "rule": "composition",
            "matching": self._matching.name,
            "order": self._order,
            "slash": self._slash,
            "spine": self._spine_license(),
            "restriction": (
                None if self._restriction is None else self._restriction.to_spec()
            ),
        }

    def to_spec(self) -> Mapping[str, JsonValue]:
        """Return the rule declaration."""
        return {
            "rule": "composition",
            "matching": self._matching.name,
            "order": self._order,
            "slash": self._slash,
            "crossed": self._crossed,
            "direction": self._direction,
            "restriction": (
                None if self._restriction is None else self._restriction.to_spec()
            ),
        }

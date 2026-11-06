"""Finite normal-form filters for derivation steps."""

import json
from itertools import product
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Never, cast, final, override

from lofi.terms import at_path

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Protocol

    from lofi.judge._ports import (
        CompositionRole,
        NormalForm,
        NormalFormDecision,
        NormalFormDecisionTable,
        Rule,
    )
    from lofi.terms import Functors, Term

    class _CompositionStep[T](Protocol):
        """Term capabilities used by exact raising constraints."""

        @property
        def functors(self) -> Functors[T]:
            """Return the rule's functor view."""

        def applied_order(self, inputs: tuple[T, ...], output: T, /) -> int | None:
            """Return the order one derivation step used."""


__all__ = [
    "AllOf",
    "EisnerNormalForm",
    "HockenmaierBiskNormalForm",
    "NoNormalForm",
]

_OTHER: Final = "OTHER"
"""Tag for lexical items and other rule results."""

_FORWARD_COMPOSITION: Final = "FC"
"""Tag for a forward-composition result of unread degree."""

_BACKWARD_COMPOSITION: Final = "BC"
"""Tag for a backward-composition result of unread degree."""

_UNDIRECTED_COMPOSITION: Final = "UC"
"""Tag for an undirected-composition result of unread degree."""

_FORWARD_RAISING: Final = "FTR"
"""Tag for a forward type-raising result."""

_BACKWARD_RAISING: Final = "BTR"
"""Tag for a backward type-raising result."""

_UNDIRECTED_RAISING: Final = "UTR"
"""Tag for an undirected type-raising result."""

_COORDINATION: Final = "COORD"
"""Tag for a coordination result."""

_UNIT_DEGREE: Final = "1"
"""Suffix of a composition tag whose degree is one."""

_HIGHER_DEGREE: Final = "n"
"""Suffix of a composition tag whose degree exceeds one."""

_EISNER_ALPHABET: Final = frozenset(
    {
        _OTHER,
        _FORWARD_COMPOSITION,
        _BACKWARD_COMPOSITION,
        _UNDIRECTED_COMPOSITION,
    }
)
"""Eisner normal-form tag alphabet."""

_HOCKENMAIER_BISK_ALPHABET: Final = frozenset(
    {
        _OTHER,
        _FORWARD_RAISING,
        _BACKWARD_RAISING,
        _UNDIRECTED_RAISING,
        _COORDINATION,
        *(
            f"{base}{suffix}"
            for base in (
                _FORWARD_COMPOSITION,
                _BACKWARD_COMPOSITION,
                _UNDIRECTED_COMPOSITION,
            )
            for suffix in ("", _UNIT_DEGREE, _HIGHER_DEGREE)
        ),
    }
)
"""Hockenmaier-Bisk normal-form tag alphabet."""


def _decisions[T](
    rules: Sequence[Rule[T]],
    alphabet: frozenset[str],
    decide: Callable[[Rule[T], tuple[str, ...]], NormalFormDecision],
) -> NormalFormDecisionTable:
    """Compile one immutable child-tag decision row per rule."""
    tags = tuple(sorted(alphabet))

    return tuple(
        MappingProxyType(
            {
                children: decide(rule, children)
                for children in product(tags, repeat=rule.arity)
            }
        )
        for rule in rules
    )


def _admit[T](rule: Rule[T], children: tuple[str, ...]) -> NormalFormDecision:
    """Return the unconditional admission decision."""
    del rule, children

    return "admit"


def _composition_role[T](rule: Rule[T]) -> CompositionRole | None:
    """Read a rule's declared composition facet, if it has one."""
    return rule.normal_form_role.composition


def _raising_tag[T](rule: Rule[T]) -> str | None:
    """Return the tag of a type-raising rule's results, or None for another rule."""
    direction = rule.normal_form_role.raising

    return None if direction is None else _raised_tag(direction)


def _raised_tag(direction: str) -> str:
    """Return the raising tag a composition in one direction consumes."""
    if direction == "undirected":
        return _UNDIRECTED_RAISING
    return _FORWARD_RAISING if direction == "forward" else _BACKWARD_RAISING


def _is_coordination[T](rule: Rule[T]) -> bool:
    """Report whether a rule declares a coordination step."""
    return rule.normal_form_role.coordination


def _valid_step[T](
    rule: Rule[T], children: tuple[str, ...], inputs: tuple[T, ...]
) -> bool:
    """Report whether a step has one child and input per premise."""
    return len(children) == rule.arity and len(inputs) == rule.arity


def _direction_tag(direction: str) -> str:
    """Return the composition tag for one direction."""
    if direction == "undirected":
        return _UNDIRECTED_COMPOSITION
    return _FORWARD_COMPOSITION if direction == "forward" else _BACKWARD_COMPOSITION


def _degree_tag(direction: str, order: int | None) -> str:
    """Return the composition tag for one direction and declared degree."""
    base = _direction_tag(direction)

    if order is None:
        return base

    return f"{base}{_UNIT_DEGREE if order == 1 else _HIGHER_DEGREE}"


def _step_order[T](
    rule: Rule[T], role: CompositionRole, inputs: tuple[T, ...], output: T
) -> int | None:
    """Return the order a composition step used, declared or derived."""
    if role.order is not None:
        return role.order

    return cast("_CompositionStep[T]", rule).applied_order(inputs, output)


def _raised_result_is_kept[T](
    rule: Rule[T], slot: int, order: int, inputs: tuple[T, ...]
) -> bool:
    """Report whether a raised primary's step concludes what lowering it would.

    The step composes the raised primary ``T|(T|X)`` with the functor its
    container's result spine reaches after ``order`` frames. Combining ``X``
    with that functor directly concludes that functor's result, so the two
    derivations conclude one exact category only when that result is ``T``
    itself, which a matching coarser than identity does not guarantee.
    """
    functors = cast("_CompositionStep[T]", rule).functors
    path = functors.result_path
    raised = cast("Term", inputs[slot])
    reached = cast("Term", inputs[1 - slot])

    for _ in range(order):
        if not functors.is_functor(cast("T", reached)):
            return False

        reached = at_path(reached, path)

    if not functors.is_functor(cast("T", reached)):
        return False

    return at_path(reached, path) is at_path(raised, path)


@final
class NoNormalForm[T]:
    """Unrestricted derivation-step filter."""

    __slots__ = ()

    @property
    def name(self) -> str:
        """Return the filter name."""
        return "none"

    @property
    def alphabet(self) -> frozenset[str]:
        """Return the singleton tag alphabet."""
        return frozenset({_OTHER})

    def lexical_tag(self) -> str:
        """Return the lexical tag."""
        return _OTHER

    def decisions(self, rules: Sequence[Rule[T]], /) -> NormalFormDecisionTable:
        """Admit every finite child-tag combination."""
        return _decisions(rules, self.alphabet, _admit)

    def tag(self, rule: Rule[T], children: tuple[str, ...], /) -> str:
        """Assign the unrestricted tag to a rule result."""
        del rule, children

        return _OTHER

    def admits(
        self,
        rule: Rule[T],
        children: tuple[str, ...],
        inputs: tuple[T, ...],
        output: T,
        /,
    ) -> bool:
        """Admit every structurally complete step."""
        del output

        return _valid_step(rule, children, inputs)


@final
class EisnerNormalForm[T]:
    """Eisner's derivation-step filter for directed composition.

    The result of a composition of positive degree is never the primary input of
    an application or composition in the same direction. Every other step is
    admitted, and every other rule, unary rules such as coercions included,
    gives its result the plain tag. The filter keeps one derivation per reading
    when composition is unbounded and unrestricted; a spine restriction or a
    bounded degree can refuse the only derivation when the required
    rebracketing is absent from the declared rule set.
    """

    __slots__ = ("_include_crossed", "_undirected")

    def __init__(
        self,
        *,
        include_crossed: bool = True,
        undirected: bool = False,
    ) -> None:
        object.__setattr__(self, "_include_crossed", include_crossed)
        object.__setattr__(self, "_undirected", undirected)

    @property
    def name(self) -> str:
        """Return the filter name."""
        return "undirected-eisner" if self._undirected else "eisner"

    @property
    def include_crossed(self) -> bool:
        """Report whether crossed compositions participate in the filter."""
        return self._include_crossed

    @property
    def undirected(self) -> bool:
        """Report whether the filter identifies the two directions."""
        return self._undirected

    def _direction(self, role: CompositionRole) -> str:
        """Return the direction represented in this filter's alphabet."""
        return "undirected" if self._undirected else role.direction

    def _composition(self, rule: Rule[T]) -> CompositionRole | None:
        """Return the composition facet this filter constrains, if any."""
        role = _composition_role(rule)
        if role is None or (role.crossed and not self._include_crossed):
            return None
        return role

    @property
    def alphabet(self) -> frozenset[str]:
        """Return the three-tag alphabet."""
        return _EISNER_ALPHABET

    def lexical_tag(self) -> str:
        """Return the lexical tag."""
        return _OTHER

    def decisions(self, rules: Sequence[Rule[T]], /) -> NormalFormDecisionTable:
        """Compile the Eisner decision for every rule and child-tag tuple."""

        def decide(rule: Rule[T], children: tuple[str, ...]) -> NormalFormDecision:
            role = self._composition(rule)

            if role is None:
                return "admit"

            direction = self._direction(role)
            primary = children[role.principal_slot]
            blocked = _direction_tag(direction)

            return "refuse" if primary == blocked else "admit"

        return _decisions(rules, self.alphabet, decide)

    def tag(self, rule: Rule[T], children: tuple[str, ...], /) -> str:
        """Tag directed composition results by direction."""
        del children
        role = self._composition(rule)

        if role is None or role.order == 0:
            return _OTHER

        return _direction_tag(self._direction(role))

    def admits(
        self,
        rule: Rule[T],
        children: tuple[str, ...],
        inputs: tuple[T, ...],
        output: T,
        /,
    ) -> bool:
        """Return whether the step satisfies the Eisner constraints."""
        del output

        if not _valid_step(rule, children, inputs):
            return False

        role = self._composition(rule)

        if role is None:
            return True

        direction = self._direction(role)
        primary = children[role.principal_slot]
        blocked = _direction_tag(direction)

        return primary != blocked


@final
class HockenmaierBiskNormalForm[T]:
    r"""Hockenmaier and Bisk's derivation-step filter for bounded composition.

    A composition result is tagged with its direction and its degree: one, more
    than one, or unread when its rule admits every degree, so that the degree is
    fixed only once a step is in hand. A type-raising result is tagged with its
    direction, and a coordination result with its own tag. Unary rules of any
    other kind, such as coercions, reset the tag, so the step after a coercion is
    never compared with the step before it.

    With $n$ the degree of the step under test and $m$ the degree of a
    child, in one direction:

    1. A composition result is never the primary input of a step with $n\leq1$.
    2. A result of degree one is never the primary input of a step with $n\geq1$.
    3. A type-raising result is never the primary input of a composition when the
       secondary input is an opposite-direction composition result with
       $m>n$.
    4. A type-raising result is never the functor of an application.
    5. A coordination result is never type-raised.

    Constraints 3 and 4 refuse a step only when combining the unraised category
    directly concludes the same exact category: under a matching coarser than
    identity, the raised category's result can differ in its features from the
    result of the functor it meets, and the raised derivation is then the only
    one of its conclusion.

    Constraints 1 and 2 presuppose, as the published form does, that each
    direction declares every composition degree up to its largest. Degrees above
    one share a tag, so constraint 3 refuses only steps of degree one. The
    published constraint on a secondary input of degree $1<m<n$ compares
    two degrees above one, which the tags do not tell apart, so it is not
    applied; no grammar whose degrees are at most two has such a step. A step
    whose degree cannot be read is admitted.
    """

    __slots__ = (
        "_composition_primary",
        "_coordination_raising",
        "_order_bound",
        "_raised_application",
        "_raised_secondary",
        "_track_child_order",
        "_unit_primary",
    )

    def __init__(
        self,
        *,
        order_bound: int = 1,
        track_child_order: bool = False,
        composition_primary: bool = True,
        unit_primary: bool = True,
        raised_secondary: bool = True,
        raised_application: bool = True,
        coordination_raising: bool = True,
    ) -> None:
        if isinstance(order_bound, bool) or not isinstance(order_bound, int):
            msg = "a Hockenmaier-Bisk order bound must be an integer"
            raise TypeError(msg)
        if order_bound < 1:
            msg = "a Hockenmaier-Bisk order bound must be positive"
            raise ValueError(msg)

        object.__setattr__(self, "_order_bound", order_bound)
        object.__setattr__(self, "_track_child_order", track_child_order)
        object.__setattr__(self, "_composition_primary", composition_primary)
        object.__setattr__(self, "_unit_primary", unit_primary)
        object.__setattr__(self, "_raised_secondary", raised_secondary)
        object.__setattr__(self, "_raised_application", raised_application)
        object.__setattr__(self, "_coordination_raising", coordination_raising)

    @property
    def order_bound(self) -> int:
        """Return the largest child order represented exactly."""
        return self._order_bound

    @property
    def track_child_order(self) -> bool:
        """Report whether tags retain each bounded child order."""
        return self._track_child_order

    @property
    def name(self) -> str:
        """Return the filter name."""
        return "hockenmaier-bisk"

    @property
    def alphabet(self) -> frozenset[str]:
        """Return the degree, raising and coordination tag alphabet."""
        if not self._track_child_order:
            return _HOCKENMAIER_BISK_ALPHABET

        composition = {
            tag
            for base in (
                _FORWARD_COMPOSITION,
                _BACKWARD_COMPOSITION,
                _UNDIRECTED_COMPOSITION,
            )
            for tag in (
                base,
                *(f"{base}:{order}" for order in range(1, self._order_bound + 1)),
                f"{base}:>{self._order_bound}",
            )
        }
        return frozenset(
            {
                _OTHER,
                _FORWARD_RAISING,
                _BACKWARD_RAISING,
                _UNDIRECTED_RAISING,
                _COORDINATION,
                *composition,
            }
        )

    def _degree_tag(self, direction: str, order: int | None) -> str:
        """Return a direction-and-order tag under the declared precision."""
        base = _direction_tag(direction)

        if order is None:
            return base
        if not self._track_child_order:
            return _degree_tag(direction, order)
        if order <= self._order_bound:
            return f"{base}:{order}"
        return f"{base}:>{self._order_bound}"

    def _child_order(self, tag: str, direction: str) -> int | None:
        """Return the exact represented order, or None when unreadable."""
        base = _direction_tag(direction)

        if self._track_child_order:
            prefix = f"{base}:"
            if tag.startswith(prefix) and not tag.startswith(f"{base}:>"):
                return int(tag.removeprefix(prefix))
            return None

        return 1 if tag == f"{base}{_UNIT_DEGREE}" else None

    def _is_composition(self, tag: str, direction: str) -> bool:
        """Report whether a child tag denotes composition in a direction."""
        return tag.startswith(_direction_tag(direction))

    def _is_higher(self, tag: str, direction: str, order: int) -> bool:
        """Report whether a child composition order is known to exceed order."""
        base = _direction_tag(direction)

        if self._track_child_order:
            if tag == f"{base}:>{self._order_bound}":
                return order <= self._order_bound
            child_order = self._child_order(tag, direction)
            return child_order is not None and child_order > order

        return tag == f"{base}{_HIGHER_DEGREE}" and order == 1

    def lexical_tag(self) -> str:
        """Return the lexical tag."""
        return _OTHER

    def decisions(self, rules: Sequence[Rule[T]], /) -> NormalFormDecisionTable:
        """Compile every decision that does not require the exact terms."""

        def decide(rule: Rule[T], children: tuple[str, ...]) -> NormalFormDecision:
            role = _composition_role(rule)

            if role is None:
                raising = _raising_tag(rule)

                return (
                    "refuse"
                    if self._coordination_raising
                    and raising is not None
                    and children[0] == _COORDINATION
                    else "admit"
                )

            direction = role.direction
            order = role.order

            if order is None:
                return "premises"

            slot = role.principal_slot
            primary = children[slot]
            secondary = children[1 - slot]
            opposite = (
                "undirected"
                if direction == "undirected"
                else "backward"
                if direction == "forward"
                else "forward"
            )
            raised = _raised_tag(direction)

            if primary == raised:
                higher = self._is_higher(secondary, opposite, order)

                inspect = (self._raised_application and order == 0) or (
                    self._raised_secondary and higher
                )
                return "premises" if inspect else "admit"

            if (
                self._composition_primary
                and self._is_composition(primary, direction)
                and order <= 1
            ):
                return "refuse"

            unit = self._child_order(primary, direction) == 1
            return "refuse" if self._unit_primary and unit else "admit"

        return _decisions(rules, self.alphabet, decide)

    def tag(self, rule: Rule[T], children: tuple[str, ...], /) -> str:
        """Return the rule's normal-form tag."""
        del children
        role = _composition_role(rule)

        if role is not None:
            direction = role.direction
            order = role.order

            return _OTHER if order == 0 else self._degree_tag(direction, order)

        raising = _raising_tag(rule)

        if raising is not None:
            return raising

        return _COORDINATION if _is_coordination(rule) else _OTHER

    def admits(
        self,
        rule: Rule[T],
        children: tuple[str, ...],
        inputs: tuple[T, ...],
        output: T,
        /,
    ) -> bool:
        """Return whether the step satisfies the Hockenmaier-Bisk constraints."""
        if not _valid_step(rule, children, inputs):
            return False

        role = _composition_role(rule)

        if role is None:
            return (
                not self._coordination_raising
                or _raising_tag(rule) is None
                or children[0] != _COORDINATION
            )

        direction = role.direction
        order = _step_order(rule, role, inputs, output)

        if order is None:
            return True

        slot = role.principal_slot
        primary = children[slot]
        secondary = children[1 - slot]
        opposite = (
            "undirected"
            if direction == "undirected"
            else "backward"
            if direction == "forward"
            else "forward"
        )
        raised = _raised_tag(direction)

        if primary == raised:
            higher = self._is_higher(secondary, opposite, order)

            return not (
                (
                    (self._raised_application and order == 0)
                    or (self._raised_secondary and higher)
                )
                and _raised_result_is_kept(rule, slot, order, inputs)
            )

        if (
            self._composition_primary
            and self._is_composition(primary, direction)
            and order <= 1
        ):
            return False

        return not (self._unit_primary and self._child_order(primary, direction) == 1)


@final
class AllOf[T]:
    """Conjunction of finite normal-form filters."""

    __slots__ = ("_alphabet", "_decode", "_encode", "_name", "_normal_forms")

    def __init__(self, normal_forms: Sequence[NormalForm[T]]) -> None:
        held = tuple(normal_forms)
        products = tuple(product(*(sorted(one.alphabet) for one in held)))
        encoded = {
            tags: json.dumps(tags, ensure_ascii=False, separators=(",", ":"))
            for tags in products
        }
        decoded = {tag: tags for tags, tag in encoded.items()}
        name = "all-of" if not held else f"all-of:{','.join(one.name for one in held)}"

        object.__setattr__(self, "_normal_forms", held)
        object.__setattr__(self, "_encode", MappingProxyType(encoded))
        object.__setattr__(self, "_decode", MappingProxyType(decoded))
        object.__setattr__(self, "_alphabet", frozenset(decoded))
        object.__setattr__(self, "_name", name)

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "a normal-form conjunction is immutable"

        raise AttributeError(msg)

    @property
    def name(self) -> str:
        """Return the conjunction name."""
        return self._name

    @property
    def alphabet(self) -> frozenset[str]:
        """Return the encoded product alphabet."""
        return self._alphabet

    @property
    def normal_forms(self) -> tuple[NormalForm[T], ...]:
        """Return the conjoined filters."""
        return self._normal_forms

    def lexical_tag(self) -> str:
        """Encode the component lexical tags."""
        return self._encode[tuple(one.lexical_tag() for one in self._normal_forms)]

    def decisions(self, rules: Sequence[Rule[T]], /) -> NormalFormDecisionTable:
        """Compile the pointwise meet of the component decision tables."""
        tables = tuple(one.decisions(rules) for one in self._normal_forms)
        positions = {id(rule): position for position, rule in enumerate(rules)}

        def decide(rule: Rule[T], children: tuple[str, ...]) -> NormalFormDecision:
            if not tables:
                return "admit"

            position = positions[id(rule)]
            component = tuple(
                table[position][self._project(children, index)]
                for index, table in enumerate(tables)
            )

            if "refuse" in component:
                return "refuse"

            return "premises" if "premises" in component else "admit"

        return _decisions(rules, self.alphabet, decide)

    def _project(self, children: tuple[str, ...], index: int) -> tuple[str, ...]:
        """Project composite child tags to one component."""
        try:
            return tuple(self._decode[tag][index] for tag in children)
        except KeyError as error:
            msg = f"unknown conjunction tag {error.args[0]!r}"
            raise ValueError(msg) from error

    def tag(self, rule: Rule[T], children: tuple[str, ...], /) -> str:
        """Encode the component tags assigned to one step."""
        tags = tuple(
            one.tag(rule, self._project(children, index))
            for index, one in enumerate(self._normal_forms)
        )

        try:
            return self._encode[tags]
        except KeyError as error:
            msg = f"component tag tuple {tags!r} is outside the product alphabet"
            raise ValueError(msg) from error

    def admits(
        self,
        rule: Rule[T],
        children: tuple[str, ...],
        inputs: tuple[T, ...],
        output: T,
        /,
    ) -> bool:
        """Admit a step accepted by every component filter."""
        if not self._normal_forms:
            return _valid_step(rule, children, inputs)

        return all(
            one.admits(rule, self._project(children, index), inputs, output)
            for index, one in enumerate(self._normal_forms)
        )

"""Conservative head-code indexing for unary and binary rules."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Never, final, override

from lofi._checks import ensure_name, ensure_non_negative
from lofi.judge._ports import BinaryJoin, GroundPremises

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from lofi.judge._ports import HeadIndex, Rule
    from lofi.terms import Matching, OpName

__all__ = ["RuleIndex", "RuleProbe"]

_BINARY_ARITY: Final = 2
"""Binary-rule arity."""

_UNARY_ARITY: Final = 1
"""Unary-rule arity."""

type _HeadConstraint = frozenset[OpName] | None
"""Permitted operator names for one premise head."""

type _ProbeAlternative = tuple[_HeadConstraint, ...]
"""Conjunctive head constraints for one rule orientation."""

type _CodeConstraint = frozenset[int] | None
"""Compiled codes for one premise-head constraint."""

type _CodeAlternative = tuple[_CodeConstraint, ...]
"""Compiled premise-head constraints for one rule orientation."""


@final
class RuleProbe:
    """Conservative premise-head constraints in disjunctive form."""

    __slots__ = ("_alternatives", "_arity")

    def __init__(
        self,
        arity: int,
        alternatives: Iterable[_ProbeAlternative],
    ) -> None:
        held_arity = ensure_non_negative("probe arity", arity)
        held: list[_ProbeAlternative] = []
        seen: set[_ProbeAlternative] = set()

        for at, alternative in enumerate(alternatives):
            if len(alternative) != held_arity:
                msg = (
                    f"probe alternative {at} has arity {len(alternative)}, "
                    f"not {held_arity}"
                )
                raise ValueError(msg)

            checked = tuple(
                None
                if constraint is None
                else frozenset(
                    ensure_name(f"probe alternative {at} head", name)
                    for name in constraint
                )
                for constraint in alternative
            )

            if checked not in seen:
                seen.add(checked)
                held.append(checked)

        object.__setattr__(self, "_arity", held_arity)
        object.__setattr__(self, "_alternatives", tuple(held))

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "a rule probe is immutable"

        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "a rule probe is immutable"

        raise AttributeError(msg)

    @override
    def __reduce__(
        self,
    ) -> tuple[type[RuleProbe], tuple[int, tuple[_ProbeAlternative, ...]]]:
        """Return the probe reconstruction call."""
        return (RuleProbe, (self._arity, self._alternatives))

    @property
    def arity(self) -> int:
        """Return the constrained premise count."""
        return self._arity

    @property
    def alternatives(self) -> tuple[_ProbeAlternative, ...]:
        """Return the disjunctive head constraints."""
        return self._alternatives

    def _identity(self) -> tuple[int, tuple[_ProbeAlternative, ...]]:
        """Return the fields in comparison order."""
        return (self._arity, self._alternatives)

    @override
    def __eq__(self, other: object) -> bool:
        """Compare probe fields."""
        if type(other) is not RuleProbe:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Hash the probe fields."""
        return hash((RuleProbe, self._identity()))

    @override
    def __repr__(self) -> str:
        """Render the constructor call."""
        return f"RuleProbe({self._arity!r}, {self._alternatives!r})"


def _codes[T](
    probe: RuleProbe,
    heads: HeadIndex[T],
    declared: frozenset[int],
) -> tuple[_CodeAlternative, ...]:
    """Compile operator-name constraints to head codes."""
    alternatives: list[_CodeAlternative] = []

    for alternative in probe.alternatives:
        compiled: list[_CodeConstraint] = []

        for constraint in alternative:
            if constraint is None:
                compiled.append(None)
                continue

            codes = frozenset(
                ensure_non_negative("named head code", code)
                for name in constraint
                if (code := heads.named(name)) is not None
            )
            unknown = codes - declared

            if unknown:
                msg = (
                    "head index mapped a declared name outside its codes: "
                    f"{sorted(unknown)}"
                )
                raise ValueError(msg)

            compiled.append(codes)

        alternatives.append(tuple(compiled))

    return tuple(alternatives)


@final
class RuleIndex[T]:
    """Unary- and binary-rule candidate tables."""

    __slots__ = (
        "_always",
        "_codes",
        "_heads",
        "_inventory",
        "_joined",
        "_positions",
        "_rules",
        "_table",
        "_unary_always",
        "_unary_ground",
        "_unary_positions",
        "_unary_rules",
        "_unary_table",
        "_unjoined_table",
        "_width",
    )

    def __init__(self, rules: Sequence[Rule[T]], heads: HeadIndex[T]) -> None:
        inventory = tuple(rules)
        unary = tuple(rule for rule in inventory if rule.arity == _UNARY_ARITY)
        binary = tuple(rule for rule in inventory if rule.arity == _BINARY_ARITY)
        declared = _validate_codes(heads.codes)
        codes = tuple(sorted(declared))
        positions = MappingProxyType({code: at for at, code in enumerate(codes)})
        unary_compiled: list[tuple[Rule[T], tuple[_CodeAlternative, ...] | None]] = []
        binary_compiled: list[tuple[Rule[T], tuple[_CodeAlternative, ...] | None]] = []

        for rule in inventory:
            probe = rule.probe

            if probe is not None and probe.arity != rule.arity:
                msg = (
                    f"rule {rule.name!r} has arity {rule.arity} but its probe "
                    f"has arity {probe.arity}"
                )
                raise ValueError(msg)

            compiled = (rule, None if probe is None else _codes(probe, heads, declared))

            if rule.arity == _UNARY_ARITY:
                unary_compiled.append(compiled)
            elif rule.arity == _BINARY_ARITY:
                binary_compiled.append(compiled)

        ground = _grounded(unary)
        grounded = frozenset(
            id(rule)
            for _, by_ground in ground
            for rules_on in by_ground.values()
            for rule in rules_on
        )
        unary_rows: list[list[Rule[T]]] = [[] for _ in codes]

        for rule, alternatives in unary_compiled:
            if id(rule) in grounded:
                continue
            selected = (
                range(len(codes))
                if alternatives is None
                else sorted(
                    {
                        positions[code]
                        for alternative in alternatives
                        for code in (
                            codes if alternative[0] is None else alternative[0]
                        )
                    }
                )
            )
            for position in selected:
                unary_rows[position].append(rule)

        table_rows: list[list[Rule[T]]] = [[] for _ in range(len(codes) ** 2)]

        for rule, alternatives in binary_compiled:
            selected = (
                range(len(table_rows))
                if alternatives is None
                else sorted(
                    {
                        positions[left] * len(codes) + positions[right]
                        for alternative in alternatives
                        for left in (
                            codes if alternative[0] is None else alternative[0]
                        )
                        for right in (
                            codes if alternative[1] is None else alternative[1]
                        )
                    }
                )
            )
            for position in selected:
                table_rows[position].append(rule)

        unary_table = tuple(tuple(row) for row in unary_rows)
        table = tuple(tuple(row) for row in table_rows)

        joined = tuple(
            rule
            for rule in binary
            if isinstance(rule, BinaryJoin) and rule.matching.has_key
        )
        joined_ids = frozenset(id(rule) for rule in joined)
        unjoined_table = tuple(
            tuple(rule for rule in row if id(rule) not in joined_ids) for row in table
        )

        object.__setattr__(self, "_inventory", inventory)
        object.__setattr__(self, "_joined", joined)
        object.__setattr__(self, "_unjoined_table", unjoined_table)
        object.__setattr__(self, "_unary_rules", unary)
        object.__setattr__(self, "_unary_ground", ground)
        object.__setattr__(
            self,
            "_unary_positions",
            MappingProxyType({id(rule): at for at, rule in enumerate(unary)}),
        )
        object.__setattr__(self, "_rules", binary)
        object.__setattr__(
            self,
            "_always",
            tuple(
                rule for rule, alternatives in binary_compiled if alternatives is None
            ),
        )
        object.__setattr__(
            self,
            "_unary_always",
            tuple(
                rule
                for rule, alternatives in unary_compiled
                if alternatives is None and id(rule) not in grounded
            ),
        )
        object.__setattr__(self, "_heads", heads)
        object.__setattr__(self, "_codes", frozenset(codes))
        object.__setattr__(self, "_positions", positions)
        object.__setattr__(self, "_width", len(codes))
        object.__setattr__(self, "_unary_table", unary_table)
        object.__setattr__(self, "_table", table)

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "a rule index is immutable"

        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "a rule index is immutable"

        raise AttributeError(msg)

    @override
    def __reduce__(
        self,
    ) -> tuple[
        type[RuleIndex[T]],
        tuple[tuple[Rule[T], ...], HeadIndex[T]],
    ]:
        """Return the index reconstruction call."""
        return (RuleIndex, (self._inventory, self._heads))

    @property
    def rules(self) -> tuple[Rule[T], ...]:
        """Return indexed binary rules in declaration order."""
        return self._rules

    @property
    def always(self) -> tuple[Rule[T], ...]:
        """Return binary rules that declined a probe."""
        return self._always

    @property
    def unary_rules(self) -> tuple[Rule[T], ...]:
        """Return indexed unary rules in declaration order."""
        return self._unary_rules

    @property
    def unary_always(self) -> tuple[Rule[T], ...]:
        """Return unary rules that declined a probe."""
        return self._unary_always

    @property
    def heads(self) -> HeadIndex[T]:
        """Return the injected signature-scoped head index."""
        return self._heads

    @property
    def codes(self) -> frozenset[int]:
        """Return the indexed head codes."""
        return self._codes

    @property
    def joined(self) -> tuple[Rule[T], ...]:
        """Return the binary rules that declare join keys under a keyed matching.

        A pair of premises sharing no key of such a rule cannot instantiate it,
        so a caller may join premises on those keys instead of trying the rule
        on every pair the head table admits.
        """
        return self._joined

    def unjoined(self, left_code: int, right_code: int, /) -> tuple[Rule[T], ...]:
        """Select the binary rules without join keys compatible with two heads.

        Parameters
        ----------
        left_code : int
            The left premise's head code.
        right_code : int
            The right premise's head code.

        Returns
        -------
        tuple[Rule[T], ...]
            The head table's candidates outside ``joined``, in declaration
            order.
        """
        left_at = self._position(left_code)
        right_at = self._position(right_code)

        return self._unjoined_table[left_at * self._width + right_at]

    def candidates(self, left: T, right: T, /) -> tuple[Rule[T], ...]:
        """Select binary rules compatible with two term heads."""
        left_at = self._position(self._heads.term(left))
        right_at = self._position(self._heads.term(right))

        return self._table[left_at * self._width + right_at]

    def unary_candidates(self, term: T, /) -> tuple[Rule[T], ...]:
        """Select unary rules compatible with one term, in declaration order.

        A rule whose premise admits one representative is selected by that
        representative rather than by head code, so a term meets only the
        ground rules written for it.

        Parameters
        ----------
        term : T
            The premise.

        Returns
        -------
        tuple[Rule[T], ...]
            The rules the head table admits for the term's head, together with
            the ground rules written for its representative.
        """
        row = self._unary_table[self._position(self._heads.term(term))]
        found = tuple(
            rule
            for matching, by_ground in self._unary_ground
            for rule in by_ground.get(
                matching.key(term) if matching.has_key else term, ()
            )
        )

        if not found:
            return row

        positions = self._unary_positions

        return tuple(sorted((*row, *found), key=lambda rule: positions[id(rule)]))

    def _position(self, code: int) -> int:
        """Return the table position for a declared head code."""
        checked = ensure_non_negative("term head code", code)

        try:
            return self._positions[checked]
        except KeyError:
            msg = f"a term head mapped outside the declared codes: {checked!r}"
            raise ValueError(msg) from None


def _grounded[T](
    rules: Sequence[Rule[T]],
) -> tuple[tuple[Matching[T], Mapping[T, tuple[Rule[T], ...]]], ...]:
    """Group unary rules whose premise admits one representative.

    Rules are grouped by matching object and then by the representative their
    premise admits, in declaration order.
    """
    groups: dict[int, tuple[Matching[T], dict[T, list[Rule[T]]]]] = {}

    for rule in rules:
        if not isinstance(rule, GroundPremises):
            continue

        (ground,) = rule.ground_premises

        if ground is None:
            continue

        matching = rule.matching
        _, by_ground = groups.setdefault(id(matching), (matching, {}))
        by_ground.setdefault(ground, []).append(rule)

    return tuple(
        (
            matching,
            MappingProxyType({key: tuple(held) for key, held in by_ground.items()}),
        )
        for matching, by_ground in groups.values()
    )


def _validate_codes(codes: Iterable[int]) -> frozenset[int]:
    """Validate a nonempty set of nonnegative integer head codes."""
    held = frozenset(codes)

    if not held:
        msg = "a head index declares no codes"
        raise ValueError(msg)

    for code in held:
        ensure_non_negative("head code", code)

    return held

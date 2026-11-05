"""Named rule inventories."""

from itertools import product
from json import dumps
from typing import TYPE_CHECKING, Final, Never, Protocol, final, override

from lofi._checks import ensure_name, ensure_non_negative, ensure_positive
from lofi._frozen import FrozenValue
from lofi.judge._errors import RuleSetValidationError
from lofi.judge._ports import DeclaredRelation
from lofi.judge._recover import recover
from lofi.judge._values import (
    RESTRICTION_MODES,
    ClosureReport,
    Diagnostic,
)
from lofi.terms import Identity

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

    from lofi._types import JsonValue
    from lofi.judge._index import RuleIndex
    from lofi.judge._ports import Direction, NormalForm, Rule
    from lofi.judge._values import RestrictionMode
    from lofi.terms import Matching

    class _CategoryMeasure[T](Protocol):
        """Owner-supplied structural measures of category terms."""

        def size(self, term: T, /) -> int:
            """Return the category's exact node count."""

        def arity(self, term: T, /) -> int:
            """Return the category's formalism-declared functor arity."""


__all__ = [
    "FORWARD_ENUMERATION",
    "ForwardEnumeration",
    "GenerativeParsing",
    "RuleRequirement",
    "RuleSet",
    "refuse_coinciding_rules",
]

_BINARY_ARITY: Final = 2
"""Indexed rule arity."""

type _RuleSetFields[T] = tuple[
    str,
    tuple[Rule[T], ...],
    Matching[T],
    NormalForm[T],
    RestrictionMode,
    tuple[tuple[T, ...] | None, ...],
    tuple[tuple[T, ...] | None, ...],
    RuleIndex[T] | None,
]
"""Fields needed to reconstruct a rule set."""

type _RuleSetReduction[T] = tuple[
    Callable[[_RuleSetFields[T]], RuleSet[T]],
    tuple[_RuleSetFields[T]],
]
"""Rule-set pickle reduction."""


def _has_complete_forward[T](rule: Rule[T]) -> bool:
    """Check whether a rule completely enumerates its conclusions."""
    return "forward" in rule.directions and rule.forward_is_complete


class RuleRequirement(Protocol):
    """A capability every rule of an inventory must have for one algorithm."""

    @property
    def name(self) -> str:
        """Return the requirement's name, as diagnostics report it."""
        ...

    def diagnose[T](self, rule: Rule[T], /) -> Diagnostic | None:
        """Return the finding for one rule, or None when it meets the requirement.

        Parameters
        ----------
        rule : Rule[T]
            The rule to check.
        """
        ...


@final
class ForwardEnumeration(FrozenValue):
    """Every rule enumerates its conclusions completely from its premises."""

    __slots__ = ()

    _noun = "forward-enumeration requirement"

    @property
    def name(self) -> str:
        """Return the requirement's name."""
        return "forward_enumeration"

    def diagnose[T](self, rule: Rule[T], /) -> Diagnostic | None:
        """Report a rule that lacks complete forward enumeration."""
        if _has_complete_forward(rule):
            return None

        return Diagnostic(
            "forward_enumeration_unavailable",
            self.name,
            f"rule {rule.name!r} lacks complete forward enumeration",
            rules=(rule.name,),
        )

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds the requirement."""
        return "ForwardEnumeration()"


FORWARD_ENUMERATION: Final = ForwardEnumeration()
"""The requirement closure and forward-only parsing place on every rule."""


@final
class GenerativeParsing(FrozenValue):
    """Every rule generates its conclusions, directly or through a proposer.

    Parameters
    ----------
    proposer_available : bool
        Whether the parse has a checked proposer for check-only rules.

    Raises
    ------
    TypeError
        If ``proposer_available`` is not a boolean.
    """

    __slots__ = ("_proposer_available",)

    _noun = "generative-parsing requirement"

    def __init__(self, *, proposer_available: bool) -> None:
        if not isinstance(proposer_available, bool):
            msg = f"proposer availability is not a boolean: {proposer_available!r}"
            raise TypeError(msg)

        object.__setattr__(self, "_proposer_available", proposer_available)

    @property
    def name(self) -> str:
        """Return the requirement's name."""
        return "generative_parsing"

    @property
    def proposer_available(self) -> bool:
        """Report whether the parse has a checked proposer."""
        return self._proposer_available

    def diagnose[T](self, rule: Rule[T], /) -> Diagnostic | None:
        """Report a rule the parse cannot draw conclusions from."""
        if _has_complete_forward(rule) or (
            "check" in rule.directions and self._proposer_available
        ):
            return None

        if "check" in rule.directions:
            return Diagnostic(
                "proposer_required",
                self.name,
                f"rule {rule.name!r} requires a checked proposer",
                rules=(rule.name,),
            )

        return Diagnostic(
            "generation_unavailable",
            self.name,
            f"rule {rule.name!r} cannot generate conclusions",
            rules=(rule.name,),
        )

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds the requirement."""
        return f"GenerativeParsing(proposer_available={self._proposer_available!r})"


def _premise_tuples[T](
    reached: tuple[T, ...],
    frontier: set[T],
    arity: int,
    *,
    first_step: bool,
) -> Iterator[tuple[T, ...]]:
    """Yield premise tuples containing a frontier category."""
    if arity == 0:
        if first_step:
            yield ()

        return

    for inputs in product(reached, repeat=arity):
        if any(term in frontier for term in inputs):
            yield inputs


def _differences(left: Mapping[str, JsonValue], right: Mapping[str, JsonValue]) -> str:
    """Name the declared parameters on which two declarations disagree."""
    differing = sorted(
        key for key in left.keys() | right.keys() if left.get(key) != right.get(key)
    )

    return ", ".join(repr(key) for key in differing) or "nothing"


def refuse_coinciding_rules[T](name: str, rules: Iterable[Rule[T]], /) -> None:
    """Refuse an inventory in which two rules denote the same relation.

    Coincidence is read from the relation each rule declares, so it follows
    from the rules' parameters and never from their names. A rule that
    declares no relation coincides with no other.

    Parameters
    ----------
    name : str
        The inventory's name, for the message.
    rules : Iterable[Rule[T]]
        The rules in declaration order.

    Raises
    ------
    ValueError
        If two rules declare equal relations. The message names every such
        pair and the declared parameters they differ in, none of which
        changes the relation.
    """
    seen: dict[str, Rule[T]] = {}
    pairs: list[str] = []

    for rule in rules:
        if not isinstance(rule, DeclaredRelation):
            continue

        relation = dumps(rule.relation, sort_keys=True, separators=(",", ":"))
        earlier = seen.get(relation)

        if earlier is None:
            seen[relation] = rule
            continue

        pairs.append(
            f"{earlier.name!r} and {rule.name!r} differ only in "
            f"{_differences(earlier.to_spec(), rule.to_spec())}"
        )

    if pairs:
        msg = (
            f"rule set {name!r} declares the same relation twice: "
            f"{'; '.join(pairs)}, which does not change the relation over "
            "categories"
        )
        raise ValueError(msg)


def _category_rows[T](
    field: str,
    rules: tuple[Rule[T], ...],
    declared: Sequence[Sequence[T] | None] | None,
) -> tuple[tuple[T, ...] | None, ...]:
    """Validate finite rule-category declarations in rule order."""
    if declared is None:
        return tuple(None for _ in rules)

    rows = tuple(None if row is None else tuple(row) for row in declared)

    if len(rows) != len(rules):
        msg = (
            f"{field} category declarations have length {len(rows)}, "
            f"not the rule count {len(rules)}"
        )
        raise ValueError(msg)

    for position, row in enumerate(rows):
        if row == ():
            msg = f"rule {position}'s {field} categories are empty"
            raise ValueError(msg)
        if row is not None and any(
            category == earlier
            for at, category in enumerate(row)
            for earlier in row[:at]
        ):
            msg = f"rule {position}'s {field} categories repeat a category"
            raise ValueError(msg)

    return rows


class RuleSet[T]:
    """Named rule inventory.

    Parameters
    ----------
    name : str
        Stable inventory name.
    rules : Sequence[Rule[T]]
        Rules in declaration order.
    matching : Matching[T]
        The grammar's congruence, which goals and consumers compare
        categories under. A rule may match its premises under its own
        congruence.
    normal_form : NormalForm[T]
        Derivation-step filter.
    restriction_mode : RestrictionMode
        Whether rule instances are unrestricted or grammar-specific.
    introduces : Sequence[Sequence[T] | None] | None
        Exhaustive finite conclusions in rule order. ``None`` leaves a rule's
        conclusions unrestricted.
    consumes : Sequence[Sequence[T] | None] | None
        Finite categories of which each application consumes at least one, in
        rule order. ``None`` leaves a rule's premises unrestricted.
    index : RuleIndex[T] | None
        Optional index over the unary and binary rules. It must cover both,
        in declaration order.

    Raises
    ------
    TypeError
        If ``name`` has the wrong type.
    ValueError
        If the name, restriction mode, rule inventory, or index is invalid.
    """

    __slots__ = (
        "_consumes",
        "_index",
        "_introduces",
        "_matching",
        "_name",
        "_normal_form",
        "_packing",
        "_restriction_mode",
        "_rules",
    )

    def __init__(
        self,
        name: str,
        rules: Sequence[Rule[T]],
        matching: Matching[T],
        normal_form: NormalForm[T],
        restriction_mode: RestrictionMode = "unrestricted",
        *,
        introduces: Sequence[Sequence[T] | None] | None = None,
        consumes: Sequence[Sequence[T] | None] | None = None,
        index: RuleIndex[T] | None = None,
    ) -> None:
        object.__setattr__(self, "_name", ensure_name("a rule set's name", name))

        if restriction_mode not in RESTRICTION_MODES:
            declared = ", ".join(RESTRICTION_MODES)
            msg = (
                f"the rule set {self._name!r} declares restriction mode "
                f"{restriction_mode!r}, not one of {declared}"
            )
            raise ValueError(msg)

        inventory = tuple(rules)
        introduced = _category_rows("introduced", inventory, introduces)
        consumed = _category_rows("consumed", inventory, consumes)
        seen: dict[str, int] = {}

        for position, rule in enumerate(inventory):
            if rule.name in seen:
                msg = (
                    f"the rule set {self._name!r} names {rule.name!r} twice, "
                    f"at positions {seen[rule.name]} and {position}"
                )
                raise ValueError(msg)

            if restriction_mode == "unrestricted" and rule.restriction is not None:
                msg = (
                    f"the unrestricted rule set {self._name!r} contains "
                    f"restricted rule {rule.name!r}"
                )
                raise ValueError(msg)

            seen[rule.name] = position

        binary = tuple(rule for rule in inventory if rule.arity == _BINARY_ARITY)
        unary = tuple(rule for rule in inventory if rule.arity == 1)

        if index is not None and (
            len(index.rules) != len(binary)
            or any(
                left is not right
                for left, right in zip(index.rules, binary, strict=True)
            )
        ):
            msg = "the rule index does not cover the binary rules in declaration order"
            raise ValueError(msg)

        if index is not None and (
            len(index.unary_rules) != len(unary)
            or any(
                left is not right
                for left, right in zip(index.unary_rules, unary, strict=True)
            )
        ):
            msg = "the rule index does not cover the unary rules in declaration order"
            raise ValueError(msg)

        object.__setattr__(self, "_rules", inventory)
        object.__setattr__(self, "_introduces", introduced)
        object.__setattr__(self, "_consumes", consumed)
        object.__setattr__(self, "_matching", matching)
        object.__setattr__(
            self,
            "_packing",
            matching
            if all(rule.matching is matching for rule in inventory)
            else Identity[T](),
        )
        object.__setattr__(self, "_normal_form", normal_form)
        object.__setattr__(self, "_restriction_mode", restriction_mode)
        object.__setattr__(self, "_index", index)

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "a rule set is immutable"

        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "a rule set is immutable"

        raise AttributeError(msg)

    @property
    def name(self) -> str:
        """Return the inventory name."""
        return self._name

    @property
    def rules(self) -> tuple[Rule[T], ...]:
        """Return rules in declaration order."""
        return self._rules

    @property
    def introduces(self) -> tuple[tuple[T, ...] | None, ...]:
        """Return finite possible conclusions aligned with the rules."""
        return self._introduces

    @property
    def consumes(self) -> tuple[tuple[T, ...] | None, ...]:
        """Return finite consumed categories aligned with the rules."""
        return self._consumes

    @property
    def matching(self) -> Matching[T]:
        """Return the grammar's term congruence."""
        return self._matching

    @property
    def packing(self) -> Matching[T]:
        """Return the congruence conclusions are packed and deduplicated under.

        It is the grammar's congruence when every rule matches its premises
        under that congruence, and structural identity otherwise, so that a
        rule matching more finely than the grammar sees every category it
        could tell apart.
        """
        return self._packing

    @property
    def normal_form(self) -> NormalForm[T]:
        """Return the normal-form filter."""
        return self._normal_form

    @property
    def restriction_mode(self) -> RestrictionMode:
        """Return the restriction mode."""
        return self._restriction_mode

    @property
    def index(self) -> RuleIndex[T] | None:
        """Return the unary- and binary-rule index, if configured."""
        return self._index

    def with_normal_form(self, normal_form: NormalForm[T], /) -> RuleSet[T]:
        """Return this inventory under another derivation-step filter.

        The rules, matching, restrictions, finite category declarations, and
        optional rule index retain their exact identities. Only the normal-form
        automaton changes.

        Parameters
        ----------
        normal_form : NormalForm[T]
            Replacement derivation-step filter.

        Returns
        -------
        RuleSet[T]
            The same declared inventory under ``normal_form``.
        """
        return RuleSet(
            self._name,
            self._rules,
            self._matching,
            normal_form,
            self._restriction_mode,
            introduces=self._introduces,
            consumes=self._consumes,
            index=self._index,
        )

    def of_arity(self, k: int) -> tuple[Rule[T], ...]:
        """Return rules taking a given number of premises.

        Parameters
        ----------
        k : int
            Required rule arity.

        Returns
        -------
        tuple[Rule[T], ...]
            Matching rules in declaration order.

        Raises
        ------
        TypeError
            If ``k`` is not an integer or is a boolean.
        ValueError
            If ``k`` is negative.
        """
        arity = ensure_non_negative("a selected arity", k)

        return tuple(rule for rule in self._rules if rule.arity == arity)

    def supporting(self, direction: Direction) -> tuple[Rule[T], ...]:
        """Return rules computing a given direction.

        Parameters
        ----------
        direction : Direction
            Computation direction to select.

        Returns
        -------
        tuple[Rule[T], ...]
            Supporting rules in declaration order.
        """
        return tuple(rule for rule in self._rules if direction in rule.directions)

    def recover(self, inputs: Sequence[T], output: T, /) -> tuple[Rule[T], ...]:
        """Return rules that relate premises to a conclusion.

        Parameters
        ----------
        inputs : Sequence[T]
            Premises in rule order.
        output : T
            Candidate conclusion.

        Returns
        -------
        tuple[Rule[T], ...]
            Matching rules in declaration order.
        """
        return recover(self._rules, inputs, output)

    def validate(
        self,
        requirement: RuleRequirement,
        *,
        strict: bool = True,
    ) -> tuple[Diagnostic, ...]:
        """Check whether every rule meets a requirement.

        Parameters
        ----------
        requirement : RuleRequirement
            The capability an algorithm needs of every rule.
        strict : bool
            Whether findings raise instead of being returned.

        Returns
        -------
        tuple[Diagnostic, ...]
            Findings in rule declaration order.

        Raises
        ------
        TypeError
            If ``strict`` is not a boolean.
        RuleSetValidationError
            If ``strict`` is true and findings exist.
        """
        if not isinstance(strict, bool):
            msg = f"strict validation is not a boolean: {strict!r}"
            raise TypeError(msg)

        findings = tuple(
            diagnostic
            for rule in self._rules
            if (diagnostic := requirement.diagnose(rule)) is not None
        )

        if strict and findings:
            raise RuleSetValidationError(
                ruleset=self._name,
                requirement=requirement.name,
                diagnostics=findings,
            )

        return findings

    def closure(
        self,
        seeds: Iterable[T],
        *,
        max_depth: int,
        budget: int,
        category_measure: _CategoryMeasure[T] | None = None,
        max_category_arity: int | None = None,
        max_category_size: int | None = None,
    ) -> ClosureReport[T]:
        """Compute a restriction-relaxed structural closure upper bound.

        Parameters
        ----------
        seeds : Iterable[T]
            Initial categories.
        max_depth : int
            Maximum number of chaining rounds.
        budget : int
            Maximum number of reached representatives.
        category_measure : _CategoryMeasure[T] | None
            Owner-declared structural measures, required by either category bound.
        max_category_arity : int | None
            Greatest admitted functor-spine arity, or no arity bound.
        max_category_size : int | None
            Greatest admitted exact category node count, or no size bound.

        Returns
        -------
        ClosureReport[T]
            Reached representatives and stopping reason.

        Raises
        ------
        TypeError
            If a bound has the wrong type.
        ValueError
            If a bound is negative or matching has no representative key.
        RuleSetValidationError
            If a rule lacks complete forward enumeration.
        """
        depth_limit = ensure_non_negative("closure depth", max_depth)
        node_limit = ensure_non_negative("closure node budget", budget)
        arity_limit = (
            None
            if max_category_arity is None
            else ensure_positive("maximum category arity", max_category_arity)
        )
        size_limit = (
            None
            if max_category_size is None
            else ensure_positive("maximum category size", max_category_size)
        )

        if (arity_limit is not None or size_limit is not None) and (
            category_measure is None
        ):
            msg = "category bounds need the grammar's declared category measure"
            raise ValueError(msg)

        if not self._packing.has_key:
            msg = (
                f"rule set {self._name!r} cannot compute erased closure because "
                f"matching {self._packing.name!r} has no representative key"
            )
            raise ValueError(msg)

        search = _ClosureSearch(
            self,
            depth_limit,
            node_limit,
            category_measure,
            arity_limit,
            size_limit,
        )

        if depth_limit == 0:
            return search.seed_only(seeds)

        self.validate(FORWARD_ENUMERATION)
        return search.run(seeds)

    @override
    def __reduce__(self) -> _RuleSetReduction[T]:
        """Return the rule-set reconstruction call."""
        fields: _RuleSetFields[T] = (
            self._name,
            self._rules,
            self._matching,
            self._normal_form,
            self._restriction_mode,
            self._introduces,
            self._consumes,
            self._index,
        )

        return (_restore_ruleset, (fields,))

    def __iter__(self) -> Iterator[Rule[T]]:
        """Iterate over rules in declaration order."""
        return iter(self._rules)

    def __len__(self) -> int:
        """Return the rule count."""
        return len(self._rules)

    @override
    def __repr__(self) -> str:
        """Render the inventory summary."""
        return (
            f"RuleSet(name={self._name!r}, rules={len(self._rules)}, "
            f"matching={self._matching.name!r}, "
            f"normal_form={self._normal_form.name!r}, "
            f"restriction_mode={self._restriction_mode!r})"
        )


def _restore_ruleset[T](fields: _RuleSetFields[T]) -> RuleSet[T]:
    """Reconstruct a rule set from frozen fields."""
    (
        name,
        rules,
        matching,
        normal_form,
        restriction_mode,
        introduces,
        consumes,
        index,
    ) = fields

    return RuleSet(
        name,
        rules,
        matching,
        normal_form,
        restriction_mode,
        introduces=introduces,
        consumes=consumes,
        index=index,
    )


class _ClosureSearch[T]:
    """Bounded semi-naive closure state."""

    __slots__ = (
        "_by_arity",
        "_depth_limit",
        "_fired",
        "_index",
        "_known",
        "_matching",
        "_max_arity",
        "_max_size",
        "_measure",
        "_node_limit",
        "_reached",
        "_rules",
        "_steps",
    )

    def __init__(
        self,
        ruleset: RuleSet[T],
        depth_limit: int,
        node_limit: int,
        measure: _CategoryMeasure[T] | None,
        max_arity: int | None,
        max_size: int | None,
    ) -> None:
        self._rules = ruleset.rules
        self._matching = ruleset.packing
        self._index = ruleset.index
        self._depth_limit = depth_limit
        self._node_limit = node_limit
        self._measure = measure
        self._max_arity = max_arity
        self._max_size = max_size
        self._reached: list[T] = []
        self._known: set[T] = set()
        self._fired: set[str] = set()
        self._steps = 0
        self._by_arity: dict[int, tuple[Rule[T], ...]] = {}

        for rule in self._rules:
            arity = ensure_non_negative(f"rule {rule.name!r} arity", rule.arity)
            self._by_arity[arity] = (*self._by_arity.get(arity, ()), rule)

    def _report(
        self,
        *,
        saturated: bool,
        escaping: Iterable[T] = (),
    ) -> ClosureReport[T]:
        """Build a report in rule declaration order."""
        return ClosureReport(
            self._reached,
            saturated=saturated,
            steps=self._steps,
            budget=self._node_limit,
            never_applicable=(
                rule.name for rule in self._rules if rule.name not in self._fired
            ),
            escaping=escaping,
        )

    def _add_seeds(self, seeds: Iterable[T]) -> ClosureReport[T] | None:
        """Add erased seeds until the node budget is full."""
        for seed in seeds:
            if not self._admitted(seed):
                return self._report(saturated=False, escaping=(seed,))

            representative = self._matching.key(seed)

            if representative in self._known:
                continue

            if len(self._reached) == self._node_limit:
                return self._report(saturated=False, escaping=(representative,))

            self._known.add(representative)
            self._reached.append(representative)

        return None

    def _admitted(self, category: T) -> bool:
        """Return whether a category lies within both declared dimensions."""
        measure = self._measure

        if measure is None:
            return True

        return (
            self._max_arity is None or measure.arity(category) <= self._max_arity
        ) and (self._max_size is None or measure.size(category) <= self._max_size)

    def _candidates(
        self,
        arity: int,
        rules: tuple[Rule[T], ...],
        inputs: tuple[T, ...],
    ) -> tuple[Rule[T], ...]:
        """Return rules selected for one premise tuple."""
        if self._index is not None:
            if arity == _BINARY_ARITY:
                return self._index.candidates(inputs[0], inputs[1])
            if arity == 1:
                return self._index.unary_candidates(inputs[0])

        return rules

    def _consider(
        self,
        rule: Rule[T],
        output: T,
        fresh: set[T],
    ) -> ClosureReport[T] | None:
        """Add one new representative or report its escape."""
        self._fired.add(rule.name)

        if not self._admitted(output):
            return self._report(saturated=False, escaping=(output,))

        representative = self._matching.key(output)

        if representative in self._known:
            return None

        if self._steps == self._depth_limit or (len(self._reached) == self._node_limit):
            return self._report(saturated=False, escaping=(representative,))

        self._known.add(representative)
        self._reached.append(representative)
        fresh.add(representative)

        return None

    def _advance(self, frontier: set[T]) -> tuple[set[T], ClosureReport[T] | None]:
        """Run one semi-naive chaining step."""
        held = tuple(self._reached)
        fresh: set[T] = set()

        for arity, rules in self._by_arity.items():
            for inputs in _premise_tuples(
                held,
                frontier,
                arity,
                first_step=self._steps == 0,
            ):
                for rule in self._candidates(arity, rules, inputs):
                    for output in rule.apply(inputs):
                        stopped = self._consider(rule, output, fresh)

                        if stopped is not None:
                            return (fresh, stopped)

        return (fresh, None)

    def run(self, seeds: Iterable[T]) -> ClosureReport[T]:
        """Run closure to a fixpoint or the first explicit bound."""
        stopped = self._add_seeds(seeds)

        if stopped is not None:
            return stopped

        frontier = set(self._reached)

        while True:
            advanced, stopped = self._advance(frontier)

            if stopped is not None:
                return stopped

            if not advanced:
                return self._report(saturated=True)

            frontier = advanced
            self._steps += 1

    def seed_only(self, seeds: Iterable[T]) -> ClosureReport[T]:
        """Return the erased seed inventory without exercising any rule."""
        stopped = self._add_seeds(seeds)

        if stopped is not None:
            return stopped

        return self._report(saturated=not self._rules)

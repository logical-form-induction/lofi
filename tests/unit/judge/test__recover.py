from itertools import permutations
from typing import TYPE_CHECKING, Final

from lofi.judge import NormalFormRole
from lofi.judge._recover import recover

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from lofi._types import JsonValue
    from lofi.judge._index import RuleProbe
    from lofi.judge._ports import Direction, Rule, Slot
    from lofi.judge._values import RuleRestriction

_CHECKING: Final[frozenset[Direction]] = frozenset({"check"})
_FORWARD: Final[frozenset[Direction]] = frozenset({"forward"})


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


class Relation:
    """A check-only arithmetic relation.

    Parameters
    ----------
    name
        The rule name.
    arity
        The number of premises.
    holds
        The predicate over premises and a conclusion.
    directions
        The directions in which the rule computes conclusions.
    """

    def __init__(
        self,
        name: str,
        arity: int,
        holds: Callable[[tuple[int, ...], int], bool],
        directions: frozenset[Direction] = _CHECKING,
    ) -> None:
        """Bind the rule to one relation."""
        self._name = name
        self._arity = arity
        self._holds = holds
        self._directions = directions

    @property
    def name(self) -> str:
        """Name the rule."""
        return self._name

    @property
    def arity(self) -> int:
        """Take as many premises as the relation was declared over."""
        return self._arity

    @property
    def directions(self) -> frozenset[Direction]:
        """Report the declared directions."""
        return self._directions

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
        """Report incomplete forward enumeration."""
        return False

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
        """Yield no forward results."""
        del inputs

        return iter(())

    def admits(self, inputs: tuple[int, ...], output: int, /) -> bool:
        """Decide the relation and reject invalid arity."""
        return len(inputs) == self._arity and self._holds(inputs, output)

    def residuate(self, output: int, known: tuple[Slot[int], ...], /) -> Iterator[int]:
        """Yield no residuals."""
        del output, known

        return iter(())

    def to_spec(self) -> Mapping[str, JsonValue]:
        """Return the declaration this rule was built from."""
        return {"rule": self._name}


def _doubling() -> Relation:
    return Relation("doubling", 1, lambda inputs, output: output == inputs[0] * 2)


def _squaring() -> Relation:
    return Relation("squaring", 1, lambda inputs, output: output == inputs[0] ** 2)


def _successor() -> Relation:
    return Relation("successor", 1, lambda inputs, output: output == inputs[0] + 1)


def _difference() -> Relation:
    return Relation(
        "difference", 2, lambda inputs, output: output == inputs[0] - inputs[1]
    )


def _names(rules: tuple[Rule[int], ...]) -> tuple[str, ...]:
    return tuple(rule.name for rule in rules)


def test_recovery_returns_every_rule_that_relates_the_step() -> None:
    inventory = (_doubling(), _squaring(), _successor())

    assert _names(recover(inventory, [2], 4)) == ("doubling", "squaring")


def test_recovery_returns_one_rule_where_only_one_relates_the_step() -> None:
    inventory = (_doubling(), _squaring(), _successor())

    assert _names(recover(inventory, [3], 6)) == ("doubling",)


def test_permuting_the_inventory_leaves_the_recovered_set_unchanged() -> None:
    inventory = (_doubling(), _squaring(), _successor())
    recovered = {
        frozenset(_names(recover(order, [2], 4))) for order in permutations(inventory)
    }

    assert recovered == {frozenset({"doubling", "squaring"})}


def test_the_recovered_rules_come_back_in_the_order_the_inventory_gave_them() -> None:
    forward = (_doubling(), _squaring())
    backward = (_squaring(), _doubling())

    assert _names(recover(forward, [2], 4)) == ("doubling", "squaring")
    assert _names(recover(backward, [2], 4)) == ("squaring", "doubling")


def test_a_step_no_rule_relates_recovers_nothing() -> None:
    inventory = (_doubling(), _squaring(), _successor())

    assert recover(inventory, [3], 7) == ()


def test_a_step_of_an_arity_no_rule_takes_recovers_nothing() -> None:
    inventory = (_doubling(), _difference())

    assert recover(inventory, [1, 2, 3], 4) == ()


def test_a_binary_rule_is_not_recovered_from_a_single_premise() -> None:
    assert recover((_difference(),), [0], 0) == ()


def test_the_premises_are_read_in_the_order_the_step_recorded_them() -> None:
    inventory = (_difference(),)

    assert _names(recover(inventory, [5, 3], 2)) == ("difference",)
    assert recover(inventory, [3, 5], 2) == ()


def test_a_rule_that_does_not_check_is_never_recovered() -> None:
    everything = Relation("everything", 1, lambda _inputs, _output: True, _FORWARD)
    inventory = (_doubling(), everything)

    assert _names(recover(inventory, [2], 4)) == ("doubling",)
    assert recover(inventory, [3], 7) == ()


def test_an_empty_inventory_recovers_nothing() -> None:
    assert recover((), [2], 4) == ()


def test_recovery_iterates_the_inventory_once() -> None:
    inventory = iter((_doubling(), _squaring()))

    assert _names(recover(inventory, [2], 4)) == ("doubling", "squaring")

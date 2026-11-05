"""Lexical category inventories generated from the inverse of a rule set.

An inventory is grown top down. Each draw starts a derivation at a goal
category and expands a node by choosing a rule and one of its residual premise
slots, filling the rule's other premises from a pool of known categories and
residuating the chosen slot against the node. A node left unexpanded is a
leaf, and the leaves of a finished derivation are lexical categories. Every
category of an inventory is therefore a leaf of a derivation of a goal category
whose other leaves are in the inventory too, and that derivation is kept as
the category's witness.
"""

from typing import TYPE_CHECKING, final

from lofi._frozen import FrozenValue
from lofi.judge import HOLE
from lofi.spec._errors import InventoryExhaustedError

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from random import Random

    from lofi.judge import Rule, RuleSet, Slot
    from lofi.spec.grammar._values import CategoryMeasure
    from lofi.terms import Term

__all__ = [
    "CategoryInventory",
    "InventoryDerivation",
    "category_depth",
    "derives",
    "generate_inventory",
]

type _Shape = tuple[str, tuple[_Shape, ...]]
"""An operator name over the shapes of its arguments."""


def category_depth(category: Term, /) -> int:
    """Return how deeply constructors nest in a category.

    Parameters
    ----------
    category : Term
        The category to measure.

    Returns
    -------
    int
        Zero for a category with no arguments, and otherwise one more than
        the deepest argument, so ``S/NP`` has depth one and ``(S/NP)/NP``
        depth two.
    """
    return 1 + max(map(category_depth, category.args)) if category.args else 0


def _shape(category: Term, /) -> _Shape:
    """Return the store-independent shape a category sorts by.

    Parameters
    ----------
    category : Term
        The category to describe.
    """
    return (category.op.name, tuple(_shape(argument) for argument in category.args))


@final
class InventoryDerivation[T](FrozenValue):
    """A derivation of a category whose leaves are lexical categories.

    Parameters
    ----------
    category : T
        The category the derivation concludes.
    rule : Rule[T] | None
        The rule of the last step, or nothing for a lexical leaf.
    premises : tuple[InventoryDerivation[T], ...]
        The derivations of the step's premises in rule order, none for a leaf.
    """

    _noun = "inventory derivation"

    __slots__ = ("_category", "_premises", "_rule")

    def __init__(
        self,
        category: T,
        rule: Rule[T] | None = None,
        premises: tuple[InventoryDerivation[T], ...] = (),
    ) -> None:
        object.__setattr__(self, "_category", category)
        object.__setattr__(self, "_rule", rule)
        object.__setattr__(self, "_premises", premises)

    @property
    def category(self) -> T:
        """Return the category the derivation concludes."""
        return self._category

    @property
    def rule(self) -> Rule[T] | None:
        """Return the rule of the last step, or nothing for a leaf."""
        return self._rule

    @property
    def premises(self) -> tuple[InventoryDerivation[T], ...]:
        """Return the premise derivations in rule order."""
        return self._premises

    def leaves(self) -> Iterator[T]:
        """Yield the lexical categories of the derivation from left to right."""
        if self._rule is None:
            yield self._category

            return

        for premise in self._premises:
            yield from premise.leaves()


@final
class CategoryInventory[T](FrozenValue):
    """Lexical categories beside the derivations that make each one usable.

    Parameters
    ----------
    categories : tuple[T, ...]
        The categories in the order they were found.
    witnesses : tuple[InventoryDerivation[T], ...]
        For each category, in the same order, a derivation of a goal
        category with that category among its leaves and every leaf in the
        inventory.
    draws : int
        How many derivations were drawn to find the categories, the rejected
        ones included.
    """

    _noun = "category inventory"

    __slots__ = ("_categories", "_draws", "_witnesses")

    def __init__(
        self,
        categories: tuple[T, ...],
        witnesses: tuple[InventoryDerivation[T], ...],
        *,
        draws: int,
    ) -> None:
        object.__setattr__(self, "_categories", categories)
        object.__setattr__(self, "_witnesses", witnesses)
        object.__setattr__(self, "_draws", draws)

    @property
    def categories(self) -> tuple[T, ...]:
        """Return the categories in the order they were found."""
        return self._categories

    @property
    def witnesses(self) -> tuple[InventoryDerivation[T], ...]:
        """Return each category's witness derivation, in category order."""
        return self._witnesses

    @property
    def draws(self) -> int:
        """Return how many derivations were drawn, the rejected ones included."""
        return self._draws


def _tag[T](derivation: InventoryDerivation[T], rules: RuleSet[T], /) -> str | None:
    """Return a derivation's normal-form tag, or nothing if a step is refused.

    Parameters
    ----------
    derivation : InventoryDerivation[T]
        The derivation to check bottom up.
    rules : RuleSet[T]
        The rule set whose rules and normal form each step must satisfy.
    """
    normal_form = rules.normal_form
    rule = derivation.rule

    if rule is None:
        return normal_form.lexical_tag()

    children: list[str] = []

    for premise in derivation.premises:
        child = _tag(premise, rules)

        if child is None:
            return None

        children.append(child)

    tags = tuple(children)
    inputs = tuple(premise.category for premise in derivation.premises)

    if rule not in rules.rules or not rule.admits(inputs, derivation.category):
        return None

    if not normal_form.admits(rule, tags, inputs, derivation.category):
        return None

    return normal_form.tag(rule, tags)


def derives[T](derivation: InventoryDerivation[T], rules: RuleSet[T], /) -> bool:
    """Return whether every step of a derivation is one the rule set admits.

    Parameters
    ----------
    derivation : InventoryDerivation[T]
        The derivation to check.
    rules : RuleSet[T]
        The rule set whose rules must admit each step and whose normal form
        must keep it.
    """
    return _tag(derivation, rules) is not None


@final
class _Grower:
    """Grows derivations top down through the residuals of a rule set.

    Parameters
    ----------
    rules : RuleSet[Term]
        The rules expanded through.
    primitives : Sequence[Term]
        The categories the premise pool starts with.
    category_measure : CategoryMeasure
        The grammar-owned functor roles used to measure arguments and order.
    max_depth : int
        The deepest category a premise or leaf may be.
    max_arguments : int | None
        The greatest argument count admitted, or no bound.
    max_order : int | None
        The greatest category order admitted, or no bound.
    derivation_depth : int
        The height below which a node may be expanded.
    expansion : float
        The chance a node below the root is expanded.
    patience : int
        How many expansions of the root are attempted before it is left a leaf.
    random : Random
        The source of every choice.
    """

    __slots__ = (
        "_category_measure",
        "_derivation_depth",
        "_expansion",
        "_max_arguments",
        "_max_depth",
        "_max_order",
        "_options",
        "_patience",
        "_pool",
        "_random",
    )

    def __init__(
        self,
        rules: RuleSet[Term],
        primitives: Sequence[Term],
        *,
        category_measure: CategoryMeasure,
        max_depth: int,
        max_arguments: int | None,
        max_order: int | None,
        derivation_depth: int,
        expansion: float,
        patience: int,
        random: Random,
    ) -> None:
        self._options = tuple(
            (rule, slot)
            for rule in rules.rules
            if "residual" in rule.directions
            for slot in sorted(rule.residual_slots)
        )
        self._pool = list(dict.fromkeys(primitives))
        self._category_measure = category_measure
        self._max_depth = max_depth
        self._max_arguments = max_arguments
        self._max_order = max_order
        self._derivation_depth = derivation_depth
        self._expansion = expansion
        self._patience = patience
        self._random = random

    def know(self, category: Term, /) -> None:
        """Add a category to the pool premises are drawn from.

        Parameters
        ----------
        category : Term
            A category newly admitted to the inventory.
        """
        if category not in self._pool:
            self._pool.append(category)

    def _admissible(self, category: Term, /) -> bool:
        """Return whether a lexical category lies within every declared bound."""
        return (
            category_depth(category) <= self._max_depth
            and (
                self._max_arguments is None
                or self._category_measure.arity(category) <= self._max_arguments
            )
            and (
                self._max_order is None
                or self._category_measure.order(category) <= self._max_order
            )
        )

    def grow(self, category: Term, height: int, /) -> InventoryDerivation[Term] | None:
        """Return a derivation of a category, or nothing if none was found.

        The root is expanded by up to ``patience`` attempts, each drawing its
        premises afresh, and is left a leaf only if every attempt fails. A node
        below the root is expanded with chance ``expansion`` by one attempt.

        Parameters
        ----------
        category : Term
            The category to derive.
        height : int
            How many steps lie between the node and the root.
        """
        if height < self._derivation_depth:
            attempts = (
                self._patience
                if height == 0
                else int(self._random.random() < self._expansion)
            )

            for _ in range(attempts):
                expanded = self._expand(category, height)

                if expanded is not None:
                    return expanded

        if not self._admissible(category):
            return None

        return InventoryDerivation(category)

    def _expand(
        self, category: Term, height: int, /
    ) -> InventoryDerivation[Term] | None:
        """Return a derivation whose last step concludes a category.

        Parameters
        ----------
        category : Term
            The conclusion of the step.
        height : int
            How many steps lie between the node and the root.
        """
        options = list(self._options)
        self._random.shuffle(options)

        for rule, slot in options:
            drawn = tuple(self._random.choice(self._pool) for _ in range(rule.arity))
            known: tuple[Slot[Term], ...] = tuple(
                HOLE if at == slot else held for at, held in enumerate(drawn)
            )
            fillers = sorted(
                (
                    filler
                    for filler in rule.residuate(category, known)
                    if self._admissible(filler)
                ),
                key=_shape,
            )

            if not fillers:
                continue

            filler = self._random.choice(fillers)
            premises: list[InventoryDerivation[Term]] = []

            for at, held in enumerate(drawn):
                premise = self.grow(filler if at == slot else held, height + 1)

                if premise is None:
                    return None

                premises.append(premise)

            return InventoryDerivation(category, rule, tuple(premises))

        return None


def generate_inventory(
    rules: RuleSet[Term],
    goals: Sequence[Term],
    primitives: Sequence[Term],
    *,
    category_measure: CategoryMeasure,
    categories: int,
    max_depth: int,
    max_arguments: int | None,
    max_order: int | None,
    derivation_depth: int,
    expansion: float,
    patience: int,
    random: Random,
) -> CategoryInventory[Term]:
    """Grow an inventory of lexical categories from the goals through the rules.

    Each draw picks a goal uniformly and grows a derivation of it. The root is
    expanded by up to ``patience`` attempts, each drawing its premises afresh,
    and is a leaf only if every attempt fails; a node below the root is
    expanded with chance ``expansion`` while it lies fewer than
    ``derivation_depth`` steps below the root, and is otherwise a leaf. To
    expand a node, the rules' residual slots are tried in a shuffled order:
    each other premise is drawn uniformly from the pool, which holds the
    primitives and every category admitted so far, and the slot's filler is
    drawn uniformly from the residuals no deeper than ``max_depth``. A
    derivation is admitted when the rule set and its normal form admit every
    step, it has a leaf not yet in the inventory, and its new leaves do not
    take the inventory past ``categories``.

    Parameters
    ----------
    rules : RuleSet[Term]
        The rules derivations are grown through.
    goals : Sequence[Term]
        The categories a derivation may conclude.
    primitives : Sequence[Term]
        The categories the premise pool starts with.
    category_measure : CategoryMeasure
        The grammar-owned functor roles used to measure arguments and order.
    categories : int
        How many distinct lexical categories the inventory holds.
    max_depth : int
        The deepest category, by :func:`category_depth`, that may be a leaf
        or a premise.
    max_arguments : int | None
        The greatest functor-spine argument count a lexical category or
        generated premise may have, or no bound.
    max_order : int | None
        The greatest category order a lexical category or generated premise
        may have, or no bound.
    derivation_depth : int
        The greatest number of steps between a derivation's root and a leaf.
    expansion : float
        The chance a node below the root and above the depth bound is
        expanded rather than left a leaf.
    patience : int
        How many consecutive derivations may fail to be admitted before the
        inventory is refused, and how many expansions of a derivation's root
        are attempted before it is left a leaf.
    random : Random
        The source of every choice, so equal states grow equal inventories.

    Returns
    -------
    CategoryInventory[Term]
        The categories in the order they were admitted, each with its
        witness derivation.

    Raises
    ------
    InventoryExhaustedError
        If ``patience`` consecutive derivations are refused before the
        inventory holds ``categories`` categories.
    """
    grower = _Grower(
        rules,
        primitives,
        category_measure=category_measure,
        max_depth=max_depth,
        max_arguments=max_arguments,
        max_order=max_order,
        derivation_depth=derivation_depth,
        expansion=expansion,
        patience=patience,
        random=random,
    )
    found: dict[Term, InventoryDerivation[Term]] = {}
    stale = 0
    draws = 0

    while len(found) < categories:
        if stale >= patience:
            raise InventoryExhaustedError(
                requested=categories, reached=len(found), patience=patience
            )

        draws += 1
        derivation = grower.grow(random.choice(goals), 0)
        fresh = (
            ()
            if derivation is None or not derives(derivation, rules)
            else tuple(
                dict.fromkeys(leaf for leaf in derivation.leaves() if leaf not in found)
            )
        )

        if derivation is None or not fresh or len(found) + len(fresh) > categories:
            stale += 1

            continue

        stale = 0

        for category in fresh:
            found[category] = derivation
            grower.know(category)

    return CategoryInventory(tuple(found), tuple(found.values()), draws=draws)

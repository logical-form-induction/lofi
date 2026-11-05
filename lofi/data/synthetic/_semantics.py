"""Logical forms assigned to lexical items, and the lexicon holding the assignment.

A logical form belongs to one of five classes. A quantifier form is a
determiner meaning built on one of the fragment's quantifiers, a connective
form joins or negates propositions with one of the fragment's connectives, a
predicate form applies a fresh symbol to as many arguments as its arity, the
identity returns its argument, and a constant names a fresh individual. The
forms of a class are its inventory, each with a rank weight. A lexical item
draws its class from the rows its category is assigned, weighted by the
class's token share, then a form within the class in proportion to the rank
weights, so the assignment is a seeded draw and never a rule.
"""

from collections import Counter
from math import log
from random import Random
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, final, override

from lofi._checks import ensure_finite, ensure_name, ensure_non_negative
from lofi._frozen import FrozenValue
from lofi.data.synthetic._values import Histogram, LexiconReport
from lofi.registry import Registry

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from lofi.data.synthetic._values import SyntheticCorpus
    from lofi.lexicon import StaticLexicon
    from lofi.spec.language import LanguageAssembly
    from lofi.terms import Term

__all__ = [
    "LogicalClass",
    "LogicalClassContext",
    "LogicalClassPlugin",
    "Meaning",
    "SemanticLexicon",
    "SemanticReport",
    "assign",
    "inventory",
    "logical_classes",
]

type LogicalClass = str
"""The class of logical form a lexical item bears."""

type ClassRow = tuple[LogicalClass, int, tuple[float, ...]]
"""A class with its inventory size and one rank weight per form."""

type ArityRow = tuple[int, int]
"""A predicate arity with how many forms of the predicate inventory bear it."""

type QuantifierRow = tuple[str, str | None]
"""A quantifier name and its declared meaning text, or the standard text."""

type AssignmentRow = tuple[LogicalClass, int | None, float]
"""A class a category may draw, the arity it forces if any, and its weight."""

type Inventory = Mapping[LogicalClass, tuple[tuple[Meaning, float], ...]]
"""The forms of each class, each with its rank weight."""

_DETERMINERS: Final[Mapping[str, tuple[str, str]]] = MappingProxyType(
    {"forall": ("forall", "if"), "exists": ("exists", "and")}
)
_PROPOSITIONS: Final = "pqrstuvw"
_IDENTITY_TEXT: Final = "λx.x"
_PREDICATE_PREFIX: Final = "p"
_CONSTANT_PREFIX: Final = "c"


@final
class Meaning:
    """One logical form, with the class and the logical constant it is built on.

    Parameters
    ----------
    logical_class : LogicalClass
        The class the form belongs to.
    symbol : str
        The logical constant the form is built on: the quantifier or
        connective name, the predicate or individual symbol, or the identity's
        own text.
    arity : int | None
        How many arguments a filterable functional form takes, or nothing when
        the form's class does not support arity-conditioned assignment.
    text : str
        The form in the surface syntax logical forms are parsed from.

    Raises
    ------
    TypeError
        If the arity is not an integer.
    ValueError
        If the class, symbol or text is empty, or a supplied arity is not
        positive.
    """

    __slots__ = ("_arity", "_logical_class", "_symbol", "_text")

    def __init__(
        self,
        logical_class: LogicalClass,
        symbol: str,
        *,
        arity: int | None = None,
        text: str,
    ) -> None:
        if arity is not None and ensure_non_negative("a meaning's arity", arity) == 0:
            msg = f"a meaning needs a positive arity, not {arity!r}"
            raise ValueError(msg)

        self._logical_class = ensure_name("a meaning's class", logical_class)
        self._symbol = ensure_name("a meaning's symbol", symbol)
        self._arity = arity
        self._text = ensure_name("a meaning's text", text)

    @property
    def logical_class(self) -> LogicalClass:
        """Return the class the form belongs to."""
        return self._logical_class

    @property
    def symbol(self) -> str:
        """Return the logical constant the form is built on."""
        return self._symbol

    @property
    def arity(self) -> int | None:
        """Return the functional form's assignment arity, when it has one."""
        return self._arity

    @property
    def text(self) -> str:
        """Return the form in the surface syntax."""
        return self._text

    def _fields(self) -> tuple[LogicalClass, str, int | None, str]:
        """Return the fields the meaning is equal on."""
        return (self._logical_class, self._symbol, self._arity, self._text)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another meaning is the same form of the same class."""
        if type(other) is not Meaning:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the class, the symbol, the arity and the text."""
        return hash((Meaning, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this meaning."""
        return (
            f"Meaning({self._logical_class!r}, {self._symbol!r}, "
            f"arity={self._arity!r}, text={self._text!r})"
        )


def _quantifier(
    name: str, template: str | None, connectives: frozenset[str]
) -> Meaning:
    """Return the determiner meaning built on a quantifier.

    A standard quantifier restricts by one connective, which the fragment
    must declare. A generalized quantifier applies to the restrictor and the
    scope as a constant.
    """
    if template is not None:
        return Meaning("quantifier", name, text=template)

    determiner = _DETERMINERS.get(name)

    if determiner is None:
        return Meaning("quantifier", name, text=f"λP.λQ.{name}(P)(Q)")

    binder, connective = determiner

    if connective not in connectives:
        msg = (
            f"quantifier {name!r} restricts by the connective {connective!r}, "
            "which the fragment does not declare"
        )
        raise ValueError(msg)

    return Meaning(
        "quantifier",
        name,
        text=f"λP.λQ.{binder} x.{connective}(P(x))(Q(x))",
    )


def _connective_arities(language: LanguageAssembly, /) -> Mapping[str, int]:
    """Return how many propositions each connective of a language takes.

    The arity of a connective is the number of arrows its declared type
    scheme takes before its result.
    """
    arrow = language.function_type
    arities: dict[str, int] = {}

    for name in language.profile.fragment.connectives:
        body = language.scheme(name).body
        arity = 0

        while body.op.name == arrow and len(body.args) == 2:
            arity += 1
            body = body.args[1]

        arities[name] = arity

    return MappingProxyType(arities)


def _connective(name: str, language: LanguageAssembly, /) -> Meaning:
    """Return the meaning joining or negating propositions with a connective."""
    arity = _connective_arities(language).get(name)

    if arity is None or not 0 < arity <= len(_PROPOSITIONS):
        msg = f"no connective form is built on {name!r}"
        raise ValueError(msg)

    variables = _PROPOSITIONS[:arity]
    binders = "".join(f"λ{variable}." for variable in variables)
    arguments = "".join(f"({variable})" for variable in variables)

    return Meaning("connective", name, arity=arity, text=f"{binders}{name}{arguments}")


def _predicate(symbol: str, arity: int) -> Meaning:
    """Return the meaning applying a symbol to as many arguments as its arity."""
    binders = "".join(f"λx{index}." for index in range(arity, 0, -1))
    arguments = "".join(f"(x{index})" for index in range(1, arity + 1))

    return Meaning(
        "predicate", symbol, arity=arity, text=f"{binders}{symbol}{arguments}"
    )


def _identity() -> Meaning:
    """Return the meaning returning its argument."""
    return Meaning("identity", _IDENTITY_TEXT, text=_IDENTITY_TEXT)


def _constant(symbol: str) -> Meaning:
    """Return the meaning naming an individual."""
    return Meaning("constant", symbol, text=symbol)


@final
class LogicalClassContext(FrozenValue):
    """Language-owned resources supplied to one logical-form class plugin."""

    _noun = "logical-form class context"
    __slots__ = ("_arities", "_connectives", "_language", "_quantifiers")

    def __init__(
        self,
        language: LanguageAssembly,
        /,
        *,
        quantifiers: Iterable[QuantifierRow],
        connectives: Iterable[str],
        arities: Iterable[ArityRow],
    ) -> None:
        object.__setattr__(self, "_language", language)
        object.__setattr__(self, "_quantifiers", tuple(quantifiers))
        object.__setattr__(self, "_connectives", tuple(connectives))
        object.__setattr__(self, "_arities", tuple(arities))

    @property
    def language(self) -> LanguageAssembly:
        """Return the compiled language owning logical constants and types."""
        return self._language

    @property
    def quantifiers(self) -> tuple[QuantifierRow, ...]:
        """Return the declared quantifier meanings in declaration order."""
        return self._quantifiers

    @property
    def connectives(self) -> tuple[str, ...]:
        """Return the declared connectives in declaration order."""
        return self._connectives

    @property
    def arities(self) -> tuple[ArityRow, ...]:
        """Return the declared predicate-arity counts in declaration order."""
        return self._arities


class LogicalClassPlugin(FrozenValue):
    """One named logical-form inventory builder under ``lofi.lf_class``."""

    _noun = "logical-form class plugin"
    __slots__ = ("_build", "_name", "_order")

    def __init__(
        self,
        name: str,
        order: int,
        build: Callable[[int, LogicalClassContext], tuple[Meaning, ...]],
        /,
    ) -> None:
        object.__setattr__(self, "_name", ensure_name("a logical-form class", name))
        object.__setattr__(
            self,
            "_order",
            ensure_non_negative("a logical-form class report order", order),
        )
        object.__setattr__(self, "_build", build)

    @property
    def name(self) -> LogicalClass:
        """Return the class name contributed by this plugin."""
        return self._name

    @property
    def order(self) -> int:
        """Return this class's position in semantic reports."""
        return self._order

    def forms(self, size: int, context: LogicalClassContext, /) -> tuple[Meaning, ...]:
        """Build this class's inventory in rank order."""
        held = self._build(
            ensure_non_negative("a logical-form class size", size), context
        )
        if len(held) != size:
            msg = (
                f"logical-form class {self._name!r} built {len(held)} forms "
                f"for a requested size of {size}"
            )
            raise ValueError(msg)
        if any(form.logical_class != self._name for form in held):
            msg = (
                f"logical-form class {self._name!r} built a form assigned "
                "to another class"
            )
            raise ValueError(msg)
        return held


def _class_plugins() -> tuple[LogicalClassPlugin, ...]:
    """Return installed logical-form class plugins in declared report order."""
    found: list[LogicalClassPlugin] = []

    for tag, member in Registry[object]("lf_class").members():
        if not isinstance(member, LogicalClassPlugin) or member.name != tag:
            msg = (
                f"logical-form class entry {tag!r} does not name "
                "its matching LogicalClassPlugin"
            )
            raise TypeError(msg)
        found.append(member)

    return tuple(sorted(found, key=lambda plugin: (plugin.order, plugin.name)))


def logical_classes() -> tuple[LogicalClass, ...]:
    """Return installed logical-form class names in semantic-report order."""
    return tuple(plugin.name for plugin in _class_plugins())


def _class_plugin(logical_class: LogicalClass, /) -> LogicalClassPlugin:
    """Return the unique installed plugin for one logical-form class."""
    matches = tuple(
        plugin for plugin in _class_plugins() if plugin.name == logical_class
    )
    if len(matches) != 1:
        names = ", ".join(logical_classes())
        msg = (
            f"logical-form class {logical_class!r} has {len(matches)} installed "
            f"entries; installed classes are {names or '(none)'}"
        )
        raise ValueError(msg)
    return matches[0]


def _closed(
    logical_class: LogicalClass, size: int, names: Iterable[str]
) -> tuple[str, ...]:
    """Return the first names of a closed class, refusing a size beyond them."""
    held = tuple(names)

    if size > len(held):
        msg = (
            f"the joint sizes the {logical_class} class at {size} but the "
            f"fragment declares {len(held)} {logical_class}s: {list(held)!r}"
        )
        raise ValueError(msg)

    return held[:size]


def _predicates(size: int, arities: Iterable[ArityRow]) -> tuple[tuple[str, int], ...]:
    """Return a symbol and an arity for each predicate form, in arity-row order."""
    rows = tuple(arities)
    total = sum(count for _, count in rows)

    if total != size:
        msg = (
            f"the joint sizes the predicate class at {size} but the arity law "
            f"places {total} forms: {rows!r}"
        )
        raise ValueError(msg)

    forms: list[tuple[str, int]] = []

    for arity, count in rows:
        held = ensure_non_negative("a predicate arity", arity)

        if held == 0:
            msg = "a predicate arity is not positive"
            raise ValueError(msg)

        first = len(forms) + 1

        forms.extend(
            (f"{_PREDICATE_PREFIX}{first + offset}", held)
            for offset in range(ensure_non_negative("an arity's form count", count))
        )

    return tuple(forms)


def _quantifier_forms(
    size: int, context: LogicalClassContext, /
) -> tuple[Meaning, ...]:
    """Build the declared prefix of the quantifier inventory."""
    if size > len(context.quantifiers):
        msg = (
            f"the joint sizes the quantifier class at {size} but the fragment "
            f"declares {len(context.quantifiers)} quantifiers: "
            f"{[name for name, _ in context.quantifiers]!r}"
        )
        raise ValueError(msg)
    connectives = frozenset(context.connectives)
    return tuple(
        _quantifier(name, template, connectives)
        for name, template in context.quantifiers[:size]
    )


def _connective_forms(
    size: int, context: LogicalClassContext, /
) -> tuple[Meaning, ...]:
    """Build the declared prefix of the connective inventory."""
    return tuple(
        _connective(name, context.language)
        for name in _closed("connective", size, context.connectives)
    )


def _predicate_forms(size: int, context: LogicalClassContext, /) -> tuple[Meaning, ...]:
    """Build fresh predicate forms following the declared arity law."""
    return tuple(
        _predicate(symbol, arity)
        for symbol, arity in _predicates(size, context.arities)
    )


def _identity_forms(size: int, context: LogicalClassContext, /) -> tuple[Meaning, ...]:
    """Build the singleton identity inventory."""
    del context
    if size != 1:
        msg = f"the identity class has one form, but the joint sizes it at {size}"
        raise ValueError(msg)
    return (_identity(),)


def _constant_forms(size: int, context: LogicalClassContext, /) -> tuple[Meaning, ...]:
    """Build the requested fresh individual-constant inventory."""
    del context
    return tuple(
        _constant(f"{_CONSTANT_PREFIX}{index}") for index in range(1, size + 1)
    )


QUANTIFIER_CLASS = LogicalClassPlugin("quantifier", 0, _quantifier_forms)
CONNECTIVE_CLASS = LogicalClassPlugin("connective", 1, _connective_forms)
PREDICATE_CLASS = LogicalClassPlugin("predicate", 2, _predicate_forms)
IDENTITY_CLASS = LogicalClassPlugin("identity", 3, _identity_forms)
CONSTANT_CLASS = LogicalClassPlugin("constant", 4, _constant_forms)


def inventory(
    classes: Iterable[ClassRow],
    *,
    language: LanguageAssembly,
    quantifiers: Iterable[QuantifierRow],
    connectives: Iterable[str],
    arities: Iterable[ArityRow] = (),
) -> Inventory:
    """Build the forms of each class, each with its rank weight.

    Parameters
    ----------
    classes : Iterable[ClassRow]
        Each class with its inventory size and one rank weight per form.
    language : LanguageAssembly
        The run's compiled language, which owns each connective's type and arity.
    quantifiers : Iterable[QuantifierRow]
        The quantifiers and optional meaning texts the fragment declares, in
        the order the quantifier class is built on.
    connectives : Iterable[str]
        The connectives the fragment declares, in the order the connective
        class is built on.
    arities : Iterable[ArityRow]
        How many forms of the predicate inventory bear each arity. The
        predicate class's rank weights run over the arity rows in order.

    Raises
    ------
    ValueError
        If a class is listed twice, a size has another number of weights, a
        closed class is larger than the fragment's inventory, a standard
        quantifier restricts by a connective the fragment lacks, the arity
        rows place another number of forms than the predicate size, or the
        identity class is sized at more than one form.
    """
    held: dict[LogicalClass, tuple[tuple[Meaning, float], ...]] = {}
    context = LogicalClassContext(
        language,
        quantifiers=quantifiers,
        connectives=connectives,
        arities=arities,
    )

    for logical_class, size, weights in classes:
        if logical_class in held:
            msg = f"the joint lists the {logical_class} class twice"
            raise ValueError(msg)

        if len(weights) != size:
            msg = (
                f"the {logical_class} class has {size} forms but "
                f"{len(weights)} rank weights"
            )
            raise ValueError(msg)

        forms = _class_plugin(logical_class).forms(size, context)
        held[logical_class] = tuple(
            (form, ensure_finite(f"a {logical_class} rank weight", weight))
            for form, weight in zip(forms, weights, strict=True)
        )

    return MappingProxyType(held)


@final
class SemanticLexicon:
    """The logical form each lexical item bears, beside the inventory drawn from.

    Parameters
    ----------
    meanings : Mapping[tuple[str, str], tuple[Meaning, ...]]
        The nonempty tuple of meanings each lexical item bears, keyed by the
        item's category text and surface form.
    inventory : Mapping[LogicalClass, tuple[Meaning, ...]]
        The forms of each class in rank order, whether or not an item drew
        them.
    shares : Mapping[LogicalClass, float]
        The declared share of token mass of each class.

    Raises
    ------
    ValueError
        If an item bears a form outside the inventory of its class, or a class
        has an inventory without a share or a share without an inventory.
    """

    __slots__ = ("_inventory", "_meanings", "_shares")

    def __init__(
        self,
        meanings: Mapping[tuple[str, str], tuple[Meaning, ...]],
        inventory: Mapping[LogicalClass, tuple[Meaning, ...]],
        shares: Mapping[LogicalClass, float],
    ) -> None:
        if set(inventory) != set(shares):
            msg = (
                f"the classes with forms are {sorted(inventory)} but the classes "
                f"with shares are {sorted(shares)}"
            )
            raise ValueError(msg)

        known = {
            logical_class: frozenset(forms)
            for logical_class, forms in inventory.items()
        }

        for (category, form), held in meanings.items():
            if not held:
                msg = f"item {form!r} as {category!r} bears no meaning"
                raise ValueError(msg)

            if len(set(held)) != len(held):
                msg = f"item {form!r} as {category!r} bears a meaning twice"
                raise ValueError(msg)

            for meaning in held:
                if meaning not in known.get(meaning.logical_class, frozenset()):
                    msg = (
                        f"item {form!r} as {category!r} bears {meaning!r}, outside "
                        f"the {meaning.logical_class} inventory"
                    )
                    raise ValueError(msg)

        self._meanings = MappingProxyType(
            {item: tuple(held) for item, held in meanings.items()}
        )
        self._inventory = MappingProxyType(
            {logical_class: tuple(forms) for logical_class, forms in inventory.items()}
        )
        self._shares = MappingProxyType(dict(shares))

    @property
    def meanings(self) -> Mapping[tuple[str, str], tuple[Meaning, ...]]:
        """Return every meaning each item bears, by category and surface form."""
        return self._meanings

    @property
    def meaning_count(self) -> int:
        """Return how many item-to-meaning assignments the lexicon holds."""
        return sum(len(held) for held in self._meanings.values())

    @property
    def inventory(self) -> Mapping[LogicalClass, tuple[Meaning, ...]]:
        """Return the forms of each class in rank order."""
        return self._inventory

    @property
    def shares(self) -> Mapping[LogicalClass, float]:
        """Return the declared share of token mass of each class."""
        return self._shares

    @property
    def sizes(self) -> Mapping[LogicalClass, int]:
        """Return how many forms each class has."""
        return MappingProxyType(
            {
                logical_class: len(forms)
                for logical_class, forms in self._inventory.items()
            }
        )

    def __getitem__(self, item: tuple[str, str]) -> tuple[Meaning, ...]:
        """Return the meanings an item bears, by category and surface form."""
        return self._meanings[item]

    def __len__(self) -> int:
        """Return how many items bear a form."""
        return len(self._meanings)

    def _fields(
        self,
    ) -> tuple[
        tuple[tuple[tuple[str, str], tuple[Meaning, ...]], ...],
        tuple[tuple[LogicalClass, tuple[Meaning, ...]], ...],
        tuple[tuple[LogicalClass, float], ...],
    ]:
        """Return the fields the lexicon is equal on."""
        return (
            tuple(self._meanings.items()),
            tuple(self._inventory.items()),
            tuple(self._shares.items()),
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another lexicon assigns the same forms from one inventory."""
        if type(other) is not SemanticLexicon:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the assignment, the inventory and the shares."""
        return hash((SemanticLexicon, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return a rendering naming the sizes of the assignment and the inventory."""
        return (
            f"SemanticLexicon(items={len(self._meanings)}, "
            f"inventory={dict(self.sizes)!r})"
        )


def _draw[T](random: Random, choices: tuple[tuple[T, float], ...]) -> T:
    """Return one choice drawn in proportion to its weight."""
    return random.choices(
        [choice for choice, _ in choices], [weight for _, weight in choices]
    )[0]


def assign(
    lexicon: StaticLexicon[Term],
    render: Mapping[Term, str],
    forms: Inventory,
    shares: Mapping[LogicalClass, float],
    assignment: Mapping[str, tuple[AssignmentRow, ...]],
    *,
    seed: int,
    meanings_per_item: tuple[int, ...] | None = None,
    placement: float = 0.0,
) -> SemanticLexicon:
    """Draw a logical form for every lexical item.

    Items are visited in the lexicon's order, surface forms ascending and a
    form's entries in their order. Each item draws a row of its category's
    assignment in proportion to the row's weight times the token share of the
    row's class, then a form of that class in proportion to the rank weights,
    among the forms of the row's arity when it forces one.

    Parameters
    ----------
    lexicon : StaticLexicon[Term]
        The lexical items, each a form bearing a category.
    render : Mapping[Term, str]
        The text of every category the lexicon assigns.
    forms : Inventory
        The forms of each class with their rank weights.
    shares : Mapping[LogicalClass, float]
        The declared share of token mass of each class in the inventory.
    assignment : Mapping[str, tuple[AssignmentRow, ...]]
        The rows every category text of the lexicon may draw.
    seed : int
        The seed of the draw. Equal seeds over equal inputs draw equal
        lexicons.

    Raises
    ------
    ValueError
        If a category of the lexicon has no rendering or no assignment row, a
        row names a class outside the inventory, or a row forces an arity no
        predicate form bears.
    """
    random = Random(seed)
    meanings: dict[tuple[str, str], tuple[Meaning, ...]] = {}
    items = tuple(
        (form, entry, render.get(entry.category))
        for form in lexicon.forms
        for entry in lexicon.entries(form)
    )
    counts = [1] * len(items) if meanings_per_item is None else list(meanings_per_item)

    if len(counts) != len(items):
        msg = (
            f"the polysemy law counts {len(counts)} items but the lexicon holds "
            f"{len(items)}"
        )
        raise ValueError(msg)

    random.shuffle(counts)

    if placement:
        ranked = sorted(range(len(items)), key=lambda at: items[at][1].weight)
        ordered = sorted(counts, reverse=placement > 0.0)
        placed = [1] * len(items)

        for at, count in zip(ranked, ordered, strict=True):
            placed[at] = count

        counts = placed

    for (form, _entry, category), count in zip(items, counts, strict=True):
        if category is None:
            msg = f"the category of item {form!r} has no rendering"
            raise ValueError(msg)

        rows = assignment.get(category)

        if rows is None:
            msg = f"the assignment has no row for category {category!r}"
            raise ValueError(msg)

        available: list[
            tuple[LogicalClass, tuple[tuple[Meaning, float], ...], float]
        ] = []

        for logical_class, arity, row_weight in rows:
            share = _share(shares, category, logical_class)
            held = forms[logical_class]
            if arity is not None:
                held = tuple(
                    (meaning, weight)
                    for meaning, weight in held
                    if meaning.arity == arity
                )

                if not held:
                    msg = (
                        f"category {category!r} forces arity {arity} but no "
                        f"predicate form bears it"
                    )
                    raise ValueError(msg)

            available.append(
                (
                    logical_class,
                    held,
                    row_weight * share,
                )
            )

        possible = {meaning for _, held, _ in available for meaning, _ in held}

        if count > len(possible):
            msg = (
                f"item {form!r} as {category!r} requests {count} meanings but "
                f"only {len(possible)} are available"
            )
            raise ValueError(msg)

        drawn: list[Meaning] = []

        for _ in range(count):
            choices = tuple(
                (
                    (
                        logical_class,
                        tuple(
                            (meaning, weight)
                            for meaning, weight in held
                            if meaning not in drawn
                        ),
                    ),
                    row_weight,
                )
                for logical_class, held, row_weight in available
                if any(meaning not in drawn for meaning, _ in held)
            )
            _, forms_left = _draw(random, choices)
            meaning = _draw(random, forms_left)
            drawn.append(meaning)

        meanings[category, form] = tuple(drawn)

    return SemanticLexicon(
        meanings,
        {
            logical_class: tuple(meaning for meaning, _ in held)
            for logical_class, held in forms.items()
        },
        shares,
    )


def _share(
    shares: Mapping[LogicalClass, float], category: str, logical_class: LogicalClass
) -> float:
    """Return a class's token share, refusing a class the inventory lacks."""
    share = shares.get(logical_class)

    if share is None:
        msg = (
            f"category {category!r} is assigned the {logical_class} class, "
            f"which the joint does not declare"
        )
        raise ValueError(msg)

    return share


def _entropy(counts: Iterable[int], total: int) -> float:
    """Return the entropy in nats of counts summing to a total."""
    return -sum(count / total * log(count / total) for count in counts if count > 0)


@final
class SemanticReport:
    """The joint of logical-form class and token frequency measured over a corpus.

    Parameters
    ----------
    lexicon : SemanticLexicon
        The assignment the corpus was measured against.
    joint : LexiconReport
        Over each class: how many of its forms were drawn as a token each
        number of times, a form no item drew or no token realized in the zero
        bucket. Over the forms: the same counts whatever the class.
    arities : Histogram
        Over the lexical items bearing a predicate: how many bear each arity.
    cells : Mapping[str, Mapping[LogicalClass, int]]
        Over each category text: how many lexical items of that category bear
        each class.

    Raises
    ------
    ValueError
        If the joint or a cell names a class outside the lexicon's inventory,
        or the cells count another number of items than the lexicon holds.
    """

    __slots__ = ("_arities", "_cells", "_joint", "_lexicon")

    def __init__(
        self,
        lexicon: SemanticLexicon,
        joint: LexiconReport,
        arities: Histogram,
        cells: Mapping[str, Mapping[LogicalClass, int]],
    ) -> None:
        known = set(lexicon.inventory)
        outside = sorted(set(joint.classes) - known)

        if outside:
            msg = f"the joint reports classes outside the inventory: {outside!r}"
            raise ValueError(msg)

        held: dict[str, Mapping[LogicalClass, int]] = {}
        items = 0

        for category, counts in cells.items():
            outside = sorted(set(counts) - known)

            if outside:
                msg = (
                    f"category {category!r} is counted under classes outside "
                    f"the inventory: {outside!r}"
                )
                raise ValueError(msg)

            held[category] = MappingProxyType(
                {
                    logical_class: ensure_non_negative(
                        f"the {logical_class} count of {category!r}", count
                    )
                    for logical_class, count in counts.items()
                }
            )
            items += sum(held[category].values())

        if items != lexicon.meaning_count:
            msg = (
                f"the cells count {items} meanings but the lexicon holds "
                f"{lexicon.meaning_count}"
            )
            raise ValueError(msg)

        self._lexicon = lexicon
        self._joint = joint
        self._arities = arities
        self._cells = MappingProxyType(held)

    @classmethod
    def of(
        cls,
        lexicon: SemanticLexicon,
        corpora: tuple[SyntheticCorpus, ...],
        /,
    ) -> SemanticReport:
        """Measure the assignment jointly over one or more corpora.

        Every leaf of every gold derivation is one token of the lexical item
        its category text and surface form name.

        Parameters
        ----------
        lexicon : SemanticLexicon
            The assignment to measure.
        corpora : tuple[SyntheticCorpus, ...]
            The nonempty sequence of corpora drawn from the shared assignment.

        Raises
        ------
        ValueError
            If no corpus is supplied or a leaf names a lexical item the
            assignment does not cover.
        """
        if not corpora:
            msg = "a semantic report measures at least one corpus"
            raise ValueError(msg)

        tokens: Counter[Meaning] = Counter()

        for corpus in corpora:
            for example in corpus.examples:
                surfaces = tuple(token.surface for token in example.item.tokens)

                for leaf in example.supervision.derivation.leaves():
                    item = (leaf.label, surfaces[leaf.span.start])
                    meanings = lexicon.meanings.get(item)

                    if meanings is None:
                        msg = f"the assignment gives {item[1]!r} as {item[0]!r} no form"
                        raise ValueError(msg)

                    tokens.update(meanings)

        cells: dict[str, Counter[LogicalClass]] = {}

        for (category, _), meanings in lexicon.meanings.items():
            cells.setdefault(category, Counter()).update(
                meaning.logical_class for meaning in meanings
            )

        return cls(
            lexicon,
            LexiconReport(
                {
                    logical_class: Histogram.of(tokens[form] for form in forms)
                    for logical_class, forms in lexicon.inventory.items()
                },
                Histogram.of(
                    tokens[form]
                    for forms in lexicon.inventory.values()
                    for form in forms
                ),
                Histogram.of(1 for forms in lexicon.inventory.values() for _ in forms),
                None,
            ),
            Histogram.of(
                meaning.arity
                for meanings in lexicon.meanings.values()
                for meaning in meanings
                if meaning.logical_class == "predicate" and meaning.arity is not None
            ),
            cells,
        )

    @property
    def lexicon(self) -> SemanticLexicon:
        """Return the assignment the corpus was measured against."""
        return self._lexicon

    @property
    def joint(self) -> LexiconReport:
        """Return the per-class and per-form token counts."""
        return self._joint

    @property
    def arities(self) -> Histogram:
        """Return how many predicate-bearing items bear each arity."""
        return self._arities

    @property
    def cells(self) -> Mapping[str, Mapping[LogicalClass, int]]:
        """Return how many items of each category bear each class."""
        return self._cells

    @property
    def per_form_meanings(self) -> Histogram:
        """Return the distinct-meaning count of each surface form."""
        by_form: dict[str, set[Meaning]] = {}

        for (_, form), meanings in self._lexicon.meanings.items():
            by_form.setdefault(form, set()).update(meanings)

        return Histogram.of(len(meanings) for meanings in by_form.values())

    @property
    def items(self) -> Mapping[LogicalClass, int]:
        """Return how many lexical items bear each class of the inventory."""
        counts = Counter[LogicalClass]()

        for held in self._cells.values():
            counts.update(held)

        return MappingProxyType(
            {
                logical_class: counts[logical_class]
                for logical_class in self._lexicon.inventory
            }
        )

    @property
    def item_shares(self) -> Mapping[LogicalClass, float]:
        """Return the fraction of the items bearing each class, zero when none."""
        items = self.items
        total = sum(items.values())

        return MappingProxyType(
            {
                logical_class: count / total if total else 0.0
                for logical_class, count in items.items()
            }
        )

    @property
    def symbols_used(self) -> Mapping[LogicalClass, int]:
        """Return how many inventory symbols of each class occur in the corpus."""
        counts: dict[LogicalClass, int] = {}

        for logical_class in self._lexicon.inventory:
            histogram = self._joint.classes[logical_class]
            counts[logical_class] = sum(
                count for frequency, count in histogram.counts.items() if frequency > 0
            )

        return MappingProxyType(counts)

    @property
    def concordance(self) -> float | None:
        """Return the normalized mutual information of category and class.

        The mutual information is divided by the smaller of the two
        entropies, so it is one when either determines the other and near
        zero when the class is drawn without regard to the category, or
        nothing when the items vary in neither.
        """
        by_category = {
            category: sum(counts.values()) for category, counts in self._cells.items()
        }
        by_class = self.items
        total = sum(by_category.values())

        if total == 0:
            return None

        floor = min(
            _entropy(by_category.values(), total), _entropy(by_class.values(), total)
        )

        if floor == 0.0:
            return None

        information = sum(
            count
            / total
            * log(count * total / (by_category[category] * by_class[logical_class]))
            for category, counts in self._cells.items()
            for logical_class, count in counts.items()
            if count > 0
        )

        return information / floor

    def _fields(
        self,
    ) -> tuple[
        SemanticLexicon,
        LexiconReport,
        Histogram,
        tuple[tuple[str, tuple[tuple[LogicalClass, int], ...]], ...],
    ]:
        """Return the fields the report is equal on."""
        return (
            self._lexicon,
            self._joint,
            self._arities,
            tuple(
                (category, tuple(counts.items()))
                for category, counts in self._cells.items()
            ),
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another report measured the same assignment the same way."""
        if type(other) is not SemanticReport:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the assignment, the joint, the arities and the cells."""
        return hash((SemanticReport, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return a rendering naming the joint, the arities and the cells."""
        cells = {category: dict(counts) for category, counts in self._cells.items()}

        return (
            f"SemanticReport({self._lexicon!r}, {self._joint!r}, {self._arities!r}, "
            f"{cells!r})"
        )

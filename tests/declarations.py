"""Generated declarations of varied shape and size.

Every strategy here draws a configuration document and validates it through
the same didactic boundary a document on disk passes, so what a test receives
is a declaration a practitioner could have written. Names are drawn too: no
primitive, constructor, feature, form or symbol is one a packaged declaration
uses, so a component that reads a packaged name instead of the declaration it
is given fails on the first example.

A :class:`Profile` bounds what is drawn and states what a run over its draws
may cost. ``SMALL`` keeps every draw cheap enough for the ordinary suite and
``LARGE`` draws declarations several times bigger for the opt-in generality
run.
"""

import _thread
import enum
import json
import signal
import sys
import threading
import time
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple, cast

import yaml
from hypothesis import strategies as st

from lofi.data.synthetic.spec.interpretation import InterpretationSpec
from lofi.data.synthetic.training import SyntheticTrainingBatch
from lofi.nn.arity import ArityBound
from lofi.run import RandomStreams, SeedPolicy, plan_training, resolve_device_policy
from lofi.spec import (
    PACKAGED_CONFIG_ROOT,
    CoercionSetSpec,
    RuleSetSpec,
    RuleSpec,
    RunSpecCodec,
)
from lofi.spec.grammar import GrammarDecl, GrammarIso, SignatureSpec
from lofi.spec.language import (
    LanguageDecl,
    registered_language,
)
from lofi.spec.provers import TruthTableProverSpec

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes
else:
    import resource
from lofi.train import (
    Featurized,
    GroundedStep,
    ParserFittingPlan,
    ParserStep,
    prepare_training,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import FrameType

    from lofi._types import JsonValue
    from lofi.run import TrainingBlueprint
    from lofi.spec import RunSpec

__all__ = [
    "LARGE",
    "SMALL",
    "Candidates",
    "DataKind",
    "DataSection",
    "Interpreted",
    "Profile",
    "ProfileExceededError",
    "Spine",
    "Witnessed",
    "atom_text",
    "candidate_sets",
    "category_language",
    "category_text",
    "data_sections",
    "exceeds_shipped_ranks",
    "grammars",
    "interpreted_grammars",
    "languages",
    "packaged_names",
    "peak_resident_bytes",
    "plan",
    "rule_names",
    "rule_sets",
    "run_document",
    "signatures",
    "train_one_step",
    "train_spec",
    "within",
    "witnessed_grammars",
]

SOURCE: Final = Path(__file__).resolve().parents[1] / "lofi"
"""The package whose declarations the drawn names avoid."""

CONFIGS: Final = SOURCE / "configs"
"""The packaged configuration tree generated run documents start from."""

PACKAGED_CODEC: Final = RunSpecCodec(
    search_path=(PACKAGED_CONFIG_ROOT / "run", PACKAGED_CONFIG_ROOT)
)
"""The codec that resolves factored packaged run documents."""


def _signature_names(document: object) -> Iterator[str]:
    """Yield the primitive, operator and constructor names a document declares.

    Parameters
    ----------
    document : object
        A parsed configuration document or a part of one.
    """
    if isinstance(document, list):
        for item in document:
            yield from _signature_names(item)

        return

    if not isinstance(document, dict):
        return

    primitives = document.get("primitives")
    operators = document.get("operators")
    constructors = document.get("constructors")

    if isinstance(primitives, list):
        yield from (name for name in primitives if isinstance(name, str))

    if isinstance(operators, list):
        yield from (
            operator["name"]
            for operator in operators
            if isinstance(operator, dict) and isinstance(operator.get("name"), str)
        )

    if isinstance(constructors, dict):
        yield from (name for name in constructors if isinstance(name, str))

    for value in document.values():
        yield from _signature_names(value)


@cache
def packaged_names() -> frozenset[str]:
    """Return the names the packaged declarations declare.

    Returns
    -------
    frozenset[str]
        Every atomic type, type constructor, constant and term former of the
        packaged logical-form languages, every registered rule kind,
        and every primitive, operator and constructor a packaged
        configuration document declares, keeping the names that hold a
        letter or a digit.
    """
    names: set[str] = set()

    for language in (
        registered_language("propositional"),
        registered_language("neo-davidsonian"),
    ):
        names.update(atomic.name for atomic in language.atomic_types)
        names.update(constructor.name for constructor in language.constructors)
        names.update(constant.name for constant in language.constants)
        names.update(
            (
                language.abstraction,
                language.application,
                language.function_type,
            )
        )

    names.update(kind for kind in RuleSpec.__variants__ if isinstance(kind, str))

    for document in sorted(SOURCE.rglob("*.yaml")):
        names.update(
            _signature_names(yaml.safe_load(document.read_text(encoding="utf-8")))
        )

    return frozenset(name for name in names if any(c.isalnum() for c in name))


class Profile(NamedTuple):
    """The bounds of one family of draws and what a run over them may cost.

    Attributes
    ----------
    fewest : int
        The fewest primitive categories a signature declares and the fewest
        lexical categories a lexicon assigns beyond its first.
    primitives : int
        The most primitive categories a signature declares.
    arguments : int
        The most arguments one lexical category takes.
    spines : int
        The most lexical categories a lexicon assigns beyond the atoms its
        categories take as arguments.
    ambiguity : int
        The most categories one form bears.
    order : int
        The highest composition order a rule inventory declares.
    length : int
        The longest sentence a corpus draws.
    size : int
        How many items a drawn corpus holds.
    candidates : int
        The most candidate grammars a comparison set holds.
    examples : int
        How many examples one property draws.
    seconds : float
        The wall clock one property over this profile may take.
    memory : int
        The peak resident bytes the process may reach during that property.
    """

    fewest: int
    primitives: int
    arguments: int
    spines: int
    ambiguity: int
    order: int
    length: int
    size: int
    candidates: int
    examples: int
    seconds: float
    memory: int


SMALL: Final = Profile(
    fewest=0,
    primitives=3,
    arguments=2,
    spines=3,
    ambiguity=2,
    order=2,
    length=3,
    size=4,
    candidates=3,
    examples=6,
    seconds=600.0,
    memory=6 * 2**30,
)
"""Draws a unit test can afford on every run."""

LARGE: Final = Profile(
    fewest=5,
    primitives=8,
    arguments=3,
    spines=12,
    ambiguity=4,
    order=3,
    length=5,
    size=12,
    candidates=6,
    examples=4,
    seconds=1_800.0,
    memory=12 * 2**30,
)
"""Draws for the opt-in generality run."""


class DataKind(enum.Enum):
    """The kinds of data a drawn data section supplies."""

    STRINGS = "strings"
    ACCEPTABILITY = "acceptability"
    PARTIAL_SUPERVISION = "partial-supervision"
    INFERENCE = "inference"


class Spine(NamedTuple):
    """A lexical category as its result and the arguments it takes.

    Attributes
    ----------
    result : str
        The category the spine yields once every argument is consumed.
    arguments : tuple[tuple[str, str], ...]
        Each argument category with the direction its functor takes it from,
        ``"forward"`` or ``"backward"``, innermost first, so the last is the
        first one consumed.
    """

    result: str
    arguments: tuple[tuple[str, str], ...]


class DataSection(NamedTuple):
    """A run document's data section and the loss its kind trains under.

    Attributes
    ----------
    kind : DataKind
        The kind of data the section supplies.
    data : dict[str, JsonValue]
        The ``data`` section.
    loss : JsonValue
        The ``loss`` section, or None to keep the packaged one.
    """

    kind: DataKind
    data: dict[str, JsonValue]
    loss: JsonValue

    @property
    def dataset(self) -> dict[str, JsonValue]:
        """Return the section's dataset declaration."""
        datasets = cast("dict[str, JsonValue]", self.data["datasets"])
        train = cast("dict[str, JsonValue]", datasets["train"])

        return cast("dict[str, JsonValue]", train["dataset"])

    @property
    def grammar(self) -> GrammarDecl:
        """Return the grammar the section's dataset declares, validated."""
        return GrammarDecl.model_validate_json(json.dumps(self.dataset["grammar"]))

    @property
    def language(self) -> LanguageDecl:
        """Return the learner language the section declares, validated."""
        document = (
            cast("dict[str, JsonValue]", self.dataset["source"])["language"]
            if self.kind is DataKind.INFERENCE
            else self.dataset["language"]
        )
        return LanguageDecl.model_validate_json(json.dumps(document))


class Interpreted(NamedTuple):
    """A grammar with an interpretation of its lexicon.

    Attributes
    ----------
    grammar : GrammarDecl
        A grammar each of whose spines yields its goal.
    language : LanguageDecl
        The learner language whose store contains categories and meanings.
    interpretation : InterpretationSpec
        The meanings of its lexical entries in a drawn language.
    """

    grammar: GrammarDecl
    language: LanguageDecl
    interpretation: InterpretationSpec


class Candidates(NamedTuple):
    """A source grammar and the candidate grammars compared against it.

    Attributes
    ----------
    source : GrammarDecl
        The grammar the data is drawn from.
    candidates : tuple[GrammarDecl, ...]
        The source and grammars that differ from it, and from one another,
        only in their rule inventories; the source is the first.
    """

    source: GrammarDecl
    candidates: tuple[GrammarDecl, ...]


class _Lexicon(NamedTuple):
    """A drawn lexicon document, the spine of each category, and a form of each."""

    document: dict[str, JsonValue]
    spines: dict[str, Spine]
    forms: dict[str, str]


class Witnessed(NamedTuple):
    """A grammar with a string its goal derives.

    Attributes
    ----------
    grammar : GrammarDecl
        The grammar.
    sentence : tuple[str, ...]
        Two forms whose categories combine by application into the goal.
    """

    grammar: GrammarDecl
    sentence: tuple[str, ...]


def _dump(
    model: GrammarDecl | SignatureSpec | RuleSetSpec | LanguageDecl,
) -> JsonValue:
    """Return the configuration document a validated declaration dumps to."""
    return json.loads(model.model_dump_json())


def _distinct(
    strategy: st.SearchStrategy[str], low: int, high: int
) -> st.SearchStrategy[list[str]]:
    """Return lists of distinct drawn names."""
    return st.lists(strategy, min_size=low, max_size=high, unique=True)


_PRIMITIVE: Final = st.from_regex(r"[A-Z][A-Z0-9]{0,3}", fullmatch=True).filter(
    lambda name: name not in packaged_names()
)
_LOWER: Final = st.from_regex(r"[a-z]{2,5}", fullmatch=True).filter(
    lambda name: name not in packaged_names()
)
_FEATURE: Final = st.from_regex(r"[a-z][0-9]", fullmatch=True)
_SPELLING: Final = st.text(alphabet="|<>~%^&*+!?=", min_size=1, max_size=2)
_DIRECTION: Final = st.sampled_from(("forward", "backward"))


def _unprefixed(spellings: list[str]) -> bool:
    """Return whether no spelling begins another."""
    return not any(a != b and b.startswith(a) for a in spellings for b in spellings)


@st.composite
def signatures(
    draw: st.DrawFn,
    profile: Profile,
    *,
    category_sort: str | None = None,
    interpretable: bool = False,
) -> SignatureSpec:
    """Draw a category signature.

    The signature declares between two and ``profile.primitives`` primitives;
    either a forward and a backward functor constructor or one self-opposite
    one, with drawn names, spellings and result and argument slots; possibly
    a featured variant of some primitives; and its category sort and syntax. An
    s-expression signature may declare a further constructor of arity one to
    three without functor roles, which infix text cannot spell.

    Parameters
    ----------
    draw : st.DrawFn
        The draw function.
    profile : Profile
        The bounds of the draw.
    interpretable : bool
        Whether the signature is one an interpretation erases to types: a
        forward and a backward functor and, in s-expression syntax, possibly
        a binary constructor without functor roles.
    category_sort : str | None
        The category sort to keep fixed, or None to draw one.
    """
    primitives = draw(_distinct(_PRIMITIVE, max(2, profile.fewest), profile.primitives))
    names = draw(_distinct(_LOWER, 3, 3))
    result, argument = draw(st.sampled_from(((0, 1), (1, 0))))
    spellings = draw(_distinct(_SPELLING, 2, 2).filter(_unprefixed))
    spelled = draw(st.booleans())
    syntax = draw(st.sampled_from(("declared", "s-expression")))
    directions = (
        ("forward", "backward")
        if interpretable or draw(st.booleans())
        else ("self-opposite",)
    )
    sort = (
        draw(_PRIMITIVE.filter(lambda name: name not in primitives))
        if category_sort is None
        else category_sort
    )
    selected = tuple(zip(names, directions, spellings, strict=False))
    constructors: list[JsonValue] = [
        {
            "name": name,
            "kind": [sort] * 3,
            "notation": {
                "style": "infix",
                "spelling": spelling if spelled else name,
                "precedence": 10,
                "associativity": "left",
            },
        }
        for name, direction, spelling in zip(names, directions, spellings, strict=False)
    ]
    functors: list[JsonValue] = []
    for position, (name, direction, _) in enumerate(selected):
        opposite = name if direction == "self-opposite" else selected[1 - position][0]
        side = {"forward": "right", "backward": "left", "self-opposite": "either"}[
            direction
        ]
        functors.append(
            {
                "name": name,
                "constructor": name,
                "argument_side": side,
                "opposite": opposite,
                "result": result,
                "argument": argument,
            }
        )

    if syntax == "s-expression" and draw(st.booleans()):
        arity = st.just(2) if interpretable else st.integers(1, 3)
        held_arity = draw(arity)
        constructors.append(
            {
                "name": names[2],
                "kind": [sort] * (held_arity + 1),
                "notation": {"style": "prefix", "spelling": names[2]},
            }
        )

    featured = draw(st.lists(st.sampled_from(primitives), unique=True, max_size=2))
    return SignatureSpec.model_validate_json(
        json.dumps(
            {
                "name": draw(_LOWER),
                "sorts": [sort],
                "category_sort": sort,
                "primitives": primitives,
                "operators": [
                    {
                        "name": f"{primitive}[{draw(_FEATURE)}]",
                        "sort": sort,
                        "erases_to": primitive,
                    }
                    for primitive in featured
                ],
                "constructors": constructors,
                "functors": functors,
                "syntax": syntax,
            }
        )
    )


def _node(signature: SignatureSpec, name: str, children: list[str]) -> str:
    """Return a constructor applied to category texts, in the signature's syntax.

    Parameters
    ----------
    signature : SignatureSpec
        The signature whose syntax the text is written in.
    name : str
        The constructor's name.
    children : list[str]
        The texts of its arguments, in slot order.
    """
    if signature.syntax == "s-expression":
        return "(" + " ".join([json.dumps(name), *children]) + ")"

    constructor = next(held for held in signature.constructors if held.name == name)
    spelling = constructor.notation.spelling
    atoms = {
        *signature.primitives,
        *(operator.name for operator in signature.operators),
    }
    left, right = (child if child in atoms else f"({child})" for child in children)

    return f"{left}{spelling}{right}"


def atom_text(signature: SignatureSpec, name: str) -> str:
    """Return a primitive or featured category in the signature's syntax.

    Parameters
    ----------
    signature : SignatureSpec
        The signature whose syntax the text is written in.
    name : str
        The category's name.
    """
    return (
        f"({json.dumps(f'#const[{signature.category_sort}]')} {json.dumps(name)})"
        if signature.syntax == "s-expression"
        else name
    )


def category_text(signature: SignatureSpec, spine: Spine) -> str:
    """Return a spine's category in the signature's syntax.

    Parameters
    ----------
    signature : SignatureSpec
        The signature whose constructors write the category.
    spine : Spine
        The category's result and argument texts.
    """
    text = spine.result

    for argument, direction in spine.arguments:
        role = next(
            view
            for view in signature.functors
            if view.argument_side
            in ({"right", "either"} if direction == "forward" else {"left", "either"})
        )
        children = ["", ""]
        children[role.result] = text
        children[role.argument] = argument
        text = _node(signature, role.constructor, children)

    return text


def _functor_names(signature: SignatureSpec) -> list[str]:
    """Return the names of the signature's functor constructors."""
    return [view.name for view in signature.functors]


@st.composite
def _arguments(
    draw: st.DrawFn, signature: SignatureSpec, names: list[str]
) -> list[str]:
    """Draw the argument categories spines may take over some primitives.

    Each named category is one, and so is each declared constructor without
    functor roles applied to named categories.
    """
    atoms = [atom_text(signature, name) for name in names]
    wrapped = []

    functor_constructors = {view.constructor for view in signature.functors}
    for constructor in signature.constructors:
        if constructor.name not in functor_constructors:
            arity = constructor.arity
            children = draw(
                st.lists(st.sampled_from(atoms), min_size=arity, max_size=arity)
            )
            wrapped.append(_node(signature, constructor.name, children))

    return [*atoms, *wrapped]


@st.composite
def _spines(
    draw: st.DrawFn,
    goal: str,
    profile: Profile,
    arguments: list[str],
    results: list[str],
) -> list[Spine]:
    """Draw lexical spines, the first deriving the goal from one argument."""
    argument = st.tuples(st.sampled_from(arguments), _DIRECTION)
    rest = st.builds(
        Spine,
        st.sampled_from(results),
        st.lists(argument, min_size=1, max_size=profile.arguments).map(tuple),
    )

    return [
        Spine(goal, (draw(argument),)),
        *draw(st.lists(rest, min_size=profile.fewest, max_size=profile.spines - 1)),
    ]


@st.composite
def _lexicon(
    draw: st.DrawFn,
    signature: SignatureSpec,
    spines: list[Spine],
    profile: Profile,
) -> _Lexicon:
    """Draw an inline lexicon over spines, with a word for every argument.

    Every argument a spine takes is the category of some word, so every
    spine derives its result over its own word and its arguments' words.
    Forms may bear several categories, and entries carry drawn log-weights.
    """
    by_text = {category_text(signature, spine): spine for spine in spines}
    atoms = sorted({argument for spine in spines for argument, _ in spine.arguments})
    categories = [*by_text, *(atom for atom in atoms if atom not in by_text)]
    forms = draw(_distinct(_LOWER, len(categories), len(categories)))
    weight = st.one_of(st.just(0.0), st.floats(-3.0, 0.0, allow_nan=False))
    assigned = {
        (form, category) for form, category in zip(forms, categories, strict=True)
    }

    for form in forms:
        extra = st.lists(
            st.sampled_from(categories), unique=True, max_size=profile.ambiguity - 1
        )
        assigned.update((form, category) for category in draw(extra))

    entries: list[JsonValue] = [
        {"form": form, "category": category, "weight": draw(weight)}
        for form, category in sorted(assigned)
    ]

    return _Lexicon(
        {"kind": "inline", "entries": entries},
        by_text,
        dict(zip(categories, forms, strict=True)),
    )


@st.composite
def rule_sets(
    draw: st.DrawFn,
    signature: SignatureSpec,
    profile: Profile,
    *,
    coercible: list[str] | None = None,
) -> RuleSetSpec:
    """Draw a rule inventory over a signature's functor constructors.

    Application is always declared, either rule by rule or by a combinator
    set; composition and crossed composition up to ``profile.order``, type
    raising to drawn targets and ground coercions between primitives may be
    added. The coercions follow a drawn order of the primitives, so no chain
    of them returns to where it began.

    Parameters
    ----------
    draw : st.DrawFn
        The draw function.
    signature : SignatureSpec
        The signature the rules are declared over.
    profile : Profile
        The bounds of the draw.
    coercible : list[str] | None
        The primitives a coercion may relate, or None for every primitive.
    """
    orders = st.lists(
        st.integers(1, profile.order), unique=True, max_size=profile.order
    )
    primitives = list(signature.primitives)
    targets = draw(
        st.none() | st.lists(st.sampled_from(primitives), unique=True, min_size=1)
    )
    raising = targets is not None and draw(st.booleans())
    explicit = draw(st.booleans())
    composition = draw(orders)
    crossed_composition = draw(orders)
    direction = draw(st.sampled_from(("unrestricted", "steedman", "kuhlmann")))

    self_opposite = any(view.argument_side == "either" for view in signature.functors)

    if direction == "unrestricted" or self_opposite:
        crossed_composition = [
            order for order in crossed_composition if order not in composition
        ]

    ranked = draw(st.permutations(primitives if coercible is None else coercible))
    pair = st.tuples(st.integers(0, len(ranked)), st.integers(0, len(ranked)))
    coercions = [
        (ranked[low], ranked[high])
        for low, high in draw(
            st.lists(
                pair.filter(lambda p: p[0] < p[1] < len(ranked)),
                unique=True,
                max_size=2 if len(ranked) > 1 else 0,
            )
        )
    ]

    return RuleSetSpec.model_validate_json(
        json.dumps(
            {
                "name": draw(_LOWER),
                "rules": [
                    *(
                        {"rule": "application", "view": name}
                        for name in (_functor_names(signature) if explicit else ())
                    ),
                    {
                        "rule": "combinators",
                        "application": not explicit,
                        "composition": composition,
                        "crossed_composition": crossed_composition,
                        "direction": direction,
                        "type_raising": raising,
                        "raising_targets": [
                            atom_text(signature, target) for target in targets
                        ]
                        if raising and targets is not None
                        else None,
                        "raising_arguments": None,
                    },
                    *(
                        {
                            "rule": "coercions",
                            "label": draw(_LOWER),
                            "entries": [
                                f"{atom_text(signature, source)} => "
                                f"{atom_text(signature, target)}"
                                for source, target in coercions
                            ],
                        }
                        for _ in range(1 if coercions else 0)
                    ),
                ],
            }
        )
    )


def _grammar(
    draw: st.DrawFn,
    signature: SignatureSpec,
    goal: str,
    lexicon: _Lexicon,
    rules: RuleSetSpec,
) -> GrammarDecl:
    """Validate a grammar document assembled from drawn parts."""
    directed = len(_functor_names(signature)) == 2
    erasing = bool(signature.operators) or draw(st.booleans())

    return GrammarDecl.model_validate_json(
        json.dumps(
            {
                "name": draw(_LOWER),
                "signature": _dump(signature),
                "rules": _dump(rules),
                "lexicon": lexicon.document,
                "goal": {
                    "kind": "root",
                    "categories": [atom_text(signature, goal)],
                },
                "matching": {"kind": "feature-erasure" if erasing else "identity"},
                "normal_form": {
                    "kind": draw(
                        st.sampled_from(
                            ("none", "eisner", "hockenmaier-bisk")
                            if directed
                            else ("none",)
                        )
                    )
                },
                "budget": {"unary_rounds": 1},
            }
        )
    )


@st.composite
def witnessed_grammars(
    draw: st.DrawFn,
    profile: Profile,
    *,
    category_sort: str | None = None,
) -> Witnessed:
    """Draw a complete grammar and a two-word string its goal derives.

    Parameters
    ----------
    draw : st.DrawFn
        The draw function.
    profile : Profile
        The bounds of the draw.
    category_sort : str | None
        The learner language's category sort, or None to draw one.
    """
    signature = draw(signatures(profile, category_sort=category_sort))
    goal = draw(st.sampled_from(signature.primitives))
    names = [*signature.primitives, *(op.name for op in signature.operators)]
    arguments = draw(_arguments(signature, names))
    results = [atom_text(signature, name) for name in names]
    spines = draw(_spines(atom_text(signature, goal), profile, arguments, results))
    lexicon = draw(_lexicon(signature, spines, profile))
    grammar = _grammar(
        draw, signature, goal, lexicon, draw(rule_sets(signature, profile))
    )
    first = spines[0]
    argument, direction = first.arguments[0]
    functor = lexicon.forms[category_text(signature, first)]
    taken = lexicon.forms[argument]
    sentence = (functor, taken) if direction == "forward" else (taken, functor)

    return Witnessed(grammar, sentence)


def grammars(
    profile: Profile, *, category_sort: str | None = None
) -> st.SearchStrategy[GrammarDecl]:
    """Return complete grammars whose goal a string of two words derives.

    Parameters
    ----------
    profile : Profile
        The bounds of the draw.
    category_sort : str | None
        The learner language's category sort, or None to draw one.
    """
    return witnessed_grammars(profile, category_sort=category_sort).map(
        lambda witnessed: witnessed.grammar
    )


@st.composite
def languages(draw: st.DrawFn) -> LanguageDecl:
    """Draw a logical-form language over the packaged propositional language.

    The fragment keeps negation, on which a contradiction is decided, and the
    binary connectives of a drawn subset of the other connective classes.

    Parameters
    ----------
    draw : st.DrawFn
        The draw function.
    """
    base = registered_language("propositional")
    binary = ("conjunction", "disjunction", "conditional", "biconditional")
    kept = set(draw(st.lists(st.sampled_from(binary), unique=True)))
    dropped = set(binary) - kept

    return base.with_(
        constants=tuple(
            constant
            for constant in base.constants
            if not dropped.intersection(constant.classes)
        )
    )


def category_language(grammar: GrammarDecl, language: LanguageDecl) -> LanguageDecl:
    """Return a language whose category sort is the grammar's declared sort."""
    category_sort = grammar.signature.category_sort
    bounds = dict(language.bounds)
    category_bound = bounds.pop(language.category_sort)
    bounds[category_sort] = category_bound

    return language.with_(category_sort=category_sort, bounds=bounds)


def _meaning(form: str, spine: Spine | None) -> tuple[str, int | None]:
    """Return the symbol a word denotes and, for a predicate, its arity."""
    if spine is None:
        return f"k{form}", None

    arity = len(spine.arguments)

    return f"p{form}{arity}", arity


def _abstraction(symbol: str, arity: int | None) -> str:
    """Return the meaning of a word denoting a symbol of an arity."""
    if arity is None:
        return symbol

    variables = [f"x{index}" for index in range(arity)]
    prefix = "".join(f"\\{variable}. " for variable in reversed(variables))

    return prefix + symbol + "".join(f"({variable})" for variable in variables)


@st.composite
def interpreted_grammars(draw: st.DrawFn, profile: Profile) -> Interpreted:
    """Draw a grammar and an interpretation of every one of its entries.

    The goal denotes a truth value and every other primitive an individual.
    Every spine yields the goal from individuals, and a word bearing it
    denotes a predicate of as many individuals as the spine takes, abstracted
    in the order the spine consumes them; a word bearing a primitive denotes
    a constant. A second word bears the first spine, so the grammar derives
    two sentences that differ in meaning and a draw that excludes trivial
    pairs can be served. Coercions relate individuals only, and each acts as
    the identity, which the rule set lists last. A functor constructor erases
    through its roles, and any other constructor through a type-term image the
    interpretation draws.

    Parameters
    ----------
    draw : st.DrawFn
        The draw function.
    profile : Profile
        The bounds of the draw.
    """
    language = draw(languages())
    language = language.with_(
        logic=language.logic.with_(prover=TruthTableProverSpec(atoms=12, timeout=5.0))
    )
    signature = draw(
        signatures(
            profile,
            category_sort=language.category_sort,
            interpretable=True,
        )
    )
    goal = draw(st.sampled_from(signature.primitives))
    others = [primitive for primitive in signature.primitives if primitive != goal]
    goal_text = atom_text(signature, goal)
    arguments = [atom_text(signature, primitive) for primitive in others]
    spines = draw(_spines(goal_text, profile, arguments, [goal_text]))
    lexicon = draw(_lexicon(signature, spines, profile))
    entries = cast("list[dict[str, JsonValue]]", lexicon.document["entries"])
    spoken = {entry["form"] for entry in entries}
    entries.append(
        {
            "form": draw(_LOWER.filter(lambda form: form not in spoken)),
            "category": next(iter(lexicon.spines)),
            "weight": 0.0,
        }
    )
    rules = draw(rule_sets(signature, profile, coercible=others))
    coercions = sum(
        len(coercion.entries)
        for coercion in rules.rules
        if isinstance(coercion, CoercionSetSpec)
    )
    constants: dict[str, JsonValue] = {}
    for entry in cast("list[dict[str, str]]", lexicon.document["entries"]):
        form, category = entry["form"], entry["category"]
        symbol, arity = _meaning(form, lexicon.spines.get(category))
        entry["meaning"] = _abstraction(symbol, arity)

        constants[symbol] = (
            {"name": symbol, "type": "e", "classes": ["individual"]}
            if arity is None
            else {
                "name": symbol,
                "type": " -> ".join(("e",) * arity + ("t",)),
                "classes": ["predicate"],
            }
        )

    grammar = _grammar(draw, signature, goal, lexicon, rules)
    names = rule_names(grammar, language)

    images = {
        name: draw(st.sampled_from(((0, 1), (1, 0))))
        for constructor in signature.constructors
        if constructor.name not in {view.constructor for view in signature.functors}
        for name in (constructor.name,)
    }
    interpretation = InterpretationSpec.model_validate_json(
        json.dumps(
            {
                "erasure": {
                    "kind": "signature-morphism",
                    "atoms": {
                        primitive: "t" if primitive == goal else "e"
                        for primitive in signature.primitives
                    },
                    "constructors": {
                        name: f"a{argument} -> a{result}"
                        for name, (result, argument) in images.items()
                    },
                },
                "constants": [constants[symbol] for symbol in sorted(constants)],
                "meanings": {"source": {"kind": "lexical"}},
                "actions": dict.fromkeys(names[len(names) - coercions :], "\\x. x"),
            }
        )
    )

    return Interpreted(grammar, language, interpretation)


@st.composite
def candidate_sets(
    draw: st.DrawFn,
    profile: Profile,
    *,
    category_sort: str | None = None,
) -> Candidates:
    """Draw a source grammar and candidates differing in their rule inventories.

    Parameters
    ----------
    draw : st.DrawFn
        The draw function.
    profile : Profile
        The bounds of the draw.
    category_sort : str | None
        The category sort to keep fixed, or None to draw one.
    """
    signature = draw(signatures(profile, category_sort=category_sort))
    goal = draw(st.sampled_from(signature.primitives))
    names = [*signature.primitives, *(op.name for op in signature.operators)]
    arguments = draw(_arguments(signature, names))
    results = [atom_text(signature, name) for name in names]
    spines = draw(_spines(atom_text(signature, goal), profile, arguments, results))
    lexicon = draw(_lexicon(signature, spines, profile))
    inventories = draw(
        st.lists(
            rule_sets(signature, profile),
            min_size=2,
            max_size=profile.candidates,
            unique_by=lambda rules: rules.model_dump_json(),
        )
    )
    source = _grammar(draw, signature, goal, lexicon, inventories[0])
    declared = cast("dict[str, JsonValue]", _dump(source))
    candidates = tuple(
        GrammarDecl.model_validate_json(json.dumps({**declared, "rules": _dump(rules)}))
        for rules in inventories
    )

    return Candidates(source, candidates)


def _packaged(*parts: str) -> JsonValue:
    """Return one packaged configuration fragment."""
    return yaml.safe_load(CONFIGS.joinpath(*parts).read_text(encoding="utf-8"))


def _length(profile: Profile) -> st.SearchStrategy[JsonValue]:
    """Return the sentence-length laws a drawn corpus may declare.

    Either law draws lengths from two words to ``profile.length``.
    """
    uniform: JsonValue = {
        "kind": "uniform",
        "min_length": 2,
        "max_length": profile.length,
    }
    negative_binomial = st.builds(
        lambda r, p: {
            "kind": "negative-binomial",
            "r": r,
            "p": p,
            "min_length": 2,
            "max_length": profile.length,
        },
        st.floats(0.5, 4.0, allow_nan=False),
        st.floats(0.2, 0.8, allow_nan=False),
    )

    return st.just(uniform) | negative_binomial


@st.composite
def data_sections(draw: st.DrawFn, profile: Profile, kind: DataKind) -> DataSection:
    """Draw a data section of one kind over a drawn grammar.

    Strings are an unlabeled corpus; acceptability labels its items, with or
    without negatives corrupted by every built-in operator in a drawn order, and
    flips labels at drawn rates; partial supervision pins a drawn share
    of the tokens to their gold lexical categories; inference judgments pair
    sentences of an interpreted grammar and label the pairs.

    Parameters
    ----------
    draw : st.DrawFn
        The draw function.
    profile : Profile
        The bounds of the draw.
    kind : DataKind
        The kind of data to draw.
    """
    seed = draw(st.integers(0, 2**31 - 1))

    if kind is DataKind.INFERENCE:
        interpreted = draw(interpreted_grammars(profile))

        return DataSection(
            kind,
            {
                "datasets": {
                    "train": {
                        "dataset": {
                            "kind": "synthetic-inference",
                            "labeling": "three-way",
                            "retained": {"validation": 0.25},
                            "grammar": _dump(interpreted.grammar),
                            "source": {
                                "language": _dump(interpreted.language),
                                "interpretation": json.loads(
                                    interpreted.interpretation.model_dump_json()
                                ),
                            },
                            "size": profile.size,
                            "seed": seed,
                            "candidates": 2 * profile.size,
                            "attempts": 50 * profile.size,
                            "premise": draw(_length(profile)),
                            "hypothesis": draw(_length(profile)),
                        },
                        "featurizer": {
                            "kind": "pair",
                            "selection": "relational-scalar",
                        },
                    }
                }
            },
            None,
        )

    language = draw(languages())
    grammar = draw(grammars(profile, category_sort=language.category_sort))
    dataset: dict[str, JsonValue] = {
        "kind": "synthetic",
        "retained": {"validation": 0.25},
        "grammar": _dump(grammar),
        "language": _dump(language),
        "size": profile.size,
        "seed": seed,
        "length": draw(_length(profile)),
    }
    rate = st.floats(0.0, 0.2, allow_nan=False)

    if kind is DataKind.ACCEPTABILITY:
        negatives = draw(st.booleans())

        if negatives:
            operators = draw(
                st.permutations(
                    (
                        {"kind": "swap", "edits": 1},
                        {"kind": "deletion", "edits": 1},
                        {"kind": "insertion", "edits": 1},
                        {"kind": "substitution", "edits": 1},
                    )
                )
            )
            entries: list[JsonValue] = [
                cast("JsonValue", {"operator": operator, "weight": 1.0})
                for operator in operators
            ]
            dataset["corruption"] = {
                "proportion": {
                    "value": draw(st.floats(0.25, 0.75, allow_nan=False)),
                    "draw": "exact",
                },
                "operators": {"entries": entries},
                "attempts": 200,
            }

        dataset["noise"] = {
            "grammatical": {"rate": draw(rate)},
            "ungrammatical": {"rate": draw(rate) if negatives else 0.0},
        }

    train: dict[str, JsonValue] = {
        "dataset": dataset,
        "featurizer": {"kind": "unlabeled"},
    }
    data: dict[str, JsonValue] = {"datasets": {"train": train}}

    if kind is not DataKind.PARTIAL_SUPERVISION:
        return DataSection(kind, data, None)

    policy = cast(
        "dict[str, JsonValue]",
        _packaged(
            "data",
            "datasets",
            "train",
            "constraint_policy",
            "syntactic",
            "partial.yaml",
        ),
    )
    source = draw(st.sampled_from(("gold-lexicon.yaml", "gold-derivation.yaml")))
    train["constraint_source"] = _packaged(
        "data", "datasets", "train", "constraint_source", source
    )
    train["constraint_policy"] = {
        "syntactic": {
            **policy,
            "coverage": draw(st.floats(0.25, 1.0, allow_nan=False)),
            "seed": seed,
        }
    }

    return DataSection(kind, data, _packaged("loss", "pinning.yaml"))


def rule_names(grammar: GrammarDecl, language: LanguageDecl) -> tuple[str, ...]:
    """Return the names of the rules a grammar builds, in declaration order.

    Parameters
    ----------
    grammar : GrammarDecl
        The grammar.
    language : LanguageDecl
        The learner language that owns the grammar's category sort.
    """
    assembly = GrammarIso().forward(grammar, language)

    return tuple(rule.name for rule in assembly.grammar.rules)


def run_document(section: DataSection) -> dict[str, JsonValue]:
    """Return the packaged parser run over a drawn data section.

    The tagged parser follows the section grammar's unary-closure budget, the
    combinator decoder family owns its binary rules, and the trainer takes one
    microbatch per step.

    Parameters
    ----------
    section : DataSection
        The data section and the loss it trains under.
    """
    run = cast(
        "dict[str, JsonValue]",
        json.loads(
            PACKAGED_CODEC(CONFIGS / "run" / "synthetic-parser.yaml").model_dump_json()
        ),
    )
    run["approximation"] = {"residual_mass": 0.9}
    loss = cast("dict[str, JsonValue]", run["loss"])
    _, coherence_term = cast("list[dict[str, JsonValue]]", loss["terms"])
    coherence = cast("dict[str, JsonValue]", coherence_term["objective"])
    coherence["divergence"] = {"kind": "fenchel-young", "alpha": 1.5}
    grammar = section.grammar
    assembly = GrammarIso().forward(grammar, section.language)
    model = cast("dict[str, JsonValue]", run["model"])
    model["parser"] = {
        "kind": "tagged",
        "width": 4,
        "unary_depth": assembly.grammar.budget.unary_rounds,
    }
    spaces = cast("dict[str, JsonValue]", model["spaces"])
    categories = cast("dict[str, JsonValue]", spaces["categories"])
    combinators = cast("dict[str, JsonValue]", categories["combinators"])
    combinators["decoders"] = {
        rule.name: "application" for rule in assembly.grammar.rules if rule.arity == 2
    }
    trainer = cast("dict[str, JsonValue]", run["trainer"])
    trainer["accumulation_steps"] = 1
    if section.kind is DataKind.INFERENCE:
        run["trainer"] = {
            "kind": "alternating",
            "base": trainer,
            "tasks": [
                {"kind": "parser", "name": "parser", "dataset": "train"},
                {"kind": "grounded", "name": "grounded", "dataset": "train"},
            ],
            "mixing": {"kind": "round-robin"},
        }
    run["data"] = section.data

    if section.loss is not None:
        run["loss"] = section.loss

    return run


def plan(
    run: dict[str, JsonValue], directory: Path
) -> tuple[RunSpec, TrainingBlueprint]:
    """Compose a run document from disk and plan its training.

    Parameters
    ----------
    run : dict[str, JsonValue]
        The run document.
    directory : Path
        Where the document is written.
    """
    path = directory / "generated.yaml"
    path.write_text(
        yaml.safe_dump(run, sort_keys=False), encoding="utf-8", newline="\n"
    )
    spec = PACKAGED_CODEC(path)

    return spec, plan_training(spec)


def train_one_step(run: dict[str, JsonValue], directory: Path) -> dict[str, float]:
    """Compose, build the data of and train one step of a run document.

    Parameters
    ----------
    run : dict[str, JsonValue]
        The run document.
    directory : Path
        Where the document is written.

    Returns
    -------
    dict[str, float]
        The metrics of the step.
    """
    spec, _ = plan(run, directory)

    return train_spec(spec)


def train_spec(spec: RunSpec) -> dict[str, float]:
    """Build the data of and train one step of a composed run.

    Parameters
    ----------
    spec : RunSpec
        The composed run.

    Returns
    -------
    dict[str, float]
        The metrics of the step.
    """
    blueprint = plan_training(spec)
    streams = RandomStreams(SeedPolicy(13, samplers=blueprint.sampler_streams))
    streams.apply()
    prepared = prepare_training(blueprint, streams, resolve_device_policy(spec))
    fitting = cast("ParserFittingPlan", prepared.plan)
    batch = fitting.batches[0]

    if isinstance(fitting.step, ParserStep):
        if not isinstance(batch, SyntheticTrainingBatch):
            msg = "a parser step needs a synthetic sentence batch"
            raise TypeError(msg)

        step = fitting.step.run(Featurized(batch, 0))

    elif isinstance(fitting.step, GroundedStep):
        if isinstance(batch, SyntheticTrainingBatch):
            msg = "a grounded step needs a synthetic inference batch"
            raise TypeError(msg)

        step = fitting.step.run(Featurized(batch, 0))

    else:
        msg = f"a parser fitting plan has an unknown step: {fitting.step!r}"
        raise TypeError(msg)

    return dict(step.result.metrics)


def exceeds_shipped_ranks(grammar: GrammarDecl) -> bool:
    """Return whether a grammar declares a constructor wider than the shipped ranks.

    Parameters
    ----------
    grammar : GrammarDecl
        The grammar.
    """
    ceiling = ArityBound().ceiling

    return any(c.arity > ceiling for c in grammar.signature.constructors)


if sys.platform == "win32":

    class _MemoryCounters(ctypes.Structure):
        """The process memory counters ``GetProcessMemoryInfo`` fills."""

        _fields_ = (
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        )

    _KERNEL32: Final = ctypes.WinDLL("kernel32", use_last_error=True)
    _KERNEL32.GetCurrentProcess.restype = wintypes.HANDLE
    _PSAPI: Final = ctypes.WinDLL("psapi", use_last_error=True)
    _PSAPI.GetProcessMemoryInfo.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(_MemoryCounters),
        wintypes.DWORD,
    )
    _PSAPI.GetProcessMemoryInfo.restype = wintypes.BOOL

    def peak_resident_bytes() -> int:
        """Return the most resident memory this process has held, in bytes."""
        counters = _MemoryCounters()
        counters.cb = ctypes.sizeof(counters)

        if not _PSAPI.GetProcessMemoryInfo(
            _KERNEL32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            raise ctypes.WinError(ctypes.get_last_error())

        return counters.PeakWorkingSetSize

elif sys.platform == "linux":

    def peak_resident_bytes() -> int:
        """Return the most resident memory this process has held, in bytes.

        Linux carries the peak of the image an ``exec`` replaced into
        ``ru_maxrss``, so a process started by a larger one would report its
        parent's peak. ``VmHWM`` in ``/proc/self/status`` is the peak of this
        image alone, in kibibytes.
        """
        status = Path("/proc/self/status").read_text(encoding="ascii")

        for line in status.splitlines():
            name, _, value = line.partition(":")

            if name == "VmHWM":
                return int(value.split()[0]) * 1024

        msg = "/proc/self/status names no VmHWM"
        raise OSError(msg)

else:

    def peak_resident_bytes() -> int:
        """Return the most resident memory this process has held, in bytes."""
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        return peak if sys.platform == "darwin" else peak * 1024


class ProfileExceededError(Exception):
    """Raised when a run over a profile's draws exceeds its time or memory."""


class _Expired(BaseException):
    """Interrupts a run whose budget is spent, past any example's handler."""


@contextmanager
def within(profile: Profile) -> Iterator[None]:
    """Run a block under a profile's wall clock and resident-memory limits.

    A watcher samples the process's peak resident memory while the block
    runs and interrupts it as soon as the peak passes the limit, so a
    runaway draw is stopped before it exhausts the machine. A peak the
    process reached before the block began is not charged to it.

    Parameters
    ----------
    profile : Profile
        The profile whose limits apply.

    Raises
    ------
    ProfileExceededError
        If the block runs past ``profile.seconds``, or the process's peak
        resident memory rises past ``profile.memory`` while it runs.
    """
    ceiling = max(profile.memory, peak_resident_bytes())
    deadline = time.monotonic() + profile.seconds
    running = threading.Event()
    finished = threading.Event()
    exceeded = threading.Event()
    expired = threading.Event()

    def watch() -> None:
        while not finished.wait(min(0.25, max(0.0, deadline - time.monotonic()))):
            if peak_resident_bytes() > ceiling:
                exceeded.set()

            elif time.monotonic() < deadline:
                continue

            expired.set()
            _thread.interrupt_main(signal.SIGINT)

            return

    def expire(signum: int, frame: FrameType | None) -> None:
        del signum, frame

        if not expired.is_set():
            raise KeyboardInterrupt

        if running.is_set():
            raise _Expired

    running.set()
    previous = signal.signal(signal.SIGINT, expire)
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()

    try:
        yield
    except _Expired:
        if exceeded.is_set():
            msg = (
                f"the run held {peak_resident_bytes()} resident bytes, "
                f"past its {profile.memory}"
            )
        else:
            msg = f"the run took longer than its {profile.seconds} seconds"

        raise ProfileExceededError(msg) from None
    finally:
        running.clear()
        finished.set()
        watcher.join()
        signal.signal(signal.SIGINT, previous)

    peak = peak_resident_bytes()

    if peak > ceiling:
        msg = f"the run held {peak} resident bytes, past its {profile.memory}"
        raise ProfileExceededError(msg)

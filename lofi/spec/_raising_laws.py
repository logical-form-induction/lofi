"""Rule-law cases for type raising."""

from random import Random
from typing import TYPE_CHECKING, Final

from hypothesis import strategies as st

from lofi.judge.laws import (
    RuleFixture,
    RuleLawCase,
    RuleLawCases,
    RuleTotalityCase,
    check_rule_laws,
)
from lofi.spec._rule import TypeRaisingSpec
from lofi.spec._rule_laws import LawGrammar

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi._types import JsonValue
    from lofi.judge import Rule
    from lofi.spec._rule_laws import LawView, SlashRole
    from lofi.terms import Term

__all__ = ["provide_rule_laws"]

_R3_DRAWS: Final = 3_000
_R3_SEED: Final = 0
_TARGETS: Final = ("t", "u")
_ATOMS: Final = ("x", *_TARGETS, "other")
_VIEWS: Final[tuple[LawView, ...]] = ("direct", "swapped", "undirected")
_ROLES: Final[tuple[SlashRole, ...]] = ("forward", "backward")


def _raised(grammar: LawGrammar, role: SlashRole, target: str) -> Term:
    """Construct the category raising ``x`` over one target."""
    functors, slash = grammar.functor(role)
    target_term = grammar.atom(target)
    inner = functors.make(functors.opposite(slash), target_term, grammar.atom("x"))

    return functors.make(slash, target_term, inner)


class _Rebuild:
    """Rebuild a type-raising rule from its declaration."""

    __slots__ = ("_atoms", "_grammar")

    def __init__(self, grammar: LawGrammar, /) -> None:
        self._grammar = grammar
        self._atoms = {grammar.atom(atom).payload: atom for atom in _ATOMS}

    def _atom(self, value: JsonValue) -> str:
        """Return the declared text of one encoded law atom.

        Raises
        ------
        TypeError
            If the encoding is not one of the law atoms.
        """
        if (
            not isinstance(value, list)
            or len(value) != 3
            or not isinstance(value[1], int)
            or value[1] not in self._atoms
        ):
            msg = f"invalid type-raising category: {value!r}"
            raise TypeError(msg)

        return self._atoms[value[1]]

    def _atoms_of(self, value: JsonValue) -> tuple[str, ...] | None:
        """Return the declared texts of an encoded atom list, or ``None``.

        Raises
        ------
        TypeError
            If the value is neither ``None`` nor a list.
        """
        if value is None:
            return None

        if not isinstance(value, list):
            msg = f"invalid type-raising category list: {value!r}"
            raise TypeError(msg)

        return tuple(self._atom(atom) for atom in value)

    def __call__(self, spec: Mapping[str, JsonValue], /) -> Rule[Term]:
        """Compile the grammar the declaration names.

        Parameters
        ----------
        spec : Mapping[str, JsonValue]
            Rule declaration.

        Raises
        ------
        KeyError
            If a required field or the slash's view is missing.
        TypeError
            If a field or a target has the wrong type.
        """
        slash = spec["slash"]

        if not isinstance(slash, str):
            msg = f"invalid type-raising declaration: {spec!r}"
            raise TypeError(msg)

        return self._grammar.rule(
            TypeRaisingSpec(
                view=self._grammar.view_of(slash),
                targets=self._atoms_of(spec["targets"]),
                arguments=self._atoms_of(spec["arguments"]),
            )
        )


def _ordinary_case(
    grammar: LawGrammar, role: SlashRole, *, targeted: bool
) -> RuleLawCase:
    """Build one ordinary type-raising case."""
    rule = grammar.rule(
        TypeRaisingSpec(
            view=grammar.view_name(role),
            targets=_TARGETS if targeted else None,
            arguments=None,
        )
    )
    outputs = tuple(_raised(grammar, role, target) for target in _TARGETS)
    atoms = tuple(grammar.atom(atom) for atom in _ATOMS)
    terms = st.sampled_from((*atoms, *outputs))
    fixture = RuleFixture(rule, terms, _Rebuild(grammar), 100)
    mode = "targeted" if targeted else "untargeted"

    def check() -> None:
        """Run every law for the captured type-raising fixture."""
        check_rule_laws(fixture)

    return RuleLawCase(f"raising/{grammar.view}/{role}/{mode}", rule.name, check)


def _totality_case(grammar: LawGrammar) -> RuleTotalityCase:
    """Build the targeted forward-raising totality case."""
    rule = grammar.rule(
        TypeRaisingSpec(
            view=grammar.view_name("forward"), targets=_TARGETS[:1], arguments=None
        )
    )
    pool = (
        *(grammar.atom(atom) for atom in _ATOMS),
        _raised(grammar, "forward", _TARGETS[0]),
    )

    def check() -> int:
        """Run the captured seed-0 totality draws."""
        random = Random(_R3_SEED)
        calls = 0

        for _ in range(_R3_DRAWS):
            inputs = (random.choice(pool),)
            output = random.choice(pool)
            rule.admits(inputs, output)
            calls += 1

        return calls

    return RuleTotalityCase("r3/raising/forward/targeted", rule.name, check)


def provide_rule_laws() -> RuleLawCases:
    """Return ordinary and totality type-raising cases."""
    grammars = {view: LawGrammar(view) for view in _VIEWS}
    ordinary = tuple(
        _ordinary_case(grammars[view], role, targeted=targeted)
        for view in _VIEWS
        for role in _ROLES
        for targeted in (True, False)
    )

    return RuleLawCases("raising", ordinary, (_totality_case(grammars["direct"]),))

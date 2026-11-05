"""Rules derived from term patterns."""

from itertools import product
from typing import TYPE_CHECKING, cast, final

from lofi._checks import ensure_name
from lofi._frozen import FrozenValue
from lofi.judge._index import RuleProbe
from lofi.judge._ports import HOLE, NormalFormRole
from lofi.judge._proof import _deterministic
from lofi.judge._spine import match_spine
from lofi.terms import (
    ArityError,
    Bindings,
    ExactMatching,
    MVar,
    Node,
    PayloadError,
    PVar,
    SortError,
    Spine,
    TermError,
    TermRuleKernel,
    UnknownOperatorError,
    build,
    match,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from lofi._types import JsonValue
    from lofi.judge._ports import Direction, SchematicSequent, Slot
    from lofi.judge._values import RuleRestriction
    from lofi.terms import (
        Operator,
        Pattern,
        Signature,
        Term,
        TermMatching,
        TermStore,
        _CompiledRule,
    )

__all__ = ["PatternRule"]


type _Variables = tuple[
    frozenset[str],
    frozenset[str],
    frozenset[str],
    frozenset[str],
]
"""Term, operator, payload, and spine variables."""

type _Context = tuple[tuple[Operator, tuple[MVar, ...]], ...]
"""Spine frames, outermost first, each an operator and its sibling variables."""


def _merge_variables(left: _Variables, right: _Variables) -> _Variables:
    """Combine two variable inventories."""
    return (
        left[0] | right[0],
        left[1] | right[1],
        left[2] | right[2],
        left[3] | right[3],
    )


def _variables_except(variables: tuple[_Variables, ...], omitted: int) -> _Variables:
    """Combine every variable inventory except one."""
    found: _Variables = (
        frozenset(),
        frozenset(),
        frozenset(),
        frozenset(),
    )

    for at, held in enumerate(variables):
        if at != omitted:
            found = _merge_variables(found, held)

    return found


def _variables(pattern: Pattern) -> _Variables:
    """Return the variables a pattern requires."""
    if isinstance(pattern, MVar):
        return (frozenset({pattern.name}), frozenset(), frozenset(), frozenset())

    if isinstance(pattern, Spine):
        terms, operators, payloads, spines = _variables(pattern.core)

        return (terms, operators, payloads, spines | {pattern.name})

    operators = (
        frozenset({pattern.op.name}) if isinstance(pattern.op, PVar) else frozenset()
    )
    payloads = (
        frozenset({pattern.payload.name})
        if isinstance(pattern.payload, PVar)
        else frozenset()
    )
    found: _Variables = (frozenset(), operators, payloads, frozenset())

    for child in pattern.args:
        found = _merge_variables(found, _variables(child))

    return found


def _ground(
    pattern: Pattern,
    store: TermStore,
    variables: _Variables | None = None,
) -> Term | None:
    """Return the one term a variable-free pattern matches, if it can be built.

    A pattern naming an operator the store cannot build matches nothing, and
    is given no ground term, so it constrains nothing.
    """
    if any(_variables(pattern) if variables is None else variables):
        return None

    try:
        return build(pattern, store, Bindings())
    except TermError:
        return None


def _available(bindings: Bindings) -> _Variables:
    """Return the variables bound in an environment."""
    operators = frozenset(
        name for name, value in bindings.payloads.items() if isinstance(value, str)
    )
    payloads = frozenset(
        name for name, value in bindings.payloads.items() if isinstance(value, int)
    )

    return (
        frozenset(bindings.terms),
        operators,
        payloads,
        frozenset(bindings.spines),
    )


def _covered(required: _Variables, available: _Variables) -> bool:
    """Return whether every required variable is available."""
    return all(need <= held for need, held in zip(required, available, strict=True))


def _term_variables(pattern: Pattern) -> tuple[MVar, ...]:
    """Return term variables in first-occurrence order."""
    found: dict[str, MVar] = {}

    def visit(current: Pattern) -> None:
        if isinstance(current, MVar):
            prior = found.get(current.name)

            if prior is not None and prior.sort != current.sort:
                msg = (
                    f"term variable {current.name!r} has both sort "
                    f"{prior.sort!r} and {current.sort!r}"
                )
                raise ValueError(msg)

            found.setdefault(current.name, current)
        elif isinstance(current, Spine):
            visit(current.core)
        else:
            for child in current.args:
                visit(child)

    visit(pattern)
    return tuple(found.values())


def _bind_terms(pattern: Pattern, bindings: Mapping[str, Term]) -> Pattern:
    """Replace declared term variables by fixed finite targets."""
    if isinstance(pattern, MVar):
        return _fixed(bindings[pattern.name]) if pattern.name in bindings else pattern

    if isinstance(pattern, Spine):
        return Spine(
            pattern.name,
            _bind_terms(pattern.core, bindings),
            pattern.ops,
            slot=pattern.slot,
            min_order=pattern.min_order,
            max_order=pattern.max_order,
        )

    return Node(
        pattern.op,
        tuple(_bind_terms(child, bindings) for child in pattern.args),
        pattern.payload,
    )


def _finite_conclusions(
    conclusion: Pattern,
    required: _Variables,
    premise_variables: _Variables,
    targets: tuple[Term, ...] | None,
) -> tuple[Pattern, ...]:
    """Instantiate unbound conclusion terms from one declared finite domain."""
    if targets is None:
        return (conclusion,) if _covered(required, premise_variables) else ()

    missing = tuple(
        variable
        for variable in _term_variables(conclusion)
        if variable.name not in premise_variables[0]
    )
    if not missing:
        msg = "finite targets require an unbound conclusion term variable"
        raise ValueError(msg)
    if any(
        not needed <= available
        for needed, available in zip(required[1:], premise_variables[1:], strict=True)
    ):
        msg = "finite targets cannot bind operator, payload, or spine variables"
        raise ValueError(msg)

    choices = tuple(
        tuple(target for target in targets if target.op.sort == variable.sort)
        for variable in missing
    )
    unavailable = next(
        (
            variable.sort
            for variable, candidates in zip(missing, choices, strict=True)
            if not candidates
        ),
        None,
    )
    if unavailable is not None:
        msg = f"finite targets contain no term of sort {unavailable!r}"
        raise ValueError(msg)

    conclusions: list[Pattern] = []
    for values in product(*choices):
        bound = _bind_terms(
            conclusion,
            {
                variable.name: value
                for variable, value in zip(missing, values, strict=True)
            },
        )
        if bound not in conclusions:
            conclusions.append(bound)

    return tuple(conclusions)


def _pattern_spec(pattern: Pattern) -> dict[str, JsonValue]:
    """Return a JSON declaration for a pattern."""
    if isinstance(pattern, MVar):
        return {"kind": "term", "name": pattern.name, "sort": pattern.sort}

    if isinstance(pattern, Spine):
        return {
            "kind": "spine",
            "name": pattern.name,
            "core": _pattern_spec(pattern.core),
            "ops": list(pattern.ops),
            "slot": pattern.slot,
            "min_order": pattern.min_order,
            "max_order": pattern.max_order,
        }

    operator: JsonValue = (
        {"variable": pattern.op.name} if isinstance(pattern.op, PVar) else pattern.op
    )
    payload: JsonValue = (
        {"variable": pattern.payload.name}
        if isinstance(pattern.payload, PVar)
        else pattern.payload
    )

    return {
        "kind": "node",
        "op": operator,
        "args": [_pattern_spec(child) for child in pattern.args],
        "payload": payload,
    }


def _fixed(term: Term) -> Node:
    """Return the ground pattern that names exactly one term."""
    return Node(term.op.name, tuple(_fixed(child) for child in term.args), term.payload)


def _keyed(
    pattern: Pattern,
    store: TermStore,
    matching: TermMatching,
    variables: _Variables | None = None,
) -> Pattern:
    """Return a pattern whose ground subpatterns name their class representative.

    A premise is compared by its representative under the rule's congruence,
    so each maximal ground subpattern is replaced by the pattern of its own
    representative, and the pattern then matches every member of the class.
    A ground subpattern naming no term of the store matches nothing and is
    kept as declared.
    """
    if isinstance(pattern, MVar):
        return pattern

    if isinstance(pattern, Spine):
        return Spine(
            pattern.name,
            _keyed(pattern.core, store, matching),
            pattern.ops,
            slot=pattern.slot,
            min_order=pattern.min_order,
            max_order=pattern.max_order,
        )

    if not any(_variables(pattern) if variables is None else variables):
        try:
            term = build(pattern, store, Bindings())
        except ArityError, PayloadError, SortError, UnknownOperatorError:
            return pattern
        else:
            return _fixed(matching.key(term))

    return Node(
        pattern.op,
        tuple(_keyed(child, store, matching) for child in pattern.args),
        pattern.payload,
    )


def _head_constraint(pattern: Pattern) -> frozenset[str] | None:
    """Return conservative root-operator names for a pattern."""
    if isinstance(pattern, MVar):
        return None

    if isinstance(pattern, Node):
        return None if isinstance(pattern.op, PVar) else frozenset({pattern.op})

    heads = frozenset(pattern.ops)

    if pattern.min_order > 0:
        return heads

    core = _head_constraint(pattern.core)

    return None if core is None else heads | core


def _root_matches(
    pattern: Pattern,
    term: Term,
    bindings: Bindings,
) -> tuple[Bindings, ...]:
    """Enumerate bindings for one root pattern."""
    if isinstance(pattern, Spine):
        return tuple(
            found.bindings for found in match_spine(pattern, term, bindings).matches
        )

    held = match(pattern, term, bindings)

    return () if held is None else (held,)


def _match_pairs(
    pairs: Sequence[tuple[Pattern, Term]],
    bindings: Bindings | None = None,
) -> tuple[Bindings, ...]:
    """Return environments satisfying a pattern-term sequence."""
    environments = (Bindings() if bindings is None else bindings,)

    for pattern, term in pairs:
        environments = tuple(
            updated
            for environment in environments
            for updated in _root_matches(pattern, term, environment)
        )

        if not environments:
            break

    return environments


def _spines(pattern: Pattern) -> Iterator[Spine]:
    """Yield every spine occurrence in a pattern, outermost first."""
    if isinstance(pattern, Spine):
        yield pattern
        yield from _spines(pattern.core)
    elif isinstance(pattern, Node):
        for child in pattern.args:
            yield from _spines(child)


def _bounded_contexts(
    occurrences: Sequence[Spine],
    signature: Signature,
) -> tuple[_Context, ...] | None:
    """Return every context that the occurrences of one spine variable admit.

    Each context fixes the spine's order and the operator of every frame, and
    gives each frame's other arguments fresh term variables at the sorts the
    operator declares, so substituting it yields a deterministic pattern.

    Parameters
    ----------
    occurrences : Sequence[Spine]
        The spine patterns sharing one context variable.
    signature : Signature
        The signature that declares the frame operators.

    Returns
    -------
    tuple[_Context, ...] | None
        The contexts in increasing order, or ``None`` when an occurrence has
        no finite maximum, which no finite deterministic family presents.
    """
    maxima = tuple(spine.max_order for spine in occurrences)

    if any(maximum is None for maximum in maxima):
        return None

    first = occurrences[0]
    low = max(spine.min_order for spine in occurrences)
    high = min(maximum for maximum in maxima if maximum is not None)

    if any(spine.slot != first.slot for spine in occurrences):
        high = min(high, 0)

    declared = {operator.name: operator for operator in signature.operators}
    operators: list[Operator] = []

    for name in dict.fromkeys(first.ops):
        operator = declared.get(name)

        if operator is None or any(name not in spine.ops for spine in occurrences):
            continue

        if first.slot < operator.arity:
            operators.append(operator)

    return tuple(
        tuple(
            (
                operator,
                tuple(
                    MVar(f"{first.name}#{depth}.{position}", sort)
                    for position, sort in enumerate(operator.arg_sorts)
                    if position != first.slot
                ),
            )
            for depth, operator in enumerate(frames)
        )
        for order in range(low, high + 1)
        for frames in product(operators, repeat=order)
    )


def _instantiate(pattern: Pattern, contexts: Mapping[str, _Context]) -> Pattern:
    """Replace every spine in a pattern by its chosen context."""
    if isinstance(pattern, MVar):
        return pattern

    if isinstance(pattern, Spine):
        built = _instantiate(pattern.core, contexts)

        for operator, siblings in reversed(contexts[pattern.name]):
            arguments: list[Pattern] = list(siblings)
            arguments.insert(pattern.slot, built)
            built = Node(operator.name, tuple(arguments))

        return built

    return Node(
        pattern.op,
        tuple(_instantiate(child, contexts) for child in pattern.args),
        payload=pattern.payload,
    )


def _schemata(
    premises: tuple[Pattern, ...],
    conclusion: Pattern,
    signature: Signature,
) -> tuple[SchematicSequent, ...]:
    """Return the deterministic sequents a pattern rule presents.

    A rule without spines presents itself when it is deterministic. A rule
    whose spines are all bounded presents one sequent per choice of context
    for each spine variable, shared by every occurrence of that variable.

    Parameters
    ----------
    premises : tuple[Pattern, ...]
        The premise patterns.
    conclusion : Pattern
        The conclusion pattern.
    signature : Signature
        The signature that declares the frame operators.

    Returns
    -------
    tuple[SchematicSequent, ...]
        The sequents, or nothing when the rule has an operator or payload
        variable or an unbounded spine.
    """
    occurrences: dict[str, list[Spine]] = {}

    for pattern in (*premises, conclusion):
        for spine in _spines(pattern):
            occurrences.setdefault(spine.name, []).append(spine)

    choices: list[tuple[tuple[str, _Context], ...]] = []

    for name, held in occurrences.items():
        contexts = _bounded_contexts(held, signature)

        if contexts is None:
            return ()

        choices.append(tuple((name, context) for context in contexts))

    schemata: list[SchematicSequent] = []

    for chosen in product(*choices):
        bound = dict(chosen)
        instance = tuple(_instantiate(premise, bound) for premise in premises)
        result = _instantiate(conclusion, bound)

        if not all(_deterministic(pattern) for pattern in (*instance, result)):
            return ()

        schemata.append((instance, result))

    return tuple(schemata)


@final
class PatternRule(FrozenValue):
    """Rule specified by premise and conclusion patterns.

    Premises are compared by their representative under the rule's
    congruence, so a ground subpattern matches every member of the class it
    names: under feature erasure ``S[to]`` matches ``S[dcl]``, and under
    identity it matches only itself.

    Parameters
    ----------
    compiled : _CompiledRule
        The compiled pattern, coercion, absorption or coordination recipe.
    name : str
        The rule name.
    premises : Sequence[Pattern]
        The premise patterns in rule order.
    conclusion : Pattern
        The conclusion pattern.
    store : TermStore
        The store the patterns' ground terms live in.
    matching : TermMatching
        The premise congruence.
    normal_form_role : NormalFormRole | None
        The role normal-form filters read, if declared.
    restriction : RuleRestriction | None
        The rule-instance restriction, if declared.
    probe : RuleProbe | None
        Premise-head constraints, inferred from the premises unless given.
    targets : Sequence[Term] | None
        Finite conclusion targets, if declared.
    """

    _noun = "pattern rule"

    __slots__ = (
        "_conclusion",
        "_conclusion_keys",
        "_directions",
        "_forward_conclusions",
        "_grounds",
        "_lowering",
        "_matching",
        "_name",
        "_normal_form_role",
        "_premise_keys",
        "_premises",
        "_probe",
        "_reachability_schemata",
        "_required",
        "_residual_slots",
        "_restriction",
        "_schemata",
        "_store",
        "_targets",
    )

    def __init__(
        self,
        compiled: _CompiledRule,
        name: str,
        premises: Sequence[Pattern],
        conclusion: Pattern,
        store: TermStore,
        matching: TermMatching,
        /,
        *,
        normal_form_role: NormalFormRole | None = None,
        restriction: RuleRestriction | None = None,
        probe: RuleProbe | None = None,
        targets: Sequence[Term] | None = None,
    ) -> None:
        held_matching = matching
        direct_schemata = compiled.direct_schemata
        held_premises = tuple(premises)

        if probe is not None and probe.arity != len(held_premises):
            msg = (
                f"pattern rule {name!r} has arity {len(held_premises)} but its "
                f"explicit probe has arity {probe.arity}"
            )
            raise ValueError(msg)

        variables = tuple(_variables(premise) for premise in held_premises)
        premise_variables: _Variables = (
            frozenset(),
            frozenset(),
            frozenset(),
            frozenset(),
        )

        for held in variables:
            premise_variables = _merge_variables(premise_variables, held)

        conclusion_variables = _variables(conclusion)
        held_targets = None if targets is None else tuple(targets)
        if held_targets == ():
            msg = "finite pattern-rule targets must be nonempty"
            raise ValueError(msg)
        forward_conclusions = _finite_conclusions(
            conclusion,
            conclusion_variables,
            premise_variables,
            held_targets,
        )
        residual_slots = frozenset(
            at
            for at, required in enumerate(variables)
            if _covered(
                required,
                _merge_variables(
                    conclusion_variables,
                    _variables_except(variables, at),
                ),
            )
        )
        directions: set[Direction] = {"check"}
        constraints = tuple(_head_constraint(premise) for premise in held_premises)
        inferred_probe = (
            None
            if not (isinstance(held_matching, ExactMatching) and held_matching.exact)
            or all(constraint is None for constraint in constraints)
            else RuleProbe(len(held_premises), (constraints,))
        )
        held_probe = inferred_probe if probe is None else probe
        held_premise_keys = (
            tuple(
                _keyed(premise, store, held_matching, required)
                for premise, required in zip(held_premises, variables, strict=True)
            )
            if held_matching.has_key
            else held_premises
        )
        admitted_conclusions = forward_conclusions or (conclusion,)
        held_conclusion_keys = tuple(
            _keyed(held, store, held_matching) if held_matching.has_key else held
            for held in admitted_conclusions
        )

        if forward_conclusions:
            directions.add("forward")

        if residual_slots:
            directions.add("residual")

        object.__setattr__(self, "_name", ensure_name("name", name))
        object.__setattr__(
            self,
            "_normal_form_role",
            NormalFormRole() if normal_form_role is None else normal_form_role,
        )
        object.__setattr__(self, "_premises", held_premises)
        object.__setattr__(self, "_conclusion", conclusion)
        object.__setattr__(self, "_targets", held_targets)
        object.__setattr__(self, "_required", conclusion_variables)
        object.__setattr__(
            self,
            "_grounds",
            tuple(
                _ground(premise, store, required)
                for premise, required in zip(
                    held_premise_keys,
                    variables,
                    strict=True,
                )
            ),
        )
        object.__setattr__(self, "_premise_keys", held_premise_keys)
        object.__setattr__(self, "_conclusion_keys", held_conclusion_keys)
        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_matching", held_matching)
        object.__setattr__(self, "_directions", frozenset(directions))
        object.__setattr__(self, "_forward_conclusions", forward_conclusions)
        object.__setattr__(self, "_probe", held_probe)
        object.__setattr__(self, "_restriction", restriction)
        object.__setattr__(self, "_residual_slots", residual_slots)
        object.__setattr__(
            self,
            "_schemata",
            (
                tuple((held_premises, held) for held in admitted_conclusions)
                if direct_schemata
                else tuple(
                    schema
                    for held in admitted_conclusions
                    for schema in _schemata(held_premises, held, store.signature)
                )
            ),
        )
        object.__setattr__(
            self,
            "_reachability_schemata",
            (
                tuple((held_premise_keys, held) for held in held_conclusion_keys)
                if direct_schemata
                else tuple(
                    schema
                    for held in held_conclusion_keys
                    for schema in _schemata(held_premise_keys, held, store.signature)
                )
            ),
        )
        object.__setattr__(
            self,
            "_lowering",
            TermRuleKernel(
                compiled.pattern_kernel(
                    held_premise_keys,
                    admitted_conclusions,
                    held_matching,
                    store,
                )
            ),
        )

    @property
    def name(self) -> str:
        """Return the rule name."""
        return self._name

    @property
    def normal_form_role(self) -> NormalFormRole:
        """Return the role declared for normal-form filters."""
        return self._normal_form_role

    @property
    def premises(self) -> tuple[Pattern, ...]:
        """Return the premise patterns."""
        return self._premises

    @property
    def proof_schemata(self) -> tuple[SchematicSequent, ...]:
        """Return the deterministic sequents the patterns present.

        A bounded spine contributes one sequent per order and choice of frame
        operators; an unbounded spine or an operator or payload variable
        presents none.
        """
        return self._schemata

    @property
    def reachability_schemata(self) -> tuple[SchematicSequent, ...]:
        """Return the consumed premises and introduced conclusions."""
        return self._reachability_schemata

    @property
    def conclusion(self) -> Pattern:
        """Return the conclusion pattern."""
        return self._conclusion

    @property
    def targets(self) -> tuple[Term, ...] | None:
        """Return the finite domain of unbound conclusion terms, if any."""
        return self._targets

    @property
    def arity(self) -> int:
        """Return the premise count."""
        return len(self._premises)

    @property
    def directions(self) -> frozenset[Direction]:
        """Return the directions derived from variable coverage."""
        return self._directions

    @property
    def spine_order(self) -> int:
        """Return the spine order of every instance."""
        return 0

    @property
    def residual_slots(self) -> frozenset[int]:
        """Return the premise slots determined by the other patterns."""
        return self._residual_slots

    @property
    def forward_is_complete(self) -> bool:
        """Report whether forward construction is complete."""
        return "forward" in self._directions

    @property
    def matching(self) -> TermMatching:
        """Return the term congruence used by the rule."""
        return self._matching

    @property
    def restriction(self) -> RuleRestriction | None:
        """Return the rule-instance restriction, if declared."""
        return self._restriction

    @property
    def probe(self) -> RuleProbe | None:
        """Return conservative premise-head constraints, if available."""
        return self._probe

    @property
    def lowering(self) -> TermRuleKernel:
        """Return the native implementation of this relation."""
        return self._lowering

    @property
    def ground_premises(self) -> tuple[Term | None, ...]:
        """Return, per premise, the one term a variable-free pattern matches."""
        return self._grounds

    def _representative(self, term: Term) -> Term:
        """Return a matching key when the congruence provides one."""
        return self._matching.key(term) if self._matching.has_key else term

    def _input_bindings(self, inputs: tuple[Term, ...]) -> tuple[Bindings, ...]:
        """Return environments licensed by the premises."""
        if len(inputs) != self.arity:
            return ()

        pairs = tuple(
            (pattern, self._representative(term))
            for pattern, term in zip(self._premise_keys, inputs, strict=True)
        )

        return _match_pairs(pairs)

    def apply(self, inputs: tuple[Term, ...], /) -> Iterator[Term]:
        """Enumerate conclusions for complete premise bindings."""
        if "forward" in self._directions:
            yield from self._lowering.apply(inputs)

    def admits(
        self,
        inputs: tuple[Term, ...],
        output: Term,
        /,
    ) -> bool:
        """Return whether the patterns relate the supplied terms."""
        if len(inputs) != self.arity:
            return False

        representative = self._representative(output)
        environments = self._input_bindings(inputs)

        return any(
            _root_matches(conclusion, representative, environment)
            for conclusion in self._conclusion_keys
            for environment in environments
        )

    def residuate(
        self,
        output: Term,
        known: tuple[Slot[Term], ...],
        /,
    ) -> Iterator[Term]:
        """Enumerate constructible fillers for one premise hole."""
        holes = tuple(at for at, value in enumerate(known) if value is HOLE)

        if len(known) != self.arity or len(holes) != 1:
            return

        missing = holes[0]

        if missing not in self._residual_slots:
            return

        pairs = tuple(
            (pattern, self._representative(cast("Term", value)))
            for at, (pattern, value) in enumerate(
                zip(self._premise_keys, known, strict=True)
            )
            if at != missing and value is not HOLE
        )
        environments = _match_pairs(pairs)
        representative = self._representative(output)
        required = _variables(self._premises[missing])
        emitted: set[Term] = set()

        for conclusion in self._conclusion_keys:
            for environment in environments:
                for bindings in _root_matches(conclusion, representative, environment):
                    if not _covered(required, _available(bindings)):
                        continue

                    term = build(self._premises[missing], self._store, bindings)

                    if term not in emitted:
                        emitted.add(term)
                        yield term

    @property
    def relation(self) -> Mapping[str, JsonValue]:
        """Return the patterns and congruence that fix the relation.

        Premises are given as the rule compares them, with each ground part
        replaced by its class representative, so patterns naming different
        members of one class denote the same relation.
        """
        return {
            "rule": "pattern",
            "signature": self._store.signature.digest,
            "matching": self._matching.name,
            "premises": [_pattern_spec(pattern) for pattern in self._premise_keys],
            "conclusion": _pattern_spec(self._conclusion),
            "targets": (
                None
                if self._targets is None
                else [_pattern_spec(_fixed(target)) for target in self._targets]
            ),
            "restriction": (
                None if self._restriction is None else self._restriction.to_spec()
            ),
        }

    def to_spec(self) -> Mapping[str, JsonValue]:
        """Return the rule's JSON declaration."""
        return {
            "rule": "pattern",
            "name": self._name,
            "signature": self._store.signature.digest,
            "matching": self._matching.name,
            "premises": [_pattern_spec(pattern) for pattern in self._premises],
            "conclusion": _pattern_spec(self._conclusion),
            "targets": (
                None
                if self._targets is None
                else [_pattern_spec(_fixed(target)) for target in self._targets]
            ),
            "restriction": (
                None if self._restriction is None else self._restriction.to_spec()
            ),
        }

"""Located constraint generation for logical forms."""

from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple, final

from lofi._checks import ensure_non_negative
from lofi.terms import Bindings, MVar, NotLocallyClosedError, Spine, at_path, build
from lofi.typecheck._unify import Unifier
from lofi.typecheck._values import Constraint

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi.terms import OpName, Path, Pattern, Sort, Term, TermStore
    from lofi.typecheck._lexsig import LexicalSignature
    from lofi.typecheck._values import Scheme, TypingRule

__all__ = ["UnknownSymbolError", "generate"]


@final
class UnknownSymbolError(ValueError):
    """A missing lexical declaration and its source location."""

    __slots__ = ("path", "symbol", "term")

    def __init__(self, symbol: int, term: Term, path: Sequence[int]) -> None:
        """Initialize the error from a symbol node and path."""
        self.symbol = ensure_non_negative("unknown symbol", symbol)
        self.term = term
        self.path = tuple(
            ensure_non_negative(f"path[{at}]", position)
            for at, position in enumerate(path)
        )
        super().__init__(
            f"lexical signature declares no symbol {self.symbol} at {self.path!r}"
        )


@final
class _FreshSupply:
    """Fresh type metavariables for one generation run."""

    __slots__ = ("_blocked", "_issued", "_next", "_sort", "_store")

    def __init__(self, store: TermStore, sort: Sort, blocked: int) -> None:
        self._store = store
        self._sort = sort
        self._blocked = blocked
        self._issued = 0
        self._next = 0

    @property
    def issued(self) -> int:
        """Return the mask of issued atom indices."""
        return self._issued

    def take(self) -> Term:
        """Return the next unblocked type metavariable."""
        while True:
            variable = self._store.fvar(self._sort, f"$type{self._next}")
            self._next += 1
            atom = variable.payload

            if atom is None:
                msg = "the type signature built a metavariable without an atom"
                raise ValueError(msg)

            bit = 1 << atom

            if not self._blocked & bit:
                self._blocked |= bit
                self._issued |= bit

                return variable


class _GenerationResult(NamedTuple):
    """Generated constraints and fresh-atom provenance."""

    assignment: Mapping[Path, Term]
    constraints: tuple[Constraint, ...]
    issued: int


def _annotations(
    term: Term,
    store: TermStore,
    annotations: Mapping[Path, Term] | None,
) -> Mapping[Path, Term]:
    """Validate and freeze caller-supplied annotations."""
    if annotations is None:
        return MappingProxyType({})

    checked: dict[Path, Term] = {}

    for supplied_path, annotation in annotations.items():
        path = tuple(
            ensure_non_negative(f"annotation path[{at}]", position)
            for at, position in enumerate(supplied_path)
        )
        node = at_path(term, path)

        if not store.contains(annotation):
            msg = f"annotation at {path!r} is outside the type store lineage"
            raise ValueError(msg)

        if not annotation.is_closed:
            raise NotLocallyClosedError(max_index=annotation.maxidx, depth=0)

        if node.args:
            if not node.op.is_binder:
                msg = f"annotation at {path!r} does not address a binder or leaf"
                raise ValueError(msg)
        elif not (
            store.signature.is_variable(node.op) or store.signature.is_constant(node.op)
        ):
            msg = f"annotation at {path!r} does not address an inferred leaf"
            raise ValueError(msg)

        checked[path] = annotation

    return MappingProxyType(checked)


def _injected(
    term: Term,
    store: TermStore,
    constraints: Mapping[Path, Term] | None,
) -> Mapping[Path, Term]:
    """Validate and freeze caller-supplied located type constraints."""
    if constraints is None:
        return MappingProxyType({})

    checked: dict[Path, Term] = {}

    for supplied_path, expected in constraints.items():
        path = tuple(
            ensure_non_negative(f"constraint path[{at}]", position)
            for at, position in enumerate(supplied_path)
        )
        at_path(term, path)

        if not store.contains(expected):
            msg = f"constraint at {path!r} is outside the type store lineage"
            raise ValueError(msg)

        if not expected.is_closed:
            raise NotLocallyClosedError(max_index=expected.maxidx, depth=0)

        checked[path] = expected

    return MappingProxyType(checked)


def _pattern_atoms(pattern: Pattern, free: frozenset[OpName]) -> int:
    """Return fixed free variable atoms named by a type pattern.

    Parameters
    ----------
    pattern : Pattern
        The type pattern.
    free : frozenset[OpName]
        The names of the type signature's free variable operators.
    """
    if isinstance(pattern, MVar):
        return 0

    if isinstance(pattern, Spine):
        return _pattern_atoms(pattern.core, free)

    held = (
        1 << pattern.payload
        if pattern.op in free
        and isinstance(pattern.payload, int)
        and not isinstance(pattern.payload, bool)
        and pattern.payload >= 0
        else 0
    )

    for child in pattern.args:
        held |= _pattern_atoms(child, free)

    return held


def _blocked_atoms(
    store: TermStore,
    signature: LexicalSignature,
    rules: Mapping[OpName, TypingRule],
    annotations: Mapping[Path, Term],
    injected: Mapping[Path, Term],
) -> int:
    """Return atoms that fresh variables must not reuse."""
    free = frozenset(
        op.name
        for op in store.signature.operators
        if store.signature.is_free_variable(op)
    )
    blocked = 0

    for scheme in signature.schemes.values():
        blocked |= scheme.body.free

    for rule in rules.values():
        for pattern in (*rule.binds, *rule.premises, rule.result):
            if pattern is not None:
                blocked |= _pattern_atoms(pattern, free)

    for annotation in annotations.values():
        blocked |= annotation.free

    for constraint in injected.values():
        blocked |= constraint.free

    return blocked


def _pattern_variables(
    pattern: Pattern,
    variables: dict[str, Sort],
) -> None:
    """Collect a type pattern's term metavariables by name."""
    if isinstance(pattern, MVar):
        held = variables.get(pattern.name)

        if held is not None and held != pattern.sort:
            msg = (
                f"type pattern variable {pattern.name!r} has both sort "
                f"{held!r} and {pattern.sort!r}"
            )
            raise ValueError(msg)

        variables[pattern.name] = pattern.sort

        return

    if isinstance(pattern, Spine):
        msg = "typing rules cannot contain spine patterns"
        raise TypeError(msg)

    for child in pattern.args:
        _pattern_variables(child, variables)


def _instantiate_rule(
    store: TermStore,
    supply: _FreshSupply,
    rule: TypingRule,
) -> tuple[tuple[Term | None, ...], tuple[Term, ...], Term]:
    """Instantiate one typing rule with fresh metavariables."""
    variables: dict[str, Sort] = {}

    for pattern in (*rule.binds, *rule.premises, rule.result):
        if pattern is not None:
            _pattern_variables(pattern, variables)

    terms = {name: supply.take() for name in variables}
    bindings = Bindings(terms=terms)
    bound = tuple(
        None if pattern is None else build(pattern, store, bindings)
        for pattern in rule.binds
    )
    premises = tuple(build(pattern, store, bindings) for pattern in rule.premises)

    return bound, premises, build(rule.result, store, bindings)


def _instantiate_scheme(
    unifier: Unifier,
    supply: _FreshSupply,
    scheme: Scheme,
) -> Term:
    """Instantiate a symbol scheme for one occurrence."""
    substitution = {atom: supply.take() for atom in scheme.quantified}

    return unifier.apply(substitution, scheme.body)


def _variable_payload(
    store: TermStore,
    term: Term,
) -> int:
    """Return a variable payload as a nonnegative integer."""
    payload = term.payload

    if not store.signature.is_variable(term.op) or payload is None:
        msg = f"malformed logical-form variable {term.op.name!r}: {term!r}"
        raise ValueError(msg)

    return ensure_non_negative("variable payload", payload)


@final
class _Generator:
    """Constraint-generation state for one term walk."""

    __slots__ = (
        "_annotation_targets",
        "_annotations",
        "_assignment",
        "_assumptions",
        "_constraints",
        "_injected",
        "_rules",
        "_signature",
        "_store",
        "_supply",
        "_unifier",
    )

    def __init__(
        self,
        store: TermStore,
        signature: LexicalSignature,
        rules: Mapping[OpName, TypingRule],
        annotations: Mapping[Path, Term],
        injected: Mapping[Path, Term],
        *,
        term_sort: Sort,
    ) -> None:
        self._store = store
        self._signature = signature
        self._rules = rules
        self._annotations = annotations
        self._injected = injected
        self._supply = _FreshSupply(
            store,
            store.signature.classifiers[term_sort],
            _blocked_atoms(store, signature, rules, annotations, injected),
        )
        self._unifier = Unifier(store)
        self._assignment: dict[Path, Term] = {}
        self._constraints: list[Constraint] = []
        self._annotation_targets: dict[Path, Term] = {}
        self._assumptions: dict[int, Term] = {}

    def run(
        self,
        term: Term,
    ) -> _GenerationResult:
        """Walk a closed term and freeze its generated values."""
        if not term.is_closed:
            raise NotLocallyClosedError(max_index=term.maxidx, depth=0)

        self._visit(term, (), ())
        annotation_constraints = tuple(
            Constraint(
                "annotation",
                self._annotation_targets[path],
                self._annotations[path],
                path,
            )
            for path in sorted(self._annotations)
        )
        injected_constraints = tuple(
            Constraint(
                "unification",
                self._assignment[path],
                self._injected[path],
                path,
            )
            for path in sorted(self._injected)
        )

        return _GenerationResult(
            MappingProxyType(self._assignment),
            (*self._constraints, *injected_constraints, *annotation_constraints),
            self._supply.issued,
        )

    def _annotate(self, path: Path, inferred: Term) -> None:
        """Record an inferred annotation target when the path has one."""
        annotation = self._annotations.get(path)

        if annotation is not None:
            self._annotation_targets[path] = inferred

    def _free(self, node: Term) -> Term:
        """Get the assumption shared by one free atom."""
        atom = _variable_payload(self._store, node)
        inferred = self._assumptions.get(atom)

        if inferred is None:
            inferred = self._supply.take()
            self._assumptions[atom] = inferred

        return inferred

    def _symbol(self, node: Term, path: Path) -> Term:
        """Instantiate one declared symbol occurrence."""
        payload = node.payload

        if payload is None:
            msg = f"malformed logical-form symbol: {node!r}"
            raise ValueError(msg)

        symbol = ensure_non_negative("symbol payload", payload)
        scheme = self._signature.schemes.get(symbol)

        if scheme is None:
            raise UnknownSymbolError(symbol, node, path)

        return _instantiate_scheme(
            self._unifier,
            self._supply,
            scheme,
        )

    def _rule(
        self,
        node: Term,
        path: Path,
        environment: tuple[Term, ...],
    ) -> Term:
        """Instantiate one rule and visit its premises."""
        rule = self._rules[node.op.name]

        if rule.op != node.op.name:
            msg = f"rule table maps {node.op.name!r} to rule {rule.op!r}"
            raise ValueError(msg)

        annotation_positions = frozenset(
            site.annotation for site in node.op.binding if site.annotation is not None
        )
        logical_positions = tuple(
            position
            for position in range(len(node.args))
            if position not in annotation_positions
        )

        if len(rule.premises) != len(logical_positions):
            msg = (
                f"typing rule {rule.op!r} has {len(rule.premises)} premises "
                f"for {len(logical_positions)} logical arguments"
            )
            raise ValueError(msg)

        bound, premises, inferred = _instantiate_rule(
            self._store,
            self._supply,
            rule,
        )

        for position, binding, expected in zip(
            logical_positions, bound, premises, strict=True
        ):
            child = node.args[position]
            count = len(node.op.scope(position))
            child_environment = self._child_environment(
                rule.op,
                position,
                count,
                binding,
                environment,
            )
            actual = self._visit(child, (*path, position), child_environment)
            self._constraints.append(Constraint("unification", actual, expected, path))

        for site, binding in zip(
            node.op.binding,
            (b for b in bound if b is not None),
            strict=True,
        ):
            if site.annotation is None:
                continue

            annotation = node.args[site.annotation]
            self._constraints.append(
                Constraint("annotation", binding, annotation, path)
            )
            self._assignment[(*path, site.annotation)] = binding

        annotated = self._annotations.get(path)

        if annotated is not None:
            introductions = tuple(binding for binding in bound if binding is not None)

            if len(introductions) != 1:
                msg = f"annotation at {path!r} does not address one binder introduction"
                raise ValueError(msg)

            self._annotation_targets[path] = introductions[0]

        return inferred

    @staticmethod
    def _child_environment(
        op: OpName,
        position: int,
        count: int,
        binding: Term | None,
        environment: tuple[Term, ...],
    ) -> tuple[Term, ...]:
        """Extend the environment for one binding premise."""
        if count and binding is None:
            msg = f"typing rule {op!r} gives no type to argument {position}'s bindings"
            raise ValueError(msg)

        if not count and binding is not None:
            msg = f"typing rule {op!r} types bindings at unbound argument {position}"
            raise ValueError(msg)

        return environment if binding is None else (binding,) * count + environment

    def _visit(
        self,
        node: Term,
        path: Path,
        environment: tuple[Term, ...],
    ) -> Term:
        """Visit one node."""
        if self._store.signature.is_bound_variable(node.op):
            inferred = environment[_variable_payload(self._store, node)]
        elif self._store.signature.is_free_variable(node.op):
            inferred = self._free(node)
        elif self._store.signature.is_constant(node.op):
            inferred = self._symbol(node, path)
        else:
            inferred = self._rule(node, path, environment)

        if not node.args:
            self._annotate(path, inferred)

        self._assignment[path] = inferred

        return inferred


def generate(
    term: Term,
    store: TermStore,
    signature: LexicalSignature,
    rules: Mapping[OpName, TypingRule],
    /,
    *,
    annotations: Mapping[Path, Term] | None = None,
    constraints: Mapping[Path, Term] | None = None,
) -> tuple[Mapping[Path, Term], tuple[Constraint, ...]]:
    """Generate provisional node types and located constraints.

    Parameters
    ----------
    term : Term
        The logical form whose nodes receive provisional types.
    store : TermStore
        The language store holding the form and its types.
    signature : LexicalSignature
        The type schemes of the form's constants.
    rules : Mapping[OpName, TypingRule]
        Structural typing rules keyed by operator name.
    annotations : Mapping[Path, Term] | None
        Surface binder or leaf annotations keyed by node path.
    constraints : Mapping[Path, Term] | None
        Additional expected node types keyed by path.
    """
    generated = _generate(
        term,
        store,
        signature,
        rules,
        annotations=annotations,
        constraints=constraints,
    )

    return generated.assignment, generated.constraints


def _generate(
    term: Term,
    store: TermStore,
    signature: LexicalSignature,
    rules: Mapping[OpName, TypingRule],
    /,
    *,
    annotations: Mapping[Path, Term] | None = None,
    constraints: Mapping[Path, Term] | None = None,
) -> _GenerationResult:
    """Generate constraints with private fresh-atom provenance."""
    held_annotations = _annotations(term, store, annotations)
    held_constraints = _injected(term, store, constraints)
    term_sort = term.op.sort

    if term_sort not in store.signature.classifiers:
        msg = f"term sort {term_sort!r} has no declared classifier sort"
        raise ValueError(msg)

    return _Generator(
        store,
        signature,
        rules,
        held_annotations,
        held_constraints,
        term_sort=term_sort,
    ).run(term)

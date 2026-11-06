"""Runtime assembly for rules over a declared constructor inventory.

A signature's category constructors may declare functor roles. The binary
functor view those roles describe is what composition, substitution, type
raising and the coercions are built over, so the same rule declarations run
over CCG's directed slash pair, an undirected arrow, or any other declared
pair of constructors.
"""

from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Protocol, cast, final

from lofi._frozen import FrozenValue, Unpicklable
from lofi.grammars.ccg._values import marked_refinement
from lofi.judge import (
    HOLE,
    NormalFormRole,
    PatternRule,
    ProofSchema,
    RuleIndex,
    RuleProbe,
    RuleRestriction,
    RuleSet,
)
from lofi.terms import (
    CategoryResolution,
    MVar,
    Node,
    RepresentativeUnavailableError,
    RootMatching,
    TermMatching,
    TermRuleKernel,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from lofi._types import JsonValue
    from lofi.grammars.ccg._functors import FunctorView
    from lofi.judge import (
        Conjunctions,
        Direction,
        NormalForm,
        RestrictionMode,
        Rule,
        SchematicSequent,
        Slot,
    )
    from lofi.terms import (
        Matching,
        OpName,
        Pattern,
        Signature,
        Sort,
        Term,
        TermStore,
        TermSyntax,
    )

    class _ClausalAdjunctDeclaration(Protocol):
        """Fields needed to build clausal adjunct coercion."""

        @property
        def name(self) -> str:
            """Return the rule name."""
            ...

        @property
        def source_view(self) -> str:
            """Return the source functor view."""
            ...

        @property
        def target_view(self) -> str:
            """Return the target functor view."""
            ...

        @property
        def source_core(self) -> str:
            """Return the required source-result core."""
            ...

        @property
        def argument(self) -> str:
            """Return the required source argument."""
            ...

        @property
        def head_policy(self) -> Literal["pass-through"]:
            """Return the lexical-head policy."""
            ...

        @property
        def restriction(self) -> RuleRestriction | None:
            """Return the native-compiled restriction."""
            ...

    class _CoordinationRuntimeDeclaration(Protocol):
        """Fields consumed by one coordination phase."""

        @property
        def name(self) -> str:
            """Return the rule name."""
            ...

        @property
        def head_policy(self) -> Literal["non-absorbing", "left-conjunct"]:
            """Return the coordination head policy."""
            ...

        @property
        def phase(self) -> Literal["mark", "merge"]:
            """Return the coordination phase."""
            ...

        @property
        def core_relation(self) -> Literal["same", "different"]:
            """Return the declared core relation."""
            ...

        @property
        def conjunction_feature(self) -> str:
            """Return the conjunction feature."""
            ...

        @property
        def targets(self) -> tuple[str, ...] | None:
            """Return finite targets, if declared."""
            ...

    class _RuleSetDeclaration(Protocol):
        """Fields needed to build a rule set."""

        @property
        def name(self) -> str:
            """Return the rule-set name."""
            ...

        @property
        def restriction_mode(self) -> RestrictionMode:
            """Return the restriction mode."""
            ...


__all__ = ["CCGAssembly"]

_BINARY_ARITY = 2


@final
class _CategoryHeads:
    """Feature-erased root-operator codes for CCG categories."""

    __slots__ = ("_codes", "_named", "_signature")

    def __init__(self, signature: Signature) -> None:
        named = {
            operator.name: signature.code(operator.erases_to or operator.name)
            for operator in signature.operators
        }
        object.__setattr__(self, "_signature", signature)
        object.__setattr__(self, "_named", MappingProxyType(named))
        object.__setattr__(self, "_codes", frozenset(named.values()))

    @property
    def codes(self) -> frozenset[int]:
        """Return every feature-erased root code."""
        return self._codes

    def term(self, term: Term, /) -> int:
        """Return a category's feature-erased root code."""
        return self._signature.code(term.core.op)

    def named(self, name: str, /) -> int | None:
        """Return a named operator's feature-erased root code, if declared."""
        return self._named.get(name)


def _pattern_probe(
    patterns: Sequence[Pattern],
    matching: Matching[Term],
    signature: Signature,
    /,
) -> RuleProbe | None:
    """Compile fixed CCG premise roots to a conservative head probe.

    A fixed premise root constrains its slot to the root operators the
    premise congruence identifies with it. A congruence that bounds no root
    leaves the slot unconstrained.
    """
    constraints = tuple(
        matching.identified_roots(premise.op, signature)
        if isinstance(premise, Node)
        and isinstance(premise.op, str)
        and isinstance(matching, RootMatching)
        else None
        for premise in patterns
    )

    return (
        None
        if all(constraint is None for constraint in constraints)
        else RuleProbe(len(patterns), (constraints,))
    )


def _fixed_pattern(term: Term) -> Node:
    """Convert a category to a fixed pattern."""
    return Node(
        term.op.name,
        tuple(_fixed_pattern(child) for child in term.args),
        term.payload,
    )


def _coordination_pairs(
    pairs: tuple[tuple[Term, Term], ...],
    feature: str,
    store: TermStore,
    /,
) -> tuple[tuple[Term, Term], ...]:
    """Select the conjunction pairs carrying one declared feature."""
    suffix = f"[{feature}]"
    held = tuple(
        pair
        for pair in pairs
        if (
            store.payload_name(pair[1])
            if store.signature.is_constant(pair[1].op)
            else pair[1].op.name
        ).endswith(suffix)
    )
    if not held:
        msg = f"no CCG conjunction pair has feature {feature!r}"
        raise ValueError(msg)
    return held


def _equivalent_member[T](
    value: T,
    members: Sequence[T],
    matching: Matching[T],
) -> bool:
    """Return whether a value matches a declared member."""
    return any(matching.equivalent(value, member) for member in members)


def _unique[T](values: Iterator[T], matching: Matching[T]) -> Iterator[T]:
    """Yield one representative from each matching class."""
    emitted: list[T] = []

    for value in values:
        if _equivalent_member(value, emitted, matching):
            continue

        emitted.append(value)
        yield value


@final
class _Absorption(Unpicklable, FrozenValue):
    """Finite binary category absorption."""

    _noun = "CCG assembly value"
    _because = "retains live term identities"

    __slots__ = (
        "_absorbing",
        "_lowering",
        "_pattern",
        "_probe",
        "_side",
        "_sources",
    )

    def __init__(
        self,
        pattern: PatternRule,
        sources: tuple[str, ...],
        absorbing: tuple[Term, ...],
        side: Literal["left", "right"],
        lowering: TermRuleKernel,
    ) -> None:
        if pattern.arity != _BINARY_ARITY:
            msg = "a CCG absorption rule requires exactly two premises"
            raise ValueError(msg)

        object.__setattr__(self, "_pattern", pattern)
        object.__setattr__(self, "_sources", sources)
        object.__setattr__(self, "_absorbing", absorbing)
        object.__setattr__(self, "_side", side)
        object.__setattr__(self, "_lowering", lowering)
        heads = frozenset(term.core.op.name for term in absorbing)
        constraints = (heads, None) if side == "left" else (None, heads)
        object.__setattr__(self, "_probe", RuleProbe(_BINARY_ARITY, (constraints,)))

    @property
    def name(self) -> str:
        """Return the rule name."""
        return self._pattern.name

    @property
    def arity(self) -> int:
        """Return the premise count."""
        return _BINARY_ARITY

    @property
    def directions(self) -> frozenset[Direction]:
        """Return every computation direction."""
        return frozenset({"forward", "check", "residual"})

    @property
    def spine_order(self) -> int:
        """Return the spine order of every instance."""
        return 0

    @property
    def residual_slots(self) -> frozenset[int]:
        """Return both recoverable premise slots."""
        return frozenset(range(_BINARY_ARITY))

    @property
    def forward_is_complete(self) -> bool:
        """Return whether forward enumeration is complete."""
        return True

    @property
    def matching(self) -> TermMatching:
        """Return the shared category congruence."""
        return self._pattern.matching

    @property
    def restriction(self) -> RuleRestriction | None:
        """Return the declared restriction, if present."""
        return self._pattern.restriction

    @property
    def proof_schemata(self) -> tuple[SchematicSequent, ...]:
        """Return the guarded pattern schema."""
        return self._pattern.proof_schemata

    @property
    def reachability_schemata(self) -> tuple[SchematicSequent, ...]:
        """Return the consumed premises and introduced conclusions."""
        absorbing_at, _ = self._positions()
        schemata: list[SchematicSequent] = []

        for premises, conclusion in self._pattern.reachability_schemata:
            for category in self._absorbing:
                guarded = list(premises)
                guarded[absorbing_at] = _fixed_pattern(self.matching.key(category))
                schemata.append((tuple(guarded), conclusion))

        return tuple(schemata)

    @property
    def probe(self) -> RuleProbe | None:
        """Return the absorbing premise's conservative head constraint."""
        return self._probe

    @property
    def normal_form_role(self) -> NormalFormRole:
        """Return the absorption rule's intrinsic structural role."""
        return self._pattern.normal_form_role

    @property
    def lowering(self) -> TermRuleKernel:
        """Return the native implementation of this relation."""
        return self._lowering

    def _positions(self) -> tuple[int, int]:
        """Return the absorbing and retained premise positions."""
        return (0, 1) if self._side == "left" else (1, 0)

    def _guard(self, inputs: tuple[Term, ...]) -> bool:
        """Check the finite exactly-one-category condition."""
        if len(inputs) != self.arity:
            return False

        absorbing_at, retained_at = self._positions()

        return _equivalent_member(
            inputs[absorbing_at], self._absorbing, self.matching
        ) and not _equivalent_member(
            inputs[retained_at], self._absorbing, self.matching
        )

    def apply(self, inputs: tuple[Term, ...], /) -> Iterator[Term]:
        """Enumerate the retained category when the patterns agree."""
        yield from self._lowering.apply(inputs)

    def admits(
        self,
        inputs: tuple[Term, ...],
        output: Term,
        /,
    ) -> bool:
        """Decide the guarded absorption relation."""
        if not self._guard(inputs):
            return False

        _, retained_at = self._positions()

        return self.matching.equivalent(
            output, inputs[retained_at]
        ) and self._pattern.admits(inputs, output)

    def residuate(
        self,
        output: Term,
        known: tuple[Slot[Term], ...],
        /,
    ) -> Iterator[Term]:
        """Enumerate the missing absorbing or retained premise."""
        if len(known) != self.arity or sum(value is HOLE for value in known) != 1:
            return

        missing = known.index(HOLE)
        absorbing_at, retained_at = self._positions()

        if missing == absorbing_at:
            candidates = iter(self._absorbing)
        elif missing == retained_at:
            candidates = iter((output,))
        else:
            return

        def admitted() -> Iterator[Term]:
            for candidate in candidates:
                inputs = tuple(candidate if value is HOLE else value for value in known)

                if self.admits(cast("tuple[Term, ...]", inputs), output):
                    yield candidate

        yield from _unique(admitted(), self.matching)

    @property
    def relation(self) -> Mapping[str, JsonValue]:
        """Return the parameters that fix the relation, without the name."""
        return {
            "rule": "ccg-absorption",
            "pattern": dict(self._pattern.relation),
            "absorbing": list(self._sources),
            "side": self._side,
        }

    def to_spec(self) -> Mapping[str, JsonValue]:
        """Return the rule declaration."""
        return {
            "rule": "ccg-absorption",
            "name": self.name,
            "pattern": dict(self._pattern.to_spec()),
            "absorbing": list(self._sources),
            "side": self._side,
            "other_must_be_nonabsorbing": True,
        }


@final
class _ClausalAdjunct(Unpicklable, FrozenValue):
    """Feature-bearing clausal adjunct coercion."""

    _noun = "CCG assembly value"
    _because = "retains live term identities"

    __slots__ = (
        "_argument",
        "_argument_source",
        "_lowering",
        "_matching",
        "_name",
        "_restriction",
        "_source_core",
        "_source_core_source",
        "_source_slash",
        "_source_slash_source",
        "_store",
        "_target_slash",
        "_target_slash_source",
    )

    def __init__(
        self,
        store: TermStore,
        matching: TermMatching,
        declaration: _ClausalAdjunctDeclaration,
        resolved: tuple[str, str, Term, Term],
        lowering: TermRuleKernel,
    ) -> None:
        source_slash, target_slash, source_core, argument = resolved

        if source_core.op.arity != 0 or source_core is not source_core.core:
            msg = "a clausal-adjunct source core must be an unfeatured atom"
            raise ValueError(msg)

        if argument.op.arity != 0:
            msg = "a clausal-adjunct argument must be an atom"
            raise ValueError(msg)

        if declaration.head_policy != "pass-through":
            msg = "a clausal-adjunct rule requires pass-through head selection"
            raise ValueError(msg)

        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_matching", matching)
        object.__setattr__(self, "_lowering", lowering)
        object.__setattr__(self, "_name", declaration.name)
        object.__setattr__(self, "_restriction", declaration.restriction)
        object.__setattr__(self, "_source_slash", source_slash)
        object.__setattr__(self, "_target_slash", target_slash)
        object.__setattr__(self, "_source_core", source_core)
        object.__setattr__(self, "_argument", argument)
        object.__setattr__(self, "_source_slash_source", declaration.source_view)
        object.__setattr__(self, "_target_slash_source", declaration.target_view)
        object.__setattr__(self, "_source_core_source", declaration.source_core)
        object.__setattr__(self, "_argument_source", declaration.argument)

    @property
    def name(self) -> str:
        """Return the rule name."""
        return self._name

    @property
    def arity(self) -> int:
        """Return the premise count."""
        return 1

    @property
    def directions(self) -> frozenset[Direction]:
        """Return the checking-only direction."""
        return frozenset({"check"})

    @property
    def spine_order(self) -> int:
        """Return the spine order of every instance."""
        return 0

    @property
    def residual_slots(self) -> frozenset[int]:
        """Return no recoverable premise slots."""
        return frozenset()

    @property
    def forward_is_complete(self) -> bool:
        """Return whether forward enumeration is complete."""
        return False

    @property
    def matching(self) -> TermMatching:
        """Return the shared category congruence."""
        return self._matching

    @property
    def restriction(self) -> RuleRestriction | None:
        """Return the declared restriction, if present."""
        return self._restriction

    @property
    def proof_schemata(self) -> tuple[SchematicSequent, ...]:
        """Return the feature-erased clausal adjunct schema."""
        variable = MVar("X", self._source_core.op.sort)
        premise = Node(
            self._source_slash,
            (
                _fixed_pattern(self._source_core),
                _fixed_pattern(self._argument),
            ),
        )
        conclusion = Node(self._target_slash, (variable, variable))

        return (((premise,), conclusion),)

    @property
    def reachability_schemata(self) -> tuple[SchematicSequent, ...]:
        """Return the consumed premises and introduced conclusions."""
        return self.proof_schemata

    @property
    def probe(self) -> None:
        """Return no premise-head index probe."""
        return None

    @property
    def normal_form_role(self) -> NormalFormRole:
        """Return the clausal adjunct rule's intrinsic structural role."""
        return NormalFormRole()

    @property
    def lowering(self) -> TermRuleKernel:
        """Return the native implementation of this relation."""
        return self._lowering

    def apply(self, inputs: tuple[Term, ...], /) -> Iterator[Term]:
        """Return no conclusions for unrestricted forward enumeration."""
        yield from self._lowering.apply(inputs)

    def _valid_source(self, source: Term) -> bool:
        """Check the declared feature-bearing source shape."""
        if not self._store.contains(source) or source.op.name != self._source_slash:
            return False

        result, argument = source.args

        return (
            result is not result.core
            and result.core is self._source_core
            and argument is self._argument
        )

    def _valid_output(self, output: Term) -> bool:
        """Check the repeated target-functor shape."""
        if not self._store.contains(output) or output.op.name != self._target_slash:
            return False

        left, right = output.args
        return self._matching.equivalent(left, right)

    def admits(
        self,
        inputs: tuple[Term, ...],
        output: Term,
        /,
    ) -> bool:
        """Decide the clausal adjunct coercion relation."""
        return (
            len(inputs) == self.arity
            and self._valid_source(inputs[0])
            and self._valid_output(output)
        )

    def residuate(
        self,
        output: Term,
        known: tuple[Slot[Term], ...],
        /,
    ) -> Iterator[Term]:
        """Return no residual candidates."""
        del output, known
        return iter(())

    @property
    def relation(self) -> Mapping[str, JsonValue]:
        """Return the parameters that fix the relation, without the name."""
        return {
            "rule": "ccg-clausal-adjunct",
            "source_slash": self._source_slash,
            "target_slash": self._target_slash,
            "source_core": self._source_core_source,
            "argument": self._argument_source,
            "matching": self._matching.name,
            "restriction": (
                None if self._restriction is None else self._restriction.to_spec()
            ),
        }

    def to_spec(self) -> Mapping[str, JsonValue]:
        """Return the runtime rule declaration."""
        return {
            "rule": "ccg-clausal-adjunct",
            "name": self._name,
            "source_view": self._source_slash_source,
            "target_view": self._target_slash_source,
            "source_core": self._source_core_source,
            "argument": self._argument_source,
            "head_policy": "pass-through",
            "matching": self._matching.name,
            "restriction": (
                None if self._restriction is None else self._restriction.to_spec()
            ),
        }


@final
class _CoordinationPhase(Unpicklable, FrozenValue):
    """One phase of finite feature-sensitive coordination."""

    _noun = "CCG assembly value"
    _because = "retains live term identities"

    __slots__ = (
        "_core_relation",
        "_feature",
        "_lowering",
        "_marking",
        "_marks_by_key",
        "_pairs",
        "_pattern",
        "_phase",
        "_probe",
        "_store",
        "_target_keys",
        "_target_sources",
        "_targets",
        "_unmarked_by_key",
    )

    def __init__(
        self,
        pattern: PatternRule,
        pairs: tuple[tuple[Term, Term], ...],
        declaration: _CoordinationRuntimeDeclaration,
        targets: tuple[Term, ...] | None,
        lowering: TermRuleKernel,
        *,
        store: TermStore,
        marking: str | None,
    ) -> None:
        if pattern.arity != _BINARY_ARITY:
            msg = "a CCG coordination phase requires exactly two premises"
            raise ValueError(msg)

        if (
            marking is None
            and targets is not None
            and any(
                not _equivalent_member(
                    target, tuple(pair[0] for pair in pairs), pattern.matching
                )
                for target in targets
            )
        ):
            msg = "a coordination target has no declared conjunction pair"
            raise ValueError(msg)

        object.__setattr__(self, "_pattern", pattern)
        object.__setattr__(self, "_pairs", pairs)
        object.__setattr__(self, "_phase", declaration.phase)
        object.__setattr__(self, "_core_relation", declaration.core_relation)
        object.__setattr__(self, "_feature", declaration.conjunction_feature)
        object.__setattr__(self, "_target_sources", declaration.targets)
        object.__setattr__(self, "_targets", targets)
        object.__setattr__(self, "_lowering", lowering)
        object.__setattr__(self, "_marking", marking)
        object.__setattr__(self, "_store", store)

        marks: dict[Term, list[Term]] = {}
        unmarked: dict[Term, list[Term]] = {}

        for plain, marked in pairs:
            marks.setdefault(pattern.matching.key(plain), []).append(marked)
            unmarked.setdefault(pattern.matching.key(marked), []).append(plain)

        object.__setattr__(
            self,
            "_marks_by_key",
            MappingProxyType({key: tuple(values) for key, values in marks.items()}),
        )
        object.__setattr__(
            self,
            "_unmarked_by_key",
            MappingProxyType({key: tuple(values) for key, values in unmarked.items()}),
        )
        object.__setattr__(
            self,
            "_target_keys",
            None
            if targets is None
            else frozenset(pattern.matching.key(target) for target in targets),
        )

        probe = pattern.probe

        if declaration.phase == "merge":
            plain_heads = (
                frozenset(plain.core.op.name for plain, _ in pairs)
                if marking is None
                else None
            )
            marked_heads = frozenset(
                {
                    *(marked.core.op.name for _, marked in pairs),
                    *(() if marking is None else (marking,)),
                }
            )
            probe = RuleProbe(
                _BINARY_ARITY,
                ((plain_heads, marked_heads),),
            )

        object.__setattr__(self, "_probe", probe)

    @property
    def name(self) -> str:
        """Return the rule name."""
        return self._pattern.name

    @property
    def arity(self) -> int:
        """Return the premise count."""
        return _BINARY_ARITY

    @property
    def directions(self) -> frozenset[Direction]:
        """Return the supported computation directions."""
        if self._phase == "merge":
            return frozenset({"forward", "check", "residual"})

        if self._core_relation == "different" and self._targets is None:
            held: set[Direction] = {"check"}

            if self._pattern.residual_slots:
                held.add("residual")

            return frozenset(held)

        held: set[Direction] = {"check"}

        if self._targets is not None or "forward" in self._pattern.directions:
            held.add("forward")

        if self._pattern.residual_slots:
            held.add("residual")

        return frozenset(held)

    @property
    def spine_order(self) -> int:
        """Return the spine order of every instance."""
        return 0

    @property
    def residual_slots(self) -> frozenset[int]:
        """Return the recoverable premise slots."""
        if self._phase == "merge":
            return frozenset(range(_BINARY_ARITY))

        if "residual" not in self.directions:
            return frozenset()

        return self._pattern.residual_slots

    @property
    def forward_is_complete(self) -> bool:
        """Return whether forward enumeration is complete."""
        return "forward" in self.directions

    @property
    def matching(self) -> TermMatching:
        """Return the shared category congruence."""
        return self._pattern.matching

    @property
    def restriction(self) -> RuleRestriction | None:
        """Return the declared restriction, if present."""
        return self._pattern.restriction

    @property
    def proof_schemata(self) -> tuple[SchematicSequent, ...]:
        """Return the feature-erased coordination schema."""
        if self._phase == "mark" and self._targets is not None:
            return tuple(
                (premises, _fixed_pattern(marked))
                for premises, _ in self._pattern.proof_schemata
                for target in self._targets
                for marked in self._marks(target)
            )

        return self._pattern.proof_schemata

    @property
    def reachability_schemata(self) -> tuple[SchematicSequent, ...]:
        """Return the consumed premises and introduced conclusions."""
        if self._phase == "merge":
            declared = tuple(
                (
                    (_fixed_pattern(plain), _fixed_pattern(marked)),
                    _fixed_pattern(plain),
                )
                for plain, marked in self._pairs
                if (
                    self._target_keys is None
                    or self.matching.key(plain) in self._target_keys
                )
                and self._pattern.admits((plain, plain), plain)
            )
            if self._marking is None:
                return declared
            base = MVar("X")
            return (*declared, ((base, Node(self._marking, (base,))), base))

        if self._targets is None:
            return tuple(
                (premises, _fixed_pattern(marked))
                for premises, _ in self._pattern.proof_schemata
                for _, marked in self._pairs
            ) + (
                ()
                if self._marking is None
                else tuple(
                    (premises, Node(self._marking, (conclusion,)))
                    for premises, conclusion in self._pattern.proof_schemata
                )
            )

        return self.proof_schemata

    @property
    def probe(self) -> RuleProbe | None:
        """Return conservative premise-head constraints."""
        return self._probe

    @property
    def normal_form_role(self) -> NormalFormRole:
        """Return the coordination rule's intrinsic structural role."""
        return self._pattern.normal_form_role

    @property
    def lowering(self) -> TermRuleKernel:
        """Return the native implementation of this relation."""
        return self._lowering

    def _marks(self, unmarked: Term) -> Iterator[Term]:
        """Enumerate declared marked counterparts, constructed ones last."""
        declared = self._marks_by_key.get(self.matching.key(unmarked), ())
        yield from declared
        if self._marking is not None:
            constructed = self._store.op(self._marking, unmarked)
            if constructed not in declared:
                yield constructed

    def _unmarked(self, marked: Term) -> Iterator[Term]:
        """Enumerate declared unmarked counterparts, constructed ones last."""
        declared = self._unmarked_by_key.get(self.matching.key(marked), ())
        yield from declared
        if (
            self._marking is not None
            and marked.op.name == self._marking
            and marked.args[0] not in declared
        ):
            yield marked.args[0]

    def _core_matches(self, source: Term, target: Term) -> bool:
        """Check the declared relation between category cores."""
        equivalent = self.matching.equivalent(source, target)

        return equivalent if self._core_relation == "same" else not equivalent

    def _admitted_candidates(
        self,
        output: Term,
        known: tuple[Slot[Term], ...],
        candidates: Iterator[Term],
    ) -> Iterator[Term]:
        """Yield candidates that satisfy the coordination relation."""
        for candidate in candidates:
            inputs = tuple(candidate if value is HOLE else value for value in known)

            if self.admits(cast("tuple[Term, ...]", inputs), output):
                yield candidate

    def _merge_base(
        self,
        inputs: tuple[Term, ...],
    ) -> tuple[Term, tuple[Term, ...]] | None:
        """Return the common unmarked base and normalized premises."""
        if len(inputs) != self.arity:
            return None

        left, right = inputs

        for base in self._unmarked(right):
            if not self.matching.equivalent(left, base):
                continue

            normalized = (left, base)

            if (
                self._target_keys is not None
                and self.matching.key(base) not in self._target_keys
            ):
                continue

            if self._pattern.admits(normalized, left):
                return left, normalized

        return None

    def apply(self, inputs: tuple[Term, ...], /) -> Iterator[Term]:
        """Enumerate conclusions for the declared coordination phase."""
        yield from self._lowering.apply(inputs)

    def admits(
        self,
        inputs: tuple[Term, ...],
        output: Term,
        /,
    ) -> bool:
        """Decide the declared coordination phase."""
        if self._phase == "merge":
            merged = self._merge_base(inputs)

            return merged is not None and self.matching.equivalent(output, merged[0])

        if len(inputs) != self.arity:
            return False

        return any(
            self._pattern.admits(inputs, base)
            and self._core_matches(inputs[1], base)
            and (
                self._target_keys is None
                or self.matching.key(base) in self._target_keys
            )
            for base in self._unmarked(output)
        )

    def residuate(
        self,
        output: Term,
        known: tuple[Slot[Term], ...],
        /,
    ) -> Iterator[Term]:
        """Enumerate a missing premise when the phase determines it."""
        if len(known) != self.arity or sum(value is HOLE for value in known) != 1:
            return

        if self._phase == "mark":
            for base in self._unmarked(output):
                candidates = self._pattern.residuate(base, known)
                admitted = self._admitted_candidates(output, known, candidates)
                yield from _unique(admitted, self.matching)

            return

        missing = known.index(HOLE)

        candidates = iter((output,)) if missing == 0 else self._marks(output)

        admitted = self._admitted_candidates(output, known, candidates)
        yield from _unique(admitted, self.matching)

    @property
    def relation(self) -> Mapping[str, JsonValue]:
        """Return the parameters that fix the relation, without the name."""
        return {
            "rule": "ccg-coordination-phase",
            "pattern": dict(self._pattern.relation),
            "phase": self._phase,
            "core_relation": self._core_relation,
            "conjunction_feature": self._feature,
            "targets": (
                None if self._target_sources is None else list(self._target_sources)
            ),
        }

    def to_spec(self) -> Mapping[str, JsonValue]:
        """Return the rule declaration."""
        return {
            "rule": "ccg-coordination-phase",
            "name": self.name,
            "pattern": dict(self._pattern.to_spec()),
            "phase": self._phase,
            "core_relation": self._core_relation,
            "conjunction_feature": self._feature,
            "targets": (
                None if self._target_sources is None else list(self._target_sources)
            ),
        }


@final
class CCGAssembly(Unpicklable, FrozenValue):
    """Runtime construction context for rules over a declared signature.

    Functor-family rules are built over the functor view; pattern,
    absorption and coordination rules match categories directly.

    Parameters
    ----------
    store : TermStore
        The category store.
    syntax : TermSyntax
        The category syntax over the store's signature.
    resolve_category : CategoryResolution
        Resolve category text once in the crossing's store.
    functors : Mapping[str, tuple[FunctorView, OpName]]
        Named functor views paired with the constructor each view selects.
    matching : TermMatching
        The declared category congruence.
    conjunctions : Conjunctions[Term] | None
        The conjunction inventory coordination runs over, or nothing for a
        grammar without one, which builds no coordination rule. With an
        inventory, categories are compared under the declared congruence
        refined to keep the inventory's conjunction marks apart wherever it
        would equate a marked category with an unmarked one.
    normal_form : NormalForm[Term]
        The derivation-step filter.
    category_sort : Sort
        The learner language's declared category sort.

    Raises
    ------
    ValueError
        If the syntax reads another signature, a functor view builds another
        sort than the category sort, a conjunction category is foreign or
        wrongly sorted, or a view selects a constructor it does not contain.
    """

    _noun = "CCG assembly value"
    _because = "retains live term identities"

    __slots__ = (
        "_categories",
        "_category_sort",
        "_conjunctions",
        "_coordination",
        "_functors",
        "_matching",
        "_normal_form",
        "_rule_matchings",
        "_store",
        "_syntax",
    )

    def __init__(
        self,
        store: TermStore,
        syntax: TermSyntax,
        resolve_category: CategoryResolution,
        functors: Mapping[str, tuple[FunctorView, OpName]],
        matching: TermMatching,
        conjunctions: Conjunctions[Term] | None,
        normal_form: NormalForm[Term],
        *,
        category_sort: Sort,
        rule_matchings: Sequence[tuple[str, TermMatching]] = (),
        categories: Sequence[str] = (),
    ) -> None:
        if syntax.signature is not store.signature:
            msg = "a CCG assembly syntax must use the store signature"
            raise ValueError(msg)

        if category_sort not in store.signature.sorts:
            msg = f"category sort {category_sort!r} is not declared by the signature"
            raise ValueError(msg)

        for name, (view, constructor) in functors.items():
            if view.category_sort != category_sort:
                msg = (
                    f"functor view {name!r} builds sort {view.category_sort!r}, "
                    f"not the category sort {category_sort!r}"
                )
                raise ValueError(msg)
            if constructor not in view.slashes:
                msg = (
                    f"functor view {name!r} selects {constructor!r}, which it "
                    "does not contain"
                )
                raise ValueError(msg)

        markers = () if conjunctions is None else conjunctions.markers
        pairs = () if conjunctions is None else conjunctions.pairs

        for marker in markers:
            self._check_category(
                store, marker, "a CCG conjunction marker", category_sort
            )

        for unmarked, marked in pairs:
            self._check_category(
                store, unmarked, "an unmarked CCG conjunction category", category_sort
            )
            self._check_category(
                store, marked, "a marked CCG conjunction category", category_sort
            )

        marking = None if conjunctions is None else conjunctions.marking

        if marking is not None:
            operator = store.signature.by_name(marking)
            if operator.sort != category_sort or operator.arg_sorts != (category_sort,):
                msg = (
                    f"the conjunction marking constructor {marking!r} does not map "
                    f"one {category_sort!r} category to another"
                )
                raise ValueError(msg)

        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_syntax", syntax)
        object.__setattr__(self, "_category_sort", category_sort)
        object.__setattr__(self, "_conjunctions", conjunctions)
        resolved = {
            source: resolve_category(source) for source in dict.fromkeys(categories)
        }
        for category in resolved.values():
            self._check_category(
                store,
                category,
                "a declared CCG category",
                category_sort,
            )
        object.__setattr__(
            self,
            "_categories",
            MappingProxyType(resolved),
        )
        coordination: dict[str, list[tuple[Term, Term]]] = {}
        for pair in pairs:
            marked = pair[1]
            name = (
                store.payload_name(marked)
                if store.signature.is_constant(marked.op)
                else marked.op.name
            )
            if "[" in name and name.endswith("]"):
                coordination.setdefault(name.rsplit("[", 1)[1][:-1], []).append(pair)
        object.__setattr__(
            self,
            "_coordination",
            MappingProxyType(
                {feature: tuple(rows) for feature, rows in coordination.items()}
            ),
        )
        object.__setattr__(self, "_functors", MappingProxyType(dict(functors)))
        object.__setattr__(
            self,
            "_matching",
            marked_refinement(matching, conjunctions),
        )
        object.__setattr__(self, "_normal_form", normal_form)
        object.__setattr__(self, "_rule_matchings", tuple(rule_matchings))

    @staticmethod
    def _check_category(
        store: TermStore,
        term: Term,
        field: str,
        category_sort: Sort,
    ) -> None:
        """Reject a foreign or wrongly sorted category."""
        if not store.contains(term):
            msg = f"{field} is outside the assembly store lineage"
            raise ValueError(msg)

        if term.op.sort != category_sort:
            msg = (
                f"{field} has sort {term.op.sort!r}, expected category sort "
                f"{category_sort!r}"
            )
            raise ValueError(msg)

    @property
    def store(self) -> TermStore:
        """Return the category store."""
        return self._store

    @property
    def syntax(self) -> TermSyntax:
        """Return the category syntax."""
        return self._syntax

    @property
    def conjunctions(self) -> Conjunctions[Term] | None:
        """Return the finite conjunction declaration, if one is declared."""
        return self._conjunctions

    @property
    def functors(self) -> Mapping[str, tuple[FunctorView, OpName]]:
        """Return the named functor views."""
        return self._functors

    @property
    def category_sort(self) -> Sort:
        """Return the signature sort categories inhabit."""
        return self._category_sort

    @property
    def default_sort(self) -> Sort:
        """Return the category sort used by an omitted pattern-variable sort."""
        return self._category_sort

    @property
    def sorts(self) -> frozenset[str]:
        """Return every sort declared by the category signature."""
        return frozenset(self._store.signature.sorts)

    @property
    def matching(self) -> TermMatching:
        """Return the shared category congruence."""
        return self._matching

    @property
    def normal_form(self) -> NormalForm[Term]:
        """Return the shared derivation normal form."""
        return self._normal_form

    def compiled_matching(self, recipe: str | None, /) -> TermMatching:
        """Return the congruence selected by one native rule recipe."""
        if recipe is None:
            return self._matching

        relation = next(
            (
                relation
                for prepared, relation in self._rule_matchings
                if prepared == recipe
            ),
            None,
        )
        if relation is None:
            msg = "native rule matching was not prepared by the grammar"
            raise RepresentativeUnavailableError(msg)

        built = marked_refinement(relation, self._conjunctions)
        return self._matching if built.name == self._matching.name else built

    def resolve_functor(self, source: str, /) -> tuple[FunctorView, OpName]:
        """Return the functor view and the constructor a rule names.

        Raises
        ------
        ValueError
            If the name is neither a functor constructor nor its spelling.
        """
        selected = self._functors.get(source)

        if selected is None:
            msg = f"{source!r} is not a functor view of this signature"
            raise ValueError(msg)

        return selected

    def functor_names(self) -> tuple[str, ...]:
        """Return the named functor views in declaration order."""
        return tuple(self._functors)

    def category(self, source: str, /) -> Term:
        """Resolve a category declaration."""
        category = self._categories.get(source)
        if category is None:
            msg = f"category source {source!r} was not declared by the CCG assembly"
            raise ValueError(msg)

        return category

    def coordination_marking(self, feature: str, /) -> str | None:
        """Return the constructor marking any category under one feature, if any.

        The declared marking constructor serves the coordination feature its
        name carries, as a marked pair's category does.
        """
        conjunctions = self._conjunctions
        marking = None if conjunctions is None else conjunctions.marking
        if marking is None or not marking.endswith(f"[{feature}]"):
            return None
        return marking

    def coordination_pairs(self, feature: str, /) -> tuple[tuple[Term, Term], ...]:
        """Return the conjunction pairs carrying one compiled feature."""
        held = self._coordination.get(feature)
        if held is None:
            msg = f"no CCG conjunction pair has feature {feature!r}"
            raise ValueError(msg)
        return held

    def constant(self, name: str, sort: str, /) -> Node:
        """Return the fixed pattern for one vocabulary constant."""
        term = self._store.const(sort, name)

        if term.payload is None:
            msg = f"constant {name!r} of sort {sort!r} has no payload"
            raise ValueError(msg)

        return Node(term.op.name, payload=term.payload)

    def rule_name(self, rule: Rule[Term], /) -> str:
        """Return a runtime rule name."""
        return rule.name

    def proof_schema(self, rule: Rule[Term], /) -> ProofSchema | None:
        """Return a rule's structural proof schema, if available."""
        return rule if isinstance(rule, ProofSchema) else None

    def rule_set(
        self,
        declaration: _RuleSetDeclaration,
        rules: tuple[Rule[Term], ...],
        introduces: tuple[tuple[Term, ...] | None, ...],
        consumes: tuple[tuple[Term, ...] | None, ...],
        /,
    ) -> RuleSet[Term]:
        """Build a rule set under the shared runtime objects.

        Raises
        ------
        ValueError
            If two of the rules denote the same relation over categories.
        """
        return RuleSet(
            declaration.name,
            rules,
            self._matching,
            self._normal_form,
            declaration.restriction_mode,
            introduces=introduces,
            consumes=consumes,
            index=RuleIndex(rules, _CategoryHeads(self._store.signature)),
        )

"""Declared neural-term decoding and supertag-factored A* evaluation."""

from math import fsum, inf
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Literal,
    NamedTuple,
    Protocol,
    cast,
    final,
    override,
    runtime_checkable,
)

import didactic.api as dx

from lofi._checks import ensure_name, ensure_positive
from lofi.deduce import (
    BINARY_WIDENING,
    FREE_ROOTS,
    GOAL_ROOTS,
    LINEAR_WIDENING,
    SUPERTAG_BEAM_PROPOSER,
    SUPERTAG_OUTSIDE,
    ZERO_HEURISTIC,
    AStarHeuristic,
    AStarOptions,
    AStarProposerFactory,
    AStarRootMode,
    WideningStrategy,
    outside_from_edges,
)
from lofi.interpret import ConstituentUtility, LogProbability, MbrUtility, unfold
from lofi.spec._errors import DeclarationError
from lofi.spec._ports import Dependency
from lofi.spec.components.run import (
    EvaluatorSpec,
    ParsePlugin,
    ParseSpec,
    parse_plugin,
)
from lofi.spec.evaluators.conventions import ConventionSettings, ConventionSpec

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi._types import JsonValue
    from lofi.deduce import AStarContext, AStarResult
    from lofi.deduce import Derivation as ForestDerivation
    from lofi.terms import Term

__all__ = [
    "AStarHeuristicSpec",
    "AStarProposerSpec",
    "AStarRootModeSpec",
    "AStarSpec",
    "AStarStrategySpec",
    "BinaryAStarStrategySpec",
    "ConstituentUtilitySpec",
    "DecodeEvaluatorSettings",
    "DecodeEvaluatorSpec",
    "DecodeSpec",
    "DerivationDecision",
    "FreeAStarRootSpec",
    "GoalAStarRootSpec",
    "LinearAStarStrategySpec",
    "MbrDecision",
    "MbrDecodeSpec",
    "MbrUtilitySpec",
    "SupertagBeamProposerSpec",
    "SupertagOutsideHeuristicSpec",
    "ViterbiDecision",
    "ViterbiDecodeSpec",
    "ZeroHeuristicSpec",
    "astar_parse_refusal",
    "decode_decision",
    "derivation_nodes",
]

type DecodeOutput = Literal["lexical-types", "logical-forms"]
"""The kind of complete neural terms a decode evaluator publishes."""


class AStarStrategySpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declares how global supertag thresholds are searched."""


class LinearAStarStrategySpec(AStarStrategySpec, extra="forbid"):
    """Probe global score levels from best to worst."""

    kind: Literal["linear"] = "linear"


class BinaryAStarStrategySpec(AStarStrategySpec, extra="forbid"):
    """Binary-search the ordered global score levels."""

    kind: Literal["binary"] = "binary"


class AStarRootModeSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declares which spanning chart items finish A* search."""


class GoalAStarRootSpec(AStarRootModeSpec, extra="forbid"):
    """Finish only at roots accepted by the grammar goal."""

    kind: Literal["goal"] = "goal"

    @property
    def needs_every_spanning_item(self) -> bool:
        """Return that goal roots survive a root-reachability parse."""
        return False


class FreeAStarRootSpec(AStarRootModeSpec, extra="forbid"):
    """Finish at every spanning chart item."""

    kind: Literal["free"] = "free"

    @property
    def needs_every_spanning_item(self) -> bool:
        """Return that free roots include items no goal root reaches."""
        return True


class AStarHeuristicSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declares an admissible remaining lexical-score bound."""


class SupertagOutsideHeuristicSpec(AStarHeuristicSpec, extra="forbid"):
    """Bound a candidate by the best tags at every other position."""

    kind: Literal["supertag-outside"] = "supertag-outside"


class ZeroHeuristicSpec(AStarHeuristicSpec, extra="forbid"):
    """Use the neutral upper bound at every position."""

    kind: Literal["zero"] = "zero"


class AStarProposerSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declares how check-only grammar rules receive candidate conclusions."""


class SupertagBeamProposerSpec(AStarProposerSpec, extra="forbid"):
    """Propose categories decoded anywhere in the sentence's lexical beams."""

    kind: Literal["supertag-beam"] = "supertag-beam"


def _linear(declaration: LinearAStarStrategySpec, /) -> WideningStrategy:
    """Build linear global-threshold widening."""
    del declaration
    return LINEAR_WIDENING


def _binary(declaration: BinaryAStarStrategySpec, /) -> WideningStrategy:
    """Build binary global-threshold widening."""
    del declaration
    return BINARY_WIDENING


def _goal(declaration: GoalAStarRootSpec, /) -> AStarRootMode:
    """Build grammar-goal root selection."""
    del declaration
    return GOAL_ROOTS


def _free(declaration: FreeAStarRootSpec, /) -> AStarRootMode:
    """Build free spanning-root selection."""
    del declaration
    return FREE_ROOTS


def _supertag_outside(declaration: SupertagOutsideHeuristicSpec, /) -> AStarHeuristic:
    """Build the supertag-outside bound."""
    del declaration
    return SUPERTAG_OUTSIDE


def _zero(declaration: ZeroHeuristicSpec, /) -> AStarHeuristic:
    """Build the zero bound."""
    del declaration
    return ZERO_HEURISTIC


def _supertag_beam(declaration: SupertagBeamProposerSpec, /) -> AStarProposerFactory:
    """Build the checked proposer over decoded supertag beams."""
    del declaration
    return SUPERTAG_BEAM_PROPOSER


LINEAR_ASTAR_STRATEGY_PLUGIN = ParsePlugin(
    spec=LinearAStarStrategySpec,
    build=_linear,
)
BINARY_ASTAR_STRATEGY_PLUGIN = ParsePlugin(
    spec=BinaryAStarStrategySpec,
    build=_binary,
)
GOAL_ASTAR_ROOT_PLUGIN = ParsePlugin(spec=GoalAStarRootSpec, build=_goal)
FREE_ASTAR_ROOT_PLUGIN = ParsePlugin(spec=FreeAStarRootSpec, build=_free)
SUPERTAG_OUTSIDE_HEURISTIC_PLUGIN = ParsePlugin(
    spec=SupertagOutsideHeuristicSpec,
    build=_supertag_outside,
)
ZERO_HEURISTIC_PLUGIN = ParsePlugin(spec=ZeroHeuristicSpec, build=_zero)
SUPERTAG_BEAM_PROPOSER_PLUGIN = ParsePlugin(
    spec=SupertagBeamProposerSpec,
    build=_supertag_beam,
)


class AStarSpec(dx.Model, extra="forbid"):
    """Declare one supertag-factored A* search setup."""

    strategy: AStarStrategySpec = LinearAStarStrategySpec()
    root_mode: AStarRootModeSpec = GoalAStarRootSpec()
    heuristic: AStarHeuristicSpec = SupertagOutsideHeuristicSpec()
    proposer: AStarProposerSpec | None = None

    def build(self) -> AStarOptions:
        """Lower every open search choice into its runtime capability."""
        return AStarOptions(
            cast(
                "WideningStrategy",
                parse_plugin("astar_strategy", self.strategy).construct(self.strategy),
            ),
            cast(
                "AStarRootMode",
                parse_plugin("astar_root_mode", self.root_mode).construct(
                    self.root_mode
                ),
            ),
            cast(
                "AStarHeuristic",
                parse_plugin("astar_heuristic", self.heuristic).construct(
                    self.heuristic
                ),
            ),
            (
                None
                if self.proposer is None
                else cast(
                    "AStarProposerFactory",
                    parse_plugin("astar_proposer", self.proposer).construct(
                        self.proposer
                    ),
                )
            ),
        )


class MbrUtilitySpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declares how an MBR candidate earns utility."""


class ConstituentUtilitySpec(MbrUtilitySpec, extra="forbid"):
    """Use posterior constituent mass as utility."""

    kind: Literal["constituent"] = "constituent"


def _constituent(declaration: ConstituentUtilitySpec, /) -> MbrUtility:
    """Build posterior-constituent utility."""
    del declaration
    return ConstituentUtility()


CONSTITUENT_UTILITY_PLUGIN = ParsePlugin(
    spec=ConstituentUtilitySpec,
    build=_constituent,
)


class DerivationDecision(Protocol):
    """Selects one derivation from a finished A* search."""

    @property
    def conventions(self) -> Mapping[str, JsonValue]:
        """Return the named decode conventions a report row carries."""
        ...

    def select(
        self, result: AStarResult[Term], context: AStarContext[Term], /
    ) -> ForestDerivation[Term, float] | None:
        """Return the selected derivation, or None when nothing was parsed."""
        ...


def derivation_nodes(
    derivation: ForestDerivation[Term, float], /
) -> tuple[ForestDerivation[Term, float], ...]:
    """Return every node of a derivation in stable depth-first order.

    Parameters
    ----------
    derivation : ForestDerivation[Term, float]
        The derivation to walk.

    Returns
    -------
    tuple[ForestDerivation[Term, float], ...]
        The root first, then each subtree left to right.
    """
    reached: list[ForestDerivation[Term, float]] = []
    active = [derivation]

    while active:
        node = active.pop()
        reached.append(node)
        active.extend(reversed(node.children))

    return tuple(reached)


def _derivation_score(
    derivation: ForestDerivation[Term, float],
    order: LogProbability,
    /,
) -> float:
    """Return one derivation's sum of local log scores."""
    return fsum(order.rank(node.edge.weight) for node in derivation_nodes(derivation))


@final
class ViterbiDecision:
    """Select the highest-scoring derivation among the search's roots."""

    __slots__ = ()

    @property
    def conventions(self) -> Mapping[str, JsonValue]:
        """Return the Viterbi decode conventions."""
        return MappingProxyType(
            {"decode_method": "viterbi", "decode_utility": None, "decode_candidates": 1}
        )

    def select(
        self, result: AStarResult[Term], context: AStarContext[Term], /
    ) -> ForestDerivation[Term, float] | None:
        """Return the highest-scoring derivation, or None for no root."""
        del context
        selected: ForestDerivation[Term, float] | None = None
        highest = -inf
        order = LogProbability()

        for root in result.roots:
            candidate = next(unfold(result.chart, root, limit=1, order=order), None)

            if candidate is None:
                continue

            score = _derivation_score(candidate, order)

            if score > highest:
                selected = candidate
                highest = score

        return selected

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another decision is also Viterbi selection."""
        return type(other) is ViterbiDecision

    @override
    def __hash__(self) -> int:
        """Return the hash shared by every Viterbi decision."""
        return hash(ViterbiDecision)

    @override
    def __repr__(self) -> str:
        """Return the constructor call."""
        return "ViterbiDecision()"


@final
class MbrDecision:
    """Select the highest-utility member of a bounded candidate set.

    Parameters
    ----------
    utility : MbrUtility
        The gain each candidate earns against posterior constituent mass.
    candidates : int
        How many highest-scoring candidates are compared.

    Raises
    ------
    ValueError
        If the candidate count is not positive.
    """

    __slots__ = ("_candidates", "_utility")

    def __init__(self, utility: MbrUtility, candidates: int) -> None:
        self._utility = utility
        self._candidates = ensure_positive("an MBR candidate count", candidates)

    @property
    def conventions(self) -> Mapping[str, JsonValue]:
        """Return the MBR utility and candidate count."""
        return MappingProxyType(
            {
                "decode_method": "mbr",
                "decode_utility": self._utility.name,
                "decode_candidates": self._candidates,
            }
        )

    def select(
        self, result: AStarResult[Term], context: AStarContext[Term], /
    ) -> ForestDerivation[Term, float] | None:
        """Return the highest-utility candidate among all selected roots."""
        posterior_goal = result.options.root_mode.goal(context.goal)
        outside = outside_from_edges(result.chart, posterior_goal)
        marginals = {
            key: outside.constituent_marginal(key)
            for span in result.chart.index.spans_by_length()
            for key in result.chart.items_at(span)
        }
        order = LogProbability()
        ranked = sorted(
            (
                candidate
                for root in result.roots
                for candidate in unfold(
                    result.chart,
                    root,
                    limit=self._candidates,
                    order=order,
                )
            ),
            key=lambda candidate: _derivation_score(candidate, order),
            reverse=True,
        )[: self._candidates]
        selected: ForestDerivation[Term, float] | None = None
        highest = -inf

        for candidate in ranked:
            gain = self._utility.gain(candidate, marginals=marginals)

            if gain > highest:
                selected = candidate
                highest = gain

        return selected

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another decision compares the same candidates."""
        return (
            type(other) is MbrDecision
            and self._utility.name == other._utility.name
            and self._candidates == other._candidates
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the utility name and candidate count."""
        return hash((MbrDecision, self._utility.name, self._candidates))

    @override
    def __repr__(self) -> str:
        """Return the utility and candidate count."""
        return (
            f"MbrDecision(utility={self._utility.name!r}, "
            f"candidates={self._candidates!r})"
        )


class DecodeMethodDeclaration(Protocol):
    """A derivation-decoding declaration."""

    @property
    def requires_marginals(self) -> bool:
        """Return whether the decoder requires posterior marginals."""
        ...


@runtime_checkable
class _RootModeDeclaration(Protocol):
    """A root-mode declaration that states which spanning items it reads."""

    @property
    def needs_every_spanning_item(self) -> bool:
        """Return whether the mode reads items that reach no goal root."""


def astar_parse_refusal(
    astar: AStarSpec,
    parse: ParseSpec,
    /,
    *,
    marginals: bool,
) -> DeclarationError | None:
    """Return why a run parse cannot serve one A* decode, or None.

    Parameters
    ----------
    astar : AStarSpec
        The declared A* search.
    parse : ParseSpec
        The run's symbolic parse declaration.
    marginals : bool
        Whether the consumer reads posterior marginals from the A* chart.

    Returns
    -------
    DeclarationError | None
        The refusal, or None when the parse serves the search. A semiring
        without the float log-probability weights A* ranks by, or without the
        marginals the consumer reads, is refused with a
        ``SemiringCapabilityError``.
    """
    if parse.retain != "forest":
        return DeclarationError(
            "A* decoding reads its roots from a packed forest; declare "
            "parse.retain: forest"
        )

    root_mode = astar.root_mode

    if not isinstance(root_mode, _RootModeDeclaration):
        return DeclarationError(
            f"the A* root mode {root_mode.kind!r} does not declare which "
            "spanning items it reads"
        )

    if parse.reachability == "roots" and root_mode.needs_every_spanning_item:
        return DeclarationError(
            f"the A* root mode {root_mode.kind!r} reads spanning items that "
            "parse.reachability: roots does not construct"
        )

    weights = parse.semiring_refusal("log-probability", consumer="the A* decode")

    if weights is not None or not marginals:
        return weights

    return parse.semiring_refusal("marginals", consumer="the decode")


class DecodeSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declares how one derivation is selected from a finished A* search."""


class ViterbiDecodeSpec(DecodeSpec, extra="forbid"):
    """Select the single highest-scoring derivation."""

    kind: Literal["viterbi"] = "viterbi"

    @property
    def requires_marginals(self) -> bool:
        """Return that Viterbi selection needs no posterior marginals."""
        return False


class MbrDecodeSpec(DecodeSpec, extra="forbid"):
    """Select the highest-utility member of a bounded candidate set."""

    kind: Literal["mbr"] = "mbr"
    utility: MbrUtilitySpec = ConstituentUtilitySpec()
    candidates: int

    __axioms__ = (
        dx.axiom(
            "candidates > 0",
            message="an MBR decode compares a positive candidate count",
        ),
    )

    @property
    def requires_marginals(self) -> bool:
        """Return that minimum Bayes risk requires posterior marginals."""
        return True


def _viterbi(declaration: ViterbiDecodeSpec, /) -> DerivationDecision:
    """Build Viterbi derivation selection."""
    del declaration
    return ViterbiDecision()


def _mbr(declaration: MbrDecodeSpec, /) -> DerivationDecision:
    """Build minimum-Bayes-risk selection under the declared utility."""
    return MbrDecision(
        cast(
            "MbrUtility",
            parse_plugin("mbr_utility", declaration.utility).construct(
                declaration.utility
            ),
        ),
        declaration.candidates,
    )


VITERBI_DECODE_PLUGIN = ParsePlugin(spec=ViterbiDecodeSpec, build=_viterbi)
MBR_DECODE_PLUGIN = ParsePlugin(spec=MbrDecodeSpec, build=_mbr)


def decode_decision(declaration: DecodeSpec, /) -> DerivationDecision:
    """Return the derivation decision one declaration's installed plugin builds.

    Parameters
    ----------
    declaration : DecodeSpec
        A validated decode-method declaration.

    Returns
    -------
    DerivationDecision
        The runtime selection rule.
    """
    return cast(
        "DerivationDecision",
        parse_plugin("decode_method", declaration).construct(declaration),
    )


class DecodeEvaluatorSettings(NamedTuple):
    """Resolved settings consumed by the decode evaluator."""

    readout: str
    output: DecodeOutput
    astar: AStarOptions | None
    decode: DerivationDecision | None
    conventions: ConventionSettings


class DecodeEvaluatorSpec(EvaluatorSpec):
    """Declare held-out neural-term beams and optional A* parsing."""

    kind: Literal["decode"] = "decode"
    readout: str
    output: DecodeOutput
    astar: AStarSpec | None = None
    decode: DecodeSpec | None = None
    conventions: ConventionSpec

    @dx.validates("readout")
    @classmethod
    def _readout_is_named(cls, value: str) -> str:
        """Return a nonempty complete-term readout role."""
        return ensure_name("a decode evaluator's term-readout role", value)

    @dx.model_validator()
    def _astar_decodes_lexical_types(self) -> DecodeEvaluatorSpec:
        """Keep symbolic A* and derivation selection on one declared route."""
        if self.output == "logical-forms" and self.astar is not None:
            msg = "logical-form beams do not enter supertag-factored A*"
            raise ValueError(msg)

        if (self.astar is None) != (self.decode is None):
            msg = "an A* decode declares both astar and decode, or neither"
            raise ValueError(msg)

        return self

    def parse_refusal(self, parse: ParseSpec, /) -> DeclarationError | None:
        """Return why the run parse cannot serve the declared A* decode."""
        astar = self.astar
        decode = cast("DecodeMethodDeclaration | None", self.decode)

        if astar is None or decode is None:
            return None

        return astar_parse_refusal(astar, parse, marginals=decode.requires_marginals)

    @override
    def dependencies(self) -> tuple[Dependency, ...]:
        """Return the declared complete-term readout dependency."""
        return (Dependency(self.readout, "terms-readout"),)

    @override
    def metrics(self, labels: tuple[str, ...] = (), /) -> tuple[str, ...]:
        """Return beam size and, when declared, A* diagnostics."""
        del labels
        if self.astar is None:
            return ("decode-candidates",)

        return (
            "decode-candidates",
            "decode-parsed",
            "decode-refused",
            "decode-attempts",
            "decode-widenings",
        )

    def settings(self) -> DecodeEvaluatorSettings:
        """Return immutable evaluator settings."""
        astar = self.astar
        decode = self.decode
        return DecodeEvaluatorSettings(
            self.readout,
            self.output,
            None if astar is None else astar.build(),
            None if decode is None else decode_decision(decode),
            self.conventions.settings(),
        )

"""Interchange-intervention pairs generated from a declared grammar and lexicon.

A pair holds one token string and two of its derivations under the same rule set.
The two agree on every cell, every split and every combinator, and disagree on the
category exactly one lexical item bears, so an intervention aligned to the pair
moves one high-level variable and nothing else. Both derivations are the
generator's own choice rather than a reading of an ambiguous gold parse, so the
split and the combinator at every cell are recorded whatever normal form the rule
set declares.
"""

from collections.abc import Mapping
from hashlib import sha256
from itertools import chain, islice
from json import dumps
from random import Random
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, NewType, override

import torch

from lofi._checks import ensure_name, ensure_non_negative
from lofi._frozen import FrozenValue
from lofi.deduce import Boolean, ParsePolicy, Retain, SpanFree, parse
from lofi.probe._causal import (
    AlignedInput,
    AlignmentPolicy,
    CombinatorAt,
    CounterfactualPair,
    JointCounterfactualPair,
    NamedVariable,
    Side,
    SplitAt,
    TokenizerAlignment,
    TypeAt,
)
from lofi.probe._provenance import Provenance
from lofi.probe._variables import extract_variables

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from torch import Tensor

    from lofi._types import JsonValue
    from lofi.deduce import (
        Cell,
        Derivation,
        Goal,
        Normalizer,
        PackedForest,
        Proposer,
        Truth,
    )
    from lofi.judge import Rule, RuleSet
    from lofi.lexicon import Clamp, Entry, LexicalSource
    from lofi.probe._causal import CellAssignment, HighLevelVariable

__all__ = [
    "ALIGNMENT_EXCLUSION_REASONS",
    "ALIGNMENT_REFUSAL_REASONS",
    "AlignmentExclusion",
    "AlignmentExclusionReason",
    "AlignmentManifest",
    "AlignmentRefusal",
    "AlignmentRefusalReason",
    "CounterfactualGeneration",
    "CounterfactualGenerator",
    "GeneratedPair",
    "GeneratedPairId",
    "ModelCausalPair",
    "ModelJointCausalPair",
    "confirm_pair",
]

GeneratedPairId = NewType("GeneratedPairId", str)
"""A stable content address for one generated counterfactual pair."""

type AlignmentRefusalReason = Literal[
    "empty-model-row-span",
    "incomplete-source-span",
    "unordered-model-row-span",
    "noncontiguous-model-row-span",
]
"""A stable reason an alignment refused a generated candidate."""

ALIGNMENT_REFUSAL_REASONS: Final[tuple[AlignmentRefusalReason, ...]] = (
    "empty-model-row-span",
    "incomplete-source-span",
    "unordered-model-row-span",
    "noncontiguous-model-row-span",
)
"""Alignment refusal reasons in report order."""

type AlignmentExclusionReason = Literal[
    "zero-piece-word",
    "crossing-piece",
    "context-overflow",
]
"""A stable reason a tokenizer could not align a candidate sentence whole."""

ALIGNMENT_EXCLUSION_REASONS: Final[tuple[AlignmentExclusionReason, ...]] = (
    "zero-piece-word",
    "crossing-piece",
    "context-overflow",
)
"""Alignment exclusion reasons in report order."""

_PAIR_ID_DOMAIN: Final = "lofi.probe.generated-pair"
_PAIR_ID_VERSION: Final = 1

type _VariableKey = tuple[str, int, int, int]
"""One high-level variable, as a key a sort can order variables by."""

type _Decisions = tuple[tuple[HighLevelVariable, str | int], ...]
"""What one derivation decided, in variable order."""

type _Reading = tuple[tuple[str, ...], CellAssignment]
"""One derivation, as its lexical categories and the variables it decided."""


class AlignmentRefusal(FrozenValue):
    """A tokenizer-specific refusal of one generated candidate.

    Parameters
    ----------
    pair_id : GeneratedPairId
        Stable content address of the rejected pair.
    tokenizer : str
        Identity of the tokenizer whose alignment rejected it.
    source_span : Cell
        Half-open source-token span that failed alignment.
    reason : AlignmentRefusalReason
        Stable machine-readable reason for the refusal.

    Raises
    ------
    TypeError
        If an identity or reason has the wrong runtime type.
    ValueError
        If an identity is empty, the source span is invalid or the reason is not
        declared.
    """

    __slots__ = ("_pair_id", "_reason", "_source_span", "_tokenizer")

    _noun = "alignment refusal"

    def __init__(
        self,
        pair_id: GeneratedPairId,
        tokenizer: str,
        source_span: Cell,
        reason: AlignmentRefusalReason,
    ) -> None:
        if not isinstance(reason, str):
            msg = f"an alignment refusal reason is not a string: {reason!r}"
            raise TypeError(msg)

        if reason not in ALIGNMENT_REFUSAL_REASONS:
            msg = f"no alignment refusal reason is called {reason!r}"
            raise ValueError(msg)

        object.__setattr__(
            self, "_pair_id", GeneratedPairId(ensure_name("a refused pair id", pair_id))
        )
        object.__setattr__(
            self,
            "_tokenizer",
            ensure_name("a refusing tokenizer's identity", tokenizer),
        )
        object.__setattr__(self, "_source_span", TypeAt(source_span).cell)
        object.__setattr__(self, "_reason", reason)

    @property
    def pair_id(self) -> GeneratedPairId:
        """Return the rejected pair's stable content address."""
        return self._pair_id

    @property
    def tokenizer(self) -> str:
        """Return the identity of the tokenizer that rejected the pair."""
        return self._tokenizer

    @property
    def source_span(self) -> Cell:
        """Return the source-token span that failed alignment."""
        return self._source_span

    @property
    def reason(self) -> AlignmentRefusalReason:
        """Return the stable reason for the refusal."""
        return self._reason

    def _identity(
        self,
    ) -> tuple[GeneratedPairId, str, Cell, AlignmentRefusalReason]:
        """Return the fields a refusal is equal on."""
        return (self._pair_id, self._tokenizer, self._source_span, self._reason)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value records the same refusal."""
        if type(other) is not AlignmentRefusal:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the refusal."""
        return hash((AlignmentRefusal, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this refusal."""
        return (
            f"AlignmentRefusal({self._pair_id!r}, {self._tokenizer!r}, "
            f"{self._source_span!r}, {self._reason!r})"
        )


class AlignmentExclusion(FrozenValue):
    """A candidate sentence one tokenizer could not align whole.

    An excluded sentence yields no candidate pair, so it is counted beside the
    refusals rather than among them.

    Parameters
    ----------
    tokens : tuple[str, ...]
        The source tokens of the excluded sentence.
    tokenizer : str
        Identity of the tokenizer that could not align it.
    reason : AlignmentExclusionReason
        Stable machine-readable reason for the exclusion.

    Raises
    ------
    TypeError
        If the reason is not a string.
    ValueError
        If the sentence has no token, the identity is empty or the reason is not
        declared.
    """

    __slots__ = ("_reason", "_tokenizer", "_tokens")

    _noun = "alignment exclusion"

    def __init__(
        self,
        tokens: tuple[str, ...],
        tokenizer: str,
        reason: AlignmentExclusionReason,
    ) -> None:
        if not isinstance(reason, str):
            msg = f"an alignment exclusion reason is not a string: {reason!r}"
            raise TypeError(msg)

        if reason not in ALIGNMENT_EXCLUSION_REASONS:
            msg = f"no alignment exclusion reason is called {reason!r}"
            raise ValueError(msg)

        held = tuple(tokens)

        if not held:
            msg = "an excluded sentence has no source token"
            raise ValueError(msg)

        object.__setattr__(self, "_tokens", held)
        object.__setattr__(
            self,
            "_tokenizer",
            ensure_name("an excluding tokenizer's identity", tokenizer),
        )
        object.__setattr__(self, "_reason", reason)

    @property
    def tokens(self) -> tuple[str, ...]:
        """Return the source tokens of the excluded sentence."""
        return self._tokens

    @property
    def tokenizer(self) -> str:
        """Return the identity of the tokenizer that could not align it."""
        return self._tokenizer

    @property
    def reason(self) -> AlignmentExclusionReason:
        """Return the stable reason for the exclusion."""
        return self._reason

    def _identity(self) -> tuple[tuple[str, ...], str, AlignmentExclusionReason]:
        """Return the fields an exclusion is equal on."""
        return (self._tokens, self._tokenizer, self._reason)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value records the same exclusion."""
        if type(other) is not AlignmentExclusion:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the exclusion."""
        return hash((AlignmentExclusion, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this exclusion."""
        return (
            f"AlignmentExclusion({self._tokens!r}, {self._tokenizer!r}, "
            f"{self._reason!r})"
        )


class AlignmentManifest(FrozenValue):
    """Counts candidate alignment under one exact policy and tokenizer order.

    Parameters
    ----------
    policy : AlignmentPolicy
        Rules applied to every tokenizer alignment.
    tokenizers : tuple[str, ...]
        Tokenizer identities in evaluation order.
    candidate_count : int
        Number of generated candidates checked.
    aligned_count : int
        Number accepted under every tokenizer alignment.

    Raises
    ------
    TypeError
        If the policy is not one or a count or identity has the wrong runtime type.
    ValueError
        If identities repeat, a count is negative or the aligned count exceeds the
        candidate count.
    """

    __slots__ = ("_aligned_count", "_candidate_count", "_policy", "_tokenizers")

    _noun = "alignment manifest"

    def __init__(
        self,
        policy: AlignmentPolicy,
        tokenizers: tuple[str, ...],
        candidate_count: int,
        aligned_count: int,
    ) -> None:
        if type(policy) is not AlignmentPolicy:
            msg = f"an alignment manifest requires an AlignmentPolicy, not {policy!r}"
            raise TypeError(msg)

        identities = tuple(
            ensure_name("a manifest tokenizer identity", tokenizer)
            for tokenizer in tokenizers
        )

        if len(set(identities)) != len(identities):
            msg = f"a manifest repeats a tokenizer identity: {identities!r}"
            raise ValueError(msg)

        candidates = ensure_non_negative(
            "an alignment candidate count", candidate_count
        )
        aligned = ensure_non_negative("an aligned candidate count", aligned_count)

        if aligned > candidates:
            msg = f"an aligned count {aligned} exceeds the candidate count {candidates}"
            raise ValueError(msg)

        if candidates and not identities:
            msg = "a nonempty alignment population names at least one tokenizer"
            raise ValueError(msg)

        object.__setattr__(self, "_policy", policy)
        object.__setattr__(self, "_tokenizers", identities)
        object.__setattr__(self, "_candidate_count", candidates)
        object.__setattr__(self, "_aligned_count", aligned)

    @property
    def policy(self) -> AlignmentPolicy:
        """Return the exact policy applied to every candidate."""
        return self._policy

    @property
    def tokenizers(self) -> tuple[str, ...]:
        """Return tokenizer identities in evaluation order."""
        return self._tokenizers

    @property
    def candidate_count(self) -> int:
        """Return how many generated candidates were checked."""
        return self._candidate_count

    @property
    def aligned_count(self) -> int:
        """Return how many candidates every tokenizer accepted."""
        return self._aligned_count

    @property
    def aligned_fraction(self) -> float | None:
        """Return the accepted fraction, or none for an empty population."""
        if self._candidate_count == 0:
            return None

        return self._aligned_count / self._candidate_count

    def _identity(self) -> tuple[AlignmentPolicy, tuple[str, ...], int, int]:
        """Return the fields a manifest is equal on."""
        return (
            self._policy,
            self._tokenizers,
            self._candidate_count,
            self._aligned_count,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value records the same alignment population."""
        if type(other) is not AlignmentManifest:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the manifest."""
        return hash((AlignmentManifest, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this manifest."""
        return (
            f"AlignmentManifest({self._policy!r}, {self._tokenizers!r}, "
            f"{self._candidate_count!r}, {self._aligned_count!r})"
        )


def _variable_key(variable: HighLevelVariable) -> _VariableKey:
    """Return an orderable key naming one high-level variable.

    Parameters
    ----------
    variable : HighLevelVariable
        The variable to name.

    Returns
    -------
    _VariableKey
        The variant, the cell's start and end, and the split a combinator names
        or minus one where the variant names none.
    """
    if isinstance(variable, NamedVariable):
        msg = "structural counterfactual generation does not admit NamedVariable"
        raise TypeError(msg)
    start, end = variable.cell
    split = variable.split if isinstance(variable, CombinatorAt) else -1

    return (variable.kind, start, end, split)


def _variable_json(variable: HighLevelVariable) -> dict[str, JsonValue]:
    """Return one high-level variable as a JSON object.

    Parameters
    ----------
    variable : HighLevelVariable
        The variable to serialize.

    Returns
    -------
    dict[str, JsonValue]
        The variant, the cell and the split a combinator names, which is null
        for the other variants.
    """
    if isinstance(variable, NamedVariable):
        msg = "structural counterfactual generation does not admit NamedVariable"
        raise TypeError(msg)
    start, end = variable.cell
    split = variable.split if isinstance(variable, CombinatorAt) else None

    return {"kind": variable.kind, "cell": [start, end], "split": split}


def _assignment_json(assignment: CellAssignment) -> list[JsonValue]:
    """Return what one derivation decided as a JSON array.

    Parameters
    ----------
    assignment : CellAssignment
        The decisions to serialize.

    Returns
    -------
    list[JsonValue]
        One object per variable, in variable order, each holding the variable
        and the value it took.
    """
    return [
        {"variable": _variable_json(variable), "value": decided}
        for variable, decided in sorted(
            assignment.items(), key=lambda item: _variable_key(item[0])
        )
    ]


def _pair_id(
    tokens: tuple[str, ...],
    base: tuple[str, ...],
    variant: tuple[str, ...],
    base_assignment: CellAssignment,
    variant_assignment: CellAssignment,
    varied: HighLevelVariable,
    grammar: str,
) -> GeneratedPairId:
    """Return the content address of a generated pair's semantic fields.

    Parameters
    ----------
    tokens : tuple[str, ...]
        The string both derivations cover.
    base : tuple[str, ...]
        The category each token bears in the base derivation.
    variant : tuple[str, ...]
        The category each token bears in the counterfactual derivation.
    base_assignment : CellAssignment
        What the base derivation decided.
    variant_assignment : CellAssignment
        What the counterfactual derivation decided.
    varied : HighLevelVariable
        The variable the pair changes.
    grammar : str
        The fingerprint of the grammar both derivations use.

    Returns
    -------
    GeneratedPairId
        The lowercase SHA-256 digest of the versioned canonical JSON record.
    """
    payload: dict[str, JsonValue] = {
        "domain": _PAIR_ID_DOMAIN,
        "version": _PAIR_ID_VERSION,
        "tokens": list(tokens),
        "base": list(base),
        "variant": list(variant),
        "base_assignment": _assignment_json(base_assignment),
        "variant_assignment": _assignment_json(variant_assignment),
        "varied": _variable_json(varied),
        "grammar": grammar,
    }
    canonical = dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()

    return GeneratedPairId(sha256(canonical).hexdigest())


def _decisions(assignment: CellAssignment) -> _Decisions:
    """Return what one derivation decided in a canonical order.

    Parameters
    ----------
    assignment : CellAssignment
        The decisions to order.

    Returns
    -------
    _Decisions
        One variable-and-value pair per variable, in variable order.
    """
    return tuple(sorted(assignment.items(), key=lambda item: _variable_key(item[0])))


def _differing(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[int, ...]:
    """Return the positions at which two lexical assignments disagree.

    Parameters
    ----------
    left : tuple[str, ...]
        One category per token.
    right : tuple[str, ...]
        One category per token, over the same tokens.

    Returns
    -------
    tuple[int, ...]
        The positions bearing different categories, in ascending order.

    Raises
    ------
    ValueError
        If the two name categories for different numbers of tokens.
    """
    return tuple(
        position
        for position, (here, there) in enumerate(zip(left, right, strict=True))
        if here != there
    )


class GeneratedPair:
    """One interchange-intervention record, with the provenance that identifies it.

    A record is a claim, not a certificate. ``confirm_pair`` is what reads it back
    and decides whether the two assignments really differ at the one variable the
    record names.

    Parameters
    ----------
    tokens : tuple[str, ...]
        The string both derivations cover.
    base : tuple[str, ...]
        The category each token bears in the base derivation.
    variant : tuple[str, ...]
        The category each token bears in the counterfactual derivation.
    base_assignment : CellAssignment
        What the base derivation decided, variable by variable.
    variant_assignment : CellAssignment
        What the counterfactual derivation decided, over the same variables.
    varied : HighLevelVariable
        The variable the record claims the two differ at.
    grammar : str
        The fingerprint of the grammar both derivations were built under.
    provenance : Provenance
        What generated the record, over what data, under what seed.

    Raises
    ------
    TypeError
        If the varied variable is named or is not exactly a type, split or
        combinator variable, or the provenance is not one.
    ValueError
        If the fingerprint is empty, if the string is empty, if a category
        sequence is a different length from the string, if the varied cell runs
        past the string, or if the provenance records no seed.
    """

    __slots__ = (
        "_base",
        "_base_assignment",
        "_grammar",
        "_id",
        "_provenance",
        "_seed",
        "_tokens",
        "_variant",
        "_variant_assignment",
        "_varied",
    )

    def __init__(
        self,
        tokens: tuple[str, ...],
        base: tuple[str, ...],
        variant: tuple[str, ...],
        base_assignment: CellAssignment,
        variant_assignment: CellAssignment,
        varied: HighLevelVariable,
        grammar: str,
        provenance: Provenance,
    ) -> None:
        if isinstance(varied, NamedVariable):
            msg = "a generated structural record does not admit NamedVariable"
            raise TypeError(msg)
        if type(varied) not in {TypeAt, SplitAt, CombinatorAt}:
            msg = (
                "a generated structural record varies exactly a TypeAt, SplitAt "
                f"or CombinatorAt, not {varied!r}"
            )
            raise TypeError(msg)

        if type(provenance) is not Provenance:
            msg = f"a record requires a Provenance, not {provenance!r}"
            raise TypeError(msg)

        if not tokens:
            msg = "a record covers a nonempty string"
            raise ValueError(msg)

        for named, categories in (("base", base), ("variant", variant)):
            if len(categories) != len(tokens):
                msg = (
                    f"a record's {named} names {len(categories)} categories for "
                    f"{len(tokens)} tokens"
                )
                raise ValueError(msg)

        if varied.cell[1] > len(tokens):
            msg = (
                f"a record varies {varied!r}, whose cell runs past the "
                f"{len(tokens)} tokens it covers"
            )
            raise ValueError(msg)

        if provenance.seed is None:
            msg = "a record names the seed its generator drew under"
            raise ValueError(msg)

        self._tokens = tokens
        self._base: tuple[str, ...] = base
        self._variant: tuple[str, ...] = variant
        self._base_assignment: CellAssignment = MappingProxyType(dict(base_assignment))
        self._variant_assignment: CellAssignment = MappingProxyType(
            dict(variant_assignment)
        )
        self._varied: TypeAt | SplitAt | CombinatorAt = varied
        self._grammar = ensure_name("a record's grammar", grammar)
        self._provenance = provenance
        self._seed = provenance.seed
        self._id = _pair_id(
            self._tokens,
            self._base,
            self._variant,
            self._base_assignment,
            self._variant_assignment,
            self._varied,
            self._grammar,
        )

    @property
    def id(self) -> GeneratedPairId:
        """Return the pair's stable content address."""
        return self._id

    @property
    def tokens(self) -> tuple[str, ...]:
        """Return the string both derivations cover."""
        return self._tokens

    @property
    def base(self) -> tuple[str, ...]:
        """Return the category each token bears in the base derivation."""
        return self._base

    @property
    def variant(self) -> tuple[str, ...]:
        """Return the category each token bears in the counterfactual derivation."""
        return self._variant

    @property
    def base_assignment(self) -> CellAssignment:
        """Return what the base derivation decided."""
        return self._base_assignment

    @property
    def variant_assignment(self) -> CellAssignment:
        """Return what the counterfactual derivation decided."""
        return self._variant_assignment

    @property
    def varied(self) -> TypeAt | SplitAt | CombinatorAt:
        """Return the variable the record claims the two derivations differ at."""
        return self._varied

    @property
    def span(self) -> Cell:
        """Return the half-open cell an intervention on the record targets."""
        return self._varied.cell

    @property
    def form(self) -> str:
        """Return the surface form whose category the record varies."""
        return self._tokens[self._varied.cell[0]]

    @property
    def grammar(self) -> str:
        """Return the fingerprint of the grammar both derivations were built under."""
        return self._grammar

    @property
    def provenance(self) -> Provenance:
        """Return what generated the record, over what data, under what seed."""
        return self._provenance

    @property
    def seed(self) -> int:
        """Return the seed the generator drew under."""
        return self._seed

    def to_json(self) -> dict[str, JsonValue]:
        """Return the record and its provenance as a JSON object.

        Returns
        -------
        dict[str, JsonValue]
            The string, both lexical assignments, both variable assignments, the
            varied variable, the span, the form varied, the grammar fingerprint
            and the provenance the record was generated under.
        """
        start, end = self._varied.cell

        return {
            "id": self._id,
            "tokens": list(self._tokens),
            "base": list(self._base),
            "variant": list(self._variant),
            "base_assignment": _assignment_json(self._base_assignment),
            "variant_assignment": _assignment_json(self._variant_assignment),
            "varied": _variable_json(self._varied),
            "span": [start, end],
            "form": self.form,
            "grammar": self._grammar,
            "provenance": self._provenance.to_json(),
        }

    @override
    def __repr__(self) -> str:
        """Return the form varied, the two categories, the span and the grammar."""
        position = self._varied.cell[0]

        return (
            f"GeneratedPair(form={self.form!r}, base={self._base[position]!r}, "
            f"variant={self._variant[position]!r}, span={self._varied.cell!r}, "
            f"grammar={self._grammar!r})"
        )


class CounterfactualGeneration(FrozenValue):
    """Accepted pairs, alignment refusals, excluded sentences and their manifest.

    Parameters
    ----------
    pairs : tuple[GeneratedPair, ...]
        Candidates accepted by every tokenizer alignment, in generated order.
    aligned_inputs : Mapping[GeneratedPairId, AlignedInput]
        The exact tokenizer evidence validated for every accepted pair.
    refusals : tuple[AlignmentRefusal, ...]
        One record for every tokenizer alignment that rejected a candidate.
    manifest : AlignmentManifest
        Policy, tokenizer order and exact population counts.
    exclusions : tuple[AlignmentExclusion, ...]
        One record for every tokenizer that could not align a candidate sentence
        whole, which therefore entered no generation.

    Raises
    ------
    TypeError
        If a member has the wrong runtime type.
    ValueError
        If pair identities repeat, retained evidence does not cover exactly the
        accepted identities, differs from a pair's tokens or manifest tokenizer
        order, the manifest's aligned count differs from the accepted pair count,
        or a refusal names an unknown tokenizer.
    """

    __slots__ = (
        "_aligned_inputs",
        "_exclusions",
        "_manifest",
        "_pairs",
        "_records",
        "_refusals",
    )

    _noun = "counterfactual generation"

    def __init__(
        self,
        pairs: tuple[GeneratedPair, ...],
        aligned_inputs: Mapping[GeneratedPairId, AlignedInput],
        refusals: tuple[AlignmentRefusal, ...],
        manifest: AlignmentManifest,
        *,
        exclusions: tuple[AlignmentExclusion, ...] = (),
    ) -> None:
        if any(type(pair) is not GeneratedPair for pair in pairs):
            msg = "a counterfactual generation contains only GeneratedPair values"
            raise TypeError(msg)

        if any(type(exclusion) is not AlignmentExclusion for exclusion in exclusions):
            msg = (
                "a counterfactual generation contains only AlignmentExclusion "
                "values among its exclusions"
            )
            raise TypeError(msg)

        if any(type(refusal) is not AlignmentRefusal for refusal in refusals):
            msg = "a counterfactual generation contains only AlignmentRefusal values"
            raise TypeError(msg)

        if type(manifest) is not AlignmentManifest:
            msg = (
                "a counterfactual generation requires an AlignmentManifest, "
                f"not {manifest!r}"
            )
            raise TypeError(msg)

        if not isinstance(aligned_inputs, Mapping):
            msg = (
                "a counterfactual generation retains an alignment mapping, "
                f"not {aligned_inputs!r}"
            )
            raise TypeError(msg)

        identities = tuple(pair.id for pair in pairs)

        if len(set(identities)) != len(identities):
            msg = "a counterfactual generation contains a duplicate pair identity"
            raise ValueError(msg)

        retained = dict(aligned_inputs)

        if set(retained) != set(identities):
            missing = tuple(
                identity for identity in identities if identity not in retained
            )
            foreign = tuple(
                identity for identity in retained if identity not in identities
            )
            msg = (
                "retained tokenizer evidence does not cover exactly the accepted "
                f"pairs; missing={missing!r}, foreign={foreign!r}"
            )
            raise ValueError(msg)

        for pair in pairs:
            aligned = retained[pair.id]

            if type(aligned) is not AlignedInput:
                msg = (
                    f"retained evidence for {pair.id!r} is an AlignedInput, "
                    f"not {aligned!r}"
                )
                raise TypeError(msg)

            if aligned.tokens != pair.tokens:
                msg = (
                    f"retained evidence for {pair.id!r} covers {aligned.tokens!r}, "
                    f"not the pair's tokens {pair.tokens!r}"
                )
                raise ValueError(msg)

            if aligned.tokenizers != manifest.tokenizers:
                msg = (
                    f"retained evidence for {pair.id!r} names tokenizer order "
                    f"{aligned.tokenizers!r}, not {manifest.tokenizers!r}"
                )
                raise ValueError(msg)

        if manifest.aligned_count != len(pairs):
            msg = (
                f"a manifest counts {manifest.aligned_count} aligned candidates "
                f"but the generation contains {len(pairs)} pairs"
            )
            raise ValueError(msg)

        rejected = manifest.candidate_count - manifest.aligned_count

        if len(refusals) < rejected:
            msg = (
                f"a manifest counts {rejected} rejected candidates but the "
                f"generation contains only {len(refusals)} refusals"
            )
            raise ValueError(msg)

        if rejected == 0 and refusals:
            msg = "a fully aligned population cannot contain alignment refusals"
            raise ValueError(msg)

        for refusal in refusals:
            if refusal.tokenizer not in manifest.tokenizers:
                msg = (
                    f"refusal tokenizer {refusal.tokenizer!r} is absent from "
                    f"manifest order {manifest.tokenizers!r}"
                )
                raise ValueError(msg)

        object.__setattr__(self, "_pairs", tuple(pairs))
        object.__setattr__(
            self,
            "_records",
            MappingProxyType(dict(zip(identities, pairs, strict=True))),
        )
        object.__setattr__(self, "_aligned_inputs", MappingProxyType(retained))
        object.__setattr__(self, "_refusals", tuple(refusals))
        object.__setattr__(self, "_manifest", manifest)
        object.__setattr__(self, "_exclusions", tuple(exclusions))

    @property
    def pairs(self) -> tuple[GeneratedPair, ...]:
        """Return accepted pairs in generated order."""
        return self._pairs

    @property
    def aligned_inputs(self) -> Mapping[GeneratedPairId, AlignedInput]:
        """Return exact tokenizer evidence keyed by accepted pair identity."""
        return self._aligned_inputs

    def pair(self, pair_id: GeneratedPairId, /) -> GeneratedPair:
        """Return the accepted record with one stable identity.

        Raises
        ------
        KeyError
            If no accepted pair has that identity.
        """
        try:
            return self._records[pair_id]
        except KeyError:
            msg = f"counterfactual generation contains no accepted pair {pair_id!r}"
            raise KeyError(msg) from None

    def aligned_input(self, pair_id: GeneratedPairId, /) -> AlignedInput:
        """Return the exact tokenizer evidence retained for one accepted pair.

        Raises
        ------
        KeyError
            If no accepted pair has that identity.
        """
        try:
            return self._aligned_inputs[pair_id]
        except KeyError:
            msg = f"counterfactual generation retains no evidence for {pair_id!r}"
            raise KeyError(msg) from None

    @property
    def refusals(self) -> tuple[AlignmentRefusal, ...]:
        """Return every tokenizer-specific rejection in generated order."""
        return self._refusals

    @property
    def exclusions(self) -> tuple[AlignmentExclusion, ...]:
        """Return every sentence a tokenizer could not align, in sentence order."""
        return self._exclusions

    @property
    def manifest(self) -> AlignmentManifest:
        """Return the policy, tokenizer order and population counts."""
        return self._manifest

    @override
    def __repr__(self) -> str:
        """Return the accepted and refused counts with the manifest."""
        return (
            f"CounterfactualGeneration(pairs={len(self._pairs)!r}, "
            f"refusals={len(self._refusals)!r}, manifest={self._manifest!r})"
        )


def confirm_pair(record: GeneratedPair) -> CounterfactualPair:
    """Return the counterfactual pair a generated record turns out to hold.

    The record is read on its own terms. The difference between its two
    assignments is recomputed here rather than taken from the variable the record
    claims to vary, the claim is held to what the recomputation found, and only
    then is the pair built.

    Parameters
    ----------
    record : GeneratedPair
        The record to confirm.

    Returns
    -------
    CounterfactualPair
        The two assignments, the variable they differ at and the cell an
        intervention targets.

    Raises
    ------
    TypeError
        If the record is not a generated pair.
    ValueError
        If the two assignments decide different variables, differ at other than
        one of them, differ at a variable that is not the category at a
        one-token cell, or disagree with the lexical categories or with the
        variable the record claims.
    """
    if type(record) is not GeneratedPair:
        msg = f"a confirmation reads a GeneratedPair, not {record!r}"
        raise TypeError(msg)

    base = dict(record.base_assignment)
    variant = dict(record.variant_assignment)

    if set(base) != set(variant):
        msg = "a record's two assignments decide different variables"
        raise ValueError(msg)

    differing = tuple(
        variable for variable, decided in base.items() if variant[variable] != decided
    )

    if len(differing) != 1:
        msg = (
            f"a record's two assignments differ at {len(differing)} variables, "
            f"so no outcome can be attributed to any one of them"
        )
        raise ValueError(msg)

    varied = differing[0]
    if isinstance(varied, NamedVariable):
        msg = "a generated structural record does not admit NamedVariable"
        raise TypeError(msg)
    start, end = varied.cell

    if varied.kind != "type" or end - start != 1:
        msg = (
            f"a record varies {varied!r}, which is not the category at a cell "
            f"one lexical item covers"
        )
        raise ValueError(msg)

    if varied != record.varied:
        msg = f"a record claims to vary {record.varied!r} but varies {varied!r}"
        raise ValueError(msg)

    positions = _differing(record.base, record.variant)

    if positions != (start,):
        named = ", ".join(str(position) for position in positions) or "no position"
        msg = (
            f"a record varies the category at position {start} but its lexical "
            f"assignments differ at {named}"
        )
        raise ValueError(msg)

    if base[varied] != record.base[start] or variant[varied] != record.variant[start]:
        msg = (
            f"a record decides {base[varied]!r} against {variant[varied]!r} at "
            f"{varied.cell!r}, which is not the pair of categories "
            f"{record.base[start]!r} and {record.variant[start]!r} its lexical "
            f"assignments give the form {record.form!r}"
        )
        raise ValueError(msg)

    return CounterfactualPair(base, variant, varied, varied.cell, record.provenance)


def _model_inputs(field: str, inputs: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
    """Return a detached snapshot of one side's named model inputs."""
    if not isinstance(inputs, Mapping):
        msg = f"{field} is a mapping of named model inputs, not {inputs!r}"
        raise TypeError(msg)

    if not inputs:
        msg = f"{field} is a nonempty mapping of named model inputs"
        raise ValueError(msg)

    held: dict[str, Tensor] = {}

    for name, tensor in inputs.items():
        named = ensure_name(f"a name in {field}", name)

        if not isinstance(tensor, torch.Tensor):
            msg = f"model input {named!r} in {field} is not a tensor: {tensor!r}"
            raise TypeError(msg)

        if tensor.ndim == 0 or int(tensor.shape[0]) != 1:
            msg = (
                f"model input {named!r} in {field} has shape {tuple(tensor.shape)}, "
                "not one leading item row"
            )
            raise ValueError(msg)

        if not bool(torch.isfinite(tensor).all().item()):
            msg = f"model input {named!r} in {field} contains a nonfinite value"
            raise ValueError(msg)

        copied = tensor.detach().clone()

        if tensor.requires_grad:
            copied.requires_grad_(tensor.requires_grad)

        held[named] = copied

    return MappingProxyType(held)


def _source_tokens(tokens: tuple[str, ...]) -> tuple[str, ...]:
    """Return a checked immutable source-token sequence."""
    if type(tokens) is not tuple:
        msg = f"a model causal pair's tokens are an immutable tuple, not {tokens!r}"
        raise TypeError(msg)

    if not tokens:
        msg = "a model causal pair has at least one source token"
        raise ValueError(msg)

    return tuple(
        ensure_name(f"source token {position}", token)
        for position, token in enumerate(tokens)
    )


def _alignment_for(
    tokens: tuple[str, ...], alignment: TokenizerAlignment, field: str
) -> TokenizerAlignment:
    """Return one side's alignment after matching its source-token count."""
    if type(alignment) is not TokenizerAlignment:
        msg = f"{field} is a TokenizerAlignment, not {alignment!r}"
        raise TypeError(msg)

    if len(alignment.token_rows) != len(tokens):
        msg = (
            f"{field} maps {len(alignment.token_rows)} source tokens, not the "
            f"{len(tokens)} tokens of the model causal pair"
        )
        raise ValueError(msg)

    return alignment


class ModelCausalPair(FrozenValue):
    """One causal pair with exact inputs for both model passes.

    Parameters
    ----------
    identity : str
        Stable identity of the model-ready pair.
    pair : CounterfactualPair
        Confirmed assignments varying one type, split, combinator or named variable.
    tokens : tuple[str, ...]
        Exact source tokens shared by both assignments.
    policy : AlignmentPolicy
        Exact policy used to select model rows from source spans.
    base_inputs : Mapping[str, Tensor]
        Keyword tensors for the model's base-side pass, each holding one item.
    variant_inputs : Mapping[str, Tensor]
        Keyword tensors for the model's variant-side pass, under the same keys and
        tensor contract as the base side.
    base_alignment : TokenizerAlignment
        Exact tokenizer rows for the base-side input.
    variant_alignment : TokenizerAlignment
        Exact tokenizer rows for the variant-side input.

    Raises
    ------
    TypeError
        If a value has the wrong runtime type or an input is not a tensor.
    ValueError
        If the model inputs are empty, malformed, nonfinite, incompatible or
        identical, if the pair runs past the tokens, or if the per-side alignments
        disagree with the tokens, tokenizer identity or selected-row extent.
    """

    __slots__ = (
        "_base_inputs",
        "_base_live_rows",
        "_base_rows",
        "_pair",
        "_pair_id",
        "_policy",
        "_tokenizer",
        "_tokens",
        "_variant_inputs",
        "_variant_live_rows",
        "_variant_rows",
    )

    _noun = "model causal pair"

    def __init__(
        self,
        identity: str,
        pair: CounterfactualPair,
        tokens: tuple[str, ...],
        policy: AlignmentPolicy,
        *,
        base_inputs: Mapping[str, Tensor],
        variant_inputs: Mapping[str, Tensor],
        base_alignment: TokenizerAlignment,
        variant_alignment: TokenizerAlignment,
    ) -> None:
        pair_id = ensure_name("a model causal pair's identity", identity)

        if type(pair) is not CounterfactualPair:
            msg = f"a model causal pair requires a CounterfactualPair, not {pair!r}"
            raise TypeError(msg)

        if type(policy) is not AlignmentPolicy:
            msg = f"a model causal pair requires an AlignmentPolicy, not {policy!r}"
            raise TypeError(msg)

        held_tokens = _source_tokens(tokens)

        if pair.span[1] > len(held_tokens):
            msg = (
                f"a model causal pair targets source span {pair.span!r} past its "
                f"{len(held_tokens)} tokens"
            )
            raise ValueError(msg)

        base_tokens = _alignment_for(held_tokens, base_alignment, "base alignment")
        variant_tokens = _alignment_for(
            held_tokens, variant_alignment, "variant alignment"
        )

        if base_tokens.tokenizer != variant_tokens.tokenizer:
            msg = (
                f"base tokenizer {base_tokens.tokenizer!r} and variant tokenizer "
                f"{variant_tokens.tokenizer!r} do not identify one model"
            )
            raise ValueError(msg)

        base_rows = base_tokens.model_rows(pair.span, policy)
        variant_rows = variant_tokens.model_rows(pair.span, policy)

        if not base_rows or not variant_rows:
            msg = "a model causal pair selects at least one tokenizer row on each side"
            raise ValueError(msg)

        if len(base_rows) != len(variant_rows):
            msg = (
                f"base and variant alignments select {len(base_rows)} and "
                f"{len(variant_rows)} rows, so they define no paired observation"
            )
            raise ValueError(msg)

        base_model = _model_inputs("base inputs", base_inputs)
        variant_model = _model_inputs("variant inputs", variant_inputs)

        if set(base_model) != set(variant_model):
            msg = (
                "base and variant model inputs name different keys: "
                f"{tuple(base_model)!r} and {tuple(variant_model)!r}"
            )
            raise ValueError(msg)

        differs = False

        for name, base_tensor in base_model.items():
            variant_tensor = variant_model[name]

            if base_tensor.dtype != variant_tensor.dtype:
                msg = (
                    f"model input {name!r} has dtype {base_tensor.dtype} on the "
                    f"base side and {variant_tensor.dtype} on the variant side"
                )
                raise ValueError(msg)

            if base_tensor.device != variant_tensor.device:
                msg = (
                    f"model input {name!r} is on {base_tensor.device} on the base "
                    f"side and {variant_tensor.device} on the variant side"
                )
                raise ValueError(msg)

            if tuple(base_tensor.shape[1:]) != tuple(variant_tensor.shape[1:]):
                msg = (
                    f"model input {name!r} has nonbatch shape "
                    f"{tuple(base_tensor.shape[1:])} on the base side and "
                    f"{tuple(variant_tensor.shape[1:])} on the variant side"
                )
                raise ValueError(msg)

            differs = differs or not torch.equal(base_tensor, variant_tensor)

        if not differs:
            msg = "base and variant model inputs are identical"
            raise ValueError(msg)

        object.__setattr__(self, "_pair_id", pair_id)
        object.__setattr__(self, "_tokens", held_tokens)
        object.__setattr__(self, "_pair", pair)
        object.__setattr__(self, "_policy", policy)
        object.__setattr__(self, "_tokenizer", base_tokens.tokenizer)
        object.__setattr__(self, "_base_inputs", base_model)
        object.__setattr__(self, "_variant_inputs", variant_model)
        object.__setattr__(self, "_base_rows", base_rows)
        object.__setattr__(self, "_variant_rows", variant_rows)
        object.__setattr__(self, "_base_live_rows", base_tokens.live_rows)
        object.__setattr__(self, "_variant_live_rows", variant_tokens.live_rows)

    @property
    def id(self) -> str:
        """Return the pair's stable identity."""
        return self._pair_id

    @property
    def tokens(self) -> tuple[str, ...]:
        """Return the source tokens shared by both sides."""
        return self._tokens

    @property
    def base(self) -> CellAssignment:
        """Return what the base side decided."""
        return self._pair.base

    @property
    def variant(self) -> CellAssignment:
        """Return what the counterfactual side decided."""
        return self._pair.variant

    @property
    def varied(self) -> HighLevelVariable:
        """Return the one decision on which the assignments differ."""
        return self._pair.varied

    @property
    def span(self) -> Cell:
        """Return the source-token span selected for intervention."""
        return self._pair.span

    @property
    def provenance(self) -> Provenance:
        """Return what generated the pair."""
        return self._pair.provenance

    @property
    def policy(self) -> AlignmentPolicy:
        """Return the policy under which tokenizer rows were selected."""
        return self._policy

    @property
    def tokenizer(self) -> str:
        """Return the tokenizer identity shared by both model sides."""
        return self._tokenizer

    def inputs(self, side: Side, /) -> Mapping[str, Tensor]:
        """Return a fresh snapshot of one side's named model inputs."""
        if not isinstance(side, Side):
            msg = f"model inputs require a Side, not {side!r}"
            raise TypeError(msg)

        held = self._base_inputs if side is Side.BASE else self._variant_inputs
        copied: dict[str, Tensor] = {}

        for name, tensor in held.items():
            value = tensor.detach().clone()

            if tensor.requires_grad:
                value.requires_grad_(tensor.requires_grad)

            copied[name] = value

        return MappingProxyType(copied)

    def rows(self, side: Side, /) -> tuple[int, ...]:
        """Return exact model rows selected for one side."""
        if not isinstance(side, Side):
            msg = f"model rows require a Side, not {side!r}"
            raise TypeError(msg)

        return self._base_rows if side is Side.BASE else self._variant_rows

    def live_rows(self, side: Side, /) -> int:
        """Return the live-row extent of one side's tokenizer input."""
        if not isinstance(side, Side):
            msg = f"a live-row extent requires a Side, not {side!r}"
            raise TypeError(msg)

        return self._base_live_rows if side is Side.BASE else self._variant_live_rows

    @override
    def __repr__(self) -> str:
        """Return the identity, tokenizer, varied decision and selected rows."""
        return (
            f"ModelCausalPair({self._pair_id!r}, tokenizer={self._tokenizer!r}, "
            f"varied={self.varied!r}, base_rows={self._base_rows!r}, "
            f"variant_rows={self._variant_rows!r})"
        )


class ModelJointCausalPair(FrozenValue):
    """One joint structural pair with exact model inputs and per-variable rows."""

    __slots__ = (
        "_base_inputs",
        "_base_live_rows",
        "_base_rows",
        "_pair",
        "_pair_id",
        "_policy",
        "_tokenizer",
        "_tokens",
        "_variant_inputs",
        "_variant_live_rows",
        "_variant_rows",
    )

    _noun = "model joint causal pair"

    def __init__(
        self,
        identity: str,
        pair: JointCounterfactualPair,
        tokens: tuple[str, ...],
        policy: AlignmentPolicy,
        *,
        base_inputs: Mapping[str, Tensor],
        variant_inputs: Mapping[str, Tensor],
        base_alignment: TokenizerAlignment,
        variant_alignment: TokenizerAlignment,
    ) -> None:
        pair_id = ensure_name("a model joint pair's identity", identity)
        if type(pair) is not JointCounterfactualPair:
            msg = f"a model joint pair requires a JointCounterfactualPair, not {pair!r}"
            raise TypeError(msg)
        if type(policy) is not AlignmentPolicy:
            msg = f"a model joint pair requires an AlignmentPolicy, not {policy!r}"
            raise TypeError(msg)

        held_tokens = _source_tokens(tokens)
        if pair.span[1] > len(held_tokens):
            msg = (
                f"a model joint pair targets source span {pair.span!r} past its "
                f"{len(held_tokens)} tokens"
            )
            raise ValueError(msg)
        base_tokens = _alignment_for(held_tokens, base_alignment, "base alignment")
        variant_tokens = _alignment_for(
            held_tokens, variant_alignment, "variant alignment"
        )
        if base_tokens.tokenizer != variant_tokens.tokenizer:
            msg = (
                f"base tokenizer {base_tokens.tokenizer!r} and variant tokenizer "
                f"{variant_tokens.tokenizer!r} do not identify one model"
            )
            raise ValueError(msg)

        base_rows: dict[HighLevelVariable, tuple[int, ...]] = {}
        variant_rows: dict[HighLevelVariable, tuple[int, ...]] = {}
        for variable in pair.variables:
            if isinstance(variable, NamedVariable):
                msg = "a model joint pair does not admit NamedVariable"
                raise TypeError(msg)
            on_base = base_tokens.model_rows(variable.cell, policy)
            on_variant = variant_tokens.model_rows(variable.cell, policy)
            if not on_base or not on_variant:
                msg = (
                    f"joint variable {variable!r} selects at least one tokenizer "
                    "row on each side"
                )
                raise ValueError(msg)
            if len(on_base) != len(on_variant):
                msg = (
                    f"joint variable {variable!r} selects {len(on_base)} base rows "
                    f"and {len(on_variant)} variant rows"
                )
                raise ValueError(msg)
            base_rows[variable] = on_base
            variant_rows[variable] = on_variant

        base_union = {row for rows in base_rows.values() for row in rows}
        variant_union = {row for rows in variant_rows.values() for row in rows}
        if len(base_union) != len(variant_union):
            msg = (
                "base and variant joint alignments select different row-union "
                f"extents, {len(base_union)} and {len(variant_union)}"
            )
            raise ValueError(msg)

        base_model = _model_inputs("base inputs", base_inputs)
        variant_model = _model_inputs("variant inputs", variant_inputs)
        if set(base_model) != set(variant_model):
            msg = (
                "base and variant model inputs name different keys: "
                f"{tuple(base_model)!r} and {tuple(variant_model)!r}"
            )
            raise ValueError(msg)

        differs = False
        for name, base_tensor in base_model.items():
            variant_tensor = variant_model[name]
            if (
                base_tensor.dtype != variant_tensor.dtype
                or base_tensor.device != variant_tensor.device
                or tuple(base_tensor.shape[1:]) != tuple(variant_tensor.shape[1:])
            ):
                msg = f"model input {name!r} has incompatible base and variant tensors"
                raise ValueError(msg)
            differs = differs or not torch.equal(base_tensor, variant_tensor)
        if not differs:
            msg = "base and variant model inputs are identical"
            raise ValueError(msg)

        object.__setattr__(self, "_pair_id", pair_id)
        object.__setattr__(self, "_pair", pair)
        object.__setattr__(self, "_tokens", held_tokens)
        object.__setattr__(self, "_policy", policy)
        object.__setattr__(self, "_tokenizer", base_tokens.tokenizer)
        object.__setattr__(self, "_base_inputs", base_model)
        object.__setattr__(self, "_variant_inputs", variant_model)
        object.__setattr__(self, "_base_rows", MappingProxyType(base_rows))
        object.__setattr__(self, "_variant_rows", MappingProxyType(variant_rows))
        object.__setattr__(self, "_base_live_rows", base_tokens.live_rows)
        object.__setattr__(self, "_variant_live_rows", variant_tokens.live_rows)

    @property
    def id(self) -> str:
        """Return the pair's stable identity."""
        return self._pair_id

    @property
    def tokens(self) -> tuple[str, ...]:
        """Return the source tokens shared by both derivations."""
        return self._tokens

    @property
    def base(self) -> CellAssignment:
        """Return what the base derivation decided."""
        return self._pair.base

    @property
    def variant(self) -> CellAssignment:
        """Return what the joint counterfactual derivation decided."""
        return self._pair.variant

    @property
    def variables(self) -> tuple[HighLevelVariable, HighLevelVariable]:
        """Return the dominant and dominated changed variables."""
        return self._pair.variables

    @property
    def span(self) -> Cell:
        """Return the source-token region containing both variables."""
        return self._pair.span

    @property
    def provenance(self) -> Provenance:
        """Return what generated the structural pair."""
        return self._pair.provenance

    @property
    def policy(self) -> AlignmentPolicy:
        """Return the policy under which tokenizer rows were selected."""
        return self._policy

    @property
    def tokenizer(self) -> str:
        """Return the tokenizer identity shared by both sides."""
        return self._tokenizer

    def inputs(self, side: Side, /) -> Mapping[str, Tensor]:
        """Return a fresh snapshot of one side's named model inputs."""
        if not isinstance(side, Side):
            msg = f"joint model inputs require a Side, not {side!r}"
            raise TypeError(msg)
        held = self._base_inputs if side is Side.BASE else self._variant_inputs
        copied: dict[str, Tensor] = {}
        for name, tensor in held.items():
            value = tensor.detach().clone()
            if tensor.requires_grad:
                value.requires_grad_(tensor.requires_grad)
            copied[name] = value
        return MappingProxyType(copied)

    def rows(self, side: Side, variable: HighLevelVariable, /) -> tuple[int, ...]:
        """Return exact model rows selected for one variable on one side."""
        if not isinstance(side, Side):
            msg = f"joint model rows require a Side, not {side!r}"
            raise TypeError(msg)
        if variable not in self._pair.variables:
            msg = f"{variable!r} is not one of this joint pair's variables"
            raise ValueError(msg)
        selected = self._base_rows if side is Side.BASE else self._variant_rows
        return selected[variable]

    def selected_rows(self, side: Side, /) -> tuple[int, ...]:
        """Return the ordered union of both variables' selected rows."""
        rows = {row for variable in self.variables for row in self.rows(side, variable)}
        return tuple(sorted(rows))

    def live_rows(self, side: Side, /) -> int:
        """Return the live-row extent of one side's tokenizer input."""
        if not isinstance(side, Side):
            msg = f"a live-row extent requires a Side, not {side!r}"
            raise TypeError(msg)
        return self._base_live_rows if side is Side.BASE else self._variant_live_rows

    @override
    def __repr__(self) -> str:
        """Return the identity, tokenizer, variables, and selected rows."""
        return (
            f"ModelJointCausalPair({self._pair_id!r}, "
            f"tokenizer={self._tokenizer!r}, variables={self.variables!r})"
        )


class _Derivable[T, W]:
    """Reports every entry another source admits as derivable.

    Parameters
    ----------
    source : LexicalSource[T, W]
        The source whose entries are reweighted.
    """

    __slots__ = ("_source",)

    def __init__(self, source: LexicalSource[T, W]) -> None:
        self._source: LexicalSource[T, W] = source

    @property
    def is_re_callable(self) -> bool:
        """Return whether the enumeration may be read more than once."""
        return self._source.is_re_callable

    @property
    def clamps(self) -> tuple[Clamp[T], ...]:
        """Return the clamps the wrapped source enforces."""
        return self._source.clamps

    def axioms(
        self, tokens: Sequence[str], /
    ) -> tuple[tuple[int, Entry[T], Truth], ...]:
        """Return the wrapped source's entries, each weighted derivable.

        Parameters
        ----------
        tokens : Sequence[str]
            Surface tokens in position order.

        Returns
        -------
        tuple[tuple[int, Entry[T], Truth], ...]
            Position, entry and truth for each admitted axiom.
        """
        return tuple(
            (position, entry, True)
            for position, entry, _ in self._source.axioms(tokens)
        )

    @override
    def __repr__(self) -> str:
        """Return the source whose entries it reweights."""
        return f"_Derivable({self._source!r})"


class _Reachable[T]:
    """Weights every axiom and every rule application as derivable."""

    __slots__ = ()

    @property
    def lowering(self) -> SpanFree:
        """Declare that every rule receives the same local weight."""
        return SpanFree()

    def axiom_weight(self, position: int, entry: Entry[T], /) -> Truth:
        """Return that a lexical axiom stands.

        Parameters
        ----------
        position : int
            The index of the token the entry was drawn for.
        entry : Entry[T]
            The entry admitted there.

        Returns
        -------
        Truth
            True, since the generator reads structure rather than weight.
        """
        del position, entry

        return True

    def edge_weight(
        self,
        rule: Rule[T],
        inputs: tuple[T, ...],
        output: T,
        span: Cell,
        split: int | None,
        /,
    ) -> Truth:
        """Return that a rule application stands.

        Parameters
        ----------
        rule : Rule[T]
            The rule that fired.
        inputs : tuple[T, ...]
            The premises, in order.
        output : T
            The conclusion drawn.
        span : Cell
            The half-open token interval the conclusion covers.
        split : int | None
            Binary split point, or None for another arity.

        Returns
        -------
        Truth
            True, since the generator reads structure rather than weight.
        """
        del rule, inputs, output, span, split

        return True

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this scorer."""
        return "_Reachable()"


def _alignment_refusal_reason(
    pair: GeneratedPair,
    alignment: TokenizerAlignment,
    policy: AlignmentPolicy,
) -> AlignmentRefusalReason | None:
    """Return why one tokenizer rejects a pair, or none when it accepts.

    Parameters
    ----------
    pair : GeneratedPair
        Candidate whose varied source span is checked.
    alignment : TokenizerAlignment
        Tokenizer evidence checked against the policy.
    policy : AlignmentPolicy
        Piece, special-row and contiguity rules to enforce.
    """
    start, end = pair.span
    groups = alignment.token_rows[start:end]

    if not groups:
        return "incomplete-source-span"

    if policy.pieces == "all":
        source_rows = tuple(row for group in groups for row in group)
    elif policy.pieces == "first":
        source_rows = tuple(group[0] for group in groups)
    else:
        source_rows = tuple(group[-1] for group in groups)

    if not source_rows:
        return "empty-model-row-span"

    rows = alignment.model_rows(pair.span, policy)

    if rows != tuple(sorted(set(rows))):
        return "unordered-model-row-span"

    if policy.contiguous and rows != tuple(range(rows[0], rows[-1] + 1)):
        return "noncontiguous-model-row-span"

    return None


class CounterfactualGenerator[T, W]:
    """Generates interchange-intervention pairs from a grammar and a lexicon.

    Parameters
    ----------
    rules : RuleSet[T]
        The rules both derivations of every pair are built under.
    source : LexicalSource[T, W]
        The lexicon supplying the categories each form may bear.
    goal : Goal[T]
        The goal a derivation of the whole string reaches.
    render : Callable[[T], str]
        How a category is written as the decision at its cell.
    grammar : str
        The fingerprint of the declared grammar, recorded on every pair.
    parse_policy : ParsePolicy[Literal[Retain.FOREST]]
        The run's declared packed-forest reachability policy.
    readings : int
        How many derivations of one string the generator reads before it stops.
    threads : int
        How many chart cells of one width the parse fills concurrently, the
        grammar budget's declared thread count.
    proposer : Proposer[T] | None
        Candidate conclusions for check-only rules, or None where the rules need
        none.
    normalizer : Normalizer[T] | None
        The normal form rule outputs take before insertion, or None.

    Raises
    ------
    ValueError
        If the fingerprint is empty or fewer than two readings are budgeted.
    """

    __slots__ = (
        "_goal",
        "_grammar",
        "_normalizer",
        "_parse_policy",
        "_proposer",
        "_readings",
        "_render",
        "_rules",
        "_source",
        "_threads",
        "_unary_depth",
    )

    def __init__(
        self,
        rules: RuleSet[T],
        source: LexicalSource[T, W],
        goal: Goal[T],
        render: Callable[[T], str],
        grammar: str,
        unary_depth: int,
        parse_policy: ParsePolicy[Literal[Retain.FOREST]],
        readings: int = 64,
        threads: int = 1,
        proposer: Proposer[T] | None = None,
        normalizer: Normalizer[T] | None = None,
    ) -> None:
        budget = ensure_non_negative("a generator's readings", readings)

        if budget < 2:
            msg = (
                f"a pair needs two derivations of one string, so a generator "
                f"reading {budget} of them generates nothing"
            )
            raise ValueError(msg)

        self._rules: RuleSet[T] = rules
        self._source: LexicalSource[T, W] = source
        self._goal: Goal[T] = goal
        self._render: Callable[[T], str] = render
        self._grammar = ensure_name("a generator's grammar", grammar)
        self._unary_depth = ensure_non_negative("a unary closure depth", unary_depth)
        self._parse_policy = parse_policy
        self._readings = budget
        self._threads = threads
        self._proposer: Proposer[T] | None = proposer
        self._normalizer: Normalizer[T] | None = normalizer

    @property
    def grammar(self) -> str:
        """Return the fingerprint recorded on every pair it generates."""
        return self._grammar

    @property
    def readings(self) -> int:
        """Return how many derivations of one string it reads."""
        return self._readings

    def generate(
        self,
        inputs: Iterable[AlignedInput],
        *,
        policy: AlignmentPolicy,
        dataset: str,
        seed: int,
        exclusions: Sequence[AlignmentExclusion] = (),
    ) -> CounterfactualGeneration:
        """Return pairs whose varied spans every supplied alignment admits.

        Parameters
        ----------
        inputs : Iterable[AlignedInput]
            Source-token strings and caller-supplied tokenizer evidence.
        policy : AlignmentPolicy
            Rules every tokenizer alignment applies.
        dataset : str
            The handle of the data the strings came from.
        seed : int
            The seed the order of one string's pairs is drawn under.
        exclusions : Sequence[AlignmentExclusion]
            The sentences of the same population some tokenizer could not align
            whole, carried into the generation's report.

        Returns
        -------
        CounterfactualGeneration
            Accepted pairs, tokenizer-specific refusals and exact population
            counts.

        Raises
        ------
        TypeError
            If the policy or an input has the wrong runtime type.
        ValueError
            If the dataset handle is empty, the seed is negative or inputs name
            different tokenizer orders.
        """
        if type(policy) is not AlignmentPolicy:
            msg = (
                f"counterfactual generation requires an AlignmentPolicy, not {policy!r}"
            )
            raise TypeError(msg)

        handle = ensure_name("a generated pair's dataset", dataset)
        drawn = ensure_non_negative("a generator's seed", seed)
        provenance = Provenance("grammar-counterfactual", handle, drawn)
        order = Random(drawn)
        tokenizers: tuple[str, ...] | None = None
        accepted: list[GeneratedPair] = []
        accepted_inputs: dict[GeneratedPairId, AlignedInput] = {}
        refusals: list[AlignmentRefusal] = []
        candidate_count = 0

        for aligned in inputs:
            if type(aligned) is not AlignedInput:
                msg = (
                    "counterfactual generation consumes AlignedInput values, "
                    f"not {aligned!r}"
                )
                raise TypeError(msg)

            if tokenizers is None:
                tokenizers = aligned.tokenizers
            elif aligned.tokenizers != tokenizers:
                msg = (
                    f"aligned inputs name tokenizer order {aligned.tokenizers!r}, "
                    f"not {tokenizers!r}"
                )
                raise ValueError(msg)

            found = self._sentence_pairs(aligned.tokens, handle, provenance)
            order.shuffle(found)
            candidate_count += len(found)

            for pair in found:
                confirmed = confirm_pair(pair)

                if confirmed.varied != pair.varied:
                    msg = (
                        f"confirmed pair {pair.id!r} varies {confirmed.varied!r}, "
                        f"not {pair.varied!r}"
                    )
                    raise ValueError(msg)

                rejected = False

                for alignment in aligned.alignments:
                    reason = _alignment_refusal_reason(pair, alignment, policy)

                    if reason is None:
                        continue

                    refusals.append(
                        AlignmentRefusal(
                            pair.id,
                            alignment.tokenizer,
                            pair.span,
                            reason,
                        )
                    )
                    rejected = True

                if not rejected:
                    if pair.id in accepted_inputs:
                        msg = (
                            f"generated pair identity {pair.id!r} repeats across "
                            "the supplied aligned inputs"
                        )
                        raise ValueError(msg)

                    accepted.append(pair)
                    accepted_inputs[pair.id] = aligned

        manifest = AlignmentManifest(
            policy,
            () if tokenizers is None else tokenizers,
            candidate_count,
            len(accepted),
        )

        return CounterfactualGeneration(
            tuple(accepted),
            accepted_inputs,
            tuple(refusals),
            manifest,
            exclusions=tuple(exclusions),
        )

    def _sentence_pairs(
        self, tokens: tuple[str, ...], dataset: str, provenance: Provenance
    ) -> list[GeneratedPair]:
        """Return every pair one string admits.

        Parameters
        ----------
        tokens : tuple[str, ...]
            The string to read derivations of.
        dataset : str
            The handle of the data the string came from.
        provenance : Provenance
            What every pair from this run records.

        Returns
        -------
        list[GeneratedPair]
            The pairs, in the order the readings were enumerated.
        """
        readings = self._readings_of(tokens, dataset)

        if len(readings) < 2:
            return []

        buckets: dict[tuple[HighLevelVariable, _Decisions], list[int]] = {}

        for index, (_, assignment) in enumerate(readings):
            decided = _decisions(assignment)

            for removed, (variable, _) in enumerate(decided):
                blank = (variable, (*decided[:removed], *decided[removed + 1 :]))
                buckets.setdefault(blank, []).append(index)

        found: list[GeneratedPair] = []

        for members in buckets.values():
            for rank, index in enumerate(members):
                for other in members[rank + 1 :]:
                    made = self._made(
                        tokens, readings[index], readings[other], provenance
                    )

                    if made is not None:
                        found.append(made)

        return found

    def _readings_of(self, tokens: tuple[str, ...], dataset: str) -> list[_Reading]:
        """Return the derivations of one string, up to the reading budget.

        Parameters
        ----------
        tokens : tuple[str, ...]
            The string to parse.
        dataset : str
            The handle of the data the string came from.

        Returns
        -------
        list[_Reading]
            One lexical assignment and one variable assignment per derivation.
        """
        forest: PackedForest[T, Truth] = parse(
            tokens,
            rules=self._rules,
            source=_Derivable(self._source),
            semiring=Boolean(),
            scorer=_Reachable[T](),
            goal=self._goal,
            proposer=self._proposer,
            normalizer=self._normalizer,
            unary_depth=self._unary_depth,
            threads=self._threads,
            policy=self._parse_policy,
        )
        trees = chain.from_iterable(
            forest.derivations(root) for root in self._goal.roots(forest)
        )
        read = (
            self._read(tokens, derivation, dataset)
            for derivation in islice(trees, self._readings)
        )

        return [found for found in read if found is not None]

    def _read(
        self, tokens: tuple[str, ...], derivation: Derivation[T, Truth], dataset: str
    ) -> _Reading | None:
        """Return one derivation's lexical and variable assignments.

        Parameters
        ----------
        tokens : tuple[str, ...]
            The string the derivation covers.
        derivation : Derivation[T, Truth]
            The derivation to read.
        dataset : str
            The handle of the data the string came from.

        Returns
        -------
        _Reading | None
            The category each token bears and what the derivation decided, or
            None where some token bears no category of its own.
        """
        lexical: dict[int, str] = {}
        pending = [derivation]

        while pending:
            node = pending.pop()

            if node.children:
                pending.extend(node.children)
                continue

            lexical[node.key.span[0]] = self._render(node.edge.output)

        if len(lexical) != len(tokens):
            return None

        assignment = extract_variables(
            derivation,
            self._rules.normal_form,
            render=self._render,
            dataset=dataset,
            arm="unconstrained",
        ).assignment

        return (tuple(lexical[position] for position in range(len(tokens))), assignment)

    def _made(
        self,
        tokens: tuple[str, ...],
        base: _Reading,
        variant: _Reading,
        provenance: Provenance,
    ) -> GeneratedPair | None:
        """Return the pair two readings of one string make, where they make one.

        Parameters
        ----------
        tokens : tuple[str, ...]
            The string both readings cover.
        base : _Reading
            The reading the intervention runs on.
        variant : _Reading
            The reading the intervened value is taken from.
        provenance : Provenance
            What the pair records as having generated it.

        Returns
        -------
        GeneratedPair | None
            The pair, or None where the two differ at other than the category of
            a single lexical item.
        """
        base_lexical, base_assignment = base
        variant_lexical, variant_assignment = variant
        positions = _differing(base_lexical, variant_lexical)

        if len(positions) != 1:
            return None

        at = positions[0]
        cell = (at, at + 1)
        varied = TypeAt(cell)

        for assignment, lexical in (
            (base_assignment, base_lexical),
            (variant_assignment, variant_lexical),
        ):
            if assignment.get(varied) != lexical[at]:
                return None

        differing = tuple(
            variable
            for variable, decided in base_assignment.items()
            if variant_assignment[variable] != decided
        )

        if differing != (varied,):
            return None

        return GeneratedPair(
            tokens,
            base_lexical,
            variant_lexical,
            base_assignment,
            variant_assignment,
            varied,
            self._grammar,
            provenance,
        )

    @override
    def __repr__(self) -> str:
        """Return the rule set, the grammar fingerprint and the reading budget."""
        return (
            f"CounterfactualGenerator(rules={self._rules.name!r}, "
            f"grammar={self._grammar!r}, readings={self._readings!r})"
        )

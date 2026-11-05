"""Type-system reconstruction metrics after label alignment."""

from collections import Counter
from math import fsum, log
from statistics import fmean
from typing import TYPE_CHECKING, NamedTuple

from lofi._checks import ensure_name
from lofi.eval._alignment import Alignment, Contingency
from lofi.eval._tree_agreement import _clone, _primitive_target
from lofi.terms import Operator, Term, TermStore, UnknownOperatorError

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

__all__ = [
    "ConstructorSetRecovery",
    "LexicalTypeRecovery",
    "PrimitiveSetRecovery",
    "TypeSystemReconstruction",
    "constructor_set_recovery",
    "lexical_type_recovery",
    "primitive_set_recovery",
    "type_system_reconstruction",
]


class PrimitiveSetRecovery(NamedTuple):
    """Primitive inventory recovery after alignment.

    Attributes
    ----------
    induced_size : int
        Number of induced primitives.
    gold_size : int
        Number of gold primitives.
    exact_set_match : bool
        Whether every induced primitive maps onto the gold inventory exactly.
    per_primitive_f1 : tuple[tuple[str, float], ...]
        Inventory F1 for each gold primitive.
    """

    induced_size: int
    gold_size: int
    exact_set_match: bool
    per_primitive_f1: tuple[tuple[str, float], ...]


class ConstructorSetRecovery(NamedTuple):
    """Aligned constructor inventory and declared-role recovery.

    Attributes
    ----------
    induced_size : int
        Number of induced constructors.
    gold_size : int
        Number of gold constructors.
    exact_set_match : bool
        Whether the constructor inventories match.
    role_mutual_information : float
        Mutual information between induced constructors and declared roles, in nats.
    """

    induced_size: int
    gold_size: int
    exact_set_match: bool
    role_mutual_information: float


class LexicalTypeRecovery(NamedTuple):
    """Lexical type recovery under ambiguity.

    Attributes
    ----------
    words : int
        Number of scored word types.
    tokens : int
        Number of scored tokens.
    type_f1 : float | None
        Mean word-level multiset F1.
    token_accuracy : float | None
        Exact type accuracy over token occurrences.
    per_word_f1 : tuple[tuple[str, float], ...]
        Multiset F1 for each word.
    """

    words: int
    tokens: int
    type_f1: float | None
    token_accuracy: float | None
    per_word_f1: tuple[tuple[str, float], ...]


class TypeSystemReconstruction(NamedTuple):
    """Primitive, constructor, and lexical reconstruction metrics.

    Attributes
    ----------
    primitives : PrimitiveSetRecovery
        Primitive inventory recovery.
    constructors : ConstructorSetRecovery
        Constructor inventory and declared-role recovery.
    lexicon : LexicalTypeRecovery
        Lexical type recovery.
    """

    primitives: PrimitiveSetRecovery
    constructors: ConstructorSetRecovery
    lexicon: LexicalTypeRecovery


def primitive_set_recovery(
    alignment: Alignment,
    contingency: Contingency,
    /,
) -> PrimitiveSetRecovery:
    """Measure primitive inventory recovery on held-out counts.

    Parameters
    ----------
    alignment : Alignment
        Primitive alignment fitted on a non-test split.
    contingency : Contingency
        Primitive co-occurrence counts from the scored split.

    Returns
    -------
    PrimitiveSetRecovery
        Inventory sizes, exact match, and per-primitive F1.

    Raises
    ------
    TypeError
        If either argument has the wrong value type.
    ValueError
        If the alignment was fitted on the scored split.
    """
    _held_out(alignment, contingency, subject="primitive")
    induced = contingency.induced_labels
    gold = contingency.gold_labels
    images = tuple(
        image for label in induced if (image := alignment.apply(label)) is not None
    )
    scores: list[tuple[str, float]] = []

    for label in gold:
        true_positive = sum(
            contingency.count(induced_label, label)
            for induced_label in induced
            if alignment.apply(induced_label) == label
        )
        false_positive = sum(
            contingency.count(induced_label, gold_label)
            for induced_label in induced
            if alignment.apply(induced_label) == label
            for gold_label in gold
            if gold_label != label
        )
        false_negative = sum(
            contingency.count(induced_label, label)
            for induced_label in induced
            if alignment.apply(induced_label) != label
        )
        denominator = 2 * true_positive + false_positive + false_negative
        score = 0.0 if denominator == 0 else 2.0 * true_positive / denominator
        scores.append((label, score))

    return PrimitiveSetRecovery(
        len(induced),
        len(gold),
        len(images) == len(induced)
        and len(induced) == len(gold)
        and len(set(images)) == len(gold)
        and set(images) == set(gold),
        tuple(scores),
    )


def constructor_set_recovery(
    alignment: Alignment,
    contingency: Contingency,
    role_pairs: Sequence[tuple[str, str]],
    /,
    *,
    roles: Collection[str],
) -> ConstructorSetRecovery:
    """Measure aligned constructor inventory and declared-role recovery.

    Parameters
    ----------
    alignment : Alignment
        Constructor alignment fitted on a non-test split.
    contingency : Contingency
        Constructor co-occurrence counts from the scored split.
    role_pairs : Sequence[tuple[str, str]]
        Induced constructor and declared child-role observations.
    roles : Collection[str]
        The complete declared child-role inventory.

    Returns
    -------
    ConstructorSetRecovery
        Inventory sizes, exact match, and role mutual information.

    Raises
    ------
    TypeError
        If an alignment or contingency has the wrong value type.
    ValueError
        If the alignment was fitted on the scored split or an observation is
        unknown.
    """
    _held_out(alignment, contingency, subject="constructor")
    induced = contingency.induced_labels
    gold = contingency.gold_labels
    declared_roles = frozenset(
        ensure_name(f"declared constructor role {position}", role)
        for position, role in enumerate(roles)
    )

    if not declared_roles:
        msg = "constructor recovery needs at least one declared role"
        raise ValueError(msg)

    aligned: list[tuple[str, str]] = []

    for position, (constructor, role) in enumerate(role_pairs):
        source = ensure_name(f"constructor observation {position}", constructor)
        target = ensure_name(f"role observation {position}", role)

        if source not in induced:
            msg = f"role observations contain unknown constructor {source!r}"
            raise ValueError(msg)

        if target not in declared_roles:
            msg = f"role observations contain undeclared role {target!r}"
            raise ValueError(msg)

        aligned.append((source, target))

    images = tuple(
        image for label in induced if (image := alignment.apply(label)) is not None
    )

    return ConstructorSetRecovery(
        len(induced),
        len(gold),
        len(images) == len(induced)
        and len(induced) == len(gold)
        and len(set(images)) == len(gold)
        and set(images) == set(gold),
        _mutual_information(aligned),
    )


def lexical_type_recovery(
    predicted: Mapping[str, Sequence[Term]],
    gold: Mapping[str, Sequence[Term]],
    predicted_tokens: Sequence[Term],
    gold_tokens: Sequence[Term],
    stores: tuple[TermStore, TermStore],
    *,
    primitive_alignment: Alignment,
    constructor_alignment: Alignment | None,
) -> LexicalTypeRecovery:
    """Measure aligned word-level multiset F1 and token accuracy.

    Parameters
    ----------
    predicted : Mapping[str, Sequence[Term]]
        Induced type multiset for each word.
    gold : Mapping[str, Sequence[Term]]
        Gold type multiset for each word.
    predicted_tokens : Sequence[Term]
        Induced type assigned to each token occurrence.
    gold_tokens : Sequence[Term]
        Gold type assigned to each token occurrence.
    stores : tuple[TermStore, TermStore]
        Stores that own the induced and gold terms.
    primitive_alignment : Alignment
        Frozen primitive alignment.
    constructor_alignment : Alignment | None
        Frozen constructor alignment, or ``None`` when development contained no
        constructor observation.

    Returns
    -------
    LexicalTypeRecovery
        Word-level multiset F1 and token-level exact accuracy.

    Raises
    ------
    TypeError
        If a lexical entry is not an interned term.
    ValueError
        If vocabularies differ, an alignment has the wrong label kind, or a term
        is foreign.
    """
    _alignment(primitive_alignment, subject="primitive")
    if constructor_alignment is not None:
        _alignment(constructor_alignment, subject="constructor")

    predicted_words = frozenset(predicted)
    gold_words = frozenset(gold)

    if predicted_words != gold_words:
        msg = "lexical recovery requires identical predicted and gold vocabularies"
        raise ValueError(msg)

    predicted_store, gold_store = stores
    predicted_lexicon = tuple(
        term for word in sorted(predicted) for term in predicted[word]
    )
    predicted_terms = (*predicted_lexicon, *predicted_tokens)
    gold_lexicon = tuple(term for word in sorted(gold) for term in gold[word])
    gold_terms = (*gold_lexicon, *gold_tokens)
    _validate_terms(predicted_terms, predicted_store, side="predicted")
    _validate_terms(
        gold_terms,
        gold_store,
        side="gold",
    )
    comparison = TermStore(gold_store.signature, max_nodes=gold_store.max_nodes)
    predicted_memo: dict[Term, Term] = {}
    gold_memo: dict[Term, Term] = {}
    scores: list[tuple[str, float]] = []

    for position, word in enumerate(sorted(predicted)):
        name = ensure_name(f"lexical word {position}", word)
        induced_counter = Counter(
            _aligned_clone(
                predicted_store,
                comparison,
                term,
                predicted_memo,
                primitive_alignment=primitive_alignment,
                constructor_alignment=constructor_alignment,
            )
            for term in predicted[word]
        )
        gold_counter = Counter(
            _clone(gold_store, comparison, term, None, gold_memo) for term in gold[word]
        )
        matched = sum((induced_counter & gold_counter).values())
        denominator = induced_counter.total() + gold_counter.total()
        score = 1.0 if denominator == 0 else 2.0 * matched / denominator
        scores.append((name, score))

    token_matches = sum(
        _aligned_clone(
            predicted_store,
            comparison,
            predicted_term,
            predicted_memo,
            primitive_alignment=primitive_alignment,
            constructor_alignment=constructor_alignment,
        )
        is _clone(gold_store, comparison, gold_term, None, gold_memo)
        for predicted_term, gold_term in zip(
            predicted_tokens, gold_tokens, strict=False
        )
    )
    tokens = max(len(predicted_tokens), len(gold_tokens))

    return LexicalTypeRecovery(
        len(scores),
        tokens,
        None if not scores else fmean(score for _, score in scores),
        None if tokens == 0 else token_matches / tokens,
        tuple(scores),
    )


def type_system_reconstruction(
    predicted: Mapping[str, Sequence[Term]],
    gold: Mapping[str, Sequence[Term]],
    predicted_tokens: Sequence[Term],
    gold_tokens: Sequence[Term],
    stores: tuple[TermStore, TermStore],
    *,
    primitive_alignment: Alignment,
    primitive_contingency: Contingency,
    constructor_alignment: Alignment,
    constructor_contingency: Contingency,
    role_pairs: Sequence[tuple[str, str]],
    roles: Collection[str],
) -> TypeSystemReconstruction:
    """Measure primitive, constructor, and lexical reconstruction.

    Parameters
    ----------
    predicted : Mapping[str, Sequence[Term]]
        Induced type multiset for each word.
    gold : Mapping[str, Sequence[Term]]
        Gold type multiset for each word.
    predicted_tokens : Sequence[Term]
        Induced type assigned to each token occurrence.
    gold_tokens : Sequence[Term]
        Gold type assigned to each token occurrence.
    stores : tuple[TermStore, TermStore]
        Stores that own the induced and gold terms.
    primitive_alignment : Alignment
        Primitive alignment fitted on a non-test split.
    primitive_contingency : Contingency
        Primitive counts from the scored split.
    constructor_alignment : Alignment
        Constructor alignment fitted on a non-test split.
    constructor_contingency : Contingency
        Constructor counts from the scored split.
    role_pairs : Sequence[tuple[str, str]]
        Induced constructor and declared child-role observations.
    roles : Collection[str]
        The complete declared child-role inventory.

    Returns
    -------
    TypeSystemReconstruction
        Reconstruction metrics for all three parts of the type system.

    Raises
    ------
    TypeError
        If a count table, alignment, or lexical term has the wrong type.
    ValueError
        If the primitive and constructor counts come from different splits or
        an alignment or lexical assignment is incompatible.
    """
    if not isinstance(primitive_contingency, Contingency):
        msg = (
            "primitive reconstruction counts are not a Contingency: "
            f"{primitive_contingency!r}"
        )
        raise TypeError(msg)

    if not isinstance(constructor_contingency, Contingency):
        msg = (
            "constructor reconstruction counts are not a Contingency: "
            f"{constructor_contingency!r}"
        )
        raise TypeError(msg)

    if primitive_contingency.split != constructor_contingency.split:
        msg = (
            "primitive and constructor reconstruction must score the same split, not "
            f"{primitive_contingency.split!r} and {constructor_contingency.split!r}"
        )
        raise ValueError(msg)

    return TypeSystemReconstruction(
        primitive_set_recovery(primitive_alignment, primitive_contingency),
        constructor_set_recovery(
            constructor_alignment,
            constructor_contingency,
            role_pairs,
            roles=roles,
        ),
        lexical_type_recovery(
            predicted,
            gold,
            predicted_tokens,
            gold_tokens,
            stores,
            primitive_alignment=primitive_alignment,
            constructor_alignment=constructor_alignment,
        ),
    )


def _alignment(value: Alignment, *, subject: str) -> Alignment:
    """Return a frozen label alignment.

    Parameters
    ----------
    value : Alignment
        Alignment to validate.
    subject : str
        Label kind used in errors.
    """
    if not isinstance(value, Alignment):
        msg = f"a {subject} recovery alignment is not frozen Alignment: {value!r}"
        raise TypeError(msg)

    return value


def _held_out(
    alignment: Alignment,
    contingency: Contingency,
    *,
    subject: str,
) -> None:
    """Validate a fitted alignment and held-out scoring table.

    Parameters
    ----------
    alignment : Alignment
        Fitted label alignment.
    contingency : Contingency
        Counts from the scored split.
    subject : str
        Label kind used in errors.
    """
    fitted = _alignment(alignment, subject=subject)

    if not isinstance(contingency, Contingency):
        msg = f"{subject} recovery counts are not a Contingency: {contingency!r}"
        raise TypeError(msg)

    if fitted.split == contingency.split:
        msg = f"{subject} recovery cannot score the {fitted.split!r} fitting split"
        raise ValueError(msg)


def _constructor_target(
    operator: Operator,
    gold_store: TermStore,
    alignment: Alignment,
) -> Operator:
    """Resolve one constructor through a frozen alignment.

    Parameters
    ----------
    operator : Operator
        Predicted constructor declaration.
    gold_store : TermStore
        Store containing gold declarations.
    alignment : Alignment
        Frozen constructor alignment.
    """
    image = alignment.apply(operator.name)

    if image is None:
        msg = f"constructor {operator.name!r} has no image in the supplied alignment"
        raise ValueError(msg)

    try:
        target = gold_store.signature.by_name(image)
    except UnknownOperatorError as error:
        msg = f"aligned constructor {image!r} is absent from the gold signature"
        raise ValueError(msg) from error

    source_shape = (
        operator.sort,
        operator.arg_sorts,
        operator.binding,
        operator.payload_sort,
        operator.family,
    )
    target_shape = (
        target.sort,
        target.arg_sorts,
        target.binding,
        target.payload_sort,
        target.family,
    )

    if source_shape != target_shape:
        msg = (
            f"predicted constructor {operator.name!r} is incompatible with aligned "
            f"constructor {image!r}"
        )
        raise ValueError(msg)

    return target


def _aligned_clone(
    source: TermStore,
    target: TermStore,
    term: Term,
    memo: dict[Term, Term],
    *,
    primitive_alignment: Alignment,
    constructor_alignment: Alignment | None,
) -> Term:
    """Rebuild one lexical type after applying both frozen correspondences.

    Parameters
    ----------
    source : TermStore
        Store owning the predicted term.
    target : TermStore
        Comparison store carrying the gold signature.
    term : Term
        Predicted term to rebuild.
    memo : dict[Term, Term]
        Already rebuilt subterms.
    primitive_alignment : Alignment
        Frozen primitive alignment.
    constructor_alignment : Alignment | None
        Frozen constructor alignment, or ``None`` when development contained no
        constructor observation.
    """
    held = memo.get(term)
    if held is not None:
        return held
    operator = term.op
    payload = term.payload
    if operator.arity == 0 and operator.is_constant:
        if payload is None:
            msg = f"constant operator {operator.name!r} has no payload"
            raise ValueError(msg)
        name = source.signature.vocabulary(operator.sort).name(payload)
        image = primitive_alignment.apply(name)
        if image is None:
            msg = f"primitive constant {name!r} has no image in the alignment"
            raise ValueError(msg)
        operator = target.signature.by_name(operator.name)
        payload = target.signature.vocabulary(operator.sort).id(image)
    elif operator.arity == 0:
        operator = _primitive_target(operator, (source, target), primitive_alignment)
        if payload is not None and term.op.is_free_variable:
            payload = target.atom(operator.sort, source.atom_name(payload))
    elif constructor_alignment is None:
        msg = (
            f"constructor {operator.name!r} occurred outside development and has "
            "no frozen alignment"
        )
        raise ValueError(msg)
    else:
        operator = _constructor_target(operator, target, constructor_alignment)
    arguments = tuple(
        _aligned_clone(
            source,
            target,
            child,
            memo,
            primitive_alignment=primitive_alignment,
            constructor_alignment=constructor_alignment,
        )
        for child in term.args
    )
    rebuilt = target.op(operator, *arguments, payload=payload)
    memo[term] = rebuilt
    return rebuilt


def _mutual_information(pairs: Sequence[tuple[str, str]]) -> float:
    """Return empirical mutual information in nats.

    Parameters
    ----------
    pairs : Sequence[tuple[str, str]]
        Induced constructor and declared-role observations.
    """
    total = len(pairs)

    if total == 0:
        return 0.0

    joint = Counter(pairs)
    constructors = Counter(constructor for constructor, _ in pairs)
    roles = Counter(role for _, role in pairs)

    return fsum(
        (count / total)
        * log((count * total) / (constructors[constructor] * roles[role]))
        for (constructor, role), count in joint.items()
    )


def _validate_terms(terms: Sequence[Term], store: TermStore, *, side: str) -> None:
    """Validate one side's lexical terms against its store.

    Parameters
    ----------
    terms : Sequence[Term]
        Terms to validate.
    store : TermStore
        Store expected to own the terms.
    side : str
        Side name used in errors.
    """
    for position, term in enumerate(terms):
        if not isinstance(term, Term):
            msg = f"{side} lexical entry {position} is not an interned term"
            raise TypeError(msg)

        if not store.contains(term):
            msg = f"{side} lexical entry {position} is outside its supplied store"
            raise ValueError(msg)

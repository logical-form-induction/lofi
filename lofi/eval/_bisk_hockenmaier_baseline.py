"""Registered production operation for the Bisk--Hockenmaier baseline."""

import json
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, cast, final

from lofi._types import ItemId, JsonValue
from lofi.data import AttachmentTree, Token
from lofi.data.corpora import (
    CcgbankSentenceIdentity,
    align_conll2008,
    read_conll2008,
)
from lofi.data.corpora.ccgbank._auto import CcgbankAutoCodec
from lofi.data.corpora.ccgbank._auto_parse import AutoLeaf, AutoNode, AutoTree
from lofi.data.corpora.ccgbank.roundtrip import (
    MarkedCategoryAlgebra,
    gold_sentence,
)
from lofi.data.deps import BiskHockenmaierDependencyProjector
from lofi.deduce import SpanFree, parse
from lofi.eval._bisk_hockenmaier import (
    BiskHockenmaierInducer,
    BiskHockenmaierSettings,
    BiskItem,
)
from lofi.eval._ports import (
    BaselineBuildContext,
    BaselineMeasurement,
    BaselineResult,
)
from lofi.grammars.ccg import FunctorView
from lofi.lexicon import Clamp, Entry
from lofi.spec._interpolation import lower_package_resource
from lofi.spec.grammar import GrammarIso

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from lofi.deduce import Cell, CellPruner, Chart, ParsePolicy, Retain
    from lofi.judge import Rule
    from lofi.spec.baselines.bisk_hockenmaier import BiskHockenmaierBaselineSettings
    from lofi.spec.grammar import CategoryMeasure, GrammarAssembly, GrammarDecl
    from lofi.terms import Functors, Matching, Term, TermStore

__all__ = ["BiskHockenmaierOperation", "build_bisk_hockenmaier_baseline"]

_REASON: Final = (
    "the published description is re-run because no prediction file was released"
)
_SYSTEM: Final = "bisk-hockenmaier-2012"
_REFERENCE_METRICS: Final = (
    "dependencies-length-10",
    "dependencies-length-20",
    "dependencies-all-lengths",
)


class _AutoSentence(NamedTuple):
    """One AUTO sentence with the fields used by induction and evaluation."""

    section: int
    item_id: ItemId
    forms: tuple[str, ...]
    pos: tuple[str, ...]
    tokens: tuple[Token, ...]
    gold_categories: tuple[str, ...]


class _Reference(NamedTuple):
    """Validated published dependency scores and their citation."""

    citation: str
    scores: Mapping[str, float]


class _Counts(NamedTuple):
    """Matched and eligible units for one metric."""

    matched: int
    total: int

    @property
    def value(self) -> float:
        """Return accuracy over the nonempty eligible population."""
        if self.total == 0:
            msg = "a Bisk--Hockenmaier evaluation band contains no scored unit"
            raise ValueError(msg)
        return self.matched / self.total


class _UnitScorer:
    """Assign neutral log weight to every lexical and rule edge."""

    __slots__ = ()

    @property
    def lowering(self) -> SpanFree:
        return SpanFree()

    def axiom_weight(self, position: int, entry: Entry[Term], /) -> float:
        del position, entry
        return 0.0

    def edge_weight(
        self,
        rule: Rule[Term],
        inputs: tuple[Term, ...],
        output: Term,
        span: Cell,
        split: int | None,
        /,
    ) -> float:
        del rule, inputs, output, span, split
        return 0.0


class _SentenceLexicon:
    """Expose the POS-admitted category superset at each sentence position."""

    __slots__ = ("_categories",)

    def __init__(self, categories: Sequence[Sequence[Term]]) -> None:
        self._categories = tuple(tuple(row) for row in categories)

    @property
    def is_re_callable(self) -> bool:
        return True

    @property
    def clamps(self) -> tuple[Clamp[Term], ...]:
        return ()

    def axioms(
        self, tokens: Sequence[str], /
    ) -> Iterable[tuple[int, Entry[Term], float]]:
        return (
            (position, Entry(form, category), 0.0)
            for position, (form, categories) in enumerate(
                zip(tokens, self._categories, strict=True)
            )
            for category in categories
        )


class _TermCategoryAlgebra:
    """Supply induced-dependency structure over interned CCG categories."""

    __slots__ = ("_functors", "_matching")

    def __init__(self, functors: Functors[Term], matching: Matching[Term]) -> None:
        self._functors = functors
        self._matching = matching

    def is_functor(self, category: Term, /) -> bool:
        return self._functors.is_functor(category)

    def result(self, category: Term, /) -> Term:
        return category.args[self._functors.result_path[0]]

    def argument(self, category: Term, /) -> Term:
        return category.args[self._functors.argument_path[0]]

    def equivalent(self, left: Term, right: Term, /) -> bool:
        return self._matching.equivalent(left, right)

    def shares_head(self, category: Term, /) -> bool:
        return self.is_functor(category) and self.equivalent(
            self.result(category), self.argument(category)
        )

    def coindices(self, category: Term, /) -> tuple[()]:
        del category
        return ()


class _Universe(NamedTuple):
    """The functor view, atoms and candidate categories of one compiled store."""

    functors: Functors[Term]
    atoms: Mapping[str, Term]
    candidates: tuple[Term, ...]


class _Grammar(NamedTuple):
    """One compiled baseline grammar and the induction roles read from it."""

    assembly: GrammarAssembly
    functors: Functors[Term]
    categories: tuple[Term, ...]
    nominal: Term
    sentence: Term
    coordination: Term
    gold_sentence_atom: str
    gold_coordination_atom: str
    headed_arguments: tuple[Term, ...]
    sentence_arguments: tuple[Term, ...]
    raising_rules: frozenset[str]
    coordination_marks: frozenset[str]


def _resolved(root: Path, declared: str, /) -> Path:
    """Resolve one declared source against the fit document's directory."""
    lowered = Path(lower_package_resource(declared))
    return lowered if lowered.is_absolute() else root / lowered


def _section(path: Path, root: Path, /) -> int | None:
    """Read a two-digit CCGbank section from a source-relative path."""
    relative = path.relative_to(root) if root.is_dir() else path
    for part in reversed(relative.parts[:-1]):
        if len(part) == 2 and part.isascii() and part.isdecimal():
            return int(part)
    stem = path.stem
    return int(stem[4:6]) if stem.startswith("wsj_") and stem[4:6].isdigit() else None


def _auto_files(source: Path, sections: frozenset[int], /) -> tuple[Path, ...]:
    """Return the exact requested AUTO sections in stable source order."""
    candidates = (
        (source,) if source.is_file() else tuple(sorted(source.rglob("*.auto")))
    )
    selected = tuple(
        path
        for path in candidates
        if path.suffix.casefold() == ".auto" and _section(path, source) in sections
    )
    found = frozenset(
        section for path in selected if (section := _section(path, source)) is not None
    )
    missing = sorted(sections - found)
    if missing:
        msg = f"CCGbank source {source} has no AUTO data for sections {missing!r}"
        raise ValueError(msg)
    return selected


def _leaves(node: AutoNode, /) -> tuple[AutoLeaf, ...]:
    """Return AUTO lexical leaves in surface order."""
    if isinstance(node, AutoLeaf):
        return (node,)
    if not isinstance(node, AutoTree):
        msg = f"an AUTO node has an unknown representation: {node!r}"
        raise TypeError(msg)
    return tuple(leaf for child in node.children for leaf in _leaves(child))


def _auto_sentences(
    source: Path, files: Sequence[Path], /
) -> tuple[_AutoSentence, ...]:
    """Parse selected AUTO files."""
    rows: list[_AutoSentence] = []
    seen: set[ItemId] = set()
    codec = CcgbankAutoCodec()
    for path in files:
        section = _section(path, source)
        if section is None:
            msg = f"AUTO source {path} has no CCGbank section"
            raise ValueError(msg)
        derivations = codec.parse(path.read_bytes())
        for position, derivation in enumerate(derivations):
            identifier_value = dict(derivation.header).get("ID")
            if identifier_value is None or not identifier_value.strip():
                msg = f"AUTO derivation {position} in {path} has no canonical ID"
                raise ValueError(msg)
            identifier = ItemId(identifier_value)
            if identifier in seen:
                msg = f"CCGbank source repeats AUTO identifier {identifier!r}"
                raise ValueError(msg)
            seen.add(identifier)
            prepared = gold_sentence(derivation.tree)
            leaves = _leaves(derivation.tree)
            pos = tuple(leaf.original_pos for leaf in leaves)
            forms = tuple(token.surface for token in prepared.tokens)
            if len(pos) != len(forms):
                msg = f"AUTO item {identifier!r} has different token and POS counts"
                raise ValueError(msg)
            rows.append(
                _AutoSentence(
                    section,
                    identifier,
                    forms,
                    pos,
                    prepared.tokens,
                    prepared.categories,
                )
            )
    return tuple(rows)


def _categories(
    functors: Functors[Term],
    atoms: Mapping[str, Term],
    settings: BiskHockenmaierBaselineSettings,
    measure: CategoryMeasure,
    /,
) -> tuple[Term, ...]:
    """Generate the finite lexical category universe declared by the baseline."""
    nominal = atoms[settings.nominal_category]
    sentence = atoms[settings.sentence_category]
    bases = (nominal, sentence)
    arguments = tuple(atoms[name] for name in settings.headed_or_modifier_arguments)
    sentence_arguments = tuple(atoms[name] for name in settings.sentence_arguments)
    held: list[Term] = list(bases)
    frontier = list(bases)
    for _ in range(settings.max_arity):
        grown: list[Term] = []
        for result in frontier:
            base = result
            while functors.is_functor(base):
                base = base.args[functors.result_path[0]]
            choices = sentence_arguments if base is sentence else arguments
            for argument in choices:
                for slash in sorted(functors.slashes):
                    category = functors.make(slash, result, argument)
                    if category not in held and category not in grown:
                        grown.append(category)
        held.extend(grown)
        frontier = grown
    if settings.second_order_modifiers:
        first_order = tuple(held)
        for category in first_order:
            if measure.arity(category) >= settings.max_arity:
                continue
            for slash in sorted(functors.slashes):
                modifier = functors.make(slash, category, category)
                if modifier not in held:
                    held.append(modifier)
    held.append(atoms[settings.coordination_category])
    return tuple(held)


def _inducer_settings(
    settings: BiskHockenmaierBaselineSettings,
    conventions: Mapping[str, JsonValue],
    /,
) -> BiskHockenmaierSettings:
    """Translate the baseline declaration into the core induction controls."""
    return BiskHockenmaierSettings(
        nominal_tags=settings.nominal_tags,
        verbal_tags=settings.verbal_tags,
        coordination_tags=settings.coordination_tags,
        other_tags=settings.other_tags,
        gerund_tag=settings.gerund_tag,
        max_arity=settings.max_arity,
        category_induction_steps=settings.category_induction_steps,
        derived_constituent_pass=settings.derived_constituent_pass,
        second_order_modifiers=settings.second_order_modifiers,
        iterations=settings.em_max_iterations,
        k=settings.k,
        probability_floor=settings.probability_floor,
        tolerance=settings.convergence_tolerance,
        conventions=conventions,
    )


def _functors(declaration: GrammarDecl, store: TermStore, /) -> Functors[Term]:
    """Return the one forward and backward functor view the grammar declares."""
    views = declaration.signature.functors
    forward = tuple(view for view in views if view.argument_side == "right")
    backward = tuple(view for view in views if view.argument_side == "left")
    if len(forward) != 1 or len(backward) != 1 or len(views) != 2:
        msg = (
            "the baseline grammar must declare exactly one forward and one "
            f"backward functor view, not {tuple(view.name for view in views)!r}"
        )
        raise ValueError(msg)
    (right,) = forward
    (left,) = backward
    if (right.result, right.argument) != (left.result, left.argument):
        msg = "the baseline grammar's functor views place their slots differently"
        raise ValueError(msg)
    return FunctorView(
        store,
        right.constructor,
        left.constructor,
        result=right.result,
        argument=right.argument,
    )


def _universe(
    assembly: GrammarAssembly,
    declaration: GrammarDecl,
    settings: BiskHockenmaierBaselineSettings,
    /,
) -> _Universe:
    """Return the functors, atoms and candidate categories of one assembly."""
    store = assembly.store
    functors = _functors(declaration, store)
    atoms = {
        name: assembly.syntax.parse(store, name)
        for name in declaration.signature.primitives
    }
    candidates = _categories(functors, atoms, settings, assembly.category_measure)
    return _Universe(functors, atoms, candidates)


def _grammar(settings: BiskHockenmaierBaselineSettings, /) -> _Grammar:
    """Compile the declared grammar over the baseline's lexical categories."""
    compiler = GrammarIso()
    declared = settings.grammar
    assembly = compiler.forward(declared, settings.language)
    universe = _universe(assembly, declared, settings)
    rules = assembly.grammar.rules
    roles = {rule.name: rule.normal_form_role for rule in rules}
    selections = assembly.head_selections
    return _Grammar(
        assembly,
        universe.functors,
        universe.candidates,
        universe.atoms[settings.nominal_category],
        universe.atoms[settings.sentence_category],
        universe.atoms[settings.coordination_category],
        settings.gold_sentence_atom,
        settings.gold_coordination_atom,
        tuple(universe.atoms[name] for name in settings.headed_or_modifier_arguments),
        tuple(universe.atoms[name] for name in settings.sentence_arguments),
        frozenset(name for name, role in roles.items() if role.raising is not None),
        frozenset(
            name
            for name, role in roles.items()
            if role.coordination and selections[name].policy == "non-absorbing"
        ),
    )


def _inducer(
    settings: BiskHockenmaierBaselineSettings,
    grammar: _Grammar,
    conventions: Mapping[str, JsonValue],
    /,
) -> BiskHockenmaierInducer:
    """Build the declared inducer before its POS-specific parse forests."""
    return BiskHockenmaierInducer(
        _inducer_settings(settings, conventions),
        grammar.functors,
        nominal=grammar.nominal,
        sentence=grammar.sentence,
        coordination=grammar.coordination,
        headed_arguments=grammar.headed_arguments,
        sentence_arguments=grammar.sentence_arguments,
    )


def _raising_targets(
    chart: Chart[Term, float], grammar: _Grammar, /
) -> Mapping[tuple[str, Term], Term]:
    """Index the declared target of every raising edge retained in one chart."""
    targets: dict[tuple[str, Term], Term] = {}
    for span in chart.index.spans_by_length():
        for parent in chart.items_at(span):
            names = frozenset(edge.rule_name for edge in chart.edges(parent))
            if not names & grammar.raising_rules:
                continue
            if not grammar.functors.is_functor(parent.reading):
                msg = f"a type-raising edge produced nonfunctor {parent.reading!r}"
                raise ValueError(msg)
            target = parent.reading.args[grammar.functors.result_path[0]]
            for name in names & grammar.raising_rules:
                targets[(name, parent.reading)] = target
    return targets


def _root_feeders(
    pos: Sequence[str],
    grammar: _Grammar,
    settings: BiskHockenmaierBaselineSettings,
    /,
) -> frozenset[Term] | None:
    """Return the only categories a POS sequence lets derive the goal, if any.

    A sentence with a nongerund verb derives the goal only from the sentence
    category; any other sentence derives it by every goal rule of the grammar.
    """
    forced_sentence = any(
        tag in settings.verbal_tags and tag != settings.gerund_tag for tag in pos
    )
    return frozenset({grammar.sentence}) if forced_sentence else None


def _item(
    sentence: _AutoSentence,
    grammar: _Grammar,
    settings: BiskHockenmaierBaselineSettings,
    inducer: BiskHockenmaierInducer,
    /,
) -> BiskItem:
    """Parse one AUTO sentence into the packed induction-item boundary."""
    assembly = grammar.assembly
    compiled = assembly.grammar
    budget = compiled.budget
    support = settings.parse
    chart = parse(
        sentence.forms,
        rules=compiled.rules,
        source=_SentenceLexicon(
            tuple(
                inducer.lexical_categories(tag, grammar.categories)
                for tag in sentence.pos
            )
        ),
        semiring=support.semiring,
        scorer=_UnitScorer(),
        goal=compiled.goal,
        category_measure=assembly.category_measure,
        max_category_arity=budget.max_category_arity,
        max_category_size=budget.max_category_size,
        unary_depth=budget.unary_rounds,
        pruner=cast("CellPruner[Term, float] | None", support.pruner),
        prune_lexical=support.prune_lexical,
        measure_pruned_mass=support.measure_pruned_mass,
        strategy=support.strategy,
        span_index=support.span_index,
        chart_store=support.chart_store,
        policy=cast("ParsePolicy[Literal[Retain.FOREST]]", support.policy),
        threads=settings.threads,
    )
    projector = BiskHockenmaierDependencyProjector(
        _TermCategoryAlgebra(grammar.functors, compiled.matching),
        grammar.coordination_marks,
    )
    return BiskItem(
        sentence.item_id,
        sentence.forms,
        sentence.pos,
        sentence.tokens,
        chart,
        compiled.goal,
        assembly.syntax,
        assembly.store,
        assembly.head_selections,
        grammar.raising_rules,
        _raising_targets(chart, grammar),
        projector,
        root_feeders=_root_feeders(sentence.pos, grammar, settings),
    )


def _gold_category(
    text: str,
    grammar: _Grammar,
    algebra: MarkedCategoryAlgebra,
    /,
) -> Term:
    """Erase gold features into the induced atomic category inventory.

    The declared gold sentence and coordination atoms erase to the induced
    sentence and coordination categories; every other gold atom erases to the
    induced nominal category.
    """
    if algebra.is_functor(text):
        result_text = algebra.result(text)
        argument_text = algebra.argument(text)
        slash = next(
            candidate
            for candidate in ("/", "\\")
            if algebra.equivalent(
                text,
                f"({result_text}){candidate}({argument_text})",
            )
        )
        return grammar.functors.make(
            slash,
            _gold_category(result_text, grammar, algebra),
            _gold_category(argument_text, grammar, algebra),
        )
    erased = text.split("[", 1)[0]
    if erased == grammar.gold_sentence_atom:
        return grammar.sentence
    if erased == grammar.gold_coordination_atom:
        return grammar.coordination
    return grammar.nominal


def _punctuation(pos: Sequence[str], tags: frozenset[str], /) -> frozenset[int]:
    """Return the one canonical one-based punctuation mask."""
    return frozenset(
        position for position, tag in enumerate(pos, start=1) if tag in tags
    )


def _tree_with_mask(tree: AttachmentTree, mask: frozenset[int], /) -> AttachmentTree:
    """Attach one canonical punctuation mask to a dependency tree."""
    return AttachmentTree(tree.forms, tree.heads, mask)


def _content_digest(root: Path, files: Sequence[Path], /) -> str:
    """Return a stable SHA-256 identity over relative names and file bytes."""
    digest = sha256()
    for path in files:
        relative = path.relative_to(root) if root.is_dir() else Path(path.name)
        name = relative.as_posix().encode()
        content = path.read_bytes()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return f"sha256:{digest.hexdigest()}"


def _length(sentence: _AutoSentence, settings: BiskHockenmaierBaselineSettings) -> int:
    """Return the declared raw or punctuation-excluded sentence length."""
    punctuation = _punctuation(sentence.pos, settings.punctuation_tags)
    return (
        len(sentence.forms) - len(punctuation)
        if settings.length_excludes_punctuation
        else len(sentence.forms)
    )


def _in_length_band(
    sentence: _AutoSentence,
    limit: int | None,
    settings: BiskHockenmaierBaselineSettings,
    /,
) -> bool:
    """Return whether one sentence belongs to a declared evaluation band."""
    return limit is None or _length(sentence, settings) <= limit


def _supertag_counts(
    predictions: Sequence[tuple[BiskItem, tuple[Term, ...] | None]],
    gold: Mapping[ItemId, tuple[Term, ...]],
    punctuation: Mapping[ItemId, frozenset[int]],
    admitted: frozenset[ItemId],
    /,
) -> _Counts:
    """Count exact nonpunctuation lexical-category decisions."""
    matched = 0
    total = 0
    for item, categories in predictions:
        if item.item_id not in admitted:
            continue
        excluded = punctuation[item.item_id]
        expected = gold[item.item_id]
        total += len(expected) - len(excluded)
        if categories is None:
            continue
        matched += sum(
            predicted.key() == target.key()
            for position, (predicted, target) in enumerate(
                zip(categories, expected, strict=True), start=1
            )
            if position not in excluded
        )
    return _Counts(matched, total)


def _dependency_counts(
    predictions: Sequence[tuple[BiskItem, AttachmentTree | None]],
    gold: Mapping[ItemId, AttachmentTree],
    punctuation: Mapping[ItemId, frozenset[int]],
    admitted: frozenset[ItemId],
    /,
) -> _Counts:
    """Count directed attachments over the same nonpunctuation positions."""
    matched = 0
    total = 0
    for item, predicted in predictions:
        if item.item_id not in admitted:
            continue
        mask = punctuation[item.item_id]
        target = _tree_with_mask(gold[item.item_id], mask)
        held_prediction = (
            None if predicted is None else _tree_with_mask(predicted, mask)
        )
        positions = tuple(
            position for position in range(1, target.length + 1) if position not in mask
        )
        total += len(positions)
        if held_prediction is not None:
            matched += sum(
                held_prediction.head_of(position) == target.head_of(position)
                for position in positions
            )
    return _Counts(matched, total)


def _reference(source: Path, evaluation_section: int, /) -> _Reference:
    """Validate the canonical Figure 4(a) dependency reference artifact."""
    value: JsonValue = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        msg = "reference artifact is not an object"
        raise TypeError(msg)
    document = value
    required = {"source", "metric", "training_band", "evaluation_section", "points"}
    if set(document) != required:
        msg = (
            "reference artifact fields differ from the canonical schema: "
            f"{sorted(document)!r}"
        )
        raise ValueError(msg)
    source_value = document["source"]
    if not isinstance(source_value, dict):
        msg = "reference artifact source is not an object"
        raise TypeError(msg)
    metadata = source_value
    source_fields = {"title", "authors", "venue", "url", "figure"}
    if set(metadata) != source_fields:
        msg = "reference artifact source fields differ from the canonical schema"
        raise ValueError(msg)
    authors = metadata["authors"]
    if (
        not isinstance(metadata["title"], str)
        or not metadata["title"].strip()
        or not isinstance(authors, list)
        or not authors
        or any(not isinstance(author, str) or not author.strip() for author in authors)
        or not isinstance(metadata["venue"], str)
        or not metadata["venue"].strip()
        or metadata["figure"] != "Figure 4(a)"
    ):
        msg = "reference artifact does not identify Figure 4(a) exactly"
        raise ValueError(msg)
    citation = metadata.get("url")
    if not isinstance(citation, str) or not citation.strip():
        msg = "reference artifact has no citation URL"
        raise ValueError(msg)
    if (
        document["metric"] != "unlabeled directed attachment"
        or document["training_band"] != "wsj10"
    ):
        msg = "reference artifact names a different metric or training band"
        raise ValueError(msg)
    if document["evaluation_section"] != evaluation_section:
        msg = "reference artifact names a different evaluation section"
        raise ValueError(msg)
    values = document["points"]
    if not isinstance(values, list):
        msg = "reference artifact points are not a list"
        raise TypeError(msg)
    scores: dict[str, float] = {}
    for position, raw in enumerate(values):
        if not isinstance(raw, dict) or set(raw) != {"metric", "score_percent"}:
            msg = f"reference point {position} does not match the canonical schema"
            raise ValueError(msg)
        point = raw
        metric = point["metric"]
        score = point["score_percent"]
        if (
            not isinstance(metric, str)
            or isinstance(score, bool)
            or not isinstance(score, int | float)
        ):
            msg = f"reference point {position} has invalid metric or score"
            raise TypeError(msg)
        if metric in scores:
            msg = f"reference point {position} repeats metric {metric!r}"
            raise ValueError(msg)
        value = float(score) / 100.0
        if not 0.0 <= value <= 1.0:
            msg = f"reference point {position} is outside the accuracy range"
            raise ValueError(msg)
        scores[metric] = value
    if tuple(scores) != _REFERENCE_METRICS:
        msg = (
            "reference dependency metrics are incomplete or reordered: "
            f"{tuple(scores)!r}"
        )
        raise ValueError(msg)
    return _Reference(citation, scores)


@final
class BiskHockenmaierOperation:
    """Fit and evaluate one completely bound baseline declaration."""

    __slots__ = ("_context", "_settings")

    def __init__(
        self,
        settings: BiskHockenmaierBaselineSettings,
        context: BaselineBuildContext,
    ) -> None:
        self._settings = settings
        self._context = context

    def __call__(self, /) -> BaselineResult:
        """Run corpus loading, induction, six scores and the reference gate."""
        settings = self._settings
        source = _resolved(self._context.root, settings.ccgbank_source)
        selected = frozenset(
            (
                *settings.training_sections,
                settings.development_section,
                settings.evaluation_section,
            )
        )
        auto_files = _auto_files(source, selected)
        sentences = _auto_sentences(source, auto_files)
        training = tuple(
            sentence
            for sentence in sentences
            if sentence.section in settings.training_sections
            and _length(sentence, settings) <= settings.training_max_length
        )
        evaluation = tuple(
            sentence
            for sentence in sentences
            if sentence.section == settings.evaluation_section
        )
        if not training or not evaluation:
            msg = (
                "Bisk--Hockenmaier fit needs nonempty training and evaluation "
                "selections"
            )
            raise ValueError(msg)
        identities = tuple(
            CcgbankSentenceIdentity(sentence.item_id, sentence.forms, sentence.pos)
            for sentence in evaluation
        )
        dependency_source = _resolved(
            self._context.root, settings.dependency_gold_source
        )
        aligned = align_conll2008(
            read_conll2008(dependency_source),
            identities,
        )
        dependency_gold = {row.item_id: row.sentence.tree for row in aligned}
        reference_source = _resolved(self._context.root, settings.reference_source)
        source_digests = {
            "ccgbank-auto": _content_digest(source, auto_files),
            "dependency-gold": _content_digest(dependency_source, (dependency_source,)),
            "published-reference": _content_digest(
                reference_source, (reference_source,)
            ),
        }
        conventions: dict[str, JsonValue] = dict(settings.conventions)
        conventions["source_digests"] = dict(sorted(source_digests.items()))
        grammar = _grammar(settings)
        inducer = _inducer(settings, grammar, conventions)
        training_items = tuple(
            _item(sentence, grammar, settings, inducer) for sentence in training
        )
        evaluation_items = tuple(
            _item(sentence, grammar, settings, inducer) for sentence in evaluation
        )
        model = inducer.fit(training_items)
        predictions = tuple((item, model.predict(item)) for item in evaluation_items)
        tag_predictions = tuple(
            (item, prediction.supertags.parser_categories)
            for item, prediction in predictions
        )
        dependency_predictions = tuple(
            (item, prediction.dependencies) for item, prediction in predictions
        )
        gold_algebra = MarkedCategoryAlgebra()
        gold_tags = {
            sentence.item_id: tuple(
                _gold_category(category, grammar, gold_algebra)
                for category in sentence.gold_categories
            )
            for sentence in evaluation
        }
        punctuation = {
            sentence.item_id: _punctuation(sentence.pos, settings.punctuation_tags)
            for sentence in evaluation
        }
        by_id = {sentence.item_id: sentence for sentence in evaluation}
        results: dict[str, _Counts] = {}
        for limit in settings.evaluation_length_bands:
            name = f"length-{limit}" if limit is not None else "all-lengths"
            admitted = frozenset(
                item_id
                for item_id, sentence in by_id.items()
                if _in_length_band(sentence, limit, settings)
            )
            results[f"supertagging-{name}"] = _supertag_counts(
                tag_predictions, gold_tags, punctuation, admitted
            )
            results[f"dependencies-{name}"] = _dependency_counts(
                dependency_predictions, dependency_gold, punctuation, admitted
            )
        reference = _reference(
            reference_source,
            settings.evaluation_section,
        )
        refused = tuple(
            (metric, results[metric].value, target)
            for metric, target in reference.scores.items()
            if abs(results[metric].value - target) > settings.reference_tolerance
        )
        if refused:
            detail = ", ".join(
                f"{metric}={observed:.6f} outside {target:.6f} "
                f"+/- {settings.reference_tolerance:.6f}"
                for metric, observed, target in refused
            )
            msg = f"Bisk--Hockenmaier Figure 4(a) reference gate failed: {detail}"
            raise ValueError(msg)
        families = {
            "supertagging": "ccg-induction:supertagging",
            "dependencies": "ccg-induction:dependencies",
        }
        metrics = tuple(
            f"{family}-{band}"
            for limit in settings.evaluation_length_bands
            for band in (f"length-{limit}" if limit is not None else "all-lengths",)
            for family in ("supertagging", "dependencies")
        )
        return BaselineResult(
            tuple(
                BaselineMeasurement(
                    families[metric.split("-", 1)[0]],
                    _SYSTEM,
                    metric,
                    "re-running",
                    reference.citation,
                    _REASON,
                    results[metric].value,
                    results[metric].total,
                    "accuracy",
                )
                for metric in metrics
            ),
            source_digests,
        )


def build_bisk_hockenmaier_baseline(
    settings: BiskHockenmaierBaselineSettings,
    context: BaselineBuildContext,
    /,
) -> BiskHockenmaierOperation:
    """Bind a validated baseline declaration to its run resources."""
    return BiskHockenmaierOperation(settings, context)

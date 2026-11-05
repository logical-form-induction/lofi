"""Declared construction of judgment items from bead templates and lexicons.

An item-construction document names bead resources: templates, read from bead
JSON Lines files or template collections or declared inline; lexicons, read
from bead JSON Lines files or delimited tables, fetched through bead's resource
adapters, declared inline, or proposed by an installed generator; the bead
filling strategy each template is filled by; the bead item template whose
elements name what each item renders and whose constraints every item
satisfies; and the attributes each item records in its ``item_metadata``.
Every resource is declared in bead's own serialized shape, identities included,
and an identity bead would draw at random is derived from the declared
namespace instead, so the same document always constructs the same items.

Items are built by bead's item constructor. An item template whose elements
name no filled template renders each filling of every template as one item; one
whose elements each name a declared template renders every combination of
those templates' fillings as one item, each element showing its template's
filling. A declared admission expression, read over a combination as an
attribute is, admits the combinations that become items.

An attribute is a bead DSL expression over one item's fillings. For an item
of one template, ``template`` is the bead template and ``slots`` maps each slot
name to the lexical item filling it, so ``slots.verb.lemma`` reads a filler's
lemma; for an item combining several templates, ``templates`` maps each
template's name to its ``template`` and ``slots``. ``item_template`` is the
bead item template and ``elements`` maps each element name to its declared
element, so template and element metadata are readable too. No slot name is
reserved. An item's identity is a UUID derived from the declared namespace and,
for every template it renders, the template's identity and the identity of the
lexical item filling each slot, in slot-name order.
"""

import json
from importlib import import_module
from itertools import product
from typing import TYPE_CHECKING, Final, Protocol, cast, final
from uuid import UUID, uuid5

from bead.dsl.errors import DSLError
from bead.dsl.evaluator import DSLEvaluator
from bead.items.adapters.registry import default_registry
from bead.items.cache import ModelOutputCache
from bead.items.constructor import ItemConstructor
from bead.items.item_template import ItemTemplate
from bead.resources.constraints import Constraint
from bead.resources.lexical_item import LexicalItem
from bead.resources.lexicon import Lexicon
from bead.resources.loaders import from_csv, from_tsv
from bead.resources.template import Template, slots_match_template
from bead.resources.template_collection import TemplateCollection
from bead.templates.filler import (
    ConstraintUnsatisfiableError,
    CSPFiller,
    FilledTemplate,
)
from bead.templates.renderers import DefaultRenderer
from bead.templates.resolver import ConstraintResolver
from bead.templates.strategies import (
    ExhaustiveStrategy,
    FillingStrategy,
    MixedFillingStrategy,
    MLMFillingStrategy,
    RandomStrategy,
    StrategyConfig,
    StrategyFiller,
    StratifiedStrategy,
)

from lofi._frozen import FrozenValue
from lofi.judgment._spec import (
    ConstraintFile,
    ConstraintFilling,
    ExhaustiveFilling,
    FillingSpec,
    GlazingLexicon,
    InlineConstraint,
    InlineItemTemplate,
    InlineLexicon,
    InlineTemplate,
    ItemConstructionSpec,
    ItemTemplateFile,
    LexiconFile,
    MaskedLMFilling,
    MaskedLMGeneratorSpec,
    MaskedLMSettings,
    MixedFilling,
    RandomFilling,
    StratifiedFilling,
    StreamingFilling,
    TableLexicon,
    TemplateCollectionFile,
    TemplateFile,
    UniMorphLexicon,
    refuse_unidentified,
)
from lofi.nn._snapshots import artifact_fetcher
from lofi.registry import ComponentPlugin, plugin_for

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence
    from pathlib import Path

    from bead.dsl.evaluator import DslValue
    from bead.items.adapters.registry import ModelAdapterRegistry
    from bead.items.item import Item, MetadataValue
    from bead.resources.adapters import GlazingAdapter, UniMorphAdapter
    from bead.templates.adapters.huggingface import HuggingFaceMLMAdapter
    from bead.templates.streaming import StreamingFiller

    from lofi.spec.lexicon_generators import LexiconGeneratorSpec


__all__ = [
    "MASKED_LM_PLUGIN",
    "FillMaskPipeline",
    "LexiconGenerator",
    "MaskedLMLexiconGenerator",
    "construct_items",
    "item_identity",
    "lexicon_generator_for",
    "masked_lm_adapter",
]

_CONTENT_FIELDS: Final = (
    "lemma",
    "form",
    "language_code",
    "features",
    "source",
    "metadata",
)
"""The lexical-item fields a derived lexical-item identity is read from."""

_SEED_BOUND: Final = 2**32
"""One past the largest seed a derived filling seed takes."""


class FillMaskPipeline(Protocol):
    """Return masked-token candidates from one rendered prompt."""

    def __call__(
        self, prompt: str, /, *, top_k: int
    ) -> tuple[Mapping[str, object], ...]:
        """Return candidates in descending model score order."""
        ...


class _PipelineFactory(Protocol):
    """Construct a Hugging Face task pipeline."""

    def __call__(self, task: str, /, *, model: str) -> FillMaskPipeline:
        """Return the pipeline over a fetched model directory."""
        ...


class LexiconGenerator(Protocol):
    """Propose one bead lexicon."""

    def generate(self, namespace: UUID, /) -> Lexicon:
        """Return the proposed lexicon, its items' identities under ``namespace``."""
        ...


@final
class MaskedLMLexiconGenerator(FrozenValue):
    """Propose a lexicon from a pinned masked language model's completions.

    Parameters
    ----------
    settings : MaskedLMSettings
        The resolved declaration.
    fill : FillMaskPipeline
        The pinned model's fill-mask pipeline.
    """

    _noun = "masked-LM lexicon generator"
    __slots__ = ("_fill", "_settings")

    def __init__(self, settings: MaskedLMSettings, fill: FillMaskPipeline, /) -> None:
        object.__setattr__(self, "_settings", settings)
        object.__setattr__(self, "_fill", fill)

    def _lemmas(self) -> tuple[str, ...]:
        """Return the highest-scoring distinct single-token completions."""
        settings = self._settings
        best: dict[str, float] = {}
        for prompt in settings.prompts:
            for candidate in self._fill(prompt, top_k=settings.count):
                token = candidate.get("token_str")
                score = candidate.get("score")
                if not isinstance(token, str):
                    msg = "a fill-mask candidate has no token_str"
                    raise TypeError(msg)
                if isinstance(score, bool) or not isinstance(score, int | float):
                    msg = "a fill-mask candidate has no numeric score"
                    raise TypeError(msg)
                lemma = token.strip()
                if not lemma or any(character.isspace() for character in lemma):
                    continue
                best[lemma] = max(best.get(lemma, float("-inf")), float(score))
        ranked = sorted(best.items(), key=lambda pair: (-pair[1], pair[0]))
        selected = tuple(lemma for lemma, _ in ranked[: settings.count])
        if len(selected) != settings.count:
            msg = (
                f"the masked language model yielded {len(selected)} distinct "
                f"single-token completions for {settings.lexicon!r}, not "
                f"{settings.count}"
            )
            raise ValueError(msg)
        return selected

    def generate(self, namespace: UUID, /) -> Lexicon:
        """Return the proposed lexicon, best completion first."""
        settings = self._settings
        source = f"masked-lm:{settings.checkpoint.identity}"
        return Lexicon(
            name=settings.lexicon,
            language_code=settings.language_code,
            items=tuple(
                LexicalItem(
                    id=uuid5(
                        namespace,
                        f"lexical-item\0{source}\0{settings.language_code}\0{lemma}",
                    ),
                    lemma=lemma,
                    language_code=settings.language_code,
                    features=dict(settings.features),
                    source=source,
                )
                for lemma in self._lemmas()
            ),
        )


def _hugging_face(settings: MaskedLMSettings, context: None, /) -> LexiconGenerator:
    """Build the pinned Hugging Face masked-LM lexicon generator."""
    del context
    module = import_module("transformers")
    factory = cast("_PipelineFactory", module.pipeline)
    directory = artifact_fetcher(settings.cache).fetch(settings.checkpoint)
    fill = factory("fill-mask", model=str(directory))
    return MaskedLMLexiconGenerator(settings, fill)


MASKED_LM_PLUGIN: Final = ComponentPlugin(
    spec=MaskedLMGeneratorSpec, build=_hugging_face
)
"""The pinned masked-language-model lexicon generator."""


class _GeneratorDeclaration(Protocol):
    """A validated generator declaration exposing immutable settings."""

    @property
    def kind(self) -> str:
        """Return the installed variant tag."""
        ...

    @classmethod
    def family(cls) -> str:
        """Return the component family."""
        ...

    def settings(self) -> object:
        """Return the resolved generator."""
        ...


def lexicon_generator_for(declaration: LexiconGeneratorSpec, /) -> LexiconGenerator:
    """Build the lexicon generator installed under the declaration's tag.

    Raises
    ------
    UnknownTagError
        If no generator is installed under the tag.
    RegistryError
        If the entry is not a plugin for the declaration's class.
    """
    declared = cast("_GeneratorDeclaration", declaration)
    plugin = cast(
        "ComponentPlugin[object, None, LexiconGenerator]",
        plugin_for(declared),
    )
    return plugin.construct(declared, None)


def item_identity(
    namespace: UUID, fillings: Mapping[UUID, Mapping[str, UUID]], /
) -> UUID:
    """Return an item's identity from the templates it renders and their fillers.

    Parameters
    ----------
    namespace : UUID
        The declared namespace.
    fillings : Mapping[UUID, Mapping[str, UUID]]
        For each template the item renders, by identity, the identity of the
        lexical item filling each slot, keyed by slot name.
    """
    parts: list[str] = []
    for template in sorted(fillings, key=str):
        slots = fillings[template]
        parts.append(str(template))
        parts.extend(f"{slot}={slots[slot]}" for slot in sorted(slots))
    return uuid5(namespace, "\0".join(("item", *parts)))


def _ensure_file(path: Path, /) -> None:
    """Refuse a declared resource file that does not exist."""
    if not path.is_file():
        msg = f"the declared resource {path.name!r} is not a file under {path.parent}"
        raise ValueError(msg)


def _identified_lines(path: Path, noun: str, /) -> tuple[str, ...]:
    """Return a JSON Lines file's records, each declaring its identity."""
    _ensure_file(path)
    lines = tuple(
        stripped
        for line in path.read_text(encoding="utf-8").splitlines()
        if (stripped := line.strip())
    )
    for number, line in enumerate(lines, start=1):
        refuse_unidentified(json.loads(line), f"{noun} on line {number} of {path.name}")
    return lines


def _templates(spec: ItemConstructionSpec, root: Path, /) -> tuple[Template, ...]:
    """Return the declared templates in construction order."""
    held: list[Template] = []
    for source in spec.templates:
        match source:
            case TemplateFile(path=declared):
                held.extend(
                    Template.model_validate_json(line)
                    for line in _identified_lines(root / declared, "a template")
                )
            case TemplateCollectionFile(path=declared):
                path = root / declared
                _ensure_file(path)
                text = path.read_text(encoding="utf-8")
                document = json.loads(text)
                templates = (
                    document.get("templates") if isinstance(document, dict) else ()
                )
                for position, template in enumerate(
                    templates if isinstance(templates, list) else ()
                ):
                    refuse_unidentified(
                        template, f"template {position} of the collection {path.name}"
                    )
                held.extend(TemplateCollection.model_validate_json(text))
            case InlineTemplate(template=template):
                held.append(template)
            case _:
                msg = f"unknown template source {source!r}"
                raise TypeError(msg)
    for template in held:
        slots_match_template(template)
    identities = tuple(template.id for template in held)
    if len(set(identities)) != len(identities):
        msg = "the declared templates repeat a template identity"
        raise ValueError(msg)
    return tuple(held)


def _identified(lexicon: Lexicon, namespace: UUID, /) -> Lexicon:
    """Return a loaded lexicon with each item's identity derived from its content.

    Raises
    ------
    ValueError
        If two items have one content.
    """
    items: list[LexicalItem] = []
    for item in lexicon:
        dumped = json.loads(item.model_dump_json())
        content = json.dumps(
            {field: dumped[field] for field in _CONTENT_FIELDS}, sort_keys=True
        )
        identity = uuid5(namespace, f"lexical-item\0{lexicon.name}\0{content}")
        items.append(item.with_(id=identity))
    identities = tuple(item.id for item in items)
    if len(set(identities)) != len(identities):
        msg = f"the lexicon {lexicon.name!r} lists one lexical item twice"
        raise ValueError(msg)
    return Lexicon(name=lexicon.name).with_items(tuple(items))


def _fetched(name: str, items: Iterable[LexicalItem], namespace: UUID, /) -> Lexicon:
    """Return the items a resource adapter fetched as one identified lexicon."""
    fetched = Lexicon(name=name).with_items(tuple(items))
    if not len(fetched):
        msg = f"the resource adapter fetched no lexical item for {name!r}"
        raise ValueError(msg)
    return _identified(fetched, namespace)


class _ResourceAdapters(Protocol):
    """bead's resource adapters, loaded only where a lexicon fetches through one."""

    GlazingAdapter: type[GlazingAdapter]
    UniMorphAdapter: type[UniMorphAdapter]


class _Streaming(Protocol):
    """bead's streaming filler, loaded only where a template streams through it."""

    StreamingFiller: type[StreamingFiller]


class _MaskedLMAdapters(Protocol):
    """bead's masked-LM adapter, loaded only where a filling declares one."""

    HuggingFaceMLMAdapter: type[HuggingFaceMLMAdapter]


def _declared_lexicon(source: object, root: Path, namespace: UUID, /) -> Lexicon:
    """Return one declared lexicon."""
    match source:
        case LexiconFile(path=declared, name=name):
            lines = _identified_lines(root / declared, "a lexical item")
            return Lexicon(
                name=name,
                items=tuple(LexicalItem.model_validate_json(line) for line in lines),
            )
        case TableLexicon() as table:
            path = root / table.path
            _ensure_file(path)
            load = from_csv if table.format == "csv" else from_tsv
            loaded = load(
                path,
                table.name,
                language_code=table.language_code,
                column_mapping=dict(table.column_mapping) or None,
                feature_columns=list(table.feature_columns) or None,
                pos=table.pos,
            )
            return _identified(loaded, namespace)
        case GlazingLexicon() as glazing:
            adapters = cast(
                "_ResourceAdapters", import_module("bead.resources.adapters")
            )
            adapter = adapters.GlazingAdapter(resource=glazing.resource)
            return _fetched(
                glazing.name,
                adapter.fetch_items(
                    glazing.query, glazing.language_code, **dict(glazing.options)
                ),
                namespace,
            )
        case UniMorphLexicon() as unimorph:
            adapters = cast(
                "_ResourceAdapters", import_module("bead.resources.adapters")
            )
            return _fetched(
                unimorph.name,
                adapters.UniMorphAdapter().fetch_items(
                    unimorph.query, unimorph.language_code, **dict(unimorph.options)
                ),
                namespace,
            )
        case InlineLexicon(lexicon=lexicon):
            return lexicon
    msg = f"unknown lexicon source {source!r}"
    raise TypeError(msg)


def _lexicon(
    spec: ItemConstructionSpec,
    root: Path,
    generator_for: Callable[[LexiconGeneratorSpec], LexiconGenerator],
    /,
) -> Lexicon:
    """Return every declared and generated lexical item as one lexicon."""
    lexicons = [
        _declared_lexicon(source, root, spec.namespace) for source in spec.lexicons
    ]
    lexicons.extend(
        generator_for(declared).generate(spec.namespace) for declared in spec.generators
    )
    if not lexicons:
        msg = "an item construction declares no lexicon and no generator"
        raise ValueError(msg)
    names = tuple(lexicon.name for lexicon in lexicons)
    if len(set(names)) != len(names):
        msg = f"the declared lexicons repeat a name: {names!r}"
        raise ValueError(msg)
    merged = Lexicon(name=" + ".join(names))
    return merged.with_items(tuple(item for lexicon in lexicons for item in lexicon))


def _item_template(spec: ItemConstructionSpec, root: Path, /) -> ItemTemplate:
    """Return the declared item template, which names at least one element."""
    match spec.item_template:
        case ItemTemplateFile(path=declared):
            path = root / declared
            _ensure_file(path)
            text = path.read_text(encoding="utf-8")
            refuse_unidentified(json.loads(text), f"the item template in {path.name}")
            template = ItemTemplate.model_validate_json(text)
        case InlineItemTemplate(item_template=declared):
            template = declared
        case source:
            msg = f"unknown item-template source {source!r}"
            raise TypeError(msg)
    if not template.elements:
        msg = f"the item template {template.name!r} declares no element"
        raise ValueError(msg)
    for element in template.elements:
        if element.is_text and element.content is None:
            msg = f"the text element {element.element_name!r} declares no content"
            raise ValueError(msg)
    return template


def _constraints(
    spec: ItemConstructionSpec, root: Path, item_template: ItemTemplate, /
) -> dict[UUID, Constraint]:
    """Return the declared constraints, exactly those the item template names."""
    held: list[Constraint] = []
    for source in spec.constraints:
        match source:
            case ConstraintFile(path=declared):
                held.extend(
                    Constraint.model_validate_json(line)
                    for line in _identified_lines(root / declared, "a constraint")
                )
            case InlineConstraint(constraint=constraint):
                held.append(constraint)
            case _:
                msg = f"unknown constraint source {source!r}"
                raise TypeError(msg)
    declared = {constraint.id: constraint for constraint in held}
    if len(declared) != len(held):
        msg = "the declared constraints repeat a constraint identity"
        raise ValueError(msg)
    named = frozenset(item_template.constraints)
    missing = sorted(str(identity) for identity in named - declared.keys())
    unused = sorted(str(identity) for identity in declared.keys() - named)
    if missing or unused:
        msg = (
            f"the item template {item_template.name!r} names the constraints "
            f"{missing!r} the construction does not declare, and the construction "
            f"declares the constraints {unused!r} the item template does not name"
        )
        raise ValueError(msg)
    return declared


def masked_lm_adapter(declared: MaskedLMFilling, /) -> HuggingFaceMLMAdapter:
    """Return bead's loaded Hugging Face masked-LM adapter for a declared filling."""
    adapters = cast(
        "_MaskedLMAdapters", import_module("bead.templates.adapters.huggingface")
    )
    directory = artifact_fetcher(declared.artifacts.settings()).fetch(declared.pinned())
    adapter = adapters.HuggingFaceMLMAdapter(str(directory), device=declared.device)
    adapter.load_model()
    return adapter


def _derived_seed(seed: int | None, namespace: UUID, template: Template, /) -> int:
    """Return a declared seed, or one derived from the namespace and template."""
    if seed is not None:
        return seed
    return uuid5(namespace, f"filling\0{template.id}").int % _SEED_BOUND


def _enumerating(
    filling: FillingSpec, namespace: UUID, template: Template, /
) -> FillingStrategy | None:
    """Return the bead strategy of an enumerating filling, or None for another."""
    match filling:
        case ExhaustiveFilling():
            return ExhaustiveStrategy()
        case RandomFilling(samples=samples, seed=seed):
            return RandomStrategy(
                n_samples=samples, seed=_derived_seed(seed, namespace, template)
            )
        case StratifiedFilling(samples=samples, grouping_property=grouping, seed=seed):
            return StratifiedStrategy(
                n_samples=samples,
                grouping_property=grouping,
                seed=_derived_seed(seed, namespace, template),
            )
    return None


def _masked_lm(
    declared: MaskedLMFilling,
    adapter_for: Callable[[MaskedLMFilling], HuggingFaceMLMAdapter],
    /,
) -> MLMFillingStrategy:
    """Return bead's masked-LM strategy for one declared filling."""
    return MLMFillingStrategy(
        ConstraintResolver(),
        adapter_for(declared),
        beam_size=declared.beam_size,
        fill_direction=declared.fill_direction,
        custom_order=(
            None if declared.custom_order is None else list(declared.custom_order)
        ),
        top_k=declared.top_k,
        budget=declared.budget,
        per_slot_max_fills=dict(declared.per_slot_max_fills),
        per_slot_enforce_unique=dict(declared.per_slot_enforce_unique),
    )


def _mixed(
    declared: MixedFilling,
    namespace: UUID,
    template: Template,
    adapter_for: Callable[[MaskedLMFilling], HuggingFaceMLMAdapter],
    /,
) -> MixedFillingStrategy:
    """Return bead's mixed strategy for one declared filling."""
    slots: dict[str, tuple[str | FillingStrategy, StrategyConfig]] = {}
    for slot, filling in declared.slots.items():
        enumerating = _enumerating(filling, namespace, template)
        if enumerating is not None:
            slots[slot] = (enumerating, {})
    masked = declared.masked_lm
    if masked is not None:
        strategy = _masked_lm(masked, adapter_for)
        for slot in declared.masked_lm_slots:
            config: StrategyConfig = {
                "resolver": strategy.resolver,
                "model_adapter": strategy.model_adapter,
                "beam_size": masked.beam_size,
                "fill_direction": masked.fill_direction,
                "top_k": masked.top_k,
            }
            if masked.custom_order is not None:
                config["custom_order"] = list(masked.custom_order)
            if masked.budget is not None:
                config["budget"] = masked.budget
            if slot in masked.per_slot_max_fills:
                config["max_fills"] = masked.per_slot_max_fills[slot]
            if slot in masked.per_slot_enforce_unique:
                config["enforce_unique"] = masked.per_slot_enforce_unique[slot]
            slots[slot] = ("mlm", config)
    default = _enumerating(declared.default, namespace, template)
    return MixedFillingStrategy(slots, default_strategy=default)


def _from_template(
    strategy: MLMFillingStrategy | MixedFillingStrategy,
    lexicon: Lexicon,
    template: Template,
    /,
) -> tuple[FilledTemplate, ...]:
    """Return the fillings a template-context strategy finds, rendered by bead."""
    renderer = DefaultRenderer()
    return tuple(
        FilledTemplate(
            template_id=str(template.id),
            template_name=template.name,
            slot_fillers=combination,
            rendered_text=renderer.render(
                template.template_string, combination, template.slots
            ),
            strategy_name=strategy.name,
            template_slots={
                name: slot.required for name, slot in template.slots.items()
            },
        )
        for combination in strategy.generate_from_template(
            template, [lexicon], template.language_code
        )
    )


def _fillings(
    filling: FillingSpec,
    lexicon: Lexicon,
    template: Template,
    namespace: UUID,
    adapter_for: Callable[[MaskedLMFilling], HuggingFaceMLMAdapter],
    /,
) -> tuple[FilledTemplate, ...]:
    """Return the fillings the declared strategy finds for one template."""
    if isinstance(filling, ConstraintFilling):
        try:
            return tuple(
                CSPFiller(lexicon, max_attempts=filling.max_attempts).fill(
                    template, language_code=template.language_code, count=filling.count
                )
            )
        except ConstraintUnsatisfiableError as refused:
            raise ValueError(str(refused)) from refused
    if template.constraints:
        msg = (
            f"the template {template.name!r} declares a template constraint, "
            "which only the csp strategy reads; declare the csp strategy to "
            "honor it"
        )
        raise ValueError(msg)
    enumerating = _enumerating(filling, namespace, template)
    if enumerating is not None:
        return tuple(
            StrategyFiller(lexicon, enumerating).fill(
                template, language_code=template.language_code
            )
        )
    match filling:
        case MaskedLMFilling() as declared:
            return _from_template(_masked_lm(declared, adapter_for), lexicon, template)
        case MixedFilling() as declared:
            mixed = _mixed(declared, namespace, template, adapter_for)
            return _from_template(mixed, lexicon, template)
        case StreamingFilling(max_combinations=bound):
            streaming = cast("_Streaming", import_module("bead.templates.streaming"))
            filler = streaming.StreamingFiller(lexicon, max_combinations=bound)
            return tuple(filler.stream(template, language_code=template.language_code))
    msg = f"unknown filling strategy {filling!r}"
    raise TypeError(msg)


def _metadata(value: object, name: str, /) -> MetadataValue:
    """Return one evaluated attribute as a bead metadata value."""
    if value is None or isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, list | tuple | set | frozenset):
        values = cast("Iterable[object]", value)
        return tuple(_metadata(member, name) for member in values)
    msg = f"attribute {name!r} evaluates to {type(value).__name__}, not a JSON value"
    raise ValueError(msg)


class _Rendered:
    """One combination of fillings an item renders, and its DSL context."""

    __slots__ = ("context", "filled", "slots")

    def __init__(
        self,
        templates: Sequence[Template],
        filled: Sequence[FilledTemplate],
        item_template: ItemTemplate,
        /,
    ) -> None:
        self.filled = {
            template.id: filling
            for template, filling in zip(templates, filled, strict=True)
        }
        self.slots = {
            template.id: {slot: item.id for slot, item in filling.slot_fillers.items()}
            for template, filling in zip(templates, filled, strict=True)
        }
        context: dict[str, DslValue] = {
            "item_template": item_template,
            "elements": {
                element.element_name: element for element in item_template.elements
            },
        }
        if len(templates) == 1:
            context["template"] = templates[0]
            context["slots"] = dict(filled[0].slot_fillers)
        else:
            context["templates"] = {
                template.name: {
                    "template": template,
                    "slots": dict(filling.slot_fillers),
                }
                for template, filling in zip(templates, filled, strict=True)
            }
        self.context = context


def _attributes(
    expressions: Mapping[str, str],
    rendered: _Rendered,
    evaluator: DSLEvaluator,
    /,
) -> dict[str, MetadataValue]:
    """Return the declared attributes of one combination of fillings."""
    held: dict[str, MetadataValue] = {}
    for name, expression in expressions.items():
        try:
            value = evaluator.evaluate(expression, rendered.context)
        except DSLError as refused:
            msg = f"attribute {name!r} cannot be read from a filling: {refused}"
            raise ValueError(msg) from refused
        held[name] = _metadata(value, name)
    return held


def _admitted(
    admit: str | None, rendered: _Rendered, evaluator: DSLEvaluator, /
) -> bool:
    """Return whether the declared admission expression admits a combination."""
    if admit is None:
        return True
    try:
        admitted = evaluator.evaluate(admit, rendered.context)
    except DSLError as refused:
        msg = f"the admission expression {admit!r} cannot be read: {refused}"
        raise ValueError(msg) from refused
    match admitted:
        case bool():
            return admitted
        case _:
            msg = (
                f"the admission expression {admit!r} yields {admitted!r}, not a "
                "truth value"
            )
            raise ValueError(msg)


def _bound(item_template: ItemTemplate, template: Template, /) -> ItemTemplate:
    """Return an item template whose filled-template elements render one template."""
    return item_template.with_(
        elements=tuple(
            element.with_(filled_template_ref_id=template.id)
            if element.is_template_ref
            else element
            for element in item_template.elements
        )
    )


def _plan(
    item_template: ItemTemplate, templates: Sequence[Template], /
) -> tuple[tuple[ItemTemplate, tuple[Template, ...]], ...]:
    """Return each item template to construct by and the templates it combines.

    Raises
    ------
    ValueError
        If some filled-template elements name a template and others do not,
        an element names an undeclared template, or a declared template is
        rendered by no element.
    """
    references = tuple(
        element.filled_template_ref_id
        for element in item_template.elements
        if element.is_template_ref
    )
    if not references:
        msg = f"the item template {item_template.name!r} renders no filled template"
        raise ValueError(msg)
    if all(reference is None for reference in references):
        return tuple(
            (_bound(item_template, template), (template,)) for template in templates
        )
    if any(reference is None for reference in references):
        msg = (
            f"some filled-template elements of {item_template.name!r} name the "
            "template they render and others do not"
        )
        raise ValueError(msg)
    by_id = {template.id: template for template in templates}
    named = tuple(dict.fromkeys(cast("tuple[UUID, ...]", references)))
    unknown = sorted(str(reference) for reference in named if reference not in by_id)
    unrendered = sorted(
        template.name for template in templates if template.id not in named
    )
    if unknown or unrendered:
        msg = (
            f"the item template {item_template.name!r} renders the undeclared "
            f"templates {unknown!r}, and no element renders the declared "
            f"templates {unrendered!r}"
        )
        raise ValueError(msg)
    return ((item_template, tuple(by_id[reference] for reference in named)),)


def _strategies(
    spec: ItemConstructionSpec, templates: Sequence[Template], /
) -> dict[UUID, FillingSpec]:
    """Return the declared strategy of every template.

    Raises
    ------
    ValueError
        If a per-template strategy names no declared template, or names one of
        several templates sharing a name.
    """
    names = [template.name for template in templates]
    unknown = sorted(name for name in spec.template_fillings if name not in names)
    shared = sorted(name for name in spec.template_fillings if names.count(name) > 1)
    if unknown or shared:
        msg = (
            f"the per-template strategies name the undeclared templates "
            f"{unknown!r} and the templates {shared!r}, which several declared "
            "templates share"
        )
        raise ValueError(msg)
    return {
        template.id: spec.template_fillings.get(template.name, spec.filling)
        for template in templates
    }


def construct_items(
    spec: ItemConstructionSpec,
    root: Path,
    /,
    *,
    generator_for: Callable[
        [LexiconGeneratorSpec], LexiconGenerator
    ] = lexicon_generator_for,
    adapter_for: Callable[[MaskedLMFilling], HuggingFaceMLMAdapter] = masked_lm_adapter,
    model_registry: ModelAdapterRegistry = default_registry,
) -> tuple[Item, ...]:
    """Construct the declared items through bead's item constructor.

    A combination of fillings the declared admission expression or the item
    template's constraints do not admit is not constructed.

    Parameters
    ----------
    spec : ItemConstructionSpec
        The validated document.
    root : Path
        The directory the document's files are relative to.
    generator_for : Callable[[LexiconGeneratorSpec], LexiconGenerator]
        Builds the generator each declared generator names.
    adapter_for : Callable[[MaskedLMFilling], HuggingFaceMLMAdapter]
        Loads the masked-LM adapter each declared masked-LM filling names.
    model_registry : ModelAdapterRegistry
        The bead registry of the models the item template's constraints call.

    Raises
    ------
    ValueError
        If a resource is refused, a template cannot be filled or reads a
        constraint its strategy ignores, an attribute cannot be read from a
        filling, or two fillings construct one identity.
    """
    templates = _templates(spec, root)
    lexicon = _lexicon(spec, root, generator_for)
    item_template = _item_template(spec, root)
    constraints = _constraints(spec, root, item_template)
    strategies = _strategies(spec, templates)
    plan = _plan(item_template, templates)
    fillings: dict[UUID, tuple[FilledTemplate, ...]] = {}
    for template in templates:
        try:
            fillings[template.id] = _fillings(
                strategies[template.id], lexicon, template, spec.namespace, adapter_for
            )
        except ValueError as refused:
            msg = f"the template {template.name!r} cannot be filled: {refused}"
            raise ValueError(msg) from refused

    constructor = ItemConstructor(model_registry, ModelOutputCache(backend="memory"))
    evaluator = DSLEvaluator()
    items: list[Item] = []
    for bound, combined in plan:
        for filled in product(*(fillings[template.id] for template in combined)):
            rendered = _Rendered(combined, filled, item_template)
            if not _admitted(spec.admit, rendered, evaluator):
                continue
            items.extend(
                item.with_(
                    id=item_identity(spec.namespace, rendered.slots),
                    item_metadata=_attributes(spec.attributes, rendered, evaluator),
                )
                for item in constructor.construct_items(
                    bound, rendered.filled, constraints
                )
            )

    identities = tuple(item.id for item in items)
    if len(set(identities)) != len(identities):
        msg = "two fillings construct the same item; each filling is drawn once"
        raise ValueError(msg)
    if not items:
        msg = "the item construction constructs no item"
        raise ValueError(msg)
    return tuple(items)

"""Gold constraint sources over a synthetic corpus and its interpretation.

Each source has one label domain. Gold derivations and the gold lexicon emit
syntactic categories. The logical-form source emits the unique meaning of a
surface form and classifies it by the logical constants it contains.
"""

from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from lofi._frozen import FrozenValue
from lofi.data._constraint_sources import (
    GoldDerivationSource,
    GoldLexiconSource,
    LogicalFormSource,
)
from lofi.lexicon import Entry, StaticLexicon
from lofi.registry import ComponentPlugin
from lofi.spec.constraint_policy import PolicyDomain
from lofi.spec.constraint_sources import (
    GoldConstraintSourceSettings,
    GoldDerivationConstraintSourceSpec,
    GoldLexiconConstraintSourceSpec,
    GoldLogicalFormConstraintSourceSpec,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi.data._selection import SelectionPolicy
    from lofi.data.synthetic._interpretation import InterpretationFunction
    from lofi.data.synthetic._semantics import Meaning
    from lofi.terms import Term

__all__ = ["ConstraintSourceContext", "UnservedPolicyError"]


class UnservedPolicyError(ValueError):
    """Raised when a constraint source cannot select under its declared policy."""


class ConstraintSourceContext(FrozenValue):
    """What a run supplies to every gold constraint source it builds.

    Parameters
    ----------
    syntactic : SelectionPolicy[str, str]
        The run's syntactic constraint-selection policy.
    semantic : SelectionPolicy[str, str]
        The run's semantic constraint-selection policy.
    lexicon : StaticLexicon[Term]
        The compiled lexicon of the corpus grammar.
    render : Callable[[Term], str]
        The canonical text of a category.
    interpretation : InterpretationFunction | None
        The interpretation the corpus was read under, or nothing when it was
        drawn without one.
    """

    __slots__ = ("_interpretation", "_lexicon", "_policies", "_render")

    _noun = "constraint-source context"

    def __init__(
        self,
        *,
        syntactic: SelectionPolicy[str, str],
        semantic: SelectionPolicy[str, str],
        lexicon: StaticLexicon[Term],
        render: Callable[[Term], str],
        interpretation: InterpretationFunction | None,
    ) -> None:
        object.__setattr__(
            self,
            "_policies",
            MappingProxyType(
                {PolicyDomain.SYNTACTIC: syntactic, PolicyDomain.SEMANTIC: semantic}
            ),
        )
        object.__setattr__(self, "_lexicon", lexicon)
        object.__setattr__(self, "_render", render)
        object.__setattr__(self, "_interpretation", interpretation)

    def policy(self, domain: PolicyDomain, /) -> SelectionPolicy[str, str]:
        """Return the run's selection policy for one label domain.

        Parameters
        ----------
        domain : PolicyDomain
            The domain a source declares it serves.
        """
        return self._policies[domain]

    @property
    def lexicon(self) -> StaticLexicon[Term]:
        """Return the compiled lexicon."""
        return self._lexicon

    @property
    def render(self) -> Callable[[Term], str]:
        """Return the canonical text of a category."""
        return self._render

    @property
    def interpretation(self) -> InterpretationFunction | None:
        """Return the corpus's interpretation, if it was drawn under one."""
        return self._interpretation


def _build_gold_derivation(
    settings: GoldConstraintSourceSettings, context: ConstraintSourceContext, /
) -> GoldDerivationSource[str, str]:
    """Build a source selecting category text from gold derivations.

    Parameters
    ----------
    settings : GoldConstraintSourceSettings
        The label domain the source serves.
    context : ConstraintSourceContext
        The run's selection policies.
    """
    return GoldDerivationSource(context.policy(settings.policy))


def _categories(context: ConstraintSourceContext, /) -> StaticLexicon[str]:
    """Return the compiled lexicon with each category as its canonical text.

    Parameters
    ----------
    context : ConstraintSourceContext
        The lexicon and the category renderer.
    """
    return StaticLexicon(
        Entry(entry.form, context.render(entry.category), weight=entry.weight)
        for form in context.lexicon.forms
        for entry in context.lexicon.entries(form)
    )


def _class_name(logical_class: str, arity: int | None, /) -> str:
    """Return the document name of one logical-constant class."""
    return logical_class if arity is None else f"{logical_class}/{arity}"


def _meaning_classes(meaning: Meaning, /) -> frozenset[str]:
    """Return the broad and arity-specific classes of a generated meaning."""
    exact = _class_name(meaning.logical_class, meaning.arity)
    return frozenset({meaning.logical_class, exact})


def _term_classes(
    interpretation: InterpretationFunction, term: Term, /
) -> frozenset[str]:
    """Return declared constant classes occurring in a listed logical form."""
    found: set[str] = set()
    pending = [term]

    while pending:
        current = pending.pop()

        if current.op.is_constant and current.payload is not None:
            symbol = interpretation.store.payload_name(current)

            for logical_class, members in interpretation.inventory.roles.items():
                arity = members.get(symbol)

                if arity is None:
                    continue

                found.add(logical_class)

                if arity > 0:
                    found.add(_class_name(logical_class, arity))

        pending.extend(current.args)

    return frozenset(found)


def _logical_forms(
    context: ConstraintSourceContext, /
) -> tuple[dict[tuple[str, ...], str], dict[tuple[str, ...], frozenset[str]]]:
    """Return each unambiguous surface form's meaning and constant classes.

    Parameters
    ----------
    context : ConstraintSourceContext
        The lexicon and the interpretation that gives every entry its meaning.

    Raises
    ------
    UnservedPolicyError
        If the corpus was drawn without an interpretation.
    """
    interpretation = context.interpretation

    if interpretation is None:
        msg = (
            "the corpus was drawn without an interpretation, so it has no "
            "logical forms to select under the semantic policy"
        )
        raise UnservedPolicyError(msg)

    semantic = interpretation.semantic_lexicon
    labels: dict[tuple[str, ...], dict[str, set[str]]] = {}

    for form in context.lexicon.forms:
        key = (form,)
        by_meaning = labels.setdefault(key, {})

        for entry in context.lexicon.entries(form):
            meanings = interpretation.entries[(form, entry.category)]
            metadata = (
                None
                if semantic is None
                else semantic.meanings[(context.render(entry.category), form)]
            )

            if metadata is not None and len(metadata) != len(meanings):
                msg = (
                    f"the generated assignment gives {form!r} as "
                    f"{context.render(entry.category)!r} {len(metadata)} classed "
                    f"meanings but the interpretation holds {len(meanings)}"
                )
                raise UnservedPolicyError(msg)

            for position, meaning in enumerate(meanings):
                rendered = interpretation.render(meaning)
                classes = (
                    _term_classes(interpretation, meaning)
                    if metadata is None
                    else _meaning_classes(metadata[position])
                )

                if not classes:
                    msg = (
                        f"the logical form {rendered!r} of {form!r} belongs to no "
                        "declared logical-constant class"
                    )
                    raise UnservedPolicyError(msg)

                by_meaning.setdefault(rendered, set()).update(classes)

    ambiguous = {
        form: tuple(sorted(meanings))
        for form, meanings in labels.items()
        if len(meanings) != 1
    }

    if ambiguous:
        msg = (
            "a logical-form source assigns one meaning per surface form, but "
            f"these forms are ambiguous: {ambiguous!r}"
        )
        raise UnservedPolicyError(msg)

    assignment = {form: next(iter(meanings)) for form, meanings in labels.items()}
    classes = {
        form: frozenset(next(iter(meanings.values())))
        for form, meanings in labels.items()
    }

    return assignment, classes


def _build_gold_lexicon(
    settings: GoldConstraintSourceSettings, context: ConstraintSourceContext, /
) -> GoldLexiconSource[str, str]:
    """Build a source over the syntactic categories of the lexicon.

    Parameters
    ----------
    settings : GoldConstraintSourceSettings
        The label domain the source serves.
    context : ConstraintSourceContext
        The run's selection policies, the lexicon and the category renderer.
    """
    return GoldLexiconSource(_categories(context), context.policy(settings.policy))


def _build_gold_logical_form(
    settings: GoldConstraintSourceSettings, context: ConstraintSourceContext, /
) -> LogicalFormSource[str, str]:
    """Build a source over the interpreted lexicon's logical forms."""
    assignment, classes = _logical_forms(context)

    return LogicalFormSource(
        assignment,
        classes,
        context.policy(settings.policy),
    )


GOLD_DERIVATION_PLUGIN: Final = ComponentPlugin(
    spec=GoldDerivationConstraintSourceSpec, build=_build_gold_derivation
)
"""The gold-derivation constraint source's plugin."""

GOLD_LEXICON_PLUGIN: Final = ComponentPlugin(
    spec=GoldLexiconConstraintSourceSpec, build=_build_gold_lexicon
)
"""The gold-lexicon constraint source's plugin."""

GOLD_LOGICAL_FORM_PLUGIN: Final = ComponentPlugin(
    spec=GoldLogicalFormConstraintSourceSpec, build=_build_gold_logical_form
)
"""The gold-logical-form constraint source's plugin."""

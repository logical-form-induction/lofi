"""What a run supplies to every neural component it builds through a plugin."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, cast, override

from lofi._frozen import FrozenValue, Unpicklable
from lofi.nn.type_directed import TypeDirectedLanguage
from lofi.registry import plugin_for

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi._types import ComponentId
    from lofi.judge import ActionSet, RuleSet
    from lofi.lf import SurfaceSyntax
    from lofi.nn.codec import NeuralCodecView
    from lofi.nn.loss._approximation import ApproximationBound
    from lofi.nn.policy import DevicePolicy
    from lofi.nn.readouts import ProofTermMeaning
    from lofi.nn.tokens.embedder import TokenVocabulary
    from lofi.nn.tokens.pretrained import PretrainedLoader
    from lofi.registry import ComponentPlugin
    from lofi.spec import ConfiguredSpec
    from lofi.spec.language import LanguageCodec
    from lofi.terms import ChoiceIndex, TargetSet, Term

__all__ = ["BuildContext", "build_component"]


class _LearnerLanguage(TypeDirectedLanguage, Protocol):
    """The learner-language capability neural construction consumes."""

    @property
    @override
    def codec(self) -> LanguageCodec:
        """Return the language-owned codecs and payload choices."""
        ...

    @property
    def syntax(self) -> SurfaceSyntax:
        """Return the language's compiled surface syntax."""
        ...


class BuildContext(FrozenValue, Unpicklable):
    """The run-wide resources and wired producers one component is built with.

    Parameters
    ----------
    component : ComponentId
        The handle the run assigned the component.
    policy : DevicePolicy
        Where the run's tensors are placed and what precision they are stored in.
    codec : NeuralCodecView
        The tensor view of the term codec every component of the run shares.
    language : _LearnerLanguage
        The exact learner language that owns the run's codecs and choices.
    vocabulary : TokenVocabulary
        The run's token-to-row assignment.
    actions : ActionSet
        The ordered action inventory of the grammar the run induces.
    rules : RuleSet[Term]
        The grammar's exact rule inventory.
    categories : tuple[Term, ...]
        The category representatives the parser may place in chart states.
    producers : Mapping[str, object]
        The built components the component reads, by the port or role it names
        each one under.
    targets : Mapping[str, TargetSet[Term]]
        The target sets the component may score, by source name.
    loader : PretrainedLoader | None
        The backend pretrained weights are loaded through, or None for a run
        whose components load no pretrained weights.
    meaning : ProofTermMeaning | None
        The run's interpretation capability when its readouts construct meanings
        compositionally.
    """

    __slots__ = (
        "_actions",
        "_approximation",
        "_categories",
        "_codec",
        "_component",
        "_language",
        "_loader",
        "_meaning",
        "_policy",
        "_producers",
        "_rules",
        "_targets",
        "_vocabulary",
    )

    _noun = "build context"
    _because = "holds built modules and a live device policy"

    def __init__(
        self,
        *,
        component: ComponentId,
        policy: DevicePolicy,
        language: _LearnerLanguage,
        codec: NeuralCodecView,
        vocabulary: TokenVocabulary,
        actions: ActionSet,
        rules: RuleSet[Term],
        categories: tuple[Term, ...],
        producers: Mapping[str, object],
        targets: Mapping[str, TargetSet[Term]],
        loader: PretrainedLoader | None,
        meaning: ProofTermMeaning | None = None,
        approximation: ApproximationBound | None = None,
    ) -> None:
        object.__setattr__(self, "_component", component)
        object.__setattr__(self, "_policy", policy)
        object.__setattr__(self, "_language", language)
        object.__setattr__(self, "_codec", codec)
        object.__setattr__(self, "_vocabulary", vocabulary)
        object.__setattr__(self, "_actions", actions)
        object.__setattr__(self, "_rules", rules)
        object.__setattr__(self, "_categories", tuple(categories))
        object.__setattr__(self, "_producers", MappingProxyType(dict(producers)))
        object.__setattr__(self, "_targets", MappingProxyType(dict(targets)))
        object.__setattr__(self, "_loader", loader)
        object.__setattr__(self, "_meaning", meaning)
        object.__setattr__(self, "_approximation", approximation)

    @property
    def component(self) -> ComponentId:
        """Return the handle the run assigned the component."""
        return self._component

    @property
    def policy(self) -> DevicePolicy:
        """Return the run's placement and precision policy."""
        return self._policy

    @property
    def codec(self) -> NeuralCodecView:
        """Return the run's shared codec view."""
        return self._codec

    @property
    def language(self) -> _LearnerLanguage:
        """Return the learner language that owns the run's symbolic choices."""
        return self._language

    @property
    def choices(self) -> ChoiceIndex:
        """Return the learner language's payload-aware alphabet."""
        return self._language.codec.choices

    @property
    def vocabulary(self) -> TokenVocabulary:
        """Return the run's token vocabulary."""
        return self._vocabulary

    @property
    def actions(self) -> ActionSet:
        """Return the grammar's action inventory."""
        return self._actions

    @property
    def rules(self) -> RuleSet[Term]:
        """Return the grammar's exact rule inventory."""
        return self._rules

    @property
    def categories(self) -> tuple[Term, ...]:
        """Return the category representatives in chart-axis order."""
        return self._categories

    @property
    def producers(self) -> Mapping[str, object]:
        """Return the built components the component reads, by port or role."""
        return self._producers

    @property
    def targets(self) -> Mapping[str, TargetSet[Term]]:
        """Return the target sets the component may score, by source name."""
        return self._targets

    @property
    def loader(self) -> PretrainedLoader:
        """Return the pretrained-weight backend.

        Raises
        ------
        LookupError
            If the run declares no artifacts cache to load pretrained weights
            through.
        """
        if self._loader is None:
            msg = (
                "the run declares no artifacts cache, so no pretrained weights "
                "are loaded"
            )
            raise LookupError(msg)

        return self._loader

    @property
    def meaning(self) -> ProofTermMeaning | None:
        """Return the run's proof-term interpretation, where one is declared."""
        return self._meaning

    @property
    def approximation(self) -> ApproximationBound | None:
        """Return the run's resolved residual-mass budget, where declared."""
        return self._approximation

    def producer(self, port: str, /) -> object:
        """Return the built component wired to one port.

        Parameters
        ----------
        port : str
            The port or role the component names its producer under.

        Returns
        -------
        object
            The producer.

        Raises
        ------
        KeyError
            If no producer is wired to the port.
        """
        try:
            return self._producers[port]

        except KeyError:
            known = ", ".join(sorted(self._producers)) or "no producer"
            msg = f"{self._component!r} reads {port!r}, and the run wired {known}"
            raise KeyError(msg) from None

    def for_component(
        self, component: ComponentId, producers: Mapping[str, object], /
    ) -> BuildContext:
        """Return this context for another component and its producers.

        Parameters
        ----------
        component : ComponentId
            The handle of the component to build.
        producers : Mapping[str, object]
            Its producers, by port or role.

        Returns
        -------
        BuildContext
            A context sharing every run-wide resource with this one.
        """
        return BuildContext(
            component=component,
            policy=self._policy,
            language=self._language,
            codec=self._codec,
            vocabulary=self._vocabulary,
            actions=self._actions,
            rules=self._rules,
            categories=self._categories,
            producers=producers,
            targets=self._targets,
            loader=self._loader,
            meaning=self._meaning,
            approximation=self._approximation,
        )

    def with_targets(self, targets: Mapping[str, TargetSet[Term]], /) -> BuildContext:
        """Return this context scoring other target sets.

        Parameters
        ----------
        targets : Mapping[str, TargetSet[Term]]
            The target sets, by source name.

        Returns
        -------
        BuildContext
            A context differing from this one in its target sets alone.
        """
        return BuildContext(
            component=self._component,
            policy=self._policy,
            language=self._language,
            codec=self._codec,
            vocabulary=self._vocabulary,
            actions=self._actions,
            rules=self._rules,
            categories=self._categories,
            producers=self._producers,
            targets=targets,
            loader=self._loader,
            meaning=self._meaning,
            approximation=self._approximation,
        )

    def with_codec(self, codec: NeuralCodecView, /) -> BuildContext:
        """Return this context for another represented sort's codec view.

        Parameters
        ----------
        codec : NeuralCodecView
            The exact codec view owned by the represented space.

        Returns
        -------
        BuildContext
            A context differing from this one in its codec view alone.
        """
        return BuildContext(
            component=self._component,
            policy=self._policy,
            language=self._language,
            codec=codec,
            vocabulary=self._vocabulary,
            actions=self._actions,
            rules=self._rules,
            categories=self._categories,
            producers=self._producers,
            targets=self._targets,
            loader=self._loader,
            meaning=self._meaning,
            approximation=self._approximation,
        )

    def with_loader(self, loader: PretrainedLoader, /) -> BuildContext:
        """Return this context loading pretrained weights through another backend.

        Parameters
        ----------
        loader : PretrainedLoader
            The backend.

        Returns
        -------
        BuildContext
            A context differing from this one in its loader alone.
        """
        return BuildContext(
            component=self._component,
            policy=self._policy,
            language=self._language,
            codec=self._codec,
            vocabulary=self._vocabulary,
            actions=self._actions,
            rules=self._rules,
            categories=self._categories,
            producers=self._producers,
            targets=self._targets,
            loader=loader,
            meaning=self._meaning,
            approximation=self._approximation,
        )


def build_component[V](declaration: ConfiguredSpec[V], context: BuildContext) -> object:
    """Build one declared component through the plugin its family lists.

    Parameters
    ----------
    declaration : ConfiguredSpec[V]
        A validated component declaration.
    context : BuildContext
        What the run supplies to the component.

    Returns
    -------
    object
        The built component, placed under the context's device policy.

    Raises
    ------
    UnknownTagError
        If the declaration's family lists no plugin under its tag.
    RegistryError
        If the entry names something other than the declaration's plugin.
    """
    plugin = cast("ComponentPlugin[V, BuildContext, object]", plugin_for(declaration))

    return plugin.construct(declaration, context)

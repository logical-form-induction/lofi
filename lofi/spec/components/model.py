"""The model section of a run and the component families it declares."""

from enum import Enum, StrEnum, auto
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, Protocol, cast, override

import didactic.api as dx

from lofi._checks import ensure_name, ensure_positive
from lofi.spec._ports import Dependency
from lofi.spec.components.base import ComponentSpec, _in_role_order
from lofi.spec.components.data import TrainingBatchKind
from lofi.spec.grammar import GrammarDecl
from lofi.spec.language import LANGUAGE_FIELD, LanguageDecl
from lofi.spec.semantics import LearnerInterpretationSpec
from lofi.spec.spaces import SpaceSpec

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.terms import TermCodec

_DEFAULT_TOKEN_ENCODER_TRAINABLE: Final = False

__all__ = [
    "ActionsSpec",
    "ControllerSpec",
    "DataModelLanguageSpec",
    "DecoderCapability",
    "GroundingSpec",
    "InlineModelLanguageSpec",
    "ModelLanguageSpec",
    "ModelRole",
    "ModelSection",
    "PairScorerSpec",
    "ParameterCapacitySpec",
    "ParserSpec",
    "ReadoutSpec",
    "SpanDecoderSpec",
    "SubwordPoolerSpec",
    "TermDecoderSpec",
    "TermEncoderSpec",
    "TokenEncoderSpec",
]


class ModelLanguageSpec(dx.TaggedUnion, discriminator="from", extra="forbid"):
    """Declare where the model's learner language comes from."""

    def declared_language(self) -> LanguageDecl | None:
        """Return the inline language, or nothing when data supplies it."""
        return None


class DataModelLanguageSpec(ModelLanguageSpec, extra="forbid"):
    """Use the one source assembly shared by every selected dataset."""

    __annotations__ = {"from": Literal["data"]}
    locals()["from"] = "data"


class InlineModelLanguageSpec(ModelLanguageSpec, extra="forbid"):
    """Declare the learner language independently of the data source.

    The interpretation is a neutral boundary declaration rather than a dataset
    implementation detail.
    """

    language: LanguageDecl = LANGUAGE_FIELD
    interpretation = None

    __annotations__ = {
        "from": Literal["inline"],
        "language": LanguageDecl,
        "grammar": GrammarDecl,
        "interpretation": LearnerInterpretationSpec | None,
    }
    locals()["from"] = "inline"

    if TYPE_CHECKING:
        grammar: GrammarDecl
        interpretation: LearnerInterpretationSpec | None

    @override
    def declared_language(self) -> LanguageDecl:
        """Return the model-owned learner language."""
        return self.language


class ParameterCapacitySpec(ComponentSpec, extra="forbid"):
    """Declare an exact total parameter-count budget for the built model."""

    kind: Literal["parameter-count"] = "parameter-count"
    target: int

    @dx.validates("target")
    @classmethod
    def _target_is_positive(cls, value: int) -> int:
        """Return a positive total parameter target."""
        return ensure_positive("a parameter-capacity target", value)


class DecoderCapability(Enum):
    """Name one decoder protocol required by a configured consumer."""

    FRONTIER = auto()
    TEACHER_FORCED = auto()
    MARGINAL = auto()


class _ControllerInputDeclaration(Protocol):
    """Expose the chart representation a controller declaration reads."""

    input: str


class TermEncoderSpec(
    ComponentSpec, dx.TaggedUnion, discriminator="kind", extra="forbid"
):
    """Declares a term encoder."""

    @classmethod
    def family(cls) -> str:
        """Return the family whose entry-point group lists the plugins."""
        return "term_encoder"


class ModelRole(StrEnum):
    """The role each single component of the model section is wired under."""

    TOKENS_ENCODER = "token_encoder"
    SUBWORD_POOLER = "subword_pooler"
    CONTROLLER = "controller"
    PARSER = "parser"
    GROUNDING = "grounding"


class TokenEncoderSpec(
    ComponentSpec, dx.TaggedUnion, discriminator="kind", extra="forbid"
):
    """Declares a token encoder."""

    @classmethod
    def family(cls) -> str:
        """Return the family whose entry-point group lists the plugins."""
        return "token_encoder"


class SubwordPoolerSpec(
    ComponentSpec, dx.TaggedUnion, discriminator="kind", extra="forbid"
):
    """Declares a subword pooler."""

    @classmethod
    def family(cls) -> str:
        """Return the family whose entry-point group lists the plugins."""
        return "subword_pooler"


class TermDecoderSpec(
    ComponentSpec, dx.TaggedUnion, discriminator="kind", extra="forbid"
):
    """Declares a decoder over terms of a represented space."""

    capabilities = frozenset[DecoderCapability]()

    def capabilities_for(self, codec: TermCodec, /) -> frozenset[DecoderCapability]:
        """Return the decoder protocols available for the represented codec."""
        del codec
        return self.capabilities

    @classmethod
    def family(cls) -> str:
        """Return the family whose entry-point group lists the plugins."""
        return "term_decoder"


class ControllerSpec(
    ComponentSpec, dx.TaggedUnion, discriminator="kind", extra="forbid"
):
    """Declares a controller and its optional differentiable raising treatment."""

    __annotations__ = {"raise": Literal["none", "soft"]}
    locals()["raise"] = "none"

    def raising(self) -> Literal["none", "soft"]:
        """Return the declared type-raising treatment."""
        return cast("Literal['none', 'soft']", getattr(self, "raise"))

    @override
    def publications(self) -> frozenset[str]:
        """Return the whole controller lent to a parser as its policy."""
        return frozenset({"policy"})

    def encoder_dependency(self, encoder: str, /) -> Dependency:
        """Return the edge from the encoder whose category cells the controller reads.

        Parameters
        ----------
        encoder : str
            The term-encoder role of the represented category space.
        """
        return Dependency(encoder, "cells")

    @classmethod
    def family(cls) -> str:
        """Return the family whose entry-point group lists the plugins."""
        return "controller"


class GroundingSpec(
    ComponentSpec, dx.TaggedUnion, discriminator="kind", extra="forbid"
):
    """Declares the inference grounding published by the model."""

    @classmethod
    def family(cls) -> str:
        """Return the grounding plugin family."""
        return "grounding"

    def nested_components(self) -> Mapping[str, ComponentSpec]:
        """Return independently registered components nested by this grounding."""
        return {}

    def form_readouts(self) -> tuple[str, str] | None:
        """Return the premise and hypothesis form readouts, when consumed."""
        return None

    def supported_signals(self) -> frozenset[str]:
        """Return the inference-training signals this grounding supports."""
        return frozenset({"likelihood", "reward"})

    def reward_declared(self) -> bool:
        """Return whether this grounding declares proof payoffs."""
        return False

    def validate_readouts(self, readouts: Mapping[str, ReadoutSpec], /) -> None:
        """Validate grounding-specific use of the model's readout roles."""


class PairScorerSpec(
    ComponentSpec, dx.TaggedUnion, discriminator="kind", extra="forbid"
):
    """Declares the evidence source used by a pairwise grounding."""

    @classmethod
    def family(cls) -> str:
        """Return the pair-scorer plugin family."""
        return "pair_scorer"


class ReadoutSpec(ComponentSpec, dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declares a chart-cell readout."""

    space: str

    @dx.validates("space")
    @classmethod
    def _space_is_named(cls, value: str) -> str:
        """Return the nonempty represented-space role this readout observes."""
        return ensure_name("a readout's space", value)

    @classmethod
    def family(cls) -> str:
        """Return the family whose entry-point group lists the plugins."""
        return "readout"

    def estimate_kind(self) -> str | None:
        """Return how complete terms are estimated, or none when not decoded."""
        return None

    def published_masses(self) -> frozenset[str]:
        """Return the ids of the target-mass pairs this readout publishes."""
        return frozenset()


class ParserSpec(ComponentSpec, dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declares a parser."""

    @classmethod
    def family(cls) -> str:
        """Return the family whose entry-point group lists the plugins."""
        return "chart"

    def controller_inputs(self) -> frozenset[str]:
        """Return the controller representations this parser can provide."""
        return frozenset()


class SpanDecoderSpec(
    ComponentSpec, dx.TaggedUnion, discriminator="kind", extra="forbid"
):
    """Declares a decoder of the words a chart span covers."""

    @classmethod
    def family(cls) -> str:
        """Return the family whose entry-point group lists the plugins."""
        return "span_decoder"


class ActionsSpec(dx.Model, extra="forbid"):
    """Declares the controller actions that apply no rule of the grammar.

    Parameters
    ----------
    identity : str
        The name of the action that leaves a cell's category as it is.
    null : str | None
        The name of the action a controller weighs when no combinator applies to
        a pair of cells, or None for a controller with no such column.
    """

    identity: str = "identity"
    null: str | None = None

    @dx.model_validator()
    def _named_actions(self) -> ActionsSpec:
        """Validate the two action names.

        Returns
        -------
        ActionsSpec
            This declaration after validation.

        Raises
        ------
        ValueError
            If either name is empty, or the null action's name is the identity's.
        """
        if not self.identity:
            msg = "the identity action needs a name"
            raise ValueError(msg)

        if self.null == "":
            msg = "a null action needs a name"
            raise ValueError(msg)

        if self.null == self.identity:
            msg = f"the null action and the identity are both named {self.null!r}"
            raise ValueError(msg)

        return self


class ModelSection(dx.Model, extra="forbid"):
    """Declares model components.

    Parameters
    ----------
    language : ModelLanguageSpec
        The learner language and grammar the model represents.
    capacity : ParameterCapacitySpec | None
        Optional exact target for the built model's total parameter count.
    parser : ParserSpec | None
        Optional parser component.
    token_encoder : TokenEncoderSpec | None
        Optional token encoder.
    subword_pooler : SubwordPoolerSpec | None
        Optional subword pooler.
    encoders : dict[str, TermEncoderSpec]
        Term encoders keyed by role.
    controller : ControllerSpec | None
        Optional controller.
    decoders : dict[str, TermDecoderSpec]
        Term decoders keyed by role.
    spaces : dict[str, SpaceSpec]
        Represented sorts bound to named encoders and decoders.
    readouts : dict[str, ReadoutSpec]
        Chart-cell readouts keyed by role.
    span_decoders : dict[str, SpanDecoderSpec]
        Span decoders keyed by role.
    actions : ActionsSpec
        The controller actions besides one per grammar rule.
    """

    language: ModelLanguageSpec
    capacity: ParameterCapacitySpec | None = None
    parser: ParserSpec | None = None
    token_encoder: TokenEncoderSpec | None = None
    subword_pooler: SubwordPoolerSpec | None = None
    controller: ControllerSpec | None = None
    grounding: GroundingSpec | None = None
    encoders: dict[str, TermEncoderSpec] = dx.field(default_factory=dict)
    decoders: dict[str, TermDecoderSpec] = dx.field(default_factory=dict)
    spaces: dict[str, SpaceSpec] = dx.field(default_factory=dict)
    readouts: dict[str, ReadoutSpec] = dx.field(default_factory=dict)
    span_decoders: dict[str, SpanDecoderSpec] = dx.field(default_factory=dict)
    actions: ActionsSpec = dx.field(default_factory=ActionsSpec)

    @dx.validates("encoders", "decoders", "readouts", "span_decoders")
    @classmethod
    def _order_roles[S: ComponentSpec](cls, roles: dict[str, S]) -> dict[str, S]:
        """Return components in role order.

        Parameters
        ----------
        roles : dict[str, S]
            Components keyed by role.
        """
        return _in_role_order(roles)

    @dx.validates("spaces")
    @classmethod
    def _order_spaces(cls, spaces: dict[str, SpaceSpec]) -> dict[str, SpaceSpec]:
        """Return represented spaces in role order."""
        return dict(sorted(spaces.items()))

    @dx.model_validator()
    def _distinct_roles(self) -> ModelSection:
        """Validate that each role names one component.

        Returns
        -------
        ModelSection
            This declaration after validation.

        Raises
        ------
        ValueError
            If a keyed component takes the name of a single component's field, or
            a role is declared in two keyed sections.
        """
        taken = set(self._single())

        if self.capacity is not None:
            taken.add("capacity")
        grounding = self.grounding

        if grounding is not None:
            nested = grounding.nested_components()
            reused = sorted(taken & nested.keys())

            if reused:
                msg = f"grounding components reuse other component roles: {reused}"
                raise ValueError(msg)

            taken |= nested.keys()

        for name, section in self._keyed():
            reused = sorted(taken & section.keys())

            if reused:
                msg = f"{name} reuse other component roles: {reused}"
                raise ValueError(msg)

            taken |= section.keys()

        reused = sorted(taken & self.spaces.keys())

        if reused:
            msg = f"spaces reuse component roles: {reused}"
            raise ValueError(msg)

        return self

    @dx.model_validator()
    def _controller_input_composes(self) -> ModelSection:
        """Refuse a controller input the selected parser cannot provide."""
        controller = self.controller
        parser = self.parser

        if controller is None:
            return self

        input_source = cast("_ControllerInputDeclaration", controller).input

        if controller.raising() == "soft" and input_source != "inside":
            msg = "model.controller: soft raising requires input 'inside', not 'span'"
            raise ValueError(msg)

        if parser is None:
            return self

        accepted = parser.controller_inputs()

        if not accepted:
            return self

        if input_source not in accepted:
            msg = (
                f"model.controller: input {input_source!r} is incompatible with "
                f"the {str(parser.kind)!r} parser; it provides controller inputs "
                f"{sorted(accepted)}"
            )
            raise ValueError(msg)

        return self

    @dx.model_validator()
    def _spaces_resolve(self) -> ModelSection:
        """Validate every space against the named component declarations."""
        sorts: dict[str, str] = {}

        for name, space in self.spaces.items():
            if space.encoder not in self.encoders:
                msg = (
                    f"space {name!r} names encoder {space.encoder!r}; declared "
                    f"encoders: {sorted(self.encoders)}"
                )
                raise ValueError(msg)

            named_decoders = {space.identity}

            if space.combinators is not None:
                named_decoders.update(space.combinators.decoders.values())

            missing = sorted(named_decoders - self.decoders.keys())

            if missing:
                msg = f"space {name!r} names undeclared decoders: {missing}"
                raise ValueError(msg)

            first = sorts.get(space.sort)

            if first is not None:
                msg = (
                    f"spaces {first!r} and {name!r} both represent sort {space.sort!r}"
                )
                raise ValueError(msg)

            sorts[space.sort] = name

        for name, readout in self.readouts.items():
            if readout.space not in self.spaces:
                msg = (
                    f"readout {name!r} names space {readout.space!r}; declared "
                    f"spaces: {sorted(self.spaces)}"
                )
                raise ValueError(msg)

        for name, space in self.spaces.items():
            relation = space.relation

            if relation is None:
                continue

            if relation.space not in self.spaces:
                msg = f"space {name!r} relates to undeclared space {relation.space!r}"
                raise ValueError(msg)

            if relation.space == name:
                msg = f"space {name!r} cannot declare a relation to itself"
                raise ValueError(msg)

        return self

    def fitted_batches(
        self, required: frozenset[TrainingBatchKind], /
    ) -> frozenset[TrainingBatchKind]:
        """Return the batch kinds a trainer's requirement asks of this model's data.

        A grounded model fits its parser on labeled inference pairs, so a
        trainer's sentence requirement is met by inference-pair batches.

        Parameters
        ----------
        required : frozenset[TrainingBatchKind]
            The batch kinds a trainer requires of one dataset.

        Returns
        -------
        frozenset[TrainingBatchKind]
            The batch kinds the dataset must yield for this model.
        """
        if self.grounding is None or TrainingBatchKind.SENTENCES not in required:
            return required

        return (required - {TrainingBatchKind.SENTENCES}) | {
            TrainingBatchKind.INFERENCE_PAIRS
        }

    def components(self) -> dict[str, ComponentSpec]:
        """Return every declared component, keyed by the role it fills.

        Returns
        -------
        dict[str, ComponentSpec]
            The declared single components under their field names, then the
            named encoders, term decoders, span decoders and readouts under their
            declared roles.
        """
        declared: dict[str, ComponentSpec] = {
            role: component
            for role, component in self._single().items()
            if component is not None
        }
        grounding = self.grounding

        if grounding is not None:
            declared.update(grounding.nested_components())

        for _, section in self._keyed():
            declared.update(section)

        return declared

    def base_components(self) -> dict[str, ComponentSpec]:
        """Return components wired independently of represented spaces."""
        declared: dict[str, ComponentSpec] = {
            role: component
            for role, component in self._single().items()
            if component is not None and role != ModelRole.GROUNDING
        }
        declared.update(self.span_decoders)

        return declared

    def checkpointed_components(self) -> Mapping[str, ComponentSpec]:
        """Return the stateful components keyed by their checkpoint role.

        Returns
        -------
        Mapping[str, ComponentSpec]
            Every declared component whose parameters a checkpoint stores, in
            sorted role order.
        """
        return MappingProxyType(
            {
                role: declaration
                for role, (_, declaration) in self._checkpointed().items()
            }
        )

    def checkpointed_paths(self) -> Mapping[str, str]:
        """Return the document path of each stateful component by role.

        Returns
        -------
        Mapping[str, str]
            The dotted structural path under which each checkpointed role is
            declared, in sorted role order.
        """
        return MappingProxyType(
            {role: path for role, (path, _) in self._checkpointed().items()}
        )

    def _checkpointed(self) -> dict[str, tuple[str, ComponentSpec]]:
        """Return each checkpointed role's structural path and declaration."""
        optional = {
            "capacity": ("model.capacity", self.capacity),
            "token_encoder": ("model.token_encoder", self.token_encoder),
            "subword_pooler": ("model.subword_pooler", self.subword_pooler),
            "controller": ("model.controller", self.controller),
            "parser": ("model.parser", self.parser),
            "grounding": ("model.grounding", self.grounding),
        }
        declarations: dict[str, tuple[str, ComponentSpec]] = {
            role: (path, declaration)
            for role, (path, declaration) in optional.items()
            if declaration is not None
        }
        grounding = self.grounding

        if grounding is not None:
            declarations.update(
                {
                    role: (f"model.grounding.{role}", declaration)
                    for role, declaration in grounding.nested_components().items()
                }
            )

        for field, section in (
            ("encoders", self.encoders),
            ("decoders", self.decoders),
            ("span_decoders", self.span_decoders),
        ):
            declarations.update(
                {
                    role: (f"model.{field}.{role}", declaration)
                    for role, declaration in section.items()
                }
            )

        declarations.update(
            {
                role: (f"model.spaces.{role}.relation.mapping", relation.mapping)
                for role, space in self.spaces.items()
                if (relation := space.relation) is not None
                and relation.mapping is not None
            }
        )

        return {
            role: declaration
            for role, declaration in sorted(declarations.items())
            if declaration[1].checkpointed()
        }

    def _single(self) -> dict[str, ComponentSpec | None]:
        """Return the single components under their field names."""
        return {
            ModelRole.TOKENS_ENCODER: self.token_encoder,
            ModelRole.SUBWORD_POOLER: self.subword_pooler,
            ModelRole.CONTROLLER: self.controller,
            ModelRole.PARSER: self.parser,
            ModelRole.GROUNDING: self.grounding,
        }

    def _keyed(self) -> tuple[tuple[str, Mapping[str, ComponentSpec]], ...]:
        """Return each keyed section under the name an error reports it by."""
        return (
            ("encoders", self.encoders),
            ("decoders", self.decoders),
            ("span decoders", self.span_decoders),
            ("readouts", self.readouts),
        )

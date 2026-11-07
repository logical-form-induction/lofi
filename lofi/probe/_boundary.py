"""Specification-boundary codecs for probing site maps and declared probes."""

import enum
from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    NamedTuple,
    Protocol,
    Self,
    cast,
    override,
)

import didactic.api as dx

from lofi._checks import ensure_finite, ensure_non_negative, ensure_positive
from lofi.nn.sites import (
    Access,
    ComponentId,
    ComponentKind,
    DTypeClass,
    Role,
    Site,
    SiteMap,
    SiteSpec,
)
from lofi.probe._acdc import AbsoluteThreshold, PruningThreshold, RelativeThreshold
from lofi.probe._das import (
    AlignmentObjective,
    Annealing,
    BoundlessSearch,
    FixedRankSearch,
    Schedule,
    SearchStrategy,
)
from lofi.probe._description import VOITA_BLOCKS, BlockSchedule
from lofi.probe._external import ExternalSite, ExternalSiteMap
from lofi.probe._interventions import MeanFrom, Noise, Resample, Shuffle, Zero
from lofi.probe._node_vocabulary import NodeVocabularyArm
from lofi.probe._roles import ensure_publishes
from lofi.probe._selectivity import ClassifierProbe, ControlTask, ProbeCapacity
from lofi.spec import ProbeSpec
from lofi.spec.artifacts import PretrainedSourceSpec, source_artifact, source_spec
from lofi.stats import Dependence, LabelPermutation

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, MutableSequence

    from torch import Tensor

    from lofi.probe._interventions import Intervention
    from lofi.probe._probing import InterventionSource
    from lofi.probe._values import Trace

_DEFAULT_SEED: Final = 0
_DEFAULT_NULL_REPLICATES: Final = 99
_DEFAULT_BOOTSTRAP_REPLICATES: Final = 999
_DEFAULT_BOOTSTRAP_LEVEL: Final = 0.95
_DEFAULT_FDR_ALPHA: Final = 0.05

__all__ = [
    "ACDCThresholdDecl",
    "AbsoluteThresholdDecl",
    "BoundlessSearchDecl",
    "CausalProbeSettings",
    "CausalProbeSpec",
    "CausalSiteDecl",
    "CircuitProbeSettings",
    "CircuitProbeSpec",
    "ClassifierProbeSettings",
    "ClassifierProbeSpec",
    "DASSearchDecl",
    "DiagnosticsProbeSettings",
    "DiagnosticsProbeSpec",
    "ExternalModelSpec",
    "ExternalSiteDecl",
    "ExternalSiteMapIso",
    "FixedRankSearchDecl",
    "InterventionDecl",
    "InterventionProbeSettings",
    "InterventionProbeSpec",
    "MeanDecl",
    "NodeVocabularyArmDecl",
    "NoiseDecl",
    "ProbeSiteDecl",
    "RelativeThresholdDecl",
    "ResampleDecl",
    "SaeProbeSettings",
    "SaeProbeSpec",
    "SearchContext",
    "ShuffleDecl",
    "SiteDecl",
    "SiteInterventionDecl",
    "SiteMapIso",
    "SiteMapSpec",
    "SiteReader",
    "SpanLabels",
    "ZeroDecl",
    "lower",
]

type SiteReader = Callable[[Tensor], None]
"""Callback that stores the tensor published at one site."""


class SiteDecl(dx.Model, extra="forbid"):
    """Stores a declared probing site.

    Parameters
    ----------
    role : str
        What the value at the point plays, by the name of the ``Role`` member.
    key : tuple[int | str, ...]
        Which point among those the component publishes for that role.
    axes : tuple[str, ...]
        The name of each axis, outermost first.
    dtype_class : str
        What the numbers there mean, by the value a ``DTypeClass`` member spells.
    differentiable : bool
        Whether a gradient flows through the value at the point.
    access : tuple[str, ...]
        What the point permits, by the names of the ``Access`` members.
    """

    role: str
    key: tuple[int | str, ...] = ()
    axes: tuple[str, ...] = ()
    dtype_class: str
    differentiable: bool
    access: tuple[str, ...]


class SiteMapSpec(dx.Model, extra="forbid"):
    """Stores a declared collection of probing sites.

    Parameters
    ----------
    component : str
        The component every point belongs to.
    kind : str
        The value of the component kind whose role inventory applies.
    sites : tuple[SiteDecl, ...]
        One entry per point the component offers, in any order.
    """

    component: str
    kind: str
    sites: tuple[SiteDecl, ...] = ()


def _kind(spelled: str) -> ComponentKind:
    """Return the component kind a declaration spells.

    Parameters
    ----------
    spelled : str
        The value of the component-kind member.

    Returns
    -------
    ComponentKind
        The member.

    Raises
    ------
    ValueError
        If no component kind spells that value.
    """
    try:
        return ComponentKind(spelled)

    except ValueError as error:
        declared = ", ".join(member.value for member in ComponentKind)
        msg = f"no component kind is spelled {spelled!r}; declared: {declared}"

        raise ValueError(msg) from error


def _role(name: str) -> Role:
    """Return the role a declaration names.

    Parameters
    ----------
    name : str
        The member name the declaration spells.

    Returns
    -------
    Role
        The member.

    Raises
    ------
    ValueError
        If no role goes by that name.
    """
    try:
        return Role[name]

    except KeyError as error:
        declared = ", ".join(member.name for member in Role)
        msg = f"no role is named {name!r}; declared: {declared}"

        raise ValueError(msg) from error


def _access(names: tuple[str, ...]) -> Access:
    """Return the access set a declaration names.

    Parameters
    ----------
    names : tuple[str, ...]
        The member names the declaration spells.

    Returns
    -------
    Access
        Their union.

    Raises
    ------
    ValueError
        If no access member goes by one of the names, or if the set is empty, which
        describes a point no observer can use.
    """
    if not names:
        msg = "a declared point permits neither observation nor patching"
        raise ValueError(msg)

    permitted = Access(0)

    for name in names:
        try:
            permitted |= Access[name]

        except KeyError as error:
            declared = ", ".join(member.name for member in Access)
            msg = f"no access is named {name!r}; declared: {declared}"

            raise ValueError(msg) from error

    return permitted


def _dtype_class(spelled: str) -> DTypeClass:
    """Return the dtype class a declaration spells.

    Parameters
    ----------
    spelled : str
        The value of a ``DTypeClass`` member.

    Returns
    -------
    DTypeClass
        The member.

    Raises
    ------
    ValueError
        If no member spells that value.
    """
    try:
        return DTypeClass(spelled)

    except ValueError as error:
        declared = ", ".join(str(member.value) for member in DTypeClass)
        msg = f"no dtype class is spelled {spelled!r}; declared: {declared}"

        raise ValueError(msg) from error


class SiteMapIso(dx.Iso[SiteMapSpec, "SiteMap"]):
    """Converts between declared and runtime site maps."""

    @override
    def forward(self, a: SiteMapSpec, /) -> SiteMap:
        """Return the interior map a declaration describes.

        Parameters
        ----------
        a : SiteMapSpec
            The declared map.

        Returns
        -------
        SiteMap
            Each point it declares, keyed by ``Site``, in the order declared.

        Raises
        ------
        ValueError
            If enums, axis names, allowed accesses, or point names are invalid.
        """
        component = ComponentId(a.component)
        kind = _kind(a.kind)
        resolved: dict[Site, SiteSpec] = {}

        for declared in a.sites:
            site = Site(
                component,
                _role(declared.role),
                declared.key,
            )

            if site in resolved:
                msg = f"a declared site map names {site!r} twice"
                raise ValueError(msg)

            resolved[site] = SiteSpec(
                declared.axes,
                _dtype_class(declared.dtype_class),
                differentiable=declared.differentiable,
                access=_access(declared.access),
            )

        if kind is ComponentKind.CUSTOM:
            return SiteMap.custom(component, resolved)

        return ensure_publishes(SiteMap(component, kind, resolved))

    @override
    def backward(self, b: SiteMap, /) -> SiteMapSpec:
        """Return the declaration an interior map would be written out as.

        Parameters
        ----------
        b : SiteMap
            The interior map.

        Returns
        -------
        SiteMapSpec
            One entry per point, in the order the map iterates in.
        """
        return SiteMapSpec(
            component=str(b.component),
            kind=b.kind.value,
            sites=tuple(
                SiteDecl(
                    role=site.role.name,
                    key=site.key,
                    axes=spec.axes,
                    dtype_class=str(spec.dtype_class.value),
                    differentiable=spec.differentiable,
                    access=tuple(
                        member.name for member in Access if member in spec.access
                    ),
                )
                for site, spec in b.items()
            ),
        )


class ExternalSiteDecl(dx.Model, extra="forbid"):
    """Stores a declared probing point inside a foreign model.

    Parameters
    ----------
    path : str
        The dotted path of the module whose output holds the value, as
        ``named_modules`` spells it.
    role : str
        What the value at the point plays, by the name of the ``Role`` member.
    layer : int
        Which layer of the model the module sits in.
    axes : tuple[str, ...]
        The name of each axis, outermost first.
    dtype_class : str
        What the numbers there mean, by the value a ``DTypeClass`` member spells.
    differentiable : bool
        Whether a gradient flows through the value at the point.
    access : tuple[str, ...]
        What the point permits, by the names of the ``Access`` members.
    select : int | None
        Which position of the module's output the value sits at, or None where the
        module returns the value itself.
    """

    path: str
    role: str
    layer: int
    axes: tuple[str, ...] = ()
    dtype_class: str
    differentiable: bool
    access: tuple[str, ...]
    select: int | None = None


class ExternalModelSpec(dx.Model, extra="forbid"):
    """Stores the declared probing points of one foreign model.

    Parameters
    ----------
    model : str
        What the foreign model is called, which every point belongs to.
    sites : tuple[ExternalSiteDecl, ...]
        One entry per point declared for it, in any order.
    checkpoint : PretrainedSourceSpec | None
        Where the pinned artifact backing the model is fetched from, or None when
        the caller supplies the model through another declared route.
    """

    model: str
    sites: tuple[ExternalSiteDecl, ...] = ()
    checkpoint: PretrainedSourceSpec | None = None

    def sites_runtime(self) -> tuple[Site, ...]:
        """Return the sites named by this validated declaration."""
        return tuple(ExternalSiteMapIso().forward(self).published)

    def site_map(self) -> ExternalSiteMap:
        """Return the resolved external site map."""
        return ExternalSiteMapIso().forward(self)


class ExternalSiteMapIso(dx.Iso[ExternalModelSpec, ExternalSiteMap]):
    """Converts between declared and resolved foreign-model site maps."""

    @override
    def forward(self, a: ExternalModelSpec, /) -> ExternalSiteMap:
        """Return the resolved map a declaration describes.

        Parameters
        ----------
        a : ExternalModelSpec
            The declared map.

        Returns
        -------
        ExternalSiteMap
            Each point it declares, in the order declared.

        Raises
        ------
        ValueError
            If enums, axis names, allowed accesses, module paths, layer indices or
            point names are invalid.
        """
        return ExternalSiteMap(
            ComponentId(a.model),
            [
                ExternalSite(
                    declared.path,
                    _role(declared.role),
                    declared.layer,
                    SiteSpec(
                        declared.axes,
                        _dtype_class(declared.dtype_class),
                        differentiable=declared.differentiable,
                        access=_access(declared.access),
                    ),
                    select=declared.select,
                )
                for declared in a.sites
            ],
            checkpoint=(
                None if a.checkpoint is None else source_artifact(a.checkpoint)
            ),
        )

    @override
    def backward(self, b: ExternalSiteMap, /) -> ExternalModelSpec:
        """Return the declaration a resolved map would be written out as.

        Parameters
        ----------
        b : ExternalSiteMap
            The resolved map.

        Returns
        -------
        ExternalModelSpec
            One entry per point, in the order the map declares them.
        """
        return ExternalModelSpec(
            model=str(b.model),
            sites=tuple(
                ExternalSiteDecl(
                    path=external.path,
                    role=external.role.name,
                    layer=external.layer,
                    axes=external.spec.axes,
                    dtype_class=str(external.spec.dtype_class.value),
                    differentiable=external.spec.differentiable,
                    access=tuple(
                        member.name
                        for member in Access
                        if member in external.spec.access
                    ),
                    select=external.select,
                )
                for external in b.sites
            ),
            checkpoint=(None if b.checkpoint is None else source_spec(b.checkpoint)),
        )


def lower(
    published: SiteMap,
    requested: Iterable[Site],
    collected: MutableSequence[tuple[Site, Tensor]],
) -> Mapping[Site, SiteReader]:
    """Return the closures a scope installs, one per point it was asked for.

    Parameters
    ----------
    published : SiteMap
        The interior map the codec produced.
    requested : Iterable[Site]
        The points the scope was opened over.
    collected : MutableSequence[tuple[Site, Tensor]]
        Where the readings go, in the order they are taken.

    Returns
    -------
    Mapping[Site, SiteReader]
        One reader per requested point and none besides, keyed by the point, in the
        order the map publishes them.

    Raises
    ------
    ValueError
        If a requested point is not one the map publishes.
    """
    asked = frozenset(requested)
    unknown = sorted(asked - frozenset(published), key=repr)

    if unknown:
        msg = (
            f"a scope was asked to read {unknown[0]!r}, which this component "
            f"does not publish among its {len(published)} points"
        )
        raise ValueError(msg)

    def reader(site: Site) -> SiteReader:
        def read(tensor: Tensor) -> None:
            collected.append((site, tensor))

        return read

    return {site: reader(site) for site in published if site in asked}


class DiagnosticsProbeSettings(NamedTuple):
    """The settings a diagnostics probe is built from.

    Attributes
    ----------
    roles : frozenset[Role] | None
        The roles whose points the probe reads, or None for every role the
        trained point publishes.
    """

    roles: frozenset[Role] | None


class DiagnosticsProbeSpec(ProbeSpec):
    """Declare the installed diagnostics a set of published roles supports.

    Every diagnostic in the ``lofi.diagnostic`` catalog whose required roles lie
    among the declared roles is run over each held-out pass of the trained point.

    Parameters
    ----------
    roles : tuple[str, ...] | None
        The names of the ``Role`` members whose points are read, or None for
        every role the trained point publishes. A declared role no component
        publishes is refused when the run is composed.
    """

    kind: Literal["diagnostics"] = "diagnostics"
    roles: tuple[str, ...] | None = None

    @dx.validates("roles")
    @classmethod
    def _roles_are_members(
        cls, value: tuple[str, ...] | None
    ) -> tuple[str, ...] | None:
        """Return role names that each name a ``Role`` member."""
        if value is not None:
            for name in value:
                _role(name)

        return value

    def settings(self) -> DiagnosticsProbeSettings:
        """Return the settings the plugin builds this probe from."""
        return DiagnosticsProbeSettings(
            roles=None if self.roles is None else frozenset(map(_role, self.roles))
        )


class SpanLabels(enum.StrEnum):
    """Enumerates what a classifier probe labels a span with."""

    #: Whether the span is a constituent of the item's gold analysis, over every
    #: span of the item.
    CONSTITUENT = "constituent"

    #: The label of the gold constituent at the span, over the gold constituents.
    CATEGORY = "category"


class ClassifierProbeSettings(NamedTuple):
    """The settings a classifier probe is built from.

    Attributes
    ----------
    role : Role
        The role of the chart points whose span vectors are classified.
    labels : SpanLabels
        What each span is labeled with.
    held_out_fraction : float
        The share of held-out items the probes are scored on rather than fitted on.
    split_seed : int
        The seed the items are divided under.
    classifier : ClassifierProbe
        The classifier, its capacity and its control task.
    schedule : BlockSchedule
        Where the online code's blocks end.
    """

    role: Role
    labels: SpanLabels
    held_out_fraction: float
    split_seed: int
    classifier: ClassifierProbe
    schedule: BlockSchedule


class ClassifierProbeSpec(ProbeSpec):
    """Declare a span classifier scored by selectivity, codelength and erasure.

    The span vectors at every chart point of the declared role are labeled from
    the held-out items' gold analyses and divided by item into a fitting and a
    scored share. Each point reports the probe's selectivity against a control
    task, the online codelength of the fitting labels, and the same probe's
    accuracies once LEACE has erased the labels' linear information from the
    vectors.

    Parameters
    ----------
    role : str
        The name of the ``Role`` member whose points hold one vector per span,
        over an ``item`` axis and then a ``span`` axis.
    labels : SpanLabels
        What each span is labeled with.
    held_out_fraction : float
        The share of items scored rather than fitted on, strictly between zero
        and one.
    split_seed : int
        The seed the items are divided under.
    hidden : int
        The probe's hidden width, zero for a linear probe.
    epochs : int
        How many full-batch steps each fit takes.
    learning_rate : float
        The fit's step size.
    seed : int
        The seed the probe's starting weights are drawn under.
    control_seed : int
        The seed the control task draws its labels under.
    block_fractions : tuple[float, ...]
        Where the online code's blocks end, ascending prefix fractions ending at
        one.
    """

    kind: Literal["classifier"] = "classifier"
    role: str = Role.CHART_CELL.name
    labels: SpanLabels = SpanLabels.CONSTITUENT
    held_out_fraction: float = 0.5
    split_seed: int = 0
    hidden: int = 0
    epochs: int = 100
    learning_rate: float = 0.1
    seed: int = _DEFAULT_SEED
    control_seed: int = 0
    block_fractions: tuple[float, ...] = VOITA_BLOCKS.fractions

    @dx.validates("role")
    @classmethod
    def _role_is_member(cls, value: str) -> str:
        """Return a role name that names a ``Role`` member."""
        _role(value)

        return value

    @dx.validates("held_out_fraction")
    @classmethod
    def _fraction_divides(cls, value: float) -> float:
        """Return a share strictly between zero and one."""
        if not 0.0 < value < 1.0:
            msg = f"a held-out fraction lies strictly between 0 and 1, not {value!r}"
            raise ValueError(msg)

        return value

    @dx.model_validator()
    def _builds(self) -> Self:
        """Refuse a capacity, control seed or block schedule no probe is built from."""
        self.settings()

        return self

    def settings(self) -> ClassifierProbeSettings:
        """Return the settings the plugin builds this probe from."""
        return ClassifierProbeSettings(
            role=_role(self.role),
            labels=self.labels,
            held_out_fraction=self.held_out_fraction,
            split_seed=self.split_seed,
            classifier=ClassifierProbe(
                ProbeCapacity(
                    hidden=self.hidden,
                    epochs=self.epochs,
                    learning_rate=self.learning_rate,
                    seed=self.seed,
                ),
                ControlTask(seed=self.control_seed),
            ),
            schedule=BlockSchedule(self.block_fractions),
        )


class ProbeSiteDecl(dx.Model, extra="forbid"):
    """Names one point a trained component publishes.

    Parameters
    ----------
    component : str
        The run role of the component publishing the point.
    role : str
        The name of the ``Role`` member the point plays.
    key : tuple[int | str, ...]
        Which point among those the component publishes for that role.
    """

    component: str
    role: str
    key: tuple[int | str, ...] = ()

    @dx.validates("role")
    @classmethod
    def _role_is_member(cls, value: str) -> str:
        """Return a role name that names a ``Role`` member."""
        _role(value)

        return value

    def site(self) -> Site:
        """Return the point this declaration names."""
        return Site(ComponentId(self.component), _role(self.role), self.key)


class SaeProbeSettings(NamedTuple):
    """The site, capacity, objective, and optimization declared for an SAE arm."""

    site: Site
    width: int
    sparsity_penalty: float
    epochs: int
    learning_rate: float
    seed: int


class SaeProbeSpec(ProbeSpec):
    """Declare a sparse autoencoder fitted to one published activation site.

    Parameters
    ----------
    site : ProbeSiteDecl
        The observable real-valued site whose final axis is activation width.
    width : int
        Number of nonnegative latent features.
    sparsity_penalty : float
        Coefficient on mean latent activation in the fitting objective.
    epochs : int
        Number of full-batch optimization steps.
    learning_rate : float
        Adam step size.
    seed : int
        Seed used only to initialize this probe's parameters.
    """

    kind: Literal["sae"] = "sae"
    site: ProbeSiteDecl
    width: int
    sparsity_penalty: float
    epochs: int
    learning_rate: float
    seed: int = _DEFAULT_SEED

    @dx.model_validator()
    def _builds(self) -> Self:
        """Refuse a declaration no sparse autoencoder can be built from."""
        self.settings()

        return self

    def settings(self) -> SaeProbeSettings:
        """Return the validated interior settings."""
        penalty = ensure_finite("an SAE sparsity penalty", self.sparsity_penalty)
        rate = ensure_finite("an SAE learning rate", self.learning_rate)

        if penalty < 0.0:
            msg = f"an SAE sparsity penalty is negative: {penalty!r}"
            raise ValueError(msg)

        if rate <= 0.0:
            msg = f"an SAE learning rate is not positive: {rate!r}"
            raise ValueError(msg)

        return SaeProbeSettings(
            self.site.site(),
            ensure_positive("an SAE width", self.width),
            penalty,
            ensure_positive("an SAE epoch count", self.epochs),
            rate,
            ensure_non_negative("an SAE seed", self.seed),
        )


class _Standing(NamedTuple):
    """An intervention that does not depend on the unpatched pass."""

    intervention: Intervention

    @property
    def access(self) -> Access:
        """Return the patch access the site must permit."""
        return Access.PATCH

    def drawn(self, clean: Trace, /) -> Intervention:
        """Return the intervention, whatever the unpatched pass read."""
        del clean

        return self.intervention


class _MeanOfClean:
    """Replaces the value at a site with the mean the unpatched pass read there."""

    __slots__ = ()

    @property
    def access(self) -> Access:
        """Return the access to read the site unpatched and then patch it."""
        return Access.OBSERVE | Access.PATCH

    def drawn(self, clean: Trace, /) -> Intervention:
        """Return the mean of the unpatched readings, filled to each reading."""
        return MeanFrom(clean)


class _ResampleClean(NamedTuple):
    """Replaces the value at a site with draws from the unpatched readings."""

    seed: int

    @property
    def access(self) -> Access:
        """Return the access to read the site unpatched and then patch it."""
        return Access.OBSERVE | Access.PATCH

    def drawn(self, clean: Trace, /) -> Intervention:
        """Return draws from the unpatched readings under the seed."""
        return Resample(clean, self.seed)


class _InterventionDeclaration(Protocol):
    """A declaration that supplies an intervention source."""

    def source(self) -> InterventionSource:
        """Return what draws the intervention from the unpatched pass."""
        ...


class InterventionDecl(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declares what replaces the value at an intervened site."""


class ZeroDecl(InterventionDecl):
    """Declare zeros in place of the value."""

    kind: Literal["zero"] = "zero"

    def source(self) -> InterventionSource:
        """Return zeros, whatever the unpatched pass read."""
        return _Standing(Zero())


class MeanDecl(InterventionDecl):
    """Declare the unpatched pass's mean at the site in place of the value."""

    kind: Literal["mean"] = "mean"

    def source(self) -> InterventionSource:
        """Return the mean the unpatched pass read at the site."""
        return _MeanOfClean()


class NoiseDecl(InterventionDecl):
    """Declare centered Gaussian noise added to the value.

    Parameters
    ----------
    sigma : float
        The standard deviation of the noise.
    seed : int
        The seed the noise is drawn under.
    """

    kind: Literal["noise"] = "noise"
    sigma: float
    seed: int = _DEFAULT_SEED

    @dx.model_validator()
    def _draws(self) -> Self:
        """Refuse a scale or seed no noise is drawn under."""
        self.source()

        return self

    def source(self) -> InterventionSource:
        """Return the noise, whatever the unpatched pass read."""
        return _Standing(Noise(self.sigma, self.seed))


class ShuffleDecl(InterventionDecl):
    """Declare the value permuted along its outermost axis.

    Parameters
    ----------
    seed : int
        The seed the permutation is drawn under.
    """

    kind: Literal["shuffle"] = "shuffle"
    seed: int = _DEFAULT_SEED

    def source(self) -> InterventionSource:
        """Return the permutation, whatever the unpatched pass read."""
        return _Standing(Shuffle(self.seed))


class ResampleDecl(InterventionDecl):
    """Declare numbers drawn from the unpatched pass's readings at the site.

    Parameters
    ----------
    seed : int
        The seed the draw runs under.
    """

    kind: Literal["resample"] = "resample"
    seed: int = _DEFAULT_SEED

    def source(self) -> InterventionSource:
        """Return draws from the unpatched pass's readings at the site."""
        return _ResampleClean(self.seed)


class SiteInterventionDecl(dx.Model, extra="forbid"):
    """Pairs a published point with what replaces the value there.

    Parameters
    ----------
    site : ProbeSiteDecl
        The point intervened on, which must permit patching.
    intervention : InterventionDecl
        What replaces the value there.
    """

    site: ProbeSiteDecl
    intervention: InterventionDecl


class InterventionProbeSettings(NamedTuple):
    """The settings an intervention probe is built from.

    Attributes
    ----------
    interventions : tuple[tuple[Site, InterventionSource], ...]
        Each intervened point and what draws its intervention, in declared order.
    outcome : Role
        The role of the points whose readings the interventions are scored on.
    """

    interventions: tuple[tuple[Site, InterventionSource], ...]
    outcome: Role


class InterventionProbeSpec(ProbeSpec):
    """Declare one intervention per site, each scored on the declared outcome.

    Each held-out pass runs once unpatched and once with each declared
    intervention alone, and each intervention reports the mean absolute change
    it caused in the readings at every point of the outcome role.

    Parameters
    ----------
    interventions : tuple[SiteInterventionDecl, ...]
        Each intervened point and what replaces the value there, at least one,
        no point twice.
    outcome : str
        The name of the ``Role`` member whose points are compared.
    """

    kind: Literal["intervention"] = "intervention"
    interventions: tuple[SiteInterventionDecl, ...]
    outcome: str = Role.LOG_PARTITION.name

    @dx.validates("outcome")
    @classmethod
    def _outcome_is_member(cls, value: str) -> str:
        """Return a role name that names a ``Role`` member."""
        _role(value)

        return value

    @dx.validates("interventions")
    @classmethod
    def _one_per_site(
        cls, value: tuple[SiteInterventionDecl, ...]
    ) -> tuple[SiteInterventionDecl, ...]:
        """Return at least one intervention, with no point named twice."""
        if not value:
            msg = "an intervention probe declares at least one intervention"
            raise ValueError(msg)

        sites = [entry.site.site() for entry in value]

        if len(frozenset(sites)) != len(sites):
            msg = "an intervention probe names a point twice"
            raise ValueError(msg)

        return value

    def settings(self) -> InterventionProbeSettings:
        """Return the settings the plugin builds this probe from."""
        return InterventionProbeSettings(
            interventions=tuple(
                (
                    entry.site.site(),
                    cast("_InterventionDeclaration", entry.intervention).source(),
                )
                for entry in self.interventions
            ),
            outcome=_role(self.outcome),
        )


class _ThresholdDeclaration(Protocol):
    """A declaration that supplies an ACDC pruning threshold."""

    def threshold(self) -> PruningThreshold:
        """Return the rule bounding an accepted divergence increment."""
        ...


class ACDCThresholdDecl(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declares the cutoff below which ACDC prunes a candidate edge."""


class AbsoluteThresholdDecl(ACDCThresholdDecl):
    """Declare a fixed bound on an accepted divergence increment.

    Parameters
    ----------
    bound : float
        The finite nonnegative bound, in the units of the divergence.
    """

    kind: Literal["absolute"] = "absolute"
    bound: float

    @dx.model_validator()
    def _bounds(self) -> Self:
        """Refuse a bound no threshold is built from."""
        self.threshold()

        return self

    def threshold(self) -> PruningThreshold:
        """Return the fixed bound."""
        return AbsoluteThreshold(self.bound)


class RelativeThresholdDecl(ACDCThresholdDecl):
    """Declare a bound that is a fraction of the full corruption divergence.

    Parameters
    ----------
    fraction : float
        The finite nonnegative fraction of the divergence of the
        counterfactual output from the clean output.
    """

    kind: Literal["relative"] = "relative"
    fraction: float

    @dx.model_validator()
    def _bounds(self) -> Self:
        """Refuse a fraction no threshold is built from."""
        self.threshold()

        return self

    def threshold(self) -> PruningThreshold:
        """Return the fraction of the full corruption divergence."""
        return RelativeThreshold(self.fraction)


class CircuitProbeSettings(NamedTuple):
    """The validated settings of one circuit arm."""

    attribution_steps: int
    acdc_trials: int
    acdc_threshold: PruningThreshold
    null_replicates: int
    seed: int
    completeness_draws: int
    bootstrap_replicates: int
    bootstrap_level: float
    sensitivity_threshold: float
    include_zero: bool
    fdr_alpha: float
    dependence: Dependence


class CircuitProbeSpec(ProbeSpec):
    """Declare EAP-IG, budgeted ACDC, faithfulness, nulls, and sensitivity.

    Parameters
    ----------
    acdc_trials : int
        The explicit hard bound on ACDC candidate forwards. It is required: an
        unbudgeted pruning declaration is not a valid circuit experiment.
    attribution_steps : int
        Right-endpoint integration steps used by EAP-IG.
    acdc_threshold : ACDCThresholdDecl
        The rule bounding the divergence increment ACDC accepts: a fixed
        bound, or a fraction of the full corruption divergence.
    null_replicates : int
        Matched-random-circuit draws for each circuitry criterion.
    seed : int
        Shared local seed for resampling and matched nulls.
    completeness_draws : int
        Random circuit subsets added to singleton and whole-circuit checks.
    bootstrap_replicates : int
        Item bootstrap draws for faithfulness effect sizes.
    bootstrap_level : float
        Coverage of faithfulness intervals.
    sensitivity_threshold : float
        Minimum normalized recovery counted as the sensitivity conclusion.
    include_zero : bool
        Whether the off-distribution zero-ablation arm is also reported.
    fdr_alpha : float
        Level of the Benjamini-Hochberg circuitry family.
    dependence : Literal["independent", "prds"]
        Dependence regime declared for that BH family.
    """

    kind: Literal["circuit"] = "circuit"
    acdc_trials: int
    attribution_steps: int = 8
    acdc_threshold: ACDCThresholdDecl = dx.field(
        default_factory=lambda: AbsoluteThresholdDecl(bound=0.01)
    )
    null_replicates: int = _DEFAULT_NULL_REPLICATES
    seed: int = _DEFAULT_SEED
    completeness_draws: int = 8
    bootstrap_replicates: int = _DEFAULT_BOOTSTRAP_REPLICATES
    bootstrap_level: float = _DEFAULT_BOOTSTRAP_LEVEL
    sensitivity_threshold: float = 0.5
    include_zero: bool = False
    fdr_alpha: float = _DEFAULT_FDR_ALPHA
    dependence: Literal["independent", "prds"] = "prds"

    @dx.model_validator()
    def _builds(self) -> Self:
        """Return a declaration whose complete interior settings are valid."""
        self.settings()

        return self

    def settings(self) -> CircuitProbeSettings:
        """Return validated interior settings."""
        steps = ensure_positive("EAP-IG integration steps", self.attribution_steps)
        trials = ensure_non_negative("an ACDC trial budget", self.acdc_trials)
        threshold = cast("_ThresholdDeclaration", self.acdc_threshold).threshold()
        null_replicates = ensure_positive(
            "matched-circuit null replicates", self.null_replicates
        )
        seed = ensure_non_negative("a circuit probe seed", self.seed)
        draws = ensure_non_negative("completeness draws", self.completeness_draws)
        bootstrap = ensure_positive(
            "faithfulness bootstrap replicates", self.bootstrap_replicates
        )
        level = ensure_finite("faithfulness bootstrap coverage", self.bootstrap_level)
        sensitivity = ensure_finite(
            "an ablation sensitivity threshold", self.sensitivity_threshold
        )
        alpha = ensure_finite("a circuit FDR level", self.fdr_alpha)

        for name, value in (
            ("faithfulness bootstrap coverage", level),
            ("ablation sensitivity threshold", sensitivity),
            ("circuit FDR level", alpha),
        ):
            if not 0.0 < value < 1.0:
                msg = f"{name} lies strictly inside the unit interval, not {value!r}"
                raise ValueError(msg)

        return CircuitProbeSettings(
            steps,
            trials,
            threshold,
            null_replicates,
            seed,
            draws,
            bootstrap,
            level,
            sensitivity,
            self.include_zero,
            alpha,
            Dependence(self.dependence),
        )


class SearchContext(NamedTuple):
    """The settings every declared search arm shares."""

    schedule: Schedule
    null: LabelPermutation[int]
    objective: AlignmentObjective
    annealing: Annealing


class _SearchDeclaration(Protocol):
    """A declaration that builds one DAS search strategy."""

    @property
    def kind(self) -> str:
        """Return the tag naming the search arm in its rows."""
        ...

    def strategy(self, context: SearchContext, /) -> SearchStrategy:
        """Return the search under the shared schedule, null and objective."""
        ...


class DASSearchDecl(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declares one distributed-alignment search arm of a causal probe."""


class FixedRankSearchDecl(DASSearchDecl):
    """Declare a search over a fixed binary prefix of a learned rotation.

    Parameters
    ----------
    rank : int
        The positive number of leading rotated coordinates swapped.
    """

    kind: Literal["fixed-rank"] = "fixed-rank"
    rank: int = 1

    @dx.validates("rank")
    @classmethod
    def _rank_is_positive(cls, value: int) -> int:
        """Refuse a rank that swaps nothing."""
        return ensure_positive("a fixed DAS rank", value)

    def strategy(self, context: SearchContext, /) -> SearchStrategy:
        """Return the fixed-rank search under the shared settings."""
        return FixedRankSearch(
            rank=self.rank,
            schedule=context.schedule,
            null=context.null,
            objective=context.objective,
        )


class BoundlessSearchDecl(DASSearchDecl):
    """Declare a search learning one ordered interval in a rotated basis.

    Parameters
    ----------
    boundary_lower : float
        The initial nonnegative lower boundary index.
    boundary_upper : float
        The initial upper boundary index, above the lower one.
    boundary_margin : float
        How far an initial boundary share is held from 0 and 1 before its logit
        is taken, in (0, 0.5).
    """

    kind: Literal["boundless"] = "boundless"
    boundary_lower: float = 0.0
    boundary_upper: float = 1.0
    boundary_margin: float = 1.0e-6

    @dx.model_validator()
    def _bounds(self) -> Self:
        """Refuse an interval no Boundless search is built from."""
        lower = ensure_finite("a Boundless DAS lower boundary", self.boundary_lower)
        upper = ensure_finite("a Boundless DAS upper boundary", self.boundary_upper)
        margin = ensure_finite("a Boundless DAS boundary margin", self.boundary_margin)

        if lower < 0.0 or lower >= upper:
            msg = (
                "Boundless DAS boundaries are nonnegative and ordered, not "
                f"{(lower, upper)!r}"
            )
            raise ValueError(msg)

        if not 0.0 < margin < 0.5:
            msg = f"a Boundless DAS boundary margin lies in (0, 0.5), not {margin!r}"
            raise ValueError(msg)

        return self

    def strategy(self, context: SearchContext, /) -> SearchStrategy:
        """Return the Boundless search under the shared settings."""
        return BoundlessSearch(
            boundaries=(self.boundary_lower, self.boundary_upper),
            margin=self.boundary_margin,
            annealing=context.annealing,
            schedule=context.schedule,
            null=context.null,
            objective=context.objective,
        )


class CausalSiteDecl(dx.Model, extra="forbid"):
    """Declare the published point of one layer-position sweep coordinate."""

    layer: int
    position: int
    site: ProbeSiteDecl


class NodeVocabularyArmDecl(dx.Model, extra="forbid"):
    """Name the node vocabulary representation one causal probe measures.

    Parameters
    ----------
    representation : str
        The node vocabulary arm the probed point's representation is.
    search : str
        The kind of the declared search arm whose sweep the arm records.
    """

    representation: Literal[
        "derivation-variables",
        "trained-transformer-sae-latents",
        "dense-linear-probe",
        "random-transformer-sae-control",
    ]
    search: str


class CausalProbeSettings(NamedTuple):
    """The validated settings of one causal arm."""

    sites: tuple[tuple[tuple[int, int], Site], ...]
    searches: tuple[tuple[str, SearchStrategy], ...]
    joint_search: str
    arm: tuple[NodeVocabularyArm, str] | None
    objective: AlignmentObjective
    annealing: Annealing
    schedule: Schedule
    null_replicates: int
    seed: int
    budget_hours: float
    bootstrap_replicates: int
    bootstrap_level: float
    bracketing_tolerance: float
    bracketing_attempts: int
    well_cutoff: float
    minimum_bucket_size: int
    minimum_accuracy_gap: float
    fdr_alpha: float
    dependence: Dependence


class CausalProbeSpec(ProbeSpec):
    """Declare DAS search arms, IIA, joint tests, controls, and buckets.

    Parameters
    ----------
    searches : tuple[DASSearchDecl, ...]
        The search arms swept over every site, each of a distinct kind.
    joint_search : str
        The kind of the declared arm the joint condition and the controls fit.
    arm : NodeVocabularyArmDecl | None
        The node vocabulary representation the probe measures, whose complete
        sweep each held-out pass records beside the probe's rows, or null where
        the probe records none.
    """

    kind: Literal["causal"] = "causal"
    sites: tuple[CausalSiteDecl, ...]
    searches: tuple[DASSearchDecl, ...] = dx.field(
        default_factory=lambda: (FixedRankSearchDecl(), BoundlessSearchDecl())
    )
    joint_search: str = "fixed-rank"
    arm: NodeVocabularyArmDecl | None = None
    objective: Literal["cross-entropy", "logit-difference"] = "logit-difference"
    annealing_initial: float = 1.0
    annealing_final: float = 0.1
    epochs: int = 10
    learning_rate: float = 0.01
    batch_size: int = 16
    seed: int = _DEFAULT_SEED
    null_replicates: int = _DEFAULT_NULL_REPLICATES
    budget_hours: float
    bootstrap_replicates: int = _DEFAULT_BOOTSTRAP_REPLICATES
    bootstrap_level: float = _DEFAULT_BOOTSTRAP_LEVEL
    bracketing_tolerance: float = 0.1
    bracketing_attempts: int = 100
    well_cutoff: float = 0.8
    minimum_bucket_size: int = 10
    minimum_accuracy_gap: float = 0.1
    fdr_alpha: float = _DEFAULT_FDR_ALPHA
    dependence: Literal["independent", "prds"] = "prds"

    @dx.model_validator()
    def _builds(self) -> Self:
        """Return a declaration whose complete settings are valid."""
        self.settings()

        return self

    def settings(self) -> CausalProbeSettings:
        """Return validated interior settings."""
        if not self.sites:
            msg = "a causal probe declares at least one layer-position site"
            raise ValueError(msg)
        indexed: dict[tuple[int, int], Site] = {}

        for declared in self.sites:
            coordinate = (
                ensure_non_negative("a causal layer", declared.layer),
                ensure_non_negative("a causal position", declared.position),
            )
            if coordinate in indexed:
                msg = f"a causal probe declares coordinate {coordinate!r} twice"
                raise ValueError(msg)
            indexed[coordinate] = declared.site.site()

        declared = tuple(cast("_SearchDeclaration", arm) for arm in self.searches)
        kinds = tuple(arm.kind for arm in declared)
        if not kinds or len(set(kinds)) != len(kinds):
            msg = "a causal probe declares distinct nonempty search arms"
            raise ValueError(msg)
        if self.joint_search not in kinds:
            msg = (
                f"a causal probe's joint_search {self.joint_search!r} names none "
                f"of its declared search arms {kinds!r}"
            )
            raise ValueError(msg)
        if self.arm is not None and self.arm.search not in kinds:
            msg = (
                f"a causal probe's arm records search {self.arm.search!r}, none "
                f"of its declared search arms {kinds!r}"
            )
            raise ValueError(msg)
        arm = (
            None
            if self.arm is None
            else (NodeVocabularyArm(self.arm.representation), self.arm.search)
        )
        annealing = Annealing(
            initial=self.annealing_initial, final=self.annealing_final
        )
        schedule = Schedule(
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            seed=self.seed,
        )
        null_replicates = ensure_positive(
            "a per-cell null replicate count", self.null_replicates
        )
        seed = ensure_non_negative("a causal probe seed", self.seed)
        budget = ensure_finite("a causal sweep budget in hours", self.budget_hours)
        bootstrap = ensure_positive(
            "a causal bootstrap replicate count", self.bootstrap_replicates
        )
        level = ensure_finite("a causal bootstrap coverage", self.bootstrap_level)
        tolerance = ensure_finite(
            "a bracketing-control tolerance", self.bracketing_tolerance
        )
        attempts = ensure_positive(
            "a bracketing-control attempt count", self.bracketing_attempts
        )
        cutoff = ensure_finite("an abstraction well cutoff", self.well_cutoff)
        bucket_size = ensure_positive(
            "a minimum abstraction bucket size", self.minimum_bucket_size
        )
        gap = ensure_finite(
            "a minimum abstraction accuracy gap", self.minimum_accuracy_gap
        )
        alpha = ensure_finite("a causal FDR level", self.fdr_alpha)
        context = SearchContext(
            schedule,
            LabelPermutation(replicates=null_replicates, seed=seed),
            AlignmentObjective(self.objective),
            annealing,
        )
        searches = tuple((arm.kind, arm.strategy(context)) for arm in declared)

        if budget < 0.0:
            msg = f"a causal sweep budget is nonnegative, not {budget!r}"
            raise ValueError(msg)
        for name, value, closed in (
            ("causal bootstrap coverage", level, False),
            ("bracketing-control tolerance", tolerance, True),
            ("abstraction well cutoff", cutoff, True),
            ("minimum abstraction accuracy gap", gap, True),
            ("causal FDR level", alpha, False),
        ):
            valid = 0.0 <= value <= 1.0 if closed else 0.0 < value < 1.0
            if not valid:
                interval = "closed" if closed else "open"
                msg = f"{name} lies outside the {interval} unit interval: {value!r}"
                raise ValueError(msg)

        return CausalProbeSettings(
            tuple(sorted(indexed.items())),
            searches,
            self.joint_search,
            arm,
            AlignmentObjective(self.objective),
            annealing,
            schedule,
            null_replicates,
            seed,
            budget,
            bootstrap,
            level,
            tolerance,
            attempts,
            cutoff,
            bucket_size,
            gap,
            alpha,
            Dependence(self.dependence),
        )

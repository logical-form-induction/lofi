"""Site maps for foreign models, declared as module paths and axis names."""

from collections.abc import Mapping
from contextlib import ExitStack
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Final,
    NamedTuple,
    Protocol,
    Self,
    override,
    runtime_checkable,
)

import torch
from torch import nn

from lofi._checks import ensure_name, ensure_non_negative
from lofi.nn.sites import Access, ComponentId, Role, Site, SiteMap, SiteSpec
from lofi.probe._causal import (
    AlignmentPolicy,
    CombinatorAt,
    NamedVariable,
    Side,
    SplitAt,
    TypeAt,
)
from lofi.probe._hooks import Hooks, module_named
from lofi.probe._ports import ModelReadyCausalPair, ModelReadyJointCausalPair
from lofi.probe._values import Observation

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from types import TracebackType

    from torch import Tensor
    from torch.nn import Module

    from lofi._artifacts import PretrainedArtifact
    from lofi.nn.ports import Patch
    from lofi.probe._causal import HighLevelVariable

__all__ = [
    "TORCH_HOOKS",
    "Attachment",
    "DecisionScorer",
    "ExternalJointSubject",
    "ExternalModel",
    "ExternalScope",
    "ExternalSite",
    "ExternalSiteMap",
    "ExternalSubject",
    "Hooking",
    "Output",
    "Recorder",
    "Release",
    "TorchHooks",
]

type Output = Tensor | tuple[object, ...]
"""What a module hands back, which is either the value or a tuple holding it."""

type Recorder = Callable[[Observation], None]
"""Callback filing one reading under the point it was taken at."""

type Release = Callable[[], None]
"""Callback removing everything one region installed."""

type DecisionScorer = Callable[
    [Output, HighLevelVariable, tuple[str | int, ...]], Tensor
]
"""Maps one external output to one score row per pair and declared decision."""


def _module_path(field: str, path: str) -> str:
    """Return a dotted module path.

    Parameters
    ----------
    field : str
        What the path is for, for the message.
    path : str
        The path to check.

    Returns
    -------
    str
        The path, unchanged.

    Raises
    ------
    TypeError
        If the path is not a string.
    ValueError
        If the path is empty or holds an empty segment, neither of which names a
        module a foreign model holds.
    """
    named = ensure_name(field, path)

    if any(not segment for segment in named.split(".")):
        msg = f"{field} holds an empty segment: {path!r}"
        raise ValueError(msg)

    return named


class ExternalSite:
    """Declares one probing point inside a foreign model.

    Parameters
    ----------
    path : str
        The dotted path of the module whose output holds the value, as
        ``named_modules`` spells it. It is matched whole against the paths the
        model publishes, never against a part of one.
    role : Role
        What the value at the point plays in the computation.
    layer : int
        Which layer of the model the module sits in, which is the point's key.
    spec : SiteSpec
        The axes, numeric class, gradient behavior and access declared there.
    select : int | None
        Which position of the module's output the value sits at, or None where the
        module returns the value itself.

    Raises
    ------
    TypeError
        If the path or the selected position is of the wrong type, or the role or
        the specification is not one.
    ValueError
        If the path is empty or holds an empty segment, or the layer index or the
        selected position is negative.
    """

    __slots__ = ("_layer", "_path", "_role", "_select", "_spec")

    def __init__(
        self,
        path: str,
        role: Role,
        layer: int,
        spec: SiteSpec,
        *,
        select: int | None = None,
    ) -> None:
        if not isinstance(role, Role):
            msg = f"a declared point's role is a Role, not {role!r}"
            raise TypeError(msg)

        if not isinstance(spec, SiteSpec):
            msg = f"what a declared point holds is not a SiteSpec: {spec!r}"
            raise TypeError(msg)

        self._path = _module_path("a declared point's module path", path)
        self._role = role
        self._layer = ensure_non_negative("a declared point's layer", layer)
        self._spec = spec
        self._select = (
            None
            if select is None
            else ensure_non_negative("a declared point's position", select)
        )

    @property
    def path(self) -> str:
        """Return the dotted path of the module whose output holds the value."""
        return self._path

    @property
    def role(self) -> Role:
        """Return what the value at the point plays in the computation."""
        return self._role

    @property
    def layer(self) -> int:
        """Return which layer of the model the module sits in."""
        return self._layer

    @property
    def spec(self) -> SiteSpec:
        """Return the axes, numeric class, gradient behavior and access declared."""
        return self._spec

    @property
    def select(self) -> int | None:
        """Return which position of the output holds the value, or None."""
        return self._select

    def _identity(self) -> tuple[str, Role, int, SiteSpec, int | None]:
        """Return the five fields a declared point is equal on."""
        return (self._path, self._role, self._layer, self._spec, self._select)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value declares the same five fields.

        Parameters
        ----------
        other : object
            The value to compare against.
        """
        if type(other) is not ExternalSite:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the five fields the declaration is equal on."""
        return hash((ExternalSite, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this declaration."""
        return (
            f"ExternalSite({self._path!r}, {self._role}, {self._layer!r}, "
            f"{self._spec!r}, select={self._select!r})"
        )


class ExternalSiteMap:
    """Declares the points one foreign model publishes, each under its module path.

    Parameters
    ----------
    model : ComponentId
        What the foreign model is called, which every point belongs to.
    sites : Iterable[ExternalSite]
        The points declared for it, in any order.
    checkpoint : PretrainedArtifact | None
        The pinned artifact backing the model, or None when the caller supplies
        the model by another declared route.

    Raises
    ------
    TypeError
        If the name is not a string, or a declared point is not one.
    ValueError
        If the name is empty, or two declarations name the same point.
    """

    __slots__ = ("_checkpoint", "_declared", "_model", "_published")

    def __init__(
        self,
        model: ComponentId,
        sites: Iterable[ExternalSite],
        *,
        checkpoint: PretrainedArtifact | None = None,
    ) -> None:
        named = ComponentId(ensure_name("an external map's model", model))

        declared: dict[Site, ExternalSite] = {}
        published: dict[Site, SiteSpec] = {}

        for external in sites:
            if not isinstance(external, ExternalSite):
                msg = f"an external map holds {external!r}, which is not a declaration"
                raise TypeError(msg)

            site = Site(named, external.role, (external.layer,))

            if site in declared:
                msg = (
                    f"{named} declares {site!r} twice, at {declared[site].path} and "
                    f"at {external.path}"
                )
                raise ValueError(msg)

            declared[site] = external
            published[site] = external.spec

        self._model = named
        self._checkpoint = checkpoint
        self._declared = MappingProxyType(declared)
        self._published = SiteMap.custom(named, published)

    @property
    def model(self) -> ComponentId:
        """Return what the foreign model is called."""
        return self._model

    @property
    def published(self) -> SiteMap:
        """Return what each declared point holds, keyed by the point."""
        return self._published

    @property
    def checkpoint(self) -> PretrainedArtifact | None:
        """Return the pinned artifact backing the model, or None."""
        return self._checkpoint

    @property
    def sites(self) -> tuple[ExternalSite, ...]:
        """Return the declarations, in the order they were given."""
        return tuple(self._declared.values())

    def declaration(self, site: Site) -> ExternalSite:
        """Return the declaration one point was made under.

        Parameters
        ----------
        site : Site
            The point to look up.

        Returns
        -------
        ExternalSite
            Its declaration.

        Raises
        ------
        ValueError
            If this model declares no such point.
        """
        found = self._declared.get(site)

        if found is None:
            msg = (
                f"{self._model} declares no {site!r}, and declares "
                f"{len(self._declared)} points"
            )
            raise ValueError(msg)

        return found

    def _identity(
        self,
    ) -> tuple[ComponentId, tuple[ExternalSite, ...], PretrainedArtifact | None]:
        """Return the model and its declarations, which a map is equal on."""
        return (self._model, self.sites, self._checkpoint)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value declares the same points for the same model.

        Parameters
        ----------
        other : object
            The value to compare against.
        """
        if type(other) is not ExternalSiteMap:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the model and its declarations."""
        return hash((ExternalSiteMap, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the model and how many points are declared for it."""
        return f"ExternalSiteMap({self._model!r}, {len(self._declared)} sites)"


class Attachment(NamedTuple):
    """What one region installs at one declared point.

    Parameters
    ----------
    site : Site
        The point the value is filed under.
    module : Module
        The module the model holds at the declared path.
    path : str
        The path it was found at, which a message names.
    select : int | None
        Which position of the module's output the value sits at, or None where the
        module returns the value itself.
    axes : tuple[str, ...]
        The name of each axis of the value, outermost first.
    observed : bool
        Whether the region keeps what is read there, as against reaching the point
        only to replace what the pass carries on with.
    patch : Patch[Observation] | None
        What replaces the value there, or None where the point is only read.
    """

    site: Site
    module: Module
    path: str
    select: int | None
    axes: tuple[str, ...]
    observed: bool
    patch: Patch[Observation] | None


def _filed(attachment: Attachment, value: Tensor, record: Recorder) -> Tensor:
    """Return what stands in place of one value, filing the reading kept of it.

    Parameters
    ----------
    attachment : Attachment
        What the region installed at the point.
    value : Tensor
        What the module put there.
    record : Recorder
        Where a reading goes.

    Returns
    -------
    Tensor
        The replacement where the point is patched, and the value itself otherwise.

    Raises
    ------
    ValueError
        If the value has a different number of axes than the declaration names.
    """
    observed = Observation(attachment.site, value, axes=attachment.axes)
    standing = value

    if attachment.patch is not None:
        standing = attachment.patch.stands_in_for(observed)
        observed = Observation(attachment.site, standing, axes=attachment.axes)

    if attachment.observed:
        record(observed)

    return standing


def _whole(
    attachment: Attachment, record: Recorder
) -> Callable[[Module, tuple[object, ...], Output], Output]:
    """Return the hook for a point declared as the whole output of its module.

    Parameters
    ----------
    attachment : Attachment
        What the region installed at the point.
    record : Recorder
        Where a reading goes.

    Returns
    -------
    Callable[[Module, tuple[object, ...], Output], Output]
        A forward hook returning what the pass carries on with. It raises
        ``TypeError`` where the module returned something other than a tensor.
    """

    def read(_module: Module, _args: tuple[object, ...], output: Output) -> Output:
        if not isinstance(output, torch.Tensor):
            msg = (
                f"{attachment.path} declares the value at {attachment.site!r} as its "
                f"whole output, and it returned {output!r}"
            )
            raise TypeError(msg)

        return _filed(attachment, output, record)

    return read


def _positioned(
    attachment: Attachment, position: int, record: Recorder
) -> Callable[[Module, tuple[object, ...], Output], Output]:
    """Return the hook for a point declared at one position of its module's output.

    Parameters
    ----------
    attachment : Attachment
        What the region installed at the point.
    position : int
        Which position of the output the value sits at.
    record : Recorder
        Where a reading goes.

    Returns
    -------
    Callable[[Module, tuple[object, ...], Output], Output]
        A forward hook returning what the pass carries on with, which is the output
        it was handed with the value at the position replaced. It raises
        ``TypeError`` where the module returned no tuple or holds no tensor at the
        position, and ``ValueError`` where the output is shorter than the position.
    """

    def read(_module: Module, _args: tuple[object, ...], output: Output) -> Output:
        if not isinstance(output, tuple):
            msg = (
                f"{attachment.path} declares the value at {attachment.site!r} at "
                f"position {position} of its output, and it returned {output!r}"
            )
            raise TypeError(msg)

        if position >= len(output):
            msg = (
                f"{attachment.path} declares the value at {attachment.site!r} at "
                f"position {position} of its output, and it returned "
                f"{len(output)} positions"
            )
            raise ValueError(msg)

        held = output[position]

        if not isinstance(held, torch.Tensor):
            msg = (
                f"{attachment.path} holds {held!r} at position {position}, where "
                f"{attachment.site!r} declares a tensor"
            )
            raise TypeError(msg)

        standing = _filed(attachment, held, record)

        return (*output[:position], standing, *output[position + 1 :])

    return read


@runtime_checkable
class Hooking(Protocol):
    """Installs the reads and writes one region needs on a foreign model."""

    def install(self, attached: Sequence[Attachment], record: Recorder, /) -> Release:
        """Install one region's reads and writes and return what removes them.

        Parameters
        ----------
        attached : Sequence[Attachment]
            One entry per point the region was opened over.
        record : Recorder
            Where a reading goes.

        Returns
        -------
        Release
            What removes everything this installed, leaving the model as it stood.
        """


class TorchHooks:
    """Installs a region's reads and writes as forward hooks."""

    __slots__ = ()

    def install(self, attached: Sequence[Attachment], record: Recorder, /) -> Release:
        """Register one forward hook per point and return what removes them.

        Parameters
        ----------
        attached : Sequence[Attachment]
            One entry per point the region was opened over.
        record : Recorder
            Where a reading goes.

        Returns
        -------
        Release
            What removes the hooks, in the reverse of the order they went on.

        Raises
        ------
        Exception
            Whatever registering a hook raised, with every hook this call had
            already installed taken back off first.
        """
        hooks = Hooks()

        with ExitStack() as rollback:
            rollback.callback(hooks.remove)

            for attachment in attached:
                hooks.on_module(
                    attachment.module,
                    _whole(attachment, record)
                    if attachment.select is None
                    else _positioned(attachment, attachment.select, record),
                )

            rollback.pop_all()

        return hooks.remove

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this installer."""
        return "TorchHooks()"


TORCH_HOOKS: Final = TorchHooks()
"""The installer a foreign model is read through unless another is given."""


class ExternalScope:
    """The region a foreign model is read and patched over.

    Parameters
    ----------
    attached : Sequence[Attachment]
        One entry per point the region was opened over, already resolved against
        the model.
    hooking : Hooking
        What installs the reads and writes.
    """

    __slots__ = ("_attached", "_hooking", "_readings", "_release")

    def __init__(self, attached: Sequence[Attachment], hooking: Hooking) -> None:
        self._attached = tuple(attached)
        self._hooking = hooking
        self._readings: dict[Site, Observation] = {}
        self._release: Release | None = None

    @property
    def readings(self) -> Mapping[Site, Observation]:
        """Return what was read in the region, keyed by the point it came from."""
        return MappingProxyType(self._readings)

    def _record(self, observation: Observation) -> None:
        """Keep one reading under the point it was taken at.

        Parameters
        ----------
        observation : Observation
            The reading to keep.
        """
        self._readings[observation.site] = observation

    def __enter__(self) -> Self:
        """Return the region, with its points read and patched.

        Raises
        ------
        RuntimeError
            If the region is already open, since what a second opening installed
            would be removed twice.
        """
        if self._release is not None:
            msg = "this region is already open, and a region opens once"
            raise RuntimeError(msg)

        self._release = self._hooking.install(self._attached, self._record)

        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
        /,
    ) -> None:
        """Remove everything the region installed, on success and on failure alike.

        Parameters
        ----------
        kind : type[BaseException] | None
            The class of the exception leaving the region, or None.
        error : BaseException | None
            The exception leaving the region, or None.
        traceback : TracebackType | None
            Where it was raised, or None.

        Raises
        ------
        RuntimeError
            If the region never opened, so there is nothing to remove.
        """
        release = self._release

        if release is None:
            msg = "this region never opened, so there is nothing to remove"
            raise RuntimeError(msg)

        self._release = None
        release()

    @override
    def __repr__(self) -> str:
        """Return how many points are installed and whether the region is open."""
        return (
            f"ExternalScope({len(self._attached)} sites, "
            f"open={self._release is not None!r})"
        )


class ExternalModel:
    """Reads and patches a foreign model at the points its declaration names.

    Parameters
    ----------
    module : Module
        The foreign model, whose modules the declaration names by path.
    declared : ExternalSiteMap
        The points declared for it.
    hooking : Hooking
        What installs a region's reads and writes. The default installs them as
        forward hooks; an intervention library supplies this protocol instead,
        which is the one place such a library would be named.

    Raises
    ------
    TypeError
        If the model is not a module, the declaration is not an external map, or
        the installer does not install.
    """

    __slots__ = ("_declared", "_hooking", "_module")

    def __init__(
        self,
        module: Module,
        declared: ExternalSiteMap,
        *,
        hooking: Hooking = TORCH_HOOKS,
    ) -> None:
        if not isinstance(module, nn.Module):
            msg = f"a foreign model is a torch module, not {module!r}"
            raise TypeError(msg)

        if not isinstance(declared, ExternalSiteMap):
            msg = f"what a foreign model publishes is not an external map: {declared!r}"
            raise TypeError(msg)

        if not isinstance(hooking, Hooking):
            msg = f"what a region installs its reads through installs none: {hooking!r}"
            raise TypeError(msg)

        self._module = module
        self._declared = declared
        self._hooking = hooking

    @property
    def declared(self) -> ExternalSiteMap:
        """Return the points declared for the foreign model."""
        return self._declared

    def site_map(self) -> SiteMap:
        """Return the points an observer may name inside the foreign model.

        Returns
        -------
        SiteMap
            Each declared point, under what is found there.
        """
        return self._declared.published

    def run(self, inputs: Mapping[str, Tensor], /) -> Output:
        """Run the foreign module with named tensor inputs.

        Parameters
        ----------
        inputs : Mapping[str, Tensor]
            Nonempty keyword inputs for one forward pass.

        Returns
        -------
        Output
            The tensor or tensor-bearing tuple returned by the module.

        Raises
        ------
        TypeError
            If inputs are not a mapping, an input name is not a string, an input is
            not a tensor, or the module returns neither a tensor nor a tuple.
        ValueError
            If no input is supplied or an input name is empty.
        """
        if not isinstance(inputs, Mapping):
            msg = f"external model inputs are a mapping, not {inputs!r}"
            raise TypeError(msg)

        if not inputs:
            msg = "an external model runs on at least one named input"
            raise ValueError(msg)

        held: dict[str, Tensor] = {}

        for name, tensor in inputs.items():
            named = ensure_name("an external model input name", name)

            if not isinstance(tensor, torch.Tensor):
                msg = f"external model input {named!r} is not a tensor: {tensor!r}"
                raise TypeError(msg)

            held[named] = tensor

        output = self._module(**held)

        if not isinstance(output, torch.Tensor | tuple):
            msg = f"an external model returned neither a tensor nor a tuple: {output!r}"
            raise TypeError(msg)

        return output

    def instrument(
        self,
        sites: frozenset[Site],
        /,
        *,
        patches: Mapping[Site, Patch[Observation]],
    ) -> ExternalScope:
        """Return the region those points are read and patched over.

        Every point named is resolved here, once, by following the path its
        declaration gives one segment at a time from the model itself. A path is
        matched whole, so a declaration naming a module the model does not hold is
        refused rather than attached to a longer path it spells part of.

        Parameters
        ----------
        sites : frozenset[Site]
            The points to read, drawn from what the declaration names.
        patches : Mapping[Site, Patch[Observation]]
            What replaces the value at a point, keyed by the point.

        Returns
        -------
        ExternalScope
            The region, not yet entered.

        Raises
        ------
        ValueError
            If a point is not declared, or the model holds no module at the path
            declared for it.
        """
        attached: list[Attachment] = []

        for site in sorted(frozenset(sites) | frozenset(patches), key=repr):
            external = self._declared.declaration(site)

            attached.append(
                Attachment(
                    site,
                    module_named(
                        f"the declaration of {site!r}", self._module, external.path
                    ),
                    external.path,
                    external.select,
                    external.spec.axes,
                    site in sites,
                    patches.get(site),
                )
            )

        return ExternalScope(attached, self._hooking)

    @override
    def __repr__(self) -> str:
        """Return the declaration the foreign model is read through."""
        return f"ExternalModel({self._declared!r})"


class _RowPatch:
    """Gather, optionally patch, and scatter one subject scope's paired rows."""

    __slots__ = ("_batch_axis", "_pairs", "_row_axis", "_scope", "_side", "_site")

    def __init__(
        self,
        scope: _ExternalSubjectScope,
        site: Site,
        pairs: tuple[ModelReadyCausalPair, ...],
        side: Side,
        batch_axis: str,
        row_axis: str,
    ) -> None:
        self._scope = scope
        self._site = site
        self._pairs = pairs
        self._side = side
        self._batch_axis = batch_axis
        self._row_axis = row_axis

    def stands_in_for(self, reading: Observation, /) -> Tensor:
        """Return the full activation with only declared pair rows replaced."""
        if reading.site != self._site:
            msg = f"row-local patch for {self._site!r} received {reading.site!r}"
            raise ValueError(msg)

        axes = reading.axes

        if self._batch_axis not in axes or self._row_axis not in axes:
            msg = (
                f"{self._site!r} declares axes {axes!r}, which do not include "
                f"batch axis {self._batch_axis!r} and row axis {self._row_axis!r}"
            )
            raise ValueError(msg)

        batch_position = axes.index(self._batch_axis)
        row_position = axes.index(self._row_axis)
        moved = reading.tensor.movedim((batch_position, row_position), (0, 1))

        if int(moved.shape[0]) != len(self._pairs):
            msg = (
                f"{self._site!r} returned {int(moved.shape[0])} item rows for "
                f"{len(self._pairs)} causal pairs"
            )
            raise ValueError(msg)

        selected: list[Tensor] = []

        for position, pair in enumerate(self._pairs):
            live = pair.live_rows(self._side)

            if int(moved.shape[1]) != live:
                msg = (
                    f"{self._site!r} returned row extent {int(moved.shape[1])} for "
                    f"pair {pair.id!r}, whose tokenizer alignment declares {live}"
                )
                raise ValueError(msg)

            rows = pair.rows(self._side)

            if rows[-1] >= int(moved.shape[1]):
                msg = (
                    f"pair {pair.id!r} selects row {rows[-1]} outside the "
                    f"extent {int(moved.shape[1])} at {self._site!r}"
                )
                raise ValueError(msg)

            selected.append(moved[position, list(rows)])

        local_moved = torch.stack(selected)
        local = local_moved.movedim((0, 1), (batch_position, row_position))
        observed = Observation(self._site, local, axes=axes)
        patch = self._scope.patch(self._site)
        replacement = local if patch is None else patch.stands_in_for(observed)

        if not isinstance(replacement, torch.Tensor):
            msg = f"a patch at {self._site!r} returned no tensor: {replacement!r}"
            raise TypeError(msg)

        if tuple(replacement.shape) != tuple(local.shape):
            msg = (
                f"a row-local patch at {self._site!r} returned shape "
                f"{tuple(replacement.shape)}, not {tuple(local.shape)}"
            )
            raise ValueError(msg)

        if replacement.dtype != local.dtype or replacement.device != local.device:
            msg = (
                f"a row-local patch at {self._site!r} returned {replacement.dtype} "
                f"on {replacement.device}, not {local.dtype} on {local.device}"
            )
            raise ValueError(msg)

        if self._scope.observes(self._site):
            self._scope.record(Observation(self._site, replacement, axes=axes))

        replacement_moved = replacement.movedim((batch_position, row_position), (0, 1))
        standing = moved.clone()

        for position, pair in enumerate(self._pairs):
            standing[position, list(pair.rows(self._side))] = replacement_moved[
                position
            ]

        return standing.movedim((0, 1), (batch_position, row_position))


class _JointRowPatch:
    """Gather and scatter the union of both variables' declared model rows."""

    __slots__ = ("_batch_axis", "_pairs", "_row_axis", "_scope", "_side", "_site")

    def __init__(
        self,
        scope: _ExternalSubjectScope,
        site: Site,
        pairs: tuple[ModelReadyJointCausalPair, ...],
        side: Side,
        batch_axis: str,
        row_axis: str,
    ) -> None:
        self._scope = scope
        self._site = site
        self._pairs = pairs
        self._side = side
        self._batch_axis = batch_axis
        self._row_axis = row_axis

    def stands_in_for(self, reading: Observation, /) -> Tensor:
        """Return the full activation with both variables' row union replaced."""
        if reading.site != self._site:
            msg = f"joint row-local patch for {self._site!r} received {reading.site!r}"
            raise ValueError(msg)
        axes = reading.axes
        if self._batch_axis not in axes or self._row_axis not in axes:
            msg = (
                f"{self._site!r} declares axes {axes!r}, which omit required "
                "joint batch or row axes"
            )
            raise ValueError(msg)
        batch_position = axes.index(self._batch_axis)
        row_position = axes.index(self._row_axis)
        moved = reading.tensor.movedim((batch_position, row_position), (0, 1))
        if int(moved.shape[0]) != len(self._pairs):
            msg = (
                f"{self._site!r} returned {int(moved.shape[0])} items for "
                f"{len(self._pairs)} joint pairs"
            )
            raise ValueError(msg)

        selected: list[Tensor] = []
        for position, pair in enumerate(self._pairs):
            if int(moved.shape[1]) != pair.live_rows(self._side):
                msg = (
                    f"{self._site!r} returned the wrong live-row extent for {pair.id!r}"
                )
                raise ValueError(msg)
            rows = pair.selected_rows(self._side)
            if rows[-1] >= int(moved.shape[1]):
                msg = f"joint pair {pair.id!r} selects a row outside {self._site!r}"
                raise ValueError(msg)
            selected.append(moved[position, list(rows)])

        local_moved = torch.stack(selected)
        local = local_moved.movedim((0, 1), (batch_position, row_position))
        observed = Observation(self._site, local, axes=axes)
        patch = self._scope.patch(self._site)
        replacement = local if patch is None else patch.stands_in_for(observed)
        if not isinstance(replacement, torch.Tensor):
            msg = f"a joint patch at {self._site!r} returned no tensor"
            raise TypeError(msg)
        if (
            tuple(replacement.shape) != tuple(local.shape)
            or replacement.dtype != local.dtype
            or replacement.device != local.device
        ):
            msg = (
                f"a joint row-local patch at {self._site!r} changed its tensor contract"
            )
            raise ValueError(msg)
        if self._scope.observes(self._site):
            self._scope.record(Observation(self._site, replacement, axes=axes))

        replacement_moved = replacement.movedim((batch_position, row_position), (0, 1))
        standing = moved.clone()
        for position, pair in enumerate(self._pairs):
            standing[position, list(pair.selected_rows(self._side))] = (
                replacement_moved[position]
            )
        return standing.movedim((0, 1), (batch_position, row_position))


class _ExternalSubjectScope:
    """One row-local observation and patching region of an external subject."""

    __slots__ = (
        "_enter",
        "_leave",
        "_open",
        "_patches",
        "_readings",
        "_sites",
    )

    def __init__(
        self,
        sites: frozenset[Site],
        patches: Mapping[Site, Patch[Observation]],
        enter: Callable[[_ExternalSubjectScope], None],
        leave: Callable[[_ExternalSubjectScope], None],
    ) -> None:
        self._enter = enter
        self._leave = leave
        self._sites = sites
        self._patches = MappingProxyType(dict(patches))
        self._readings: dict[Site, Observation] = {}
        self._open = False

    @property
    def readings(self) -> Mapping[Site, Observation]:
        """Return row-local observations keyed by their declared sites."""
        return MappingProxyType(self._readings)

    @property
    def targets(self) -> frozenset[Site]:
        """Return every site this region reads or patches."""
        return self._sites | frozenset(self._patches)

    def observes(self, site: Site) -> bool:
        """Return whether this region records one site."""
        return site in self._sites

    def patch(self, site: Site) -> Patch[Observation] | None:
        """Return what replaces one site's paired rows, if anything."""
        return self._patches.get(site)

    def record(self, observation: Observation) -> None:
        """Keep one row-local observation under its site."""
        self._readings[observation.site] = observation

    def __enter__(self) -> Self:
        """Open this region on its subject."""
        if self._open:
            msg = "this external-subject region is already open"
            raise RuntimeError(msg)

        self._enter(self)
        self._open = True

        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
        /,
    ) -> None:
        """Close this region on success or failure."""
        del kind, error, traceback

        if not self._open:
            msg = "this external-subject region never opened"
            raise RuntimeError(msg)

        self._leave(self)
        self._open = False


def _decision(field: str, value: str | int) -> str | int:
    """Return one declared decision after validating its public representation."""
    if isinstance(value, bool) or not isinstance(value, str | int):
        msg = f"{field} is a nonempty name or nonnegative position, not {value!r}"
        raise TypeError(msg)

    return (
        ensure_name(field, value)
        if isinstance(value, str)
        else ensure_non_negative(field, value)
    )


class ExternalSubject:
    """Scores causal pairs through row-local interventions on an external model.

    Parameters
    ----------
    model : ExternalModel
        The declared external model to execute and instrument.
    policy : AlignmentPolicy
        The required source-to-model row policy.
    tokenizer : str
        The tokenizer identity every pair must carry.
    decisions : Mapping[HighLevelVariable, tuple[str | int, ...]]
        Ordered decision inventories, keyed by the high-level variable scored.
    scorer : DecisionScorer
        Differentiable mapping from model output to one row per pair over the
        requested decision inventory.
    batch_axis : str
        Axis naming model-input items at every causal site.
    row_axis : str
        Axis naming tokenizer rows at every causal site.

    Raises
    ------
    TypeError
        If the model, policy, decision mapping, decision values or scorer has the
        wrong runtime type.
    ValueError
        If the tokenizer or an axis name is empty, the two axes agree, or a
        decision inventory is empty or contains a duplicate.
    """

    __slots__ = (
        "_active",
        "_batch_axis",
        "_decisions",
        "_model",
        "_policy",
        "_row_axis",
        "_scorer",
        "_tokenizer",
    )

    def __init__(
        self,
        model: ExternalModel,
        policy: AlignmentPolicy,
        tokenizer: str,
        decisions: Mapping[HighLevelVariable, tuple[str | int, ...]],
        scorer: DecisionScorer,
        *,
        batch_axis: str,
        row_axis: str,
    ) -> None:
        if type(model) is not ExternalModel:
            msg = f"an external subject requires an ExternalModel, not {model!r}"
            raise TypeError(msg)

        if type(policy) is not AlignmentPolicy:
            msg = f"an external subject requires an AlignmentPolicy, not {policy!r}"
            raise TypeError(msg)

        if not callable(scorer):
            msg = f"an external subject scorer is not callable: {scorer!r}"
            raise TypeError(msg)

        if not isinstance(decisions, Mapping):
            msg = (
                f"an external subject requires decision inventories, not {decisions!r}"
            )
            raise TypeError(msg)

        batch = ensure_name("an external subject's batch axis", batch_axis)
        row = ensure_name("an external subject's row axis", row_axis)

        if batch == row:
            msg = f"batch and row axes are distinct, not both {batch!r}"
            raise ValueError(msg)

        inventories: dict[HighLevelVariable, tuple[str | int, ...]] = {}

        for variable, inventory in decisions.items():
            if type(variable) not in {
                TypeAt,
                SplitAt,
                CombinatorAt,
                NamedVariable,
            }:
                msg = f"a decision inventory is keyed by a variable, not {variable!r}"
                raise TypeError(msg)

            if type(inventory) is not tuple:
                msg = (
                    f"decision inventory at {variable!r} is an immutable tuple, "
                    f"not {inventory!r}"
                )
                raise TypeError(msg)

            held = tuple(
                _decision(f"decision {position} at {variable!r}", value)
                for position, value in enumerate(inventory)
            )

            if not held:
                msg = f"decision inventory at {variable!r} is empty"
                raise ValueError(msg)

            if len(set(held)) != len(held):
                msg = f"decision inventory at {variable!r} contains a duplicate"
                raise ValueError(msg)

            inventories[variable] = held

        if not inventories:
            msg = "an external subject declares at least one decision inventory"
            raise ValueError(msg)

        self._model = model
        self._policy = policy
        self._tokenizer = ensure_name("an external subject's tokenizer", tokenizer)
        self._decisions = MappingProxyType(inventories)
        self._scorer = scorer
        self._batch_axis = batch
        self._row_axis = row
        self._active: _ExternalSubjectScope | None = None

    @property
    def policy(self) -> AlignmentPolicy:
        """Return the required source-to-model row policy."""
        return self._policy

    def site_map(self) -> SiteMap:
        """Return the points published by the external model."""
        return self._model.site_map()

    def decisions(self, variable: HighLevelVariable, /) -> tuple[str | int, ...]:
        """Return the declared decisions at one variable in score order.

        Raises
        ------
        ValueError
            If the subject declares no inventory for the variable.
        """
        found = self._decisions.get(variable)

        if found is None:
            msg = f"an external subject declares no decisions at {variable!r}"
            raise ValueError(msg)

        return found

    def instrument(
        self,
        sites: frozenset[Site],
        /,
        *,
        patches: Mapping[Site, Patch[Observation]],
    ) -> _ExternalSubjectScope:
        """Return a region gathering and patching only pair-declared model rows."""
        for site in sites | frozenset(patches):
            declaration = self._model.declared.declaration(site)
            axes = declaration.spec.axes

            if site in sites and Access.OBSERVE not in declaration.spec.access:
                msg = f"{site!r} does not permit observation"
                raise ValueError(msg)

            if site in patches and Access.PATCH not in declaration.spec.access:
                msg = f"{site!r} does not permit patching"
                raise ValueError(msg)

            if self._batch_axis not in axes or self._row_axis not in axes:
                msg = (
                    f"{site!r} declares axes {axes!r}, not required batch and row "
                    f"axes {self._batch_axis!r} and {self._row_axis!r}"
                )
                raise ValueError(msg)

        return _ExternalSubjectScope(
            sites, patches, self._open_scope, self._close_scope
        )

    def _open_scope(self, scope: _ExternalSubjectScope) -> None:
        """Install one row-local subject scope."""
        if self._active is not None:
            msg = "an external subject opens one instrumentation region at a time"
            raise RuntimeError(msg)

        self._active = scope

    def _close_scope(self, scope: _ExternalSubjectScope) -> None:
        """Remove the currently installed row-local subject scope."""
        if self._active is not scope:
            msg = "the external-subject region being closed is not open"
            raise RuntimeError(msg)

        self._active = None

    def run(self, pairs: Sequence[ModelReadyCausalPair], side: Side, /) -> Tensor:
        """Run stored inputs from one side and score the pairs' shared variable.

        Raises
        ------
        TypeError
            If a pair is not model ready or the side is not one.
        ValueError
            If the population is empty, repeats an identity, mixes variables,
            policies, tokenizers, decisions, input contracts or row extents, or
            if the scorer returns a malformed or nonfinite score matrix.
        """
        if not isinstance(side, Side):
            msg = f"an external subject run requires a Side, not {side!r}"
            raise TypeError(msg)

        held = tuple(pairs)

        if not held:
            msg = "an external subject runs at least one causal pair"
            raise ValueError(msg)

        if any(not isinstance(pair, ModelReadyCausalPair) for pair in held):
            msg = "an external subject runs only model-ready causal pairs"
            raise TypeError(msg)

        identities = tuple(pair.id for pair in held)

        if len(set(identities)) != len(identities):
            msg = "an external subject run repeats a causal pair identity"
            raise ValueError(msg)

        variable = held[0].varied
        inventory = self.decisions(variable)
        first_inputs = held[0].inputs(side)
        names = tuple(first_inputs)
        first_rows = len(held[0].rows(side))
        first_live = held[0].live_rows(side)
        gathered: dict[str, list[Tensor]] = {name: [] for name in names}

        for pair in held:
            if pair.varied != variable:
                msg = f"an external subject run mixes {variable!r} with {pair.varied!r}"
                raise ValueError(msg)

            if pair.policy != self._policy:
                msg = (
                    f"pair {pair.id!r} uses policy {pair.policy!r}, not the "
                    f"subject policy {self._policy!r}"
                )
                raise ValueError(msg)

            if pair.tokenizer != self._tokenizer:
                msg = (
                    f"pair {pair.id!r} uses tokenizer {pair.tokenizer!r}, not "
                    f"{self._tokenizer!r}"
                )
                raise ValueError(msg)

            for assignment in (pair.base, pair.variant):
                decided = assignment[variable]

                if decided not in inventory:
                    msg = (
                        f"pair {pair.id!r} decides {variable!r} as {decided!r}, "
                        f"outside {inventory!r}"
                    )
                    raise ValueError(msg)

            if len(pair.rows(side)) != first_rows:
                msg = (
                    f"pair {pair.id!r} selects {len(pair.rows(side))} rows, not "
                    f"the population's paired extent {first_rows}"
                )
                raise ValueError(msg)

            if pair.live_rows(side) != first_live:
                msg = (
                    f"pair {pair.id!r} declares {pair.live_rows(side)} live rows, "
                    f"not the population's extent {first_live}"
                )
                raise ValueError(msg)

            inputs = pair.inputs(side)

            if set(inputs) != set(names):
                msg = (
                    f"pair {pair.id!r} names input keys {tuple(inputs)!r}, not "
                    f"the population keys {names!r}"
                )
                raise ValueError(msg)

            for name in names:
                tensor = inputs[name]
                expected = first_inputs[name]

                if (
                    tensor.dtype != expected.dtype
                    or tensor.device != expected.device
                    or tuple(tensor.shape) != tuple(expected.shape)
                ):
                    msg = (
                        f"model input {name!r} for pair {pair.id!r} has "
                        f"{tensor.dtype}, {tensor.device}, {tuple(tensor.shape)}, "
                        f"not {expected.dtype}, {expected.device}, "
                        f"{tuple(expected.shape)}"
                    )
                    raise ValueError(msg)

                gathered[name].append(tensor)

        batched = {
            name: torch.cat(tuple(tensors), dim=0) for name, tensors in gathered.items()
        }
        active = self._active

        if active is None:
            output = self._model.run(batched)
        else:
            row_patches = {
                site: _RowPatch(
                    active,
                    site,
                    held,
                    side,
                    self._batch_axis,
                    self._row_axis,
                )
                for site in active.targets
            }

            with self._model.instrument(frozenset(), patches=row_patches):
                output = self._model.run(batched)

        scores = self._scorer(output, variable, inventory)

        if not isinstance(scores, torch.Tensor) or not scores.is_floating_point():
            msg = f"an external subject scorer returned no floating tensor: {scores!r}"
            raise TypeError(msg)

        expected_shape = (len(held), len(inventory))

        if tuple(scores.shape) != expected_shape:
            msg = (
                f"an external subject scorer returned shape {tuple(scores.shape)}, "
                f"not {expected_shape}"
            )
            raise ValueError(msg)

        if not bool(torch.isfinite(scores).all().item()):
            msg = "an external subject scorer returned a nonfinite score"
            raise ValueError(msg)

        return scores

    @override
    def __repr__(self) -> str:
        """Return the external declaration, tokenizer and inventory count."""
        return (
            f"ExternalSubject({self._model.declared!r}, "
            f"tokenizer={self._tokenizer!r}, decisions={len(self._decisions)!r})"
        )


class ExternalJointSubject:
    """Score both variables of joint pairs through published row-local sites."""

    __slots__ = (
        "_active",
        "_batch_axis",
        "_decisions",
        "_model",
        "_policy",
        "_row_axis",
        "_scorer",
        "_tokenizer",
    )

    def __init__(
        self,
        model: ExternalModel,
        policy: AlignmentPolicy,
        tokenizer: str,
        decisions: Mapping[HighLevelVariable, tuple[str | int, ...]],
        scorer: DecisionScorer,
        *,
        batch_axis: str,
        row_axis: str,
    ) -> None:
        if type(model) is not ExternalModel:
            msg = f"an external joint subject requires an ExternalModel, not {model!r}"
            raise TypeError(msg)
        if type(policy) is not AlignmentPolicy:
            msg = "an external joint subject requires an AlignmentPolicy"
            raise TypeError(msg)
        if not callable(scorer):
            msg = "an external joint subject scorer is callable"
            raise TypeError(msg)
        if not isinstance(decisions, Mapping):
            msg = "an external joint subject requires decision inventories"
            raise TypeError(msg)

        batch = ensure_name("an external joint subject's batch axis", batch_axis)
        row = ensure_name("an external joint subject's row axis", row_axis)
        if batch == row:
            msg = f"joint batch and row axes are distinct, not both {batch!r}"
            raise ValueError(msg)
        inventories: dict[HighLevelVariable, tuple[str | int, ...]] = {}
        for variable, inventory in decisions.items():
            if isinstance(variable, NamedVariable):
                msg = "an external joint subject does not admit NamedVariable"
                raise TypeError(msg)
            if type(variable) not in {TypeAt, SplitAt, CombinatorAt}:
                msg = (
                    f"a joint decision inventory is keyed by no variable: {variable!r}"
                )
                raise TypeError(msg)
            if type(inventory) is not tuple:
                msg = f"joint decisions at {variable!r} are an immutable tuple"
                raise TypeError(msg)
            held = tuple(
                _decision(f"decision {position} at {variable!r}", value)
                for position, value in enumerate(inventory)
            )
            if not held or len(set(held)) != len(held):
                msg = f"joint decisions at {variable!r} are nonempty and unique"
                raise ValueError(msg)
            inventories[variable] = held
        if len(inventories) < 2:
            msg = "an external joint subject declares at least two decision inventories"
            raise ValueError(msg)

        self._model = model
        self._policy = policy
        self._tokenizer = ensure_name(
            "an external joint subject's tokenizer", tokenizer
        )
        self._decisions = MappingProxyType(inventories)
        self._scorer = scorer
        self._batch_axis = batch
        self._row_axis = row
        self._active: _ExternalSubjectScope | None = None

    @property
    def policy(self) -> AlignmentPolicy:
        """Return the required source-to-model row policy."""
        return self._policy

    def site_map(self) -> SiteMap:
        """Return the points published by the external model."""
        return self._model.site_map()

    def decisions(self, variable: HighLevelVariable, /) -> tuple[str | int, ...]:
        """Return the declared decisions at one joint variable."""
        found = self._decisions.get(variable)
        if found is None:
            msg = f"an external joint subject declares no decisions at {variable!r}"
            raise ValueError(msg)
        return found

    def instrument(
        self,
        sites: frozenset[Site],
        /,
        *,
        patches: Mapping[Site, Patch[Observation]],
    ) -> _ExternalSubjectScope:
        """Return a region restricted to published joint-pair rows."""
        for site in sites | frozenset(patches):
            declaration = self._model.declared.declaration(site)
            axes = declaration.spec.axes
            if site in sites and Access.OBSERVE not in declaration.spec.access:
                msg = f"{site!r} does not permit observation"
                raise ValueError(msg)
            if site in patches and Access.PATCH not in declaration.spec.access:
                msg = f"{site!r} does not permit patching"
                raise ValueError(msg)
            if self._batch_axis not in axes or self._row_axis not in axes:
                msg = f"{site!r} omits the external joint subject's batch or row axis"
                raise ValueError(msg)
        return _ExternalSubjectScope(
            sites, patches, self._open_scope, self._close_scope
        )

    def _open_scope(self, scope: _ExternalSubjectScope) -> None:
        if self._active is not None:
            msg = "an external joint subject opens one region at a time"
            raise RuntimeError(msg)
        self._active = scope

    def _close_scope(self, scope: _ExternalSubjectScope) -> None:
        if self._active is not scope:
            msg = "the external joint subject region being closed is not open"
            raise RuntimeError(msg)
        self._active = None

    def run(
        self, pairs: Sequence[ModelReadyJointCausalPair], side: Side, /
    ) -> tuple[Tensor, Tensor]:
        """Run one side once and score its two changed variables separately."""
        if not isinstance(side, Side):
            msg = f"an external joint subject run requires a Side, not {side!r}"
            raise TypeError(msg)
        held = tuple(pairs)
        if not held:
            msg = "an external joint subject runs at least one pair"
            raise ValueError(msg)
        if any(not isinstance(pair, ModelReadyJointCausalPair) for pair in held):
            msg = "an external joint subject runs only model-ready joint pairs"
            raise TypeError(msg)
        identities = tuple(pair.id for pair in held)
        if len(set(identities)) != len(identities):
            msg = "an external joint subject run repeats an identity"
            raise ValueError(msg)

        variables = held[0].variables
        inventories = tuple(self.decisions(variable) for variable in variables)
        first_inputs = held[0].inputs(side)
        names = tuple(first_inputs)
        first_rows = len(held[0].selected_rows(side))
        first_live = held[0].live_rows(side)
        gathered: dict[str, list[Tensor]] = {name: [] for name in names}

        for pair in held:
            if pair.variables != variables:
                msg = "an external joint subject run mixes variable pairs"
                raise ValueError(msg)
            if pair.policy != self._policy or pair.tokenizer != self._tokenizer:
                msg = (
                    f"joint pair {pair.id!r} differs from the subject's model contract"
                )
                raise ValueError(msg)
            for variable, inventory in zip(variables, inventories, strict=True):
                for assignment in (pair.base, pair.variant):
                    if assignment[variable] not in inventory:
                        msg = f"joint pair {pair.id!r} decides outside {inventory!r}"
                        raise ValueError(msg)
            if (
                len(pair.selected_rows(side)) != first_rows
                or pair.live_rows(side) != first_live
            ):
                msg = "an external joint subject run mixes row extents"
                raise ValueError(msg)
            inputs = pair.inputs(side)
            if set(inputs) != set(names):
                msg = "an external joint subject run mixes model input names"
                raise ValueError(msg)
            for name in names:
                tensor = inputs[name]
                expected = first_inputs[name]
                if (
                    tensor.dtype != expected.dtype
                    or tensor.device != expected.device
                    or tuple(tensor.shape) != tuple(expected.shape)
                ):
                    msg = f"joint model input {name!r} changes its tensor contract"
                    raise ValueError(msg)
                gathered[name].append(tensor)

        batched = {
            name: torch.cat(tuple(tensors), dim=0) for name, tensors in gathered.items()
        }
        active = self._active
        if active is None:
            output = self._model.run(batched)
        else:
            row_patches = {
                site: _JointRowPatch(
                    active,
                    site,
                    held,
                    side,
                    self._batch_axis,
                    self._row_axis,
                )
                for site in active.targets
            }
            with self._model.instrument(frozenset(), patches=row_patches):
                output = self._model.run(batched)

        scored = tuple(
            self._scorer(output, variable, inventory)
            for variable, inventory in zip(variables, inventories, strict=True)
        )
        for scores, inventory in zip(scored, inventories, strict=True):
            if not isinstance(scores, torch.Tensor) or not scores.is_floating_point():
                msg = "an external joint scorer returned no floating tensor"
                raise TypeError(msg)
            if tuple(scores.shape) != (len(held), len(inventory)):
                msg = (
                    f"external joint scores have malformed shape {tuple(scores.shape)}"
                )
                raise ValueError(msg)
            if not bool(torch.isfinite(scores).all().item()):
                msg = "an external joint scorer returned a nonfinite score"
                raise ValueError(msg)
        return (scored[0], scored[1])

    @override
    def __repr__(self) -> str:
        """Return the external declaration, tokenizer, and inventory count."""
        return (
            f"ExternalJointSubject({self._model.declared!r}, "
            f"tokenizer={self._tokenizer!r}, decisions={len(self._decisions)!r})"
        )

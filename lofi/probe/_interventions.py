"""Interventions on published model sites."""

from typing import TYPE_CHECKING, ClassVar, Final, override

import torch

from lofi._checks import ensure_finite, ensure_non_negative
from lofi.nn.sites import Site
from lofi.probe._circuits import Unit
from lofi.probe._erasure import LinearErasure
from lofi.probe._values import Observation, Trace

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import Tensor

__all__ = [
    "Interchange",
    "Intervention",
    "JointInterchange",
    "MeanFrom",
    "Noise",
    "PatchFrom",
    "Resample",
    "Shuffle",
    "Sparing",
    "Subspace",
    "Within",
    "Zero",
]

#: Maximum orthogonality error for a rotation matrix.
_ORTHOGONALITY_ATOL: Final[float] = 1e-4

#: Number of axes in a matrix.
_MATRIX: Final[int] = 2


def _checked_seed(seed: int) -> int:
    """Return the seed a stochastic variant draws under.

    Parameters
    ----------
    seed : int
        The seed to check.

    Returns
    -------
    int
        The seed, unchanged.

    Raises
    ------
    TypeError
        If the seed is not an integer.
    ValueError
        If the seed is negative.
    """
    return ensure_non_negative("an intervention's seed", seed)


def _generator(seed: int) -> torch.Generator:
    """Return the host generator one application draws from.

    Parameters
    ----------
    seed : int
        The seed to draw under.

    Returns
    -------
    torch.Generator
        A generator on the host, seeded.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)

    return generator


def _closed(trace: Trace, what: str) -> Trace:
    """Return a trace an intervention may read.

    Parameters
    ----------
    trace : Trace
        The trace to check.
    what : str
        What the trace is for, for the message.

    Returns
    -------
    Trace
        The trace, unchanged.

    Raises
    ------
    TypeError
        If the value is not a trace.
    ValueError
        If the trace has not closed.
    """
    if not isinstance(trace, Trace):
        msg = f"the {what} is not a Trace: {trace!r}"
        raise TypeError(msg)

    if not trace.closed:
        msg = f"the {what} is still open, so what it holds could still change"
        raise ValueError(msg)

    return trace


def _pool(trace: Trace, site: Site) -> Tensor:
    """Return every unmasked number a trace recorded at one point, flattened.

    Parameters
    ----------
    trace : Trace
        The trace to read.
    site : Site
        The point to read it at.

    Returns
    -------
    Tensor
        Rank 1 over the recorded entries the mask keeps, in the recorded dtype.

    Raises
    ------
    KeyError
        If the trace read nothing at that point.
    ValueError
        If everything recorded there is masked off.
    """
    recorded = trace[site]
    kept = recorded.tensor[recorded.covering()]

    if kept.numel() == 0:
        msg = f"everything recorded at {site!r} is masked off"
        raise ValueError(msg)

    return kept


class Zero:
    """Replaces the value at the point with zeros.

    Attributes
    ----------
    tag : ClassVar[str]
        The discriminator this variant answers to.
    """

    __slots__ = ()

    tag: ClassVar[str] = "zero"

    def stands_in_for(self, observation: Observation, /) -> Tensor:
        """Return zeros in place of what was read.

        Parameters
        ----------
        observation : Observation
            What is at the point now, which fixes the shape, dtype and placement of what
            replaces it.

        Returns
        -------
        Tensor
            Zeros, of the reading's shape, dtype and placement.
        """
        return torch.zeros_like(observation.tensor)

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this intervention."""
        return "Zero()"


class MeanFrom:
    """Replaces the value at the point with the mean an earlier trace saw there.

    Attributes
    ----------
    tag : ClassVar[str]
        The discriminator this variant answers to.
    trace : Trace
        The closed trace the mean is taken over.

    Parameters
    ----------
    trace : Trace
        The trace to read, which must have closed.

    Raises
    ------
    TypeError
        If the trace is not a trace.
    ValueError
        If the trace is still open.
    """

    __slots__ = ("_trace",)

    tag: ClassVar[str] = "mean-from"

    def __init__(self, trace: Trace) -> None:
        self._trace = _closed(trace, "trace a mean is taken over")

    @property
    def trace(self) -> Trace:
        """Return the closed trace the mean is taken over."""
        return self._trace

    def stands_in_for(self, observation: Observation, /) -> Tensor:
        """Return the recorded mean, filled to the reading's shape.

        Parameters
        ----------
        observation : Observation
            What is at the point now.

        Returns
        -------
        Tensor
            The reading's shape, dtype and placement, every entry the mean of what the
            trace recorded at that point.

        Raises
        ------
        KeyError
            If the trace read nothing at that point.
        TypeError
            If the reading is not floating point.
        ValueError
            If everything it recorded there is masked off.
        """
        tensor = observation.tensor

        if not tensor.is_floating_point():
            msg = f"a mean cannot be written into a reading held in {tensor.dtype}"
            raise TypeError(msg)

        pool = _pool(self._trace, observation.site)

        return torch.full_like(tensor, pool.to(torch.float64).mean().item())

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this intervention."""
        return f"MeanFrom({self._trace!r})"


class Resample:
    """Replaces the value at the point with numbers drawn from an earlier trace.

    Attributes
    ----------
    tag : ClassVar[str]
        The discriminator this variant answers to.
    source : Trace
        The closed trace the numbers are drawn from.
    seed : int
        The seed the draw runs under.

    Parameters
    ----------
    source : Trace
        The trace to draw from, which must have closed.
    seed : int
        The seed the draw runs under.

    Raises
    ------
    TypeError
        If the source is not a trace, or the seed is not an integer.
    ValueError
        If the source is still open, or the seed is negative.
    """

    __slots__ = ("_seed", "_source")

    tag: ClassVar[str] = "resample"

    def __init__(self, source: Trace, seed: int) -> None:
        self._source = _closed(source, "trace numbers are drawn from")
        self._seed = _checked_seed(seed)

    @property
    def source(self) -> Trace:
        """Return the closed trace the numbers are drawn from."""
        return self._source

    @property
    def seed(self) -> int:
        """Return the seed the draw runs under."""
        return self._seed

    def stands_in_for(self, observation: Observation, /) -> Tensor:
        """Return numbers drawn from the source, one per entry of the reading.

        Parameters
        ----------
        observation : Observation
            What is at the point now.

        Returns
        -------
        Tensor
            The reading's shape, dtype and placement, every entry drawn with replacement
            from what the source recorded at that point.

        Raises
        ------
        KeyError
            If the source read nothing at that point.
        ValueError
            If everything it recorded there is masked off.
        """
        tensor = observation.tensor
        pool = _pool(self._source, observation.site)
        drawn = torch.randint(
            pool.numel(),
            tuple(tensor.shape),
            generator=_generator(self._seed),
        )

        return pool.reshape(-1)[drawn].to(device=tensor.device, dtype=tensor.dtype)

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this intervention."""
        return f"Resample({self._source!r}, seed={self._seed!r})"


class PatchFrom:
    """Replaces the value at the point with what a trace held at another point.

    Attributes
    ----------
    tag : ClassVar[str]
        The discriminator this variant answers to.
    trace : Trace
        The closed trace the replacement is read from.
    site : Site
        The point in that trace the replacement is read at.

    Parameters
    ----------
    trace : Trace
        The trace to read, which must have closed.
    site : Site
        The point in it to read at.

    Raises
    ------
    TypeError
        If the trace is not a trace, or the site is not a site.
    ValueError
        If the trace is still open.
    """

    __slots__ = ("_site", "_trace")

    tag: ClassVar[str] = "patch-from"

    def __init__(self, trace: Trace, site: Site) -> None:
        if not isinstance(site, Site):
            msg = f"the point a patch is read at is not a Site: {site!r}"
            raise TypeError(msg)

        self._trace = _closed(trace, "trace a patch is read from")
        self._site = site

    @property
    def trace(self) -> Trace:
        """Return the closed trace the replacement is read from."""
        return self._trace

    @property
    def site(self) -> Site:
        """Return the point in that trace the replacement is read at."""
        return self._site

    def stands_in_for(self, observation: Observation, /) -> Tensor:
        """Return what the trace held at the source point.

        Parameters
        ----------
        observation : Observation
            What is at the point now, which fixes the shape the replacement must have
            and the placement it is moved to.

        Returns
        -------
        Tensor
            The recorded numbers, at the reading's dtype and placement.

        Raises
        ------
        KeyError
            If the trace read nothing at the source point.
        TypeError
            If one of the two is floating point and the other is not.
        ValueError
            If what it read there is of a different shape than the reading, which is a
            patch between two points that do not correspond.
        """
        tensor = observation.tensor
        recorded = self._trace[self._site].tensor

        if recorded.is_floating_point() != tensor.is_floating_point():
            msg = (
                f"a reading held in {recorded.dtype} cannot replace one held "
                f"in {tensor.dtype}"
            )
            raise TypeError(msg)

        if tuple(recorded.shape) != tuple(tensor.shape):
            msg = (
                f"the reading at {self._site!r} is of shape "
                f"{tuple(recorded.shape)} and cannot replace one of shape "
                f"{tuple(tensor.shape)}"
            )
            raise ValueError(msg)

        return recorded.to(device=tensor.device, dtype=tensor.dtype)

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this intervention."""
        return f"PatchFrom({self._trace!r}, {self._site!r})"


class Noise:
    """Adds centered Gaussian noise of a declared scale to the value at the point.

    Attributes
    ----------
    tag : ClassVar[str]
        The discriminator this variant answers to.
    sigma : float
        The standard deviation of the noise.
    seed : int
        The seed the draw runs under.

    Parameters
    ----------
    sigma : float
        The standard deviation of the noise, which is positive: a scale of zero is a
        no-op wearing the shape of an intervention.
    seed : int
        The seed the draw runs under.

    Raises
    ------
    TypeError
        If the scale is not a number, or the seed is not an integer.
    ValueError
        If the scale is not finite and positive, or the seed is negative.
    """

    __slots__ = ("_seed", "_sigma")

    tag: ClassVar[str] = "noise"

    def __init__(self, sigma: float, seed: int) -> None:
        scale = ensure_finite("an intervention's noise scale", sigma)

        if scale <= 0.0:
            msg = f"an intervention's noise scale is positive, not {scale}"
            raise ValueError(msg)

        self._sigma = scale
        self._seed = _checked_seed(seed)

    @property
    def sigma(self) -> float:
        """Return the standard deviation of the noise."""
        return self._sigma

    @property
    def seed(self) -> int:
        """Return the seed the draw runs under."""
        return self._seed

    def stands_in_for(self, observation: Observation, /) -> Tensor:
        """Return the reading with centered noise of the declared scale added.

        Parameters
        ----------
        observation : Observation
            What is at the point now.

        Returns
        -------
        Tensor
            The reading's shape, dtype and placement, each entry moved by an independent
            draw.

        Raises
        ------
        TypeError
            If the reading is not floating point.
        """
        tensor = observation.tensor

        if not tensor.is_floating_point():
            msg = f"noise cannot be added to a reading held in {tensor.dtype}"
            raise TypeError(msg)

        drawn = torch.randn(
            tuple(tensor.shape),
            generator=_generator(self._seed),
            dtype=torch.float32,
        )

        return tensor + self._sigma * drawn.to(device=tensor.device, dtype=tensor.dtype)

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this intervention."""
        return f"Noise({self._sigma!r}, seed={self._seed!r})"


class Shuffle:
    """Permutes the value at the point along its outermost axis.

    Attributes
    ----------
    tag : ClassVar[str]
        The discriminator this variant answers to.
    seed : int
        The seed the permutation is drawn under.

    Parameters
    ----------
    seed : int
        The seed the permutation is drawn under.

    Raises
    ------
    TypeError
        If the seed is not an integer.
    ValueError
        If the seed is negative.
    """

    __slots__ = ("_seed",)

    tag: ClassVar[str] = "shuffle"

    def __init__(self, seed: int) -> None:
        self._seed = _checked_seed(seed)

    @property
    def seed(self) -> int:
        """Return the seed the permutation is drawn under."""
        return self._seed

    def stands_in_for(self, observation: Observation, /) -> Tensor:
        """Return the reading permuted along its outermost axis.

        Parameters
        ----------
        observation : Observation
            What is at the point now.

        Returns
        -------
        Tensor
            The reading's shape, dtype and placement, its outermost positions in a drawn
            order.

        Raises
        ------
        ValueError
            If the reading is scalar and has no axis to permute.
        """
        tensor = observation.tensor

        if tensor.ndim == 0:
            msg = "a single number has no axis to shuffle along"
            raise ValueError(msg)

        order = torch.randperm(
            int(tensor.shape[0]), generator=_generator(self._seed)
        ).to(tensor.device)

        return tensor[order]

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this intervention."""
        return f"Shuffle(seed={self._seed!r})"


def _checked_rotation(rotation: Tensor) -> Tensor:
    """Return the rotation a subspace is read in.

    Parameters
    ----------
    rotation : Tensor
        The matrix to check.

    Returns
    -------
    Tensor
        The same matrix with its gradient connection intact.

    Raises
    ------
    TypeError
        If it is not a tensor, or is not floating point.
    ValueError
        If it is not a square matrix, or is not orthogonal.
    """
    if not isinstance(rotation, torch.Tensor):
        msg = f"a subspace's rotation is not a tensor: {rotation!r}"
        raise TypeError(msg)

    if not rotation.is_floating_point():
        msg = f"a subspace's rotation is held in {rotation.dtype}"
        raise TypeError(msg)

    if rotation.ndim != _MATRIX or rotation.shape[0] != rotation.shape[1]:
        msg = (
            "a subspace's rotation is a square matrix, not one of shape "
            f"{tuple(rotation.shape)}"
        )
        raise ValueError(msg)

    exact = rotation.detach().to(torch.float64)
    identity = torch.eye(exact.shape[0], dtype=torch.float64, device=exact.device)

    if not torch.allclose(
        exact.T @ exact, identity, rtol=0.0, atol=_ORTHOGONALITY_ATOL
    ):
        msg = "a subspace's rotation is not orthogonal, so it defines no projection"
        raise ValueError(msg)

    return rotation


def _checked_rank(rotation: Tensor, rank: int) -> int:
    """Return a rank as a number of leading coordinates of a rotated basis.

    Parameters
    ----------
    rotation : Tensor
        The checked rotation the basis is read in.
    rank : int
        The number of leading coordinates.

    Returns
    -------
    int
        The rank.

    Raises
    ------
    TypeError
        If the rank is not an integer.
    ValueError
        If the rank is not between one and the matrix's side.
    """
    side = int(rotation.shape[0])
    held = ensure_non_negative("a subspace's rank", rank)

    if held < 1 or held > side:
        msg = (
            f"a basis of {side} coordinates holds a subspace of rank 1 to "
            f"{side}, not {held}"
        )
        raise ValueError(msg)

    return held


def _rotated(rotation: Tensor, tensor: Tensor, subject: str) -> Tensor:
    """Return a reading's innermost axis rewritten in a rotated basis.

    Parameters
    ----------
    rotation : Tensor
        The orthogonal matrix the basis is read in.
    tensor : Tensor
        The reading to rotate.
    subject : str
        What the reading is, for a message.

    Returns
    -------
    Tensor
        The reading times the rotation, at the reading's dtype and placement.

    Raises
    ------
    TypeError
        If the reading is not floating point.
    ValueError
        If the reading's innermost axis is not the rotation's side.
    """
    if not tensor.is_floating_point():
        msg = f"{subject} held in {tensor.dtype} cannot be rotated"
        raise TypeError(msg)

    side = int(rotation.shape[0])

    if tensor.ndim == 0 or int(tensor.shape[-1]) != side:
        msg = (
            f"a rotation over {side} coordinates does not act on the "
            f"innermost axis of {subject} of shape {tuple(tensor.shape)}"
        )
        raise ValueError(msg)

    return tensor @ rotation.to(device=tensor.device, dtype=tensor.dtype)


class Subspace:
    """Projects the named coordinates of a rotated basis out of the value.

    Attributes
    ----------
    tag : ClassVar[str]
        The discriminator this variant answers to.
    rotation : Tensor
        The orthogonal matrix the subspace is read in.
    rank : int
        How many leading coordinates of that basis the alignment was fitted at.
    coordinates : tuple[int, ...]
        Which of those coordinates this intervention clears.

    Parameters
    ----------
    rotation : Tensor
        The orthogonal matrix the subspace is read in, square and floating point.
    rank : int
        How many leading coordinates of that basis were fitted, at least one and at most
        the matrix's side.
    coordinates : Iterable[int]
        Which of those coordinates to clear, each below the rank and each named once.

    Raises
    ------
    TypeError
        If the rotation is not a floating point tensor, or a coordinate or the rank is
        not an integer.
    ValueError
        If the rotation is not a square orthogonal matrix, if the rank is not between
        one and its side, or if a coordinate is repeated or is at or beyond the rank.
    """

    __slots__ = ("_coordinates", "_rank", "_rotation")

    tag: ClassVar[str] = "subspace"

    def __init__(self, rotation: Tensor, rank: int, coordinates: Iterable[int]) -> None:
        checked = _checked_rotation(rotation).detach()
        held = _checked_rank(checked, rank)
        chosen: list[int] = []

        for coordinate in coordinates:
            index = ensure_non_negative("a subspace's coordinate", coordinate)

            if index >= held:
                msg = f"coordinate {index} is at or beyond a subspace of rank {held}"
                raise ValueError(msg)

            if index in chosen:
                msg = f"a subspace names coordinate {index} twice"
                raise ValueError(msg)

            chosen.append(index)

        self._rotation = checked
        self._rank = held
        self._coordinates = tuple(chosen)

    @property
    def rotation(self) -> Tensor:
        """Return the orthogonal matrix the subspace is read in."""
        return self._rotation

    @property
    def rank(self) -> int:
        """Return how many leading coordinates of that basis were fitted."""
        return self._rank

    @property
    def coordinates(self) -> tuple[int, ...]:
        """Return which of those coordinates this intervention clears."""
        return self._coordinates

    def stands_in_for(self, observation: Observation, /) -> Tensor:
        """Return the reading with the named coordinates of the basis cleared.

        Parameters
        ----------
        observation : Observation
            What is at the point now.

        Returns
        -------
        Tensor
            The reading's shape, dtype and placement, with the component in the named
            coordinates of the rotated basis removed.

        Raises
        ------
        TypeError
            If the reading is not floating point.
        ValueError
            If the reading's innermost axis is not the rotation's side.
        """
        tensor = observation.tensor
        rotated = _rotated(self._rotation, tensor, "a reading")

        if self._coordinates:
            cleared = torch.tensor(self._coordinates, device=tensor.device)
            rotated = rotated.index_fill(-1, cleared, 0.0)

        return rotated @ self._rotation.to(device=tensor.device, dtype=tensor.dtype).T

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this intervention."""
        return (
            f"Subspace(<{int(self._rotation.shape[0])}x"
            f"{int(self._rotation.shape[0])}>, rank={self._rank!r}, "
            f"coordinates={self._coordinates!r})"
        )


class Interchange:
    """Blends a source into rotated coordinates under one explicit mask.

    Attributes
    ----------
    tag : ClassVar[str]
        The discriminator this variant answers to.
    rotation : Tensor
        The orthogonal matrix the subspace is read in.
    mask : Tensor
        The share of every rotated coordinate taken from the source.
    source : Tensor
        The value the coordinates are taken from, of the reading's own shape.

    Parameters
    ----------
    rotation : Tensor
        The orthogonal matrix the subspace is read in, square and floating point. It
        retains its gradient connection, so replacement gradients reach the parameters
        from which the rotation was built.
    mask : Tensor
        A floating rank-one tensor with one finite share from zero to one per
        rotated coordinate. It retains its gradient connection.
    source : Tensor
        The value those coordinates are read from, floating point and of the shape of
        the reading it will stand in for.

    Raises
    ------
    TypeError
        If the rotation, mask or source is not a floating point tensor.
    ValueError
        If the rotation is not a square orthogonal matrix, the mask does not have one
        finite share per coordinate at the rotation's dtype and placement, or the
        source's innermost axis is not that side.
    """

    __slots__ = ("_mask", "_rotation", "_source")

    tag: ClassVar[str] = "interchange"

    def __init__(self, rotation: Tensor, mask: Tensor, source: Tensor) -> None:
        checked = _checked_rotation(rotation)

        if not isinstance(mask, torch.Tensor):
            msg = f"an interchange's mask is not a tensor: {mask!r}"
            raise TypeError(msg)

        if not mask.is_floating_point():
            msg = f"an interchange's mask is held in {mask.dtype}"
            raise TypeError(msg)

        if not isinstance(source, torch.Tensor):
            msg = f"an interchange's source is not a tensor: {source!r}"
            raise TypeError(msg)

        if not source.is_floating_point():
            msg = f"an interchange's source is held in {source.dtype}"
            raise TypeError(msg)

        side = int(checked.shape[0])

        if mask.ndim != 1 or tuple(mask.shape) != (side,):
            msg = (
                f"a rotation over {side} coordinates needs a mask of shape "
                f"{(side,)}, not {tuple(mask.shape)}"
            )
            raise ValueError(msg)

        if mask.dtype != checked.dtype:
            msg = (
                f"an interchange's mask is held in {mask.dtype}, not the "
                f"rotation's {checked.dtype}"
            )
            raise ValueError(msg)

        if mask.device != checked.device:
            msg = (
                f"an interchange's mask is on {mask.device}, not the rotation's "
                f"{checked.device}"
            )
            raise ValueError(msg)

        if not bool(torch.isfinite(mask).all()):
            msg = "an interchange's mask contains a nonfinite share"
            raise ValueError(msg)

        if bool(((mask < 0.0) | (mask > 1.0)).any()):
            msg = "an interchange's mask contains a share outside zero to one"
            raise ValueError(msg)

        if source.ndim == 0 or int(source.shape[-1]) != side:
            msg = (
                f"a rotation over {side} coordinates does not act on the "
                f"innermost axis of a source of shape {tuple(source.shape)}"
            )
            raise ValueError(msg)

        if source.dtype != checked.dtype:
            msg = (
                f"an interchange's source is held in {source.dtype}, not the "
                f"rotation's {checked.dtype}"
            )
            raise ValueError(msg)

        if source.device != checked.device:
            msg = (
                f"an interchange's source is on {source.device}, not the rotation's "
                f"{checked.device}"
            )
            raise ValueError(msg)

        self._rotation = checked
        self._mask = mask
        self._source = source.detach()

    @property
    def rotation(self) -> Tensor:
        """Return the orthogonal matrix the subspace is read in."""
        return self._rotation

    @property
    def mask(self) -> Tensor:
        """Return the share of every rotated coordinate taken from the source."""
        return self._mask

    @property
    def source(self) -> Tensor:
        """Return the value the coordinates are taken from."""
        return self._source

    def stands_in_for(self, observation: Observation, /) -> Tensor:
        """Return the reading with its masked rotated coordinates from the source.

        Parameters
        ----------
        observation : Observation
            What is at the point now.

        Returns
        -------
        Tensor
            The reading's shape, dtype and placement, blending the source and reading
            components by the mask in the rotated basis.

        Raises
        ------
        TypeError
            If the reading is not floating point.
        ValueError
            If the reading's innermost axis is not the rotation's side, or the source is
            not of the reading's shape.
        """
        tensor = observation.tensor

        if tuple(self._source.shape) != tuple(tensor.shape):
            msg = (
                f"an interchange from a source of shape {tuple(self._source.shape)} "
                f"cannot stand in for a reading of shape {tuple(tensor.shape)}"
            )
            raise ValueError(msg)

        if tensor.dtype != self._source.dtype or tensor.device != self._source.device:
            msg = (
                "an interchange's source and reading do not share a dtype and placement"
            )
            raise ValueError(msg)

        rotation = self._rotation.to(device=tensor.device, dtype=tensor.dtype)
        rotated = _rotated(rotation, tensor, "a reading")
        taken = _rotated(rotation, self._source.to(device=tensor.device), "a source")
        mask = self._mask.to(device=tensor.device, dtype=tensor.dtype)

        return (mask * taken + (1.0 - mask) * rotated) @ rotation.T

    @override
    def __repr__(self) -> str:
        """Return the rotation side, mask shape and source shape."""
        side = int(self._rotation.shape[0])

        return (
            f"Interchange(<{side}x{side}>, mask=<{tuple(self._mask.shape)}>, "
            f"source=<{tuple(self._source.shape)}>)"
        )


class JointInterchange:
    """Patch two disjoint rotated coordinate sets from one joint source at once."""

    __slots__ = ("_first", "_rotation", "_second", "_source")

    tag: ClassVar[str] = "joint-interchange"

    def __init__(
        self,
        rotation: Tensor,
        first_mask: Tensor,
        second_mask: Tensor,
        source: Tensor,
    ) -> None:
        checked = _checked_rotation(rotation)
        side = int(checked.shape[0])

        for name, mask in (("first", first_mask), ("second", second_mask)):
            if not isinstance(mask, torch.Tensor) or not mask.is_floating_point():
                msg = f"a joint interchange's {name} mask is a floating tensor"
                raise TypeError(msg)
            if tuple(mask.shape) != (side,):
                msg = (
                    f"a joint interchange's {name} mask has shape "
                    f"{tuple(mask.shape)}, not {(side,)}"
                )
                raise ValueError(msg)
            if mask.dtype != checked.dtype or mask.device != checked.device:
                msg = (
                    f"a joint interchange's {name} mask does not share its "
                    "rotation's dtype and placement"
                )
                raise ValueError(msg)
            if not bool(torch.isfinite(mask).all()) or bool(
                ((mask < 0.0) | (mask > 1.0)).any()
            ):
                msg = (
                    f"a joint interchange's {name} mask is not finite within "
                    "zero to one"
                )
                raise ValueError(msg)

        if bool(((first_mask != 0.0) & (second_mask != 0.0)).any()):
            msg = "a joint interchange's two masks overlap"
            raise ValueError(msg)
        if not bool((first_mask != 0.0).any()) or not bool((second_mask != 0.0).any()):
            msg = "a joint interchange assigns at least one coordinate to each mask"
            raise ValueError(msg)
        if not isinstance(source, torch.Tensor) or not source.is_floating_point():
            msg = "a joint interchange's source is a floating point tensor"
            raise TypeError(msg)
        if source.ndim == 0 or int(source.shape[-1]) != side:
            msg = (
                f"a rotation over {side} coordinates does not act on the "
                f"innermost axis of a source of shape {tuple(source.shape)}"
            )
            raise ValueError(msg)
        if source.dtype != checked.dtype or source.device != checked.device:
            msg = (
                "a joint interchange's source does not share its rotation's "
                "dtype and placement"
            )
            raise ValueError(msg)

        self._rotation = checked
        self._first = first_mask
        self._second = second_mask
        self._source = source.detach()

    @property
    def rotation(self) -> Tensor:
        """Return the one orthogonal matrix shared by both masks."""
        return self._rotation

    @property
    def masks(self) -> tuple[Tensor, Tensor]:
        """Return the two disjoint masks in dominance order."""
        return (self._first, self._second)

    @property
    def source(self) -> Tensor:
        """Return the joint counterfactual source."""
        return self._source

    def stands_in_for(self, observation: Observation, /) -> Tensor:
        """Return one simultaneous, order-independent joint patch."""
        tensor = observation.tensor
        if tuple(self._source.shape) != tuple(tensor.shape):
            msg = (
                f"a joint source of shape {tuple(self._source.shape)} cannot stand "
                f"in for a reading of shape {tuple(tensor.shape)}"
            )
            raise ValueError(msg)
        if tensor.dtype != self._source.dtype or tensor.device != self._source.device:
            msg = (
                "a joint interchange's source and reading differ in dtype or placement"
            )
            raise ValueError(msg)

        rotation = self._rotation.to(device=tensor.device, dtype=tensor.dtype)
        current = _rotated(rotation, tensor, "a reading")
        source = _rotated(rotation, self._source, "a source")
        selected = (self._first + self._second).to(
            device=tensor.device, dtype=tensor.dtype
        )
        return (selected * source + (1.0 - selected) * current) @ rotation.T

    @override
    def __repr__(self) -> str:
        """Return the rotation side and the two mask widths."""
        side = int(self._rotation.shape[0])
        widths = (
            int((self._first != 0.0).sum().item()),
            int((self._second != 0.0).sum().item()),
        )
        return f"JointInterchange(<{side}x{side}>, widths={widths!r})"


def _checked_units(units: Iterable[Unit]) -> frozenset[Unit]:
    """Return the units a scoped intervention indexes with.

    Parameters
    ----------
    units : Iterable[Unit]
        The units, in any order and with repeats.

    Returns
    -------
    frozenset[Unit]
        The units, each once.

    Raises
    ------
    TypeError
        If a member is not a unit.
    """
    held = frozenset(units)

    for unit in held:
        if type(unit) is not Unit:
            msg = f"a scoped intervention indexes with units, not {unit!r}"
            raise TypeError(msg)

    return held


def _checked_variant(intervention: Intervention) -> Intervention:
    """Return an intervention a scope may apply.

    Parameters
    ----------
    intervention : Intervention
        The value to check.

    Returns
    -------
    Intervention
        The intervention, unchanged.

    Raises
    ------
    TypeError
        If the value is not one of the declared variants.
    """
    if type(intervention) not in {
        Zero,
        MeanFrom,
        Resample,
        PatchFrom,
        Noise,
        Shuffle,
        Subspace,
        Interchange,
        JointInterchange,
        Within,
        Sparing,
        LinearErasure,
    }:
        msg = f"a scope applies one of the declared interventions, not {intervention!r}"
        raise TypeError(msg)

    return intervention


def _covered(units: frozenset[Unit], observation: Observation) -> Tensor:
    """Return which entries of a reading some unit indexes.

    Parameters
    ----------
    units : frozenset[Unit]
        The units, each at the reading's point. A unit with no coordinate indexes
        every entry.
    observation : Observation
        The reading.

    Returns
    -------
    Tensor
        Of the reading's shape and placement, holding truth values, true where some
        unit indexes.

    Raises
    ------
    KeyError
        If a unit names an axis the reading does not declare.
    ValueError
        If a unit is at another point than the reading, or a coordinate lies beyond
        its axis.
    """
    covered = torch.zeros_like(observation.tensor, dtype=torch.bool)
    # the units naming one set of axes, with their coordinates in that axis order
    rows: dict[tuple[str, ...], list[tuple[Unit, tuple[int, ...]]]] = {}

    for unit in units:
        if unit.site != observation.site:
            msg = (
                f"{unit!r} is not at {observation.site!r}, so it indexes nothing there"
            )
            raise ValueError(msg)

        axes = tuple(sorted(unit.index))
        rows.setdefault(axes, []).append(
            (unit, tuple(unit.index[axis] for axis in axes))
        )

    for axes, held in rows.items():
        index: list[slice | Tensor] = [slice(None)] * observation.rank
        columns = torch.tensor(
            [coordinates for _, coordinates in held], device=covered.device
        ).reshape(len(held), len(axes))

        for offset, axis in enumerate(axes):
            extent = observation.extent(axis)
            beyond = columns[:, offset] >= extent

            if bool(beyond.any()):
                unit, coordinates = held[int(beyond.nonzero()[0])]
                msg = (
                    f"{unit!r} indexes coordinate {coordinates[offset]} on axis "
                    f"{axis!r}, which has {extent} positions"
                )
                raise ValueError(msg)

            index[observation.position(axis)] = columns[:, offset]

        covered[tuple(index)] = True

    return covered


def _replacement(intervention: Intervention, observation: Observation) -> Tensor:
    """Return what an intervention puts at a point, checked against the reading.

    Parameters
    ----------
    intervention : Intervention
        What replaces the value at the point.
    observation : Observation
        What is at the point now.

    Returns
    -------
    Tensor
        The replacement, of the reading's shape.

    Raises
    ------
    ValueError
        If the intervention returned a value of a different shape than the reading.
    """
    replacement = intervention.stands_in_for(observation)

    if tuple(replacement.shape) != tuple(observation.tensor.shape):
        msg = (
            f"{intervention!r} stood in for the reading at {observation.site!r} "
            f"with a value of shape {tuple(replacement.shape)} rather than "
            f"{tuple(observation.tensor.shape)}"
        )
        raise ValueError(msg)

    return replacement


class Within:
    """Applies an intervention at the entries some units index and nowhere else.

    Attributes
    ----------
    tag : ClassVar[str]
        The discriminator this variant answers to.
    units : frozenset[Unit]
        The units whose entries are replaced.
    intervention : Intervention
        What replaces them.

    Parameters
    ----------
    units : Iterable[Unit]
        The units whose entries to replace, each at the point this is applied at. A
        unit with no coordinate covers the whole point. With no units, nothing is
        replaced.
    intervention : Intervention
        What replaces them.

    Raises
    ------
    TypeError
        If a member is not a unit, or the intervention is not one of the declared
        variants.
    """

    __slots__ = ("_intervention", "_units")

    tag: ClassVar[str] = "within"

    def __init__(self, units: Iterable[Unit], intervention: Intervention) -> None:
        self._units = _checked_units(units)
        self._intervention = _checked_variant(intervention)

    @property
    def units(self) -> frozenset[Unit]:
        """Return the units whose entries are replaced."""
        return self._units

    @property
    def intervention(self) -> Intervention:
        """Return what replaces them."""
        return self._intervention

    def stands_in_for(self, observation: Observation, /) -> Tensor:
        """Return the reading with the intervention applied at the indexed entries.

        Parameters
        ----------
        observation : Observation
            What is at the point now.

        Returns
        -------
        Tensor
            The reading's shape, dtype and placement, the replacement at every entry
            some unit indexes and the reading's own number elsewhere.

        Raises
        ------
        KeyError
            If a unit names an axis the reading does not declare.
        ValueError
            If a unit is at another point than the reading, a coordinate lies beyond
            its axis, or the intervention returned a value of another shape.
        """
        if not self._units:
            return observation.tensor

        replacement = _replacement(self._intervention, observation)

        return torch.where(
            _covered(self._units, observation), replacement, observation.tensor
        )

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this intervention."""
        return f"Within({sorted(self._units, key=repr)!r}, {self._intervention!r})"


class Sparing:
    """Applies an intervention everywhere at the point except where some unit indexes.

    Attributes
    ----------
    tag : ClassVar[str]
        The discriminator this variant answers to.
    units : frozenset[Unit]
        The units whose entries keep the reading's own numbers.
    intervention : Intervention
        What replaces every other entry.

    Parameters
    ----------
    units : Iterable[Unit]
        The units whose entries to leave alone, each at the point this is applied
        at. A unit with no coordinate keeps the whole point. With no units, every
        entry is replaced.
    intervention : Intervention
        What replaces every other entry.

    Raises
    ------
    TypeError
        If a member is not a unit, or the intervention is not one of the declared
        variants.
    """

    __slots__ = ("_intervention", "_units")

    tag: ClassVar[str] = "sparing"

    def __init__(self, units: Iterable[Unit], intervention: Intervention) -> None:
        self._units = _checked_units(units)
        self._intervention = _checked_variant(intervention)

    @property
    def units(self) -> frozenset[Unit]:
        """Return the units whose entries keep the reading's own numbers."""
        return self._units

    @property
    def intervention(self) -> Intervention:
        """Return what replaces every other entry."""
        return self._intervention

    def stands_in_for(self, observation: Observation, /) -> Tensor:
        """Return the reading with the intervention applied outside the indexed entries.

        Parameters
        ----------
        observation : Observation
            What is at the point now.

        Returns
        -------
        Tensor
            The reading's shape, dtype and placement, the reading's own number at every
            entry some unit indexes and the replacement elsewhere.

        Raises
        ------
        KeyError
            If a unit names an axis the reading does not declare.
        ValueError
            If a unit is at another point than the reading, a coordinate lies beyond
            its axis, or the intervention returned a value of another shape.
        """
        replacement = _replacement(self._intervention, observation)

        if not self._units:
            return replacement

        return torch.where(
            _covered(self._units, observation), observation.tensor, replacement
        )

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this intervention."""
        return f"Sparing({sorted(self._units, key=repr)!r}, {self._intervention!r})"


type Intervention = (
    Zero
    | MeanFrom
    | Resample
    | PatchFrom
    | Noise
    | Shuffle
    | Subspace
    | Interchange
    | JointInterchange
    | Within
    | Sparing
    | LinearErasure
)
"""Specifies what replaces the value at a published point, as one closed vocabulary."""

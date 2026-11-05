"""Realized submodule sharing between built components.

A producer publishes the submodules a consumer may hold, each with how the
consumer registers it. :func:`module_sharing_matches` compares what two built
modules actually hold in common with what the consumer declared.
"""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Mapping, Sequence

    from lofi._types import ModuleSharing, SubmoduleDeclaration

__all__ = [
    "ModuleSharingProducer",
    "ModuleSharingSubject",
    "SizedTensor",
    "StateTensor",
    "SubmodulePublication",
    "module_sharing_matches",
]


@runtime_checkable
class SizedTensor(Protocol):
    """Exposes how many elements a tensor holds."""

    def numel(self) -> int:
        """Return the number of stored elements."""


@runtime_checkable
class StateTensor(SizedTensor, Protocol):
    """Exposes the storage bytes a tensor's elements occupy."""

    def untyped_storage(self) -> Hashable:
        """Return the underlying storage."""

    def storage_offset(self) -> int:
        """Return the first element's storage offset."""

    def element_size(self) -> int:
        """Return the number of bytes one element occupies."""

    def size(self) -> Sequence[int]:
        """Return the extent of each axis."""

    def stride(self) -> Sequence[int]:
        """Return the storage step of each axis, in elements."""


@runtime_checkable
class ModuleSharingSubject[P, S: StateTensor](Protocol):
    """Exposes registered parameters and saved state for sharing validation."""

    def parameters(self) -> Iterable[P]:
        """Return the registered parameters."""

    def state_dict(self) -> Mapping[str, S]:
        """Return the named saved tensors."""


class SubmodulePublication[P, S: StateTensor](Protocol):
    """A published module and its registered-state semantics."""

    @property
    def module(self) -> ModuleSharingSubject[P, S]:
        """Return the published module."""

    @property
    def sharing(self) -> ModuleSharing:
        """Return how a consumer registers the module."""


@runtime_checkable
class ModuleSharingProducer[P, S: StateTensor](
    ModuleSharingSubject[P, S],
    Protocol,
):
    """A module component that publishes shareable submodules."""

    @property
    def shared(self) -> Mapping[str, SubmodulePublication[P, S]]:
        """Return published submodules keyed by name."""


def _parameter_identities[P, S: StateTensor](
    module: ModuleSharingSubject[P, S],
) -> frozenset[int]:
    """Collect parameter identities from a module."""
    return frozenset(id(parameter) for parameter in module.parameters())


type _Region = tuple[Hashable, int, int]
"""A storage and the half-open byte range a tensor's elements span within it."""


def _region(tensor: StateTensor, /) -> _Region:
    """Return the storage bytes from a tensor's first to its last element."""
    width = tensor.element_size()
    last = sum(
        (extent - 1) * step
        for extent, step in zip(tensor.size(), tensor.stride(), strict=True)
    )
    start = tensor.storage_offset() * width

    return (tensor.untyped_storage(), start, start + (last + 1) * width)


def _state_regions[P, S: StateTensor](
    module: ModuleSharingSubject[P, S],
) -> frozenset[_Region]:
    """Collect the storage regions of a module's nonempty saved tensors.

    Saved extra state that is not a tensor occupies no storage and is skipped.
    """
    return frozenset(
        _region(tensor)
        for tensor in module.state_dict().values()
        if isinstance(tensor, StateTensor) and tensor.numel() > 0
    )


def _overlapping(
    regions: frozenset[_Region], others: frozenset[_Region]
) -> frozenset[_Region]:
    """Return the regions that share at least one byte with any other region."""
    return frozenset(
        region
        for region in regions
        if any(
            region[0] == other[0] and region[1] < other[2] and other[1] < region[2]
            for other in others
        )
    )


def module_sharing_matches[P, S: StateTensor](
    consumer: ModuleSharingSubject[P, S],
    producer: ModuleSharingProducer[P, S],
    declaration: SubmoduleDeclaration | None,
) -> bool:
    """Compare realized parameter and saved-state sharing with a declaration.

    Saved state is shared wherever a consumer tensor and a producer tensor span
    any common byte of one storage, so a slice or other partial alias of a
    producer tensor counts as sharing. Declared state sharing holds only when
    the consumer's overlapping tensors are exactly the published tensors.

    Parameters
    ----------
    consumer : ModuleSharingSubject[P, S]
        Dependent module being checked.
    producer : ModuleSharingProducer[P, S]
        Module supplied through the dependency edge and its publications, which
        are read only where a declaration selects one.
    declaration : SubmoduleDeclaration | None
        Selected publication and expected sharing, or ``None`` when the modules
        must be disjoint.

    Returns
    -------
    bool
        Whether parameter and saved-state sharing both match the declaration.
    """
    consumer_parameters = _parameter_identities(consumer)
    producer_parameters = _parameter_identities(producer)
    consumer_state = _state_regions(consumer)
    producer_state = _state_regions(producer)

    if declaration is None:
        expected_parameters: frozenset[int] = frozenset()
        expected_state: frozenset[_Region] = frozenset()
    else:
        published = producer.shared.get(declaration.name)

        if published is None or published.sharing != declaration.sharing:
            return False

        published_parameters = _parameter_identities(published.module)
        published_state = _state_regions(published.module)

        if not published_parameters <= producer_parameters:
            return False

        if not published_state <= producer_state:
            return False

        expected_parameters = (
            published_parameters if declaration.sharing.in_parameters else frozenset()
        )
        expected_state = (
            published_state if declaration.sharing.in_state_dict else frozenset()
        )

    return (
        consumer_parameters & producer_parameters == expected_parameters
        and _overlapping(consumer_state, producer_state) == expected_state
    )

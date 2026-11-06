from types import MappingProxyType
from typing import TYPE_CHECKING, override

import torch
from torch import Tensor, nn

from lofi._types import ModuleSharing, SharedSubmodule, SubmoduleDeclaration
from lofi.nn.sharing import (
    ModuleSharingSubject,
    SizedTensor,
    StateTensor,
    module_sharing_matches,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


class TiedHolder(nn.Module):
    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder


class PublishedModule(nn.Module):
    def __init__(self, module: nn.Module, sharing: ModuleSharing) -> None:
        super().__init__()
        self.module = module
        self._sharing = sharing

    @property
    def shared(self) -> Mapping[str, SharedSubmodule[nn.Module]]:
        return MappingProxyType({"target": SharedSubmodule(self.module, self._sharing)})


class BufferModule(nn.Module):
    def __init__(self, values: Tensor) -> None:
        super().__init__()
        self.register_buffer("values", values)


def _declared(*, in_parameters: bool, in_state_dict: bool) -> SubmoduleDeclaration:
    return SubmoduleDeclaration("target", ModuleSharing(in_parameters, in_state_dict))


def test_a_built_module_and_its_saved_tensors_satisfy_the_sharing_protocols() -> None:
    encoder = nn.Linear(2, 2)

    assert isinstance(TiedHolder(encoder), ModuleSharingSubject)
    assert all(
        isinstance(tensor, StateTensor) and isinstance(tensor, SizedTensor)
        for tensor in encoder.state_dict().values()
    )


def test_the_standard_validator_compares_both_sharing_surfaces() -> None:
    encoder = nn.Linear(2, 2)
    producer = PublishedModule(
        encoder, ModuleSharing(in_parameters=True, in_state_dict=True)
    )
    holder = TiedHolder(encoder)

    assert module_sharing_matches(
        holder, producer, _declared(in_parameters=True, in_state_dict=True)
    )
    assert not module_sharing_matches(
        holder, producer, _declared(in_parameters=False, in_state_dict=True)
    )
    assert not module_sharing_matches(
        holder, producer, _declared(in_parameters=True, in_state_dict=False)
    )


def test_undeclared_sharing_is_refused_and_disjoint_modules_are_accepted() -> None:
    encoder = nn.Linear(2, 2)
    producer = PublishedModule(
        encoder, ModuleSharing(in_parameters=True, in_state_dict=True)
    )

    assert not module_sharing_matches(TiedHolder(encoder), producer, None)
    assert module_sharing_matches(nn.Linear(2, 2), producer, None)


def test_a_declaration_selecting_an_unpublished_submodule_is_refused() -> None:
    encoder = nn.Linear(2, 2)
    producer = PublishedModule(
        encoder, ModuleSharing(in_parameters=True, in_state_dict=True)
    )
    missing = SubmoduleDeclaration(
        "missing", ModuleSharing(in_parameters=True, in_state_dict=True)
    )

    assert not module_sharing_matches(TiedHolder(encoder), producer, missing)


def test_empty_state_tensors_do_not_create_false_sharing() -> None:
    target = BufferModule(torch.empty(0))
    producer = PublishedModule(
        target, ModuleSharing(in_parameters=False, in_state_dict=True)
    )

    assert module_sharing_matches(BufferModule(torch.empty(0)), producer, None)
    assert module_sharing_matches(
        TiedHolder(target),
        producer,
        _declared(in_parameters=False, in_state_dict=True),
    )


class SlicedHolder(nn.Module):
    def __init__(self, encoder: nn.Linear) -> None:
        super().__init__()
        self.weight = nn.Parameter(encoder.weight.data[1:])


def test_a_partial_alias_of_a_producer_tensor_counts_as_sharing() -> None:
    encoder = nn.Linear(2, 2)
    producer = PublishedModule(
        encoder, ModuleSharing(in_parameters=True, in_state_dict=True)
    )
    holder = SlicedHolder(encoder)

    assert not module_sharing_matches(holder, producer, None)
    assert not module_sharing_matches(
        holder, producer, _declared(in_parameters=False, in_state_dict=True)
    )


def test_a_declared_full_publication_is_accepted_beside_disjoint_state() -> None:
    encoder = nn.Linear(2, 2)
    producer = PublishedModule(
        encoder, ModuleSharing(in_parameters=True, in_state_dict=True)
    )
    holder = TiedHolder(encoder)
    holder.extra = nn.Linear(2, 2)

    assert module_sharing_matches(
        holder, producer, _declared(in_parameters=True, in_state_dict=True)
    )


class StampedModule(nn.Linear):
    """A module that saves a string beside its tensors."""

    @override
    def get_extra_state(self) -> str:
        return "stamp"

    @override
    def set_extra_state(self, state: str) -> None:
        del state


def test_saved_extra_state_that_is_not_a_tensor_shares_no_storage() -> None:
    stamped = StampedModule(2, 2)
    producer = PublishedModule(
        stamped, ModuleSharing(in_parameters=True, in_state_dict=True)
    )
    holder = TiedHolder(stamped)

    assert "encoder._extra_state" in holder.state_dict()
    assert module_sharing_matches(
        holder, producer, _declared(in_parameters=True, in_state_dict=True)
    )
    assert not module_sharing_matches(holder, producer, None)
    assert module_sharing_matches(StampedModule(2, 2), producer, None)

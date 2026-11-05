from typing import override

import pytest
import torch
from torch import Tensor, nn

from lofi.run._modules import (
    FrozenModuleReference,
    ParameterCount,
    parameter_count,
    register_module_collection,
)
from lofi.run._ports import ModuleRegistrar


class Decoder(nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self._readout = nn.Linear(2, 1, bias=False)
        self.register_buffer("_offset", torch.tensor(offset))

    @override
    def forward(self, values: Tensor) -> Tensor:
        return self._readout(values) + self._offset


class Parser(nn.Module):
    def __init__(self, decoders: tuple[Decoder, ...]) -> None:
        super().__init__()
        self.decoders = register_module_collection(self, "decoder", decoders)


class Holder(nn.Module):
    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.reference = FrozenModuleReference(encoder)


class TiedHolder(nn.Module):
    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder


class BufferModule(nn.Module):
    def __init__(self, values: Tensor) -> None:
        super().__init__()
        self.register_buffer("values", values)


def test_a_registered_decoder_collection_is_owned_and_moves_with_its_parser() -> None:
    decoders = (Decoder(1.0), Decoder(2.0), Decoder(3.0))
    parser = Parser(decoders)
    decoder_parameters = {
        id(parameter) for decoder in decoders for parameter in decoder.parameters()
    }

    assert parser.decoders == decoders
    assert {id(parameter) for parameter in parser.parameters()} == decoder_parameters
    assert set(parser.state_dict()) == {
        f"decoder_{index}.{name}"
        for index, decoder in enumerate(decoders)
        for name in decoder.state_dict()
    }

    parser.to("meta")

    assert all(
        tensor.device.type == "meta"
        for decoder in parser.decoders
        for tensor in (*decoder.parameters(), *decoder.buffers())
    )


def test_a_module_collection_requires_a_name() -> None:
    parser = Parser(())

    with pytest.raises(ValueError, match="module collection is empty"):
        register_module_collection(parser, "", (Decoder(1.0),))


def test_a_frozen_reference_does_not_register_its_module() -> None:
    encoder = nn.Linear(2, 2)
    holder = Holder(encoder)

    assert isinstance(holder, ModuleRegistrar)
    assert not tuple(holder.parameters())
    assert not holder.state_dict()
    assert all(not parameter.requires_grad for parameter in encoder.parameters())


def test_a_count_is_the_elements_of_every_parameter() -> None:
    decoder = Decoder(1.0)
    linear = nn.Linear(3, 4)

    assert parameter_count(()) == ParameterCount(0, 0)
    assert parameter_count(decoder.parameters()) == ParameterCount(2, 2)
    assert parameter_count(linear.parameters()) == ParameterCount(16, 16)
    assert parameter_count((*decoder.parameters(), *linear.parameters())) == (
        ParameterCount(18, 18)
    )


def test_a_count_holds_no_buffer_and_trains_no_frozen_parameter() -> None:
    buffered = BufferModule(torch.ones(5))
    encoder = nn.Linear(2, 2)
    frozen = nn.Linear(3, 1).requires_grad_(requires_grad=False)

    assert parameter_count(buffered.parameters()) == ParameterCount(0, 0)
    assert parameter_count(Holder(nn.Linear(2, 2)).parameters()) == (
        ParameterCount(0, 0)
    )
    assert parameter_count((*encoder.parameters(), *frozen.parameters())) == (
        ParameterCount(10, 6)
    )


def test_a_parameter_listed_twice_is_counted_once() -> None:
    encoder = nn.Linear(2, 2)
    tied = TiedHolder(encoder)
    independent = nn.Linear(2, 2)
    alone = parameter_count(encoder.parameters())

    assert parameter_count((*encoder.parameters(), *tied.parameters())) == alone
    assert parameter_count(
        (*encoder.parameters(), *independent.parameters())
    ).total == (2 * alone.total)

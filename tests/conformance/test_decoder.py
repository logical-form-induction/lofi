"""The independent decoder protocols agree on payload-aware choice rows."""

from typing import Final

import pytest
import torch
from tests.conformance.decoder import (
    DecoderAgreementSuite,
    DecoderConformance,
    FrontierDecoderSuite,
    MarginalDecoderSuite,
    TeacherForcedDecoderSuite,
)

from lofi.nn._values import Encoded, LogDistribution
from lofi.terms import (
    BatchEncoding,
    ChoiceIndex,
    HoleContext,
    Operator,
    OperatorIndex,
    PreorderArityLayout,
    Signature,
    Term,
    TermCodec,
    TermStore,
)

SLOTS: Final = 3
SIGNATURE: Final = Signature(
    "decoder-conformance",
    ("T",),
    (
        Operator("one", "T"),
        Operator("wrap", "T", arg_sorts=("T",)),
    ),
)
INDEX: Final = OperatorIndex(SIGNATURE)
CHOICES: Final = ChoiceIndex(SIGNATURE, depth_bounds={"T": SLOTS})
CODEC: Final = TermCodec(INDEX, PreorderArityLayout(SLOTS))
STORE: Final = TermStore(SIGNATURE)
ONE: Final = STORE.op(SIGNATURE.operator(SIGNATURE.code("one")))
WRAP: Final = SIGNATURE.operator(SIGNATURE.code("wrap"))
TERMS: Final = CODEC.encode_batch((STORE.op(WRAP, ONE), ONE))
INPUTS: Final = Encoded(
    torch.tensor(((0.5,), (-0.25,)), dtype=torch.float64),
    ("term", "width"),
)
ROOT: Final = HoleContext((), "T", {}, {}, SLOTS)


class ChoiceRows:
    """A deterministic fixture implementing all three decoder protocols."""

    def _rows(self, inputs: Encoded) -> torch.Tensor:
        slots = torch.arange(1, SLOTS + 1, dtype=inputs.tensor.dtype)
        choices = torch.arange(CHOICES.width, dtype=inputs.tensor.dtype)
        logits = (
            inputs.tensor[:, :1, None]
            * slots.reshape(1, SLOTS, 1)
            * choices.reshape(1, 1, CHOICES.width)
        )
        return logits.log_softmax(dim=-1)

    def rows(self, inputs: Encoded, /, *, terms: BatchEncoding) -> LogDistribution:
        del terms
        return LogDistribution(
            self._rows(inputs),
            ("term", "slot", "choice"),
            normalized_over="choice",
        )

    def marginal_rows(self, inputs: Encoded, /, *, root_sort: str) -> LogDistribution:
        if root_sort != "T":
            msg = f"this fixture represents T, not {root_sort!r}"
            raise ValueError(msg)
        return LogDistribution(
            self._rows(inputs),
            ("term", "slot", "choice"),
            normalized_over="choice",
        )

    def start(self, inputs: Encoded, /) -> torch.Tensor:
        return inputs.tensor[0]

    def weigh(self, state: torch.Tensor, /, *, context: HoleContext) -> LogDistribution:
        del context
        choices = torch.arange(CHOICES.width, dtype=state.dtype)
        return LogDistribution(
            (state[0] * choices).log_softmax(dim=0),
            ("choice",),
            normalized_over="choice",
        )

    def fill(self, state: torch.Tensor, /, *, choices: torch.Tensor) -> torch.Tensor:
        return state + choices

    def admits(self, term: Term, /) -> bool:
        del term
        return True


class TestAChoiceDecoder(
    TeacherForcedDecoderSuite,
    MarginalDecoderSuite,
    FrontierDecoderSuite,
    DecoderAgreementSuite,
):
    @pytest.fixture
    def conformance(self) -> DecoderConformance[ChoiceRows]:
        return DecoderConformance(
            build=ChoiceRows,
            inputs=INPUTS,
            terms=TERMS,
            index=CHOICES,
            root=ROOT,
            root_sort="T",
            slots=SLOTS,
        )

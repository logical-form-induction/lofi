from typing import TYPE_CHECKING, Final, NamedTuple, NoReturn

import pytest
import torch

from lofi._checks import LawViolationError
from lofi.nn._values import Encoded
from lofi.nn.arity import ArityUnsupportedError
from lofi.nn.ports import ArityScheme

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "ARITY_SCHEME_LAWS",
    "AritySchemeConformance",
    "AritySchemeSuite",
]

_CELL_AXES: Final = ("cell", "width")
_WIDTH_AXES: Final = ("width",)


def _fail(detail: str, *, law: str) -> NoReturn:
    message = f"{law}: {detail}"

    raise LawViolationError(message)


class AritySchemeConformance(NamedTuple):
    build: Callable[[], ArityScheme]


class AritySchemeSuite:
    pytestmark = pytest.mark.conformance

    def _state(self, scheme: ArityScheme, seed: int) -> Encoded:
        """Draw one unmasked state of the scheme's own width."""
        generator = torch.Generator().manual_seed(seed)

        return Encoded(torch.randn(scheme.width, generator=generator), _WIDTH_AXES)

    def _cells(self, scheme: ArityScheme, mask: tuple[bool, ...]) -> Encoded:
        """Build a state over a cell axis, valid where the mask is true."""
        return Encoded(
            torch.ones(len(mask), scheme.width),
            _CELL_AXES,
            torch.tensor(mask),
        )

    def test_the_scheme_answers_the_published_call_surface(
        self, conformance: AritySchemeConformance
    ) -> None:
        scheme = conformance.build()

        if not isinstance(scheme, ArityScheme):
            _fail(
                f"{type(scheme).__name__} does not provide factoring and combination",
                law="the_scheme_answers_the_published_call_surface",
            )

        parent = self._state(scheme, 101)

        for rank in scheme.ranks:
            try:
                children = scheme.factor(parent, arity=rank)
                scheme.combine(children)

            except TypeError as widened:
                _fail(
                    f"factoring at rank {rank} refused the declared call: {widened}",
                    law="the_scheme_answers_the_published_call_surface",
                )

    def test_factoring_emits_one_state_per_argument_position(
        self, conformance: AritySchemeConformance
    ) -> None:
        scheme = conformance.build()
        parent = self._state(scheme, 102)

        if scheme.factor(parent, arity=0) != ():
            _fail(
                "a nullary operator was factored into children",
                law="factoring_emits_one_state_per_argument_position",
            )

        for rank in scheme.ranks:
            children = scheme.factor(parent, arity=rank)

            if len(children) != rank:
                _fail(
                    f"rank {rank} was factored into {len(children)} children",
                    law="factoring_emits_one_state_per_argument_position",
                )

            for position, child in enumerate(children):
                if child.axes != parent.axes:
                    _fail(
                        f"child {position} of rank {rank} runs over {child.axes}, "
                        f"not over {parent.axes}",
                        law="factoring_emits_one_state_per_argument_position",
                    )

                if child.tensor.shape != parent.tensor.shape:
                    _fail(
                        f"child {position} of rank {rank} is shaped "
                        f"{tuple(child.tensor.shape)}, not "
                        f"{tuple(parent.tensor.shape)}",
                        law="factoring_emits_one_state_per_argument_position",
                    )

    def test_combination_returns_one_state_on_the_children_axes(
        self, conformance: AritySchemeConformance
    ) -> None:
        scheme = conformance.build()
        parent = self._state(scheme, 103)

        for rank in scheme.ranks:
            combined = scheme.combine(scheme.factor(parent, arity=rank))

            if combined.axes != parent.axes:
                _fail(
                    f"a rank {rank} combination runs over {combined.axes}, "
                    f"not over {parent.axes}",
                    law="combination_returns_one_state_on_the_children_axes",
                )

            if combined.tensor.shape != parent.tensor.shape:
                _fail(
                    f"a rank {rank} combination is shaped "
                    f"{tuple(combined.tensor.shape)}, not "
                    f"{tuple(parent.tensor.shape)}",
                    law="combination_returns_one_state_on_the_children_axes",
                )

    def test_a_combined_state_is_valid_only_where_every_child_is(
        self, conformance: AritySchemeConformance
    ) -> None:
        scheme = conformance.build()

        for rank in scheme.ranks:
            masks = tuple(
                tuple(position != at for position in range(rank + 1))
                for at in range(rank)
            )
            combined = scheme.combine(
                tuple(self._cells(scheme, mask) for mask in masks)
            )
            expected = [all(column) for column in zip(*masks, strict=True)]

            if combined.mask is None:
                _fail(
                    f"a rank {rank} combination of masked children has no mask",
                    law="a_combined_state_is_valid_only_where_every_child_is",
                )

            if combined.mask.tolist() != expected:
                _fail(
                    f"a rank {rank} combination is valid at "
                    f"{combined.mask.tolist()}, not at {expected}",
                    law="a_combined_state_is_valid_only_where_every_child_is",
                )

    def test_an_arity_the_scheme_does_not_serve_is_refused(
        self, conformance: AritySchemeConformance
    ) -> None:
        scheme = conformance.build()
        unserved = max(scheme.ranks, default=0) + 1
        parent = self._state(scheme, 104)

        with pytest.raises(ArityUnsupportedError):
            scheme.factor(parent, arity=unserved)

        with pytest.raises(ArityUnsupportedError):
            scheme.combine((parent,) * unserved)

    def test_a_child_of_another_width_is_refused(
        self, conformance: AritySchemeConformance
    ) -> None:
        scheme = conformance.build()
        narrow = Encoded(torch.ones(scheme.width + 1), _WIDTH_AXES)

        for rank in scheme.ranks:
            with pytest.raises(ValueError, match="wide"):
                scheme.combine((narrow,) * rank)

            with pytest.raises(ValueError, match="innermost"):
                scheme.combine((Encoded(torch.ones(scheme.width), ("symbol",)),) * rank)

        with pytest.raises(ValueError, match="wide"):
            scheme.factor(narrow, arity=0)


ARITY_SCHEME_LAWS: Final = (
    AritySchemeSuite.test_the_scheme_answers_the_published_call_surface,
    AritySchemeSuite.test_factoring_emits_one_state_per_argument_position,
    AritySchemeSuite.test_combination_returns_one_state_on_the_children_axes,
    AritySchemeSuite.test_a_combined_state_is_valid_only_where_every_child_is,
    AritySchemeSuite.test_an_arity_the_scheme_does_not_serve_is_refused,
    AritySchemeSuite.test_a_child_of_another_width_is_refused,
)

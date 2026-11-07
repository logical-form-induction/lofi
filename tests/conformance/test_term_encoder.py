from typing import TYPE_CHECKING, Final, override

import pytest
import torch
from tests.conformance._sites import SiteObservation
from tests.conformance.term_encoder import (
    SOFT_TERM_ENCODER_LAWS,
    TERM_ENCODER_LAWS,
    SoftTermEncoderSuite,
    TermEncoderConformance,
    TermEncoderSuite,
)

from lofi._checks import LawViolationError
from lofi.nn._values import Encoded, LogDistribution
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
from lofi.terms import BatchEncoding, Encoding

if TYPE_CHECKING:
    from collections.abc import Mapping

PAD: Final = 0
ALPHABET: Final = 4

ROOTED: Final = (Encoding.NO_PARENT, 0, 0)
LEVELS: Final = (0, 1, 1)


def row(symbols: tuple[int, ...]) -> Encoding:
    """Return one term of three slots, occupied wherever its code is not padding.

    Parameters
    ----------
    symbols
        The operator code at each slot.

    Returns
    -------
    Encoding
        The linearization, rooted at its first slot.
    """
    return Encoding(symbols, ROOTED, LEVELS, tuple(code != PAD for code in symbols))


TERMS: Final = BatchEncoding((row((3, 1, 2)), row((2, 1, PAD))))

COMPONENT: Final = ComponentId("code-sum")
ENCODING_SITE: Final = Site(COMPONENT, Role.TERM_ENCODING)
NODE_SITE: Final = Site(COMPONENT, Role.NODE_ENCODING)

SITES: Final = SiteMap(
    COMPONENT,
    ComponentKind.TERM_ENCODER,
    {
        ENCODING_SITE: SiteSpec(
            ("term", "width"),
            DTypeClass.REAL,
            differentiable=True,
            access=Access.OBSERVE,
        ),
        NODE_SITE: SiteSpec(
            ("term", "slot", "width"),
            DTypeClass.REAL,
            differentiable=True,
            access=Access.OBSERVE,
        ),
    },
)


class CodeSum(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))

    @property
    def output_width(self) -> int:
        return 1

    def _codes(self, terms: BatchEncoding) -> torch.Tensor:
        return torch.tensor(terms.symbols, dtype=torch.int64)

    def encode(self, terms: BatchEncoding, /) -> Encoded:
        totals = self._codes(terms).sum(dim=1, keepdim=True).double()

        return Encoded(totals * self.scale, ("term", "width"))

    def encode_nodes(self, terms: BatchEncoding, /) -> Encoded:
        codes = self._codes(terms).unsqueeze(-1).double()

        return Encoded(
            codes * self.scale, ("term", "slot", "width"), self._node_mask(terms)
        )

    def _node_mask(self, terms: BatchEncoding) -> torch.Tensor | None:
        return torch.tensor(terms.mask)

    def encode_soft(
        self,
        choices: LogDistribution,
        /,
        *,
        structure: BatchEncoding,
    ) -> Encoded:
        del structure
        moments = self._moment(choices)

        return Encoded(
            moments.sum(dim=choices.axis("slot"), keepdim=True) * self.scale,
            ("term", "width"),
        )

    def _moment(self, choices: LogDistribution) -> torch.Tensor:
        alphabet = torch.arange(choices.size("choice"), dtype=torch.float64)

        return (choices.tensor.exp() * alphabet).sum(dim=choices.axis("choice"))

    def site_map(self) -> SiteMap:
        return SITES

    def instrument(self, terms: BatchEncoding) -> Mapping[Site, Encoded]:
        return {
            ENCODING_SITE: self.encode(terms),
            NODE_SITE: self.encode_nodes(terms),
        }

    def placed(self, placement: str) -> CodeSum:
        return self.to(torch.device(placement))


class Renamed(CodeSum):
    @override
    def encode_nodes(self, terms: BatchEncoding, /) -> Encoded:
        encoded = super().encode_nodes(terms)

        return Encoded(encoded.tensor, ("term", "node", "width"), encoded.mask)

    @override
    def site_map(self) -> SiteMap:
        changed = dict(super().site_map())
        changed[NODE_SITE] = SiteSpec(
            ("term", "node", "width"),
            DTypeClass.REAL,
            differentiable=True,
            access=Access.OBSERVE,
        )

        return SiteMap(COMPONENT, ComponentKind.TERM_ENCODER, changed)


class Narrow(CodeSum):
    @property
    @override
    def output_width(self) -> int:
        return 2


class Permissive(CodeSum):
    @override
    def encode(self, terms: BatchEncoding, /, **extra: int) -> Encoded:
        del extra

        return super().encode(terms)


class Growing(CodeSum):
    @override
    def encode(self, terms: BatchEncoding, /) -> Encoded:
        self.add_module(f"spare{len(self._modules)}", torch.nn.Identity())

        return super().encode(terms)


class Homesick(CodeSum):
    @override
    def placed(self, placement: str) -> Homesick:
        moved = self.to(torch.device(placement))

        with torch.no_grad():
            moved.scale.mul_(2.0)

        return moved


class Unmasked(CodeSum):
    @override
    def _node_mask(self, terms: BatchEncoding) -> torch.Tensor | None:
        del terms

        return None


class Silent(CodeSum):
    @override
    def instrument(self, terms: BatchEncoding) -> Mapping[Site, Encoded]:
        del terms

        return {}


class Skewed(CodeSum):
    @override
    def _moment(self, choices: LogDistribution) -> torch.Tensor:
        alphabet = torch.arange(choices.size("choice"), dtype=torch.float64)

        return (choices.tensor.exp() * alphabet.square()).sum(
            dim=choices.axis("choice")
        )


def _against(build: type[CodeSum]) -> TermEncoderConformance[CodeSum]:
    """Build the conformance fixture for an encoder class.

    Parameters
    ----------
    build
        The encoder to build.

    Returns
    -------
    TermEncoderConformance
        The fixture for ``TERMS``.
    """
    return TermEncoderConformance(
        build=build,
        terms=TERMS,
        choices=lambda terms: terms.symbols,
        choice_width=ALPHABET,
        sites=SiteObservation(
            build=build,
            observe=lambda subject: subject.instrument(TERMS),
            read=lambda subject: (subject.encode(TERMS).tensor,),
        ),
        submodules=lambda subject: tuple(
            name for name, _ in subject.named_modules() if name
        ),
        place=lambda subject, placement: subject.placed(placement),
    )


def _red(conformance: TermEncoderConformance[CodeSum]) -> frozenset[str]:
    """Collect the laws that fail for a conformance fixture.

    Parameters
    ----------
    conformance
        The fixture to check.

    Returns
    -------
    frozenset of str
        Names of the failing laws.
    """
    suite = TermEncoderSuite()
    red: list[str] = []

    for law in TERM_ENCODER_LAWS:
        try:
            law(suite, conformance)
        except LawViolationError:
            red.append(law.__name__)

    return frozenset(red)


def _red_soft(conformance: TermEncoderConformance[CodeSum]) -> frozenset[str]:
    """Collect the soft-encoder laws that fail for a conformance fixture."""
    suite = SoftTermEncoderSuite()
    red: list[str] = []

    for law in SOFT_TERM_ENCODER_LAWS:
        try:
            law(suite, conformance)
        except LawViolationError:
            red.append(law.__name__)

    return frozenset(red)


_BROKEN: Final[tuple[tuple[str, type[CodeSum]], ...]] = (
    ("test_rank_and_axes_are_as_declared", Renamed),
    ("test_the_declared_width_is_the_trailing_axis", Narrow),
    ("test_the_signature_accepts_no_extra_keyword", Permissive),
    ("test_encoding_is_pure", Growing),
    ("test_placement_does_not_change_the_encoding", Homesick),
    ("test_a_node_no_term_occupies_is_masked", Unmasked),
    ("test_the_site_map_matches_what_instrumentation_yields", Silent),
)

_BROKEN_SOFT: Final[tuple[tuple[str, type[CodeSum]], ...]] = (
    ("test_a_one_hot_distribution_encodes_as_the_term", Skewed),
)


class TestATermEncoderThatSumsItsCodes(TermEncoderSuite, SoftTermEncoderSuite):
    @pytest.fixture
    def conformance(self) -> TermEncoderConformance[CodeSum]:
        return _against(CodeSum)


def test_every_law_holds_of_an_encoder_that_sums_its_codes() -> None:
    assert _red(_against(CodeSum)) == frozenset()
    assert _red_soft(_against(CodeSum)) == frozenset()


@pytest.mark.parametrize(
    ("law", "build"),
    _BROKEN,
    ids=[law.removeprefix("test_") for law, _ in _BROKEN],
)
def test_exactly_the_broken_law_goes_red(law: str, build: type[CodeSum]) -> None:
    assert _red(_against(build)) == frozenset({law})


@pytest.mark.parametrize(
    ("law", "build"),
    _BROKEN_SOFT,
    ids=[law.removeprefix("test_") for law, _ in _BROKEN_SOFT],
)
def test_exactly_the_broken_soft_law_goes_red(law: str, build: type[CodeSum]) -> None:
    assert _red_soft(_against(build)) == frozenset({law})


def test_the_laws_are_stated_once_each() -> None:
    assert len(TERM_ENCODER_LAWS) == len(_BROKEN)
    assert len({law.__name__ for law in TERM_ENCODER_LAWS}) == len(TERM_ENCODER_LAWS)
    assert {law.__name__ for law in TERM_ENCODER_LAWS} == {law for law, _ in _BROKEN}
    assert len(SOFT_TERM_ENCODER_LAWS) == len(_BROKEN_SOFT)
    assert {law.__name__ for law in SOFT_TERM_ENCODER_LAWS} == {
        law for law, _ in _BROKEN_SOFT
    }

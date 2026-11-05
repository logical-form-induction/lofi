"""Registered strategies a term readout decodes complete terms with."""

from typing import TYPE_CHECKING, Final, Protocol, final, runtime_checkable

from lofi._frozen import FrozenValue
from lofi.nn._values import Encoded
from lofi.nn.ports import TeacherForcedDecoder
from lofi.nn.sampling import sampling_strategy
from lofi.nn.strategies import beam, exhaustive, greedy, sample, score
from lofi.registry import Registry, RegistryError

if TYPE_CHECKING:
    from lofi.nn.codec import NeuralCodecView
    from lofi.nn.forms import WeightedTerms
    from lofi.nn.loss._approximation import ApproximationBound
    from lofi.nn.ports import FrontierDecoder
    from lofi.spec.decoding import DecodingSettings
    from lofi.terms import ChoiceIndex, Term, TermStore

__all__ = [
    "BEAM",
    "EXHAUSTIVE",
    "GREEDY",
    "SAMPLE",
    "DecodingStrategy",
    "decoding_strategy",
]


@runtime_checkable
class DecodingStrategy(Protocol):
    """Decode complete terms from a frontier decoder and state what it needs."""

    @property
    def rescores(self) -> bool:
        """Return whether decoded weights are checked against teacher forcing."""
        ...

    @property
    def enumerates(self) -> bool:
        """Return whether decoding enumerates terms under a residual-mass bound."""
        ...

    def frontier(self, settings: DecodingSettings, /) -> int:
        """Return how many terms one decoded frontier holds at most."""
        ...

    def decode[S](
        self,
        decoder: FrontierDecoder[S],
        inputs: Encoded,
        settings: DecodingSettings,
        /,
        *,
        codec: NeuralCodecView,
        index: ChoiceIndex,
        store: TermStore,
        root_sort: str,
        max_nodes: int,
        clamp: Term | None,
        approximation: ApproximationBound | None,
    ) -> WeightedTerms:
        """Return the weighted terms decoded from one input vector."""
        ...


@final
class _Greedy(FrozenValue):
    """Take the highest-scoring choice at every frontier step."""

    _noun = "greedy decoding strategy"
    __slots__ = ()

    @property
    def rescores(self) -> bool:
        return False

    @property
    def enumerates(self) -> bool:
        return False

    def frontier(self, settings: DecodingSettings, /) -> int:
        del settings

        return 1

    def decode[S](
        self,
        decoder: FrontierDecoder[S],
        inputs: Encoded,
        settings: DecodingSettings,
        /,
        *,
        codec: NeuralCodecView,
        index: ChoiceIndex,
        store: TermStore,
        root_sort: str,
        max_nodes: int,
        clamp: Term | None,
        approximation: ApproximationBound | None,
    ) -> WeightedTerms:
        del settings, codec, approximation

        return greedy(
            decoder,
            inputs,
            index=index,
            store=store,
            root_sort=root_sort,
            max_nodes=max_nodes,
            clamp=clamp,
        )


@final
class _Beam(FrozenValue):
    """Keep the declared number of highest-scoring partial terms."""

    _noun = "beam decoding strategy"
    __slots__ = ()

    @property
    def rescores(self) -> bool:
        return False

    @property
    def enumerates(self) -> bool:
        return False

    def frontier(self, settings: DecodingSettings, /) -> int:
        if settings.width is None:
            msg = "a beam declaration resolved without a width"
            raise ValueError(msg)

        return settings.width

    def decode[S](
        self,
        decoder: FrontierDecoder[S],
        inputs: Encoded,
        settings: DecodingSettings,
        /,
        *,
        codec: NeuralCodecView,
        index: ChoiceIndex,
        store: TermStore,
        root_sort: str,
        max_nodes: int,
        clamp: Term | None,
        approximation: ApproximationBound | None,
    ) -> WeightedTerms:
        del codec, approximation

        return beam(
            decoder,
            inputs,
            index=index,
            store=store,
            root_sort=root_sort,
            max_nodes=max_nodes,
            width=self.frontier(settings),
            clamp=clamp,
        )


@final
class _Exhaustive(FrozenValue):
    """Enumerate terms until the residual mass falls under the declared bound."""

    _noun = "exhaustive decoding strategy"
    __slots__ = ()

    @property
    def rescores(self) -> bool:
        return False

    @property
    def enumerates(self) -> bool:
        return True

    def frontier(self, settings: DecodingSettings, /) -> int:
        if settings.cap is None:
            msg = "an exhaustive declaration resolved without a cap"
            raise ValueError(msg)

        return settings.cap

    def decode[S](
        self,
        decoder: FrontierDecoder[S],
        inputs: Encoded,
        settings: DecodingSettings,
        /,
        *,
        codec: NeuralCodecView,
        index: ChoiceIndex,
        store: TermStore,
        root_sort: str,
        max_nodes: int,
        clamp: Term | None,
        approximation: ApproximationBound | None,
    ) -> WeightedTerms:
        del codec

        if approximation is None or settings.exact_residual is None:
            msg = (
                "an exhaustive declaration resolved without its residual bound "
                "or exactness residual"
            )
            raise ValueError(msg)

        weighted = exhaustive(
            decoder,
            inputs,
            index=index,
            store=store,
            root_sort=root_sort,
            max_nodes=max_nodes,
            residual_mass=approximation.residual_mass,
            exact_residual=settings.exact_residual,
            cap=self.frontier(settings),
            clamp=clamp,
        )
        return approximation.check_terms(weighted, component="term decoding")


@final
class _Sample(FrozenValue):
    """Draw terms under a restriction, then rescore them under teacher forcing."""

    _noun = "sampling decoding strategy"
    __slots__ = ()

    @property
    def rescores(self) -> bool:
        return True

    @property
    def enumerates(self) -> bool:
        return False

    def frontier(self, settings: DecodingSettings, /) -> int:
        if settings.draws is None:
            msg = "a sampling declaration resolved without a draw count"
            raise ValueError(msg)

        return settings.draws

    def decode[S](
        self,
        decoder: FrontierDecoder[S],
        inputs: Encoded,
        settings: DecodingSettings,
        /,
        *,
        codec: NeuralCodecView,
        index: ChoiceIndex,
        store: TermStore,
        root_sort: str,
        max_nodes: int,
        clamp: Term | None,
        approximation: ApproximationBound | None,
    ) -> WeightedTerms:
        del approximation
        restriction = settings.sampling
        tolerance = settings.rescore_tolerance

        if restriction is None or tolerance is None:
            msg = (
                "a sampling declaration resolved without a restriction or "
                "rescore tolerance"
            )
            raise ValueError(msg)

        if not isinstance(decoder, TeacherForcedDecoder):
            msg = (
                f"{type(decoder).__name__} samples terms but provides no "
                "teacher-forced rows for rescoring"
            )
            raise TypeError(msg)

        drawn = sample(
            decoder,
            inputs,
            index=index,
            store=store,
            root_sort=root_sort,
            max_nodes=max_nodes,
            restriction=sampling_strategy(restriction),
            draws=self.frontier(settings),
            clamp=clamp,
        )

        if not drawn.terms:
            return drawn

        count = len(drawn.terms)
        repeated = Encoded(
            inputs.tensor.expand(count, *inputs.tensor.shape[1:]),
            inputs.axes,
        )
        rescored = score(
            decoder,
            repeated,
            codec.codec.encode_batch(drawn.terms),
            index=index,
        )

        if rescored.tensor.shape != drawn.log_weights.shape:
            msg = (
                "sample-then-rescore returned weights of shape "
                f"{tuple(rescored.tensor.shape)} for draws of shape "
                f"{tuple(drawn.log_weights.shape)}"
            )
            raise ValueError(msg)

        disagreement = (rescored.tensor.detach() - drawn.log_weights.detach()).abs()

        if disagreement.numel() and bool(disagreement.max() > tolerance):
            msg = (
                "sampled term weights disagree with the same model's rescoring "
                f"by {float(disagreement.max())}, beyond declared tolerance "
                f"{tolerance}"
            )
            raise ValueError(msg)

        return drawn


GREEDY: Final[DecodingStrategy] = _Greedy()
"""Stepwise argmax decoding."""

BEAM: Final[DecodingStrategy] = _Beam()
"""Bounded beam decoding."""

EXHAUSTIVE: Final[DecodingStrategy] = _Exhaustive()
"""Enumeration under a residual-mass bound and a term cap."""

SAMPLE: Final[DecodingStrategy] = _Sample()
"""Repeated restricted sampling checked against teacher-forced rescoring."""


def decoding_strategy(name: str, /) -> DecodingStrategy:
    """Return the decoding strategy installed under ``name``.

    Raises
    ------
    RegistryError
        If no installed ``lofi.decoding_strategy`` entry is named ``name``, or
        the entry names something other than a decoding strategy.
    """
    member = Registry[object]("decoding_strategy").get(name)

    if not isinstance(member, DecodingStrategy):
        msg = (
            f"the decoding_strategy entry {name!r} names {member!r}, "
            "not a decoding strategy"
        )
        raise RegistryError(msg, family="decoding_strategy")

    return member

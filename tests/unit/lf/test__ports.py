from typing import TYPE_CHECKING, Final

from tests.language import language

from lofi.lf import EntailmentOracle, Proof, Proved, ProverBound, TruthTableOracle

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.lf import Verdict
    from lofi.terms import Term


LANGUAGE: Final = language(("rain",))
BOUND: Final = ProverBound(
    "truth-table",
    LANGUAGE.profile.fragment,
)


def test_a_shipped_oracle_satisfies_the_port() -> None:
    oracle = TruthTableOracle(
        LANGUAGE.store,
        LANGUAGE.profile,
        bound=BOUND,
        atoms=12,
    )

    assert isinstance(oracle, EntailmentOracle)


class _NoBound:
    """A decision procedure missing the profile bound required by the port."""

    __slots__ = ()

    def entails(
        self,
        premise: Term,
        hypothesis: Term,
        /,
        *,
        theory: Sequence[Term] = (),
        timeout: float,
    ) -> Verdict:
        del premise, hypothesis, theory, timeout

        return Proved(Proof("fixture", "always"))


def test_an_oracle_missing_the_bound_does_not_satisfy_the_port() -> None:
    assert not isinstance(_NoBound(), EntailmentOracle)

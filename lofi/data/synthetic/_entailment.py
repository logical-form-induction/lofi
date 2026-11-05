"""The logical form of an item and the certificate of one entailment decision."""

from typing import TYPE_CHECKING, Final, Literal, override

from lofi._checks import ensure_finite, ensure_name
from lofi.lf import Graded, Proved, Refuted

if TYPE_CHECKING:
    from lofi._types import ItemId
    from lofi.lf import Verdict

__all__ = [
    "ENTAILMENT_LABELS",
    "EntailmentCertificate",
    "EntailmentLabel",
    "GoldForm",
]

type EntailmentLabel = Literal["entailment", "non-entailment", "undecided"]
"""What the proof system established about a premise and a hypothesis."""

ENTAILMENT_LABELS: Final[tuple[EntailmentLabel, ...]] = (
    "entailment",
    "non-entailment",
    "undecided",
)
"""The entailment labels in report order."""


def _label(verdict: Verdict) -> EntailmentLabel:
    """Return the label a verdict establishes.

    Parameters
    ----------
    verdict : Verdict
        The oracle's answer.
    """
    if isinstance(verdict, Proved):
        return "entailment"

    if isinstance(verdict, Refuted):
        return "non-entailment"

    if isinstance(verdict, Graded):
        if verdict.degree == 1.0:
            return "entailment"

        if verdict.degree == 0.0:
            return "non-entailment"

    return "undecided"


def _timeout(value: float) -> float:
    """Return a positive number of seconds."""
    held = ensure_finite("a timeout", value)

    if held <= 0.0:
        msg = f"a timeout is a positive number of seconds: {value!r}"
        raise ValueError(msg)

    return held


class GoldForm:
    """The logical form the declared interpretation assigns to one item.

    Parameters
    ----------
    item_id : ItemId
        The item the form belongs to.
    text : str
        The form written in the logical-form surface syntax, which reads back
        into the interpretation's store as the same interned term.

    Raises
    ------
    ValueError
        If the text is empty.
    """

    __slots__ = ("_item_id", "_text")

    def __init__(self, item_id: ItemId, text: str) -> None:
        self._item_id = item_id
        self._text = ensure_name(f"the logical form of {item_id!r}", text)

    @property
    def item_id(self) -> ItemId:
        """Return the item the form belongs to."""
        return self._item_id

    @property
    def text(self) -> str:
        """Return the form in the logical-form surface syntax."""
        return self._text

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value assigns the same form to the same item."""
        if type(other) is not GoldForm:
            return NotImplemented

        return (self._item_id, self._text) == (other._item_id, other._text)

    @override
    def __hash__(self) -> int:
        """Return a hash over the item and the form."""
        return hash((GoldForm, self._item_id, self._text))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this form."""
        return f"GoldForm({self._item_id!r}, {self._text!r})"


class EntailmentCertificate:
    """The verdict one decision reached and the proof system that reached it.

    Parameters
    ----------
    verdict : Verdict
        A proof with its justification, a refutation with its countermodel,
        or an abstention with the reason the decision was exhausted.
    prover : str
        The name of the oracle that decided.
    profile_digest : str
        The digest of the logic profile the decision assumed.
    timeout : float
        The wall clock in seconds the decision was allowed.

    Raises
    ------
    TypeError
        If the timeout is not numeric.
    ValueError
        If the prover or the digest is empty, or the timeout is not positive.
    """

    __slots__ = ("_profile_digest", "_prover", "_timeout", "_verdict")

    def __init__(
        self, verdict: Verdict, *, prover: str, profile_digest: str, timeout: float
    ) -> None:
        self._verdict = verdict
        self._prover = ensure_name("a certificate's prover", prover)
        self._profile_digest = ensure_name(
            "a certificate's profile digest", profile_digest
        )
        self._timeout = _timeout(timeout)

    @property
    def verdict(self) -> Verdict:
        """Return the answer the decision reached."""
        return self._verdict

    @property
    def label(self) -> EntailmentLabel:
        """Return the label the verdict establishes."""
        return _label(self._verdict)

    @property
    def prover(self) -> str:
        """Return the name of the oracle that decided."""
        return self._prover

    @property
    def profile_digest(self) -> str:
        """Return the digest of the logic profile the decision assumed."""
        return self._profile_digest

    @property
    def timeout(self) -> float:
        """Return the wall clock in seconds the decision was allowed."""
        return self._timeout

    def _fields(self) -> tuple[Verdict, str, str, float]:
        """Return the fields the certificate is equal on."""
        return (self._verdict, self._prover, self._profile_digest, self._timeout)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another certificate records the same decision."""
        if type(other) is not EntailmentCertificate:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the verdict, the prover, the profile and the clock."""
        return hash((EntailmentCertificate, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this certificate."""
        return (
            f"EntailmentCertificate({self._verdict!r}, prover={self._prover!r}, "
            f"profile_digest={self._profile_digest!r}, timeout={self._timeout!r})"
        )

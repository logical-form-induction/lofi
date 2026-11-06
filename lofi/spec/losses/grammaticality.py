"""Grammaticality loss declaration."""

from typing import TYPE_CHECKING, Literal, NamedTuple, override

import didactic.api as dx

from lofi._checks import ensure_finite, ensure_name
from lofi.spec.components.loss import LossSpec

if TYPE_CHECKING:
    from lofi.spec.components.model import ModelSection

__all__ = ["GrammaticalityLossSettings", "GrammaticalityLossSpec"]


class GrammaticalityLossSettings(NamedTuple):
    """The settings a grammaticality objective is built from.

    Attributes
    ----------
    readout : str
        The identifier of the published root target mass pair.
    noise : float
        The probability that an observed grammaticality label was flipped.
    """

    readout: str
    noise: float


class GrammaticalityLossSpec(LossSpec):
    """Declares binary grammaticality scoring against a published root mass.

    Attributes
    ----------
    kind : Literal["grammaticality"]
        Registry tag.
    readout : str
        The identifier of the published root target mass pair.
    noise : float
        The probability that an observed grammaticality label was flipped.
    """

    kind: Literal["grammaticality"] = "grammaticality"
    readout: str
    noise: float = 0.0

    @dx.validates("readout")
    @classmethod
    def _readout_is_named(cls, value: str) -> str:
        """Validate the published-readout name."""
        return ensure_name("a published readout", value)

    @dx.validates("noise")
    @classmethod
    def _noise_is_informative(cls, value: float) -> float:
        """Validate a finite label-noise rate below one half."""
        held = ensure_finite("the label-noise rate", value)

        if not 0.0 <= held < 0.5:
            msg = f"a label-noise rate lies in [0, 0.5), not {value!r}"

            raise ValueError(msg)

        return held

    @override
    def refuse_model(self, model: ModelSection, path: str, /) -> None:
        """Refuse a root mass that no declared readout publishes.

        Raises
        ------
        ValueError
            If no readout of the model publishes a target mass pair under
            ``readout``.
        """
        published = sorted(
            {
                mass
                for readout in model.readouts.values()
                for mass in readout.published_masses()
            }
        )

        if self.readout not in published:
            msg = (
                f"{path}.readout: the objective reads the root mass "
                f"{self.readout!r}, which no declared readout publishes; they "
                f"publish {published}"
            )
            raise ValueError(msg)

    def settings(self) -> GrammaticalityLossSettings:
        """Return the settings the plugin builds this objective from."""
        return GrammaticalityLossSettings(readout=self.readout, noise=self.noise)

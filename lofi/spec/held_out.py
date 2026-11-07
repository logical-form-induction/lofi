"""The rules that predict a held-out participant's responses."""

import enum

__all__ = ["HeldOutParticipantPolicy"]


class HeldOutParticipantPolicy(enum.StrEnum):
    """Selects how an unseen participant's response is predicted.

    A fit that holds a participant out predicts that participant either by
    marginalizing over the population's participant effects (``marginal``) or
    by fixing the participant effects at their population means (``plug_in``).
    """

    MARGINAL = "marginal"
    PLUG_IN = "plug_in"

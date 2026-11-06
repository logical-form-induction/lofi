"""LoFI's public package.

The package publishes the error every law suite raises, so one ``except``
clause catches a violation whichever layer's laws found it.
"""

from lofi._checks import LawViolationError

__all__ = ["LawViolationError"]

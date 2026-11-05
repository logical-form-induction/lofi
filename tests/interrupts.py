"""Interrupting a child process from the keyboard's console on every platform.

POSIX delivers ``SIGINT`` to one child process. Windows cannot send Ctrl-C to
one child, but it can send Ctrl-Break to a child that leads its own process
group, and Python receives that as ``SIGBREAK``. A child that handles
``INTERRUPT`` with ``signal.default_int_handler`` therefore sees the same
``KeyboardInterrupt`` on both.
"""

import signal
import subprocess
import sys
from typing import Final

if sys.platform == "win32":
    INTERRUPT: Final = signal.SIGBREAK
    """The signal the console interrupt arrives as in the child."""

    NEW_GROUP: Final = subprocess.CREATE_NEW_PROCESS_GROUP
    """The creation flags of a child the console interrupt can reach."""

    def interrupt(process: subprocess.Popen[str] | subprocess.Popen[bytes], /) -> None:
        """Send the console interrupt to a child started with ``NEW_GROUP``."""
        process.send_signal(signal.CTRL_BREAK_EVENT)

else:
    INTERRUPT: Final = signal.SIGINT
    """The signal the console interrupt arrives as in the child."""

    NEW_GROUP: Final = 0
    """The creation flags of a child the console interrupt can reach."""

    def interrupt(process: subprocess.Popen[str] | subprocess.Popen[bytes], /) -> None:
        """Send the console interrupt to a child started with ``NEW_GROUP``."""
        process.send_signal(signal.SIGINT)

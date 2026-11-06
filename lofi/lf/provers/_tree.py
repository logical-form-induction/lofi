"""Killing a prover process together with every process it started."""

import os
import signal
import sys
from contextlib import suppress
from typing import Final, Self, final

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

__all__ = ["ProcessTree"]

if sys.platform == "win32":
    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION: Final = 9
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE: Final = 0x2000
    _PROCESS_SET_QUOTA: Final = 0x0100
    _PROCESS_TERMINATE: Final = 0x0001
    _KILLED: Final = 1
    _NOT_INHERITED: Final = 0

    class _BasicLimits(ctypes.Structure):
        """The basic limits of a Windows job object."""

        _fields_ = (
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        )

    class _IoCounters(ctypes.Structure):
        """The I/O counters a Windows job object reports."""

        _fields_ = (
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        )

    class _ExtendedLimits(ctypes.Structure):
        """The extended limits of a Windows job object."""

        _fields_ = (
            ("BasicLimitInformation", _BasicLimits),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        )

    _KERNEL32: Final = ctypes.WinDLL("kernel32", use_last_error=True)
    _KERNEL32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
    _KERNEL32.CreateJobObjectW.restype = wintypes.HANDLE
    _KERNEL32.SetInformationJobObject.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    )
    _KERNEL32.SetInformationJobObject.restype = wintypes.BOOL
    _KERNEL32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    _KERNEL32.OpenProcess.restype = wintypes.HANDLE
    _KERNEL32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    _KERNEL32.AssignProcessToJobObject.restype = wintypes.BOOL
    _KERNEL32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
    _KERNEL32.TerminateJobObject.restype = wintypes.BOOL
    _KERNEL32.CloseHandle.argtypes = (wintypes.HANDLE,)
    _KERNEL32.CloseHandle.restype = wintypes.BOOL

    def _checked(succeeded: object, /) -> None:
        """Raise the last Windows error if a call reported failure."""
        if not succeeded:
            raise ctypes.WinError(ctypes.get_last_error())

    @final
    class ProcessTree:
        """A process and every process it starts, killed together.

        On Windows the process is placed in a job object that its children
        join as they start, and killing the tree terminates the job. The job
        is created to kill its processes when it is closed, so leaving the
        context kills whatever is still running. A child the process starts
        before it is placed in the job is not reached.

        Parameters
        ----------
        pid : int
            The process identifier, of a process started with
            ``start_new_session=True``.
        """

        __slots__ = ("_job",)

        def __init__(self, pid: int, /) -> None:
            job = _KERNEL32.CreateJobObjectW(None, None)
            _checked(job)
            self._job = job
            limits = _ExtendedLimits()
            limits.BasicLimitInformation.LimitFlags = (
                _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            )
            _checked(
                _KERNEL32.SetInformationJobObject(
                    job,
                    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                    ctypes.byref(limits),
                    ctypes.sizeof(limits),
                )
            )
            process = _KERNEL32.OpenProcess(
                _PROCESS_SET_QUOTA | _PROCESS_TERMINATE, _NOT_INHERITED, pid
            )
            _checked(process)

            try:
                _checked(_KERNEL32.AssignProcessToJobObject(job, process))

            finally:
                _KERNEL32.CloseHandle(process)

        def kill(self) -> None:
            """Kill the process and every process it started."""
            _KERNEL32.TerminateJobObject(self._job, _KILLED)

        def __enter__(self) -> Self:
            """Return the tree for the duration of a block."""
            return self

        def __exit__(self, *_: object) -> None:
            """Kill the tree and release its job object."""
            self.kill()
            _KERNEL32.CloseHandle(self._job)

else:

    @final
    class ProcessTree:
        """A process and every process it starts, killed together.

        On POSIX the process leads its own session, and killing the tree
        kills every process of that session. Leaving the context kills
        whatever is still running.

        Parameters
        ----------
        pid : int
            The process identifier, of a process started with
            ``start_new_session=True``.
        """

        __slots__ = ("_pid",)

        def __init__(self, pid: int, /) -> None:
            self._pid = pid

        def kill(self) -> None:
            """Kill the process and every process it started."""
            with suppress(ProcessLookupError, PermissionError):
                os.killpg(self._pid, signal.SIGKILL)

        def __enter__(self) -> Self:
            """Return the tree for the duration of a block."""
            return self

        def __exit__(self, *_: object) -> None:
            """Kill the tree."""
            self.kill()

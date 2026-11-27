"""Windows job supervisor for temporary, artificial pilot worker fixtures only.

The child is created suspended and assigned to the job before it can run.
Closing the job on timeout terminates its entire process tree. This module
cannot issue a GO or accept a production repository/destination.
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
import tempfile
import time
from ctypes import wintypes
from pathlib import Path
from typing import Any

try:
    import _winapi
except ImportError:  # Windows-only implementation; import remains safe elsewhere.
    _winapi = None

_KILL_ON_CLOSE = 0x2000
_JOB_MEMORY_LIMIT = 0x200
_EXTENDED_LIMIT_INFORMATION = 9
_WAIT_TIMEOUT = 0x102
_CREATE_SUSPENDED = 0x4


class _BasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
    )]


class _ExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def run_synthetic_worker(
    command: tuple[str, ...], *, cwd: Path, environment: dict[str, str],
    timeout_seconds: float, memory_bytes: int,
) -> dict[str, Any]:
    """Run one Python fixture under an OS-enforced job and return its outcome."""
    root = Path(cwd).resolve()
    if (sys.platform != "win32" or _winapi is None or not root.is_relative_to(
            Path(tempfile.gettempdir()).resolve())
            or not root.is_dir() or len(command) < 2
            or Path(command[0]).resolve() != Path(sys.executable).resolve()
            or not 0 < timeout_seconds <= 60 or not 16 * 1024**2 <= memory_bytes <= 512 * 1024**2):
        raise ValueError("PILOT_SYNTHETIC_SUPERVISOR_SCOPE_INVALID")
    return _run_windows_job(
        command, root=root, environment=environment,
        timeout_seconds=timeout_seconds, memory_bytes=memory_bytes,
    )


def _run_windows_job(
    command: tuple[str, ...], *, root: Path, environment: dict[str, str],
    timeout_seconds: float, memory_bytes: int,
) -> dict[str, Any]:
    """Apply one aggregate commit limit and wall timeout to a child tree."""
    if (sys.platform != "win32" or _winapi is None or not root.is_dir()
            or len(command) < 2
            or Path(command[0]).resolve() != Path(sys.executable).resolve()
            or not 0 < timeout_seconds <= 1800
            or not 16 * 1024**2 <= memory_bytes <= 1024**3):
        raise ValueError("PILOT_JOB_CONFIGURATION_INVALID")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.SetInformationJobObject.restype = wintypes.BOOL
    kernel.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
    ]
    kernel.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.ResumeThread.restype = wintypes.DWORD
    kernel.ResumeThread.argtypes = [wintypes.HANDLE]
    kernel.QueryInformationJobObject.restype = wintypes.BOOL
    kernel.QueryInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
        ctypes.c_void_p,
    ]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise OSError(ctypes.get_last_error(), "PILOT_JOB_CREATE_FAILED")
    limits = _ExtendedLimitInformation()
    limits.BasicLimitInformation.LimitFlags = _KILL_ON_CLOSE | _JOB_MEMORY_LIMIT
    limits.JobMemoryLimit = memory_bytes
    process = thread = None
    started = time.monotonic()
    try:
        if not kernel.SetInformationJobObject(
            job, _EXTENDED_LIMIT_INFORMATION, ctypes.byref(limits),
            ctypes.sizeof(limits),
        ):
            raise OSError(ctypes.get_last_error(), "PILOT_JOB_LIMIT_FAILED")
        applied = _ExtendedLimitInformation()
        if not kernel.QueryInformationJobObject(
            job, _EXTENDED_LIMIT_INFORMATION, ctypes.byref(applied),
            ctypes.sizeof(applied), None,
        ):
            raise OSError(ctypes.get_last_error(), "PILOT_JOB_APPLIED_LIMIT_QUERY_FAILED")
        if (int(applied.JobMemoryLimit) != memory_bytes
                or int(applied.BasicLimitInformation.LimitFlags)
                   & (_KILL_ON_CLOSE | _JOB_MEMORY_LIMIT)
                   != (_KILL_ON_CLOSE | _JOB_MEMORY_LIMIT)):
            raise OSError("PILOT_JOB_APPLIED_LIMIT_DIVERGENCE")
        startup = subprocess.STARTUPINFO()
        process, thread, pid, _tid = _winapi.CreateProcess(
            command[0], subprocess.list2cmdline(command), None, None, False,
            _CREATE_SUSPENDED | subprocess.CREATE_NO_WINDOW,
            environment, str(root), startup,
        )
        if not kernel.AssignProcessToJobObject(job, process):
            raise OSError(ctypes.get_last_error(), "PILOT_JOB_ASSIGN_FAILED")
        if kernel.ResumeThread(thread) == 0xFFFFFFFF:
            raise OSError(ctypes.get_last_error(), "PILOT_WORKER_RESUME_FAILED")
        waited = _winapi.WaitForSingleObject(process, int(timeout_seconds * 1000))
        peak = _ExtendedLimitInformation()
        if not kernel.QueryInformationJobObject(
            job, _EXTENDED_LIMIT_INFORMATION, ctypes.byref(peak),
            ctypes.sizeof(peak), None,
        ):
            raise OSError(ctypes.get_last_error(), "PILOT_JOB_QUERY_FAILED")
        if (int(peak.JobMemoryLimit) != memory_bytes
                or int(peak.BasicLimitInformation.LimitFlags)
                   & (_KILL_ON_CLOSE | _JOB_MEMORY_LIMIT)
                   != (_KILL_ON_CLOSE | _JOB_MEMORY_LIMIT)):
            raise OSError("PILOT_JOB_APPLIED_LIMIT_DIVERGENCE")
        status = "timeout" if waited == _WAIT_TIMEOUT else (
            "complete" if _winapi.GetExitCodeProcess(process) == 0 else "worker_failed"
        )
        return {
            "status": status, "pid": pid,
            "exit_code": None if status == "timeout" else _winapi.GetExitCodeProcess(process),
            "peak_job_memory_bytes": int(peak.PeakJobMemoryUsed),
            "memory_limit_bytes": memory_bytes,
            "job_memory_limit_bytes": int(peak.JobMemoryLimit),
            "job_limit_flags": int(peak.BasicLimitInformation.LimitFlags),
            "elapsed_seconds": time.monotonic() - started,
        }
    finally:
        # This OS action kills descendants, including on timeout or exception.
        kernel.CloseHandle(job)
        if thread is not None:
            _winapi.CloseHandle(thread)
        if process is not None:
            _winapi.CloseHandle(process)


def verify_current_operational_job(*, memory_bytes: int = 1024**3) -> None:
    """Child-side guard: reject direct unsupervised operational entry."""
    if sys.platform != "win32" or memory_bytes != 1024**3:
        raise ValueError("PILOT_OPERATIONAL_JOB_REQUIRED")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.IsProcessInJob.restype = wintypes.BOOL
    kernel.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE,
                                    ctypes.POINTER(wintypes.BOOL)]
    kernel.QueryInformationJobObject.restype = wintypes.BOOL
    kernel.QueryInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
        ctypes.c_void_p,
    ]
    included = wintypes.BOOL()
    if not kernel.IsProcessInJob(kernel.GetCurrentProcess(), None, ctypes.byref(included)):
        raise OSError(ctypes.get_last_error(), "PILOT_OPERATIONAL_JOB_QUERY_FAILED")
    if not included.value:
        raise ValueError("PILOT_OPERATIONAL_JOB_REQUIRED")
    observed = _ExtendedLimitInformation()
    if not kernel.QueryInformationJobObject(
        None, _EXTENDED_LIMIT_INFORMATION, ctypes.byref(observed),
        ctypes.sizeof(observed), None,
    ):
        raise OSError(ctypes.get_last_error(), "PILOT_OPERATIONAL_JOB_QUERY_FAILED")
    if (int(observed.JobMemoryLimit) != memory_bytes
            or int(observed.BasicLimitInformation.LimitFlags)
               & (_KILL_ON_CLOSE | _JOB_MEMORY_LIMIT)
               != (_KILL_ON_CLOSE | _JOB_MEMORY_LIMIT)):
        raise ValueError("PILOT_OPERATIONAL_JOB_LIMIT_DIVERGENCE")

"""CPU, memory, and disk for ``/health``.

Read on demand and never cached: a health check exists to say what is true
right now, and a stale number is worse than no number.

Two things about these figures are worth knowing before trusting them.

**They describe the machine, not this process, and in a container that is the
*host* machine.** ``psutil`` reads ``/proc``, which inside Docker still shows
the host's CPU count and total memory rather than whatever cgroup limit the
container was given. A container capped at 512MB on a 32GB host reports 32GB
here and looks idle right up until the OOM killer takes it. On any
request-scoped host the numbers describe a sandbox frozen between requests,
which makes the CPU figure close to meaningless.

**Nothing here may fail the health check.** Every reading is taken
independently and a failure is logged and dropped, so a platform that refuses
one of them still answers ``{"status": "ok"}`` with the other two. A liveness
endpoint that 500s because it could not stat a filesystem would take the
service out of a load balancer over nothing.
"""

from __future__ import annotations

import logging
import os

import psutil

from app.api.schemas import CpuStats, DiskStats, MachineStats, MemoryStats

logger = logging.getLogger(__name__)

# Which filesystem to report on. The one the process is running from by
# default -- that is the one whose filling up stops this service working.
# ``disk_usage`` reports the mount containing the path, so this does not have
# to name a mount point itself.
DISK_PATH = os.environ.get("RAG_HEALTH_DISK_PATH") or os.getcwd()


def primeCpuPercent() -> None:
    """Start the clock that ``cpuStats`` measures against.

    ``cpu_percent(interval=None)`` reports the average since the *previous*
    call, so the first call in a process has nothing to compare against and
    always answers 0.0. Priming at import means the first real health check
    gets a figure covering the time since startup rather than a zero that
    looks like an idle machine.

    The alternative -- ``interval=0.1`` -- measures properly but blocks for
    100ms, and this endpoint is the one a load balancer hits every few
    seconds. Blocking the event loop on every one of those to sharpen a
    diagnostic number is the wrong trade.
    """
    psutil.cpu_percent(interval=None)


def cpuStats() -> CpuStats | None:
    try:
        # Non-blocking: the window is "since the last call", which for a
        # regularly polled health endpoint is the interval between polls. Two
        # checks arriving back to back therefore make the second one's figure
        # near-meaningless -- it covers milliseconds. Averaged over ordinary
        # polling it is representative, and that is what it is for.
        return CpuStats(
            used_percent=psutil.cpu_percent(interval=None),
            count=psutil.cpu_count(logical=True) or 0,
        )
    except Exception:
        logger.warning("Could not read CPU usage.", exc_info=True)
        return None


def memoryStats() -> MemoryStats | None:
    try:
        memory = psutil.virtual_memory()
    except Exception:
        logger.warning("Could not read memory usage.", exc_info=True)
        return None

    # `available` rather than `total - used`: it counts reclaimable cache as
    # free, which is what "how much can I still allocate" means on Linux.
    # `used` excludes that cache, so the three do not add up, deliberately.
    return MemoryStats(
        total_bytes=memory.total,
        available_bytes=memory.available,
        used_bytes=memory.used,
        used_percent=memory.percent,
    )


def diskStats(path: str = DISK_PATH) -> DiskStats | None:
    try:
        disk = psutil.disk_usage(path)
    except Exception:
        logger.warning("Could not read disk usage for '%s'.", path, exc_info=True)
        return None

    return DiskStats(
        path=path,
        total_bytes=disk.total,
        free_bytes=disk.free,
        used_bytes=disk.used,
        used_percent=disk.percent,
    )


def machineStats() -> MachineStats | None:
    """Everything ``/health`` reports about the machine.

    Whichever readings failed are ``None`` and are omitted from the response
    rather than sent as null, so a caller can tell "not available here" from
    "zero". If none of the three could be read, this is ``None`` in turn and
    ``machine`` disappears from the response altogether -- an empty object
    would say the same thing with more noise.
    """
    stats = MachineStats(cpu=cpuStats(), memory=memoryStats(), disk=diskStats())
    if (stats.cpu, stats.memory, stats.disk) == (None, None, None):
        return None
    return stats

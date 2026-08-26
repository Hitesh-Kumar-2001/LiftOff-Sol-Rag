"""The unversioned endpoints: liveness, and the machine figures on it."""

import pytest
from fastapi.testclient import TestClient

from app.infra import machineStats as machineStatsModule
from app.main import app

client = TestClient(app)


def health() -> dict:
    response = client.get("/health")
    assert response.status_code == 200
    return response.json()


def testHealthReportsOk() -> None:
    """The field anything already watching this endpoint is checking. Adding
    machine figures alongside it must not have moved or renamed it."""
    assert health()["status"] == "ok"


def testHealthReportsCpuMemoryAndDisk() -> None:
    machine = health()["machine"]

    assert set(machine) == {"cpu", "memory", "disk"}


def testCpuUsageIsAPercentageOfARealCoreCount() -> None:
    cpu = health()["machine"]["cpu"]

    assert 0.0 <= cpu["usedPercent"] <= 100.0
    assert cpu["count"] >= 1


def testMemoryUsageIsInternallyConsistent() -> None:
    memory = health()["machine"]["memory"]

    assert memory["totalBytes"] > 0
    assert 0 <= memory["usedBytes"] <= memory["totalBytes"]
    assert 0 <= memory["availableBytes"] <= memory["totalBytes"]
    assert 0.0 <= memory["usedPercent"] <= 100.0


def testDiskUsageIsInternallyConsistent() -> None:
    disk = health()["machine"]["disk"]

    assert disk["path"]
    assert disk["totalBytes"] > 0
    assert disk["usedBytes"] + disk["freeBytes"] <= disk["totalBytes"]
    assert 0.0 <= disk["usedPercent"] <= 100.0


@pytest.mark.parametrize(
    "reading, section",
    [("virtual_memory", "memory"), ("disk_usage", "disk"), ("cpu_percent", "cpu")],
)
def testAReadingThatFailsIsOmittedRatherThanFailingTheCheck(
    monkeypatch, reading: str, section: str
) -> None:
    """A liveness endpoint that 500s because it could not stat a filesystem
    would pull a working service out of a load balancer over nothing."""

    def refuse(*args, **kwargs):
        raise OSError("not permitted in this sandbox")

    monkeypatch.setattr(machineStatsModule.psutil, reading, refuse)

    payload = health()

    assert payload["status"] == "ok"
    assert section not in payload["machine"]


def testEveryReadingFailingStillReportsOk(monkeypatch) -> None:
    """The whole point: no machine figure is worth failing liveness for."""
    for reading in ("virtual_memory", "disk_usage", "cpu_percent", "cpu_count"):

        def refuse(*args, **kwargs):
            raise OSError("not permitted in this sandbox")

        monkeypatch.setattr(machineStatsModule.psutil, reading, refuse)

    payload = health()

    # `machine` itself is omitted once nothing inside it could be read.
    assert payload == {"status": "ok"}


def testRoot() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "RAG API is running"}


# --- startup: what must, and must not, stop the process coming up ----------


def testAFailingCpuPrimeDoesNotStopStartup(monkeypatch) -> None:
    """The measurement primer runs in the lifespan hook, so unlike the readings
    it guards nothing downstream -- an exception there propagates out of startup
    and the application never comes up. On a platform where psutil cannot read
    /proc that is a container crash-looping over a diagnostic number."""

    def broken(*args, **kwargs):
        raise RuntimeError("no /proc here")

    monkeypatch.setattr(machineStatsModule.psutil, "cpu_percent", broken)

    with TestClient(app) as started:
        assert started.get("/health").status_code == 200


def testAMisconfiguredDeploymentFailsAtStartup(monkeypatch) -> None:
    """Deliberately fatal, and deliberately *at startup*.

    Left to build lazily, the first request would raise instead: every request
    would 500 while /health went on answering ok, so a platform would shift all
    traffic onto a task that cannot serve any of it and keep it there. Refusing
    to start stops the rollout and leaves the previous version running.
    """
    from app.main import checkConfiguration
    from app.stores.chatStore import getChatStore
    from app.stores.projectStore import getProjectStore

    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)
    getProjectStore.cache_clear()
    getChatStore.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="GCP_PROJECT_ID"):
            checkConfiguration()
    finally:
        # The caches are process-wide; leaving a failed build cached, or a
        # store built without a GCP project, would follow every later test.
        getProjectStore.cache_clear()
        getChatStore.cache_clear()

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
